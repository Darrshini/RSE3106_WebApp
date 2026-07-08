/**
 * NavAssist — ai.js  (LAPTOP WEBCAM TEST VERSION)
 * OWNER: Kim Hyeonghu (original) + audio/FSM fixes (Darrshini, see comments)
 *
 * This is the WEBCAM version — same camera source as your teammate's
 * original file (getUserMedia, <video id="camVideo">) — kept ONLY so you
 * can verify the red-light / multi-post audio fixes locally before
 * switching the live server over to the ESP32 version (ai_corrected.js).
 *
 * DO NOT push this file to the server — it expects <video id="camVideo">
 * in index.html, which the ESP32 version replaces with <canvas id="camFrame">.
 * This file pairs with your CURRENT (unchanged) index.html/style.css.
 *
 * Fixes applied vs the original, all inside handleGuidance():
 *   1. Calls window.navassist.onTrafficLightVisible() when a traffic-light
 *      post is detected during SCANNING — previously never called, so the
 *      state machine would get stuck at SCANNING forever.
 *   2. Announces LEFT/RIGHT when multiple traffic-light posts are seen.
 *   3. Speaks "Still red. Please continue to wait." periodically while
 *      WAITING and a red man is detected — previously silent.
 */

// ============================================================
// Config
// ============================================================
const MODEL_URL   = 'models/pedestrian.onnx';
const CLASSES     = ['red', 'green', 'traffic-light'];   // must match training order
const CLASS_COLOR = { red: '#ff1744', green: '#00c853', 'traffic-light': '#2979ff' };
const INPUT_SIZE  = 640;      // model input (letterboxed square)
const CONF_THRESH = 0.35;     // detection confidence threshold
const IOU_THRESH  = 0.45;     // NMS IoU threshold
const INFER_EVERY_MS = 180;   // throttle inference (video still renders every frame)
const GREEN_SPEAK_COOLDOWN_MS = 4000;
const RED_SPEAK_COOLDOWN_MS = 6000;   // less frequent than green -- it's a "keep waiting" reminder

// ============================================================
// State
// ============================================================
let session = null;
let video, overlay, octx;
let preCanvas, preCtx;        // offscreen canvas for letterbox pre-processing
let latestDetections = [];
let started = false;
let modelStatus = 'idle';
let lastGreenSpeakAt = 0;
let lastRedSpeakAt = 0;
let fps = 0, lastFrameTs = 0;

// onnxruntime-web: fetch the wasm binaries from the CDN
if (window.ort) {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
    ort.env.wasm.numThreads = 1;
}

// ============================================================
// Public entry points
// ============================================================
window.navassist = window.navassist || {};
window.navassist.startCameraAI = startCameraAI;

function handleCameraFrame() { /* no-op: webcam drives detection directly */ }

window.addEventListener('load', () => {
    const splashBtn = document.getElementById('splashButton');
    const splash    = document.getElementById('splashScreen');
    const kickoff = () => startCameraAI();
    if (splashBtn) splashBtn.addEventListener('click', kickoff, { once: true });
    if (splash)    splash.addEventListener('touchstart', kickoff, { once: true, passive: true });
});

// ============================================================
// Camera + model bootstrap
// ============================================================
async function startCameraAI() {
    if (started) return;
    started = true;

    video   = document.getElementById('camVideo');
    overlay = document.getElementById('camOverlay');
    if (!video || !overlay) { started = false; return; }
    octx = overlay.getContext('2d');

    preCanvas = document.createElement('canvas');
    preCanvas.width = preCanvas.height = INPUT_SIZE;
    preCtx = preCanvas.getContext('2d', { willReadFrequently: true });

    try {
        await startWebcam();
    } catch (e) {
        modelStatus = 'camera-error';
        setContext('Camera access failed: ' + e.message + '. Allow camera permission and reload.');
        window.navassist.debugLog && window.navassist.debugLog('getUserMedia error: ' + e.message);
        return;
    }

    modelStatus = 'loading-model';
    setContext('Loading detection model…');
    try {
        session = await ort.InferenceSession.create(MODEL_URL, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        modelStatus = 'ready';
        setContext('Point the camera at a pedestrian light.');
        window.navassist.debugLog && window.navassist.debugLog(
            'Model loaded. in=' + session.inputNames + ' out=' + session.outputNames);
    } catch (e) {
        modelStatus = 'model-error';
        setContext('Model failed to load (' + e.message + '). Is models/pedestrian.onnx present?');
        window.navassist.debugLog && window.navassist.debugLog('Model load error: ' + e.message);
    }

    requestAnimationFrame(renderLoop);
    detectLoop();
}

async function startWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
    });
    video.srcObject = stream;
    await video.play();
    const sync = () => {
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        const cv = overlay.parentElement;
        if (cv && video.videoWidth) cv.style.aspectRatio = video.videoWidth + ' / ' + video.videoHeight;
    };
    if (video.videoWidth) sync(); else video.addEventListener('loadedmetadata', sync, { once: true });
}

// ============================================================
// Inference loop (throttled)
// ============================================================
async function detectLoop() {
    while (started) {
        const t0 = performance.now();
        if (session && video.readyState >= 2 && video.videoWidth) {
            try {
                const pre = preprocess();
                const feeds = {}; feeds[session.inputNames[0]] = pre.tensor;
                const results = await session.run(feeds);
                const out = results[session.outputNames[0]];
                latestDetections = postprocess(out, pre);
                handleGuidance(latestDetections);
            } catch (e) {
                window.navassist.debugLog && window.navassist.debugLog('Inference error: ' + e.message);
            }
        }
        const elapsed = performance.now() - t0;
        await sleep(Math.max(0, INFER_EVERY_MS - elapsed));
    }
}

function preprocess() {
    const iw = video.videoWidth, ih = video.videoHeight, S = INPUT_SIZE;
    const scale = Math.min(S / iw, S / ih);
    const nw = Math.round(iw * scale), nh = Math.round(ih * scale);
    const padX = Math.floor((S - nw) / 2), padY = Math.floor((S - nh) / 2);

    preCtx.fillStyle = 'rgb(114,114,114)';
    preCtx.fillRect(0, 0, S, S);
    preCtx.drawImage(video, padX, padY, nw, nh);

    const rgba = preCtx.getImageData(0, 0, S, S).data;
    const area = S * S;
    const data = new Float32Array(3 * area);
    for (let i = 0; i < area; i++) {
        data[i]            = rgba[i * 4]     / 255;
        data[i + area]     = rgba[i * 4 + 1] / 255;
        data[i + 2 * area] = rgba[i * 4 + 2] / 255;
    }
    return { tensor: new ort.Tensor('float32', data, [1, 3, S, S]), scale, padX, padY };
}

function postprocess(out, pre) {
    const dims = out.dims;
    const num  = dims[2];
    const nCls = dims[1] - 4;
    const d    = out.data;
    const { scale, padX, padY } = pre;

    const boxes = [];
    for (let a = 0; a < num; a++) {
        let best = 0, bestC = 0;
        for (let c = 0; c < nCls; c++) {
            const s = d[(4 + c) * num + a];
            if (s > best) { best = s; bestC = c; }
        }
        if (best < CONF_THRESH) continue;
        const cx = d[a], cy = d[num + a], w = d[2 * num + a], h = d[3 * num + a];
        boxes.push({
            x1: (cx - w / 2 - padX) / scale,
            y1: (cy - h / 2 - padY) / scale,
            x2: (cx + w / 2 - padX) / scale,
            y2: (cy + h / 2 - padY) / scale,
            score: best, cls: bestC
        });
    }
    return nms(boxes, IOU_THRESH);
}

function nms(boxes, iouThr) {
    boxes.sort((a, b) => b.score - a.score);
    const keep = [], dead = new Array(boxes.length).fill(false);
    for (let i = 0; i < boxes.length; i++) {
        if (dead[i]) continue;
        keep.push(boxes[i]);
        for (let j = i + 1; j < boxes.length; j++) {
            if (!dead[j] && boxes[i].cls === boxes[j].cls && iou(boxes[i], boxes[j]) > iouThr) dead[j] = true;
        }
    }
    return keep;
}

function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1), areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-6);
}

// ============================================================
// Rendering
// ============================================================
function renderLoop(ts) {
    if (!started) return;
    if (lastFrameTs) fps = 0.9 * fps + 0.1 * (1000 / (ts - lastFrameTs));
    lastFrameTs = ts;
    drawOverlay(latestDetections);
    requestAnimationFrame(renderLoop);
}

function drawOverlay(dets) {
    const W = overlay.width, H = overlay.height;
    if (!W || !H) return;
    octx.clearRect(0, 0, W, H);
    const lw = Math.max(2, W * 0.004);
    const fontPx = Math.max(14, Math.round(W * 0.022));

    for (const det of dets) {
        const name = CLASSES[det.cls], color = CLASS_COLOR[name] || '#ffffff';
        octx.lineWidth = lw; octx.strokeStyle = color;
        octx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
        const label = `${name} ${Math.round(det.score * 100)}%`;
        octx.font = `bold ${fontPx}px sans-serif`;
        const tw = octx.measureText(label).width;
        const ty = Math.max(det.y1, fontPx + 6);
        octx.fillStyle = color;
        octx.fillRect(det.x1, ty - fontPx - 6, tw + 10, fontPx + 6);
        octx.fillStyle = '#000';
        octx.fillText(label, det.x1 + 5, ty - 6);
    }

    const greens = dets.filter(d => CLASSES[d.cls] === 'green');
    if (greens.length) {
        const g = greens.reduce((a, b) => (a.score > b.score ? a : b));
        const gx = (g.x1 + g.x2) / 2, gy = (g.y1 + g.y2) / 2;
        const ox = W / 2, oy = H - lw;
        drawArrow(ox, oy, gx, gy, '#00e5ff', W);
        const dir = getDirection(gx / W);
        drawBanner(`🟢 GREEN — GO ${dir}`, '#00c853', W, H, fontPx);
    }

    octx.font = `${Math.max(12, Math.round(W * 0.016))}px monospace`;
    octx.fillStyle = 'rgba(0,0,0,0.55)';
    const hud = `${modelStatus}  ${dets.length} det  ${fps.toFixed(0)} fps`;
    const hw = octx.measureText(hud).width;
    octx.fillRect(6, 6, hw + 12, 22);
    octx.fillStyle = '#0f0';
    octx.fillText(hud, 12, 22);
}

function drawArrow(x1, y1, x2, y2, color, W) {
    const head = Math.max(14, W * 0.03);
    const ang = Math.atan2(y2 - y1, x2 - x1);
    octx.strokeStyle = color; octx.fillStyle = color;
    octx.lineWidth = Math.max(4, W * 0.008);
    octx.lineCap = 'round';
    octx.beginPath(); octx.moveTo(x1, y1); octx.lineTo(x2, y2); octx.stroke();
    octx.beginPath();
    octx.moveTo(x2, y2);
    octx.lineTo(x2 - head * Math.cos(ang - Math.PI / 6), y2 - head * Math.sin(ang - Math.PI / 6));
    octx.lineTo(x2 - head * Math.cos(ang + Math.PI / 6), y2 - head * Math.sin(ang + Math.PI / 6));
    octx.closePath(); octx.fill();
    octx.beginPath(); octx.arc(x1, y1, Math.max(5, W * 0.01), 0, Math.PI * 2); octx.fill();
}

function drawBanner(text, color, W, H, fontPx) {
    octx.font = `bold ${Math.round(fontPx * 1.2)}px sans-serif`;
    const tw = octx.measureText(text).width, pad = 14;
    const bw = tw + pad * 2, bh = fontPx * 1.9;
    const bx = (W - bw) / 2, by = H - bh - 14;
    octx.fillStyle = 'rgba(0,0,0,0.65)';
    roundRect(bx, by, bw, bh, 10); octx.fill();
    octx.fillStyle = color;
    octx.fillText(text, bx + pad, by + bh - fontPx * 0.7);
}

function roundRect(x, y, w, h, r) {
    octx.beginPath();
    octx.moveTo(x + r, y);
    octx.arcTo(x + w, y, x + w, y + h, r);
    octx.arcTo(x + w, y + h, x, y + h, r);
    octx.arcTo(x, y + h, x, y, r);
    octx.arcTo(x, y, x + w, y, r);
    octx.closePath();
}

// ============================================================
// Guidance: post detection, direction, light-state, FSM callbacks
// ============================================================
function handleGuidance(dets) {
    if (!dets.length) return;

    const frameW = overlay.width || 1;

    // FIX 1 + 2: announce traffic-light post(s), with left/right disambiguation
    // when more than one is visible. Only meaningful while SCANNING.
    const posts = dets.filter(d => CLASSES[d.cls] === 'traffic-light');
    if (posts.length &&
        window.navassist.currentState &&
        window.navassist.currentState() === window.navassist.STATES.SCANNING) {

        if (posts.length > 1) {
            const dirs = posts
                .map(p => getDirection(((p.x1 + p.x2) / 2) / frameW))
                .filter((v, i, arr) => arr.indexOf(v) === i);
            const desc = dirs.length > 1
                ? `Multiple traffic light posts detected: ${dirs.join(' and ').toLowerCase()}.`
                : `Traffic light posts detected to your ${dirs[0].toLowerCase()}.`;
            window.navassist.speak && window.navassist.speak(desc);
        } else {
            const dir = getDirection(((posts[0].x1 + posts[0].x2) / 2) / frameW);
            if (dir !== 'CENTRE') {
                window.navassist.speak && window.navassist.speak(
                    `Traffic light post detected to your ${dir.toLowerCase()}.`);
            }
        }

        const post = posts.reduce((a, b) => (a.score > b.score ? a : b));
        window.navassist.onTrafficLightVisible && window.navassist.onTrafficLightVisible(post.score);
    }

    const best = dets.reduce((a, b) => (a.score > b.score ? a : b));
    const cx = (best.x1 + best.x2) / 2;
    const direction = getDirection(cx / frameW);

    const greens = dets.filter(d => CLASSES[d.cls] === 'green');
    if (greens.length) {
        const now = Date.now();
        if (now - lastGreenSpeakAt > GREEN_SPEAK_COOLDOWN_MS) {
            lastGreenSpeakAt = now;
            const g = greens.reduce((a, b) => (a.score > b.score ? a : b));
            const gdir = getDirection(((g.x1 + g.x2) / 2) / frameW);
            window.navassist.speak && window.navassist.speak(
                gdir === 'CENTRE' ? 'Green man ahead. You may cross.' : `Green man to your ${gdir.toLowerCase()}.`);
        }
        window.navassist.onGreenDetected && window.navassist.onGreenDetected();
    }

    // FIX 3: red-light reassurance while WAITING (previously completely silent)
    const reds = dets.filter(d => CLASSES[d.cls] === 'red');
    if (reds.length &&
        window.navassist.currentState &&
        window.navassist.currentState() === window.navassist.STATES.WAITING) {
        const now = Date.now();
        if (now - lastRedSpeakAt > RED_SPEAK_COOLDOWN_MS) {
            lastRedSpeakAt = now;
            window.navassist.speak && window.navassist.speak('Still red. Please continue to wait.');
        }
    }

    const boxH = (best.y2 - best.y1) / (overlay.height || 1);
    if (window.navassist.currentState && window.navassist.currentState() === window.navassist.STATES.NAVIGATING) {
        if (boxH > 0.4) window.navassist.onArrived && window.navassist.onArrived();
        else window.navassist.onDirectionDecided && window.navassist.onDirectionDecided(direction);
    }
}

function getDirection(normX) {
    if (normX < 0.4) return 'LEFT';
    if (normX > 0.6) return 'RIGHT';
    return 'CENTRE';
}

// ============================================================
// Helpers
// ============================================================
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function setContext(text) {
    const el = document.getElementById('contextMessage');
    if (el) el.textContent = text;
}
