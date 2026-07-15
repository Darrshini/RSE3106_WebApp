/**
 * NavAssist — webcam.js
 *
 * Runs the pedestrian-light model (models/pedestrian.onnx, classes red/green/
 * traffic-light) live off the LAPTOP WEBCAM, instead of off the Pi's WebSocket
 * camera frames the way ai.js does. Everything else is the same pipeline: the
 * model runs in js/inference.worker.js (unchanged, shared with ai.js) so it
 * never blocks the video, on WebGPU where available and WASM otherwise.
 *
 * The crossing segmentation model (/api/infer, server-side) is layered on top
 * via the existing js/crossing.js, drawing into its own canvas — so both models
 * run against the same webcam at once. Toggle it off to give pedestrian.onnx
 * the whole CPU/GPU.
 *
 * There is deliberately NO state machine here: no GPS, no compass, no haptics,
 * no WebSocket. This is a perception test bench for a PC, not the assistive app.
 */
(function () {

// ============================================================
// Config — mirrors ai.js so detections match what the real app sees
// ============================================================
const MODEL_URL   = 'models/pedestrian.onnx';
const ORT_PATHS   = 'vendor/onnxruntime/';       // self-hosted, no CDN
const CLASSES     = ['red', 'green', 'traffic-light'];
const CLASS_COLOR = { red: '#ff1744', green: '#00c853', 'traffic-light': '#2979ff' };
const INPUT_SIZE  = 640;
const IOU_THRESH  = 0.45;
const INFER_EVERY_MS = 100;   // floor between inferences; inferBusy paces it on slow devices

let confThresh = 0.35;        // live-adjustable from the slider

// ============================================================
// State
// ============================================================
let video, overlay, octx, preCanvas, preCtx;
let worker = null, workerReady = false, inferBusy = false;
let detections = [];
let backend = '', lastInferMs = 0, fps = 0, lastTs = 0, lastInferAt = 0;
let running = false, speechMuted = false;
let statusEl, hudOn = true;

// crossing.js calls window.navassist.speak if present, so defining it here is
// also how the mute checkbox silences it without touching that file.
window.navassist = window.navassist || {};
window.navassist.speak = function (text) {
    if (speechMuted || !window.speechSynthesis) return;
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(text));
};
window.navassist.debugLog = function (m) { console.log('[NavAssist]', m); };

// ============================================================
// Boot
// ============================================================
window.addEventListener('load', () => {
    video     = document.getElementById('feed');
    overlay   = document.getElementById('detOverlay');
    statusEl  = document.getElementById('status');
    octx      = overlay.getContext('2d');

    preCanvas = document.createElement('canvas');
    preCanvas.width = preCanvas.height = INPUT_SIZE;
    preCtx = preCanvas.getContext('2d', { willReadFrequently: true });

    document.getElementById('startBtn').addEventListener('click', start);
    document.getElementById('muteChk').addEventListener('change', e => {
        speechMuted = e.target.checked;
        if (speechMuted && window.speechSynthesis) speechSynthesis.cancel();
    });
    document.getElementById('hudChk').addEventListener('change', e => { hudOn = e.target.checked; });
    document.getElementById('confRange').addEventListener('input', e => {
        confThresh = Number(e.target.value) / 100;
        document.getElementById('confVal').textContent = confThresh.toFixed(2);
    });
});

async function start() {
    if (running) return;
    const btn = document.getElementById('startBtn');
    btn.disabled = true;

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } },
            audio: false
        });
        video.srcObject = stream;
        await video.play();
    } catch (e) {
        setStatus('Camera failed: ' + e.message + ' — needs https:// or localhost.');
        btn.disabled = false;
        return;
    }

    running = true;
    btn.textContent = 'Running';
    setStatus('Camera on. Loading pedestrian.onnx…');

    // Same worker file ai.js uses, loaded the same way. Both URLs must be
    // absolute: a relative one would resolve against the worker script's own
    // location (js/), not the app root.
    worker = new Worker('js/inference.worker.js');
    worker.onmessage = onWorkerMessage;
    worker.onerror = (e) => setStatus('Worker failed to start: ' + (e.message || 'unknown'));
    worker.postMessage({
        type: 'load',
        modelUrl:  new URL(MODEL_URL, location.href).href,
        wasmPaths: new URL(ORT_PATHS, location.href).href,
        numThreads: 1,
        providers: ['webgpu', 'wasm']   // WebGPU needs a secure context (localhost is one)
    });

    // Crossing segmentation model (server-side /api/infer), drawn into its own
    // canvas by crossing.js. Optional — unchecking it leaves only pedestrian.onnx.
    if (document.getElementById('crossChk').checked && window.Crossing) {
        window.Crossing.start(video, document.getElementById('crossOverlay'), (state, msg) => {
            document.getElementById('crossState').textContent = state + (msg ? ' — ' + msg : '');
        });
    }

    requestAnimationFrame(renderLoop);
}

function onWorkerMessage(e) {
    const msg = e.data;
    if (msg.type === 'ready') {
        workerReady = true;
        backend = msg.backend || '?';
        setStatus('Model ready on ' + backend.toUpperCase() + '. Point the camera at a pedestrian light.');
    } else if (msg.type === 'result') {
        inferBusy = false;
        lastInferMs = msg.inferMs || 0;
        detections = msg.detections || [];
    } else if (msg.type === 'error') {
        inferBusy = false;
        if (!workerReady) setStatus('Model failed to load: ' + msg.message);
        console.warn('[worker]', msg.message);
    } else if (msg.type === 'log') {
        console.log('[worker]', msg.message);
    }
}

// ============================================================
// Per-frame: size canvases, kick off inference, draw
// ============================================================
function renderLoop(ts) {
    if (!running) return;
    requestAnimationFrame(renderLoop);

    if (lastTs) fps = 0.9 * fps + 0.1 * (1000 / (ts - lastTs));
    lastTs = ts;

    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return;

    // Detections come back in ORIGINAL video pixels, so the overlay works in
    // that same space and CSS stretches it over the video.
    if (overlay.width !== vw || overlay.height !== vh) {
        overlay.width = vw; overlay.height = vh;
        const cv = overlay.parentElement;
        if (cv) cv.style.aspectRatio = vw + ' / ' + vh;
    }

    maybeInfer();
    draw();
}

function maybeInfer() {
    if (!workerReady || inferBusy) return;
    const now = performance.now();
    if (now - lastInferAt < INFER_EVERY_MS) return;
    lastInferAt = now;

    const pre = preprocess();
    inferBusy = true;
    // Transfer (zero-copy) rather than clone the buffer across threads.
    worker.postMessage({
        type: 'infer',
        data: pre.data.buffer,
        dims: [1, 3, INPUT_SIZE, INPUT_SIZE],
        scale: pre.scale, padX: pre.padX, padY: pre.padY,
        confThresh, iouThresh: IOU_THRESH
    }, [pre.data.buffer]);
}

// Letterbox the current video frame into a 640x640 CHW float buffer.
// Identical to ai.js's preprocess(), except the source is the <video> element
// directly rather than a canvas holding a decoded JPEG.
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
    return { data, scale, padX, padY };
}

// ============================================================
// Overlay: boxes + arrow to the strongest green + HUD
// ============================================================
function draw() {
    const W = overlay.width, H = overlay.height;
    octx.clearRect(0, 0, W, H);
    const lw = Math.max(2, W * 0.004);
    const fontPx = Math.max(14, Math.round(W * 0.022));

    for (const det of detections) {
        const name = CLASSES[det.cls], color = CLASS_COLOR[name] || '#fff';
        octx.lineWidth = lw; octx.strokeStyle = color;
        octx.strokeRect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);

        const label = name + ' ' + Math.round(det.score * 100) + '%';
        octx.font = 'bold ' + fontPx + 'px sans-serif';
        const tw = octx.measureText(label).width;
        const ty = Math.max(det.y1, fontPx + 6);
        octx.fillStyle = color;
        octx.fillRect(det.x1, ty - fontPx - 6, tw + 10, fontPx + 6);
        octx.fillStyle = '#000';
        octx.fillText(label, det.x1 + 5, ty - 6);
    }

    // Direction vector from the user (bottom-centre) to the strongest green man,
    // same convention as ai.js's drawOverlay.
    const greens = detections.filter(d => CLASSES[d.cls] === 'green');
    if (greens.length) {
        const g = greens.reduce((a, b) => (a.score > b.score ? a : b));
        const gx = (g.x1 + g.x2) / 2, gy = (g.y1 + g.y2) / 2;
        arrow(W / 2, H - lw, gx, gy, '#00e5ff', W);
    }

    if (hudOn) {
        octx.font = Math.max(12, Math.round(W * 0.016)) + 'px monospace';
        const hud = (workerReady ? 'ready' : 'loading') + '  ' + (backend || '-') + '  ' +
                    detections.length + ' det  ' + fps.toFixed(0) + ' fps  ' +
                    lastInferMs.toFixed(0) + 'ms  conf>' + confThresh.toFixed(2);
        const hw = octx.measureText(hud).width;
        octx.fillStyle = 'rgba(0,0,0,0.6)';
        octx.fillRect(6, 6, hw + 12, 24);
        octx.fillStyle = '#0f0';
        octx.fillText(hud, 12, 23);
    }
}

function arrow(x1, y1, x2, y2, color, W) {
    const head = Math.max(14, W * 0.03);
    const ang = Math.atan2(y2 - y1, x2 - x1);
    octx.strokeStyle = color; octx.fillStyle = color;
    octx.lineWidth = Math.max(4, W * 0.008); octx.lineCap = 'round';
    octx.beginPath(); octx.moveTo(x1, y1); octx.lineTo(x2, y2); octx.stroke();
    octx.beginPath(); octx.moveTo(x2, y2);
    octx.lineTo(x2 - head * Math.cos(ang - Math.PI / 6), y2 - head * Math.sin(ang - Math.PI / 6));
    octx.lineTo(x2 - head * Math.cos(ang + Math.PI / 6), y2 - head * Math.sin(ang + Math.PI / 6));
    octx.closePath(); octx.fill();
    octx.beginPath(); octx.arc(x1, y1, Math.max(5, W * 0.01), 0, Math.PI * 2); octx.fill();
}

function setStatus(msg) { if (statusEl) statusEl.textContent = msg; }

})();
