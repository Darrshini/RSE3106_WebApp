/**
 * NavAssist — ai.js
 * OWNER: Kim Hyeonghu (adapted: Raspberry Pi camera frames instead of laptop webcam)
 *
 * Camera frame ingestion + crossing overlay + vision-driven FSM.
 *
 * Uses a SINGLE model: crossing_seg.onnx, which runs on the NODE SERVER.
 * Audio logic follows crossing.js (live.html): vision-driven transitions
 * with NO tap confirmations. The model detects 'dotted line' and
 * 'pedestrian light' classes, using independent green/red boolean signals
 * and a RED_HOLD timer to distinguish blink-off from constant red.
 *
 * Flow:
 *   1. app.js calls handleCameraFrame(buf) with raw JPEG bytes from /live.
 *   2. Frames are decoded, rotated, and drawn onto #camFrame.
 *   3. The server runs crossing_seg.onnx and pushes 'crossing/result' JSON.
 *   4. handleCrossingPerception() draws the overlay AND drives state
 *      transitions directly (no tap confirmations).
 */

// ============================================================
// Config
// ============================================================
const END_NO_DASH_FRAMES = 6;
const END_LIGHT_AREA_FRAC = 0.02;
const RED_HOLD_MS = 2000;
const SPEAK_COOLDOWN_MS = 3500;

// ============================================================
// State
// ============================================================
let frameCanvas, overlay, frameCtx, octx;
let crossOverlay, crossCtx, crossMaskCanvas, crossMaskCtx;
let lastCrossDrawAt = 0;
let decodeInFlight = false;
let pendingFrame = null;

let rotateDeg = 90;

let noDashStreak = 0;
let started = false;
let modelStatus = 'idle';
let fps = 0, lastFrameTs = 0;

// crossing.js-style temporal state
let redSince = null;
let hurried = false;
let lastCat = 'NONE';
let lastDirWord = '';
let lastSpeakAt = 0;
let lastKnownState = null;

// ============================================================
// Public entry points
// ============================================================
window.navassist = window.navassist || {};
window.navassist.startCameraAI = startCameraAI;
window.navassist.setFrameRotation = (deg) => { rotateDeg = deg; };
window.navassist.handleCrossingResult = handleCrossingPerception;

window.addEventListener('load', () => {
    const splashBtn = document.getElementById('splashButton');
    const splash    = document.getElementById('splashScreen');
    const kickoff = () => startCameraAI();
    if (splashBtn) splashBtn.addEventListener('click', kickoff, { once: true });
    if (splash)    splash.addEventListener('touchstart', kickoff, { once: true, passive: true });
});

// ============================================================
// Bootstrap — canvas setup only (model runs on the server)
// ============================================================
function startCameraAI() {
    if (started) return;
    started = true;

    frameCanvas = document.getElementById('camFrame');
    overlay     = document.getElementById('camOverlay');
    if (!frameCanvas || !overlay) { started = false; return; }
    frameCtx = frameCanvas.getContext('2d');
    octx     = overlay.getContext('2d');

    crossOverlay = document.getElementById('crossOverlay');
    if (crossOverlay) {
        crossCtx = crossOverlay.getContext('2d');
        crossMaskCanvas = document.createElement('canvas');
        crossMaskCtx = crossMaskCanvas.getContext('2d');
    }

    modelStatus = 'waiting-pi';
    setContext('Waiting for glasses camera feed…');
    requestAnimationFrame(renderLoop);
}

// ============================================================
// Camera frame ingestion
// ============================================================
function handleCameraFrame(buf) {
    if (!buf || !buf.byteLength) return;
    if (decodeInFlight) {
        pendingFrame = buf;
        return;
    }
    decodeFrame(buf);
}

function decodeFrame(buf) {
    decodeInFlight = true;
    createImageBitmap(new Blob([buf], { type: 'image/jpeg' }))
        .then(drawFrame)
        .catch(() => {
            window.navassist.debugLog && window.navassist.debugLog('Bad frame: failed to decode JPEG');
        })
        .finally(onDecodeSettled);
}

function drawFrame(bmp) {
    const quarter = (rotateDeg === 90 || rotateDeg === 270);
    const cw = quarter ? bmp.height : bmp.width;
    const ch = quarter ? bmp.width  : bmp.height;
    if (frameCanvas.width !== cw || frameCanvas.height !== ch) {
        frameCanvas.width = overlay.width = cw;
        frameCanvas.height = overlay.height = ch;
        const cv = overlay.parentElement;
        if (cv && cw) cv.style.aspectRatio = cw + ' / ' + ch;
    }

    frameCtx.save();
    if (rotateDeg === 90)       { frameCtx.translate(cw, 0);  frameCtx.rotate(Math.PI / 2); }
    else if (rotateDeg === 180) { frameCtx.translate(cw, ch); frameCtx.rotate(Math.PI); }
    else if (rotateDeg === 270) { frameCtx.translate(0, ch);  frameCtx.rotate(-Math.PI / 2); }
    frameCtx.drawImage(bmp, 0, 0);
    frameCtx.restore();
    bmp.close();

    if (modelStatus === 'waiting-pi') modelStatus = 'ready';
}

function onDecodeSettled() {
    decodeInFlight = false;
    if (pendingFrame !== null) {
        const next = pendingFrame;
        pendingFrame = null;
        decodeFrame(next);
    }
}

// ============================================================
// Throttled speak (crossing.js-style)
// ============================================================
function speakThrottled(text, force) {
    const now = Date.now();
    if (!force && now - lastSpeakAt < SPEAK_COOLDOWN_MS) return false;
    lastSpeakAt = now;
    window.navassist.speak && window.navassist.speak(text);
    return true;
}

// ============================================================
// Server-side crossing perception — vision-driven FSM
// ============================================================
function handleCrossingPerception(result) {
    drawCrossingOverlay(result);

    const state = window.navassist.currentState && window.navassist.currentState();
    const STATES = window.navassist.STATES;
    if (!state || !STATES) return;

    // Reset temporal state on state transitions
    if (state !== lastKnownState) {
        if (state === STATES.WAITING || state === STATES.SCANNING) {
            redSince = null;
            hurried = false;
            lastCat = 'NONE';
            lastDirWord = '';
            noDashStreak = 0;
        }
        lastKnownState = state;
    }

    const frameW = result.w || 1;
    const frameH = result.h || 1;

    // --- SCANNING: detect the pedestrian light to advance to NAVIGATING ---
    if (state === STATES.SCANNING && result.light) {
        const lightCx = (result.light.box[0] + result.light.box[2]) / 2;
        const lightDir = getDirection(lightCx / frameW);
        window.navassist.onTrafficLightVisible &&
            window.navassist.onTrafficLightVisible(result.light.conf, lightDir);
        return;
    }

    // --- NAVIGATING: guide toward the pedestrian light, detect arrival ---
    if (state === STATES.NAVIGATING && result.light) {
        const lightCx = (result.light.box[0] + result.light.box[2]) / 2;
        const direction = getDirection(lightCx / frameW);

        const lightBoxH = (result.light.box[3] - result.light.box[1]) / frameH;
        if (lightBoxH > 0.10 || (result.signals && result.signals.corridorAhead)) {
            window.navassist.onArrived && window.navassist.onArrived();
        } else {
            window.navassist.onDirectionDecided &&
                window.navassist.onDirectionDecided(direction);
        }
        return;
    }

    // --- WAITING + CROSSING: crossing.js-style temporal logic ---
    // Uses independent green/red boolean signals from the server, NOT the
    // combined .state string. This matches crossing.js's decide() function.
    const light = result.light;
    const green = !!(light && light.green);
    const red = !!(light && light.red);
    const crossingSeen = !!(result.signals && result.signals.corridorAhead);
    const cat = green ? (red ? 'GREEN_RED' : 'GREEN_ONLY') : (red ? 'RED_ONLY' : 'NONE');
    const changed = cat !== lastCat;
    lastCat = cat;

    // Red timer: any green resets it; sustained RED-only for RED_HOLD_MS
    // means the light has switched to constant red (not just a blink-off).
    if (green) redSince = null;
    else if (cat === 'RED_ONLY' && redSince == null) redSince = performance.now();
    const constantRed = redSince != null && (performance.now() - redSince) >= RED_HOLD_MS;

    if (state === STATES.WAITING) {
        if (cat === 'GREEN_ONLY' && crossingSeen) {
            window.navassist.onGreenCross && window.navassist.onGreenCross();
        } else if (cat === 'GREEN_RED') {
            if (changed) speakThrottled('Green is flashing, wait for the next green.');
        } else if (cat === 'RED_ONLY') {
            if (changed) speakThrottled('Red man. Please wait.');
        }

        // Directional haptic toward the green man
        if (light && (cat === 'GREEN_ONLY' || cat === 'GREEN_RED')) {
            const lightCx = (light.box[0] + light.box[2]) / 2;
            const lightDir = getDirection(lightCx / frameW);
            window.navassist.onGreenDirection &&
                window.navassist.onGreenDirection(lightDir);
        }
    } else if (state === STATES.CROSSING) {
        // Corridor direction guidance (spoken + haptic)
        if (result.corridor && result.corridor.has) {
            noDashStreak = 0;
            const ang = result.corridor.angleDeg;
            let dirWord = 'straight ahead';
            if (ang > -65 && ang <= 0) dirWord = 'veer right';
            else if (ang < -115 && ang >= -180) dirWord = 'veer left';

            if (dirWord !== lastDirWord) {
                if (speakThrottled(dirWord, lastDirWord === '')) lastDirWord = dirWord;
            }

            // Haptic via app.js callback
            let corridorDir = 'CENTRE';
            if (ang > -65 && ang <= 0) corridorDir = 'RIGHT';
            else if (ang < -115 && ang >= -180) corridorDir = 'LEFT';
            window.navassist.onCorridorDirection &&
                window.navassist.onCorridorDirection(corridorDir);
        } else {
            noDashStreak++;
        }

        // Light switched to constant red mid-crossing
        if (constantRed && !hurried) {
            hurried = true;
            speakThrottled('The light is red. Hurry to finish crossing.', true);
        }

        // End detection: dashes ran out + light close/gone
        const lightBig = light && light.areaFrac >= END_LIGHT_AREA_FRAC;
        if (noDashStreak >= END_NO_DASH_FRAMES && (lightBig || !light)) {
            window.navassist.onVisionArrivalSignal &&
                window.navassist.onVisionArrivalSignal();
        }
    }
}

// ============================================================
// Crossing-model overlay drawing
// ============================================================
function drawCrossingOverlay(r) {
    if (!crossCtx || !r || !r.w) return;
    if (crossOverlay.width !== r.w || crossOverlay.height !== r.h) {
        crossOverlay.width = r.w; crossOverlay.height = r.h;
    }
    const W = crossOverlay.width, H = crossOverlay.height;
    const lw = Math.max(2, W * 0.004), fp = Math.max(13, Math.round(W * 0.022));
    crossCtx.clearRect(0, 0, W, H);

    if (r.dotted) for (const d of r.dotted) if (d.mask) drawCrossMask(d.mask);

    if (r.light && r.light.box) {
        const st = r.light.state;
        const col = st === 'GREEN' ? '#00e676' : st === 'RED' ? '#ff1744'
                  : st === 'GREENRED' ? '#ffab00' : '#9e9e9e';
        const b = r.light.box;
        crossCtx.lineWidth = lw * 1.3; crossCtx.strokeStyle = col;
        crossCtx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
        drawCrossTag('light ' + st, b[0], b[1], col, fp);
    }
    if (r.corridor && r.corridor.has) drawCrossArrow(r.corridor.near, r.corridor.far, '#00e5ff', W);

    lastCrossDrawAt = performance.now();
}

function drawCrossMask(m) {
    crossMaskCanvas.width = m.mw; crossMaskCanvas.height = m.mh;
    const n = m.mw * m.mh, bin = atob(m.data);
    const img = crossMaskCtx.createImageData(m.mw, m.mh), dt = img.data;
    for (let i = 0; i < n; i++) {
        if ((bin.charCodeAt(i >> 3) >> (i & 7)) & 1) {
            const o = i * 4; dt[o] = 0; dt[o + 1] = 224; dt[o + 2] = 255; dt[o + 3] = 125;
        }
    }
    crossMaskCtx.putImageData(img, 0, 0);
    const b = m.box;
    crossCtx.imageSmoothingEnabled = true;
    crossCtx.drawImage(crossMaskCanvas, b[0], b[1], b[2] - b[0], b[3] - b[1]);
}

function drawCrossArrow(a, b, color, W) {
    const head = Math.max(14, W * 0.03), ang = Math.atan2(b.y - a.y, b.x - a.x);
    crossCtx.strokeStyle = color; crossCtx.fillStyle = color;
    crossCtx.lineWidth = Math.max(4, W * 0.008); crossCtx.lineCap = 'round';
    crossCtx.beginPath(); crossCtx.moveTo(a.x, a.y); crossCtx.lineTo(b.x, b.y); crossCtx.stroke();
    crossCtx.beginPath(); crossCtx.moveTo(b.x, b.y);
    crossCtx.lineTo(b.x - head * Math.cos(ang - Math.PI / 6), b.y - head * Math.sin(ang - Math.PI / 6));
    crossCtx.lineTo(b.x - head * Math.cos(ang + Math.PI / 6), b.y - head * Math.sin(ang + Math.PI / 6));
    crossCtx.closePath(); crossCtx.fill();
    crossCtx.beginPath(); crossCtx.arc(a.x, a.y, Math.max(5, W * 0.01), 0, 6.283); crossCtx.fill();
}

function drawCrossTag(text, x, y, color, fp) {
    crossCtx.font = 'bold ' + fp + 'px sans-serif';
    const tw = crossCtx.measureText(text).width, ty = Math.max(y, fp + 6);
    crossCtx.fillStyle = color; crossCtx.fillRect(x, ty - fp - 6, tw + 10, fp + 6);
    crossCtx.fillStyle = '#000'; crossCtx.fillText(text, x + 5, ty - 6);
}

// ============================================================
// Render loop
// ============================================================
function renderLoop(ts) {
    if (!started) return;
    if (lastFrameTs) fps = 0.9 * fps + 0.1 * (1000 / (ts - lastFrameTs));
    lastFrameTs = ts;

    if (crossCtx && lastCrossDrawAt && performance.now() - lastCrossDrawAt > 1500) {
        crossCtx.clearRect(0, 0, crossOverlay.width, crossOverlay.height);
        lastCrossDrawAt = 0;
    }

    const W = overlay.width, H = overlay.height;
    if (W && H) {
        octx.clearRect(0, 0, W, H);
        const hudFont = Math.max(12, Math.round(W * 0.016));
        octx.font = hudFont + 'px monospace';
        octx.fillStyle = 'rgba(0,0,0,0.55)';
        const hud = modelStatus + '  ' + fps.toFixed(0) + ' fps';
        const hw = octx.measureText(hud).width;
        octx.fillRect(6, 6, hw + 12, 22);
        octx.fillStyle = '#0f0';
        octx.fillText(hud, 12, 22);
    }

    requestAnimationFrame(renderLoop);
}

// ============================================================
// Helpers
// ============================================================
function getDirection(normX) {
    if (normX < 0.4) return 'LEFT';
    if (normX > 0.6) return 'RIGHT';
    return 'CENTRE';
}

function setContext(text) {
    const el = document.getElementById('contextMessage');
    if (el) el.textContent = text;
}
