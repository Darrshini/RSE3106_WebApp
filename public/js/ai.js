/**
 * NavAssist — ai.js
 * OWNER: Kim Hyeonghu (adapted: Raspberry Pi camera frames instead of laptop webcam)
 *
 * Camera processing + AI inference + direction logic.
 *
 * Runs a locally-trained YOLO11 pedestrian-traffic-light model
 * (red / green / traffic-light). onnxruntime-web runs the model inside
 * js/inference.worker.js (a Web Worker), NOT on this thread, so inference
 * never freezes the camera feed / gestures / heartbeat. Video source is the
 * glasses camera (Pi Zero 2W + Camera Module v3), delivered as raw JPEG bytes
 * over the /live WebSocket, NOT the laptop/phone's own webcam.
 *
 * TWO MODELS, TWO MACHINES, ONE FRAME. Neither runs on the Pi -- a Zero 2W is
 * 4x Cortex-A53 @1GHz, where a YOLO11 pass takes seconds, not milliseconds:
 *   pedestrian.onnx    HERE, in the browser (js/inference.worker.js).
 *   crossing_seg.onnx  on the NODE SERVER, against the same JPEG the Pi already
 *                      sent it. We never upload a frame back -- the server has
 *                      it already, so results simply arrive on the socket as
 *                      'crossing/result' and app.js routes them to
 *                      handleCrossingResult() below.
 *
 * Flow:
 *   1. app.js calls handleCameraFrame(buf) with the raw JPEG bytes of every
 *      binary frame that arrives on /live.
 *   2. Each frame is decoded (off-thread, via createImageBitmap), rotated
 *      upright, drawn onto the #camFrame canvas, and -- throttled to
 *      INFER_EVERY_MS -- pre-processed and handed to the inference worker,
 *      which runs the model and returns detections.
 *   3. Detections are drawn on the #camOverlay canvas: bounding boxes,
 *      a direction arrow to the strongest GREEN light, and a status HUD.
 *   4. FSM callbacks (window.navassist.*) drive app.js's state machine and
 *      audio announcements.
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
const INFER_EVERY_MS = 150;   // MINIMUM gap between inferences. Inference runs in
                              // a Web Worker and inferBusy already prevents overlap,
                              // so this is just a floor to stop a fast device pegging
                              // a core at 100%. On a slow device where one inference
                              // takes longer than this, inferBusy paces it instead
                              // and this floor never bites. Watch the "…ms" in the
                              // HUD: that's the real per-inference cost, and the
                              // detection overlay lags the live video by roughly
                              // that much (plus this floor). If it's high, the fix
                              // is faster inference (WebGPU / smaller model), NOT a
                              // lower floor here.
const RED_SPEAK_COOLDOWN_MS = 6000;   // less frequent than green -- it's a "keep waiting" reminder, not new info

// Flashing green-man detection (see handleGuidance for the tracking logic).
// Window covers roughly the last 2-3 seconds of inference results; needs a
// few observed transitions before confidently calling it "flashing" rather
// than a single missed/noisy frame.
const FLASH_HISTORY_WINDOW = 12;
const FLASH_MIN_TRANSITIONS = 3;

// ============================================================
// State
// ============================================================
let worker = null;            // inference.worker.js — runs the model off the main thread
let workerReady = false;      // true once the worker has loaded the model
let inferBusy = false;        // true while the worker is mid-inference (one at a time)
let frameCanvas, overlay, frameCtx, octx;
let preCanvas, preCtx;        // offscreen canvas for letterbox pre-processing
// Crossing-model overlay: a THIRD canvas (#crossOverlay), below the pedestrian
// overlay, on which we draw the server's crossing_seg.onnx output -- the same
// visuals pi.html shows. Draw-only: app.js owns the decisions (see
// handleCrossingPerception). maskCanvas unpacks each bit-packed segmentation
// mask before it's stretched onto the overlay.
let crossOverlay, crossCtx, crossMaskCanvas, crossMaskCtx;
let lastCrossDrawAt = 0;      // performance.now() of the last crossing result drawn (for stale-clear)
let latestDetections = [];
let latestFrameImg = null;    // inference source: the frameCanvas AFTER rotation
let decodeInFlight = false;   // true while a frame is mid-decode (frame-drop guard)
let pendingFrame = null;      // newest frame that arrived while a decode was in flight

// The camera is mounted sideways on the glasses, so frames land 90° CCW of
// upright and every incoming frame is rotated back at the one point it enters
// (decodeFrame). The Pi deliberately does NOT rotate: libcamera's hardware
// transform can't do a quarter turn, so rotating there would mean a software
// re-encode and the loss of the hardware JPEG encoder -- the whole reason a Zero
// 2W keeps up at all. Here it's free (the GPU does it), and on the server it's
// free too (it decodes the JPEG anyway).
//
// The SERVER is the single source of truth for the angle: it sends its PI_ROTATE
// on connect and app.js hands it to setFrameRotation(). Both ends therefore
// always agree on which way is up, so the boxes can't silently land sideways.
let rotateDeg = 90;

// Server-side crossing perception -- dotted-line corridor + light-state reading
// via the YOLO11-seg model. Runs on the SERVER, on the frame the Pi already sent
// it, and arrives here as pushed 'crossing/result' messages (see
// handleCrossingResult). Independent of, and parallel to, the pedestrian.onnx
// inference above: it ADDS corridor guidance and a more accurate flashing-light
// read; it does not replace the traffic-light-post detection used in SCANNING.
let noDashStreak = 0;
const END_NO_DASH_FRAMES = 6;             // consecutive no-dash frames before treating it as "corridor ran out"
const END_LIGHT_AREA_FRAC = 0.02;         // light this big in frame => plausibly at the far side
let started = false;
let modelStatus = 'idle';
let lastRedSpeakAt = 0;
let greenFlashHistory = [];  // rolling true/false history of green detection, most recent last
let lastInferAt = 0;
let lastInferMs = 0;          // last inference wall-time (ms), for the HUD
let backend = '';             // execution provider in use: 'webgpu' or 'wasm'
let fps = 0, lastFrameTs = 0;

// onnxruntime-web runs inside inference.worker.js, not on this (main) thread.
// It's self-hosted under public/vendor/onnxruntime/ (NOT a CDN) so the model
// loads with no internet -- important when the glasses run on a laptop hotspot
// with no upstream link. That dir holds the WebGPU build, which bundles both
// the WebGPU (GPU) and WASM (CPU) execution providers. Path is passed to the
// worker in the 'load' message below.
const ORT_WASM_PATHS = 'vendor/onnxruntime/';

// ============================================================
// Public entry points
// ============================================================
window.navassist = window.navassist || {};
window.navassist.startCameraAI = startCameraAI;
// Called by app.js when the server reports the rotation it applies to Pi frames.
window.navassist.setFrameRotation = (deg) => { rotateDeg = deg; };
// Called by app.js for every 'crossing/result' the server pushes down /live.
window.navassist.handleCrossingResult = handleCrossingPerception;

// Kick off model loading on the first user gesture (keeps behaviour consistent
// with the splash-tap pattern the rest of the app uses; no getUserMedia needed
// anymore, but this still cleanly ties "start" to a tap).
window.addEventListener('load', () => {
    const splashBtn = document.getElementById('splashButton');
    const splash    = document.getElementById('splashScreen');
    const kickoff = () => startCameraAI();
    if (splashBtn) splashBtn.addEventListener('click', kickoff, { once: true });
    if (splash)    splash.addEventListener('touchstart', kickoff, { once: true, passive: true });
});

// ============================================================
// Model bootstrap (no camera acquisition here — frames come from the Pi)
// ============================================================
function startCameraAI() {
    if (started) return;
    started = true;

    frameCanvas = document.getElementById('camFrame');
    overlay     = document.getElementById('camOverlay');
    if (!frameCanvas || !overlay) { started = false; return; }
    frameCtx = frameCanvas.getContext('2d');
    octx     = overlay.getContext('2d');

    // Optional: only present on index.html. If it's missing (e.g. a page that
    // reuses ai.js without it) the crossing draw path simply no-ops.
    crossOverlay = document.getElementById('crossOverlay');
    if (crossOverlay) {
        crossCtx = crossOverlay.getContext('2d');
        crossMaskCanvas = document.createElement('canvas');
        crossMaskCtx = crossMaskCanvas.getContext('2d');
    }

    preCanvas = document.createElement('canvas');
    preCanvas.width = preCanvas.height = INPUT_SIZE;
    preCtx = preCanvas.getContext('2d', { willReadFrequently: true });

    modelStatus = 'loading-model';
    setContext('Loading detection model…');

    // Spin up the inference worker and have it load the model. The worker does
    // all the ONNX work; this thread only pre-processes frames and draws.
    worker = new Worker('js/inference.worker.js');
    worker.onmessage = handleWorkerMessage;
    worker.onerror = (e) => {
        modelStatus = 'model-error';
        setContext('Inference worker failed to start (' + (e.message || 'unknown error') + ').');
        window.navassist.debugLog && window.navassist.debugLog('Worker onerror: ' + (e.message || e.filename));
    };
    // Model path must be absolute — a relative URL would resolve against the
    // worker script's location (js/), not the app root.
    worker.postMessage({
        type: 'load',
        // Both must be absolute — a relative URL would resolve against the
        // worker script's location (js/), not the app root.
        modelUrl: new URL(MODEL_URL, location.href).href,
        wasmPaths: new URL(ORT_WASM_PATHS, location.href).href,
        numThreads: 1,
        // Prefer the GPU (WebGPU); fall back to CPU (WASM) on devices/browsers
        // that lack it. NOTE: WebGPU needs a secure context — it's only offered
        // over HTTPS or on localhost, so an http://<lan-ip> page will fall back
        // to WASM even on a WebGPU-capable phone.
        providers: ['webgpu', 'wasm']
    });
    // renderLoop starts once the worker signals 'ready' (see handleWorkerMessage).
}

// Messages coming back from inference.worker.js.
function handleWorkerMessage(e) {
    const msg = e.data;
    switch (msg.type) {
        case 'ready':
            workerReady = true;
            backend = msg.backend || '?';   // 'webgpu' or 'wasm' — shown in the HUD
            modelStatus = 'waiting-pi';
            setContext('Model ready (' + backend + '). Waiting for glasses camera feed…');
            requestAnimationFrame(renderLoop);   // draw overlay every frame (smooth)
            break;
        case 'result':
            // One inference finished: free the worker and act on the detections.
            inferBusy = false;
            lastInferMs = msg.inferMs || 0;
            latestDetections = msg.detections || [];
            handleGuidance(latestDetections);
            break;
        case 'error':
            inferBusy = false;
            if (!workerReady) {   // failed during model load, not a per-frame hiccup
                modelStatus = 'model-error';
                setContext('Model failed to load (' + msg.message + '). Is models/pedestrian.onnx present?');
            }
            window.navassist.debugLog && window.navassist.debugLog('Worker error: ' + msg.message);
            break;
        case 'log':
            window.navassist.debugLog && window.navassist.debugLog(msg.message);
            break;
    }
}

// ============================================================
// Camera frame ingestion — called by app.js for every binary frame on /live
// ============================================================
// `buf` is an ArrayBuffer of raw JPEG bytes, straight from the Pi, byte for
// byte. It is NOT base64: the old ESP32 protocol wrapped each frame in
// {"topic":"camera/image","payload":"<base64>"}, which inflates every frame 33%
// and makes both ends parse a ~100KB string per frame. On the Zero 2W's
// 2.4GHz-only WiFi that inflation was the actual throughput ceiling.
function handleCameraFrame(buf) {
    if (!buf || !buf.byteLength) return;

    // Frame-drop guard: only ever decode ONE frame at a time. If a decode is
    // already running, stash just the newest frame and drop everything that
    // arrived in between -- we only care about the latest. Without this, a
    // camera that sends frames faster than we can decode/draw them (the Pi at
    // 15fps does, unlike the old ~1fps ESP32-CAM) builds an ever-growing
    // backlog and the on-screen feed falls further and further behind reality.
    // That's the "laggy only through the webserver" symptom: the camera is
    // fine, the browser consumer just can't keep up. Dropping is always right
    // here -- a fresher frame is already on its way.
    if (decodeInFlight) {
        pendingFrame = buf;
        return;
    }
    decodeFrame(buf);
}

function decodeFrame(buf) {
    decodeInFlight = true;
    // createImageBitmap decodes OFF the main thread. The old path -- new Image()
    // with a data: URL -- forced a base64 round trip through a string and decoded
    // on the main thread, which is a real cost at 15fps.
    createImageBitmap(new Blob([buf], { type: 'image/jpeg' }))
        .then(drawFrame)
        .catch(() => {
            // A torn/partial JPEG. Skip it; another is right behind it.
            window.navassist.debugLog && window.navassist.debugLog('Bad frame: failed to decode JPEG');
        })
        .finally(onDecodeSettled);
}

// Rotate the frame upright here, at the single point it enters, so the display
// AND the browser worker agree with the server's coordinates (the server rotates
// the same frame by the same angle inside its own pipeline). A quarter turn
// swaps width and height.
function drawFrame(bmp) {
    const quarter = (rotateDeg === 90 || rotateDeg === 270);
    const cw = quarter ? bmp.height : bmp.width;    // display/canvas width
    const ch = quarter ? bmp.width  : bmp.height;   // display/canvas height
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
    bmp.close();                                    // release the decoded frame immediately

    latestFrameImg = frameCanvas;                   // rotated frame is the inference source
    if (modelStatus === 'waiting-pi') modelStatus = 'ready';
    maybeRunInference();                            // browser worker (pedestrian.onnx)
    // No crossing call here: the server already has this exact frame and runs
    // crossing_seg.onnx on it unprompted, pushing 'crossing/result' back down
    // /live. Re-encoding and uploading it would be a pointless round trip.
}

// After a decode finishes (success or failure), immediately pick up the newest
// frame that queued up while we were busy, if any. This keeps the display as
// fresh as possible while still never running two decodes concurrently.
function onDecodeSettled() {
    decodeInFlight = false;
    if (pendingFrame !== null) {
        const next = pendingFrame;
        pendingFrame = null;
        decodeFrame(next);
    }
}

// ============================================================
// Server-side crossing perception (dotted-line corridor + light state)
// ============================================================
// Nothing to call and nothing to upload: the server runs crossing_seg.onnx on
// the Pi frames it is already relaying, and pushes each result down /live.
// app.js routes them straight here via window.navassist.handleCrossingResult.
//
// Server-side perception is an ADDITION, not a dependency -- if the model is
// slow or the results stop arriving, the client-side pipeline (traffic-light-post
// detection, phone-compass drift correction) keeps working on its own.
//
// Routes server-side perception into app.js's existing, already
// safety-gated callbacks -- this file never decides or speaks on its own
// behalf, same pattern as every other detection path in this file.
function handleCrossingPerception(result) {
    // Draw the crossing model's output on every result, regardless of app state,
    // so the second model is visibly working the moment the Pi is streaming --
    // exactly like pi.html. This is purely a view; the decision/speech logic
    // below is unchanged and remains the single source of truth.
    drawCrossingOverlay(result);

    const state = window.navassist.currentState && window.navassist.currentState();
    const STATES = window.navassist.STATES;
    if (!state || !STATES) return;

    // --- Light state while WAITING: more accurate than blink-counting,
    // since it reads green/red presence independently per frame instead of
    // needing a rolling history window first. RED is intentionally left to
    // the existing pedestrian.onnx-driven "still red" reminder, so there's
    // only one code path speaking that reminder, not two.
    if (state === STATES.WAITING && result.light) {
        const frameW = result.w || 1;
        const lightDir = getDirection(((result.light.box[0] + result.light.box[2]) / 2) / frameW);
        if (result.light.state === 'GREEN') {
            window.navassist.onGreenDetected && window.navassist.onGreenDetected(lightDir);
        } else if (result.light.state === 'GREENRED') {
            window.navassist.onGreenFlashing && window.navassist.onGreenFlashing(lightDir);
        }
    }

    // --- Corridor (dotted-line) haptic guidance + vision-based arrival
    // signal, only relevant while actually crossing.
    if (state === STATES.CROSSING) {
        if (result.corridor && result.corridor.has) {
            noDashStreak = 0;
            // ang ~ -90 = straight ahead (up). Same threshold convention as
            // the original crossing.js dirWord() logic.
            const ang = result.corridor.angleDeg;
            let corridorDir = 'CENTRE';
            if (ang > -65 && ang <= 0) corridorDir = 'RIGHT';
            else if (ang < -115 && ang >= -180) corridorDir = 'LEFT';
            window.navassist.onCorridorDirection && window.navassist.onCorridorDirection(corridorDir);
        } else {
            noDashStreak++;
        }

        const lightBig = result.light && result.light.areaFrac >= END_LIGHT_AREA_FRAC;
        if (noDashStreak >= END_NO_DASH_FRAMES && (lightBig || !result.light)) {
            // Corridor ran out AND the light looks close/gone -- plausible
            // sign of reaching the far side. This is a SIGNAL, not a
            // decision: it feeds into the SAME tap-confirmation gate GPS
            // distance already triggers (promptArrivalConfirmation), it
            // does not complete the crossing on its own.
            window.navassist.onVisionArrivalSignal && window.navassist.onVisionArrivalSignal();
        }
    }
}

// ============================================================
// Crossing-model overlay (draw-only) — mirrors pi.html's crossing.js visuals
// ============================================================
// index.html previously consumed crossing/result for DECISIONS only and never
// drew it, so the page looked like it ran a single model. This renders the same
// three things pi.html does -- cyan dotted-line segmentation masks, the corridor
// arrow, and the pedestrian-light box coloured by state -- on the dedicated
// #crossOverlay layer (below the pedestrian boxes). It deliberately runs NO
// state machine and speaks nothing: app.js already owns every decision, so this
// duplicates only the drawing, not crossing.js's FSM. Coordinates are the
// server's upright r.w x r.h space, the SAME space the rotated #camFrame and the
// #camOverlay use, so all three layers line up (see the rotation notes above).
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
        const st = r.light.state;   // GREEN | GREENRED | RED | NONE
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

// Unpack a bit-packed segmentation bitmap into a small canvas, then stretch it
// (smoothed) onto its box in overlay coords. Ported from crossing.js drawMask().
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
// Inference (throttled, triggered per incoming frame)
// ============================================================
// Pre-process the current frame on this thread (it needs the DOM canvas) and
// ship the resulting float buffer to the worker, which runs the model and the
// box decode. At most one inference is in flight at a time (inferBusy) so a
// slow model can't build a queue -- newer frames just wait for the next slot.
function maybeRunInference() {
    if (!workerReady || !latestFrameImg || inferBusy) return;
    const now = performance.now();
    if (now - lastInferAt < INFER_EVERY_MS) return;
    lastInferAt = now;

    const pre = preprocess();
    inferBusy = true;
    // Transfer the buffer (zero-copy) rather than cloning it across threads.
    worker.postMessage({
        type: 'infer',
        data: pre.data.buffer,
        dims: [1, 3, INPUT_SIZE, INPUT_SIZE],
        scale: pre.scale, padX: pre.padX, padY: pre.padY,
        confThresh: CONF_THRESH, iouThresh: IOU_THRESH
    }, [pre.data.buffer]);
}

// Letterbox the current frame into a 640x640 CHW float buffer for the model.
function preprocess() {
    const iw = latestFrameImg.width, ih = latestFrameImg.height, S = INPUT_SIZE;
    const scale = Math.min(S / iw, S / ih);
    const nw = Math.round(iw * scale), nh = Math.round(ih * scale);
    const padX = Math.floor((S - nw) / 2), padY = Math.floor((S - nh) / 2);

    preCtx.fillStyle = 'rgb(114,114,114)';
    preCtx.fillRect(0, 0, S, S);
    preCtx.drawImage(latestFrameImg, padX, padY, nw, nh);

    const rgba = preCtx.getImageData(0, 0, S, S).data;
    const area = S * S;
    const data = new Float32Array(3 * area);
    for (let i = 0; i < area; i++) {
        data[i]            = rgba[i * 4]     / 255;  // R
        data[i + area]     = rgba[i * 4 + 1] / 255;  // G
        data[i + 2 * area] = rgba[i * 4 + 2] / 255;  // B
    }
    // The YOLO output decode + NMS live in the worker now (postprocess there).
    return { data, scale, padX, padY };
}

// ============================================================
// Rendering: boxes + green direction vector (every animation frame)
// ============================================================
function renderLoop(ts) {
    if (!started) return;
    if (lastFrameTs) fps = 0.9 * fps + 0.1 * (1000 / (ts - lastFrameTs));
    lastFrameTs = ts;
    drawOverlay(latestDetections);
    // Crossing results arrive only as fast as the server infers; if they stop
    // (Pi disconnected), clear the crossing layer once so a stale mask/arrow
    // doesn't sit frozen over a dead feed.
    if (crossCtx && lastCrossDrawAt && performance.now() - lastCrossDrawAt > 1500) {
        crossCtx.clearRect(0, 0, crossOverlay.width, crossOverlay.height);
        lastCrossDrawAt = 0;
    }
    requestAnimationFrame(renderLoop);
}

function drawOverlay(dets) {
    const W = overlay.width, H = overlay.height;
    if (!W || !H) return;
    octx.clearRect(0, 0, W, H);
    const lw = Math.max(2, W * 0.004);
    const fontPx = Math.max(14, Math.round(W * 0.022));

    // Bounding boxes
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

    // Direction vector to the strongest GREEN light
    const greens = dets.filter(d => CLASSES[d.cls] === 'green');
    if (greens.length) {
        const g = greens.reduce((a, b) => (a.score > b.score ? a : b));
        const gx = (g.x1 + g.x2) / 2, gy = (g.y1 + g.y2) / 2;
        const ox = W / 2, oy = H - lw;                 // origin: bottom-centre = the user
        drawArrow(ox, oy, gx, gy, '#00e5ff', W);
        const dir = getDirection(gx / W);
        drawBanner(`🟢 GREEN — GO ${dir}`, '#00c853', W, H, fontPx);
    }

    // Small status HUD (top-left)
    octx.font = `${Math.max(12, Math.round(W * 0.016))}px monospace`;
    octx.fillStyle = 'rgba(0,0,0,0.55)';
    const hud = `${modelStatus}  ${backend}  ${dets.length} det  ${fps.toFixed(0)} fps  ${lastInferMs.toFixed(0)}ms`;
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

    // Announce spotting the traffic-light post(s). Only meaningful while
    // SCANNING — app.js's callbacks guard on that state too, so this is
    // safe to call on every frame without extra cooldown logic; once it
    // transitions away from SCANNING, this block stops firing (state check
    // below naturally prevents re-triggering the choice prompt every frame).
    const posts = dets.filter(d => CLASSES[d.cls] === 'traffic-light');
    if (posts.length &&
        window.navassist.currentState &&
        window.navassist.currentState() === window.navassist.STATES.SCANNING) {

        if (posts.length > 1) {
            // Multiple posts in frame — disambiguate by direction so the
            // user knows there's more than one option, not just "a post".
            const dirs = posts
                .map(p => getDirection(((p.x1 + p.x2) / 2) / frameW))
                .filter((v, i, arr) => arr.indexOf(v) === i); // unique, order-preserving

            if (dirs.length === 2 && dirs.includes('LEFT') && dirs.includes('RIGHT')) {
                // Clean two-way split (one clearly left, one clearly right) --
                // let the user actually choose which one to head toward,
                // rather than silently auto-picking the higher-confidence one.
                window.navassist.onMultiplePostsChoice && window.navassist.onMultiplePostsChoice();
                return;
            }

            // Anything messier than a clean left/right split (3+ directions,
            // or multiple boxes clustered in the same direction bucket) --
            // fall back to the existing behaviour: announce it, then
            // auto-proceed with whichever detection scored highest. Trying
            // to build a 3+-way choice gesture isn't worth the complexity
            // for what should be a rare edge case.
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
            // dir === 'CENTRE' is left unannounced here — app.js's own
            // "Traffic light post detected" message (fired below) already
            // covers the straight-ahead case without sounding redundant.
        }

        const post = posts.reduce((a, b) => (a.score > b.score ? a : b));
        window.navassist.onTrafficLightVisible && window.navassist.onTrafficLightVisible(post.score);
    }

    const best = dets.reduce((a, b) => (a.score > b.score ? a : b));
    const cx = (best.x1 + best.x2) / 2;
    const direction = getDirection(cx / frameW);

    const greens = dets.filter(d => CLASSES[d.cls] === 'green');
    const greenDetectedThisFrame = greens.length > 0;

    // Flashing-light detection: a physically flashing green man produces
    // gaps in detection (off during the "off" phase of the blink) that a
    // steady green never does. Track the last N frames as a rolling
    // true/false history and count how many times it flips -- no new model
    // training needed, this is pure temporal pattern tracking on top of the
    // EXISTING 'green' class. Singapore crossings typically flash green for
    // the last ~10s before turning red (per LTA/GMCD research), rather than
    // showing a numeric countdown, so this covers the common case.
    greenFlashHistory.push(greenDetectedThisFrame);
    if (greenFlashHistory.length > FLASH_HISTORY_WINDOW) greenFlashHistory.shift();

    if (greens.length) {
        const g = greens.reduce((a, b) => (a.score > b.score ? a : b));
        const gdir = getDirection(((g.x1 + g.x2) / 2) / frameW);

        // Directional haptic: buzz the side the green man is on. Has its own
        // cooldown inside app.js, so it's safe to call every frame here.
        window.navassist.onGreenDirection && window.navassist.onGreenDirection(gdir);

        if (isGreenFlashing()) {
            // Signal is ending soon -- do NOT invite the user to cross.
            // app.js decides exactly what to say/do with this; ai.js only
            // reports the pattern it observed.
            window.navassist.onGreenFlashing && window.navassist.onGreenFlashing(gdir);
        } else {
            // Speech for "green man detected, you may cross" now lives
            // entirely in app.js's onGreenDetected -- previously this also
            // spoke here, causing two overlapping/back-to-back messages for
            // the same event. Direction is passed through so app.js can
            // build ONE message that includes both the direction and the
            // "double tap to confirm" instruction, instead of splitting
            // that across two speak() calls.
            window.navassist.onGreenDetected && window.navassist.onGreenDetected(gdir);
        }
    }

    // Red man: only relevant while WAITING (i.e. user already confirmed a
    // crossing and is standing there). Reassures periodically rather than
    // every frame -- this is a "still waiting" reminder, not new information,
    // and onGreenDetected() above is what actually ends the wait.
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

// Counts on/off transitions in the recent green-detection history. A
// physically flashing light produces multiple flips within a short window;
// a steady light produces zero or one (e.g. the initial detection itself).
// Requires a FULL window of data before returning true, so a partially-
// filled history (e.g. right after WAITING begins) can't false-positive.
function isGreenFlashing() {
    if (greenFlashHistory.length < FLASH_HISTORY_WINDOW) return false;
    let transitions = 0;
    for (let i = 1; i < greenFlashHistory.length; i++) {
        if (greenFlashHistory[i] !== greenFlashHistory[i - 1]) transitions++;
    }
    return transitions >= FLASH_MIN_TRANSITIONS;
}

function getDirection(normX) {
    if (normX < 0.4) return 'LEFT';
    if (normX > 0.6) return 'RIGHT';
    return 'CENTRE';
}

// ============================================================
// Helpers
// ============================================================
function setContext(text) {
    const el = document.getElementById('contextMessage');
    if (el) el.textContent = text;
}
