/**
 * NavAssist — pi.js
 *
 * Real-time test bench for the RASPBERRY PI CAMERA, the counterpart to webcam.js
 * (which drives the same two models off a laptop webcam via getUserMedia).
 *
 * The two models run in two different places, and neither runs on the Pi:
 *
 *   pedestrian.onnx   HERE, in js/inference.worker.js (unchanged, shared with
 *                     ai.js/webcam.js) -- WebGPU where available, else WASM.
 *   crossing_seg.onnx on the NODE SERVER, on a worker thread, against the JPEG
 *                     the Pi already sent it. We never upload a frame back to
 *                     the server; results just arrive on this socket.
 *
 * The Pi Zero 2W is a camera and nothing else. It has no realistic hope of
 * running either model at a useful frame rate, so it does the one thing its
 * hardware is good at -- capture and JPEG-encode -- and ships the bytes.
 *
 * WHY THE FRAMES ARRIVE AS A SOCKET OF JPEGs AND NOT A <video>:
 * there is no WebRTC/HLS path here, just discrete JPEGs relayed over a
 * WebSocket, so this file has to do by hand what a <video> element would
 * otherwise do for free -- decode, pace, and drop.
 *
 * ROTATION: the camera is mounted sideways, so frames land 90 deg CCW of upright.
 * The server un-rotates them inside its sharp pipeline before inference (free --
 * it decodes there anyway), and we un-rotate them here on the canvas (free -- the
 * GPU does it). So the server's coordinates and this canvas's coordinates are the
 * same upright space, and the two overlays line up. The server tells us the angle
 * it used on connect, so the two can never silently disagree.
 */
(function () {

// ============================================================
// Config -- mirrors ai.js/webcam.js so detections match the real app
// ============================================================
const MODEL_URL   = 'models/pedestrian.onnx';
const ORT_PATHS   = 'vendor/onnxruntime/';       // self-hosted, no CDN
const CLASSES     = ['red', 'green', 'traffic-light'];
const CLASS_COLOR = { red: '#ff1744', green: '#00c853', 'traffic-light': '#2979ff' };
const INPUT_SIZE  = 640;
const IOU_THRESH  = 0.45;
const INFER_EVERY_MS = 100;   // floor between inferences; inferBusy is what actually paces a slow device

let confThresh = 0.35;        // live-adjustable from the slider

// ============================================================
// State
// ============================================================
let frameCanvas, fctx, detOverlay, octx, preCanvas, preCtx, view;
let ws = null, wsRetry = null;
let rotateDeg = 90;           // authoritative value comes from the server on connect

// Frame decode: exactly one createImageBitmap in flight, newest frame wins.
// Without this, a camera that outsends our decode rate builds an unbounded
// backlog and the picture drifts steadily further behind reality -- the classic
// "the feed is laggy but only through the web app" symptom. Dropping stale
// frames is always right here: a newer one is already on its way.
let decodeBusy = false, pendingFrame = null;

let worker = null, workerReady = false, inferBusy = false;
let detections = [];
let backend = '', lastInferMs = 0, crossMs = 0;
let linkFps = 0, lastFrameAt = 0, lastInferAt = 0;
let frameW = 0, frameH = 0;
let piConnected = false;
let speechMuted = false, hudOn = true, pedOn = true, crossOn = true;

// crossing.js speaks through window.navassist.speak when it exists, so defining
// it here is also how the mute checkbox silences it without touching that file.
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
    frameCanvas = document.getElementById('camFrame');
    detOverlay  = document.getElementById('detOverlay');
    view        = frameCanvas.parentElement;
    fctx = frameCanvas.getContext('2d');
    octx = detOverlay.getContext('2d');

    preCanvas = document.createElement('canvas');
    preCanvas.width = preCanvas.height = INPUT_SIZE;
    preCtx = preCanvas.getContext('2d', { willReadFrequently: true });

    document.getElementById('muteChk').addEventListener('change', e => {
        speechMuted = e.target.checked;
        if (speechMuted && window.speechSynthesis) speechSynthesis.cancel();
    });
    document.getElementById('hudChk').addEventListener('change', e => { hudOn = e.target.checked; });
    document.getElementById('pedChk').addEventListener('change', e => {
        pedOn = e.target.checked;
        if (!pedOn) detections = [];
    });
    document.getElementById('crossChk').addEventListener('change', e => {
        crossOn = e.target.checked;
        // Tell the SERVER to stop inferring, not just to stop drawing. The two
        // models are on different machines now, but you are probably viewing this
        // page on the same laptop that runs the server -- so an idle crossing
        // model still steals CPU from pedestrian.onnx unless the server backs off.
        sendConfig();
        if (!crossOn) {
            document.getElementById('crossOverlay').getContext('2d')
                .clearRect(0, 0, 1e4, 1e4);
            document.getElementById('crossState').textContent = '';
        }
    });
    document.getElementById('confRange').addEventListener('input', e => {
        confThresh = Number(e.target.value) / 100;
        document.getElementById('confVal').textContent = confThresh.toFixed(2);
    });

    document.getElementById('hapLeft').addEventListener('click',  () => sendHaptic('left'));
    document.getElementById('hapRight').addEventListener('click', () => sendHaptic('right'));
    document.getElementById('hapBoth').addEventListener('click',  () => sendHaptic('both'));

    startWorker();
    window.Crossing.startExternal(document.getElementById('crossOverlay'), (state, msg) => {
        document.getElementById('crossState').textContent = state + (msg ? ' — ' + msg : '');
    });
    connect();
    requestAnimationFrame(renderLoop);
});

// ============================================================
// pedestrian.onnx -- same worker file ai.js and webcam.js use
// ============================================================
function startWorker() {
    worker = new Worker('js/inference.worker.js');
    worker.onmessage = onWorkerMessage;
    worker.onerror = (e) => setStatus('Worker failed to start: ' + (e.message || 'unknown'));
    worker.postMessage({
        type: 'load',
        // Both URLs must be absolute: a relative one would resolve against the
        // worker script's own location (js/), not the app root.
        modelUrl:  new URL(MODEL_URL, location.href).href,
        wasmPaths: new URL(ORT_PATHS, location.href).href,
        numThreads: 1,
        providers: ['webgpu', 'wasm']   // WebGPU needs a secure context; localhost is one
    });
}

function onWorkerMessage(e) {
    const msg = e.data;
    if (msg.type === 'ready') {
        workerReady = true;
        backend = msg.backend || '?';
        setStatus('pedestrian.onnx ready on ' + backend.toUpperCase() + '.');
    } else if (msg.type === 'result') {
        inferBusy = false;
        lastInferMs = msg.inferMs || 0;
        detections = pedOn ? (msg.detections || []) : [];
    } else if (msg.type === 'error') {
        inferBusy = false;
        if (!workerReady) setStatus('pedestrian.onnx failed to load: ' + msg.message);
        console.warn('[worker]', msg.message);
    } else if (msg.type === 'log') {
        console.log('[worker]', msg.message);
    }
}

// ============================================================
// The /live socket: JPEG frames in (binary), crossing results in (text)
// ============================================================
function connect() {
    clearTimeout(wsRetry);
    const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/live';
    ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => sendConfig();

    ws.onmessage = (ev) => {
        // Binary => a JPEG frame straight from the Pi, byte-for-byte. Text =>
        // JSON (crossing results, connection events). Frames are deliberately NOT
        // base64-in-JSON: that inflates every frame 33% and makes both ends parse
        // a ~100KB string, which on the Pi's 2.4GHz-only WiFi is the actual
        // throughput ceiling.
        if (typeof ev.data === 'string') {
            let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
            handleJson(m);
        } else {
            onFrame(ev.data);
        }
    };

    ws.onclose = () => {
        setPi(false, 'Server disconnected. Reconnecting…');
        wsRetry = setTimeout(connect, 2000);
    };
    ws.onerror = () => ws.close();
}

function sendConfig() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ topic: 'live/config', payload: { crossing: crossOn } }));
}

// ============================================================
// Manual haptic test
// ============================================================
// This page has NO state machine, so unlike app.js it never decides to buzz
// anything by itself -- detecting a light or a direction here will never move a
// motor, which is exactly the thing that looks like a broken haptic and isn't.
// These buttons send the same 'haptic/command' envelope app.js does, so they
// exercise the real path end to end: browser -> server (relays it) -> Pi
// (read_commands -> HapticDriver) -> motor.
function sendHaptic(motor) {
    const el = document.getElementById('hapticState');
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        el.textContent = 'Haptic: no server connection.';
        return;
    }
    if (!piConnected) {
        // Worth calling out rather than sending into the void: the server drops
        // the command and warns, but from this page it would just look like a
        // dead motor.
        el.textContent = 'Haptic: Pi is not connected — the server has nowhere to send it.';
        return;
    }
    ws.send(JSON.stringify({
        topic: 'haptic/command',
        timestamp: Date.now(),
        payload: { motor, pattern: 'pulse', intensity: 0.8, duration_ms: 400 },
    }));
    el.textContent = `Haptic: sent ${motor} @0.8 for 400ms — expect "[cmd] haptic:" on the Pi.`;
    console.log('[NavAssist] haptic/command ->', motor);
}

function handleJson(m) {
    if (m.topic === 'connection/event') {
        const p = m.payload || {};
        if (typeof p.rotate === 'number') rotateDeg = p.rotate;   // server is the source of truth
        if (p.event === 'pi_connected')  setPi(true,  'Pi camera streaming.');
        if (p.event === 'pi_disconnected') {
            setPi(false, 'Pi camera not connected. Run navassist_pi_camera.py on the Pi.');
            detections = [];
        }
    } else if (m.topic === 'crossing/result') {
        crossMs = (m.payload && m.payload.inferMs) || 0;
        if (crossOn) window.Crossing.push(m.payload);
    } else if (m.topic === 'system/heartbeat') {
        setPi(true, 'Pi camera streaming.');
    }
}

function setPi(on, msg) {
    piConnected = on;
    document.getElementById('piDot').classList.toggle('on', on);
    document.getElementById('piState').textContent = msg;
}

// ============================================================
// Frame in -> decode -> rotate upright -> draw -> feed the model
// ============================================================
function onFrame(arrayBuf) {
    if (decodeBusy) { pendingFrame = arrayBuf; return; }   // newest wins, stale ones are dropped
    decodeBusy = true;

    // createImageBitmap decodes OFF the main thread. The obvious alternative --
    // new Image() with a data: URL, the way ai.js does it -- forces a base64
    // round trip through a string and decodes on the main thread, which is a real
    // cost at 15fps.
    createImageBitmap(new Blob([arrayBuf], { type: 'image/jpeg' }))
        .then(drawFrame)
        .catch(() => { /* a torn/partial JPEG: skip it, another is coming */ })
        .finally(() => {
            decodeBusy = false;
            if (pendingFrame) { const next = pendingFrame; pendingFrame = null; onFrame(next); }
        });
}

function drawFrame(bmp) {
    const now = performance.now();
    if (lastFrameAt) linkFps = 0.9 * linkFps + 0.1 * (1000 / (now - lastFrameAt));
    lastFrameAt = now;

    // Un-rotate to upright. A quarter turn swaps width and height.
    const quarter = (rotateDeg === 90 || rotateDeg === 270);
    const cw = quarter ? bmp.height : bmp.width;
    const ch = quarter ? bmp.width  : bmp.height;

    if (frameCanvas.width !== cw || frameCanvas.height !== ch) {
        frameCanvas.width = detOverlay.width = cw;
        frameCanvas.height = detOverlay.height = ch;
        if (view) view.style.aspectRatio = cw + ' / ' + ch;
        frameW = cw; frameH = ch;
    }

    fctx.save();
    if (rotateDeg === 90)       { fctx.translate(cw, 0);  fctx.rotate(Math.PI / 2); }
    else if (rotateDeg === 180) { fctx.translate(cw, ch); fctx.rotate(Math.PI); }
    else if (rotateDeg === 270) { fctx.translate(0, ch);  fctx.rotate(-Math.PI / 2); }
    fctx.drawImage(bmp, 0, 0);
    fctx.restore();
    bmp.close();                                   // release the decoded frame immediately

    maybeInfer();
}

// ============================================================
// pedestrian.onnx inference, throttled two ways
// ============================================================
function maybeInfer() {
    if (!pedOn || !workerReady || inferBusy || !frameW) return;
    const now = performance.now();
    if (now - lastInferAt < INFER_EVERY_MS) return;   // floor, so a fast machine doesn't peg a core
    lastInferAt = now;

    const pre = preprocess();
    inferBusy = true;                                 // hard guarantee: one inference at a time
    worker.postMessage({
        type: 'infer',
        data: pre.data.buffer,                        // transferred, not cloned
        dims: [1, 3, INPUT_SIZE, INPUT_SIZE],
        scale: pre.scale, padX: pre.padX, padY: pre.padY,
        confThresh, iouThresh: IOU_THRESH
    }, [pre.data.buffer]);
}

// Letterbox the current (already upright) frame into a 640x640 CHW float buffer.
// Same as ai.js/webcam.js preprocess(); the source is frameCanvas.
function preprocess() {
    const iw = frameW, ih = frameH, S = INPUT_SIZE;
    const scale = Math.min(S / iw, S / ih);
    const nw = Math.round(iw * scale), nh = Math.round(ih * scale);
    const padX = Math.floor((S - nw) / 2), padY = Math.floor((S - nh) / 2);

    preCtx.fillStyle = 'rgb(114,114,114)';            // grey pad, matching Ultralytics
    preCtx.fillRect(0, 0, S, S);
    preCtx.drawImage(frameCanvas, padX, padY, nw, nh);

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
// Overlay: boxes + arrow to the strongest green man + HUD
// ============================================================
function renderLoop() {
    requestAnimationFrame(renderLoop);
    if (!frameW) return;

    const W = detOverlay.width, H = detOverlay.height;
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
        arrow(W / 2, H - lw, (g.x1 + g.x2) / 2, (g.y1 + g.y2) / 2, '#00e5ff', W);
    }

    if (hudOn) {
        octx.font = Math.max(12, Math.round(W * 0.016)) + 'px monospace';
        const hud = (piConnected ? linkFps.toFixed(0) + ' fps' : 'no pi') + '  ' +
                    frameW + '×' + frameH + '  ' +
                    (workerReady ? backend + ' ' + lastInferMs.toFixed(0) + 'ms' : 'loading') + '  ' +
                    'seg ' + (crossOn ? crossMs.toFixed(0) + 'ms' : 'off') + '  ' +
                    detections.length + ' det';
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

function setStatus(msg) {
    const el = document.getElementById('status');
    if (el) el.textContent = msg;
}

})();
