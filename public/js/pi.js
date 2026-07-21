/**
 * NavAssist — pi.js
 *
 * Real-time test bench for the RASPBERRY PI CAMERA, the counterpart to webcam.js
 * (which drives the same model off a laptop webcam via getUserMedia).
 *
 * ONE MODEL, ON THE SERVER: crossing_seg.onnx runs on the NODE SERVER, on a
 * worker thread, against the JPEG the Pi already sent it. We never upload a frame
 * back to the server; results just arrive on this socket as 'crossing/result'.
 * There is no in-browser model here anymore -- crossing.js owns the overlay + FSM
 * for that pushed result, and this file just does the camera plumbing around it:
 * decode each JPEG, rotate it upright, draw it, and pace/drop stale frames.
 *
 * The Pi Zero 2W is a camera and nothing else -- it has no realistic hope of
 * running the model at a useful frame rate, so it captures + JPEG-encodes and
 * ships the bytes.
 *
 * ROTATION: the camera is mounted sideways, so frames land 90 deg CCW of upright.
 * The server un-rotates them inside its sharp pipeline before inference (free --
 * it decodes there anyway), and we un-rotate them here on the canvas (free -- the
 * GPU does it). The server tells us the angle it used on connect, so the two
 * coordinate spaces can never silently disagree.
 */
(function () {

// ============================================================
// State
// ============================================================
let frameCanvas, fctx, detOverlay, octx, view;
let ws = null, wsRetry = null;
let rotateDeg = 90;           // authoritative value comes from the server on connect

// Frame decode: exactly one createImageBitmap in flight, newest frame wins.
// Without this, a camera that outsends our decode rate builds an unbounded
// backlog and the picture drifts steadily further behind reality.
let decodeBusy = false, pendingFrame = null;

let crossMs = 0;
let linkFps = 0, lastFrameAt = 0;
let frameW = 0, frameH = 0;
let piConnected = false;
let speechMuted = false, hudOn = true, crossOn = true;

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

    document.getElementById('muteChk').addEventListener('change', e => {
        speechMuted = e.target.checked;
        if (speechMuted && window.speechSynthesis) speechSynthesis.cancel();
    });
    document.getElementById('hudChk').addEventListener('change', e => { hudOn = e.target.checked; });
    document.getElementById('crossChk').addEventListener('change', e => {
        crossOn = e.target.checked;
        // Tell the SERVER to stop inferring, not just to stop drawing -- you are
        // probably viewing this page on the same laptop that runs the server, so
        // an idle crossing model still steals CPU unless the server backs off.
        sendConfig();
        if (!crossOn) {
            document.getElementById('crossOverlay').getContext('2d').clearRect(0, 0, 1e4, 1e4);
            document.getElementById('crossState').textContent = '';
        }
    });

    document.getElementById('hapLeft').addEventListener('click',  () => sendHaptic('left'));
    document.getElementById('hapRight').addEventListener('click', () => sendHaptic('right'));
    document.getElementById('hapBoth').addEventListener('click',  () => sendHaptic('both'));

    window.Crossing.startExternal(document.getElementById('crossOverlay'), (state, msg) => {
        document.getElementById('crossState').textContent = state + (msg ? ' — ' + msg : '');
    });
    connect();
    requestAnimationFrame(renderLoop);
});

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
        // a ~100KB string, which on the Pi's 2.4GHz-only WiFi is the throughput
        // ceiling.
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
// motor. These buttons send the same 'haptic/command' envelope app.js does, so
// they exercise the real path end to end: browser -> server (relays it) -> Pi
// (read_commands -> HapticController) -> motor.
function sendHaptic(motor) {
    const el = document.getElementById('hapticState');
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        el.textContent = 'Haptic: no server connection.';
        return;
    }
    if (!piConnected) {
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
        if (p.event === 'pi_disconnected') setPi(false, 'Pi camera not connected. Run navassist_pi_camera.py on the Pi.');
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
// Frame in -> decode -> rotate upright -> draw
// ============================================================
function onFrame(arrayBuf) {
    if (decodeBusy) { pendingFrame = arrayBuf; return; }   // newest wins, stale ones are dropped
    decodeBusy = true;

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
}

// ============================================================
// Overlay: HUD only (crossing.js draws the masks/light/arrow on #crossOverlay)
// ============================================================
function renderLoop() {
    requestAnimationFrame(renderLoop);
    if (!frameW) return;

    const W = detOverlay.width, H = detOverlay.height;
    octx.clearRect(0, 0, W, H);

    if (hudOn) {
        octx.font = Math.max(12, Math.round(W * 0.016)) + 'px monospace';
        const hud = (piConnected ? linkFps.toFixed(0) + ' fps' : 'no pi') + '  ' +
                    frameW + '×' + frameH + '  ' +
                    'seg ' + (crossOn ? crossMs.toFixed(0) + 'ms' : 'off');
        const hw = octx.measureText(hud).width;
        octx.fillStyle = 'rgba(0,0,0,0.6)';
        octx.fillRect(6, 6, hw + 12, 24);
        octx.fillStyle = '#0f0';
        octx.fillText(hud, 12, 23);
    }
}

})();
