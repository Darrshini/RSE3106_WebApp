/**
 * NavAssist — webcam.js
 *
 * Runs the project's ONE perception model (crossing_seg.onnx: dotted-line
 * crossing + pedestrian light with GREEN/RED/GREENRED state) live off the LAPTOP
 * WEBCAM, instead of off the Pi's WebSocket camera frames the way ai.js does.
 *
 * The model runs on the SERVER (/api/infer, onnxruntime-node). This page just
 * hands its <video> to js/crossing.js, which captures frames, POSTs them, and
 * draws the overlay + runs its own crossing FSM. There is deliberately NO in-
 * browser model here anymore, and no state machine of the app's own: no GPS, no
 * compass, no haptics, no WebSocket. This is a perception test bench for a PC.
 */
(function () {

let video, overlay, octx, statusEl;
let running = false, speechMuted = false, hudOn = true;
let fps = 0, lastTs = 0;

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
    video    = document.getElementById('feed');
    overlay  = document.getElementById('detOverlay');
    statusEl = document.getElementById('status');
    octx     = overlay.getContext('2d');

    document.getElementById('startBtn').addEventListener('click', start);
    document.getElementById('muteChk').addEventListener('change', e => {
        speechMuted = e.target.checked;
        if (speechMuted && window.speechSynthesis) speechSynthesis.cancel();
    });
    document.getElementById('hudChk').addEventListener('change', e => { hudOn = e.target.checked; });
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
    setStatus('Camera on. Point it at a pedestrian crossing / light.');

    // The crossing segmentation model (server-side /api/infer), drawn into its
    // own canvas by crossing.js. This is the whole perception pipeline now.
    if (window.Crossing) {
        window.Crossing.start(video, document.getElementById('crossOverlay'), (state, msg) => {
            document.getElementById('crossState').textContent = state + (msg ? ' — ' + msg : '');
        });
    }

    requestAnimationFrame(renderLoop);
}

// ============================================================
// Per-frame: size the HUD canvas, draw a small fps readout. All the perception
// visuals (masks, light box, corridor arrow, banner) are crossing.js's job.
// ============================================================
function renderLoop(ts) {
    if (!running) return;
    requestAnimationFrame(renderLoop);

    if (lastTs) fps = 0.9 * fps + 0.1 * (1000 / (ts - lastTs));
    lastTs = ts;

    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return;
    if (overlay.width !== vw || overlay.height !== vh) {
        overlay.width = vw; overlay.height = vh;
        const cv = overlay.parentElement;
        if (cv) cv.style.aspectRatio = vw + ' / ' + vh;
    }

    const W = overlay.width;
    octx.clearRect(0, 0, W, overlay.height);
    if (hudOn) {
        octx.font = Math.max(12, Math.round(W * 0.016)) + 'px monospace';
        const hud = 'crossing_seg (server)  ' + fps.toFixed(0) + ' fps';
        const hw = octx.measureText(hud).width;
        octx.fillStyle = 'rgba(0,0,0,0.6)';
        octx.fillRect(6, 6, hw + 12, 24);
        octx.fillStyle = '#0f0';
        octx.fillText(hud, 12, 23);
    }
}

function setStatus(msg) { if (statusEl) statusEl.textContent = msg; }

})();
