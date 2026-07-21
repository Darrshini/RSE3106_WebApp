/**
 * NavAssist — ai.js
 * OWNER: Kim Hyeonghu (adapted: Raspberry Pi camera frames instead of laptop webcam)
 *
 * Camera processing + direction logic.
 *
 * ONE MODEL NOW. The whole perception stack is crossing_seg.onnx (a YOLO11-seg
 * model, classes: 'dotted line', 'pedestrian light'), run on the NODE SERVER via
 * onnxruntime-node, on the same JPEG the Pi already sent it. It is a superset of
 * the old browser pedestrian.onnx: the 'pedestrian light' class is the crossing
 * signal (what SCANNING aims at), the server reads its GREEN/RED/GREENRED/NONE
 * state from the lit pixels, AND it segments the dotted-line crossing corridor.
 * So there is NO in-browser model and NO inference worker anymore -- the browser
 * never runs a neural net.
 *
 * We never upload a frame back: the server has it already, so results simply
 * arrive on the /live socket as 'crossing/result' and app.js routes them to
 * handleCrossingResult() below.
 *
 * Flow:
 *   1. app.js calls handleCameraFrame(buf) with the raw JPEG bytes of every
 *      binary frame that arrives on /live.
 *   2. Each frame is decoded (off-thread, via createImageBitmap), rotated
 *      upright, and drawn onto the #camFrame canvas.
 *   3. 'crossing/result' messages (pushed by the server for the same frame) are
 *      drawn on #crossOverlay (dotted-line masks, corridor arrow, light box) and
 *      folded into app.js's state machine via the window.navassist.* callbacks.
 */

// ============================================================
// Config
// ============================================================
const RED_SPEAK_COOLDOWN_MS = 6000;   // "keep waiting" reminder while a red light shows -- not new info, so infrequent

// ============================================================
// State
// ============================================================
let frameCanvas, overlay, frameCtx, octx;
// Crossing-model overlay: a THIRD canvas (#crossOverlay), below the HUD/banner
// overlay, on which we draw the server's crossing_seg.onnx output -- the same
// visuals pi.html shows. Draw-only: app.js owns the decisions (see
// handleCrossingPerception). maskCanvas unpacks each bit-packed segmentation
// mask before it's stretched onto the overlay.
let crossOverlay, crossCtx, crossMaskCanvas, crossMaskCtx;
let lastCrossDrawAt = 0;      // performance.now() of the last crossing result drawn (for stale-clear)
let latestCrossing = null;    // the most recent 'crossing/result' payload (for the HUD + banner)
let lastSegMs = 0;            // server-reported crossing inference time (ms), for the HUD
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

// End-of-crossing vision signal (dotted-line corridor runs out + light close/gone).
let noDashStreak = 0;
const END_NO_DASH_FRAMES = 6;             // consecutive no-dash frames before treating it as "corridor ran out"
const END_LIGHT_AREA_FRAC = 0.02;         // light this big in frame => plausibly at the far side
let started = false;
let modelStatus = 'idle';
let lastRedSpeakAt = 0;
let fps = 0, lastFrameTs = 0;

// ============================================================
// Public entry points
// ============================================================
window.navassist = window.navassist || {};
window.navassist.startCameraAI = startCameraAI;
// Called by app.js when the server reports the rotation it applies to Pi frames.
window.navassist.setFrameRotation = (deg) => { rotateDeg = deg; };
// Called by app.js for every 'crossing/result' the server pushes down /live.
window.navassist.handleCrossingResult = handleCrossingPerception;

// Kick off on the first user gesture (keeps behaviour consistent with the
// splash-tap pattern the rest of the app uses).
window.addEventListener('load', () => {
    const splashBtn = document.getElementById('splashButton');
    const splash    = document.getElementById('splashScreen');
    const kickoff = () => startCameraAI();
    if (splashBtn) splashBtn.addEventListener('click', kickoff, { once: true });
    if (splash)    splash.addEventListener('touchstart', kickoff, { once: true, passive: true });
});

// ============================================================
// Bootstrap (no camera acquisition, no model load -- frames + perception both
// come from the server)
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

    modelStatus = 'waiting-pi';
    setContext('Waiting for glasses camera feed…');
    requestAnimationFrame(renderLoop);   // draw the HUD/banner overlay every frame (smooth)
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
// AND the server's coordinates agree (the server rotates the same frame by the
// same angle inside its own pipeline). A quarter turn swaps width and height.
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

    if (modelStatus === 'waiting-pi') { modelStatus = 'live'; setContext('Glasses feed live.'); }
    // No inference here: the server already has this exact frame and runs
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
// Server-side perception — the ONE model's output, folded into the FSM
// ============================================================
// crossing_seg.onnx runs on the server against the Pi frames it is already
// relaying, and pushes each result down /live. app.js routes them straight here
// via window.navassist.handleCrossingResult.
//
// This is where EVERY perception decision the old pedestrian.onnx used to make
// now comes from, driven by result.light(s) (the pedestrian light + its state)
// and result.corridor (the dotted-line crossing):
//   SCANNING   -> spotting the pedestrian-light post(s), left/right choice
//   WAITING    -> green (cross) / green-flashing (wait) / red (keep waiting)
//   NAVIGATING -> approach direction toward the light + "arrived" when it's big
//   CROSSING   -> corridor drift guidance + vision-based arrival signal
// This file never decides or speaks on its own behalf beyond the red "keep
// waiting" reminder; every callback is state-gated inside app.js.
function handleCrossingPerception(result) {
    if (!result) return;
    latestCrossing = result;
    lastSegMs = result.inferMs || lastSegMs;

    // Draw the model's output on every result, regardless of app state, so the
    // perception is visibly working the moment the Pi is streaming -- exactly
    // like pi.html. This is purely a view; the decision logic below is the
    // single source of truth.
    drawCrossingOverlay(result);

    const state = window.navassist.currentState && window.navassist.currentState();
    const STATES = window.navassist.STATES;
    if (!state || !STATES) return;

    const frameW = result.w || 1;
    const light = result.light;
    const lights = result.lights || (light ? [light] : []);
    const lightDir = (l) => getDirection(((l.box[0] + l.box[2]) / 2) / frameW);

    // --- SCANNING: spotting the pedestrian-light post(s). The 'pedestrian
    // light' class is the crossing signal the user aims at -- the direct
    // replacement for pedestrian.onnx's old 'traffic-light' post. app.js's
    // callbacks guard on SCANNING too, so this is safe to call every frame.
    if (state === STATES.SCANNING && lights.length) {
        if (lights.length > 1) {
            const dirs = lights.map(lightDir).filter((v, i, a) => a.indexOf(v) === i);   // unique, order-preserving
            if (dirs.length === 2 && dirs.includes('LEFT') && dirs.includes('RIGHT')) {
                // Clean two-way split -- let the user choose which post to head
                // toward rather than silently auto-picking the higher-confidence one.
                window.navassist.onMultiplePostsChoice && window.navassist.onMultiplePostsChoice();
                return;
            }
            const desc = dirs.length > 1
                ? `Multiple pedestrian lights detected: ${dirs.join(' and ').toLowerCase()}.`
                : `Pedestrian lights detected to your ${dirs[0].toLowerCase()}.`;
            window.navassist.speak && window.navassist.speak(desc);
        } else {
            const dir = lightDir(lights[0]);
            if (dir !== 'CENTRE') {
                window.navassist.speak && window.navassist.speak(
                    `Pedestrian light detected to your ${dir.toLowerCase()}.`);
            }
            // dir === 'CENTRE' is left to app.js's own "detected" message below.
        }
        const best = lights.reduce((a, b) => (a.conf > b.conf ? a : b));
        window.navassist.onTrafficLightVisible && window.navassist.onTrafficLightVisible(best.conf);
    }

    // --- Green-man directional nudge. Fires while SCANNING or WAITING (app.js
    // gates it and owns the cooldown), so pointing at a green man off to one
    // side buzzes that side even before the GPS->confirm->navigate chain.
    if (light && light.state === 'GREEN') {
        window.navassist.onGreenDirection && window.navassist.onGreenDirection(lightDir(light));
    }

    // --- WAITING: read the light and either invite the cross, warn it's
    // ending, or reassure that it's still red. The server reports green and red
    // presence independently per frame, so we get the state directly instead of
    // needing a rolling blink-history window first:
    //   GREEN     = constant walk        -> you may cross
    //   GREENRED  = clearance (flashing)  -> wait for the next green
    //   RED       = don't walk            -> keep waiting
    if (state === STATES.WAITING && light) {
        if (light.state === 'GREEN') {
            window.navassist.onGreenDetected && window.navassist.onGreenDetected(lightDir(light));
        } else if (light.state === 'GREENRED') {
            window.navassist.onGreenFlashing && window.navassist.onGreenFlashing(lightDir(light));
        } else if (light.state === 'RED') {
            const now = Date.now();
            if (now - lastRedSpeakAt > RED_SPEAK_COOLDOWN_MS) {
                lastRedSpeakAt = now;
                window.navassist.speak && window.navassist.speak('Still red. Please continue to wait.');
            }
        }
    }

    // --- NAVIGATING: walk toward the pedestrian light. Direction from its box
    // centre; "arrived" once the box fills enough of the frame (the light is
    // close). Mirrors the old pedestrian.onnx approach logic, on the light box.
    if (state === STATES.NAVIGATING && light) {
        const boxH = (light.box[3] - light.box[1]) / (result.h || 1);
        if (boxH > 0.4) {
            window.navassist.onArrived && window.navassist.onArrived();
        } else {
            window.navassist.onDirectionDecided && window.navassist.onDirectionDecided(lightDir(light));
        }
    }

    // --- CROSSING: dotted-line corridor haptic guidance + vision-based arrival.
    if (state === STATES.CROSSING) {
        if (result.corridor && result.corridor.has) {
            noDashStreak = 0;
            // ang ~ -90 = straight ahead (up). Same threshold convention as crossing.js.
            const ang = result.corridor.angleDeg;
            let corridorDir = 'CENTRE';
            if (ang > -65 && ang <= 0) corridorDir = 'RIGHT';
            else if (ang < -115 && ang >= -180) corridorDir = 'LEFT';
            window.navassist.onCorridorDirection && window.navassist.onCorridorDirection(corridorDir);
        } else {
            noDashStreak++;
        }

        const lightBig = light && light.areaFrac >= END_LIGHT_AREA_FRAC;
        if (noDashStreak >= END_NO_DASH_FRAMES && (lightBig || !light)) {
            // Corridor ran out AND the light looks close/gone -- a plausible sign
            // of reaching the far side. A SIGNAL, not a decision: it feeds the
            // SAME tap-confirmation gate GPS distance triggers.
            window.navassist.onVisionArrivalSignal && window.navassist.onVisionArrivalSignal();
        }
    }
}

// ============================================================
// Crossing-model overlay (draw-only) — mirrors pi.html's crossing.js visuals
// ============================================================
// Renders the model's output on the dedicated #crossOverlay layer: cyan
// dotted-line segmentation masks, the corridor arrow, and the pedestrian-light
// box coloured by state. Coordinates are the server's upright r.w x r.h space,
// the SAME space the rotated #camFrame uses, so the layers line up.
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
// Rendering: HUD + green banner (every animation frame)
// ============================================================
function renderLoop(ts) {
    if (!started) return;
    if (lastFrameTs) fps = 0.9 * fps + 0.1 * (1000 / (ts - lastFrameTs));
    lastFrameTs = ts;
    drawOverlay();
    // Crossing results arrive only as fast as the server infers; if they stop
    // (Pi disconnected), clear the crossing layer once so a stale mask/arrow
    // doesn't sit frozen over a dead feed.
    if (crossCtx && lastCrossDrawAt && performance.now() - lastCrossDrawAt > 1500) {
        crossCtx.clearRect(0, 0, crossOverlay.width, crossOverlay.height);
        lastCrossDrawAt = 0;
        latestCrossing = null;
    }
    requestAnimationFrame(renderLoop);
}

// The pedestrian boxes + green arrow that used to live here are gone with the
// browser model; the crossing overlay (drawCrossingOverlay) draws the light box
// and corridor arrow. This layer now carries just the HUD and a GREEN-GO banner.
function drawOverlay() {
    const W = overlay.width, H = overlay.height;
    if (!W || !H) return;
    octx.clearRect(0, 0, W, H);
    const fontPx = Math.max(14, Math.round(W * 0.022));

    const light = latestCrossing && latestCrossing.light;
    if (light && light.state === 'GREEN') {
        const frameW = (latestCrossing.w || W);
        const dir = getDirection(((light.box[0] + light.box[2]) / 2) / frameW);
        drawBanner(`🟢 GREEN — GO ${dir}`, '#00c853', W, H, fontPx);
    }

    // Small status HUD (top-left)
    octx.font = `${Math.max(12, Math.round(W * 0.016))}px monospace`;
    octx.fillStyle = 'rgba(0,0,0,0.55)';
    const nLights = latestCrossing && latestCrossing.lights ? latestCrossing.lights.length : 0;
    const hud = `${modelStatus}  seg ${lastSegMs.toFixed(0)}ms  ${nLights} light  ${fps.toFixed(0)} fps`;
    const hw = octx.measureText(hud).width;
    octx.fillRect(6, 6, hw + 12, 22);
    octx.fillStyle = '#0f0';
    octx.fillText(hud, 12, 22);
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
