/**
 * NavAssist — crossing.js  (client)
 *
 * Shared by both pages (live Pi-camera + upload-video test). Grabs frames from
 * a <video>, POSTs them to the server (/api/infer, which runs the YOLO11-seg
 * model + perception), draws the overlay (dotted-line boxes, pedestrian-light +
 * state, corridor direction vector), and runs the stateful decision FSM:
 *
 * The pedestrian light is read as two INDEPENDENT signals per frame (from the
 * server): green-man present, and red present (red man OR the red countdown
 * numeral). The three meaningful readings:
 *   GREEN only  = constant WALK (no countdown)         -> cross
 *   GREEN + RED = clearance (flashing green + red count) -> if not started, wait
 *   RED only    = clearance OFF-blink OR constant red   -> treat as still-green
 *                 until red persists RED_HOLD (2s), then it's constant red.
 *
 *   WAITING  -> cross when GREEN-only + a crossing (dotted lines) is in view.
 *               GREEN+RED (clearance) or RED -> keep waiting for the next constant
 *               green. (Arriving mid-clearance => wait for the next turn.)
 *   CROSSING -> guide along the corridor. Any green (GREEN or GREEN+RED) resets a
 *               red timer; RED-only starts it. If red holds >= RED_HOLD (2s) the
 *               light has switched to constant red -> "hurry to finish crossing".
 *   DONE     -> reached the far side; returns to WAITING once the light is red.
 *
 * Exposes window.Crossing = { start(videoEl, overlayEl, onStatus), stop() }.
 */
(function () {
    const CFG = {
        INFER_MS: 150,            // min gap between inference requests (fast enough to sample the ~0.5s green blink)
        LOOP_MS: 40,
        JPEG_Q: 0.6,
        CAP_W: 960,               // downscale frame sent to the server (speed)
        LIGHT_CLOSE_AREA: 0.004,  // light this big (frac of frame) => trust its state
        END_LIGHT_AREA: 0.02,     // light this big => you're at the far side
        END_NO_DASH_FRAMES: 6,    // consecutive frames with no dashes ahead => corridor ran out
        SPEAK_COOLDOWN: 3500,
        // --- pedestrian-light timing (SG clearance blink = 500ms on / 500ms off) ---
        RED_HOLD_MS: 2000         // red-only (no green) sustained this long => constant red (light switched back).
                                  // Bridges a blink-off AND a missed on-phase (off 500 + missed-on 500 + off 500),
                                  // so a clearance blink never trips the constant-red timer.
    };
    const S = { WAITING: 'WAITING', CROSSING: 'CROSSING', DONE: 'DONE' };

    let video, overlay, octx, cap, capctx, maskCanvas, maskCtx, onStatus;
    let running = false, inFlight = false, lastInferAt = 0, latest = null;
    let state = S.WAITING, noDash = 0, lastSpeakAt = 0, lastDirWord = '';
    // light timing state: redSince = when the current RED-only run began (null once any green shows);
    // hurried = "red, hurry" already announced this crossing.
    let redSince = null, hurried = false, lastCat = 'NONE';

    // returns true if it actually spoke (false if throttled) so callers can retry
    function speak(text, force) {
        const now = Date.now();
        if (!force && now - lastSpeakAt < CFG.SPEAK_COOLDOWN) return false;
        lastSpeakAt = now;
        if (window.navassist && window.navassist.speak) window.navassist.speak(text);
        else if (window.speechSynthesis) { speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(text)); }
        return true;
    }
    function status(msg) { if (onStatus) onStatus(state, msg); }

    function start(videoEl, overlayEl, statusCb) {
        video = videoEl; overlay = overlayEl; onStatus = statusCb;
        octx = overlay.getContext('2d');
        cap = document.createElement('canvas'); capctx = cap.getContext('2d', { willReadFrequently: true });
        maskCanvas = document.createElement('canvas'); maskCtx = maskCanvas.getContext('2d');
        state = S.WAITING; noDash = 0; running = true;
        redSince = null; hurried = false; lastCat = 'NONE'; lastDirWord = ''; lastSpeakAt = 0;
        requestAnimationFrame(draw);
        loop();
    }
    function stop() { running = false; }

    // ---- inference loop (throttled, one request at a time) ----
    async function loop() {
        while (running) {
            const now = performance.now();
            if (!inFlight && video.readyState >= 2 && video.videoWidth && now - lastInferAt >= CFG.INFER_MS) {
                lastInferAt = now; inFlight = true;
                try { await runInfer(); } catch (e) { /* transient network/frame error */ }
                finally { inFlight = false; }
            }
            await new Promise(r => setTimeout(r, CFG.LOOP_MS));
        }
    }

    async function runInfer() {
        const vw = video.videoWidth, vh = video.videoHeight, s = Math.min(1, CFG.CAP_W / vw);
        cap.width = Math.round(vw * s); cap.height = Math.round(vh * s);
        capctx.drawImage(video, 0, 0, cap.width, cap.height);
        const dataUrl = cap.toDataURL('image/jpeg', CFG.JPEG_Q);
        const res = await fetch('/api/infer', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl })
        });
        const r = await res.json();
        if (r && r.w) { latest = r; decide(r, performance.now()); }
    }

    // ---- decision FSM (driven by the two independent light signals) ----
    function decide(r, t) {
        const light = r.light;
        const green = !!(light && light.green);      // green man present this frame
        const red = !!(light && light.red);          // red present (red man OR red countdown numeral)
        const close = !!(light && light.areaFrac >= CFG.LIGHT_CLOSE_AREA);
        const crossingSeen = !!(r.signals && r.signals.corridorAhead);
        const cat = green ? (red ? 'GREEN_RED' : 'GREEN_ONLY') : (red ? 'RED_ONLY' : 'NONE');
        const changed = cat !== lastCat; lastCat = cat;

        // Red timer: ANY green resets it; a run of RED-only advances it. Sustained
        // RED-only for RED_HOLD means the light has switched to constant red.
        if (green) redSince = null;
        else if (cat === 'RED_ONLY' && redSince == null) redSince = t;
        const constantRed = redSince != null && (t - redSince) >= CFG.RED_HOLD_MS;

        // WAITING vs CROSSING IS the "initial reading" distinction: green+red / red while
        // WAITING (haven't started) => wait; the SAME readings after we've started (CROSSING)
        // just keep us going. So in WAITING we start the moment we see constant green.
        if (state === S.WAITING) {
            if (cat === 'GREEN_ONLY' && close && crossingSeen) {             // constant green (no numeral) at a crossing => GO NOW
                state = S.CROSSING; hurried = false; noDash = 0;
                speak('Green man. You may cross now.', true);
                status('CROSS NOW');
            } else if (cat === 'GREEN_RED' && close) {                        // clearance (green + countdown) as initial => wait
                if (changed) speak('Green is flashing, wait for the next green.');
                status('waiting — clearance, wait for next green');
            } else if (cat === 'RED_ONLY' && close) {
                if (changed) speak('Red man. Please wait.');
                status('waiting — red');
            } else if (cat === 'GREEN_ONLY' && close) {
                status('green — looking for the crossing…');                 // constant green but no crossing detected yet
            } else {
                status('looking for the pedestrian light…');
            }
        } else if (state === S.CROSSING) {
            if (r.corridor && r.corridor.has) {
                noDash = 0;
                const word = dirWord(r.corridor, r.w);
                if (word && word !== lastDirWord) {
                    if (speak(word, lastDirWord === '')) lastDirWord = word;  // force the FIRST cue; commit only if spoken
                }
            } else { noDash++; }
            if (constantRed) {                                               // light switched to constant red mid-crossing
                if (!hurried) { hurried = true; speak('The light is red. Hurry to finish crossing.', true); }
                status('crossing — RED, hurry!');
            } else {
                status('crossing' + (lastDirWord ? ' — ' + lastDirWord : ''));
            }
            const lightBig = light && light.areaFrac >= CFG.END_LIGHT_AREA;
            if (noDash >= CFG.END_NO_DASH_FRAMES && (lightBig || !light)) {   // dashes ran out + light close/gone => far side
                state = S.DONE;
                speak('You have reached the other side. Use your cane to confirm the kerb.', true);
                status('reached the other side');
            }
        } else if (state === S.DONE) {                                       // end of crossing -> ready for the next turn
            if (cat === 'RED_ONLY' || constantRed) { state = S.WAITING; redSince = null; hurried = false; }
        }
    }

    // corridor heading -> spoken guidance
    function dirWord(cor, w) {
        const ang = cor.angleDeg;                       // ~ -90 = straight ahead (up)
        if (ang > -65 && ang <= 0) return 'veer right';
        if (ang < -115 && ang >= -180) return 'veer left';
        return 'straight ahead';
    }

    // ---- overlay drawing (every animation frame, from the latest result) ----
    function draw() {
        if (!running) return;
        requestAnimationFrame(draw);
        if (!latest) return;
        const r = latest;
        if (overlay.width !== r.w || overlay.height !== r.h) {
            overlay.width = r.w; overlay.height = r.h;
            const cv = overlay.parentElement; if (cv) cv.style.aspectRatio = r.w + ' / ' + r.h;
        }
        const W = overlay.width, H = overlay.height, lw = Math.max(2, W * 0.004), fp = Math.max(13, Math.round(W * 0.022));
        octx.clearRect(0, 0, W, H);

        for (const d of r.dotted) if (d.mask) drawMask(d.mask);   // the model's segmentation, not a box

        if (r.light) {
            const st = r.light.state;   // GREEN | GREENRED | RED | NONE
            const col = st === 'GREEN' ? '#00e676' : st === 'RED' ? '#ff1744' : st === 'GREENRED' ? '#ffab00' : '#9e9e9e';
            const b = r.light.box;
            octx.lineWidth = lw * 1.3; octx.strokeStyle = col; octx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
            tag(octx, 'light ' + st, b[0], b[1], col, fp);
        }
        if (r.corridor && r.corridor.has) arrow(octx, r.corridor.near, r.corridor.far, '#00e5ff', W);

        banner(octx, bannerText(), bannerColor(), W, H, fp);
    }

    function bannerText() {
        if (state === S.CROSSING) return hurried ? '🏃 RED — hurry across' : '🟢 CROSS — ' + (lastDirWord || 'straight ahead');
        if (state === S.DONE) return '✓ Reached the other side';
        if (lastCat === 'GREEN_ONLY') return '🟢 Green — go';
        if (lastCat === 'GREEN_RED') return '🟡 Clearance — wait for next green';
        if (lastCat === 'RED_ONLY') return '🔴 Wait';
        return 'Looking for the light…';
    }
    function bannerColor() {
        if (state === S.CROSSING) return hurried ? '#ff1744' : '#00c853';
        if (state === S.DONE) return '#2979ff';
        if (lastCat === 'GREEN_ONLY') return '#00c853';
        if (lastCat === 'GREEN_RED') return '#ffab00';
        if (lastCat === 'RED_ONLY') return '#ff1744';
        return '#555';
    }

    // Render a dotted-line SEGMENTATION mask: unpack the bit-packed bitmap into a
    // small canvas, then stretch it (smoothed) onto its box in overlay coords.
    function drawMask(m) {
        maskCanvas.width = m.mw; maskCanvas.height = m.mh;
        const n = m.mw * m.mh, bin = atob(m.data), img = maskCtx.createImageData(m.mw, m.mh), dt = img.data;
        for (let i = 0; i < n; i++) {
            if ((bin.charCodeAt(i >> 3) >> (i & 7)) & 1) {
                const o = i * 4; dt[o] = 0; dt[o + 1] = 224; dt[o + 2] = 255; dt[o + 3] = 125;
            }
        }
        maskCtx.putImageData(img, 0, 0);
        const b = m.box;
        octx.imageSmoothingEnabled = true;
        octx.drawImage(maskCanvas, b[0], b[1], b[2] - b[0], b[3] - b[1]);
    }

    function arrow(ctx, a, b, color, W) {
        const head = Math.max(14, W * 0.03), ang = Math.atan2(b.y - a.y, b.x - a.x);
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = Math.max(4, W * 0.008); ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(b.x, b.y);
        ctx.lineTo(b.x - head * Math.cos(ang - Math.PI / 6), b.y - head * Math.sin(ang - Math.PI / 6));
        ctx.lineTo(b.x - head * Math.cos(ang + Math.PI / 6), b.y - head * Math.sin(ang + Math.PI / 6));
        ctx.closePath(); ctx.fill();
        ctx.beginPath(); ctx.arc(a.x, a.y, Math.max(5, W * 0.01), 0, 6.283); ctx.fill();
    }
    function tag(ctx, text, x, y, color, fp) {
        ctx.font = 'bold ' + fp + 'px sans-serif'; const tw = ctx.measureText(text).width, ty = Math.max(y, fp + 6);
        ctx.fillStyle = color; ctx.fillRect(x, ty - fp - 6, tw + 10, fp + 6);
        ctx.fillStyle = '#000'; ctx.fillText(text, x + 5, ty - 6);
    }
    function banner(ctx, text, color, W, H, fp) {
        ctx.font = 'bold ' + Math.round(fp * 1.25) + 'px sans-serif';
        const tw = ctx.measureText(text).width, pad = 16, bw = tw + pad * 2, bh = fp * 2, bx = (W - bw) / 2, by = H - bh - 16;
        ctx.fillStyle = 'rgba(0,0,0,0.7)'; ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(bx, by, bw, bh, 12); else ctx.rect(bx, by, bw, bh);
        ctx.fill(); ctx.fillStyle = color; ctx.fillText(text, bx + pad, by + bh - fp * 0.75);
    }

    window.Crossing = { start, stop };
})();
