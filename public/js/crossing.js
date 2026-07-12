/**
 * NavAssist — crossing.js  (client)
 *
 * Shared by both pages (live Pi-camera + upload-video test). Grabs frames from
 * a <video>, POSTs them to the server (/api/infer, which runs the YOLO11-seg
 * model + perception), draws the overlay (dotted-line boxes, pedestrian-light +
 * state, corridor direction vector), and runs the stateful decision FSM:
 *
 *   WAITING  -> if a crossing (dotted lines) is seen AND the green man is STEADY
 *               (non-blinking) -> cross now. In Singapore the green man is solid
 *               for the first seconds after red, THEN blinks 500ms on/500ms off
 *               during the clearance countdown. If it is already FLASHING before
 *               you start -> do NOT start; wait for the next steady green. RED is
 *               announced only after green has been ABSENT a sustained time, so an
 *               off-blink never reads as RED. smoothLight() tells the three apart.
 *   CROSSING -> guide along the corridor; announce "reached the other side" when
 *               the dashes run out ahead AND the light is close/gone (far side).
 *   DONE     -> returns to WAITING when a red is seen again.
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
        // --- blinking green man smoothing (SG clearance blink = 500ms on / 500ms off) ---
        GREEN_HOLD_MS: 2000,      // RED only after green is ABSENT this long. Must exceed one whole blink cycle PLUS a
                                  // missed on-phase (off 500 + missed-on 500 + off 500 = 1500ms) so a blink -- even with
                                  // an undetected on-phase -- never reads as RED. Green gone 2s straight => truly red.
        MIN_BLINK_OFF_MS: 300,    // an observed off sustained this long = a real blink (=> FLASHING), not 1-frame noise
        MAX_ON_GAP_MS: 400,       // trust green as continuous only across sample-gaps below this. Must sit ABOVE the
                                  // real frame gap (~150-350ms) yet BELOW the physical off-phase (500ms), so a real
                                  // blink-off can never hide unobserved inside a "continuous" run.
        STEADY_CONFIRM_MS: 1100,  // green must be CONTINUOUSLY on this long to count as STEADY (crossable) green
                                  // (> a full 1s blink cycle, so a flash-on can never masquerade as steady)
        HIST_MS: 3000             // timestamped-sample history window the smoother recomputes from each frame
    };
    const S = { WAITING: 'WAITING', CROSSING: 'CROSSING', DONE: 'DONE' };

    let video, overlay, octx, cap, capctx, maskCanvas, maskCtx, onStatus;
    let running = false, inFlight = false, lastInferAt = 0, latest = null;
    let state = S.WAITING, noDash = 0, lastSpeakAt = 0, lastDirWord = '';
    // temporal light smoother: a rolling buffer of {t, g} samples (see smoothLight)
    let smSamples = [], lastLightEff = 'UNKNOWN';

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
        smSamples = []; lastLightEff = 'UNKNOWN'; lastDirWord = ''; lastSpeakAt = 0;
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

    // ---- temporal light smoother (blinking green man) ----
    // The server classifies each FRAME independently; blinking + irregular sampling
    // (the effective frame rate is bounded by inference latency, not INFER_MS) are
    // temporal. We keep a short timestamped history and RECOMPUTE the effective
    // state from scratch every frame -- no persistent latches, so per-frame noise
    // can't corrupt a whole episode. Returns:
    //   'GREEN'     steady, crossable  (green CONTINUOUSLY observed on >= STEADY_CONFIRM)
    //   'GREEN_NEW' green on/held but not yet confirmed steady
    //   'FLASHING'  clearance blink   (a sustained off observed within the green episode)
    //   'RED' | 'UNKNOWN'
    // Safety: GREEN requires truly-continuous on-time -- ANY observed off resets it,
    // so a ~0.5s flash-on can never reach STEADY_CONFIRM => never a mid-clearance cross.
    // A lone green frame (needs >=2) never opens an episode, so noise can't mask RED.
    function smoothLight(light, t) {
        const greenNow = !!(light && light.state === 'GREEN');
        smSamples.push({ t, g: greenNow });
        while (smSamples.length && t - smSamples[0].t > CFG.HIST_MS) smSamples.shift();

        let lastG = -1; for (const s of smSamples) if (s.g) lastG = s.t;
        if (lastG < 0 || (t - lastG) > CFG.GREEN_HOLD_MS) {       // green truly gone -> end the episode
            smSamples = [{ t, g: greenNow }];                    // clear so the next green starts a FRESH episode
            return (light && light.state === 'RED') ? 'RED' : 'UNKNOWN';
        }
        const greens = smSamples.filter(s => s.g);
        if (greens.length < 2) return (light && light.state === 'RED') ? 'RED' : 'UNKNOWN';   // 1 lone green = noise

        // Continuous-on run: contiguous green samples, but ONLY credit a gap between
        // consecutive samples if it is smaller than MIN_BLINK_OFF -- otherwise a real
        // off could have hidden unobserved in that gap (coarse/aliased sampling), so we
        // must NOT assume the green was continuous across it. Consequence: if the frame
        // rate is too slow to catch a blink-off, onRun never reaches STEADY_CONFIRM and
        // we stay GREEN_NEW (no cross) rather than risk a false cross during clearance.
        let onRun = 0;
        if (greenNow) {
            let start = t, prev = t;
            for (let i = smSamples.length - 1; i >= 0 && smSamples[i].g; i--) {
                if (prev - smSamples[i].t > CFG.MAX_ON_GAP_MS) break;   // gap too big -> a blink-off may hide here
                start = smSamples[i].t; prev = smSamples[i].t;
            }
            onRun = t - start;
        }

        // longest OFF run observed within this episode (after the first green)
        const firstG = greens[0].t;
        let maxOff = 0, offStart = -1;
        for (const s of smSamples) {
            if (s.t < firstG) continue;
            if (s.g) offStart = -1;
            else { if (offStart < 0) offStart = s.t; if (s.t - offStart > maxOff) maxOff = s.t - offStart; }
        }
        if (!greenNow && offStart >= 0 && t - offStart > maxOff) maxOff = t - offStart;   // extend a trailing off to now

        if (greenNow && onRun >= CFG.STEADY_CONFIRM_MS) return 'GREEN';   // steady wins (safety: needs full on-run)
        if (maxOff >= CFG.MIN_BLINK_OFF_MS) return 'FLASHING';            // a real off happened => clearance blink
        return 'GREEN_NEW';
    }

    // ---- decision FSM ----
    function decide(r, now) {
        const light = r.light;
        const eff = smoothLight(light, now);
        const close = !!(light && light.areaFrac >= CFG.LIGHT_CLOSE_AREA);
        const changed = eff !== lastLightEff;
        lastLightEff = eff;

        const crossingSeen = !!(r.signals && r.signals.corridorAhead);   // a pedestrian crossing (dotted lines) is in view

        if (state === S.WAITING) {
            if (eff === 'GREEN' && close && crossingSeen) {          // STEADY (non-blinking) green at a crossing => go
                state = S.CROSSING; noDash = 0;
                speak('Green man. You may cross now.', true);
                status('CROSS NOW');
            } else if (eff === 'FLASHING' && close) {                // already blinking before we started => do NOT start
                if (changed) speak('Green is flashing, do not start. Wait for the next green.');
                status('waiting — flashing, do not start');
            } else if (eff === 'RED' && close) {
                if (changed) speak('Red man. Please wait.');
                status('waiting — red');
            } else if (eff === 'GREEN' && close) {                   // steady green but no crossing detected (yet)
                status('green — looking for the crossing…');
            } else if (eff === 'GREEN_NEW' && close) {
                status('green — confirming it is steady…');          // just came on; wait until steady-confirmed
            }
        } else if (state === S.CROSSING) {
            if (r.corridor && r.corridor.has) {
                noDash = 0;
                const word = dirWord(r.corridor, r.w);
                if (word && word !== lastDirWord) {
                    if (speak(word, lastDirWord === '')) lastDirWord = word;   // force FIRST cue; commit only when actually spoken
                }
                status('crossing — ' + word);
            } else {
                noDash++;
                status('crossing');
            }
            const lightBig = light && light.areaFrac >= CFG.END_LIGHT_AREA;
            if (noDash >= CFG.END_NO_DASH_FRAMES && (lightBig || !light)) {
                state = S.DONE; speak('You have reached the other side. Use your cane to confirm the kerb.', true); status('reached the other side');
            }
        } else if (state === S.DONE) {
            if (eff === 'RED' && close) { state = S.WAITING; }       // ready for the next crossing
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
            const b = r.light.box, col = r.light.state === 'GREEN' ? '#00e676' : r.light.state === 'RED' ? '#ff1744' : '#ffab00';
            octx.lineWidth = lw * 1.3; octx.strokeStyle = col; octx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
            tag(octx, 'light ' + r.light.state, b[0], b[1], col, fp);
        }
        if (r.corridor && r.corridor.has) arrow(octx, r.corridor.near, r.corridor.far, '#00e5ff', W);

        banner(octx, bannerText(), bannerColor(), W, H, fp);
    }

    function bannerText() {
        if (state === S.CROSSING) return '🟢 CROSS — ' + (lastDirWord || 'straight ahead');
        if (state === S.DONE) return '✓ Reached the other side';
        if (lastLightEff === 'GREEN') return '🟢 Green — steady';
        if (lastLightEff === 'GREEN_NEW') return '🟢 Green — confirming…';
        if (lastLightEff === 'FLASHING') return '🟡 Flashing — don’t start';
        if (lastLightEff === 'RED') return '🔴 Wait';
        return 'Looking for the crossing…';
    }
    function bannerColor() {
        if (state === S.CROSSING) return '#00c853';
        if (state === S.DONE) return '#2979ff';
        if (lastLightEff === 'FLASHING') return '#ffab00';
        if (lastLightEff === 'RED') return '#ff1744';
        if (lastLightEff === 'GREEN' || lastLightEff === 'GREEN_NEW') return '#00c853';
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
