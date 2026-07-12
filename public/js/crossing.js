/**
 * NavAssist — crossing.js  (client)
 *
 * Shared by both pages (live Pi-camera + upload-video test). Grabs frames from
 * a <video>, POSTs them to the server (/api/infer, which runs the YOLO11-seg
 * model + perception), draws the overlay (dotted-line boxes, pedestrian-light +
 * state, corridor direction vector), and runs the stateful decision FSM:
 *
 *   WAITING  -> must see RED (arm), then a fresh RED->GREEN turn to say "cross".
 *               A green that's ALREADY showing on arrival = wait for the next.
 *   CROSSING -> guide along the corridor; announce "reached the other side" when
 *               the dashes run out ahead AND the light is close/gone (far side).
 *   DONE     -> re-arms to WAITING if a red is seen again.
 *
 * Exposes window.Crossing = { start(videoEl, overlayEl, onStatus), stop() }.
 */
(function () {
    const CFG = {
        INFER_MS: 250,            // min gap between inference requests
        LOOP_MS: 40,
        JPEG_Q: 0.6,
        CAP_W: 960,               // downscale frame sent to the server (speed)
        LIGHT_CLOSE_AREA: 0.004,  // light this big (frac of frame) => trust its state
        END_LIGHT_AREA: 0.02,     // light this big => you're at the far side
        END_NO_DASH_FRAMES: 6,    // consecutive frames with no dashes ahead => corridor ran out
        SPEAK_COOLDOWN: 3500
    };
    const S = { WAITING: 'WAITING', CROSSING: 'CROSSING', DONE: 'DONE' };

    let video, overlay, octx, cap, capctx, onStatus;
    let running = false, inFlight = false, lastInferAt = 0, latest = null;
    let state = S.WAITING, armed = false, noDash = 0, lastSpeakAt = 0, lastDirWord = '';

    function speak(text, force) {
        const now = Date.now();
        if (!force && now - lastSpeakAt < CFG.SPEAK_COOLDOWN) return;
        lastSpeakAt = now;
        if (window.navassist && window.navassist.speak) window.navassist.speak(text);
        else if (window.speechSynthesis) { speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(text)); }
    }
    function status(msg) { if (onStatus) onStatus(state, msg); }

    function start(videoEl, overlayEl, statusCb) {
        video = videoEl; overlay = overlayEl; onStatus = statusCb;
        octx = overlay.getContext('2d');
        cap = document.createElement('canvas'); capctx = cap.getContext('2d', { willReadFrequently: true });
        state = S.WAITING; armed = false; noDash = 0; running = true;
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
        if (r && r.w) { latest = r; decide(r); }
    }

    // ---- decision FSM ----
    function decide(r) {
        const light = r.light, ls = light ? light.state : 'UNKNOWN';
        const close = light && light.areaFrac >= CFG.LIGHT_CLOSE_AREA;

        if (state === S.WAITING) {
            if (close && ls === 'RED') { armed = true; speak('Red man. Please wait.'); status('waiting — red'); }
            else if (ls === 'GREEN') {
                if (armed) { state = S.CROSSING; armed = false; noDash = 0; speak('Green man. You may cross now.', true); status('CROSS NOW'); }
                else if (close) { speak('Green is already showing. Wait for the next green before you cross.'); status('waiting — let this green pass'); }
            }
        } else if (state === S.CROSSING) {
            if (r.corridor && r.corridor.has) {
                noDash = 0;
                const word = dirWord(r.corridor, r.w);
                if (word && word !== lastDirWord) { lastDirWord = word; speak(word); }
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
            if (close && ls === 'RED') { state = S.WAITING; armed = true; }
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

        octx.lineWidth = lw; octx.strokeStyle = '#00b0ff';
        for (const d of r.dotted) { const b = d.box; octx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]); }

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
        const l = latest && latest.light;
        if (l && l.state === 'GREEN') return armed ? '🟢 GREEN' : '⏸ Wait for next green';
        if (l && l.state === 'RED') return '🔴 Wait';
        return 'Looking for the crossing…';
    }
    function bannerColor() {
        if (state === S.CROSSING) return '#00c853';
        if (state === S.DONE) return '#2979ff';
        const l = latest && latest.light;
        if (l && l.state === 'RED') return '#ff1744';
        return '#555';
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
