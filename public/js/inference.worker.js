/**
 * NavAssist — inference.worker.js
 *
 * Runs the YOLO ONNX model OFF the main thread. onnxruntime-web's WASM backend
 * is single-threaded and blocks whatever thread it runs on for the whole of
 * each session.run(). On the main thread that froze the camera redraw, gesture
 * handling and the WebSocket heartbeat for a few hundred ms every inference --
 * the periodic "hitch" in the feed. Here it runs in a Worker, so the main
 * thread stays free to decode/draw frames smoothly while the model chews.
 *
 * Message protocol (main <-> worker):
 *   main -> worker  { type:'load',  modelUrl, wasmPaths, numThreads, providers }
 *     wasmPaths is the self-hosted onnxruntime dir (public/vendor/onnxruntime/),
 *     which holds ort.webgpu.min.js + the wasm binary. No CDN, so it works offline.
 *     providers is the ordered EP preference, e.g. ['webgpu','wasm'].
 *   worker -> main  { type:'ready', backend }           // backend = EP that won
 *                   { type:'error', message }            // during load or infer
 *                   { type:'log',   message }            // debug passthrough
 *   main -> worker  { type:'infer', data:ArrayBuffer, dims, scale, padX, padY,
 *                     confThresh, iouThresh }             // data buffer transferred
 *   worker -> main  { type:'result', detections:[{x1,y1,x2,y2,score,cls}], inferMs }
 *
 * The main thread does the canvas-bound pre-processing (it needs the DOM
 * canvas) and hands over a ready-made CHW float buffer. The worker owns model
 * loading, session.run and the box decode + NMS, so all the heavy work lives
 * here -- on the GPU (WebGPU) when available, else the CPU (WASM).
 */

// NOTE: do NOT name this `ort` -- importScripts of the ort bundle declares a
// global `ort`, and a `let ort` here collides with it ("already declared").
let ortLib = null;    // = self.ort, set after importScripts in the 'load' handler
let session = null;
let ready = false;

self.onmessage = async (e) => {
    const msg = e.data;

    if (msg.type === 'load') {
        try {
            // ort.webgpu.min.js is the UMD build that bundles BOTH the WebGPU
            // and WASM execution providers. importable into a classic worker;
            // sits alongside the wasm binary in wasmPaths.
            importScripts(msg.wasmPaths + 'ort.webgpu.min.js');
            ortLib = self.ort;
            ortLib.env.wasm.wasmPaths = msg.wasmPaths;
            ortLib.env.wasm.numThreads = msg.numThreads || 1;   // multi-thread needs COOP/COEP; keep at 1

            // Try the preferred execution providers in order (webgpu, then wasm)
            // and keep the first that loads. Done one-at-a-time -- rather than
            // passing the whole list to one create() call -- so we can report
            // exactly which backend actually won, which is the whole point of
            // this exercise (is the GPU being used on this device or not?).
            const providers = msg.providers && msg.providers.length ? msg.providers : ['wasm'];
            let backend = null, lastErr = null;
            for (const ep of providers) {
                try {
                    session = await ortLib.InferenceSession.create(msg.modelUrl, {
                        executionProviders: [ep],
                        graphOptimizationLevel: 'all'
                    });
                    backend = ep;
                    break;
                } catch (err) {
                    lastErr = err;
                    self.postMessage({ type: 'log',
                        message: `EP '${ep}' unavailable (${err.message}); trying next…` });
                }
            }
            if (!backend) throw lastErr || new Error('no execution provider available');

            ready = true;
            self.postMessage({ type: 'log',
                message: `Worker model loaded on '${backend}'. in=${session.inputNames} out=${session.outputNames}` });
            self.postMessage({ type: 'ready', backend });
        } catch (err) {
            self.postMessage({ type: 'error', message: err.message });
        }
        return;
    }

    if (msg.type === 'infer') {
        if (!ready || !session) return;
        try {
            const t0 = performance.now();
            const tensor = new ortLib.Tensor('float32', new Float32Array(msg.data), msg.dims);
            const feeds = {}; feeds[session.inputNames[0]] = tensor;
            const results = await session.run(feeds);
            const out = results[session.outputNames[0]];
            const detections = postprocess(out, msg);
            const inferMs = performance.now() - t0;
            self.postMessage({ type: 'result', detections, inferMs });
        } catch (err) {
            self.postMessage({ type: 'error', message: err.message });
        }
        return;
    }
};

// Decode YOLO output [1, 4+nc, 8400] -> boxes in original-frame pixels, then NMS.
// Reads the output shape dynamically (num anchors, class count), so a new model
// with a different class count or anchor grid drops in with no change here.
function postprocess(out, meta) {
    const dims = out.dims;            // e.g. [1, 7, 8400]
    const num  = dims[2];
    const nCls = dims[1] - 4;
    const d    = out.data;
    const { scale, padX, padY, confThresh, iouThresh } = meta;

    const boxes = [];
    for (let a = 0; a < num; a++) {
        let best = 0, bestC = 0;
        for (let c = 0; c < nCls; c++) {
            const s = d[(4 + c) * num + a];
            if (s > best) { best = s; bestC = c; }
        }
        if (best < confThresh) continue;
        const cx = d[a], cy = d[num + a], w = d[2 * num + a], h = d[3 * num + a];
        boxes.push({
            x1: (cx - w / 2 - padX) / scale,
            y1: (cy - h / 2 - padY) / scale,
            x2: (cx + w / 2 - padX) / scale,
            y2: (cy + h / 2 - padY) / scale,
            score: best, cls: bestC
        });
    }
    return nms(boxes, iouThresh);
}

function nms(boxes, iouThr) {
    boxes.sort((a, b) => b.score - a.score);
    const keep = [], dead = new Array(boxes.length).fill(false);
    for (let i = 0; i < boxes.length; i++) {
        if (dead[i]) continue;
        keep.push(boxes[i]);
        for (let j = i + 1; j < boxes.length; j++) {
            if (!dead[j] && boxes[i].cls === boxes[j].cls && iou(boxes[i], boxes[j]) > iouThr) dead[j] = true;
        }
    }
    return keep;
}

function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1), areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-6);
}
