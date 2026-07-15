/**
 * crossing_worker.js — runs crossing_infer.js on a worker thread.
 *
 * Why this exists: crossing_infer does a lot of work in plain JS on whatever
 * thread calls it -- filling the 640x640x3 input tensor, and (the expensive one)
 * evaluating each dotted-line segmentation mask, which is a 32-coefficient dot
 * product per prototype cell over a 160x160 grid. On the main thread that blocks
 * the event loop, which means it also blocks the WebSocket relay: camera frames
 * from the Pi would sit in the queue waiting for an inference to finish, and the
 * live feed visibly stutters. Here it blocks nothing but itself.
 *
 * Protocol (see server.js):
 *   in   { type:'infer', id, buf: ArrayBuffer(jpeg), rotate }   -- buf is TRANSFERRED
 *   out  { type:'ready' }
 *        { type:'result', id, result }     result.inferMs = wall time in here
 *        { type:'error',  id, message }
 *
 * Requests are handled strictly one at a time, in arrival order. The caller is
 * responsible for not queueing stale frames (server.js keeps only the latest).
 */
const { parentPort } = require('worker_threads');
const crossing = require('./crossing_infer');

crossing.load()
    .then(() => parentPort.postMessage({ type: 'ready' }))
    .catch(e => parentPort.postMessage({ type: 'error', message: 'model load failed: ' + e.message }));

parentPort.on('message', async (msg) => {
    if (!msg || msg.type !== 'infer') return;
    const t0 = Date.now();
    try {
        const result = await crossing.infer(Buffer.from(msg.buf), { rotate: msg.rotate });
        result.inferMs = Date.now() - t0;
        parentPort.postMessage({ type: 'result', id: msg.id, result });
    } catch (e) {
        parentPort.postMessage({ type: 'error', id: msg.id, message: e.message });
    }
});
