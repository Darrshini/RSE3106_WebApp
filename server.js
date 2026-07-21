/**
 * NavAssist — server.js
 *
 * Does exactly two things:
 * 1. Serves the web app files (HTML/CSS/JS) to any browser that connects
 * 2. Relays WebSocket messages between the browser and the ESP32
 *
 * The server itself does NOT process any sensor data -- that logic
 * lives in the browser-side app.js and ai.js files.
 *
 * How to run:
 *   npm install        (first time only)
 *   node server.js     (or npm start)
 *
 * Then open http://localhost:3000 in your browser,
 * and point your ESP32 at ws://<your-laptop-ip>:3000
 */

require('dotenv').config();
const express = require('express');
const { WebSocketServer } = require('ws');
const http = require('http');
const path = require('path');

const PORT = process.env.PORT || 3000;

// ============================================================
// Express -- serves your HTML/CSS/JS files to the browser
// ============================================================

const app = express();
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json({ limit: '15mb' })); // large limit for base64 camera frames posted to /api/infer

// ============================================================
// API routes
// ============================================================

// Non-secret runtime info the settings page can display. There are NO API keys
// -- the model runs locally on this server and map data comes from the keyless
// Overpass API, so nothing sensitive is served here. See README.
app.get('/api/config', (req, res) => {
    res.json({
        crossingModel: 'crossing_seg.onnx (server, onnxruntime-node)',
    });
});

// ============================================================
// Crossing perception -- server-side YOLO11-seg inference.
// The browser POSTs a base64 JPEG frame; we run the model + logic and return
// the pedestrian-light state, the crossing-corridor direction vector, and the
// per-frame end-of-crossing signals. The stateful decisions (state machine,
// gestures, confirmations) all still live in app.js/ai.js -- this endpoint
// is purely perception, stays stateless.
// ============================================================
// The model runs on a WORKER THREAD (crossing_worker.js), never here. See that
// file for why: on the main thread its mask decode blocks the event loop, which
// stalls the WebSocket camera relay behind every inference.
const { Worker } = require('worker_threads');

// Degrees CLOCKWISE to rotate incoming Pi frames before inference. The Camera
// Module is mounted sideways on the glasses, so its frames land 90 deg CCW of
// upright. Applied inside crossing_infer's sharp pipeline (free -- it decodes
// there anyway), so every coordinate we hand back is already upright. Set
// PI_ROTATE=0 if the camera is ever remounted the right way up.
const PI_ROTATE = Number(process.env.PI_ROTATE ?? 90);

let crossWorker = null;
let crossReady  = false;
let crossBusy   = false;          // one inference in flight at a time
let latestFrame = null;           // newest Pi JPEG not yet inferred (older ones are dropped)
let crossSeq    = 0;
const crossPending = new Map();   // id -> {resolve, reject}, for /api/infer callers

function startCrossingWorker() {
    crossWorker = new Worker(path.join(__dirname, 'crossing_worker.js'));

    crossWorker.on('message', (m) => {
        if (m.type === 'ready') {
            crossReady = true;
            console.log('[infer] crossing model loaded (worker thread), Pi rotation ' + PI_ROTATE + ' deg CW');
            pumpCrossing();
            return;
        }
        if (m.type === 'error' && m.id === undefined) {   // load-time failure, not a per-frame one
            console.error('[infer] ' + m.message);
            return;
        }

        const waiter = crossPending.get(m.id);
        if (waiter) {
            crossPending.delete(m.id);
            if (m.type === 'result') waiter.resolve(m.result);
            else waiter.reject(new Error(m.message));
        } else if (m.type === 'result') {
            // No waiter => this was a live Pi frame. Push it to the viewers.
            crossBusy = false;
            broadcastLiveJson({ topic: 'crossing/result', timestamp: Date.now(), payload: m.result });
            pumpCrossing();
        } else {
            crossBusy = false;
            console.warn('[infer] frame failed:', m.message);
            pumpCrossing();
        }
    });

    crossWorker.on('error', (e) => {
        console.error('[infer] worker crashed:', e.message);
        crossReady = false; crossBusy = false;
        for (const w of crossPending.values()) w.reject(new Error('inference worker crashed'));
        crossPending.clear();
    });

    crossWorker.on('exit', (code) => {
        if (code === 0) return;
        console.warn('[infer] worker exited (' + code + ') -- restarting in 1s');
        setTimeout(startCrossingWorker, 1000);
    });
}
startCrossingWorker();

// Hand the worker the newest Pi frame, if it's idle and anyone is watching.
// Deliberately does NOT queue: if inference is slower than the camera, the
// frames that piled up in between are dropped and we always infer the freshest
// one. That's what keeps the overlay tracking reality instead of drifting
// further behind it the longer the app runs.
function pumpCrossing() {
    if (!crossReady || crossBusy || !latestFrame) return;
    let wanted = false;
    for (const c of liveClients) if (c.wantsCrossing && c.readyState === 1) { wanted = true; break; }
    if (!wanted) return;   // nobody is watching, or everyone turned it off -- don't burn CPU
    const buf = latestFrame;
    latestFrame = null;
    crossBusy = true;
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    crossWorker.postMessage({ type: 'infer', id: undefined, buf: ab, rotate: PI_ROTATE }, [ab]);
}

// One-off inference, used by /api/infer.
function inferOnce(buf, rotate) {
    return new Promise((resolve, reject) => {
        if (!crossWorker) return reject(new Error('inference worker not running'));
        const id = ++crossSeq;
        crossPending.set(id, { resolve, reject });
        const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
        crossWorker.postMessage({ type: 'infer', id, buf: ab, rotate: rotate || 0 }, [ab]);
    });
}

// Still used by webcam.html / live.html / test.html, which pull frames from a
// <video> and POST them. The Pi path does NOT use this -- the server already has
// those frames, so making the browser re-encode and upload them back would be a
// pointless round trip. Callers can pass `rotate` (degrees CW) if their source
// is sideways; a webcam or an uploaded video is upright, so they omit it.
app.post('/api/infer', async (req, res) => {
    try {
        const b64 = (req.body && req.body.image) || '';
        const data = b64.replace(/^data:image\/\w+;base64,/, '');
        if (!data) return res.status(400).json({ error: 'no image' });
        res.json(await inferOnce(Buffer.from(data, 'base64'), Number(req.body.rotate) || 0));
    } catch (e) {
        console.error('[infer] error:', e.message);
        res.status(500).json({ error: e.message });
    }
});

// ============================================================
// HTTP server -- wraps Express so WebSocket can share the port
// ============================================================

const server = http.createServer(app);

// ============================================================
// WebSocket server -- handles real-time comms
// Both the browser AND the ESP32 connect here as clients
// The server relays messages between them
// ============================================================

const wss = new WebSocketServer({ server });

// Track connected clients by type
// browserClients is a Set (not a single slot) so multiple people can view
// the live feed simultaneously -- every message gets broadcast to all of them.
let browserClients = new Set();
let esp32Client = null;

// The real-time Pi path, kept entirely separate from the legacy /browser +
// /esp32 relay above so it can't disturb index.html:
//   /pi    <- Raspberry Pi. Sends each JPEG as a BINARY frame (raw bytes, no
//            base64, no JSON envelope). base64-in-JSON inflates every frame by
//            33% and makes both ends parse a ~100KB string per frame; on a Zero
//            2W's 2.4GHz-only WiFi that inflation is the actual throughput
//            ceiling. Text frames on this socket are still JSON (heartbeats).
//   /live  -> viewers (pi.html). Gets the same JPEG bytes straight through, plus
//            crossing/result JSON pushed from the worker. Viewers never upload a
//            frame back to us.
let liveClients = new Set();
let piClient = null;
let piFrames = 0, piBytes = 0;   // running totals, reported once a second

wss.on('connection', (ws, req) => {
    const clientIp = req.socket.remoteAddress;
    const url = req.url;

    console.log(`[WS] New connection from ${clientIp} path: ${url}`);

    // Identify client type by connection path:
    // Browser connects to /browser
    // ESP32 connects to /esp32
    if (url === '/browser') {
        browserClients.add(ws);
        console.log(`[WS] Browser connected (${browserClients.size} total viewer(s))`);

        // Tell THIS newly-connected browser the current ESP32 status --
        // sent directly to it, not broadcast, so other already-connected
        // viewers don't get a redundant duplicate status message.
        const statusMsg = JSON.stringify({
            topic: 'connection/event',
            timestamp: Date.now(),
            payload: {
                event: esp32Client ? 'esp32_connected' : 'esp32_disconnected'
            }
        });
        if (ws.readyState === 1) ws.send(statusMsg);

        ws.on('message', (data) => {
            // Browser → ESP32 (e.g. haptic commands)
            try {
                const message = JSON.parse(data.toString());
                console.log(`[WS] Browser→ESP32: ${message.topic}`);
                sendToEsp32(message);
            } catch (e) {
                console.warn('[WS] Invalid message from browser:', e.message);
            }
        });

        ws.on('close', () => {
            browserClients.delete(ws);
            console.log(`[WS] Browser disconnected (${browserClients.size} remaining)`);
        });

    } else if (url === '/esp32') {
        esp32Client = ws;
        console.log('[WS] ESP32 connected');

        // Tell browser the ESP32 just connected
        sendToBrowser({
            topic: 'connection/event',
            timestamp: Date.now(),
            payload: { event: 'esp32_connected' }
        });

        ws.on('message', (data) => {
            // ESP32 → Browser (camera frames, IMU, heartbeat)
            try {
                const message = JSON.parse(data.toString());
                // Only log non-image topics to avoid flooding the console
                if (message.topic !== 'camera/image') {
                    console.log(`[WS] ESP32→Browser: ${message.topic}`);
                }
                sendToBrowser(message);
            } catch (e) {
                console.warn('[WS] Invalid message from ESP32:', e.message);
            }
        });

        ws.on('close', () => {
            console.log('[WS] ESP32 disconnected');
            esp32Client = null;
            // Tell browser the ESP32 disconnected
            sendToBrowser({
                topic: 'connection/event',
                timestamp: Date.now(),
                payload: { event: 'esp32_disconnected' }
            });
        });

    } else if (url === '/pi') {
        piClient = ws;
        console.log('[WS] Raspberry Pi camera connected');
        broadcastLiveJson({ topic: 'connection/event', timestamp: Date.now(),
                            payload: { event: 'pi_connected', rotate: PI_ROTATE } });

        ws.on('message', (data, isBinary) => {
            if (isBinary) {
                piFrames++; piBytes += data.length;
                latestFrame = data;        // newest wins; anything not yet inferred is dropped
                broadcastLiveBinary(data);
                pumpCrossing();
            } else {
                try {
                    const message = JSON.parse(data.toString());
                    broadcastLiveJson(message);     // heartbeats etc.
                } catch (e) {
                    console.warn('[WS] Invalid text message from Pi:', e.message);
                }
            }
        });

        ws.on('close', () => {
            console.log('[WS] Raspberry Pi camera disconnected');
            piClient = null;
            latestFrame = null;
            broadcastLiveJson({ topic: 'connection/event', timestamp: Date.now(),
                                payload: { event: 'pi_disconnected' } });
        });

    } else if (url === '/live') {
        ws.wantsCrossing = true;
        liveClients.add(ws);
        console.log(`[WS] Live viewer connected (${liveClients.size} total)`);
        if (ws.readyState === 1) {
            // Tell the viewer the rotation we apply, so its canvas and our
            // coordinates can never silently disagree about which way is up.
            ws.send(JSON.stringify({
                topic: 'connection/event', timestamp: Date.now(),
                payload: { event: piClient ? 'pi_connected' : 'pi_disconnected', rotate: PI_ROTATE }
            }));
        }
        pumpCrossing();   // a viewer just showed up: start inferring again

        ws.on('message', (data) => {
            try {
                const message = JSON.parse(data.toString());
                if (message.topic === 'live/config' && message.payload) {
                    // Unchecking "crossing model" has to stop the SERVER inferring,
                    // not just stop the browser drawing -- you are almost certainly
                    // viewing this page on the same laptop that runs the server, so
                    // an unwanted seg model still competes with pedestrian.onnx for
                    // the same CPU.
                    ws.wantsCrossing = message.payload.crossing !== false;
                    pumpCrossing();
                } else {
                    // Anything else is for the glasses -- haptic/command from
                    // app.js. The Pi picks these up in read_commands().
                    sendToPi(message);
                }
            } catch (e) {
                console.warn('[WS] Invalid message from live viewer:', e.message);
            }
        });

        ws.on('close', () => {
            liveClients.delete(ws);
            console.log(`[WS] Live viewer disconnected (${liveClients.size} remaining)`);
        });

    } else {
        console.log(`[WS] Unknown path ${url} -- closing connection`);
        ws.close();
    }
});

// ============================================================
// Helper functions
// ============================================================

// Camera frames dropped so far, plus when we last logged -- so we can report
// drops occasionally instead of flooding the console every frame.
let droppedCamFrames = 0;
let lastDropLogAt = 0;

function sendToBrowser(message) {
    const data = JSON.stringify(message);
    const isCameraFrame = message.topic === 'camera/image';

    for (const client of browserClients) {
        if (client.readyState !== 1) continue;

        // Backpressure-aware frame dropping. WebSocket is reliable + ordered,
        // so if a browser can't drain frames as fast as the camera produces
        // them, they don't drop -- they queue in this socket's send buffer and
        // the on-screen feed falls further and further behind (the lag you see
        // only once the server is in the loop). bufferedAmount is how many
        // bytes are still waiting to go out to THIS client. If more than
        // roughly one whole frame is already queued, skip sending this frame to
        // that client -- it'd only add latency, and a newer frame is coming.
        // Only camera frames are dropped; heartbeat / IMU / connection events
        // are small and important, so they always go through.
        if (isCameraFrame && client.bufferedAmount > data.length) {
            droppedCamFrames++;
            continue;
        }

        client.send(data);
    }

    // Occasional summary so it's visible that dropping is happening (and how
    // much), without logging on every single frame.
    if (isCameraFrame && droppedCamFrames > 0) {
        const now = Date.now();
        if (now - lastDropLogAt > 2000) {
            console.log(`[WS] Dropped ${droppedCamFrames} stale camera frame(s) in the last ~2s to keep the feed low-latency`);
            droppedCamFrames = 0;
            lastDropLogAt = now;
        }
    }
}

function sendToEsp32(message) {
    if (esp32Client && esp32Client.readyState === 1) {
        esp32Client.send(JSON.stringify(message));
    } else {
        console.warn('[WS] Cannot send to ESP32 -- not connected');
    }
}

// Browser (/live) -> Pi (/pi). Haptic commands, as text frames; the Pi's binary
// frames only ever flow the other way.
//
// This logs every relayed topic on purpose. When the motors don't buzz, the very
// first question is "did a command even get this far", and a silent pass-through
// cannot answer it -- you end up unable to tell a browser that never sent one
// from a Pi that never received it. The line below splits those two cases.
function sendToPi(message) {
    if (piClient && piClient.readyState === 1) {
        piClient.send(JSON.stringify(message));
        console.log(`[WS] browser -> Pi  ${message.topic}  ${JSON.stringify(message.payload || {})}`);
    } else {
        console.warn(`[WS] Cannot send to Pi -- not connected (dropped ${message.topic})`);
    }
}

// ---- /live broadcast (Pi frames + crossing results) --------------------------

let droppedLiveFrames = 0;

// Same backpressure rule as sendToBrowser: WebSocket is reliable and ordered, so
// a viewer that can't keep up doesn't drop frames, it accumulates them in the
// send buffer and falls steadily further behind. If roughly a whole frame is
// already queued for this client, skip -- a fresher one is right behind it.
function broadcastLiveBinary(buf) {
    for (const client of liveClients) {
        if (client.readyState !== 1) continue;
        if (client.bufferedAmount > buf.length) { droppedLiveFrames++; continue; }
        client.send(buf, { binary: true });
    }
}

function broadcastLiveJson(message) {
    const data = JSON.stringify(message);
    for (const client of liveClients) {
        if (client.readyState === 1) client.send(data);
    }
}

// Once a second, print what the link is actually sustaining. This is the number
// to watch when tuning CAP_W/FPS/JPEG_Q on the Pi: if fps here is well below the
// Pi's configured FPS, the bottleneck is the WiFi link, not the models.
setInterval(() => {
    if (!piFrames && !droppedLiveFrames) return;
    const kbps = (piBytes / 1024).toFixed(0);
    console.log(`[pi] ${piFrames} fps  ${kbps} KB/s` +
                (droppedLiveFrames ? `  (dropped ${droppedLiveFrames} to slow viewers)` : ''));
    piFrames = 0; piBytes = 0; droppedLiveFrames = 0;
}, 1000);

// ============================================================
// Start
// ============================================================

server.listen(PORT, '0.0.0.0', () => {
    const { networkInterfaces } = require('os');
    const nets = networkInterfaces();
    const candidates = [];

    // Collect all non-internal IPv4 addresses
    for (const [name, ifaces] of Object.entries(nets)) {
        for (const iface of ifaces) {
            if (iface.family === 'IPv4' && !iface.internal) {
                candidates.push({ name, address: iface.address });
            }
        }
    }

    // Prefer WiFi interface over others (Ethernet, VPN, virtual adapters)
    const wifi = candidates.find(c =>
        c.name.toLowerCase().includes('wi-fi') ||
        c.name.toLowerCase().includes('wlan') ||
        c.name.toLowerCase().includes('wireless')
    );
    const best = wifi || candidates[0];
    const localIp = best ? best.address : 'unknown';

    console.log('\n========================================');
    console.log(' NavAssist Server Running');
    console.log('========================================');
    console.log(`\n Open the web app at:`);
    console.log(`   http://localhost:${PORT}          (on this laptop)`);
    console.log(`   http://${localIp}:${PORT}   (on phone/any device on same WiFi)\n`);
    console.log(` Raspberry Pi camera test bench:`);
    console.log(`   http://localhost:${PORT}/pi.html\n`);
    console.log(` Set SERVER_HOST in navassist_pi_camera.py to:`);
    console.log(`   ${localIp}      (the Pi connects to ws://${localIp}:${PORT}/pi)\n`);
    console.log(` ESP32 should connect to:`);
    console.log(`   ws://${localIp}:${PORT}/esp32\n`);
    console.log(` Browser WebSocket connects to:`);
    console.log(`   ws://${localIp}:${PORT}/browser`);

    // Show ALL detected IPs so you can pick the right one if needed
    if (candidates.length > 1) {
        console.log('\n All detected network interfaces:');
        candidates.forEach(c => console.log(`   ${c.name}: ${c.address}`));
        console.log(' If the IP above looks wrong, use ipconfig to find the correct one.');
    }

    console.log('\n----------------------------------------');
    console.log(' Waiting for connections...\n');
});
