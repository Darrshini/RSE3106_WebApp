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
app.use(express.json()); // for parsing API request bodies

// ============================================================
// API routes -- browser can call these to get API keys safely
// Keys stay on the server, never exposed in browser JS files
// ============================================================

app.get('/api/config', (req, res) => {
    // Browser asks for config -- server responds with keys
    // This way keys never appear in your committed JS files
    res.json({
        roboflowApiKey: process.env.ROBOFLOW_API_KEY,
        roboflowModelId: process.env.ROBOFLOW_MODEL_ID,
        roboflowModelVersion: process.env.ROBOFLOW_MODEL_VERSION,
        googleMapsApiKey: process.env.GOOGLE_MAPS_API_KEY,
    });
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

    } else {
        console.log(`[WS] Unknown path ${url} -- closing connection`);
        ws.close();
    }
});

// ============================================================
// Helper functions
// ============================================================

function sendToBrowser(message) {
    const data = JSON.stringify(message);
    for (const client of browserClients) {
        if (client.readyState === 1) {
            client.send(data);
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
