# NavAssist Web App

Assistive navigation web app for visually impaired users — RSE3106 James Dyson Award Submission

## Team 7

| Member | File | Responsibility |
|---|---|---|
| Darrshini | `public/js/app.js` | ESP32 comms, GPS, haptic commands, state machine |
| Kim Hyeonghu | `public/js/ai.js` | Camera processing, AI inference, direction logic |

---

## How it works

```
Browser (index.html)
    ↕ WebSocket ws://laptop-ip:80/browser
Node.js server (server.js) — running on laptop
    ↕ WebSocket ws://laptop-ip:80/esp32
ESP32-CAM (glasses hardware)
```

The Node.js server relays messages between the browser and the ESP32. All intelligence runs in the browser.
 - ENSURE YOU HAVE NODE.JS INSTALLED ON YOUR PC OR LAPTOP. CHECK IN TERMINAL : node --version
 - if you see an error , install Node.js from this site: https://nodejs.org/en/download
 - ensure node.js is installed before starting the next setup instructions below:

---

## Quick start

### 1. Clone and install
```bash
git clone https://github.com/Darrshini/RSE3106_WebApp.git
cd RSE3106_WebApp
npm install
```

### 2. Set up API keys
```bash
cp .env.example .env
# Edit .env and fill in your actual keys
```

### 3. Start the server
```bash
npm start
```

You'll see:
```
Open the web app at:
  http://localhost:80         (on this laptop)
  http://192.168.x.x:80       (on any device on same WiFi)

ESP32 should connect to:
  ws://192.168.x.x:80/esp32
```

### 4. Open the web app
Open `http://localhost:80` in Chrome on your phone or laptop.

### 5. Flash the ESP32
Update the ESP32 firmware with:
- WiFi credentials (your laptop's hotspot or same router)
- WebSocket URL: `ws://192.168.x.x:80/esp32`

---

## Message protocol

Every message uses this envelope:
```json
{ "topic": "...", "timestamp": 1234567890, "payload": { ... } }
```

| Topic | Direction | Description |
|---|---|---|
| `camera/image` | ESP32 → Browser | JPEG frame as base64 |
| `imu/orientation` | ESP32 → Browser | Heading, pitch, roll |
| `system/heartbeat` | ESP32 → Browser | Keep-alive every 2s |
| `haptic/command` | Browser → ESP32 | Fire left/right motor |
| `connection/event` | Server → Browser | ESP32 connected/disconnected |

---

## Branch strategy

- `main` — stable, working code only
- `feature/esp32-comms` — Darrshini: WebSocket, GPS, haptic
- `feature/ai-inference` — [Teammate]: camera processing, Roboflow

Merge to `main` only when your branch is tested and working.

---

## Getting API keys

**Roboflow:**
1. Sign up at [app.roboflow.com](https://app.roboflow.com)
2. Profile → Settings → Roboflow API → copy key

**Google Maps:**
1. [console.cloud.google.com](https://console.cloud.google.com)
2. Enable Places API + Geocoding API
3. Create an API key, restrict to Places API
