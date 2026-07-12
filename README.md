# NavAssist Web App
Assistive navigation web app for visually impaired users — RSE3106 James Dyson Award Submission

The instructions below are to set up the app locally on your PC or laptop if you want to.

I have hosted the webapp on an AWS instance, you can access it through this link: [Web app prototype](https://rse3106.duckdns.org/)
I have changed it to HTTPS, however you will get a warning sign saying it's unsecure — this is normal, since I used a self-signed certificate, but it should work fine, just click
**Advanced > Continue to Site**


I needed to change from HTTP to HTTPS because in HTTP, GPS and the phone's compass will not work.

## Team 7

| Member | File | Responsibility |
|---|---|---|
| Darrshini | `public/js/app.js` | GPS, haptic commands, state machine |
| Kim Hyeonghu | `public/js/ai.js` | Camera processing, AI inference, direction logic |
| Neo Zhen Ye | `public/js/app.js` | Overall integration with Raspberry Pi and system logic |

---

## How it works

```
Browser (index.html)
    ↕ WebSocket wss://rse3106.duckdns.org/browser
Node.js server (server.js) — running on AWS EC2
    ↕ WebSocket wss://rse3106.duckdns.org/esp32
Raspberry Pi 4 + Camera Module 3 (glasses hardware)
```

The Node.js server relays messages between the browser and the Raspberry Pi. All intelligence — camera AI inference, GPS/crossing logic, and heading tracking — runs in the browser, not on the server or the Pi.

**Note on hardware:** this project originally used an ESP32-CAM, then moved to a Raspberry Pi 4 with Camera Module 3 for the camera feed. The Pi's Python script (`navassist_pi_camera.py`) captures JPEG frames using the hardware-accelerated MJPEG encoder and streams them to the server over WebSocket, using the same message protocol the ESP32 originally used — so `server.js`/`app.js`/`ai.js` didn't need structural changes for the hardware swap, just a new device-side script.

**Note on heading/orientation:** we decided not to use a physical accelerometer/gyroscope on the glasses hardware. Instead, `app.js` reads heading directly from the **phone's own built-in compass** (via the browser's `DeviceOrientationEvent`) to drive the "you've turned away" warning and the haptic drift-correction while crossing. This is a deliberate tradeoff — phone compass data can be noisier than a dedicated IMU, and assumes the phone stays reasonably oriented with the user, but it saved significant hardware/integration time. **This only works on mobile browsers**, not laptops — most laptops don't have the sensors to produce this data at all, so heading-based features can't be tested on a laptop, only on an actual phone. On iPhone specifically, the first time the app is opened it'll show a one-time "Motion & Orientation" permission popup — you must tap **Allow**, otherwise heading data silently never arrives.

 - ENSURE YOU HAVE NODE.JS INSTALLED ON YOUR PC OR LAPTOP. CHECK IN TERMINAL: `node --version`
 - if you see an error, install Node.js from this site: https://nodejs.org/en/download
 - ensure Node.js is installed before starting the next setup instructions below:

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
  http://localhost:3000         (on this laptop)
  http://192.168.x.x:3000       (on any device on same WiFi)
Raspberry Pi should connect to:
  ws://192.168.x.x:3000/esp32
```

### 4. Open the web app
Open `http://localhost:3000` in Chrome on your phone or laptop.
(For the hosted AWS version, just open https://rse3106.duckdns.org — no local setup needed.)

### 5. Run the camera script on the Raspberry Pi
The Pi runs `navassist_pi_camera.py`, which captures frames from Camera Module 3 and streams them to the server.

Install dependencies on the Pi first:
```bash
sudo apt install -y python3-picamera2
pip install websockets --break-system-packages
```

Then edit the top of the script to point at whichever server you're using:
- **Local testing** (Pi and laptop on the same WiFi): set `WS_HOST` to your laptop's local IP (shown when you run `npm start`), `WS_PORT = 3000`, and use plain `ws://` (no SSL).
- **Production (AWS)**: `WS_HOST = "rse3106.duckdns.org"`, `WS_PORT = 443`, `wss://` with certificate verification disabled (self-signed cert, same reasoning as the browser warning above).

Run it:
```bash
python3 navassist_pi_camera.py
```
You should see `Connected to relay server.` printed, and the web app's status bar should show "Glasses connected."

---

## Message protocol
Every message uses this envelope:
```json
{ "topic": "...", "timestamp": 1234567890, "payload": { ... } }
```

| Topic | Direction | Description |
|---|---|---|
| `camera/image` | Raspberry Pi → Browser | JPEG frame as base64 |
| `system/heartbeat` | Raspberry Pi → Browser | Keep-alive every 2s |
| `haptic/command` | Browser → Raspberry Pi | Fire left/right motor |
| `connection/event` | Server → Browser | Raspberry Pi connected/disconnected |

**Note:** `imu/orientation` (Pi/ESP32 → Browser) is no longer used — heading now comes directly from the phone's own compass in the browser (see the note above), not from a message over WebSocket. The code path for handling `imu/orientation` still exists in `app.js` for backward compatibility, but nothing currently sends this message.

---

## Getting API keys

**Google Maps:**
1. [console.cloud.google.com](https://console.cloud.google.com)
2. Enable Places API + Geocoding API
3. Create an API key, restrict to Places API
