# NavAssist Web App
Assistive navigation web app for visually impaired users — RSE3106 James Dyson Award Submission

> **Branch note — `test/pure-crossing`:** this branch is the full app (`index.html` runs
> `app.js` + `ai.js`, same as `main`), with the crossing-**selection** confirmations removed:
> it no longer asks the user to double/triple-tap to pick a crossing or a traffic-light post —
> it identifies the crossing and heads straight for it. The two safety gates are kept, each as a
> single confirming tap: **CONFIRM_CROSSING** (tap to confirm you are crossing on the green man)
> and **CONFIRM_ARRIVAL** (tap to confirm you felt the tactile indicators). See `HAPTICS.md` for
> the state-by-state haptic flow.

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
Browser (index.html)  — app.js + ai.js
    ↕ WebSocket wss://rse3106.duckdns.org/live
Node.js server (server.js) — running on AWS EC2
    ↕ WebSocket wss://rse3106.duckdns.org/pi
Raspberry Pi Zero 2W + Camera Module v3 (glasses hardware)
```

**The Pi is a camera, not a brain.** It captures JPEG frames and ships the bytes; that is all
it does. The two AI models run in two *other* places, and neither runs on the Pi:

| Model | Classes | Runs |
|---|---|---|
| `pedestrian.onnx` | `red`, `green`, `traffic-light` | **In the browser**, via onnxruntime-web in a Web Worker (`js/inference.worker.js`). WebGPU where available, else WASM. |
| `crossing_seg.onnx` | `dotted line`, `pedestrian light` | **On the Node server**, via onnxruntime-node on a worker thread (`crossing_worker.js`). |

A Zero 2W is 4× Cortex-A53 @1GHz with 512 MB of RAM — a YOLO11 pass on it takes *seconds*,
not milliseconds. Moving inference onto the Pi wouldn't make the system faster, it would stop
it working at all. So the Pi encodes and sends, the server segments, and the browser detects.
All the *decisions* (state machine, GPS/crossing logic, confirmations, speech) live in the
browser, in `app.js`.

### Hardware history

This project used an ESP32-CAM, then a Raspberry Pi 4, and is now on a **Pi Zero 2W with
Camera Module v3**. The ESP32 path (`/esp32` + `/browser`, base64 frames) still exists in
`server.js` but **nothing uses it anymore** — `index.html` no longer connects to it. It is
dead code kept only for reference and can be deleted once nobody wants the history.

The Zero 2W move changed the wire protocol, and that change is what makes the feed real-time:
frames now travel as **raw binary JPEG bytes**, not base64 inside a JSON envelope. base64
inflates every frame by 33% and makes both ends parse a ~100 KB string per frame; on the Zero
2W — whose WiFi is 2.4 GHz-only — that inflation *was* the throughput ceiling. See
`PI_REALTIME.md` for the full design and the reasoning behind every knob.

### Note on heading/orientation

We decided not to use a physical accelerometer/gyroscope on the glasses. Instead, `app.js`
reads heading directly from the **phone's own built-in compass** (via the browser's
`DeviceOrientationEvent`) to drive the "you've turned away" warning and the haptic
drift-correction while crossing. This is a deliberate tradeoff — phone compass data is noisier
than a dedicated IMU, and it assumes the phone stays reasonably oriented with the user — but it
saved significant hardware and integration time.

**This only works on mobile browsers**, not laptops — most laptops don't have the sensors to
produce this data at all, so heading-based features can't be tested on a laptop, only on an
actual phone. On iPhone specifically, the first time the app is opened it shows a one-time
"Motion & Orientation" permission popup — you must tap **Allow**, otherwise heading data
silently never arrives.

---

## Quick start

Ensure you have Node.js installed. Check in a terminal: `node --version`. If you see an error,
install it from https://nodejs.org/en/download.

### 1. Clone and install
```bash
git clone https://github.com/Darrshini/RSE3106_WebApp.git
cd RSE3106_WebApp
npm install
```

### 2. (Optional) Configuration — no API keys needed
The app needs **no API keys**. It finds nearby pedestrian crossings from OpenStreetMap via the
free, keyless **Overpass API**, so you can go straight to step 3.

A `.env` file is optional — copy it only if you want to change the port:
```bash
cp .env.example .env   # then edit PORT if 3000 is taken
```
The `GOOGLE_MAPS_API_KEY` and `ROBOFLOW_*` entries in `.env.example` are **legacy and unused**:
the map lookup moved to Overpass, and both models now run locally (browser + Node), not Roboflow.

### 3. Start the server
```bash
npm start
```
You'll see the local IP printed, plus the exact URL the Pi should connect to.

### 4. Open the web app
Open `http://localhost:3000` in Chrome on your phone or laptop.
(For the hosted AWS version, just open https://rse3106.duckdns.org — no local setup needed.)

### 5. Run the camera script on the Raspberry Pi

Install dependencies on the Pi first:
```bash
sudo apt install -y python3-picamera2
pip install websockets --break-system-packages
```

Then run it, pointing at whichever server you're using:
```bash
# Local testing (Pi and laptop on the same WiFi).
# --host is the IP that `npm start` printed on the laptop.
python3 navassist_pi_camera.py --host 192.168.1.42

# Production (AWS). --tls skips cert verification (self-signed, same as the browser warning).
python3 navassist_pi_camera.py --host rse3106.duckdns.org --port 443 --tls
```

You should see `Connected to relay server.` printed on the Pi, `[pi] 15 fps ...` ticking on the
server, and the web app's status bar should show "Glasses connected."

---

## The pages

| Page | What it's for |
|---|---|
| `index.html` | **The real app.** Pi feed + both models + GPS + state machine + speech. |
| `pi.html` | Pi camera test bench — the same feed and both models, with an FPS/latency HUD and a live confidence slider, but no GPS or state machine. **Use this to debug the vision stack.** |
| `webcam.html` | Same two models against a **laptop webcam**, no Pi involved. See `WEBCAM_REALTIME.md`. |
| `test.html` | Crossing model against an **uploaded video file**. |
| `model_test.html` | Single-image `pedestrian.onnx` debugging, with a very low confidence threshold. |
| `settings.html` | App settings. |

---

## Message protocol

The Pi's camera frames are **binary WebSocket frames** — raw JPEG bytes, no envelope, no
base64. Everything else is JSON in this envelope:
```json
{ "topic": "...", "timestamp": 1234567890, "payload": { ... } }
```

| Topic | Direction | Description |
|---|---|---|
| *(binary frame)* | Pi → Server → Browser | JPEG frame, raw bytes. Not JSON. |
| `connection/event` | Server → Browser | `pi_connected` / `pi_disconnected`, plus the `rotate` angle the server applies |
| `crossing/result` | Server → Browser | `crossing_seg.onnx` output for the latest frame: light state, corridor heading, masks |
| `system/heartbeat` | Pi → Browser | Keep-alive every 2s |
| `haptic/command` | Browser → Server → Pi | Fire left/right motor. The Pi drives these as PWM on **GPIO 12 (right) / GPIO 13 (left)** — see `HAPTICS.md`. |
| `live/config` | Browser → Server | Turn the server-side crossing model on/off for this viewer |

Two topics are **no longer sent by anything**, and are kept only so old code doesn't break:
- `camera/image` — the old base64 frame envelope, replaced by binary frames.
- `imu/orientation` — heading now comes from the phone's compass, not the glasses.

---

## Location data (no API key)

The app has **no API keys to obtain.** Nearby signal-controlled pedestrian crossings and their
street names come from **OpenStreetMap**, queried live through the public **Overpass API**
(`overpass-api.de` and mirrors, tried in order) — free and keyless. The queries are:

- crossings: `node["highway"="crossing"]["crossing"="traffic_signals"](around:R,lat,lon)`
- road name: `way(around:R,lat,lon)["highway"]["name"]`

The only location input the app needs is the phone's own **GPS** (`navigator.geolocation`), which
supplies the `lat,lon` those queries are centred on.

> An earlier version used **Google Places** for this and required a Google Maps API key. It was
> replaced because Places is a business directory — it kept returning shops instead of crossings.
> Any `GOOGLE_MAPS_API_KEY` still shown in `.env.example` is leftover and unused.
