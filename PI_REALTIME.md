# The Raspberry Pi camera path — design notes

Context for whoever picks this up next (human or Claude). This is the **main** camera path:
it is what `index.html` runs on. It replaced the ESP32/base64 protocol.

Hardware: **Raspberry Pi Zero 2W + Camera Module v3**, worn on the glasses, mounted sideways.

---

## The one-paragraph version

The Pi captures JPEGs with its hardware encoder and pushes the raw bytes over a WebSocket. The
Node server relays those bytes untouched to every browser watching, and — on a worker thread, on
the same frame it just relayed — runs the crossing segmentation model and pushes the result down
the same socket. The browser (`index.html`) just draws the frame and acts on the result; it runs
**no model of its own**. Nobody re-uploads a frame to anybody.

```
Pi Zero 2W                      Node server                     Browser (index.html)
──────────                      ───────────                     ────────────────────
picamera2                                                       app.js  — socket, GPS, FSM
  └ hardware MJPEG              /pi  ──┬── relay bytes ──►      ai.js   — decode, rotate, draw
      └ raw JPEG bytes ─────────►      │                          (no browser inference)
        (binary WS frame)              └── crossing_worker.js
                                            └ crossing_seg.onnx
        ◄──────── haptic/command ────────┘        │
                                                  └── crossing/result ──► /live
```

## Which model runs where, and why not on the Pi

The real app (`index.html`) uses a **single model, on the Node server** — the browser runs none:

| Model | Classes | Where | Used by |
|---|---|---|---|
| `crossing_seg.onnx` | dotted line, pedestrian light | **Node server** — onnxruntime-node in `crossing_worker.js`, a worker thread | **`index.html` (the real app)**, plus `pi.html` / `webcam.html`. The server derives the light state (red / green / flashing) and the crossing corridor from it and pushes both as `crossing/result`. |
| `pedestrian.onnx` | red, green, traffic-light | **Browser** — onnxruntime-web in `js/inference.worker.js`, WebGPU → WASM, self-hosted under `public/vendor/onnxruntime/` (no CDN, works offline) | **Dev/debug pages only** (`pi.html`, `webcam.html`, `model_test.html`). `index.html` no longer runs it. |

**Neither runs on the Pi, and neither should.** A Zero 2W is 4× Cortex-A53 @1GHz with 512 MB of
RAM. A YOLO11 pass on it takes seconds. Moving inference there wouldn't speed the system up, it
would stop it working. The Pi does the one thing its silicon is genuinely good at — capture and
JPEG-encode — and ships the bytes.

The crossing model is on a **worker thread**, not the main thread, and that matters: its mask
decode blocks the event loop, which would stall the WebSocket camera relay behind every single
inference. See the header of `crossing_worker.js`.

---

## The decisions that make it real-time

Change any of the first three casually and the lag comes straight back.

### 1. Binary frames, not base64-in-JSON
The old ESP32 protocol wrapped every JPEG in `{"topic":"camera/image","payload":"<base64>"}`.
base64 inflates each frame by **33%**, and both ends then have to parse a ~100 KB string per
frame. On a Zero 2W — 2.4 GHz-only WiFi, sharing its antenna budget with everything else — that
inflation **is** the throughput ceiling. Now: raw bytes on the wire, `createImageBitmap` in the
browser (which decodes *off* the main thread), and the server reads them straight into the
model.

### 2. No compression on the WebSocket
Python's `websockets` negotiates `permessage-deflate` by default, which would try to
DEFLATE-compress JPEG bytes that are **already entropy-coded**. Real CPU on a weak core, for
approximately zero saving. Turned off explicitly (`compression=None`).

### 3. Latest frame wins, everywhere
Every stage drops stale frames rather than queueing them, and this is the single most important
property in the whole pipeline:

| Stage | Mechanism |
|---|---|
| Pi encoder → sender | `FrameBus`: a **single-slot** mailbox. The encoder overwrites; the sender takes whatever is there. |
| Server → viewers | `bufferedAmount > frame size` ⇒ skip this client for this frame (`broadcastLiveBinary`). |
| Server → crossing model | `latestFrame` is one slot, not a queue (`pumpCrossing`). |
| Browser decode | `decodeInFlight` + `pendingFrame`: one decode at a time, newest wins (`ai.js`). |
| Browser inference | `inferBusy`: one inference in flight at a time (`ai.js`). |

WebSocket is reliable and ordered, so a consumer that can't keep up **does not drop frames — it
accumulates them**, and the picture drifts steadily further behind reality the longer you run.
That is the exact failure people misdiagnose as "the camera is slow." Dropping is always the
right call here: a fresher frame is already on its way.

### 4. Capture at 960×720 — the aspect ratio matters more than the pixel count

Both models letterbox their input to 640×640, and the frame is rotated **upright** before
inference. That makes the capture's **shape** matter more than its size, because the horizontal
pixels are exactly the ones letterboxing throws away:

| Capture | Rotated | What the model actually gets |
|---|---|---|
| **960×720 (4:3)** — *default* | 720×960 | **480×640** — full width, plus real extra detail |
| 640×480 (4:3) | 480×640 | **480×640**, at scale 1.0 — pixel for pixel, no downscale |
| 1280×720 (16:9) | 720×1280 | **360×640** — *less* across, for *more* bytes |

So a "higher resolution" 720p (16:9) capture is actively **worse** for detection: it hands the
model 360px across where 4:3 gives 480px, and charges ~1.3× the bytes over the Pi's weakest link
for the privilege. 960×720 is the sweet spot — full width for the model, and more real detail
than VGA in which to find small distant lights.

```bash
python3 navassist_pi_camera.py --width 640 --height 480     # cheapest on WiFi
python3 navassist_pi_camera.py --width 1280 --height 720    # 16:9, nicer-looking demo only
```

### 5. Bitrate scales with resolution — or the picture goes grey
`MJPEGEncoder` takes a **bitrate, not a quality**: it hits the number you give it no matter what
that costs the picture. Starve it and JPEG discards **chroma first** — the colour channels get
quantised into near-nothing while luma survives, so the image doesn't go blocky, it goes
**washed-out and grey**. A hard-coded bitrate silently rots the moment anyone changes the
resolution:

```
4 Mbit/s @  640x480 x15  = 0.87 bits/pixel   fine
4 Mbit/s @ 1280x720 x15  = 0.29 bits/pixel   grey mush
```

So it's derived instead — `width × height × fps × 0.9 bits/pixel`, clamped to [2, 16] Mbit/s.
At the 960×720 default that's **9.3 Mbit/s**. The script prints what it picked and warns if it
lands below 0.5 bits/pixel (grey) or above 8 Mbit/s (more than the Zero 2W's 2.4 GHz WiFi
comfortably carries). Override with `--bitrate` if you need to.

### 6. Auto exposure and auto white balance, pinned on
Explicitly enabled (`AeEnable` / `AwbEnable`), not left to inheritance. This is a wearable that
walks from shade into direct sun: if AE were ever off, the frame blows out or goes black the
moment the light changes and the model stops seeing the light it needs. AE metering is
**centre-weighted**, so it exposes for what the wearer is facing rather than for a bright sky
filling the top of the shot — which would otherwise silhouette the traffic light into
uselessness. AWB matters for the same reason: *"is that light red or green"* is a colour
question, and drifting white balance is exactly what would corrupt the answer.

Note that `FrameRate` sets the exposure **ceiling**: at 15fps the AE loop can integrate for at
most ~66ms. That's the right way round for us — a long exposure would motion-blur a walking
wearer's frame, and a blurred pedestrian light is a missed one. AE trades to gain instead.

---

## Rotation: who turns the picture upright

The camera is mounted sideways, so frames land 90° off. Three places could rotate them, and
only one of them is wrong:

- **The Pi: no.** libcamera's hardware transform can flip and do 180°, but **not a quarter
  turn**. A 90° rotation there means pulling the frame into numpy, transposing, and re-encoding
  in software — which throws away the hardware JPEG encoder, i.e. the entire reason a Zero 2W
  can keep up. The frame stays exactly as the sensor produced it.
- **The server: yes, free.** It rotates inside its `sharp` pipeline, which decodes the JPEG
  anyway.
- **The browser: yes, free.** It rotates on the canvas; the GPU does it.

`PI_ROTATE` on the server (default **90° CW**, override with the env var) is the **single source
of truth**. The server sends its value to each browser in the `connection/event` payload on
connect, and `app.js` hands it to `ai.js` via `window.navassist.setFrameRotation()`. That is why
the server's coordinates and the browser's canvas can never silently disagree about which way is
up — which, if it ever did, would land every bounding box sideways.

Remounted the camera upright? Set `PI_ROTATE=0`. Nothing else needs touching.

---

## Wire protocol

Camera frames are **binary** WebSocket frames — raw JPEG bytes, no envelope. Everything else is
JSON: `{ topic, timestamp, payload }`.

| Path | Who | Carries |
|---|---|---|
| `/pi` | the Raspberry Pi | binary JPEGs up; `system/heartbeat` up; `haptic/command` down |
| `/live` | browsers (`index.html`, `pi.html`) | binary JPEGs down; `connection/event`, `crossing/result` down; `haptic/command`, `live/config` up |

- `connection/event` → `{ event: 'pi_connected' | 'pi_disconnected', rotate: 90 }`
- `crossing/result` → `{ w, h, light, dotted, corridor, signals, inferMs }`, in **upright**
  coordinates (the server already un-rotated the frame)
- `live/config` → `{ crossing: false }` tells the **server** to stop inferring for this viewer.
  It has to stop the server, not just stop the browser drawing: you're probably viewing the page
  on the same laptop that runs the server, so an unwanted seg model still steals CPU from
  `pedestrian.onnx`.

`/browser` and `/esp32` still exist in `server.js` but **nothing connects to them**. That's the
dead ESP32 path; delete it whenever you like.

---

## Files

| File | Role |
|---|---|
| `navassist_pi_camera.py` | Runs **on the Pi**. Capture → hardware MJPEG → binary WS, plus the haptic motors: `haptic/command` → PWM on **GPIO 12 (right) / GPIO 13 (left)**. Still no inference — `app.js` decides *when* to buzz, the Pi only actuates. |
| `server.js` | `/pi` + `/live` relay, backpressure, `PI_ROTATE`, pumps frames into the crossing worker. |
| `crossing_worker.js` | Worker thread. Keeps `crossing_seg.onnx` off the event loop. |
| `crossing_infer.js` | The actual seg model + `sharp` preprocessing + mask decode. |
| `public/js/app.js` | `/live` socket, binary vs JSON split, GPS, the vision-driven state machine (no tap confirmations; single-tap start/reset only), speech, and haptics — including the steady crossing-guidance cadence and the crossing lock. |
| `public/js/ai.js` | Decode → rotate → draw the frame + the `crossing/result` overlay, and fold that result into the state machine (light state → cues, corridor → guidance). Runs **no browser model**. |
| `public/js/pi.js` | `pi.html`'s test bench. Same feed, same models, plus HUD and confidence slider. |

---

## Running it

```bash
npm install
npm start
```
On the Pi:
```bash
sudo apt install -y python3-picamera2
pip install websockets --break-system-packages
python3 navassist_pi_camera.py --host <the IP npm start printed>
```
Then open `http://localhost:3000` (the real app) or `http://localhost:3000/pi.html` (the bench).

Useful flags: `--fps`, `--width`/`--height`, `--quality`, `--encoder jpeg` (software fallback),
`--tls` (for the AWS box; skips self-signed cert checks).

### The numbers to watch

Both ends print once a second, and together they tell you exactly where a bottleneck is:

- **Pi** — `[net] captured 15  sent 15  dropped  0   118 KB/s`.
  `dropped` climbing means the **link** can't keep up with the camera. Lower `--fps`,
  `--quality`, or `--width` before you go blaming the models.
- **Server** — `[pi] 15 fps  118 KB/s`.
  If this is well below the Pi's configured FPS, the bottleneck is the WiFi link, not inference.
- **Browser HUD** (`pi.html`) — `15 fps  480×640  webgpu 42ms  seg 190ms  2 det`.

---

## Gotchas

- **The nested folder.** The zip extracts `RSE3106_WebApp-main/RSE3106_WebApp-main/`. The real
  root is the one with `server.js` + `package.json` in it.
- **WebGPU needs a secure context.** On `localhost` you get `webgpu` in the HUD. Over plain
  `http://<lan-ip>` it silently falls back to `wasm` — noticeably slower, still functional. The
  AWS box's HTTPS solves this properly.
- **First load stalls for a few seconds.** `pedestrian.onnx` is ~38 MB and must download before
  the worker signals ready. Normal, not a hang.
- **Camera Module v3 focus is pinned at infinity.** The pedestrian light we care about is across
  the road, so the script sets manual focus at infinity (`AfMode=Manual`, `LensPosition=0.0`
  dioptres) rather than letting continuous AF hunt onto near clutter — the wearer's own body, a
  passing hand — and get caught mid-hunt (blurred) exactly when a light appears. Infinity is sharp
  from a few metres out to the horizon, which is the whole range that matters. It prints
  `[cam] focus fixed at infinity (manual, LensPosition 0.0)`. On a v2 (fixed focus) the control is
  unavailable and skipped, which is fine — a v2 is already focused near infinity.
- **Seeing no detections?** Try `pi.html` and slide the confidence threshold down before
  concluding anything is broken. `model_test.html` drops its threshold to 0.08 specifically to
  chase near-misses, which tells you this model can score low on real scenes.

---

## What has been verified, and what hasn't

**Verified end-to-end** (Windows 11, Node v24.18.0) by standing up the server and driving `/pi`
and `/live` with a simulated Pi pushing real JPEGs at 15fps:

- `connection/event` arrives with `{ event: 'pi_connected', rotate: 90 }`.
- Binary frames relay to the viewer (52 frames over 4s, ~13fps sustained).
- `crossing_seg.onnx` loads on the worker thread and pushes `crossing/result` unprompted —
  14 results over the same 4s — with **no upload from the browser**.
- Results come back as `480×640`: the sideways 640×480 sensor frame, correctly rotated upright.
  So the server's coordinate space and the browser's canvas agree.
- `haptic/command` sent by the browser on `/live` **reaches the Pi** on `/pi`.
- Heartbeats relay. No errors in the server log.

**Not verified:** the live render against a real scene — real camera, real crossing, real
detection quality. Driving physical hardware wasn't possible in that session. The plumbing is
proven; the perception isn't. If detections look wrong, suspect the confidence threshold (above)
before suspecting the decode.
