# Real-time webcam mode (`webcam.html`) — notes

A page that runs both of the project's AI models in real time against a **laptop/PC webcam**,
with no Raspberry Pi, no WebSocket relay, and no phone involved.

> **Read `PI_REALTIME.md` first if you're working on the real app.** The Pi is the actual
> hardware; `index.html` runs off it. This page is a **development convenience** — it exists so
> you can exercise the vision stack on a laptop when there's no Pi on your desk.

| File | Role |
|---|---|
| `public/webcam.html` | The page: video element, two stacked overlay canvases, controls |
| `public/js/webcam.js` | Grabs webcam frames, drives `pedestrian.onnx`, starts the crossing model |

---

## Why this exists

The stock app gets its camera frames from the Pi over a WebSocket and its heading from the
phone's compass. Without a Pi you get no frames, and without a phone you get no compass — so on
a bare laptop you can't exercise the perception stack at all.

`webcam.html` bypasses both and feeds the **webcam** into the same two models.

If you *do* have the Pi on hand, prefer **`pi.html`** instead: it's the same idea but against
the real camera, so what you see is what the real app sees (same lens, same mounting angle, same
frame size, same JPEG artefacts). `webcam.html` is the fallback when the Pi isn't there.

---

## The project has TWO models, in two different places

This trips people up, so it's worth stating plainly.

**1. `public/models/pedestrian.onnx`** — classes `['red', 'green', 'traffic-light']`.
Runs **in the browser**, via `onnxruntime-web`, inside a Web Worker
(`public/js/inference.worker.js`) so inference never blocks the video or the UI. Tries WebGPU
first, falls back to WASM. The runtime is **self-hosted** under `public/vendor/onnxruntime/`
(deliberately not a CDN, so it works with no internet).

**2. `public/models/crossing_seg.onnx`** — a YOLO11-**seg** model, classes
`['dotted line', 'pedestrian light']`. Runs **on the Node server**, via `onnxruntime-node`, on a
worker thread. It is stateless per frame; all temporal decisions live client-side.

`webcam.html` runs **both at once, against the same webcam feed.**

### How the crossing model gets its frames here — and why that differs from the Pi

This is the one real asymmetry between this page and the Pi path, and it's worth understanding
before you "fix" anything:

- **`webcam.html`** POSTs a JPEG to **`POST /api/infer`** and gets a result back. It has to:
  the frame originates in the browser, and the server has never seen it.
- **`index.html` / `pi.html`** do **not**. The server already *has* those frames — the Pi sent
  them — so it infers on them unprompted and **pushes** `crossing/result` down the `/live`
  socket. Making the browser re-encode and upload a frame the server already holds would be a
  pointless round trip.

Both paths reach the same `crossing_infer.js`. Only the delivery differs. `/api/infer` remains
in use by `webcam.html`, `live.html`, and `test.html`.

---

## How `webcam.js` works

1. `getUserMedia()` → a `<video>` element (`#feed`).
2. On every animation frame (`renderLoop`), it sizes the overlay canvas to the video's native
   pixel dimensions and calls `maybeInfer()`.
3. `maybeInfer()` is throttled two ways, and both matter:
   - `INFER_EVERY_MS = 100` — a *floor* between inferences, so a fast machine doesn't peg a core.
   - `inferBusy` — a hard guarantee that only **one** inference is in flight at a time. On a slow
     machine this is what actually paces things; the floor never bites. This is why a slow model
     degrades to a lower frame rate instead of building an ever-growing backlog.
4. `preprocess()` letterboxes the current video frame into a 640×640 CHW `Float32Array` (grey
   `rgb(114,114,114)` padding, matching Ultralytics).
5. The buffer is **transferred** (zero-copy, not cloned) to `inference.worker.js`, which runs
   `session.run()`, decodes the YOLO output, does NMS, and posts back `{ detections, inferMs }`.
6. `draw()` renders boxes, a cyan arrow from bottom-centre (the "user") to the strongest green
   man, and a HUD.

The crossing model needs none of this plumbing: `crossing.js` exposes
`window.Crossing.start(videoEl, overlayEl, onStatus)` and runs its own capture → `/api/infer` →
draw loop. `webcam.js` just hands it the same `<video>` and a **second, separate canvas**, so the
two models' overlays never fight over one drawing context.

### Coordinate spaces (important if you touch the drawing code)

The two overlays work in **different pixel spaces**, and that is fine:

- `#detOverlay` is sized to `video.videoWidth × videoHeight`, because `inference.worker.js`
  returns boxes already mapped back to original-frame pixels (it undoes the letterbox using
  `scale`/`padX`/`padY`).
- `#crossOverlay` is sized by `crossing.js` to the server's reported `r.w × r.h`, which is the
  **downscaled** capture (`CAP_W = 960`).

Both canvases are CSS-stretched over the same video box, so they line up visually despite the
differing internal resolutions. Don't "fix" this by forcing them to match — you'd break one of
the two mappings.

### Speech muting

`crossing.js` speaks via `window.navassist.speak` if it exists, else falls back to
`speechSynthesis` directly. `webcam.js` therefore **defines** `window.navassist.speak` itself,
and that function is what the mute checkbox gates. That's why muting works without editing
`crossing.js`.

### No rotation here

`ai.js` un-rotates incoming Pi frames (the glasses camera is mounted sideways; the server is the
source of truth for the angle — see `PI_REALTIME.md`). A webcam is already upright, so
`webcam.js` does **no** rotation. If you ever point this at a sideways camera, that's what to
add.

---

## Running it

```bash
cd RSE3106_WebApp-main        # NB: the zip extracts a NESTED folder of the same name.
                              # The real root is the one containing server.js + package.json.
npm install
npm start
```

Then open **http://localhost:3000/webcam.html** and click "Start camera".

You do **not** need a `.env` file. `webcam.html` never calls `/api/config`, and no API keys are
involved — the Overpass/GPS/Google-Maps paths belong to `index.html` only.

### Controls on the page

- **Crossing model (server)** — uncheck to disable `/api/infer` and give `pedestrian.onnx` the
  whole machine.
- **Mute speech** — silences the crossing FSM's spoken cues.
- **HUD** — top-left readout: `ready | backend | N det | fps | inference ms | conf threshold`.
- **Confidence slider** — live-adjustable detection threshold. **Use this before concluding the
  model is broken.** The default is 0.35, but `model_test.html` drops its threshold to 0.08
  specifically to debug near-misses, which suggests this model can score low on real scenes. If
  you see no boxes, slide it down.

---

## Gotchas

- **`getUserMedia` needs a secure context.** `localhost` counts as secure, so plain HTTP works
  there. Opening the page from a phone at `http://<lan-ip>:3000` will fail to get the camera at
  all. For local webcam work, just stay on `localhost`.
- **WebGPU also needs a secure context**, for the same reason. On `localhost` you should get
  `webgpu` in the HUD; over LAN HTTP it silently falls back to `wasm` — slower, still functional.
- **First load stalls for a few seconds.** `pedestrian.onnx` is ~38 MB and must download before
  the worker signals ready. Normal, not a hang.
- **The `<video>` uses `object-fit: cover`**, so a webcam whose aspect ratio differs from 16:9 is
  visually cropped while the model still sees the full uncropped frame. Cosmetic, but it can make
  a box look slightly misplaced near the edges.

---

## History

This page was originally added as a self-contained two-file addition that touched nothing else.
**That is no longer true**, and the old version of this document claiming so is out of date:

- `crossing.js` gained `startExternal()` and `push()` so `pi.js` could drive it from
  WebSocket-pushed results instead of its own `/api/infer` loop. `webcam.js`'s use of
  `Crossing.start()` is unaffected.
- `server.js` gained the `/pi` and `/live` paths and moved the crossing model onto a worker
  thread (`crossing_worker.js`).
- `app.js` and `ai.js` were migrated onto the Pi's binary `/live` feed.

`webcam.html` and `webcam.js` themselves still work exactly as described above.

**Verified:** `npm install` is clean (`onnxruntime-node` and `sharp` both ship Windows prebuilts,
no native compile step); `crossing_seg.onnx` loads at server start; `POST /api/infer` returns a
well-formed result; `pedestrian.onnx`'s graph output is `[1, 7, 8400]`, so the worker's
`nCls = dims[1] - 4` derives **3**, exactly matching `['red','green','traffic-light']` — the main
thing that could have silently produced garbage boxes.

**Not verified:** the live webcam render — camera permission prompt, box drawing against a real
scene, detection quality. Driving a physical webcam wasn't possible in that session.
