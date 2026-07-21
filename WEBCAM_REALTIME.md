# Real-time webcam mode (`webcam.html`) — notes

A page that runs the project's AI model in real time against a **laptop/PC webcam**, with no
Raspberry Pi, no WebSocket relay, and no phone involved.

> **Read `PI_REALTIME.md` first if you're working on the real app.** The Pi is the actual
> hardware; `index.html` runs off it. This page is a **development convenience** — it exists so
> you can exercise the vision stack on a laptop when there's no Pi on your desk.

| File | Role |
|---|---|
| `public/webcam.html` | The page: video element, two stacked overlay canvases, controls |
| `public/js/webcam.js` | Grabs the webcam, hands the `<video>` to `crossing.js`, draws a small HUD |
| `public/js/crossing.js` | The actual pipeline: captures frames, POSTs to `/api/infer`, draws the overlay, runs the crossing FSM |

---

## Why this exists

The stock app gets its camera frames from the Pi over a WebSocket and its heading from the
phone's compass. Without a Pi you get no frames, and without a phone you get no compass — so on
a bare laptop you can't exercise the perception stack at all.

`webcam.html` bypasses both and feeds the **webcam** into the model.

If you *do* have the Pi on hand, prefer **`pi.html`** instead: it's the same idea but against
the real camera, so what you see is what the real app sees (same lens, same mounting angle, same
frame size, same JPEG artefacts). `webcam.html` is the fallback when the Pi isn't there.

---

## The project has ONE model, on the server

This used to be two models; it is now one. Worth stating plainly because older notes (and older
code) assumed two.

**`public/models/crossing_seg.onnx`** — a YOLO11-**seg** model, classes
`['dotted line', 'pedestrian light']`. Runs **on the Node server**, via `onnxruntime-node`, on a
worker thread. It is stateless per frame; all temporal decisions live client-side. It does the
whole perception job: it segments the dotted-line crossing corridor *and* detects the pedestrian
light, whose red/green/clearance state the server reads from the lit pixels (HSV).

There is **no in-browser model**. An earlier `pedestrian.onnx` (classes `red`/`green`/
`traffic-light`) used to run here in a Web Worker (`js/inference.worker.js`) via
`onnxruntime-web`; it was a subset of `crossing_seg.onnx`, so it and its worker were removed. The
browser now runs no neural net — it only captures webcam frames, POSTs them, and draws.

### How the crossing model gets its frames — and why that differs from the Pi

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

It is deliberately thin now that there is no browser model to drive:

1. `getUserMedia()` → a `<video>` element (`#feed`).
2. Hands that `<video>` (and the `#crossOverlay` canvas) to **`window.Crossing.start()`** from
   `crossing.js`. From there `crossing.js` owns everything: it captures frames off the video,
   POSTs them to `/api/infer`, draws the masks / light box / corridor arrow / banner onto
   `#crossOverlay`, and runs the crossing decision FSM.
3. `webcam.js`'s own `renderLoop` does just two things: size the `#detOverlay` canvas to the
   video, and draw a small **fps HUD** on it. No inference, no boxes.
4. It also defines `window.navassist.speak` — that's the hook `crossing.js` speaks through, and
   gating it is how the **Mute speech** checkbox silences the FSM's spoken cues without touching
   `crossing.js`.

All the heavy lifting — capture throttling (`INFER_MS` floor + one request in flight at a time),
letterbox, decode, NMS, the light-state read — happens in `crossing.js` and on the server, not
here.

### Coordinate spaces (important if you touch the drawing code)

The two overlay canvases work in **different pixel spaces**, and that is fine:

- `#detOverlay` is sized to `video.videoWidth × videoHeight` and now carries **only the HUD**.
- `#crossOverlay` is sized by `crossing.js` to the server's reported `r.w × r.h`, which is the
  **downscaled** capture (`CAP_W = 960`).

Both canvases are CSS-stretched over the same video box, so they line up visually despite the
differing internal resolutions. Don't "fix" this by forcing them to match — you'd break the
crossing overlay's mapping.

### No rotation here

`ai.js` un-rotates incoming Pi frames (the glasses camera is mounted sideways; the server is the
source of truth for the angle — see `PI_REALTIME.md`). A webcam is already upright, so nothing on
this page rotates. If you ever point this at a sideways camera, that's what to add.

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

- **Start camera** — asks for webcam permission and begins.
- **Mute speech** — silences the crossing FSM's spoken cues.
- **HUD** — top-left readout on `#detOverlay`: `crossing_seg (server) · fps`.

The confidence threshold is **not** adjustable from this page anymore — it's a server-side
constant (`CONF` in `crossing_infer.js`, default `0.30`), since the model runs on the server. If
the model scores low on a real scene and you want to chase near-misses, lower `CONF` there and
restart the server.

---

## Gotchas

- **`getUserMedia` needs a secure context.** `localhost` counts as secure, so plain HTTP works
  there. Opening the page from a phone at `http://<lan-ip>:3000` will fail to get the camera at
  all. For local webcam work, just stay on `localhost`.
- **No in-browser model anymore**, so the old WebGPU / WASM / ~38 MB-model-download gotchas are
  gone. If the overlay is blank, the thing to check is that the **server is running** (the model
  loads there at startup — `[infer] model loaded …` in the server log) and that `/api/infer` is
  returning results (watch the Network tab, or use `model_test.html`).
- **The `<video>` uses `object-fit: cover`**, so a webcam whose aspect ratio differs from 16:9 is
  visually cropped while the model still sees the full uncropped frame. Cosmetic, but it can make
  a box look slightly misplaced near the edges.

---

## History

This page began as a self-contained two-file addition (`webcam.html` + `webcam.js`) that ran a
browser-side `pedestrian.onnx` and layered the server crossing model on top. It has changed twice
since:

- `crossing.js` gained `startExternal()` and `push()` so `pi.js` could drive it from
  WebSocket-pushed results instead of its own `/api/infer` loop. `webcam.js`'s use of
  `Crossing.start()` is unaffected.
- `server.js` gained the `/pi` and `/live` paths and moved the crossing model onto a worker
  thread (`crossing_worker.js`).
- **The browser `pedestrian.onnx` and its worker were removed.** `crossing_seg.onnx` is a
  superset, so `webcam.js` was reduced to a thin webcam-capture + HUD shell around `crossing.js`,
  which is now the entire perception pipeline on this page.

---

**Verified:** `npm install` is clean (`onnxruntime-node` and `sharp` both ship Windows prebuilts,
no native compile step); `crossing_seg.onnx` loads at server start; `POST /api/infer` returns a
well-formed result (`{ w, h, light, lights, dotted, corridor, signals, inferMs }`) for a posted
frame — confirmed by driving the endpoint directly.

**Not verified:** the live webcam render — camera permission prompt, `crossing.js` drawing
against a real scene, detection quality. Driving a physical webcam wasn't possible in that
session.
