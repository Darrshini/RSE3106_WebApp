# Crossing guidance arrows — the misleading light arrow is gone; corridor reliability remains

**Status: the "two arrows, one pointing at the traffic light" problem is now RESOLVED IN
SOFTWARE** — but for a structural reason, not a targeted fix: the browser `pedestrian.onnx` model
that drew the light-referenced arrow was removed entirely (the app is now one model,
`crossing_seg.onnx`, on the server — see `README.md` / `PI_REALTIME.md`). What is still *not*
fixed is the underlying camera-angle problem that makes the remaining, correct arrow unreliable.
This doc is context for the next session (human or Claude).

## The original observation (2026-07-15)

While actually crossing at a signalised junction, the app drew **sometimes two cyan arrows**, and
they **mostly pointed toward the traffic light** rather than following the painted pedestrian
crossing. The expectation was that guidance would reference the **crossing itself** (the dotted
lines / zebra you walk on), not the light across the road.

## Why there were two arrows — and why there is now only one

Back then, `index.html` overlaid **two independent cyan arrows** from two different models:

| Arrow | Drawn by | Referenced | Status now |
|---|---|---|---|
| **Green-light arrow** | `pedestrian.onnx` (browser), in `ai.js` `drawOverlay()` | the **traffic light** | **GONE** — `pedestrian.onnx` and all browser inference were removed. `drawOverlay()` no longer draws any arrow; it draws only the HUD and a `GREEN — GO` **banner** (text, no arrow). |
| **Corridor arrow** | `crossing_seg.onnx` (server), in `ai.js` `drawCrossingOverlay()` | the **pedestrian crossing** | **KEPT** — this is the one we want. Built server-side in `crossing_infer.js` `corridor()` from the dotted lines' vanishing point: user(bottom-centre) → the far kerb between the boundary lines. |

So the misleading light-referenced arrow simply no longer exists. The only arrow the app can draw
now is the corridor arrow, which references the crossing by construction. The consolidation onto
`crossing_seg.onnx` removed the source of the confusion as a side effect.

## What is still not fixed: the corridor arrow is often absent

Removing the wrong arrow doesn't guarantee the right one is *there*. The corridor arrow only draws
when `crossing_seg.onnx` gets a clean read of the dotted lines (`r.corridor.has === true`), and a
**forward-facing glasses camera barely sees the ground crossing:**

- The painted crossing is on the **ground**, seen at a shallow grazing angle → foreshortened, low
  in the frame or below it. So the dotted-line segmentation frequently fails, `r.corridor.has` is
  false, and **no arrow draws at all**.
- The traffic light, by contrast, sits across the road at ~eye level → sharp and central for a
  forward-aimed, infinity-focused camera. (This asymmetry is exactly why the light used to
  dominate.)

So the current visible-guidance failure mode has flipped from *"an arrow pointing at the wrong
thing"* to *"no arrow, because the crossing isn't in frame."*

## What already works (don't "fix" this part)

The **haptic guidance while crossing is not light-based**, and never was. During `STATES.CROSSING`:
- `onCorridorDirection` reads the dotted-line corridor angle (`r.corridor.angleDeg`) and nudges
  the matching side — `app.js`.
- Phone-compass **drift correction** (`checkHeadingDrift`, `app.js`) buzzes the user back toward
  the heading captured when they started crossing.
These share one cooldown so they never fire competing pulses. When the dotted lines *are* visible
the corridor haptic leads; when they aren't (the camera problem above), the compass drift
correction still covers it. So the crossing haptics degrade gracefully even when the visible
corridor arrow is missing.

## Options for next time (pick with the user)

1. **Hardware — angle the camera down a few degrees.** This is now the *main* open item. Getting
   the ground crossing into frame is what makes `crossing_seg.onnx` segment the dotted lines
   reliably, which is what makes the corridor arrow (and the corridor haptic) actually appear.
   Currently the mount aim is unconfirmed — ask the user: level/forward, or tilted down?
2. **Software — nothing further needed for the "wrong arrow" itself.** It's gone with the model.
   If anything, consider whether the `GREEN — GO` banner in `ai.js` `drawOverlay()` should be
   suppressed during `STATES.CROSSING` (it's a WAITING-phase cue), but it's a text banner, not a
   misleading arrow, so this is minor.

## Key files, for orientation

- `public/js/ai.js` — `drawCrossingOverlay()` draws the dotted-line masks, the corridor arrow,
  and the light box coloured by state (the crossing overlay). `drawOverlay()` now draws only the
  HUD + `GREEN — GO` banner (no arrow, no boxes — the browser model is gone).
  `handleCrossingPerception()` routes `crossing/result` into the FSM.
- `public/js/app.js` — the state machine + all haptic decisions (`onCorridorDirection`, compass
  drift). Arrows are drawn by `ai.js`; **decisions/haptics live here.**
- `crossing_infer.js` — server-side: builds the corridor vector from the dotted-line segmentation
  (`corridor()`), returns `{ near, far, angleDeg, lightAgrees }`, plus the pedestrian-light box +
  state that replaced everything `pedestrian.onnx` used to provide.
- `PI_REALTIME.md` / `HAPTICS.md` — the camera path and the haptic transport, for background.
