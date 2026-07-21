# Haptics — what was built, and how to test it

**Status: the motors + `HapticController` are bench-tested and working (`test_haptic.py`).
What is NOT yet proven is the full networked path through `navassist_pi_camera.py` — server →
`/pi` → this script → motor.** Read "How to test" below and start there.

Updated 2026-07-21. Hardware: two ERM vibration motors. The pin mapping lives in **`haptic.py`**
(`HapticController`), not in this file.

---

## The one-paragraph version

The haptic *signal* already existed and already reached the Pi — `app.js` decides when to
buzz, `server.js` relays it, and the Pi script receives it. `navassist_pi_camera.py` now hands
each `haptic/command` to **`HapticController` from `haptic.py`** — the *same* controller
`test_haptic.py` and `main.py` drive, so the pin mapping and PWM that were already proven on the
bench are exactly what runs here. `navassist_pi_camera.py` no longer has its own motor driver; it
only relays. Nothing about the vision stack changed.

```
app.js  sendHaptic()          server.js  sendToPi()        navassist_pi_camera.py
  │  decides WHEN + which side    │  relays, logs             │  read_commands()
  └─ 'haptic/command' ──/live──►  └──────────/pi──────────►   └─ HapticCommandRunner.handle()
       {motor,pattern,                                             └─ HapticController.buzz()  ← haptic.py
        intensity,duration_ms}                                          (pins + PWM live here)
```

The Pi still makes **no decisions**. It was a camera; it is now a camera and two motors, and the
motor half is your bench-tested `haptic.py`.

> **Why the rewrite:** `navassist_pi_camera.py` used to carry its own `HapticDriver` (raw
> `gpiozero` + a duty window), which had **never touched hardware**. Your `haptic.py`
> `HapticController` *had* — proven with `test_haptic.py`. So the untested driver was removed and
> this script now delegates to the proven one, using the same `asyncio.to_thread(buzz, …)` +
> cooldown pattern your `main.py` already runs.

---

## How to test

Three terminals. The point of the logging is that you can watch one button-press travel the
whole chain and see exactly where it stops.

### 1. Get the script onto the Pi — *next to `haptic.py`*

`navassist_pi_camera.py` does `from haptic import HapticController`, so it must sit in the **same
directory** as `haptic.py` (alongside `main.py` / `test_haptic.py`):

```bash
scp navassist_pi_camera.py pi@<pi-ip>:~/     # same folder as haptic.py
```

One-time, on the Pi (needed by `haptic.py`):
```bash
sudo apt install -y python3-gpiozero python3-lgpio
```

### 2. Server (laptop)

```bash
cd "C:\UniThings\Yr2Tri3\SEP2 things\RSE3106_WebApp"   # the folder with server.js
npm start
```
Note the LAN IP it prints. Keep this terminal visible — it is checkpoint #2.

### 3. Pi (over SSH)

```bash
python3 navassist_pi_camera.py --host <the-IP-npm-start-printed> --haptic-test
```

`--haptic-test` buzzes left → right → both at startup and **then carries on streaming
normally**, so you know the motor half works before a single frame moves. Expect:

```
[haptic] HapticController ready (haptic.py)
[haptic] self-test: LEFT
[haptic] self-test: RIGHT
[haptic] self-test: BOTH
[haptic] self-test done. Felt on the wrong side? Fix the pin mapping in haptic.py.
[cam] 960x720 @ 15fps, mjpeg encoder
Connected to relay server.
[net] captured 15  sent 15  dropped  0   118 KB/s
```

**Motors feel swapped?** The `--haptic-*-pin` flags are gone — the side mapping lives in
`haptic.py` now, so fix it there (the same place `test_haptic.py` proved it).

### 4. The bench

Open **http://localhost:3000/pi.html**. Camera feed, traffic-light boxes, dotted-line masks,
and a **Haptic test** row: Left / Right / Both.

### 5. Press Left, and watch it travel

| # | Where | What you should see |
|---|---|---|
| 1 | Browser (status line + console) | `Haptic: sent left @0.8 for 400ms` |
| 2 | Server (`npm start` terminal) | `[WS] browser -> Pi  haptic/command  {"motor":"left",...}` |
| 3 | Pi (SSH terminal) | `[cmd] haptic: {'motor': 'left', ...}` then `[haptic] left intensity=0.80 duration=400ms pattern=pulse` |
| 4 | The glasses | Left motor buzzes for ~400 ms (intensity → PWM per `haptic.py`) |

**Wherever the sequence stops is the fault line.** That is what the logging is for:

| Last thing you see | What's wrong |
|---|---|
| Nothing in the browser | `/live` socket is down — check the status dot |
| Browser: "Pi is not connected" | Pi isn't on `/pi` — go back to step 3 |
| Server: `Cannot send to Pi -- not connected (dropped haptic/command)` | Server sees the browser but not the Pi |
| Server logs the relay, but no `[cmd]` on the Pi | Network/socket between server and Pi |
| `[cmd]` prints but no `[haptic]` line | `HapticController` didn't init — see the `[haptic] unavailable (…)` line at startup (missing `haptic.py`, or `--no-haptics`) |
| `[haptic]` prints, no buzz | GPIO/wiring/`haptic.py` — but `--haptic-test` should already have caught that |

---

## What this test does NOT prove

The buttons prove the **transport**: browser → server → Pi → motor. They do **not** prove the
AI triggers a buzz, and this is the thing that already wasted an evening once:

**`pi.html` has no state machine.** No GPS, no compass, **no automatic haptics** — the Left/Right/
Both buttons are the only haptic path on that page. A traffic light detected on `pi.html` will
**never** move a motor on its own. Same for `webcam.html`. That is by design, not a bug.

Haptics only fire automatically from `app.js`, i.e. **`index.html`**, and only in specific states.
The flow is **vision-driven** — there are **no double/triple-tap confirmations**; the only taps
are a single tap to start scanning and a single tap to reset after a crossing:
- `onDirectionDecided` (`app.js:590`) buzzes toward the traffic-light post only while **`NAVIGATING`**
- `onGreenDirection` (`app.js:639`) buzzes the green-man side while **`WAITING`** or **`SCANNING`**
- **Crossing guidance** — while **`CROSSING`**, `crossingHapticTick()` (`app.js:556`) fires a
  steady **0.5 s pulse every 2 s** on the side(s) matching the corridor direction: **both = go
  straight**, one side = veer that way. `onCorridorDirection` (`app.js:656`) feeds it the vision
  direction; compass drift is a fallback when the corridor isn't visible. (Tunables:
  `CROSSING_HAPTIC_INTERVAL_MS`, `CROSSING_HAPTIC_DURATION_MS`, `CROSSING_HAPTIC_INTENSITY`.)

Reaching those states needs **no taps**: a traffic-light *post* advances **SCANNING → NAVIGATING**
automatically, and a green man advances **WAITING → CROSSING** automatically (`onGreenCross`,
`app.js:609`). Once crossing starts the app is **locked into `CROSSING`** (`crossingLocked`,
`app.js:63`) — nothing can pull the state out of it except a genuine finish or losing the glasses,
so a GPS glitch or a stray tap can't interrupt a wearer mid-road.

So opening `index.html` and pointing at a **green man** *does* buzz toward it (via `onGreenDirection`
while `SCANNING`/`WAITING`) — but the steady crossing cadence only begins once you're actually
**`CROSSING`**. The green-direction buzz has its own cooldown
(`GREEN_DIRECTION_HAPTIC_COOLDOWN_MS = 1000`) so it won't fire every frame.

**Get the transport green on `pi.html` first.** If a button buzzes, any failure on
`index.html` is state-machine logic, not plumbing — a much smaller haystack.

---

## Intensity: the mapping lives in `haptic.py` now

`app.js` sends `intensity` as **0.0–1.0** in the `haptic/command` payload. What that becomes on
the pin — the PWM duty, the floor that overcomes an ERM motor's static friction, the comfort
ceiling — is entirely `HapticController.buzz()`'s job in **`haptic.py`**. `navassist_pi_camera.py`
does **not** reshape it; it passes `motor`, `duration_s`, `intensity` straight through, exactly
as `main.py` and `test_haptic.py` do.

What this script *does* still guard, before calling `buzz()`:
- **Zero is off.** `intensity <= 0` (or `duration <= 0`) sends no buzz at all — silence is a
  valid cue.
- **Never buzz forever.** `duration_ms` is clamped to `MAX_PULSE_MS` (3000 ms), so a malformed
  value can't latch a motor on.
- **Bursts are dropped.** A 150 ms cooldown collapses a rapid stream of commands into one buzz
  (see below).

If a low-intensity nudge (`app.js` sends `0.3` for its CENTRE "on track" cue) doesn't start your
motors, or full intensity is unpleasant, that's a `haptic.py` tuning matter now — the same file
`test_haptic.py`'s intensity keys (`1`–`5`) let you feel.

---

## Three things in `HapticCommandRunner` that are load-bearing

Don't "simplify" these without reading why.

1. **It never blocks the event loop.** `HapticController.buzz()` is synchronous and sleeps for
   the whole pulse. Calling it directly would stall the camera relay for that entire time —
   dropping frames every time the user gets a nudge, which is precisely when they can least afford
   it. So it's pushed to a worker thread with `asyncio.to_thread(self.haptic.buzz, …)`, and the
   relay and heartbeats keep flowing. (This is exactly what your `main.py` does.)

2. **It drops bursts.** `app.js`'s `onDirectionDecided` re-fires the NAVIGATING nudge on *every
   frame*, far faster than a motor can give a distinct pulse. A 150 ms cooldown collapses a burst
   into one clean buzz instead of a long, blurred one. No generation counter or stop-task
   bookkeeping is needed, because `buzz()` owns its own on→sleep→off lifecycle.

3. **It fails soft.** No `haptic.py`, no GPIO, or `--no-haptics` → warn, run with haptics
   disabled, **keep the camera running**. A wearer with video and no buzzing is degraded; a
   wearer with neither is blind. And because `buzz()` self-terminates, a link that dies mid-pulse
   needs no explicit stop — the worker thread finishes the pulse and the motor goes quiet on its
   own within `duration_ms`. (That's why the old `stop_all()` on disconnect is gone.)

---

## Files changed

| File | Change |
|---|---|
| `navassist_pi_camera.py` | **Removed the untested `HapticDriver`**; now delegates to `HapticController` from `haptic.py` via a `HapticCommandRunner` (async `handle()` → `asyncio.to_thread(buzz, …)` + 150 ms cooldown), mirroring `main.py`. `read_commands()` `await`s it; `run()` inits the controller fail-soft and closes it on shutdown. |
| `server.js` | `sendToPi()` **logs every relayed topic** (was a silent pass-through — you couldn't tell "browser never sent one" from "Pi never got one"). |
| `public/pi.html` | **Haptic test** row (Left/Right/Both) + a `#hapticState` status line. |
| `public/js/pi.js` | `sendHaptic()` + button wiring. Sends the same envelope `app.js` does. |

CLI flags: **`--haptic-test`** (self-test then stream) and **`--no-haptics`** (camera only). The
old `--haptic-left-pin` / `--haptic-right-pin` / `--haptic-freq` / `--haptic-min-duty` /
`--haptic-max-duty` / `--haptic-active-low` flags are **removed** — those knobs live in `haptic.py`.

---

## Verified / not verified

**Verified:**
- **Motors + `HapticController` on the bench**, via `test_haptic.py` — sides, intensity sweep,
  pulse pattern all work (this is your existing, confirmed-working setup).
- The relay pattern this script uses (`asyncio.to_thread(buzz, …)` + cooldown) is the *same* one
  `main.py` already runs against the same controller.
- `navassist_pi_camera.py` byte-compiles (`python -m py_compile`); `server.js` and `pi.js` pass
  `node --check`.

**NOT verified — do this on the Pi:**
- The **full path through `navassist_pi_camera.py`** end-to-end: server → `/pi` → `read_commands`
  → `HapticCommandRunner` → `buzz`. `main.py` proved this over the *old* `/esp32` base64 protocol;
  this script is the *new* `/pi` binary protocol, so re-run `--haptic-test` + the `pi.html` buttons
  to confirm the swap didn't break the relay.
- The `index.html` state-machine path (`NAVIGATING` → `onDirectionDecided` → buzz), i.e. a buzz
  triggered by the AI rather than a button.

**Note on running the Pi:** `navassist_pi_camera.py` streams **binary frames on `/pi`**, which is
what the current `server.js` / `index.html` / `pi.html` consume. `main.py` streams **base64 on
`/esp32`** (the old path nothing reads anymore). Run **`navassist_pi_camera.py`** to get the
real-time feed *and* your haptics together.

**Hardware caveat:** a GPIO pin sources ~16 mA; an ERM motor wants roughly an order of magnitude
more. If the motors are on the pins *directly* rather than behind a transistor/MOSFET or a driver
breakout, you will eventually take the pin — or the SoC — with them.

---

## If you're picking this up cold

Read `PI_REALTIME.md` first — it's the main camera path and explains why the Pi runs neither
model. The short version of the model layout, since it confuses everyone:

| Model | Classes | Runs | Used by |
|---|---|---|---|
| `crossing_seg.onnx` | **dotted line**, pedestrian light | **Node server**, onnxruntime-node on a worker thread | **`index.html` (the real app)**, plus `pi.html` / `webcam.html` |
| `pedestrian.onnx` | red, green, traffic-light | **Browser**, onnxruntime-web in a worker | dev/debug pages only (`pi.html`, `webcam.html`, `model_test.html`) |

Neither runs on the Pi. The dotted-line/crossing model's output is drawn on `pi.html` and
`webcam.html`, and now also on `index.html` (a draw-only overlay was added — see `ai.js`
`drawCrossingOverlay`), while on `index.html` it *additionally* drives the state machine — the
corridor direction feeds the steady crossing-guidance cadence and the light state (red / green /
flashing) drives the spoken cues. Motors are driven only from `app.js` (`index.html`), never from
`pi.js`/`webcam.js`.
