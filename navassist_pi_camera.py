#!/usr/bin/env python3
"""
navassist_pi_camera.py — Raspberry Pi Zero 2W + Camera Module v3 -> NavAssist server.

Captures JPEG frames and streams them to the Node server over a WebSocket, as
BINARY messages. That is all it does. The Pi is a camera, not a brain:

  * pedestrian.onnx  runs in the BROWSER (onnxruntime-web, in a worker).
  * crossing_seg.onnx runs on the NODE SERVER (onnxruntime-node, worker thread).
  * the Pi runs NEITHER. A Zero 2W is 4x Cortex-A53 @1GHz with 512MB of RAM; a
    YOLO11 pass on it takes seconds, not milliseconds. Trying to move inference
    here would not make the system faster, it would stop it working at all.

Three decisions in here are what make the feed real-time. Change them casually
and the lag comes back:

  1. BINARY FRAMES, NOT base64-in-JSON. The old ESP32 protocol wrapped each JPEG
     in {"topic":"camera/image","payload":"<base64>"}. base64 inflates every
     frame by 33%, and both ends then have to parse a ~100KB string per frame. On
     a Zero 2W -- whose WiFi is 2.4GHz-only and shares its antenna budget with
     everything else -- that inflation IS the throughput ceiling. Raw bytes on
     the wire; the server reads them straight into the model.

  2. NO COMPRESSION ON THE WEBSOCKET. websockets negotiates permessage-deflate
     by default, which would try to DEFLATE-compress JPEG bytes that are already
     entropy-coded. That is real CPU on a weak core for approximately zero
     saving, so it is turned off explicitly.

  3. LATEST FRAME WINS. The encoder thread overwrites a single slot; the sender
     takes whatever is there when it is free. If WiFi stalls, the frames that
     pile up in between are DROPPED, never queued. A queue would mean the
     picture drifts further behind reality the longer you run -- which is exactly
     the failure people mistake for "the camera is slow".

HAPTICS
The one thing the Pi does besides the camera. app.js decides WHEN to buzz and
which side (sendHaptic -> 'haptic/command'), the server relays it down /pi, and
we hand it to HapticController from haptic.py -- the SAME controller that
test_haptic.py and main.py drive, so the pin mapping and PWM that were already
proven on the bench are exactly what runs here. app.js sends motor
('left'|'right'|'both'), intensity (0..1) and duration_ms; the Pi still makes no
decisions, it just actuates.

HapticController.buzz() BLOCKS for the pulse, so it runs OFF the event loop
(asyncio.to_thread) and the camera relay never stalls behind a vibration. Pin
numbers and the intensity->PWM mapping live in haptic.py now, not in this file.

NB: a GPIO pin sources ~16mA, an ERM vibration motor wants an order of magnitude
more. Drive it through a transistor/MOSFET or a motor driver breakout, not off
the pin directly, or you will eventually take the pin (or the SoC) with it.

WHY THIS SCRIPT DOES NOT ROTATE THE IMAGE
The camera is mounted sideways on the glasses, so frames come out 90 degrees off.
Rotating here is the one place you should NOT do it:
  * libcamera's hardware transform can flip and do 180, but NOT a quarter turn --
    so a 90 degree rotation means pulling the frame into numpy, transposing it,
    and re-encoding in software. That throws away the hardware JPEG encoder, i.e.
    the entire reason a Zero 2W can keep up at all.
  * both consumers get the rotation for free anyway. The server rotates inside
    its sharp pipeline (it decodes the JPEG there regardless), and the browser
    rotates on the canvas (the GPU does it). Neither pays measurably.
So the frame stays as the sensor produced it, and PI_ROTATE on the server (default
90 CW) is the single source of truth for which way is up.

SETUP (on the Pi)
    sudo apt install -y python3-picamera2 python3-gpiozero python3-lgpio
    pip install websockets --break-system-packages

RUN
    python3 navassist_pi_camera.py --host 192.168.1.42
    (--host is the IP that `npm start` prints on the laptop)

    python3 navassist_pi_camera.py --host rse3106.duckdns.org --port 443 --tls
    (the hosted AWS server; --tls skips cert verification, self-signed)

Then open http://<laptop-ip>:3000/pi.html to see both models running on the feed.
"""

import argparse
import asyncio
import io
import json
import ssl
import sys
import threading
import time

try:
    import websockets
except ImportError:
    sys.exit("websockets is missing.  pip install websockets --break-system-packages")

try:
    from picamera2 import Picamera2
    from picamera2.outputs import FileOutput
except ImportError:
    sys.exit("picamera2 is missing.  sudo apt install -y python3-picamera2")


# ---------------------------------------------------------------------------
# Defaults. Override any of them on the command line.
# ---------------------------------------------------------------------------
SERVER_HOST = "192.168.1.42"   # laptop IP printed by `npm start` -- CHANGE THIS
SERVER_PORT = 3000
CAP_WIDTH   = 960              # 4:3, NOT 16:9 -- see the note below, the shape matters
CAP_HEIGHT  = 720
FPS         = 15
JPEG_Q      = 75               # software encoder only
BITRATE     = None             # hardware MJPEG encoder. None = auto, scaled to the resolution

# BITRATE MUST SCALE WITH RESOLUTION, AND THIS IS WHY THE IMAGE GOES GREY IF IT DOESN'T.
#
# MJPEGEncoder takes a BITRATE, not a quality: it will hit the number you give it
# no matter what that costs the picture. Starve it and JPEG throws away CHROMA
# first -- the colour channels get subsampled and quantised into near-oblivion
# while luma (brightness) survives. The image does not get obviously blocky, it
# goes WASHED OUT AND GREY. That is the symptom, and it is a bitrate problem, not
# a colour-format one.
#
# A fixed bitrate silently rots the moment anyone changes the resolution:
#
#   4 Mbit/s @  640x480 x15  = 0.87 bits/pixel   fine
#   4 Mbit/s @ 1280x720 x15  = 0.29 bits/pixel   grey mush  <-- this bug
#
# So derive it instead. ~0.9 bits/pixel is a reasonable MJPEG working point.
BITS_PER_PIXEL = 0.9
BITRATE_FLOOR  = 2_000_000
BITRATE_CEIL   = 16_000_000    # past here the Zero 2W's 2.4GHz WiFi is the real limit anyway

# ---------------------------------------------------------------------------
# Haptics. Pin mapping and the intensity->PWM curve live in haptic.py's
# HapticController -- the bench-tested one shared with test_haptic.py and
# main.py. This file only relays 'haptic/command' into HapticController.buzz();
# the one knob it keeps is a hard duration cap so a malformed duration_ms can't
# leave a motor buzzing forever.
# ---------------------------------------------------------------------------
MAX_PULSE_MS = 3000        # hard ceiling; a bad duration_ms must not buzz forever


def auto_bitrate(args):
    """Bitrate for the hardware encoder, scaled to what we are actually capturing."""
    if args.bitrate:
        return args.bitrate      # explicit --bitrate always wins
    bps = int(args.width * args.height * args.fps * BITS_PER_PIXEL)
    return max(BITRATE_FLOOR, min(bps, BITRATE_CEIL))

# WHY 960x720 (4:3) AND NOT 1280x720 (16:9), WHICH IS THE COUNTERINTUITIVE BIT:
#
# Both models letterbox their input to 640x640, and our frame is rotated UPRIGHT
# before inference. That makes the capture's ASPECT RATIO matter more than its
# pixel count, because the horizontal pixels are the ones letterboxing throws away:
#
#   capture           rotated      what the model actually gets
#   960x720  (4:3)    720x960      480x640     <-- this. full width, extra detail
#   640x480  (4:3)    480x640      480x640     full width, at scale 1.0
#   1280x720 (16:9)   720x1280     360x640     LESS across, for MORE bytes
#
# So a "higher resolution" 720p (16:9) capture is actively worse: it hands the
# model 360px across where 4:3 gives 480px, and charges 1.3x the bytes for the
# privilege -- over the Pi's weakest link, its 2.4GHz-only WiFi.
#
# 960x720 is the sweet spot: the model gets its full 480px across, AND more real
# detail than VGA to find small distant lights in.
#
#     python3 navassist_pi_camera.py --width 640 --height 480    # cheapest on WiFi
#     python3 navassist_pi_camera.py --width 1280 --height 720   # 16:9, for a nicer-looking demo only


class FrameBus(io.BufferedIOBase):
    """Single-slot mailbox between the encoder thread and the asyncio sender.

    picamera2 calls write() from its own encoder thread, once per complete JPEG.
    We keep only the newest frame: if the sender is still busy pushing the last
    one out over congested WiFi, the ones that arrive in between are dropped on
    purpose. Latency stays flat instead of growing without bound.
    """

    def __init__(self):
        self.loop = None
        self.event = None
        self.lock = threading.Lock()
        self.latest = None
        self.captured = 0
        self.dropped = 0

    def bind(self, loop):
        self.loop = loop
        self.event = asyncio.Event()

    def write(self, buf):
        with self.lock:
            if self.latest is not None:
                self.dropped += 1          # sender never got to the previous one
            self.latest = bytes(buf)
            self.captured += 1
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.event.set)
        return len(buf)

    def take(self):
        with self.lock:
            buf, self.latest = self.latest, None
            return buf


class HapticCommandRunner:
    """Turns 'haptic/command' messages from the browser into motor buzzes.

    Delegates to HapticController (haptic.py) -- the SAME controller test_haptic.py
    and main.py drive, so the pin mapping and PWM that were already proven on the
    bench are exactly what runs here. app.js owns every decision about when to buzz
    and which side; this class only actuates.

    1. NEVER BLOCK THE EVENT LOOP. HapticController.buzz() is synchronous and
       sleeps for the whole pulse, so it is pushed to a worker thread with
       asyncio.to_thread -- the camera relay and heartbeats keep flowing while a
       motor is buzzing, instead of dropping frames on every nudge.

    2. DROP BURSTS. app.js's onDirectionDecided re-fires the NAVIGATING nudge
       every frame, far faster than a motor can give a distinct pulse. A short
       cooldown collapses a burst into one buzz rather than a long, blurred one.
       (This mirrors main.py's HapticCommandRunner, which is the proven path.)

    3. FAIL SOFT. With no controller -- haptic.py missing, GPIO unavailable, or
       --no-haptics -- every command is a no-op and the camera is unaffected. A
       wearer with video and no buzzing is degraded; a wearer with neither is
       blind.

    buzz() self-terminates (on, sleep, off), so a link that dies mid-pulse needs
    no explicit stop: the worker thread finishes the pulse and the motor goes
    quiet on its own within duration_ms (capped at MAX_PULSE_MS).
    """

    def __init__(self, haptic, cooldown_s=0.15):
        self.haptic = haptic            # HapticController instance, or None
        self.cooldown_s = cooldown_s
        self._last = 0.0

    async def handle(self, payload):
        """Act on one 'haptic/command'. Awaited by read_commands; the blocking
        buzz is run on a worker thread so this never stalls the socket loop."""
        if self.haptic is None or not isinstance(payload, dict):
            return

        motor = str(payload.get("motor", "both")).lower()
        if motor not in ("left", "right", "both"):
            print("  [haptic] ignoring unknown motor: %r" % motor)
            return

        try:
            intensity = float(payload.get("intensity", 0.8))
        except (TypeError, ValueError):
            intensity = 0.8              # same default app.js uses when it omits one
        intensity = max(0.0, min(1.0, intensity))

        try:
            ms = int(payload.get("duration_ms") or 300)
        except (TypeError, ValueError):
            ms = 300
        ms = max(0, min(ms, MAX_PULSE_MS))       # never buzz forever on a bad value
        duration_s = ms / 1000.0

        # Zero intensity/duration means "no cue" -- stay silent, don't buzz.
        if intensity <= 0.0 or duration_s <= 0.0:
            return

        now = time.monotonic()
        if now - self._last < self.cooldown_s:
            return
        self._last = now

        print("  [haptic] %s intensity=%.2f duration=%dms pattern=%s"
              % (motor, intensity, ms, payload.get("pattern")))
        try:
            # buzz(side, duration_s, intensity) -- same call test_haptic.py/main.py
            # make. It blocks for duration_s, so hand it to a worker thread.
            await asyncio.to_thread(self.haptic.buzz, motor, duration_s, intensity)
        except Exception as exc:
            print("  [haptic] buzz failed: %s" % exc)

    async def self_test(self):
        """--haptic-test: buzz left, then right, then both, before any network
        traffic -- confirm the wiring without the server, browser, or a crossing."""
        if self.haptic is None:
            print("[haptic] self-test skipped -- haptics unavailable")
            return
        for label, motor in (("LEFT", "left"), ("RIGHT", "right"), ("BOTH", "both")):
            print("[haptic] self-test: %s" % label)
            try:
                await asyncio.to_thread(self.haptic.buzz, motor, 0.6, 0.9)
            except Exception as exc:
                print("[haptic] self-test buzz failed: %s" % exc)
            await asyncio.sleep(0.4)
        print("[haptic] self-test done. Felt on the wrong side? "
              "Fix the pin mapping in haptic.py.")

    def close(self):
        """Release the controller on shutdown. Nothing to force-stop mid-pulse:
        buzz() ends on its own (see the class docstring)."""
        if self.haptic is None:
            return
        try:
            self.haptic.close()
        except Exception:
            pass
        self.haptic = None


def make_encoder(args):
    """Hardware MJPEG by default; software JPEG as a fallback.

    The Zero 2W's VideoCore IV has a hardware JPEG block, and using it is what
    frees the CPU for capture and networking. MJPEGEncoder drives it. JpegEncoder
    is a multi-threaded software encoder -- it works, it honours --quality
    directly, and it costs real CPU. Only reach for it if MJPEG misbehaves.
    """
    from picamera2.encoders import JpegEncoder, MJPEGEncoder
    if args.encoder == "mjpeg":
        br = auto_bitrate(args)
        bpp = br / float(args.width * args.height * args.fps)
        print("[cam] MJPEG bitrate %.1f Mbit/s (%.2f bits/pixel)" % (br / 1e6, bpp))
        if bpp < 0.5:
            print("[cam] WARNING: under ~0.5 bits/pixel JPEG starts dumping colour -- "
                  "expect a washed-out, greyish image. Raise --bitrate or lower --width/--fps.")
        if br > 8_000_000:
            print("[cam] NOTE: >8 Mbit/s is a lot for the Zero 2W's 2.4GHz WiFi. If [net] starts "
                  "reporting dropped frames, that is the link, not the camera.")
        return MJPEGEncoder(bitrate=br)
    return JpegEncoder(q=args.quality)


def start_camera(args, bus):
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (args.width, args.height)},
        controls={
            "FrameRate": args.fps,
            # AUTO EXPOSURE, on explicitly rather than by luck. It is picamera2's
            # default, but this is a wearable that walks from shade into direct
            # sun and back, so it is worth pinning rather than inheriting: if AE
            # is ever off, the frame blows out or goes black the moment the light
            # changes and the model simply stops seeing the light it needs.
            #
            # Note FrameRate above sets the exposure CEILING: at 15fps the AE loop
            # can integrate for at most ~66ms. That is the right way round for us
            # -- a long exposure would motion-blur a walking wearer's frame, and a
            # blurred pedestrian light is a missed one. AE trades to gain instead.
            "AeEnable": True,
            # Auto white balance too, for the same reason: 'is that light red or
            # green' is a COLOUR question, and a drifting white balance is exactly
            # the thing that would corrupt the answer.
            "AwbEnable": True,
        },
    )
    picam2.configure(config)

    # Enum-valued controls have to come from libcamera, and it may not be importable
    # on every setup -- so they go here, after the camera is already configured and
    # working, where failing to set them costs us tuning but not the camera.
    try:
        from libcamera import controls as libcontrols

        # Camera Module v3 has autofocus, but we PIN IT AT INFINITY rather than let
        # it hunt: the pedestrian light we care about is across the road (far), and
        # continuous AF wastes time racking focus onto near clutter -- the wearer's
        # own body, passers-by, a hand -- and can be caught mid-hunt (blurred) at
        # exactly the moment a light appears. Manual focus held at infinity is sharp
        # for everything from a few metres out to the horizon, which is the whole
        # range that matters here. LensPosition is in DIOPTRES (1/metres), so 0.0 =
        # infinity. Silently skipped on a v2 (fixed focus, no such control) -- which
        # is already focused near infinity anyway.
        picam2.set_controls({
            "AfMode": libcontrols.AfModeEnum.Manual,
            "LensPosition": 0.0,          # dioptres; 0.0 = focus at infinity
        })
        print("[cam] focus fixed at infinity (manual, LensPosition 0.0)")

        # AE mode + metering. 'Normal' is the standard AE loop (as opposed to Short/
        # Long, which bias the exposure/gain tradeoff). Centre-weighted metering is
        # what we want over the default: the traffic light is what the wearer is
        # facing, so it is near the middle of frame, and we would rather expose for
        # THAT than for a bright sky filling the top of the shot -- which is exactly
        # what would otherwise silhouette the light into uselessness.
        picam2.set_controls({
            "AeEnable": True,
            "AeExposureMode": libcontrols.AeExposureModeEnum.Normal,
            "AeMeteringMode": libcontrols.AeMeteringModeEnum.CentreWeighted,
            "AwbEnable": True,
            "AwbMode": libcontrols.AwbModeEnum.Auto,
        })
        print("[cam] auto exposure (normal, centre-weighted) + auto white balance enabled")
    except Exception as exc:
        # AeEnable/AwbEnable were already set in the config above, so plain auto
        # exposure is still on even if we land here. Only the fine-tuning is lost.
        print("[cam] AF/AE fine-tuning unavailable, plain auto exposure still on: %s" % exc)

    try:
        picam2.start_recording(make_encoder(args), FileOutput(bus))
        print("[cam] %dx%d @ %dfps, %s encoder"
              % (args.width, args.height, args.fps, args.encoder))
    except Exception as exc:
        if args.encoder != "mjpeg":
            raise
        print("[cam] hardware MJPEG encoder failed (%s) -- falling back to software JPEG" % exc)
        args.encoder = "jpeg"
        picam2.start_recording(make_encoder(args), FileOutput(bus))
        print("[cam] %dx%d @ %dfps, software JPEG q=%d"
              % (args.width, args.height, args.fps, args.quality))

    return picam2


async def send_frames(ws, bus, stats):
    """Push the newest frame, forever.

    `await ws.send(...)` is the backpressure valve: if the link is congested it
    blocks here, the encoder keeps overwriting bus.latest behind our back, and
    when we come back around we pick up the FRESHEST frame rather than working
    through a backlog of stale ones.
    """
    while True:
        await bus.event.wait()
        bus.event.clear()
        buf = bus.take()
        if not buf:
            continue
        await ws.send(buf)                     # binary frame: raw JPEG bytes
        stats["sent"] += 1
        stats["bytes"] += len(buf)


async def send_heartbeats(ws):
    """Text frames, the same envelope the rest of the project uses."""
    while True:
        await ws.send(json.dumps({
            "topic": "system/heartbeat",
            "timestamp": int(time.time() * 1000),
            "payload": {},
        }))
        await asyncio.sleep(2)


async def read_commands(ws, haptics):
    """Drain what the server sends us and act on it.

    Today that is haptic/command, originating in app.js's state machine
    (sendHaptic -> sendToPi) and relayed by server.js. handle() pushes the
    blocking buzz to a worker thread, so awaiting it here doesn't hold up the
    camera relay (a separate task) or the socket's own ping/close handling.
    """
    async for message in ws:
        if isinstance(message, bytes):
            continue
        try:
            envelope = json.loads(message)
        except ValueError:
            continue
        if envelope.get("topic") == "haptic/command":
            payload = envelope.get("payload") or {}
            print("[cmd] haptic: %s" % payload)
            await haptics.handle(payload)


async def report(bus, stats):
    """Once a second, the numbers you actually need to tune this."""
    while True:
        await asyncio.sleep(1)
        sent, kb = stats["sent"], stats["bytes"] / 1024.0
        captured, dropped = bus.captured, bus.dropped
        bus.captured = bus.dropped = 0
        stats["sent"] = stats["bytes"] = 0
        if captured or sent:
            # dropped >> 0 means the LINK cannot keep up with the camera: lower
            # --fps, --quality, or --width before you go blaming the models.
            print("[net] captured %2d  sent %2d  dropped %2d  %5.0f KB/s"
                  % (captured, sent, dropped, kb))


async def run(args):
    bus = FrameBus()
    bus.bind(asyncio.get_running_loop())

    # Haptics delegate to HapticController in haptic.py -- the bench-tested one.
    # Import it HERE, not at module top, and fail soft: a missing haptic.py or an
    # unavailable GPIO disables buzzing but must never stop the camera.
    haptic = None
    if args.no_haptics:
        print("[haptic] disabled (--no-haptics)")
    else:
        try:
            from haptic import HapticController
            haptic = HapticController()
            print("[haptic] HapticController ready (haptic.py)")
        except Exception as exc:
            print("[haptic] unavailable (%s) -- haptics OFF, camera unaffected." % exc)
    haptics = HapticCommandRunner(haptic)
    if args.haptic_test:
        await haptics.self_test()

    picam2 = start_camera(args, bus)

    scheme = "wss" if args.tls else "ws"
    url = "%s://%s:%d/pi" % (scheme, args.host, args.port)

    ssl_ctx = None
    if args.tls:
        # The project's AWS box uses a self-signed certificate -- the same one the
        # browser warns about. Verification is disabled for that reason and no
        # other.
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    stats = {"sent": 0, "bytes": 0}
    reporter = asyncio.create_task(report(bus, stats))

    try:
        while True:
            try:
                print("[net] connecting to %s ..." % url)
                async with websockets.connect(
                    url,
                    ssl=ssl_ctx,
                    compression=None,       # never DEFLATE an already-compressed JPEG
                    ping_interval=20,
                    ping_timeout=20,
                    open_timeout=10,
                ) as ws:
                    print("Connected to relay server.")
                    await asyncio.gather(
                        send_frames(ws, bus, stats),
                        send_heartbeats(ws),
                        read_commands(ws, haptics),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # No stop needed on a mid-pulse drop: buzz() self-terminates, so
                # the worker thread finishes the pulse and the motor goes quiet on
                # its own within duration_ms -- no phantom "turn left" that outlasts
                # the reconnect.
                print("[net] link down (%s) -- retrying in 2s" % exc)
                await asyncio.sleep(2)
    finally:
        reporter.cancel()
        haptics.close()
        picam2.stop_recording()
        print("[cam] stopped")


def parse_args():
    p = argparse.ArgumentParser(description="Stream Pi Camera frames to the NavAssist server.")
    p.add_argument("--host", default=SERVER_HOST, help="server IP/hostname (default: %(default)s)")
    p.add_argument("--port", type=int, default=SERVER_PORT, help="server port (default: %(default)s)")
    p.add_argument("--tls", action="store_true", help="use wss:// (AWS; skips self-signed cert checks)")
    p.add_argument("--width", type=int, default=CAP_WIDTH, help="capture width (default: %(default)s)")
    p.add_argument("--height", type=int, default=CAP_HEIGHT, help="capture height (default: %(default)s)")
    p.add_argument("--fps", type=int, default=FPS, help="capture frame rate (default: %(default)s)")
    p.add_argument("--encoder", choices=("mjpeg", "jpeg"), default="mjpeg",
                   help="mjpeg = hardware (default), jpeg = software")
    p.add_argument("--quality", type=int, default=JPEG_Q, help="software encoder quality (default: %(default)s)")
    p.add_argument("--bitrate", type=int, default=BITRATE,
                   help="hardware encoder bitrate in bits/sec (default: auto, ~%.1f bits/pixel "
                        "scaled to --width/--height/--fps)" % BITS_PER_PIXEL)

    h = p.add_argument_group("haptics")
    # Pin numbers, PWM frequency and the intensity->duty curve now live in
    # haptic.py's HapticController (shared with test_haptic.py). Tune them there,
    # not with CLI flags.
    h.add_argument("--haptic-test", action="store_true",
                   help="buzz left, right, then both at startup and carry on -- check the wiring")
    h.add_argument("--no-haptics", action="store_true",
                   help="do not initialise HapticController at all (camera only)")
    return p.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print("\nbye")
