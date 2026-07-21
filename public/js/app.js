/**
 * NavAssist — app.js
 * OWNER: Darrshini
 *
 * Responsibilities:
 * 1. Connect to the Node.js server over the /live WebSocket
 * 2. Receive camera frames (binary JPEG) and crossing results from the Raspberry
 *    Pi Zero 2W glasses, via the server
 * 3. Get GPS location from phone browser API
 * 4. Send haptic motor commands back to the Pi
 * 5. Manage app state machine
 * 6. Audio announcements for accessibility
 *
 * Heading comes from the PHONE's compass (DeviceOrientationEvent), not from an
 * IMU on the glasses -- see handleImuReading / the heading section below.
 */

// ============================================================
// Config
// ============================================================

let config = {};

async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        config = await res.json();
        debugLog('Config loaded');
    } catch (e) {
        debugLog('Config load failed: ' + e.message);
    }
}

// ============================================================
// State Machine
// ============================================================

const STATES = {
    IDLE:             'IDLE',
    READY:            'READY',
    SCANNING:         'SCANNING',
    CHOOSE_POST:      'CHOOSE_POST',
    RESOLVING:        'RESOLVING',
    TARGET_DETECTED:  'TARGET_DETECTED',
    CONFIRM_TARGET:   'CONFIRM_TARGET',
    NAVIGATING:       'NAVIGATING',
    REACHED:          'REACHED',
    WAITING:          'WAITING',
    CONFIRM_CROSSING: 'CONFIRM_CROSSING',
    CROSSING:         'CROSSING',
    CONFIRM_ARRIVAL:  'CONFIRM_ARRIVAL',
    COMPLETED:        'COMPLETED',
    LOST_CONNECTION:  'LOST_CONNECTION'
};

let currentState = STATES.IDLE;
let armedGestures = [];

function transitionTo(newState, message) {
    debugLog('State: ' + currentState + ' → ' + newState);
    currentState = newState;
    updateUI(newState, message);
    announceState(newState, message);
}

const GESTURE_RULES = {
    [STATES.READY]:            { single_tap: 'START_SCAN' },
    [STATES.CONFIRM_TARGET]:   { double_tap: 'CONFIRM_YES', triple_tap: 'CONFIRM_NO' },
    [STATES.CHOOSE_POST]:      { double_tap: 'CHOOSE_LEFT', triple_tap: 'CHOOSE_RIGHT' },
    [STATES.CONFIRM_CROSSING]: { double_tap: 'CONFIRM_YES', triple_tap: 'CONFIRM_NO' },
    [STATES.CONFIRM_ARRIVAL]:  { single_tap: 'CONFIRM_YES' },
    [STATES.COMPLETED]:        { single_tap: 'RESET' }
};

// ============================================================
// WebSocket
// ============================================================

let ws = null;
// /live, not the old /browser. The glasses are a Raspberry Pi Zero 2W now, and
// it sends each JPEG as a BINARY WebSocket frame -- raw bytes, no base64, no
// JSON envelope (see navassist_pi_camera.py for why: base64 inflates every frame
// 33%, and on the Zero 2W's 2.4GHz-only WiFi that inflation was the throughput
// ceiling). The server relays those bytes to us untouched, and pushes the
// crossing model's results down the same socket as JSON.
const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/live';

function connectWebSocket() {
    ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        debugLog('Connected to server');
    };

    ws.onmessage = (event) => {
        // Binary => a JPEG frame from the Pi. Text => JSON (crossing results,
        // heartbeats, connection events).
        if (typeof event.data !== 'string') {
            if (typeof handleCameraFrame === 'function') handleCameraFrame(event.data);
            return;
        }
        try {
            handleIncomingMessage(JSON.parse(event.data));
        } catch (e) {
            debugLog('Parse error: ' + e.message);
        }
    };

    ws.onclose = () => {
        debugLog('Server disconnected. Retrying...');
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = () => ws.close();
}

// Browser → Pi (haptic commands). The server forwards anything that isn't
// live/config straight on to the Pi, which handles it in read_commands().
function sendToPi(topic, payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ topic, timestamp: Date.now(), payload }));
}

// ============================================================
// Incoming message router
// ============================================================

// Camera frames are NOT routed here -- they arrive as binary, handled in
// ws.onmessage above. Everything on this socket that is text is JSON.
function handleIncomingMessage(envelope) {
    switch (envelope.topic) {
        case 'connection/event': handleConnectionEvent(envelope.payload); break;
        case 'system/heartbeat': handleHeartbeat(envelope.payload); break;
        case 'imu/orientation':  handleImuReading(envelope.payload); break;
        case 'crossing/result':
            // The server ran crossing_seg.onnx on the frame the Pi sent it and
            // pushed the result. ai.js folds it into the state machine.
            if (window.navassist.handleCrossingResult) {
                window.navassist.handleCrossingResult(envelope.payload);
            }
            break;
        default:
            debugLog('Unknown topic: ' + envelope.topic);
    }
}

// ============================================================
// Connection events
// ============================================================

let heartbeatTimer = null;

function handleConnectionEvent(payload) {
    const event = payload.event;
    debugLog('Connection: ' + event);

    // The server tells us the angle it rotates Pi frames by, and it is the single
    // source of truth for it. Handing it to ai.js is what stops the server's
    // coordinates and our canvas silently disagreeing about which way is up.
    if (typeof payload.rotate === 'number' && window.navassist.setFrameRotation) {
        window.navassist.setFrameRotation(payload.rotate);
    }

    if (event === 'pi_connected') {
        clearTimeout(heartbeatTimer);
        updateConnectionStatus(true);
        speak('Glasses connected successfully. Tap anywhere to start scanning.', true);
        transitionTo(STATES.READY);
    } else if (event === 'pi_disconnected') {
        updateConnectionStatus(false);
        if (pendingConfirmation) {
            pendingConfirmation = null;
            if (pendingConfirmationTimerId) {
                clearTimeout(pendingConfirmationTimerId);
                pendingConfirmationTimerId = null;
            }
        }
        if (crossingStallCheckId) {
            clearInterval(crossingStallCheckId);
            crossingStallCheckId = null;
        }
        if (crossingFallbackTimerId) {
            clearTimeout(crossingFallbackTimerId);
            crossingFallbackTimerId = null;
        }
        if (currentState !== STATES.IDLE) {
            speak('Glasses disconnected. Please check the connection.', true);
            transitionTo(STATES.LOST_CONNECTION);
        } else {
            speak('Waiting for glasses to connect. Please ensure your glasses are powered on and connected to WiFi.', true);
        }
    }
}

function handleHeartbeat(payload) {
    clearTimeout(heartbeatTimer);
    heartbeatTimer = setTimeout(() => {
        debugLog('Heartbeat timeout');
        handleConnectionEvent({ event: 'pi_disconnected' });
    }, 6000);
}

// ============================================================
// IMU + compass
// ============================================================

let currentHeading = 0;
let imuCalibrated = false;
let phoneCompassAvailable = false;
const SIGNIFICANT_TURN_DEG = 45;

// Reference headings captured at the start of a phase, compared against the
// live phone heading to detect drift. Using the phone's own compass as the
// sole heading source (team decision: no glasses IMU, accept compass noise
// as a tradeoff for time) -- this replaces the old gyroscope-rate-integration
// approach, which depended entirely on glasses hardware that no longer sends
// imu/orientation data at all.
let navigatingStartHeading = null;
let lastTurnWarningAt = 0;
const TURN_WARNING_COOLDOWN_MS = 5000;

let crossingStartHeading = null;
let lastCrossingHapticAt = 0;
const CROSSING_DRIFT_THRESHOLD_DEG = 20;   // how far off-line before nudging
const CROSSING_HAPTIC_COOLDOWN_MS = 1500;  // avoid buzzing on every reading

// Signed shortest-path angle difference, handles 0/360 wraparound correctly
function angleDiffDeg(a, b) {
    let diff = a - b;
    while (diff > 180) diff -= 360;
    while (diff < -180) diff += 360;
    return diff;
}

function initPhoneCompass() {
    if (typeof DeviceOrientationEvent === 'undefined') return;

    if (typeof DeviceOrientationEvent.requestPermission === 'function') {
        DeviceOrientationEvent.requestPermission()
            .then(response => {
                if (response === 'granted') {
                    window.addEventListener('deviceorientation', handleDeviceOrientation);
                    phoneCompassAvailable = true;
                }
            })
            .catch(() => {});
    } else {
        window.addEventListener('deviceorientation', handleDeviceOrientation);
        phoneCompassAvailable = true;
    }
}

function handleDeviceOrientation(event) {
    if (event.alpha === null) return;
    currentHeading = event.alpha;
    imuCalibrated = true;
    checkHeadingDrift();
}

// Runs on every phone compass update. Handles both the "you've turned away"
// warning during NAVIGATING and the haptic drift-correction during CROSSING
// -- both compare the live heading against a reference captured at the start
// of that phase, rather than integrating a rotation rate (which needed a
// physical gyroscope we no longer have).
function checkHeadingDrift() {
    if (currentState === STATES.NAVIGATING && navigatingStartHeading !== null) {
        const drift = angleDiffDeg(currentHeading, navigatingStartHeading);
        const now = Date.now();
        if (Math.abs(drift) > SIGNIFICANT_TURN_DEG && now - lastTurnWarningAt > TURN_WARNING_COOLDOWN_MS) {
            lastTurnWarningAt = now;
            speak('You have turned away. Follow the haptic feedback to re-orient toward the crossing.');
            navigatingStartHeading = currentHeading;  // reset reference so this only re-fires after another significant turn
        }
    }

    // Crossing drift correction: nudge the user back toward a straight line
    // if their heading drifts too far from the heading captured the moment
    // they started crossing. NOTE: verify the sign below (which motor fires
    // for which drift direction) against the real motor wiring -- this
    // assumes compass-style clockwise degrees, where a positive drift means
    // the user turned right and needs a left-nudge to correct back.
    if (currentState === STATES.CROSSING && crossingStartHeading !== null) {
        const drift = angleDiffDeg(currentHeading, crossingStartHeading);
        const now = Date.now();
        if (Math.abs(drift) > CROSSING_DRIFT_THRESHOLD_DEG &&
            now - lastCrossingHapticAt > CROSSING_HAPTIC_COOLDOWN_MS) {
            lastCrossingHapticAt = now;
            if (drift > 0) sendHaptic('left', 'pulse', 0.6, 250);
            else            sendHaptic('right', 'pulse', 0.6, 250);
        }
    }
}

function handleImuReading(payload) {
    // Kept for backward compatibility in case a physical IMU is ever added
    // back later -- currently unused, since heading comes from the phone's
    // own compass (see handleDeviceOrientation / checkHeadingDrift above).
    if (!phoneCompassAvailable && payload.heading_deg) {
        currentHeading = payload.heading_deg;
        imuCalibrated = payload.calibrated || false;
        checkHeadingDrift();
    }

    debugLog('IMU: heading=' + currentHeading.toFixed(1) + '°');
}

window.navassist = window.navassist || {};
window.navassist.getCurrentHeading = () => currentHeading;
window.navassist.isImuCalibrated = () => imuCalibrated;

// ============================================================
// GPS
// ============================================================

let currentLocation = null;

// Crossing completion: PRIMARY trigger is real GPS distance moved (unchanged
// -- still fires the moment someone has genuinely covered a crossing's
// worth of distance). The FALLBACK, though, no longer assumes a fixed
// duration means "done" -- a fixed number can't be right for both a narrow
// 3-lane crossing and a wide multi-lane road with a median. Instead it
// tracks whether GPS distance is STILL INCREASING (still walking, however
// long the crossing is) versus STALLED (hasn't meaningfully progressed in
// a while -- either arrived and stopped, or something's wrong either way).
// A stall is what triggers the check-in prompt, not elapsed time.
let crossingStartLocation = null;
let crossingFallbackTimerId = null;
let crossingStallCheckId = null;
let lastCrossingDistance = 0;
let lastCrossingProgressAt = 0;
let crossingStallWarned = false;
const CROSSING_DISTANCE_THRESHOLD_M = 8;      // typical minimum pedestrian crossing width
const CROSSING_PROGRESS_MIN_DELTA_M = 1;      // ignore GPS jitter smaller than this as "not real progress"
const CROSSING_STALL_WARNING_MS = 8000;       // gentle heads-up before the actual prompt
const CROSSING_STALL_PROMPT_MS = 12000;       // no forward progress for this long -> prompt
const CROSSING_ABSOLUTE_CEILING_MS = 90000;   // last-resort only, e.g. total GPS loss with zero updates at all

// Haversine formula: great-circle distance between two lat/lng points in meters
function distanceMeters(lat1, lng1, lat2, lng2) {
    const R = 6371000; // Earth radius in meters
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLng = (lng2 - lng1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 +
        Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
        Math.sin(dLng / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// Runs on a fixed interval while CROSSING (independent of how often GPS
// updates actually arrive, since those can be sparse/irregular). Checks how
// long it's been since distance last meaningfully increased.
function checkCrossingStall() {
    if (currentState !== STATES.CROSSING) return;
    const stalledFor = Date.now() - lastCrossingProgressAt;

    if (stalledFor >= CROSSING_STALL_PROMPT_MS) {
        crossingStallWarned = false;  // reset for next time, if this prompt gets cancelled/ignored
        promptArrivalConfirmation();
    } else if (stalledFor >= CROSSING_STALL_WARNING_MS && !crossingStallWarned) {
        crossingStallWarned = true;
        speak('Still walking? Let me know once you have crossed.');
    }
}

// Fired once the user has moved far enough (GPS distance) or progress has
// stalled for a while -- either way, this does NOT complete the crossing
// automatically anymore. It prompts the user to check for tactile ground
// indicators with their cane and explicitly confirm before the crossing is
// marked complete. This exists specifically to prevent a false "crossing
// complete" while the user might still be halfway across -- GPS alone isn't
// trusted to make that call on its own.
function promptArrivalConfirmation() {
    if (currentState !== STATES.CROSSING) return;  // already prompted or moved on -- avoid double-firing
    if (crossingFallbackTimerId) {
        clearTimeout(crossingFallbackTimerId);
        crossingFallbackTimerId = null;
    }
    if (crossingStallCheckId) {
        clearInterval(crossingStallCheckId);
        crossingStallCheckId = null;
    }
    transitionTo(STATES.CONFIRM_ARRIVAL,
        'You may have reached the other side. If you can feel the tactile ground indicators with your cane, tap once to confirm you have crossed safely.');
}

// Only called after the user has explicitly confirmed (through the
// double-confirmation flow) that they feel the tactile indicators.
function completeCrossing() {
    if (currentState !== STATES.CONFIRM_ARRIVAL) return;
    crossingStartHeading = null;
    crossingStartLocation = null;
    transitionTo(STATES.COMPLETED, 'Crossing complete. You are safely across. Tap once to scan again.');
}

function startGpsTracking() {
    if (!navigator.geolocation) {
        debugLog('GPS not available');
        return;
    }

    navigator.geolocation.watchPosition(
        (position) => {
            currentLocation = {
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
                accuracyMeters: position.coords.accuracy
            };
            debugLog('GPS: ' + currentLocation.latitude.toFixed(5) + ', ' +
                currentLocation.longitude.toFixed(5) + ' ±' +
                currentLocation.accuracyMeters.toFixed(0) + 'm');

            if (currentState === STATES.RESOLVING) {
                resolveJunction();
            }

            if (currentState === STATES.CROSSING && crossingStartLocation) {
                const moved = distanceMeters(
                    crossingStartLocation.latitude, crossingStartLocation.longitude,
                    currentLocation.latitude, currentLocation.longitude
                );
                debugLog('Crossing distance moved: ' + moved.toFixed(1) + 'm');

                if (moved - lastCrossingDistance >= CROSSING_PROGRESS_MIN_DELTA_M) {
                    lastCrossingDistance = moved;
                    lastCrossingProgressAt = Date.now();
                    crossingStallWarned = false;  // fresh progress -- reset the warning so it can fire again if they stall later
                }

                if (moved >= CROSSING_DISTANCE_THRESHOLD_M) {
                    promptArrivalConfirmation();
                }
            }
        },
        (error) => {
            debugLog('GPS error: ' + error.message);
        },
        { enableHighAccuracy: true, maximumAge: 3000, timeout: 10000 }
    );
}

window.navassist.getCurrentLocation = () => currentLocation;

// ============================================================
// Junction resolution
// ============================================================

let nearbyCrossings = [];
let selectedCrossingIndex = 0;

async function resolveJunction() {
    if (currentLocation) {
        await resolveJunctionWithGPS();
    } else {
        resolveJunctionWithCameraOnly();
    }
}

// Overpass API endpoints, tried in order -- the public instance can be slow
// or rate-limited under load, so we fall through to mirrors rather than
// failing outright on the first one.
const OVERPASS_ENDPOINTS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
    'https://lz4.overpass-api.de/api/interpreter'
];
const CROSSING_SEARCH_RADIUS_M = 100;
const ROAD_NAME_SEARCH_RADIUS_M = 20;   // small radius anchored on the crossing itself

async function overpassQuery(query) {
    let lastError = null;
    for (const endpoint of OVERPASS_ENDPOINTS) {
        try {
            const res = await fetch(endpoint, {
                method: 'POST',
                body: 'data=' + encodeURIComponent(query)
            });
            if (!res.ok) { lastError = new Error('HTTP ' + res.status); continue; }
            const data = await res.json();
            return data;
        } catch (e) {
            lastError = e;
            // try the next mirror
        }
    }
    throw lastError || new Error('All Overpass endpoints failed');
}

// Finds the name of whatever road the crossing sits on, by querying a small
// radius directly around the crossing's own coordinates (not the user's) --
// a crossing node should be right on top of, or immediately adjacent to,
// the road way it belongs to.
async function findRoadNameNear(lat, lon) {
    try {
        const query = '[out:json][timeout:10];' +
            'way(around:' + ROAD_NAME_SEARCH_RADIUS_M + ',' + lat + ',' + lon + ')["highway"]["name"];' +
            'out tags;';
        const data = await overpassQuery(query);
        if (data.elements && data.elements.length && data.elements[0].tags && data.elements[0].tags.name) {
            return data.elements[0].tags.name;
        }
    } catch (e) {
        debugLog('Road name lookup failed: ' + e.message);
    }
    return null;  // caller falls back to a generic name -- never blocks on this
}

async function resolveJunctionWithGPS() {
    if (currentLocation.accuracyMeters > 30) {
        speak('GPS signal is weak. Please wait or move to an open area.');
        setTimeout(resolveJunction, 3000);
        return;
    }

    try {
        // Query real OpenStreetMap infrastructure data for signal-controlled
        // pedestrian crossings -- highway=crossing + crossing=traffic_signals
        // is the actual OSM tag for exactly this, confirmed against OSM's
        // own wiki. This replaces the previous Google Places lookup, which
        // used a "traffic_signals" type that doesn't exist in Google's
        // Places API at all (it's a business directory, not road
        // infrastructure) -- that mismatch is what caused nearby stores to
        // come back instead of crossings.
        const query = '[out:json][timeout:15];' +
            'node["highway"="crossing"]["crossing"="traffic_signals"]' +
            '(around:' + CROSSING_SEARCH_RADIUS_M + ',' +
            currentLocation.latitude + ',' + currentLocation.longitude + ');' +
            'out body;';

        const data = await overpassQuery(query);

        if (!data.elements || !data.elements.length) {
            // No real crossings found nearby. Fail safe: fall back to
            // camera-only scanning rather than ever guessing or widening
            // the search blindly -- if GPS is subtly off (right area, wrong
            // street), presenting a distant or wrong result would be worse
            // than presenting none.
            debugLog('No pedestrian crossings found via Overpass within ' + CROSSING_SEARCH_RADIUS_M + 'm');
            resolveJunctionWithCameraOnly();
            return;
        }

        // Compute distance for every candidate first, sort, then only look
        // up road names for the closest few -- keeps this fast and limits
        // how many extra network requests a single scan can trigger.
        const candidates = data.elements.map(el => ({
            latitude: el.lat,
            longitude: el.lon,
            distance: haversineMeters(
                currentLocation.latitude, currentLocation.longitude,
                el.lat, el.lon
            )
        })).sort((a, b) => a.distance - b.distance);

        const topCandidates = candidates.slice(0, 5);
        const namePromises = topCandidates.map(c => findRoadNameNear(c.latitude, c.longitude));
        const names = await Promise.all(namePromises);

        nearbyCrossings = topCandidates.map((c, i) => ({
            name: names[i] || 'a nearby pedestrian crossing',
            distance: c.distance
        }));

        nearbyCrossings.sort((a, b) => a.distance - b.distance);
        selectedCrossingIndex = 0;
        announceCurrentCrossing();

    } catch (e) {
        debugLog('Maps error: ' + e.message);
        resolveJunctionWithCameraOnly();
    }
}

function resolveJunctionWithCameraOnly() {
    speak('Scanning for a pedestrian crossing. Please walk slowly toward the crossing you want to use.');
    transitionTo(STATES.SCANNING);
}

function announceCurrentCrossing() {
    if (!nearbyCrossings.length) {
        resolveJunctionWithCameraOnly();
        return;
    }

    const crossing = nearbyCrossings[selectedCrossingIndex];
    const distText = crossing.distance < 20
        ? 'very close'
        : 'about ' + Math.round(crossing.distance) + ' metres away';

    const more = nearbyCrossings.length > 1
        ? 'Triple tap to hear the next crossing. There are ' + nearbyCrossings.length + ' crossings nearby.'
        : 'Triple tap if this is not correct.';

    const message = 'Crossing found: ' + crossing.name + ', ' + distText + '. ' +
        'Double tap to confirm this is your crossing. ' + more;

    transitionTo(STATES.TARGET_DETECTED);
    setTimeout(() => {
        if (currentState === STATES.TARGET_DETECTED) {
            transitionTo(STATES.CONFIRM_TARGET, message);
        }
    }, 1500);
}

function tryNextCrossing() {
    if (nearbyCrossings.length <= 1) {
        speak('No other crossings found nearby. Scanning again.');
        transitionTo(STATES.RESOLVING);
        setTimeout(resolveJunction, 1000);
        return;
    }
    selectedCrossingIndex = (selectedCrossingIndex + 1) % nearbyCrossings.length;
    announceCurrentCrossing();
}

function haversineMeters(lat1, lng1, lat2, lng2) {
    const R = 6371000;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLng = (lng2 - lng1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLng/2) * Math.sin(dLng/2);
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

// ============================================================
// Haptic commands
// ============================================================

function sendHaptic(motor, pattern, intensity, durationMs) {
    intensity = intensity || 0.8;
    durationMs = durationMs || 300;
    sendToPi('haptic/command', { motor, pattern, intensity, duration_ms: durationMs });
    debugLog('Haptic: ' + motor);
}

// Callbacks for ai.js
window.navassist.onTrafficLightVisible = function(confidence) {
    if (currentState === STATES.SCANNING) {
        const message = 'Traffic light post detected. ' +
            'Double tap to confirm this is your crossing, triple tap to keep scanning.';
        transitionTo(STATES.TARGET_DETECTED);
        setTimeout(() => {
            if (currentState === STATES.TARGET_DETECTED) {
                transitionTo(STATES.CONFIRM_TARGET, message);
            }
        }, 1500);
    }
};

// Called by ai.js when exactly two traffic-light posts are visible at once,
// clearly separated left/right -- instead of silently picking whichever one
// scored higher confidence, the user actually gets to choose which one to
// head toward. Goes through the same double-confirmation gate as every
// other decision in the app (CHOOSE_LEFT/CHOOSE_RIGHT are wrapped by the
// pending-confirmation layer above, same as CONFIRM_YES/CONFIRM_NO).
window.navassist.onMultiplePostsChoice = function() {
    if (currentState !== STATES.SCANNING) return;
    transitionTo(STATES.CHOOSE_POST,
        'Two traffic light posts detected: one on your left, one on your right. ' +
        'Double tap to choose the left post, triple tap to choose the right post.');
};

// Haptic-only direction guidance while walking toward the traffic light post.
// (Reverted back from audio -- team decided haptic during NAVIGATING after all.)
window.navassist.onDirectionDecided = function(direction) {
    if (currentState !== STATES.NAVIGATING) return;
    if (direction === 'LEFT')   sendHaptic('left',  'pulse');
    if (direction === 'RIGHT')  sendHaptic('right', 'pulse');
    if (direction === 'CENTRE') sendHaptic('both',  'pulse', 0.3, 200);
};

window.navassist.onArrived = function() {
    if (currentState === STATES.NAVIGATING) {
        navigatingStartHeading = null;
        transitionTo(STATES.REACHED, 'You have reached the crossing. Please wait.');
        setTimeout(() => {
            if (currentState === STATES.REACHED) {
                // Neutral message here -- we don't yet know the actual light
                // color. ai.js reads the crossing_seg light state (from
                // crossing/result) and speaks the real state within moments of
                // entering WAITING.
                transitionTo(STATES.WAITING, 'Checking the signal…');
            }
        }, 2000);
    }
};

window.navassist.onGreenDetected = function(direction) {
    if (currentState === STATES.WAITING) {
        // Single consolidated message -- previously ai.js ALSO spoke a
        // separate "Green man to your left." message for the same event,
        // causing two overlapping/back-to-back announcements. Now there's
        // exactly one, built here since this function already owns the
        // "you may cross" wording and the confirm instruction.
        const dirText = (direction && direction !== 'CENTRE') ? ` to your ${direction.toLowerCase()}` : '';
        transitionTo(STATES.CONFIRM_CROSSING,
            `Green man${dirText}. You may cross now. Double tap to confirm.`);
        sendHaptic('both', 'pulse', 1.0, 800);
    }
};

// Called by ai.js on every frame where a green man is detected. Gives a light
// directional nudge toward whichever side the green man is on -- this is the
// haptic that matches the "GREEN — GO LEFT/RIGHT" banner ai.js draws.
//
// Fires while SCANNING or WAITING:
//   * SCANNING -- so the REAL app (index.html) buzzes toward a green man the
//     moment it's seen off to one side, without first walking the whole
//     GPS -> confirm -> navigate -> reach chain. This is what makes "point the
//     camera at a green man on the left and feel the left motor" work directly.
//   * WAITING  -- the original behaviour, once the user is standing at the
//     crossing waiting for the signal.
// NOT in NAVIGATING/CROSSING: those states have their OWN directional haptics
// (onDirectionDecided toward the post, onCorridorDirection along the crossing),
// and a second green-man buzz on top would fight them.
//
// Has its OWN cooldown here (not in ai.js) since ai.js calls this every frame
// -- without throttling, this would buzz continuously while a green man is in view.
let lastGreenDirectionHapticAt = 0;
const GREEN_DIRECTION_HAPTIC_COOLDOWN_MS = 1000;

window.navassist.onGreenDirection = function(direction) {
    if (currentState !== STATES.WAITING && currentState !== STATES.SCANNING) return;
    const now = Date.now();
    if (now - lastGreenDirectionHapticAt < GREEN_DIRECTION_HAPTIC_COOLDOWN_MS) return;
    lastGreenDirectionHapticAt = now;

    // Left/right buzz the matching side; CENTRE buzzes both softly so "go
    // straight" still has a cue (the banner shows GO CENTRE for a green man
    // dead ahead). getDirection() in ai.js splits at 0.4 / 0.6 of frame width.
    if (direction === 'LEFT')       sendHaptic('left',  'pulse', 0.5, 200);
    else if (direction === 'RIGHT') sendHaptic('right', 'pulse', 0.5, 200);
    else                            sendHaptic('both',  'pulse', 0.3, 200);
};

// Called by ai.js when green is detected but flashing -- i.e. the signal is
// about to end. Deliberately does NOT invite the user to cross (does not
// call anything that would transition to CONFIRM_CROSSING) -- stays in
// WAITING and just warns, so the user waits for the next full green cycle
// instead of starting to cross on limited/expiring time.
let lastFlashWarningAt = 0;
const FLASH_WARNING_COOLDOWN_MS = 4000;

window.navassist.onGreenFlashing = function() {
    if (currentState !== STATES.WAITING) return;
    const now = Date.now();
    if (now - lastFlashWarningAt < FLASH_WARNING_COOLDOWN_MS) return;
    lastFlashWarningAt = now;
    speak('The signal is ending soon. Please wait for the next green light.');
};

// Called by ai.js with dotted-line corridor direction from the server-side
// segmentation model, while CROSSING. Deliberately shares the SAME cooldown
// (lastCrossingHapticAt / CROSSING_HAPTIC_COOLDOWN_MS) as the phone-compass
// heading-drift correction in checkHeadingDrift() -- whichever signal is
// ready first within a given cooldown window fires, the other is skipped
// for that window. This means vision-based corridor guidance and compass
// drift correction never stack or fire competing haptic pulses at once;
// when the dotted lines are visible they naturally take priority (they
// update faster than GPS/compass), and compass drift correction still
// covers moments where the corridor isn't visible (occlusion, etc).
window.navassist.onCorridorDirection = function(direction) {
    if (currentState !== STATES.CROSSING) return;
    const now = Date.now();
    if (now - lastCrossingHapticAt < CROSSING_HAPTIC_COOLDOWN_MS) return;
    lastCrossingHapticAt = now;

    if (direction === 'LEFT')       sendHaptic('left',  'pulse', 0.6, 250);
    else if (direction === 'RIGHT') sendHaptic('right', 'pulse', 0.6, 250);
    // CENTRE: already tracking the corridor centreline, no correction needed
};

// Called by ai.js when the dotted-line corridor runs out AND the light
// looks close/gone -- a vision-based hint that the user may have reached
// the far side. This is only a SIGNAL: it feeds into the exact same
// tap-confirmation gate GPS distance already triggers, it never completes
// the crossing by itself. Safe to call repeatedly -- promptArrivalConfirmation
// already guards on currentState === CROSSING internally.
window.navassist.onVisionArrivalSignal = function() {
    promptArrivalConfirmation();
};

// ============================================================
// Gesture detection -- ENTIRE SCREEN
// ============================================================

const TAP_WINDOW_MS = 400;
let tapCount = 0;
let tapTimer = null;

document.addEventListener('touchstart', handleScreenTap, { passive: true });
document.addEventListener('click', handleScreenTap);

function handleScreenTap(e) {
    if (e.target.closest('.splash-settings-link')) return;
    if (e.target.closest('.back-button')) return;
    if (e.target.closest('.settings-link')) return;

    tapCount++;
    if (navigator.vibrate) navigator.vibrate(20);

    clearTimeout(tapTimer);
    tapTimer = setTimeout(() => {
        classifyGesture(tapCount);
        tapCount = 0;
    }, TAP_WINDOW_MS);
}

// ============================================================
// Double-confirmation safety layer
// ============================================================
// Every CONFIRM_YES / CONFIRM_NO decision (choosing a crossing, confirming
// it's safe to cross, cycling to a different crossing) now requires TWO
// taps to actually commit. The first tap is echoed back out loud so the
// user can hear what was detected; only a FOLLOW-UP single tap actually
// executes it. This exists specifically to catch gesture misdetection --
// a double-tap misread as a triple-tap, or vice versa -- before it can
// silently commit to the wrong crossing decision. Given how serious a
// wrong crossing choice could be, the extra tap is worth it every time.
let pendingConfirmation = null;
let pendingConfirmationTimerId = null;
const PENDING_CONFIRMATION_TIMEOUT_MS = 5000;

const CONFIRMATION_ECHO_MESSAGES = {
    CONFIRM_TARGET: {
        CONFIRM_YES: 'You selected: yes, this is my crossing. Tap once more to confirm.',
        CONFIRM_NO:  'You selected: no, try a different crossing. Tap once more to confirm.'
    },
    CONFIRM_CROSSING: {
        CONFIRM_YES: 'You selected: yes, I will cross now. Tap once more to confirm.',
        CONFIRM_NO:  'You selected: no, keep waiting. Tap once more to confirm.'
    },
    CONFIRM_ARRIVAL: {
        CONFIRM_YES: 'You selected: yes, I have crossed safely. Tap once more to confirm.'
    },
    CHOOSE_POST: {
        CHOOSE_LEFT:  'You selected: the post on your left. Tap once more to confirm.',
        CHOOSE_RIGHT: 'You selected: the post on your right. Tap once more to confirm.'
    }
};

function armPendingConfirmation(intent) {
    pendingConfirmation = { intent, originState: currentState };
    const echo = (CONFIRMATION_ECHO_MESSAGES[currentState] || {})[intent] ||
        'Tap once more to confirm your selection.';
    speak(echo);

    if (pendingConfirmationTimerId) clearTimeout(pendingConfirmationTimerId);
    pendingConfirmationTimerId = setTimeout(() => {
        if (pendingConfirmation) {
            pendingConfirmation = null;
            speak('No confirmation received. Please make your selection again.');
        }
    }, PENDING_CONFIRMATION_TIMEOUT_MS);
}

function resolvePendingConfirmation(gesture) {
    const confirmed = pendingConfirmation;
    pendingConfirmation = null;
    if (pendingConfirmationTimerId) {
        clearTimeout(pendingConfirmationTimerId);
        pendingConfirmationTimerId = null;
    }

    if (gesture !== 'single_tap') {
        speak('Cancelled. Please make your selection again.');
        return;
    }

    // Fail-safe: if the state changed while the confirmation was pending
    // (e.g. glasses disconnected, or the light changed underneath the
    // user), the original decision no longer applies -- discard it rather
    // than executing a now-stale intent against a different situation.
    if (currentState !== confirmed.originState) {
        debugLog('State changed during pending confirmation -- discarding stale intent');
        return;
    }

    debugLog('Confirmation acknowledged -> ' + confirmed.intent);
    handleIntent(confirmed.intent);
}

function classifyGesture(count) {
    const gestureMap = { 1: 'single_tap', 2: 'double_tap', 3: 'triple_tap' };
    const gesture = gestureMap[Math.min(count, 3)];
    if (!gesture) return;

    // If a confirmation is currently pending, THIS tap decides it --
    // single tap confirms, anything else cancels. Normal gesture rules
    // are bypassed entirely while a confirmation is pending.
    if (pendingConfirmation) {
        resolvePendingConfirmation(gesture);
        return;
    }

    const rules = GESTURE_RULES[currentState] || {};
    const intent = rules[gesture];

    if (!intent) {
        switch (currentState) {
            case STATES.IDLE:
                speak('Waiting for glasses to connect. Please ensure your glasses are powered on.');
                break;
            case STATES.LOST_CONNECTION:
                speak('Glasses disconnected. Please check that your glasses are powered on and connected to WiFi.');
                break;
            case STATES.SCANNING:
            case STATES.RESOLVING:
                speak('Scanning in progress. Please wait.');
                break;
            case STATES.NAVIGATING:
                speak('Follow the haptic feedback on the glasses to reach the crossing.');
                break;
            case STATES.WAITING:
                speak('Please wait for the green man signal.');
                break;
            case STATES.CROSSING:
                speak('Cross now. Walk straight ahead.');
                break;
            case STATES.CONFIRM_ARRIVAL:
                speak('If you feel the tactile ground indicators, tap once to confirm you have crossed safely.');
                break;
            case STATES.READY:
                speak('Tap anywhere once to start scanning for a crossing.');
                break;
            case STATES.CONFIRM_TARGET:
            case STATES.CONFIRM_CROSSING:
                speak('Double tap anywhere to confirm yes, or triple tap to try again.');
                break;
            case STATES.CHOOSE_POST:
                speak('Double tap for the post on your left, or triple tap for the post on your right.');
                break;
            default:
                announceState(currentState);
        }
        return;
    }

    debugLog('Gesture: ' + gesture + ' → ' + intent);

    // CONFIRM_YES / CONFIRM_NO are the safety-critical decisions (choosing
    // a crossing, confirming it's safe to cross, cycling to another
    // crossing) -- require a follow-up tap before actually committing,
    // instead of acting on the very first tap alone.
    if (intent === 'CONFIRM_YES' || intent === 'CONFIRM_NO' ||
        intent === 'CHOOSE_LEFT' || intent === 'CHOOSE_RIGHT') {
        armPendingConfirmation(intent);
        return;
    }

    handleIntent(intent);
}

function handleIntent(intent) {
    switch (intent) {
        case 'START_SCAN':
            transitionTo(STATES.RESOLVING, 'Identifying nearby crossings. Please wait.');
            resolveJunction();
            break;
        case 'CONFIRM_YES':
            if (currentState === STATES.CONFIRM_TARGET) {
                navigatingStartHeading = currentHeading;
                transitionTo(STATES.NAVIGATING, 'Confirmed. Follow the haptic feedback.');
            } else if (currentState === STATES.CONFIRM_CROSSING) {
                crossingStartHeading = currentHeading;
                crossingStartLocation = currentLocation;  // may be null if no GPS fix yet -- handled below
                transitionTo(STATES.CROSSING, 'Cross now. Walk straight ahead.');

                lastCrossingDistance = 0;
                lastCrossingProgressAt = Date.now();
                crossingStallWarned = false;

                if (!crossingStartLocation) {
                    debugLog('No GPS fix at crossing start -- stall detection will not have distance data, only the absolute ceiling applies');
                }
                // Progress-stall detection: checks every few seconds whether
                // GPS distance is still increasing (still walking, however
                // long the crossing is) versus stalled (arrived and stopped,
                // or something's wrong) -- see checkCrossingStall(). This
                // replaces a fixed-duration assumption, which can't be
                // correct for both a narrow crossing and a wide one with a
                // median.
                crossingStallCheckId = setInterval(checkCrossingStall, 2000);

                // Absolute last-resort ceiling: only matters if there's
                // truly zero GPS data at all (no fix ever, e.g. testing
                // indoors) so the app can never get stuck waiting forever.
                // Deliberately generous -- this should almost never be what
                // actually fires if GPS is working.
                crossingFallbackTimerId = setTimeout(promptArrivalConfirmation, CROSSING_ABSOLUTE_CEILING_MS);
            } else if (currentState === STATES.CONFIRM_ARRIVAL) {
                completeCrossing();
            }
            break;
        case 'CONFIRM_NO':
            if (currentState === STATES.CONFIRM_TARGET) {
                tryNextCrossing();
            } else if (currentState === STATES.CONFIRM_CROSSING) {
                transitionTo(STATES.WAITING, 'Understood. Continuing to wait for green man.');
            }
            break;
        case 'CHOOSE_LEFT':
        case 'CHOOSE_RIGHT':
            if (currentState === STATES.CHOOSE_POST) {
                const chosenSide = intent === 'CHOOSE_LEFT' ? 'left' : 'right';
                const message = `Heading toward the post on your ${chosenSide}. ` +
                    'Double tap to confirm this is your crossing, triple tap to keep scanning.';
                transitionTo(STATES.TARGET_DETECTED);
                setTimeout(() => {
                    if (currentState === STATES.TARGET_DETECTED) {
                        transitionTo(STATES.CONFIRM_TARGET, message);
                    }
                }, 1500);
            }
            break;
        case 'RESET':
            transitionTo(STATES.READY, 'Ready. Tap once to scan again.');
            break;
    }
}

// ============================================================
// UI updates
// ============================================================

const UI_CONFIG = {
    IDLE:             { icon: '⏸', label: 'Waiting...', cls: '', hint: '', msg: 'Waiting for glasses to connect.' },
    READY:            { icon: '▶', label: 'Tap to start', cls: '', hint: '1 tap to start', msg: 'Glasses connected. Tap to scan.' },
    SCANNING:         { icon: '🔍', label: 'Scanning...', cls: 'state-confirm', hint: '', msg: 'Looking for a crossing.' },
    RESOLVING:        { icon: '📍', label: 'Locating...', cls: 'state-confirm', hint: '', msg: 'Finding nearby crossings.' },
    TARGET_DETECTED:  { icon: '🚦', label: 'Found!', cls: 'state-success', hint: '', msg: 'Traffic light found.' },
    CONFIRM_TARGET:   { icon: '?', label: 'Confirm', cls: 'state-confirm', hint: '2 taps = Yes  •  3 taps = No', msg: 'Is this the right crossing?' },
    CHOOSE_POST:      { icon: '↔', label: 'Choose post', cls: 'state-confirm', hint: '2 taps = Left  •  3 taps = Right', msg: 'Which traffic light post?' },
    NAVIGATING:       { icon: '→', label: 'Guiding...', cls: 'state-confirm', hint: '', msg: 'Follow haptic feedback.' },
    REACHED:          { icon: '✓', label: 'Arrived', cls: 'state-success', hint: '', msg: 'You have arrived.' },
    WAITING:          { icon: '🔴', label: 'Red man', cls: 'state-danger', hint: '', msg: 'Please wait.' },
    CONFIRM_CROSSING: { icon: '🟢', label: 'Cross now?', cls: 'state-confirm', hint: '2 taps = Yes  •  3 taps = Wait', msg: 'Green man. Ready to cross?' },
    CROSSING:         { icon: '🚶', label: 'Crossing', cls: 'state-success', hint: '', msg: 'Cross now.' },
    CONFIRM_ARRIVAL:  { icon: '?', label: 'Confirm arrival', cls: 'state-confirm', hint: '1 tap = confirm', msg: 'Feel the tactile indicators? Tap to confirm.' },
    COMPLETED:        { icon: '✓', label: 'Tap to reset', cls: 'state-success', hint: '1 tap to scan again', msg: 'Crossing complete.' },
    LOST_CONNECTION:  { icon: '✕', label: 'Disconnected', cls: '', hint: '', msg: 'Glasses disconnected.' }
};

function updateUI(state, customMessage) {
    const cfg = UI_CONFIG[state] || UI_CONFIG.IDLE;
    const btn = document.getElementById('mainButton');
    if (!btn) return;

    const trulyDisabled = (state === STATES.IDLE || state === STATES.LOST_CONNECTION);
    btn.disabled = trulyDisabled;
    btn.className = 'main-button ' + cfg.cls;
    btn.setAttribute('aria-label', cfg.label);
    document.getElementById('buttonIcon').textContent = cfg.icon;
    document.getElementById('buttonLabel').textContent = cfg.label;
    document.getElementById('gestureHint').textContent = cfg.hint;
    document.getElementById('contextMessage').textContent = customMessage || cfg.msg;
    document.getElementById('appStateLabel').textContent = state;
    armedGestures = Object.keys(GESTURE_RULES[state] || {});
}

function updateConnectionStatus(connected) {
    const el  = document.getElementById('connectionStatus');
    const txt = document.getElementById('connectionText');
    if (!el || !txt) return;
    el.className = 'status-indicator ' + (connected ? 'status-connected' : 'status-disconnected');
    txt.textContent = connected ? 'Glasses connected' : 'Glasses disconnected';
}

// ============================================================
// Audio
// ============================================================

function speak(text, interrupt) {
    if (!window.speechSynthesis) return;
    if (interrupt !== false) window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    const settings = JSON.parse(localStorage.getItem('navassist_settings') || '{}');
    u.rate   = (settings.speechRate  || 105) / 100;
    u.volume = (settings.audioVolume || 85)  / 100;
    window.speechSynthesis.speak(u);
}

const STATE_SPEECH = {
    READY:            'Ready. Tap anywhere once to start scanning for a crossing.',
    RESOLVING:        'Identifying nearby crossings. Please wait.',
    TARGET_DETECTED:  'Traffic light found.',
    CONFIRM_TARGET:   'Is this the right crossing? Double tap for yes, triple tap to try again.',
    CHOOSE_POST:      'Two traffic light posts detected. Double tap for the left post, triple tap for the right post.',
    NAVIGATING:       'Confirmed. Follow the haptic feedback toward the crossing.',
    REACHED:          'You have reached the crossing.',
    WAITING:          'Checking the signal…',
    CONFIRM_CROSSING: 'Green man. You may cross. Double tap to confirm.',
    CROSSING:         'Cross now. Walk straight ahead.',
    CONFIRM_ARRIVAL:  'If you feel the tactile ground indicators, tap once to confirm you have crossed safely.',
    COMPLETED:        'Crossing complete. Well done. Tap once to scan again.',
    LOST_CONNECTION:  'Glasses disconnected. Please check the connection.'
};

function announceState(state, customMessage) {
    const msg = customMessage || STATE_SPEECH[state];
    if (msg) speak(msg);
}

// ============================================================
// Debug log
// ============================================================

function debugLog(msg) {
    console.log('[NavAssist]', msg);
    const settings = JSON.parse(localStorage.getItem('navassist_settings') || '{}');
    const debugPanel = document.getElementById('debugPanel');
    if (debugPanel && settings.debugMode) {
        debugPanel.classList.add('visible');
        const el = document.getElementById('debugLog');
        if (el) {
            const line = document.createElement('div');
            line.textContent = new Date().toLocaleTimeString() + ' ' + msg;
            el.appendChild(line);
            el.scrollTop = el.scrollHeight;
            while (el.children.length > 20) el.removeChild(el.firstChild);
        }
    }
}

window.navassist.debugLog = debugLog;
window.navassist.speak = speak;
window.navassist.currentState = () => currentState;
window.navassist.STATES = STATES;
window.navassist.transitionTo = transitionTo;

// ============================================================
// Bootstrap
// ============================================================

window.addEventListener('load', async () => {
    await loadConfig();

    const settings = JSON.parse(localStorage.getItem('navassist_settings') || '{}');
    if (settings.highContrast) {
        document.body.classList.add('high-contrast');
    }

    const splashScreen = document.getElementById('splashScreen');
    const mainApp      = document.getElementById('mainApp');
    const splashButton = document.getElementById('splashButton');

    if (splashScreen) {
        const cameFromSettings = document.referrer.includes('settings.html');

        setTimeout(() => {
            if (cameFromSettings) {
                speak('Settings saved. Tap anywhere to start.', false);
            } else {
                speak('I am TOPH. Your Traffic Optical Personal Helper. Tap anywhere to start.', false);
            }
        }, 500);

        function startApp() {
            splashScreen.classList.add('hidden');
            mainApp.classList.remove('hidden');
            connectWebSocket();
            startGpsTracking();
            initPhoneCompass();
            updateUI(STATES.IDLE);
            speak('Waiting for glasses to connect. Please ensure your glasses are powered on and connected to WiFi.', false);
        }

        splashButton.addEventListener('click', startApp);

        document.addEventListener('touchstart', (e) => {
            if (e.target.closest('.splash-settings-link')) return;
            if (splashScreen && !splashScreen.classList.contains('hidden')) {
                startApp();
            }
        }, { passive: true });
    }
});
