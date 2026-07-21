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
 * Audio logic follows the same approach as crossing.js (live.html):
 * vision-driven transitions with NO tap confirmations. The only taps
 * are: single-tap to START scanning, and single-tap to RESET after completion.
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
    RESOLVING:        'RESOLVING',
    NAVIGATING:       'NAVIGATING',
    REACHED:          'REACHED',
    WAITING:          'WAITING',
    CROSSING:         'CROSSING',
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
    [STATES.READY]:     { single_tap: 'START_SCAN' },
    [STATES.COMPLETED]: { single_tap: 'RESET' }
};

// ============================================================
// WebSocket
// ============================================================

let ws = null;
const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/live';

function connectWebSocket() {
    ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        debugLog('Connected to server');
    };

    ws.onmessage = (event) => {
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

function sendToPi(topic, payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ topic, timestamp: Date.now(), payload }));
}

// ============================================================
// Incoming message router
// ============================================================

function handleIncomingMessage(envelope) {
    switch (envelope.topic) {
        case 'connection/event': handleConnectionEvent(envelope.payload); break;
        case 'system/heartbeat': handleHeartbeat(envelope.payload); break;
        case 'imu/orientation':  handleImuReading(envelope.payload); break;
        case 'crossing/result':
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

let navigatingStartHeading = null;
let lastTurnWarningAt = 0;
const TURN_WARNING_COOLDOWN_MS = 5000;

let crossingStartHeading = null;
let lastCrossingHapticAt = 0;
const CROSSING_DRIFT_THRESHOLD_DEG = 20;
const CROSSING_HAPTIC_COOLDOWN_MS = 1500;

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

function checkHeadingDrift() {
    if (currentState === STATES.NAVIGATING && navigatingStartHeading !== null) {
        const drift = angleDiffDeg(currentHeading, navigatingStartHeading);
        const now = Date.now();
        if (Math.abs(drift) > SIGNIFICANT_TURN_DEG && now - lastTurnWarningAt > TURN_WARNING_COOLDOWN_MS) {
            lastTurnWarningAt = now;
            speak('You have turned away. Follow the haptic feedback to re-orient toward the crossing.');
            navigatingStartHeading = currentHeading;
        }
    }

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

let crossingStartLocation = null;
let crossingFallbackTimerId = null;
let crossingStallCheckId = null;
let lastCrossingDistance = 0;
let lastCrossingProgressAt = 0;
let crossingStallWarned = false;
const CROSSING_DISTANCE_THRESHOLD_M = 8;
const CROSSING_PROGRESS_MIN_DELTA_M = 1;
const CROSSING_STALL_WARNING_MS = 8000;
const CROSSING_ABSOLUTE_CEILING_MS = 90000;

function distanceMeters(lat1, lng1, lat2, lng2) {
    const R = 6371000;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLng = (lng2 - lng1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 +
        Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
        Math.sin(dLng / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function checkCrossingStall() {
    if (currentState !== STATES.CROSSING) return;
    const stalledFor = Date.now() - lastCrossingProgressAt;
    if (stalledFor >= CROSSING_STALL_WARNING_MS && !crossingStallWarned) {
        crossingStallWarned = true;
        speak('Still walking? Keep going straight.');
    }
}

function completeCrossing() {
    if (currentState !== STATES.CROSSING) return;
    crossingStartHeading = null;
    crossingStartLocation = null;
    if (crossingStallCheckId) {
        clearInterval(crossingStallCheckId);
        crossingStallCheckId = null;
    }
    if (crossingFallbackTimerId) {
        clearTimeout(crossingFallbackTimerId);
        crossingFallbackTimerId = null;
    }
    transitionTo(STATES.COMPLETED,
        'You have reached the other side. Crossing complete. Tap once to scan again.');
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
                    crossingStallWarned = false;
                }

                if (moved >= CROSSING_DISTANCE_THRESHOLD_M) {
                    completeCrossing();
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

const OVERPASS_ENDPOINTS = [
    'https://overpass-api.de/api/interpreter',
    'https://overpass.kumi.systems/api/interpreter',
    'https://lz4.overpass-api.de/api/interpreter'
];
const CROSSING_SEARCH_RADIUS_M = 100;
const ROAD_NAME_SEARCH_RADIUS_M = 20;

async function resolveJunction() {
    if (currentLocation) {
        await resolveJunctionWithGPS();
    } else {
        resolveJunctionWithCameraOnly();
    }
}

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
        }
    }
    throw lastError || new Error('All Overpass endpoints failed');
}

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
    return null;
}

async function resolveJunctionWithGPS() {
    if (currentLocation.accuracyMeters > 30) {
        speak('GPS signal is weak. Please wait or move to an open area.');
        setTimeout(resolveJunction, 3000);
        return;
    }

    try {
        const query = '[out:json][timeout:15];' +
            'node["highway"="crossing"]["crossing"="traffic_signals"]' +
            '(around:' + CROSSING_SEARCH_RADIUS_M + ',' +
            currentLocation.latitude + ',' + currentLocation.longitude + ');' +
            'out body;';

        const data = await overpassQuery(query);

        if (!data.elements || !data.elements.length) {
            debugLog('No pedestrian crossings found via Overpass within ' + CROSSING_SEARCH_RADIUS_M + 'm');
            resolveJunctionWithCameraOnly();
            return;
        }

        const candidates = data.elements.map(el => ({
            latitude: el.lat,
            longitude: el.lon,
            distance: haversineMeters(
                currentLocation.latitude, currentLocation.longitude,
                el.lat, el.lon
            )
        })).sort((a, b) => a.distance - b.distance);

        const nearest = candidates[0];
        const name = await findRoadNameNear(nearest.latitude, nearest.longitude);

        nearbyCrossings = [{
            name: name || 'a nearby pedestrian crossing',
            distance: nearest.distance
        }];

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

    const crossing = nearbyCrossings[0];
    const distText = crossing.distance < 20
        ? 'very close'
        : 'about ' + Math.round(crossing.distance) + ' metres away';

    speak('Crossing found: ' + crossing.name + ', ' + distText + '. Scanning for the traffic light.');
    transitionTo(STATES.SCANNING);
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

// --- Callbacks for ai.js (all vision-driven, no tap confirmations) ---

window.navassist.onTrafficLightVisible = function(confidence, direction) {
    if (currentState === STATES.SCANNING) {
        const dirText = (direction && direction !== 'CENTRE') ? ' to your ' + direction.toLowerCase() : '';
        navigatingStartHeading = currentHeading;
        transitionTo(STATES.NAVIGATING,
            'Traffic light post detected' + dirText + '. Guiding you toward the crossing.');
    }
};

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
                transitionTo(STATES.WAITING, 'Checking the signal…');
            }
        }, 2000);
    }
};

window.navassist.onGreenCross = function() {
    if (currentState === STATES.WAITING) {
        crossingStartHeading = currentHeading;
        crossingStartLocation = currentLocation;
        transitionTo(STATES.CROSSING, 'Green man. You may cross now.');
        sendHaptic('both', 'pulse', 1.0, 800);

        lastCrossingDistance = 0;
        lastCrossingProgressAt = Date.now();
        crossingStallWarned = false;

        if (!crossingStartLocation) {
            debugLog('No GPS fix at crossing start -- stall detection will not have distance data');
        }
        crossingStallCheckId = setInterval(checkCrossingStall, 2000);
        crossingFallbackTimerId = setTimeout(completeCrossing, CROSSING_ABSOLUTE_CEILING_MS);
    }
};

let lastGreenDirectionHapticAt = 0;
const GREEN_DIRECTION_HAPTIC_COOLDOWN_MS = 1000;

window.navassist.onGreenDirection = function(direction) {
    if (currentState !== STATES.WAITING && currentState !== STATES.SCANNING) return;
    const now = Date.now();
    if (now - lastGreenDirectionHapticAt < GREEN_DIRECTION_HAPTIC_COOLDOWN_MS) return;
    lastGreenDirectionHapticAt = now;

    if (direction === 'LEFT')       sendHaptic('left',  'pulse', 0.5, 200);
    else if (direction === 'RIGHT') sendHaptic('right', 'pulse', 0.5, 200);
    else                            sendHaptic('both',  'pulse', 0.3, 200);
};

window.navassist.onCorridorDirection = function(direction) {
    if (currentState !== STATES.CROSSING) return;
    const now = Date.now();
    if (now - lastCrossingHapticAt < CROSSING_HAPTIC_COOLDOWN_MS) return;
    lastCrossingHapticAt = now;

    if (direction === 'LEFT')       sendHaptic('left',  'pulse', 0.6, 250);
    else if (direction === 'RIGHT') sendHaptic('right', 'pulse', 0.6, 250);
};

window.navassist.onVisionArrivalSignal = function() {
    completeCrossing();
};

// ============================================================
// Gesture detection — ENTIRE SCREEN
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

function classifyGesture(count) {
    const gestureMap = { 1: 'single_tap', 2: 'double_tap', 3: 'triple_tap' };
    const gesture = gestureMap[Math.min(count, 3)];
    if (!gesture) return;

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
            case STATES.READY:
                speak('Tap anywhere once to start scanning for a crossing.');
                break;
            default:
                announceState(currentState);
        }
        return;
    }

    debugLog('Gesture: ' + gesture + ' → ' + intent);
    handleIntent(intent);
}

function handleIntent(intent) {
    switch (intent) {
        case 'START_SCAN':
            transitionTo(STATES.RESOLVING, 'Identifying nearby crossings. Please wait.');
            resolveJunction();
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
    NAVIGATING:       { icon: '→', label: 'Guiding...', cls: 'state-confirm', hint: '', msg: 'Follow haptic feedback.' },
    REACHED:          { icon: '✓', label: 'Arrived', cls: 'state-success', hint: '', msg: 'You have arrived.' },
    WAITING:          { icon: '🔴', label: 'Waiting', cls: 'state-danger', hint: '', msg: 'Please wait for green.' },
    CROSSING:         { icon: '🚶', label: 'Crossing', cls: 'state-success', hint: '', msg: 'Cross now.' },
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
    NAVIGATING:       'Follow the haptic feedback toward the crossing.',
    REACHED:          'You have reached the crossing.',
    WAITING:          'Checking the signal…',
    CROSSING:         'Cross now. Walk straight ahead.',
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
