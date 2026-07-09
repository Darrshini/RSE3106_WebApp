/**
 * NavAssist — app.js
 * OWNER: Darrshini
 *
 * Responsibilities:
 * 1. Connect to Node.js server via WebSocket
 * 2. Receive sensor data from ESP32 (IMU, heartbeat, camera frames)
 * 3. Get GPS location from phone browser API
 * 4. Send haptic motor commands back to ESP32
 * 5. Manage app state machine
 * 6. Audio announcements for accessibility
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
    TARGET_DETECTED:  'TARGET_DETECTED',
    CONFIRM_TARGET:   'CONFIRM_TARGET',
    NAVIGATING:       'NAVIGATING',
    REACHED:          'REACHED',
    WAITING:          'WAITING',
    CONFIRM_CROSSING: 'CONFIRM_CROSSING',
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
    [STATES.READY]:            { single_tap: 'START_SCAN' },
    [STATES.CONFIRM_TARGET]:   { double_tap: 'CONFIRM_YES', triple_tap: 'CONFIRM_NO' },
    [STATES.CONFIRM_CROSSING]: { double_tap: 'CONFIRM_YES', triple_tap: 'CONFIRM_NO' },
    [STATES.COMPLETED]:        { single_tap: 'RESET' }
};

// ============================================================
// WebSocket
// ============================================================

let ws = null;
const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/browser';

function connectWebSocket() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        debugLog('Connected to server');
    };

    ws.onmessage = (event) => {
        try {
            const envelope = JSON.parse(event.data);
            handleIncomingMessage(envelope);
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

function sendToEsp32(topic, payload) {
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
        case 'camera/image':
            if (typeof handleCameraFrame === 'function') {
                handleCameraFrame(envelope.payload);
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

    if (event === 'esp32_connected') {
        clearTimeout(heartbeatTimer);
        updateConnectionStatus(true);
        speak('Glasses connected successfully. Tap anywhere to start scanning.', true);
        transitionTo(STATES.READY);
    } else if (event === 'esp32_disconnected') {
        updateConnectionStatus(false);
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
        handleConnectionEvent({ event: 'esp32_disconnected' });
    }, 6000);
}

// ============================================================
// IMU + compass
// ============================================================

let currentHeading = 0;
let imuCalibrated = false;
let phoneCompassAvailable = false;
let cumulativeGyroZ = 0;
let lastGyroTime = Date.now();
const SIGNIFICANT_TURN_DEG = 45;

// Crossing drift correction
let crossingStartHeading = null;
let lastCrossingHapticAt = 0;
const CROSSING_DRIFT_THRESHOLD_DEG = 20;   // how far off-line before nudging
const CROSSING_HAPTIC_COOLDOWN_MS = 1500;  // avoid buzzing every IMU sample

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
    if (event.alpha !== null) {
        currentHeading = event.alpha;
        imuCalibrated = true;
    }
}

function handleImuReading(payload) {
    const gyroZ = payload.gyro_z || 0;

    if (!phoneCompassAvailable && payload.heading_deg) {
        currentHeading = payload.heading_deg;
        imuCalibrated = payload.calibrated || false;
    }

    // Track cumulative rotation for turn detection
    const now = Date.now();
    const dt = (now - lastGyroTime) / 1000;
    lastGyroTime = now;
    cumulativeGyroZ += gyroZ * dt;

    if (Math.abs(cumulativeGyroZ) > SIGNIFICANT_TURN_DEG) {
        if (currentState === STATES.NAVIGATING) {
            speak('You have turned away. Follow the haptic feedback to re-orient toward the crossing.');
        }
        cumulativeGyroZ = 0;
    }

    // Crossing drift correction: nudge the user back toward a straight line
    // if their heading drifts too far from the heading captured the moment
    // they started crossing. NOTE: verify the sign below (which motor fires
    // for which drift direction) against the real ESP32 heading_deg
    // convention and motor wiring -- this assumes compass-style clockwise
    // degrees, where a positive drift means the user turned right and needs
    // a left-nudge to correct back.
    if (currentState === STATES.CROSSING && crossingStartHeading !== null) {
        const drift = angleDiffDeg(currentHeading, crossingStartHeading);
        const nowMs = Date.now();
        if (Math.abs(drift) > CROSSING_DRIFT_THRESHOLD_DEG &&
            nowMs - lastCrossingHapticAt > CROSSING_HAPTIC_COOLDOWN_MS) {
            lastCrossingHapticAt = nowMs;
            if (drift > 0) sendHaptic('left', 'pulse', 0.6, 250);
            else            sendHaptic('right', 'pulse', 0.6, 250);
        }
    }

    debugLog('IMU: heading=' + currentHeading.toFixed(1) + '° gyroZ=' + gyroZ.toFixed(1));
}

window.navassist = window.navassist || {};
window.navassist.getCurrentHeading = () => currentHeading;
window.navassist.isImuCalibrated = () => imuCalibrated;

// ============================================================
// GPS
// ============================================================

let currentLocation = null;

function startGpsTracking() {
    if (location.protocol === 'http:' && location.hostname !== 'localhost') {
        debugLog('GPS blocked on HTTP -- needs HTTPS');
        return;
    }

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
    if (currentLocation && location.protocol === 'https:') {
        await resolveJunctionWithGPS();
    } else {
        resolveJunctionWithCameraOnly();
    }
}

async function resolveJunctionWithGPS() {
    if (currentLocation.accuracyMeters > 30) {
        speak('GPS signal is weak. Please wait or move to an open area.');
        setTimeout(resolveJunction, 3000);
        return;
    }

    try {
        const url = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json' +
            '?location=' + currentLocation.latitude + ',' + currentLocation.longitude +
            '&radius=80&type=traffic_signals' +
            '&key=' + config.googleMapsApiKey;

        const res = await fetch(url);
        const data = await res.json();

        if (data.status !== 'OK' || !data.results.length) {
            resolveJunctionWithCameraOnly();
            return;
        }

        nearbyCrossings = data.results.map(place => ({
            name: place.name || 'Pedestrian crossing',
            distance: haversineMeters(
                currentLocation.latitude, currentLocation.longitude,
                place.geometry.location.lat,
                place.geometry.location.lng
            )
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
            cumulativeGyroZ = 0;
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
    sendToEsp32('haptic/command', { motor, pattern, intensity, duration_ms: durationMs });
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
                cumulativeGyroZ = 0;
            }
        }, 1500);
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
        transitionTo(STATES.REACHED, 'You have reached the crossing. Please wait.');
        setTimeout(() => {
            if (currentState === STATES.REACHED) {
                // Neutral message here -- we don't yet know the actual light
                // color. ai.js's red/green detection (handleGuidance) speaks
                // the real state within moments of entering WAITING.
                transitionTo(STATES.WAITING, 'Checking the signal…');
            }
        }, 2000);
    }
};

window.navassist.onGreenDetected = function() {
    if (currentState === STATES.WAITING) {
        transitionTo(STATES.CONFIRM_CROSSING,
            'Green man. You may cross now. Double tap to confirm.');
        sendHaptic('both', 'pulse', 1.0, 800);
    }
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
            case STATES.CONFIRM_TARGET:
            case STATES.CONFIRM_CROSSING:
                speak('Double tap anywhere to confirm yes, or triple tap to try again.');
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
        case 'CONFIRM_YES':
            if (currentState === STATES.CONFIRM_TARGET) {
                transitionTo(STATES.NAVIGATING, 'Confirmed. Follow the haptic feedback.');
            } else if (currentState === STATES.CONFIRM_CROSSING) {
                crossingStartHeading = currentHeading;
                transitionTo(STATES.CROSSING, 'Cross now. Walk straight ahead.');
                setTimeout(() => {
                    if (currentState === STATES.CROSSING) {
                        crossingStartHeading = null;
                        transitionTo(STATES.COMPLETED, 'Crossing complete. Tap once to scan again.');
                    }
                }, 15000);
            }
            break;
        case 'CONFIRM_NO':
            if (currentState === STATES.CONFIRM_TARGET) {
                tryNextCrossing();
            } else if (currentState === STATES.CONFIRM_CROSSING) {
                transitionTo(STATES.WAITING, 'Understood. Continuing to wait for green man.');
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
    NAVIGATING:       { icon: '→', label: 'Guiding...', cls: 'state-confirm', hint: '', msg: 'Follow haptic feedback.' },
    REACHED:          { icon: '✓', label: 'Arrived', cls: 'state-success', hint: '', msg: 'You have arrived.' },
    WAITING:          { icon: '🔴', label: 'Red man', cls: 'state-danger', hint: '', msg: 'Please wait.' },
    CONFIRM_CROSSING: { icon: '🟢', label: 'Cross now?', cls: 'state-confirm', hint: '2 taps = Yes  •  3 taps = Wait', msg: 'Green man. Ready to cross?' },
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
    TARGET_DETECTED:  'Traffic light found.',
    CONFIRM_TARGET:   'Is this the right crossing? Double tap for yes, triple tap to try again.',
    NAVIGATING:       'Confirmed. Follow the haptic feedback toward the crossing.',
    REACHED:          'You have reached the crossing.',
    WAITING:          'Checking the signal…',
    CONFIRM_CROSSING: 'Green man. You may cross. Double tap to confirm.',
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
                speak('NavAssist. Settings saved. Tap anywhere to start.', false);
            } else {
                speak('NavAssist. Assistive navigation for pedestrian crossings. Tap anywhere to start.', false);
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
