/**
 * NavAssist — app.js
 * OWNER: Darrshini
 *
 * Responsibilities:
 * 1. Connect to Node.js server via WebSocket
 * 2. Receive sensor data from ESP32 (IMU, heartbeat, camera frames)
 * 3. Get GPS location from phone's browser API
 * 4. Send haptic motor commands back to ESP32
 * 5. Manage app state machine
 * 6. Audio announcements for accessibility
 *
 * Does NOT handle: AI inference, bounding box processing (→ ai.js)
 */

// ============================================================
// Config -- fetched from server so keys stay off client
// ============================================================

let config = {};

async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        config = await res.json();
        debugLog('Config loaded from server');
    } catch (e) {
        debugLog('Failed to load config: ' + e.message);
    }
}

// ============================================================
// State Machine
// ============================================================

const STATES = {
    IDLE: 'IDLE',
    READY: 'READY',
    SCANNING: 'SCANNING',
    RESOLVING: 'RESOLVING',
    TARGET_DETECTED: 'TARGET_DETECTED',
    CONFIRM_TARGET: 'CONFIRM_TARGET',
    NAVIGATING: 'NAVIGATING',
    REACHED: 'REACHED',
    WAITING: 'WAITING',
    CONFIRM_CROSSING: 'CONFIRM_CROSSING',
    CROSSING: 'CROSSING',
    COMPLETED: 'COMPLETED',
    LOST_CONNECTION: 'LOST_CONNECTION'
};

let currentState = STATES.IDLE;
let armedGestures = [];

function transitionTo(newState, message = null) {
    debugLog(`State: ${currentState} → ${newState}`);
    currentState = newState;
    updateUI(newState, message);
    announceState(newState, message);
}

// Which gestures are valid in each state
const GESTURE_RULES = {
    [STATES.READY]:           { single_tap: 'START_SCAN' },
    [STATES.CONFIRM_TARGET]:  { double_tap: 'CONFIRM_YES', triple_tap: 'CONFIRM_NO' },
    [STATES.CONFIRM_CROSSING]:{ double_tap: 'CONFIRM_YES', triple_tap: 'CONFIRM_NO' },
    [STATES.COMPLETED]:       { single_tap: 'RESET' }
};

// ============================================================
// WebSocket connection to Node.js server
// ============================================================

let ws = null;
const WS_URL = `ws://${location.host}/browser`;

function connectWebSocket() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        debugLog('Connected to NavAssist server');
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
        debugLog('Server disconnected. Retrying in 2s...');
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = () => ws.close();
}

function sendToEsp32(topic, payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
        topic,
        timestamp: Date.now(),
        payload
    }));
}

// ============================================================
// Incoming message router
// ============================================================

function handleIncomingMessage(envelope) {
    switch (envelope.topic) {
        case 'connection/event':
            handleConnectionEvent(envelope.payload);
            break;
        case 'system/heartbeat':
            handleHeartbeat(envelope.payload);
            break;
        case 'imu/orientation':
            handleImuReading(envelope.payload);
            break;
        case 'camera/image':
            // Forward to ai.js for processing
            if (typeof handleCameraFrame === 'function') {
                handleCameraFrame(envelope.payload);
            }
            break;
        default:
            debugLog(`Unknown topic: ${envelope.topic}`);
    }
}

// ============================================================
// Connection events
// ============================================================

let heartbeatTimer = null;

function handleConnectionEvent(payload) {
    const event = payload.event;
    debugLog('Connection event: ' + event);

    if (event === 'esp32_connected') {
        clearTimeout(heartbeatTimer);
        updateConnectionStatus(true);
        transitionTo(STATES.READY, 'Glasses connected. Tap once to start scanning.');
    } else if (event === 'esp32_disconnected') {
        updateConnectionStatus(false);
        if (currentState !== STATES.IDLE) {
            transitionTo(STATES.LOST_CONNECTION, 'Glasses disconnected. Please reconnect.');
        }
    }
}

function handleHeartbeat(payload) {
    // Reset the missed-heartbeat timer on every heartbeat received
    clearTimeout(heartbeatTimer);
    heartbeatTimer = setTimeout(() => {
        debugLog('Heartbeat timeout -- connection lost');
        handleConnectionEvent({ event: 'esp32_disconnected' });
    }, 6000); // 3 missed heartbeats (at 2s each) = connection lost
}

// ============================================================
// IMU data -- exposed globally so ai.js can read current heading
// ============================================================

let currentHeading = 0;
let imuCalibrated = false;

function handleImuReading(payload) {
    currentHeading = payload.heading_deg || 0;
    imuCalibrated = payload.calibrated || false;
    debugLog(`IMU: heading=${currentHeading.toFixed(1)}° calibrated=${imuCalibrated}`);
}

// Expose for ai.js
window.navassist = window.navassist || {};
window.navassist.getCurrentHeading = () => currentHeading;
window.navassist.isImuCalibrated = () => imuCalibrated;

// ============================================================
// GPS location -- using phone's browser Geolocation API
// ============================================================

let currentLocation = null;

function startGpsTracking() {
    if (!navigator.geolocation) {
        debugLog('GPS not available in this browser');
        return;
    }

    navigator.geolocation.watchPosition(
        (position) => {
            currentLocation = {
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
                accuracyMeters: position.coords.accuracy
            };
            debugLog(`GPS: ${currentLocation.latitude.toFixed(5)}, ${currentLocation.longitude.toFixed(5)} ±${currentLocation.accuracyMeters.toFixed(0)}m`);

            // If currently resolving junction, retry with new fix
            if (currentState === STATES.RESOLVING) {
                resolveJunction();
            }
        },
        (error) => {
            debugLog('GPS error: ' + error.message);
        },
        {
            enableHighAccuracy: true,
            maximumAge: 3000,
            timeout: 10000
        }
    );
}

// Expose for ai.js
window.navassist.getCurrentLocation = () => currentLocation;

// ============================================================
// Junction resolution -- GPS + Google Maps
// ============================================================

async function resolveJunction() {
    if (!currentLocation) {
        speak('Waiting for GPS signal. Please step outside if indoors.');
        return;
    }

    if (currentLocation.accuracyMeters > 25) {
        speak('GPS signal is weak. Please wait a moment.');
        return;
    }

    debugLog(`Resolving junction at ${currentLocation.latitude.toFixed(5)},${currentLocation.longitude.toFixed(5)}`);

    try {
        const url = `https://maps.googleapis.com/maps/api/place/nearbysearch/json` +
            `?location=${currentLocation.latitude},${currentLocation.longitude}` +
            `&radius=80&type=traffic_signals` +
            `&key=${config.googleMapsApiKey}`;

        const res = await fetch(url);
        const data = await res.json();

        if (data.status !== 'OK' || !data.results.length) {
            debugLog('No crossings found nearby');
            speak('No pedestrian crossing found nearby. Keep walking.');
            return;
        }

        // Calculate bearing from user to each crossing
        const crossings = data.results.map(place => ({
            name: place.name,
            bearing: calculateBearing(
                currentLocation.latitude, currentLocation.longitude,
                place.geometry.location.lat,
                place.geometry.location.lng
            )
        }));

        debugLog(`Found ${crossings.length} crossings`);

        // Pick the crossing closest to the user's current heading
        const target = crossings.reduce((best, crossing) => {
            const diff = headingDifference(currentHeading, crossing.bearing);
            return diff < headingDifference(currentHeading, best.bearing) ? crossing : best;
        });

        const headingDiff = headingDifference(currentHeading, target.bearing);

        if (headingDiff <= 30) {
            // User is facing this crossing -- good match
            transitionTo(STATES.TARGET_DETECTED,
                `Crossing found: ${target.name}. Is this correct?`);
            setTimeout(() => {
                if (currentState === STATES.TARGET_DETECTED) {
                    transitionTo(STATES.CONFIRM_TARGET,
                        'Double tap to confirm this crossing, triple tap to try again.');
                }
            }, 2000);
        } else {
            speak(`Found ${crossings.length} crossings nearby. Please face the crossing you want to use.`);
        }

    } catch (e) {
        debugLog('Maps API error: ' + e.message);
        speak('Could not look up nearby crossings. Please try again.');
    }
}

function calculateBearing(lat1, lng1, lat2, lng2) {
    const dLng = (lng2 - lng1) * Math.PI / 180;
    const lat1R = lat1 * Math.PI / 180;
    const lat2R = lat2 * Math.PI / 180;
    const y = Math.sin(dLng) * Math.cos(lat2R);
    const x = Math.cos(lat1R) * Math.sin(lat2R) -
              Math.sin(lat1R) * Math.cos(lat2R) * Math.cos(dLng);
    return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}

function headingDifference(a, b) {
    const diff = Math.abs(a - b) % 360;
    return diff > 180 ? 360 - diff : diff;
}

// ============================================================
// Haptic commands -- sent back to ESP32
// ============================================================

function sendHaptic(motor, pattern, intensity = 0.8, durationMs = 300) {
    sendToEsp32('haptic/command', { motor, pattern, intensity, duration_ms: durationMs });
    debugLog(`Haptic: ${motor} ${pattern} ${intensity}`);
}

// Expose for ai.js to call when it has direction guidance
window.navassist.sendHapticLeft  = () => sendHaptic('left',  'pulse');
window.navassist.sendHapticRight = () => sendHaptic('right', 'pulse');
window.navassist.sendHapticBoth  = () => sendHaptic('both',  'pulse', 1.0, 500);

// Called by ai.js when it determines the navigation direction
window.navassist.onDirectionDecided = function(direction) {
    if (currentState !== STATES.NAVIGATING) return;
    if (direction === 'LEFT')   sendHaptic('left',  'pulse');
    if (direction === 'RIGHT')  sendHaptic('right', 'pulse');
    if (direction === 'CENTRE') sendHaptic('both',  'pulse', 0.3, 200);
};

// Called by ai.js when it determines the user has arrived
window.navassist.onArrived = function() {
    if (currentState === STATES.NAVIGATING) {
        transitionTo(STATES.REACHED, 'You have reached the crossing. Please wait.');
        setTimeout(() => {
            if (currentState === STATES.REACHED) {
                transitionTo(STATES.WAITING, 'Red man. Please wait for the green signal.');
            }
        }, 2000);
    }
};

// Called by ai.js when it detects green light
window.navassist.onGreenDetected = function() {
    if (currentState === STATES.WAITING) {
        transitionTo(STATES.CONFIRM_CROSSING,
            'Green man. You may cross. Double tap to confirm.');
        sendHaptic('both', 'pulse', 1.0, 800);
    }
};

// ============================================================
// Button gesture detection
// ============================================================

const mainButton = document.getElementById('mainButton');
const TAP_WINDOW_MS = 400;
let tapCount = 0;
let tapTimer = null;

mainButton.addEventListener('click', () => {
    tapCount++;
    // Tiny immediate vibration so user knows tap registered
    if (navigator.vibrate) navigator.vibrate(20);

    clearTimeout(tapTimer);
    tapTimer = setTimeout(() => {
        classifyGesture(tapCount);
        tapCount = 0;
    }, TAP_WINDOW_MS);
});

mainButton.addEventListener('touchstart', () => {
    const label = mainButton.getAttribute('aria-label') || 'Button';
    speak(label, false);
}, { passive: true });

function classifyGesture(count) {
    const gestureMap = { 1: 'single_tap', 2: 'double_tap', 3: 'triple_tap' };
    const gesture = gestureMap[Math.min(count, 3)];
    if (!gesture) return;

    const rules = GESTURE_RULES[currentState] || {};
    const intent = rules[gesture];

    if (!intent) {
        speak('This action is not available right now.');
        return;
    }

    debugLog(`Gesture: ${gesture} → ${intent}`);
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
                transitionTo(STATES.CROSSING, 'Cross now. Walk straight ahead.');
                // Auto-complete crossing after 15 seconds
                setTimeout(() => {
                    if (currentState === STATES.CROSSING) {
                        transitionTo(STATES.COMPLETED, 'Crossing complete. Tap once to scan again.');
                    }
                }, 15000);
            }
            break;
        case 'CONFIRM_NO':
            if (currentState === STATES.CONFIRM_TARGET) {
                transitionTo(STATES.RESOLVING, 'Trying again. Please face the crossing you want.');
                setTimeout(resolveJunction, 1000);
            } else if (currentState === STATES.CONFIRM_CROSSING) {
                transitionTo(STATES.WAITING, 'Understood. Continuing to wait for green man.');
            }
            break;
        case 'RESET':
            transitionTo(STATES.READY, 'Ready. Tap once to start scanning.');
            break;
    }
}

// ============================================================
// UI updates
// ============================================================

const UI_CONFIG = {
    IDLE:             { icon: '⏸', label: 'Waiting...', enabled: false, cls: '', hint: '', msg: 'Waiting for connection.' },
    READY:            { icon: '▶', label: 'Tap to start', enabled: true, cls: '', hint: '1 tap to start', msg: 'Ready to scan.' },
    SCANNING:         { icon: '🔍', label: 'Scanning...', enabled: false, cls: 'state-confirm', hint: '', msg: 'Scanning...' },
    RESOLVING:        { icon: '📍', label: 'Locating...', enabled: false, cls: 'state-confirm', hint: '', msg: 'Finding nearby crossings.' },
    TARGET_DETECTED:  { icon: '🚦', label: 'Found!', enabled: false, cls: 'state-success', hint: '', msg: 'Traffic light found.' },
    CONFIRM_TARGET:   { icon: '?', label: 'Confirm', enabled: true, cls: 'state-confirm', hint: '2 taps = Yes  •  3 taps = No', msg: 'Is this the right crossing?' },
    NAVIGATING:       { icon: '→', label: 'Guiding...', enabled: false, cls: 'state-confirm', hint: '', msg: 'Follow haptic feedback.' },
    REACHED:          { icon: '✓', label: 'Arrived', enabled: false, cls: 'state-success', hint: '', msg: 'You have arrived.' },
    WAITING:          { icon: '🔴', label: 'Red man', enabled: false, cls: 'state-danger', hint: '', msg: 'Please wait.' },
    CONFIRM_CROSSING: { icon: '🟢', label: 'Cross now?', enabled: true, cls: 'state-confirm', hint: '2 taps = Yes  •  3 taps = Wait', msg: 'Green man. Ready to cross?' },
    CROSSING:         { icon: '🚶', label: 'Crossing', enabled: false, cls: 'state-success', hint: '', msg: 'Cross now.' },
    COMPLETED:        { icon: '✓', label: 'Tap to reset', enabled: true, cls: 'state-success', hint: '1 tap to scan again', msg: 'Crossing complete.' },
    LOST_CONNECTION:  { icon: '✕', label: 'Disconnected', enabled: false, cls: '', hint: '', msg: 'Glasses disconnected.' }
};

function updateUI(state, customMessage = null) {
    const cfg = UI_CONFIG[state] || UI_CONFIG.IDLE;
    const btn = document.getElementById('mainButton');
    btn.disabled = !cfg.enabled;
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
    const el = document.getElementById('connectionStatus');
    const txt = document.getElementById('connectionText');
    el.className = 'status-indicator ' + (connected ? 'status-connected' : 'status-disconnected');
    txt.textContent = connected ? 'Glasses connected' : 'Glasses disconnected';
}

// ============================================================
// Audio -- speaks every state change aloud
// ============================================================

function speak(text, interrupt = true) {
    if (!window.speechSynthesis) return;
    if (interrupt) window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.05; u.volume = 1.0;
    window.speechSynthesis.speak(u);
}

const STATE_SPEECH = {
    READY:            'Ready. Tap once to start scanning for a crossing.',
    RESOLVING:        'Identifying your location and nearby crossings.',
    TARGET_DETECTED:  'Traffic light found.',
    CONFIRM_TARGET:   'Is this the right crossing? Double tap for yes, triple tap to try again.',
    NAVIGATING:       'Confirmed. Follow the haptic feedback toward the crossing.',
    REACHED:          'You have reached the crossing.',
    WAITING:          'Red man. Please wait.',
    CONFIRM_CROSSING: 'Green man. You may cross. Double tap to confirm.',
    CROSSING:         'Cross now. Walk straight ahead.',
    COMPLETED:        'Crossing complete. Well done. Tap once to scan again.',
    LOST_CONNECTION:  'Glasses disconnected. Please check the connection.'
};

function announceState(state, customMessage = null) {
    const msg = customMessage || STATE_SPEECH[state];
    if (msg) speak(msg);
}

// ============================================================
// Debug log -- visible at bottom of screen during development
// ============================================================

function debugLog(msg) {
    console.log('[NavAssist]', msg);
    const el = document.getElementById('debugLog');
    if (el) {
        const line = document.createElement('div');
        line.textContent = `${new Date().toLocaleTimeString()} ${msg}`;
        el.appendChild(line);
        el.scrollTop = el.scrollHeight;
        // Keep only last 20 lines
        while (el.children.length > 20) el.removeChild(el.firstChild);
    }
}

// Expose for ai.js
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
    connectWebSocket();
    startGpsTracking();
    updateUI(STATES.IDLE);

    setTimeout(() => {
        speak('NavAssist loaded. Waiting for glasses to connect.', false);
    }, 1000);
});
