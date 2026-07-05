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
        // Clear announcement so user knows glasses are ready
        speak('Glasses connected successfully. Tap anywhere to start scanning.', true);
        transitionTo(STATES.READY);
    } else if (event === 'esp32_disconnected') {
        updateConnectionStatus(false);
        if (currentState !== STATES.IDLE) {
            speak('Glasses disconnected. Please check the connection.', true);
            transitionTo(STATES.LOST_CONNECTION);
        } else {
            // Still in IDLE -- just update status, no state change needed
            speak('Waiting for glasses to connect. Please ensure glasses are powered on and connected to WiFi.', true);
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
// IMU data -- from ESP32 glasses (relative motion)
// AND phone compass (absolute heading) combined
// ============================================================

let currentHeading = 0;      // absolute compass heading from phone
let imuCalibrated = false;
let phoneCompassAvailable = false;

// Phone's own compass -- primary source for absolute heading
// This works regardless of whether ESP32 IMU has a magnetometer
function initPhoneCompass() {
    if (typeof DeviceOrientationEvent === 'undefined') {
        debugLog('Phone compass not available in this browser');
        return;
    }

    // iOS 13+ requires permission
    if (typeof DeviceOrientationEvent.requestPermission === 'function') {
        DeviceOrientationEvent.requestPermission()
            .then(response => {
                if (response === 'granted') {
                    window.addEventListener('deviceorientation', handleDeviceOrientation);
                    phoneCompassAvailable = true;
                    debugLog('Phone compass permission granted');
                }
            })
            .catch(err => debugLog('Compass permission denied: ' + err));
    } else {
        // Android -- no permission needed
        window.addEventListener('deviceorientation', handleDeviceOrientation);
        phoneCompassAvailable = true;
        debugLog('Phone compass initialised');
    }
}

function handleDeviceOrientation(event) {
    // event.alpha = compass heading (0-360°, 0=North) on Android
    if (event.alpha !== null) {
        currentHeading = event.alpha;
        imuCalibrated = true;
    }
}

function handleImuReading(payload) {
    // ESP32 IMU gives us pitch/roll/gyro -- useful for detecting
    // if user is turning, but phone compass gives absolute heading
    const pitch  = payload.pitch_deg || 0;
    const roll   = payload.roll_deg  || 0;
    const gyroZ  = payload.gyro_z    || 0;

    // Only use ESP32 heading if phone compass unavailable
    if (!phoneCompassAvailable && payload.heading_deg) {
        currentHeading = payload.heading_deg;
        imuCalibrated  = payload.calibrated || false;
    }

    debugLog(`IMU: heading=${currentHeading.toFixed(1)}° pitch=${pitch.toFixed(1)}° gyroZ=${gyroZ.toFixed(1)}`);
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
    // GPS requires HTTPS or localhost in modern browsers
    // If on HTTP (e.g. AWS without SSL), GPS will be blocked
    if (location.protocol === 'http:' && location.hostname !== 'localhost') {
        debugLog('GPS blocked: requires HTTPS. GPS features disabled on HTTP.');
        // Don't show error to user -- GPS is enhancement, not blocker
        // Junction lookup will be skipped gracefully if no location
        return;
    }

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
// Gesture detection -- listens on ENTIRE SCREEN
// Visually impaired users cannot reliably find a specific button
// so any tap anywhere on the screen counts as a gesture.
// The button on screen is purely visual reference for sighted helpers.
// ============================================================

const TAP_WINDOW_MS = 400;
let tapCount = 0;
let tapTimer = null;
let lastTapTime = 0;

// Listen on the whole document, not just the button
document.addEventListener('touchstart', handleScreenTap, { passive: true });
document.addEventListener('click', handleScreenTap);

function handleScreenTap(e) {
    // Ignore taps on the settings link -- let that navigate normally
    if (e.target.closest('.settings-link')) return;
    // Ignore taps on the back button in settings
    if (e.target.closest('.back-button')) return;

    tapCount++;
    lastTapTime = Date.now();

    // Immediate feedback -- tiny vibration so user knows tap registered
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
        // Give a helpful, state-specific message rather than a vague one
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
                // Re-announce current state message
                announceState(currentState);
        }
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

    // Button is never truly disabled -- tapping anywhere on screen works.
    // We just change the visual appearance to reflect state for sighted helpers.
    // Keep disabled only for IDLE and LOST_CONNECTION where nothing should happen.
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

    // Read settings saved by settings.js
    const settings = JSON.parse(localStorage.getItem('navassist_settings') || '{}');
    u.rate   = (settings.speechRate   || 105) / 100;
    u.volume = (settings.audioVolume  || 85)  / 100;

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

    // Only show in debug panel if debug mode is enabled in settings
    const settings = JSON.parse(localStorage.getItem('navassist_settings') || '{}');
    const debugPanel = document.getElementById('debugPanel');
    if (debugPanel) {
        if (settings.debugMode) {
            debugPanel.classList.add('visible');
        }
        const el = document.getElementById('debugLog');
        if (el && settings.debugMode) {
            const line = document.createElement('div');
            line.textContent = `${new Date().toLocaleTimeString()} ${msg}`;
            el.appendChild(line);
            el.scrollTop = el.scrollHeight;
            while (el.children.length > 20) el.removeChild(el.firstChild);
        }
    }
}

// Expose for ai.js
window.navassist.debugLog = debugLog;
window.navassist.speak = speak;
window.navassist.currentState = () => currentState;
window.navassist.STATES = STATES;
window.navassist.transitionTo = transitionTo;

// ============================================================
// Bootstrap -- splash screen handles the autoplay unlock
// ============================================================

window.addEventListener('load', async () => {
    await loadConfig();

    // Apply saved high contrast setting immediately on load
    const settings = JSON.parse(localStorage.getItem('navassist_settings') || '{}');
    if (settings.highContrast) {
        document.body.classList.add('high-contrast');
    }

    const splashScreen = document.getElementById('splashScreen');
    const mainApp = document.getElementById('mainApp');
    const splashButton = document.getElementById('splashButton');

    splashButton.addEventListener('click', () => {
        // This tap unlocks browser audio -- everything after can play sound
        splashScreen.classList.add('hidden');
        mainApp.classList.remove('hidden');

        connectWebSocket();
        startGpsTracking();
        initPhoneCompass();
        updateUI(STATES.IDLE);

        // Clear announcement of current state
        speak(
            'NavAssist started. Waiting for glasses to connect. ' +
            'Please ensure your glasses are powered on and connected to WiFi.',
            false
        );
    });
});
