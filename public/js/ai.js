/**
 * NavAssist — ai.js
 * OWNER: [Teammate's name]
 *
 * Responsibilities:
 * 1. Receive camera frames from the ESP32 (via app.js)
 * 2. Call Roboflow API for traffic light detection
 * 3. Process bounding boxes to determine direction (LEFT/CENTRE/RIGHT)
 * 4. Detect when user has arrived at the crossing (proximity)
 * 5. Detect green/red light state changes
 * 6. Call window.navassist callbacks to trigger haptic/audio/state changes
 *
 * Interface with app.js (DO NOT MODIFY these function names):
 *
 * Called BY app.js:
 *   handleCameraFrame(payload)  -- called whenever a new frame arrives from ESP32
 *
 * Call FROM this file into app.js:
 *   window.navassist.onDirectionDecided(direction)  -- 'LEFT', 'RIGHT', or 'CENTRE'
 *   window.navassist.onArrived()                    -- user is close enough to crossing
 *   window.navassist.onGreenDetected()              -- light changed to green
 *   window.navassist.debugLog(message)              -- add to debug log
 *   window.navassist.speak(text)                    -- speak text aloud
 *   window.navassist.currentState()                 -- get current FSM state string
 *   window.navassist.STATES                         -- state name constants
 *
 * Reading sensor data from app.js:
 *   window.navassist.getCurrentHeading()            -- current IMU heading in degrees
 *   window.navassist.getCurrentLocation()           -- { latitude, longitude, accuracyMeters }
 */

// ============================================================
// Roboflow API config -- loaded from server via app.js config
// ============================================================

// NOTE: config is loaded by app.js and stored in the global `config` variable.
// You can access it here since both files run in the same browser context.
// Wait until window.load fires before using it (same as app.js does).

const ROBOFLOW_BASE_URL = 'https://serverless.roboflow.com';
const CONFIDENCE_THRESHOLD = 0.5;

// ============================================================
// Frame rate throttling
// Only run inference every N frames -- the ESP32 may send faster
// than Roboflow can process, and each call costs API quota
// ============================================================

let frameCount = 0;
const INFERENCE_EVERY_N_FRAMES = 3; // run AI every 3rd frame

// ============================================================
// Main entry point -- called by app.js for every camera frame
// ============================================================

async function handleCameraFrame(payload) {
    // Only run inference while actively navigating
    const state = window.navassist.currentState();
    const activeStates = [
        window.navassist.STATES.NAVIGATING,
        window.navassist.STATES.WAITING,
        window.navassist.STATES.SCANNING,
        window.navassist.STATES.RESOLVING
    ];

    if (!activeStates.includes(state)) return;

    // Throttle -- skip frames to stay within API limits
    frameCount++;
    if (frameCount % INFERENCE_EVERY_N_FRAMES !== 0) return;

    // Get base64 image data from payload
    const imageData = payload.data;
    if (!imageData) {
        window.navassist.debugLog('Camera frame missing image data');
        return;
    }

    try {
        const detection = await callRoboflowApi(imageData);
        if (detection) {
            processDetection(detection, payload.width || 320, payload.height || 240);
        }
    } catch (e) {
        window.navassist.debugLog('AI inference error: ' + e.message);
    }
}

// ============================================================
// Roboflow API call
// ============================================================

async function callRoboflowApi(base64ImageData) {
    // TODO: fill in from config loaded by app.js
    const apiKey = config?.roboflowApiKey;
    const modelId = config?.roboflowModelId || 'ono-gedd7/pedestrian-traffic-light-puf4a';
    const modelVersion = config?.roboflowModelVersion || '3';

    if (!apiKey) {
        window.navassist.debugLog('Roboflow API key not loaded yet');
        return null;
    }

    const url = `${ROBOFLOW_BASE_URL}/${modelId}/${modelVersion}?api_key=${apiKey}`;

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: base64ImageData
    });

    const data = await response.json();
    return data;
}

// ============================================================
// Detection processing -- YOUR TEAMMATE'S MAIN LOGIC GOES HERE
// ============================================================

function processDetection(data, imageWidth, imageHeight) {
    const predictions = data.predictions || [];

    if (predictions.length === 0) {
        window.navassist.debugLog('No traffic light detected in frame');
        return;
    }

    // Filter by confidence threshold
    const confident = predictions.filter(p => p.confidence >= CONFIDENCE_THRESHOLD);
    if (confident.length === 0) return;

    // Pick the highest confidence prediction
    const best = confident.reduce((a, b) => a.confidence > b.confidence ? a : b);

    window.navassist.debugLog(
        `Detected: ${best.class} confidence=${(best.confidence * 100).toFixed(0)}% ` +
        `x=${best.x.toFixed(0)} y=${best.y.toFixed(0)}`
    );

    // Normalize coordinates to 0-1 range
    const centerX = best.x / imageWidth;
    const boxHeight = best.height / imageHeight;

    // --- DIRECTION LOGIC ---
    // Determine if the traffic light post is left, centre, or right of user
    const direction = getDirection(centerX);

    // --- PROXIMITY LOGIC ---
    // Larger bounding box height = closer to the light post
    const isVeryClose = boxHeight > 0.4;

    // --- LIGHT STATE ---
    const lightState = getLightState(best.class);

    // --- CALL BACK INTO app.js ---
    const state = window.navassist.currentState();

    if (state === window.navassist.STATES.NAVIGATING) {
        if (isVeryClose) {
            window.navassist.onArrived();
        } else {
            window.navassist.onDirectionDecided(direction);
        }
    }

    if (state === window.navassist.STATES.WAITING && lightState === 'GREEN') {
        window.navassist.onGreenDetected();
    }
}

// ============================================================
// Helper functions
// ============================================================

function getDirection(normalizedCenterX) {
    if (normalizedCenterX < 0.4) return 'LEFT';
    if (normalizedCenterX > 0.6) return 'RIGHT';
    return 'CENTRE';
}

function getLightState(className) {
    const lower = className.toLowerCase();
    if (lower.includes('green')) return 'GREEN';
    if (lower.includes('red'))   return 'RED';
    return 'UNKNOWN';
}
