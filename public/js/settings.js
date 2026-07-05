/**
 * NavAssist — settings.js
 * Handles all settings page interactions.
 * Saves to localStorage so settings persist between sessions.
 */

// ============================================================
// Load saved settings on page open
// ============================================================

const defaults = {
    hapticIntensity: 70,
    audioVolume: 85,
    speechRate: 105,
    highContrast: false,
    debugMode: false
};

function loadSettings() {
    try {
        const saved = JSON.parse(localStorage.getItem('navassist_settings') || '{}');
        return { ...defaults, ...saved };
    } catch (e) {
        return defaults;
    }
}

function saveSettings(settings) {
    localStorage.setItem('navassist_settings', JSON.stringify(settings));
}

// ============================================================
// DOM elements
// ============================================================

const hapticSlider   = document.getElementById('hapticIntensity');
const audioSlider    = document.getElementById('audioVolume');
const speechSlider   = document.getElementById('speechRate');
const contrastBtn    = document.getElementById('highContrast');
const debugBtn       = document.getElementById('debugMode');
const hapticValueEl  = document.getElementById('hapticValue');
const audioValueEl   = document.getElementById('audioValue');
const speechValueEl  = document.getElementById('speechValue');
const serverInfoEl   = document.getElementById('serverInfo');

// ============================================================
// Apply settings to UI
// ============================================================

function applySettings(settings) {
    hapticSlider.value = settings.hapticIntensity;
    audioSlider.value  = settings.audioVolume;
    speechSlider.value = settings.speechRate;
    setToggle(contrastBtn, settings.highContrast);
    setToggle(debugBtn, settings.debugMode);
    updateLabels(settings);

    // Apply high contrast to this page too
    document.body.classList.toggle('high-contrast', settings.highContrast);
}

function updateLabels(settings) {
    hapticValueEl.textContent = settings.hapticIntensity + '%';
    audioValueEl.textContent  = settings.audioVolume + '%';
    speechValueEl.textContent = speechRateLabel(settings.speechRate);
}

function speechRateLabel(rate) {
    if (rate < 80)  return 'Slow';
    if (rate < 130) return 'Normal';
    if (rate < 170) return 'Fast';
    return 'Very fast';
}

function setToggle(btn, value) {
    btn.setAttribute('aria-checked', String(value));
    const label = btn.getAttribute('aria-label') || '';
    btn.setAttribute('aria-label', label.replace(/on$|off$/, value ? 'on' : 'off'));
}

// ============================================================
// Slider interactions
// ============================================================

hapticSlider.addEventListener('input', () => {
    const settings = loadSettings();
    settings.hapticIntensity = parseInt(hapticSlider.value);
    updateLabels(settings);
    saveSettings(settings);
    speak(`Haptic intensity ${settings.hapticIntensity} percent`);
    // Brief test vibration so user feels the new intensity
    if (navigator.vibrate) navigator.vibrate(200);
});

audioSlider.addEventListener('input', () => {
    const settings = loadSettings();
    settings.audioVolume = parseInt(audioSlider.value);
    updateLabels(settings);
    saveSettings(settings);
    speak(`Audio volume ${settings.audioVolume} percent`);
});

speechSlider.addEventListener('input', () => {
    const settings = loadSettings();
    settings.speechRate = parseInt(speechSlider.value);
    updateLabels(settings);
    saveSettings(settings);
    speak(`Speech rate ${speechRateLabel(settings.speechRate)}`);
});

// ============================================================
// Toggle interactions
// ============================================================

contrastBtn.addEventListener('click', () => {
    const settings = loadSettings();
    settings.highContrast = !settings.highContrast;
    setToggle(contrastBtn, settings.highContrast);
    document.body.classList.toggle('high-contrast', settings.highContrast);
    saveSettings(settings);
    speak(`High contrast ${settings.highContrast ? 'on' : 'off'}`);
});

debugBtn.addEventListener('click', () => {
    const settings = loadSettings();
    settings.debugMode = !settings.debugMode;
    setToggle(debugBtn, settings.debugMode);
    saveSettings(settings);
    speak(`Debug mode ${settings.debugMode ? 'on' : 'off'}`);
});

// ============================================================
// Text to speech -- simple version for settings page
// ============================================================

function speak(text) {
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();

    const sayIt = () => {
        const u = new SpeechSynthesisUtterance(text);
        const s = loadSettings();
        u.rate   = (s.speechRate  || 105) / 100;
        u.volume = (s.audioVolume || 85)  / 100;
        window.speechSynthesis.speak(u);
    };

    // Desktop Chrome needs voices to load first
    const voices = window.speechSynthesis.getVoices();
    if (voices.length > 0) {
        sayIt();
    } else {
        window.speechSynthesis.addEventListener('voiceschanged', sayIt, { once: true });
    }
}

// ============================================================
// Audio unlock -- required for desktop browsers
// ============================================================

let audioUnlocked = false;

function unlockAudio() {
    if (audioUnlocked) return;
    if (window.speechSynthesis) {
        const u = new SpeechSynthesisUtterance('');
        u.volume = 0;
        window.speechSynthesis.speak(u);
        audioUnlocked = true;
    }
}

document.addEventListener('click',      unlockAudio, { once: true });
document.addEventListener('touchstart', unlockAudio, { once: true });

// ============================================================
// Bootstrap
// ============================================================

const settings = loadSettings();
applySettings(settings);
loadServerInfo();

async function loadServerInfo() {
    try {
        const res = await fetch('/api/config');
        const cfg = await res.json();
        serverInfoEl.textContent =
            `Connected to: ${location.host}\n` +
            `Model: ${cfg.roboflowModelId || 'not configured'}`;
    } catch (e) {
        serverInfoEl.textContent = `Server: ${location.host}`;
    }
}

// ============================================================
// Back button -- returns to correct page based on where user came from
// ============================================================

const backButton = document.getElementById('backButton');
if (backButton) {
    backButton.addEventListener('click', (e) => {
        e.preventDefault();
        speak('Going back.', true);
        // Always go back to index -- the page itself handles
        // whether to show splash or main app based on referrer
        setTimeout(() => {
            window.location.href = 'index.html';
        }, 400); // slight delay so "going back" speech starts
    });
}

// ============================================================
// Announce settings page on load
// ============================================================

setTimeout(() => {
    speak('Settings. Adjust haptic intensity, audio volume, speech rate, and display options. Go back when done.', false);
}, 300);
