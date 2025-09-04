import time
import re
import html
from modules import shared
from extensions.KokoroTTS_4_TGUI.src.debug import *


# These next two function mad it easier to jest pairs without modifying script.py
def create_ai_audio_html(url, speed, text):
    # return make_standard_autoplay(url,speed, text)
    # return create_speaker_button_html(url,speed, text)
    return create_hidden_audio_html(url,speed, text)
    # return create_pure_audioplayr_js(url, speed, text)

def create_ai_audio_js():
    return create_hidden_audio_js()




def create_hidden_audio_html(url, speed, text):
    """Complete self-contained audio component - all logic inline"""
    abbrev = "chah"
    timestamp = int(time.time() * 1000)
    ts_url = f"{url}?t={timestamp}"

    result = f"""
    <div id="kokoro-hidden-audio" style="display: none;">
        <audio id="kokoro-current-audio" preload="auto" autoplay data-speed="{speed}">
            <source src="{ts_url}" type="audio/wav">
        </audio>
        <script>
            console.log("KokoroTTS ({abbrev}): Embedded audio component loaded");

            function setSpeakButtonEnabled(enabled, initialText) {{
                var btnOriginal = document.getElementById("kokoro-audio-control");
                var btnClone = document.getElementById("kokoro-audio-control-clone");

                [btnOriginal, btnClone].forEach(btn => {{
                    if (btn) {{
                        btn.disabled = !enabled;
                        btn.style.opacity = enabled ? "1" : "0.5";
                        if (!enabled) {{
                            btn.textContent = "Speak";
                            //btn.className = "lg secondary svelte-cmf5ev"; // Disabled state
                        }} else {{
                            // When enabling, set correct "Speak" state styling
                            //btn.textContent = "Speak";
                            btn.textContent = initialText;
                            btn.className = initialText === 'Speak' ? "lg primary svelte-cmf5ev" : "lg secondary svelte-cmf5ev";
                            //btn.className = "lg primary svelte-cmf5ev"; // Speak state (orange border)
                        }}
                    }}
                }});
                console.log("KokoroTTS ({abbrev}_setBtn): Button enabled:", enabled);
            }}
            function forceButtonUpdate(text) {{
                var btnClone = document.getElementById('kokoro-audio-control-clone');
                var btnOriginal = document.getElementById('kokoro-audio-control');

                if (btnClone) {{
                    btnClone.textContent = text;
                    if (text === 'Speak') {{
                        btnClone.className = "lg primary svelte-cmf5ev";
                    }} else {{
                        btnClone.className = "lg secondary svelte-cmf5ev";
                    }}
                }}

                if (btnOriginal) {{
                    btnOriginal.textContent = text;
                    if (text === 'Speak') {{
                        btnOriginal.className = "lg primary svelte-cmf5ev";
                    }} else {{
                        btnOriginal.className = "lg secondary svelte-cmf5ev";
                    }}
                }}

                setTimeout(() => {{
                    if (btnClone) btnClone.textContent = text;
                    if (btnOriginal) btnOriginal.textContent = text;
                }}, 100);
                setTimeout(() => {{
                    if (btnClone) btnClone.textContent = text;
                    if (btnOriginal) btnOriginal.textContent = text;
                }}, 500);
            }}

            window.kokoroToggleAudio = function() {{
                var audio = document.getElementById('kokoro-current-audio');

                if (audio) {{
                    window.kokoroCurrentAudio = audio;

                    if (audio.paused || audio.ended) {{
                        if (audio.ended) audio.currentTime = 0;
                        audio.play();
                        forceButtonUpdate('Pause');
                        setSpeakButtonEnabled(true);
                        console.log("KokoroTTS ({abbrev}_toggle): Playing audio");
                    }} else {{
                        audio.pause();
                        forceButtonUpdate('Speak');
                        setSpeakButtonEnabled(true);
                        console.log("KokoroTTS ({abbrev}_toggle): Pausing audio");
                    }}
                }}
            }};

            function initializeAudioComponent() {{
                // Clean up any existing old audio elements first
                var existingAudios = document.querySelectorAll('audio[id="kokoro-current-audio"]');
                if (existingAudios.length > 1) {{
                    // Remove all but the last one (newest)
                    for (var i = 0; i < existingAudios.length - 1; i++) {{
                        existingAudios[i].remove();
                        console.log("KokoroTTS ({abbrev}): Removed old audio element");
                    }}
                }}

                var audio = document.getElementById('kokoro-current-audio');
                if (audio && !audio.kokoroInitialized) {{
                    audio.kokoroInitialized = true;

                    audio.addEventListener('playing', function() {{
                        forceButtonUpdate('Pause');
                        setSpeakButtonEnabled(true);
                        console.log("KokoroTTS ({abbrev}): Audio playing - button set to Pause");
                    }});

                    audio.addEventListener('pause', function() {{
                        forceButtonUpdate('Speak');
                        console.log("KokoroTTS ({abbrev}): Audio paused - button set to Speak");
                    }});

                    audio.addEventListener('ended', function() {{
                        forceButtonUpdate('Speak');
                        console.log("KokoroTTS ({abbrev}): Audio ended - button set to Speak");
                    }});

                    console.log("KokoroTTS ({abbrev}): Audio component fully initialized");

                    // Check if audio is already playing (autoplay detection)
                    setTimeout(() => {{
                        if (audio && !audio.paused && !audio.ended) {{
                            setSpeakButtonEnabled(true, 'Pause');  // Enable with correct text immediately
                            console.log("KokoroTTS ({abbrev}): Detected autoplay, enabled as Pause");
                        }} else {{
                            setSpeakButtonEnabled(true, 'Speak');  // Enable with Speak text
                        }}
                    }}, 500);

                    // Check if audio is already playing (autoplay detection)
                    setTimeout(() => {{
                        if (audio && !audio.paused && !audio.ended) {{
                            forceButtonUpdate('Pause');
                            console.log("KokoroTTS ({abbrev}): Detected autoplay, set button to Pause");
                        }}
                    }}, 500);
                }}
            }}

            initializeAudioComponent();
        </script>
    </div>
    """

    return result


def create_button_cloning_js(abbrev):
    """Reusable button cloning functionality"""
    result = f"""
    function cloneKokoroButtons() {{
        var originalBtn = document.getElementById("kokoro-audio-control");
        var generateBtn = document.getElementById("Generate");

        if (originalBtn && generateBtn && !document.getElementById("kokoro-audio-control-clone")) {{
            var clonedBtn = originalBtn.cloneNode(true);
            clonedBtn.id = "kokoro-audio-control-clone";

            clonedBtn.classList.remove("kokoro-hidden-original");

            generateBtn.insertAdjacentElement("afterend", clonedBtn);
            clonedBtn.style.setProperty("margin-left", "-10px");

            // Force clone to correct initial state regardless of original
            clonedBtn.disabled = true;
            clonedBtn.style.opacity = "0.5";
            clonedBtn.textContent = "Speak";

            [clonedBtn, originalBtn].forEach(btn => {{
                btn.addEventListener("click", function() {{
                    if (window.kokoroToggleAudio) window.kokoroToggleAudio();
                }});
            }});

            console.log("KokoroTTS ({abbrev}): Button cloned");
        }}
    }}
    """
    return result


def create_hidden_audio_js():
    """Generic utility to enable HTML components with embedded scripts"""
    abbrev = "chaj"
    button_cloning_js = create_button_cloning_js(abbrev)
    css_rule = '.kokoro-hidden-original { display: none !important; }'
    result = f"""
    console.log("KokoroTTS ({abbrev}): Script execution utility loaded");

    // Inject CSS
    var style = document.createElement('style');
    style.textContent = '{css_rule}';
    document.head.appendChild(style);


    // Generic script executor for any HTML component
    function executeEmbeddedScripts() {{
        var containers = document.querySelectorAll('[id*="kokoro-"]');
        containers.forEach(function(container) {{
            if (!container.scriptsExecuted) {{
                container.scriptsExecuted = true;
                var scripts = container.getElementsByTagName('script');
                for (var i = 0; i < scripts.length; i++) {{
                    if (scripts[i].innerHTML) {{
                        console.log("KokoroTTS ({abbrev}): Executing embedded script");
                        eval(scripts[i].innerHTML);
                    }}
                }}
            }}
        }});
    }}

    {button_cloning_js}

    // Run utilities
        // Initialize buttons to disabled state on startup
    function initializeButtonState() {{
        var audio = document.getElementById('kokoro-current-audio');
        if (!audio) {{
            var btnOriginal = document.getElementById("kokoro-audio-control");
            var btnClone = document.getElementById("kokoro-audio-control-clone");
            [btnOriginal, btnClone].forEach(btn => {{
                if (btn) {{
                    btn.disabled = true;
                    btn.style.opacity = "0.5";
                    btn.textContent = "Speak";
                }}
            }});
            console.log("KokoroTTS ({abbrev}): Set buttons to disabled startup state");
        }}
    }}

    setInterval(executeEmbeddedScripts, 300);
    setTimeout(initializeButtonState, 100);  // Set initial state after buttons exist
    cloneKokoroButtons();
    """
    return result




def create_pure_audioplayr_js(url, speed, text):
    name = "cpapjs"
    result = f"""
    <div id="kokoro-hidden-audio" style="border: 1px solid white; padding: 10px; margin: 5px;">
        <button id="kokoro-audio-control" style="display: block;" disabled>Speak</button>
        <p id="kokoro-original-text" style="display: block;">{text}</p>
        <script>
            let audioContext;
            let audioBuffer;
            let sourceNode;
            let isPlaying = false;
            const playbackSpeed = {speed};

            console.log('KokoroTTS ({name}_init): Starting audio initialization');

            function loadAudio(url) {{
                console.log('KokoroTTS ({name}_load): Attempting to load audio from:', url);

                // Test with regular HTML5 audio first
                var testAudio = new Audio(url);
                testAudio.oncanplaythrough = function() {{
                    console.log('KokoroTTS ({name}_load): HTML5 Audio can play this file');
                }};
                testAudio.onerror = function(e) {{
                    console.log('KokoroTTS ({name}_load): HTML5 Audio failed:', e);
                }};

                try {{
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    console.log('KokoroTTS ({name}_load): AudioContext created, state:', audioContext.state);
                }} catch(e) {{
                    console.error('KokoroTTS ({name}_load): Failed to create AudioContext:', e);
                    return;
                }}

                fetch(url)
                    .then(response => {{
                        console.log('KokoroTTS ({name}_load): Fetch response:', response.status, response.ok);
                        if (!response.ok) {{
                            throw new Error('Network response was not ok: ' + response.statusText);
                        }}
                        return response.arrayBuffer();
                    }})
                    .then(data => {{
                        console.log('KokoroTTS ({name}_load): ArrayBuffer received, size:', data.byteLength);
                        return audioContext.decodeAudioData(data);
                    }})
                    .then(buffer => {{
                        console.log('KokoroTTS ({name}_load): Audio decoded successfully, duration:', buffer.duration);
                        audioBuffer = buffer;
                        document.getElementById('kokoro-audio-control').disabled = false;
                        console.log('KokoroTTS ({name}_load): Button enabled');
                    }})
                    .catch(error => {{
                        console.error('KokoroTTS ({name}_load): Error in audio loading chain:', error);
                    }});
            }}

            function toggleAudio() {{
                console.log('KokoroTTS ({name}_toggle): Toggle called, audioBuffer exists:', !!audioBuffer);
                if (!audioBuffer) return;
                if (isPlaying) {{
                    pauseAudio();
                }} else {{
                    playAudio();
                }}
            }}

            function playAudio() {{
                console.log('KokoroTTS ({name}_play): Starting playback');
                sourceNode = audioContext.createBufferSource();
                sourceNode.buffer = audioBuffer;
                sourceNode.playbackRate.value = playbackSpeed;
                sourceNode.connect(audioContext.destination);
                sourceNode.start(0);
                isPlaying = true;
                document.getElementById('kokoro-audio-control').textContent = 'Pause';

                sourceNode.onended = function() {{
                    console.log('KokoroTTS ({name}_play): Playback ended');
                    isPlaying = false;
                    document.getElementById('kokoro-audio-control').textContent = 'Speak';
                }};
            }}

            function pauseAudio() {{
                console.log('KokoroTTS ({name}_pause): Pausing audio');
                if (sourceNode) {{
                    sourceNode.stop();
                    isPlaying = false;
                    document.getElementById('kokoro-audio-control').textContent = 'Speak';
                }}
            }}

            document.getElementById('kokoro-audio-control').addEventListener('click', toggleAudio);
            console.log('KokoroTTS ({name}_init): Click listener attached');

            loadAudio('{url}');
        </script>
    </div>
    """
    return result

# This one is destined to work in tandem with create_pure_audioplayr_js
def create_pure_audio_js():
    name = "cpaj"
    result = f"""
    console.log('KokoroTTS ({name}_init): Script loaded');

    // Function to manually execute scripts in dynamically loaded content
    function executeScriptsInElement(element) {{
        var scripts = element.getElementsByTagName('script');
        for (var i = 0; i < scripts.length; i++) {{
            var script = scripts[i];
            if (script.innerHTML) {{
                console.log('KokoroTTS ({name}_exec): Executing inline script');
                try {{
                    eval(script.innerHTML);
                }} catch(e) {{
                    console.error('KokoroTTS ({name}_exec): Script execution failed:', e);
                }}
            }}
        }}
    }}

    // Monitor for new audio divs with scripts
    function checkForNewScripts() {{
        var audioDiv = document.getElementById('kokoro-hidden-audio');
        if (audioDiv && !audioDiv.scriptsExecuted) {{
            audioDiv.scriptsExecuted = true;
            console.log('KokoroTTS ({name}_monitor): Found new audio div, executing scripts');
            executeScriptsInElement(audioDiv);
        }}
    }}

    // Button cloning code
    function tryCloneButton(attempts) {{
        attempts = attempts || 0;
        if (attempts > 10) return;

        var originalBtn = document.getElementById('kokoro-audio-control');
        var generateBtn = document.getElementById('Generate');

        console.log('KokoroTTS ({name}_clone): Attempt', attempts + 1, '- Original:', !!originalBtn, 'Generate:', !!generateBtn);

        if (originalBtn && generateBtn && !document.getElementById('kokoro-audio-control-clone')) {{
            var clonedBtn = originalBtn.cloneNode(true);
            clonedBtn.id = 'kokoro-audio-control-clone';
            generateBtn.insertAdjacentElement('afterend', clonedBtn);
            clonedBtn.style.setProperty('margin-left', '-10px');

            clonedBtn.addEventListener('click', function() {{
                console.log('KokoroTTS ({name}_clone): Clone clicked');
                originalBtn.click();
            }});

            console.log('KokoroTTS ({name}_clone): Button cloned successfully');
            startButtonSync();
        }} else if (!document.getElementById('kokoro-audio-control-clone')) {{
            setTimeout(function() {{ tryCloneButton(attempts + 1); }}, 200);
        }}
    }}

    function startButtonSync() {{
        function syncButtons() {{
            var original = document.getElementById('kokoro-audio-control');
            var clone = document.getElementById('kokoro-audio-control-clone');

            if (original && clone) {{
                if (clone.textContent !== original.textContent) {{
                    clone.textContent = original.textContent;
                    console.log('KokoroTTS ({name}_sync): Synced button text to:', original.textContent);
                }}
                clone.disabled = original.disabled;
                clone.style.opacity = original.disabled ? '0.5' : '1';
            }}
        }}

        setInterval(syncButtons, 100);
    }}

    // Start monitoring and cloning
    setInterval(checkForNewScripts, 300);
    tryCloneButton();
    """
    return result


def _create_hidden_audio_html(url, speed, text):
    """
    Create audio that plays in hidden div, with control via external button.
    Uses single static ID for simplicity.
    """
    timestamp=int(time.time() * 1000)
    ts_url=f"{url}?t={timestamp}"
#
#         <div id="kokoro-hidden-audio" class="block svelte-12cmxck padded hide-container" style="border-style: solid; overflow: visible; min-width: min(0px, 100%); border-width: var(--block-border-width);">
    audio_html = f"""
        <div id="kokoro-hidden-audio" class="block svelte-12cmxck padded" style="border-style: solid; overflow: visible; min-width: min(0px, 100%); border-width: var(--block-border-width);">
            <div class="wrap center full svelte-z7cif2 hide" style="position: absolute; padding: 0px;"></div>
            <div class="svelte-1ed2p3z">
                <div class="prose svelte-1ybaih5">
                    <audio id="kokoro-current-audio" style="display: none;" preload="auto" autoplay>
                        <source src="{ts_url}" type="audio/wav">
                    </audio>
                    <div>
                        <span id="audio-url" style="display: block; margin-top: 10px;">Audio URL: {ts_url}</span>
                    </div>
                    <script>
                        (function() {{
                            console.log('KokoroTTS (chah): Setting up audio player');
                            var audio = document.getElementById('kokoro-current-audio');
                            var controlBtnClone = document.getElementById('kokoro-audio-control-clone');
                            var controlBtnOriginal = document.getElementById('kokoro-audio-control');
                            console.log("KokoroTTS (chah): Audio element found:", audio);
                            console.log("KokoroTTS (chah): Control button clone found:", controlBtnClone);
                            console.log("KokoroTTS (chah): Control button original found:", controlBtnOriginal);
                            if (audio && (controlBtnClone || controlBtnOriginal)) {{
                                audio.playbackRate = {speed};
                                audio.volume = 0.9;
                                window.kokoroCurrentAudio = audio;  // This line is crucial
                                console.log('KokoroTTS (chah): Set audio reference');

                                function showPauseText() {{
                                    if (controlBtnClone) controlBtnClone.textContent = 'Pause';
                                    if (controlBtnOriginal) controlBtnOriginal.textContent = 'Pause';
                                }}

                                function showSpeakText() {{
                                    if (controlBtnClone) controlBtnClone.textContent = 'Speak';
                                    if (controlBtnOriginal) controlBtnOriginal.textContent = 'Speak';
                                }}

                                audio.onplay = showPauseText;
                                audio.onplaying = showPauseText;
                                audio.onpause = showSpeakText;
                                audio.onended = showSpeakText;

                                audio.play().then(() => {{
                                    showPauseText();
                                }}).catch(e => {{
                                    console.log('KokoroTTS (chah): Audio play failed:', e);
                                    showSpeakText();
                                }});
                            }} else {{
                                console.log("KokoroTTS (chah): Audio or control button not found, cannot set up audio player.");
                            }}
                        }})();
                    </script>
                </div>
            </div>
        </div>
    """

    return audio_html



def _create_hidden_audio_js():
    """
    Returns custom javascript as a string. It is applied whenever the web UI is loaded.
    This manages audio playback toggle, button cloning, synced state, tooltips, etc.
    """
    return """
    console.log("KokoroTTS (cjs) script loaded");

    // Global function to toggle audio play/pause and update both buttons' text
    window.kokoroToggleAudio = function() {
        var audio = window.kokoroCurrentAudio;
        var controlBtnClone = document.getElementById('kokoro-audio-control-clone');
        var controlBtnOriginal = document.getElementById('kokoro-audio-control');

        if (!audio) {
            console.log("KokoroTTS (cjs): Audio element not found; trying document.getElementById('kokoro-current-audio')");
            audio = document.getElementById('kokoro-current-audio');
            if (audio) {
                window.kokoroCurrentAudio = audio;
                console.log("KokoroTTS (cjs): Audio element found and cached.");
            } else {
                console.log("KokoroTTS (cjs): Audio element not found; cannot toggle");
                return; // no audio to toggle
            }
        }

        // Play or pause audio and update button text accordingly
        if (audio.paused) {
            audio.play()
                .then(() => {
                    if (controlBtnClone) controlBtnClone.textContent = 'Pause';
                    if (controlBtnOriginal) controlBtnOriginal.textContent = 'Pause';
                    console.log("KokoroTTS (cjs): Audio playing; button text set to 'Pause'");
                })
                .catch(err => {
                    if (controlBtnClone) controlBtnClone.textContent = 'Speak';
                    if (controlBtnOriginal) controlBtnOriginal.textContent = 'Speak';
                    console.error("KokoroTTS (cjs): Audio play failed:", err);
                });
        // Manual fallback: set immediately to Pause in case onplay missed firing
        if (controlBtnClone) controlBtnClone.textContent = 'Pause';
        if (controlBtnOriginal) controlBtnOriginal.textContent = 'Pause';

        } else {
            audio.pause();
            if (controlBtnClone) controlBtnClone.textContent = 'Speak';
            if (controlBtnOriginal) controlBtnOriginal.textContent = 'Speak';
            console.log("KokoroTTS (cjs): KokoroTTS (cjs): Audio paused; button text set to 'Speak'");
        }

        // Listen for audio play/pause/end events to update buttons text automatically
        audio.onplay = function() {
            if (controlBtnClone) controlBtnClone.textContent = 'Pause';
            if (controlBtnOriginal) controlBtnOriginal.textContent = 'Pause';
        };
        audio.onpause = function() {
            if (controlBtnClone) controlBtnClone.textContent = 'Speak';
            if (controlBtnOriginal) controlBtnOriginal.textContent = 'Speak';
        };
        audio.onended = function() {
            if (controlBtnClone) controlBtnClone.textContent = 'Speak';
            if (controlBtnOriginal) controlBtnOriginal.textContent = 'Speak';
        };
    };

    // Enable/disable buttons and update their tooltips and opacity
    function setSpeakButtonEnabled(enabled) {
        var btnOriginal = document.getElementById("kokoro-audio-control");
        var btnClone = document.getElementById("kokoro-audio-control-clone");
        var tooltipText = enabled ? "Click to replay last audio response" : "No audio output yet";

        if (btnOriginal) {
            btnOriginal.disabled = !enabled;
            btnOriginal.style.opacity = enabled ? "1" : "0.5";
            btnOriginal.title = tooltipText;
        }
        if (btnClone) {
            btnClone.disabled = !enabled;
            btnClone.style.opacity = enabled ? "1" : "0.5";
            btnClone.title = tooltipText;
        }
    }

    // Initialize buttons as disabled on page load
    setSpeakButtonEnabled(false);
    window.setSpeakButtonEnabled = setSpeakButtonEnabled;

    // Update tooltips based on button enabled state
    function updateSpeakButtonTooltip() {
        var btnOriginal = document.getElementById("kokoro-audio-control");
        var btnClone = document.getElementById("kokoro-audio-control-clone");
        if (btnOriginal) {
            btnOriginal.title = btnOriginal.disabled ? "No audio output yet" : "Click to replay last audio response";
        }
        if (btnClone) {
            btnClone.title = btnClone.disabled ? "No audio output yet" : "Click to replay last audio response";
        }
    }
    new MutationObserver(updateSpeakButtonTooltip).observe(
        document.getElementById('kokoro-audio-control'),
        { attributes: true, attributeFilter: ['disabled'] }
    );
    updateSpeakButtonTooltip();

    // Sync cloned button state with original (disabled, title, opacity)
    function syncClonedButtonState() {
        var originalBtn = document.getElementById("kokoro-audio-control");
        var clonedBtn = document.getElementById("kokoro-audio-control-clone");
        if (!originalBtn || !clonedBtn) return;
        clonedBtn.disabled = originalBtn.disabled;
        clonedBtn.title = originalBtn.title;
        clonedBtn.style.opacity = originalBtn.disabled ? "0.5" : "1";
    }
    syncClonedButtonState();
    new MutationObserver(syncClonedButtonState).observe(
        document.getElementById("kokoro-audio-control"),
        { attributes: true, attributeFilter: ["disabled", "title"] }
    );

    // Clone the Speak button with retry mechanism and add click event to cloned button
    function tryCloneButton(attempts) {
        attempts = attempts || 0;
        if (attempts > 10) {
            console.log("KokoroTTS (cjs): Gave up cloning after 10 attempts");
            return;
        }
        var originalBtn = document.getElementById("kokoro-audio-control");
        var generateBtn = document.getElementById("Generate");
        console.log("KokoroTTS (cjs): Attempt", attempts + 1, "- Original button:", !!originalBtn, "Generate button:", !!generateBtn);
        if (originalBtn && generateBtn && !document.getElementById("kokoro-audio-control-clone")) {
            var clonedBtn = originalBtn.cloneNode(true);
            clonedBtn.id = "kokoro-audio-control-clone";
            generateBtn.insertAdjacentElement("afterend", clonedBtn);
            clonedBtn.style.setProperty("margin-left", "-10px");
            clonedBtn.style.display = "inline-block";
            clonedBtn.style.visibility = "visible";
            clonedBtn.addEventListener("click", function() {
                console.log("KokoroTTS (cjs): Cloned button clicked");
                window.kokoroToggleAudio();
            });
            console.log("KokoroTTS (cjs): Button cloned successfully");
            syncClonedButtonState();
            new MutationObserver(syncClonedButtonState).observe(
                originalBtn,
                { attributes: true, attributeFilter: ["disabled", "title"] }
            );
        } else if (!document.getElementById("kokoro-audio-control-clone")) {
            setTimeout(function() { tryCloneButton(attempts + 1); }, 200);
        }
    }
    tryCloneButton();

    // Add click listener to original button to toggle audio as well
    var originalBtn = document.getElementById('kokoro-audio-control');
    if (originalBtn) {
        originalBtn.addEventListener('click', function() {
            console.log("KokoroTTS (cjs): Original button clicked");
            window.kokoroToggleAudio();
        });
    }
    // Poll for Speak button becoming enabled, then trigger toggle once and stop polling
    (function waitForEnabledButton() {
        var btnClone = document.getElementById("kokoro-audio-control-clone");
        var btnOriginal = document.getElementById("kokoro-audio-control");
        var btn = btnClone || btnOriginal;
    if (!btn.disabled) {
        console.log("KokoroTTS (cjs): Speak button enabled, updating button text to 'Pause'");
        if (btnClone) {
            btnClone.textContent = 'Pause';
            btnClone.style.opacity = "1";
            btnClone.title = "Click to pause current audio response";
        }
        if (btnOriginal) {
            btnOriginal.textContent = 'Pause';
            btnOriginal.style.opacity = "1";
            btnOriginal.title = "Click to replay last audio response";
        }
        // Stop polling: don't call setTimeout again
        } else {
            // Button still disabled, check again soon
            setTimeout(waitForEnabledButton, 200);
        }
    })();

    """


def old_create_hidden_audio_html(url, speed, text):
    """
    Create audio that plays in hidden div, with control via external button.
    Uses single static ID for simplicity.
    """
    audio_html = f"""
    <audio id="kokoro-current-audio" style="display: none;" preload="auto" autoplay>
        <source src="{url}?t={int(time.time() * 1000)}" type="audio/wav">
    </audio>
    <script>
        (function() {{
            console.log('KokoroTTS (chah): Setting up audio player');
            var audio = document.getElementById('kokoro-current-audio');
            var controlBtn = document.getElementById('kokoro-audio-control-clone') || document.getElementById('kokoro-audio-control');

            if (audio && controlBtn) {{
                audio.playbackRate = {speed};
                audio.volume = 0.9;

                // Always set this as the current audio
                window.kokoroCurrentAudio = audio;
                console.log('KokoroTTS (chah): Set audio reference');

                audio.onplay = function() {{
                    controlBtn.textContent = 'Pause';
                }};

                audio.onended = function() {{
                    controlBtn.textContent = 'Speak';
                }};

                audio.onpause = function() {{
                    controlBtn.textContent = 'Speak';
                }};

                audio.play().catch(function(e) {{
                    controlBtn.textContent = 'Speak';
                }});
            }}
        }})();
    </script>
    """
    return audio_html


def create_speaker_button_html(audio_url, speed, text):
    """
    Create auto-playing audio with play/pause toggle button.
    Icon style is read from shared.args.kokoro_icon_style
    """
    import time
    from modules import shared

    timestamp = int(time.time() * 1000)
    icon_style = getattr(shared.args, 'kokoro_icon_style', 'emoji')

    # Define icons based on style
    if icon_style == "png":
        play_icon = '<img src="/file/extensions/KokoroTTS_4_TGUI/images/play.png" width="50" height="50" style="vertical-align: middle;">'
        pause_icon = '<img src="/file/extensions/KokoroTTS_4_TGUI/images/pause.png" width="50" height="50" style="vertical-align: middle;">'
    else:  # emoji default
        play_icon = "🔊"
        pause_icon = "⏸️"

    # Escape for inline JavaScript
    play_icon_escaped = play_icon.replace('"', '&quot;').replace("'", "&#39;")
    pause_icon_escaped = pause_icon.replace('"', '&quot;').replace("'", "&#39;")

    return f"""
    <audio id="kokoro-preview-{timestamp}" style="display: none;" preload="auto" autoplay>
        <source src="{audio_url}?t={timestamp}" type="audio/wav">
    </audio>
    <button id="kokoro-btn-{timestamp}"
            onmousedown="var a=document.getElementById('kokoro-preview-{timestamp}'); if(a.paused){{a.play();this.innerHTML='{pause_icon_escaped}';this.title='Pause voice preview';}}else{{a.pause();this.innerHTML='{play_icon_escaped}';this.title='Play voice preview';}}"
            style="background: none; border: none; cursor: pointer; padding: 2px; margin-left: 5px; vertical-align: middle;"
            title="Pause voice preview">
        {pause_icon}
    </button>
    """

def make_standard_autoplay(url, speed, text):
    timestamp = int(time.time() * 1000)
    log(f"make_test_autoplay called: url={url}, speed={speed}")

    #<audio id="kokoro-preview-{timestamp}" style="display: none;" preload="auto"; autoplay>
    audio_html = f"""
    <audio id="kokoro-preview-{timestamp}" preload="auto"; autoplay>
        <source src="{url}" type="audio/wav">
    </audio>
    <script>
        (function() {{
            console.log('KokoroTTS (chah): Auto-playing TTS audio');
            var audio = document.getElementById('kokoro-preview-' + timestamp);
            if (audio) {{
                audio.playbackRate = {speed};
                audio.volume = 0.8; // Adjust volume as needed

                // Try to play immediately
                audio.play().then(function() {{
                    console.log('KokoroTTS (chah): Auto-play successful');
                }}).catch(function(e) {{
                    console.log('KokoroTTS (chah): Auto-play blocked, waiting for user interaction');
                    // If autoplay fails, play on next user interaction
                    document.addEventListener('click', function playOnClick() {{
                        audio.play().then(function() {{
                            console.log('KokoroTTS (chah): Audio played after user interaction');
                        }}).catch(function(e2) {{
                            console.error('KokoroTTS (chah): Failed to play after interaction:', e2);
                        }});
                        document.removeEventListener('click', playOnClick);
                    }}, {{ once: true }});
                }});


                audio.onended = function() {{ audio.remove(); }};
                audio.onerror = function() {{ audio.remove(); }};
                audio.load();
            }}
        }})();
    </script>
    """

    log(f"Generated auto-TTS HTML for URL: {url}")
    log(f"Returning HTML length: {len(audio_html)}")
    return audio_html

def make_html_autoplay(url,speed,text):
    log(f"make_html_autoplay called: url={url}, speed={speed}")
    timestamp = int(time.time() * 1000)  # Prevent caching issues
    audio_html = f"""
    <audio id="kokoro-single-{timestamp}" style="display: none;" preload="auto">
        <source src="file/{url}?v={timestamp}" type="audio/wav">
    </audio>
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            var audio = document.getElementById('kokoro-single-{timestamp}');
            if (audio) {{
                audio.playbackRate = {speed};
                audio.play().catch(function(e) {{
                    console.log('KokoroTTS (chah): Autoplay failed, will play on user interaction');
                    document.addEventListener('click', function() {{
                        audio.play();
                    }}, {{ once: true }});
                }});
            }}
        }})();
    </script>
    """
    # audio_html = makehtml.make_test_autoplay(file_url, speed, clean_text[:50] + "..." if len(clean_text) > 50 else clean_text)

    result = original_string + audio_html
    log(f"output_standard returning length: {len(result)}")
    log(f"Contains 'DEBUG': {'DEBUG' in result}")

    return result
    # return auto_audio_html


def make_js_autoplay(url_or_urls, speed, text):
    """
    Create HTML with JavaScript autoplay for single audio file or chunked audio files.

    Args:
        url_or_urls: Single URL string or list of URLs for chunked playback
        speed: Playback speed
    """
    log(f"make_js_autoplay called: url={url}, speed={speed}")
    timestamp = int(time.time() * 1000)  # Prevent caching issues

    # Handle single URL (existing behavior)
    if isinstance(url_or_urls, str):
        url = url_or_urls
        auto_audio_html = f"""
        <audio id="kokoro-auto-{timestamp}" style="display: none;" preload="auto">
            <source src="{url}?t={timestamp}" type="audio/wav">
        </audio>
        <script>
            (function() {{
                console.log('KokoroTTS (chah): Setting up auto-play for chat response');
                var audio = document.getElementById('kokoro-auto-{timestamp}');
                if (audio) {{
                    audio.playbackRate = {speed};
                    audio.volume = 0.9;

                    // Event handlers
                    audio.onloadeddata = function() {{
                        console.log('KokoroTTS (chah): Chat audio loaded, attempting to play');
                        audio.play().then(function() {{
                            console.log('KokoroTTS (chah): Chat audio playing successfully');
                        }}).catch(function(e) {{
                            console.log('KokoroTTS (chah): Chat auto-play blocked:', e.message);
                            // Try again on next user interaction
                            var playOnInteraction = function() {{
                                console.log('KokoroTTS (chah): Playing chat audio after user interaction');
                                audio.play().catch(function(e2) {{
                                    console.error('KokoroTTS (chah): Failed to play after interaction:', e2);
                                }});
                                document.removeEventListener('click', playOnInteraction);
                                document.removeEventListener('keydown', playOnInteraction);
                            }};
                            document.addEventListener('click', playOnInteraction, {{ once: true }});
                            document.addEventListener('keydown', playOnInteraction, {{ once: true }});
                        }});
                    }};

                    audio.onended = function() {{
                        console.log('KokoroTTS (chah): Chat audio finished playing');
                        // Clean up by removing the audio element
                        setTimeout(function() {{ audio.remove(); }}, 100);
                    }};

                    audio.onerror = function() {{
                        console.error('KokoroTTS (chah): Chat audio failed to load:', audio.src);
                        audio.remove();
                    }};

                    // Force load the audio
                    audio.load();
                }} else {{
                    console.error('KokoroTTS (chah): Could not find audio element');
                }}
            }})();
        </script>
        """
        return auto_audio_html

    # Handle multiple URLs (chunked playback)
    elif isinstance(url_or_urls, list):
        audio_urls = url_or_urls

        # Create audio elements for each chunk
        audio_elements = ""
        for i, url in enumerate(audio_urls):
            audio_elements += f'<audio id="kokoro-chunk-{timestamp}-{i}" style="display: none;" preload="auto"><source src="{url}?v={timestamp}" type="audio/wav"></audio>'

        js_code = f"""
        <script>
            (function() {{
                console.log('KokoroTTS (chah): Starting chunked playback with {len(audio_urls)} chunks');
                var chunkCount = {len(audio_urls)};
                var currentChunk = 0;
                var timestamp = {timestamp};
                var speed = {speed};

                function playNext() {{
                    if (currentChunk >= chunkCount) {{
                        console.log('KokoroTTS (chah): All chunks played');
                        return;
                    }}

                    var audio = document.getElementById('kokoro-chunk-' + timestamp + '-' + currentChunk);
                    if (audio) {{
                        console.log('KokoroTTS (chah): Playing chunk ' + currentChunk);
                        audio.playbackRate = speed;
                        audio.volume = 0.9;

                        audio.onended = function() {{
                            console.log('KokoroTTS (chah): Chunk ' + currentChunk + ' finished');
                            currentChunk++;
                            setTimeout(playNext, 200);
                        }};

                        audio.onerror = function() {{
                            console.error('KokoroTTS (chah): Error playing chunk ' + currentChunk);
                            currentChunk++;
                            setTimeout(playNext, 100);
                        }};

                        audio.play().then(function() {{
                            console.log('KokoroTTS (chah): Chunk ' + currentChunk + ' playing successfully');
                        }}).catch(function(e) {{
                            console.log('KokoroTTS (chah): Chunk ' + currentChunk + ' autoplay blocked:', e.message);
                            // Try again on next user interaction for this chunk
                            var playOnInteraction = function() {{
                                console.log('KokoroTTS (chah): Playing chunk after user interaction');
                                audio.play().catch(function(e2) {{
                                    console.error('KokoroTTS (chah): Failed to play chunk after interaction:', e2);
                                    currentChunk++;
                                    setTimeout(playNext, 100);
                                }});
                                document.removeEventListener('click', playOnInteraction);
                                document.removeEventListener('keydown', playOnInteraction);
                            }};
                            document.addEventListener('click', playOnInteraction, {{ once: true }});
                            document.addEventListener('keydown', playOnInteraction, {{ once: true }});
                        }});
                    }} else {{
                        console.error('KokoroTTS (chah): Could not find audio element for chunk ' + currentChunk);
                        currentChunk++;
                        setTimeout(playNext, 100);
                    }}
                }}

                // Start playing after a short delay
                setTimeout(playNext, 500);
            }})();
        </script>
        """

        return audio_elements + js_code

    else:
        log('KokoroTTS (chah): Invalid URL parameter - must be string or list')
        return ""
