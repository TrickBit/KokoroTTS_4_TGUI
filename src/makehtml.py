import time
import re
import html
from modules import shared
from extensions.KokoroTTS_4_TGUI.src.debug import *



def get_console_log_prefix():
    """Get the appropriate console logging prefix based on debug setting"""
    debug_setting = getattr(shared.args, 'kokoro_enable_debug', 'errors')

    if debug_setting == 'off' or debug_setting is False:
        return "//", "//"  # Both info and error disabled
    elif debug_setting == 'errors' or debug_setting is True:
        return "//", "console.error"  # Only errors enabled
    elif debug_setting == 'all':
        return "console.log", "console.error"  # Both enabled
    else:
        return "//", "console.error"  # Default to errors only



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
    <audio id="kokoro-preview-{timestamp}" style="display: none;" preload="auto" autoplay onloadeddata="this.playbackRate={speed};">
        <source src="{audio_url}?t={timestamp}" type="audio/wav">
    </audio>
    <button id="kokoro-btn-{timestamp}"
            onmousedown="var a=document.getElementById('kokoro-preview-{timestamp}'); if(a.paused){{a.play();this.innerHTML='{pause_icon_escaped}';this.title='Pause voice preview';}}else{{a.pause();this.innerHTML='{play_icon_escaped}';this.title='Play voice preview';}}"
            style="background: none; border: none; cursor: pointer; padding: 2px; margin-left: 5px; vertical-align: middle;"
            title="Pause voice preview">
        {pause_icon}
    </button>
    """

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

    # Get logging prefixes for info and error messages
    info_log, error_log = get_console_log_prefix()

    result = f"""
    <div id="kokoro-hidden-audio" style="display: none;">
        <audio id="kokoro-current-audio" preload="auto" autoplay onloadeddata="this.playbackRate={speed};">
            <source src="{ts_url}" type="audio/wav">
        </audio>
        <script>
            {info_log}("KokoroTTS ({abbrev}): Embedded audio component loaded");

            function setSpeakButtonEnabled(enabled, initialText) {{
                var btnOriginal = document.getElementById("kokoro-audio-control");
                var btnClone = document.getElementById("kokoro-audio-control-clone");

                if (btnClone) {{
                    btnClone.disabled = !enabled;
                    btnClone.style.opacity = enabled ? "1" : "0.5";
                    if (!enabled) {{
                        btnClone.textContent = "Speak";
                    }} else {{
                        btnClone.textContent = initialText;
                        btnClone.className = initialText === 'Speak' ? "lg primary svelte-cmf5ev" : "lg secondary svelte-cmf5ev";
                    }}
                }}
                if (btnOriginal) {{
                    btnOriginal.disabled = !enabled;
                    // Only update text content, no visual styling
                    if (!enabled) {{
                        btnOriginal.textContent = "Speak";
                    }} else {{
                        btnOriginal.textContent = initialText;
                    }}
                    // Keep it hidden
                    btnOriginal.classList.add("kokoro-hidden-original");
                }}
                {info_log}("KokoroTTS ({abbrev}_setBtn): Button enabled:", enabled);
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
                    // Don't set className - keep it hidden
                    btnOriginal.classList.add("kokoro-hidden-original");

                    //if (text === 'Speak') {{
                    //    btnOriginal.className = "lg primary svelte-cmf5ev";
                    //}} else {{
                    //    btnOriginal.className = "lg secondary svelte-cmf5ev";
                    //}}
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
                        audio.play().then(() => {{
                            forceButtonUpdate('Pause');
                            setSpeakButtonEnabled(true);
                            {info_log}("KokoroTTS ({abbrev}_toggle): Playing audio");
                        }}).catch(e => {{
                            {error_log}("KokoroTTS ({abbrev}_toggle): Audio play failed:", e);
                        }});
                    }} else {{
                        audio.pause();
                        forceButtonUpdate('Speak');
                        setSpeakButtonEnabled(true);
                        {info_log}("KokoroTTS ({abbrev}_toggle): Pausing audio");
                    }}
                }} else {{
                    {error_log}("KokoroTTS ({abbrev}_toggle): Audio element not found");
                }}
            }};

            function initializeAudioComponent() {{
                // Clean up any existing old audio elements first
                var existingAudios = document.querySelectorAll('audio[id="kokoro-current-audio"]');
                if (existingAudios.length > 1) {{
                    // Remove all but the last one (newest)
                    for (var i = 0; i < existingAudios.length - 1; i++) {{
                        try {{
                            existingAudios[i].remove();
                            {info_log}("KokoroTTS ({abbrev}): Removed old audio element");
                        }} catch(e) {{
                            {error_log}("KokoroTTS ({abbrev}): Error removing old audio element:", e);
                        }}
                    }}
                }}

                var audio = document.getElementById('kokoro-current-audio');
                if (audio && !audio.kokoroInitialized) {{
                    audio.kokoroInitialized = true;
                    / Apply the speed from data-speed attribute
                    var speed = parseFloat(audio.getAttribute('data-speed')) || 1.0;
                    audio.playbackRate = speed;
                    {info_log}("KokoroTTS (chah): Set playback rate to:", speed);
                    audio.addEventListener('playing', function() {{
                        forceButtonUpdate('Pause');
                        setSpeakButtonEnabled(true);
                        {info_log}("KokoroTTS ({abbrev}): Audio playing - button set to Pause");
                    }});

                    audio.addEventListener('pause', function() {{
                        forceButtonUpdate('Speak');
                        {info_log}("KokoroTTS ({abbrev}): Audio paused - button set to Speak");
                    }});

                    audio.addEventListener('ended', function() {{
                        forceButtonUpdate('Speak');
                        {info_log}("KokoroTTS ({abbrev}): Audio ended - button set to Speak");
                    }});

                    audio.addEventListener('error', function(e) {{
                        {error_log}("KokoroTTS ({abbrev}): Audio error:", e);
                        forceButtonUpdate('Speak');
                        setSpeakButtonEnabled(false);
                    }});

                    {info_log}("KokoroTTS ({abbrev}): Audio component fully initialized");

                    // Check if audio is already playing (autoplay detection)
                    setTimeout(() => {{
                        if (audio && !audio.paused && !audio.ended) {{
                            setSpeakButtonEnabled(true, 'Pause');
                            {info_log}("KokoroTTS ({abbrev}): Detected autoplay, enabled as Pause");
                        }} else {{
                            setSpeakButtonEnabled(true, 'Speak');
                        }}
                    }}, 500);

                    setTimeout(() => {{
                        if (audio && !audio.paused && !audio.ended) {{
                            forceButtonUpdate('Pause');
                            {info_log}("KokoroTTS ({abbrev}): Detected autoplay, set button to Pause");
                        }}
                    }}, 500);
                }} else if (!audio) {{
                    {error_log}("KokoroTTS ({abbrev}): Audio element not found during initialization");
                }}
            }}

            try {{
                initializeAudioComponent();
            }} catch(e) {{
                {error_log}("KokoroTTS ({abbrev}): Error during audio component initialization:", e);
            }}
        </script>
    </div>
    """

    return result


def create_button_cloning_js(abbrev):
    """Reusable button cloning functionality"""
    info_log, error_log = get_console_log_prefix()

    result = f"""
    function cloneKokoroButtons() {{
        try {{
            var originalBtn = document.getElementById("kokoro-audio-control");
            var generateBtn = document.getElementById("Generate");

            if (originalBtn && generateBtn && !document.getElementById("kokoro-audio-control-clone")) {{
                var clonedBtn = originalBtn.cloneNode(true);
                clonedBtn.id = "kokoro-audio-control-clone";

                // Hide the original and show the clone
                originalBtn.classList.add("kokoro-hidden-original");
                clonedBtn.classList.remove("kokoro-hidden-original");

                generateBtn.insertAdjacentElement("afterend", clonedBtn);
                clonedBtn.style.setProperty("margin-left", "-10px");

                // Force clone to correct initial state regardless of original
                clonedBtn.disabled = true;
                clonedBtn.style.opacity = "0.5";
                clonedBtn.textContent = "Speak";

                [clonedBtn, originalBtn].forEach(btn => {{
                    btn.addEventListener("click", function() {{
                        if (window.kokoroToggleAudio) {{
                            window.kokoroToggleAudio();
                        }} else {{
                            {error_log}("KokoroTTS ({abbrev}): kokoroToggleAudio function not found");
                        }}
                    }});
                }});

                {info_log}("KokoroTTS ({abbrev}): Button cloned successfully");
            }} else {{
                if (!originalBtn) {{
                    {error_log}("KokoroTTS ({abbrev}): Original button not found for cloning");
                }}
                if (!generateBtn) {{
                    {error_log}("KokoroTTS ({abbrev}): Generate button not found for cloning");
                }}
                if (document.getElementById("kokoro-audio-control-clone")) {{
                    {info_log}("KokoroTTS ({abbrev}): Button already cloned, skipping");
                }}
            }}
        }} catch(e) {{
            {error_log}("KokoroTTS ({abbrev}): Error in cloneKokoroButtons:", e);
        }}
    }}
    """
    return result


def create_hidden_audio_js():
    """Generic utility to enable HTML components with embedded scripts"""
    abbrev = "chaj"
    info_log, error_log = get_console_log_prefix()

    # Use the refactored button cloning function
    button_cloning_js = create_button_cloning_js(abbrev)

    css_rule = '.kokoro-hidden-original { display: none !important; }'
    result = f"""
    {info_log}("KokoroTTS ({abbrev}): Script execution utility loaded");

    // Inject CSS
    try {{
        var style = document.createElement('style');
        style.textContent = '{css_rule}';
        document.head.appendChild(style);
    }} catch(e) {{
        {error_log}("KokoroTTS ({abbrev}): Error injecting CSS:", e);
    }}

    // Generic script executor for any HTML component
    function executeEmbeddedScripts() {{
        try {{
            var containers = document.querySelectorAll('[id*="kokoro-"]');
            containers.forEach(function(container) {{
                if (!container.scriptsExecuted) {{
                    container.scriptsExecuted = true;
                    var scripts = container.getElementsByTagName('script');
                    for (var i = 0; i < scripts.length; i++) {{
                        if (scripts[i].innerHTML) {{
                            {info_log}("KokoroTTS ({abbrev}): Executing embedded script");
                            try {{
                                eval(scripts[i].innerHTML);
                            }} catch(scriptError) {{
                                {error_log}("KokoroTTS ({abbrev}): Script execution error:", scriptError);
                            }}
                        }}
                    }}
                }}
            }});
        }} catch(e) {{
            {error_log}("KokoroTTS ({abbrev}): Error in executeEmbeddedScripts:", e);
        }}
    }}

    {button_cloning_js}

    // Initialize buttons to disabled state on startup
    function initializeButtonState() {{
        try {{
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
                {info_log}("KokoroTTS ({abbrev}): Set buttons to disabled startup state");
            }}
        }} catch(e) {{
            {error_log}("KokoroTTS ({abbrev}): Error in initializeButtonState:", e);
        }}
    }}

    setInterval(executeEmbeddedScripts, 300);
    setTimeout(initializeButtonState, 100);
    cloneKokoroButtons();
    """
    return result


