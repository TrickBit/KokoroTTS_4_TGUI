"""
KokoroTTS HTML Generation Module

This module handles the generation of HTML components for audio playback and UI controls.
It provides self-contained audio components with embedded JavaScript logic and
utility functions for enabling script execution in Gradio environments.

Key components:
- Self-contained audio playback with embedded controls
- Button cloning and synchronization system
- Configurable logging based on debug settings
- Speaker button for preview functionality

Architecture:
- create_ai_audio_html() and create_ai_audio_js() are the main entry points
- create_hidden_audio_html() creates self-contained audio components
- create_hidden_audio_js() provides script execution utilities
- create_speaker_button_html() handles preview audio with play/pause buttons
"""

import time
import re
import html
from modules import shared
from extensions.KokoroTTS_4_TGUI.src.debug import *

def get_console_log_prefix():
    """
    Get the appropriate console logging prefix based on debug setting.

    Returns:
        tuple: (info_log_prefix, error_log_prefix) for JavaScript console logging
               info_log_prefix: "console.log" or "//" based on debug level
               error_log_prefix: "console.error" or "//" based on debug level

    Debug levels:
        'off' or False: No logging at all
        'errors' or True: Only error logging
        'all': Full logging (info + errors)
    """
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
    Create auto-playing audio with play/pause toggle button for voice previews.
    Uses PNG icons by default and applies speed control via onloadeddata event.

    This function creates a simple audio element with inline event handlers
    that work within Gradio's constraints. The audio auto-plays and the button
    allows toggling between play and pause states.

    Args:
        audio_url (str): URL to the audio file
        speed (float): Playback speed multiplier (e.g., 1.0 = normal, 0.5 = half speed)
        text (str): Text being spoken (for debugging/context, not displayed)

    Returns:
        str: HTML string containing audio element and speaker button

    Features:
        - Auto-playing audio with speed control
        - Icon-based play/pause button
        - Configurable icon style (PNG vs emoji)
        - Inline event handlers (Gradio compatible)
    """
    timestamp = int(time.time() * 1000)
    icon_style = getattr(shared.args, 'kokoro_icon_style', 'emoji')

    # Define icons based on style preference
    if icon_style == "png":
        play_icon = '<img src="/file/extensions/KokoroTTS_4_TGUI/images/play.png" width="50" height="50" style="vertical-align: middle;">'
        pause_icon = '<img src="/file/extensions/KokoroTTS_4_TGUI/images/pause.png" width="50" height="50" style="vertical-align: middle;">'
    else:  # emoji default
        play_icon = "🔊"
        pause_icon = "⏸️"

    # Escape icons for safe inline JavaScript usage
    play_icon_escaped = play_icon.replace('"', '&quot;').replace("'", "&#39;")
    pause_icon_escaped = pause_icon.replace('"', '&quot;').replace("'", "&#39;")

    return f"""
    <audio id="kokoro-preview-{timestamp}" style="display: none;" preload="auto" autoplay
           onloadeddata="this.playbackRate={speed};"
           onended="document.getElementById('kokoro-btn-{timestamp}').innerHTML='{play_icon_escaped}';document.getElementById('kokoro-btn-{timestamp}').title='Play voice preview';"
           onpause="document.getElementById('kokoro-btn-{timestamp}').innerHTML='{play_icon_escaped}';document.getElementById('kokoro-btn-{timestamp}').title='Play voice preview';"
           onplay="document.getElementById('kokoro-btn-{timestamp}').innerHTML='{pause_icon_escaped}';document.getElementById('kokoro-btn-{timestamp}').title='Pause voice preview';">
        <source src="{audio_url}?t={timestamp}" type="audio/wav">
    </audio>
    <button id="kokoro-btn-{timestamp}"
            onmousedown="var a=document.getElementById('kokoro-preview-{timestamp}'); if(a.paused){{a.play();}}else{{a.pause();}}"
            style="background: none; border: none; cursor: pointer; padding: 2px; margin-left: 5px; vertical-align: middle;"
            title="Pause voice preview">
        {pause_icon}
    </button>
    """

# MAIN ENTRY POINTS - Used by script.py

def create_ai_audio_html(url, speed, text):
    """
    Create the main audio component for AI responses.
    This is the primary entry point called by script.py for main TTS output.

    Args:
        url (str): URL to the audio file
        speed (float): Playback speed multiplier
        text (str): Text being spoken (for context)

    Returns:
        str: HTML string for audio component
    """
    return create_hidden_audio_html(url, speed, text)

def create_ai_audio_js():
    """
    Create the main JavaScript utility for enabling audio components.
    This is the primary entry point called by script.py's custom_js() function.

    Returns:
        str: JavaScript code for audio component management
    """
    return create_hidden_audio_js()

# HIDDEN AUDIO SYSTEM - Main implementation

def create_hidden_audio_html(url, speed, text):
    """
    Complete self-contained audio component with all logic embedded inline.
    This creates a hidden div containing an audio element and all necessary
    JavaScript for managing playback state and button synchronization.

    This is the core audio component that handles the main TTS output from AI responses.
    It uses a hidden div approach with embedded JavaScript that gets executed by
    the utility script in create_hidden_audio_js().

    Features:
        - Auto-playing audio with speed control
        - Button state management (Speak/Pause)
        - Automatic cleanup of old audio elements
        - Synchronization between original and cloned buttons
        - Error handling and logging
        - Original button stays hidden while clone is visible

    Architecture:
        The embedded JavaScript defines functions that:
        1. Manage button states (setSpeakButtonEnabled, forceButtonUpdate)
        2. Handle audio playback (kokoroToggleAudio - global function)
        3. Initialize audio component and event listeners
        4. Clean up old audio elements to prevent conflicts

    Args:
        url (str): URL to the audio file
        speed (float): Playback speed multiplier
        text (str): Text being spoken (for debugging context)

    Returns:
        str: HTML string with hidden div containing audio and JavaScript
    """
    abbrev = "chah"  # Abbreviation for logging identification
    timestamp = int(time.time() * 1000)
    ts_url = f"{url}?t={timestamp}"  # Add timestamp to prevent caching

    # Get logging prefixes based on current debug settings
    info_log, error_log = get_console_log_prefix()

    result = f"""
    <div id="kokoro-hidden-audio" style="display: none;">
        <audio id="kokoro-current-audio" preload="auto" autoplay onloadeddata="this.playbackRate={speed};">
            <source src="{ts_url}" type="audio/wav">
        </audio>
        <script>
            {info_log}("KokoroTTS ({abbrev}): Embedded audio component loaded");

            /**
             * Enable/disable speak buttons and set their initial state.
             * Operates on both original (hidden) and cloned (visible) buttons.
             *
             * The original button must stay hidden at all times - only its state
             * is updated for consistency. The cloned button gets full visual updates.
             *
             * @param {{boolean}} enabled - Whether buttons should be enabled
             * @param {{string}} initialText - Text to display on buttons ("Speak" or "Pause")
             */
            function setSpeakButtonEnabled(enabled, initialText) {{
                var btnOriginal = document.getElementById("kokoro-audio-control");
                var btnClone = document.getElementById("kokoro-audio-control-clone");

                // Update cloned button (the visible one)
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

                // Update original button (keep hidden but maintain state)
                if (btnOriginal) {{
                    btnOriginal.disabled = !enabled;
                    // Only update text content, no visual styling
                    if (!enabled) {{
                        btnOriginal.textContent = "Speak";
                    }} else {{
                        btnOriginal.textContent = initialText;
                    }}
                    // Ensure it stays hidden
                    btnOriginal.classList.add("kokoro-hidden-original");
                }}

                {info_log}("KokoroTTS ({abbrev}_setBtn): Button enabled:", enabled);
            }}

            /**
             * Force update button text and styling for both original and cloned buttons.
             * Used when audio state changes (play/pause/end).
             *
             * This function is called by audio event listeners to ensure button
             * state stays synchronized with actual audio playback state.
             *
             * @param {{string}} text - Text to display ("Speak" or "Pause")
             */
            function forceButtonUpdate(text) {{
                var btnClone = document.getElementById('kokoro-audio-control-clone');
                var btnOriginal = document.getElementById('kokoro-audio-control');

                // Update visible cloned button
                if (btnClone) {{
                    btnClone.textContent = text;
                    if (text === 'Speak') {{
                        btnClone.className = "lg primary svelte-cmf5ev";
                    }} else {{
                        btnClone.className = "lg secondary svelte-cmf5ev";
                    }}
                }}

                // Update hidden original button state (but keep it hidden)
                if (btnOriginal) {{
                    btnOriginal.textContent = text;
                    // Don't set className - keep it hidden
                    btnOriginal.classList.add("kokoro-hidden-original");
                }}

                // Delayed updates to ensure state persistence across Gradio updates
                setTimeout(() => {{
                    if (btnClone) btnClone.textContent = text;
                    if (btnOriginal) {{
                        btnOriginal.textContent = text;
                        btnOriginal.classList.add("kokoro-hidden-original");
                    }}
                }}, 100);
                setTimeout(() => {{
                    if (btnClone) btnClone.textContent = text;
                    if (btnOriginal) {{
                        btnOriginal.textContent = text;
                        btnOriginal.classList.add("kokoro-hidden-original");
                    }}
                }}, 500);
            }}

            /**
             * Global function to toggle audio playback.
             * Called by both original and cloned button click events.
             *
             * This is the main user interaction function - it handles play/pause
             * toggle and updates button states accordingly. Made global so both
             * the original and cloned buttons can call it.
             */
            window.kokoroToggleAudio = function() {{
                var audio = document.getElementById('kokoro-current-audio');

                if (audio) {{
                    window.kokoroCurrentAudio = audio;

                    if (audio.paused || audio.ended) {{
                        // Start/resume playback
                        if (audio.ended) audio.currentTime = 0;
                        audio.play().then(() => {{
                            forceButtonUpdate('Pause');
                            setSpeakButtonEnabled(true);
                            {info_log}("KokoroTTS ({abbrev}_toggle): Playing audio");
                        }}).catch(e => {{
                            {error_log}("KokoroTTS ({abbrev}_toggle): Audio play failed:", e);
                        }});
                    }} else {{
                        // Pause playback
                        audio.pause();
                        forceButtonUpdate('Speak');
                        setSpeakButtonEnabled(true);
                        {info_log}("KokoroTTS ({abbrev}_toggle): Pausing audio");
                    }}
                }} else {{
                    {error_log}("KokoroTTS ({abbrev}_toggle): Audio element not found");
                }}
            }};

            /**
             * Initialize the audio component and set up event listeners.
             *
             * This function handles:
             * 1. Cleanup of old audio elements (prevents multiple audio conflicts)
             * 2. Setting up event listeners for audio state changes
             * 3. Autoplay detection and button state initialization
             *
             * The initialization happens when the script is embedded and executed
             * by the utility script from create_hidden_audio_js().
             */
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

                    // Set up audio event listeners for state management
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
                    // Use delayed checks to ensure audio has time to start
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

            // Initialize the audio component with error handling
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
    """
    Reusable button cloning functionality with proper error handling.

    Creates a clone of the original audio control button and positions it
    next to the Generate button for better UX. The original button stays
    hidden while the clone becomes the visible, interactive element.

    This function is called by create_hidden_audio_js() to set up the
    button cloning system that allows the audio controls to appear
    next to the Generate button instead of in the KokoroTTS accordion.

    Args:
        abbrev (str): Abbreviation for logging identification

    Returns:
        str: JavaScript code for button cloning functionality

    Features:
        - Clones original button and positions next to Generate button
        - Hides original, shows clone
        - Sets up click event handlers on both buttons
        - Handles missing elements gracefully
        - Prevents duplicate cloning
    """
    info_log, error_log = get_console_log_prefix()

    result = f"""
    /**
     * Clone the original audio control button and position it next to Generate button.
     *
     * This function implements the button cloning system that allows users to
     * control audio playback from a button positioned next to the Generate button
     * rather than having to scroll down to the KokoroTTS settings section.
     *
     * The original button remains hidden while the clone is visible and functional.
     */
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

                // Position clone next to Generate button
                generateBtn.insertAdjacentElement("afterend", clonedBtn);
                clonedBtn.style.setProperty("margin-left", "-10px");

                // Set initial state for clone regardless of original state
                clonedBtn.disabled = true;
                clonedBtn.style.opacity = "0.5";
                clonedBtn.textContent = "Speak";

                // Add click event listeners to both buttons
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
                // Log specific missing elements for debugging
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
    """
    Generic utility to enable HTML components with embedded scripts.

    This is the main JavaScript utility that enables the hidden audio system
    to work within Gradio's constraints. It provides:

    1. Script execution system for embedded scripts
    2. Button cloning functionality
    3. CSS injection for hiding original buttons
    4. Initialization and monitoring loops

    The core challenge this solves is that Gradio doesn't execute embedded
    <script> tags in dynamically inserted HTML. This utility monitors for
    new HTML components and executes their embedded scripts using eval().

    Architecture:
        - executeEmbeddedScripts(): Finds and executes scripts in kokoro elements
        - Button cloning: Creates visible clone of hidden original button
        - CSS injection: Ensures proper hiding of original elements
        - Monitoring loops: Continuously checks for new components

    Returns:
        str: JavaScript code for complete audio component management system
    """
    abbrev = "chaj"  # Abbreviation for logging identification
    info_log, error_log = get_console_log_prefix()

    # Use the refactored button cloning function
    button_cloning_js = create_button_cloning_js(abbrev)

    css_rule = '.kokoro-hidden-original { display: none !important; }'

    result = f"""
    {info_log}("KokoroTTS ({abbrev}): Script execution utility loaded");

    // Inject CSS for hiding original elements
    try {{
        var style = document.createElement('style');
        style.textContent = '{css_rule}';
        document.head.appendChild(style);
    }} catch(e) {{
        {error_log}("KokoroTTS ({abbrev}): Error injecting CSS:", e);
    }}

    /**
     * Generic script executor for any HTML component with embedded scripts.
     *
     * This function solves the core problem that Gradio doesn't execute
     * embedded <script> tags in dynamically inserted HTML. It finds all
     * kokoro-related elements and executes their embedded scripts using eval().
     *
     * The function marks elements as processed to avoid re-execution and
     * handles errors gracefully to prevent breaking the UI.
     */
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

    /**
     * Initialize buttons to disabled state on startup.
     *
     * This function ensures that audio control buttons start in a consistent
     * disabled state when the page loads, before any audio is available.
     * It only runs if no audio component is currently present.
     */
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

    // Start monitoring and initialization
    // Script execution runs frequently to catch new components
    setInterval(executeEmbeddedScripts, 300);

    // Button initialization runs once after a delay to let UI load
    setTimeout(initializeButtonState, 100);

    // Button cloning runs once to set up the clone
    cloneKokoroButtons();
    """
    return result
