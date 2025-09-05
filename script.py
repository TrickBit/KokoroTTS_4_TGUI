import importlib
import pathlib
import html
import time
import subprocess
from pydub import AudioSegment
import yaml
import tempfile
import os
from extensions.KokoroTTS_4_TGUI.src import generate
from extensions.KokoroTTS_4_TGUI.src.voices import VOICES
from extensions.KokoroTTS_4_TGUI.src.debug import *
from extensions.KokoroTTS_4_TGUI.src.splittext import *
from extensions.KokoroTTS_4_TGUI.src import makehtml

import gradio as gr
from modules import shared

# Extension configuration constants
OurName = "KokoroTTS_4_TGUI"
DEFAULT_ICON_STYLE = "png"  # Options: "emoji", "png", "svg_simple", "svg_modern", "svg_3d"

def input_modifier(string, state):
    """
    Modify the input string to indicate that a voice message is being recorded.
    Called by text-generation-webui when processing user input.

    Args:
        string: The input text from the AI
        state: The current chat state

    Returns:
        str: The unmodified input string (we don't modify input)
    """
    shared.processing_message = "*Is recording a voice message...*"
    return string

def generate_default_preview_text():
    """
    Generate the default preview text template with placeholders for voice name and region.

    Returns:
        str: Template string with {VNAME} and {VREGION} placeholders
    """
    return "Hello, I'm {VNAME}{VREGION}, how are you today?"

def substitute_placeholders(text, voice):
    """
    Replace placeholders in the text with the voice name and region.

    Args:
        text (str): Text containing {VNAME} and {VREGION} placeholders
        voice (str): Voice identifier (e.g., 'bf_emma', 'am_john')

    Returns:
        str: Text with placeholders replaced by actual voice name and region
    """
    # Extract voice name (remove prefix if present)
    voice_name = voice[3:].capitalize() if voice.startswith(('bf_', 'af_', 'bm_', 'am_')) else voice.capitalize()

    # Determine region based on voice prefix
    region = " from the UK" if voice.startswith(('bf_', 'bm_')) else " from the US" if voice.startswith(('af_', 'am_')) else ""

    # Replace placeholders
    text = text.replace("{VNAME}", voice_name)
    text = text.replace("{VREGION}", region)
    return text

def sort_voices(voices):
    """
    Sort the voices list with priority: British female, American female, British male, American male, others.

    Args:
        voices (list): List of voice identifiers

    Returns:
        list: Sorted voice list with preferred ordering
    """
    def sort_key(voice):
        if voice.startswith('bf_'):
            return (0, voice.lower())  # British female - highest priority
        elif voice.startswith('af_'):
            return (1, voice.lower())  # American female
        elif voice.startswith('bm_'):
            return (2, voice.lower())  # British male
        elif voice.startswith('am_'):
            return (3, voice.lower())  # American male
        else:
            return (4, voice.lower())  # Others - lowest priority

    return sorted(voices, key=sort_key)

def save_setting(setting, value):
    """
    Save a setting to settings.yaml and update shared.args (only in non-multi-user mode).
    In multi-user mode, settings are not persisted to avoid conflicts.

    Args:
        setting (str): Setting name (without 'kokoro_' prefix)
        value: Setting value to save
    """
    # Skip saving in multi-user mode to avoid conflicts
    if getattr(shared.args, 'multi_user', False):
        log(f"Skipping save_setting for {setting} in multi-user mode")
        return

    settings_file = pathlib.Path(__file__).parent / 'settings.yaml'
    settings = {}

    try:
        # Load existing settings
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                settings = yaml.safe_load(f) or {}

        # Update setting with kokoro_ prefix
        settings[f"kokoro_{setting}"] = value

        # Save back to file
        with open(settings_file, 'w') as f:
            yaml.safe_dump(settings, f)

        log(f"Settings saved to {settings_file}: {setting}={value}")
    except Exception as e:
        log(f"Error saving setting {setting}: {e}")

    # Update runtime setting
    setattr(shared.args, f"kokoro_{setting}", value)

def save_voice(voice):
    """
    Save the selected voice name to settings.

    Args:
        voice (str): Voice identifier to save
    """
    log(f"Saving voice: {voice}")
    save_setting("voice", voice)

def save_preview_text(text):
    """
    Save the preview text to settings.yaml and update the template state.
    Returns default text if input is empty.

    Args:
        text (str): Preview text to save

    Returns:
        str: The saved text (or default if empty)
    """
    log(f"Saving preview text: {text}")
    if not text or text.isspace():
        text = generate_default_preview_text()
    save_setting("preview_text", text)
    return text

# EXPERIMENTAL MODE FEATURES - CURRENTLY DISABLED
# These functions are stubbed out for future development

def on_experimental_mode_change(is_enabled):
    """
    Handle changes to the experimental mode setting.

    Args:
        is_enabled (bool): Whether experimental mode is enabled

    Returns:
        str: Status message about the change
    """
    # Reload modules for hot-reloading during development
    makehtml = importlib.reload(importlib.import_module('extensions.KokoroTTS_4_TGUI.src.makehtml'))
    splittext = importlib.reload(importlib.import_module('extensions.KokoroTTS_4_TGUI.src.splittext'))
    generate = importlib.reload(importlib.import_module('extensions.KokoroTTS_4_TGUI.src.generate'))

    log(f"Experimental mode changed: {is_enabled}")
    save_setting("experimental", is_enabled)
    return f"{OurName} experimental mode {'enabled' if is_enabled else 'disabled'}."

def output_chunked(original_string, clean_text):
    """
    EXPERIMENTAL: Chunked output mode - generates multiple audio files for streaming playback.
    Currently disabled - falls back to standard mode.

    Args:
        original_string (str): Original response text
        clean_text (str): Cleaned text for TTS

    Returns:
        str: HTML string with embedded audio (falls back to standard mode)
    """
    log("Experimental chunked mode called - falling back to standard mode")
    return output_standard(original_string, clean_text)

def simple_chunk_text(text):
    """
    EXPERIMENTAL: Simple text chunking for experimental mode.
    Split text into sentences for incremental playback.

    Args:
        text (str): Text to chunk

    Returns:
        list: List of text chunks (currently returns single chunk)
    """
    log("Simple chunking called - returning single chunk")
    return [text] if text.strip() else []

# END EXPERIMENTAL FEATURES

def on_voice_change(selected_voice, template_state):
    """
    Handle voice selection changes, save the voice, and load it.

    Args:
        selected_voice (str): The newly selected voice
        template_state: Current template state from Gradio

    Returns:
        tuple: Updated dropdown component and template state
    """
    log(f"Voice changed: {selected_voice}")
    save_voice(selected_voice)
    generate.load_voice(selected_voice)
    return (
        gr.Dropdown(choices=sort_voices(VOICES), value=selected_voice, label="Voice", info="Select Voice", interactive=True),
        template_state
    )

def load_settings():
    """
    Load settings from settings.yaml or use defaults for first run.
    Handles backward compatibility for old boolean debug settings.
    """
    log("Entering load_settings")

    # Default settings for first run
    defaults = {
        'kokoro_enable_tts': False,
        'kokoro_speed': 1.0,
        'kokoro_pitch': 1.0,
        'kokoro_voice': VOICES[0] if VOICES else 'bf_emma',
        'kokoro_preview_text': generate_default_preview_text(),
        'kokoro_experimental': False,
        'kokoro_preprocess_code': False,
        'kokoro_enable_debug': 'errors',
        'kokoro_icon_style': DEFAULT_ICON_STYLE,
        'kokoro_multi_user_mode': False,  # Kept for future consideration
        'autoplay': False,  # Legacy compatibility
    }

    settings_file = pathlib.Path(__file__).parent / 'settings.yaml'
    settings = {}
    is_multi_user = getattr(shared.args, 'multi_user', False)
    log(f"Multi-user mode: {is_multi_user}")

    # Load settings file (only in non-multi-user mode)
    if not is_multi_user:
        try:
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    settings = yaml.safe_load(f) or {}
                log(f"Settings loaded from {settings_file}: {settings}")
        except Exception as e:
            log(f"Error loading settings from {settings_file}: {e}")

    # Merge defaults with user settings (user overrides defaults)
    final_settings = {**defaults, **settings}

    # Extract settings with fallback to defaults
    default_enable_tts = bool(final_settings.get('kokoro_enable_tts', final_settings.get('autoplay', False)))
    default_speed = float(final_settings.get('kokoro_speed'))
    default_pitch = float(final_settings.get('kokoro_pitch'))
    default_preview_text = final_settings.get('kokoro_preview_text')
    default_experimental = bool(final_settings.get('kokoro_experimental'))
    default_preprocess_code = bool(final_settings.get('kokoro_preprocess_code'))
    default_voice = final_settings.get('kokoro_voice')
    default_icon_style = final_settings.get('kokoro_icon_style')

    # Handle debug mode - convert old boolean setting to new string setting
    debug_setting = final_settings.get('kokoro_enable_debug')
    if isinstance(debug_setting, bool):
        default_debug_mode = 'all' if debug_setting else 'off'
    else:
        default_debug_mode = debug_setting if debug_setting in ['off', 'errors', 'all'] else 'errors'

    # Validate voice exists
    if default_voice not in VOICES:
        default_voice = VOICES[0] if VOICES else 'bf_emma'

    # Apply settings to shared.args for runtime access
    setattr(shared.args, 'kokoro_speed', default_speed)
    setattr(shared.args, 'kokoro_pitch', default_pitch)
    setattr(shared.args, 'kokoro_enable_tts', default_enable_tts)
    setattr(shared.args, 'kokoro_preview_text', default_preview_text)
    setattr(shared.args, 'kokoro_experimental', default_experimental)
    setattr(shared.args, 'kokoro_preprocess_code', default_preprocess_code)
    setattr(shared.args, 'kokoro_enable_debug', default_debug_mode)
    setattr(shared.args, 'kokoro_voice', default_voice)
    setattr(shared.args, 'kokoro_icon_style', default_icon_style)

    log(f"Settings applied: speed={default_speed}, pitch={default_pitch}, enable_tts={default_enable_tts}, "
        f"preview_text={default_preview_text[:50]}..., experimental={default_experimental}, "
        f"preprocess_code={default_preprocess_code}, debug_mode={default_debug_mode}, voice={default_voice}")

def output_modifier(string, state):
    """
    Generate and automatically play TTS audio for AI responses.
    This is called by text-generation-webui after the AI generates a response.

    Args:
        string (str): The AI's response text
        state: Current chat state

    Returns:
        str: The original string (audio is handled separately via hidden div)
    """
    # Check if TTS is enabled
    if not getattr(shared.args, 'kokoro_enable_tts', False):
        log("TTS disabled, skipping audio generation")
        return string

    log("TTS enabled, generating audio")
    log(f"Original string length: {len(string)}")

    # get the setting for code/markdown preprocessing
    preprocess_code = getattr(shared.args, 'kokoro_preprocess_code', True)
    # Clean the text for TTS processing
    string_for_tts = clean_text_for_tts(string, preprocess_code)
    log(f"Cleaned string: '{string_for_tts[:100]}...'")

    # Skip if no meaningful text remains after cleaning
    if not string_for_tts.strip():
        log("No text to speak after cleaning")
        return string

    # Check experimental mode (currently disabled)
    experimental_mode = getattr(shared.args, 'kokoro_experimental', False)
    log(f"Experimental mode: {experimental_mode}")

    if experimental_mode:
        # Experimental mode - currently falls back to standard
        log("Taking experimental chunked path (fallback to standard)")
        return output_chunked(string, string_for_tts)
    else:
        # Standard mode - single audio file with hidden div
        log("Taking standard output path")
        return output_standard(string, string_for_tts)

def output_standard(original_string, clean_text):
    """
    Standard output mode - generates single audio file with hidden div integration.

    Args:
        original_string (str): Original AI response text
        clean_text (str): Cleaned text ready for TTS

    Returns:
        str: Original string (audio HTML is stored separately for button access)
    """
    log("=== ENTERING output_standard ===")
    speed = getattr(shared.args, 'kokoro_speed', 1.0)
    pitch = getattr(shared.args, 'kokoro_pitch', 1.0)
    save_setting("speed", speed)
    save_setting("pitch", pitch)

    # Hot-reload makehtml for development
    try:
        makehtml = importlib.reload(importlib.import_module('extensions.KokoroTTS_4_TGUI.src.makehtml'))
    except Exception as e:
        log(f"Error reloading makehtml: {e}")
        return original_string

    # Ensure voice is loaded
    voice_name = getattr(generate, 'voice_name', None)
    if voice_name is None:
        default_voice = getattr(shared.args, 'kokoro_voice', 'bf_emma')
        generate.load_voice(default_voice)

    try:
        # Generate audio file
        msg_id = generate.run(clean_text, pitch=pitch)
        if msg_id is None:
            log("ERROR: generate.run() returned None!")
            return original_string

        # Create file URL for web interface
        file_url = f"/file/extensions/{OurName}/audio/{msg_id}.wav"

        # Create hidden audio HTML component
        audio_html = makehtml.create_ai_audio_html(file_url, speed, clean_text[:50])
        log(f"Generated audio HTML: {audio_html[:200]}...")

        # Store the audio HTML globally so the button can access it
        setattr(shared, 'kokoro_current_audio_html', audio_html)

        log(f"Generated audio HTML for hidden div: {len(audio_html)} chars")

        # Return just the original text - audio is handled via hidden div
        return original_string

    except Exception as e:
        log(f"Error in standard output: {e}")
        return original_string

def voice_preview(preview_text):
    """
    Generate voice preview with audio playback and speaker button.

    Args:
        preview_text (str): Text to use for preview

    Returns:
        str: HTML string with preview text and audio controls
    """
    log("Generating voice preview")
    speed = getattr(shared.args, 'kokoro_speed', 1.0)
    pitch = getattr(shared.args, 'kokoro_pitch', 1.0)
    save_setting("speed", speed)
    save_setting("pitch", pitch)


    voice_name = getattr(shared.args, 'kokoro_voice', VOICES[0])
    substituted_text = substitute_placeholders(preview_text, voice_name)

    # Clean substituted_text for TTS (always use standard [preprocess_code=False] mode for preview)
    # get the setting for code/markdown preprocessing - here for documentaion only
    # preprocess_code = getattr(shared.args, 'kokoro_preprocess_code', True)
    preprocess_code = False
    # Clean the substituted_text for TTS processing
    string_for_tts = clean_text_for_tts(substituted_text, preprocess_code )
    log(f"Cleaned string: '{string_for_tts[:100]}...'")

    # Hot-reload makehtml for development
    makehtml = importlib.reload(importlib.import_module('extensions.KokoroTTS_4_TGUI.src.makehtml'))

    try:
        # Generate preview audio
        msg_id = generate.run(string_for_tts, preview=True, pitch=pitch)
        if msg_id:
            file_url = f"/file/extensions/{OurName}/audio/preview.wav"
            # Create speaker button with audio controls
            play_html =  makehtml.create_speaker_button_html(file_url, speed, preview_text)
            return f"Preview: {substituted_text} {play_html}"
    except Exception as e:
        log(f"Error generating preview: {e}")
        return f"Preview: {substituted_text} (Audio generation failed)"

    return f"Preview: {substituted_text}"

def ui():
    """
    Create the Gradio UI for the KokoroTTS extension.
    Builds all the UI components and sets up event handlers.
    """
    log("Creating UI")

    # Load settings and initialize
    load_settings()
    default_voice = getattr(shared.args, 'kokoro_voice')
    default_speed = getattr(shared.args, 'kokoro_speed')
    default_enable_tts = getattr(shared.args, 'kokoro_enable_tts')
    default_preview_text = getattr(shared.args, 'kokoro_preview_text')
    default_experimental = getattr(shared.args, 'kokoro_experimental')
    default_preprocess_code = getattr(shared.args, 'kokoro_preprocess_code')
    default_debug_mode = getattr(shared.args, 'kokoro_enable_debug')
    default_pitch = getattr(shared.args, 'kokoro_pitch', 1.0)

    # Load the default voice
    generate.load_voice(default_voice)

    # Voice selection info text
    info_voice = (
        "Select a Voice.\nThe default voice is a 50-50 mix of Bella & Sarah.\nVoices starting with 'a' are American "
        "english, voices with 'b' are British english"
    )

    # Create the main UI accordion
    with gr.Accordion(f"{OurName}"):
        sorted_voices = sort_voices(VOICES)

        # Audio control section
        with gr.Row():
            # Hidden audio div for main playback
            hidden_audio_div = gr.HTML(
                value="Player Audio Code Goes Here",
                elem_id="kokoro-hidden-audio",
                visible=False
            )

            # Audio control button (gets cloned and positioned next to Generate button)
            audio_control_btn = gr.Button(
                "Speak",
                elem_id="kokoro-audio-control",
                visible=True,
                elem_classes=["lg", "primary", "svelte-cmf5ev", "kokoro-hidden-original"]
            )

        # Voice selection
        voice = gr.Dropdown(
            choices=sorted_voices,
            value=default_voice,
            label="Voice",
            info=info_voice,
            interactive=True
        )
        template_state = gr.State(value=default_preview_text)

        # Preview text
        preview_text = gr.Textbox(
            value=default_preview_text,
            label="Preview text",
            info="Text to use for voice preview. Use {VNAME} for the voice name and {VREGION} for the region.",
            interactive=True
        )

        # Preview button and output
        preview = gr.Button("Voice preview", variant="secondary")
        preview_output = gr.HTML()

        # Speed control
        speed = gr.Slider(
            minimum=0.5,
            maximum=1.5,
            step=0.1,
            value=default_speed,
            label="Voice speed",
            info="Adjust voice playback speed",
            interactive=True
        )

        # Experimental pitch control
        pitch = gr.Slider(
            minimum=0.5,
            maximum=1.5,
            step=0.1,
            value=default_pitch,
            label="Voice pitch (experimental - not implemented yet)",
            info="1.0 = normal pitch, <1.0 = lower, >1.0 = higher. Currently does nothing.",
            interactive=True
        )

        # TTS enable toggle
        enable_tts = gr.Checkbox(
            label="Enable TTS",
            value=default_enable_tts,
            info="Enable to automatically generate and play TTS audio",
            interactive=True
        )

        # Text splitting method
        splitting_method = gr.Radio(
            ["Split by sentence", "Split by Word"],
            info="Kokoro only supports 510 tokens. Split method for long text.",
            value="Split by sentence",
            label="Splitting method",
            interactive=True
        )

        # Code preprocessing options
        preprocess_code = gr.Checkbox(
            label="Preprocess code blocks and markdown",
            value=default_preprocess_code,
            info="Convert code blocks and markdown to speech-friendly text",
            interactive=True
        )

        # Debug mode selection
        debug_mode = gr.Radio(
            choices=["off", "errors", "all"],
            value=default_debug_mode,
            label="Debug mode",
            info="off: no console logging, errors: only errors, all: full logging including info",
            interactive=True
        )

        # EXPERIMENTAL MODE - CURRENTLY COMMENTED OUT
        # Uncomment these lines when experimental features are ready
        #
        # experimental_mode = gr.Checkbox(
        #     label="Enable experimental mode (chunked playback)",
        #     value=default_experimental,
        #     info="Experimental chunked audio for more responsive playback",
        #     interactive=True
        # )

        # Status textbox for feedback
        status_textbox = gr.Textbox(label="Status", interactive=False)

    # Event handlers setup

    # Voice selection events
    voice.change(on_voice_change, inputs=[voice, template_state], outputs=[voice, template_state])

    # Speed and pitch control events - doing this in the speak is actually created now
    # speed.change(lambda x: save_setting("speed", x), inputs=[speed])
    # pitch.change(lambda x: save_setting("pitch", x), inputs=[pitch])

    # Audio control button handler
    def handle_audio_button_click():
        """Return stored audio HTML to update the hidden div"""
        audio_html = getattr(shared, 'kokoro_current_audio_html', '')
        log(f"Button clicked, returning audio HTML: {len(audio_html)} chars")
        return audio_html

    audio_control_btn.click(handle_audio_button_click, outputs=[hidden_audio_div])

    # TTS toggle with button visibility
    def handle_tts_toggle(x):
        save_setting("enable_tts", x)
        return gr.Button(visible=x)

    enable_tts.change(handle_tts_toggle, inputs=[enable_tts], outputs=[audio_control_btn])

    # Preview text and voice preview
    preview_text.change(save_preview_text, inputs=[preview_text], outputs=[template_state])
    preview.click(voice_preview, inputs=[preview_text], outputs=preview_output)

    # Text processing settings
    splitting_method.change(generate.set_splitting_type, inputs=[splitting_method])
    preprocess_code.change(lambda x: save_setting("preprocess_code", x), inputs=[preprocess_code])
    debug_mode.change(lambda x: save_setting("enable_debug", x), inputs=[debug_mode])

    # EXPERIMENTAL MODE HANDLERS - CURRENTLY COMMENTED OUT
    # Uncomment when experimental features are ready
    #
    # experimental_mode.change(on_experimental_mode_change, inputs=[experimental_mode], outputs=[status_textbox])

    # Auto-load stored audio when AI responds
    def load_stored_audio():
        """Load stored audio HTML when display updates - only if new audio available"""
        audio_html = getattr(shared, 'kokoro_current_audio_html', '')
        last_audio_html = getattr(shared, 'kokoro_last_loaded_audio_html', '')

        # Only act if there's new audio content
        if audio_html and audio_html != last_audio_html:
            # Mark this audio as loaded to prevent re-loading
            setattr(shared, 'kokoro_last_loaded_audio_html', audio_html)
            log(f"Auto-loading NEW audio into hidden div: {len(audio_html)} chars")
            return audio_html, gr.Button(value="Pause", interactive=True)
        elif audio_html:
            # Same audio, no update needed - return current state
            return gr.update(), gr.update()  # No change
        else:
            # No audio available
            return "", gr.Button(value="Speak", interactive=False)

    # Chain from the AI response display component update
    if 'display' in shared.gradio:
        shared.gradio['display'].change(
            load_stored_audio,
            outputs=[hidden_audio_div, audio_control_btn]
        )
        log("Event chaining established: display -> hidden_audio_div + audio_control_btn")
    else:
        log("Warning: shared.gradio['display'] not available for chaining")

def custom_js():
    """
    Returns custom JavaScript as a string. It is applied whenever the web UI is loaded.
    This manages audio playback toggle, button cloning, synced state, tooltips, etc.

    Returns:
        str: JavaScript code for audio management and UI enhancements
    """
    return makehtml.create_ai_audio_js()
