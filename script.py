import importlib
import pathlib
import html
import time
import subprocess
from pydub import AudioSegment
import yaml
import tempfile
import os
import gradio as gr
from modules import shared

# ONNX-only backend - PyTorch support removed
extension_dir = pathlib.Path(__file__).parent
extension_name = extension_dir.name

setattr(shared.args, f"kokoro_backend", "onnx")
setattr(shared.args, f"kokoro_homedir", extension_dir)
setattr(shared.args, f"kokoro_extension_name", extension_name)
setattr(shared.args, f"kokoro_icon_style", "png")

# ONNX backend imports only
generate = importlib.import_module('extensions.KokoroTTS_4_TGUI.src.onnx.generate')
voices_module = importlib.import_module('extensions.KokoroTTS_4_TGUI.src.onnx.voices')
blender = importlib.import_module('extensions.KokoroTTS_4_TGUI.src.onnx.blender')
VOICES = voices_module.VOICES

from extensions.KokoroTTS_4_TGUI.src.debug import *
from extensions.KokoroTTS_4_TGUI.src.splittext import *
from extensions.KokoroTTS_4_TGUI.src import makehtml

del extension_dir, extension_name


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
        'kokoro_language': 'en-us',
        'kokoro_experimental': False,
        'kokoro_preprocess_code': False,
        'kokoro_enable_debug': 'errors',
        'kokoro_icon_style': getattr(shared.args, 'kokoro_icon_style'),
        'kokoro_multi_user_mode': False,
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
    default_language = final_settings.get('kokoro_language')

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
    setattr(shared.args, 'kokoro_language', default_language)

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
        extension_name = getattr(shared.args, 'kokoro_extension_name')
        file_url = f"/file/extensions/{extension_name}/audio/{msg_id}.wav"

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
    preprocess_code = False
    # Clean the substituted_text for TTS processing
    string_for_tts = clean_text_for_tts(substituted_text, preprocess_code)
    log(f"Cleaned string: '{string_for_tts[:100]}...'")

    # Hot-reload makehtml for development
    makehtml = importlib.reload(importlib.import_module('extensions.KokoroTTS_4_TGUI.src.makehtml'))

    try:
        # Generate preview audio
        msg_id = generate.run(string_for_tts, preview=True, pitch=pitch)
        if msg_id:
            extension_name = getattr(shared.args, 'kokoro_extension_name')
            file_url = f"/file/extensions/{extension_name}/audio/preview.wav"
            # Create speaker button with audio controls
            play_html = makehtml.create_speaker_button_html(file_url, speed, preview_text)
            return f"Preview: {substituted_text} {play_html}"
    except Exception as e:
        log(f"Error generating preview: {e}")
        return f"Preview: {substituted_text} (Audio generation failed)"

    return f"Preview: {substituted_text}"

def substitute_placeholders(text, voice):
    """
    Replace placeholders in the text with the voice name and region.

    Args:
        text (str): Text containing {VNAME} and {VREGION} placeholders
        voice (str): Voice identifier (e.g., 'bf_emma', 'mf_jane', 'mx_box')

    Returns:
        str: Text with placeholders replaced by actual voice name and region
    """
    voice_name = ""
    region = ""

    # Handle blended voices (mf_, mm_, mx_)
    if voice.startswith('mf_'):
        voice_name = voice[3:].capitalize()
        region = " (blended female voice)"
    elif voice.startswith('mm_'):
        voice_name = voice[3:].capitalize()
        region = " (blended male voice)"
    elif voice.startswith('mx_'):
        voice_name = voice[3:].capitalize()
        region = " (blended character voice)"

    # Handle physical voices - English
    elif voice.startswith(('bf_', 'af_', 'bm_', 'am_')):
        voice_name = voice[3:].capitalize()
        if voice.startswith(('bf_', 'bm_')):
            region = " from the UK"
        elif voice.startswith(('af_', 'am_')):
            region = " from the US"

    # Handle other language voices
    elif voice.startswith('ef_'):
        voice_name = voice[3:].capitalize()
        region = " from Europe"
    elif voice.startswith('ff_'):
        voice_name = voice[3:].capitalize()
        region = " from France"
    elif voice.startswith(('hf_', 'hm_')):
        voice_name = voice[3:].capitalize()
        region = " From India"
    elif voice.startswith(('if_', 'im_')):
        voice_name = voice[3:].capitalize()
        region = " from Italy"
    elif voice.startswith(('jf_', 'jm_')):
        voice_name = voice[3:].capitalize()
        region = " from Japan"
    elif voice.startswith(('pf_', 'pm_')):
        voice_name = voice[3:].capitalize()
        region = " from Portugal"
    elif voice.startswith(('zf_', 'zm_')):
        voice_name = voice[3:].capitalize()
        region = " from China"

    # Fallback for unknown voice formats
    else:
        voice_name = voice.capitalize()
        region = ""

    # Replace placeholders
    text = text.replace("{VNAME}", voice_name)
    text = text.replace("{VREGION}", region)

    return text

def get_voice_display_mapping():
    """Create mapping between display names and actual voice IDs"""
    from extensions.KokoroTTS_4_TGUI.src.onnx.voices import VOICES, get_voice_info

    display_mapping = {}

    for voice_id in VOICES:
        if voice_id.startswith('m'):  # Blended voices (mf_, mm_, mx_)
            if voice_id.startswith('mf_'):
                display_name = f"{voice_id[3:].title()} - Blended Female"
            elif voice_id.startswith('mm_'):
                display_name = f"{voice_id[3:].title()} - Blended Male"
            elif voice_id.startswith('mx_'):
                display_name = f"{voice_id[3:].title()} - Blended Character"
            else:
                display_name = voice_id
        else:  # Physical voices
            voice_info = get_voice_info(voice_id)
            display_name = f"{voice_info['display_name']} - {voice_info['country']} {voice_info['gender']}"

        display_mapping[display_name] = voice_id

    return display_mapping

def get_sorted_display_names():
    """Get sorted display names for dropdown, grouped by language then gender"""
    mapping = get_voice_display_mapping()

    # Sort with priority: Language groups, then gender within each language
    def sort_key(display_name):
        # Extract the suffix after " - " to determine the category
        if " - " in display_name:
            suffix = display_name.split(" - ", 1)[1]
            name = display_name.split(" - ", 1)[0]
        else:
            suffix = display_name
            name = display_name

        # Language and gender priority
        if 'American Female' in suffix:
            return (0, 0, name)  # English first, females first
        elif 'American Male' in suffix:
            return (0, 1, name)
        elif 'British Female' in suffix:
            return (1, 0, name)
        elif 'British Male' in suffix:
            return (1, 1, name)
        elif 'European Female' in suffix:
            return (2, 0, name)
        elif 'European Male' in suffix:
            return (2, 1, name)
        elif 'French Female' in suffix:
            return (3, 0, name)
        elif 'French Male' in suffix:
            return (3, 1, name)
        elif 'Italian Female' in suffix:
            return (4, 0, name)
        elif 'Italian Male' in suffix:
            return (4, 1, name)
        elif 'Japanese Female' in suffix:
            return (5, 0, name)
        elif 'Japanese Male' in suffix:
            return (5, 1, name)
        elif 'Portuguese Female' in suffix:
            return (6, 0, name)
        elif 'Portuguese Male' in suffix:
            return (6, 1, name)
        elif 'Chinese Female' in suffix:
            return (7, 0, name)
        elif 'Chinese Male' in suffix:
            return (7, 1, name)
        elif 'Hindi Female' in suffix:
            return (8, 0, name)
        elif 'Hindi Male' in suffix:
            return (8, 1, name)
        elif 'Blended Female' in suffix:
            return (9, 0, name)
        elif 'Blended Male' in suffix:
            return (9, 1, name)
        elif 'Blended Character' in suffix:
            return (9, 2, name)
        else:
            return (99, 99, name)  # Unknown goes last

    sorted_names = sorted(mapping.keys(), key=sort_key)
    return sorted_names, mapping

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
    default_language = getattr(shared.args, 'kokoro_language', 'en-us')

    # Load the default voice
    generate.load_voice(default_voice)

    # Enhanced voice selection with display names
    sorted_display_names, voice_mapping = get_sorted_display_names()
    default_voice_id = getattr(shared.args, 'kokoro_voice')

    # Find display name for default voice
    default_display_name = None
    for display_name, voice_id in voice_mapping.items():
        if voice_id == default_voice_id:
            default_display_name = display_name
            break

    # Fallback if default voice not found in mapping
    if default_display_name is None:
        if sorted_display_names:
            default_display_name = sorted_display_names[0]
            log(f"Warning: Default voice {default_voice_id} not found, using {default_display_name}")
        else:
            default_display_name = "No voices available"
            log("Warning: No voices available")

    # Voice selection info text
    info_voice = (
        "Select a Voice with full description.\n"
        "British and American voices use regional pronunciation.\n"
        "Blended voices combine multiple voice characteristics."
    )

    kokoro_title = f"{getattr(shared.args, 'kokoro_extension_name')} - ONNX Backend"

    # Create the main UI accordion
    with gr.Accordion(f"{kokoro_title}"):
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

        # Enhanced voice selection dropdown
        voice_dropdown = gr.Dropdown(
            choices=sorted_display_names,
            value=default_display_name,
            label="Voice",
            info=info_voice,
            interactive=True
        )

        # Language selection for pronunciation (ONNX-only feature)
        language_dropdown = gr.Dropdown(
            choices=[
                ("English (US)", "en-us"),
                ("English (UK)", "en-gb"),
                ("French", "fr-fr"),
                ("Italian", "it"),
                ("Japanese", "ja"),
                ("Portuguese", "pt"),
                ("Chinese", "zh"),
                ("Hindi", "hi")
            ],
            value=default_language,
            label="Pronunciation Language",
            info="Language for phonemization - voices will speak English with this accent/pronunciation",
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
            label="Voice pitch (experimental)",
            info="1.0 = normal pitch, <1.0 = lower, >1.0 = higher. May affect quality.",
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

        # Status textbox for feedback
        status_textbox = gr.Textbox(label="Status", interactive=False)

    # Event handlers setup

    # Updated voice change handler
    def on_voice_change_enhanced(selected_display_name, template_state):
        current_mapping = get_voice_display_mapping()
        actual_voice_id = current_mapping.get(selected_display_name)

        if actual_voice_id:
            log(f"Voice changed: '{selected_display_name}' → {actual_voice_id}")
            save_voice(actual_voice_id)
            generate.load_voice(actual_voice_id)
        else:
            log(f"Warning: Could not find voice ID for display name: {selected_display_name}")

        # Get updated choices in case voices changed
        updated_display_names, updated_mapping = get_sorted_display_names()

        return (
            gr.Dropdown(choices=updated_display_names, value=selected_display_name),
            template_state
        )

    # Voice selection events
    voice_dropdown.change(
        on_voice_change_enhanced,
        inputs=[voice_dropdown, template_state],
        outputs=[voice_dropdown, template_state]
    )

    # Keep immediate runtime updates:
    speed.change(lambda x: setattr(shared.args, 'kokoro_speed', x), inputs=[speed])
    pitch.change(lambda x: setattr(shared.args, 'kokoro_pitch', x), inputs=[pitch])

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
    # Language change handler for ONNX backend
    language_dropdown.change(lambda x: save_setting("language", x), inputs=[language_dropdown])

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

    # Voice Blending Section (ONNX only)
    blender.create_blending_ui(sorted_display_names)

def custom_js():
    """
    Returns custom JavaScript as a string. It is applied whenever the web UI is loaded.
    This manages audio playbook toggle, button cloning, synced state, tooltips, etc.

    Returns:
        str: JavaScript code for audio management and UI enhancements
    """
    return makehtml.create_ai_audio_js()
