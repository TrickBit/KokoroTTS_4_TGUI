"""
KokoroTTS_4_TGUI UI Layer
Gradio interface components and event handlers only.
Business logic handled by trickbit.py core module.
"""

import gradio as gr
import importlib
from modules import shared
from extensions.KokoroTTS_4_TGUI.src.kshared import ksettings, kruntime
from extensions.KokoroTTS_4_TGUI.src import trickbit
from extensions.KokoroTTS_4_TGUI.src import makehtml
from extensions.KokoroTTS_4_TGUI.src.onnx import blender
from extensions.KokoroTTS_4_TGUI.src.debug import error_log, info_log, debug_log

# ============================================================================
# TEXT-GENERATION-WEBUI ENTRY POINTS
# ============================================================================

def input_modifier(string, state):
    """
    Purpose: Entry point for text-generation-webui - processes user input
    Pre: Valid input string from user
    Post: Processing message set via trickbit
    Args: string (str) - User input text
          state - Chat state object
    Returns: str - Unmodified input string (we don't modify input)
    """
    if not ksettings.get('enable_tts', False):
        return string  # Do nothing, don't even set processing message
    return trickbit.handle_input_modifier(string, state)

def output_modifier(string, state):
    """
    Purpose: Entry point for text-generation-webui - processes AI responses
    Pre: AI response generated
    Post: TTS audio processed via trickbit
    Args: string (str) - AI response text
          state - Chat state object
    Returns: str - Original string (audio handled separately)
    """
    if not ksettings.get('enable_tts', False):
        return string  # Do nothing, don't even set processing message
    return trickbit.handle_output_modifier(string, state)

def custom_js():
    """
    Purpose: Entry point for text-generation-webui - provides custom JavaScript
    Pre: Makehtml module available
    Post: None
    Args: None
    Returns: str - JavaScript code for audio management
    """
    return makehtml.create_ai_audio_js()

# ============================================================================
# UI EVENT HANDLER FUNCTIONS
# ============================================================================

def handle_voice_change(selected_display_name, template_state):
    """
    Purpose: Handle voice selection change from UI dropdown
    Pre: Valid display name selected, template state available
    Post: Voice loaded and settings saved
    Args: selected_display_name (str or list) - Selected voice display name
          template_state - Gradio state object for UI consistency
    Returns: template_state - Unchanged state object for Gradio
    """
    try:
        voice_id = trickbit.get_voice_id_from_display(selected_display_name)
        if voice_id:
            ksettings.set('voice', voice_id)
            ksettings.save()
            # Load voice for generation
            generate = importlib.import_module('extensions.KokoroTTS_4_TGUI.src.onnx.generate')
            generate.load_voice(voice_id)
            info_log(f"Voice changed to: {voice_id}")
    except Exception as e:
        error_log(f"Error changing voice: {e}")
    return template_state


def handle_tts_toggle(enabled):
    """
    Purpose: Handle TTS enable/disable toggle from UI
    Pre: Valid boolean input provided
    Post: TTS setting saved and button visibility updated
    Args: enabled (bool) - TTS enabled state
    Returns: gr.Button - Updated button with visibility
    """
    try:
        ksettings.set('enable_tts', enabled)
        ksettings.save()
        info_log(f"TTS {'enabled' if enabled else 'disabled'}")
        return gr.Button(visible=enabled)
    except Exception as e:
        error_log(f"Error toggling TTS: {e}")
        return gr.Button(visible=False)


def handle_preview_text_change(text):
    """
    Purpose: Handle preview text change with automatic default fallback
    Pre: Text may be empty or whitespace-only
    Post: Valid preview text saved to settings
    Args: text (str) - Preview text from user input
    Returns: str - Actual text saved (default if input was empty)
    """
    try:
        if not text or not text.strip():
            text = trickbit.generate_default_preview_text()
        ksettings.set('preview_text', text)
        ksettings.save()
        info_log(f"Preview text saved: {text[:30]}...")
        return text
    except Exception as e:
        error_log(f"Error saving preview text: {e}")
        return text


def handle_speed_change(speed):
    """
    Purpose: Handle speed slider change (runtime only, not persisted)
    Pre: Valid float speed value provided
    Post: Speed setting updated in runtime only
    Args: speed (float) - Playback speed multiplier
    Returns: None
    """
    try:
        ksettings.set('speed', speed)
        # Don't save immediately to avoid disk thrash during slider dragging
        info_log(f"Speed changed to: {speed}")
    except Exception as e:
        error_log(f"Error changing speed: {e}")


def handle_pitch_change(pitch):
    """
    Purpose: Handle pitch slider change (runtime only, not persisted)
    Pre: Valid float pitch value provided
    Post: Pitch setting updated in runtime only
    Args: pitch (float) - Pitch multiplier for audio generation
    Returns: None
    """
    try:
        ksettings.set('pitch', pitch)
        # Don't save immediately to avoid disk thrash during slider dragging
        info_log(f"Pitch changed to: {pitch}")
    except Exception as e:
        error_log(f"Error changing pitch: {e}")


def handle_splitting_method_change(method):
    """
    Purpose: Handle text splitting method change from UI
    Pre: Valid splitting method string provided
    Post: Splitting method applied to audio generator
    Args: method (str) - Splitting method name ('Split by sentence', etc.)
    Returns: None
    """
    try:
        trickbit.apply_splitting_method(method)
        info_log(f"Splitting method changed to: {method}")
    except Exception as e:
        error_log(f"Error changing splitting method: {e}")


def handle_preprocess_toggle(enabled):
    """
    Purpose: Handle code preprocessing toggle from UI
    Pre: Valid boolean input provided
    Post: Preprocessing setting saved
    Args: enabled (bool) - Preprocessing enabled state
    Returns: None
    """
    try:
        ksettings.set('preprocess_code', enabled)
        ksettings.save()
        info_log(f"Code preprocessing {'enabled' if enabled else 'disabled'}")
    except Exception as e:
        error_log(f"Error toggling preprocessing: {e}")


def handle_language_change(language):
    """
    Purpose: Handle pronunciation language change from UI
    Pre: Valid language code provided
    Post: Language setting saved for phonemization
    Args: language (str) - Language code (e.g., 'en-us', 'en-gb')
    Returns: None
    """
    try:
        ksettings.set('language', language)
        ksettings.save()
        info_log(f"Pronunciation language changed to: {language}")
    except Exception as e:
        error_log(f"Error changing language: {e}")


def handle_log_level_change(level):
    """
    Purpose: Handle log level change from UI
    Pre: Valid log level string provided
    Post: Log level setting saved
    Args: level (str) - Log level ('quiet', 'verbose')
    Returns: None
    """
    try:
        ksettings.set('log_level', level)
        ksettings.save()
        info_log(f"Log level changed to: {level}")
    except Exception as e:
        error_log(f"Error changing log level: {e}")


def handle_audio_button_click():
    """
    Purpose: Handle audio control button click
    Pre: Audio HTML may be stored in kruntime
    Post: Audio HTML returned for hidden div update
    Args: None
    Returns: str - Current audio HTML for playback
    """
    try:
        audio_html = trickbit.get_current_audio_html()
        info_log(f"Audio button clicked, returning HTML: {len(audio_html)} chars")
        return audio_html
    except Exception as e:
        error_log(f"Error getting audio HTML: {e}")
        return ""


def load_stored_audio_ui():
    """
    Purpose: Load stored audio HTML when display updates (AI responds)
    Pre: Audio generation may have completed
    Post: UI updated with new audio controls if available
    Args: None
    Returns: tuple - (audio_html, button_update) for UI components
    """
    try:
        audio_html, button_update = trickbit.load_stored_audio_for_ui()

        # Handle the different return scenarios
        if audio_html is None and button_update is None:
            # No change needed
            return gr.update(), gr.update()
        elif audio_html == "":
            # No audio available
            return "", gr.Button(value="Speak", interactive=False)
        else:
            # New audio available
            return audio_html, gr.Button(value="Pause", interactive=True)
    except Exception as e:
        error_log(f"Error loading stored audio: {e}")
        return gr.update(), gr.update()

# ============================================================================
# UI COMPONENTS AND LAYOUT
# ============================================================================

def ui():
    """
    Purpose: Create main Gradio UI interface
    Pre: Core system available via trickbit
    Post: UI components created with event handlers
    Args: None
    Returns: None
    """
    info_log("Creating UI")

    # Get default values from settings with fallbacks
    default_voice = ksettings.get('voice', 'bf_emma')
    default_speed = ksettings.get('speed', 1.0)
    default_enable_tts = ksettings.get('enable_tts', False)
    default_preview_text = ksettings.get('preview_text', trickbit.generate_default_preview_text())
    default_experimental = ksettings.get('experimental', False)
    default_preprocess_code_setting = ksettings.get('preprocess_code', False)
    default_debug_mode = ksettings.get('enable_debug', 'errors')
    default_pitch = ksettings.get('pitch', 1.0)
    default_language = ksettings.get('language', 'en-us')

    # Get enhanced voice selection with display names
    try:
        sorted_display_names, voice_mapping = trickbit.get_sorted_display_names()
    except Exception as e:
        error_log(f"Error getting voice names: {e}")
        sorted_display_names = ["Emma - British Female"]
        voice_mapping = {"Emma - British Female": "bf_emma"}

    # Find display name for default voice
    default_display_name = None
    for display_name, voice_id in voice_mapping.items():
        if voice_id == default_voice:
            default_display_name = display_name
            break

    # Fallback if default voice not found in mapping
    if default_display_name is None:
        if sorted_display_names:
            default_display_name = sorted_display_names[0]
            info_log(f"Warning: Default voice {default_voice} not found, using {default_display_name}")
        else:
            default_display_name = "No voices available"
            error_log("Warning: No voices available")

    # UI text and info
    info_voice = (
        "Select a Voice with full description.\n"
        "British and American voices use regional pronunciation.\n"
        "Blended voices combine multiple voice characteristics."
    )

    kokoro_title = f"{kruntime.get('extension_name', 'KokoroTTS_4_TGUI')} - ONNX Backend"

    # ========================================================================
    # MAIN UI ACCORDION
    # ========================================================================

    with gr.Accordion(f"{kokoro_title}"):

        # ====================================================================
        # AUDIO CONTROL SECTION
        # ====================================================================

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

        # ====================================================================
        # VOICE SELECTION SECTION
        # ====================================================================

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

        # Template state for preview text
        template_state = gr.State(value=default_preview_text)

        # ====================================================================
        # PREVIEW SECTION
        # ====================================================================

        # Preview text input
        preview_text = gr.Textbox(
            value=default_preview_text,
            label="Preview text",
            info="Text to use for voice preview. Use {VNAME} for the voice name and {VREGION} for the region.",
            interactive=True
        )

        # Preview button and output
        preview_btn = gr.Button("Voice preview", variant="secondary")
        preview_output = gr.HTML()

        # ====================================================================
        # AUDIO CONTROL SETTINGS
        # ====================================================================

        # Speed control
        speed_slider = gr.Slider(
            minimum=0.5,
            maximum=1.5,
            step=0.1,
            value=default_speed,
            label="Voice speed",
            info="Adjust voice playback speed",
            interactive=True
        )

        # Experimental pitch control
        pitch_slider = gr.Slider(
            minimum=0.5,
            maximum=1.5,
            step=0.1,
            value=default_pitch,
            label="Voice pitch (experimental)",
            info="1.0 = normal pitch, <1.0 = lower, >1.0 = higher. May affect quality.",
            interactive=True
        )

        # ====================================================================
        # TTS SETTINGS SECTION
        # ====================================================================

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

        # Code preprocessing options (renamed variable to avoid conflict)
        preprocess_code_checkbox = gr.Checkbox(
            label="Preprocess code blocks and markdown",
            value=default_preprocess_code_setting,
            info="Convert code blocks and markdown to speech-friendly text",
            interactive=True
        )

        # ====================================================================
        # DEBUG AND ADVANCED SETTINGS
        # ====================================================================

        log_level = gr.Radio(
            choices=["quiet", "verbose"],
            value="quiet",
            label="Logging Level",
            info="quiet: errors only, verbose: detailed progress messages",
            interactive=True
        )

        # Status textbox for feedback
        status_textbox = gr.Textbox(label="Status", interactive=False)

    # ========================================================================
    # VOICE BLENDING SECTION
    # ========================================================================

    # Voice blending UI (ONNX only)
    try:
        blender.create_blending_ui(sorted_display_names)
    except Exception as e:
        error_log(f"Error creating blending UI: {e}")

    # ========================================================================
    # EVENT HANDLERS SETUP
    # ========================================================================

    # Voice selection events
    voice_dropdown.change(
        handle_voice_change,
        inputs=[voice_dropdown, template_state],
        outputs=[template_state]
    )

    # Audio control settings - immediate runtime updates
    speed_slider.change(
        handle_speed_change,
        inputs=[speed_slider]
    )

    pitch_slider.change(
        handle_pitch_change,
        inputs=[pitch_slider]
    )

    # Audio control button handler
    audio_control_btn.click(
        handle_audio_button_click,
        outputs=[hidden_audio_div]
    )

    # TTS toggle with button visibility
    enable_tts.change(
        handle_tts_toggle,
        inputs=[enable_tts],
        outputs=[audio_control_btn]
    )

    # Preview text and voice preview
    preview_text.change(
        handle_preview_text_change,
        inputs=[preview_text],
        outputs=[template_state]
    )

    preview_btn.click(
        trickbit.generate_voice_preview,
        inputs=[preview_text],
        outputs=[preview_output]
    )

    # Text processing settings
    splitting_method.change(
        handle_splitting_method_change,
        inputs=[splitting_method]
    )

    preprocess_code_checkbox.change(
        handle_preprocess_toggle,
        inputs=[preprocess_code_checkbox]
    )

    log_level.change(
        handle_log_level_change,
        inputs=[log_level]
    )

    # Language change handler for ONNX backend
    language_dropdown.change(
        handle_language_change,
        inputs=[language_dropdown]
    )

    # ========================================================================
    # AUTO-LOAD AUDIO WHEN AI RESPONDS
    # ========================================================================

    # Chain from the AI response display component update
    try:
        if 'display' in shared.gradio:
            shared.gradio['display'].change(
                load_stored_audio_ui,
                outputs=[hidden_audio_div, audio_control_btn]
            )
            info_log("Event chaining established: display -> hidden_audio_div + audio_control_btn")
        else:
            info_log("Warning: shared.gradio['display'] not available for chaining")
    except Exception as e:
        error_log(f"Error setting up event chaining: {e}")

    info_log("UI creation completed successfully")
