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

OurName = "KokoroTTS_4_TGUI"
DEFAULT_ICON_STYLE = "png"  # Options: "emoji", "png", "svg_simple", "svg_modern", "svg_3d"
#default_enable_debug=True #toggle this to OFF/False for release - single user is still able to turn it on for the session

def input_modifier(string, state):
    """
    Modify the input string to indicate that a voice message is being recorded.
    """
    shared.processing_message = "*Is recording a voice message...*"
    return string

def generate_default_preview_text():
    """
    Generate the default preview text template with placeholders for voice name and region.
    """
    return "Hello, I'm {VNAME}{VREGION}, how are you today?"

def substitute_placeholders(text, voice):
    """
    Replace placeholders in the text with the voice name and region.
    """
    voice_name = voice[3:].capitalize() if voice.startswith(('bf_', 'af_', 'bm_', 'am_')) else voice.capitalize()
    region = " from the UK" if voice.startswith(('bf_', 'bm_')) else " from the US" if voice.startswith(('af_', 'am_')) else ""
    text = text.replace("{VNAME}", voice_name)
    text = text.replace("{VREGION}", region)
    return text

def sort_voices(voices):
    """
    Sort the voices list with priority: British female, American female, British male, American male, others.
    """
    def sort_key(voice):
        if voice.startswith('bf_'):
            return (0, voice.lower())
        elif voice.startswith('af_'):
            return (1, voice.lower())
        elif voice.startswith('bm_'):
            return (2, voice.lower())
        elif voice.startswith('am_'):
            return (3, voice.lower())
        else:
            return (4, voice.lower())
    return sorted(voices, key=sort_key)

def save_setting(setting, value):
    """
    Save a setting to settings.yaml and update shared.args (only in non-multi-user mode).
    """
    if getattr(shared.args, 'multi_user', False):
        log(f"Skipping save_setting for {setting} in multi-user mode")
        return
    settings_file = pathlib.Path(__file__).parent / 'settings.yaml'
    settings = {}
    try:
        if settings_file.exists():
            with open(settings_file, 'r') as f:
                settings = yaml.safe_load(f) or {}
        settings[f"kokoro_{setting}"] = value
        with open(settings_file, 'w') as f:
            yaml.safe_dump(settings, f)
        log(f"Settings saved to {settings_file}: {setting}={value}")
    except Exception as e:
        log(f"Error saving setting {setting}: {e}")
    setattr(shared.args, f"kokoro_{setting}", value)

def save_voice(voice):
    """
    Save the selected voice name.
    """
    log(f"Saving voice: {voice}")
    save_setting("voice", voice)

def save_preview_text(text):
    """
    Save the preview text to settings.yaml and update the template state.
    """
    log(f"Saving preview text: {text}")
    if not text or text.isspace():
        text = generate_default_preview_text()
    save_setting("preview_text", text)
    return text

def on_experimental_mode_change(is_enabled):
    """
    Handle changes to the experimental mode setting.
    """
    makehtml = importlib.reload(importlib.import_module('extensions.KokoroTTS_4_TGUI.src.makehtml'))
    splittext = importlib.reload(importlib.import_module('extensions.KokoroTTS_4_TGUI.src.splittext'))
    generate = importlib.reload(importlib.import_module('extensions.KokoroTTS_4_TGUI.src.generate'))

    log(f"Experimental mode changed: {is_enabled}")
    save_setting("experimental", is_enabled)
    return f"{OurName} experimental mode {'enabled' if is_enabled else 'disabled'}."

def on_voice_change(selected_voice, template_state):
    """
    Handle voice selection changes, save the voice, and load it.
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
    Load settings from settings.yaml (non-multi-user) or set defaults.
    """
    log("Entering load_settings")
    settings_file = pathlib.Path(__file__).parent / 'settings.yaml'
    settings = {}
    is_multi_user = getattr(shared.args, 'multi_user', False)
    log(f"Multi-user mode: {is_multi_user}")
    if not is_multi_user:
        try:
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    settings = yaml.safe_load(f) or {}
                log(f"Settings loaded from {settings_file}: {settings}")
        except Exception as e:
            log(f"Error loading settings from {settings_file}: {e}")

    default_enable_tts = bool(settings.get('kokoro_enable_tts', settings.get('autoplay', False)))
    default_speed = float(settings.get('kokoro_speed', 1.0))
    default_preview_text = settings.get('kokoro_preview_text', generate_default_preview_text())
    default_experimental = bool(settings.get('kokoro_experimental', False))
    default_preprocess_code = bool(settings.get('kokoro_preprocess_code', False))
    default_debug_mode = bool(settings.get('kokoro_enable_debug', True))  # Add this line
    default_voice = settings.get('kokoro_voice', VOICES[0])
    # Add icon style setting
    default_icon_style = settings.get('kokoro_icon_style', DEFAULT_ICON_STYLE)

    if default_voice not in VOICES:
        default_voice = VOICES[0]



    setattr(shared.args, 'kokoro_speed', default_speed)
    setattr(shared.args, 'kokoro_enable_tts', default_enable_tts)
    setattr(shared.args, 'kokoro_preview_text', default_preview_text)
    setattr(shared.args, 'kokoro_experimental', default_experimental)
    setattr(shared.args, 'kokoro_preprocess_code', default_preprocess_code)
    setattr(shared.args, 'kokoro_enable_debug', default_debug_mode)
    setattr(shared.args, 'kokoro_voice', default_voice)
    setattr(shared.args, 'kokoro_icon_style', default_icon_style)

    log(f"Settings applied: speed={default_speed}, enable_tts={default_enable_tts}, preview_text={default_preview_text}, experimental={default_experimental}, preprocess_code={default_preprocess_code}, debug_mode={default_debug_mode}, voice={default_voice}")  # Update this line


def output_modifier(string, state):
    """
    Generate and automatically play TTS audio.
    """
    # Check if TTS is enabled
    if not getattr(shared.args, 'kokoro_enable_tts', False):
        log("TTS disabled, skipping audio generation")
        return string

    log("TTS enabled, generating audio")
    log(f"Original string length: {len(string)}")

    # Clean the text for TTS
    string_for_tts = clean_text_for_tts(string)
    log(f"Cleaned string: '{string_for_tts[:100]}...'")

    if not string_for_tts.strip():
        log("No text to speak after cleaning")
        return string

    # Check experimental mode
    experimental_mode = getattr(shared.args, 'kokoro_experimental', False)
    log(f"Experimental mode: {experimental_mode}")

    if experimental_mode:
        # Experimental: Use chunking for better responsiveness
        log("Taking experimental chunked path")
        return output_chunked(string, string_for_tts)
    else:
        # Regular: Use standard generate.run() like original
        log("Taking standard output path")
        return output_standard(string, string_for_tts)

def output_chunked(original_string, clean_text):
    """
    Experimental chunked output mode - generates multiple audio files and uses make_js_autoplay.
    """
    log("Using experimental chunked output mode")

    # Import makehtml fresh each time for hot-reloading
    makehtml = importlib.reload(importlib.import_module('extensions.KokoroTTS_4_TGUI.src.makehtml'))

    # Use simple_chunk_text (defined in this file) instead of splittext version
    chunks = simple_chunk_text(clean_text)

    if not chunks:
        return output_standard(original_string, clean_text)

    try:
        output_dir = pathlib.Path(__file__).parent / 'audio'
        os.makedirs(output_dir, exist_ok=True)

        audio_urls = []
        for i, chunk in enumerate(chunks):
            # Create unique filename for each chunk
            chunk_filename = f'kokoro_chunk_{int(time.time() * 1000)}_{i}.wav'
            audio_path = output_dir / chunk_filename

            try:
                msg_id = generate.run(chunk, output_path=audio_path)
                if os.path.exists(audio_path):
                    # Format URL properly for web interface
                    file_url = f"file/{audio_path.as_posix()}"
                    audio_urls.append(file_url)
                    log(f"Generated chunk {i}: {file_url}")
            except Exception as e:
                log(f"Error generating chunk {i}: {e}")
                continue

        if not audio_urls:
            log("No chunks generated successfully, falling back to standard mode")
            return output_standard(original_string, clean_text)

        # Get speed for client-side playback
        speed = getattr(shared.args, 'kokoro_speed', 1.0)

        # Use make_js_autoplay for chunked audio - remove the extra 'text' parameter
        audio_html = makehtml.make_js_autoplay(audio_urls, speed)

        return original_string + audio_html

    except Exception as e:
        log(f"Error in chunked output: {e}")
        return output_standard(original_string, clean_text)


def simple_chunk_text(text):
    """
    Simple text chunking for experimental mode.
    """
    import re
    chunks = re.split(r'([.!?]+\s+)', text)
    result = []

    for i in range(0, len(chunks), 2):
        sentence = chunks[i].strip() if i < len(chunks) else ""
        punct = chunks[i+1] if i+1 < len(chunks) else ""

        if sentence and len(sentence) > 5:
            result.append(sentence + punct)

    return result


def output_standard(original_string, clean_text):
    """
    Standard output mode - uses hidden audio div instead of inline buttons.
    """
    log("=== ENTERING output_standard ===")
    # setattr(shared, 'kokoro_current_output', clean_text)
    # return original_string
    # pass

    # Import makehtml fresh each time for hot-reloading
    try:
        makehtml = importlib.reload(importlib.import_module('extensions.KokoroTTS_4_TGUI.src.makehtml'))
    except Exception as e:
        log(f"Error reloading makehtml: {e}")
        return original_string

    # Check if voice is loaded
    voice_name = getattr(generate, 'voice_name', None)
    if voice_name is None:
        default_voice = getattr(shared.args, 'kokoro_voice', 'bf_emma')
        generate.load_voice(default_voice)

    try:
        # Generate audio
        msg_id = generate.run(clean_text)
        if msg_id is None:
            log("ERROR: generate.run() returned None!")
            return original_string

        # Create file URL
        file_url = f"/file/extensions/{OurName}/audio/{msg_id}.wav"
        speed = getattr(shared.args, 'kokoro_speed', 1.0)

        # setattr(shared, 'kokoro_current_audio_file', file_url)

        # Create hidden audio HTML - this should go to the hidden div
        audio_html = makehtml.create_ai_audio_html(file_url, speed, clean_text[:50])
        log(f"Generated audio HTML: {audio_html[:200]}...")  # Show first 200 chars

        # Store the audio HTML globally so the button can access it
        setattr(shared, 'kokoro_current_audio_html', audio_html)

        log(f"Generated audio HTML for hidden div: {len(audio_html)} chars")

        # Return just the original text - no HTML additions to chat
        return original_string

    except Exception as e:
        log(f"Error in standard output: {e}")
        return original_string

def voice_preview(preview_text):
    """
    Generate voice preview with hidden audio and clickable speaker button.
    """
    log("Generating voice preview")
    voice_name = getattr(shared.args, 'kokoro_voice', VOICES[0])
    substituted_text = substitute_placeholders(preview_text, voice_name)

    # Always use standard mode for preview
    clean_text = clean_text_for_tts(substituted_text)
    makehtml = importlib.reload(importlib.import_module('extensions.KokoroTTS_4_TGUI.src.makehtml'))

    try:
        msg_id = generate.run(clean_text, preview=True)
        if msg_id:
            file_url = f"/file/extensions/{OurName}/audio/preview.wav"
            speed = getattr(shared.args, 'kokoro_speed', 1.0)

            play_html = makehtml.create_speaker_button_html(file_url, speed, preview_text)
            return f"Preview: {substituted_text} {play_html }"
    except Exception as e:
        log(f"Error generating preview: {e}")
        return f"Preview: {substituted_text} (Audio generation failed)"

    return f"Preview: {substituted_text}"

def ui():
    """
    Create the Gradio UI for the KokoroTTS extension.
    """
    log("Creating UI")

    setattr(shared.args, 'kokoro_enable_debug', True)  # change this to false for release

    load_settings()
    default_voice = getattr(shared.args, 'kokoro_voice')
    default_speed = getattr(shared.args, 'kokoro_speed')
    default_enable_tts = getattr(shared.args, 'kokoro_enable_tts')
    default_preview_text = getattr(shared.args, 'kokoro_preview_text')
    default_experimental = getattr(shared.args, 'kokoro_experimental')
    default_preprocess_code = getattr(shared.args, 'kokoro_preprocess_code')
    default_debug_mode = getattr(shared.args, 'kokoro_enable_debug')  # Add this line

    is_multi_user = getattr(shared.args, 'multi_user', False)
    generate.load_voice(default_voice)

    info_voice = (
        "Select a Voice.\nThe default voice is a 50-50 mix of Bella & Sarah.\nVoices starting with 'a' are American "
        "english, voices with 'b' are British english"
    )

    with gr.Accordion(f"{OurName}"):
        sorted_voices = sort_voices(VOICES)


        with gr.Row():
        # player_output = gr.HTML()
            hidden_audio_div = gr.HTML(
                    value="Player Audio Code Goes Here",
                    elem_id="kokoro-hidden-audio",
                    visible=False #default_enable_tts
                )
            # audio_control_btn = gr.Button(
            #     "Speak",
            #     elem_id="kokoro-audio-control",
            #     elem_classes="custom-button",
            #     variant="primary",
            #     visible=False , #default_enable_tts,
            #     interactive=False  # Start disabled when no audio
            # )
            audio_control_btn = gr.Button(
                "Speak",
                elem_id="kokoro-audio-control",
                visible=True,  # Keep visible for cloning
                elem_classes=["lg", "primary", "svelte-cmf5ev", "kokoro-hidden-original"]  # Add CSS class
            )


        voice = gr.Dropdown(choices=sorted_voices, value=default_voice, label="Voice", info=info_voice, interactive=True)
        template_state = gr.State(value=default_preview_text)

        if not is_multi_user:
            preview_text = gr.Textbox(
                value=default_preview_text,
                label="Preview text",
                info="Text to use for voice preview. Use {VNAME} for the voice name and {VREGION} for the region.",
                interactive=True
            )

        preview = gr.Button("Voice preview", variant="secondary")
        preview_output = gr.HTML()

        speed = gr.Slider(minimum=0.25, maximum=2.0, step=0.1, value=default_speed, label="Voice speed", info="Adjust voice playback speed", interactive=True)

        enable_tts = gr.Checkbox(
            label="Enable TTS",
            value=default_enable_tts,
            info="Enable to automatically generate and play TTS audio",
            interactive=True
        )

        if not is_multi_user:
            # with gr.Row():

            splitting_method = gr.Radio(["Split by sentence", "Split by Word"],
                                      info="Kokoro only supports 510 tokens. Split method for long text.",
                                      value="Split by sentence", label="Splitting method", interactive=True)
            experimental_mode = gr.Checkbox(
                label="Enable experimental mode (chunked playback)",
                value=default_experimental,
                info="Experimental chunked audio for more responsive playback",
                interactive=True
            )
            preprocess_code = gr.Checkbox(
                label="Preprocess code blocks and markdown",
                value=default_preprocess_code,
                info="Convert code blocks and markdown to speech-friendly text",
                interactive=True
            )
            debug_mode = gr.Checkbox(  # Add this
                label="Enable debug mode",
                value=default_debug_mode,
                info="Show detailed status messages and enhanced logging",
                interactive=True
            )
            status_textbox = gr.Textbox(label="Status", interactive=False)

    # Event handlers
    voice.change(on_voice_change, inputs=[voice, template_state], outputs=[voice, template_state])
    speed.change(lambda x: save_setting("speed", x), inputs=[speed])

    # Audio control event handler should be same for both single and multi user
    def handle_audio_button_click():
        # Return the stored audio HTML to update the hidden div
        audio_html = getattr(shared, 'kokoro_current_audio_html', '')
        log(f"Button clicked, returning audio HTML: {len(audio_html)} chars")
        return audio_html

    audio_control_btn.click(handle_audio_button_click, outputs=[hidden_audio_div])



    if is_multi_user:
        enable_tts.change(lambda x: save_setting("enable_tts", x), inputs=[enable_tts])
        preview.click(voice_preview, inputs=[gr.State(generate_default_preview_text())], outputs=preview_output)
    else:
        # Handle both saving and button visibility for non-multi-user
        def handle_tts_toggle(x):
            save_setting("enable_tts", x)
            return gr.Button(visible=x)

        enable_tts.change(handle_tts_toggle, inputs=[enable_tts], outputs=[audio_control_btn])

        preview_text.change(save_preview_text, inputs=[preview_text], outputs=[template_state])
        preview.click(voice_preview, inputs=[preview_text], outputs=preview_output)
        splitting_method.change(generate.set_splitting_type, inputs=[splitting_method])
        experimental_mode.change(on_experimental_mode_change, inputs=[experimental_mode], outputs=[status_textbox])
        preprocess_code.change(lambda x: save_setting("preprocess_code", x), inputs=[preprocess_code])
        debug_mode.change(lambda x: save_setting("debug_mode", x), inputs=[debug_mode])


        # Update audio button visibility when TTS is toggled
        enable_tts.change(
            lambda x: [save_setting("enable_tts", x), gr.Button(visible=x)],
            inputs=[enable_tts],
            outputs=[audio_control_btn]
        )
         # Event chaining - automatically load stored audio when AI responds

        def load_stored_audio_1():
            """Load stored audio HTML when display updates"""
            audio_html = getattr(shared, 'kokoro_current_audio_html', '')
            if audio_html:
                log(f"Auto-loading stored audio into hidden div: {len(audio_html)} chars")
                # Return audio HTML + button set to "Pause" (since audio will autoplay)
                return audio_html, gr.Button(value="Pause", interactive=True)
            else:
                log("No stored audio found for auto-loading")
                return "", gr.Button(value="Speak", interactive=False)

        def load_stored_audio_3():
            # Button goes from disbled to enabled but not to pause
            audio_html = getattr(shared, 'kokoro_current_audio_html', '')
            if audio_html:
                # Set button to show "Pause" since audio will auto-play
                button_update = gr.Button.update(value="Pause", interactive=True)
                return audio_html, button_update
            else:
                button_update = gr.Button.update(value="Speak", interactive=False)
                return "", button_update

        # # Assuming hidden_audio_div and speak_button are Gradio components
        # hidden_audio_div = shared.gradio['html-default']
        # audio_control_btn = shared.gradio['audio_control_btn']

        # Chain from the AI response display component update
        if 'display' in shared.gradio:
            shared.gradio['display'].change(
                load_stored_audio_1,
                outputs=[hidden_audio_div, audio_control_btn]
            )
            log("Event chaining established: display -> hidden_audio_div + audio_control_btn (speak button)")
        else:
            log("Warning: shared.gradio['display'] not available for chaining")


        # hidden_audio_div.value = makehtml.create_ai_audio_html("", default_speed, preview_text)


def custom_js():
    """
    Returns custom javascript as a string. It is applied whenever the web UI is loaded.
    This manages audio playback toggle, button cloning, synced state, tooltips, etc.
    """
    return makehtml.create_ai_audio_js()
