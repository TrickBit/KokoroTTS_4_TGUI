import pathlib
import html
import time
import subprocess
from pydub import AudioSegment
import yaml
from extensions.KokoroTTS_4_TGUI.src.generate import run, load_voice, set_plitting_type
from extensions.KokoroTTS_4_TGUI.src.voices import VOICES
import gradio as gr
import time
from modules import shared

def input_modifier(string, state):
    shared.processing_message = "*Is recording a voice message...*"
    return string

def voice_update(voice):
    load_voice(voice)
    # Strip bf/af/bm/am prefix and capitalize voice name
    voice_name = voice[3:].capitalize() if voice.startswith(('bf_', 'af_', 'bm_', 'am_')) else voice.capitalize()
    # Set region based on first letter
    region = "England" if voice.startswith(('bf_', 'bm_')) else "America" if voice.startswith(('af_', 'am_')) else ""
    default_preview_text = f"Hello, I'm {voice_name} from {region}, how are you today?" if region else f"Hello, I'm {voice_name}, how are you today?"
    return [
        gr.Dropdown(choices=VOICES, value=voice, label="Voice", info="Select Voice", interactive=True),
        gr.Textbox(value=default_preview_text, label="Preview text", info="Text to use for voice preview")
    ]

def adjust_audio_speed(input_path, output_path, speed):
    """Adjust audio speed using pydub, preserving pitch."""
    audio = AudioSegment.from_file(input_path)
    # Change speed (pydub's speed change also adjusts duration)
    adjusted = audio.speedup(playback_speed=speed) if speed >= 1.0 else audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * speed)})
    adjusted.export(output_path, format="wav")

def voice_preview(preview_text):
    # Generate original audio
    run(preview_text, preview=True)
    audio_dir = pathlib.Path(__file__).parent / 'audio' / 'preview.wav'
    speed = getattr(shared.args, 'kokoro_speed', 1.0)
    if speed != 1.0:
        # Create temporary file for adjusted audio
        adjusted_audio_dir = pathlib.Path(__file__).parent / 'audio' / 'preview_adjusted.wav'
        adjust_audio_speed(audio_dir, adjusted_audio_dir, speed)
        audio_url = f'{adjusted_audio_dir.as_posix()}?v=f{int(time.time())}'
    else:
        audio_url = f'{audio_dir.as_posix()}?v=f{int(time.time())}'
    return f'<audio controls autoplay playbackRate="{speed}"><source src="file/{audio_url}" type="audio/mpeg"></audio>'

def save_speed(speed):
    settings_file = pathlib.Path(__file__).parent.parent.parent / 'user_data' / 'settings.yaml'
    settings = {}
    if settings_file.exists():
        with open(settings_file, 'r') as f:
            settings = yaml.safe_load(f) or {}
    settings['kokoro_speed'] = speed
    with open(settings_file, 'w') as f:
        yaml.safe_dump(settings, f)
    setattr(shared.args, 'kokoro_speed', speed)

def save_preview_text(text):
    settings_file = pathlib.Path(__file__).parent.parent.parent / 'user_data' / 'settings.yaml'
    settings = {}
    if settings_file.exists():
        with open(settings_file, 'r') as f:
            settings = yaml.safe_load(f) or {}
    settings['kokoro_preview_text'] = text
    with open(settings_file, 'w') as f:
        yaml.safe_dump(settings, f)

def ui():
    settings_file = pathlib.Path(__file__).parent.parent.parent / 'user_data' / 'settings.yaml'
    settings = {}
    if settings_file.exists():
        with open(settings_file, 'r') as f:
            settings = yaml.safe_load(f) or {}
    default_speed = settings.get('kokoro_speed', 1.0)
    default_autoplay = settings.get('kokoro_autoplay', False)

    info_voice = """Select a Voice. \nThe default voice is a 50-50 mix of Bella & Sarah\nVoices starting with 'a' are American
     english, voices with 'b' are British english"""
    with gr.Accordion("Kokoro"):
        voice = gr.Dropdown(choices=VOICES, value=VOICES[0], label="Voice", info=info_voice, interactive=True)
        # Initialize default preview text based on initial voice
        initial_voice_name = VOICES[0][3:].capitalize() if VOICES[0].startswith(('bf_', 'af_', 'bm_', 'am_')) else VOICES[0].capitalize()
        initial_region = "England" if VOICES[0].startswith(('bf_', 'bm_')) else "America" if VOICES[0].startswith(('af_', 'am_')) else ""
        initial_preview_text = f"Hello, I'm {initial_voice_name} from {initial_region}, how are you today?" if initial_region else f"Hello, I'm {initial_voice_name}, how are you today?"
        preview_text = gr.Textbox(value=initial_preview_text, label="Preview text", info="Text to use for voice preview", interactive=True)
        preview = gr.Button("Voice preview", type="secondary")
        speed = gr.Slider(minimum=0.25, maximum=2.0, step=0.1, value=default_speed, label="Voice speed", info="Adjust voice playback speed (0.25x to 2.0x)", interactive=True)
        preview_output = gr.HTML()
        info_splitting = """Kokoro only supports 510 tokens. One method to split the text is by sentence (default), the other way
        is by word up to 510 tokens."""
        spltting_method = gr.Radio(["Split by sentence", "Split by Word"], info=info_splitting, value="Split by sentence", label_lines=2, interactive=True)
        autoplay = gr.Checkbox(label="Autoplay generated audio", value=default_autoplay, info="Enable to automatically play audio in the browser when generated", interactive=True)


    voice.change(voice_update, inputs=voice, outputs=[voice, preview_text])
    preview.click(fn=voice_preview, inputs=preview_text, outputs=preview_output)
    spltting_method.change(set_plitting_type, spltting_method)
    autoplay.change(lambda x: setattr(shared.args, 'kokoro_autoplay', x), autoplay)
    speed.change(save_speed, speed)

def output_modifier(string, state):
    string_for_tts = html.unescape(string)
    string_for_tts = string_for_tts.replace('*', '')
    string_for_tts = string_for_tts.replace('`', '')
    msg_id = run(string_for_tts)
    audio_dir = pathlib.Path(__file__).parent / 'audio' / f'{msg_id}.wav'
    speed = getattr(shared.args, 'kokoro_speed', 1.0)
    if speed != 1.0:
        adjusted_audio_dir = pathlib.Path(__file__).parent / 'audio' / f'{msg_id}_adjusted.wav'
        adjust_audio_speed(audio_dir, adjusted_audio_dir, speed)
        audio_url = adjusted_audio_dir.as_posix()
    else:
        audio_url = audio_dir.as_posix()
    autoplay_attr = " autoplay" if getattr(shared.args, 'kokoro_autoplay', False) else ""
    string += f'<audio controls{autoplay_attr} playbackRate="{speed}"><source src="file/{audio_url}" type="audio/mpeg"></audio>'

    # Optional: Play audio server-side with aplay (not recommended for web interface)
    # try:
    #     subprocess.run(["aplay", audio_url], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # except (subprocess.CalledProcessError, FileNotFoundError) as e:
    #     print(f"Warning: Failed to play audio with aplay: {e}")

    return string
