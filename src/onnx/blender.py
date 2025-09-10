"""
ONNX Voice Blending Module

Handles voice embedding arithmetic and blending operations for ONNX backend.
Only loaded when ONNX backend is active.
"""

import gradio as gr
import numpy as np
from modules import shared
from extensions.KokoroTTS_4_TGUI.src.debug import *

def blend_voices(primary_voice, secondary_voice, blend_ratio=0.6):
    """
    Blend two voice embeddings using linear interpolation.

    Args:
        primary_voice (np.array): Primary voice embedding
        secondary_voice (np.array): Secondary voice embedding
        blend_ratio (float): 0.0-1.0, weight of primary voice

    Returns:
        np.array: Blended voice embedding
    """
    try:
        # Ensure both voices have same shape
        if primary_voice.shape != secondary_voice.shape:
            log(f"Voice shape mismatch: {primary_voice.shape} vs {secondary_voice.shape}")
            return primary_voice

        # Simple linear interpolation
        blended = primary_voice * blend_ratio + secondary_voice * (1 - blend_ratio)

        log(f"Blended voices: {blend_ratio:.1%} primary + {100-blend_ratio:.1%} secondary")
        log(f"Blended voice shape: {blended.shape}, range: {np.min(blended):.3f} to {np.max(blended):.3f}")

        return blended.astype(np.float32)

    except Exception as e:
        log(f"Error blending voices: {e}")
        return primary_voice


def create_blended_voice_embedding(primary_voice_name, secondary_voice_name, blend_ratio):
    """Create a blended voice embedding from two voice names"""
    try:
        # Import the voices data
        from extensions.KokoroTTS_4_TGUI.src.onnx.generate import VOICES_DATA

        if primary_voice_name not in VOICES_DATA:
            log(f"Primary voice {primary_voice_name} not found")
            return None

        if secondary_voice_name not in VOICES_DATA:
            log(f"Secondary voice {secondary_voice_name} not found")
            return None

        primary_embedding = VOICES_DATA[primary_voice_name]
        secondary_embedding = VOICES_DATA[secondary_voice_name]

        blended_embedding = blend_voices(primary_embedding, secondary_embedding, blend_ratio)

        return blended_embedding

    except Exception as e:
        log(f"Error creating blended voice: {e}")
        return None

def get_blend_configuration(mode, voices, weights):
    """Create a standardized blend configuration object"""
    return {
        'mode': mode,
        'voices': voices,
        'weights': weights,
        'timestamp': time.time()
    }

def apply_blend_configuration(config):
    """Apply a saved blend configuration to create blended voice"""
    # Stub for future implementation
    log(f"Applying blend config: {config['mode']} with {len(config['voices'])} voices")
    return None

def save_blend_to_file(config, name):
    """Save blend configuration to file for later loading"""
    # Stub for future implementation
    log(f"Saving blend '{name}' to file")
    return True

def load_saved_blends():
    """Load all saved blend configurations"""
    # Stub for future implementation
    return []



def create_blending_ui(sorted_display_names):
    """Create the voice blending UI accordion with working preview"""

    with gr.Accordion("Voice Blending (Experimental)", open=False):
        gr.Markdown("Create custom voices by blending characteristics from multiple speakers.")

        with gr.Row():
            blend_enable = gr.Checkbox(
                label="Enable Voice Blending",
                value=False,
                info="Mix two voices to create unique characteristics"
            )

        with gr.Group(visible=False) as blend_controls:
            with gr.Row():
                primary_voice = gr.Dropdown(
                    choices=sorted_display_names,
                    label="Primary Voice (Base)",
                    info="Main voice characteristics"
                )
                secondary_voice = gr.Dropdown(
                    choices=sorted_display_names,
                    label="Secondary Voice (Blend)",
                    info="Voice to blend in"
                )

            blend_ratio = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                step=0.1,
                value=0.6,
                label="Blend Ratio",
                info="0.1 = mostly secondary, 0.9 = mostly primary"
            )

            blend_preview_btn = gr.Button("Preview Blend", variant="secondary")
            blend_preview_output = gr.HTML()

        # Event handlers
        blend_enable.change(
            lambda x: gr.Group(visible=x),
            inputs=[blend_enable],
            outputs=[blend_controls]
        )

        # Working preview handler
        def handle_blend_preview(primary_display, secondary_display, ratio):
            """Generate actual audio preview of blended voice"""
            try:
                # Get voice mapping (we need access to the mapping from script.py)
                from extensions.KokoroTTS_4_TGUI.script import get_voice_display_mapping
                current_mapping = get_voice_display_mapping()

                primary_voice_id = current_mapping.get(primary_display)
                secondary_voice_id = current_mapping.get(secondary_display)

                if not primary_voice_id or not secondary_voice_id:
                    return "Error: Could not find voice IDs for selected voices"

                # Create blended voice embedding
                blended_embedding = create_blended_voice_embedding(primary_voice_id, secondary_voice_id, ratio)

                if blended_embedding is None:
                    return "Error: Could not create blended voice"

                # Generate preview text and audio
                preview_text = "Hello, this is a preview of the blended voice."

                # Process text to tokens
                from extensions.KokoroTTS_4_TGUI.src.onnx.phoneme import process_text_for_onnx
                tokens, phonemes, normalized = process_text_for_onnx(preview_text, lang='a')

                # Generate audio with blended voice
                from extensions.KokoroTTS_4_TGUI.src.onnx.generate import generate_onnx_audio
                audio_output = generate_onnx_audio(tokens, blended_embedding)

                if audio_output is None:
                    return "Error: Could not generate blended audio"

                # Save blend preview audio
                from pydub import AudioSegment
                import time

                audio_output = audio_output / np.max(np.abs(audio_output)) if np.max(np.abs(audio_output)) > 0 else audio_output
                audio_int16 = np.int16(audio_output * 32767)

                segment = AudioSegment(
                    data=audio_int16.tobytes(),
                    sample_width=2,
                    frame_rate=24000,
                    channels=1
                )

                # Save to unique blend preview file
                extension_name = getattr(shared.args, 'kokoro_extension_name')
                audio_dir = getattr(shared.args, f"kokoro_homedir") / 'audio'
                audio_dir.mkdir(exist_ok=True)

                blend_filename = f'blend_preview_{int(time.time())}.wav'
                blend_path = audio_dir / blend_filename
                segment.export(blend_path, format="wav")

                # Create speaker button HTML for blend preview
                from extensions.KokoroTTS_4_TGUI.src import makehtml
                file_url = f"/file/extensions/{extension_name}/audio/{blend_filename}"
                play_html = makehtml.create_speaker_button_html(file_url, 1.0, preview_text)

                return f"Blend Preview: {ratio:.1%} {primary_display} + {100-ratio:.1%} {secondary_display} {play_html}"

            except Exception as e:
                log(f"Error in blend preview: {e}")
                return f"Error generating blend preview: {e}"

        blend_preview_btn.click(
            handle_blend_preview,
            inputs=[primary_voice, secondary_voice, blend_ratio],
            outputs=[blend_preview_output]
        )
