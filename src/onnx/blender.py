"""
ONNX Voice Blending Module

Handles voice embedding arithmetic and blending operations for ONNX backend.
Only loaded when ONNX backend is active.
"""

def adjust_tgui_path():
    """
    Adjust sys.path to include the TGUI root directory (containing 'extensions/' and 'modules/').
    Call this before any imports to ensure consistent import paths in TGUI or standalone mode.
    """
    import os
    import sys

    start_dir = os.path.dirname(os.path.abspath(__file__))
    current = start_dir
    while current != os.path.dirname(current):  # Stop at filesystem root
        if (os.path.exists(os.path.join(current, 'extensions')) and
            os.path.exists(os.path.join(current, 'modules'))):
            if current not in sys.path:
                sys.path.insert(0, current)  # Prepend to prioritize
            return
        current = os.path.dirname(current)
    raise RuntimeError("Could not find TGUI root containing both 'extensions/' and 'modules/'")

# Call before any extension imports
adjust_tgui_path()

import gradio as gr
import numpy as np
import time
import pathlib
from extensions.KokoroTTS_4_TGUI.src.kshared import kruntime
from extensions.KokoroTTS_4_TGUI.src.debug import error_log, info_log, debug_log


def blend_voices(primary_voice, secondary_voice, blend_ratio=0.6):
    """
    Purpose: Blend two voice embeddings using linear interpolation
    Pre: Valid voice embedding arrays with matching shapes
    Post: Blended voice embedding created using specified ratio
    Args: primary_voice (np.array) - Primary voice embedding
          secondary_voice (np.array) - Secondary voice embedding
          blend_ratio (float) - 0.0-1.0, weight of primary voice (0.6 = 60% primary)
    Returns: np.array - Blended voice embedding as float32 array
    """
    try:
        # Ensure both voices have same shape
        if primary_voice.shape != secondary_voice.shape:
            error_log(f"Voice shape mismatch: {primary_voice.shape} vs {secondary_voice.shape}")
            return primary_voice

        # Simple linear interpolation
        blended = primary_voice * blend_ratio + secondary_voice * (1 - blend_ratio)

        info_log(f"Blended voices: {blend_ratio:.1%} primary + {(1-blend_ratio):.1%} secondary")
        debug_log(f"Blended voice shape: {blended.shape}, range: {np.min(blended):.3f} to {np.max(blended):.3f}")

        return blended.astype(np.float32)

    except Exception as e:
        error_log(f"Error blending voices: {e}")
        return primary_voice


def create_blended_voice_embedding(primary_voice_name, secondary_voice_name, blend_ratio):
    """
    Purpose: Create a blended voice embedding from two voice names
    Pre: Voice names exist in voices data
    Post: Blended embedding created from specified voices
    Args: primary_voice_name (str) - Name of primary voice
          secondary_voice_name (str) - Name of secondary voice
          blend_ratio (float) - Blend ratio for interpolation
    Returns: np.array - Blended voice embedding, or None if voices not found
    """
    try:
        voices_data = kruntime.get('onnx_voices_data', {})

        if primary_voice_name not in voices_data:
            error_log(f"Primary voice {primary_voice_name} not found")
            return None

        if secondary_voice_name not in voices_data:
            error_log(f"Secondary voice {secondary_voice_name} not found")
            return None

        primary_embedding = voices_data[primary_voice_name]
        secondary_embedding = voices_data[secondary_voice_name]

        blended_embedding = blend_voices(primary_embedding, secondary_embedding, blend_ratio)

        return blended_embedding

    except Exception as e:
        error_log(f"Error creating blended voice: {e}")
        return None


def get_blend_configuration(mode, voices, weights):
    """
    Purpose: Create a standardized blend configuration object for saving/loading
    Pre: Valid mode string and voice/weight data
    Post: Configuration object created with timestamp
    Args: mode (str) - Blending mode identifier
          voices (list) - List of voice names involved in blend
          weights (list) - List of corresponding blend weights
    Returns: dict - Standardized configuration object with metadata
    """
    return {
        'mode': mode,
        'voices': voices,
        'weights': weights,
        'timestamp': time.time()
    }


def apply_blend_configuration(config):
    """
    Purpose: Apply a saved blend configuration to create blended voice
    Pre: Valid configuration object with required fields
    Post: Blended voice created according to saved configuration
    Args: config (dict) - Blend configuration object
    Returns: np.array - Blended voice embedding, or None if failed
    """
    # Stub for future implementation
    info_log(f"Applying blend config: {config['mode']} with {len(config['voices'])} voices")
    return None


def save_blend_to_file(config, name):
    """
    Purpose: Save blend configuration to file for later loading
    Pre: Valid configuration object and save name
    Post: Configuration saved to persistent storage
    Args: config (dict) - Blend configuration to save
          name (str) - Name for saved configuration
    Returns: bool - True if save successful, False otherwise
    """
    # Stub for future implementation
    info_log(f"Saving blend '{name}' to file")
    return True


def load_saved_blends():
    """
    Purpose: Load all saved blend configurations from storage
    Pre: Blend storage system available
    Post: All saved configurations loaded into memory
    Args: None
    Returns: list - List of saved blend configuration objects
    """
    # Stub for future implementation
    return []


def create_blending_ui(sorted_display_names):
    """
    Purpose: Create the voice blending UI accordion with working preview and speed control
    Pre: Sorted list of display names available
    Post: Complete blending UI created with all controls and event handlers
    Args: sorted_display_names (list) - List of voice display names for dropdowns
    Returns: None - UI components created in Gradio context
    """
    extension_name = kruntime.get('extension_name', 'KokoroTTS_4_TGUI')
    voice_blending_title = f"{extension_name} - Voice Blending"

    with gr.Accordion(f"{voice_blending_title}", open=False):
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
                    info="Main voice characteristics",
                    value=None
                )
                secondary_voice = gr.Dropdown(
                    choices=sorted_display_names,
                    label="Secondary Voice (Blend)",
                    info="Voice to blend in",
                    value=None
                )

            blend_ratio = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.6,
                label="Primary Voice Weight",
                info="0.0 = pure primary voice, 1.0 = pure secondary voice"
            )

            # Speed control for blended voice preview
            blend_speed = gr.Slider(
                minimum=0.5,
                maximum=1.5,
                step=0.1,
                value=1.0,
                label="Preview Speed",
                info="Playback speed for blended voice preview"
            )

            blend_preview_btn = gr.Button("Preview Blend", variant="secondary")
            blend_preview_output = gr.HTML()

    # Dynamic filtering functions
        def update_secondary_choices(primary_selection):
            """
            Purpose: Filter secondary dropdown to exclude selected primary voice
            Pre: Primary voice selection made or cleared
            Post: Secondary dropdown updated with filtered choices, preserving existing selection
            Args: primary_selection (str) - Currently selected primary voice
            Returns: gr.Dropdown - Updated secondary dropdown without primary selection
            """
            if primary_selection:
                filtered_choices = [choice for choice in sorted_display_names if choice != primary_selection]
                return gr.Dropdown(choices=filtered_choices)
            return gr.Dropdown(choices=sorted_display_names)

        def update_primary_choices(secondary_selection):
            """
            Purpose: Filter primary dropdown to exclude selected secondary voice
            Pre: Secondary voice selection made or cleared
            Post: Primary dropdown updated with filtered choices, preserving existing selection
            Args: secondary_selection (str) - Currently selected secondary voice
            Returns: gr.Dropdown - Updated primary dropdown without secondary selection
            """
            if secondary_selection:
                filtered_choices = [choice for choice in sorted_display_names if choice != secondary_selection]
                return gr.Dropdown(choices=filtered_choices)
            return gr.Dropdown(choices=sorted_display_names)

        # Working preview handler with speed control
        def handle_blend_preview(primary_display, secondary_display, blend_ratio, speed):
            """
            Purpose: Generate actual audio preview of blended voice with speed control
            Pre: Valid voice selections and blend parameters
            Post: Blended audio generated and returned as HTML player
            Args: primary_display (str) - Display name of primary voice
                  secondary_display (str) - Display name of secondary voice
                  blend_ratio (float) - Primary voice blend_ratio (0.0-1.0)
                  speed (float) - Playback speed multiplier
            Returns: str - HTML with audio player for blended voice preview
            """
            try:
                # Validate selections
                if not primary_display or not secondary_display:
                    return "Error: Please select both primary and secondary voices"

                if primary_display == secondary_display:
                    return "Error: Primary and secondary voices cannot be the same"

                # Get voice mapping
                from extensions.KokoroTTS_4_TGUI.src.trickbit import get_voice_display_mapping
                current_mapping = get_voice_display_mapping()

                primary_voice_id = current_mapping.get(primary_display)
                secondary_voice_id = current_mapping.get(secondary_display)

                if not primary_voice_id or not secondary_voice_id:
                    return "Error: Could not find voice IDs for selected voices"

                # Create blended voice embedding
                blended_embedding = create_blended_voice_embedding(primary_voice_id, secondary_voice_id, 1.0 - blend_ratio)

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

                audio_output = audio_output / np.max(np.abs(audio_output)) if np.max(np.abs(audio_output)) > 0 else audio_output
                audio_int16 = np.int16(audio_output * 32767)

                segment = AudioSegment(
                    data=audio_int16.tobytes(),
                    sample_width=2,
                    frame_rate=24000,
                    channels=1
                )

                # Save to unique blend preview file
                extension_name = kruntime.get('extension_name', 'KokoroTTS_4_TGUI')
                audio_dir = kruntime.get('extension_dir') / 'audio'
                audio_dir.mkdir(exist_ok=True)

                blend_filename = f'blend_preview_{int(time.time())}.wav'
                blend_path = audio_dir / blend_filename
                segment.export(blend_path, format="wav")

                # Create speaker button HTML for blend preview with speed control
                from extensions.KokoroTTS_4_TGUI.src import makehtml
                file_url = f"/file/extensions/{extension_name}/audio/{blend_filename}"
                play_html = makehtml.create_speaker_button_html(file_url, speed, preview_text)

                return f"Blend Preview: {(1-blend_ratio):.1%} {primary_display} + {blend_ratio:.1%} {secondary_display} (Speed: {speed}x) {play_html}"

            except Exception as e:
                error_log(f"Error in blend preview: {e}")
                return f"Error generating blend preview: {e}"

        # Event handlers
        blend_enable.change(
            lambda x: gr.Group(visible=x),
            inputs=[blend_enable],
            outputs=[blend_controls]
        )

        # Dynamic filtering event handlers
        primary_voice.change(
            update_secondary_choices,
            inputs=[primary_voice],
            outputs=[secondary_voice]
        )

        secondary_voice.change(
            update_primary_choices,
            inputs=[secondary_voice],
            outputs=[primary_voice]
        )

        # Preview generation
        blend_preview_btn.click(
            handle_blend_preview,
            inputs=[primary_voice, secondary_voice, blend_ratio, blend_speed],
            outputs=[blend_preview_output]
        )
