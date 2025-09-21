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

import gc
import shutil
import uuid
import numpy as np
import pathlib
import time
import requests
from pydub import AudioSegment
import re
import html
import importlib

# Now we can import from extensions with confidence
from extensions.KokoroTTS_4_TGUI.src.kshared import kruntime
from extensions.KokoroTTS_4_TGUI.src.onnx.phoneme import process_text_for_onnx
from extensions.KokoroTTS_4_TGUI.src.onnx.voices import ID2SPEAKER
from extensions.KokoroTTS_4_TGUI.src.debug import error_log, info_log, debug_log
from extensions.KokoroTTS_4_TGUI.src.splittext import simple_sentence_split


def get_model_files():
    """
    Purpose: Get model file paths, downloading if necessary
    Pre: Extension directory accessible
    Post: Model files available for use
    Args: None
    Returns: tuple - (model_path, voices_path) as pathlib.Path objects
    """
    # Only import downloader when we need it
    try:
        from extensions.KokoroTTS_4_TGUI.src.onnx.model_downloader import get_model_paths, download_all_models

        model_path, voices_path, both_exist = get_model_paths()

        if not both_exist:
            info_log("Model files not found, downloading...")
            model_path, voices_path = download_all_models()
        else:
            debug_log("Model files already available")

        return model_path, voices_path

    except ImportError:
        error_log("Model downloader not available")
        raise RuntimeError("Model downloader required for ONNX backend")


def cleanup_old_audio_files(max_files=10, max_age_hours=24):
    """
    Purpose: Clean up old audio files to manage disk usage
    Pre: Extension directory exists
    Post: Old audio files removed, keeping only recent files
    Args: max_files (int) - Maximum number of files to keep
          max_age_hours (int) - Maximum age in hours before deletion
    Returns: None
    """
    try:
        audio_dir = pathlib.Path(kruntime.get('extension_dir')) / 'audio'
        if not audio_dir.exists():
            return

        # Get all .wav files with their modification times
        wav_files = list(audio_dir.glob('*.wav'))
        if len(wav_files) <= max_files:
            return

        # Sort by modification time (newest first)
        wav_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Remove old files beyond max_files limit
        files_to_remove = wav_files[max_files:]
        current_time = time.time()

        for file_path in files_to_remove:
            try:
                # Also check age
                file_age_hours = (current_time - file_path.stat().st_mtime) / 3600
                if file_age_hours > max_age_hours or len(wav_files) > max_files:
                    file_path.unlink()
                    debug_log(f"Cleaned up old audio file: {file_path.name}")
            except Exception as e:
                error_log(f"Error removing file {file_path}: {e}")

    except Exception as e:
        error_log(f"Error in audio cleanup: {e}")


def apply_pitch_shift(audio_data, sample_rate, pitch_factor):
    """
    Purpose: Apply pitch shifting to audio data using numpy (simple approach for ONNX)
    Pre: Valid audio data array and sample rate
    Post: Audio data pitch-shifted by specified factor
    Args: audio_data (np.array) - Audio samples
          sample_rate (int) - Sample rate (24000 for Kokoro)
          pitch_factor (float) - Pitch multiplier (1.0 = normal, 0.5 = octave down, 2.0 = octave up)
    Returns: np.array - Pitch-shifted audio data
    """
    if abs(pitch_factor - 1.0) < 0.01:
        return audio_data  # No change needed

    try:
        # Try to use librosa if available for better quality
        import librosa
        semitones = 12 * np.log2(pitch_factor)
        shifted = librosa.effects.pitch_shift(
            y=audio_data,
            sr=sample_rate,
            n_steps=semitones,
            bins_per_octave=12
        )
        return shifted
    except ImportError:
        # Fallback: simple resampling-based pitch shift (lower quality)
        info_log("librosa not available, using simple pitch shift")
        # Resample to change pitch (crude but functional)
        original_length = len(audio_data)
        new_length = int(original_length / pitch_factor)

        # Simple linear interpolation
        indices = np.linspace(0, original_length - 1, new_length)
        shifted = np.interp(indices, np.arange(original_length), audio_data)

        # Pad or trim to original length
        if len(shifted) < original_length:
            shifted = np.pad(shifted, (0, original_length - len(shifted)), 'constant')
        else:
            shifted = shifted[:original_length]

        return shifted


def initialize_onnx_model():
    """
    Purpose: Initialize ONNX runtime session
    Pre: ONNX model file downloaded
    Post: Model session initialized and stored in runtime
    Args: None
    Returns: ONNX InferenceSession object
    """
    # Check if already initialized
    model_session = kruntime.get('onnx_model_session', None)
    if model_session is not None:
        return model_session

    # Import onnxruntime locally
    import onnxruntime as ort

    if ort is None:
        raise RuntimeError("onnxruntime not available. Install with: pip install onnxruntime")

    try:
        model_path, _ = get_model_files()

        # Create ONNX runtime session
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
            info_log("Using CUDA for ONNX inference")
        else:
            info_log("Using CPU for ONNX inference")

        session = ort.InferenceSession(str(model_path), providers=providers)

        # Store in runtime
        kruntime.set('onnx_model_session', session)
        info_log("ONNX model loaded successfully")

        return session

    except Exception as e:
        error_log(f"Error initializing ONNX model: {e}")
        raise


def load_voices_binary():
    """
    Purpose: Load voices from NPZ file
    Pre: Voices binary file downloaded
    Post: Voice embeddings loaded and stored in runtime
    Args: None
    Returns: dict - Voice embeddings dictionary
    """
    # Check if already loaded
    voices_data = kruntime.get('onnx_voices_data', None)
    if voices_data:
        return voices_data

    try:
        _, voices_path = get_model_files()

        # Load as NPZ file
        voices_npz = np.load(voices_path)
        info_log(f"Loaded NPZ file with {len(voices_npz.keys())} voices")

        voices_data = {}
        # Load each voice from the NPZ
        for voice_id in voices_npz.keys():
            voice_embedding = voices_npz[voice_id]
            debug_log(f"Loaded voice {voice_id}: shape={voice_embedding.shape}")
            voices_data[voice_id] = voice_embedding

        # Store in runtime
        kruntime.set('onnx_voices_data', voices_data)
        info_log(f"Successfully loaded {len(voices_data)} voices")
        return voices_data

    except Exception as e:
        error_log(f"Error loading voices NPZ: {e}")
        return create_fallback_voices()


def create_fallback_voices():
    """
    Purpose: Create basic synthetic voice embeddings for testing
    Pre: Voices loading failed
    Post: Fallback voice embeddings created and stored
    Args: None
    Returns: dict - Fallback voice embeddings
    """
    error_log("Creating fallback voice embeddings - voices file not loaded properly")

    # Create a reasonable baseline voice embedding
    np.random.seed(42)  # Deterministic for consistency
    base_embedding = np.random.normal(0, 0.1, (510, 1, 256)).astype(np.float32)

    # Create a few test voices with slight variations
    test_voices = ['af_bella', 'bf_emma', 'am_adam', 'bm_daniel']

    voices_data = {}
    for i, voice_id in enumerate(test_voices):
        # Small variations per voice
        variation = np.random.normal(0, 0.02, (510, 1, 256)).astype(np.float32)
        voice_embedding = base_embedding + variation
        voices_data[voice_id] = voice_embedding
        debug_log(f"Created fallback voice: {voice_id}")

    # Store in runtime
    kruntime.set('onnx_voices_data', voices_data)
    error_log(f"Using {len(voices_data)} fallback voices - download proper voices file to fix this")
    return voices_data


def normalize_text(text):
    """
    Purpose: Normalize text for TTS processing (simplified version for ONNX)
    Pre: Valid text string
    Post: Text normalized for TTS processing
    Args: text (str) - Text to normalize
    Returns: str - Normalized text
    """
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    text = re.sub(r'[^\S \n]', ' ', text)
    text = re.sub(r'  +', ' ', text)

    # Basic number handling
    text = re.sub(r'\b\d{4}\b', lambda m: f" {m.group()} ", text)

    return text.strip()


def generate_audio_chunks(text_chunks):
    """
    Purpose: Generate audio chunks from text chunks - ONNX implementation
    Pre: Text chunks available, voice embedding loaded
    Post: Audio segments generated for all chunks
    Args: text_chunks (list) - List of text strings to process
    Returns: list - AudioSegment objects
    """
    segments = []

    for i, text_chunk in enumerate(text_chunks):
        debug_log(f"Processing chunk {i+1}/{len(text_chunks)}: {text_chunk[:50]}...")

        try:
            # Use the ported phoneme processing
            voice_name = kruntime.get('onnx_voice_name', 'bf_emma')
            lang_code = voice_name[0] if voice_name else 'a'  # 'a' for American, 'b' for British
            tokens, phonemes, normalized = process_text_for_onnx(text_chunk, lang=lang_code)

            debug_log(f"Chunk {i+1}: '{normalized[:30]}...' -> {len(tokens)} tokens")

            # Generate audio for this chunk
            current_voice_embedding = kruntime.get('onnx_current_voice_embedding', None)
            audio_output = generate_onnx_audio(tokens, current_voice_embedding)

            # Continue with AudioSegment conversion...
            if audio_output is not None and len(audio_output) > 0:
                audio_output = audio_output / np.max(np.abs(audio_output)) if np.max(np.abs(audio_output)) > 0 else audio_output
                audio_int16 = np.int16(audio_output * 32767)

                segment = AudioSegment(
                    data=audio_int16.tobytes(),
                    sample_width=2,
                    frame_rate=24000,
                    channels=1
                )
                segments.append(segment)
                debug_log(f"Generated audio segment {i+1}: {len(audio_output)} samples")
            else:
                error_log(f"No audio generated for chunk {i+1}")

        except Exception as e:
            error_log(f"Error processing chunk {i+1}: {e}")
            continue

    info_log(f"Generated {len(segments)} audio segments")
    return segments


def generate_onnx_audio(tokens, voice_embedding, speed=1.0):
    """
    Purpose: Generate audio using ONNX model
    Pre: ONNX model initialized, tokens and voice embedding available
    Post: Audio array generated
    Args: tokens (list) - Token sequence for TTS
          voice_embedding (np.array) - Voice style embedding
          speed (float) - Playback speed multiplier
    Returns: np.array - Generated audio samples
    """
    session = initialize_onnx_model()

    try:
        # Add start/end tokens like PyTorch version: [0, *tokens, 0]
        padded_tokens = [0] + tokens + [0]
        tokens_array = np.array([padded_tokens], dtype=np.int64)
        debug_log(f"Padded tokens: {padded_tokens}")

        # Use the voice embedding for the sequence length
        seq_len = len(padded_tokens)
        if seq_len <= voice_embedding.shape[0]:
            voice_style = voice_embedding[:seq_len]  # Take first seq_len frames
        else:
            # Pad if needed
            padding = np.repeat(voice_embedding[-1:], seq_len - voice_embedding.shape[0], axis=0)
            voice_style = np.concatenate([voice_embedding, padding], axis=0)

        # Average to get style vector
        voice_style = np.mean(voice_style, axis=0).reshape(1, -1)  # (1, 256)
        debug_log(f"Voice style for {seq_len} tokens: shape={voice_style.shape}")

        inputs = {
            'tokens': tokens_array,
            'style': voice_style.astype(np.float32),
            'speed': np.array([speed], dtype=np.float32)
        }

        debug_log(f"Input shapes: tokens={tokens_array.shape}, style={voice_style.shape}, speed={inputs['speed'].shape}")

        # Run inference
        outputs = session.run(None, inputs)
        audio_output = outputs[0]

        # Validate output
        debug_log(f"Generated audio shape: {audio_output.shape}")
        debug_log(f"Audio data range: min={np.min(audio_output)}, max={np.max(audio_output)}")

        if np.any(np.isnan(audio_output)):
            error_log("ONNX model generated NaN values!")
            return None

        if audio_output.ndim > 1:
            audio_output = audio_output.flatten()

        return audio_output

    except Exception as e:
        error_log(f"Error in ONNX inference: {e}")
        raise


def load_voice(voice=None):
    """
    Purpose: Load a voice by name - ONNX implementation
    Pre: Voices data loaded, valid voice name provided
    Post: Voice embedding loaded and stored in runtime
    Args: voice (str) - Voice identifier (e.g., 'bf_emma')
    Returns: None
    """
    # Initialize voices if not loaded
    voices_data = load_voices_binary()


    # Get voices list when needed
    voices_module = importlib.import_module('extensions.KokoroTTS_4_TGUI.src.onnx.voices')
    VOICES = voices_module.get_voices()

    # Set voice name with fallback
    voice_name = voice or (VOICES[0] if VOICES else 'af_bella')

    if voice_name not in voices_data:
        error_log(f"Voice {voice_name} not found, using first available voice")
        voice_name = list(voices_data.keys())[0] if voices_data else None

    if voice_name and voice_name in voices_data:
        # Store in runtime
        kruntime.set('onnx_voice_name', voice_name)
        kruntime.set('onnx_current_voice_embedding', voices_data[voice_name])
        info_log(f'Loaded ONNX voice: {voice_name}')
    else:
        raise RuntimeError(f"No voices available or voice {voice_name} not found")


def resample_audio_to_48k(audio_segment):
    """
    Purpose: Convert 24kHz audio to 48kHz for better hardware compatibility
    Pre: Valid AudioSegment object
    Post: Audio resampled to 48kHz
    Args: audio_segment (AudioSegment) - Audio to resample
    Returns: AudioSegment - Resampled audio
    """
    try:
        # Export to temporary raw audio
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            audio_segment.export(temp_file.name, format="wav")

            # Use ffmpeg to resample if available
            try:
                import subprocess
                output_path = temp_file.name.replace('.wav', '_48k.wav')
                subprocess.run([
                    'ffmpeg', '-i', temp_file.name,
                    '-ar', '48000',
                    '-y', output_path
                ], check=True, capture_output=True)

                # Load the resampled audio
                resampled = AudioSegment.from_wav(output_path)

                # Clean up temp files
                pathlib.Path(temp_file.name).unlink()
                pathlib.Path(output_path).unlink()

                return resampled

            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback: use pydub's built-in resampling
                return audio_segment.set_frame_rate(48000)

    except Exception as e:
        debug_log(f"Resampling failed, using original: {e}")
        return audio_segment


def run(text, output_path=None, preview=False, pitch=1.0):
    """
    Purpose: Generate audio from text with optional pitch control - ONNX implementation
    Pre: Text provided, ONNX system initialized
    Post: Audio file generated at specified or default path
    Args: text (str) - Text to convert to speech
          output_path (str, optional) - Output file path
          preview (bool, optional) - Whether this is a preview generation
          pitch (float, optional) - Pitch adjustment factor
    Returns: str - Message ID for generated audio file
    """
    # Clean up old files before generating new ones
    if not preview:
        cleanup_old_audio_files()

    # Ensure model and voice are loaded
    initialize_onnx_model()
    current_voice_embedding = kruntime.get('onnx_current_voice_embedding', None)
    if current_voice_embedding is None:
        load_voice()  # Load default voice

    embedding = kruntime.get('onnx_current_voice_embedding', None)
    if embedding is not None:
        debug_log(f"Voice embedding shape: {embedding.shape}")
        debug_log(f"Voice embedding range: {np.min(embedding):.3f} to {np.max(embedding):.3f}")
        debug_log(f"Voice embedding dtype: {embedding.dtype}")
    else:
        error_log("No voice embedding in runtime!")

    msg_id = str(uuid.uuid4())

    try:
        # Process text through the pipeline
        text_chunks = split_text(text)
        segments = generate_audio_chunks(text_chunks)

        if not segments:
            error_log("No audio segments generated")
            return None

        # Concatenate audio segments
        full_audio = concatenate_audio_segments(segments)

        # Apply pitch shifting if requested
        if abs(pitch - 1.0) > 0.01:
            debug_log(f"Applying pitch shift: {pitch}")
            # Convert AudioSegment to numpy array
            audio_array = np.array(full_audio.get_array_of_samples(), dtype=np.float32)
            audio_array = audio_array / np.max(np.abs(audio_array))  # Normalize

            # Apply pitch shift
            shifted_array = apply_pitch_shift(audio_array, 24000, pitch)

            # Convert back to AudioSegment
            shifted_array = np.int16(shifted_array * 32767)
            full_audio = AudioSegment(
                data=shifted_array.tobytes(),
                sample_width=2,
                frame_rate=24000,
                channels=1
            )

        # Save audio file
        audio_dir = pathlib.Path(kruntime.get('extension_dir')) / 'audio'
        audio_dir.mkdir(exist_ok=True)
        default_audio_path = audio_dir / f'{"preview" if preview else msg_id}.wav'
        audio_path = output_path or default_audio_path
        full_audio.export(audio_path, format="wav")

        kruntime.set('soundfile', audio_path)

        info_log(f"Generated audio: {audio_path}")
        return msg_id

    except Exception as e:
        error_log(f"Error in run(): {e}")
        return None


def set_splitting_type(method="Split by sentence"):
    """
    Purpose: Set the splitting method for the text - ONNX implementation
    Pre: Valid method string provided
    Post: Splitting type stored in runtime
    Args: method (str) - Splitting method ("Split by sentence" or other)
    Returns: None
    """
    sentence_based = True if method == "Split by sentence" else False
    kruntime.set('onnx_sentence_based', sentence_based)
    info_log(f'ONNX splitting method: {"sentence" if sentence_based else "word"}')


def split_text(text):
    """
    Purpose: Split the text into chunks of sentences or words up to 510 tokens - ONNX implementation
    Pre: Valid text string, voice loaded for tokenization
    Post: Text split into processable chunks
    Args: text (str) - Text to split into chunks
    Returns: list - List of text chunks ready for TTS processing
    """
    max_token = 510

    # Get splitting preference from runtime
    sentence_based = kruntime.get('onnx_sentence_based', True)

    # Use the simple_sentence_split function we imported at the top
    if sentence_based:
        # Use simple sentence splitting
        text_parts = simple_sentence_split(text, max_chunk_length=200)
    else:
        # Split by words
        text_parts = text.split()

    chunks = []
    current_text_parts = []
    current_chunk_len = 0

    for text_part in text_parts:
        # FIXED: Use proper phonemization and tokenization instead of rough estimate
        try:
            voice_name = kruntime.get('onnx_voice_name', 'bf_emma')
            lang_code = voice_name[0] if voice_name else 'a'  # 'a' for American, 'b' for British
            tokens, phonemes, normalized = process_text_for_onnx(text_part, lang=lang_code)
            additional_tokens = len(tokens) + 1
        except Exception as e:
            error_log(f"Error tokenizing text part, using fallback estimate: {e}")
            # Fallback to rough estimate if phonemization fails
            additional_tokens = len(text_part.split()) * 2

        if current_chunk_len + additional_tokens > max_token and current_text_parts:
            # Create chunk from accumulated parts
            current_text = ' '.join(current_text_parts)
            chunks.append(current_text)
            current_text_parts = []
            current_chunk_len = 0

        current_text_parts.append(text_part)
        current_chunk_len += additional_tokens

    # Add remaining parts as final chunk
    if current_text_parts:
        current_text = ' '.join(current_text_parts)
        chunks.append(current_text)

    debug_log(f"Split text into {len(chunks)} chunks")
    return chunks


def concatenate_audio_segments(segments):
    """
    Purpose: Concatenate audio segments - ONNX implementation
    Pre: List of AudioSegment objects available
    Post: Single concatenated AudioSegment created
    Args: segments (list) - List of AudioSegment objects to concatenate
    Returns: AudioSegment - Concatenated audio or None if no segments
    """
    if not segments:
        return None

    audio_segment = segments[0]
    for segment in segments[1:]:
        audio_segment += segment

    return audio_segment


def play_audio_file(audio_path):
    """
    Purpose: Play audio file using available system tools
    Pre: Valid audio file path
    Post: Audio played through system audio player
    Args: audio_path (pathlib.Path) - Path to audio file to play
    Returns: bool - True if playback successful, False otherwise
    """
    import subprocess
    import platform

    try:
        system = platform.system()
        if system == "Linux":
            # Try common Linux audio players
            for player in ['aplay', 'paplay', 'mpv', 'vlc', 'ffplay']:
                try:
                    subprocess.run([player, str(audio_path)], check=True, capture_output=True)
                    info_log(f"Played audio using {player}")
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
        elif system == "Darwin":  # macOS
            subprocess.run(['afplay', str(audio_path)], check=True)
            return True
        elif system == "Windows":
            subprocess.run(['start', str(audio_path)], shell=True, check=True)
            return True

        error_log("No suitable audio player found")
        return False

    except Exception as e:
        error_log(f"Error playing audio: {e}")
        return False


if __name__ == '__main__':
    # Test the ONNX implementation
    try:
        load_voice('af_bella')
        result = run("Hello, this is a test of the ONNX TTS system.")
        if result:
            info_log(f"Test successful! Generated audio: {result}")

            # Get the audio file path and try to play it
            audio_path = kruntime.get('soundfile', None)
            if audio_path and pathlib.Path(audio_path).exists():
                info_log(f"Audio saved to: {audio_path}")
                info_log("Attempting to play audio...")
                if play_audio_file(audio_path):
                    info_log("Audio playback completed")
                else:
                    info_log("Could not play audio automatically")
                    info_log(f"You can manually play: {audio_path}")
            else:
                info_log("Audio file not found")
        else:
            error_log("Test failed - no audio generated")
    except Exception as e:
        error_log(f"Test error: {e}")
