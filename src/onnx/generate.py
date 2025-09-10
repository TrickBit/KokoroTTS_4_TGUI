import gc
import shutil
import uuid
import numpy as np
import pathlib
import os
import time
import requests
from pydub import AudioSegment
import re
import html

# Conditional imports - handle both standalone and TGUI environment
STANDALONE_MODE = False
try:
    # TGUI environment
    from modules import shared
    from extensions.KokoroTTS_4_TGUI.src.onnx.phoneme import process_text_for_onnx
    from extensions.KokoroTTS_4_TGUI.src.onnx.voices import VOICES, ID2SPEAKER
    from extensions.KokoroTTS_4_TGUI.src.debug import *
    from extensions.KokoroTTS_4_TGUI.src.splittext import simple_sentence_split
except ImportError:
    # Standalone testing mode
    STANDALONE_MODE = True

    # Mock shared module
    class MockShared:
        class Args:
            pass
        args = Args()
    shared = MockShared()

    # Set test directory
    shared.args.kokoro_homedir = pathlib.Path(__file__).parent / 'test_output'
    shared.args.kokoro_homedir.mkdir(exist_ok=True)

    # Mock debug functions
    def log(message):
        print(f"KokoroTTS_4_TGUI DEBUG {message}")
    log("Trying standalone imports")
    from phoneme import process_text_for_onnx
    from voices import VOICES, ID2SPEAKER


    # Simple sentence splitter
    def simple_sentence_split(text, max_chunk_length=200):
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

# # Import voices module (should work in both modes)
# try:
#     from voices import VOICES, ID2SPEAKER
#     # Don't try to import initialize_voices since we removed it
# except ImportError as e:
#     print(f"Failed to import voices: {e}")
#     # Emergency fallback
#     VOICES = ['af_bella', 'bf_emma', 'am_adam', 'bm_daniel']
#     ID2SPEAKER = {0: 'af_bella', 1: 'bf_emma', 2: 'am_adam', 3: 'bm_daniel'}

# Global variables for ONNX model and voice data
MODEL_SESSION = None
VOICES_DATA = {}
voice_name = None
current_voice_embedding = None


def early_log(string):
    print(f"KokoroTTS_4_TGUI DEBUG: {string}")


def cleanup_old_audio_files(max_files=10, max_age_hours=24):
    """
    Clean up old audio files to manage disk usage.
    Keeps only the most recent files and removes files older than max_age_hours.
    """
    try:
        audio_dir = getattr(shared.args, f"kokoro_homedir") / 'audio'
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
                    log(f"Cleaned up old audio file: {file_path.name}")
            except Exception as e:
                log(f"Error removing file {file_path}: {e}")

    except Exception as e:
        log(f"Error in audio cleanup: {e}")

def apply_pitch_shift(audio_data, sample_rate, pitch_factor):
    """
    Apply pitch shifting to audio data using numpy (simple approach for ONNX).
    For more advanced pitch shifting, could integrate with librosa if available.

    Args:
        audio_data (np.array): Audio samples
        sample_rate (int): Sample rate (24000 for Kokoro)
        pitch_factor (float): Pitch multiplier (1.0 = normal, 0.5 = octave down, 2.0 = octave up)

    Returns:
        np.array: Pitch-shifted audio data
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
        log("librosa not available, using simple pitch shift")
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

def download_file_robust(url, path, max_retries=3):
    """Download a file with retry logic and resume capability"""
    import urllib.request
    from urllib.error import URLError, HTTPError

    path = pathlib.Path(path)
    temp_path = path.with_suffix(path.suffix + '.tmp')

    for attempt in range(max_retries):
        try:
            log(f"Download attempt {attempt + 1}/{max_retries}: {url}")

            # Check if partial file exists
            resume_byte_pos = 0
            if temp_path.exists():
                resume_byte_pos = temp_path.stat().st_size
                log(f"Resuming download from byte {resume_byte_pos}")

            # Create request with range header for resume
            req = urllib.request.Request(url)
            if resume_byte_pos > 0:
                req.add_header('Range', f'bytes={resume_byte_pos}-')

            with urllib.request.urlopen(req) as response:
                total_size = int(response.headers.get('content-length', 0))
                if resume_byte_pos > 0:
                    total_size += resume_byte_pos

                mode = 'ab' if resume_byte_pos > 0 else 'wb'
                with open(temp_path, mode) as f:
                    downloaded = resume_byte_pos
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if downloaded % (1024 * 1024 * 50) == 0:  # Log every 50 MB
                                log(f"Download progress: {percent:.1f}%")

            # Move temp file to final location
            temp_path.rename(path)
            log(f"Download completed: {path}")
            return

        except (URLError, HTTPError, ConnectionError) as e:
            log(f"Download attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Clean up partial file on final failure
                if temp_path.exists():
                    temp_path.unlink()
                raise
            log(f"Retrying in 2 seconds...")
            time.sleep(2)

_file_check_cache = {}

def download_onnx_files():
    """Download ONNX model and voices binary with robust downloading"""
    global _file_check_cache

    base_dir = pathlib.Path(__file__).parent

    # URLs for ONNX files
    model_url = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"
    voices_url = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"

    model_path = base_dir / "kokoro-v1.0.onnx"
    voices_path = base_dir / "voices-v1.0.bin"

    # Download model if not exists or incomplete
    model_cache_key = f"model_check_{model_path}"
    if not model_path.exists():
        try:
            download_file_robust(model_url, model_path)
        except Exception as e:
            early_log(f"Error downloading model: {e}")
            raise
    else:
        # Only log once per session
        if model_cache_key not in _file_check_cache:
            model_size = model_path.stat().st_size
            if model_size < 300 * 1024 * 1024:  # Less than 300MB indicates incomplete
                log(f"Model file appears incomplete ({model_size} bytes), re-downloading...")
                model_path.unlink()
                download_file_robust(model_url, model_path)
            else:
                early_log(f"Model already exists: {model_path} ({model_size} bytes)")
            _file_check_cache[model_cache_key] = True

    # Download voices if not exists or incomplete
    voices_cache_key = f"voices_check_{voices_path}"
    if not voices_path.exists():
        try:
            download_file_robust(voices_url, voices_path)
        except Exception as e:
            early_log(f"Error downloading voices: {e}")
            raise
    else:
        # Only log once per session
        if voices_cache_key not in _file_check_cache:
            voices_size = voices_path.stat().st_size
            expected_size = 26124436  # Correct size: 6,531,109 float32 values * 4 bytes
            if abs(voices_size - expected_size) > 1024:  # Allow 1KB tolerance
                log(f"Voices file size mismatch (got {voices_size}, expected {expected_size}), re-downloading...")
                voices_path.unlink()
                download_file_robust(voices_url, voices_path)
            else:
                early_log(f"Voices already exist: {voices_path} ({voices_size} bytes)")
            _file_check_cache[voices_cache_key] = True

    return model_path, voices_path



def initialize_onnx_model():
    """Initialize ONNX runtime session"""
    global MODEL_SESSION

     # Add this line to import ort locally if not already imported
    import onnxruntime as ort


    if MODEL_SESSION is not None:
        return MODEL_SESSION

    if ort is None:
        raise RuntimeError("onnxruntime not available. Install with: pip install onnxruntime")

    try:
        model_path, _ = download_onnx_files()

        # Create ONNX runtime session
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
            log("Using CUDA for ONNX inference")
        else:
            log("Using CPU for ONNX inference")

        MODEL_SESSION = ort.InferenceSession(str(model_path), providers=providers)
        log(f"ONNX model loaded successfully from {model_path}")

        return MODEL_SESSION

    except Exception as e:
        log(f"Error initializing ONNX model: {e}")
        raise

def load_voices_binary():
    """Load voices from NPZ file"""
    global VOICES_DATA

    if VOICES_DATA:
        return VOICES_DATA

    try:
        _, voices_path = download_onnx_files()

        # Load as NPZ file instead of raw binary
        voices_npz = np.load(voices_path)
        log(f"Loaded NPZ file with {len(voices_npz.keys())} voices")

        # Load each voice from the NPZ
        for voice_id in voices_npz.keys():
            voice_embedding = voices_npz[voice_id]
            log(f"Loaded voice {voice_id}: shape={voice_embedding.shape}, range={np.min(voice_embedding):.3f} to {np.max(voice_embedding):.3f}")
            VOICES_DATA[voice_id] = voice_embedding

        log(f"Successfully loaded {len(VOICES_DATA)} real voices from NPZ file")
        return VOICES_DATA

    except Exception as e:
        log(f"Error loading voices NPZ: {e} - using fallback")
        return create_fallback_voices()

def create_fallback_voices():
    """Create basic synthetic voice embeddings for testing"""
    global VOICES_DATA

    log("Creating fallback voice embeddings...")

    # Create a reasonable baseline voice embedding
    np.random.seed(42)  # Deterministic for consistency
    base_embedding = np.random.normal(0, 0.1, (510, 1, 256)).astype(np.float32)

    # Create a few test voices with slight variations
    test_voices = ['af_bella', 'bf_emma', 'am_adam', 'bm_daniel']

    for i, voice_id in enumerate(test_voices):
        # Small variations per voice
        variation = np.random.normal(0, 0.02, (510, 1, 256)).astype(np.float32)
        voice_embedding = base_embedding + variation
        VOICES_DATA[voice_id] = voice_embedding
        log(f"Created fallback voice: {voice_id}")

    return VOICES_DATA

def normalize_text(text):
    """
    Normalize text for TTS processing (simplified version for ONNX)
    """
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    text = re.sub(r'[^\S \n]', ' ', text)
    text = re.sub(r'  +', ' ', text)

    # Basic number handling
    text = re.sub(r'\b\d{4}\b', lambda m: f" {m.group()} ", text)

    return text.strip()


def generate_audio_chunks(text_chunks):
    """Generate audio chunks from text chunks - ONNX implementation"""
    segments = []

    for i, text_chunk in enumerate(text_chunks):
        log(f"Processing chunk {i+1}/{len(text_chunks)}: {text_chunk[:50]}...")

        try:
            # Use the ported phoneme processing
            lang_code = voice_name[0] if voice_name else 'a'  # 'a' for American, 'b' for British
            tokens, phonemes, normalized = process_text_for_onnx(text_chunk, lang=lang_code)

            log(f"Chunk {i+1}: '{normalized[:30]}...' -> {len(tokens)} tokens")

            # Generate audio for this chunk
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
                log(f"Generated audio segment {i+1}: {len(audio_output)} samples")
            else:
                log(f"Warning: No audio generated for chunk {i+1}")

        except Exception as e:
            log(f"Error processing chunk {i+1}: {e}")
            continue

    log(f"Generated {len(segments)} audio segments")
    return segments

def generate_onnx_audio(tokens, voice_embedding, speed=1.0):
    """Generate audio using ONNX model"""
    session = initialize_onnx_model()

    try:
        # Add start/end tokens like PyTorch version: [0, *tokens, 0]
        padded_tokens = [0] + tokens + [0]
        tokens_array = np.array([padded_tokens], dtype=np.int64)
        log(f"Padded tokens: {padded_tokens}")

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
        log(f"Voice style for {seq_len} tokens: shape={voice_style.shape}")

        inputs = {
            'tokens': tokens_array,
            'style': voice_style.astype(np.float32),
            'speed': np.array([speed], dtype=np.float32)
        }

        log(f"Input shapes: tokens={tokens_array.shape}, style={voice_style.shape}, speed={inputs['speed'].shape}")

        # Run inference
        outputs = session.run(None, inputs)
        audio_output = outputs[0]

        # Validate output
        log(f"Generated audio shape: {audio_output.shape}")
        log(f"Audio data range: min={np.min(audio_output)}, max={np.max(audio_output)}")

        if np.any(np.isnan(audio_output)):
            log("ERROR: ONNX model generated NaN values!")
            return None

        if audio_output.ndim > 1:
            audio_output = audio_output.flatten()

        return audio_output

    except Exception as e:
        log(f"Error in ONNX inference: {e}")
        raise

def load_voice(voice=None):
    """Load a voice by name - ONNX implementation"""
    global voice_name, current_voice_embedding

    # Initialize voices if not loaded
    voices_data = load_voices_binary()

    # from .voices import VOICES
    voice_name = voice or (VOICES[0] if VOICES else 'af_bella')

    if voice_name not in voices_data:
        log(f"Warning: Voice {voice_name} not found, using first available voice")
        voice_name = list(voices_data.keys())[0] if voices_data else None

    if voice_name and voice_name in voices_data:
        current_voice_embedding = voices_data[voice_name]
        log(f'Loaded ONNX voice: {voice_name}')
    else:
        raise RuntimeError(f"No voices available or voice {voice_name} not found")


def resample_audio_to_48k(audio_segment):
    """Convert 24kHz audio to 48kHz for better hardware compatibility"""
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
        log(f"Resampling failed, using original: {e}")
        return audio_segment


def run(text, output_path=None, preview=False, pitch=1.0):
    """Generate audio from text with optional pitch control - ONNX implementation"""

    # Clean up old files before generating new ones
    if not preview:
        cleanup_old_audio_files()

    global current_voice_embedding, voice_name

    # Ensure model and voice are loaded
    initialize_onnx_model()
    if current_voice_embedding is None:
        load_voice()  # Load default voice

    msg_id = str(uuid.uuid4())

    try:
        # Process text through the pipeline
        text_chunks = split_text(text)
        segments = generate_audio_chunks(text_chunks)

        if not segments:
            log("No audio segments generated")
            return None

        # Concatenate audio segments
        full_audio = concatenate_audio_segments(segments)

        # Apply pitch shifting if requested
        if abs(pitch - 1.0) > 0.01:
            log(f"Applying pitch shift: {pitch}")
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

        full_audio = resample_audio_to_48k(full_audio)
        # Save audio file
        audio_dir = getattr(shared.args, f"kokoro_homedir") / 'audio'
        audio_dir.mkdir(exist_ok=True)
        default_audio_path = audio_dir / f'{"preview" if preview else msg_id}.wav'
        audio_path = output_path or default_audio_path
        full_audio.export(audio_path, format="wav")

        setattr(shared.args, 'kokoro_soundfile', audio_path)

        log(f"Generated audio: {audio_path}")
        return msg_id

    except Exception as e:
        log(f"Error in run(): {e}")
        return None

# Text splitting variables and functions
sentence_based = True

def set_splitting_type(method="Split by sentence"):
    """Set the splitting method for the text - ONNX implementation"""
    global sentence_based
    sentence_based = True if method == "Split by sentence" else False
    log(f'ONNX splitting method: {"sentence" if sentence_based else "word"}')

def split_text(text):
    """Split the text into chunks of sentences or words up to 510 tokens - ONNX implementation"""
    max_token = 510

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
        # Estimate token length (rough approximation)
        estimated_tokens = len(text_part.split()) * 2  # Rough estimate

        if current_chunk_len + estimated_tokens > max_token and current_text_parts:
            # Create chunk from accumulated parts
            current_text = ' '.join(current_text_parts)
            chunks.append(current_text)
            current_text_parts = []
            current_chunk_len = 0

        current_text_parts.append(text_part)
        current_chunk_len += estimated_tokens

    # Add remaining parts as final chunk
    if current_text_parts:
        current_text = ' '.join(current_text_parts)
        chunks.append(current_text)

    log(f"Split text into {len(chunks)} chunks")
    return chunks

def generate_audio_chunks(text_chunks):
    """Generate audio chunks from text chunks - ONNX implementation"""
    segments = []

    for i, text_chunk in enumerate(text_chunks):
        log(f"Processing chunk {i+1}/{len(text_chunks)}: {text_chunk[:50]}...")

        try:
            lang_code = voice_name[0] if voice_name else 'a'  # 'a' for American, 'b' for British
            tokens, phonemes, normalized = process_text_for_onnx(text_chunk, lang=lang_code)

            log(f"Chunk {i+1}: '{normalized[:30]}...' -> {len(tokens)} tokens")

            # Generate audio for this chunk
            audio_output = generate_onnx_audio(tokens, current_voice_embedding)

            # Convert to AudioSegment
            if audio_output is not None and len(audio_output) > 0:
                log(f"Audio array min/max before int16 conversion: {np.min(audio_output)}/{np.max(audio_output)}")
                audio_output = audio_output / np.max(np.abs(audio_output)) if np.max(np.abs(audio_output)) > 0 else audio_output
                audio_int16 = np.int16(audio_output * 32767)

                # ADD MORE DEBUGGING HERE:
                log(f"Int16 audio min/max: {np.min(audio_int16)}/{np.max(audio_int16)}")
                log(f"Int16 audio first 10 samples: {audio_int16[:10]}")

                segment = AudioSegment(
                    data=audio_int16.tobytes(),
                    sample_width=2,
                    frame_rate=24000,
                    channels=1
                )
                segments.append(segment)
                log(f"Generated audio segment {i+1}: {len(audio_output)} samples")
            else:
                log(f"Warning: No audio generated for chunk {i+1}")

        except Exception as e:
            log(f"Error processing chunk {i+1}: {e}")
            continue

    log(f"Generated {len(segments)} audio segments")
    return segments

def concatenate_audio_segments(segments):
    """Concatenate audio segments - ONNX implementation"""
    if not segments:
        return None

    audio_segment = segments[0]
    for segment in segments[1:]:
        audio_segment += segment

    return audio_segment

# Initialize on import
try:
    download_onnx_files()
    log("Initializing ONNX backend...")
    model_path, voices_path = download_onnx_files()

    # Initialize voices
    # from .voices import initialize_voices
    load_voices_binary()

    log("ONNX backend initialized successfully")
except Exception as e:
    early_log(f"Error during ONNX initialization: {e}")

def play_audio_file(audio_path):
    """Play audio file using available system tools"""
    import subprocess
    import platform

    try:
        system = platform.system()
        if system == "Linux":
            # Try common Linux audio players
            for player in ['aplay', 'paplay', 'mpv', 'vlc', 'ffplay']:
                try:
                    subprocess.run([player, str(audio_path)], check=True, capture_output=True)
                    log(f"Played audio using {player}")
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
        elif system == "Darwin":  # macOS
            subprocess.run(['afplay', str(audio_path)], check=True)
            return True
        elif system == "Windows":
            subprocess.run(['start', str(audio_path)], shell=True, check=True)
            return True

        log("No suitable audio player found")
        return False

    except Exception as e:
        log(f"Error playing audio: {e}")
        return False

if __name__ == '__main__':
    # Test the ONNX implementation
    try:
        load_voice('af_bella')
        # result = run("Hello, this is a test of the ONNX TTS system.")
        result = run("Hello, you secy fukker")
        if result:
            print(f"Test successful! Generated audio: {result}")

            # Get the audio file path and try to play it
            audio_path = getattr(shared.args, 'kokoro_soundfile', None)
            if audio_path and pathlib.Path(audio_path).exists():
                print(f"Audio saved to: {audio_path}")
                print("Attempting to play audio...")
                if play_audio_file(audio_path):
                    print("Audio playback completed")
                else:
                    print("Could not play audio automatically")
                    print(f"You can manually play: {audio_path}")
            else:
                print("Audio file not found")
        else:
            print("Test failed - no audio generated")
    except Exception as e:
        print(f"Test error: {e}")
