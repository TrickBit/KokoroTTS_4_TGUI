"""
ONNX Model Downloader

Handles downloading of ONNX model files and voice embeddings.
Only imported when model files are missing to keep main code clean.
Downloads to extensions/KokoroTTS_4_TGUI/models/ directory.
"""

import pathlib
import time
import urllib.request
from urllib.error import URLError, HTTPError
from extensions.KokoroTTS_4_TGUI.src.debug import info_log, error_log, debug_log


def download_file_robust(url, path, max_retries=3):
    """
    Purpose: Download a file with retry logic and resume capability
    Pre: Valid URL and destination path provided
    Post: File downloaded to specified path
    Args: url (str) - URL to download from
          path (pathlib.Path) - Destination path
          max_retries (int) - Maximum number of retry attempts
    Returns: None
    Raises: Exception if download fails after all retries
    """
    path = pathlib.Path(path)
    temp_path = path.with_suffix(path.suffix + '.tmp')

    for attempt in range(max_retries):
        try:
            info_log(f"Download attempt {attempt + 1}/{max_retries}: {url}")

            # Check if partial file exists
            resume_byte_pos = 0
            if temp_path.exists():
                resume_byte_pos = temp_path.stat().st_size
                info_log(f"Resuming download from byte {resume_byte_pos}")

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
                                info_log(f"Download progress: {percent:.1f}%")

            # Move temp file to final location
            temp_path.rename(path)
            info_log(f"Download completed: {path}")
            return

        except (URLError, HTTPError, ConnectionError) as e:
            error_log(f"Download attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                # Clean up partial file on final failure
                if temp_path.exists():
                    temp_path.unlink()
                raise
            info_log(f"Retrying in 2 seconds...")
            time.sleep(2)


def get_model_directory():
    """
    Purpose: Get the models directory path, creating it if necessary
    Pre: Extension directory structure exists
    Post: Models directory exists and path returned
    Args: None
    Returns: pathlib.Path - Path to models directory
    """
    # Calculate from this file's location: src/onnx/ -> models/
    extension_dir = pathlib.Path(__file__).parent.parent.parent
    models_dir = extension_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    debug_log(f"Models directory: {models_dir}")
    return models_dir


def download_onnx_model():
    """
    Purpose: Download ONNX model file with validation
    Pre: Models directory accessible and writable
    Post: ONNX model file downloaded and validated
    Args: None
    Returns: pathlib.Path - Path to downloaded model file
    """
    models_dir = get_model_directory()
    model_url = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx"
    model_path = models_dir / "kokoro-v1.0.onnx"

    # Download model if not exists or incomplete
    if not model_path.exists():
        info_log("Downloading ONNX model...")
        try:
            download_file_robust(model_url, model_path)
        except Exception as e:
            error_log(f"Error downloading ONNX model: {e}")
            raise
    else:
        # Check model file size
        model_size = model_path.stat().st_size
        if model_size < 300 * 1024 * 1024:  # Less than 300MB indicates incomplete
            info_log(f"Model file appears incomplete ({model_size} bytes), re-downloading...")
            model_path.unlink()
            download_file_robust(model_url, model_path)
        else:
            info_log(f"ONNX model already exists: {model_path} ({model_size} bytes)")

    return model_path


def download_voices_binary():
    """
    Purpose: Download voices binary file with validation
    Pre: Models directory accessible and writable
    Post: Voices binary file downloaded and validated
    Args: None
    Returns: pathlib.Path - Path to downloaded voices file
    """
    models_dir = get_model_directory()
    voices_url = "https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin"
    voices_path = models_dir / "voices-v1.0.bin"

    # Download voices if not exists or incomplete
    if not voices_path.exists():
        info_log("Downloading voices binary...")
        try:
            download_file_robust(voices_url, voices_path)
        except Exception as e:
            error_log(f"Error downloading voices binary: {e}")
            raise
    else:
        # Check voices file size
        voices_size = voices_path.stat().st_size
        expected_size = 26124436  # Correct size: 6,531,109 float32 values * 4 bytes
        if abs(voices_size - expected_size) > 1024:  # Allow 1KB tolerance
            info_log(f"Voices file size mismatch (got {voices_size}, expected {expected_size}), re-downloading...")
            voices_path.unlink()
            download_file_robust(voices_url, voices_path)
        else:
            info_log(f"Voices binary already exists: {voices_path} ({voices_size} bytes)")

    return voices_path


def download_all_models():
    """
    Purpose: Download all required ONNX model files
    Pre: Network connection available, models directory writable
    Post: All model files downloaded and validated
    Args: None
    Returns: tuple - (model_path, voices_path) as pathlib.Path objects
    """
    info_log("Starting ONNX model download process...")

    try:
        model_path = download_onnx_model()
        voices_path = download_voices_binary()

        info_log("All ONNX model files downloaded successfully")
        return model_path, voices_path

    except Exception as e:
        error_log(f"Failed to download ONNX models: {e}")
        raise


def get_model_paths():
    """
    Purpose: Get paths to model files, checking if they exist
    Pre: None
    Post: None
    Args: None
    Returns: tuple - (model_path, voices_path, both_exist) where both_exist is bool
    """
    models_dir = get_model_directory()
    model_path = models_dir / "kokoro-v1.0.onnx"
    voices_path = models_dir / "voices-v1.0.bin"

    both_exist = model_path.exists() and voices_path.exists()
    debug_log(f"Model files exist: {both_exist}")

    return model_path, voices_path, both_exist


# Export main functions
__all__ = [
    'download_all_models',    # Download all required files
    'get_model_paths',        # Check if files exist and get paths
    'get_model_directory'     # Get models directory path
]
