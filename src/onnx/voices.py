"""
ONNX Voices Module

This module provides voice mappings and language information for the ONNX backend.
All voice data is managed through the kruntime system for collision safety.
Voice loading and initialization is handled on-demand through function calls.
"""

def adjust_tgui_path():
    """
    Purpose: Adjust sys.path to include the TGUI root directory (containing 'extensions/' and 'modules/')
    Pre: Extension directory structure exists
    Post: TGUI root directory added to sys.path for consistent imports
    Args: None
    Returns: None
    Raises: RuntimeError if TGUI root directory cannot be found
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

import numpy as np
import pathlib

# Now we can import from extensions with confidence
from extensions.KokoroTTS_4_TGUI.src.kshared import ksettings, kruntime
from extensions.KokoroTTS_4_TGUI.src.debug import error_log, info_log, debug_log


# Complete voice mapping from sherpa-onnx implementation
# This maps binary file indices to voice identifiers
ID2SPEAKER = {
    # American Female voices
    0: "af_alloy", 1: "af_aoede", 2: "af_bella", 3: "af_heart", 4: "af_jessica",
    5: "af_kore", 6: "af_nicole", 7: "af_nova", 8: "af_river", 9: "af_sarah", 10: "af_sky",

    # American Male voices
    11: "am_adam", 12: "am_echo", 13: "am_eric", 14: "am_fenrir", 15: "am_liam",
    16: "am_michael", 17: "am_onyx", 18: "am_puck", 19: "am_santa",

    # British Female voices
    20: "bf_alice", 21: "bf_emma", 22: "bf_isabella", 23: "bf_lily",

    # British Male voices
    24: "bm_daniel", 25: "bm_fable", 26: "bm_george", 27: "bm_lewis",

    # European voices
    28: "ef_dora", 29: "em_alex", 30: "em_santa",

    # French voices
    31: "ff_siwis",

    # Hindi/Indian voices
    32: "hf_alpha", 33: "hf_beta", 34: "hm_omega", 35: "hm_psi",

    # Italian voices
    36: "if_sara", 37: "im_nicola",

    # Japanese voices
    38: "jf_alpha", 39: "jf_gongitsune", 40: "jf_nezumi", 41: "jf_tebukuro", 42: "jm_kumo",

    # Portuguese voices
    43: "pf_dora", 44: "pm_alex", 45: "pm_santa",

    # Chinese voices
    46: "zf_xiaobei", 47: "zf_xiaoni", 48: "zf_xiaoxiao", 49: "zf_xiaoyi",
    50: "zm_yunjian", 51: "zm_yunxia", 52: "zm_yunxi", 53: "zm_yunyang",
}

# Voice language mapping for pronunciation
VOICE_LANGUAGES = {
    # English voices
    'af_': 'en-us', 'am_': 'en-us',  # American English
    'bf_': 'en-gb', 'bm_': 'en-gb',  # British English

    # Other languages
    'ef_': 'en-us', 'em_': 'en-us',  # European (default to US English)
    'ff_': 'fr-fr',                  # French
    'hf_': 'hi', 'hm_': 'hi',       # Hindi
    'if_': 'it', 'im_': 'it',       # Italian
    'jf_': 'ja', 'jm_': 'ja',       # Japanese
    'pf_': 'pt', 'pm_': 'pt',       # Portuguese
    'zf_': 'zh', 'zm_': 'zh',       # Chinese
}


def get_voice_language(voice_name):
    """
    Purpose: Get the language code for a voice based on its prefix
    Pre: Valid voice name string provided
    Post: None
    Args: voice_name (str) - Voice identifier (e.g., 'af_bella', 'bf_emma')
    Returns: str - Language code (e.g., 'en-us', 'fr-fr', 'ja')
    """
    if len(voice_name) >= 3 and voice_name[2] == '_':
        prefix = voice_name[:3]
        return VOICE_LANGUAGES.get(prefix, 'en-us')
    return 'en-us'


def get_voice_info(voice_name):
    """
    Purpose: Extract comprehensive information about a voice from its name
    Pre: Valid voice name string provided
    Post: None
    Args: voice_name (str) - Voice identifier (e.g., 'af_bella')
    Returns: dict - Voice information including gender, country, language, display name, and special status
    """
    if len(voice_name) < 3 or voice_name[2] != '_':
        return {
            'gender': 'unknown',
            'country': 'unknown',
            'language': 'en-us',
            'display_name': voice_name.capitalize(),
            'is_special': False
        }

    country_code = voice_name[0]
    gender_code = voice_name[1]
    name = voice_name[3:].capitalize()

    # Country mapping
    country_map = {
        'a': 'American', 'b': 'British', 'e': 'European', 'f': 'French',
        'h': 'Hindi', 'i': 'Italian', 'j': 'Japanese', 'p': 'Portuguese',
        'z': 'Chinese', 'm': 'Mixed'  # For mf_, mm_, mx_ blended voices
    }

    # Gender mapping
    gender_map = {
        'f': 'Female', 'm': 'Male', 'x': 'Non-gendered'  # For mx_ voices like mx_box, mx_narrator
    }

    # Special voice detection
    is_special = name.lower() in ['santa', 'alpha', 'beta', 'omega', 'psi']

    return {
        'gender': gender_map.get(gender_code, 'Unknown'),
        'country': country_map.get(country_code, 'Unknown'),
        'language': get_voice_language(voice_name),
        'display_name': name,
        'is_special': is_special
    }


def get_voices():
    """
    Purpose: Get the current list of available voices, loading if necessary
    Pre: kruntime system available
    Post: Voice list loaded into kruntime if not already cached
    Args: None
    Returns: list - List of available voice identifiers
    """
    # Check if voices are already cached in kruntime
    voices_list = kruntime.get('onnx_voices_list', None)
    if voices_list:
        debug_log(f"Returning cached voices: {len(voices_list)} voices")
        return voices_list

    # Not cached, try to initialize
    debug_log("Voices not cached, initializing...")
    try:
        initialize_voices()
        voices_list = kruntime.get('onnx_voices_list', None)
        if voices_list:
            info_log(f"Loaded {len(voices_list)} voices on demand")
            return voices_list
    except Exception as e:
        error_log(f"Error loading voices: {e}")

    # Fallback to safe default voices
    fallback_voices = ['af_bella', 'bf_emma', 'am_adam', 'bm_daniel']
    error_log(f"Using fallback voices: {fallback_voices}")
    return fallback_voices


def initialize_voices():
    """
    Purpose: Initialize the voices list from NPZ file and store in kruntime
    Pre: Voices binary file exists and is readable
    Post: Voice list populated in kruntime system
    Args: None
    Returns: None
    Raises: Exception if voices file cannot be loaded
    """
    # Check if already initialized to avoid duplicate work
    if kruntime.get('onnx_voices_list', None):
        debug_log("Voices already initialized, skipping")
        return

    try:
        # Calculate voices path dynamically from file location (works in all modes)
        extension_dir = pathlib.Path(__file__).parent.parent.parent  # Go up to extension root
        voices_path = extension_dir / 'models' / "voices-v1.0.bin"
        debug_log(f"Looking for voices at: {voices_path}")

        if voices_path.exists():
            # Load as NPZ file
            voices_npz = np.load(voices_path)
            info_log(f"Loaded NPZ file with {len(voices_npz.keys())} voices")

            # Populate voices list in kruntime
            voices_list = list(voices_npz.keys())
            kruntime.set('onnx_voices_list', voices_list)
            info_log(f"Initialized {len(voices_list)} ONNX voices")
            debug_log(f"Available voices: {voices_list[:5]}...")
        else:
            error_log(f"Voices file not found at {voices_path}")
            # Set fallback voices in kruntime
            fallback_voices = ['af_bella', 'bf_emma', 'am_adam', 'bm_daniel']
            kruntime.set('onnx_voices_list', fallback_voices)
            error_log("Using fallback voice list - download voices file to get all voices")

    except Exception as e:
        error_log(f"Error initializing voices: {e}")
        # Emergency fallback
        fallback_voices = ['af_bella', 'bf_emma', 'am_adam', 'bm_daniel']
        kruntime.set('onnx_voices_list', fallback_voices)
        error_log("Emergency fallback voice list applied")
        raise  # Re-raise for calling code to handle


# Auto-initialize when module is imported (non-blocking)
try:
    initialize_voices()
    debug_log("Voice module initialization completed successfully")
except Exception as e:
    error_log(f"Voice module initialization deferred: {e}")
