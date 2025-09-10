"""
ONNX Voices Module - Simplified

This module provides voice mappings and language information for the ONNX backend.
All downloading, loading, and initialization is handled by generate.py.
"""
import numpy as np
from pathlib import Path
import pathlib  # Add this line

# Conditional imports - handle both standalone and TGUI environment
try:
    from extensions.KokoroTTS_4_TGUI.src.debug import *
    STANDALONE_MODE = False
except ImportError:
    STANDALONE_MODE = True
    def log(message):
        print(f"KokoroTTS_4_TGUI onnx voices: {message}")



# Complete voice mapping from sherpa-onnx implementation
# This maps binary file indices to voice identifiers
ID2SPEAKER = {
    # American Female voices
    0: "af_alloy",
    1: "af_aoede",
    2: "af_bella",
    3: "af_heart",
    4: "af_jessica",
    5: "af_kore",
    6: "af_nicole",
    7: "af_nova",
    8: "af_river",
    9: "af_sarah",
    10: "af_sky",

    # American Male voices
    11: "am_adam",
    12: "am_echo",
    13: "am_eric",
    14: "am_fenrir",
    15: "am_liam",
    16: "am_michael",
    17: "am_onyx",
    18: "am_puck",
    19: "am_santa",

    # British Female voices
    20: "bf_alice",
    21: "bf_emma",
    22: "bf_isabella",
    23: "bf_lily",

    # British Male voices
    24: "bm_daniel",
    25: "bm_fable",
    26: "bm_george",
    27: "bm_lewis",

    # European voices
    28: "ef_dora",    # European Female
    29: "em_alex",    # European Male
    30: "em_santa",   # European Male Santa (special)

    # French voices
    31: "ff_siwis",   # French Female

    # Hindi/Indian voices
    32: "hf_alpha",   # Hindi Female
    33: "hf_beta",    # Hindi Female
    34: "hm_omega",   # Hindi Male
    35: "hm_psi",     # Hindi Male

    # Italian voices
    36: "if_sara",    # Italian Female
    37: "im_nicola",  # Italian Male

    # Japanese voices
    38: "jf_alpha",
    39: "jf_gongitsune",
    40: "jf_nezumi",
    41: "jf_tebukuro",
    42: "jm_kumo",    # Japanese Male

    # Portuguese voices
    43: "pf_dora",    # Portuguese Female
    44: "pm_alex",    # Portuguese Male
    45: "pm_santa",   # Portuguese Male Santa (special)

    # Chinese voices
    46: "zf_xiaobei",
    47: "zf_xiaoni",
    48: "zf_xiaoxiao",
    49: "zf_xiaoyi",
    50: "zm_yunjian",
    51: "zm_yunxia",
    52: "zm_yunxi",
    53: "zm_yunyang",
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

# Dynamic voice list - populated by generate.py
VOICES = []

def get_voice_language(voice_name):
    """
    Get the language code for a voice based on its prefix.

    Args:
        voice_name (str): Voice identifier (e.g., 'af_bella', 'bf_emma')

    Returns:
        str: Language code (e.g., 'en-us', 'fr-fr', 'ja')
    """
    if len(voice_name) >= 3 and voice_name[2] == '_':
        prefix = voice_name[:3]
        return VOICE_LANGUAGES.get(prefix, 'en-us')
    return 'en-us'

def get_voice_info(voice_name):
    """
    Extract information about a voice from its name.

    Args:
        voice_name (str): Voice identifier (e.g., 'af_bella')

    Returns:
        dict: Voice information including gender, country, language, and display name
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
        'a': 'American',
        'b': 'British',
        'e': 'European',
        'f': 'French',
        'h': 'Hindi',
        'i': 'Italian',
        'j': 'Japanese',
        'p': 'Portuguese',
        'z': 'Chinese',
        'm': 'Mixed'  # For mf_, mm_, mx_ blended voices
    }

    # Gender mapping
    gender_map = {
        'f': 'Female',
        'm': 'Male',
        'x': 'Non-gendered'  # For mx_ voices like mx_box, mx_narrator
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


def initialize_voices():
    """Initialize voices when module is imported"""
    global VOICES
    try:
        # Use the same NPZ loading logic from generate.py
        voices_path = pathlib.Path(__file__).parent / "voices-v1.0.bin"

        if voices_path.exists():
            # Load as NPZ file
            voices_npz = np.load(voices_path)
            log(f"Loaded NPZ file with {len(voices_npz.keys())} voices")

            # Populate VOICES list with available voice names
            VOICES.extend(list(voices_npz.keys()))
            log(f"Initialized {len(VOICES)} ONNX voices: {VOICES[:5]}...")
        else:
            log(f"Voices file not found at {voices_path}")
            # Fallback voices
            VOICES.extend(['af_bella', 'bf_emma', 'am_adam', 'bm_daniel'])

    except Exception as e:
        log(f"Error initializing voices: {e}")
        # Emergency fallback
        if not VOICES:
            VOICES.extend(['af_bella', 'bf_emma', 'am_adam', 'bm_daniel'])

# Auto-initialize when module is imported
initialize_voices()
