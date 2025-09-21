"""
ONNX Phoneme Processing Module

Ported from pytorch/kokoro.py with PyTorch dependencies removed.
Handles text normalization, phonemization, and tokenization for ONNX backend.
All global state moved to kruntime for collision safety.
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

import re
import numpy as np
import time

# Handle phonemizer import gracefully
try:
    import phonemizer
    import phonemizer.backend
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False

# Now we can import from extensions with confidence
from extensions.KokoroTTS_4_TGUI.src.kshared import ksettings, kruntime
from extensions.KokoroTTS_4_TGUI.src.debug import error_log, info_log, debug_log


def split_num(num):
    """
    Purpose: Convert numbers and time formats to spoken text
    Pre: Valid regex match object containing number/time string
    Post: Number converted to speakable format
    Args: num (Match) - Regex match object containing number string
    Returns: str - Spoken format of the number
    """
    num = num.group()
    if '.' in num:
        return num
    elif ':' in num:
        h, m = [int(n) for n in num.split(':')]
        if m == 0:
            return f"{h} o'clock"
        elif m < 10:
            return f'{h} oh {m}'
        return f'{h} {m}'
    year = int(num[:4])
    if year < 1100 or year % 1000 < 10:
        return num
    left, right = num[:2], int(num[2:4])
    s = 's' if num.endswith('s') else ''
    if 100 <= year % 1000 <= 999:
        if right == 0:
            return f'{left} hundred{s}'
        elif right < 10:
            return f'{left} oh {right}{s}'
    return f'{left} {right}{s}'


def flip_money(m):
    """
    Purpose: Convert money symbols to spoken text
    Pre: Valid regex match object containing money string
    Post: Money amount converted to speakable format
    Args: m (Match) - Regex match object containing money string
    Returns: str - Spoken format of the money amount
    """
    m = m.group()
    bill = 'dollar' if m[0] == '$' else 'pound'
    if m[-1].isalpha():
        return f'{m[1:]} {bill}s'
    elif '.' not in m:
        s = '' if m[1:] == '1' else 's'
        return f'{m[1:]} {bill}{s}'
    b, c = m[1:].split('.')
    s = '' if b == '1' else 's'
    c = int(c.ljust(2, '0'))
    coins = f"cent{'' if c == 1 else 's'}" if m[0] == '$' else ('penny' if c == 1 else 'pence')
    return f'{b} {bill}{s} and {c} {coins}'


def point_num(num):
    """
    Purpose: Convert decimal numbers to spoken text
    Pre: Valid regex match object containing decimal number
    Post: Decimal converted to speakable format
    Args: num (Match) - Regex match object containing decimal string
    Returns: str - Spoken format with "point" separator
    """
    a, b = num.group().split('.')
    return ' point '.join([a, ' '.join(b)])


def normalize_text(text):
    """
    Purpose: Normalize text for TTS processing with comprehensive replacements
    Pre: Valid text string
    Post: Text normalized with proper punctuation, abbreviations, and number handling
    Args: text (str) - Raw text to normalize
    Returns: str - Normalized text ready for phonemization
    """
    text = text.replace(chr(8216), "'").replace(chr(8217), "'")
    text = text.replace('«', chr(8220)).replace('»', chr(8221))
    text = text.replace(chr(8220), '"').replace(chr(8221), '"')
    text = text.replace('(', '«').replace(')', '»')
    for a, b in zip('，。！，：；？', ',.!,:;?'):
        text = text.replace(a, b+' ')
    text = re.sub(r'[^\S \n]', ' ', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'(?<=\n) +(?=\n)', '', text)
    text = re.sub(r'\bD[Rr]\.(?= [A-Z])', 'Doctor', text)
    text = re.sub(r'\b(?:Mr\.|MR\.(?= [A-Z]))', 'Mister', text)
    text = re.sub(r'\b(?:Ms\.|MS\.(?= [A-Z]))', 'Miss', text)
    text = re.sub(r'\b(?:Mrs\.|MRS\.(?= [A-Z]))', 'Mrs', text)
    text = re.sub(r'\betc\.(?! [A-Z])', 'etc', text)
    text = re.sub(r'(?i)\b(y)eah?\b', r"\1e'a", text)
    text = re.sub(r'\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)', split_num, text)
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    text = re.sub(r'(?i)[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b', flip_money, text)
    text = re.sub(r'\d*\.\d+', point_num, text)
    text = re.sub(r'(?<=\d)-(?=\d)', ' to ', text)
    text = re.sub(r'(?<=\d)S', ' S', text)
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
    text = re.sub(r"(?<=X')S\b", 's', text)
    text = re.sub(r'(?:[A-Za-z]\.){2,} [a-z]', lambda m: m.group().replace('.', '-'), text)
    text = re.sub(r'(?i)(?<=[A-Z])\.(?=[A-Z])', '-', text)
    return text.strip()


def get_vocab():
    """
    Purpose: Get vocabulary mapping for tokenization - extracted from working kokoro-onnx library
    Pre: None
    Post: Authoritative vocabulary mapping from kokoro-onnx tokenizer.vocab
    Args: None
    Returns: dict - Mapping from phoneme characters to token IDs (exact from working implementation)
    """
    # Working vocabulary extracted from kokoro-onnx tokenizer (114 entries)
    vocab_mapping = {
        ';': 1, ':': 2, ',': 3, '.': 4, '!': 5, '?': 6, '—': 9, '…': 10,
        '"': 11, '(': 12, ')': 13, '"': 14, '"': 15, ' ': 16, '̃': 17,
        'ʣ': 18, 'ʥ': 19, 'ʦ': 20, 'ʨ': 21, 'ᵝ': 22, 'ꭧ': 23, 'A': 24,
        'I': 25, 'O': 31, 'Q': 33, 'S': 35, 'T': 36, 'W': 39, 'Y': 41,
        'ᵊ': 42, 'a': 43, 'b': 44, 'c': 45, 'd': 46, 'e': 47, 'f': 48,
        'h': 50, 'i': 51, 'j': 52, 'k': 53, 'l': 54, 'm': 55, 'n': 56,
        'o': 57, 'p': 58, 'q': 59, 'r': 60, 's': 61, 't': 62, 'u': 63,
        'v': 64, 'w': 65, 'x': 66, 'y': 67, 'z': 68, 'ɑ': 69, 'ɐ': 70,
        'ɒ': 71, 'æ': 72, 'β': 75, 'ɔ': 76, 'ɕ': 77, 'ç': 78, 'ɖ': 80,
        'ð': 81, 'ʤ': 82, 'ə': 83, 'ɚ': 85, 'ɛ': 86, 'ɜ': 87, 'ɟ': 90,
        'ɡ': 92, 'ɥ': 99, 'ɨ': 101, 'ɪ': 102, 'ʝ': 103, 'ɯ': 110,
        'ɰ': 111, 'ŋ': 112, 'ɳ': 113, 'ɲ': 114, 'ɴ': 115, 'ø': 116,
        'ɸ': 118, 'θ': 119, 'œ': 120, 'ɹ': 123, 'ɾ': 125, 'ɻ': 126,
        'ʁ': 128, 'ɽ': 129, 'ʂ': 130, 'ʃ': 131, 'ʈ': 132, 'ʧ': 133,
        'ʊ': 135, 'ʋ': 136, 'ʌ': 138, 'ɣ': 139, 'ɤ': 140, 'χ': 142,
        'ʎ': 143, 'ʒ': 147, 'ʔ': 148, 'ˈ': 156, 'ˌ': 157, 'ː': 158,
        'ʰ': 162, 'ʲ': 164, '↓': 169, '→': 171, '↗': 172, '↘': 173, 'ᵻ': 177
    }
    return vocab_mapping


def initialize_phonemizers():
    """
    Purpose: Initialize phonemizer backends and store in runtime with thread safety
    Pre: eSpeak NG installed and available
    Post: Phonemizers loaded and cached in kruntime
    Args: None
    Returns: dict - Dictionary of initialized phonemizer backends
    """
    # Check if already initialized (thread-safe check)
    if (kruntime.get('phonemizers_initialized', False) and
        kruntime.get('phonemizers', {})):
        return kruntime.get('phonemizers', {})

    info_log("Initializing phonemizers...")
    phonemizers = {}

    # Check if phonemizer library is available
    if not PHONEMIZER_AVAILABLE:
        error_log("phonemizer library not installed - install with: pip install phonemizer")
        # Create empty phonemizers dict to avoid crashes
        kruntime.set('phonemizers', {})
        kruntime.set('vocab_mapping', get_vocab())
        kruntime.set('phonemizers_initialized', True)
        return {}

    # Always try to load English (required)
    try:
        phonemizers['a'] = phonemizer.backend.EspeakBackend(
            language='en-us', preserve_punctuation=True, with_stress=True)
        info_log("Loaded en-us phonemizer")
    except Exception as e:
        error_log(f"Failed to load en-us phonemizer: {e}")

    try:
        phonemizers['b'] = phonemizer.backend.EspeakBackend(
            language='en-gb', preserve_punctuation=True, with_stress=True)
        info_log("Loaded en-gb phonemizer")
    except Exception as e:
        error_log(f"Failed to load en-gb phonemizer: {e}")

    # Optional language phonemizers (fail gracefully if not available)
    optional_langs = [
        ('f', 'fr', 'French'),
        ('i', 'it', 'Italian'),
        ('j', 'ja', 'Japanese'),
        ('p', 'pt', 'Portuguese'),
        ('z', 'zh', 'Chinese'),
        ('h', 'hi', 'Hindi')
    ]

    for code, lang, name in optional_langs:
        try:
            phonemizers[code] = phonemizer.backend.EspeakBackend(
                language=lang, preserve_punctuation=True, with_stress=True)
            info_log(f"Loaded {name} phonemizer")
        except Exception as e:
            debug_log(f"Skipped {name} phonemizer (not installed): {e}")

    # Store in runtime atomically
    kruntime.set('phonemizers', phonemizers)
    kruntime.set('vocab_mapping', get_vocab())
    kruntime.set('phonemizers_initialized', True)

    info_log(f"Initialized {len(phonemizers)} phonemizers: {list(phonemizers.keys())}")
    return phonemizers


def tokenize(ps):
    """
    Purpose: Convert phoneme string to token IDs using cached vocabulary
    Pre: Phoneme string provided, vocabulary initialized
    Post: Phoneme string converted to numerical tokens
    Args: ps (str) - Phoneme string to tokenize
    Returns: list - List of token IDs corresponding to phonemes
    """
    # Get vocabulary from runtime, initialize if needed
    vocab = kruntime.get('vocab_mapping', None)
    if not vocab:
        vocab = get_vocab()
        kruntime.set('vocab_mapping', vocab)

    return [i for i in map(vocab.get, ps) if i is not None]


def phonemize(text, lang, norm=True):
    """
    Purpose: Convert text to phonemes using appropriate language backend
    Pre: Text provided, phonemizers initialized, valid language code
    Post: Text converted to phoneme representation
    Args: text (str) - Text to phonemize
          lang (str) - Language code (e.g., 'a' for en-us, 'b' for en-gb)
          norm (bool) - Whether to normalize text before phonemization
    Returns: str - Phonemized text string
    """
    # Ensure phonemizers are initialized
    phonemizers = initialize_phonemizers()

    debug_log(f"Available phonemizers: {list(phonemizers.keys()) if phonemizers else 'None'}")
    debug_log(f"Requested lang: {lang}")

    # Check if language is available
    if lang not in phonemizers:
        error_log(f"Language {lang} not available, falling back to 'a' (en-us)")
        lang = 'a' if 'a' in phonemizers else list(phonemizers.keys())[0]

    if not phonemizers:
        error_log("No phonemizers available!")
        return ""

    # Normalize text if requested
    if norm:
        text = normalize_text(text)

    try:
        ps = phonemizers[lang].phonemize([text])
        ps = ps[0] if ps else ''

        # Kokoro-specific phoneme corrections
        # https://en.wiktionary.org/wiki/kokoro#English
        ps = ps.replace('kəkˈoːɹoʊ', 'kˈoʊkəɹoʊ').replace('kəkˈɔːɹəʊ', 'kˈəʊkəɹəʊ')
        ps = ps.replace('ʲ', 'j').replace('r', 'ɹ').replace('x', 'k').replace('ɬ', 'l')
        ps = re.sub(r'(?<=[a-zɹˌ])(?=hˈʌndɹɪd)', ' ', ps)
        ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»"" ]|$)', 'z', ps)

        # American English specific corrections
        if lang == 'a':
            ps = re.sub(r'(?<=nˈaɪn)ti(?!ˌ)', 'di', ps)

        # Filter out characters not in vocabulary
        vocab = kruntime.get('vocab_mapping', get_vocab())
        ps = ''.join(filter(lambda p: p in vocab, ps))

        return ps.strip()

    except Exception as e:
        error_log(f"Error in phonemization: {e}")
        return ""


def process_text_for_onnx(text, lang='a'):
    """
    Purpose: Complete text processing pipeline for ONNX TTS system
    Pre: Text provided, phonemizers and vocabulary available
    Post: Text converted through normalization -> phonemes -> tokens
    Args: text (str) - Input text to process
          lang (str) - Language code for phonemization
    Returns: tuple - (tokens, phonemes, normalized_text)
    """
    try:
        # Check for pronunciation language override from settings
        pronunciation_lang = ksettings.get('language', None)
        if pronunciation_lang:
            # Convert language code to phonemizer language
            lang_map = {
                'en-us': 'a', 'en-gb': 'b', 'fr-fr': 'f', 'it': 'i',
                'ja': 'j', 'pt': 'p', 'zh': 'z', 'hi': 'h'
            }
            lang = lang_map.get(pronunciation_lang, lang)
            debug_log(f"Using pronunciation language: {pronunciation_lang} -> {lang}")
        else:
            debug_log(f"Using default language: {lang}")

        # Normalize text
        normalized = normalize_text(text)
        debug_log(f"Normalized: '{normalized}'")

        # Convert to phonemes
        phonemes = phonemize(normalized, lang=lang, norm=False)
        debug_log(f"Phonemes: '{phonemes}'")

        # Convert to tokens
        tokens = tokenize(phonemes)
        debug_log(f"Tokens ({len(tokens)}): {tokens}")

        # Ensure within token limit
        if len(tokens) > 510:
            tokens = tokens[:510]

        return tokens, phonemes, normalized

    except Exception as e:
        error_log(f"Error in text processing: {e}")
        # Fallback: simple character tokenization
        simple_tokens = [ord(c) % 178 for c in text[:510]]
        return simple_tokens, text, text


# Initialize phoneme system
try:
    info_log("Phoneme module loaded - phonemizers will initialize on first use")
except Exception as e:
    error_log(f"Phoneme module initialization deferred: {e}")
