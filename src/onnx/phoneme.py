"""
ONNX Phoneme Processing Module

Ported from pytorch/kokoro.py with PyTorch dependencies removed.
Handles text normalization, phonemization, and tokenization for ONNX backend.
"""

import phonemizer
import re
import numpy as np

def log(message):
    print(f"KokoroTTS_4_TGUI phoneme DEBUG {message}")

def split_num(num):
    """Convert numbers and time formats to spoken text"""
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
    """Convert money symbols to spoken text"""
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
    """Convert decimal numbers to spoken text"""
    a, b = num.group().split('.')
    return ' point '.join([a, ' '.join(b)])

def normalize_text(text):
    """Normalize text for TTS processing"""
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
    """Get vocabulary mapping for tokenization - from working kokoro-onnx library"""
    # Build vocabulary from the working library's exact mapping
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

VOCAB = get_vocab()

def tokenize(ps):
    """Convert phoneme string to token IDs"""
    return [i for i in map(VOCAB.get, ps) if i is not None]


# Initialize phonemizers with graceful fallback
phonemizers = {}

# Always try to load English (required)
try:
    phonemizers['a'] = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
    log("Loaded en-us phonemizer")
except Exception as e:
    log(f"Failed to load en-us phonemizer: {e}")

try:
    phonemizers['b'] = phonemizer.backend.EspeakBackend(language='en-gb', preserve_punctuation=True, with_stress=True)
    log("Loaded en-gb phonemizer")
except Exception as e:
    log(f"Failed to load en-gb phonemizer: {e}")

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
        phonemizers[code] = phonemizer.backend.EspeakBackend(language=lang, preserve_punctuation=True, with_stress=True)
        log(f"Loaded {name} phonemizer")
    except Exception as e:
        log(f"Skipped {name} phonemizer (not installed): {e}")

log(f"Initialized {len(phonemizers)} phonemizers: {list(phonemizers.keys())}")

def phonemize(text, lang, norm=True):
    """Convert text to phonemes"""
    if norm:
        text = normalize_text(text)
    ps = phonemizers[lang].phonemize([text])
    ps = ps[0] if ps else ''
    # https://en.wiktionary.org/wiki/kokoro#English
    ps = ps.replace('kəkˈoːɹoʊ', 'kˈoʊkəɹoʊ').replace('kəkˈɔːɹəʊ', 'kˈəʊkəɹəʊ')
    ps = ps.replace('ʲ', 'j').replace('r', 'ɹ').replace('x', 'k').replace('ɬ', 'l')
    ps = re.sub(r'(?<=[a-zɹˌ])(?=hˈʌndɹɪd)', ' ', ps)
    ps = re.sub(r' z(?=[;:,.!?¡¿—…"«»"" ]|$)', 'z', ps)
    if lang == 'a':
        ps = re.sub(r'(?<=nˈaɪn)ti(?!ˌ)', 'di', ps)
    ps = ''.join(filter(lambda p: p in VOCAB, ps))
    return ps.strip()

# In process_text_for_onnx(), add more detailed debugging:
def process_text_for_onnx(text, lang='a'):
    """
    Complete text processing pipeline for ONNX:
    text -> normalized -> phonemes -> tokens
    """
    try:
        # Check for pronunciation language override from settings
        try:
            from modules import shared
            pronunciation_lang = getattr(shared.args, 'kokoro_language', 'en-us')
            # Convert language code to phonemizer language
            lang_map = {
                'en-us': 'a', 'en-gb': 'b', 'fr-fr': 'f', 'it': 'i',
                'ja': 'j', 'pt': 'p', 'zh': 'z', 'hi': 'h'
            }
            lang = lang_map.get(pronunciation_lang, lang)
            log(f"Using pronunciation language: {pronunciation_lang} -> {lang}")
        except (ImportError, AttributeError):
            log(f"Using default language: {lang}")

        # Normalize text
        normalized = normalize_text(text)
        log(f"Normalized: '{normalized}'")

        # Convert to phonemes
        phonemes = phonemize(normalized, lang=lang, norm=False)
        log(f"Phonemes: '{phonemes}'")

        # Convert to tokens
        tokens = tokenize(phonemes)
        log(f"Tokens ({len(tokens)}): {tokens}")

        # Ensure within token limit
        if len(tokens) > 510:
            tokens = tokens[:510]

        return tokens, phonemes, normalized

    except Exception as e:
        log(f"Error in text processing: {e}")
        # Fallback: simple character tokenization
        simple_tokens = [ord(c) % 178 for c in text[:510]]
        return simple_tokens, text, text
