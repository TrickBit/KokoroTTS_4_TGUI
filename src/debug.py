from modules import shared

def log(string):
    """Enhanced logging with support for debug levels: off, errors, all"""
    debug_mode = getattr(shared.args, 'kokoro_enable_debug', 'errors')
    if debug_mode != 'off':
        print(f"KokoroTTS_4_TGUI DEBUG: {string}")

def error_log(string):
    """Error-level logging - always shown unless debug is 'off'"""
    debug_mode = getattr(shared.args, 'kokoro_enable_debug', 'errors')
    if debug_mode in ['errors', 'all']:
        print(f"KokoroTTS_4_TGUI ERROR: {string}")
