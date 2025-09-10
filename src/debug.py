from modules import shared

def log(string):
    if getattr(shared.args, 'kokoro_enable_debug', False):
        print(f"KokoroTTS_4_TGUI DEBUG: {string}")
