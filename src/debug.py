"""
Debug and Logging Module

Provides clean, configurable logging functionality with three levels:
- error_log(): Always visible (configuration/runtime problems, but program continues)
- info_log(): User controlled via 'log_level' setting ('quiet' vs 'verbose')
- debug_log(): Developer controlled via set_debug_mode() function

For fatal errors that should stop execution, use exceptions instead of logging.

Key features:
- No circular import dependencies (uses dynamic imports)
- Developer debug mode independent of user logging preferences
- Clean separation between user-facing and developer logging
- Graceful fallback during initialization
"""

import importlib

def adjust_tgui_path():
    """
    Adjust sys.path to include the TGUI root directory (containing 'extensions/' or 'modules/').
    Call this before any imports to ensure consistent import paths in TGUI or standalone mode.
    """
    import os
    import sys
    start_dir = os.path.dirname(os.path.abspath(__file__))
    current = start_dir
    while current != os.path.dirname(current):  # Stop at filesystem root
        if os.path.exists(os.path.join(current, 'extensions')) and os.path.exists(os.path.join(current, 'modules')):
            if current not in sys.path:
                sys.path.insert(0, current)  # Prepend to prioritize
            return
        current = os.path.dirname(current)
    raise RuntimeError("Could not find TGUI root containing 'extensions/' or 'modules/'")

adjust_tgui_path()

# Private debug flag - NEVER touch this directly, use set_debug_mode() function
_kokoro_internal_debug_flag_do_not_touch_directly = True

def set_debug_mode(enabled):
    """
    Purpose: Control developer debug logging - ONLY way to control debug output
    Pre: Valid boolean provided
    Post: Debug logging enabled or disabled globally
    Args: enabled (bool) - True to enable debug_log() output, False to disable
    Returns: None

    Note: Never touch _kokoro_internal_debug_flag_do_not_touch_directly directly!
    """
    global _kokoro_internal_debug_flag_do_not_touch_directly
    _kokoro_internal_debug_flag_do_not_touch_directly = enabled

def _get_user_log_level():
    """
    Purpose: Get user's logging preference with graceful fallback during initialization
    Pre: None (handles all error conditions)
    Post: None
    Args: None
    Returns: str - Log level ('quiet' or 'verbose') with fallback to 'verbose'

   Simple fallback chain for user log level"""
    try:
        kshared = importlib.import_module('extensions.KokoroTTS_4_TGUI.src.kshared')
        return kshared.ksettings.get('log_level', 'verbose')
    except:
        return 'verbose'  # Safe default during initialization

def error_log(string):
    """
    Purpose: Error logging - always visible (configuration/runtime problems)
    Pre: Valid string message provided
    Post: Message logged to console with ERROR prefix
    Args: string (str) - Error message to log
    Returns: None

    Note: For problems that affect user experience but don't stop execution.
    For fatal errors, use exceptions instead.
    """
    print(f"KokoroTTS_4_TGUI ERROR: {string}")

def info_log(string):
    """
    Purpose: Info logging - user controlled via 'log_level' setting
    Pre: Valid string message provided
    Post: Message logged to console with INFO prefix if user has verbose logging enabled
    Args: string (str) - Info message to log
    Returns: None

    Shows in 'verbose' mode only. Use for progress messages and status updates
    that help users understand what's happening.
    """
    if _get_user_log_level() == 'verbose':
        print(f"KokoroTTS_4_TGUI INFO : {string}")

def debug_log(string):
    """
    Purpose: Debug logging - developer controlled via set_debug_mode()
    Pre: Valid string message provided
    Post: Message logged to console with DEBUG prefix if debug mode enabled
    Args: string (str) - Debug message to log
    Returns: None

    Shows only when set_debug_mode(True) has been called. Independent of user
    log_level setting. Use for detailed developer debugging information.
    """
    if _kokoro_internal_debug_flag_do_not_touch_directly:
        print(f"KokoroTTS_4_TGUI DEBUG: {string}")

# Backward compatibility aliases
def log(string):
    """
    Purpose: Backward compatibility - maps to info_log()
    Pre: Valid string message provided
    Post: Message handled by info_log() system
    Args: string (str) - Message to log
    Returns: None

    Note: Use info_log() directly in new code for clarity
    """
    info_log(string)

def dlog(string):
    """
    Purpose: Backward compatibility - maps to debug_log()
    Pre: Valid string message provided
    Post: Message handled by debug_log() system
    Args: string (str) - Debug message to log
    Returns: None

    Note: Use debug_log() directly in new code for clarity
    """
    debug_log(string)

def elog(string):
    """
    Purpose: Backward compatibility - maps to error_log()
    Pre: Valid string message provided
    Post: Message handled by error_log() system
    Args: string (str) - Error message to log
    Returns: None

    Note: Use error_log() directly in new code for clarity
    """
    error_log(string)

# Export all logging functions
__all__ = [
    'error_log',        # Always visible errors
    'info_log',         # User-controlled info messages
    'debug_log',        # Developer-controlled debug messages
    'set_debug_mode',   # Control debug output
    'log',              # Backward compatibility
    'dlog',             # Backward compatibility
    'elog'              # Backward compatibility
]
