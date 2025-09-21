"""
KokoroTTS_4_TGUI Settings Management Module

Provides KSettings and KRuntime classes for unified settings and state management.
Clean separation between extension settings (persisted) and runtime state (temporary).

Usage:
    from modules import shared  # For TGUI global access
    from extensions.KokoroTTS_4_TGUI.src.kshared import ksettings, kruntime

    # Extension settings (saved to YAML)
    ksettings.set('speed', 1.5)
    speed = ksettings.get('speed', 1.0)
    ksettings.save()

    # Runtime state (not saved)
    kruntime.set('current_audio_html', '<audio>...</audio>')
    html = kruntime.get('current_audio_html', '')

    # TGUI global settings
    model_name = getattr(shared.args, 'model_name', 'default')
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

import yaml
import pathlib
from modules import shared

# Import our centralized logging system
from extensions.KokoroTTS_4_TGUI.src.debug import error_log, info_log, debug_log

class KSettings:
    """
    Purpose: Settings management for KokoroTTS extension using shared.args.k4tgui['settings']
    Pre: shared module available
    Post: Settings managed through clean wrapper interface

    Manages settings persistence through YAML files while keeping runtime data
    in shared.args.k4tgui['runtime']. Extension settings are stored directly
    in our own YAML file without prefixes for clean configuration.
    """

    def __init__(self, yaml_file_path):
        """
        Purpose: Initialize KSettings with YAML file path
        Pre: Valid file path provided
        Post: Settings loaded from file and k4tgui structure initialized
        Args: yaml_file_path (pathlib.Path) - Path to settings YAML file
        Returns: None
        """
        self.yaml_file = pathlib.Path(yaml_file_path)
        self._ensure_k4tgui_structure()
        self._load_from_file()

    def _ensure_k4tgui_structure(self):
        """
        Purpose: Ensure shared.args.k4tgui structure exists with defaults
        Pre: shared.args available
        Post: k4tgui structure initialized with settings and runtime sections
        Args: None
        Returns: None
        """
        if not hasattr(shared.args, 'k4tgui'):
            shared.args.k4tgui = {
                'settings': {},
                'runtime': {}
            }
            debug_log("Created k4tgui structure in shared.args")

        # Ensure both sections exist
        if 'settings' not in shared.args.k4tgui:
            shared.args.k4tgui['settings'] = {}
        if 'runtime' not in shared.args.k4tgui:
            shared.args.k4tgui['runtime'] = {}

        # Set setting defaults if not already present
        defaults = {
            'enable_tts': False,
            'speed': 1.0,
            'pitch': 1.0,
            'voice': 'bf_emma',
            'language': 'en-us',
            'experimental': False,
            'preprocess_code': False,
            'enable_debug': 'errors',
            'preview_text': "Hello, I'm {VNAME}{VREGION}, how are you today?",
        }

        for key, default_value in defaults.items():
            if key not in shared.args.k4tgui['settings']:
                shared.args.k4tgui['settings'][key] = default_value
                debug_log(f"Set default setting {key} = {default_value}")

    def _load_from_file(self):
        """
        Purpose: Load settings from YAML file with error handling
        Pre: yaml_file path set, k4tgui structure exists
        Post: Settings from YAML merged into k4tgui['settings']
        Args: None
        Returns: None
        """
        if not self.yaml_file.exists():
            debug_log(f"Settings file {self.yaml_file} not found, using defaults")
            return

        try:
            with open(self.yaml_file, 'r', encoding='utf-8') as f:
                loaded_data = yaml.safe_load(f)
                if not loaded_data:
                    return

            loaded_count = 0
            for key, value in loaded_data.items():
                shared.args.k4tgui['settings'][key] = value
                loaded_count += 1

            debug_log(f"Loaded {loaded_count} settings from {self.yaml_file}")

        except yaml.YAMLError as e:
            error_log(f"YAML error loading settings: {e}")
        except Exception as e:
            error_log(f"Error loading settings from {self.yaml_file}: {e}")

    def get(self, key, default=None):
        """
        Purpose: Get extension setting from k4tgui['settings'] dictionary
        Pre: k4tgui structure exists
        Post: None
        Args: key (str) - Setting name (stored directly in YAML)
              default - Default value if setting not found
        Returns: Setting value from k4tgui['settings'] or default
        """
        k4tgui_dict = getattr(shared.args, 'k4tgui', {})
        settings_dict = k4tgui_dict.get('settings', {})
        return settings_dict.get(key, default)

    def set(self, key, value):
        """
        Purpose: Set extension setting in k4tgui['settings'] dictionary
        Pre: shared.args available
        Post: Setting stored in k4tgui['settings'] for immediate access
        Args: key (str) - Setting name (stored directly in YAML)
              value - Value to store
        Returns: None
        """
        if not hasattr(shared.args, 'k4tgui'):
            shared.args.k4tgui = {'settings': {}, 'runtime': {}}
        if 'settings' not in shared.args.k4tgui:
            shared.args.k4tgui['settings'] = {}

        shared.args.k4tgui['settings'][key] = value

    def save(self):
        """
        Purpose: Save current k4tgui['settings'] to YAML file
        Pre: k4tgui structure exists with current settings
        Post: YAML file updated with all current extension settings
        Args: None
        Returns: bool - True if save successful, False otherwise
        """
        try:
            k4tgui_dict = getattr(shared.args, 'k4tgui', {})
            settings_dict = k4tgui_dict.get('settings', {})

            # Convert settings to YAML (no prefix needed)
            yaml_data = {}
            for key, value in settings_dict.items():
                # Skip non-serializable values
                if callable(value):
                    continue
                yaml_data[key] = value

            # Ensure parent directory exists
            self.yaml_file.parent.mkdir(parents=True, exist_ok=True)

            # Write to YAML file
            with open(self.yaml_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump(yaml_data, f, default_flow_style=False, sort_keys=True)

            debug_log(f"Saved {len(yaml_data)} settings to {self.yaml_file}")
            return True

        except Exception as e:
            error_log(f"Error saving settings to {self.yaml_file}: {e}")
            return False

    def get_all_settings(self):
        """
        Purpose: Get all current extension settings as dictionary
        Pre: k4tgui structure exists
        Post: None
        Args: None
        Returns: dict - All current settings (clean keys, no prefixes)
        """
        k4tgui_dict = getattr(shared.args, 'k4tgui', {})
        return k4tgui_dict.get('settings', {}).copy()


class KRuntime:
    """
    Purpose: Runtime state management for KokoroTTS extension using shared.args.k4tgui['runtime']
    Pre: shared module available
    Post: Runtime state managed through clean wrapper interface

    Manages non-persistent runtime data like current audio HTML, extension paths, etc.
    This data is not saved to YAML files.
    """

    def __init__(self):
        """
        Purpose: Initialize KRuntime state manager
        Pre: shared.args available
        Post: Runtime defaults set up
        Args: None
        Returns: None
        """
        self._ensure_runtime_defaults()

    def _ensure_runtime_defaults(self):
        """
        Purpose: Ensure runtime section exists with default values
        Pre: shared.args available
        Post: Runtime defaults initialized
        Args: None
        Returns: None
        """
        if not hasattr(shared.args, 'k4tgui'):
            shared.args.k4tgui = {'settings': {}, 'runtime': {}}
        if 'runtime' not in shared.args.k4tgui:
            shared.args.k4tgui['runtime'] = {}

        # Calculate extension paths
        extension_dir = pathlib.Path(__file__).parent.parent  # Go up from src/ to extension root
        extension_name = extension_dir.name

        # Set runtime defaults if not already present
        defaults = {
            'extension_dir': extension_dir,
            'extension_name': extension_name,
            'backend': 'onnx',
            'icon_style': 'png',
            'current_audio_html': '',
            'last_loaded_audio_html': '',
            'soundfile': None,
        }

        for key, default_value in defaults.items():
            if key not in shared.args.k4tgui['runtime']:
                shared.args.k4tgui['runtime'][key] = default_value
                debug_log(f"Set default runtime {key} = {default_value}")

    def get(self, key, default=None):
        """
        Purpose: Get runtime state from k4tgui['runtime'] dictionary
        Pre: k4tgui structure exists
        Post: None
        Args: key (str) - Runtime state name
              default - Default value if state not found
        Returns: Runtime value from k4tgui['runtime'] or default
        """
        k4tgui_dict = getattr(shared.args, 'k4tgui', {})
        runtime_dict = k4tgui_dict.get('runtime', {})
        return runtime_dict.get(key, default)

    def set(self, key, value):
        """
        Purpose: Set runtime state in k4tgui['runtime'] dictionary
        Pre: shared.args available
        Post: Runtime state stored in k4tgui['runtime']
        Args: key (str) - Runtime state name
              value - Value to store
        Returns: None
        """
        if not hasattr(shared.args, 'k4tgui'):
            shared.args.k4tgui = {'settings': {}, 'runtime': {}}
        if 'runtime' not in shared.args.k4tgui:
            shared.args.k4tgui['runtime'] = {}

        shared.args.k4tgui['runtime'][key] = value

    def get_all_runtime(self):
        """
        Purpose: Get all current runtime state as dictionary
        Pre: k4tgui structure exists
        Post: None
        Args: None
        Returns: dict - All current runtime state
        """
        k4tgui_dict = getattr(shared.args, 'k4tgui', {})
        return k4tgui_dict.get('runtime', {}).copy()


def init_ksettings_and_kruntime():
    """
    Purpose: Initialize KSettings and KRuntime instances
    Pre: shared module available
    Post: Global ksettings and kruntime instances available for import
    Args: None
    Returns: tuple - (ksettings_instance, kruntime_instance)
    """
    # Calculate extension paths from this file's location
    extension_dir = pathlib.Path(__file__).parent.parent  # Go up from src/ to extension root
    settings_file = extension_dir / 'settings.yaml'

    # Create global instances
    ksettings_instance = KSettings(settings_file)
    kruntime_instance = KRuntime()

    debug_log("KSettings and KRuntime instances initialized")
    return ksettings_instance, kruntime_instance


# Initialize settings and runtime on module import
ksettings, kruntime = init_ksettings_and_kruntime()

# Convenience functions for backward compatibility
def get_setting(setting_name, default=None):
    """
    Purpose: Get a setting using KSettings (backward compatibility)
    Pre: ksettings initialized
    Post: None
    Args: setting_name (str) - Setting name (stored directly in YAML)
          default - Default value if not found
    Returns: Setting value
    """
    return ksettings.get(setting_name, default)

def set_setting(setting_name, value):
    """
    Purpose: Set a setting using KSettings (backward compatibility)
    Pre: ksettings initialized
    Post: Setting stored in k4tgui dictionary
    Args: setting_name (str) - Setting name (stored directly in YAML)
          value - Value to store
    Returns: None
    """
    ksettings.set(setting_name, value)

def save_settings():
    """
    Purpose: Save all settings using KSettings (backward compatibility)
    Pre: ksettings initialized
    Post: Settings saved to YAML file
    Args: None
    Returns: bool - Success status
    """
    return ksettings.save()

# Export everything for easy access
__all__ = [
    'ksettings',        # Main KSettings instance
    'kruntime',         # Main KRuntime instance
    # 'KSettings',        # Class for advanced usage
    # 'KRuntime',         # Class for advanced usage
    'get_setting',      # Backward compatibility
    'set_setting',      # Backward compatibility
    'save_settings',    # Backward compatibility
]
