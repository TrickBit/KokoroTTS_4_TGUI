# KokoroTTS_4_TGUI : KokoroTTS for Text Generation WebUI - Enhanced Edition

A comprehensive Text-to-Speech extension for [Oobabooga Text Generation WebUI](https://github.oobabooga/text-generation-webui) featuring the Kokoro TTS model with advanced audio controls, clean architecture, and optimized performance.

## ✨ Key Features

- **🎙️ High-Quality Voice Synthesis** - Powered by Kokoro-82M model
- **🎛️ Advanced Audio Controls** - Speed and pitch adjustment
- **🔄 Smart Text Processing** - Intelligent handling of code blocks and markdown
- **🎯 Self-Contained Installation** - Everything stays within the extension directory
- **🔊 Integrated Audio Player** - Auto-playing audio with play/pause controls
- **⚙️ Comprehensive Settings** - Persistent configuration with intelligent defaults
- **🧹 Automatic Cleanup** - Smart audio file management
- **🐛 Robust Error Handling** - Configurable debug logging

## 🚀 What's New in This Version

This enhanced version significantly improves upon the original with:

- **Clean Architecture** - No scattered files in user directories
- **Integrated Audio System** - Built-in player with speed/pitch controls
- **Smart Settings Management** - YAML-based configuration with hot-reloading
- **Optimized Performance** - Efficient audio generation and cleanup
- **Better Text Processing** - Configurable code block handling
- **Professional UI** - Streamlined interface with grouped controls

## 📋 Requirements

### System Dependencies
- **eSpeak NG**: Download from [eSpeak NG Releases](https://github.com/espeak-ng/espeak-ng/releases)
- **FFmpeg**: Download from [FFmpeg Downloads](https://ffmpeg.org/download.html)

### Python Dependencies
The extension will automatically install required packages on first run.

## 🛠️ Installation

### Prerequisites
- **Text Generation WebUI** must be installed and working
- **eSpeak NG**: Download from [eSpeak NG Releases](https://github.com/espeak-ng/espeak-ng/releases)
- **FFmpeg**: Download from [FFmpeg Downloads](https://ffmpeg.org/download.html)

### Installation Steps

1. **Navigate to your text-generation-webui extensions directory**:
   ```bash
   cd /path/to/text-generation-webui/extensions/
   ```

2. Clone this extension:
    ```bash
    git clone https://github.com/TrickBit/KokoroTTS_4_TGUI.git
   ```
3. Activate your text-generation-webui Python environment:
   ```bash
   # If using venv (adjust path as needed):
    source ../venv/bin/activate

    # Or if using conda:
    conda activate textgen

    # Or if using the webui's start script environment
   ```
4. Install Python dependencies:
   ```bash
   pip install -r KokoroTTS_4_TGUI/requirements.txt
   ```
5. Start text-generation-webui and navigate to the Interface tab to enable the KokoroTTS extension


First Run

The extension will automatically download the Kokoro model weights on first use
NLTK language data will be downloaded to the extension directory
Default settings will be created in extensions/KokoroTTS_4_TGUI/settings.yaml


## 🎮 Usage

### Basic Setup
1. **Enable TTS** - Check "Enable TTS" to automatically generate speech for AI responses
2. **Select Voice** - Choose from British or American English voices
3. **Adjust Settings** - Customize speed, pitch, and text processing options

### Voice Controls
- **Speed Control** (0.5x - 1.5x) - Adjust playback speed without pitch change
- **Pitch Control** (0.5x - 1.5x) - Modify voice pitch for creative effects
- **Voice Preview** - Test settings with customizable preview text

### Text Processing
- **Smart Code Handling** - Choose whether to speak or skip code blocks
- **Sentence Splitting** - Intelligent text segmentation for long inputs
- **Markdown Processing** - Clean handling of formatted text

### Audio Management
- **Integrated Player** - Auto-playing audio with play/pause controls
- **Smart Cleanup** - Automatic removal of old audio files
- **Speed Optimization** - Efficient generation and playback

## ⚙️ Configuration

Settings are automatically saved to `settings.yaml` and include:

- Voice selection and preview text
- Speed and pitch preferences
- Text processing options
- Debug logging levels
- Audio management settings

## 🎯 Technical Details

### Supported Languages
- **English** (British and American variants)

### Token Limitations
- Kokoro TTS supports up to 510 tokens per segment
- The extension automatically splits longer text into manageable chunks
- Sentence-based splitting preserves context and audio quality

### GPU Support
- Automatically detects and uses available GPU
- Falls back to CPU if no GPU is available
- Modify `device` variable in `src/generate.py` for specific GPU selection

## 🔧 Troubleshooting

### Audio Issues
- Ensure eSpeak NG and FFmpeg are properly installed
- Check debug logs (configurable in settings)
- Verify GPU/CPU resources are available

### Performance Optimization
- Enable audio cleanup to manage disk usage
- Adjust chunk splitting method based on your content
- Monitor debug logs for performance insights

## 📜 License & Credits

## 🙏 Acknowledgments

### Original Author
Special thanks to **h43lb1t0** for creating the original KokoroTTS extension for text-generation-webui. This enhanced version builds upon their foundational work in integrating Kokoro TTS with the Oobabooga interface.

- **Original Repository**: [h43lb1t0/KokoroTtsTexGernerationWebui](https://github.com/h43lb1t0/KokoroTtsTexGernerationWebui)
- **Original Concept**: Text segmentation and Kokoro model integration

### Additional Credits
- **Kokoro TTS Model**: [hexgrad](https://huggingface.co/hexgrad) for the Kokoro-82M model
- **Text Generation WebUI**: [oobabooga](https://github.com/oobabooga/text-generation-webui) for the base platform
- **Community Contributors**: Various community members who provided feedback and suggestions

### Project License
This enhanced extension is released under the [MIT License](LICENSE)

### Original Sources
- **Base Extension**: [h43lb1t0/KokoroTtsTexGernerationWebui](https://github.com/h43lb1t0/KokoroTtsTexGernerationWebui)
- **Kokoro Model**: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0 License)
- **Model Weights**: Downloaded directly from Hugging Face under Apache 2.0 License

### Enhancement Credits
- Enhanced architecture and audio system
- Integrated settings and UI improvements
- Performance optimizations and cleanup systems
- Advanced text processing and error handling

## 🤝 Contributing

Contributions are welcome! This version focuses on stability and clean architecture. Please:

1. **Fork the repository**
2. **Create a feature branch**
3. **Submit a pull request** with detailed description
4. **Follow existing code style** and documentation patterns

## 🗺️ Roadmap

- **Multi-language Support** - Expand beyond English
- **Voice Mixing** - Blend multiple voice characteristics
- **Advanced Effects** - Additional audio processing options
- **API Integration** - External TTS service support

---

**Version**: v1.0 - Enhanced Edition
**Compatibility**: Text Generation WebUI (Oobabooga)
**Model**: Kokoro-82M (English)
**Status**: Stable Release
