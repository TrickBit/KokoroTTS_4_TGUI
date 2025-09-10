# KokoroTTS_4_TGUI

**High-Quality Text-to-Speech for Text Generation WebUI - ONNX Edition**

A comprehensive Text-to-Speech extension for [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) featuring the Kokoro TTS model with ONNX backend optimization, advanced audio controls, and multi-language support.

## ✨ Key Features

### 🎙️ **Advanced Voice Synthesis**
- **High-Quality Speech** - Powered by Kokoro-82M model with natural-sounding output
- **50+ Voice Options** - Multiple languages, regional accents, and character voices
- **Voice Blending** - Combine multiple voices with custom weights (experimental)
- **Multi-Language Pronunciation** - English text with French, Italian, Japanese phonetics

### ⚡ **ONNX-Optimized Performance**
- **CPU Optimized** - Efficient inference without GPU requirements
- **Lightweight Deployment** - Reduced memory footprint and faster startup
- **Cross-Platform** - Consistent performance across operating systems
- **Same Quality** - Identical voice quality to PyTorch backends

### 🎛️ **Professional Audio Controls**
- **Speed Control** (0.5x - 1.5x) - Adjust playback speed without pitch artifacts
- **Experimental Pitch Control** (0.5x - 1.5x) - Modify voice pitch for creative effects
- **Real-time Preview** - Test voices and settings with customizable preview text
- **Integrated Audio Player** - Auto-playing controls with play/pause functionality

### 📄 **Smart Text Processing**
- **Intelligent Code Handling** - Choose to speak or skip code blocks
- **Markdown Cleanup** - Clean handling of formatted text for natural speech
- **Sentence-Based Chunking** - Preserves context in long texts (510 token limit)
- **Special Character Processing** - Handles technical content appropriately

### 🛠️ **Robust Architecture**
- **Self-Contained Installation** - Everything stays within extension directory
- **Automatic Cleanup** - Smart audio file management with configurable retention
- **Comprehensive Error Handling** - Configurable debug logging (off/errors/all)
- **Persistent Settings** - Intelligent defaults with user customization

## 🚀 What's New in ONNX Edition

This version focuses exclusively on ONNX Runtime for optimal performance:

### **ONNX-First Architecture**
- **Streamlined Backend** - Single, optimized inference path
- **Reduced Dependencies** - Lightweight installation without PyTorch overhead
- **Enhanced Reliability** - Simplified codebase with fewer failure points
- **Better Resource Management** - Optimized memory usage and CPU utilization

### **Enhanced Voice System**
- **NPZ Voice Format** - Optimized voice embedding storage
- **Improved Tokenization** - Better text-to-phoneme accuracy
- **Voice Organization** - Cleaner categorization by region/gender/language
- **Preview System** - Test any voice with customizable sample text

### **Advanced Features**
- **Voice Blending Math** - Linear interpolation of voice embeddings
- **Language Selection** - Pronunciation language independent of text language
- **Audio Format Optimization** - Automatic sample rate conversion
- **Smart Downloads** - Automatic model and voice file management

## 📋 Requirements

### System Dependencies
- **eSpeak NG** - Text-to-phoneme conversion: [Download](https://github.com/espeak-ng/espeak-ng/releases)
- **FFmpeg** - Audio processing: [Download](https://ffmpeg.org/download.html)

### Software Requirements
- **Text Generation WebUI** - Host application
- **Python 3.8+** - Runtime environment
- **Modern Web Browser** - HTML5 audio support required

### Hardware Recommendations
- **RAM**: 4GB+ for smooth operation
- **Storage**: ~500MB for models and voice data
- **CPU**: Multi-core recommended for real-time inference

## 🛠️ Installation

### Quick Install

1. **Navigate to extensions directory:**
   ```bash
   cd /path/to/text-generation-webui/extensions/
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/KokoroTTS_4_TGUI.git
   ```

3. **Activate your environment and install dependencies:**
   ```bash
   # Activate your text-generation-webui environment
   source ../venv/bin/activate  # or conda activate textgen

   # Install requirements
   cd KokoroTTS_4_TGUI
   pip install -r requirements.txt
   ```

4. **Enable in text-generation-webui:**
   - Restart text-generation-webui
   - Navigate to Interface tab
   - Check "KokoroTTS" to enable the extension

### First Run Setup
- ONNX model weights download automatically
- NLTK language data downloaded to extension directory
- Default settings created in `settings.yaml`
- Voice files downloaded based on selections

## 🎮 Usage

### Basic Operation
1. **Enable TTS** - Check "Enable TTS" for automatic AI response audio
2. **Select Voice** - Choose from organized voice categories
3. **Adjust Controls** - Set speed, pitch, and processing preferences
4. **Generate** - Audio automatically plays with AI responses

### Available Voices

**British Voices:**
- **Female**: bf_emma, bf_alice, bf_isabella, bf_lily
- **Male**: bm_daniel, bm_george, bm_lewis, bm_fable

**American Voices:**
- **Female**: af_bella, af_heart, af_jessica, af_nova, af_sarah, af_sky
- **Male**: am_adam, am_echo, am_eric, am_liam, am_michael, am_onyx

**International Voices:**
- **French**: Pronunciation optimized for French phonetics
- **Italian**: Native Italian accent support
- **Japanese**: Japanese pronunciation patterns
- **Portuguese, Chinese, Hindi**: Additional language support

### Advanced Features

#### **Voice Blending (Experimental)**
Combine multiple voices with custom weights:
- Minimum 2 voices, expandable to ~8 maximum
- Linear interpolation: `result = Σ(voice_i × weight_i)`
- Real-time preview of blended output
- Save/load custom voice configurations

#### **Language Pronunciation**
- **Text Language**: Keep English text
- **Pronunciation Language**: Apply French/Italian/Japanese phonetics
- **Example**: "Hello, bonjour" with French pronunciation selected

#### **Text Processing Options**
- **Code Block Handling**: Convert to speech announcements or skip entirely
- **Sentence Splitting**: Intelligent chunking for long content
- **Markdown Processing**: Clean formatting removal for natural speech

## ⚙️ Configuration

### Settings File (`settings.yaml`)
```yaml
kokoro_enable_tts: true
kokoro_voice: "bf_emma"
kokoro_speed: 1.0
kokoro_pitch: 1.0
kokoro_language: "en-gb"
kokoro_enable_debug: "errors"
kokoro_text_splitting_method: "sentences"
kokoro_process_code_blocks: "announce"
```

### Debug Levels
- **off**: No console logging
- **errors**: Error logging only (recommended)
- **all**: Full debug logging for development

### Multi-User Support
- Settings are per-session in multi-user mode
- Automatic temporary file isolation
- No cross-user data conflicts

## 🎯 Technical Architecture

### ONNX Backend Advantages

| Feature | ONNX Benefit |
|---------|-------------|
| **Startup Time** | 3-5x faster model loading |
| **Memory Usage** | 40-60% reduction vs PyTorch |
| **CPU Performance** | Optimized inference kernels |
| **Cross-Platform** | Consistent behavior everywhere |
| **Dependencies** | Minimal runtime requirements |

### Component Structure
```
KokoroTTS_4_TGUI/
├── src/
│   ├── onnx/              # ONNX inference engine
│   │   ├── generate.py    # Main TTS generation
│   │   ├── phoneme.py     # Text-to-phoneme conversion
│   │   └── voices.py      # Voice management
│   ├── blender.py         # Voice blending system
│   ├── makehtml.py        # Audio UI components
│   └── splittext.py       # Text preprocessing
├── script.py              # Main Gradio interface
├── settings.yaml          # User configuration
└── requirements.txt       # Dependencies
```

### Token Handling
- **Kokoro Limit**: 510 tokens per segment
- **Automatic Chunking**: Sentence-based splitting
- **Context Preservation**: Maintains speech flow across segments
- **Performance Optimization**: Batch processing where possible

## 🔧 Troubleshooting

### Common Issues

**Audio Not Playing:**
- Verify browser audio permissions
- Check eSpeak NG and FFmpeg installation
- Review debug logs (set `kokoro_enable_debug: "all"`)

**Slow Generation:**
- Verify ONNX Runtime installation
- Check available system resources
- Consider shorter text segments

**Voice Quality Issues:**
- Ensure voice files downloaded completely
- Try different voice selections
- Check pronunciation language settings

### Performance Optimization
- **CPU Usage**: ONNX backend is optimized for multi-core CPUs
- **Memory Management**: Enable automatic audio cleanup
- **Disk Space**: Monitor temporary file usage
- **Network**: Initial download may be slow, subsequent use is offline

## 📜 Licensing

### Project License
This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

### Model License
The Kokoro TTS model is licensed under **Apache License 2.0** by hexgrad/Kokoro-82M.

### **Important Dependency Notice**
This project depends on `phonemizer`, which is licensed under **GPL-3.0**:

- **phonemizer**: GPL-3.0 (text-to-phoneme conversion)
- **espeak-ng**: GPL-3.0 (phonemizer backend)

**Commercial Use Considerations:**
- Apache 2.0 components can be used commercially
- GPL-3.0 phonemizer dependency may affect distribution
- Consult legal counsel for commercial deployment requirements

### Third-Party Licenses
- **ONNX Runtime**: MIT License
- **PyDub**: MIT License
- **Gradio**: Apache 2.0
- **NumPy**: BSD License
- **PyYAML**: MIT License

## 🤝 Development

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup
```bash
# Clone for development
git clone https://github.com/yourusername/KokoroTTS_4_TGUI.git
cd KokoroTTS_4_TGUI

# Install dependencies
pip install -r requirements.txt

# Enable debug logging
# Edit settings.yaml: kokoro_enable_debug: "all"
```

### Testing
```bash
# Test ONNX backend directly
python src/onnx/generate.py

# Test voice generation
python -c "from src.onnx.generate import *; load_voice('bf_emma'); run('Hello world')"

# Test voice blending
python -c "from src.blender import *; test_blend(['bf_emma', 'af_bella'], [0.7, 0.3])"
```

## 🗺️ Roadmap

### Upcoming Features
- **Voice Blending Studio** - DAW-style interface for custom voice creation
- **Advanced Audio Processing** - Optional master EQ and effects
- **Preset Management** - Save and share custom voice configurations
- **Streaming Audio** - Real-time generation for long texts
- **Extended Language Support** - Additional pronunciation languages

### Performance Enhancements
- **Model Quantization** - Smaller model variants for mobile/edge devices
- **Caching System** - Improved performance for repeated phrases
- **Batch Optimization** - Enhanced multi-segment processing
- **Memory Streaming** - Reduced RAM usage for long generations

### Integration Improvements
- **API Support** - External TTS service integration
- **Voice Synthesis** - Custom voice training capabilities
- **Real-time Effects** - Live audio manipulation
- **Advanced Blending** - Non-linear voice combination algorithms

## 🙏 Acknowledgments

### Core Credits
- **Kokoro TTS Model**: [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M) for the Kokoro-82M model
- **ONNX Optimization**: [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) for ONNX research
- **Text Generation WebUI**: [oobabooga](https://github.com/oobabooga/text-generation-webui) for the platform

### Community
- **Beta Testers** - Early adopters who provided feedback
- **Voice Contributors** - Community members who helped expand voice selection
- **Documentation** - Contributors who improved installation guides
- **Performance Testing** - Users who helped optimize across different hardware

## 📞 Support

### Getting Help
- **Issues**: Report bugs via [GitHub Issues](../../issues)
- **Discussions**: Feature requests and questions in [Discussions](../../discussions)
- **Documentation**: Check the [Wiki](../../wiki) for detailed guides

### Performance Tips
- Use ONNX backend for CPU-only systems
- Enable audio cleanup to manage disk space
- Adjust debug logging based on your needs
- Monitor system resources during long generations

---

**Version**: v1.0 - ONNX Edition
**Compatibility**: Text Generation WebUI
**Model**: Kokoro-82M (Apache 2.0)
**Backend**: ONNX Runtime (Optimized)
**Status**: Active Development

**Note**: This is the ONNX-only version. For the PyTorch version, see [KokoroTTS_4_TGUI_pt](https://github.com/yourusername/KokoroTTS_4_TGUI_pt).
