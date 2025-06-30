# 🚀 AutoYT-Advance: Ultra-Intelligent YouTube Automation Pipeline

## ⚡ **The Most Advanced AI-Powered Video Generation System**

**AutoYT-Advance** is a revolutionary, **enterprise-grade YouTube automation pipeline** that creates **cinema-quality videos** with **zero human intervention**. This isn't just automation—it's **AI orchestration** at its finest.

### 🎯 **What Makes This Extraordinary?**

🧠 **AI Director**: Advanced scene detection with emotional arc analysis  
🎨 **Ultra-Quality Visuals**: ComfyUI integration for **2K+ resolution** images  
🗣️ **Professional Narration**: ElevenLabs with **natural Hindi/English** synthesis  
🎵 **Smart Music Selection**: Content-aware background music with **dynamic volume**  
🔄 **Self-Improving**: Automated feedback loops with **quality optimization**  
📊 **Cost-Effective**: **$0.50-1.39 per video** vs $100-500 traditional production  
⚡ **Lightning Fast**: **95% time savings** - 15 minutes vs 8-16 hours  
🗂️ **Production-Ready**: Enterprise folder structure with **security built-in**

### 🌟 **Unique Advanced Features**

- **🤖 AI Scene Detection**: Automatically identifies optimal scene breaks using GPT-4
- **🎬 Dynamic Pacing**: Adapts scene duration based on content complexity and emotion
- **🔍 Quality Control**: Multi-stage AI quality assessment with automatic regeneration
- **📈 Retention Optimization**: Strategic engagement hooks for maximum viewer retention
- **🎭 Emotional Intelligence**: Semantic analysis for mood-appropriate music selection
- **🖼️ Metaphor-Rich Visuals**: Advanced prompt engineering for cinematic storytelling
- **📱 Caption System**: Burned-in subtitles with Hindi/English font optimization
- **🔧 Video Interpolation**: RIFE/DAIN frame interpolation for smooth transitions

> **For Basic YouTube Automation**: Check out our simpler pipeline at [ispeedbiz/autoyt](https://github.com/ispeedbiz/autoyt) 
> 
> **For Enterprise-Grade AI Production**: You're in the right place! 🎬

## 🚀 Recent Updates & Fixes

### ✅ **Python 3.11 Compatibility Confirmed**
- **Issue Resolved**: moviepy import errors with Python 3.13
- **Solution**: Use Python 3.11 + moviepy 1.0.3 (stable version)
- **Status**: All features working including burned-in captions

### ✅ **Caption System Operational**
- **Hindi Captions**: Noto Sans Devanagari font support
- **English Captions**: Arial Bold font support  
- **Features**: Semi-transparent background, bottom positioning
- **Requirement**: ImageMagick (install with `brew install imagemagick`)

### ✅ **Background Music Volume Fixed**
- **Updated**: Volume levels from -20dB to -10dB range
- **Result**: More audible background music while preserving narration clarity
- **Smart Selection**: Content-aware music matching

### ✅ **Feedback Loop System Active**
- **Status**: Feedback files are being generated
- **Location**: `videos/[video_folder]/feedback/`
- **Structure**: Ready for team input and iterative improvement

## 🎬 Features

### 🤖 **Core Video Generation**
- **AI-Powered Script Generation** - Creates engaging, emotionally resonant scripts in Hindi/English
- **Advanced Image Generation** - Ultra-quality images with metaphor-rich prompts and ComfyUI integration
- **Professional Audio Narration** - ElevenLabs TTS with natural Hindi/English voice synthesis
- **Intelligent Background Music** - Content-aware music selection from 6 curated tracks
- **Dynamic Scene Duration** - Adaptive scene timing based on content type and emotional tone
- **Automated Video Assembly** - Seamless integration of images, audio, and background music
- **Burned-in Captions** - Multi-language support with proper fonts and positioning

### 🤖 **Advanced AI Features**
- **Semantic Content Analysis** - Deep understanding of script structure and emotional arcs
- **AI Scene Detection** - Intelligent scene break detection using GPT-4
- **Dynamic Pacing Optimization** - Content-aware timing adjustments for better engagement
- **Retention Optimization** - Strategic placement of engagement hooks and transitions
- **Automated Feedback Loop** - Continuous quality improvement through AI analysis
- **Quality Filtering** - Multi-criteria quality assessment for generated content

### 🎵 **Intelligent Audio Processing**
- **Content-Aware Music Selection** - Analyzes script content to select optimal background music
- **Dynamic Volume Adjustment** - Emotion-based volume balancing for optimal listening experience
- **6 Curated Music Tracks** - Professionally selected instrumental tracks for different moods:
  - **Pensive Piano** - Contemplative, low energy (stories, lessons)
  - **Serenity** - Peaceful, uplifting (motivation, reflection)
  - **Renunciation** - Dramatic, medium energy (tense content)
  - **T'as où les vaches** - Playful, positive (motivation, stories)
  - **Dreamland** - Dreamy, reflective (philosophical content)
  - **Allégro** - Energetic, fast tempo (exciting content)

### 📊 **Smart Content Optimization**
- **Duration-Aware Script Splitting** - Adapts scene count to target video length
- **Flexible Content Processing** - Respects natural breaks and narrative flow
- **Emotional Arc Analysis** - Optimizes pacing based on content emotional journey
- **Complexity Profiling** - Adjusts scene duration based on content complexity
- **User Feedback Integration** - Learns from user preferences for continuous improvement

### 🔧 **Production Features**
- **Google Sheets Integration** - Automated workflow management
- **YouTube Upload** - Direct upload with custom thumbnails and metadata
- **Batch Processing** - Handle multiple videos efficiently
- **Error Recovery** - Robust fallback mechanisms and error handling
- **Quality Assurance** - Multi-stage quality checks and validation

## 🎯 What's New

### 1. RIFE/DAIN Video Interpolation
- **Smooth transitions** between images using AI frame interpolation
- **Cinematic quality** with professional-looking video output
- **Configurable methods** - RIFE (faster) or DAIN (higher quality)
- **Automatic fallback** to traditional slideshow if interpolation fails

### 2. LLM Video Generation
- **Complete video planning** using advanced LLMs
- **Scene breakdowns** with detailed visual and audio instructions
- **Editing guidance** with professional post-production recommendations
- **Thumbnail concepts** generated from video content analysis

### 3. Automated Feedback Loop
- **Self-improving content** that regenerates failed outputs
- **Quality optimization** with multiple attempts until success
- **Comprehensive feedback tracking** with detailed reports
- **Intelligent prompt improvement** based on quality analysis

## 🎬 Duration-Aware Script Splitting

The system now intelligently splits scripts based on target video duration, ensuring optimal scene distribution:

### Duration-Based Scene Allocation
- **1 minute videos**: 5 scenes (~12 seconds each)
- **2 minute videos**: 8 scenes (~15 seconds each)  
- **3 minute videos**: 10 scenes (~18 seconds each)
- **5 minute videos**: 12 scenes (~25 seconds each)
- **Longer videos**: 15+ scenes (~20 seconds each)

### Smart Scene Distribution
- Automatically parses duration from Google Sheets data
- Splits scripts into optimally-sized blocks
- Ensures no empty scenes or overly long segments
- Maintains narrative flow and timing consistency
- Each scene limited to maximum 20 seconds for engagement

### Benefits
- ✅ **Efficient resource usage**: Only generates needed images
- ✅ **Better pacing**: Scenes match video duration requirements
- ✅ **Consistent timing**: Each scene optimized for viewer engagement
- ✅ **Automatic scaling**: Adapts to any video length
- ✅ **Quality assurance**: No empty or malformed scenes

## 📋 System Requirements

### **Recommended Setup**
- **Python 3.11** (confirmed compatible - recommended)
- **moviepy 1.0.3** (stable version that works)
- **ImageMagick** (for caption text rendering)
- **FFmpeg** (for video processing)
- **macOS/Linux** (tested and working)

### **Python Dependencies**
```bash
# Essential packages
openai>=1.0.0
elevenlabs>=2.0.0
moviepy==1.0.3  # Specific version for compatibility
pydub>=0.25.1
google-api-python-client>=2.0.0
gspread>=6.0.0
pillow>=10.0.0
requests>=2.31.0
```

## 🛠️ Installation

### **1. Prerequisites**
```bash
# Install system dependencies
brew install imagemagick ffmpeg  # macOS
# sudo apt-get install imagemagick ffmpeg  # Ubuntu/Debian
```

### **2. ComfyUI Setup (Ultra-Quality Image Generation)**

**⚠️ CRITICAL: ComfyUI server must be running before executing main.py**

```bash
# Install ComfyUI (one-time setup)
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install ComfyUI dependencies
pip install -r requirements.txt

# Download SDXL model (required for ultra-quality)
# Place in: ComfyUI/models/checkpoints/
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# Start ComfyUI server (MUST be running during video generation)
python main.py --listen 127.0.0.1 --port 8188

# Keep this terminal open - server must stay running!
# Access ComfyUI interface: http://127.0.0.1:8188
```

**🚨 Before Running main.py:**
1. ✅ Start ComfyUI server: `python main.py` in ComfyUI folder
2. ✅ Verify server running: Visit http://127.0.0.1:8188
3. ✅ Check model loaded: SDXL should appear in ComfyUI interface
4. ✅ Test workflow: Load `workflow/prompt_ultra_quality.json`

### **3. Python Environment Setup**
```bash
# Clone repository
git clone https://github.com/ispeedbiz/AutoYT-Advance.git
cd AutoYT-Advance

# Create Python 3.11 virtual environment
python3.11 -m venv venv311
source venv311/bin/activate  # On Windows: venv311\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **3. Configuration**
```bash
# Copy environment template
cp env.example .env

# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_api_key
# ELEVENLABS_API_KEY=your_elevenlabs_api_key  
# HINDI_VOICE_ID=your_hindi_voice_id
# ENGLISH_VOICE_ID=your_english_voice_id (optional)
```

### **4. Google Services Setup**
1. Download `service_account.json` for Google Sheets API
2. Download `client_secret.json` for YouTube API
3. Run authentication: `python -c "from utils import main; main()"`

### **5. Verification**
```bash
python verify_setup.py
```

## 🎬 Usage

### **Basic Usage**

**🔥 Quick Start (2 Terminal Setup):**

**Terminal 1 - Start ComfyUI Server:**
```bash
cd ComfyUI
python main.py --listen 127.0.0.1 --port 8188
# Keep this running! ✅ Server must stay active
```

**Terminal 2 - Run Video Generation:**
```bash
cd AutoYT-Advance
source venv311/bin/activate

# Verify ComfyUI is running
curl http://127.0.0.1:8188 || echo "❌ Start ComfyUI first!"

# Run the pipeline
python main.py
```

**⚡ Pro Tip:** Set up ComfyUI as a service for automatic startup:
```bash
# Create systemd service (Linux)
sudo nano /etc/systemd/system/comfyui.service

# Or use screen/tmux for persistent sessions
screen -S comfyui
cd ComfyUI && python main.py --listen 127.0.0.1 --port 8188
# Ctrl+A, D to detach
```

### **Caption Features**
The system now supports burned-in captions:
- **Automatic font selection** based on script language
- **Hindi scripts**: Noto Sans Devanagari font
- **English scripts**: Arial Bold font
- **Positioning**: Bottom center with semi-transparent background
- **Integration**: Seamlessly embedded in final videos

### **Advanced Features**

#### **Video Interpolation**
```python
from video_interpolation import create_interpolated_video

success = create_interpolated_video(
    image_paths=["scene1.png", "scene2.png"],
    audio_path="narration.mp3",
    output_path="final_video.mp4",
    interpolation_method="rife"  # or "dain"
)
```

#### **LLM Video Generation**
```python
from llm_video_generator import generate_video_plan

video_plan = generate_video_plan(
    topic="The Power of Morning Routines",
    tone="Motivational",
    duration="5 minutes",
    audience="Young professionals",
    highlights="Productivity, mental health, success habits",
    openai_api_key="your_api_key"
)
```

#### **Automated Feedback Loop**
```python
from automated_feedback_loop import create_feedback_loop

feedback_loop = create_feedback_loop(
    openai_api_key="your_api_key",
    max_attempts=3,
    quality_threshold=0.7
)
```

## 🔧 Project Structure

```
Youtube_auto/
├── main.py                      # Main pipeline orchestrator
├── utils.py                     # Core utilities & caption functions
├── prompt_utils.py              # Advanced prompt generation
├── quality_filter.py            # Quality control system
├── video_interpolation.py       # RIFE/DAIN interpolation
├── llm_video_generator.py       # LLM video planning
├── automated_feedback_loop.py   # Self-improving feedback
├── music_selector.py            # Intelligent music selection
├── semantic_analyzer.py         # Content analysis
├── ai_scene_detector.py         # AI-powered scene detection
├── comfyui_integration/         # Image generation
├── workflow/                    # ComfyUI workflows
├── backgrounds/                 # Background music files
├── docs/                        # Documentation
├── archive/                     # Legacy code and experiments
├── requirements.txt             # Python dependencies
├── env.example                  # Environment template
└── README.md                    # This file
```

## 🎯 Current Status

### ✅ **Working Features**
- ✅ Script generation (Hindi/English)
- ✅ Ultra-quality image generation
- ✅ Professional audio synthesis
- ✅ Background music integration
- ✅ Video assembly and upload
- ✅ Burned-in captions (Hindi/English)
- ✅ Automated feedback system
- ✅ Quality control pipeline
- ✅ Google Sheets integration
- ✅ YouTube API upload

### 🔧 **Technical Status**
- ✅ Python 3.11 compatibility confirmed
- ✅ moviepy 1.0.3 working (caption support)
- ✅ All dependencies stable
- ✅ ComfyUI integration operational
- ✅ Background music volume optimized
- ✅ Feedback loop structure implemented

## 🚨 Troubleshooting

### **moviepy Issues**
- **Problem**: `ModuleNotFoundError: No module named 'moviepy.editor'`
- **Solution**: Use Python 3.11 + moviepy 1.0.3
- **Command**: `pip install moviepy==1.0.3`

### **ComfyUI Issues**
- **Problem**: `Connection refused` or `ComfyUI not responding`
- **Solution**: 
  ```bash
  # Check if ComfyUI server is running
  curl http://127.0.0.1:8188 || echo "ComfyUI server not running"
  
  # Restart ComfyUI server
  cd ComfyUI
  python main.py --listen 127.0.0.1 --port 8188
  ```
- **Problem**: `Model not loaded` or generation fails
- **Solution**: Ensure SDXL model is in `ComfyUI/models/checkpoints/`
- **Problem**: `Image generation timeout`
- **Solution**: Increase `max_wait_sec` in `comfyui_integration/generate_image.py`

### **Caption Issues**
- **Problem**: `ImageMagick not found`
- **Solution**: `brew install imagemagick` (macOS) or `sudo apt-get install imagemagick` (Linux)

### **Audio Issues**
- **Problem**: pydub not working
- **Solution**: Use Python 3.11 (pyaudioop compatibility)

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify your Python version: `python --version` (should be 3.11.x)
3. Ensure virtual environment is activated
4. Review generated logs in `videos/` folders

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 💰 Cost Analysis

### **Quick Cost Breakdown for 2-Minute Video**

| Component | Cost per Video | Monthly (30 videos) | Type |
|-----------|----------------|---------------------|------|
| **OpenAI GPT-4** (Script + Prompts) | $0.15 - $0.25 | $4.50 - $7.50 | AI Service |
| **ElevenLabs TTS** (Voice) | $0.18 - $0.30 | $5.40 - $9.00 | AI Service |
| **OpenAI DALL-E** (Thumbnail) | $0.04 | $1.20 | AI Service |
| **ComfyUI** (Scene Images) | $0.00 | $0.00 | **FREE** (Local) |
| **Local Processing** | $0.10 - $0.20 | $3.00 - $6.00 | Hardware |
| **Software & Tools** | $0.00 | $0.00 | **FREE** (Open Source) |
| **TOTAL** | **$0.50 - $1.39** | **$15.00 - $41.70** | |

### **Value Comparison**
- **Traditional Production**: $100-500 per video, 8-16 hours
- **Our Pipeline**: $0.50-1.39 per video, 10-15 minutes
- **Savings**: 99% cost reduction, 95% time savings

📊 **[View Detailed Cost Analysis](COST_ANALYSIS.md)** - Complete breakdown with ROI calculations, scaling economics, and optimization strategies.

---

## DISCLAIMER

This channel uses AI-generated and public-domain content for educational use. Book summaries and visuals follow Indian copyright fair use policy (§52).

## 👨‍💻 Contributors

Made with ❤️ for Indian creators. **Back to Zero YouTube** 

**Connect with the Creator:**
- 👔 **LinkedIn**: [Jagdish Lade](https://www.linkedin.com/in/jagdishlade/)
- 🎥 **YouTube Channel**: [Subscribe & Support](https://www.youtube.com/@JagdishLade)

## 📜 License

MIT — free to use, modify, and contribute.

**Ready to create viral YouTube content with AI automation! 🚀**