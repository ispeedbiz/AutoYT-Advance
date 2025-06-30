# 🛠️ Complete Setup Guide

This guide covers all technical requirements and setup steps for the YouTube Auto pipeline.

## 📋 Prerequisites

### System Requirements
- **OS**: macOS, Windows, or Linux
- **Python**: 3.11 (recommended) or 3.12
- **RAM**: Minimum 8GB, recommended 16GB+
- **GPU**: NVIDIA GPU with 6GB+ VRAM (for ComfyUI)
- **Storage**: 10GB+ free space
- **Internet**: Stable connection for API calls

### Software Requirements
- **Python 3.11**: [Download here](https://www.python.org/downloads/)
- **Git**: [Download here](https://git-scm.com/downloads)
- **FFmpeg**: For video processing
- **ComfyUI**: For AI image generation

## 🎨 ComfyUI Setup

### 1. Install ComfyUI

```bash
# Clone ComfyUI repository
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Download Required Models

Create the following directory structure in ComfyUI:
```
ComfyUI/
├── models/
│   ├── checkpoints/
│   │   └── sd_xl_base_1.0.safetensors  # Download from HuggingFace
│   └── vae/
│       └── sdxl_vae.safetensors        # Download from HuggingFace
```

**Download Links:**
- **SDXL Base Model**: [Download here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors) (6.9GB)
- **SDXL VAE**: [Download here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_vae.safetensors) (335MB)

### 3. Start ComfyUI

```bash
# Navigate to ComfyUI directory
cd ComfyUI

# Start ComfyUI server
python main.py --listen 0.0.0.0 --port 8188
```

**Verify Installation:**
- Open browser: `http://localhost:8188`
- You should see the ComfyUI interface
- Load a workflow and test image generation

### 4. GPU Configuration

**For NVIDIA GPUs:**
```bash
# Install CUDA toolkit (if not already installed)
# Download from: https://developer.nvidia.com/cuda-downloads

# Verify GPU detection
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

**For Apple Silicon (M1/M2):**
```bash
# Install PyTorch for Apple Silicon
pip install torch torchvision torchaudio
```

**For CPU-only:**
```bash
# Install CPU version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📊 Google Sheets Setup

### 1. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable the following APIs:
   - Google Sheets API
   - Google Drive API
   - YouTube Data API v3

### 2. Create Service Account

1. In Google Cloud Console, go to **IAM & Admin** > **Service Accounts**
2. Click **Create Service Account**
3. Fill in details:
   - **Name**: `youtube-auto-sheets`
   - **Description**: `Service account for YouTube Auto pipeline`
4. Click **Create and Continue**
5. Grant **Editor** role
6. Click **Done**

### 3. Generate Service Account Key

1. Click on the created service account
2. Go to **Keys** tab
3. Click **Add Key** > **Create New Key**
4. Select **JSON** format
5. Download the JSON file
6. Rename to `service_account.json`
7. Place in project root directory

### 4. Create Google Sheet

1. Go to [Google Sheets](https://sheets.google.com/)
2. Create new sheet named: **"Back to Zero – Input Sheet"**
3. Share the sheet with your service account email (found in `service_account.json`)
4. Grant **Editor** permissions

### 5. Sheet Structure

Your Google Sheet should have these columns (A to S):

| Column | Header | Description |
|--------|--------|-------------|
| A | Book/Topic | Main topic for the video |
| B | Language | Hindi or English |
| C | Custom Title (optional) | Video title |
| D | Notes (optional) | Additional notes |
| E | Publish Date | YYYY-MM-DD HH:MM format |
| F | Status | Auto-filled by pipeline |
| G | Link | Auto-filled with YouTube URL |
| H | Timestamp | Auto-filled |
| I | Error | Auto-filled if errors occur |
| J | Processing Time | Auto-filled |
| K | Thumbnail Text | Text for thumbnail |
| L | Style/Tone | Motivational, Storytelling, etc. |
| M | Key Highlights | Main points |
| N | Target Audience | Who should watch |
| O | CTA | Call to action |
| P | Duration | Short, Medium, Long |
| Q | Thumbnail Source | AUTO, BOOK, PERSON |
| R | Attribution Note | Credits |
| S | Image URL | Optional image URL |

## 🎬 YouTube API Setup

### 1. Enable YouTube Data API

1. In Google Cloud Console, go to **APIs & Services** > **Library**
2. Search for "YouTube Data API v3"
3. Click **Enable**

### 2. Create OAuth 2.0 Credentials

1. Go to **APIs & Services** > **Credentials**
2. Click **Create Credentials** > **OAuth 2.0 Client IDs**
3. Select **Desktop application**
4. Name: `YouTube Auto Upload`
5. Download the JSON file
6. Rename to `client_secret.json`
7. Place in project root directory

### 3. Generate Access Token

```bash
# Run the token generation script
python utils.py
```

This will:
1. Open browser for Google OAuth
2. Ask you to authorize the application
3. Generate `token.pickle` file
4. Enable YouTube uploads

## 🔧 Environment Configuration

### 1. Create Environment File

```bash
# Copy example environment file
cp env.example .env
```

### 2. Configure Environment Variables

Edit `.env` file with your API keys:

```bash
# OpenAI API Key (for script generation)
OPENAI_API_KEY=sk-your-openai-api-key-here

# ElevenLabs API Key (for TTS)
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here

# ElevenLabs Voice IDs
HINDI_VOICE_ID=your-hindi-voice-id
ENGLISH_VOICE_ID=your-english-voice-id
```

### 3. Get API Keys

**OpenAI API Key:**
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up/login
3. Go to **API Keys**
4. Create new secret key
5. Copy and paste into `.env`

**ElevenLabs API Key:**
1. Go to [ElevenLabs](https://elevenlabs.io/)
2. Sign up/login
3. Go to **Profile** > **API Key**
4. Copy your API key
5. Paste into `.env`

**ElevenLabs Voice IDs:**
1. In ElevenLabs, go to **Voice Library**
2. Select a voice
3. Copy the Voice ID from URL or settings
4. Paste into `.env`

## 🎵 Background Music Setup

### 1. Add Background Music

Place your background music files in the `backgrounds/` directory:

```bash
backgrounds/
├── Pensive Piano - Audionautix.mp3
├── Serenity - Aakash Gandhi.mp3
├── Renunciation - Asher Fulero.mp3
└── ... (add more tracks)
```

**Recommended Sources:**
- [Audionautix](https://audionautix.com/) (Free with attribution)
- [YouTube Audio Library](https://studio.youtube.com/channel/UC/music)
- [Free Music Archive](https://freemusicarchive.org/)

### 2. Attribution

For tracks requiring attribution, the pipeline automatically creates `music_attribution.txt` files.

## 🚀 Installation Steps

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd Youtube_auto
```

### 2. Create Virtual Environment

```bash
# Create Python 3.11 virtual environment
python3.11 -m venv venv311

# Activate environment
source venv311/bin/activate  # On macOS/Linux
# or
venv311\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Test all components
python -c "
from main import PYDUB_AVAILABLE, MOVIEPY_AVAILABLE
from comfyui_integration.generate_image import generate_image
from utils import load_env
print('✅ All systems ready!')
print(f'PYDUB_AVAILABLE: {PYDUB_AVAILABLE}')
print(f'MOVIEPY_AVAILABLE: {MOVIEPY_AVAILABLE}')
"
```

## 🔍 Troubleshooting

### ComfyUI Issues

**ComfyUI not starting:**
```bash
# Check if port 8188 is available
lsof -i :8188

# Kill process if needed
kill -9 <PID>

# Start ComfyUI again
python main.py --listen 0.0.0.0 --port 8188
```

**GPU not detected:**
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Google Sheets Issues

**Permission denied:**
1. Check service account email in `service_account.json`
2. Ensure sheet is shared with service account
3. Verify sheet name matches exactly: "Back to Zero – Input Sheet"

**API quota exceeded:**
1. Check Google Cloud Console quotas
2. Enable billing if needed
3. Request quota increase

### YouTube Upload Issues

**Token expired:**
```bash
# Regenerate token
rm token.pickle
python utils.py
```

**Upload failed:**
1. Check `client_secret.json` exists
2. Verify YouTube API is enabled
3. Check internet connection

### Audio Issues

**Background music not working:**
```bash
# Check Python version (should be 3.11)
python --version

# Reinstall pydub
pip uninstall pydub
pip install pydub
```

## 📞 Support

If you encounter issues:

1. **Check logs**: Look for error messages in terminal output
2. **Verify setup**: Run the verification commands above
3. **Check documentation**: Review this setup guide
4. **Common issues**: See troubleshooting section
5. **Create issue**: If problem persists, create a GitHub issue

## 🎯 Next Steps

After setup:

1. **Test the pipeline**: Run `python main.py`
2. **Add video topics**: Fill your Google Sheet
3. **Monitor progress**: Check sheet status updates
4. **Review results**: Check generated videos in `videos/` directory

---

**Happy content creation! 🎬✨** 