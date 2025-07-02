# AutoYT‑Advance

**Ultra‑Intelligent YouTube Automation Pipeline**

Create cinema‑quality videos in minutes—fully automated, cost‑efficient, and scalable.

---

## 📚 Table of Contents

1. Quick Overview
2. Core Features
3. Architecture & Folder Layout
4. System Requirements
5. Installation & First Run
6. Day‑to‑Day Usage
7. Troubleshooting Guide
8. Cost Breakdown
9. Testing & Validation
10. Support & License

---

## 1. Quick Overview

- **What it is:** 100  % automated pipeline that writes scripts, generates images, narrates, edits, and uploads YouTube videos.
- **Why it matters:** Cuts production cost from **₹8,000+** to **₹50–120** per video and reduces delivery time from days to **≈15 min**.
- **Who it is for:** Content creators, educators, and businesses that want consistent, high‑quality output without hiring a full studio.

---

## 2. Core Features

### 2.1 Content & Engagement Intelligence

| Capability               | What it does                                              | Result                     |
| ------------------------ | --------------------------------------------------------- | -------------------------- |
| **AI Content Analyst**   | Breaks scripts into logical scenes, predicts ideal length | Smooth pacing, no dead air |
| **Engagement Optimiser** | Uses YouTube retention patterns to fine-tune hooks & CTAs | Higher watch-time and CTR  |
| **LLM Video Planner (NEW)** | Generates full JSON plan (script, scenes, edit notes, thumbnail concept) | Enterprise-grade control  |
| **Self-Improving Loop**  | Automatically regenerates weak scenes                     | Continuous quality gains   |

### 2.2 Visual & Audio Engine

- **ComfyUI + SDXL** — 2 K+ Pixar‑style images with perfect visual consistency.
- **Dynamic Negative-Prompt Builder (NEW)** — Auto-adds context-aware negatives to reduce AI artefacts (extra limbs, blur, etc.).
- **ElevenLabs TTS** — Natural Hindi / English voices.
- **Smart Music Mixer** — Picks background tracks and balances volume automatically based on emotional arc.
- **Caption Builder** — Burned‑in bilingual subtitles using ImageMagick.

### 2.3 Automation & Integrations

- **Google Sheets** — One row = one video; status updates in real time.
- **YouTube Data API** — One‑click upload with title, description, tags, and scheduled publish time.
- **Systemd / PM2 templates** for hands‑free 24×7 operation.

### 2.4 Quality Control

- Four‑level fallback for every critical step.
- Detailed logs and feedback files inside each video folder.
- 100 % test coverage for content intelligence and engagement modules.

### 2.5 Quality & Self-Healing (NEW)

| Module | What it does | Benefit |
|--------|--------------|---------|
| **Multi-Stage QA** | Vision-GPT checks every generated image for blur, bad-anatomy etc. | Fewer unusable frames |
| **4-Level Fallback Chain** | Automatically retries with alternative prompt, style, or upscale | Near-100 % success rate |
| **Automated Feedback Loop** | Saves issues to `videos/…/feedback/` and regenerates weak scenes | Continuous improvement |

### Cinematic Scene Transitions (NEW)
* **Wide + Close-Up per Scene** – Two ultra-quality images for depth.
* **Slow Zoom / Pan** – Ken-Burns movement, 6-10 s, quadratic easing.
* **Gentle Cross-Fade** – Non-distracting blend to close-up.
* **Dynamic Duration** – Movement length auto-matches narration length.

### Frame-Interpolation Fallback
* **RIFE / DAIN** – Optional AI interpolation for extra-smooth motion.
* **Auto-Detect** – Falls back to traditional slideshow if GPU not available.

### 2.6  Dynamic Scaling & Caption System (NEW)
* **1–30 min Videos** – Scene count auto-scales; timing error ≤ 15 %.
* **Bilingual Captions** – Hindi (Noto Sans Devanagari) / English (Arial Bold) burned in via ImageMagick.

### 2.7 Advanced AI Features (NEW)
- **AI Scene Detection** — GPT-4-powered semantic boundary finding for perfect cuts.
- **Dynamic Pacing Optimisation** — Adjusts scene length based on emotional intensity.
- **Retention Optimisation** — Places engagement hooks at known drop-off points.
- **Vision-GPT QA** — Multi-criteria image filter (blur, bad anatomy, style drift).
- **Automated Prompt Improvement** — Re-writes prompts and re-generates images until QA passes.

---

## 3. Architecture & Folder Layout

```
Youtube_auto/
├─ main.py                 # Pipeline orchestration
├─ content_intelligence.py # Scene & duration logic
├─ engagement_optimizer.py # Retention logic
├─ comfyui_integration/    # Image generation helpers
├─ videos/                 # Output (one sub‑folder per video)
├─ backgrounds/            # Royalty‑free music
└─ ...                     # See full tree in docs/
```

A visual flowchart can be generated anytime by running:

```bash
python pipeline_flow_diagram.py
```

---

## 4. System Requirements

| Component | Minimum   | Recommended              |
| --------- | --------- | ------------------------ |
| Python    | 3.9       | **3.11**                 |
| FFmpeg    | 4.x       | Latest                   |
| ComfyUI   | Any       | Latest commit + SDXL 1.0 |
| RAM       | 8 GB      | 16 GB                    |
| GPU       | 4 GB VRAM | 8 GB+ VRAM (CUDA)        |
| Google Cloud Service Account | – | JSON key file for Sheets API 📄 |
| YouTube OAuth Client ID      | – | `client_secret.json` for YouTube upload |

> **Tip:** The pipeline runs on CPU but image generation is ~5× faster with an NVIDIA GPU.

---

### 🔑 Google API Credentials (NEW)
1. Create a **Service Account** in Google Cloud Console; enable *Google Sheets API*; download `service_account.json` into project root.
2. Enable the *YouTube Data API v3*; create OAuth client; download `client_secret.json`.
3. First run will open a browser—log in once; token is cached in `token.pickle`.

Full step-by-step guide in **[`docs/GITHUB_SETUP.md`](docs/GITHUB_SETUP.md)**.

---

## 5. Installation & First Run

1. **Clone and set up Python env:**
   ```bash
   git clone https://github.com/ispeedbiz/AutoYT-Advance.git
   cd AutoYT-Advance
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure API keys:**
   ```bash
   cp env.example .env  # then edit .env
   ```
3. **Install & launch ComfyUI:** (see docs/comfyui\_setup.md)
4. **Verify setup:**
   ```bash
   python verify_setup.py
   ```
5. **Generate your first video:**
   ```bash
   python main.py
   ```
   Watch the progress in terminal; output appears in `videos/`.

4. **Google API Keys:**
```bash
# Place JSON files
mv ~/Downloads/service_account.json .
mv ~/Downloads/client_secret.json .
```

### 5.1 ComfyUI (full guide)
Read **[`docs/COMFYUI_SETUP.md`](docs/COMFYUI_SETUP.md)** for GPU optimisation, model links and troubleshooting.

---

## 6. Day‑to‑Day Usage

| Task              | Command                                                 | Notes                                                |
| ----------------- | ------------------------------------------------------- | ---------------------------------------------------- |
| Run pipeline      | `python main.py`                                        | Reads next row marked **🕓 Waiting** in Google Sheet |
| Manual topic      | `python main.py --topic "Stoic Quotes" --duration 2m`   | Bypasses Sheet                                       |
| Regenerate images | `python regenerate_images.py --video 2025‑07‑02_123456` | Uses saved prompts                                   |
| Upload only       | `python upload_only.py --video <folder>`                | Skip rendering                                       |

---

## 7. Troubleshooting Guide

### ComfyUI errors

- **Symptoms:** `Connection refused`, blank images.\
  **Fix:** Check if ComfyUI is running on port 8188 and model file exists in `checkpoints/`.

### moviepy import

- Use exactly `moviepy==1.0.3` on Python 3.11. Older or newer combos may fail.

### Vision-GPT rejects images

- **Symptoms:** Vision-GPT rejects images.\
  **Fix:** Check `videos/<folder>/feedback/` for reasoning; tweak `quality_threshold`.

### Interpolation too slow

- **Symptoms:** Interpolation too slow.\
  **Fix:** Use `--interpolation rife` or disable via `.env`.

### Caption text overflow

- **Symptoms:** Caption text overflow.\
  **Fix:** Adjust font-size / wrap in `utils.create_captioned_scene()`.

Full FAQ lives in `docs/troubleshooting.md`.

---

## 8. Cost Breakdown (2‑min video)

| Element                               | Cost (USD)   |
| ------------------------------------- | ------------ |
| OpenAI GPT‑4o                         | \$0.20       |
| ElevenLabs TTS                        | \$0.25       |
| DALL·E Thumbnail                      | \$0.04       |
| Local compute                         | \$0.10       |
| **Total**                             | **≈ \$0.60** |
| *Traditional studio cost: \$100‑500.* |              |

---

## 9. Testing & Validation

Run the full suite:

```bash
pytest -q                 # unit tests (100 % coverage on intelligence layers)
python verify_setup.py     # env & dependency check
```

Diagnostic tools:

```bash
python utils.py --diagnose videos/2025-07-02_Fitness_Journey
```

Outputs audio/image duration tables and pacing warnings.

---

## 10. Support & License

- **Issues:** Open a ticket or email `support@ispeedbiz.com`.
- **Docs:** Additional guides live in the `docs/` folder.
- **License:** MIT—free for commercial and personal use.

> Made with dedication by the **Back to Zero** team to empower Indian creators.

 **\*\*Connect with the Creator:\*\***

\- 👔 **\*\*LinkedIn\*\***: [Jagdish Lade]\([https://www.linkedin.com/in/jagdishlade/](https://www.linkedin.com/in/jagdishlade/))

\- 🎥 **\*\*YouTube Channel\*\***: [Subscribe & Support]\([https://www.youtube.com/@JagdishLade](https://www.youtube.com/@JagdishLade))

