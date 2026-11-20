---
title: AI Greetings AV Studio
emoji: 🎬
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
---

# 🎬 AI Greetings AV Studio

An AI-powered vertical (`9:16`, `1088x1920`) video greeting studio that transforms natural-language prompts into personalized, broadcast-grade MP4 greeting videos. Built with a portable, modular Python architecture combining **Google Gemini 2.5 Flash** creative direction, a **4-Tier Hardware-Agnostic Image Engine (`FLUX.1-schnell` / `Z-Image-Turbo`)**, a **4-Tier Multilingual Voiceover Waterfall**, and **local FFmpeg + Pillow** visual choreography.

Designed to run seamlessly across **Hugging Face Spaces**, **Kaggle Notebooks**, **Google Colab**, **Azure / Linux VMs**, and **Local Laptops (Windows, macOS, Linux)**.

---

## ✨ Key Capabilities

1. **AI Creative Director (`ai_director.py` & `AI_prompt.txt`)**
   * Powered by `gemini-2.5-flash` with structured JSON schema output and automatic retry resilience.
   * Dynamically designs a 3-layer poetic visual scene (Foreground, Midground Hero, Background Vista), color-harmonized luxury glass card palette, typography pairing, and emotionally adaptive greeting copy (celebratory, health recovery, comforting, or solemn).
   * Enforces self-contained headlines (`overlay_title`), concise 1–2 line blessings (`overlay_subtitle`), and smooth spoken scripts (`tts_script`) free of hesitation-triggering punctuation.

2. **4-Tier Hardware-Agnostic Image Engine (`image_gen.py`)**
   * Defaults to **`FLUX.1-schnell`** (`IMAGE_ENGINE=FLUX`) with automatic failover to **`Z-Image-Turbo`** (`Z_IMAGE`) on gated-token issues, hardware constraints, or runtime errors.
   * Inspects GPU count, single-card VRAM, combined VRAM, and CUDA Compute Capability at startup—zero vendor-specific configuration required.
   * Automatically activates multi-GPU balanced sharding (`device_map="balanced"`), `<32 GiB` combined-VRAM routing to `Z-Image-Turbo`, overflow-protected FP16 Tensor-Core dispatch on pre-Ampere GPUs (`sm_75`, e.g., Tesla T4), and tiled VAE decoding.

3. **Native Multilingual Support (11 Indian Languages + Global English)**
   * **Supported Languages:** English (`en-IN` / `en-US`), Hindi (`hi-IN`), Tamil (`ta-IN`), Telugu (`te-IN`), Kannada (`kn-IN`), Malayalam (`ml-IN`), Bengali (`bn-IN`), Marathi (`mr-IN`), Gujarati (`gu-IN`), Punjabi (`pa-IN`), and Odia (`od-IN`).
   * **Native Script Overlay & Complex Text Shaping:** Automatically detects Indic Unicode blocks and renders crisp regional typography using Google Noto Sans Indic fonts + `libraqm` complex ligature shaping (`local_renderer.py`).

4. **Fault-Tolerant 4-Tier TTS Voice Waterfall (`media_services.py` & `config.py`)**
   * Includes a pre-TTS punctuation sanitizer (`_sanitize_tts_text`), multi-chunk base64 audio joining, and `48kHz` broadcast sample rate:
   * **Regional Indian Languages:**
     1. **Tier 1:** **Sarvam AI (`bulbul:v4-flash`)** — Expressive narration personas (`tanya_hi_narration` / `tarun_te_narration`) with `enable_preprocessing: True` at `48kHz`.
     2. **Tier 2:** **Sarvam AI (`bulbul:v3`)** — Dravidian-tuned (`kavitha`, `shruti`, `mani`, `gokul`, `vijay`) and General Indian speaker pools at `48kHz` with stabilized temperature (`0.35`).
     3. **Tier 3:** **Azure Regional Neural Voices (`edge-tts`)** — Zero-key native neural voices across all 11 Indian languages.
     4. **Tier 4:** **ElevenLabs (`eleven_multilingual_v2`)** — Final backup (`Matilda` / `Will`).
   * **Indian English (`en-IN`):** `Sarvam AI (bulbul:v4-flash)` (`ritu_en_reels`, `simran_en_narration`, `sunny_en_social`, `shubh_en_sports`) → `Azure Neural` → `ElevenLabs`.
   * **Universal / Global English:** `ElevenLabs` → `Azure Neural` → `Sarvam AI (bulbul:v4-flash)`.

5. **9 Curated Visual Art Styles**
   * `Auto - AI Choice`, `Cinematic`, `Movie Still`, `Bokeh & Light Trails`, `Oil Painting`, `Sci-Fi Futuristic`, `35mm Analog`, `Luminous Gouache`, `Magical Realism`, and `Luxury Still-Life`.

6. **Dynamic Video Choreography & Multi-Track Audio Ducking (`local_renderer.py`)**
   * **5 Randomized Intro Materialization Effects (Anti-Repeat):** *Mosaic Pixel Resolve*, *Iris Portal Bloom*, *Slit Tile Blinds*, *Golden Shimmer Sweep*, and *Clean Cinematic* (all preserving a 100% sharp Frame-0 thumbnail for WhatsApp previews).
   * **3 Randomized Camera Presets (Anti-Repeat):** *Slow Push-In*, *Slow Pull-Back Reveal*, and *Horizon Float with Warm Luminance Breathing*.
   * **3 Randomized Card & Text Unveils (Anti-Repeat):** *Soft Vertical Float*, *Gentle Cascade*, and *Pure Silk Dissolve*, positioned cleanly above mobile video player scrubber bars.
   * **Multi-Track BGM Library:** Mood-matched background music (`festive`, `calm`, `energetic`, `elegant`, `spiritual`, `romantic`, `sad`) with live URL availability checks, anti-repeat shuffling, and dynamic FFmpeg audio ducking during voiceover playback.

7. **Cross-Platform Black-Box Telemetry (`app.py`)**
   * Thread-safe execution logger records every run (`user_prompt`, `image_gen_prompt`, `tts_script`, `overlay_title`, `overlay_subtitle`, active TTS tier/model, and choreography presets) into `execution_logs.json`.
   * Optionally syncs `execution_logs.json` in the background to a private Hugging Face Dataset (`TRACE_REPO_ID`) with remote-reset detection.

---

## 🖥️ Hardware Tiers & Expected Performance

`image_gen.py` automatically classifies the host machine at boot into one of four universal hardware tiers:

| Hardware Tier | Detection Rule | Typical Environments | Active Strategy | Approx. End-to-End Time |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: `HIGH_VRAM`** | HF ZeroGPU **or** Single GPU ≥ 22 GiB | HF ZeroGPU (H200/A100), Azure/Cloud A100/L40S/A10G, Local RTX 3090/4090 | Direct `pipe.to("cuda")` running primary `FLUX.1-schnell` | ~15 – 25 sec |
| **Tier 2: `MULTI_GPU`** | ≥ 2 GPUs with Combined VRAM ≥ 20 GiB | Kaggle (`GPU T4 x2` = 29.12 GiB), Dual-GPU Cloud/Local rigs | `device_map="balanced"` across GPUs; auto-routes `<32 GiB` to `Z-Image-Turbo` + `sm_75` FP16 Tensor-Core dispatch | ~2 – 2.5 min |
| **Tier 3: `SINGLE_LOW_VRAM`** | 1 GPU with < 22 GiB VRAM | Google Colab (`1x T4`), Azure `NC4as_T4_v3`, Local RTX 3060/4060/4070 (8–16 GB) | Component-level model CPU offload (≥14 GiB) or sequential offload (<14 GiB) + tiled VAE + `sm_75` FP16 dispatch | ~2 – 4 min |
| **Tier 4: `CPU`** | No CUDA GPU detected | Standard laptops without NVIDIA GPU, CPU containers | Standard CPU execution (`pipe.to("cpu")`) | Slow (CPU-bound) |

---

## ⚙️ Environment Variables

Copy `env.example` to `.env` (for local / VM deployments) or configure these as platform Secrets:

| Variable | Required? | Default | Purpose |
| :--- | :---: | :--- | :--- |
| `GEMINI_API_KEY` | **Yes** | — | Powers the AI Creative Director (`gemini-2.5-flash`). |
| `HF_TOKEN` | **Recommended** | — | Required for gated `black-forest-labs/FLUX.1-schnell` access and optional `TRACE_REPO_ID` dataset sync. |
| `SARVAM_API_KEY` | Optional | — | Enables Tier 1 & Tier 2 Indian TTS (`bulbul:v4-flash` / `bulbul:v3`). If omitted, falls back to free Azure Neural (`edge-tts`). |
| `ELEVENLABS_API_KEY` | Optional | — | Enables global English & fallback TTS (`eleven_multilingual_v2`). |
| `APP_PASSCODE` | Optional | `""` | Locks generation behind a UI passcode when set. Leave empty for open access. |
| `IMAGE_ENGINE` | Optional | `FLUX` | Primary image engine (`FLUX` or `Z_IMAGE`). The alternate engine is always used as automatic fallback. |
| `RENDER_ENGINE` | Optional | `LOCAL` | Video renderer (`LOCAL` FFmpeg/Pillow or `CREATOMATE`). |
| `MAX_DAILY_GREETINGS` | Optional | `3` | Daily greeting quota per deployment (resets midnight IST). |
| `CREATOMATE_API_KEY` | Optional | — | Only needed if `RENDER_ENGINE=CREATOMATE`. |
| `TRACE_REPO_ID` | Optional | — | Private Hugging Face Dataset repo ID for background `execution_logs.json` sync. |

> **Zero-Cost TTS Note:** If neither `SARVAM_API_KEY` nor `ELEVENLABS_API_KEY` is provided, the studio automatically uses **Azure Regional Neural Voices (`edge-tts`)** at zero API cost across all 11 Indian languages and English.

---

## 🚀 Platform Quickstarts

### 1. Hugging Face Spaces (ZeroGPU / Dedicated GPU)
1. Duplicate or push this repository to a Gradio Space (keep the YAML header at the top of `README.md` intact).
2. In **Settings → Hardware**, select **ZeroGPU** (recommended) or a dedicated GPU (≥ 24 GiB VRAM).
3. In **Settings → Variables and secrets**, add `GEMINI_API_KEY` and `HF_TOKEN` (plus optional `SARVAM_API_KEY`, `ELEVENLABS_API_KEY`, `APP_PASSCODE`).
4. Hugging Face automatically installs system packages from `packages.txt` and Python packages from `requirements.txt`.

---

### 2. Kaggle Notebooks (`GPU T4 x2`)
1. Create a new Kaggle Notebook. In the right sidebar, set **Accelerator → GPU T4 x2** and **Internet → On**.
2. Go to **Add-ons → Secrets**, add your keys (`GEMINI_API_KEY`, `HF_TOKEN`, and any optional keys), and toggle them **On** for the notebook.
3. Run this single cell to clone, install, and launch (Gradio automatically generates a public `.gradio.live` URL):

```python
import os
from kaggle_secrets import UserSecretsClient

# 1. Export secrets from Kaggle Add-ons -> Secrets
secrets = UserSecretsClient()
for key in ["APP_PASSCODE", "GEMINI_API_KEY", "HF_TOKEN", "SARVAM_API_KEY", "ELEVENLABS_API_KEY"]:
    try:
        os.environ[key] = secrets.get_secret(key)
    except Exception:
        pass

# 2. Clone (or sync latest commit), install dependencies, and launch
if not os.path.exists("/kaggle/working/AI-Greetings-Studio"):
    %cd /kaggle/working
    !git clone [https://github.com/gslrn2023/AI-Greetings-Studio.git](https://github.com/gslrn2023/AI-Greetings-Studio.git)

%cd /kaggle/working/AI-Greetings-Studio
!git fetch origin main && !git reset --hard origin/main
!pip install -q -r requirements.txt
!python app.py
```

> **Kaggle Tip:** Keep the notebook tab active while generating and download your `.mp4` promptly—Kaggle suspends idle interactive tunnels after ~20–30 minutes of inactivity.

---

### 3. Google Colab (`T4` / `L4` / `A100`)
1. Open a Colab notebook and select **Runtime → Change runtime type → T4 / L4 / A100 GPU**.
2. Open the **Secrets (🔑)** panel in the left sidebar and add your API keys with **Notebook access** enabled.
3. Run this cell:

```python
import os
from google.colab import userdata

for key in ["APP_PASSCODE", "GEMINI_API_KEY", "HF_TOKEN", "SARVAM_API_KEY", "ELEVENLABS_API_KEY"]:
    try:
        val = userdata.get(key)
        if val:
            os.environ[key] = val
    except Exception:
        pass

!git clone [https://github.com/gslrn2023/AI-Greetings-Studio.git](https://github.com/gslrn2023/AI-Greetings-Studio.git)
%cd AI-Greetings-Studio
!apt-get -qq update && apt-get -qq install -y ffmpeg libraqm-dev fonts-dejavu-core
!pip install -q -r requirements.txt
!python app.py
```

---

### 4. Azure / Cloud Linux VM (Ubuntu / Debian GPU VM)
Recommended SKUs: `Standard_NC4as_T4_v3` (1x T4), `Standard_NV36adms_A10_v5` (A10 24 GB), or any NVIDIA CUDA Linux VM.

```bash
# 1. Install system dependencies (required for FFmpeg video rendering & Indic font shaping)
sudo apt-get update && sudo apt-get install -y ffmpeg libraqm-dev fonts-dejavu-core git python3-venv

# 2. Clone repository & create virtual environment
git clone [https://github.com/gslrn2023/AI-Greetings-Studio.git](https://github.com/gslrn2023/AI-Greetings-Studio.git)
cd AI-Greetings-Studio
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# 3. Configure secrets
cp env.example .env
nano .env   # Add GEMINI_API_KEY, HF_TOKEN, etc.

# 4. Launch app
python app.py
```

* Access the UI via `http://<VM_PUBLIC_IP>:7860` (ensure TCP port `7860` is allowed in your Azure NSG / firewall, or use an SSH tunnel: `ssh -L 7860:localhost:7860 user@<VM_IP>`).

---

### 5. Local Laptop / Desktop (Windows, macOS, Linux)

**Prerequisites:** Python `3.10+`, Git, and **FFmpeg** installed on your system `PATH` (an 8–16+ GB NVIDIA CUDA GPU is strongly recommended; CPU-only machines will run slowly).

1. **Install FFmpeg:**
   * **Windows:** `winget install Gyan.FFmpeg` (then restart your terminal)
   * **macOS:** `brew install ffmpeg libraqm`
   * **Ubuntu / Debian:** `sudo apt-get install -y ffmpeg libraqm-dev fonts-dejavu-core`

2. **Clone, Configure & Run:**
   ```bash
   git clone [https://github.com/gslrn2023/AI-Greetings-Studio.git](https://github.com/gslrn2023/AI-Greetings-Studio.git)
   cd AI-Greetings-Studio

   python -m venv .venv
   # Activate on Windows (PowerShell): .venv\Scripts\Activate.ps1
   # Activate on macOS / Linux:       source .venv/bin/activate

   pip install -r requirements.txt
   cp env.example .env
   ```
   Edit `.env` to add your `GEMINI_API_KEY` (and optional keys), then launch:
   ```bash
   python app.py
   ```
   Open `http://localhost:7860` in your browser.

---

## 🗂️ Project Structure

```text
├── app.py                   # Gradio Web UI, 4-step pipeline & background HF Dataset trace sync
├── config.py                # Env switchboard, 4-tier voice waterfall, style map, BGM selector & quota tracker
├── ai_director.py           # Gemini 2.5 Flash blueprint generator with adaptive tone & TTS rules
├── AI_prompt.txt            # System instruction & JSON schema rules for AI Creative Director
├── image_gen.py             # 4-tier hardware-agnostic 9:16 image engine (FLUX.1-schnell / Z-Image-Turbo)
├── media_services.py        # 48kHz TTS waterfall (Sarvam v4/v3 -> Azure -> ElevenLabs) & sanitizer
├── local_renderer.py        # Pillow split-layer compositor & FFmpeg renderer (intro FX + camera + unveil)
├── creatomate_renderer.py   # Optional cloud rendering module
├── packages.txt             # Debian/Ubuntu system dependencies (ffmpeg, libraqm-dev, fonts-dejavu-core)
├── requirements.txt         # Python dependencies
└── env.example              # Environment variable configuration template
```