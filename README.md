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

An AI-powered vertical (`9:16`, `1088x1920`) video greeting studio that transforms natural-language prompts into personalized, broadcast-grade MP4 greeting videos. Built with a portable, modular Python architecture combining **Google Gemini 2.5 Flash** creative direction, **FLUX.1 / Z-Image-Turbo** artwork generation, a **4-Tier Multilingual Voiceover Waterfall**, and **local FFmpeg + Pillow** visual choreography.

Designed to run seamlessly across **Hugging Face Spaces**, **Kaggle Notebooks**, **Google Colab**, **Azure / Docker**, and **Local Laptops (Windows, macOS, Linux)**.

---

## ✨ Key Capabilities

1. **AI Creative Director (`ai_director.py` & `AI_prompt.txt`)**
   * Powered by `gemini-2.5-flash` with structured JSON schema output and automatic retry resilience.
   * Dynamically designs a 3-layer poetic visual scene (Foreground, Midground Hero, Background Vista), color-harmonized luxury glass card palette, typography pairing, and emotionally adaptive greeting copy (celebratory, health recovery, comforting, or solemn).
   * Enforces self-contained headlines (`overlay_title`), concise 1–2 line blessings (`overlay_subtitle`), and smooth spoken scripts (`tts_script`) free of hesitation-triggering punctuation.

2. **Native Multilingual Support (11 Indian Languages + Global English)**
   * **Supported Languages:** English (`en-IN` / `en-US`), Hindi (`hi-IN`), Tamil (`ta-IN`), Telugu (`te-IN`), Kannada (`kn-IN`), Malayalam (`ml-IN`), Bengali (`bn-IN`), Marathi (`mr-IN`), Gujarati (`gu-IN`), Punjabi (`pa-IN`), and Odia (`od-IN`).
   * **Native Script Overlay & Complex Text Shaping:** Automatically detects Indic Unicode blocks and renders crisp regional typography using Google Noto Sans Indic fonts + `libraqm` complex ligature shaping (`local_renderer.py`).

3. **Fault-Tolerant 4-Tier TTS Voice Waterfall (`media_services.py` & `config.py`)**
   * Includes a pre-TTS punctuation sanitizer (`_sanitize_tts_text`), multi-chunk base64 audio joining, and `48kHz` broadcast sample rate:
   * **Regional Indian Languages:**
     1. **Tier 1:** **Sarvam AI (`bulbul:v4-flash`)** — Expressive narration personas (`tanya_hi_narration` / `tarun_te_narration`) with `enable_preprocessing: True` at `48kHz`.
     2. **Tier 2:** **Sarvam AI (`bulbul:v3`)** — Dravidian-tuned (`kavitha`, `shruti`, `mani`, `gokul`, `vijay`) and General Indian speaker pools at `48kHz` with stabilized temperature (`0.35`).
     3. **Tier 3:** **Azure Regional Neural Voices (`edge-tts`)** — Zero-key native neural voices across all 11 Indian languages.
     4. **Tier 4:** **ElevenLabs (`eleven_multilingual_v2`)** — Final backup (`Matilda` / `Will`).
   * **Indian English (`en-IN`):** `Sarvam AI (bulbul:v4-flash)` (`ritu_en_reels`, `simran_en_narration`, `sunny_en_social`, `shubh_en_sports`) $\rightarrow$ `Azure Neural` $\rightarrow$ `ElevenLabs`.
   * **Universal / Global English:** `ElevenLabs` $\rightarrow$ `Azure Neural` $\rightarrow$ `Sarvam AI (bulbul:v4-flash)`.

4. **9 Curated Visual Art Styles**
   * `Auto - AI Choice`, `Cinematic`, `Movie Still`, `Bokeh & Light Trails`, `Oil Painting`, `Sci-Fi Futuristic`, `35mm Analog`, `Luminous Gouache`, `Magical Realism`, and `Luxury Still-Life`.

5. **Dynamic Video Choreography & Multi-Track Audio Ducking (`local_renderer.py`)**
   * **5 Randomized Intro Materialization Effects (Anti-Repeat):** *Mosaic Pixel Resolve*, *Iris Portal Bloom*, *Slit Tile Blinds*, *Golden Shimmer Sweep*, and *Clean Cinematic* (all preserving a 100% sharp Frame-0 thumbnail for WhatsApp previews).
   * **3 Randomized Camera Presets (Anti-Repeat):** *Slow Push-In*, *Slow Pull-Back Reveal*, and *Horizon Float with Warm Luminance Breathing*.
   * **3 Randomized Card & Text Unveils (Anti-Repeat):** *Soft Vertical Float*, *Gentle Cascade*, and *Pure Silk Dissolve*, positioned cleanly above mobile video player scrubber bars.
   * **Multi-Track BGM Library:** Mood-matched background music (`festive`, `calm`, `energetic`, `elegant`, `spiritual`, `romantic`, `sad`) with live URL availability checks, anti-repeat shuffling, and dynamic FFmpeg audio ducking during voiceover playback.

6. **Cross-Platform Black-Box Telemetry (`app.py`)**
   * Thread-safe execution logger records every run (`user_prompt`, `image_gen_prompt`, `tts_script`, `overlay_title`, `overlay_subtitle`, active TTS tier/model, and choreography presets) into `execution_logs.json`.
   * Optionally syncs `execution_logs.json` in the background to a private Hugging Face Dataset (`TRACE_REPO_ID`) with remote-reset detection.

---

## 🗂️ Project Structure

```text
├── app.py                   # Gradio Web UI, 4-step pipeline & background HF Dataset trace sync
├── config.py                # Env switchboard, 4-tier voice waterfall, style map, BGM selector & quota tracker
├── ai_director.py           # Gemini 2.5 Flash blueprint generator with adaptive tone & TTS rules
├── AI_prompt.txt            # System instruction & JSON schema rules for AI Creative Director
├── image_gen.py             # 9:16 artwork generation (FLUX.1-schnell / Z-Image-Turbo)
├── media_services.py        # 48kHz TTS waterfall (Sarvam v4/v3 -> Azure -> ElevenLabs) & sanitizer
├── local_renderer.py        # Pillow split-layer compositor & FFmpeg renderer (intro FX + camera + unveil)
├── creatomate_renderer.py   # Optional cloud rendering module
├── packages.txt             # Debian/Ubuntu system dependencies (ffmpeg, libraqm-dev, fonts-dejavu-core)
├── requirements.txt         # Python dependencies
└── env.example              # Environment variable configuration template