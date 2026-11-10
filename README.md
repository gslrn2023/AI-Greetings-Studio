---
title: AI Greetings AV Studio
emoji: 🎬
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 5.0.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# 🎬 AI Greetings AV Studio

An AI-powered vertical (`9:16`, `1088x1920`) video greeting studio that transforms natural-language prompts into personalized, broadcast-grade MP4 greeting videos. Built with a modular Python architecture combining **Google Gemini 2.5 Flash** creative direction, **FLUX.1 / Z-Image-Turbo** artwork generation, a **3-Tier Multilingual Voiceover Waterfall**, and **local FFmpeg + Pillow** motion choreography.

---

## ✨ Key Capabilities

1. **AI Creative Director (`ai_director.py`)**
   * Powered by `gemini-2.5-flash` with structured JSON schema output and automatic 30-second retry resilience.
   * Dynamically designs a 3-layer poetic visual scene (Foreground, Midground Hero, Background Vista), color-harmonized luxury card palette, typography pairing, and culturally authentic spoken greeting script.

2. **Native Multilingual Support (11 Indian Languages + Global English)**
   * **Supported Languages:** English (`en-IN`), Hindi (`hi-IN`), Tamil (`ta-IN`), Telugu (`te-IN`), Kannada (`kn-IN`), Malayalam (`ml-IN`), Bengali (`bn-IN`), Marathi (`mr-IN`), Gujarati (`gu-IN`), Punjabi (`pa-IN`), and Odia (`od-IN`).
   * **Native Script Overlay & Complex Text Shaping:** Automatically detects Indic Unicode blocks and renders crisp native regional typography using Google Noto Sans Indic fonts + `libraqm` complex ligature shaping (`local_renderer.py`).

3. **Fault-Tolerant 3-Tier TTS Voice Waterfall (`media_services.py` & `config.py`)**
   * **Indian Context / Regional Languages:**
     1. **Tier 1:** **Sarvam AI (`bulbul:v3`)** — 48kHz output with Dravidian-tuned and General Indian speaker pools and stabilized temperature/pace parameters.
     2. **Tier 2:** **Azure Regional Neural Voices (`edge-tts`)** — Free, zero-key native neural voices across all 11 Indian languages.
     3. **Tier 3:** **ElevenLabs (`eleven_multilingual_v2`)** — Final backup.
   * **Global Context:** `ElevenLabs` $\rightarrow$ `Azure Neural` $\rightarrow$ `Sarvam AI`.

4. **9 Curated Visual Art Styles**
   * `Auto - AI Choice`, `Cinematic`, `Movie Still`, `Bokeh & Light Trails`, `Oil Painting`, `Sci-Fi Futuristic`, `35mm Analog`, `Luminous Gouache`, `Magical Realism`, and `Luxury Still-Life`.

5. **Dynamic Video Choreography & Multi-Track Audio Ducking (`local_renderer.py`)**
   * **Randomized Camera Presets (Anti-Repeat):** *Slow Push-In*, *Slow Pull-Back Reveal*, and *Horizon Float with Warm Luminance Breathing*.
   * **Randomized Card & Text Unveils (Anti-Repeat):** *Soft Vertical Float*, *Gentle Cascade*, and *Pure Silk Dissolve*.
   * **Multi-Track BGM Library:** Mood-matched background music (`festive`, `calm`, `energetic`, `elegant`, `spiritual`, `romantic`, `sad`) with live URL availability verification, anti-repeat shuffling, and dynamic FFmpeg audio ducking during voiceover playback.
   * **WhatsApp Instant Preview Ready:** Frame-0 full-brightness rendering and `-movflags +faststart` metadata ensure rich thumbnails when shared on WhatsApp and mobile messengers.

---

## 🗂️ Project Structure

```text
├── app.py                   # Gradio Web UI & 4-step orchestration pipeline
├── config.py                # Environment variables, voice waterfall, style map, BGM selector & quota manager
├── ai_director.py           # Gemini 2.5 Flash blueprint generator with auto-retry
├── AI_prompt.txt            # System instruction & JSON schema rules for AI Creative Director
├── image_gen.py             # 9:16 artwork generation (FLUX.1-schnell / Z-Image-Turbo)
├── media_services.py        # 3-Tier TTS waterfall (Sarvam -> Azure -> ElevenLabs) & image hosting
├── local_renderer.py        # Pillow split-layer card/text compositor & FFmpeg video renderer
├── creatomate_renderer.py   # Optional cloud rendering module
├── packages.txt             # System-level dependencies (ffmpeg, libraqm-dev, fonts-dejavu-core)
├── requirements.txt         # Python dependencies
└── .env.example             # Environment variable template