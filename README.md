<p align="center">
  <img src="assets/banner.png" width="600">
</p>

### **Forensic-Grade Conversation Analyzer & Speaker Identity Workbench**
### ⚡ Quick Start
```bash
# 1. Clone
git clone https://github.com/Rakile/VocalTrace.git
cd VocalTrace

# 2. Create environment (conda)
conda create -n vocaltrace python=3.12
conda activate vocaltrace

# 3. Install deps
conda install -c conda-forge ffmpeg==7.1.1
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 4. Run
python launch.py
```
---

## 🧭 About

---
>VocalTrace is an advanced forensic audio workbench that combines **Biometric Speaker Identification**, **Human-in-the-Loop Diarization**, and **LLM-based Analysis**.
>
>Unlike standard transcription tools, VocalTrace allows you to build a "Voice Bank" of known speakers, manually correct AI mistakes (which are then enforced as ground truth), and chat with the transcript using RAG (Retrieval-Augmented Generation) to extract evidence and contradictions.
>
> Speaker diarization, transcription, and automatic speech recognition using **Pyannote**, **OpenAI Whisper**, and **KBLab’s KB-Whisper** (auto-selected for Swedish audio).

## 🖥️ User Interface

_Segmented, speaker-labeled transcript with waveform-linked playback (“sync-on-click”)._

![VocalTrace-Transcript Tab](assets/Transcript_Tab.png)

_LLM summary, psychological profile, themes_

![VocalTrace-Analyze Tab](assets/Raw_JSON_Tab.png)

_Manage known speakers and biometric signatures_

![VocalTrace-Voice Bank Tab](assets/Voice_Bank_Tab.png)

_Lets you chat with the transcript using RAG + LLM_

![VocalTrace-Evidence Chat Tab](assets/Chat_With_Evidence_Tab.png)

_Precision audio labeling tool + commit corrected segments to Voice Bank_

![VocalTrace-Audio Snipper Workbench](assets/Audio_Snipper.png)

## 🌟 Key Features

🕵️‍♂️ Biometric Speaker Identification  
Create a voice bank. VocalTrace identifies real speakers across recordings using learned voiceprints.

✏️ Inline Transcript Editing  
Edit transcript lines directly with instant waveform sync and commit corrected segments as verified ground truth.

✂️ Audio Snipper Workbench  
Slice audio visually, correct diarization errors, and add verified samples to the Voice Bank.

🔒 Ground Truth Persistence  
Corrected segments become locked. Future runs only transcribe the unverified gaps.

🧠 LLM-Based Conversation Analysis  
Generate summaries, psychological profiles, conflict maps, and more using ChatGPT or Gemini.

💬 Evidence Chat (RAG)  
Ask: “Did John admit anything about the contract?”  
Receive answers with exact timestamps.

⏯️ Sync-on-Click  
Click any transcript line to jump the audio player to that moment.

## 🛠️ Installation

VocalTrace requires **Python 3.10-3.12** (Tested primarily on 3.12).  
A GPU is highly recommended for pyannote + Whisper.

### 1. Prerequisites
*   **NVIDIA GPU** (Recommended)
*   **Conda installed**
*   **HuggingFace Account:** (for pyannote models, and you must accept the user agreements for [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).)

### 2. Setup Environment

```bash
# 1. Create a clean Conda environment
conda create -n vocaltrace python=3.12
conda activate vocaltrace

# 2. Install FFmpeg (Required for TorchCodec)
# Why Conda FFMpeg? Pyannote & TorchCodec require system-level FFmpeg libraries. The `conda-forge` build bundles the correct versions, avoiding DLL load failures.
conda install -c conda-forge ffmpeg==7.1.1

# 3. Install PyTorch (GPU Version)
# CRITICAL: Do this BEFORE installing requirements.
# Pyannote 4.0.3 requires Torch 2.8.0. We point to the CUDA 12.8 wheel index:
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 4. Install Application
# This will install pyannote.audio, torchcodec, and the GUI.
pip install -r requirements.txt
```

### 3. API Keys
Create an `.env` file (or set system environment variables) with the following:

### 🔐 Diarization Model Keys
> **Default model:** `pyannote/speaker-diarization-community-1`  
> Requires: HF_TOKEN_TRANSCRIBE=hf_...

> **Optional premium model:** `pyannote/speaker-diarization-precision-2`  
> Requires: PYANNOTE_PRECISION2_API_KEY=sk_6..

### 🔐 LLM model API Keys for subscribed Analysis & Chat (Choose one or both)
> **ChatGpt 5.1:**  
> Requires: OPENAI_API_KEY=sk-...

> **Gemini 2.5-Pro:**  
> Requires: GEMINI_TRANSCRIBE_ANALYSIS_API_KEY=AIza...


## 🚀 Usage
Run the launcher from the project root:
```bash
python launch.py
```

### Workflow
1.  **Load Audio:** Open an MP3/WAV file.
2.  **Voice Bank:** (Optional) Select known voices in the "Voice Bank" tab to target specific people.
3.  **Run Analysis:** Select "Transcribe + analyze audio".
4.  **Refine:**
    *   Right-click any line in the transcript to **"Refine in Snipper"**. Verified segments become ground truth, meaning future transcriptions cannot overwrite them.
    *   Adjust the waveform, correct the text, and click **Commit**.
    *   This saves the segment as "Verified" (Green).
5.  **Chat:** Go to the "Chat" tab to ask questions about the conversation.

## 🐛 Troubleshooting

<ins>**"DLL Load Failed" on startup?**</ins>  
VocalTrace includes a `bootstrap.py` system to handle conflicts between Conda's FFmpeg and PySide6. Ensure you are running via `launch.py` or `src/main.py`, which triggers this fix automatically.

<ins>**FlashAttention error? ("RuntimeError: Failed to load CUDA kernels for FlashAttention")**</ins>  
Some environments auto-install `flash-attn`, which breaks Whisper on Windows.  
Fix: `pip uninstall flash-attn`

## ✍️ Roadmap
* [ ] Export analysis report as PDF/Word
* [ ] Batch processing / queue mode
* [ ] Visual heatmap of speaker overlap & confidence
* [ ] CLI interface for headless servers
> (Developer note: legacy CLI entrypoints still exist for experimentation.)
> > ```
> > usage: transcription_engine.py [-h] [--out OUT] [--language LANGUAGE]
> >                               [--model MODEL] [--hf-token HF_TOKEN]
> >                               [--num-speakers NUM_SPEAKERS] [--names NAMES]
> >                               [--fingerprints FINGERPRINTS] [--srt]
> >                               [--enable_analysis] [--gap GAP]
> >                               audio
> >```
> > ```
> > usage: analysis_engine.py [-h] [--analysis-language {sv,en,auto}]
> >                           [--backend {gemini,chatgpt}]
> >                           transcript_filename
> >```


## 📜 License
CC-BY-4.0 Licensed. See `LICENSE` for details.

---
⚠️ VocalTrace is under active development.  
APIs and internal file formats may change between versions.

Developed by **Rakile** with assistance **Gemini** (...and ChatGpt 😊).