# VocalTrace 🎙️🔎 (By Rakile in colaboration with Gemini)
**The Conversation Analyzer**

**VocalTrace** is an advanced forensic audio workbench that combines **Biometric Speaker Identification**, **Human-in-the-Loop Diarization**, and **LLM-based Analysis**.

Unlike standard transcription tools, VocalTrace allows you to build a "Voice Bank" of known speakers, manually correct AI mistakes (which are then enforced as ground truth), and chat with the transcript using RAG (Retrieval-Augmented Generation) to extract evidence and contradictions.

Speaker Diarization, Transcription and Audio Speech Recognition using Pyannote's "speaker-diarization-community-1"-model, OpenAI's "Whisper"-model, and KBLab's "Kb-Whisper"-model (if Swedish Audio).

![VocalTrace-Transcript Tab](assets/Transcript_Tab.png)
![VocalTrace-Analyze Tab](assets/Raw_JSON_Tab.png)
![VocalTrace-Voice Bank Tab](assets/Voice_Bank_Tab.png)
![VocalTrace-Voice Bank Tab](assets/Chat_With_Evidence_Tab.png)
![VocalTrace-Audio Snipper Workbench](assets/Audio_Snipper.png)
## 🌟 Key Features

*   **🕵️‍♂️ Biometric Identification:** Build a `voices.json` database. The AI identifies specific people (e.g., "Dad", "Suspect", "Customer") instead of just `SPEAKER_01`.
*   **✂️ Audio Snipper Workbench:** Visually slice audio, verify segments, and add them to your Voice Bank in seconds.
*   **🔒 Ground Truth Persistence:** Manually corrected segments are saved and **enforced**. Future AI runs will respect your edits and only transcribe the gaps.
*   **🧠 LLM Analysis:** Generates psychological profiles, conflict assessments, and summaries using ChatGPT or Gemini models.
*   **💬 Evidence Chat:** Ask questions like *"Did John mention the contract?"* and get answers cited with specific timestamps.
*   **⏯️ Sync-on-Click:** Click any line in the transcript to instantly jump the audio player to that moment for rapid verification.

## 🛠️ Installation

VocalTrace requires **Python 3.10+** (Tested on 3.12) and uses **Conda** to manage system dependencies.

### 1. Prerequisites
*   **NVIDIA GPU** (Recommended).
*   **[Miniconda](https://docs.conda.io/en/latest/miniconda.html)** installed.
*   **HuggingFace Account:** You must accept the user agreements for [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).

### 2. Setup Environment

```bash
# 1. Create a clean Conda environment
conda create -n vocaltrace python=3.12
conda activate vocaltrace

# 2. Install FFmpeg (Required for TorchCodec)
# We use conda-forge because it provides the necessary shared libraries.
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

```ini
# Required for Speaker Diarization
HF_TOKEN_TRANSCRIBE=hf_... 

# Required for Analysis & Chat (Choose one or both)
OPENAI_API_KEY=sk-...
GEMINI_TRANSCRIBE_ANALYSIS_API_KEY=AIza...
```

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
    *   Right-click any line in the transcript to **"Refine in Snipper"**.
    *   Adjust the waveform, correct the text, and click **Commit**.
    *   This saves the segment as "Verified" (Green).
5.  **Chat:** Go to the "Chat" tab to ask questions about the conversation.

## 🐛 Troubleshooting

**"DLL Load Failed" on startup?**
VocalTrace includes a `bootstrap.py` system to handle conflicts between Conda's FFmpeg and PySide6. Ensure you are running via `launch.py` or `src/main.py`, which triggers this fix automatically.

## 📜 License
CC-BY-4.0 Licensed. See `LICENSE` for details.

---