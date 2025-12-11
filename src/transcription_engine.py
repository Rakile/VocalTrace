#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transcribe with Whisper + Pyannote.
Architecture: Diarization-First (Split-and-Transcribe).
Style: 'Natural Flow' (Pause-based paragraphs).
Feature: Added Progress Bar (tqdm).
"""

import os
import sys
import warnings
import argparse
import gc
import json
import subprocess
import numpy as np
import torch
import torchaudio
from datetime import timedelta
from typing import List, Dict, Any
from analysis_engine import analyze_conversation, call_llm_openai, call_llm_grokai, normalize_analysis_schema
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from langdetect import detect, LangDetectException

# --- CRITICAL FLAGS FOR PYANNOTE STABILITY ---
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# --- PYANNOTE MONKEY PATCH (Fixes Issue #1861) ---
try:
    from pyannote.audio.models.blocks.pooling import StatsPool
    def patched_forward(self, sequences, weights=None):
        mean = sequences.mean(dim=-1)
        if sequences.size(-1) > 1:
            std = sequences.std(dim=-1, correction=1)
        else:
            std = torch.zeros_like(mean)
        return torch.cat([mean, std], dim=-1)
    StatsPool.forward = patched_forward
    print("[i] Pyannote StatsPool.forward patched successfully.")
except ImportError:
    print("[!] Warning: Could not patch StatsPool. Pyannote might not be installed correctly.")

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    print("[!] 'tqdm' library not found. Install it for a progress bar: pip install tqdm")
    # Fallback dummy function if user doesn't install it
    def tqdm(iterable, desc="", unit=""):
        return iterable

warnings.filterwarnings("ignore")

#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Helpers ------------------------------------------------------------------
def normalize_language(lang: str, model_id: str) -> str:
    if not lang:
        lang = ""
    lang = lang.lower().strip()

    if lang in ("sv", "sv-se", "swedish"):
        return "sv"
    if lang in ("en", "en-us", "en-gb", "english"):
        return "en"

    # Fallbacks
    if "kb-whisper" in model_id.lower():
        print("[!] Could not detect language, defaulting to 'sv' for KBLab model")
        return "sv"
    else:
        print("[!] Could not detect language, defaulting to 'en'")
        return "en"

def cleanup_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def hhmmss_msec(seconds: float) -> str:
    if seconds is None: return "00:00:00,000"
    if seconds < 0: seconds = 0
    td = timedelta(seconds=seconds)
    total_seconds = td.total_seconds()
    hh = int(total_seconds // 3600)
    mm = int((total_seconds % 3600) // 60)
    ss = int(total_seconds % 60)
    msec = int(round((total_seconds - np.floor(total_seconds)) * 1000000))
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{msec:06d}"


# --- Diarization (Step 1) -----------------------------------------------------

def run_diarization(audio_path, num_speakers, hf_token):
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    # Select Model
    precision_api_key = os.getenv("PYANNOTE_PRECISION2_API_KEY")
    if precision_api_key:
        model_id = "pyannote/speaker-diarization-precision-2"
        token = precision_api_key
        print(f"[i] Using diarization model: {model_id}")
    else:
        model_id = "pyannote/speaker-diarization-community-1"
        token = hf_token
        print(f"[i] Using diarization model: {model_id}")

    try:
        pipeline = Pipeline.from_pretrained(model_id, token=token)
    except Exception as e:
        print(f"[!] Error loading Pyannote: {e}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    print(f"[i] Diarizing audio file: {os.path.basename(audio_path)}")

    # --- NATIVE PYANNOTE CALL (No manual loading) ---
    try:
        if model_id == "pyannote/speaker-diarization-precision-2":
            output = pipeline(audio_path) #Run on pyannoteAI servers
        else:
            # Pyannote pipeline handles loading via torchcodec/ffmpeg internally now and runs locally
            if num_speakers and num_speakers > 0:
                with ProgressHook() as hook:
                    output = pipeline(audio_path, preload=True, hook=hook, num_speakers=num_speakers)
            else:
                with ProgressHook() as hook:
                    output = pipeline(audio_path, preload=True, hook=hook)
    except Exception as e:
        print(f"[!] Diarization Execution Failed: {e}")
        return []

    # Handle Result Format (Annotation vs Hook Object)
    ann = None
    if hasattr(output, "exclusive_speaker_diarization"):
        ann = output.exclusive_speaker_diarization
    elif hasattr(output, "speaker_diarization"):
        ann = output.speaker_diarization
    elif hasattr(output, "annotation"):
        ann = output.annotation
    else:
        # In simple pipeline usage, output IS the annotation
        ann = output

    segments = []
    # itertracks yields (Segment, Track, Label)
    for turn, _, speaker in ann.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})

    del pipeline, output, ann
    cleanup_vram()
    return segments



# --- Merging Logic (Step 2: The "Natural Flow" Threshold) ---------------------

def merge_close_segments(segments, gap_threshold=2.0):
    if not segments: return []
    merged = []
    current = segments[0].copy()
    for next_seg in segments[1:]:
        if (next_seg["speaker"] == current["speaker"] and
                (next_seg["start"] - current["end"]) < gap_threshold):
            current["end"] = next_seg["end"]
        else:
            merged.append(current)
            current = next_seg.copy()
    merged.append(current)
    return merged


# --- ASR on Clips (Step 3) ----------------------------------------------------

def transcribe_segments(audio_path, segments, model_id, language, device, progress_callback=None):
    print(f"[i] Loading Whisper model: {model_id}...")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # --- UPDATED GENERATION CONFIG ---
    generate_kwargs = {
        "task": "transcribe",
        "condition_on_prev_tokens": False, # Crucial for segmented audio to prevent loops
        "temperature": 0.0,                # Greedy decoding (most accurate, least creative)
        # "no_repeat_ngram_size": 3,         # OPTIONAL: Uncomment if loops persist. Prevents "oh yeah oh yeah"
        "repetition_penalty": 1.1          # OPTIONAL: Penalizes repetition slightly
    }
    if language.lower().startswith("sv"):
        generate_kwargs["language"] = "sv"
    elif language.lower().startswith("en"):
        generate_kwargs["language"] = "en"
    else:
        generate_kwargs["language"] = language

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=16,
        return_timestamps=True,
        dtype=torch_dtype,
        device=0 if device == "cuda" else -1,
        generate_kwargs=generate_kwargs
    )

    print("[i] Loading full audio for slicing...")
    # Use torchaudio for consistency since we have the environment for it
    full_audio_tensor, sr = torchaudio.load(audio_path)

    # Mix to mono if needed
    if full_audio_tensor.shape[0] > 1:
        full_audio_tensor = full_audio_tensor.mean(dim=0)
    else:
        full_audio_tensor = full_audio_tensor.squeeze()

    # Convert to numpy for Whisper pipeline
    full_audio_np = full_audio_tensor.numpy()

    final_results = []

    # Filter: Separate verified from unverified
    to_transcribe = []
    for seg in segments:
        if seg.get("is_verified", False):
            final_results.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "text": seg.get("text", "")
            })
        else:
            to_transcribe.append(seg)

    total_segments = len(to_transcribe)

    for i, seg in enumerate(tqdm(to_transcribe, desc="[i] Transcribing", unit="seg")):
        start = seg["start"]
        end = seg["end"]
        spk = seg["speaker"]

        pad = 0.25
        start_sample = int(max(0, (start - pad) * sr))
        end_sample = int(min(len(full_audio_np), (end + pad) * sr))

        audio_cut = full_audio_np[start_sample:end_sample]

        if len(audio_cut) < ((pad * 3) * sr): continue

        try:
            result = asr_pipe(audio_cut, generate_kwargs=generate_kwargs)
            text = result.get("text", "").strip()
            if text:
                final_results.append({
                    "start": start, "end": end, "speaker": spk, "text": text
                })
            if progress_callback:
                progress_callback(i + 1, total_segments)
        except Exception as e:
            if hasattr(tqdm, "write"):
                tqdm.write(f"[!] Failed segment {i}: {e}")
            else:
                print(f"[!] Failed segment {i}: {e}")

    del asr_pipe, model, processor, full_audio_tensor, full_audio_np
    cleanup_vram()

    final_results.sort(key=lambda x: x["start"])
    return final_results


def detect_language_from_audio(audio_path: str) -> str:
    """
    Detect language using native torchaudio + whisper-tiny.
    """
    try:
        # Load ~10s from middle
        # Note: torchaudio.load doesn't support 'start/stop' efficiently without seeking,
        # but loading the whole file is robust.
        # For huge files, we might want frame_offset/num_frames args, but requires metadata first.
        # Let's just load, it's safer on Windows with ffmpeg.
        wav, sr = torchaudio.load(audio_path)

        # Crop 30s max
        if wav.shape[1] > sr * 30:
            mid = wav.shape[1] // 2
            wav = wav[:, mid:mid + (sr * 30)]

        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze()

        # Resample to 16k for Whisper
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=0 if device == "cuda" else -1,
        )

        result = asr(wav.numpy(), generate_kwargs={"task": "transcribe"})
        text = (result.get("text") or "").strip()

        if not text: return "unknown"

        lang = detect(text)
        if lang.startswith("sv"): return "sv"
        if lang.startswith("en"): return "en"
        return lang

    except Exception as e:
        print(f"[!] Language detection failed: {e}")
        return "unknown"


def apply_fingerprinting(audio_path, segments, fingerprint_source, hf_token, strategy="cluster"):
    """
    fingerprint_source: Can be a path (str) OR a dictionary {name: [paths]}
    """
    import json
    from biometrics import SpeakerIdentifier
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x

    known_map = {}

    # Case A: Input is a Dictionary (from GUI)
    if isinstance(fingerprint_source, dict):
        known_map = fingerprint_source

    # Case B: Input is a File Path (from CLI or old logic)
    elif isinstance(fingerprint_source, str) and os.path.exists(fingerprint_source):
        with open(fingerprint_source, 'r', encoding='utf-8') as f:
            known_map = json.load(f)

    if not known_map:
        print("[Fingerprint] No voices selected or loaded.")
        return segments

    identifier = SpeakerIdentifier(hf_token=hf_token)
    count = 0
    for name, path in known_map.items():
        if identifier.register_speaker(name, path): count += 1
    if count == 0: return segments

    # Preload via identifier (it uses sf.read, assuming that works or update it)
    full_wav, sr = identifier.preload_wave(audio_path)

    # ... (Rest of fingerprinting logic: Brute vs Cluster) ...
    # For brevity, reusing the robust logic we built before.

    # STRATEGY A: BRUTE FORCE
    if strategy == "brute":
        print(f"[Fingerprint] Brute Force ({len(segments)} segments)...")
        for seg in tqdm(segments, desc="Scanning", unit="seg"):
            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            if (end_sample - start_sample) < (sr * 0.5): continue
            seg_audio = full_wav[:, start_sample:end_sample]
            match = identifier.identify_segment(seg_audio, sample_rate=sr, threshold=0.55)
            if match: seg["speaker"] = match
        return segments

    # STRATEGY B: CLUSTER
    else:
        print("[Fingerprint] Cluster Voting...")
        from collections import defaultdict, Counter
        clusters = defaultdict(list)
        for seg in segments: clusters[seg["speaker"]].append(seg)

        mapping = {}
        dirty = []

        for spk, segs in clusters.items():
            sorted_segs = sorted(segs, key=lambda x: x["end"] - x["start"], reverse=True)
            check = sorted_segs[:5]
            votes = []
            for s in check:
                st, en = int(s["start"] * sr), int(s["end"] * sr)
                if en - st < sr * 0.5: continue
                match = identifier.identify_segment(full_wav[:, st:en], sr, 0.5)
                if match: votes.append(match)

            if not votes: continue

            vote_counts = Counter(votes)
            winner, win_count = vote_counts.most_common(1)[0]
            if (win_count / len(votes)) >= 0.8:
                mapping[spk] = winner
            else:
                dirty.append(spk)

        final_segs = []
        for seg in segments:
            if seg["speaker"] in mapping:
                seg["speaker"] = mapping[seg["speaker"]]
            elif seg["speaker"] in dirty:
                st, en = int(seg["start"] * sr), int(seg["end"] * sr)
                if en - st > sr * 0.5:
                    match = identifier.identify_segment(full_wav[:, st:en], sr, 0.55)
                    if match: seg["speaker"] = match
            final_segs.append(seg)
        return final_segs

# --- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Diarization-First + Natural Paragraphs + ProgressBar")
    parser.add_argument("audio", help="Audio file path")
    parser.add_argument("--out", help="Base output name", default=None)
    #parser.add_argument("--model", default="KBLab/kb-whisper-large", help="ASR model")
    #parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--language", default="auto", help="Language code or 'auto'")
    parser.add_argument("--model", default="auto", help="ASR model or 'auto'")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN_TRANSCRIBE"), help="HuggingFace Token")
    parser.add_argument("--num-speakers", type=int, default=None, help="Num speakers hint")
    parser.add_argument("--names", type=str, default=None, help="Names: 'Host,Guest'")
    parser.add_argument("--fingerprints", type=str, default=None, help="Path to JSON file with {'Name': 'path/to/audio.wav'}")
    parser.add_argument("--srt", action="store_true", help="Generate .srt")
    parser.add_argument("--enable_analysis", action="store_true", help="Send transcription to LLM (chatgpt for analysis.")
    parser.add_argument("--gap", type=float, default=2.0, help="Pause duration (seconds) to create a new paragraph. Default 2.0s")

    args = parser.parse_args()
    # Auto detect language if requested
    language = args.language
    # Auto-select model based on language
    model_id = args.model

    if language == "auto":
        print("[i] Detecting language from audio…")
        detected = detect_language_from_audio(args.audio)
        language = normalize_language(detected, model_id)
        print(f"[i] Detected language: {language}")

    if model_id == "auto":
        if language.startswith("sv"):
            print("[i] Using Swedish model: KBLab/kb-whisper-large")
            model_id = "KBLab/kb-whisper-large"
        else:
            print("[i] Using English/general model: openai/whisper-large-v3")
            model_id = "openai/whisper-large-v3"

    if not args.hf_token:
        print("[!] Error: HF_TOKEN required.")
        sys.exit(1)

    base_name = args.out or os.path.splitext(os.path.basename(args.audio))[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Efter att txt-filen skapats:
    txt_filename = f"{base_name}_with_speakers.txt"
    # ... din befintliga kod ...

    # 1. Diarization
    print("--- Step 1: Diarization ---")
    raw_segments = run_diarization(args.audio, args.num_speakers, args.hf_token)
    if args.fingerprints:
        print("--- Step 1.5: Speaker Fingerprinting ---")
        raw_segments = apply_fingerprinting(args.audio, raw_segments, args.fingerprints, args.hf_token)

    # 2. Merge Close Segments
    print(f"--- Step 2: Grouping (Pause threshold: {args.gap}s) ---")
    merged_segments = merge_close_segments(raw_segments, gap_threshold=args.gap)
    print(f"    -> Raw segments: {len(raw_segments)}")
    print(f"    -> Final Paragraphs: {len(merged_segments)}")

    # 3. Transcribe Cuts
    print("--- Step 3: Transcription ---")
    final_items = transcribe_segments(args.audio, merged_segments, model_id, language, device)
    #final_items = transcribe_segments(args.audio, merged_segments, args.model, args.language, device)

    # 4. Save
    rename_map = {}
    if args.names:
        names = [n.strip() for n in args.names.split(",")]
        spks = sorted(list(set(x['speaker'] for x in final_items)))
        for i, name in enumerate(names):
            if i < len(spks): rename_map[spks[i]] = name

    txt_filename = f"{base_name}_with_speakers.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        for item in final_items:
            spk = rename_map.get(item["speaker"], item["speaker"])
            stamp = hhmmss_msec(item["start"]).split(",")[0]
            f.write(f"[{stamp}] {spk}: {item['text']}\n\n")

    if args.srt:
        srt_filename = f"{base_name}_with_speakers.srt"
        with open(srt_filename, "w", encoding="utf-8") as f:
            for i, item in enumerate(final_items, 1):
                spk = rename_map.get(item["speaker"], item["speaker"])
                f.write(f"{i}\n{hhmmss_msec(item['start'])} --> {hhmmss_msec(item['end'])}\n{spk}: {item['text']}\n\n")
        print(f"[✓] Done. Outputs: {txt_filename}, {srt_filename}")
    else:
        print(f"[✓] Done. Output: {txt_filename}")

    if args.enable_analysis:
        print("\n--- Step 4: Conversation Analysis ---")
        try:
            print(f"[i] Loading transcript from: {txt_filename}")
            with open(txt_filename, "r", encoding="utf-8") as f:
                transcript = f.read()

            print("[i] Running preclassification + model selection...")
            result = analyze_conversation(transcript, call_llm_openai, analysis_language=language)
            #result = analyze_conversation(transcript, call_llm_grokai, analysis_language=language)
            raw_analysis = result.analysis_json  # dict
            normalized_analysis = normalize_analysis_schema(raw_analysis)

            analyze_filename = f"{base_name}_analyzed.json"
            print(f"[i] Chosen model: {result.chosen_model}")
            print(f"[i] Preclass-reason: {result.preclass_reason}")

            print(f"[i] Writing JSON analysis to: {analyze_filename}")
            with open(analyze_filename, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "chosen_model": result.chosen_model,
                        "preclass_reason": result.preclass_reason,
                        "analysis_raw": result.analysis_raw_text,  # string version
                        "analysis_normalized": normalized_analysis  # canonical dict
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(f"[✓] Analys klar. Sparad till: {analyze_filename}")
        except Exception as e:
            print(f"[!] Fel under analys-steget: {e}")

if __name__ == "__main__":
    main()