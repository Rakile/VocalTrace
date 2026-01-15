from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import torch

# Reuse your existing logic from the transcription script

from transcription_engine import (
    run_diarization,
    merge_close_segments,
    transcribe_segments,
    hhmmss_msec,
    detect_language_from_audio,
    normalize_language,
    apply_fingerprinting,
)

# Reuse analysis + schema-normalization from your analysis pipeline
from analysis_engine import (
    analyze_conversation,
    normalize_analysis_schema,
)


# ---------------------------------------------------------------------------
# 1) Transcription engine (audio -> transcript + segments)
# ---------------------------------------------------------------------------

def transcribe_audio_to_text(
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        language: str = "auto",
        model_id: Optional[str] = "auto",
        hf_token: Optional[str] = None,
        gap: float = 2.0,
        names: Optional[str] = None,
        device: Optional[str] = None,
        fingerprints_path: Optional[str] = None,
        fingerprints_dict: Optional[Dict] = None, # NEW argument
        id_strategy: str = "cluster",
        diarization_model: str = "unknown",
        transcription_model: str = "auto",
        progress_callback: Optional[Callable[[int, int], None]] = None,
        verified_segments: Optional[List[Dict]] = None,
        **kwargs
) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    if progress_callback: progress_callback(0, 0)

    # Language
    if language == "auto":
        language = normalize_language(detect_language_from_audio(audio_path), model_id)

    if transcription_model == "auto":
        if language.startswith("sv"):
            model_id = "KBLab/kb-whisper-large"
        else:
            model_id = "openai/whisper-large-v3"
    else:
        model_id = transcription_model

    # 1. Diarization
    raw_segments = run_diarization(audio_path, num_speakers, hf_token)

    # --- 1.5) Fingerprinting ---
    # Prioritize Dict, fallback to Path
    fp_source = fingerprints_dict if fingerprints_dict else fingerprints_path

    if fp_source:
        print(f"[core] Step 1.5: Applying Fingerprints...")
        raw_segments = apply_fingerprinting(
            audio_path,
            raw_segments,
            fp_source,
            hf_token,
            strategy=id_strategy
        )

    # 3. ENFORCE MANUAL TRUTH
    if verified_segments:
        print(f"[core] Enforcing {len(verified_segments)} verified segments.")
        raw_segments = enforce_manual_segments(raw_segments, verified_segments)

    # 4. Merge
    merged_segments = merge_close_segments(raw_segments, gap_threshold=gap)

    # 5. Transcribe (skips verified text)
    final_items = transcribe_segments(
        audio_path, merged_segments, model_id, language, device,
        progress_callback=progress_callback,
    )

    # 6. Format
    rename_map = {}
    if names:
        # ... (name mapping logic) ...
        pass

    lines = []
    for item in final_items:
        spk = rename_map.get(item["speaker"], item["speaker"])
        lines.append(f"[{hhmmss_msec(item['start'])}]-[{hhmmss_msec(item['end'])}] {spk}: {item['text']}")
        lines.append("")

    return {
        "audio_path": audio_path,
        "base_name": base_name,
        "transcript": "\n".join(lines),
        "segments": final_items,
        "rename_map": rename_map,
        "language": language,
    }


# ---------------------------------------------------------------------------
# 2) Analysis engine (transcript -> analysis JSON)
# ---------------------------------------------------------------------------

def enforce_manual_segments(ai_segments, manual_segments):
    """
    Overwrites AI segments with manual ones (Truth Persistence).
    """
    if not manual_segments: return ai_segments

    # Prepare Manual List
    clean_manual = []
    for m in manual_segments:
        clean_manual.append({
            "start": float(m["start"]),
            "end": float(m["end"]),
            "speaker": m["speaker"],
            "text": m.get("text", ""),
            "is_verified": True
        })

    final_segments = []

    # Filter AI segments
    for ai in ai_segments:
        ai_start = ai["start"]
        ai_end = ai["end"]
        keep_ai = True

        for man in clean_manual:
            man_start = man["start"]
            man_end = man["end"]

            # Subtraction Logic
            overlap_start = max(ai_start, man_start)
            overlap_end = min(ai_end, man_end)
            overlap_dur = max(0, overlap_end - overlap_start)
            ai_dur = ai_end - ai_start

            # If >50% overlap, kill AI segment
            if ai_dur > 0 and (overlap_dur / ai_dur) > 0.5:
                keep_ai = False
                break

        if keep_ai:
            final_segments.append(ai)

    final_segments.extend(clean_manual)
    final_segments.sort(key=lambda x: x["start"])
    return final_segments

def analyze_transcript(transcript, call_llm, analysis_language="auto", backend="openai", progress_callback=None):
    if progress_callback: progress_callback(0, 0)
    result = analyze_conversation(transcript, call_llm, analysis_language=analysis_language, backend=backend)
    norm = normalize_analysis_schema(result.analysis_json)
    return {
        "chosen_model": result.chosen_model,
        "preclass_reason": result.preclass_reason,
        "analysis_raw": result.analysis_raw_text,
        "analysis_normalized": norm,
    }


# ---------------------------------------------------------------------------
# 3) Full pipeline (audio -> transcript -> analysis)
# ---------------------------------------------------------------------------

def run_full_pipeline(audio_path, call_llm, verified_segments=None, fingerprints_path: Optional[str] = None, fingerprints_dict: Optional[Dict] = None, **kwargs):
    # Pass verified_segments to transcriber
    trans = transcribe_audio_to_text(audio_path, verified_segments=verified_segments,fingerprints_path=fingerprints_path, fingerprints_dict=fingerprints_dict, **kwargs)

    analysis = analyze_transcript(
        trans["transcript"], call_llm=call_llm,
        analysis_language=trans["language"], backend=kwargs.get("backend", "openai"),
        progress_callback=kwargs.get("progress_callback")
    )
    return {
        "transcript": trans["transcript"],
        "analysis": analysis,
        # ...
    }
