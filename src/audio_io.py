"""
Audio load helpers that intentionally avoid TorchCodec / torchaudio.load().

Recent torchaudio builds route torchaudio.load() through TorchCodec, which on
Windows often fails (FFmpeg DLL / version mismatch) even when system ffmpeg
works. VocalTrace decodes with soundfile and/or the system ffmpeg CLI instead,
and passes in-memory tensors to Pyannote/Whisper.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

PathLike = Union[str, Path]


def resolve_ffmpeg_exe(ffmpeg_path: str | None = None) -> str:
    """Prefer system ffmpeg (not Conda libs)."""
    exe = (
        ffmpeg_path
        or os.environ.get("VOCALTRACE_FFMPEG")
        or os.environ.get("FFMPEG_PATH")
        or shutil.which("ffmpeg")
    )
    if not exe:
        raise RuntimeError(
            "ffmpeg executable not found. Install system ffmpeg and put it on PATH, "
            "or set VOCALTRACE_FFMPEG / FFMPEG_PATH. "
            "Do not use Conda ffmpeg on Windows (DLL conflicts with PySide6)."
        )
    return exe


def load_audio_via_ffmpeg(
    path: PathLike,
    target_sr: int = 16000,
    mono: bool = True,
    ffmpeg_path: str | None = None,
) -> dict:
    """
    Decode with system ffmpeg CLI → Pyannote-compatible dict:
      {"waveform": Tensor [channels, time], "sample_rate": int}
    """
    path = str(path)
    exe = resolve_ffmpeg_exe(ffmpeg_path)

    cmd = [
        exe,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-ar",
        str(target_sr),
    ]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-f", "f32le", "pipe:1"]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed (code {proc.returncode}): {err}")

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if not audio.flags.writeable:
        audio = audio.copy()

    # Raw PCM with -ac 1 → mono. Multi-channel without -ac needs known channel count;
    # VocalTrace uses mono for diarization/ASR.
    waveform = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
    return {"waveform": waveform, "sample_rate": int(target_sr)}


def load_waveform(
    path: PathLike,
    *,
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load audio as float32 tensor [channels, samples] without TorchCodec.

    Order:
      1) soundfile (wav/flac/ogg/…)
      2) system ffmpeg (mp3/m4a/… and anything soundfile rejects)
    """
    path = str(path)
    waveform: torch.Tensor | None = None
    sr: int | None = None
    sf_error: Exception | None = None

    try:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        # soundfile: (frames, channels) → (channels, frames)
        waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    except Exception as exc:  # noqa: BLE001 — fall back to ffmpeg for any codec issue
        sf_error = exc
        if target_sr is None:
            # Decode at native rate via ffmpeg: use high quality then leave as-is;
            # we pick 48000 as intermediate if target unknown, then user can resample.
            # Prefer decoding at 16000 when target is unspecified for ASR paths.
            decode_sr = 16000
        else:
            decode_sr = target_sr
        try:
            audio_dict = load_audio_via_ffmpeg(
                path, target_sr=decode_sr, mono=mono
            )
            waveform = audio_dict["waveform"]
            sr = int(audio_dict["sample_rate"])
        except Exception as ff_exc:
            raise RuntimeError(
                f"Failed to load audio '{path}'. "
                f"soundfile error: {sf_error}; ffmpeg error: {ff_exc}"
            ) from ff_exc

    assert waveform is not None and sr is not None

    if mono and waveform.ndim == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    elif mono and waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if target_sr is not None and sr != target_sr:
        # Resample without torchaudio.load / TorchCodec — transforms only need torch
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform.contiguous().float(), int(sr)


def load_mono_numpy(
    path: PathLike,
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load mono float32 numpy array at target_sr (for AudioDenoiser etc.)."""
    wav, sr = load_waveform(path, target_sr=target_sr, mono=True)
    audio_np = wav.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return audio_np, sr
