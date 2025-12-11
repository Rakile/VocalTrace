# speaker_fingerprinting.py

import torch
import numpy as np
import soundfile as sf
import torchaudio
import os
from pyannote.audio import Model, Inference
from scipy.spatial.distance import cdist


class SpeakerIdentifier:
    def __init__(self, hf_token=None):
        print("[Fingerprint] Loading Embedding Model...")
        self.model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", use_auth_token=hf_token)
        self.inference = Inference(self.model, window="whole")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference.to(self.device)

        # CHANGED: Now stores a LIST of embeddings: {'Name': [emb1, emb2, ...]}
        self.known_speakers = {}

    def preload_wave(self, audio_path):
        wav, sr = sf.read(audio_path, always_2d=True)
        if wav.shape[1] > 1:
            wav = wav.mean(axis=1, keepdims=True)
        waveform = torch.from_numpy(wav.T).float()
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            sr = 16000
        return waveform, sr

    def _safe_crop(self, waveform, sample_rate, max_seconds=60):
        max_samples = int(max_seconds * sample_rate)
        total_samples = waveform.shape[1]
        if total_samples > max_samples:
            if total_samples > (sample_rate * 600):  # > 10 mins
                start = total_samples // 2
                return waveform[:, start: start + max_samples]
            return waveform[:, :max_samples]
        return waveform

    def get_embedding(self, audio_path=None, waveform=None, sample_rate=16000):
        if audio_path:
            waveform, sample_rate = self.preload_wave(audio_path)
            waveform = self._safe_crop(waveform, sample_rate)

        waveform = waveform.to(self.device)
        embedding = self.inference({"waveform": waveform, "sample_rate": sample_rate})

        if hasattr(embedding, "numpy"): embedding = embedding.cpu().numpy()
        if embedding.ndim == 1: embedding = embedding.reshape(1, -1)
        return embedding

    def register_speaker(self, name, paths):
        """
        Accepts either a single path (str) or a list of paths (list).
        Stores all valid embeddings for that speaker.
        """
        # Handle single string input for backward compatibility
        if isinstance(paths, str):
            paths = [paths]

        print(f"[Fingerprint] Registering profile for {name} ({len(paths)} samples)...")

        embeddings_list = []

        for p in paths:
            if not os.path.exists(p):
                print(f"[Fingerprint] Warning: File not found {p}")
                continue
            try:
                # We calculate embedding for EACH sample
                emb = self.get_embedding(audio_path=p)
                embeddings_list.append(emb)
            except Exception as e:
                print(f"[!] Failed to load sample {p}: {e}")

        if embeddings_list:
            self.known_speakers[name] = embeddings_list
            return True
        return False

    def identify_segment(self, waveform, sample_rate=16000, threshold=0.5):
        if not self.known_speakers: return None

        # Crop for speed during identification
        waveform = self._safe_crop(waveform, sample_rate, max_seconds=30)
        current_emb = self.get_embedding(waveform=waveform, sample_rate=sample_rate)

        best_name = None
        best_dist = 10.0

        # Compare against EVERY sample of EVERY speaker
        for name, emb_list in self.known_speakers.items():
            for ref_emb in emb_list:
                dist = cdist(current_emb, ref_emb, metric="cosine")[0, 0]

                # If this specific sample is the closest match so far...
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

        if best_dist < threshold:
            return best_name

        return None