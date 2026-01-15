import numpy as np
import noisereduce as nr
from scipy import signal
from scipy.io import wavfile
import soundfile as sf
import torchaudio
import torch
class AudioDenoiser:
    """
    A comprehensive class to clean and denoise audio for improved speech transcription.

    Features:
    - Spectral noise reduction
    - High-pass filtering
    - Pre-emphasis
    - Intelligent level normalization
    - ClearVoice deep learning enhancement
    """

    def __init__(self,
                 sample_rate=16000,
                 noise_reduce_strength=0.7,
                 highpass_cutoff=80,
                 apply_preemphasis=True,
                 preemphasis_coef=0.10,
                 normalize=True):
        """
        Initialize the AudioDenoiser.

        Args:
            sample_rate (int): Target sample rate for audio processing
            noise_reduce_strength (float): Strength of noise reduction (0-1)
            highpass_cutoff (int): High-pass filter cutoff frequency in Hz
            apply_preemphasis (bool): Apply pre-emphasis filter for speech
            preemphasis_coef (float): Pre-emphasis coefficient (default 0.10)
            normalize (bool): Normalize audio to [-1, 1] range
        """
        self.sample_rate = sample_rate
        self.noise_reduce_strength = noise_reduce_strength
        self.highpass_cutoff = highpass_cutoff
        self.apply_preemphasis = apply_preemphasis
        self.preemphasis_coef = preemphasis_coef
        self.normalize = normalize
        self._clearvoice = None

    # ==================== AUDIO I/O ====================

    def load_audio(self, file_path):
        """Load audio, enforce self.sample_rate, and return mono float32 numpy array.

        This is critical because downstream ASR components must be told the *true*
        sampling rate of the waveform. We enforce a single SR across the entire
        processing pipeline to avoid subtle time-scale/pitch distortions.
        """
        audio, sr = torchaudio.load(file_path)  # shape: (channels, samples)

        # Resample FIRST to preserve channel alignment
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            sr = self.sample_rate

        # Mix to mono if needed
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio.squeeze()

        audio_np = audio.detach().cpu().numpy().astype(np.float32, copy=False)
        return audio_np, sr

    def save_audio(self, audio, output_path, sr=None):
        """Save audio to file."""
        if sr is None:
            sr = self.sample_rate
        sf.write(output_path, audio, sr, subtype="PCM_16")

    # ==================== BASIC FILTERING ====================

    def _highpass_filter(self, audio, sr):
        """Apply high-pass filter to remove low-frequency noise."""
        nyquist = sr / 2
        normalized_cutoff = self.highpass_cutoff / nyquist
        b, a = signal.butter(5, normalized_cutoff, btype='high')
        filtered = signal.filtfilt(b, a, audio)
        return filtered

    def _preemphasis(self, audio, coef=None):
        """Apply pre-emphasis filter to boost high frequencies in speech."""
        if coef is None:
            coef = self.preemphasis_coef
        return np.append(audio[0], audio[1:] - coef * audio[:-1])

    def _normalize_audio(self, audio):
        """Normalize audio to [-1, 1] range."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio

    # ==================== ENERGY-BASED ANALYSIS ====================

    def frame_rms_db(self, x, sr, frame_ms=30, hop_ms=10, eps=1e-12):
        """
        Calculate RMS energy in dB for each frame of audio.

        Args:
            x (np.ndarray): Audio signal
            sr (int): Sample rate
            frame_ms (int): Frame length in milliseconds
            hop_ms (int): Hop length in milliseconds
            eps (float): Small value to prevent log(0)

        Returns:
            tuple: (db_values, frame_samples, hop_samples, padded_length)
        """
        frame = int(sr * frame_ms / 1000)
        hop = int(sr * hop_ms / 1000)

        if frame <= 0 or hop <= 0:
            raise ValueError("frame/hop too small for given sample rate")

        # Pad to fit frames cleanly
        n = len(x)
        pad = (frame - (n - frame) % hop) % hop
        xpad = np.pad(x, (0, pad), mode="constant")

        # Framing via stride trick
        shape = ((len(xpad) - frame) // hop + 1, frame)
        strides = (xpad.strides[0] * hop, xpad.strides[0])
        frames = np.lib.stride_tricks.as_strided(xpad, shape=shape, strides=strides)

        # Calculate RMS and convert to dB
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        db = 20.0 * np.log10(rms + eps)

        return db, frame, hop, len(xpad)

    def smooth(self, x, win=7):
        """Apply moving average smoothing to signal."""
        win = max(1, int(win))
        if win == 1:
            return x
        kernel = np.ones(win) / win
        return np.convolve(x, kernel, mode="same")

    def low_segments_from_db(self, db, low_enter_db=-25.0, low_exit_db=-22.0,
                             min_low_ms=150, bridge_ms=120, sr=16000, hop=160):
        """
        Detect low-energy segments using hysteresis thresholding.

        Args:
            db (np.ndarray): Energy values in dB
            low_enter_db (float): Threshold to enter low-energy state
            low_exit_db (float): Threshold to exit low-energy state
            min_low_ms (int): Minimum duration for a segment (ms)
            bridge_ms (int): Maximum gap to bridge between segments (ms)
            sr (int): Sample rate
            hop (int): Hop size in samples

        Returns:
            list: List of (start_sample, end_sample) tuples
        """
        # Hysteresis state machine
        low = np.zeros_like(db, dtype=bool)
        state = False

        for i, v in enumerate(db):
            if not state and v < low_enter_db:
                state = True
            elif state and v > low_exit_db:
                state = False
            low[i] = state

        # Find runs of True
        idx = np.flatnonzero(low)
        if idx.size == 0:
            return []

        # Identify continuous runs
        runs = []
        start = idx[0]
        prev = idx[0]

        for j in idx[1:]:
            if j == prev + 1:
                prev = j
            else:
                runs.append((start, prev))
                start = prev = j
        runs.append((start, prev))

        # Bridge small gaps
        bridged = [runs[0]]
        max_gap_frames = int((bridge_ms / 1000) * sr / hop)

        for a, b in runs[1:]:
            pa, pb = bridged[-1]
            if a - pb - 1 <= max_gap_frames:
                bridged[-1] = (pa, b)  # Merge
            else:
                bridged.append((a, b))

        # Filter by minimum duration
        min_frames = int((min_low_ms / 1000) * sr / hop)
        segments = []

        for a, b in bridged:
            if (b - a + 1) >= min_frames:
                # Convert frames to samples
                segments.append((a * hop, (b + 1) * hop))

        return segments

    def apply_upward_segment_gain(self, x, sr, segments, target_rms_db=-16.0,
                                  max_gain_db=15.0, fade_ms=10, eps=1e-12):
        """
        Apply upward gain to quiet segments to improve speech intelligibility.

        Args:
            x (np.ndarray): Audio signal
            sr (int): Sample rate
            segments (list): List of (start, end) sample positions
            target_rms_db (float): Target RMS level in dB
            max_gain_db (float): Maximum gain to apply in dB
            fade_ms (int): Fade duration at segment edges (ms)
            eps (float): Small value to prevent division by zero

        Returns:
            np.ndarray: Audio with gain applied
        """
        y = x.copy().astype(np.float32)
        fade = int(sr * fade_ms / 1000)

        for s, e in segments:
            s = max(0, s)
            e = min(len(y), e)

            if e <= s:
                continue

            # Calculate current RMS
            seg = y[s:e].astype(np.float64)
            rms = np.sqrt(np.mean(seg ** 2)) + eps
            seg_db = 20.0 * np.log10(rms)

            # Calculate gain (upward only, capped at max_gain_db)
            gain_db = np.clip(target_rms_db - seg_db, 0.0, max_gain_db)
            gain_linear = 10.0 ** (gain_db / 20.0)

            # Apply with fades to avoid clicks
            if fade > 0 and (e - s) > 2 * fade:
                ramp = np.ones(e - s, dtype=np.float32)
                ramp[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
                ramp[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
                y[s:e] = y[s:e] * (1 - ramp) + (y[s:e] * gain_linear) * ramp
            else:
                y[s:e] *= gain_linear

        return y

    # ==================== PROCESSING METHODS ====================

    def denoise(self, audio, sr=None, noise_sample=None):
        """
        Apply traditional denoising pipeline to audio.

        Args:
            audio (np.ndarray): Input audio signal
            sr (int): Sample rate (uses class default if None)
            noise_sample (np.ndarray): Optional noise profile sample

        Returns:
            np.ndarray: Cleaned audio signal
        """
        if sr is None:
            sr = self.sample_rate

        # High-pass filter
        audio_filtered = self._highpass_filter(audio, sr)

        # Spectral noise reduction
        if noise_sample is not None:
            audio_denoised = nr.reduce_noise(
                y=audio_filtered,
                sr=sr,
                y_noise=noise_sample,
                prop_decrease=self.noise_reduce_strength
            )
        else:
            audio_denoised = nr.reduce_noise(
                y=audio_filtered,
                sr=sr,
                stationary=True,
                prop_decrease=self.noise_reduce_strength
            )

        # Pre-emphasis
        if self.apply_preemphasis:
            audio_denoised = self._preemphasis(audio_denoised)

        # Normalize
        if self.normalize:
            audio_denoised = self._normalize_audio(audio_denoised)

        return audio_denoised

    def intelligent_normalize(self, audio, sr, low_enter_db=-26, low_exit_db=-24,
                              min_low_ms=150, bridge_ms=120, target_rms_db=-18,
                              max_gain_db=15, fade_ms=10, smooth_win=7):
        """
        Apply intelligent level normalization that boosts quiet segments.

        This method analyzes the audio to find low-energy segments and
        selectively boosts them to improve overall intelligibility.

        Args:
            audio (np.ndarray): Input audio signal
            sr (int): Sample rate
            low_enter_db (float): Threshold to detect low segments
            low_exit_db (float): Threshold to exit low segment detection
            min_low_ms (int): Minimum low segment duration (ms)
            bridge_ms (int): Gap bridging duration (ms)
            target_rms_db (float): Target RMS for low segments
            max_gain_db (float): Maximum gain to apply
            fade_ms (int): Fade duration at edges
            smooth_win (int): Smoothing window size

        Returns:
            np.ndarray: Normalized audio
        """
        # Calculate frame-wise energy
        db, frame, hop, _ = self.frame_rms_db(audio, sr, frame_ms=30, hop_ms=10)
        db_smoothed = self.smooth(db, win=smooth_win)

        # Find low-energy segments
        segments = self.low_segments_from_db(
            db_smoothed,
            low_enter_db=low_enter_db,
            low_exit_db=low_exit_db,
            min_low_ms=min_low_ms,
            bridge_ms=bridge_ms,
            sr=sr,
            hop=hop
        )

        # Apply selective gain
        normalized = self.apply_upward_segment_gain(
            audio, sr, segments,
            target_rms_db=target_rms_db,
            max_gain_db=max_gain_db,
            fade_ms=fade_ms
        )

        return normalized

    def enhance_with_clearvoice(self, input_path, output_path=None,
                                model_names=['MossFormer2_SE_48K']):
        """
        Apply ClearVoice deep learning enhancement.

        Args:
            input_path (str): Path to input audio file
            output_path (str): Optional path to save output
            model_names (list): ClearVoice model names to use

        Returns:
            tuple: (enhanced_audio, sample_rate) - audio as float32 at target SR
        """
        if self._clearvoice is None:
            try:
                from clearvoice import ClearVoice  # type: ignore
            except Exception as e:
                raise ImportError(
                    "ClearVoice is not installed. "
                    "Install it (and its deps) or set use_clearvoice=False."
                ) from e
            self._clearvoice = ClearVoice(
                task='speech_enhancement',
                model_names=model_names
            )

        # Process with ClearVoice (outputs at 48kHz)
        output_wav = self._clearvoice(input_path=input_path, online_write=False)

        # ClearVoice returns shape (1, n_samples), squeeze to (n_samples,)
        if output_wav.ndim > 1:
            output_wav = np.squeeze(output_wav)

        # Convert from float64 to float32
        output_wav = output_wav.astype(np.float32)

        # Resample to target sample rate
        clearvoice_sr = 48000  # ClearVoice outputs at 48kHz
        if self.sample_rate != clearvoice_sr:
            # Prefer torchaudio resampler to avoid extra deps
            wav_t = torch.from_numpy(output_wav).unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, clearvoice_sr, self.sample_rate)
            output_wav = wav_t.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

        if output_path:
            self.save_audio(output_wav, output_path, self.sample_rate)

        return output_wav, self.sample_rate

    # ==================== COMPLETE PIPELINES ====================

    def process_file(self, input_path, output_path=None, noise_sample_path=None,
                     method='denoise'):
        """
        Process an audio file with specified method.

        Args:
            input_path (str): Path to input audio file
            output_path (str): Path to save cleaned audio (optional)
            noise_sample_path (str): Path to noise sample file (optional)
            method (str): Processing method - 'denoise', 'normalize', or 'clearvoice'

        Returns:
            tuple: (processed_audio, sample_rate)
        """
        if method == 'clearvoice':
            # ClearVoice returns (audio, sample_rate) tuple
            cleaned_audio, sr = self.enhance_with_clearvoice(input_path, output_path)
        else:
            # Load audio
            audio, sr = self.load_audio(input_path)

            # Load noise sample if provided
            noise_sample = None
            if noise_sample_path:
                noise_sample, _ = self.load_audio(noise_sample_path)

            # Apply processing
            if method == 'denoise':
                cleaned_audio = self.denoise(audio, sr, noise_sample)
            elif method == 'normalize':
                cleaned_audio = self.intelligent_normalize(audio, sr)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Save if output path provided
            if output_path:
                self.save_audio(cleaned_audio, output_path, sr)

        return cleaned_audio, sr

    def full_pipeline(self, input_path, output_path=None, use_clearvoice=False):
        """
        Apply complete processing pipeline.

        Args:
            input_path (str): Path to input audio file
            output_path (str): Path to save result (optional)
            use_clearvoice (bool): Whether to use ClearVoice enhancement

        Returns:
            tuple: (processed_audio, sample_rate)
        """
        if use_clearvoice:
            # ClearVoice handles everything
            return self.process_file(input_path, output_path, method='clearvoice')
        else:
            # Traditional pipeline
            audio, sr = self.load_audio(input_path)

            # Step 1: Denoise
            audio = self.denoise(audio, sr)

            # Step 2: Intelligent normalization
            audio = self.intelligent_normalize(audio, sr)

            if output_path:
                self.save_audio(audio, output_path, sr)

            return audio, sr


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    denoiser = AudioDenoiser(
        sample_rate=16000,
        noise_reduce_strength=0.7,
        highpass_cutoff=80,
        apply_preemphasis=True,
        preemphasis_coef=0.10
    )

    # Method 1: Traditional denoising
    cleaned, sr = denoiser.process_file(
        "noisy_audio.wav",
        "cleaned_traditional.wav",
        method='denoise'
    )

    # Method 2: Intelligent normalization
    normalized, sr = denoiser.process_file(
        "noisy_audio.wav",
        "cleaned_normalized.wav",
        method='normalize'
    )

    # Method 3: ClearVoice enhancement
    enhanced, sr = denoiser.process_file(
        "noisy_audio.wav",
        "cleaned_clearvoice.wav",
        method='clearvoice'
    )

    # Method 4: Full pipeline
    final, sr = denoiser.full_pipeline(
        "noisy_audio.wav",
        "cleaned_full.wav",
        use_clearvoice=False
    )

    print(f"Processing complete! Sample rate: {sr} Hz")