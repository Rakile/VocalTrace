import os
import tempfile
import uuid

from AudioDenoiser import AudioDenoiser


class AudioProcessingProfiles:
    """
    Manages different audio processing profiles optimized for specific models.

    - Pyannote (diarization): Minimal processing to preserve speaker characteristics
    - Whisper (transcription): Aggressive cleaning for better text recognition
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.temp_files = []

    def cleanup_temp_files(self):
        """Remove all temporary files created during processing."""
        for path in self.temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                print(f"[Cleanup] Warning: Could not remove {path}: {e}")
        self.temp_files = []

    def __del__(self):
        """Ensure temp files are cleaned up."""
        self.cleanup_temp_files()

    def _create_temp_path(self, original_path, suffix):
        """Create a unique temporary file path.

        Uses the system temp directory to avoid collisions and accidental overwrites
        when processing the same input multiple times or in parallel.
        """
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        token = uuid.uuid4().hex[:8]
        filename = f"{base_name}_{suffix}_{token}.wav"
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        self.temp_files.append(temp_path)
        return temp_path

    # ==================== PYANNOTE PROFILE ====================

    def process_for_pyannote(self, audio_path, use_processing=True):
        """
        Light processing for speaker diarization.

        Goal: Preserve speaker voice characteristics while reducing noise.
        - Very light noise reduction (0.2 strength)
        - Minimal high-pass filtering (80 Hz)
        - NO pre-emphasis (preserves natural voice timbre)
        - NO normalization (preserves relative loudness differences)

        Args:
            audio_path (str): Path to original audio
            use_processing (bool): If False, returns original path

        Returns:
            str: Path to processed audio (or original if use_processing=False)
        """
        if not use_processing:
            print("[Pyannote] Using original audio (no processing)")
            return audio_path

        print("[Pyannote] Applying light processing for diarization...")

        denoiser = AudioDenoiser(
            sample_rate=self.sample_rate,
            noise_reduce_strength=0.2,  # Very light
            highpass_cutoff=80,  # Remove rumble only
            apply_preemphasis=False,  # Keep natural voice
            normalize=False  # Keep volume dynamics
        )

        output_path = self._create_temp_path(audio_path, "pyannote")
        denoiser.process_file(
            input_path=audio_path,
            output_path=output_path,
            method='denoise'
        )

        print(f"[Pyannote] Processed audio saved: {output_path}")
        return output_path

    # ==================== WHISPER PROFILE ====================

    def process_for_whisper(self, audio_path, *, use_processing: bool = True, use_clearvoice: bool = False,
                            aggressive_cleaning: bool = False):
        """
        Heavy processing for speech transcription.

        Goal: Maximum intelligibility for text recognition.
        - Moderate to strong noise reduction
        - High-pass filtering
        - Pre-emphasis for clarity
        - Intelligent normalization
        - Optional: ClearVoice deep learning enhancement

        Args:
            audio_path (str): Path to original audio
            use_clearvoice (bool): Apply AI-based enhancement
            aggressive_cleaning (bool): Use stronger noise reduction

        Returns:
            tuple: (processed_audio_array, sample_rate)
        """
        print("[Whisper] Applying transcription-optimized processing...")

        # Optional bypass: still enforce resample + mono via AudioDenoiser.load_audio()
        if not use_processing:
            passthrough = AudioDenoiser(
                sample_rate=self.sample_rate,
                noise_reduce_strength=0.0,
                highpass_cutoff=0,
                apply_preemphasis=False,
                normalize=False,
            )
            audio_np, sr = passthrough.load_audio(audio_path)
            return audio_np, sr

        # Step 1: Traditional cleaning
        strength = 0.5 if aggressive_cleaning else 0.2

        denoiser = AudioDenoiser(
            sample_rate=self.sample_rate,
            noise_reduce_strength=strength,
            highpass_cutoff=80,
            apply_preemphasis=True,
            preemphasis_coef=0.10,
            normalize=True
        )

        # Process in stages with temp files
        stage1_path = self._create_temp_path(audio_path, "whisper_stage1")

        # Stage 1: Denoise + normalize
        audio, sr = denoiser.full_pipeline(
            input_path=audio_path,
            output_path=stage1_path,
            use_clearvoice=False
        )

        # Stage 2: Optional ClearVoice
        if use_clearvoice:
            print("[Whisper] Applying ClearVoice enhancement...")
            stage2_path = self._create_temp_path(audio_path, "whisper_stage2")
            audio, sr = denoiser.enhance_with_clearvoice(
                input_path=stage1_path,
                output_path=stage2_path
            )
            print(f"[Whisper] ClearVoice processed: {stage2_path}")

        print(f"[Whisper] Final audio shape: {audio.shape}, SR: {sr}")
        return audio, sr

    # ==================== CONVENIENCE METHODS ====================

    def get_processing_config(self, model_type):
        """
        Get recommended processing settings for a model.

        Args:
            model_type (str): 'pyannote', 'whisper', 'embeddings'

        Returns:
            dict: Configuration dictionary
        """
        configs = {
            'pyannote': {
                'noise_reduce_strength': 0.2,
                'highpass_cutoff': 80,
                'apply_preemphasis': False,
                'normalize': False,
                'description': 'Minimal processing for speaker diarization'
            },
            'whisper': {
                'noise_reduce_strength': 0.5,
                'highpass_cutoff': 80,
                'apply_preemphasis': True,
                'normalize': True,
                'use_clearvoice': True,
                'description': 'Aggressive cleaning for transcription'
            },
            'embeddings': {
                'noise_reduce_strength': 0.3,
                'highpass_cutoff': 80,
                'apply_preemphasis': False,
                'normalize': False,
                'description': 'Moderate processing for speaker identification'
            }
        }

        return configs.get(model_type, configs['pyannote'])


# ==================== INTEGRATION EXAMPLE ====================

def example_full_pipeline(audio_path):
    """
    Example of how to use profiles in your transcription pipeline.
    """
    processor = AudioProcessingProfiles(sample_rate=16000)

    try:
        # 1. Process for Pyannote (diarization)
        pyannote_audio = processor.process_for_pyannote(audio_path)

        # Run diarization with this audio
        # segments = run_diarization(pyannote_audio, ...)

        # 2. Process for Whisper (transcription)
        whisper_audio, sr = processor.process_for_whisper(
            audio_path,
            use_clearvoice=True,
            aggressive_cleaning=True
        )

        # Run transcription with this audio
        # results = transcribe_segments(..., full_audio_np=whisper_audio, ...)

        print("[Pipeline] Both models processed with optimal settings!")

    finally:
        # Always cleanup temp files
        processor.cleanup_temp_files()


if __name__ == "__main__":
    # Test with your audio
    processor = AudioProcessingProfiles()

    # Show configurations
    for model in ['pyannote', 'whisper', 'embeddings']:
        config = processor.get_processing_config(model)
        print(f"\n{model.upper()} Config:")
        for key, value in config.items():
            print(f"  {key}: {value}")