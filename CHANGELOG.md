# Changelog



All notable changes to this project are documented in this file.



The format is inspired by \*Keep a Changelog\* and follows semantic versioning where practical.



---



## \[0.3.0] – 2026-01-XX



### Highlights
- Added Diarization Model and Transcription Model selection

- \*\*Major Windows stability improvement\*\* by removing Conda ffmpeg dependency

- Audio pipeline cleanup and clearer responsibility boundaries

- Optional FlashAttention2 support with safe automatic fallback

- PySide6 version pinning for Qt6 stability



---



### Added

- Support for \*\*in-memory audio diarization\*\* with Pyannote using:

```python
{"waveform": torch.Tensor, "sample\_rate": int}

