# Changelog



All notable changes to this project are documented in this file.



The format is inspired by \*Keep a Changelog\* and follows semantic versioning where practical.



---



## \[0.3.1] – 2026-07-17

### Added
- Guided installer: detect GPU, ask for **FlashAttention** / **ClearVoice**, create `run.bat`/`run.sh`,
  optional launch, `.env.example`
- `installation.bat` / `installation.sh` — find Python **3.12.10+**, create `.venv`, install
  tested **torch/torchaudio 2.8.0** (cu128 auto on NVIDIA) + pinned transformers 4.57.x / pyannote 4.0.3

### Fixed
- Install docs now require **Python 3.12.10+** (not generic “3.12”), documenting the known
  `AttributeError: '_SixMetaPathImporter' object has no attribute '_path'` on 3.12.0–3.12.9
- `launch.py` exits early with a clear message if the interpreter is not 3.12.10+
- Windows activate path documented as `.venv\Scripts\activate.bat`
- **TorchCodec must be absent**: audio I/O uses soundfile + system ffmpeg
  (`src/audio_io.py`); `torchaudio.load()` removed from `AudioDenoiser`
- Installers uninstall torchcodec after deps; **`launch.py` auto-removes it**
  before startup (app will not start reliably while torchcodec is installed)
- **Transcription failures `Failed segment N: 0`**: decode path was using
  `batch_decode(..., output_offsets=True)` + `.get("text")`, which raises
  `KeyError(0)` on transformers 5.x — fixed to plain `skip_special_tokens` decode
- GUI **diarization model** selection was ignored (`run_diarization` always got default 3.1)

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

