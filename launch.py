import sys
import os
import logging
import warnings
import subprocess
import importlib
import importlib.util


def _check_python_version() -> None:
    """VocalTrace requires Python 3.12.10+ (3.12.x only).

    Early 3.12 builds (before 3.12.10) can raise:
      AttributeError: '_SixMetaPathImporter' object has no attribute '_path'
    which is a known CPython packaging bug fixed in 3.12.10.
    """
    v = sys.version_info
    ok = v.major == 3 and v.minor == 12 and v.micro >= 10
    if ok:
        return
    print(
        "[!] Unsupported Python version: "
        f"{v.major}.{v.minor}.{v.micro}\n"
        "\n"
        "VocalTrace requires Python 3.12.10 or a later 3.12.x patch.\n"
        "  - Not 3.11 / 3.13+\n"
        "  - Not 3.12.0–3.12.9 (broken six/importlib meta path bug)\n"
        "\n"
        "Install 3.12.10+ from https://www.python.org/downloads/\n"
        "Then recreate the venv, e.g.:\n"
        "  py -3.12 -m venv .venv\n"
        "  .venv\\Scripts\\activate.bat\n"
        "  python -c \"import sys; print(sys.version)\"  # confirm 3.12.10+\n",
        file=sys.stderr,
    )
    sys.exit(1)


def _torchcodec_dist_installed() -> bool:
    """True if the torchcodec *package* is installed (even if import is broken)."""
    try:
        import importlib.metadata as md

        md.version("torchcodec")
        return True
    except Exception:
        return False


def _purge_torchcodec_modules() -> None:
    for name in list(sys.modules):
        if name == "torchcodec" or name.startswith("torchcodec."):
            del sys.modules[name]
    importlib.invalidate_caches()


def _ensure_torchcodec_absent() -> None:
    """Remove torchcodec before any torch/torchaudio/pyannote import.

    A half-broken torchcodec install (common on Windows with torch 2.8+)
    prevents VocalTrace from starting. VocalTrace never uses TorchCodec;
    audio is loaded via soundfile + system ffmpeg.
    """
    if not _torchcodec_dist_installed() and importlib.util.find_spec("torchcodec") is None:
        return

    print(
        "[!] torchcodec is installed in this environment.\n"
        "    VocalTrace cannot start reliably with it present (broken FFmpeg DLLs).\n"
        "    Uninstalling torchcodec now…",
        flush=True,
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchcodec"],
        capture_output=True,
        text=True,
    )
    _purge_torchcodec_modules()

    still_there = _torchcodec_dist_installed() or (
        importlib.util.find_spec("torchcodec") is not None
    )
    if still_there or proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        print(
            "[ERROR] Could not remove torchcodec automatically.\n"
            "        Run this, then start VocalTrace again:\n"
            f"          \"{sys.executable}\" -m pip uninstall -y torchcodec\n",
            file=sys.stderr,
        )
        if detail:
            print(detail, file=sys.stderr)
        sys.exit(1)

    print("[OK] torchcodec removed. Continuing startup…", flush=True)


_check_python_version()
_ensure_torchcodec_absent()

# Project root (directory that contains launch.py)
_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load API keys from .env in the project root (installer creates it from .env.example).
# Does not override variables already set in the real environment.
def _load_dotenv() -> None:
    env_path = os.path.join(_ROOT, ".env")
    try:
        from dotenv import load_dotenv

        if os.path.isfile(env_path):
            load_dotenv(env_path, override=False)
            return
        # Fallback: allow starting without .env
        load_dotenv(override=False)
    except ImportError:
        # Minimal parser if python-dotenv is not installed yet
        if not os.path.isfile(env_path):
            return
        try:
            with open(env_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except OSError:
            pass


_load_dotenv()

# pyannote.audio.core.io always warns when torchcodec is missing. That is the
# desired state for VocalTrace (we pass in-memory waveforms). Silence it.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pyannote\.audio\.core\.io",
)
warnings.filterwarnings("ignore", message=r"(?s).*torchcodec.*")
warnings.filterwarnings("ignore", message=r"(?s).*libtorchcodec.*")
warnings.filterwarnings("ignore", message=r"(?s).*built-in audio decoding will fail.*")
warnings.filterwarnings("ignore", message=r".*No module named 'torchcodec'.*")

# Quiet HF / httpx request spam (working env does not flood the console)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.utils").setLevel(logging.WARNING)

import torch  # noqa: E402  — after version + torchcodec gate

# Ensure the 'src' directory is in the python path
current_dir = _ROOT
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)
log = logging.getLogger(__name__)
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
    log.debug("StatsPool patch applied")
except ImportError:
    print("[!] Warning: Could not patch StatsPool. Pyannote might not be installed correctly.")

# Now we can import the main app from src
# Basic logging (override with your own config as needed)
logging.basicConfig(level=os.environ.get("VOCALTRACE_LOGLEVEL", "INFO"))

from main import MainWindow, QApplication  # noqa: E402

if __name__ == "__main__":
    # Ensure working directory is Project Root (so voices/ folder is found)
    os.chdir(current_dir)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
