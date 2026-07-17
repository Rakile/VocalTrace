#!/usr/bin/env bash
# VocalTrace — one-shot installer (Linux / macOS)
#   ./installation.sh              # interactive
#   ./installation.sh --yes        # defaults (CUDA if nvidia-smi, no clearvoice)
#   ./installation.sh cuda|cpu|mps [--yes]
set -euo pipefail
cd "$(dirname "$0")"

TORCH_VER="2.8.0"
VENV_DIR=".venv"
VENV_PY="${VENV_DIR}/bin/python"
TORCH_MODE="auto"
NONINTERACTIVE=0
WANT_FLASH=""
WANT_CLEARVOICE=""

while [[ $# -gt 0 ]]; do
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    --yes|-y) NONINTERACTIVE=1; shift ;;
    cuda|cpu|mps|auto) TORCH_MODE="$1"; shift ;;
    *) echo "[WARN] Unknown argument: $1"; shift ;;
  esac
done

echo
echo "============================================================"
echo "  VocalTrace installer"
echo "============================================================"
echo
echo "  This will:"
echo "    1. Find Python 3.12.10+"
echo "    2. Create .venv"
echo "    3. Install PyTorch ${TORCH_VER} + requirements"
echo "    4. Optionally FlashAttention and/or ClearVoice"
echo "    5. Write ./run.sh to start the app"
echo

has_nvidia=0
if command -v nvidia-smi >/dev/null 2>&1; then
  has_nvidia=1
  echo "  [detect] NVIDIA GPU:"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
else
  echo "  [detect] No nvidia-smi"
fi
echo

is_good_python() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || return 1
  "$cmd" -c 'import sys; v=sys.version_info; raise SystemExit(0 if (v.major==3 and v.minor==12 and v.micro>=10) else 1)' 2>/dev/null
}

resolve_python() {
  local candidates=(python3.12 python3 python)
  local cmd
  for cmd in "${candidates[@]}"; do
    if is_good_python "$cmd"; then
      command -v "$cmd"
      return 0
    fi
    if command -v "$cmd" >/dev/null 2>&1; then
      echo "[skip] $cmd is not Python 3.12.10+: $($cmd -c 'import sys; print(sys.version)' 2>/dev/null || true)" >&2
    fi
  done
  return 1
}

if ! PYTHON_EXE="$(resolve_python)"; then
  echo "[ERROR] No suitable Python found." >&2
  echo "Need Python 3.12.10+ (not early 3.12.x)." >&2
  echo "https://www.python.org/downloads/release/python-31210/" >&2
  exit 1
fi

echo "Using host Python:"
"$PYTHON_EXE" -c 'import sys; print(" ", sys.executable); print(" ", sys.version)'
echo

ask_yn() {
  # $1 prompt  $2 default Y|N
  local prompt="$1" def="$2" ans
  if [[ "$NONINTERACTIVE" -eq 1 ]]; then
    echo "$def"
    return
  fi
  read -r -p "$prompt" ans || true
  ans="${ans:-$def}"
  echo "$ans"
}

if [[ "$TORCH_MODE" == "auto" ]]; then
  uname_s="$(uname -s)"
  if [[ "$uname_s" == "Darwin" ]]; then
    TORCH_MODE="mps"
  elif [[ "$has_nvidia" -eq 1 ]]; then
    TORCH_MODE="cuda"
  else
    TORCH_MODE="cpu"
  fi
  if [[ "$NONINTERACTIVE" -eq 0 ]]; then
    echo "  Selected PyTorch target: ${TORCH_MODE}"
    echo "  Press Enter to keep, or type cuda / cpu / mps"
    read -r -p "  [ ${TORCH_MODE} ] : " override || true
    if [[ -n "${override:-}" ]]; then
      TORCH_MODE="$(echo "$override" | tr '[:upper:]' '[:lower:]')"
    fi
  fi
fi

if [[ "$TORCH_MODE" == "cuda" ]]; then
  WANT_FLASH="$(ask_yn "  Install FlashAttention2 if available? [Y/n]: " "Y")"
else
  WANT_FLASH="N"
  echo "  FlashAttention: skipped (not CUDA)."
fi
WANT_CLEARVOICE="$(ask_yn "  Install ClearVoice (optional AI enhancement)? [y/N]: " "N")"
echo

NEED_VENV=0
if [[ ! -x "$VENV_PY" ]]; then
  NEED_VENV=1
elif ! "$VENV_PY" -c 'import sys; v=sys.version_info; raise SystemExit(0 if (v.major==3 and v.minor==12 and v.micro>=10) else 1)' 2>/dev/null; then
  NEED_VENV=1
fi

if [[ "$NEED_VENV" -eq 1 ]]; then
  if [[ -d "$VENV_DIR" ]]; then
    echo "Existing venv is missing or uses wrong Python — recreating..."
    rm -rf "$VENV_DIR"
  else
    echo "Creating virtual environment in ${VENV_DIR} ..."
  fi
  "$PYTHON_EXE" -m venv "$VENV_DIR"
else
  echo "Reusing existing venv: ${VENV_DIR}"
fi

echo "Venv Python:"
"$VENV_PY" -c 'import sys; print(" ", sys.executable); print(" ", sys.version)'
echo

case "$(echo "$TORCH_MODE" | tr '[:upper:]' '[:lower:]')" in
  cuda)
    TORCH_INDEX="https://download.pytorch.org/whl/cu128"
    TORCH_LABEL="CUDA 12.8 (cu128) — torch==${TORCH_VER}"
    ;;
  cpu)
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_LABEL="CPU — torch==${TORCH_VER}"
    ;;
  mps)
    TORCH_INDEX=""
    TORCH_LABEL="macOS wheels — torch==${TORCH_VER}"
    ;;
  *)
    echo "[ERROR] Unknown TORCH_MODE: $TORCH_MODE" >&2
    exit 1
    ;;
esac

echo "[1/5] Installing PyTorch: ${TORCH_LABEL}"
"$VENV_PY" -m pip install --upgrade pip
if [[ -n "$TORCH_INDEX" ]]; then
  "$VENV_PY" -m pip install --upgrade \
    "torch==${TORCH_VER}" "torchaudio==${TORCH_VER}" \
    --index-url "$TORCH_INDEX"
else
  "$VENV_PY" -m pip install --upgrade \
    "torch==${TORCH_VER}" "torchaudio==${TORCH_VER}"
fi

echo
echo "[2/5] Installing VocalTrace requirements..."
"$VENV_PY" -m pip install -r requirements.txt
"$VENV_PY" -m pip install "transformers>=4.57.3,<5.0.0" "pyannote.audio==4.0.3"

echo
echo "[3/5] Optional FlashAttention..."
case "$(echo "$WANT_FLASH" | tr '[:upper:]' '[:lower:]')" in
  y|yes)
    # Prefer local wheel; else try pip (Linux may have a build path)
    WHEEL=""
    shopt -s nullglob
    for f in flash_attn*.whl ./wheels/flash_attn*.whl; do
      WHEEL="$f"
      break
    done
    shopt -u nullglob
    if [[ -n "$WHEEL" ]]; then
      echo "  Installing $WHEEL with --no-deps..."
      "$VENV_PY" -m pip install --no-deps "$WHEEL" || echo "  [WARN] wheel install failed"
    else
      echo "  No local wheel. Trying: pip install flash-attn --no-build-isolation"
      "$VENV_PY" -m pip install flash-attn --no-build-isolation \
        || echo "  [WARN] flash-attn install failed — app works without it"
    fi
    "$VENV_PY" -c 'import flash_attn; print("  [OK]", flash_attn.__version__)' 2>/dev/null \
      || echo "  [WARN] flash_attn import failed"
    ;;
  *)
    echo "  Skipped."
    ;;
esac

echo
echo "[4/5] Optional ClearVoice..."
case "$(echo "$WANT_CLEARVOICE" | tr '[:upper:]' '[:lower:]')" in
  y|yes)
    "$VENV_PY" -m pip install clearvoice || echo "  [WARN] clearvoice failed"
    echo "  Re-applying requirements pins..."
    "$VENV_PY" -m pip install -r requirements.txt
    "$VENV_PY" -m pip install "transformers>=4.57.3,<5.0.0" "pyannote.audio==4.0.3"
    ;;
  *)
    echo "  Skipped."
    ;;
esac

echo
echo "[5/5] Cleanup torchcodec (not used by VocalTrace)..."
"$VENV_PY" -m pip uninstall -y torchcodec 2>/dev/null \
  && echo "  [OK] torchcodec removed." \
  || echo "  torchcodec not present — OK."

echo
if command -v ffmpeg >/dev/null 2>&1; then
  echo "[OK] ffmpeg found on PATH."
  ffmpeg -version 2>/dev/null | head -n 1 || true
else
  echo "[WARN] ffmpeg not found on PATH."
  echo "       sudo apt-get install ffmpeg   # Debian/Ubuntu"
  echo "       brew install ffmpeg           # macOS"
fi

if [[ ! -f .env && -f .env.example ]]; then
  cp .env.example .env
  echo "[OK] Created .env from .env.example"
fi

cat > run.sh << 'EOF'
#!/usr/bin/env bash
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Virtual environment missing. Run ./installation.sh first."
  exit 1
fi
exec .venv/bin/python launch.py
EOF
chmod +x run.sh installation.sh 2>/dev/null || true

echo
echo "============================================================"
echo "  Install complete"
echo "============================================================"
"$VENV_PY" -c '
import torch
print("  torch         ", torch.__version__, "| cuda:", torch.cuda.is_available())
import transformers
print("  transformers  ", transformers.__version__)
try:
    import flash_attn
    print("  flash_attn    ", flash_attn.__version__)
except Exception:
    print("  flash_attn     not installed (optional)")
try:
    import clearvoice  # noqa: F401
    print("  clearvoice     installed")
except Exception:
    print("  clearvoice     not installed (optional)")
'
echo
echo "  Start VocalTrace:"
echo "      ./run.sh"
echo
echo "  Recommended in-app defaults:"
echo "      Diarization: pyannote/speaker-diarization-3.1"
echo "      Transcription: auto or KBLab/kb-whisper-large"
echo
echo "  Optional API keys: edit .env"
echo "============================================================"
echo

if [[ "$NONINTERACTIVE" -eq 0 ]]; then
  read -r -p "Launch VocalTrace now? [Y/n]: " launch || true
  launch="${launch:-Y}"
  case "$(echo "$launch" | tr '[:upper:]' '[:lower:]')" in
    n|no) ;;
    *) exec "$VENV_PY" launch.py ;;
  esac
fi
