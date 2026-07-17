"""In-app API key / settings editor (writes project-root .env)."""

from __future__ import annotations

import os
import re
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Keys we expose in the dialog (order matters for .env writes)
SETTINGS_KEYS = (
    "HF_TOKEN_TRANSCRIBE",
    "OPENAI_API_KEY",
    "GEMINI_TRANSCRIBE_ANALYSIS_API_KEY",
)


def project_root() -> Path:
    # src/ui/settings_dialog.py → parents[2] == project root
    return Path(__file__).resolve().parents[2]


def env_path() -> Path:
    return project_root() / ".env"


def read_env_file(path: Path | None = None) -> dict[str, str]:
    path = path or env_path()
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, _, val = raw.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            values[key] = val
    return values


def write_env_keys(updates: dict[str, str], path: Path | None = None) -> Path:
    """Update keys in .env, preserve comments/other lines when possible."""
    path = path or env_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Start from example if no .env yet
    if not path.is_file():
        example = path.parent / ".env.example"
        if example.is_file():
            path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []

    remaining = dict(updates)
    out: list[str] = []
    key_re = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*)$")

    for line in lines:
        m = key_re.match(line)
        if m and m.group(2) in remaining:
            key = m.group(2)
            val = remaining.pop(key)
            out.append(f"{key}={val}")
        else:
            out.append(line)

    if remaining:
        if out and out[-1].strip():
            out.append("")
        out.append("# --- Saved from VocalTrace Settings ---")
        for key, val in remaining.items():
            out.append(f"{key}={val}")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


def apply_env_to_process(values: dict[str, str]) -> None:
    """Push values into os.environ for the current session."""
    for key, val in values.items():
        if val:
            os.environ[key] = val
        elif key in os.environ:
            # Empty means clear for this session if user blanked it
            del os.environ[key]


def _env_or_file(*keys: str) -> str | None:
    for key in keys:
        val = (os.getenv(key) or "").strip()
        if val:
            return val
    file_vals = read_env_file()
    for key in keys:
        val = (file_vals.get(key) or "").strip()
        if val:
            return val
    return None


def get_hf_token() -> str | None:
    return _env_or_file("HF_TOKEN_TRANSCRIBE", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN")


def get_openai_api_key() -> str | None:
    return _env_or_file("OPENAI_API_KEY")


def get_gemini_api_key() -> str | None:
    return _env_or_file(
        "GEMINI_TRANSCRIBE_ANALYSIS_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    )


def key_status_summary() -> str:
    """Short status for the main window status strip."""
    parts = []
    parts.append("HF ✓" if get_hf_token() else "HF ✗")
    parts.append("OpenAI ✓" if get_openai_api_key() else "OpenAI ✗")
    parts.append("Gemini ✓" if get_gemini_api_key() else "Gemini ✗")
    return "  ·  ".join(parts)


class SettingsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("VocalTrace Settings — API keys")
        self.setMinimumWidth(580)

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Keys are saved to <b>.env</b> in the project folder and applied immediately. "
            "They are not uploaded anywhere by VocalTrace except to the service you choose "
            "(Hugging Face / OpenAI / Google) when you run analysis."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()

        def _secret_row(placeholder: str) -> tuple[QLineEdit, QHBoxLayout]:
            row = QHBoxLayout()
            edit = QLineEdit()
            edit.setEchoMode(QLineEdit.EchoMode.Password)
            edit.setPlaceholderText(placeholder)
            show = QCheckBox("Show")
            show.toggled.connect(
                lambda on, e=edit: e.setEchoMode(
                    QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
                )
            )
            row.addWidget(edit, 1)
            row.addWidget(show)
            return edit, row

        # HF token
        self.hf_edit, hf_row = _secret_row("hf_…  required for Pyannote diarization")
        form.addRow("Hugging Face token", hf_row)
        hf_help = QLabel(
            "<b>HF_TOKEN_TRANSCRIBE</b> — diarization only.<br>"
            'Token: <a href="https://huggingface.co/settings/tokens">huggingface.co/settings/tokens</a> '
            "(read access is enough).<br>"
            "Also accept model access on the same account, e.g. "
            '<a href="https://huggingface.co/pyannote/speaker-diarization-3.1">speaker-diarization-3.1</a> · '
            '<a href="https://huggingface.co/pyannote/speaker-diarization-community-1">community-1</a>.'
        )
        hf_help.setOpenExternalLinks(True)
        hf_help.setWordWrap(True)
        hf_help.setTextFormat(Qt.TextFormat.RichText)
        form.addRow("", hf_help)

        # OpenAI
        self.openai_edit, oai_row = _secret_row("sk-…  for GPT backends / chat analysis")
        form.addRow("OpenAI API key", oai_row)
        oai_help = QLabel(
            "<b>OPENAI_API_KEY</b> — used when Backend is a <b>gpt-*</b> model "
            "(summary, themes, chat-with-evidence).<br>"
            'Create a key: <a href="https://platform.openai.com/api-keys">platform.openai.com/api-keys</a><br>'
            "Not required for “Transcribe audio only”."
        )
        oai_help.setOpenExternalLinks(True)
        oai_help.setWordWrap(True)
        oai_help.setTextFormat(Qt.TextFormat.RichText)
        form.addRow("", oai_help)

        # Gemini
        self.gemini_edit, gem_row = _secret_row("for gemini-* analysis backends")
        form.addRow("Gemini API key", gem_row)
        gem_help = QLabel(
            "<b>GEMINI_TRANSCRIBE_ANALYSIS_API_KEY</b> — used when Backend is "
            "<b>gemini-2.5-flash</b> or <b>gemini-2.5-pro</b>.<br>"
            'Get a key: <a href="https://aistudio.google.com/apikey">aistudio.google.com/apikey</a><br>'
            "Aliases also accepted: <code>GEMINI_API_KEY</code>, <code>GOOGLE_API_KEY</code>. "
            "Not required for “Transcribe audio only”."
        )
        gem_help.setOpenExternalLinks(True)
        gem_help.setWordWrap(True)
        gem_help.setTextFormat(Qt.TextFormat.RichText)
        form.addRow("", gem_help)

        layout.addLayout(form)

        link_row = QHBoxLayout()
        open_env = QPushButton("Open .env folder")
        open_env.clicked.connect(self._open_env_folder)
        link_row.addWidget(open_env)
        link_row.addStretch()
        layout.addLayout(link_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load()

    def _load(self) -> None:
        self.hf_edit.setText(get_hf_token() or "")
        self.openai_edit.setText(get_openai_api_key() or "")
        self.gemini_edit.setText(get_gemini_api_key() or "")

    def _save(self) -> None:
        updates = {
            "HF_TOKEN_TRANSCRIBE": self.hf_edit.text().strip(),
            "OPENAI_API_KEY": self.openai_edit.text().strip(),
            "GEMINI_TRANSCRIBE_ANALYSIS_API_KEY": self.gemini_edit.text().strip(),
        }
        try:
            path = write_env_keys(updates)
            apply_env_to_process(updates)
        except OSError as e:
            QMessageBox.critical(self, "Could not save", str(e))
            return
        QMessageBox.information(
            self,
            "Saved",
            f"Keys written to:\n{path}\n\nThey apply to this session immediately.",
        )
        self.accept()

    def _open_env_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(project_root())))
