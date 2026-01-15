import sys
import os
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtCore import QThread, Signal, QObject, Qt, QUrl
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QComboBox, QTabWidget, QMessageBox, QCheckBox,
    QProgressBar, QGroupBox, QSlider, QStyle
)

import json
import re
from typing import Callable

# Audio Imports
from ui.voice_manager_tab import VoiceManagerTab
from ui.chat_tab import ChatTab

from pipeline_manager import (
    run_full_pipeline,
    analyze_transcript,
    transcribe_audio_to_text,
)

from analysis_engine import (
    call_llm_openai,
    call_llm_gemini_25_pro,
    call_llm_lmstudio
)


def extract_text_content(lines_str):
    """
    Input: Multi-line string with timestamps/speakers
    Output: Combined text content only
    """
    lines_str = lines_str.replace('\u2029', '\n').replace('\u2028', '\n')
    clean_parts = []
    pattern = r"^\[.+?\](?:-\[.+?\])?\s+.*?:(.*?)$"

    for line in lines_str.split('\n'):
        line = line.strip()
        if not line: continue

        match = re.search(pattern, line)
        if match:
            content = match.group(1).strip()
            clean_parts.append(content)
        else:
            clean_parts.append(line)

    return " ".join(clean_parts)


def timestamp_to_seconds(ts):
    try:
        parts = ts.split(':')
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return (hours * 3600) + (minutes * 60) + seconds
    except:
        return 0.0


def fmt_ms(ms):
    seconds = (ms // 1000) % 60
    minutes = (ms // 60000) % 60
    hours = (ms // 3600000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _pick_backend(name: str) -> Callable[[str, str], str]:
    if "gemini" in name.lower():
        return call_llm_gemini_25_pro
    elif "lmstudio" in name.lower():
            return call_llm_lmstudio
    else:
        return call_llm_openai


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #os.environ["PYANNOTE_METRICS_ENABLED"] = "false"
        self.setWindowTitle("VocalTrace - The Conversation Analyzer")
        self.resize(1200, 900)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ------------------------------------------------------------------ #
        # 1. Top controls: Audio File
        # ------------------------------------------------------------------ #
        top_files = QHBoxLayout()
        root.addLayout(top_files)

        self.audio_label = QLabel("No audio file selected")
        self.btn_audio = QPushButton("Open audio...")
        self.btn_audio.clicked.connect(self.choose_audio_file)
        top_files.addWidget(self.audio_label, 3)
        top_files.addWidget(self.btn_audio, 1)
        self._original_audio_path = None
        # ------------------------------------------------------------------ #
        # 2. Fingerprints & Strategy
        # ------------------------------------------------------------------ #
        top_fingerprints = QHBoxLayout(); root.addLayout(top_fingerprints)
        self.fingerprints_label = QLabel("Voice Matching:")
        self.fingerprints_label.setStyleSheet("font-weight: bold;")
        #¤self.btn_fingerprints = QPushButton("Select Voices JSON...")
        #self.btn_fingerprints.clicked.connect(self.choose_fingerprints_file)

        self.combo_id_strategy = QComboBox()
        self.combo_id_strategy.addItems(["Method: Fast Voting (Clusters)", "Method: Brute Force"])

        self.combo_id_strategy.setToolTip(
            "Fast Voting: Checks 5 clips per speaker. Good for clean audio.\n"
            "Brute Force: Checks every single sentence. Best for messy audio."
        )

        self.btn_open_voices = QPushButton("Manage Voices")
        self.btn_open_voices.clicked.connect(lambda: self.tabs.setCurrentWidget(self.voice_tab))

        top_fingerprints.addWidget(self.fingerprints_label)
        top_fingerprints.addWidget(self.combo_id_strategy)
        top_fingerprints.addWidget(self.btn_open_voices)
        top_fingerprints.addStretch()

        # ------------------------------------------------------------------ #
        # 2.5. Diarization Model and Transcription Model
        # ------------------------------------------------------------------ #
        top_diarization_transcription_model = QHBoxLayout(); root.addLayout(top_diarization_transcription_model)

        self.diarization_model_label = QLabel("Diarization Model:")
        self.diarization_model_label.setStyleSheet("font-weight: bold;")
        self.combo_diarization_model = QComboBox()
        self.combo_diarization_model.addItems([
            "pyannote/speaker-diarization-community-1",
            "pyannote/speaker-diarization-3.1",
        ])
        self.combo_diarization_model.setToolTip(
            "Which Pyannote diarization model to use."
        )

        self.transcription_model_label = QLabel("Transcription Model:")
        self.transcription_model_label.setStyleSheet("font-weight: bold;")
        self.combo_transcription_model = QComboBox()
        self.combo_transcription_model.addItems([
            "auto",
            "openai/whisper-large-v3",
            "openai/whisper-large-v3-turbo",
            "openai/whisper-tiny",
            "KBLab/kb-whisper-large",
        ])
        self.combo_transcription_model.setToolTip(
            "Which Whisper transcription model to use."
        )


        top_diarization_transcription_model.addWidget(self.diarization_model_label)
        top_diarization_transcription_model.addWidget(self.combo_diarization_model)
        top_diarization_transcription_model.addWidget(self.transcription_model_label)
        top_diarization_transcription_model.addWidget(self.combo_transcription_model)
        top_diarization_transcription_model.addStretch(1)

        # ------------------------------------------------------------------ #
        # 3. Transcript File
        # ------------------------------------------------------------------ #
        top_transcript = QHBoxLayout()
        root.addLayout(top_transcript)

        self.transcript_label = QLabel("No transcript file selected")
        self.btn_transcript = QPushButton("Open transcript...")
        self.btn_transcript.clicked.connect(self.choose_transcript_file)
        top_transcript.addWidget(self.transcript_label, 3)
        top_transcript.addWidget(self.btn_transcript, 1)

        # ------------------------------------------------------------------ #
        # 4. Options Row (Lang, Backend, Mode)
        # ------------------------------------------------------------------ #
        options = QHBoxLayout()
        root.addLayout(options)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["auto", "sv", "en"])

        self.backend_combo = QComboBox()
        self.backend_combo.addItems([
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-5.1",
            "LMStudio/gpt-oss-20b"
        ])
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Transcribe + analyze audio",
            "Transcribe audio only (no analysis)",
            "Analyze transcript only",
        ])

        self.run_btn = QPushButton("Run analysis")
        self.run_btn.clicked.connect(self.run_analysis)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        self.save_btn = QPushButton("Save Transcript...")
        self.save_btn.clicked.connect(self.save_transcript)
        self.save_btn.setEnabled(False)

        # Add widgets to options layout
        options.addWidget(QLabel("Language:"))
        options.addWidget(self.lang_combo)
        options.addWidget(QLabel("Backend:"))
        options.addWidget(self.backend_combo)
        options.addWidget(QLabel("Mode:"))
        options.addWidget(self.mode_combo)
        options.addStretch(1)
        options.addWidget(self.save_btn)
        options.addWidget(self.run_btn)


        # ------------------------------------------------------------------ #
        # 4.5 Audio Cleaning Options
        # ------------------------------------------------------------------ #
        self.clean_group = QGroupBox("Audio Cleaning (optional)")
        clean_layout = QHBoxLayout()
        self.clean_group.setLayout(clean_layout)

        self.chk_clean_transcription = QCheckBox("Enable cleaning for transcription")
        self.chk_clean_transcription.setChecked(False)
        self.chk_clean_transcription.setToolTip(
            "If enabled, VocalTrace will preprocess audio before ASR.\n"
            "Disable this to run raw audio through the ASR model (good for debugging)."
        )

        self.combo_clean_level = QComboBox()
        self.combo_clean_level.addItems(["Light", "Aggressive"])
        self.combo_clean_level.setCurrentText("Light")
        self.combo_clean_level.setToolTip(
            "Light: gentle denoise (recommended default)\n"
            "Aggressive: stronger denoise + higher risk of artifacts"
        )

        self.chk_use_clearvoice = QCheckBox("Use ClearVoice (AI)")
        self.chk_use_clearvoice.setChecked(False)
        self.chk_use_clearvoice.setToolTip(
            "Optional deep-learning enhancement via ClearVoice.\n"
            "Only works if the clearvoice package is installed."
        )

        self.chk_clean_diarization = QCheckBox("Light cleaning for diarization")
        self.chk_clean_diarization.setChecked(False)
        self.chk_clean_diarization.setToolTip(
            "Optional light preprocessing before diarization.\n"
            "May help in very noisy files, but can also alter speaker characteristics."
        )

        clean_layout.addWidget(self.chk_clean_transcription)
        clean_layout.addWidget(QLabel("Level:"))
        clean_layout.addWidget(self.combo_clean_level)
        clean_layout.addWidget(self.chk_use_clearvoice)
        clean_layout.addWidget(self.chk_clean_diarization)

        self.btn_denoise_now = QPushButton("Clean Audio Now")
        self.btn_denoise_now.clicked.connect(self.run_denoise_process)
        self.btn_denoise_now.setStyleSheet("background-color: #e6f3ff; font-weight: bold;")
        clean_layout.addWidget(self.btn_denoise_now)

        clean_layout.addStretch(1)

        root.addWidget(self.clean_group)
        # Progress Bar Row
        prog_row = QHBoxLayout()
        prog_row.addWidget(self.progress_bar)
        root.addLayout(prog_row)

        # ------------------------------------------------------------------ #
        # 5. AUDIO PLAYER TOOLBAR (New)
        # ------------------------------------------------------------------ #
        self.player_group = QGroupBox("Audio Player & Sync")
        player_layout = QHBoxLayout()
        self.player_group.setLayout(player_layout)

        self.icon_play = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        self.icon_pause = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)

        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.icon_play)
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_play.setEnabled(False)

        self.lbl_time = QLabel("00:00:00 / 00:00:00")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.seek_audio)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)

        # Sync Checkboxes
        self.chk_sync_click = QCheckBox("Sync on Click")
        self.chk_sync_click.setToolTip("Clicking text jumps audio to that timestamp")
        self.chk_sync_click.setChecked(True)

        self.chk_auto_play = QCheckBox("Auto-play")
        self.chk_auto_play.setToolTip("Start playing immediately when jumping to a timestamp")
        self.chk_auto_play.setChecked(False)

        self.chk_stop_at_end = QCheckBox("Stop at End")
        self.chk_stop_at_end.setToolTip("Automatically pause when playback reaches the end of the selected segment.")
        self.chk_stop_at_end.setChecked(False)

        player_layout.addWidget(self.btn_play)
        player_layout.addWidget(self.lbl_time)
        player_layout.addWidget(self.slider)
        player_layout.addWidget(self.chk_sync_click)
        player_layout.addWidget(self.chk_auto_play)
        player_layout.addWidget(self.chk_stop_at_end)

        root.addWidget(self.player_group)

        # ------------------------------------------------------------------ #
        # 6. Tabs
        # ------------------------------------------------------------------ #
        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        self.transcript_edit = QTextEdit()
        self.transcript_edit.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.transcript_edit.customContextMenuRequested.connect(self.show_transcript_menu)

        # Connect cursor change for Sync
        self.transcript_edit.cursorPositionChanged.connect(self.on_cursor_changed)

        self.summary_edit = QTextEdit()
        self.summary_edit.setReadOnly(True)
        self.raw_json_edit = QTextEdit()
        self.raw_json_edit.setReadOnly(True)

        self.voice_tab = VoiceManagerTab(
            json_path="voices.json",
            hf_token=os.getenv("HF_TOKEN_TRANSCRIBE")
        )

        self.chat_tab = ChatTab()
        self.chat_tab.set_model_config(self.backend_combo.currentText())
        self.chat_tab.btn_refresh.clicked.connect(self.refresh_chat_index)
        self.backend_combo.currentTextChanged.connect(self.chat_tab.set_model_config)

        self.tabs.addTab(self.transcript_edit, "Transcript")
        self.tabs.addTab(self.summary_edit, "Summary & Themes")
        self.tabs.addTab(self.raw_json_edit, "Raw JSON")
        self.tabs.addTab(self.voice_tab, "Voice Bank 🎤")
        self.tabs.addTab(self.chat_tab, "Chat with Evidence 💬")

        # Internal state
        self._audio_path: str | None = None
        self._transcript_path: str | None = None
        self._fingerprints_path: str | None = None
        self._input_mode: str = "none"
        self._current_worker: Worker | None = None

        # Ground Truth Storage
        self.verified_segments = []

        # Player State
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)
        self.player.playbackStateChanged.connect(self.on_state_changed)

        self.is_slider_dragged = False
        self.last_cursor_block_number = -1
        self.play_limit_ms = None  # Limit for stop-at-end

        self.transcript_edit.textChanged.connect(self._on_transcript_edited)
        self.voice_tab.request_transcript_replace.connect(self.perform_transcript_replace)

    # ------------------------------------------------------------------ #
    # AUDIO PLAYER LOGIC
    # ------------------------------------------------------------------ #

    def choose_audio_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio file",
            "",
            "Audio Files (*.wav *.mp3 *.m4a *.flac);;All Files (*.*)",
        )
        if path:
            self._audio_path = path
            self._original_audio_path = path
            self.audio_label.setText(f"🎧 ORIGINAL: {os.path.basename(path)}")
            self._input_mode = "audio"
            self.clear_verified_segments()
            self.voice_tab.set_current_audio(path)

            # Init Player
            self.player.setSource(QUrl.fromLocalFile(path))
            self.btn_play.setEnabled(True)
            self.player_group.setTitle(f"Player: {os.path.basename(path)}")

    def run_denoise_process(self):
        if not self._audio_path:
            QMessageBox.warning(self, "No Audio", "Please open an audio file first.")
            return
        if self._audio_path != self._original_audio_path:
            QMessageBox.warning(self, "Audio has already been cleaned before.", "Please reload the original audio file again before cleaning.")
            return
        self._set_running(True)  # Disables all buttons as you intended
        self.progress_bar.setRange(0, 0)  # Pulsing progress

        level = self.combo_clean_level.currentText()
        use_cv = self.chk_use_clearvoice.isChecked()

        self.denoise_worker = DenoiseWorker(self._audio_path, level, use_cv)
        self.denoise_worker.finished.connect(self.on_denoise_finished)
        self.denoise_worker.error.connect(self.on_denoise_error)
        self.denoise_worker.start()

    def on_denoise_finished(self, cleaned_path):
        self._set_running(False)
        self.progress_bar.setRange(0, 100)

        # THE SWITCH:
        self._audio_path = cleaned_path
        self.audio_label.setText(f"🎧 ORIGINAL {os.path.basename(self._original_audio_path)} → ✨ CLEANED: {os.path.basename(cleaned_path)}")
        self.audio_label.setStyleSheet("color: green; font-weight: bold;")

        # Update all dependent components
        self.player.setSource(QUrl.fromLocalFile(cleaned_path))
        self.voice_tab.set_current_audio(cleaned_path)

        QMessageBox.information(self, "Success", "Audio cleaned and set as primary source.")

    def on_denoise_error(self, err):
        self._set_running(False)
        QMessageBox.critical(self, "Denoise Error", err)

    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def on_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.btn_play.setIcon(self.icon_pause)
        else:
            self.btn_play.setIcon(self.icon_play)

    def on_position_changed(self, position):
        if not self.is_slider_dragged:
            self.slider.setValue(position)
        self.update_time_label(position, self.player.duration())

        # Check Stop-At-End Limit
        if self.play_limit_ms is not None and self.chk_stop_at_end.isChecked():
            # Stop if we hit or exceed limit
            if position >= self.play_limit_ms:
                self.player.pause()
                self.play_limit_ms = None  # Reset limit

    def on_duration_changed(self, duration):
        self.slider.setRange(0, duration)
        self.update_time_label(self.player.position(), duration)

    def update_time_label(self, pos, dur):
        self.lbl_time.setText(f"{fmt_ms(pos)} / {fmt_ms(dur)}")

    def seek_audio(self, pos):
        self.player.setPosition(pos)
        self.play_limit_ms = None  # Manual seek cancels auto-stop

    def on_slider_pressed(self):
        self.is_slider_dragged = True

    def on_slider_released(self):
        self.is_slider_dragged = False
        self.player.setPosition(self.slider.value())

    def on_cursor_changed(self):
        """
        Called when user clicks or types in the transcript.
        Syncs audio to the timestamp of the current line.
        """
        if not self.chk_sync_click.isChecked(): return
        if self.tabs.currentIndex() != 0: return

        cursor = self.transcript_edit.textCursor()
        block_num = cursor.blockNumber()

        if block_num == self.last_cursor_block_number: return
        self.last_cursor_block_number = block_num

        line_text = cursor.block().text()

        # Regex to find [Start] and optionally [End]
        match = re.search(r"\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\](?:-\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\])?", line_text)

        if match:
            start_str = match.group(1)
            end_str = match.group(2)

            seconds = timestamp_to_seconds(start_str)
            ms = int(seconds * 1000)

            # Set Limit if checkbox active
            if self.chk_stop_at_end.isChecked() and end_str:
                self.play_limit_ms = int(timestamp_to_seconds(end_str) * 1000)
            else:
                self.play_limit_ms = None

            # Force jump (no threshold check anymore)
            self.player.setPosition(ms)

            if self.chk_auto_play.isChecked():
                self.player.play()
            else:
                self.player.pause()

    def set_verified_segments(self, segs: list[dict]):
        self.verified_segments = segs or []
        self.update_json_tab_with_verified()
        self.auto_save_sidecar()

    def clear_verified_segments(self):
        self.set_verified_segments([])

    # ------------------------------------------------------------------ #
    # VERIFIED SEGMENTS & EDITING LOGIC
    # ------------------------------------------------------------------ #

    def update_json_tab_with_verified(self):
        """
        Syncs self.verified_segments into the Raw JSON tab.
        """
        current_text = self.raw_json_edit.toPlainText()
        data = {}

        if current_text.strip():
            try:
                data = json.loads(current_text)
            except Exception:
                pass

        data["verified_segments"] = self.verified_segments

        pretty = json.dumps(data, ensure_ascii=False, indent=2)
        self.raw_json_edit.setPlainText(pretty)

    def perform_transcript_replace(self, new_block):
        """
        Called when Snipper 'Commits'. Replaces text AND updates Ground Truth.
        """
        if hasattr(self, 'pending_replacement_cursor') and self.pending_replacement_cursor:
            cursor = self.pending_replacement_cursor
            cursor.beginEditBlock()
            cursor.insertText(new_block)
            cursor.endEditBlock()
            self.transcript_edit.setTextCursor(cursor)
            self.transcript_edit.ensureCursorVisible()
            self.pending_replacement_cursor = cursor
        else:
            self.transcript_edit.append(new_block)

        #self.mark_selection_verified(new_block, silent=True)
        self.toggle_selection_verified(new_block, silent=True)


    def toggle_selection_verified(self, text_block: str, silent: bool = False):
        """
        Toggle ground-truth status for the selected transcript lines.
        - If a segment already exists in verified_segments -> remove it (unverify)
        - Else -> add it (verify)
        """
        lines = text_block.strip().split('\n')
        pattern = r"^\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]-\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]\s+(.*?):\s+(.*)$"

        added = 0
        removed = 0
        tol = 0.05  # seconds tolerance for float comparisons

        for line in lines:
            line = line.strip()
            if not line:
                continue

            m = re.search(pattern, line)
            if not m:
                continue

            start_str, end_str, speaker, text = m.groups()
            start_sec = timestamp_to_seconds(start_str)
            end_sec = timestamp_to_seconds(end_str)

            # Find "same" segment in existing verified list
            match_idx = None
            for i, s in enumerate(self.verified_segments):
                if (
                        abs(s.get("start", -1) - start_sec) <= tol and
                        abs(s.get("end", -1) - end_sec) <= tol and
                        s.get("speaker") == speaker
                ):
                    match_idx = i
                    break

            if match_idx is not None:
                self.verified_segments.pop(match_idx)
                removed += 1
            else:
                # Remove overlaps to avoid duplicates/clashes
                self.verified_segments = [
                    s for s in self.verified_segments
                    if not (s["start"] < end_sec and s["end"] > start_sec)
                ]
                self.verified_segments.append({
                    "start": start_sec,
                    "end": end_sec,
                    "speaker": speaker,
                    "text": text,
                    "is_verified": True
                })
                added += 1

        if added or removed:
            self.update_json_tab_with_verified()
            self.auto_save_sidecar()
            if not silent:
                QMessageBox.information(
                    self,
                    "Ground Truth Updated",
                    f"✅ Verified: {added}\n↩️ Unverified: {removed}"
                )
        elif not silent:
            QMessageBox.warning(
                self,
                "Error",
                "Could not parse selection. Format must be:\n"
                "[00:00:00]-[00:00:00] Speaker: Text"
            )


    def mark_selection_verified(self, text_block, silent=False):
        """
        Parses text block, updates verified_segments, updates JSON tab, auto-saves.
        """
        lines = text_block.strip().split('\n')
        pattern = r"^\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]-\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]\s+(.*?):\s+(.*)$"

        count = 0
        for line in lines:
            line = line.strip()
            if not line: continue

            match = re.search(pattern, line)
            if match:
                start_str, end_str, speaker, text = match.groups()
                start_sec = timestamp_to_seconds(start_str)
                end_sec = timestamp_to_seconds(end_str)

                # Remove overlaps
                self.verified_segments = [
                    s for s in self.verified_segments
                    if not (s["start"] < end_sec and s["end"] > start_sec)
                ]

                self.verified_segments.append({
                    "start": start_sec,
                    "end": end_sec,
                    "speaker": speaker,
                    "text": text,
                    "is_verified": True
                })
                count += 1

        if count > 0:
            print(f"[GUI] Verified {count} segments.")
            self.update_json_tab_with_verified()
            self.auto_save_sidecar()
            if not silent:
                QMessageBox.information(self, "Verified", f"✅ {count} segment(s) marked as Ground Truth.")
        elif not silent:
            QMessageBox.warning(self, "Error", "Could not parse selection. Format must be:\n[00:00:00]-[00:00:00] Speaker: Text")

    def auto_save_sidecar(self):
        source = self._transcript_path or self._audio_path
        if not source: return

        json_text = self.raw_json_edit.toPlainText()
        if not json_text.strip(): return

        base = os.path.splitext(source)[0]
        json_path = f"{base}_analyzed.json"

        try:
            data = json.loads(json_text)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to auto-save sidecar: {e}")

    def verify_selection(self, text_block: str, silent: bool = False):
        # “Set to verified” (verified stay verified; unverified become verified)
        self._set_selection_verified(text_block, make_verified=True, silent=silent)

    def unverify_selection(self, text_block: str, silent: bool = False):
        # “Set to unverified” (verified removed; unverified stay unverified)
        self._set_selection_verified(text_block, make_verified=False, silent=silent)

    def _set_selection_verified(self, text_block: str, make_verified: bool, silent: bool = False):
        lines = [ln.strip() for ln in text_block.strip().split("\n") if ln.strip()]
        pattern = r"^\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]-\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]\s+(.*?):\s+(.*)$"
        tol = 0.05

        changed = 0

        for line in lines:
            m = re.search(pattern, line)
            if not m:
                continue

            start_str, end_str, speaker, text = m.groups()
            start_sec = timestamp_to_seconds(start_str)
            end_sec = timestamp_to_seconds(end_str)

            # find existing match
            match_idx = None
            for i, s in enumerate(self.verified_segments):
                if (
                        abs(s.get("start", -1) - start_sec) <= tol and
                        abs(s.get("end", -1) - end_sec) <= tol and
                        s.get("speaker") == speaker
                ):
                    match_idx = i
                    break

            if make_verified:
                if match_idx is None:
                    # remove overlaps to avoid duplicates/clashes
                    self.verified_segments = [
                        s for s in self.verified_segments
                        if not (s["start"] < end_sec and s["end"] > start_sec)
                    ]
                    self.verified_segments.append({
                        "start": start_sec,
                        "end": end_sec,
                        "speaker": speaker,
                        "text": text,
                        "is_verified": True
                    })
                    changed += 1
            else:
                if match_idx is not None:
                    self.verified_segments.pop(match_idx)
                    changed += 1

        if changed:
            self.update_json_tab_with_verified()
            self.auto_save_sidecar()
            if not silent:
                QMessageBox.information(
                    self,
                    "Ground Truth Updated",
                    f"{'✅ Verified' if make_verified else '↩️ Unverified'}: {changed}"
                )
        elif not silent:
            pass
            #QMessageBox.information(self, "No changes", "Selection was already in the requested state.")

    def _iter_selected_segments(self, text_block: str):
        """
        Yields parsed segments from a selected transcript block.
        Only yields lines that match the expected format.
        """
        lines = [ln.strip() for ln in text_block.strip().split("\n") if ln.strip()]
        pattern = r"^\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]-\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]\s+(.*?):\s+(.*)$"

        for line in lines:
            m = re.search(pattern, line)
            if not m:
                continue
            start_str, end_str, speaker, text = m.groups()
            yield (timestamp_to_seconds(start_str), timestamp_to_seconds(end_str), speaker.strip(), text)

    def selection_verification_state(self, text_block: str) -> str:
        """
        Returns: "none" | "all" | "mixed"
        Robust: uses the same parsing rules as verify/unverify and
        does NOT misclassify mixed selections when some lines fail to parse.
        """
        tol = 0.05

        parsed = list(self._iter_selected_segments(text_block))
        if not parsed:
            return "none"

        def is_verified(seg) -> bool:
            start_sec, end_sec, speaker, _text = seg
            # Match by time; speaker is optional for state purposes (less surprising UX)
            for s in self.verified_segments:
                if (
                        abs(s.get("start", -1) - start_sec) <= tol and
                        abs(s.get("end", -1) - end_sec) <= tol
                ):
                    return True
            return False

        flags = [is_verified(seg) for seg in parsed]
        if all(flags):
            return "all"
        if any(flags):
            return "mixed"
        return "none"

    def selection_looks_verified(self, text_block: str) -> bool:
        pattern = r"^\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]-\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]\s+(.*?):\s+"
        tol = 0.05

        for line in text_block.strip().split("\n"):
            m = re.search(pattern, line.strip())
            if not m:
                continue
            start_sec = timestamp_to_seconds(m.group(1))
            end_sec = timestamp_to_seconds(m.group(2))
            speaker = m.group(3)

            for s in self.verified_segments:
                if (
                        abs(s.get("start", -1) - start_sec) <= tol and
                        abs(s.get("end", -1) - end_sec) <= tol and
                        s.get("speaker") == speaker
                ):
                    return True
        return False

    # ------------------------------------------------------------------ #
    # CONTEXT MENU
    # ------------------------------------------------------------------ #

    def show_transcript_menu(self, pos):
        main_cursor = self.transcript_edit.textCursor()
        click_cursor = self.transcript_edit.cursorForPosition(pos)
        click_pos = click_cursor.position()

        target_cursor = None
        if main_cursor.hasSelection() and (main_cursor.selectionStart() <= click_pos <= main_cursor.selectionEnd()):
            target_cursor = main_cursor
        else:
            target_cursor = click_cursor
            target_cursor.select(target_cursor.SelectionType.BlockUnderCursor)

        # Smart Snap
        start_pos = target_cursor.selectionStart() + 1
        end_pos = target_cursor.selectionEnd()
        temp_cursor = self.transcript_edit.textCursor()

        temp_cursor.setPosition(start_pos)
        temp_cursor.movePosition(temp_cursor.MoveOperation.StartOfBlock)
        real_start = temp_cursor.position()

        temp_cursor.setPosition(end_pos)
        if temp_cursor.atBlockStart() and temp_cursor.position() > real_start:
            temp_cursor.movePosition(temp_cursor.MoveOperation.PreviousBlock)
        temp_cursor.movePosition(temp_cursor.MoveOperation.EndOfBlock)
        real_end = temp_cursor.position()

        final_cursor = self.transcript_edit.textCursor()
        final_cursor.setPosition(real_start)
        final_cursor.setPosition(real_end, final_cursor.MoveMode.KeepAnchor)
        self.transcript_edit.setTextCursor(final_cursor)

        # Extraction
        selected_text = final_cursor.selectedText().replace('\u2029', '\n').replace('\u2028', '\n')

        # Determine if selection is parseable for verify/unverify
        parsed = list(self._iter_selected_segments(selected_text))
        has_parsed_segments = bool(parsed)

        # Derive "times" for Snipper (your existing logic)
        ts_pattern = r"\[(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]"
        speaker_pattern = r"\]\s+(.*?):\s"
        all_times = re.findall(ts_pattern, selected_text)
        times = None

        if all_times:
            seconds = [timestamp_to_seconds(t) for t in all_times]
            if seconds:
                start_s = min(seconds)
                end_s = max(seconds)
                clean_text = extract_text_content(selected_text)

                spk = None
                spk_match = re.search(speaker_pattern, selected_text)
                if spk_match:
                    spk = spk_match.group(1).strip()

                times = (start_s, end_s, clean_text, spk)

        # Create Menu
        menu = self.transcript_edit.createStandardContextMenu()
        menu.addSeparator()

        state = self.selection_verification_state(selected_text)

        action_verify = None
        action_unverify = None

        if state == "none":
            action_verify = menu.addAction("✅ Verify selection")
        elif state == "all":
            action_unverify = menu.addAction("↩️ Unverify selection")
        else:  # mixed
            action_verify = menu.addAction("✅ Verify selected (set all)")
            action_unverify = menu.addAction("↩️ Unverify selected (set all)")

        action_snip = menu.addAction("✂️ Refine / Split / Merge in Snipper")

        # Enable/disable actions
        action_snip.setEnabled(bool(times))
        if action_verify is not None:
            action_verify.setEnabled(has_parsed_segments)
        if action_unverify is not None:
            action_unverify.setEnabled(has_parsed_segments)

        action = menu.exec(self.transcript_edit.mapToGlobal(pos))

        if action == action_snip and times:
            start_s, end_s, text_content, spk = times
            self.pending_replacement_cursor = final_cursor
            self.voice_tab.open_snipper_at_range(start_s, end_s, initial_text=text_content, initial_speaker=spk)

        elif action == action_verify:
            self.verify_selection(selected_text, silent=False)

        elif action == action_unverify:
            self.unverify_selection(selected_text, silent=False)

    # ------------------------------------------------------------------ #
    # PIPELINE & STANDARD LOGIC
    # ------------------------------------------------------------------ #

    def refresh_chat_index(self):
        current_text = self.transcript_edit.toPlainText()
        if not current_text.strip():
            QMessageBox.warning(self, "Empty", "No text to index.")
            return
        source = self._transcript_path or self._audio_path or "Manual Edit"
        self.chat_tab.load_transcript(current_text, path=source)

    def _on_transcript_edited(self):
        text = self.transcript_edit.toPlainText()
        has_text = bool(text.strip())
        self.save_btn.setEnabled(has_text)
        if has_text:
            self._input_mode = "transcript"
        else:
            if self._input_mode == "transcript": self._input_mode = "none"

    def choose_fingerprints_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select JSON", "", "JSON (*.json);;All Files (*.*)")
        if path:
            self._fingerprints_path = path
            self.fingerprints_label.setText(os.path.basename(path))
            self.fingerprints_label.setStyleSheet("color: black;")

    def choose_transcript_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select transcript", "", "Text (*.txt);;All Files (*.*)")
        if path:
            self._transcript_path = path
            self.transcript_label.setText(os.path.basename(path))
            self.transcript_edit.setPlainText("Loading file... please wait...")
            self.transcript_edit.setEnabled(False)
            self.loader_worker = FileLoaderWorker(path)
            self.loader_worker.finished.connect(self.on_transcript_loaded)
            self.loader_worker.error.connect(self.on_loader_error)
            self.loader_worker.start()

    def on_loader_error(self, err_msg):
        self.transcript_edit.setEnabled(True)
        self.transcript_edit.setPlainText("")
        QMessageBox.critical(self, "Load Error", f"Failed to read file:\n{err_msg}")

    def on_transcript_loaded(self, result):
        self.transcript_edit.setEnabled(True)
        text = result["text"]
        path = result["path"]
        json_data = result["json_data"]

        self.transcript_edit.setPlainText(text)
        self._input_mode = "transcript"

        analysis_found = False
        persona_to_set = None
        self.raw_json_edit.clear()
        self.summary_edit.setPlainText("No analysis loaded.")

        if json_data:
            print(f"[GUI] Loaded sidecar: {result['json_path']}")

            # Load Verified Segments (canonical)
            self.set_verified_segments(json_data.get("verified_segments", []))
            print(f"[GUI] Loaded {len(self.verified_segments)} verified segments.")

            try:
                pretty = json.dumps(json_data, ensure_ascii=False, indent=2)
                self.raw_json_edit.setPlainText(pretty)

                norm = json_data.get("analysis_normalized")
                if norm:
                    analysis_found = True
                    summary = norm.get("summary_short") or json_data.get("summary_short", "")
                    self.summary_edit.setPlainText(summary)
                    persona_to_set = norm.get("recommended_ai_persona")
                else:
                    self.summary_edit.setPlainText("Transcription metadata loaded.\nNo LLM analysis found.")
            except Exception as e:
                print(f"[GUI] Error processing JSON data: {e}")
        else:
            self.set_verified_segments([])

        self.chat_tab.load_transcript(text, path=path, has_analysis=analysis_found)
        if persona_to_set:
            self.chat_tab.set_persona(persona_to_set, is_generic=False)
        else:
            self.chat_tab.set_persona("You are a helpful assistant analyzing a transcript.", is_generic=True)

    def run_analysis(self):
        transcript_text = self.transcript_edit.toPlainText().strip()
        lang = self.lang_combo.currentText()
        backend_label = self.backend_combo.currentText()
        call_llm = _pick_backend(backend_label)
        mode = self.mode_combo.currentText()

        if mode.startswith("Transcribe + analyze"):
            if not self._audio_path:
                QMessageBox.information(self, "No audio", "Select an audio file.")
                return
            self._do_full_pipeline(lang, call_llm, backend_label)
            return

        if mode.startswith("Transcribe audio only"):
            if not self._audio_path:
                QMessageBox.information(self, "No audio", "Select an audio file.")
                return
            self._do_transcribe_only(lang)
            return

        if mode.startswith("Analyze transcript"):
            if not transcript_text:
                QMessageBox.information(self, "No transcript", "Load or paste transcript.")
                return
            self._do_analyze_transcript(transcript_text, lang, call_llm, backend_label)
            return

    def _set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.run_btn.setText("Running…" if running else "Run analysis")
        self.save_btn.setEnabled(False if running else bool(self.transcript_edit.toPlainText().strip()))
        for w in [self.btn_audio, self.btn_transcript, self.combo_id_strategy, self.combo_diarization_model, self.combo_transcription_model, self.btn_open_voices, self.lang_combo, self.backend_combo, self.mode_combo, self.tabs, self.btn_denoise_now]:
            w.setEnabled(not running)

    def _do_analyze_transcript(self, transcript: str, analysis_language: str, call_llm, backend_label: str):
        self._set_running(True)
        self.progress_bar.setValue(0)

        def job(progress_callback=None):
            return analyze_transcript(
                transcript, call_llm=call_llm, analysis_language=analysis_language,
                backend=backend_label, progress_callback=progress_callback
            )

        worker = Worker(job)
        self._current_worker = worker
        worker.progress.connect(self._handle_progress)

        def on_finished(out: dict):
            self._current_worker = None
            self.progress_bar.setValue(0)
            self._populate_outputs(transcript, out)
            self._set_running(False)
            self._handle_progress(0)
            self.auto_save_sidecar()

        worker.finished.connect(on_finished)
        worker.error.connect(lambda msg: (self._set_running(False), QMessageBox.critical(self, "Error", msg)))
        worker.start()

    def _do_transcribe_only(self, language: str):
        self._set_running(True)
        self.progress_bar.setValue(0)

        # Get Checked Voices
        active_voices = self.voice_tab.get_active_fingerprints()
        count = len(active_voices)
        print(f"[GUI] Using {count} selected voices for identification.")

        strat_text = self.combo_id_strategy.currentText()
        strategy = "brute" if "Brute" in strat_text else "cluster"

        diarization_model_text = self.combo_diarization_model.currentText()
        transcription_model_text = self.combo_transcription_model.currentText()

        #strategy = "brute" if "Brute" in strat_text else "cluster"


        def job(progress_callback=None):
            return transcribe_audio_to_text(
                self._audio_path,
                language=language,
                hf_token=os.getenv("HF_TOKEN_TRANSCRIBE"),
                fingerprints_dict=active_voices,  # Pass DICT now
                fingerprints_path=None,
                id_strategy=strategy,
                diarization_model= diarization_model_text,
                transcription_model=transcription_model_text,
                progress_callback=progress_callback,
                verified_segments=self.verified_segments  # Pass Truth
            )

        worker = Worker(job)
        self._current_worker = worker
        worker.progress.connect(self._handle_progress)

        def on_finished(out: dict):
            self._current_worker = None
            self.progress_bar.setValue(0)
            transcript = out.get("transcript", "")
            self.transcript_edit.setPlainText(transcript)

            minimal = {
                "audio_path": out.get("audio_path"),
                "base_name": out.get("base_name"),
                "segments": out.get("segments", []),
                "verified_segments": self.verified_segments,
                "language": out.get("language"),
                "note": "Audio-only transcription."
            }
            pretty = json.dumps(minimal, ensure_ascii=False, indent=2)
            self.raw_json_edit.setPlainText(pretty)
            self.summary_edit.setPlainText("Audio-only transcription mode.")
            self._set_running(False)
            self._handle_progress(0)

        worker.finished.connect(on_finished)
        worker.error.connect(lambda msg: (self._set_running(False), QMessageBox.critical(self, "Error", msg)))
        worker.start()

    def _do_full_pipeline(self, language: str, call_llm, backend_label: str):
        self._set_running(True)
        self.progress_bar.setValue(0)

        # Get Checked Voices
        active_voices = self.voice_tab.get_active_fingerprints()
        count = len(active_voices)
        print(f"[GUI] Using {count} selected voices for identification.")

        strat_text = self.combo_id_strategy.currentText()
        strategy = "brute" if "Brute" in strat_text else "cluster"

        diarization_model_text = self.combo_diarization_model.currentText()
        transcription_model_text = self.combo_transcription_model.currentText()

        def job(progress_callback=None):
            return run_full_pipeline(
                self._audio_path,
                call_llm=call_llm,
                language=language,
                #analysis_language=language,
                hf_token=os.getenv("HF_TOKEN_TRANSCRIBE"),
                backend=backend_label,
                fingerprints_dict=active_voices,
                fingerprints_path=None,
                id_strategy=strategy,
                diarization_model=diarization_model_text,
                transcription_model=transcription_model_text,
                progress_callback=progress_callback,
                verified_segments=self.verified_segments,
            )

        worker = Worker(job)
        self._current_worker = worker
        worker.progress.connect(self._handle_progress)

        def on_finished(out: dict):
            self._current_worker = None
            self.progress_bar.setValue(0)
            self._populate_outputs(out["transcript"], out["analysis"])
            self._set_running(False)
            self._handle_progress(0)
            self.auto_save_sidecar()

        worker.finished.connect(on_finished)
        worker.error.connect(lambda msg: (self._set_running(False), QMessageBox.critical(self, "Error", msg)))
        worker.start()

    def _handle_progress(self, value: int):
        if value < 0:
            if self.progress_bar.maximum() != 0: self.progress_bar.setRange(0, 0)
        else:
            if self.progress_bar.maximum() == 0: self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(value)

    def _populate_outputs(self, transcript: str, analysis_dict: dict):
        self.transcript_edit.setPlainText(transcript)

        # Inject verified info
        analysis_dict["verified_segments"] = self.verified_segments

        pretty = json.dumps(analysis_dict, ensure_ascii=False, indent=2)
        self.raw_json_edit.setPlainText(pretty)

        norm = analysis_dict.get("analysis_normalized", {})
        self.summary_edit.setPlainText(norm.get("summary_short", ""))

        persona = norm.get("recommended_ai_persona")
        if persona:
            self.chat_tab.set_persona(persona, is_generic=False)
        else:
            self.chat_tab.set_persona("Generic Assistant", is_generic=True)

        source = self._audio_path or self._transcript_path or "Generated"
        self.chat_tab.load_transcript(transcript, path=source, has_analysis=True)

    def save_transcript(self):
        text = self.transcript_edit.toPlainText()
        if not text.strip(): return

        suggested = "transcript.txt"
        if self._audio_path:
            suggested = f"{os.path.splitext(os.path.basename(self._audio_path))[0]}_edited.txt"
        elif self._transcript_path:
            suggested = f"{os.path.splitext(os.path.basename(self._transcript_path))[0]}_edited.txt"

        path, _ = QFileDialog.getSaveFileName(self, "Save Transcript", suggested, "Text (*.txt);;All Files (*.*)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)

                # Save Sidecar
                json_text = self.raw_json_edit.toPlainText()
                saved_json = None
                if json_text.strip():
                    base = os.path.splitext(path)[0]
                    json_path = f"{base}_analyzed.json"
                    data = json.loads(json_text)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    saved_json = json_path

                msg = f"Transcript saved: {os.path.basename(path)}"
                if saved_json: msg += f"\nSidecar saved: {os.path.basename(saved_json)}"
                QMessageBox.information(self, "Saved", msg)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))


class DenoiseWorker(QThread):
    finished = Signal(str)  # Returns the path to the cleaned file
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, audio_path, profile_level="Light", use_clearvoice=False):
        super().__init__()
        self.audio_path = audio_path
        self.profile_level = profile_level
        self.use_clearvoice = use_clearvoice

    def run(self):
        try:
            from audio_processing_profiles import AudioProcessingProfiles
            # We don't use a 'temp' directory here because we want this file
            # to persist for the duration of the user's session.
            processor = AudioProcessingProfiles()

            # Use the Whisper profile logic as it's the most aggressive/clean
            # but we return the path to the file on disk.
            cleaned_np, sr = processor.process_for_whisper(
                self.audio_path,
                use_processing=True,
                use_clearvoice=self.use_clearvoice,
                aggressive_cleaning=(self.profile_level == "Aggressive")
            )

            # Save the result as a 'sidecar' cleaned file
            base, _ = os.path.splitext(self.audio_path)
            out_path = f"{base}_cleaned.wav"

            import soundfile as sf
            sf.write(out_path, cleaned_np, sr)

            self.finished.emit(out_path)
        except Exception as e:
            self.error.emit(str(e))

class FileLoaderWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, path):
        super().__init__(); self.path = path

    def run(self):
        try:
            res = {"text": "", "path": self.path, "json_data": None}
            with open(self.path, "r", encoding="utf-8") as f:
                res["text"] = f.read()

            base = os.path.splitext(self.path)[0]
            for p in [f"{base}_analyzed.json", f"{base}.json"]:
                if os.path.exists(p):
                    with open(p, 'r', encoding='utf-8') as jf:
                        res["json_data"] = json.load(jf)
                        res["json_path"] = p
                    break
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class Worker(QThread):
    finished = Signal(object);
    error = Signal(str);
    progress = Signal(int)

    def __init__(self, fn, *args, **kwargs):
        super().__init__(); self._fn = fn; self._args = args; self._kwargs = kwargs

    def run(self):
        try:
            self.finished.emit(self._fn(progress_callback=lambda c, t: self.progress.emit(-1 if t == 0 else int(c / t * 100))))
        except TypeError:  # Fallback for no progress
            try:
                self.finished.emit(self._fn(*self._args, **self._kwargs))
            except Exception as e:
                self.error.emit(str(e))
        except Exception as e:
            self.error.emit(str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())