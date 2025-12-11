import os
import soundfile as sf
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QMessageBox, QGroupBox, QCheckBox,
    QLineEdit, QTextEdit
)
from PySide6.QtCore import QUrl, QTimer, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

class AudioSnipperDialog(QDialog):
    clip_saved = Signal(str, str)
    transcript_replacement_requested = Signal(str)

    def __init__(self, audio_path, existing_profiles, output_folder="voices", initial_text="", initial_speaker=None):
        super().__init__()
        self.setWindowTitle("Audio Snipper Workbench")
        self.resize(900, 700)  # Taller window for the batch view

        self.audio_path = audio_path
        self.profiles = existing_profiles
        self.output_folder = output_folder
        self.sr = 16000

        # State
        self.start_ms = 0
        self.end_ms = 0
        self.staged_lines = []  # List to hold the split lines

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # --- MEDIA PLAYER ---
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.setSource(QUrl.fromLocalFile(audio_path))

        self.timer = QTimer()
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.update_playback_ui)
        self.timer.start()

        # --- UI LAYOUT ---
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 1. Info
        self.lbl_info = QLabel(f"Editing: {os.path.basename(audio_path)}")
        self.lbl_info.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.lbl_info)

        instructions = QLabel("Drag on waveform to select. Spacebar to play/pause.")
        instructions.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(instructions)

        # 2. WAVEFORM
        self.waveform = WaveformWidget()
        self.waveform.load_audio(audio_path)
        self.waveform.selectionChanged.connect(self.on_waveform_select)
        layout.addWidget(self.waveform, 1)

        # 3. Transport
        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play Selection")
        self.btn_play.clicked.connect(self.play_selection)
        self.btn_play.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.lbl_sel_info = QLabel("Selection: 00:00:00 - 00:00:00")
        self.lbl_sel_info.setStyleSheet("font-family: Consolas, Monospace;")

        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.lbl_sel_info)
        layout.addLayout(ctrl_layout)

        # -------------------------------------------------------------
        # 4. TRANSCRIPT BUILDER (The "Splitter" Logic)
        # -------------------------------------------------------------
        edit_group = QGroupBox("Transcript Editor / Splitter")
        edit_layout = QVBoxLayout()
        edit_group.setLayout(edit_layout)

        # Input Row
        input_row = QHBoxLayout()
        self.txt_content = QLineEdit()
        self.txt_content.setText(initial_text)
        self.txt_content.setPlaceholderText("Text for this specific segment...")

        self.chk_auto_save = QCheckBox("Save Voice")
        self.chk_auto_save.setToolTip("Auto-save voice clip when adding segment")
        self.chk_auto_save.setStyleSheet("color: #2e8b57; font-weight: bold;")

        self.btn_add_segment = QPushButton("⬇ Add Segment")
        self.btn_add_segment.clicked.connect(self.add_segment_to_batch)

        self.btn_add_segment.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_add_segment.setStyleSheet("background-color: #f0f0f0; font-weight: bold;")

        input_row.addWidget(self.txt_content, 1)
        input_row.addWidget(self.chk_auto_save)
        input_row.addWidget(self.btn_add_segment)
        edit_layout.addLayout(input_row)

        # Batch Preview Area
        self.lbl_batch = QLabel("Staged Segments (Click 'Commit' to replace original selection with these):")
        edit_layout.addWidget(self.lbl_batch)

        self.batch_preview = QTextEdit()
        self.batch_preview.setReadOnly(True)
        self.batch_preview.setPlaceholderText("Added segments will appear here...")
        self.batch_preview.setMaximumHeight(150)
        self.batch_preview.setStyleSheet("background-color: #f9f9f9; color: #333;")
        self.batch_preview.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        edit_layout.addWidget(self.batch_preview)

        # Commit Button
        btn_commit_layout = QHBoxLayout()
        self.btn_clear_batch = QPushButton("Clear List")
        self.btn_clear_batch.clicked.connect(self.clear_batch)
        self.btn_clear_batch.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.btn_commit = QPushButton("✅ Commit / Replace in Transcript")
        self.btn_commit.clicked.connect(self.commit_changes)
        self.btn_commit.setStyleSheet("background-color: #add8e6; font-weight: bold; padding: 8px;")
        self.btn_commit.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        btn_commit_layout.addWidget(self.btn_clear_batch)
        btn_commit_layout.addWidget(self.btn_commit)
        edit_layout.addLayout(btn_commit_layout)

        layout.addWidget(edit_group)

        # 5. Export Clip (Existing)
        save_group = QGroupBox("Export Audio Clip (Voice Bank)")
        save_layout = QHBoxLayout()
        save_group.setLayout(save_layout)

        self.combo_profiles = QComboBox()
        self.combo_profiles.addItems(self.profiles)
        self.combo_profiles.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.btn_save = QPushButton("✂️ Save Audio Only")
        self.btn_save.clicked.connect(self.save_clip)
        self.btn_save.setStyleSheet("background-color: #d0f0c0; padding: 5px;")
        self.btn_save.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        save_layout.addWidget(QLabel("Assign to:"))
        save_layout.addWidget(self.combo_profiles, 1)
        save_layout.addWidget(self.btn_save)

        layout.addWidget(save_group)

        try:
            info = sf.info(self.audio_path)
            self.sr = info.samplerate
        except:
            pass

        if initial_speaker:
            self.set_current_profile(initial_speaker)

    # --- HELPERS ---
    def fmt_timestamp(self, ms):
        seconds = (ms // 1000) % 60
        minutes = (ms // 60000) % 60
        hours = (ms // 3600000)
        msec = int(ms % 1000)
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{msec:03d}"

    def fmt_duration(self, ms):
        seconds = ms / 1000.0
        if seconds < 60:
            return f"{seconds:.3f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60):02d}m {seconds % 60:06.3f}s"
        else:
            return f"{int(seconds // 3600):02d}h {int((seconds % 3600) // 60):02d}m {int(seconds % 60):02d}s"

    # --- LOGIC ---
    def update_playback_ui(self):
        current_pos = self.player.position()
        self.waveform.set_playhead(current_pos)
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            if self.end_ms > self.start_ms:
                if current_pos >= self.end_ms:
                    self.player.pause()
                    self.btn_play.setText("Play Selection")
                    self.waveform.set_playhead(self.end_ms)
                    self.player.setPosition(int(self.end_ms))

    def on_waveform_select(self, start_ms, end_ms):
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.txt_content.clearFocus()  # Ensure Spacebar works

        t_start = self.fmt_timestamp(start_ms)
        t_end = self.fmt_timestamp(end_ms)
        t_dur = self.fmt_duration(end_ms - start_ms)
        self.lbl_sel_info.setText(f"Selection: {t_start} - {t_end} (Duration: {t_dur})")

        if self.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            self.player.setPosition(int(start_ms))

    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.btn_play.setText("Play Selection")
        else:
            current_pos = self.player.position()
            if self.start_ms <= current_pos < (self.end_ms - 100):
                self.player.play()
            else:
                self.player.setPosition(int(self.start_ms))
                self.player.play()
            self.btn_play.setText("Pause")

    def play_selection(self):
        self.player.setPosition(int(self.start_ms))
        self.player.play()
        self.btn_play.setText("Pause")

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Space:
            if not self.txt_content.hasFocus():
                self.toggle_playback()
            else:
                super().keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.player.stop()
        super().closeEvent(event)

    # --- BATCH LOGIC ---

    def set_current_profile(self, speaker_name):
        index = self.combo_profiles.findText(speaker_name)
        if index >= 0:
            self.combo_profiles.setCurrentIndex(index)
        else:
            # Optional: Add "Unknown" or the raw speaker name temporarily?
            # For now, just ignore if not in list, or user selects manually
            pass

    def add_segment_to_batch(self):
        content = self.txt_content.text().strip()
        if not content:
            QMessageBox.warning(self, "Empty Text", "Enter text for this segment.")
            return
        if self.end_ms <= self.start_ms: return

        profile = self.combo_profiles.currentText()
        t_start = self.fmt_timestamp(self.start_ms)
        t_end = self.fmt_timestamp(self.end_ms)

        # Create line
        line = f"[{t_start}]-[{t_end}] {profile}: {content}"
        self.staged_lines.append(line)

        # Update View
        self.batch_preview.setText("\n\n".join(self.staged_lines))
        self.batch_preview.moveCursor(self.batch_preview.textCursor().MoveOperation.End)

        # --- Auto Save ---

        if self.chk_auto_save.isChecked():
            # We call save_clip internally, but suppress the "Saved" messagebox
            # to keep the workflow fluid
            self.save_clip(silent=True)

        # Prepare for next segment:
        # Move start time to current end time
        next_start = self.end_ms
        # Default next segment length? Let's say 2 seconds or just 0
        next_end = next_start + 2000

        self.set_selection_range(next_start, next_end)
        self.txt_content.clear()
        self.txt_content.setFocus()  # Ready to type next line

    def clear_batch(self):
        self.staged_lines = []
        self.batch_preview.clear()

    def commit_changes(self):
        # Case 1: Batch Mode (Multiple lines staged)
        if self.staged_lines:
            # Join with double newlines for readability
            final_block = "\n\n".join(self.staged_lines)
            self.transcript_replacement_requested.emit(final_block)
            self.lbl_info.setText("✅ Batch Sent!")
            #QTimer.singleShot(1500, self.accept)  # Close dialog on success? Or just clear?
            # Usually closing is expected here.

        # Case 2: Quick Mode (No batch, just replace using current inputs)
        else:
            content = self.txt_content.text().strip()
            if not content:
                QMessageBox.warning(self, "Nothing to commit", "Add segments to the list or type text to replace.")
                return

            profile = self.combo_profiles.currentText()
            t_start = self.fmt_timestamp(self.start_ms)
            t_end = self.fmt_timestamp(self.end_ms)
            line = f"[{t_start}]-[{t_end}] {profile}: {content}"

            self.transcript_replacement_requested.emit(line)
            self.lbl_info.setText("✅ Sent!")
            #QTimer.singleShot(1000, lambda: self.lbl_info.setText(f"Editing: {os.path.basename(self.audio_path)}"))
            #QTimer.singleShot(1500, self.accept)  # Close dialog on success? Or just clear?

    # --- EXPORT AUDIO ---
    def save_clip(self, silent=False):
        if self.end_ms <= self.start_ms + 100: return
        profile = self.combo_profiles.currentText()
        if not profile: return

        safe = "".join(x for x in profile if x.isalnum() or x in " _-")
        path = os.path.join(self.output_folder, f"{safe}_{int(self.start_ms)}.wav")

        try:
            sf.write(path, self.read_audio_chunk(), self.sr)  # Simplified read
            if not silent:
                self.lbl_info.setText(f"✅ Saved: {os.path.basename(path)}")
            else:
                self.lbl_info.setText(f"✅ Added & Saved Voice.")
            self.clip_saved.emit(path, profile)
        except Exception as e:
            if not silent: QMessageBox.critical(self, "Error", str(e))

    def read_audio_chunk(self):
        start = int((self.start_ms/1000)*self.sr)
        end = int((self.end_ms/1000)*self.sr)
        d, _ = sf.read(self.audio_path, start=start, stop=end)
        return d

    # --- Helper for programmatic set ---
    def set_selection_range(self, start_ms, end_ms):
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.waveform.sel_start_sec = start_ms / 1000.0
        self.waveform.sel_end_sec = end_ms / 1000.0

        # Zoom logic
        duration_ms = end_ms - start_ms
        duration_sec = duration_ms / 1000.0
        center_sec = (start_ms + (duration_ms / 2)) / 1000.0
        view_width = max(0.1, duration_sec * 1.2)
        new_view_start = max(0.0, center_sec - (view_width / 2))
        total_duration = float(self.waveform.duration_seconds)
        new_view_end = min(total_duration, new_view_start + view_width)

        self.waveform.view_start_sec = new_view_start
        self.waveform.view_end_sec = new_view_end
        self.waveform.update()

        # Update Labels
        self.on_waveform_select(start_ms, end_ms)  # Reuse this for label update

        # Seek
        self.player.setPosition(int(start_ms))
        self.waveform.set_playhead(int(start_ms))

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QFontMetrics, QCursor
import numpy as np
import math

class WaveformWidget(QWidget):
    selectionChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setMouseTracking(True)  # Essential for hover cursors

        # Colors & Config
        self.ruler_height = 25
        self.ruler_bg_color = QColor("#333333")
        self.wave_bg_color = QColor("#2b2b2b")
        self.text_color = QColor("#aaaaaa")
        self.handle_width = 8  # How close (pixels) to grab the edge

        # Data
        self.peaks = None
        self.duration_seconds = 0
        self.sr = 16000

        # Viewport
        self.view_start_sec = 0.0
        self.view_end_sec = 10.0

        # Selection
        self.sel_start_sec = 0.0
        self.sel_end_sec = 0.0

        # Playhead
        self.playhead_ms = -1

        # Interaction State
        self.drag_mode = "none"  # "none", "new", "adjust_start", "adjust_end", "pan"
        self.last_mouse_x = 0

    def load_audio(self, path):
        try:
            info = sf.info(path)
            self.duration_seconds = info.duration
            self.sr = info.samplerate

            self.view_end_sec = self.duration_seconds

            target_points = 50000
            total_samples = info.frames
            samples_per_chunk = max(1600, total_samples // target_points)

            peaks = []
            with sf.SoundFile(path) as f:
                while f.tell() < f.frames:
                    data = f.read(samples_per_chunk)
                    if len(data.shape) > 1: data = data.mean(axis=1)
                    if len(data) > 0: peaks.append(np.max(np.abs(data)))

            self.peaks = np.array(peaks, dtype=np.float32)
            self.update()
        except Exception as e:
            print(f"Error loading waveform: {e}")

    def set_playhead(self, ms):
        self.playhead_ms = ms
        self.update()

    def time_to_x(self, t_sec):
        view_duration = self.view_end_sec - self.view_start_sec
        if view_duration <= 0: return 0
        rel_time = t_sec - self.view_start_sec
        return (rel_time / view_duration) * self.width()

    def x_to_time(self, x):
        view_duration = self.view_end_sec - self.view_start_sec
        rel_pct = x / self.width()
        return self.view_start_sec + (rel_pct * view_duration)

    def calculate_tick_interval(self):
        view_duration = self.view_end_sec - self.view_start_sec
        width = self.width()
        if width == 0: return 10

        pixels_per_sec = width / view_duration
        target_interval_sec = 120 / pixels_per_sec

        intervals = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0]

        best_interval = intervals[-1]
        for interval in intervals:
            if interval >= target_interval_sec:
                best_interval = interval
                break
        return best_interval

    def format_time_label(self, t_sec, interval):
        hours = int(t_sec // 3600)
        minutes = int((t_sec % 3600) // 60)
        seconds = int(t_sec % 60)
        base_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        if interval < 1.0:
            ms = int((t_sec - int(t_sec)) * 1000)
            return f"{base_str}.{ms:03d}"
        return base_str

    def paintEvent(self, event):
        painter = QPainter(self)
        width = self.width()
        height = self.height()

        # Backgrounds
        painter.fillRect(0, 0, width, self.ruler_height, self.ruler_bg_color)
        painter.fillRect(0, self.ruler_height, width, height - self.ruler_height, self.wave_bg_color)

        if self.peaks is None: return

        # Draw Ruler
        painter.setPen(QPen(self.text_color, 1))
        painter.setFont(self.font())
        fm = QFontMetrics(painter.font())

        tick_interval = self.calculate_tick_interval()
        first_tick_time = math.ceil(self.view_start_sec / tick_interval) * tick_interval
        current_time = first_tick_time
        last_label_end_x = -100

        while current_time < self.view_end_sec:
            x = self.time_to_x(current_time)
            painter.setPen(QColor("#777777"))
            painter.drawLine(int(x), 0, int(x), self.ruler_height)
            label = self.format_time_label(current_time, tick_interval)
            text_w = fm.horizontalAdvance(label)
            text_x = x + 4
            if (text_x + text_w < width) and (x > last_label_end_x + 10):
                painter.setPen(self.text_color)
                painter.drawText(int(text_x), self.ruler_height - 6, label)
                last_label_end_x = text_x + text_w
            current_time += tick_interval

        # Draw Waveform
        painter.setPen(QPen(QColor("#00ccff"), 1))
        wave_h = height - self.ruler_height
        mid_y = self.ruler_height + (wave_h / 2)
        total_chunks = len(self.peaks)
        seconds_per_chunk = self.duration_seconds / total_chunks if total_chunks else 1

        start_idx = int(self.view_start_sec / seconds_per_chunk)
        end_idx = int(self.view_end_sec / seconds_per_chunk) + 1
        start_idx = max(0, start_idx)
        end_idx = min(total_chunks, end_idx)

        visible_peaks = self.peaks[start_idx:end_idx]
        step = max(1, len(visible_peaks) // width)

        for i in range(0, len(visible_peaks), step):
            peak = visible_peaks[i]
            time_sec = (start_idx + i) * seconds_per_chunk
            x = self.time_to_x(time_sec)
            h = peak * wave_h
            painter.drawLine(int(x), int(mid_y - h / 2), int(x), int(mid_y + h / 2))

        # ----------------------------------------
        # Draw Selection
        # ----------------------------------------
        draw_start = min(self.sel_start_sec, self.sel_end_sec)
        draw_end = max(self.sel_start_sec, self.sel_end_sec)

        if draw_end > draw_start:
            x1 = self.time_to_x(draw_start)
            x2 = self.time_to_x(draw_end)
            x1, x2 = max(0, x1), min(width, x2)

            if x2 > x1:
                rect = QRectF(x1, 0, x2 - x1, height)
                painter.fillRect(rect, QColor(255, 255, 0, 40))

                # Draw Edges Thicker to imply "Draggable"
                painter.setPen(QPen(QColor("yellow"), 2))  # Thicker line
                painter.drawLine(int(x1), 0, int(x1), height)
                painter.drawLine(int(x2), 0, int(x2), height)

        # Draw Playhead
        if self.playhead_ms >= 0:
            ph_sec = self.playhead_ms / 1000.0
            if self.view_start_sec <= ph_sec <= self.view_end_sec:
                x_ph = self.time_to_x(ph_sec)
                painter.setPen(QPen(QColor("#ff3333"), 2))
                painter.drawLine(int(x_ph), 0, int(x_ph), height)
                painter.setBrush(QColor("#ff3333"))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPolygon([QPointF(x_ph, 0), QPointF(x_ph - 6, 12), QPointF(x_ph + 6, 12)])

    # --- INTERACTION LOGIC (UPDATED) ---

    def get_hover_zone(self, x):
        """
        Returns 'start', 'end', or None based on x position.
        Uses normalized time values.
        """
        real_start = min(self.sel_start_sec, self.sel_end_sec)
        real_end = max(self.sel_start_sec, self.sel_end_sec)

        # Don't grab handles if selection is tiny/non-existent
        if (real_end - real_start) <= 0:
            return None

        x_start = self.time_to_x(real_start)
        x_end = self.time_to_x(real_end)

        # Check distance
        if abs(x - x_start) <= self.handle_width:
            return "adjust_start"
        if abs(x - x_end) <= self.handle_width:
            return "adjust_end"
        return None

    def mousePressEvent(self, event):
        self.last_mouse_x = event.position().x()

        if event.button() == Qt.MouseButton.LeftButton:
            # 1. Normalize current selection first
            real_start = min(self.sel_start_sec, self.sel_end_sec)
            real_end = max(self.sel_start_sec, self.sel_end_sec)
            self.sel_start_sec = real_start
            self.sel_end_sec = real_end

            # 2. Check if clicking a handle
            zone = self.get_hover_zone(self.last_mouse_x)

            if zone:
                self.drag_mode = zone  # "adjust_start" or "adjust_end"
            else:
                # Start fresh selection
                self.drag_mode = "new"
                t = self.x_to_time(self.last_mouse_x)
                self.sel_start_sec = max(0, min(t, self.duration_seconds))
                self.sel_end_sec = self.sel_start_sec

            self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            self.drag_mode = "pan"

    def mouseMoveEvent(self, event):
        current_x = event.position().x()

        # --- Cursor Logic (Hover) ---
        if self.drag_mode == "none":
            zone = self.get_hover_zone(current_x)
            if zone:
                self.setCursor(Qt.CursorShape.SizeHorCursor)  # ↔ Arrow
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)  # Standard

        # --- Drag Logic ---
        t = self.x_to_time(current_x)
        t = max(0, min(t, self.duration_seconds))  # Clamp

        if self.drag_mode == "new":
            self.sel_end_sec = t
            self.emit_selection()
            self.update()

        elif self.drag_mode == "adjust_start":
            # Constraint: Start cannot cross End?
            # Actually, standard behavior allows crossing (swapping roles),
            # but for simple handles, hard stop is often nicer.
            # Let's allow crossing because paintEvent handles min/max anyway.
            self.sel_start_sec = t
            self.emit_selection()
            self.update()

        elif self.drag_mode == "adjust_end":
            self.sel_end_sec = t
            self.emit_selection()
            self.update()

        elif self.drag_mode == "pan":
            pixel_delta = self.last_mouse_x - current_x
            view_width_time = self.view_end_sec - self.view_start_sec
            time_delta = (pixel_delta / self.width()) * view_width_time

            new_start = self.view_start_sec + time_delta
            new_end = self.view_end_sec + time_delta

            # Boundary checks
            if new_start < 0:
                diff = -new_start
                new_start += diff
                new_end += diff
            if new_end > self.duration_seconds:
                diff = new_end - self.duration_seconds
                new_start -= diff
                new_end -= diff
            if new_start < 0: new_start = 0  # Short file safety

            self.view_start_sec = new_start
            self.view_end_sec = new_end
            self.last_mouse_x = current_x
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Normalize final selection so start < end
            if self.sel_start_sec > self.sel_end_sec:
                self.sel_start_sec, self.sel_end_sec = self.sel_end_sec, self.sel_start_sec

            self.drag_mode = "none"
            self.emit_selection()
            self.update()

        elif event.button() == Qt.MouseButton.RightButton:
            self.drag_mode = "none"

    def emit_selection(self):
        s = min(self.sel_start_sec, self.sel_end_sec)
        e = max(self.sel_start_sec, self.sel_end_sec)
        self.selectionChanged.emit(s * 1000, e * 1000)

    def wheelEvent(self, event):
        zoom_factor = 0.9 if event.angleDelta().y() > 0 else 1.1
        mouse_t = self.x_to_time(event.position().x())
        current_view_len = self.view_end_sec - self.view_start_sec
        new_view_len = current_view_len * zoom_factor
        ratio = (mouse_t - self.view_start_sec) / current_view_len
        new_start = mouse_t - (new_view_len * ratio)
        new_end = new_start + new_view_len
        if new_start < 0: new_start = 0
        if new_end > self.duration_seconds: new_end = self.duration_seconds
        if (new_end - new_start) < 0.1: return
        self.view_start_sec = new_start
        self.view_end_sec = new_end
        self.update()