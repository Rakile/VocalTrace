import os
import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QPushButton, QLabel, QFileDialog, QLineEdit, QMessageBox, QGroupBox,
    QListWidgetItem, QMenu, QSplitter, QFrame, QCheckBox
)
from PySide6.QtCore import Qt, Signal, QThread

from biometrics import SpeakerIdentifier
from ui.audio_utils import AudioSnipperDialog

# --- Worker for Testing (Same as before) ---
class VoiceTestWorker(QThread):
    finished_result = Signal(str, str)
    error = Signal(str)

    def __init__(self, identifier, audio_path, sample_rate_threshold=0.55):
        super().__init__()
        self.identifier = identifier
        self.audio_path = audio_path
        self.threshold = sample_rate_threshold

    def run(self):
        try:
            wav, sr = self.identifier.preload_wave(self.audio_path)
            total_seconds = wav.shape[1] / sr
            analyzed_duration = min(total_seconds, 30)
            location_msg = "Middle" if total_seconds > 600 else "Start"
            match = self.identifier.identify_segment(wav, sample_rate=sr, threshold=self.threshold)
            info_text = f"(Checked {int(analyzed_duration)}s at {location_msg})"
            if match:
                msg = f"Result: ✅ MATCH: {match} {info_text}"
                style = "background: #d0f0c0; padding: 10px; font-weight: bold; color: black;"
            else:
                msg = f"Result: ❌ NO MATCH {info_text}"
                style = "background: #ffcccb; padding: 10px; font-weight: bold; color: black;"
            self.finished_result.emit(msg, style)
        except Exception as e:
            self.error.emit(str(e))


# --- NEW UI LAYOUT ---
class VoiceManagerTab(QWidget):
    request_transcript_replace = Signal(str)

    def __init__(self, json_path="voices.json", hf_token=None):
        super().__init__()
        self.json_path = json_path
        self.hf_token = hf_token
        self.identifier = None
        self.current_worker = None
        self.current_audio_path = None

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # TOP: Split View
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- LEFT PANE: Profiles ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Header with Select All/None
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("1. Profiles (Check to include in Analysis)"))

        self.btn_all = QPushButton("All")
        self.btn_all.setMaximumWidth(40)
        self.btn_all.clicked.connect(lambda: self.toggle_all(True))

        self.btn_none = QPushButton("None")
        self.btn_none.setMaximumWidth(55)
        self.btn_none.clicked.connect(lambda: self.toggle_all(False))

        header_layout.addStretch()
        header_layout.addWidget(self.btn_all)
        header_layout.addWidget(self.btn_none)

        left_layout.addLayout(header_layout)

        self.profile_list = QListWidget()
        self.profile_list.itemClicked.connect(self.on_profile_selected)
        # Enable context menu if needed later

        # Profile Controls
        prof_ctrl = QHBoxLayout()
        self.new_profile_input = QLineEdit()
        self.new_profile_input.setPlaceholderText("New Name...")
        self.btn_add_profile = QPushButton("Create")
        self.btn_add_profile.clicked.connect(self.add_profile)
        self.btn_del_profile = QPushButton("Delete")
        self.btn_del_profile.setStyleSheet("color: red")
        self.btn_del_profile.clicked.connect(self.delete_profile)

        prof_ctrl.addWidget(self.new_profile_input)
        prof_ctrl.addWidget(self.btn_add_profile)
        prof_ctrl.addWidget(self.btn_del_profile)

        left_layout.addWidget(self.profile_list)
        left_layout.addLayout(prof_ctrl)

        # --- RIGHT PANE: Samples (Unchanged) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.lbl_samples = QLabel("2. Voice Samples")
        right_layout.addWidget(self.lbl_samples)
        self.sample_list = QListWidget()

        samp_ctrl = QHBoxLayout()
        self.btn_add_sample = QPushButton("Add Audio Clip...")
        self.btn_add_sample.clicked.connect(self.add_sample)
        self.btn_snip = QPushButton("✂️ Snip from Main Audio...")
        self.btn_snip.clicked.connect(self.open_snipper)
        self.btn_del_sample = QPushButton("Remove Clip")
        self.btn_del_sample.clicked.connect(self.remove_sample)

        samp_ctrl.addWidget(self.btn_add_sample)
        samp_ctrl.addWidget(self.btn_snip)
        samp_ctrl.addWidget(self.btn_del_sample)

        right_layout.addWidget(self.sample_list)
        right_layout.addLayout(samp_ctrl)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        main_layout.addWidget(splitter, 1)

        # BOTTOM: Test Zone (Unchanged)
        test_group = QGroupBox("Test Verification")
        test_layout = QVBoxLayout()
        test_group.setLayout(test_layout)
        self.btn_test = QPushButton("Select Unknown Clip to Identify...")
        self.btn_test.clicked.connect(self.test_voice)
        self.test_result_label = QLabel("Result: [Waiting]")
        self.test_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.test_result_label.setStyleSheet("background: #f0f0f0; padding: 8px; border: 1px solid #ccc;")
        test_layout.addWidget(self.btn_test)
        test_layout.addWidget(self.test_result_label)
        main_layout.addWidget(test_group)

        self.load_json()

    # --- DATA LOGIC ---
    def set_current_audio(self, path):
        self.current_audio_path = path
        # Optional: Update button text to show it's ready
        if path:
            self.btn_snip.setText(f"✂️ Snip from: {os.path.basename(path)}")
            self.btn_snip.setToolTip(path)

    request_transcript_replace = Signal(str)

    def open_snipper_at_range(self, start_sec, end_sec, initial_text="", initial_speaker=None):
        # 1. Ensure we have an audio file
        if not self.current_audio_path or not os.path.exists(self.current_audio_path):
            QMessageBox.warning(self, "No Audio", "Please open the main audio file first (via 'Open audio...').")
            return

        # 2. Check if Snipper is already open
        if hasattr(self, 'snipper_dlg') and self.snipper_dlg.isVisible():
            # Update existing window
            self.snipper_dlg.set_selection_range(start_sec * 1000, end_sec * 1000)
            self.snipper_dlg.txt_content.setText(initial_text)

            if initial_speaker:
                self.snipper_dlg.set_current_profile(initial_speaker)

            self.snipper_dlg.raise_()
            self.snipper_dlg.activateWindow()
        else:
            # 3. Create NEW window
            # --- FIX: Calculate profile_names here ---
            profile_names = self.get_profile_names()
            if not profile_names:
                QMessageBox.warning(self, "No Profiles", "Create a profile in the list first.")
                return
            # -----------------------------------------

            self.snipper_dlg = AudioSnipperDialog(
                self.current_audio_path, self.get_profile_names(),
                output_folder="voices", initial_text=initial_text, initial_speaker=initial_speaker
            )

            # Connect signals
            self.snipper_dlg.clip_saved.connect(self.on_clip_saved_externally)
            self.snipper_dlg.transcript_replacement_requested.connect(self.request_transcript_replace.emit)

            self.snipper_dlg.show()

            # Set range immediately after showing
            self.snipper_dlg.set_selection_range(start_sec * 1000, end_sec * 1000)

    def get_profile_names(self):
        names = []
        for i in range(self.profile_list.count()):
            names.append(self.profile_list.item(i).data(Qt.ItemDataRole.UserRole))
        return names

    def open_snipper(self):
        path = self.current_audio_path
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Select Source Audio", "", "Audio (*.wav *.mp3)")
        if not path: return

        # Use helper
        profile_names = self.get_profile_names()
        if not profile_names:
            QMessageBox.warning(self, "No Profiles", "Create a profile first.")
            return

        self.snipper_dlg = AudioSnipperDialog(path, profile_names, output_folder="voices")
        self.snipper_dlg.clip_saved.connect(self.on_clip_saved_externally)

        # Connect replacement signal here too (just in case opened manually)
        self.snipper_dlg.transcript_replacement_requested.connect(self.request_transcript_replace.emit)

        self.snipper_dlg.show()

    def on_clip_saved_externally(self, new_path, profile_name):
        # Find profile
        for i in range(self.profile_list.count()):
            item = self.profile_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == profile_name:
                # Add to data
                current = item.data(Qt.ItemDataRole.UserRole + 1)
                current.append(new_path)
                item.setData(Qt.ItemDataRole.UserRole + 1, current)

                # Save JSON
                self.save_data()

                # Refresh UI if selected
                if item.isSelected():
                    self.on_profile_selected(item)
                break

    def toggle_all(self, state):
        for i in range(self.profile_list.count()):
            item = self.profile_list.item(i)
            item.setCheckState(Qt.CheckState.Checked if state else Qt.CheckState.Unchecked)

    def get_active_fingerprints(self):
        """Returns a dictionary of ONLY the checked profiles."""
        active_map = {}
        for i in range(self.profile_list.count()):
            item = self.profile_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                name = item.data(Qt.ItemDataRole.UserRole)
                samples = item.data(Qt.ItemDataRole.UserRole + 1)
                active_map[name] = samples
        return active_map

    def load_json(self):
        self.profile_list.clear()
        self.sample_list.clear()

        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for name, content in data.items():
                    if isinstance(content, str): content = [content]

                    item = QListWidgetItem(f"👤 {name}")
                    item.setData(Qt.ItemDataRole.UserRole, name)
                    item.setData(Qt.ItemDataRole.UserRole + 1, content)

                    # Make Checkable
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    # Default to Checked? Or Unchecked? Let's default to Checked for ease.
                    item.setCheckState(Qt.CheckState.Checked)

                    self.profile_list.addItem(item)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"JSON Load Error: {e}")

    def on_profile_selected(self, item):
        self.sample_list.clear()
        name = item.data(Qt.ItemDataRole.UserRole)
        samples = item.data(Qt.ItemDataRole.UserRole + 1)

        self.lbl_samples.setText(f"Samples for: {name}")

        for path in samples:
            short_name = os.path.basename(path)
            s_item = QListWidgetItem(f"💿 {short_name}")
            s_item.setToolTip(path)
            s_item.setData(Qt.ItemDataRole.UserRole, path)
            self.sample_list.addItem(s_item)

    def save_data(self):
        # Reconstruct JSON from UI List
        new_data = {}
        for i in range(self.profile_list.count()):
            item = self.profile_list.item(i)
            name = item.data(Qt.ItemDataRole.UserRole)
            samples = item.data(Qt.ItemDataRole.UserRole + 1)
            new_data[name] = samples

        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, indent=4, ensure_ascii=False)
            self.identifier = None  # Force reload model
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # --- ACTIONS ---

    def add_profile(self):
        name = self.new_profile_input.text().strip()
        if not name: return

        # Check duplicate
        for i in range(self.profile_list.count()):
            if self.profile_list.item(i).data(Qt.ItemDataRole.UserRole) == name:
                QMessageBox.warning(self, "Error", "Profile already exists.")
                return

        item = QListWidgetItem(f"👤 {name}")
        item.setData(Qt.ItemDataRole.UserRole, name)
        item.setData(Qt.ItemDataRole.UserRole + 1, [])  # Empty list
        self.profile_list.addItem(item)
        self.save_data()
        self.new_profile_input.clear()

    def delete_profile(self):
        row = self.profile_list.currentRow()
        if row < 0: return
        name = self.profile_list.item(row).data(Qt.ItemDataRole.UserRole)

        if QMessageBox.question(self, "Delete", f"Delete profile '{name}'?") == QMessageBox.StandardButton.Yes:
            self.profile_list.takeItem(row)
            self.sample_list.clear()
            self.save_data()

    def add_sample(self):
        prof_row = self.profile_list.currentRow()
        if prof_row < 0:
            QMessageBox.warning(self, "Select Profile", "Please select a person on the left first.")
            return

        paths, _ = QFileDialog.getOpenFileNames(self, "Select Audio", "", "Audio (*.wav *.mp3 *.m4a)")
        if not paths: return

        item = self.profile_list.item(prof_row)
        current_samples = item.data(Qt.ItemDataRole.UserRole + 1)

        current_samples.extend(paths)
        item.setData(Qt.ItemDataRole.UserRole + 1, current_samples)

        self.save_data()
        self.on_profile_selected(item)  # Refresh view

    def remove_sample(self):
        prof_row = self.profile_list.currentRow()
        samp_row = self.sample_list.currentRow()

        if prof_row < 0 or samp_row < 0: return

        prof_item = self.profile_list.item(prof_row)
        samples = prof_item.data(Qt.ItemDataRole.UserRole + 1)

        # Remove at index
        del samples[samp_row]

        prof_item.setData(Qt.ItemDataRole.UserRole + 1, samples)
        self.save_data()
        self.on_profile_selected(prof_item)

    # --- TESTING (Updated to register full structure) ---
    def test_voice(self):
        if self.identifier is None:
            self.test_result_label.setText("Result: ⏳ Initializing Model...")
            self.repaint()
            try:
                self.identifier = SpeakerIdentifier(hf_token=self.hf_token)
                if os.path.exists(self.json_path):
                    with open(self.json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # register_speaker now handles lists automatically!
                        for n, content in data.items():
                            self.identifier.register_speaker(n, content)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                return

        path, _ = QFileDialog.getOpenFileName(self, "Select Clip", "", "Audio (*.wav *.mp3)")
        if not path: return

        self.btn_test.setEnabled(False)
        self.test_result_label.setText("Result: 🏃 Analyzing...")
        self.current_worker = VoiceTestWorker(self.identifier, path)
        self.current_worker.finished_result.connect(self.on_test_finished)
        self.current_worker.error.connect(self.on_test_error)
        self.current_worker.start()

    def on_test_finished(self, msg, style):
        self.test_result_label.setText(msg)
        self.test_result_label.setStyleSheet(style)
        self.btn_test.setEnabled(True)

    def on_test_error(self, err):
        self.test_result_label.setText("Error")
        QMessageBox.critical(self, "Error", err)
        self.btn_test.setEnabled(True)