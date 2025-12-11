from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser,
    QLineEdit, QPushButton, QLabel, QMessageBox, QGroupBox, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal
import os

# Import Backend
from rag_engine import EvidenceRetriever, generate_rag_prompt
from analysis_engine import call_llm_openai, call_llm_gemini_25_pro


# --- WORKER 1: CHAT RESPONSE ---
class ChatWorker(QThread):
    finished = Signal(str)

    def __init__(self, query, retriever, provider="gemini", model_name="gemini-2.5-flash", role=""):
        super().__init__()
        self.query = query
        self.retriever = retriever
        self.provider = provider
        self.model_name = model_name
        self.role = role

    def run(self):
        try:
            relevant_chunks = self.retriever.search(self.query, top_k=15)

            if not relevant_chunks:
                self.finished.emit("<i>System: No relevant data found.</i>")
                return

            prompt = generate_rag_prompt(self.query, relevant_chunks, role_instruction=self.role)

            if "gemini" in self.provider.lower():
                response = call_llm_gemini_25_pro(self.model_name, prompt)
            else:
                response = call_llm_openai(self.model_name, prompt)

            self.finished.emit(response)

        except Exception as e:
            self.finished.emit(f"<b>Error:</b> {str(e)}")


# --- WORKER 2: INDEXING (New) ---
class IndexingWorker(QThread):
    """
    Runs the heavy embedding generation in the background
    so the GUI doesn't freeze when loading a 7-hour file.
    """
    finished = Signal(bool)  # Emits 'has_analysis' flag back to UI
    error = Signal(str)

    def __init__(self, retriever, text, has_analysis_flag):
        super().__init__()
        self.retriever = retriever
        self.text = text
        self.flag = has_analysis_flag

    def run(self):
        try:
            # This is the heavy blocking call
            self.retriever.ingest_transcript(self.text)
            self.finished.emit(self.flag)
        except Exception as e:
            self.error.emit(str(e))


class ChatTab(QWidget):
    def __init__(self):
        super().__init__()

        # State Variables
        self.retriever = None
        self.current_transcript_path = None
        self.chat_history = []
        self.indexing_worker = None  # Keep reference

        # Default Persona
        self.current_role = "You are a helpful assistant analyzing a transcript."

        # State to store selected model
        self.current_provider = "gemini"
        self.current_model_name = "gemini-2.5-flash"

        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- HEADER ---
        header_layout = QHBoxLayout()

        self.lbl_persona = QLabel(f"Context: General Assistant")
        self.lbl_persona.setStyleSheet("font-weight: bold; color: #555;")

        self.status_label = QLabel("Status: No transcript loaded.")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.btn_refresh = QPushButton("🔄 Re-index")
        self.btn_refresh.setToolTip("Click this if you have edited the transcript text manually.")

        header_layout.addWidget(self.lbl_persona)
        header_layout.addStretch()
        header_layout.addWidget(self.status_label)
        header_layout.addWidget(self.btn_refresh)

        layout.addLayout(header_layout)

        # 2. Chat Display
        self.chat_display = QTextBrowser()
        self.chat_display.setOpenExternalLinks(True)
        layout.addWidget(self.chat_display, 1)

        # 3. Input Area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask a question about the evidence...")
        self.input_field.returnPressed.connect(self.send_message)

        self.btn_send = QPushButton("Send")
        self.btn_send.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_field, 1)
        input_layout.addWidget(self.btn_send)
        layout.addLayout(input_layout)

    def set_persona(self, instruction_text, is_generic=False):
        if not instruction_text: return
        self.current_role = instruction_text

        display_text = (instruction_text[:50] + '..') if len(instruction_text) > 50 else instruction_text

        if is_generic:
            self.lbl_persona.setText(f"⚠️ Context: {display_text} (Unanalyzed)")
            self.lbl_persona.setStyleSheet("font-weight: bold; color: #d35400; background-color: #fdebd0; padding: 4px; border-radius: 4px;")
            self.lbl_persona.setToolTip("This transcript has not been analyzed. Using generic persona.")
        else:
            self.lbl_persona.setText(f"🧠 Context: {display_text}")
            self.lbl_persona.setStyleSheet("font-weight: bold; color: #27ae60; background-color: #eafaf1; padding: 4px; border-radius: 4px;")
            self.lbl_persona.setToolTip(instruction_text)

    def set_model_config(self, provider_label):
        provider_label = provider_label.lower()
        if "gemini" in provider_label:
            self.current_provider = "gemini"
            self.current_model_name = provider_label
        elif "gpt" in provider_label:
            self.current_provider = "openai"
            self.current_model_name = provider_label
        else:
            self.current_provider = "openai"
            self.current_model_name = provider_label

    def load_transcript(self, transcript_text, path="Unknown", has_analysis=False):
        """
        Starts background indexing.
        """
        self.current_transcript_path = path
        self.status_label.setText(f"Status: ⏳ Indexing '{os.path.basename(path)}'...")
        self.chat_display.clear()
        self.chat_display.append(f"<b>System:</b> Reading and indexing {os.path.basename(path)}... (Please wait)")

        # Disable input while indexing
        self.input_field.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.btn_refresh.setEnabled(False)

        # Initialize Engine (Lazy)
        if self.retriever is None:
            self.retriever = EvidenceRetriever()

        # START WORKER
        self.indexing_worker = IndexingWorker(self.retriever, transcript_text, has_analysis)
        self.indexing_worker.finished.connect(self.on_indexing_finished)
        self.indexing_worker.error.connect(self.on_indexing_error)
        self.indexing_worker.start()

    def on_indexing_finished(self, has_analysis):
        self.status_label.setText("Status: ✅ Ready.")

        # Re-enable input
        self.input_field.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.btn_refresh.setEnabled(True)

        if has_analysis:
            self.chat_display.append("<b>System:</b> Indexing complete. Context loaded from analysis. Ask away!")
        else:
            warning_html = (
                "<div style='background-color: #fff3cd; color: #856404; padding: 10px; border: 1px solid #ffeeba;'>"
                "<b>⚠️ Notice: Raw Transcript Loaded</b><br>"
                "This file has not been analyzed by the AI yet.<br>"
                "<i>The Chat Assistant is running in <b>Generic Mode</b>. For better accuracy and a specific persona, "
                "please run 'Analyze transcript only'.</i>"
                "</div><br>"
            )
            self.chat_display.append(warning_html)
            # We don't set persona here because Main Window usually calls set_persona immediately after calling load_transcript

    def on_indexing_error(self, err_msg):
        self.status_label.setText("Status: ❌ Error.")
        self.chat_display.append(f"<b style='color:red'>System Error: Failed to index transcript. {err_msg}</b>")
        self.btn_refresh.setEnabled(True)  # Allow retry

    def send_message(self):
        text = self.input_field.text().strip()
        if not text: return

        if not self.retriever or not self.retriever.corpus_embeddings is not None:
            QMessageBox.warning(self, "No Context", "Please load/transcribe a file first.")
            return

        self.chat_display.append(f"<p style='color: #2c3e50'><b>You:</b> {text}</p>")
        self.input_field.clear()
        self.btn_send.setEnabled(False)
        self.status_label.setText("Status: Thinking...")

        self.worker = ChatWorker(
            text,
            self.retriever,
            provider=self.current_provider,
            model_name=self.current_model_name,
            role=self.current_role
        )
        self.worker.finished.connect(self.on_response)
        self.worker.start()

    def on_response(self, response):
        formatted = response.replace("\n", "<br>")
        self.chat_display.append(f"<p style='color: #27ae60'><b>Analyst:</b> {formatted}</p><hr>")
        self.btn_send.setEnabled(True)
        self.status_label.setText(f"Status: Ready.")