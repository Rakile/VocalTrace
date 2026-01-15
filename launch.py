import sys
import os
import logging
import torch
# Ensure the 'src' directory is in the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
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

from main import MainWindow, QApplication

if __name__ == "__main__":
    # Ensure working directory is Project Root (so voices/ folder is found)
    os.chdir(current_dir)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())