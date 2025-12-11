import sys
import os

# Ensure the 'src' directory is in the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
sys.path.insert(0, src_dir)

# Now we can import the main app from src
# This automatically triggers bootstrap.py inside src/main.py
from main import MainWindow, QApplication

if __name__ == "__main__":
    # Ensure working directory is Project Root (so voices/ folder is found)
    os.chdir(current_dir)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())