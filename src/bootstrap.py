import os
import sys
import time

def bootstrap_pyside():
    """
    Temporarily hides the Conda 'Library/bin' folder to allow PySide6 to
    load its own DLLs (zlib, SSL, etc.) without conflict.
    """
    # 1. Setup Paths
    conda_base = os.path.join(sys.prefix, "Library")
    bin_dir = os.path.join(conda_base, "bin")
    hidden_dir = os.path.join(conda_base, "bin_hidden_temp")

    renamed = False

    try:
        # 2. HIDE CONDA BIN
        if os.path.exists(bin_dir) and not os.path.exists(hidden_dir):
            try:
                os.rename(bin_dir, hidden_dir)
                renamed = True
                # print("[Boot] Conda bin hidden.")
            except OSError:
                print("[Boot] Warning: Could not rename Conda bin. DLL Hell imminent.")

        # 3. FORCE LOAD CRITICAL MODULES
        # We import the modules that trigger DLL loads.
        # Once loaded, they stay in RAM.
        import PySide6.QtCore
        import PySide6.QtGui
        import PySide6.QtWidgets
        import PySide6.QtMultimedia

        # print("[Boot] PySide6 loaded successfully.")

    except ImportError as e:
        print(f"[Boot] CRITICAL: PySide6 import failed: {e}")
        sys.exit(1)

    finally:
        # 4. RESTORE CONDA BIN (Crucial for ffmpeg/torchcodec)
        if renamed and os.path.exists(hidden_dir):
            for i in range(5):
                try:
                    os.rename(hidden_dir, bin_dir)
                    # print("[Boot] Conda bin restored.")
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                print(f"[Boot] ERROR: Could not restore {bin_dir}. Please rename manually!")


# Run immediately on import
bootstrap_pyside()