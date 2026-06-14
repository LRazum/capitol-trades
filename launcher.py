"""
Capitol Trades Dashboard — launcher
====================================
Press ▶  in VS Code (or F5) to start the app.
Opens automatically at http://localhost:8501
"""

import os
import subprocess
import sys


def main() -> None:
    app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

    if not os.path.exists(app):
        print(f"[launcher] app.py not found at: {app}")
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run", app,
        "--server.port=8501",
        "--browser.gatherUsageStats=false",
    ]

    print(f"[launcher] Starting: {' '.join(cmd)}")
    print("[launcher] Dashboard → http://localhost:8501  (Ctrl+C to stop)")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[launcher] Stopped.")
    except subprocess.CalledProcessError as e:
        print(f"[launcher] Streamlit exited with code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()