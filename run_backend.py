"""
run_backend.py – PROPCAST development server launcher.
Run from the propcast/ directory:
    python run_backend.py
"""
import subprocess
import sys

if __name__ == "__main__":
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    print("Starting PROPCAST API server at http://0.0.0.0:8000")
    print("Interactive docs: http://localhost:8000/docs")
    print("Press CTRL+C to stop.\n")
    subprocess.run(cmd)
