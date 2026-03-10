"""Start the FastAPI backend on http://localhost:8000"""
import subprocess
from pathlib import Path

ROOT   = Path(__file__).parent
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"

try:
    subprocess.run(
        [str(PYTHON), "-m", "uvicorn", "api:app", "--reload", "--port", "8000"],
        cwd=ROOT,
    )
except KeyboardInterrupt:
    pass
