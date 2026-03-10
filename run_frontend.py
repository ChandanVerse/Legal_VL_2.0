"""Start the Vite dev server on http://localhost:5173 (proxies /chat to :8000)"""
import subprocess
from pathlib import Path

FRONTEND = Path(__file__).parent / "frontend"

try:
    subprocess.run(["npm", "run", "dev"], cwd=FRONTEND, shell=True)
except KeyboardInterrupt:
    pass
