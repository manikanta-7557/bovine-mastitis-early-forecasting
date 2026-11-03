"""Build-if-needed launcher for the single-origin research application.

DEFENSE GUIDE
  Why launch through Python? It serves the compiled React interface and FastAPI API
    from one local process, so a review laptop does not need a separate web server.
  Alternatives rejected: duplicating model inference in the browser or requiring a
    separate development server for normal demonstrations.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
DIST = FRONTEND / "dist"


def ensure_frontend() -> None:
    if (DIST / "index.html").is_file():
        return
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("The frontend build is missing and npm is unavailable. Install Node.js, run npm install in frontend, then run npm run build.")
    if not (FRONTEND / "node_modules").is_dir():
        subprocess.run([npm, "install"], cwd=FRONTEND, check=True)
    subprocess.run([npm, "run", "build"], cwd=FRONTEND, check=True)
    if not (DIST / "index.html").is_file():
        raise RuntimeError("The frontend build completed without producing frontend/dist/index.html.")


def main() -> None:
    ensure_frontend()
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
