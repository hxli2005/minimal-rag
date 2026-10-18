"""Runtime assets belong to the working directory, never to site-packages."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("MINIMAL_RAG_HOME", Path.cwd())).expanduser().resolve()
PACKAGE_DIR = Path(__file__).resolve().parent
