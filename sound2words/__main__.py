import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1200")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

try:
    from sound2words.ui.main_window import run
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from sound2words.ui.main_window import run


if __name__ == "__main__":
    run()
