"""Scan the project root for supported input files."""

from pathlib import Path
from typing import List, Optional

from glm_ocr_tester.config import SETTINGS

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
IGNORED_DIRS = {
    "outputs",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "glm_ocr_tester",
    ".kimi",
    ".pytest_cache",
    "node_modules",
}


def scan(root: Optional[str] = None) -> List[Path]:
    """Return a sorted list of supported file paths found directly in the project root.

    We intentionally do NOT recurse into subdirectories so that code folders and
    the outputs directory are ignored automatically.
    """
    root_path = Path(root or SETTINGS.project_root).resolve()
    files = [
        p
        for p in root_path.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
        and p.name.startswith(".") is False
    ]
    return sorted(files)
