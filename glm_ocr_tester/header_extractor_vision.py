"""Vision-based extractor for the header block.

Uses a local vision model (Qwen 3.5 9B via mlx_vlm) to read the header crop
and extract employee name, employee number, month, year, and correction number.

See prompts/header_v1.txt for the exact prompt template.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional

from glm_ocr_tester.vision_client import run_vision_extraction

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "header_v1.txt"


def _load_prompt() -> str:
    """Load the header prompt from the version-controlled prompt file."""
    path = _DEFAULT_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Header prompt not found at {path}. "
            "Run from project root or ensure prompts/header_v1.txt exists."
        )
    return path.read_text(encoding="utf-8")


def extract_header(image_path: str, prompt: Optional[str] = None) -> dict:
    """Extract header information from a cropped image.

    Args:
        image_path: Path to the cropped header image (PNG/JPG).
        prompt: Optional override prompt. If None, loads prompts/header_v1.txt.

    Returns:
        Dict with keys:
            - "employee": {"name": ..., "employee_number": ...}
            - "payroll_period": {"month": ..., "year": ..., "correction_number": ...}
            - "_vision_raw": the raw JSON dict from the vision model
            - "_error": str if extraction failed
    """
    prompt_text = prompt or _load_prompt()
    raw = run_vision_extraction(image_path, prompt_text)

    if "_error" in raw:
        logger.warning("Header vision extraction failed: %s", raw["_error"])
        return {"_error": raw["_error"], "_vision_raw": raw}

    result: Dict[str, any] = {"_vision_raw": raw}

    employee = raw.get("employee", {}) or {}
    payroll = raw.get("payroll_period", {}) or {}

    result["employee"] = {
        "name": employee.get("name") or None,
        "employee_number": employee.get("employee_number") or None,
    }

    result["payroll_period"] = {
        "month": payroll.get("month") or None,
        "year": payroll.get("year") or None,
        "correction_number": payroll.get("correction_number") or None,
    }

    # Language detection from month
    month = payroll.get("month", "")
    if month:
        my_lower = month.lower()
        german_months = ["januar", "februar", "märz", "marz", "april", "mai", "juni",
                         "juli", "august", "september", "oktober", "november", "dezember"]
        english_months = ["january", "february", "march", "april", "may", "june",
                          "july", "august", "september", "october", "november", "december"]
        if any(re.search(rf"\b{gm}\b", my_lower) for gm in german_months):
            result["language_detected"] = "de"
        elif any(re.search(rf"\b{em}\b", my_lower) for em in english_months):
            result["language_detected"] = "en"
        else:
            result["language_detected"] = "en"

    return result


def extract_header_safe(image_path: str, prompt: Optional[str] = None) -> dict:
    """Same as extract_header but never raises; returns error dict on failure."""
    try:
        return extract_header(image_path, prompt)
    except Exception as exc:
        logger.exception("Unexpected error in header extraction")
        return {"_error": str(exc), "_vision_raw": {}}
