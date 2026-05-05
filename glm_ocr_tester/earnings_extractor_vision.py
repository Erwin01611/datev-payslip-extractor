"""Vision-based extractor for the earnings block.

Uses a local vision model (Qwen 3.5 9B via mlx_vlm) to read the earnings crop
and extract line items (wage code, description, amount) and total gross.

Validated on:
- German payslips (DATEV format)
- English payslips
- Retroactive sections (Nachberechnung)

See prompts/earnings_v1.txt for the exact prompt template.
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from glm_ocr_tester.common_utils import parse_german_amount
from glm_ocr_tester.vision_client import run_vision_extraction

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "earnings_v1.txt"

# Markers that indicate an employer contribution
def _is_employer_contribution(description: str) -> bool:
    """Check if a line item is a pure employer cost (not part of Gesamt-Brutto).

    AG-Anteil VWL is excluded because it IS part of gross on most payslips.
    """
    desc_lower = description.lower()
    if "vwl" in desc_lower:
        return False
    markers = [
        "ag-leistung", "direktvers.", "betriebbl.av", "betriebl.av",
        "employer", "ag contribution", "company contribution",
        "ag-", "ag ", "ag_",
    ]
    return any(m in desc_lower for m in markers)


def _load_prompt() -> str:
    """Load the earnings prompt from the version-controlled prompt file."""
    path = _DEFAULT_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Earnings prompt not found at {path}. "
            "Run from project root or ensure prompts/earnings_v1.txt exists."
        )
    return path.read_text(encoding="utf-8")


def _strip_footnote(raw: str) -> str:
    """Remove footnote characters appended to an amount string."""
    if not raw:
        return raw
    raw = raw.split()[0]
    raw = re.sub(r"(-)[^\d,]+$", r"\1", raw)
    raw = re.sub(r"(\d)[^\d,\-]+$", r"\1", raw)
    return raw


def _parse_amount(val: Optional[str]) -> Tuple[Optional[float], str]:
    """Parse an amount string with footnote stripping and negative handling."""
    if not val:
        return None, ""
    cleaned = _strip_footnote(str(val))
    parsed, _ = parse_german_amount(cleaned)
    return parsed, cleaned


def extract_earnings(image_path: str, prompt: Optional[str] = None) -> dict:
    """Extract earnings line items from a cropped image.

    Args:
        image_path: Path to the cropped earnings image (PNG/JPG).
        prompt: Optional override prompt. If None, loads prompts/earnings_v1.txt.

    Returns:
        Dict with keys matching the Earnings model:
            - "earnings": {"line_items": [...], "total_gross": float}
            - "_vision_raw": the raw JSON dict from the vision model
            - "_error": str if extraction failed
    """
    prompt_text = prompt or _load_prompt()
    raw = run_vision_extraction(image_path, prompt_text)

    if "_error" in raw:
        logger.warning("Earnings vision extraction failed: %s", raw["_error"])
        return {"_error": raw["_error"], "_vision_raw": raw}

    result: Dict[str, any] = {"_vision_raw": raw}

    # Parse line items
    line_items: List[dict] = []
    for item in raw.get("line_items", []) or []:
        if not isinstance(item, dict):
            continue
        amount, amount_raw = _parse_amount(item.get("amount"))
        description = str(item.get("description", "")).strip()
        # Remove single-letter flags that might have been included
        description = re.sub(r'\b[LPFNJ]\b\s*$', '', description).strip()
        line_items.append({
            "wage_code": str(item.get("wage_code", "")),
            "description": description,
            "amount": amount,
            "raw_amount": amount_raw,
            "is_employer_contribution": _is_employer_contribution(description),
            "is_retroactive": False,
        })

    # Parse retroactive items
    retroactive_items: List[dict] = []
    for item in raw.get("retroactive_items", []) or []:
        if not isinstance(item, dict):
            continue
        amount, amount_raw = _parse_amount(item.get("amount"))
        description = str(item.get("description", "")).strip()
        description = re.sub(r'\b[LPFNJ]\b\s*$', '', description).strip()
        retroactive_items.append({
            "wage_code": str(item.get("wage_code", "")),
            "description": description,
            "amount": amount,
            "raw_amount": amount_raw,
            "is_employer_contribution": _is_employer_contribution(description),
            "is_retroactive": True,
        })

    all_items = line_items + retroactive_items

    # Parse total gross
    total_gross_raw = raw.get("total_gross")
    total_gross, _ = _parse_amount(total_gross_raw)

    result["earnings"] = {
        "line_items": all_items,
        "total_gross": total_gross,
    }
    return result


def extract_earnings_safe(image_path: str, prompt: Optional[str] = None) -> dict:
    """Same as extract_earnings but never raises; returns error dict on failure."""
    try:
        return extract_earnings(image_path, prompt)
    except Exception as exc:
        logger.exception("Unexpected error in earnings extraction")
        return {"_error": str(exc), "_vision_raw": {}}
