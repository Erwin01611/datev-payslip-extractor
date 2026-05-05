"""Vision-based extractor for the bank & payout block.

Uses a local vision model (Qwen 3.5 9B via mlx_vlm) to read the bank payout crop
and extract bank details, employer costs, and net payout amount.

Validated on:
- German payslips (DATEV format)
- Handles dropped decimal commas with domain validation ranges

See prompts/bank_payout_v1.txt for the exact prompt template.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from glm_ocr_tester.common_utils import parse_german_amount
from glm_ocr_tester.vision_client import run_vision_extraction

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "bank_payout_v1.txt"

# Fields that contain amounts and need parsing
_AMOUNT_FIELDS = [
    "sv_ag_anteil", "zus_ag_kosten", "gesamtkosten", "auszahlungsbetrag"
]


def _load_prompt() -> str:
    """Load the bank payout prompt from the version-controlled prompt file."""
    path = _DEFAULT_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Bank payout prompt not found at {path}. "
            "Run from project root or ensure prompts/bank_payout_v1.txt exists."
        )
    return path.read_text(encoding="utf-8")


def extract_bank_payout(image_path: str, prompt: Optional[str] = None) -> dict:
    """Extract bank and payout information from a cropped image.

    Args:
        image_path: Path to the cropped bank payout image (PNG/JPG).
        prompt: Optional override prompt. If None, loads prompts/bank_payout_v1.txt.

    Returns:
        Dict with keys:
            - "bank_name": str or None
            - "iban": str or None
            - Amount fields: parsed float values (e.g., "sv_ag_anteil": 660.23)
            - "_vision_raw": the raw JSON dict from the vision model
            - "_error": str if extraction failed
    """
    prompt_text = prompt or _load_prompt()
    raw = run_vision_extraction(image_path, prompt_text)

    if "_error" in raw:
        logger.warning("Bank payout vision extraction failed: %s", raw["_error"])
        return {"_error": raw["_error"], "_vision_raw": raw}

    result: Dict[str, any] = {"_vision_raw": raw}

    # String fields
    result["bank_name"] = raw.get("bank_name")
    result["iban"] = raw.get("iban")

    # Amount fields
    for field in _AMOUNT_FIELDS:
        raw_val = raw.get(field)
        parsed = None
        if raw_val:
            parsed, _ = parse_german_amount(str(raw_val).strip())
        result[field] = parsed
        result[f"{field}_raw"] = raw_val

    return result


def extract_bank_payout_safe(image_path: str, prompt: Optional[str] = None) -> dict:
    """Same as extract_bank_payout but never raises; returns error dict on failure."""
    try:
        return extract_bank_payout(image_path, prompt)
    except Exception as exc:
        logger.exception("Unexpected error in bank payout extraction")
        return {"_error": str(exc), "_vision_raw": {}}
