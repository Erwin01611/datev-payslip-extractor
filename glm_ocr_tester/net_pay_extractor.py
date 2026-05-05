"""Vision-based extractor for the net pay / net adjustments block.

Uses a local vision model (Qwen 3.5 9B via mlx_vlm) to read the right half of
the YTD statement crop (Netto-Bezüge/Netto-Abzüge + Netto-Verdienst).

Validated on:
- German payslips (DATEV format)
- English payslips
- Empty tables
- Single and multiple rows
- Multi-line descriptions

See prompts/net_pay_v2.txt for the exact prompt template.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from glm_ocr_tester.common_utils import parse_german_amount
from glm_ocr_tester.vision_client import run_vision_extraction

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "net_pay_v2.txt"


def _load_prompt() -> str:
    """Load the net pay prompt from the version-controlled prompt file."""
    path = _DEFAULT_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Net pay prompt not found at {path}. "
            "Run from project root or ensure prompts/net_pay_v2.txt exists."
        )
    return path.read_text(encoding="utf-8")


def extract_net_pay(image_path: str, prompt: Optional[str] = None) -> dict:
    """Extract net income and net adjustments from a cropped image.

    Args:
        image_path: Path to the cropped YTD/net pay image (PNG/JPG).
        prompt: Optional override prompt. If None, loads prompts/net_pay_v2.txt.

    Returns:
        Dict with keys:
            - "net_income": float or None
            - "net_adjustments": list of dicts with "wage_code", "description", "amount"
            - "_vision_raw": the raw JSON dict from the vision model (for debugging)
            - "_error": str if extraction failed
    """
    prompt_text = prompt or _load_prompt()
    raw = run_vision_extraction(image_path, prompt_text)

    if "_error" in raw:
        logger.warning("Net pay vision extraction failed: %s", raw["_error"])
        return {"_error": raw["_error"], "_vision_raw": raw}

    result: Dict[str, any] = {"_vision_raw": raw}

    # Parse net_income — try multiple possible keys from the model
    net_income_raw = None
    for key in ["net_income", "Netto-Verdienst", "netto_verdienst", "netto", "Netto-Verdienst"]:
        if key in raw and raw[key] is not None:
            net_income_raw = raw[key]
            break
    if net_income_raw:
        val, _ = parse_german_amount(str(net_income_raw))
        result["total_net_ytd"] = val
        if val is None:
            logger.warning("Net income raw value could not be parsed: %r", net_income_raw)
    else:
        result["total_net_ytd"] = None
        logger.warning("Net income key missing from vision output; raw keys: %s", list(raw.keys()))

    # Parse net_adjustments
    adjustments_raw = raw.get("net_adjustments", [])
    adjustments: List[dict] = []
    if isinstance(adjustments_raw, list):
        for item in adjustments_raw:
            if not isinstance(item, dict):
                continue
            amount_raw = item.get("amount")
            amount_val = None
            if amount_raw:
                parsed, _ = parse_german_amount(str(amount_raw))
                amount_val = parsed
            adjustments.append({
                "wage_code": str(item.get("wage_code", "")),
                "description": str(item.get("description", "")),
                "amount": amount_val,
                "amount_raw": str(amount_raw) if amount_raw else None,
            })
    result["net_adjustments"] = adjustments
    result["net_adjustment_count"] = len(adjustments)

    # Compute total net adjustments for validation
    total_adjustments = sum(
        adj["amount"] for adj in adjustments if adj["amount"] is not None
    )
    result["net_adjustments_total"] = total_adjustments

    return result


def extract_net_pay_safe(image_path: str, prompt: Optional[str] = None) -> dict:
    """Same as extract_net_pay but never raises; returns error dict on failure."""
    try:
        return extract_net_pay(image_path, prompt)
    except Exception as exc:
        logger.exception("Unexpected error in net pay extraction")
        return {"_error": str(exc), "total_net_ytd": None, "net_adjustments": []}
