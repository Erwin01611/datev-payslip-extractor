"""Vision-based extractor for the deductions block (tax + social security).

Uses a local vision model (Qwen 3.5 9B via mlx_vlm) to read the deductions crop
and extract all tax and social security amounts for both L (current) and N
(retroactive) rows.

The prompt returns German field names:
  lohnsteuer, kirchensteuer, solidaritaetszuschlag, steuerrechtliche_abzuege
  kv_beitrag, rv_beitrag, av_beitrag, pv_beitrag, sv_rechtliche_abzuege

These are mapped to the internal English schema names.

Validated on:
- German payslips (DATEV format)
- English payslips
- Handles dropped decimal commas, negative amounts, and footnote stripping

See prompts/deductions_v1.txt for the exact prompt template.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from glm_ocr_tester.common_utils import parse_german_amount
from glm_ocr_tester.vision_client import run_vision_extraction

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "deductions_v1.txt"

# Mapping from German field names (prompt output) to English schema names
_GERMAN_TO_ENGLISH = {
    "lohnsteuer": "wage_tax",
    "kirchensteuer": "church_tax",
    "solidaritaetszuschlag": "solidarity_surcharge",
    "steuerrechtliche_abzuege": "total_tax_deductions",
    "kv_beitrag": "hi_contribution",
    "rv_beitrag": "pi_contribution",
    "av_beitrag": "ui_contribution",
    "pv_beitrag": "ci_contribution",
    "sv_rechtliche_abzuege": "total_ss_deductions",
}

# Internal schema field names (used in result dict and validation)
_AMOUNT_FIELDS = list(_GERMAN_TO_ENGLISH.values())


def _load_prompt() -> str:
    """Load the deductions prompt from the version-controlled prompt file."""
    path = _DEFAULT_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Deductions prompt not found at {path}. "
            "Run from project root or ensure prompts/deductions_v1.txt exists."
        )
    return path.read_text(encoding="utf-8")


def _parse_ln_value(val: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    """Parse an L/N amount string using the unified parse_german_amount.

    parse_german_amount already handles footnote stripping internally.

    Returns:
        (parsed_float, cleaned_string)
    """
    if val is None:
        return None, None
    cleaned = str(val).strip()
    parsed, _ = parse_german_amount(cleaned)
    return parsed, cleaned


def extract_deductions(image_path: str, prompt: Optional[str] = None) -> dict:
    """Extract tax and social security deductions from a cropped image.

    Args:
        image_path: Path to the cropped deductions image (PNG/JPG).
        prompt: Optional override prompt. If None, loads prompts/deductions_v1.txt.

    Returns:
        Dict with keys:
            - Tax fields: wage_tax, church_tax, solidarity_surcharge, total_tax_deductions
            - SS fields: hi_contribution, pi_contribution, ui_contribution, ci_contribution, total_ss_deductions
            - Each field is a dict with "L" and "N" keys containing parsed floats
            - "_vision_raw": the raw JSON dict from the vision model
            - "_error": str if extraction failed
    """
    prompt_text = prompt or _load_prompt()
    raw = run_vision_extraction(image_path, prompt_text)

    if "_error" in raw:
        logger.warning("Deductions vision extraction failed: %s", raw["_error"])
        return {"_error": raw["_error"], "_vision_raw": raw}

    # Translate German field names to English schema names
    mapped_raw: Dict[str, any] = {}
    for german_key, english_key in _GERMAN_TO_ENGLISH.items():
        if german_key in raw:
            mapped_raw[english_key] = raw[german_key]

    result: Dict[str, any] = {"_vision_raw": mapped_raw, "validation_notes": []}

    for field in _AMOUNT_FIELDS:
        ln_data = mapped_raw.get(field, {})
        if not isinstance(ln_data, dict):
            ln_data = {}

        l_raw = ln_data.get("L")
        n_raw = ln_data.get("N")

        l_parsed, l_cleaned = _parse_ln_value(l_raw)
        n_parsed, n_cleaned = _parse_ln_value(n_raw)

        result[field] = {
            "L": l_parsed,
            "N": n_parsed,
            "L_raw": l_cleaned,
            "N_raw": n_cleaned,
        }

    return result


def extract_deductions_safe(image_path: str, prompt: Optional[str] = None) -> dict:
    """Same as extract_deductions but never raises; returns error dict on failure."""
    try:
        return extract_deductions(image_path, prompt)
    except Exception as exc:
        logger.exception("Unexpected error in deductions extraction")
        return {"_error": str(exc), "_vision_raw": {}}
