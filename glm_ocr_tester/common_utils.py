"""Shared utilities for amount parsing and string cleaning."""

import re
from typing import Optional, Tuple


def parse_german_amount(raw: str) -> Tuple[Optional[float], bool]:
    """Parse a payslip amount string using digit extraction.

    The vision model returns amounts in various formats:
    - Properly formatted: "697,50", "1.671,25", "0,65"
    - Without comma: "69750", "9750" (model dropped comma)
    - With footnote: "9750 1", "43,98 Z", "0,65-Z", "191-1"
    - Negative: "16,92-", "12,34-"
    - Standalone footnote: "1", "Z" (not an amount)

    Core rule: ALL payslip monetary amounts have exactly 2 decimal places.
    We extract digits only and divide by 100.

    Returns:
        (parsed_float, is_negative)
    """
    if not raw:
        return None, False

    raw = str(raw).strip()
    if not raw:
        return None, False

    # Standalone footnote / single digit: not an amount
    if re.fullmatch(r"[1-9A-Za-z]", raw):
        return None, False

    # Remove simple space-separated footnote:
    # "9750 1" -> "9750", "43,98 Z" -> "43,98"
    # Only strip if second token is a single footnote char
    parts = raw.split()
    if (
        len(parts) == 2
        and re.fullmatch(r"[\d.,-]+", parts[0])
        and re.fullmatch(r"[1-9A-Za-z]", parts[1])
    ):
        raw = parts[0]

    # Detect negative before removing noise
    is_negative = "-" in raw

    # Handle amount-minus-footnote pattern: "191-1" -> "191", negative
    if re.fullmatch(r"\s*\d+\s*-\s*[1-9A-Za-z]\s*", raw):
        main = re.split(r"-", raw)[0]
        digits = re.sub(r"\D", "", main)
    else:
        digits = re.sub(r"\D", "", raw)

    if len(digits) < 2:
        return None, False

    amount = int(digits) / 100
    return (-amount if is_negative else amount), is_negative


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences and <think> blocks from model output."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()
