"""Pure vision-model pipeline: no OpenRouter, no regex, no LLM APIs.

Takes pre-cropped payslip blocks and runs the local mlx_vlm vision model on each
crop to extract structured data. Maps each extractor's output into PayslipSchema
partial dicts, then returns them for merge_blocks().

Validated on:
- German DATEV-format payslips
- English payslips
- All 5 blocks: header, earnings, deductions, ytd_statement, bank_payout
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from glm_ocr_tester.header_extractor_vision import extract_header_safe
from glm_ocr_tester.earnings_extractor_vision import extract_earnings_safe
from glm_ocr_tester.deductions_extractor_vision import extract_deductions_safe
from glm_ocr_tester.net_pay_extractor import extract_net_pay_safe
from glm_ocr_tester.bank_payout_extractor_vision import extract_bank_payout_safe

logger = logging.getLogger(__name__)


def _map_header_vision(result: dict) -> dict:
    """Map header vision output to PayslipSchema partial dict."""
    if "_error" in result:
        logger.warning("Header vision error: %s", result["_error"])
        return {}

    partial = {}
    if "language_detected" in result:
        partial["language_detected"] = result["language_detected"]

    emp = result.get("employee", {})
    if emp.get("name") or emp.get("employee_number"):
        partial["employee"] = {
            "name": emp.get("name"),
            "employee_number": emp.get("employee_number"),
        }

    payroll = result.get("payroll_period", {})
    if payroll.get("month") or payroll.get("year"):
        partial["payroll_period"] = {
            "month_year": f"{payroll.get('month', '')} {payroll.get('year', '')}".strip() or None,
            "correction_number": payroll.get("correction_number"),
        }

    return partial


def _map_earnings_vision(result: dict) -> dict:
    """Map earnings vision output to PayslipSchema partial dict."""
    if "_error" in result:
        logger.warning("Earnings vision error: %s", result["_error"])
        return {}

    earnings = result.get("earnings", {})
    if not earnings:
        return {}

    # Build schema-compatible line items
    schema_items = []
    for item in earnings.get("line_items", []):
        schema_items.append({
            "wage_code": item.get("wage_code"),
            "description": item.get("description"),
            "amount": item.get("amount"),
            "ytd_amount": None,
        })

    return {
        "earnings": {
            "line_items": schema_items,
            "total_gross": earnings.get("total_gross"),
        }
    }


def _map_deductions_vision(result: dict) -> dict:
    """Map deductions vision output to PayslipSchema partial dict.

    Vision returns flat L/N fields. We use the L (current) row as primary amounts.
    """
    if "_error" in result:
        logger.warning("Deductions vision error: %s", result["_error"])
        return {}

    def _ln(field: str) -> dict:
        data = result.get(field, {})
        if isinstance(data, dict):
            return {"L": data.get("L"), "N": data.get("N")}
        return {"L": None, "N": None}

    wage_tax = _ln("wage_tax")

    def _sanitize_tax(val):
        if val is None:
            return None
        total_tax_l = _ln("total_tax_deductions").get("L")
        for ref in [v for v in [wage_tax.get("L"), total_tax_l] if v is not None]:
            if abs(val - ref) < 0.05:
                return None
        if wage_tax.get("L") is not None and val > wage_tax["L"] * 1.5:
            return None
        return val

    tax_details = {
        "lohnsteuer": wage_tax,
        "kirchensteuer": {"L": _sanitize_tax(_ln("church_tax").get("L")), "N": _ln("church_tax").get("N")},
        "solidaritaetszuschlag": {"L": _sanitize_tax(_ln("solidarity_surcharge").get("L")), "N": _ln("solidarity_surcharge").get("N")},
    }

    social_security = {
        "rentenversicherung": _ln("pi_contribution"),
        "arbeitslosenversicherung": _ln("ui_contribution"),
        "krankenversicherung": _ln("hi_contribution"),
        "pflegeversicherung": _ln("ci_contribution"),
    }

    line_items = []
    for field, label in [
        ("wage_tax", "Lohnsteuer / Wage tax"),
        ("church_tax", "Kirchensteuer / Church tax"),
        ("solidarity_surcharge", "Solidaritätszuschlag / Solidarity surcharge"),
        ("hi_contribution", "Krankenversicherung / Health insurance"),
        ("pi_contribution", "Rentenversicherung / Pension insurance"),
        ("ui_contribution", "Arbeitslosenversicherung / Unemployment insurance"),
        ("ci_contribution", "Pflegeversicherung / Care insurance"),
    ]:
        val = _ln(field).get("L")
        if val is not None:
            line_items.append({
                "description": label,
                "employee_amount": val,
                "employer_amount": None,
                "ytd_amount": None,
            })

    total_deductions = None
    tax_total = _ln("total_tax_deductions").get("L")
    ss_total = _ln("total_ss_deductions").get("L")
    if tax_total is not None and ss_total is not None:
        total_deductions = tax_total + ss_total
    elif tax_total is not None:
        total_deductions = tax_total
    elif ss_total is not None:
        total_deductions = ss_total

    return {
        "deductions": {
            "line_items": line_items,
            "total_deductions": total_deductions,
            "total_tax_deductions": _ln("total_tax_deductions"),
            "total_ss_deductions": _ln("total_ss_deductions"),
            "tax_details": tax_details,
            "social_security": social_security,
        }
    }


def _map_net_pay_vision(result: dict) -> dict:
    """Map net pay (YTD statement) vision output to PayslipSchema partial dict."""
    if "_error" in result:
        logger.warning("Net pay vision error: %s", result["_error"])
        return {}

    partial = {}
    total_net_ytd = result.get("total_net_ytd")
    if total_net_ytd is not None:
        partial["ytd_summary"] = {"total_net_ytd": total_net_ytd}

    adjustments = result.get("net_adjustments", [])
    if adjustments:
        schema_adjustments = []
        for adj in adjustments:
            schema_adjustments.append({
                "wage_code": adj.get("wage_code"),
                "description": adj.get("description"),
                "amount": adj.get("amount"),
            })
        ytd = partial.get("ytd_summary", {})
        ytd["net_adjustments"] = schema_adjustments
        partial["ytd_summary"] = ytd

    return partial


def _map_bank_payout_vision(result: dict) -> dict:
    """Map bank payout vision output to PayslipSchema partial dict."""
    if "_error" in result:
        logger.warning("Bank payout vision error: %s", result["_error"])
        return {}

    partial = {}
    net_pay = {}

    auszahlung = result.get("auszahlungsbetrag")
    if auszahlung is not None:
        net_pay["net_amount"] = auszahlung

    iban = result.get("iban")
    if iban:
        net_pay["bank_account_iban"] = iban

    bank_name = result.get("bank_name")
    if bank_name:
        net_pay["bank_name"] = bank_name
        net_pay["payment_method"] = "Bank"
    if net_pay:
        partial["net_pay"] = net_pay

    employer_costs = {}
    for key in ["sv_ag_anteil", "zus_ag_kosten", "gesamtkosten"]:
        val = result.get(key)
        if val is not None:
            employer_costs[key] = val
    if employer_costs:
        partial["employer_costs"] = employer_costs

    return partial


# Mapping from block name to (extractor_fn, mapper_fn)
_VISION_EXTRACTORS = {
    "header": (extract_header_safe, _map_header_vision),
    "earnings": (extract_earnings_safe, _map_earnings_vision),
    "deductions": (extract_deductions_safe, _map_deductions_vision),
    "ytd_statement": (extract_net_pay_safe, _map_net_pay_vision),
    "bank_payout": (extract_bank_payout_safe, _map_bank_payout_vision),
}


def extract_all_from_crops(
    blocks: List[Tuple],
    crops_dir: Path,
    stem: str,
) -> List[dict]:
    """Run vision extraction on all available crop images for a payslip.

    Args:
        blocks: List of (region, text) tuples from slice_payslip().
        crops_dir: Directory containing saved crop PNGs.
        stem: File stem used in crop filenames (e.g., "Payslip_German").

    Returns:
        List of partial dicts ready for merge_blocks().
    """
    block_results: List[dict] = []

    for region, text in blocks:
        crop_path = crops_dir / f"{stem}_{region.name}.png"
        if not crop_path.exists():
            logger.warning("Crop not found for %s: %s", region.name, crop_path)
            continue

        extractor_fn, mapper_fn = _VISION_EXTRACTORS.get(region.name)
        if extractor_fn is None:
            logger.warning("No vision extractor for block %s", region.name)
            continue

        try:
            raw_result = extractor_fn(str(crop_path))
            partial = mapper_fn(raw_result)
            if partial:
                partial["_block_name"] = region.name
                block_results.append(partial)
                logger.info(
                    "Vision %s extracted for %s",
                    region.name,
                    stem,
                )
            else:
                logger.warning("Vision %s returned empty for %s", region.name, stem)
        except Exception as exc:
            logger.warning("Vision %s extraction error for %s: %s", region.name, stem, exc)

    return block_results
