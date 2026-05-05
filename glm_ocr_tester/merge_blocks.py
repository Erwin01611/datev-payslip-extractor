"""Merge per-block partial dicts into a single PayslipSchema."""

from typing import Any, Dict, List

from glm_ocr_tester.models import PayslipSchema


def _deep_merge(base: dict, override: dict, path: str = "") -> dict:
    """Recursively merge override into base. override values win."""
    for key, val in override.items():
        if isinstance(val, dict) and key in base and isinstance(base[key], dict):
            _deep_merge(base[key], val, f"{path}.{key}")
        else:
            base[key] = val
    return base


# Precedence rules: when multiple blocks return the same field,
# which block's value should win?
_BLOCK_PRECEDENCE = {
    "net_pay.net_amount": ["bank_payout", "ytd_statement"],
    "net_pay.bank_account_iban": ["bank_payout"],

    "net_pay.payment_method": ["bank_payout"],
    "ytd_summary": ["ytd_statement"],
    "employer": ["header"],
    "employee": ["header"],
    "payroll_period": ["header"],
    "language_detected": ["header"],
    "earnings": ["earnings"],
    "deductions": ["deductions"],
}


def merge_blocks(block_results: List[Dict[str, Any]]) -> PayslipSchema:
    """Merge per-block partial dicts into a single PayslipSchema.

    Args:
        block_results: List of dicts, each with a subset of PayslipSchema fields.

    Returns:
        A fully populated PayslipSchema.
    """
    merged: dict = {
        "document_type": "payslip",
        "language_detected": None,
        "employee": {"name": None, "employee_number": None},
        "payroll_period": {"month_year": None, "correction_number": None},
        "earnings": {"line_items": [], "total_gross": None},
        "deductions": {"line_items": [], "total_deductions": None, "total_tax_deductions": {"L": None, "N": None}, "total_ss_deductions": {"L": None, "N": None}, "tax_details": {"lohnsteuer": {"L": None, "N": None}, "kirchensteuer": {"L": None, "N": None}, "solidaritaetszuschlag": {"L": None, "N": None}}, "social_security": {"rentenversicherung": {"L": None, "N": None}, "arbeitslosenversicherung": {"L": None, "N": None}, "krankenversicherung": {"L": None, "N": None}, "pflegeversicherung": {"L": None, "N": None}}},
        "net_pay": {"net_amount": None, "payment_method": None, "bank_account_iban": None, "bank_name": None},
        "employer_costs": {"sv_ag_anteil": None, "zus_ag_kosten": None, "gesamtkosten": None},
        "ytd_summary": {"total_net_ytd": None, "net_adjustments": []},
        "validation_notes": {"missing_fields": [], "ambiguous_fields": [], "arithmetic_warnings": []},
    }

    contributors: Dict[str, str] = {}

    for result in block_results:
        for key, val in result.items():
            if key == "document_type":
                continue
            if key == "validation_notes":
                if isinstance(val, dict):
                    for arr_key in ["missing_fields", "ambiguous_fields", "arithmetic_warnings"]:
                        if arr_key in val and isinstance(val[arr_key], list):
                            merged["validation_notes"][arr_key].extend(val[arr_key])
                continue

            precedence = _BLOCK_PRECEDENCE.get(key, [])
            current_block = contributors.get(key)
            new_block = result.get("_block_name", "unknown")

            if current_block is None:
                merged[key] = val
                contributors[key] = new_block
            elif precedence:
                try:
                    current_rank = precedence.index(current_block)
                except ValueError:
                    current_rank = 999
                try:
                    new_rank = precedence.index(new_block)
                except ValueError:
                    new_rank = 999

                if new_rank < current_rank:
                    merged[key] = val
                    contributors[key] = new_block
                elif new_rank == current_rank:
                    if isinstance(val, dict) and isinstance(merged.get(key), dict):
                        _deep_merge(merged[key], val)
                    else:
                        merged[key] = val
            else:
                if isinstance(val, dict) and isinstance(merged.get(key), dict):
                    _deep_merge(merged[key], val)
                else:
                    merged[key] = val

    return PayslipSchema(**merged)
