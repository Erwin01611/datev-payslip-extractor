"""Post-extraction validation checks. Returns a list of warning dicts.

All checks run AFTER the full payslip is extracted and merged.
No per-block validation — this is the single source of truth for correctness.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

TOLERANCE = 0.05


def _v(data: dict, key: str, suffix: str = "L") -> Optional[float]:
    """Get a value from a nested L/N dict."""
    node = data.get(key) if isinstance(data, dict) else None
    return node.get(suffix) if isinstance(node, dict) else None


def _warn(base: dict, check_name: str, expected, actual, message: str) -> dict:
    """Build a standardized warning dict."""
    return {
        **base,
        "check_name": check_name,
        "expected": expected,
        "actual": actual,
        "difference": round(actual - expected, 2) if expected is not None else None,
        "message": message,
    }


def validate_all(payslips: List[dict]) -> List[dict]:
    """Run all validation checks on extracted payslips."""
    warnings: List[dict] = []
    for p in payslips:
        warnings.extend(_validate_one(p))
    return warnings


def _validate_one(p: dict) -> List[dict]:
    warnings: List[dict] = []
    emp = p.get("employee", {}) or {}
    payroll = p.get("payroll_period", {}) or {}
    earnings = p.get("earnings", {}) or {}
    deductions = p.get("deductions", {}) or {}
    net_pay = p.get("net_pay", {}) or {}
    ytd = p.get("ytd_summary", {}) or {}

    base = {
        "employee_name": emp.get("name", ""),
        "employee_number": emp.get("employee_number", ""),
        "month": payroll.get("month_year", ""),
    }

    # ---- 1. Earnings sum vs Gesamt-Brutto ----
    total_gross = earnings.get("total_gross")
    item_sum = sum((item.get("amount") or 0) for item in earnings.get("line_items", []))
    if total_gross is not None and abs(item_sum - total_gross) > TOLERANCE:
        warnings.append(_warn(
            base, "earnings_gross_mismatch", total_gross, item_sum,
            f"Sum of earnings ({item_sum:.2f}) != Gesamt-Brutto ({total_gross:.2f})"
        ))

    # ---- 2. Tax items vs Steuer_abzuege (L and N) ----
    tax = deductions.get("tax_details", {}) or {}
    tax_total = deductions.get("total_tax_deductions", {}) or {}
    for suffix in ["L", "N"]:
        tax_sum = sum(filter(None, [
            _v(tax, "lohnsteuer", suffix),
            _v(tax, "kirchensteuer", suffix),
            _v(tax, "solidaritaetszuschlag", suffix),
        ]))
        total = tax_total.get(suffix) if isinstance(tax_total, dict) else None
        if total is not None and abs(tax_sum - total) > TOLERANCE:
            warnings.append(_warn(
                base, f"tax_total_mismatch_{suffix}", total, tax_sum,
                f"Tax items sum ({tax_sum:.2f}) != Steuer_abzuege_{suffix} ({total:.2f})"
            ))

    # ---- 3. SS items vs SV_abzuege (L and N) ----
    ss = deductions.get("social_security", {}) or {}
    ss_total = deductions.get("total_ss_deductions", {}) or {}
    for suffix in ["L", "N"]:
        ss_sum = sum(filter(None, [
            _v(ss, "krankenversicherung", suffix),
            _v(ss, "rentenversicherung", suffix),
            _v(ss, "arbeitslosenversicherung", suffix),
            _v(ss, "pflegeversicherung", suffix),
        ]))
        total = ss_total.get(suffix) if isinstance(ss_total, dict) else None
        if total is not None and abs(ss_sum - total) > TOLERANCE:
            warnings.append(_warn(
                base, f"ss_total_mismatch_{suffix}", total, ss_sum,
                f"SS items sum ({ss_sum:.2f}) != SV_abzuege_{suffix} ({total:.2f})"
            ))

    # ---- 4. Net income equation ----
    net_income = ytd.get("total_net_ytd")
    tax_l = _v(deductions, "total_tax_deductions", "L") or 0
    tax_n = _v(deductions, "total_tax_deductions", "N") or 0
    ss_l = _v(deductions, "total_ss_deductions", "L") or 0
    ss_n = _v(deductions, "total_ss_deductions", "N") or 0

    if total_gross is not None and net_income is not None:
        computed_net = total_gross - (tax_l + tax_n) - (ss_l + ss_n)
        if abs(computed_net - net_income) > TOLERANCE:
            warnings.append(_warn(
                base, "net_income_mismatch", net_income, computed_net,
                f"Netto mismatch: {total_gross:.2f} - ({tax_l:.2f}+{tax_n:.2f}) - ({ss_l:.2f}+{ss_n:.2f}) = {computed_net:.2f} != {net_income:.2f}"
            ))

    # ---- 5. Payout equation ----
    auszahlung = net_pay.get("net_amount")
    adj_sum = sum((adj.get("amount") or 0) for adj in ytd.get("net_adjustments", []))
    if net_income is not None and auszahlung is not None:
        computed_payout = net_income + adj_sum
        if abs(computed_payout - auszahlung) > TOLERANCE:
            warnings.append(_warn(
                base, "payout_mismatch", auszahlung, computed_payout,
                f"Payout mismatch: {net_income:.2f} + {adj_sum:.2f} = {computed_payout:.2f} != {auszahlung:.2f}"
            ))

    return warnings


def validate_payslip(payslip: dict) -> List[dict]:
    """Validate a single payslip (wrapper for runner.py compatibility)."""
    return _validate_one(payslip)


def write_warnings_csv(warnings: List[dict], out_path: Path) -> None:
    """Write warnings to a CSV file."""
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "employee_name", "employee_number", "month",
        "check_name", "expected", "actual", "difference", "message",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for w in warnings:
            writer.writerow(w)
