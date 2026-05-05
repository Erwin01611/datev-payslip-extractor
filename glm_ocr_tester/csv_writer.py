"""Convert PayslipSchema objects to a wide pivot CSV + glossary."""

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(data: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key, default)
        if data is None:
            return default
    return data


def _fmt(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def _safe_col_name(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name).strip("_")


def _wage_col_key(code: str) -> str:
    code = (code or "").strip()
    return code or "unknown"


def _ytd_col_key(code: str, description: str) -> str:
    """Build YTD column key: Y_{code} if present, else Y_{sanitized_description}."""
    code = (code or "").strip()
    if code:
        return f"Y_{code}"
    desc = _safe_col_name(description or "").strip("_")
    return f"Y_{desc}" if desc else "Y_unknown"


# ---------------------------------------------------------------------------
# Static deduction field definitions: (json_path, csv_prefix)
# ---------------------------------------------------------------------------
_DEDUCTION_LN_FIELDS = [
    ("tax_details.lohnsteuer", "lohnsteuer"),
    ("tax_details.kirchensteuer", "kirchensteuer"),
    ("tax_details.solidaritaetszuschlag", "solidaritaetszuschlag"),
    ("social_security.krankenversicherung", "kv"),
    ("social_security.rentenversicherung", "rv"),
    ("social_security.arbeitslosenversicherung", "av"),
    ("social_security.pflegeversicherung", "pv"),
    ("total_tax_deductions", "steuer_abzuege"),
    ("total_ss_deductions", "sv_abzuege"),
]


def _build_deduction_columns() -> List[str]:
    cols = []
    for _, prefix in _DEDUCTION_LN_FIELDS:
        cols.append(f"{prefix}_L")
        cols.append(f"{prefix}_N")
    return cols


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def payslip_to_row(payslip: dict, wage_columns: List[str]) -> dict:
    """Convert a single PayslipSchema dict to a flat CSV row dict."""
    emp = payslip.get("employee", {}) or {}
    payroll = payslip.get("payroll_period", {}) or {}
    earnings = payslip.get("earnings", {}) or {}
    deductions = payslip.get("deductions", {}) or {}
    net_pay = payslip.get("net_pay", {}) or {}
    employer_costs = payslip.get("employer_costs", {}) or {}

    month_year = _get(payroll, "month_year", default="")
    month, year = "", ""
    if month_year and " " in month_year:
        parts = month_year.rsplit(" ", 1)
        month = parts[0]
        year = parts[1]

    row: Dict[str, str] = {
        "employee_name": _get(emp, "name", default=""),
        "employee_number": _get(emp, "employee_number", default=""),
        "month": month,
        "year": year,
        "correction_number": _get(payroll, "correction_number", default=""),
        "language": payslip.get("language_detected", ""),
        "total_gross": _fmt(_get(earnings, "total_gross")),
        "net_amount": _fmt(_get(net_pay, "net_amount")),
        "bank_name": _get(net_pay, "bank_name", default=""),
        "iban": _get(net_pay, "bank_account_iban", default=""),
        "sv_ag_anteil": _fmt(_get(employer_costs, "sv_ag_anteil")),
        "zus_ag_kosten": _fmt(_get(employer_costs, "zus_ag_kosten")),
        "gesamtkosten": _fmt(_get(employer_costs, "gesamtkosten")),
        "auszahlungsbetrag": _fmt(_get(net_pay, "net_amount")),
    }

    # Deductions L / N
    for json_path, prefix in _DEDUCTION_LN_FIELDS:
        keys = json_path.split(".")
        data = deductions
        for key in keys:
            data = data.get(key, {}) if isinstance(data, dict) else {}
        l_val = data.get("L") if isinstance(data, dict) else None
        n_val = data.get("N") if isinstance(data, dict) else None
        row[f"{prefix}_L"] = _fmt(l_val)
        row[f"{prefix}_N"] = _fmt(n_val)

    # Wage type pivot: init all wage columns to empty
    for col in wage_columns:
        row[col] = ""

    # Fill in earnings wage amounts
    for item in earnings.get("line_items", []):
        col_key = _wage_col_key(item.get("wage_code", ""))
        if col_key in row:
            row[col_key] = _fmt(item.get("amount"))

    # Fill in ytd adjustment amounts (prefixed with Y_)
    ytd = payslip.get("ytd_summary", {}) or {}
    for adj in ytd.get("net_adjustments", []):
        col_key = _ytd_col_key(adj.get("wage_code", ""), adj.get("description", ""))
        if col_key in row:
            row[col_key] = _fmt(adj.get("amount"))

    return row


# ---------------------------------------------------------------------------
# Glossary builder
# ---------------------------------------------------------------------------

def build_glossary(payslips: List[dict]) -> List[Dict[str, str]]:
    """Build a glossary of unique wage codes → descriptions (deduplicated by code)."""
    seen: Set[str] = set()
    code_to_desc: Dict[str, str] = {}
    for p in payslips:
        # Earnings wage types
        earnings = p.get("earnings", {}) or {}
        for item in earnings.get("line_items", []):
            code = str(item.get("wage_code", "")).strip()
            desc = str(item.get("description", "")).strip()
            if not code:
                continue
            if code not in seen:
                seen.add(code)
                code_to_desc[code] = desc
            else:
                # Keep the longer/more complete description
                if len(desc) > len(code_to_desc.get(code, "")):
                    code_to_desc[code] = desc
        # YTD adjustments
        ytd = p.get("ytd_summary", {}) or {}
        for adj in ytd.get("net_adjustments", []):
            code = str(adj.get("wage_code", "")).strip()
            desc = str(adj.get("description", "")).strip()
            key = code if code else _safe_col_name(desc)
            if not key:
                continue
            if key not in seen:
                seen.add(key)
                code_to_desc[key] = desc
            else:
                if len(desc) > len(code_to_desc.get(key, "")):
                    code_to_desc[key] = desc
    glossary = [{"wage_code": code, "description": desc} for code, desc in code_to_desc.items()]
    glossary.sort(
        key=lambda x: (0, int(x["wage_code"])) if x["wage_code"].isdigit() else (1, x["wage_code"]),
    )
    return glossary


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------

def write_payslips_csv(payslips: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect all unique wage codes (earnings + ytd)
    wage_cols_set: Set[str] = set()
    for p in payslips:
        earnings = p.get("earnings", {}) or {}
        for item in earnings.get("line_items", []):
            col_key = _wage_col_key(item.get("wage_code", ""))
            if col_key:
                wage_cols_set.add(col_key)
        ytd = p.get("ytd_summary", {}) or {}
        for adj in ytd.get("net_adjustments", []):
            col_key = _ytd_col_key(adj.get("wage_code", ""), adj.get("description", ""))
            if col_key:
                wage_cols_set.add(col_key)

    wage_columns = sorted(
        wage_cols_set,
        key=lambda x: (0, int(x.replace("Y_", ""))) if x.replace("Y_", "").isdigit() else (1, x),
    )

    static_cols = [
        "employee_name", "employee_number", "month", "year",
        "correction_number", "language",
        "total_gross", "net_amount",
        "bank_name", "iban",
        "sv_ag_anteil", "zus_ag_kosten", "gesamtkosten", "auszahlungsbetrag",
    ]
    deduction_cols = _build_deduction_columns()
    fieldnames = static_cols + deduction_cols + wage_columns

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for payslip in payslips:
            writer.writerow(payslip_to_row(payslip, wage_columns))


def write_glossary_csv(payslips: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    glossary = build_glossary(payslips)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["wage_code", "description"])
        writer.writeheader()
        for row in glossary:
            writer.writerow(row)
