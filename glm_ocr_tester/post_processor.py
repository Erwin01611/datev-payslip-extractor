"""Post-processing layer to fix known edge cases on DATEV payslips."""

import re
from typing import Optional

from glm_ocr_tester.models import PayslipSchema


def _parse_amount(s: str) -> Optional[float]:
    """Parse a German/European amount string into a float."""
    s = s.strip()
    negative = s.endswith("-")
    if negative:
        s = s[:-1].strip()
    # Remove thousands separators (periods), convert decimal comma
    s = s.replace(".", "").replace(",", ".")
    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return None


def _find_amount_after_keyword(text: str, keyword: str) -> Optional[float]:
    """Search for keyword followed by an amount, handling same-line and next-line cases."""
    # Try same-line first
    same_line = re.compile(
        rf"{re.escape(keyword)}\s+([\d\.]+,\d{{2}}-?|\d{{1,3}}(?:\.\d{{3}})*,\d{{2}}-?|\d+\.\d{{2}}-?)",
        re.IGNORECASE,
    )
    match = same_line.search(text)
    if match:
        parsed = _parse_amount(match.group(1))
        if parsed is not None:
            return parsed

    # If not found, search within a 200-char window after the keyword
    idx = text.lower().find(keyword.lower())
    if idx == -1:
        return None
    window = text[idx : idx + 300]
    # Look for any valid German amount in the window
    amount_pattern = re.compile(
        r"[\d\.]+,\d{2}-?|\d{1,3}(?:\.\d{3})*,\d{2}-?",
    )
    for m in amount_pattern.finditer(window):
        parsed = _parse_amount(m.group())
        if parsed is not None:
            return parsed
    return None


def apply_post_processing(payslip: PayslipSchema, raw_text: str) -> PayslipSchema:
    """Apply deterministic corrections based on raw pdfplumber text.

    Args:
        payslip: The PayslipSchema produced by the LLM refiner.
        raw_text: The raw text extracted directly from the PDF.

    Returns:
        The corrected PayslipSchema.
    """
    _fix_net_pay(payslip, raw_text)
    _fix_social_security_from_line_items(payslip, raw_text)
    _fix_missing_taxes(payslip, raw_text)
    _prune_spurious_taxes(payslip)
    _fix_total_deductions(payslip)
    _remove_duplicate_taxes(payslip)
    _arithmetic_reconciliation(payslip, raw_text)
    return payslip


def _fix_net_pay(payslip: PayslipSchema, raw_text: str) -> None:
    """Rule 1 + 3: Force correct net pay from raw text."""
    # German payslips
    if "Auszahlungsbetrag" in raw_text:
        val = _find_amount_after_keyword(raw_text, "Auszahlungsbetrag")
        if val is not None:
            payslip.net_pay.net_amount = val
            return

    # English payslips — if current net pay seems implausibly small
    if payslip.earnings.total_gross and payslip.net_pay.net_amount:
        if payslip.earnings.total_gross > 1000 and payslip.net_pay.net_amount < 100:
            val = _find_amount_after_keyword(raw_text, "Net pay")
            if val and val > 100:
                payslip.net_pay.net_amount = val


def _fix_social_security_from_line_items(payslip: PayslipSchema, raw_text: str) -> None:
    """Rule 5: Populate social_security fields from deduction line_items when missing."""
    if not payslip.deductions.social_security:
        return

    ss = payslip.deductions.social_security
    items = payslip.deductions.line_items

    mapping = [
        ("rentenversicherung", ["RV-Beitrag", "PI Contribution", "pension"]),
        ("krankenversicherung", ["KV-Beitrag", "HI Contribution", "health insurance"]),
        ("arbeitslosenversicherung", ["AV-Beitrag", "UI Contribution", "unemployment"]),
        ("pflegeversicherung", ["PV-Beitrag", "CI Contribution", "pflege"]),
    ]

    for field_name, keywords in mapping:
        # Always override from line items when there's a clear keyword match,
        # because the LLM often misassigns social_security schema fields.
        text_lower = raw_text.lower()
        has_keyword = any(kw.lower() in text_lower for kw in keywords)
        if not has_keyword:
            continue

        for item in items:
            desc_lower = (item.description or "").lower()
            if any(kw.lower() in desc_lower for kw in keywords):
                if item.employee_amount is not None:
                    ln_obj = getattr(ss, field_name)
                    if ln_obj is not None:
                        ln_obj.L = item.employee_amount
                    break


def _fix_missing_taxes(payslip: PayslipSchema, raw_text: str) -> None:
    """Rule 2: Try to locate missing kirchensteuer / solidaritaetszuschlag in raw text."""
    td = payslip.deductions.tax_details
    if td is None:
        return

    # Search for standalone tax amounts near their labels
    if td.kirchensteuer.L is None:
        val = _find_amount_after_keyword(raw_text, "Kirchensteuer")
        if val is not None and (td.lohnsteuer.L is None or abs(val) < td.lohnsteuer.L * 2):
            td.kirchensteuer.L = val

    if td.solidaritaetszuschlag.L is None:
        val = _find_amount_after_keyword(raw_text, "Solidaritätszuschlag")
        if val is not None and (td.lohnsteuer.L is None or abs(val) < td.lohnsteuer.L * 2):
            td.solidaritaetszuschlag.L = val

    # For English payslips
    if td.kirchensteuer.L is None:
        val = _find_amount_after_keyword(raw_text, "Church tax")
        if val is not None:
            td.kirchensteuer.L = val

    if td.solidaritaetszuschlag.L is None:
        val = _find_amount_after_keyword(raw_text, "Solidarity surcharge")
        if val is not None:
            td.solidaritaetszuschlag.L = val


def _prune_spurious_taxes(payslip: PayslipSchema) -> None:
    """If kirchensteuer / solidaritaetszuschlag break gross - net arithmetic, remove them."""
    gross = payslip.earnings.total_gross
    net = payslip.net_pay.net_amount
    td = payslip.deductions.tax_details
    ss = payslip.deductions.social_security

    if gross is None or net is None or td is None:
        return

    expected = gross - net
    def _lnv(obj):
        return obj.L if obj is not None else None

    ss_total = sum(
        abs(v) for v in [
            _lnv(ss.rentenversicherung) if ss else None,
            _lnv(ss.arbeitslosenversicherung) if ss else None,
            _lnv(ss.krankenversicherung) if ss else None,
            _lnv(ss.pflegeversicherung) if ss else None,
        ] if v is not None
    )

    tax_current = sum(abs(v) for v in [_lnv(td.lohnsteuer), _lnv(td.kirchensteuer), _lnv(td.solidaritaetszuschlag)] if v is not None)
    if abs(expected - (tax_current + ss_total)) <= 50:
        return  # arithmetic checks out

    # Try without kirchensteuer + solidaritaetszuschlag
    tax_alt = sum(abs(v) for v in [_lnv(td.lohnsteuer)] if v is not None)
    if abs(expected - (tax_alt + ss_total)) <= 50:
        # Prune from tax_details
        had_kirch = td.kirchensteuer.L is not None
        had_soli = td.solidaritaetszuschlag.L is not None
        td.kirchensteuer.L = None
        td.solidaritaetszuschlag.L = None

        # Also prune matching line_items
        for item in payslip.deductions.line_items:
            desc = (item.description or "").lower()
            if had_kirch and ("kirchen" in desc or "church" in desc):
                item.employee_amount = None
            if had_soli and ("solidar" in desc or "solidarity" in desc):
                item.employee_amount = None


def _fix_total_deductions(payslip: PayslipSchema) -> None:
    """Populate total_deductions if it's null by summing taxes + social security."""
    if payslip.deductions.total_deductions is not None:
        return

    def _lnv(obj):
        return obj.L if obj is not None else None

    total = 0.0
    if payslip.deductions.tax_details:
        for val in [_lnv(payslip.deductions.tax_details.lohnsteuer), _lnv(payslip.deductions.tax_details.kirchensteuer), _lnv(payslip.deductions.tax_details.solidaritaetszuschlag)]:
            if val is not None:
                total += abs(val)
    if payslip.deductions.social_security:
        for val in [_lnv(payslip.deductions.social_security.rentenversicherung), _lnv(payslip.deductions.social_security.arbeitslosenversicherung), _lnv(payslip.deductions.social_security.krankenversicherung), _lnv(payslip.deductions.social_security.pflegeversicherung)]:
            if val is not None:
                total += abs(val)

    if total > 0:
        payslip.deductions.total_deductions = round(total, 2)


def _remove_duplicate_taxes(payslip: PayslipSchema) -> None:
    """If the LLM duplicated lohnsteuer into other tax fields, clear the duplicates."""
    td = payslip.deductions.tax_details
    if not td or td.lohnsteuer.L is None:
        return
    if td.kirchensteuer.L == td.lohnsteuer.L:
        td.kirchensteuer.L = None
    if td.solidaritaetszuschlag.L == td.lohnsteuer.L:
        td.solidaritaetszuschlag.L = None


def _arithmetic_reconciliation(payslip: PayslipSchema, raw_text: str) -> None:
    """Rule 4: If gross - net - known deductions leaves a gap that matches a missing tax, fill it."""
    gross = payslip.earnings.total_gross
    net = payslip.net_pay.net_amount
    td = payslip.deductions.tax_details
    ss = payslip.deductions.social_security

    if gross is None or net is None:
        return

    def _lnv(obj):
        return obj.L if obj is not None else None

    # Sum known deductions
    known = 0.0
    if td:
        for val in [_lnv(td.lohnsteuer), _lnv(td.kirchensteuer), _lnv(td.solidaritaetszuschlag)]:
            if val is not None:
                known += abs(val)
    if ss:
        for val in [_lnv(ss.rentenversicherung), _lnv(ss.arbeitslosenversicherung), _lnv(ss.krankenversicherung), _lnv(ss.pflegeversicherung)]:
            if val is not None:
                known += abs(val)

    gap = gross - net - known
    if abs(gap) < 0.5:
        return  # Already balanced

    # If only one tax is missing and the gap matches a reasonable tax amount, fill it
    missing = []
    if td:
        if td.kirchensteuer.L is None:
            missing.append(("kirchensteuer", td))
        if td.solidaritaetszuschlag.L is None:
            missing.append(("solidaritaetszuschlag", td))

    if len(missing) == 1:
        field_name, parent = missing[0]
        # Sanity check: missing tax should typically be < 20% of lohnsteuer
        if td and td.lohnsteuer.L is not None and abs(gap) < td.lohnsteuer.L * 0.5:
            getattr(parent, field_name).L = round(gap, 2)
