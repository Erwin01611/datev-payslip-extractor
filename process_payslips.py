#!/usr/bin/env python3
"""Main entry point: batch-process payslip PDFs → JSON + CSV per month.

Usage:
    .venv/bin/python process_payslips.py inputs/

Output:
    outputs/2025-08/payslips_YYYYMMDD_HHMMSS.json
    outputs/2025-08/payslips_YYYYMMDD_HHMMSS.csv
    outputs/crops/2025-08/00002_Babett_Moll/header.png
    ...
"""

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

# Setup logging before any imports that might log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from glm_ocr_tester.block_slicer import get_page_count, render_page, slice_payslip
from glm_ocr_tester.config import SETTINGS
from glm_ocr_tester.csv_writer import write_payslips_csv, write_glossary_csv
from glm_ocr_tester.validator import validate_all, write_warnings_csv
from glm_ocr_tester.file_scanner import scan
from glm_ocr_tester.header_extractor_vision import extract_header_safe
from glm_ocr_tester.earnings_extractor_vision import extract_earnings_safe
from glm_ocr_tester.deductions_extractor_vision import extract_deductions_safe
from glm_ocr_tester.net_pay_extractor import extract_net_pay_safe
from glm_ocr_tester.bank_payout_extractor_vision import extract_bank_payout_safe
from glm_ocr_tester.merge_blocks import merge_blocks
from glm_ocr_tester.models import PayslipSchema


def _sanitize_folder_name(name: str) -> str:
    """Replace characters that don't belong in folder names."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _normalize_wage_code(code: Optional[str]) -> Optional[str]:
    """Strip leading 'N ' prefix from wage codes (e.g. 'N 110' → '110').

    The 'N' prefix is printed on correction payslips for retroactive items.
    Stripping it ensures regular and correction rows share the same CSV column.
    """
    if not code:
        return code
    code = code.strip()
    if code.upper().startswith("N "):
        return code[2:].strip() or None
    return code


def _normalize_correction_number(correction: Optional[str]) -> str:
    """Extract numeric prefix from correction numbers (e.g. '1.C' → '1', '1.NB' → '1').

    English corrections are labeled '1.C', '2.C', etc.
    German corrections are labeled '1.NB', '2.NB', etc. (NB = Nachberechnung).
    Normalizing to the numeric part allows English and German versions of the
    same correction to collide in the dedup dict.
    """
    if not correction:
        return ""
    m = re.search(r"^(\d+)", str(correction).strip())
    return m.group(1) if m else ""


def _extract_correction_from_ocr(text: str) -> Optional[str]:
    """Extract correction number from Tesseract OCR header text.

    German headers: '(1. NB)'  → '1.NB'
    English headers: '(1.C)'   → '1.C'
    """
    m = re.search(r"\(\s*(\d+)\s*\.\s*(NB|C)\s*\)", text, re.IGNORECASE)
    if m:
        return f"{m.group(1)}.{m.group(2).upper()}"
    return None


def _detect_language_from_ocr(text: str) -> str:
    """Detect language from Tesseract OCR header text.

    More reliable than vision-model language detection because the prompt
    forces English month names, breaking lang detection for German headers.
    """
    text_lower = text.lower()
    german_markers = ["abrechnung", "brutto/netto", "personal-nr", "pers.-nr.", "blatt:", "geburtstag"]
    english_markers = ["pay advice", "pers. no.", "page:", "date of birth", "social security number"]

    g_score = sum(1 for m in german_markers if m in text_lower)
    e_score = sum(1 for m in english_markers if m in text_lower)

    if g_score > e_score:
        return "de"
    elif e_score > g_score:
        return "en"
    return ""


def _extract_header_first(image: Image.Image, tmp_dir: Path) -> Optional[dict]:
    """Extract header block first to get employee metadata.

    Returns header vision result or None on failure.
    Saves header crop to tmp_dir/header.png temporarily.
    """
    from glm_ocr_tester.block_slicer import BlockRegion

    # Slice the page to get header region
    # We need to call slice_payslip which requires a PDF path, not an image.
    # So we save the image temporarily and pass it... wait, slice_payslip needs pdf_path.
    # Alternative: we can just crop the top 40% of the page as a heuristic header region.
    # But that's hacky.

    # Better approach: the caller already has the regions from slice_payslip.
    # So this function should receive the header crop directly.
    pass  # Will be handled in process_page


def process_page(
    pdf_path: str,
    page_idx: int,
    crops_base_dir: Path,
) -> Optional[dict]:
    """Process one PDF page end-to-end.

    1. Render page → image
    2. Slice into 5 blocks
    3. Header vision FIRST → get name, Pers.-Nr., month, year
    4. Create crop folder
    5. Save all 5 crops
    6. Run remaining 4 vision extractions
    7. Merge → return PayslipSchema dict

    Returns None if critical failure.
    """
    stem = Path(pdf_path).stem
    logger.info("Processing page %d of %s", page_idx, stem)

    try:
        blocks = slice_payslip(pdf_path, language="auto", page_idx=page_idx)
    except Exception as exc:
        logger.warning("Slicing failed for %s page %d: %s", stem, page_idx, exc)
        return None

    if len(blocks) < 3:
        logger.warning("Only %d blocks found for %s page %d", len(blocks), stem, page_idx)
        return None

    # Render the page for cropping
    try:
        page_image = render_page(pdf_path, page_idx=page_idx, dpi=150)
    except Exception as exc:
        logger.warning("Could not render %s page %d: %s", stem, page_idx, exc)
        return None

    # Build a lookup: region_name → (region, text)
    block_map = {region.name: (region, text) for region, text in blocks}

    # --- Step 1: Header FIRST ---
    header_region, header_text = block_map.get("header", (None, ""))
    if header_region is None:
        logger.warning("No header block found for %s page %d", stem, page_idx)
        return None

    # Save header crop temporarily for vision
    header_crop = page_image.crop(header_region.bbox_px)
    header_tmp = crops_base_dir / "_tmp" / f"{stem}_p{page_idx}_header.png"
    header_tmp.parent.mkdir(parents=True, exist_ok=True)
    header_crop.save(header_tmp)

    header_result = extract_header_safe(str(header_tmp))
    header_tmp.unlink(missing_ok=True)

    if "_error" in header_result:
        logger.warning("Header vision failed for %s page %d: %s", stem, page_idx, header_result["_error"])

    # --- OCR fallback: correction number + language ---
    # Tesseract OCR is 100% reliable for these patterns; vision model is not.
    ocr_correction = _extract_correction_from_ocr(header_text)
    ocr_language = _detect_language_from_ocr(header_text)
    if ocr_correction:
        header_result.setdefault("payroll_period", {})["correction_number"] = ocr_correction
        logger.info("OCR correction override: %s → %s", page_idx, ocr_correction)
    if ocr_language:
        header_result["language_detected"] = ocr_language
        logger.info("OCR language override: %s → %s", page_idx, ocr_language)

    # Extract metadata from header
    emp = header_result.get("employee", {})
    payroll = header_result.get("payroll_period", {})

    pers_nr = emp.get("employee_number") or "unknown"
    name = emp.get("name") or "unknown"
    month = payroll.get("month") or "unknown"
    year = payroll.get("year") or "unknown"
    correction = payroll.get("correction_number")
    language = header_result.get("language_detected", "")

    # Build crop folder name
    folder_name = f"{pers_nr}_{_sanitize_folder_name(name)}"
    if correction:
        folder_name += f"_{correction}"
    crop_dir = crops_base_dir / f"{year}-{month}" / folder_name
    crop_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Crops → %s", crop_dir)

    # --- Step 2: Save ALL 5 crops ---
    crop_paths: Dict[str, Path] = {}
    for region, text in blocks:
        crop = page_image.crop(region.bbox_px)
        crop_path = crop_dir / f"{region.name}.png"
        crop.save(crop_path)
        crop_paths[region.name] = crop_path

    # Save block text for debugging
    for region, text in blocks:
        txt_path = crop_dir / f"{region.name}.txt"
        txt_path.write_text(text, encoding="utf-8")

    # --- Step 3: Run all vision extractions ---
    block_results: List[dict] = []

    # Header (already extracted, just map it)
    if "_error" not in header_result:
        partial = _map_header(header_result)
        if partial:
            partial["_block_name"] = "header"
            block_results.append(partial)

    # Earnings
    if "earnings" in crop_paths:
        result = extract_earnings_safe(str(crop_paths["earnings"]))
        partial = _map_earnings(result)
        if partial:
            partial["_block_name"] = "earnings"
            block_results.append(partial)

    # Deductions
    if "deductions" in crop_paths:
        result = extract_deductions_safe(str(crop_paths["deductions"]))
        partial = _map_deductions(result)
        if partial:
            partial["_block_name"] = "deductions"
            block_results.append(partial)

    # Net pay (YTD statement)
    if "ytd_statement" in crop_paths:
        result = extract_net_pay_safe(str(crop_paths["ytd_statement"]))
        partial = _map_net_pay(result)
        if partial:
            partial["_block_name"] = "ytd_statement"
            block_results.append(partial)

    # Bank payout
    if "bank_payout" in crop_paths:
        result = extract_bank_payout_safe(str(crop_paths["bank_payout"]))
        partial = _map_bank_payout(result)
        if partial:
            partial["_block_name"] = "bank_payout"
            block_results.append(partial)

    # --- Step 4: Merge ---
    if not block_results:
        logger.warning("No blocks extracted for %s page %d", stem, page_idx)
        return None

    try:
        payslip = merge_blocks(block_results)
        return payslip.model_dump()
    except Exception as exc:
        logger.warning("Merge failed for %s page %d: %s", stem, page_idx, exc)
        return None


# ---------------------------------------------------------------------------
# Schema mappers (adapted from vision_pipeline.py)
# ---------------------------------------------------------------------------

def _map_header(result: dict) -> dict:
    """Map header vision output to PayslipSchema partial dict."""
    if "_error" in result:
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


def _map_earnings(result: dict) -> dict:
    """Map earnings vision output to PayslipSchema partial dict."""
    if "_error" in result:
        return {}
    earnings = result.get("earnings", {})
    if not earnings:
        return {}
    schema_items = []
    for item in earnings.get("line_items", []):
        schema_items.append({
            "wage_code": _normalize_wage_code(item.get("wage_code")),
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


def _map_deductions(result: dict) -> dict:
    """Map deductions vision output to PayslipSchema partial dict.

    Returns L and N values for all deduction fields.
    """
    if "_error" in result:
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


def _map_net_pay(result: dict) -> dict:
    """Map net pay vision output to PayslipSchema partial dict."""
    if "_error" in result:
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
                "wage_code": _normalize_wage_code(adj.get("wage_code")),
                "description": adj.get("description"),
                "amount": adj.get("amount"),
            })
        ytd = partial.get("ytd_summary", {})
        ytd["net_adjustments"] = schema_adjustments
        partial["ytd_summary"] = ytd

    return partial


def _map_bank_payout(result: dict) -> dict:
    """Map bank payout vision output to PayslipSchema partial dict."""
    if "_error" in result:
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


# ---------------------------------------------------------------------------
# Deduplication & grouping
# ---------------------------------------------------------------------------

def deduplicate(payslips: List[dict]) -> List[dict]:
    """Deduplicate payslips. German wins over English.

    Corrections are deduplicated across languages too (e.g. '1.C' English
    and '1.NB' German are the same document — keep German).
    """
    seen: Dict[tuple, dict] = {}

    for p in payslips:
        if not p:
            continue

        emp = p.get("employee", {}) or {}
        payroll = p.get("payroll_period", {}) or {}

        pers_nr = emp.get("employee_number") or emp.get("name") or ""
        month_year = payroll.get("month_year") or ""
        correction = payroll.get("correction_number") or ""
        lang = p.get("language_detected", "")

        norm_correction = _normalize_correction_number(correction)
        key = (pers_nr, month_year, norm_correction)

        existing = seen.get(key)
        if existing is None:
            seen[key] = p
            continue

        # Duplicate detected — German wins
        existing_lang = existing.get("language_detected", "")
        if lang == "de" and existing_lang != "de":
            seen[key] = p  # Replace with German
        elif lang == existing_lang:
            # Same language — keep the one with fewer nulls
            if _count_nulls(p) < _count_nulls(existing):
                seen[key] = p
        # else: existing is German, current is not → keep existing

    return list(seen.values())


def _count_nulls(data: dict) -> int:
    """Rough count of null/empty fields."""
    count = 0
    def walk(obj):
        nonlocal count
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif obj is None or obj == "" or obj == []:
            count += 1
    walk(data)
    return count


def group_by_month(payslips: List[dict]) -> Dict[str, List[dict]]:
    """Group payslips by month-year from payroll_period."""
    groups: Dict[str, List[dict]] = {}
    for p in payslips:
        payroll = p.get("payroll_period", {}) or {}
        month_year = payroll.get("month_year", "")
        if not month_year or month_year == " ":
            month_year = "unknown"
        groups.setdefault(month_year, []).append(p)
    return groups


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: str, crops_base_dir: Path) -> List[dict]:
    """Process all pages of one PDF."""
    results: List[dict] = []
    try:
        page_count = get_page_count(pdf_path)
    except Exception as exc:
        logger.warning("Could not get page count for %s: %s", pdf_path, exc)
        return results

    logger.info("%s has %d page(s)", Path(pdf_path).name, page_count)

    for page_idx in range(page_count):
        result = process_page(pdf_path, page_idx, crops_base_dir)
        if result:
            results.append(result)

    return results


def main():
    outputs_dir = Path(SETTINGS.outputs_dir)
    crops_base_dir = outputs_dir / "crops"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Discover PDFs from arguments (files or directories)
    pdf_files: List[Path] = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            p = Path(arg)
            if p.is_file() and p.suffix.lower() == ".pdf":
                pdf_files.append(p)
            elif p.is_dir():
                pdf_files.extend(p.glob("**/*.pdf"))
            else:
                logger.warning("Argument ignored (not a PDF or directory): %s", arg)
    else:
        pdf_files = list(Path(SETTINGS.inputs_dir).glob("**/*.pdf"))

    pdf_files = sorted(set(pdf_files))
    if not pdf_files:
        logger.warning("No PDFs found")
        return

    logger.info("Found %d PDF(s) to process", len(pdf_files))

    # Process all PDFs, all pages
    all_payslips: List[dict] = []
    for pdf_path in pdf_files:
        logger.info("Processing: %s", pdf_path.name)
        page_results = process_pdf(str(pdf_path), crops_base_dir)
        all_payslips.extend(page_results)
        logger.info("  %s: %d page(s) extracted", pdf_path.name, len(page_results))

    logger.info("Total payslips before dedup: %d", len(all_payslips))

    # Deduplicate
    deduped = deduplicate(all_payslips)
    logger.info("Total payslips after dedup: %d", len(deduped))

    # Group by month
    groups = group_by_month(deduped)

    # Write outputs
    for month_year, payslips in groups.items():
        safe_month = month_year.replace(" ", "_")
        out_dir = outputs_dir / safe_month
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"payslips_{timestamp}.json"
        csv_path = out_dir / f"payslips_{timestamp}.csv"

        json_path.write_text(
            json.dumps(payslips, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_payslips_csv(payslips, csv_path)

        glossary_path = out_dir / f"glossary_{timestamp}.csv"
        write_glossary_csv(payslips, glossary_path)

        # Validation warnings
        warnings = validate_all(payslips)
        if warnings:
            warnings_path = out_dir / f"warnings_{timestamp}.csv"
            write_warnings_csv(warnings, warnings_path)
            logger.info("  %d warning(s) → %s", len(warnings), warnings_path)

        logger.info("%s: %d payslip(s) → %s", month_year, len(payslips), out_dir)

    logger.info("Done. Output in: %s", outputs_dir)


if __name__ == "__main__":
    main()
