"""Orchestrates file discovery, block slicing, vision extraction, and output writing.

Pure vision-model pipeline — no OpenRouter, no regex extractors, no LLM APIs.
Each PDF is sliced into 5 semantic blocks, each block is fed to the local
mlx_vlm vision model, and the results are merged into a PayslipSchema.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from glm_ocr_tester.config import SETTINGS

from glm_ocr_tester.file_scanner import scan
from glm_ocr_tester.models import PayslipSchema, ValidationIssue
from glm_ocr_tester.post_processor import apply_post_processing
from glm_ocr_tester.validator import validate_payslip, write_warnings_csv
from glm_ocr_tester.block_slicer import slice_payslip
from glm_ocr_tester.merge_blocks import merge_blocks
from glm_ocr_tester.vision_pipeline import extract_all_from_crops

logger = logging.getLogger(__name__)

RAW_DIRS = {
    "markdown": Path(SETTINGS.outputs_dir) / "raw_markdown",
    "table": Path(SETTINGS.outputs_dir) / "raw_tables",
    "json": Path(SETTINGS.outputs_dir) / "structured_json",
    "blocks": Path(SETTINGS.outputs_dir) / "raw_blocks",
}

CROPS_DIR = Path(SETTINGS.outputs_dir) / "crops"
VALIDATION_DIR = Path(SETTINGS.outputs_dir) / "validation_reports"
REVIEW_DIR = Path(SETTINGS.outputs_dir) / "review_queue"


def _ensure_dirs() -> None:
    for d in list(RAW_DIRS.values()) + [CROPS_DIR, VALIDATION_DIR, REVIEW_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    """Remove problematic characters from filenames."""
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _count_nulls(schema: PayslipSchema) -> int:
    """Rough heuristic: count how many top-level string/number fields are null."""
    flat = schema.model_dump()
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
    walk(flat)
    return count


class Runner:
    """Main orchestrator — vision-only pipeline."""

    def __init__(self, prompts: Optional[List[str]] = None):

        self.prompts = prompts or ["json"]
        _ensure_dirs()

    def run(self) -> Dict[str, dict]:
        """Process all discovered files and return a summary dict."""
        files = scan()
        if not files:
            logger.warning("No supported files found in project root.")
            return {}

        summary: Dict[str, dict] = {}
        logger.info("Found %d file(s) to process.", len(files))

        for file_path in files:
            file_key = file_path.name
            summary[file_key] = {"status": "success", "prompts": {}}
            logger.info("Processing: %s", file_key)

            try:
                if file_path.suffix.lower() == ".pdf":
                    self._process_pdf(file_path, summary[file_key])
                else:
                    logger.warning("Non-PDF file %s skipped (vision pipeline only supports PDFs)", file_key)
                    summary[file_key]["status"] = "skipped"
            except Exception as exc:
                logger.exception("Failed to process %s: %s", file_key, exc)
                summary[file_key]["status"] = "error"
                summary[file_key]["error"] = str(exc)

        return summary

    def _process_pdf(self, file_path: Path, file_summary: dict) -> None:
        """Run vision pipeline on a single PDF."""
        stem = _sanitize_filename(file_path.stem)

        # Slice into blocks
        try:
            blocks = slice_payslip(str(file_path), language="auto")
        except Exception as exc:
            logger.warning("Block slicing failed for %s: %s", stem, exc)
            file_summary["prompts"]["json"] = {"status": "slicing_error", "error": str(exc)}
            return

        if len(blocks) < 3:
            logger.warning("Only %d blocks found for %s, insufficient.", len(blocks), stem)
            file_summary["prompts"]["json"] = {"status": "too_few_blocks", "block_count": len(blocks)}
            return

        # Render page for crop images
        page_image = None
        try:
            from glm_ocr_tester.block_slicer import render_page
            page_image = render_page(str(file_path), page_idx=0, dpi=150)
        except Exception as exc:
            logger.warning("Could not render page for vision crops: %s", exc)

        # Save block text and crops
        for region, text in blocks:
            block_path = RAW_DIRS["blocks"] / f"{stem}_{region.name}.txt"
            block_path.write_text(text, encoding="utf-8")

            if page_image is not None:
                try:
                    crop = page_image.crop(region.bbox_px)
                    crop_path = CROPS_DIR / f"{stem}_{region.name}.png"
                    crop.save(crop_path)
                except Exception as exc:
                    logger.warning("Failed to save crop for %s (%s): %s", region.name, stem, exc)

        # Run vision extraction on all crops
        logger.info("Running vision pipeline on %d blocks for %s", len(blocks), stem)
        block_results = extract_all_from_crops(blocks, CROPS_DIR, stem)

        if not block_results:
            logger.warning("No vision blocks extracted for %s", stem)
            file_summary["prompts"]["json"] = {"status": "no_vision_results"}
            return

        # Merge into full schema
        try:
            payslip = merge_blocks(block_results)
        except Exception as exc:
            logger.warning("Block merge failed for %s: %s", stem, exc)
            file_summary["prompts"]["json"] = {"status": "merge_error", "error": str(exc)}
            return

        # Post-process with all block texts
        all_text = "\n\n".join(text for _, text in blocks)
        try:
            payslip = apply_post_processing(payslip, all_text)
        except Exception as exc:
            logger.warning("Post-processing failed for %s: %s", stem, exc)

        # Save merged JSON
        parsed_path = RAW_DIRS["json"] / f"{stem}_parsed.json"
        parsed_path.write_text(payslip.model_dump_json(indent=2), encoding="utf-8")

        file_summary["prompts"]["json"] = {
            "status": "ok",
            "parsed": True,
            "null_field_count": _count_nulls(payslip),
            "block_based": True,
            "vision_only": True,
        }

        # Validate
        try:
            issues = validate_payslip(payslip)
            report = {
                "file": stem,
                "valid": len(issues) == 0,
                "issues": [issue.model_dump() for issue in issues],
                "language": payslip.language_detected,
                "null_field_count": _count_nulls(payslip),
            }
            report_path = VALIDATION_DIR / f"{stem}_validation.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            file_summary["prompts"]["json"]["validation"] = report
        except Exception as exc:
            logger.warning("Validation failed for %s: %s", stem, exc)
            issues = []

        # Decide if review is needed
        reasons = []
        if issues:
            reasons.append("validation_failure")
        if _count_nulls(payslip) > 40:
            reasons.append("low_completeness")
        if not payslip.earnings.line_items:
            reasons.append("no_line_items")

        if reasons:
            self._flag_for_review(
                stem,
                file_path,
                payslip,
                issues,
                ", ".join(reasons),
            )

    def _flag_for_review(
        self,
        stem: str,
        original_file: Path,
        payslip: Optional[PayslipSchema],
        issues: List[ValidationIssue],
        reason: str,
    ) -> None:
        """Copy original file and write review metadata."""
        review_meta = {
            "file_name": original_file.name,
            "extracted_schema": payslip.model_dump() if payslip else None,
            "validation_issues": [issue.model_dump() for issue in issues],
            "missing_field_count": _count_nulls(payslip) if payslip else None,
            "reason_for_review": reason,
        }
        review_path = REVIEW_DIR / f"{stem}_review.json"
        review_path.write_text(json.dumps(review_meta, indent=2), encoding="utf-8")

        # Copy original PDF into review queue for convenience
        dest = REVIEW_DIR / original_file.name
        shutil.copy2(str(original_file), str(dest))
        logger.info("Flagged %s for review: %s", stem, reason)
