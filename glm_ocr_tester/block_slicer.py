"""Visually-anchored block slicer for DATEV-format payslips.

Pipeline:
1. Render PDF page to image (pdf2image)
2. Find visual section anchors with Tesseract OCR
3. Compute semantic crop regions from anchors
4. Extract native PDF text per band using pdfplumber
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


@dataclass
class BlockRegion:
    """A semantic region of the payslip page."""

    name: str
    label: str
    bbox_px: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in image pixels
    bbox_pdf: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in PDF points


# ---------------------------------------------------------------------------
# 1. Render
# ---------------------------------------------------------------------------

def get_page_count(pdf_path: str) -> int:
    """Return total number of pages in a PDF."""
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def render_page(pdf_path: str, page_idx: int = 0, dpi: int = 150) -> Image.Image:
    """Render a single PDF page to a PIL Image."""
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise ImportError(
            "pdf2image is required. Install it with: pip install pdf2image"
        ) from exc

    images = convert_from_path(pdf_path, dpi=dpi, first_page=page_idx + 1, last_page=page_idx + 1)
    if not images:
        raise RuntimeError(f"Could not render page {page_idx} from {pdf_path}")
    return images[0]


# ---------------------------------------------------------------------------
# 2. Anchor detection
# ---------------------------------------------------------------------------

# Single-word anchors: patterns must all appear in the SAME Tesseract word chunk.
_GERMAN_ANCHORS = {
    "brutto_bezuege": ["brutto", "bez"],
    "steuer_sozial": ["steuer", "sozial"],
    "net_income": ["netto-verdienst"],  # Tesseract often sees this as one hyphenated word
    "verdienst": ["verdienstbescheinigung"],
    "bank": ["bank"],  # works for all banks (Postbank, ING-DiBa, Deutsche Bank, etc.)
    "payout": ["auszahlungsbetrag"],  # fallback when Tesseract misreads "Bank"
}

_ENGLISH_ANCHORS = {
    "payments": ["payments"],
    "tax_social": ["tax", "social"],
    "bank": ["bank"],  # single-word match; "acc.no" is separate in Tesseract
}

# Multi-word phrases that Tesseract often splits into separate words.
# Format: key -> (first_word, second_word, max_x_gap, max_y_gap)
_GERMAN_MULTIWORD = {
    # Fallback if Tesseract splits "Netto-Verdienst" into two words
    "net_income": ("netto", "verdienst", 200, 5),
}

_ENGLISH_MULTIWORD = {
    "net_income": ("net", "income", 200, 5),
    "statement": ("statement", "earnings", 300, 5),
    "payout": ("net", "pay", 200, 20),  # fallback when "Bank" is misread (y-tol=20 for line spacing)
}


def find_visual_anchors(
    image: Image.Image, language: str = "auto", min_y: int = 200
) -> Dict[str, dict]:
    """Run Tesseract on the image and locate section-header anchors.

    Args:
        image: PIL Image of the rendered PDF page.
        language: "german", "english", or "auto" (tries both).
        min_y: Ignore anchors above this y-pixel to avoid title-text false positives.

    Returns:
        Dict mapping anchor_name -> {"x": int, "y": int, "text": str}
    """
    try:
        import pytesseract
    except ImportError as exc:
        raise ImportError(
            "pytesseract is required. Install it with: pip install pytesseract"
        ) from exc

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Build a list of all word objects for proximity searches
    words = []
    for i, text in enumerate(data["text"]):
        text_stripped = text.strip()
        if not text_stripped:
            continue
        words.append({
            "text": text_stripped,
            "lower": text_stripped.lower(),
            "x": data["left"][i],
            "y": data["top"][i],
            "width": data["width"][i],
        })

    def _match_single_word(lang_anchors: dict) -> dict:
        """Match anchors where all patterns must be in the SAME word chunk."""
        found: dict = {}
        for w in words:
            if w["y"] < min_y:
                continue
            for key, patterns in lang_anchors.items():
                if all(p in w["lower"] for p in patterns):
                    if key not in found or w["y"] < found[key]["y"]:
                        found[key] = {
                            "x": w["x"],
                            "y": w["y"],
                            "text": w["text"],
                        }
        return found

    def _match_multi_word_phrases(found: dict, phrases: dict) -> dict:
        """Look for multi-word phrases (e.g. 'Net income') where Tesseract
        split them into separate words. We look for the first word, then check
        if the second word appears within a small horizontal+vertical window."""
        for key, (first_word, second_word, x_window, y_window) in phrases.items():
            if key in found:
                continue  # Already found by single-word matcher
            for i, w1 in enumerate(words):
                if w1["y"] < min_y:
                    continue
                if first_word not in w1["lower"]:
                    continue
                # Look for second word nearby
                for w2 in words[i + 1 : i + 6]:
                    if second_word not in w2["lower"]:
                        continue
                    if abs(w2["y"] - w1["y"]) > y_window:
                        continue
                    if abs(w2["x"] - (w1["x"] + w1["width"])) > x_window:
                        continue
                    # Found the phrase
                    found[key] = {
                        "x": w1["x"],
                        "y": w1["y"],
                        "text": f"{w1['text']} {w2['text']}",
                    }
                    break
                if key in found:
                    break
        return found

    def _match(lang_anchors: dict, multi_word: dict = None) -> dict:
        found = _match_single_word(lang_anchors)
        if multi_word:
            found = _match_multi_word_phrases(found, multi_word)
        return found

    if language.lower() in ("de", "german", "deutsch"):
        return _match(_GERMAN_ANCHORS, _GERMAN_MULTIWORD)
    elif language.lower() in ("en", "english"):
        return _match(_ENGLISH_ANCHORS, _ENGLISH_MULTIWORD)
    else:
        # Auto: try German first, then English, merge results
        de = _match(_GERMAN_ANCHORS, _GERMAN_MULTIWORD)
        en = _match(_ENGLISH_ANCHORS, _ENGLISH_MULTIWORD)
        merged = {**en, **de}  # German wins on collision (de keys overwrite en)
        return merged


# ---------------------------------------------------------------------------
# 3. Crop region computation
# ---------------------------------------------------------------------------

def compute_crop_regions(
    anchors: Dict[str, dict],
    image_size: Tuple[int, int],
    language: str = "auto",
) -> List[BlockRegion]:
    """Turn detected anchors into 5 semantic BlockRegions.

    IMPORTANT: The Netto-Verdienst / Net income row is intentionally
    placed inside the YTD/statement block (04) so it is not split
    across the deductions and statement blocks.
    """
    w, h = image_size
    regions: List[BlockRegion] = []

    # Helper to map image y -> PDF y
    def _pdf_y(y_px: int, dpi: int = 150) -> float:
        return y_px * 72.0 / dpi

    def _region(name: str, label: str, y1_px: int, y2_px: int) -> BlockRegion:
        return BlockRegion(
            name=name,
            label=label,
            bbox_px=(0, max(0, y1_px), w, min(h, y2_px)),
            bbox_pdf=(0, _pdf_y(max(0, y1_px)), w, _pdf_y(min(h, y2_px))),
        )

    is_german = bool(
        "brutto_bezuege" in anchors or "steuer_sozial" in anchors
    )

    if is_german:
        # German DATEV layout
        y_brutto = anchors["brutto_bezuege"]["y"]
        y_steuer = anchors["steuer_sozial"]["y"]
        y_verdienst = anchors.get("verdienst", {}).get("y", int(h * 0.70))
        y_bank = anchors.get("bank", {}).get("y") or anchors.get("payout", {}).get("y") or int(h * 0.88)

        # Net income row is between deductions and verdienst.
        # If Tesseract found it, use it directly; otherwise estimate.
        y_net = anchors.get("net_income", {}).get("y")
        if y_net is None:
            # Estimate: roughly 80 % of the way from steuer to verdienst
            y_net = y_steuer + int((y_verdienst - y_steuer) * 0.80)

        regions.append(_region("header", "Header / Meta", 0, y_brutto - 10))
        regions.append(_region("earnings", "Brutto-Bezuege / Earnings", y_brutto - 5, y_steuer + 30))
        # Deductions end BEFORE the Net income row so Netto-Verdienst stays in YTD block
        regions.append(_region("deductions", "Steuer/Sozialversicherung / Deductions", y_steuer + 15, y_net - 5))
        regions.append(_region("ytd_statement", "Verdienstbescheinigung / YTD Statement", y_net - 5, y_bank - 10))
        regions.append(_region("bank_payout", "Bank & Auszahlungsbetrag / Bank & Payout", y_bank - 10, h))

    else:
        # English layout
        y_payments = anchors.get("payments", {}).get("y", int(h * 0.35))
        y_tax = anchors.get("tax_social", {}).get("y", int(h * 0.50))
        y_statement = anchors.get("statement", {}).get("y", int(h * 0.68))
        y_bank = anchors.get("bank", {}).get("y") or anchors.get("payout", {}).get("y") or int(h * 0.85)

        y_net = anchors.get("net_income", {}).get("y")
        if y_net is None:
            y_net = y_tax + int((y_statement - y_tax) * 0.85)

        regions.append(_region("header", "Header / Meta", 0, y_payments - 10))
        regions.append(_region("earnings", "Payments / Earnings", y_payments - 5, y_tax + 25))
        regions.append(_region("deductions", "Tax/Social Security / Deductions", y_tax + 15, y_net - 5))
        regions.append(_region("ytd_statement", "Statement of earnings / YTD Statement", y_net - 5, y_bank - 10))
        regions.append(_region("bank_payout", "Bank & Net pay / Bank & Payout", y_bank - 10, h))

    return regions


# ---------------------------------------------------------------------------
# 4. PDF band text extraction
# ---------------------------------------------------------------------------

def extract_band_text(
    pdf_path: str,
    page_idx: int,
    y_start_pdf: float,
    y_end_pdf: float,
) -> str:
    """Extract native text from a horizontal band of a PDF page.

    Uses pdfplumber to read words whose 'top' coordinate falls within
    [y_start_pdf, y_end_pdf], then reconstructs reading order with
    simple top-to-bottom, left-to-right sorting.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required. Install it with: pip install pdfplumber"
        ) from exc

    words: List[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_idx]
        for w in page.extract_words():
            top = float(w["top"])
            if y_start_pdf <= top <= y_end_pdf:
                words.append(w)

    if not words:
        return ""

    # Sort by y (top) then x (left) to reconstruct reading order
    words.sort(key=lambda w: (float(w["top"]), float(w["x0"])))

    # Group into lines by similar y-coordinate
    lines: List[List[str]] = []
    current_line: List[Tuple[float, str]] = []
    last_y = None
    y_tolerance = 3.0  # PDF points

    for w in words:
        y = float(w["top"])
        if last_y is None or abs(y - last_y) <= y_tolerance:
            current_line.append((float(w["x0"]), w["text"]))
            last_y = y
        else:
            current_line.sort(key=lambda t: t[0])
            lines.append([t[1] for t in current_line])
            current_line = [(float(w["x0"]), w["text"])]
            last_y = y

    if current_line:
        current_line.sort(key=lambda t: t[0])
        lines.append([t[1] for t in current_line])

    return "\n".join(" ".join(line_words) for line_words in lines)


# ---------------------------------------------------------------------------
# 5. Convenience: full slice pipeline for a single PDF
# ---------------------------------------------------------------------------

def slice_payslip(
    pdf_path: str,
    language: str = "auto",
    dpi: int = 150,
    page_idx: int = 0,
) -> List[Tuple[BlockRegion, str]]:
    """Run the full slice pipeline and return (region, text) pairs.

    Args:
        pdf_path: Path to the PDF file.
        language: "german", "english", or "auto".
        dpi: Rendering resolution for anchor detection.
        page_idx: Which page to process (default 0 for single-page payslips).

    Returns:
        List of (BlockRegion, block_text) tuples.
    """
    image = render_page(pdf_path, page_idx=page_idx, dpi=dpi)
    anchors = find_visual_anchors(image, language=language)
    regions = compute_crop_regions(anchors, image.size, language=language)

    results = []
    for region in regions:
        text = extract_band_text(
            pdf_path,
            page_idx,
            region.bbox_pdf[1],  # y1
            region.bbox_pdf[3],  # y2
        )
        results.append((region, text))

    return results
