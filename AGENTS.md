# GLM-OCR Payslip Pipeline — Agent Guide

## Project Overview

Extract structured data from German/English DATEV-format payslip PDFs using a **pure local vision pipeline**. No cloud APIs, no external LLM services. Runs on Apple Silicon via `mlx-vlm`.

## Architecture

```
PDF page (any page index)
    → Render @ 150 DPI (pdf2image)
    → Slice into 5 blocks (Tesseract anchor detection)
        → header, earnings, deductions, ytd_statement, bank_payout
    → Run vision model (mlx_vlm Qwen 3.5 9B) on each crop
        → 5 JSON responses per page
    → Merge into PayslipSchema dict
    → Deduplicate globally (German wins, fewer-nulls tiebreaker)
    → Validate arithmetic
    → Write JSON + CSV + glossary + warnings per month
```

## File Map

### Entry point
| File | Role |
|------|------|
| `process_payslips.py` | Main orchestrator. Accepts `sys.argv[1:]` PDF paths. Calls `process_page()` per page, then `deduplicate()`, `group_by_month()`, CSV/JSON writers. |

### Core pipeline (`glm_ocr_tester/`)
| File | Role |
|------|------|
| `block_slicer.py` | Render PDF → find semantic anchors with Tesseract (`lang='deu+eng'`) → compute 5 crop regions → save PNG crops. Anchors: `brutto_bezuege`, `steuer_sozial`, `net_income`/`verdienstbescheinigung`, `bank`/`auszahlungsbetrag`, `payments`, `tax_social`, `net pay`, `payout`, `statement`. |
| `vision_client.py` | Subprocess wrapper around `python -m mlx_vlm generate`. Parses JSON from model output, strips `<think>` blocks and markdown fences. `DEFAULT_TIMEOUT = 600`. |
| `header_extractor_vision.py` | Extracts employee name, number, month, year, correction number from header crop. Prompt: `prompts/header_v1.txt`. Language detection is unreliable (prompt forces English month names); see OCR fallback below. |
| `earnings_extractor_vision.py` | Extracts wage code, description, amount, total gross. Prompt: `prompts/earnings_v1.txt`. Strips `"N "` prefix from wage codes in `_map_earnings`. |
| `deductions_extractor_vision.py` | Extracts tax + social security details (L/N rows). Prompt: `prompts/deductions_v1.txt`. |
| `net_pay_extractor.py` | Extracts YTD net income + net adjustments. Prompt: `prompts/net_pay_v2.txt` (NOT v1). |
| `bank_payout_extractor_vision.py` | Extracts bank details, employer costs, payout amount. Prompt: `prompts/bank_payout_v1.txt`. |
| `merge_blocks.py` | Deep-merges per-block partial dicts into a single `PayslipSchema` object using `_BLOCK_PRECEDENCE`. |
| `csv_writer.py` | Pivot CSV writer. Collects unique wage codes → builds wide columns (`110`, `200`, `Y_9070`, etc.). |
| `validator.py` | Arithmetic validation: earnings gross total, tax total, SS total, net equation, payout equation. |
| `models.py` | Pydantic schemas: `PayslipSchema`, `Earnings`, `Deductions`, `LNValue`, `YtdSummary`, etc. |
| `config.py` | `Settings` dataclass with `inputs_dir`, `outputs_dir`, `pdf_dpi`. |
| `file_scanner.py` | PDF discovery logic. |
| `common_utils.py` | `parse_german_amount()` — handles comma decimal, thousands dot, trailing minus. |

### Prompts (`prompts/`)
| File | Used By | Status |
|------|---------|--------|
| `header_v1.txt` | `header_extractor_vision.py` | Active |
| `earnings_v1.txt` | `earnings_extractor_vision.py` | Active |
| `deductions_v1.txt` | `deductions_extractor_vision.py` | Active |
| `bank_payout_v1.txt` | `bank_payout_extractor_vision.py` | Active |
| `net_pay_v2.txt` | `net_pay_extractor.py` | Active |

### Test scripts
| File | Role |
|------|------|
| `test_vision_header.py` | Single-block header test |
| `test_vision_earnings.py` | Single-block earnings test |
| `test_vision_deductions.py` | Single-block deductions test |
| `test_vision_net_pay.py` | Single-block net pay test |
| `test_vision_bank_payout.py` | Single-block bank payout test |
| `test_vision_pipeline.py` | End-to-end single-PDF test |

### Dead files (not referenced by `process_payslips.py`)
- `runner.py`, `vision_pipeline.py`, `post_processor.py` — from earlier iterations, unused.

## Data Flow

### Per page (`process_page()`)
1. `slice_payslip(pdf_path, page_idx)` → returns `[(BlockRegion, tesseract_text), ...]`
2. Save header crop → `extract_header_safe()` → vision JSON
3. **OCR fallback**: `_extract_correction_from_ocr(header_text)` + `_detect_language_from_ocr(header_text)` override vision result
4. Create crop folder: `outputs/crops/{year}-{month}/{persnr}_{name}/` (or `..._{correction}/`)
5. Save all 5 crops + Tesseract text files
6. Run remaining 4 vision extractions
7. `_map_header()`, `_map_earnings()`, `_map_deductions()`, `_map_net_pay()`, `_map_bank_payout()` → partial dicts
8. `merge_blocks(partial_dicts)` → `PayslipSchema`
9. Return `payslip.model_dump()`

### Global (`main()`)
1. Collect all page dicts from all PDFs
2. `deduplicate(payslips)` — see below
3. `group_by_month(payslips)` — by `payroll_period.month_year`
4. Write `payslips_{timestamp}.json`, `.csv`, `glossary_{timestamp}.csv`, `warnings_{timestamp}.csv`

## Deduplication

```python
key = (pers_nr, month_year, norm_correction)
```

Where `norm_correction = _normalize_correction_number(correction)` extracts the leading digit:
- `"1.C"` → `"1"`
- `"1.NB"` → `"1"`
- `null` → `""`

**Rules:**
1. German wins over English (`lang == "de"` replaces `lang != "de"`)
2. Same language → keep the one with fewer null fields (`_count_nulls()`)
3. Existing German + new non-German → keep existing

**Key insight:** Corrections ARE deduplicated across languages. English `1.C` and German `1.NB` share the same key `(pers_nr, month_year, "1")` → German wins.

## OCR Fallback (Tesseract)

The vision model's header prompt says *"Return month as the English month name"*, which breaks language detection for German headers. Also, correction number extraction is non-deterministic (e.g., misses `1.NB`).

**Fallback in `process_page()`:**
- `_extract_correction_from_ocr(header_text)` — regex `\(\s*(\d+)\s*\.\s*(NB|C)\s*\)` extracts `1.NB` / `1.C` from Tesseract text
- `_detect_language_from_ocr(header_text)` — keyword scoring (`abrechnung` → `de`, `pay advice` → `en`)

These override the vision model result before `_map_header()` runs.

## Wage Code Normalization

Correction payslips print retroactive items with `"N "` prefix (e.g., `"N 110"`). The `_normalize_wage_code()` helper strips this prefix during schema mapping so regular and correction rows share the same CSV column (`110` instead of `N_110`).

Applied in:
- `_map_earnings()` — for earnings line items
- `_map_net_pay()` — for YTD adjustments

## Output Structure

```
outputs/
    {year}-{month}/
        payslips_{YYYYMMDD_HHMMSS}.json
        payslips_{YYYYMMDD_HHMMSS}.csv
        glossary_{YYYYMMDD_HHMMSS}.csv
        warnings_{YYYYMMDD_HHMMSS}.csv
    crops/
        {year}-{month}/
            {persnr}_{name}/
                header.png, header.txt
                earnings.png, earnings.txt
                deductions.png, deductions.txt
                ytd_statement.png, ytd_statement.txt
                bank_payout.png, bank_payout.txt
```

## Key Design Decisions

1. **Subprocess per vision call** — `subprocess.run([python, -m, mlx_vlm, generate, ...])` for each crop. Model loads from disk every time (~3-4 min cold start, but cached on SSD). Isolates crashes.

2. **Page-independent processing** — Every page is processed standalone. No pre-grouping by file or employee. Deduplication happens globally after all pages are extracted.

3. **5 semantic blocks** — header, earnings, deductions, ytd_statement, bank_payout. Vision model receives only the relevant crop with a focused prompt.

4. **Pre-cropped vision** — Instead of passing the full page to the model, we crop to the relevant section. This eliminates layout noise and improves accuracy.

5. **Universal 2-decimal rule** — Every monetary amount has exactly 2 cents digits. The last two digits are ALWAYS cents. Baked into every prompt.

6. **German wins dedup** — German payslips are the authoritative source. English versions are discarded when both exist.

## Prompt Versioning

Prompts are plain text files in `prompts/`. To update:
1. Copy `prompts/{name}_v1.txt` → `prompts/{name}_v2.txt`
2. Edit the prompt
3. Update the extractor's `_DEFAULT_PROMPT_PATH` to point to the new file
4. Test on all example types before committing

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `mlx_vlm` not found | Not installed | `pip install mlx-vlm` |
| Model download timeout | Slow internet / ~9 GB | Retry |
| Vision returns empty JSON | `max_tokens` too low | Increase to 800+ |
| Slicing fails | Bad scan / unusual format | Log warning, page skipped |
| Header vision fails | Blurry header | Falls back to OCR for correction + language |
| German correction detected as regular | Vision model misses `1.NB` | OCR fallback detects `(1. NB)` in Tesseract text |
| `N_110` column in CSV | Correction has `N 110` wage code | `_normalize_wage_code()` strips `N ` prefix |
| Duplicate rows for same employee | EN + DE versions both kept | Dedup should catch them if correction numbers match |
| `ss_total_mismatch_L` warning | Vision model reads wrong column | Check deductions crop; may be 8-column layout issue |
