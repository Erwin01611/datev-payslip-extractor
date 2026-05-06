# Payslip OCR Processor

Extract structured data (JSON + CSV) from German/English DATEV-format payslip PDFs using a **local vision-language model**. No cloud APIs. No data leaves your Mac.

## What it does

Drop one or more payslip PDFs into a folder, run the script, and get:

- **JSON** — one file per month with all employee payslip data
- **CSV** — one file per month in pivot-table format (one row per employee)
- **Glossary** — mapping of wage codes to descriptions
- **Warnings** — validation issues (e.g., totals that don't add up)
- **Crops** — PNG snapshots of each payslip section for debugging

Supports bilingual payslips (German + English), corrections (Nachberechnung / 1.C, 1.NB), and mixed batches.

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **16 GB RAM** minimum (the model alone uses ~6 GB; 8 GB Macs will swap heavily)
- **Python 3.12**
- **Homebrew** packages: `tesseract` (with German language data), `poppler`

```bash
brew install tesseract tesseract-lang poppler
```

## Installation

```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install MLX vision model runner
pip install mlx-vlm

# Download the model (first run only, ~9 GB)
python -m mlx_vlm generate \
  --model mlx-community/Qwen3.5-9B-MLX-4bit \
  --max-tokens 10 \
  --prompt "hi" \
  --image /dev/null
```

> **Note:** The model is cached globally in `~/.cache/huggingface/`. Even if you re-install the app or create a new virtual environment, the model is reused — it is **never** downloaded twice for the same user.

## How to run

### Desktop app (recommended)

Double-click **`Start Payslip Processor (Tkinter).command`** or run:

```bash
source .venv/bin/activate
python tkinter_app.py
```

> **First run only:** The app downloads a ~9 GB AI model from HuggingFace. This takes 10–25 minutes depending on your connection. The status bar will show "Downloading AI model..." during this time. Subsequent runs are fast.

### Command line

```bash
# Single folder
.venv/bin/python process_payslips.py inputs/

# Multiple PDFs
.venv/bin/python process_payslips.py inputs/batch1.pdf inputs/batch2.pdf
```

## What happens

1. **Slice** — each PDF page is split into 5 sections (header, earnings, deductions, year-to-date, bank)
2. **Read** — a local vision model (Qwen 3.5 9B) reads each section and extracts data
3. **Merge** — all sections are combined into one structured record per page
4. **Deduplicate** — if the same employee appears in German and English, only the German version is kept
5. **Validate** — arithmetic checks (e.g., do earnings add up to the gross total?)
6. **Export** — JSON + CSV + glossary + warnings are saved per month

**Speed:** ~2 minutes per payslip (5 sections × ~25 seconds each).

## Output

Results are saved to a **timestamped** folder on your Desktop (e.g., `~/Desktop/Payslip_Results_20250403_143022`). Each run gets its own folder — nothing is overwritten.

```
Payslip_Results_20250403_143022/
├── 2025-August/
│   ├── payslips_20250505_013610.json
│   ├── payslips_20250505_013610.csv
│   ├── glossary_20250505_013610.csv
│   └── warnings_20250505_013610.csv
├── 2025-July/
│   └── ...
└── crops/
    └── 2025-August/
        └── 00002_FirstName_SecondName/
            ├── header.png
            ├── earnings.png
            ├── deductions.png
            ├── ytd_statement.png
            └── bank_payout.png
```

## Privacy

- **100% local** — the model runs on your Apple Silicon GPU
- **No cloud APIs** — no OpenAI, no OpenRouter, no external services
- **No internet needed** after the initial model download
- Your payroll data never leaves your machine

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `tesseract` not found | `brew install tesseract tesseract-lang` |
| `pdf2image` error | `brew install poppler` |
| Model download fails | Retry; first download is ~9 GB |
| Slicing fails on a page | Skipped automatically; check logs |
| Wrong wage code column (`N_110` vs `110`) | Fixed automatically — corrections use the same column as regular payslips |
| Duplicate rows for same employee | German + English versions are deduplicated automatically |
