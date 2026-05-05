"""Tkinter desktop app for the Payslip OCR Processor.

Runs the existing process_payslips.py pipeline in a background thread
with a responsive native GUI. Zero changes to the underlying pipeline.
"""

import json
import logging
import queue
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

# ---------------------------------------------------------------------------
# Setup logging BEFORE importing process_payslips (which calls basicConfig)
# ---------------------------------------------------------------------------
_LOG_MSG_QUEUE = queue.Queue()


class _TkinterLogHandler(logging.Handler):
    """Custom handler that pushes log records to a thread-safe queue."""

    def __init__(self, target_queue: queue.Queue):
        super().__init__()
        self.target_queue = target_queue

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.target_queue.put(("log", msg))


# Attach handler to root logger so all pipeline logs are captured
_log_handler = _TkinterLogHandler(_LOG_MSG_QUEUE)
_log_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
)
logging.getLogger().addHandler(_log_handler)
logging.getLogger().setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Import existing pipeline (must come after logging setup)
# ---------------------------------------------------------------------------
from glm_ocr_tester.config import SETTINGS
from glm_ocr_tester.csv_writer import write_glossary_csv, write_payslips_csv
from glm_ocr_tester.validator import validate_all, write_warnings_csv
from glm_ocr_tester.block_slicer import get_page_count
from process_payslips import deduplicate, group_by_month, process_page


# ---------------------------------------------------------------------------
class PayslipApp:
    """Main application window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Payslip Processor")
        self.root.geometry("950x750")
        self.root.minsize(800, 600)

        # Threading primitives
        self.stop_event = threading.Event()
        self.msg_queue: queue.Queue = queue.Queue()
        self.pipeline_thread: threading.Thread | None = None

        # State
        self.pdf_paths: list[str] = []
        self.output_dir = Path(SETTINGS.outputs_dir).expanduser().resolve()
        self.temp_dir: Path | None = None
        self.results: list[dict] | None = None

        self._build_ui()
        self._poll_queues()

        # Handle window close while processing
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding="12")
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)

        # Header
        ttk.Label(main, text="📄  Payslip Processor", font=("Helvetica", 18, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 2)
        )
        ttk.Label(
            main, text="100% local  ·  No cloud  ·  Apple Silicon", foreground="gray"
        ).grid(row=1, column=0, sticky="w", pady=(0, 12))

        # --- File selection ---
        file_frame = ttk.LabelFrame(main, text="PDF Files", padding="10")
        file_frame.grid(row=2, column=0, sticky="ew", pady=4)
        file_frame.columnconfigure(0, weight=1)

        self.file_list = tk.Listbox(file_frame, height=5, selectmode=tk.EXTENDED)
        self.file_list.grid(row=0, column=0, sticky="ew")

        file_btn_frame = ttk.Frame(file_frame)
        file_btn_frame.grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Button(file_btn_frame, text="➕ Add PDFs", command=self._add_files).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(file_btn_frame, text="🗑 Clear", command=self._clear_files).pack(
            side=tk.LEFT
        )

        # --- Output folder ---
        out_frame = ttk.LabelFrame(main, text="Output Folder", padding="10")
        out_frame.grid(row=3, column=0, sticky="ew", pady=4)
        out_frame.columnconfigure(0, weight=1)

        self.out_var = tk.StringVar(value=str(self.output_dir))
        ttk.Entry(out_frame, textvariable=self.out_var).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(out_frame, text="Browse…", command=self._browse_output).grid(
            row=0, column=1
        )

        # --- Progress ---
        prog_frame = ttk.LabelFrame(main, text="Progress", padding="10")
        prog_frame.grid(row=4, column=0, sticky="ew", pady=4)
        prog_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            prog_frame, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew", columnspan=2, pady=(0, 6))

        self.status_var = tk.StringVar(value="Ready — add PDFs and click Process")
        ttk.Label(prog_frame, textvariable=self.status_var, font=("Helvetica", 10)).grid(
            row=1, column=0, sticky="w"
        )

        self.stop_btn = ttk.Button(
            prog_frame, text="⏹  Stop", command=self._stop, state="disabled"
        )
        self.stop_btn.grid(row=1, column=1, sticky="e")

        # --- Log viewer ---
        log_frame = ttk.LabelFrame(main, text="Processing Logs", padding="10")
        log_frame.grid(row=5, column=0, sticky="nsew", pady=4)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main.rowconfigure(5, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, height=12, state=tk.DISABLED
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        # --- Control buttons ---
        ctrl_frame = ttk.Frame(main)
        ctrl_frame.grid(row=6, column=0, sticky="ew", pady=(10, 0))

        self.start_btn = ttk.Button(
            ctrl_frame, text="▶️  Process", command=self._start, state="disabled"
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            ctrl_frame, text="📂 Open Output Folder", command=self._open_output
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(ctrl_frame, text="❓ About", command=self._show_about).pack(
            side=tk.RIGHT
        )

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------
    def _add_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="Select Payslip PDFs",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        for f in files:
            if f not in self.pdf_paths:
                self.pdf_paths.append(f)
                self.file_list.insert(tk.END, Path(f).name)
        self._update_buttons()

    def _clear_files(self) -> None:
        self.pdf_paths.clear()
        self.file_list.delete(0, tk.END)
        self._update_buttons()

    def _browse_output(self) -> None:
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.out_var.set(folder)
            self.output_dir = Path(folder)

    def _update_buttons(self) -> None:
        state = "normal" if self.pdf_paths else "disabled"
        self.start_btn.config(state=state)

    # ------------------------------------------------------------------
    # Processing control
    # ------------------------------------------------------------------
    def _start(self) -> None:
        if not self.pdf_paths:
            messagebox.showwarning("No files", "Please add at least one PDF.")
            return

        self.stop_event.clear()
        self.results = None

        # Prepare directories
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = Path(self.out_var.get()).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        crops_dir = self.output_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)

        # Copy PDFs to temp dir (so originals are never touched)
        temp_paths = []
        for src in self.pdf_paths:
            dst = self.temp_dir / Path(src).name
            shutil.copy2(src, dst)
            temp_paths.append(dst)

        # Reset UI
        self.progress_var.set(0)
        self.status_var.set("Starting…")
        self._clear_log()

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        # Launch background thread
        self.pipeline_thread = threading.Thread(
            target=self._run_pipeline,
            args=(temp_paths, crops_dir),
            daemon=True,
        )
        self.pipeline_thread.start()

    def _stop(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping after current page…")
        self.stop_btn.config(state="disabled")

    def _run_pipeline(self, pdf_paths: list[Path], crops_dir: Path) -> None:
        """Background thread: runs the pipeline and pushes messages to queue."""
        try:
            all_payslips: list[dict] = []
            total_pdfs = len(pdf_paths)

            for i, pdf_path in enumerate(pdf_paths):
                if self.stop_event.is_set():
                    self.msg_queue.put(("status", "Stopped by user."))
                    break

                self.msg_queue.put(
                    (
                        "progress",
                        int((i / total_pdfs) * 100),
                        f"Processing {pdf_path.name}  ({i + 1}/{total_pdfs})",
                    )
                )

                # Iterate pages directly so stop works between every page
                page_count = get_page_count(str(pdf_path))
                for page_idx in range(page_count):
                    if self.stop_event.is_set():
                        break
                    self.msg_queue.put((
                        "status",
                        f"{pdf_path.name} — page {page_idx + 1}/{page_count}",
                    ))
                    result = process_page(str(pdf_path), page_idx, crops_dir)
                    if result:
                        all_payslips.append(result)

                pages_done = sum(1 for p in all_payslips if str(pdf_path.name) in str(p.get("_source", "")))
                self.msg_queue.put(
                    (
                        "log",
                        f"✓ {pdf_path.name}: page loop done",
                    )
                )

            if self.stop_event.is_set():
                self.msg_queue.put(("done", []))
                return

            self.msg_queue.put(("status", "Deduplicating…"))
            deduped = deduplicate(all_payslips)

            self.msg_queue.put(("status", f"Grouping {len(deduped)} payslip(s) by month…"))
            groups = group_by_month(deduped)

            # Write outputs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_meta: list[dict] = []

            for month_year, payslips in groups.items():
                safe_month = month_year.replace(" ", "_")
                month_dir = self.output_dir / safe_month
                month_dir.mkdir(parents=True, exist_ok=True)

                json_path = month_dir / f"payslips_{timestamp}.json"
                csv_path = month_dir / f"payslips_{timestamp}.csv"
                glossary_path = month_dir / f"glossary_{timestamp}.csv"

                json_path.write_text(
                    json.dumps(payslips, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                write_payslips_csv(payslips, csv_path)
                write_glossary_csv(payslips, glossary_path)

                warnings = validate_all(payslips)
                warnings_path = None
                if warnings:
                    warnings_path = month_dir / f"warnings_{timestamp}.csv"
                    write_warnings_csv(warnings, warnings_path)

                results_meta.append(
                    {
                        "month": month_year,
                        "count": len(payslips),
                        "csv": str(csv_path),
                        "json": str(json_path),
                        "glossary": str(glossary_path),
                        "warnings": str(warnings_path) if warnings else None,
                    }
                )

            self.msg_queue.put(("progress", 100, "Complete"))
            self.msg_queue.put(("done", results_meta))

        except Exception as exc:
            logging.exception("Pipeline failed")
            self.msg_queue.put(("error", str(exc)))

        finally:
            # Always clean up temp files
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                self.temp_dir = None

    # ------------------------------------------------------------------
    # Queue polling (main thread)
    # ------------------------------------------------------------------
    def _poll_queues(self) -> None:
        """Called every 100 ms to drain queues and update UI."""
        # Pipeline messages
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass

        # Log messages
        try:
            while True:
                msg = _LOG_MSG_QUEUE.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass

        self.root.after(100, self._poll_queues)

    def _handle_msg(self, msg: tuple) -> None:
        msg_type = msg[0]

        if msg_type == "progress":
            _, value, text = msg
            self.progress_var.set(value)
            self.status_var.set(text)

        elif msg_type == "status":
            _, text = msg
            self.status_var.set(text)

        elif msg_type == "log":
            _, text = msg
            self._append_log(text)

        elif msg_type == "done":
            _, results = msg
            self.results = results
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            if results:
                months = ", ".join(r["month"] for r in results)
                messagebox.showinfo(
                    "Complete",
                    f"Processing finished!\n\n"
                    f"Months: {months}\n"
                    f"Output: {self.output_dir}",
                )
            else:
                messagebox.showinfo("Stopped", "Processing stopped by user.")

        elif msg_type == "error":
            _, err = msg
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            messagebox.showerror("Error", f"Pipeline failed:\n\n{err}")

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------
    def _append_log(self, text: str) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _open_output(self) -> None:
        subprocess.run(["open", str(self.output_dir)])

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "Payslip Processor\n"
            "Local MLX Vision Pipeline\n\n"
            "• 100% local — no cloud APIs\n"
            "• Runs on Apple Silicon GPU\n"
            "• Supports German & English payslips\n"
            "• Corrections & bilingual deduplication",
        )

    def _on_close(self) -> None:
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            if messagebox.askyesno(
                "Quit", "Processing is still running. Stop and quit?"
            ):
                self.stop_event.set()
                self.pipeline_thread.join(timeout=5)
            else:
                return
        self.root.destroy()


# ---------------------------------------------------------------------------
def main() -> None:
    root = tk.Tk()
    PayslipApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
