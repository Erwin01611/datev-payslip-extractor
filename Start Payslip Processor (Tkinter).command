#!/bin/bash
# Double-click to start the Payslip Processor (Tkinter desktop app).
# Uses the bundled Python environment directly (no activation needed).

cd "$(dirname "$0")"
.venv/bin/python tkinter_app.py
