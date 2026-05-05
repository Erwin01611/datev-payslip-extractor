#!/bin/bash
# Double-click to start the Payslip Processor (Tkinter desktop app).
# Activates the existing virtual environment and launches the GUI.

cd "$(dirname "$0")"
source .venv/bin/activate
python tkinter_app.py
