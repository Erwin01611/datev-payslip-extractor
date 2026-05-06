#!/bin/bash
# Double-click to start the Payslip Processor (Tkinter desktop app).
# Opens a fresh Terminal window so the app works regardless of your current directory.

# Try multiple ways to find the app folder (handles different unzip locations)
if [ -f "$(dirname "$0")/tkinter_app.py" ]; then
    APP_DIR="$(cd "$(dirname "$0")" && pwd)"
elif [ -f "$PWD/tkinter_app.py" ]; then
    APP_DIR="$PWD"
elif [ -f "$HOME/Desktop/datev-payslip-extractor-v1.0.0-macos/tkinter_app.py" ]; then
    APP_DIR="$HOME/Desktop/datev-payslip-extractor-v1.0.0-macos"
else
    osascript -e 'display alert "Payslip Processor not found" message "Could not locate the app folder. Please run .venv/bin/python tkinter_app.py from the app folder."'
    exit 1
fi

osascript <<EOF
tell application "Terminal"
    do script "cd '$APP_DIR' && .venv/bin/python tkinter_app.py"
    activate
end tell
EOF
