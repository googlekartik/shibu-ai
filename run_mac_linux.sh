#!/bin/bash
# One-click setup for the Rule-Based Intelligent Chatbot (Mac / Linux)
# Run:  bash run_mac_linux.sh   (or double-click in Finder after chmod +x)
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed. Please install Python 3.11 or newer, then run this again."
  exit 1
fi
if [ ! -d venv ]; then
  echo "Creating virtual environment (first run only)..."
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
echo "Installing requirements (first run only, then it is quick)..."
pip install -r requirements.txt
# Open the browser in the background, then start the app in the foreground.
if command -v open >/dev/null 2>&1; then
  (sleep 2; open http://127.0.0.1:5000) &
elif command -v xdg-open >/dev/null 2>&1; then
  (sleep 2; xdg-open http://127.0.0.1:5000) &
fi
echo ""
echo "Chatbot is starting. Keep this window open while you use it."
echo "To stop it, press Ctrl+C in this window."
python3 app.py
