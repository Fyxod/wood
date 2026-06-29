#!/usr/bin/env bash
set -euo pipefail

echo "[wood] Python:"
python --version

echo "[wood] Installing WOOD dependencies into the active environment"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

echo "[wood] Done"
