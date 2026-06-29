$ErrorActionPreference = "Stop"

Write-Host "[wood] Python:"
python --version

Write-Host "[wood] Installing WOOD dependencies into the active environment"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

Write-Host "[wood] Done"
