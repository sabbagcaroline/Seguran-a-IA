#!/usr/bin/env bash
# Reproduz TODOS os resultados e figuras (≈1 min em CPU comum, sem GPU, sem internet).
set -euo pipefail
cd "$(dirname "$0")/src"
export PYTHONWARNINGS=ignore
python3 evasion.py
python3 evasion_tentativa1.py
python3 poisoning.py
python3 rag_injection.py
python3 supply_chain.py
echo "OK: resultados em results/*.json e figuras em figures/*.png"
