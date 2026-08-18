#!/usr/bin/env bash
# End-to-end setup and notebook execution for PS4 submission.
set -euo pipefail
cd "$(dirname "$0")"

echo "Installing dependencies..."
pip3 install -r requirements.txt -q
python3 -m spacy download en_core_web_sm

if [ "$(find data/documents -name '*.txt' 2>/dev/null | wc -l | tr -d ' ')" -lt 30 ]; then
  echo "Collecting Wikipedia documents..."
  python3 -c "from pathlib import Path; from src.data_collection import collect_documents; collect_documents(Path('data/documents'))"
fi

echo "Building notebook..."
python3 scripts/build_notebook.py

echo "Executing notebook (this may take several minutes)..."
python3 -m jupyter nbconvert \
  --to notebook \
  --execute PS4_Knowledge_Grounded_RAG.ipynb \
  --output PS4_Knowledge_Grounded_RAG.ipynb \
  --ExecutePreprocessor.timeout=900

echo "Done. Open PS4_Knowledge_Grounded_RAG.ipynb and update Section 1 member details."
