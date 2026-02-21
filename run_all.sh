#!/usr/bin/env bash
set -euo pipefail

python scripts/generate_data.py
python scripts/train_model.py
python experiments/ab_test_simulation.py || true
echo "Starting app..."
streamlit run app/streamlit_app.py
