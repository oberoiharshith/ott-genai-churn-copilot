# GenAI Churn Copilot (POC)

A product-shaped POC that matches:
- churn prediction on OTT-like behavioral data (traditional ML + SQL features)
- GenAI layer that generates retention messaging + a plain-English reason for churn risk
- lightweight “deployment” via Streamlit
- experimentation thinking via a simulated A/B test

## What this demonstrates
- End-to-end DS ownership: problem framing → features → model → evaluation → product integration
- Generative AI applied to customer behavior: intervention generation + stakeholder-friendly summaries
- Clear KPI story: retention lift for a targeted cohort

## Setup

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Generate synthetic data
```bash
python scripts/generate_data.py
```

### 3) Train churn model + produce artifacts
```bash
python scripts/train_model.py
```

### 4) (Optional) Simulate an A/B test for the high-risk cohort
```bash
python experiments/ab_test_simulation.py
```

### 5) Run the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

## Using a real LLM (optional)
If you want the app to call an LLM, set:
```bash
export LLM_API_KEY="your_key"
export LLM_MODEL="..."
```
If no key is set, the app uses a safe fallback output so it still demos end-to-end.

## Artifacts produced
- `artifacts/lgbm_model.txt`
- `artifacts/metrics.json`
- `artifacts/feature_importance.csv`
- `artifacts/scored_users.csv`
- `artifacts/ab_test_summary.json` (optional)

## Notes
All data is synthetic and generated locally. No copyrighted titles are used in recommendations.
