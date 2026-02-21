import os
import json
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from genai.llm import generate_retention_copy

BASE = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(BASE, "..", "artifacts")


st.set_page_config(page_title="Quickplay Churn Copilot (POC)", layout="wide")
st.title("Quickplay Churn Copilot (POC)")
st.caption("Churn prediction + GenAI retention messaging. Synthetic data. Lightweight deployment demo.")

metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
scored_path = os.path.join(ARTIFACT_DIR, "scored_users.csv")
fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.csv")
ab_path = os.path.join(ARTIFACT_DIR, "ab_test_summary.json")

if not os.path.exists(scored_path):
    st.error(
    """Missing artifacts. Run:
1) python scripts/generate_data.py
2) python scripts/train_model.py
3) (optional) python experiments/ab_test_simulation.py"""
)

df = pd.read_csv(scored_path)
fi = pd.read_csv(fi_path) if os.path.exists(fi_path) else None
metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
ab = json.load(open(ab_path)) if os.path.exists(ab_path) else None

c1, c2, c3 = st.columns(3)
c1.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
c2.metric("Avg Precision", f"{metrics.get('avg_precision', 0):.3f}")
c3.metric("Rows", f"{metrics.get('n_rows', len(df))}")

left, right = st.columns([1,1])

with left:
    st.subheader("User selection")
    uid = st.selectbox("Pick a user_id", df["user_id"].head(1000).tolist())
    row = df[df["user_id"] == uid].iloc[0].to_dict()

    st.write("**Churn probability**")
    st.progress(float(row["churn_probability"]))
    st.write(f"{row['churn_probability']:.3f}")

    st.write("**Profile**")
    st.json({
        "country": row.get("country"),
        "plan": row.get("plan"),
        "tenure_days": int(row.get("tenure_days")),
        "sessions_7d": int(row.get("sessions_7d")),
        "sessions_30d": int(row.get("sessions_30d")),
        "recency_days": int(row.get("recency_days")),
        "avg_completion_90d": float(row.get("avg_completion_90d")),
        "top_genre_90d": row.get("top_genre_90d"),
    })

with right:
    st.subheader("GenAI output")
    drivers = "• recency_days\n• sessions_7d\n• avg_completion_90d\n• sessions_30d\n• total_minutes_90d"
    profile = {
        "country": row.get("country"),
        "plan": row.get("plan"),
        "tenure_days": int(row.get("tenure_days")),
        "sessions_7d": int(row.get("sessions_7d")),
        "sessions_30d": int(row.get("sessions_30d")),
        "recency_days": int(row.get("recency_days")),
        "avg_completion_90d": float(row.get("avg_completion_90d")),
        "top_genre_90d": row.get("top_genre_90d"),
    }
    if st.button("Generate retention copy"):
        out = generate_retention_copy(profile, drivers)
        st.code(out)

    st.write("Tip: set `OPENAI_API_KEY` to use the real model. Without it, this app uses a safe fallback template.")

st.divider()
st.subheader("A/B test simulation (optional)")
if ab:
    st.json(ab)
else:
    st.info("Run `python experiments/ab_test_simulation.py` to generate artifacts/ab_test_summary.json")

st.divider()
st.subheader("Feature importance (top 15)")
if fi is not None:
    st.dataframe(fi.head(15), use_container_width=True)
