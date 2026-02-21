import os
import json
import sys
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure repo root is on the Python path so `genai/` can be imported when Streamlit runs from /app
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from genai.llm import generate_retention_copy

BASE = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(BASE, "..", "artifacts")

st.set_page_config(page_title="OTT Churn Copilot", layout="wide")
st.title("OTT Churn Copilot")
st.caption("Churn prediction + optional GenAI retention messaging. Synthetic data. Streamlit demo.")

metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
scored_path = os.path.join(ARTIFACT_DIR, "scored_users.csv")
fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.csv")
ab_path = os.path.join(ARTIFACT_DIR, "ab_test_summary.json")


def run_cmd(cmd: list[str]):
    """Run a command from the repo root (works on Streamlit Cloud)."""
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


# Bootstrap artifacts on Streamlit Cloud (or any clean environment)
if not os.path.exists(scored_path):
    st.warning("Artifacts not found. Build them once to initialize the app.")

    with st.expander("Build steps (manual)"):
        st.code(
            "\n".join(
                [
                    "python scripts/generate_data.py",
                    "python scripts/train_model.py",
                    "(optional) python experiments/ab_test_simulation.py",
                ]
            )
        )

    if st.button("Build artifacts now (1-time)"):
        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        with st.spinner("Generating synthetic data..."):
            rc, out, err = run_cmd(["python", "scripts/generate_data.py"])
            if rc != 0:
                st.error("Data generation failed.")
                st.code((out or "") + "\n" + (err or ""))
                st.stop()

        with st.spinner("Training churn model..."):
            rc, out, err = run_cmd(["python", "scripts/train_model.py"])
            if rc != 0:
                st.error("Model training failed.")
                st.code((out or "") + "\n" + (err or ""))
                st.stop()

        with st.spinner("Running A/B simulation (optional)..."):
            rc, out, err = run_cmd(["python", "experiments/ab_test_simulation.py"])
            if rc != 0:
                st.warning("A/B simulation failed (optional).")
                st.code((out or "") + "\n" + (err or ""))

        st.success("Artifacts built. Reloading app...")
        st.rerun()

    st.stop()  # CRITICAL: prevents pandas read_csv crash before artifacts exist


# Load artifacts
df = pd.read_csv(scored_path)
fi = pd.read_csv(fi_path) if os.path.exists(fi_path) else None
metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
ab = json.load(open(ab_path)) if os.path.exists(ab_path) else None

# Header metrics
c1, c2, c3 = st.columns(3)
c1.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")
c2.metric("Avg Precision", f"{metrics.get('avg_precision', 0):.3f}")
c3.metric("Rows", f"{metrics.get('n_rows', len(df))}")

left, right = st.columns([1, 1])

with left:
    st.subheader("User selection")
    uid = st.selectbox("Pick a user_id", df["user_id"].head(1000).tolist())
    row = df[df["user_id"] == uid].iloc[0].to_dict()

    st.write("**Churn probability**")
    st.progress(float(row["churn_probability"]))
    st.write(f"{row['churn_probability']:.3f}")

    st.write("**Profile**")
    st.json(
        {
            "country": row.get("country"),
            "plan": row.get("plan"),
            "tenure_days": int(row.get("tenure_days")),
            "sessions_7d": int(row.get("sessions_7d")),
            "sessions_30d": int(row.get("sessions_30d")),
            "recency_days": int(row.get("recency_days")),
            "avg_completion_90d": float(row.get("avg_completion_90d")),
            "top_genre_90d": row.get("top_genre_90d"),
        }
    )

with right:
    st.subheader("GenAI output")

    drivers = "\n".join(
        [
            "• recency_days",
            "• sessions_7d",
            "• avg_completion_90d",
            "• sessions_30d",
            "• total_minutes_90d",
        ]
    )

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

    st.caption(
        "Tip: set OPENAI_API_KEY to use a real model. If not set, the app uses a safe fallback output."
    )

st.divider()
st.subheader("A/B test simulation (optional)")
if ab:
    st.json(ab)
else:
    st.info("No A/B summary found yet. Run `python experiments/ab_test_simulation.py` to generate it.")

st.divider()
st.subheader("Feature importance (top 15)")
if fi is not None:
    st.dataframe(fi.head(15), use_container_width=True)
else:
    st.info("No feature importance file found.")
