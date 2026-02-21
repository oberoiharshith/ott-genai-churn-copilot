import os
import json
import numpy as np
import pandas as pd

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")

def main():
    path = os.path.join(ARTIFACT_DIR, "scored_users.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Missing artifacts/scored_users.csv. Run: python scripts/train_model.py")

    df = pd.read_csv(path)
    # Target top 20% risk
    df["target"] = (df["churn_probability"] >= df["churn_probability"].quantile(0.80)).astype(int)
    cohort = df[df["target"] == 1].copy()

    rng = np.random.default_rng(42)
    cohort["variant"] = rng.choice(["control","treatment"], size=len(cohort))
    # Baseline retention probability depends on churn prob
    base_retention = 1.0 - cohort["churn_probability"]
    # Treatment adds small absolute lift (simulate 2-6 pts) with diminishing returns
    lift = 0.03 + 0.05 * (cohort["churn_probability"])
    p_ret = np.where(cohort["variant"]=="treatment", np.clip(base_retention + lift, 0, 1), base_retention)
    cohort["retained"] = rng.binomial(1, p_ret)

    summary = cohort.groupby("variant")["retained"].agg(["mean","count"]).reset_index()
    diff = float(summary.loc[summary.variant=="treatment","mean"].iloc[0] - summary.loc[summary.variant=="control","mean"].iloc[0])

    out = {
        "cohort": "top_20pct_churn_risk",
        "n_users": int(len(cohort)),
        "control_retention": float(summary.loc[summary.variant=="control","mean"].iloc[0]),
        "treatment_retention": float(summary.loc[summary.variant=="treatment","mean"].iloc[0]),
        "absolute_lift": diff,
    }
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACT_DIR, "ab_test_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Wrote artifacts/ab_test_summary.json")
    print(out)

if __name__ == "__main__":
    main()
