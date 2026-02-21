import os
import json
import duckdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "..", "data")
SQL_PATH = os.path.join(BASE, "..", "sql", "feature_engineering.sql")
ARTIFACT_DIR = os.path.join(BASE, "..", "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def load_csv(con, name):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run: python scripts/generate_data.py")
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_csv_auto('{path}')")

def main():
    con = duckdb.connect(database=":memory:")
    for t in ["users","user_sessions","watch_history","content_metadata","subscription_status"]:
        load_csv(con, t)

    con.execute(open(SQL_PATH, "r", encoding="utf-8").read())
    df = con.execute("SELECT * FROM features").df()

    # Basic preprocessing
    y = df["label"].astype(int)
    X = df.drop(columns=["label"])

    cat_cols = ["country","plan","top_genre_90d"]
    for c in cat_cols:
        X[c] = X[c].astype("category")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X_train, y_train, categorical_feature=cat_cols)

    p_test = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, p_test)
    ap = average_precision_score(y_test, p_test)

    # Feature importance
    fi = pd.DataFrame({"feature": model.feature_name_, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.csv")
    fi.to_csv(fi_path, index=False)

    # Save a small scoring table for the app
    scored = X.copy()
    scored["churn_probability"] = model.predict_proba(X)[:,1]
    scored["label"] = y.values
    scored = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    scored_path = os.path.join(ARTIFACT_DIR, "scored_users.csv")
    scored.to_csv(scored_path, index=False)

    # Persist model via lightgbm's built-in booster serialization
    booster_path = os.path.join(ARTIFACT_DIR, "lgbm_model.txt")
    model.booster_.save_model(booster_path)

    metrics = {"roc_auc": float(auc), "avg_precision": float(ap), "n_rows": int(df.shape[0])}
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved artifacts to:", ARTIFACT_DIR)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
