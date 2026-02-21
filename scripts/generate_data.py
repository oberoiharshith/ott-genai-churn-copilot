import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

n_users = 5000
n_titles = 800
start = datetime(2025, 11, 1)
days = 90

# content metadata
genres = ["Drama", "Comedy", "Crime", "Sci-Fi", "Action", "Romance", "Documentary", "Kids"]
content = pd.DataFrame({
    "content_id": np.arange(1, n_titles + 1),
    "genre": rng.choice(genres, size=n_titles, replace=True),
    "duration_min": rng.integers(18, 65, size=n_titles),
    "is_original": rng.choice([0, 1], size=n_titles, p=[0.7, 0.3]),
    "release_year": rng.integers(1995, 2026, size=n_titles)
})
content.to_csv(os.path.join(OUT_DIR, "content_metadata.csv"), index=False)

# users
countries = ["CA", "US", "UK", "AU", "IN"]
users = pd.DataFrame({
    "user_id": np.arange(1, n_users + 1),
    "country": rng.choice(countries, size=n_users, p=[0.35, 0.35, 0.1, 0.1, 0.1]),
    "plan": rng.choice(["basic", "standard", "premium"], size=n_users, p=[0.35, 0.45, 0.2]),
    "signup_date": [start - timedelta(days=int(x)) for x in rng.integers(0, 365, size=n_users)],
})
users.to_csv(os.path.join(OUT_DIR, "users.csv"), index=False)

# sessions
# A user has baseline engagement; churn is more likely if engagement decays.
baseline = rng.lognormal(mean=0.2, sigma=0.8, size=n_users)  # avg sessions/week-ish
decay = rng.uniform(0.0, 0.25, size=n_users)                 # engagement decay per week
price_sens = rng.normal(0.0, 1.0, size=n_users)

session_rows = []
watch_rows = []

for uid in range(1, n_users + 1):
    base_rate = baseline[uid-1]
    d = decay[uid-1]
    for day in range(days):
        date = start + timedelta(days=day)
        week = day / 7.0
        lam = max(0.01, base_rate * (1.0 - d * week))
        n_sess = rng.poisson(lam=lam)
        for _ in range(n_sess):
            minutes = int(rng.integers(5, 120))
            session_rows.append((uid, date.date().isoformat(), minutes, int(rng.choice([0,1], p=[0.25,0.75])))) # 1 = mobile
            # each session has 0-3 watches
            n_watches = rng.integers(0, 4)
            for __ in range(n_watches):
                cid = int(rng.integers(1, n_titles+1))
                duration = int(content.loc[content.content_id == cid, "duration_min"].iloc[0])
                # completion depends on minutes watched
                watched = int(rng.integers(1, duration+1))
                completion = min(1.0, watched / max(1, duration))
                watch_rows.append((uid, date.date().isoformat(), cid, watched, completion))

sessions = pd.DataFrame(session_rows, columns=["user_id", "event_date", "minutes", "is_mobile"])
watches = pd.DataFrame(watch_rows, columns=["user_id", "event_date", "content_id", "watched_min", "completion_rate"])

# subscription + churn label
# Create churn label based on last 7 days inactivity + engagement decay + plan sensitivity
last_day = start + timedelta(days=days-1)
cutoff = (last_day - timedelta(days=7)).date().isoformat()

user_last_event = sessions.groupby("user_id")["event_date"].max().reindex(users.user_id).fillna("1900-01-01")
inactive_7d = (user_last_event < cutoff).astype(int).values

plan_map = {"basic": 0.25, "standard": 0.15, "premium": 0.1}
plan_risk = users["plan"].map(plan_map).values

# churn probability
logit = -1.2 + 1.6*inactive_7d + 0.9*decay + 0.2*price_sens + plan_risk
p = 1/(1+np.exp(-logit))
churn = rng.binomial(1, np.clip(p, 0.01, 0.95))

subs = users[["user_id","plan"]].copy()
subs["as_of_date"] = last_day.date().isoformat()
subs["churn_next_7d"] = churn
subs.to_csv(os.path.join(OUT_DIR, "subscription_status.csv"), index=False)

sessions.to_csv(os.path.join(OUT_DIR, "user_sessions.csv"), index=False)
watches.to_csv(os.path.join(OUT_DIR, "watch_history.csv"), index=False)

print("Wrote:")
for f in ["content_metadata.csv","users.csv","subscription_status.csv","user_sessions.csv","watch_history.csv"]:
    print(" -", os.path.join(OUT_DIR, f))
