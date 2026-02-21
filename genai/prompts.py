SYSTEM = """You are a data science copilot for an OTT streaming product team.
You write concise, factual summaries. No hype. No promises. Keep it practical.
"""

USER_TEMPLATE = """Given this user profile and churn drivers, produce:
1) a 1-sentence plain-English reason for churn risk
2) a 1-sentence product intervention (retention message)
3) 3 recommended genres/titles themes (no specific copyrighted titles)

User profile:
- Country: {country}
- Plan: {plan}
- Tenure (days): {tenure_days}
- Sessions last 7d: {sessions_7d}
- Sessions last 30d: {sessions_30d}
- Recency (days since last session): {recency_days}
- Avg completion: {avg_completion_90d:.2f}
- Top genre: {top_genre_90d}

Top churn drivers (from model importance, approximate):
{drivers}

Output format (exact):
REASON: ...
INTERVENTION: ...
RECS: ...
"""
