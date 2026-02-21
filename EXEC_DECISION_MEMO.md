# Executive Decision Memo — GenAI Churn Copilot

## Question
If we can invest in only one data science initiative next quarter, which will most directly impact growth?

## North-star metric
Weekly retained users (7-day retention)

## Guardrails
- Notification opt-out rate
- Content discovery satisfaction (proxy: completion rate, repeat sessions)
- Customer support contacts per user

## Decision
Prioritize: **GenAI Churn Copilot for high-risk users**

This initiative combines:
- **churn prediction** to target the right users
- **GenAI interventions** to personalize retention actions
- **experiment design** to prove lift and de-risk rollout

## Why this lever
- Retention compounds: improving retention increases future sessions and long-term revenue.
- The model provides a scalable way to prioritize outreach/product surfaces.
- GenAI adds incremental value by tailoring the intervention and making drivers interpretable to PMs.

## Where to start
Target the **top 20% churn-risk cohort** and run an A/B test:
- Control: generic recommendations row
- Treatment: GenAI-generated intervention + tailored row
- Primary metric: 7-day retention
- Secondary: watch time, sessions, completion rate

## What “success” looks like
- Positive, statistically credible lift in 7-day retention
- No guardrail regressions
- Clear segmentation insights (which plans/countries/genres respond best)

## POC included in this repo
- Feature engineering in SQL (DuckDB)
- LightGBM churn model + metrics
- Streamlit demo showing churn risk + generated intervention text
- Optional A/B test simulation output
