-- DuckDB SQL: build a feature table for churn prediction
-- Inputs (CSV):
--   data/users.csv
--   data/user_sessions.csv
--   data/watch_history.csv
--   data/subscription_status.csv
-- Output:
--   features table in DuckDB connection

CREATE OR REPLACE TABLE features AS
WITH params AS (
  SELECT
    MAX(CAST(as_of_date AS DATE)) AS as_of_date
  FROM subscription_status
),
sessions_agg AS (
  SELECT
    user_id,
    SUM(minutes) AS total_minutes_90d,
    AVG(minutes) AS avg_minutes_per_session_90d,
    COUNT(*) AS sessions_90d,
    SUM(CASE WHEN is_mobile = 1 THEN 1 ELSE 0 END) * 1.0 / NULLIF(COUNT(*), 0) AS mobile_share_90d,
    MAX(CAST(event_date AS DATE)) AS last_session_date,
    SUM(CASE WHEN CAST(event_date AS DATE) >= (SELECT as_of_date FROM params) - INTERVAL 7 DAY THEN 1 ELSE 0 END) AS sessions_7d,
    SUM(CASE WHEN CAST(event_date AS DATE) >= (SELECT as_of_date FROM params) - INTERVAL 14 DAY THEN 1 ELSE 0 END) AS sessions_14d,
    SUM(CASE WHEN CAST(event_date AS DATE) >= (SELECT as_of_date FROM params) - INTERVAL 30 DAY THEN 1 ELSE 0 END) AS sessions_30d
  FROM user_sessions
  GROUP BY 1
),
watch_agg AS (
  SELECT
    wh.user_id,
    AVG(wh.completion_rate) AS avg_completion_90d,
    COUNT(*) AS titles_started_90d,
    COUNT(DISTINCT wh.content_id) AS distinct_titles_90d
  FROM watch_history wh
  GROUP BY 1
),
genre_pref AS (
  SELECT
    wh.user_id,
    cm.genre,
    COUNT(*) AS genre_events,
    ROW_NUMBER() OVER (PARTITION BY wh.user_id ORDER BY COUNT(*) DESC) AS rn
  FROM watch_history wh
  JOIN content_metadata cm ON cm.content_id = wh.content_id
  GROUP BY 1,2
),
top_genre AS (
  SELECT user_id, genre AS top_genre_90d
  FROM genre_pref
  WHERE rn = 1
),
base AS (
  SELECT
    u.user_id,
    u.country,
    u.plan,
    CAST(u.signup_date AS DATE) AS signup_date,
    (SELECT as_of_date FROM params) AS as_of_date
  FROM users u
)
SELECT
  b.user_id,
  b.country,
  b.plan,
  DATE_DIFF('day', b.signup_date, b.as_of_date) AS tenure_days,
  COALESCE(sa.total_minutes_90d, 0) AS total_minutes_90d,
  COALESCE(sa.avg_minutes_per_session_90d, 0) AS avg_minutes_per_session_90d,
  COALESCE(sa.sessions_90d, 0) AS sessions_90d,
  COALESCE(sa.mobile_share_90d, 0) AS mobile_share_90d,
  COALESCE(wa.avg_completion_90d, 0) AS avg_completion_90d,
  COALESCE(wa.titles_started_90d, 0) AS titles_started_90d,
  COALESCE(wa.distinct_titles_90d, 0) AS distinct_titles_90d,
  COALESCE(sa.sessions_7d, 0) AS sessions_7d,
  COALESCE(sa.sessions_14d, 0) AS sessions_14d,
  COALESCE(sa.sessions_30d, 0) AS sessions_30d,
  CASE
    WHEN sa.last_session_date IS NULL THEN 9999
    ELSE DATE_DIFF('day', sa.last_session_date, b.as_of_date)
  END AS recency_days,
  COALESCE(tg.top_genre_90d, 'Unknown') AS top_genre_90d,
  ss.churn_next_7d AS label
FROM base b
LEFT JOIN sessions_agg sa USING(user_id)
LEFT JOIN watch_agg wa USING(user_id)
LEFT JOIN top_genre tg USING(user_id)
JOIN subscription_status ss USING(user_id);
