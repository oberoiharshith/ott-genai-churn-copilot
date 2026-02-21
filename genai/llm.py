import os
from openai import OpenAI as LLMClient
from .prompts import SYSTEM, USER_TEMPLATE


def _fallback(profile, drivers_text):
    reason = "Recent inactivity and lower completion suggest the user is disengaging."
    intervention = "Send a short personalized note plus a tailored row of content to restart a habit."
    recs = f"{profile.get('top_genre_90d', 'Drama')}, short episodes, trending originals"
    return f"REASON: {reason}\nINTERVENTION: {intervention}\nRECS: {recs}"


def generate_retention_copy(profile, drivers_text):
    # Safe: if env var is missing, os.getenv returns None
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    if not api_key:
        return _fallback(profile, drivers_text)

    model = (os.getenv("LLM_MODEL") or "gpt-4.1-mini").strip()

    client = LLMClient(api_key=api_key)
    prompt = USER_TEMPLATE.format(drivers=drivers_text, **profile)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return (resp.choices[0].message.content or "").strip()