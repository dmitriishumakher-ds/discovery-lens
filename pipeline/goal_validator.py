"""
goal_validator.py — lightweight LLM gate on the goal statement.

Scores the PM's goal on three dimensions before the pipeline starts:
  1. measurable_metric — names a metric, percentage, count, or concrete outcome
  2. user_segment      — references a user type, persona, team, or segment
  3. timeframe         — includes a time period, deadline, or scope boundary

Passes if ≥ 2 of 3 dimensions are met.

On failure, returns coaching feedback and an LLM-generated rewrite suggestion.
The PM can accept the rewrite, edit it manually, or override and proceed anyway.

Token budget: ~200 in, ~200 out. Safe on Groq free tier.
On any API or parse failure, the validator passes silently — never blocks the PM.
"""

from __future__ import annotations

import json
import os

from groq import Groq

_MODEL = "llama-3.1-8b-instant"

_SYSTEM_PROMPT = """\
You are a product management coach. A PM has written a product goal statement.
Score it on three dimensions and suggest an improved rewrite if needed.

Return ONLY valid JSON — no preamble, no markdown fences:
{
  "measurable_metric": true or false,
  "user_segment": true or false,
  "timeframe": true or false,
  "feedback": "one sentence: what is missing, or confirm it is strong",
  "rewrite": "improved one-sentence goal, or null if the goal already passes all three"
}

Scoring rules:
- measurable_metric: true if the goal names a metric, percentage, count, or concrete outcome
- user_segment: true if the goal references a user type, persona, team, or segment
- timeframe: true if the goal includes a time period, deadline, or scope boundary\
"""

_PASS_SILENT = {
    "passed": True,
    "score": 3,
    "measurable_metric": True,
    "user_segment": True,
    "timeframe": True,
    "feedback": "",
    "rewrite": None,
}


def validate_goal(goal: str) -> dict:
    """
    Validate a PM goal statement with a lightweight Groq LLM call.

    Parameters
    ----------
    goal : str
        The PM's goal statement.

    Returns
    -------
    dict with keys:
        passed           : bool        — True if ≥ 2 of 3 dimensions met
        score            : int         — number of dimensions met (0–3)
        measurable_metric: bool
        user_segment     : bool
        timeframe        : bool
        feedback         : str         — coaching note shown to the PM
        rewrite          : str | None  — suggested rewrite, or None if not needed
    """
    try:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f'Goal statement: "{goal}"'},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
    except Exception:
        return _PASS_SILENT

    measurable = bool(parsed.get("measurable_metric"))
    segment = bool(parsed.get("user_segment"))
    timeframe = bool(parsed.get("timeframe"))
    score = sum([measurable, segment, timeframe])

    return {
        "passed": score >= 2,
        "score": score,
        "measurable_metric": measurable,
        "user_segment": segment,
        "timeframe": timeframe,
        "feedback": parsed.get("feedback", ""),
        "rewrite": parsed.get("rewrite") if score < 3 else None,
    }
