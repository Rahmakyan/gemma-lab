"""Answers in the shape of the Gemini Decisions API, from option
probabilities, so a local student is a drop-in for gemini-decision-1."""

from __future__ import annotations

from .students.common import options


def answer(q: dict, probs: list[float]) -> dict:
    if len(probs) != (2 if q["type"] == "noul" else len(options(q))):
        raise ValueError(f"{len(probs)} probabilities for a question with {len(options(q))} options")
    if q["type"] == "noul":
        return {"type": "noul", "noul": round(probs[0], 4), "confidence": round(max(probs), 4)}
    names = options(q)
    best = max(range(len(probs)), key=probs.__getitem__)
    out = {"type": q["type"], "confidence": round(probs[best], 4), "probabilities": {n: round(p, 4) for n, p in zip(names, probs)}}
    if q["type"] == "choice":
        out["choice"] = names[best]
    else:
        out["score"] = round(sum(i * p for i, p in enumerate(probs)), 4)
        out["legend"] = {str(i): c for i, c in enumerate(q["criteria"])}
    return out


def check(questions: dict) -> str | None:
    """Why a set of questions is turned down, or None."""
    if not 1 <= len(questions) <= 1000:
        return "it takes 1 to 1000 questions"
    for name, q in questions.items():
        kind = q.get("type")
        if kind not in ("noul", "choice", "score"):
            return f"{name}: type must be noul, choice or score"
        if not q.get("instructions"):
            return f"{name}: instructions are missing"
        if kind == "choice" and not (isinstance(q.get("criteria"), dict) and 2 <= len(q["criteria"]) <= 26):
            return f"{name}: a choice takes criteria, a map of 2 to 26 options (26 letters for the local Gemma)"
        if kind == "score" and not (isinstance(q.get("criteria"), list) and 2 <= len(q["criteria"]) <= 26):
            return f"{name}: a score takes criteria, a list of 2 to 26 levels"
    return None
