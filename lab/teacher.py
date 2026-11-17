"""The teacher labels: gemini-decision-1 answers every question in
task.QUESTIONS for every scenario. Its full probabilities are kept, so
Gemma learns the whole distribution (soft labels), not only the winner.

A label is {id, target: {question: [p per option]}} with the options in
the order of the question's criteria ([yes, no] for a noul).
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import gemini
from .paths import LABELS, SCENARIOS, read_jsonl
from .task import QUESTIONS

WORKERS = 8


def target(q: dict, answer: dict) -> list[float]:
    """The teacher's probabilities in option order (a score's levels are
    keyed "0", "1", …)."""
    if q["type"] == "noul":
        return [answer["noul"], 1 - answer["noul"]]
    probs = answer["probabilities"]
    keys = [str(i) for i in range(len(q["criteria"]))] if q["type"] == "score" else list(q["criteria"])
    total = sum(probs.values())
    return [probs[k] / total for k in keys]


def label(row: dict) -> dict | None:
    """The label of one scenario; None when the call failed (it is tried
    again on the next run)."""
    try:
        out = gemini.decide(row["message"], QUESTIONS)
        answers = out["answers"]
        return {"id": row["id"], "target": {name: [round(p, 5) for p in target(q, answers[name])] for name, q in QUESTIONS.items()},
                "tokens": out.get("usage", {}).get("total_input_tokens")}
    except gemini.Blocked as e:
        return {"id": row["id"], "blocked": str(e)}
    except Exception as e:  # one failed call must not lose the others
        print(f"  ✗ {row['id']}: {type(e).__name__}: {str(e)[:200]}")
        return None


def run() -> list[dict]:
    """Labels the scenarios that have no label yet. Each label is added to
    data/labels.jsonl as soon as it comes back, so a stop loses nothing that
    was paid for; the file is not touched when nothing is new."""
    have = read_jsonl(LABELS)
    done = {r["id"] for r in have}
    todo = [r for r in read_jsonl(SCENARIOS) if r["id"] not in done]
    new, failed = [], 0
    if todo:
        LABELS.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(WORKERS) as pool, LABELS.open("a") as f:
            for fut in as_completed([pool.submit(label, r) for r in todo]):
                row = fut.result()
                if row is None:
                    failed += 1
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                new.append(row)
    blocked = sum("blocked" in r for r in new)
    tokens = sum(r.get("tokens") or 0 for r in new)
    print(f"  {len(new)} labelled by {gemini.DECIDER} ({blocked} blocked, {failed} failed, {tokens} input tokens, about ${tokens * 0.10 / 1e6:.4f})")
    if failed:
        print(f"  ✗ {failed} calls failed: run `runit label` again to retry them")
    return have + new
