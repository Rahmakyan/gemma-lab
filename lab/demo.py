"""The data of the demo page (GET /demo): the questions, test messages with
Gemini's answers and the local student's answers, inbox messages for the
search, and test voicemails. Only fake data from data/ goes into the page."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .paths import ADAPTERS, LABELS, SCENARIOS, read_jsonl
from .task import QUESTIONS

PAGE = Path(__file__).with_name("demo.html")
CLIP_ID = re.compile(r"s\d{5}")


def options(q: dict) -> list[str]:
    if q["type"] == "noul":
        return ["yes", "no"]
    if q["type"] == "score":
        return [c.split(":")[0] for c in q["criteria"]]
    return list(q["criteria"])


def slips(n_per_class: int = 2) -> list[dict]:
    """Test messages with Gemini's answers and the current student's, two for
    each triage class (Gemini's top answer)."""
    pointer = ADAPTERS / "current"
    if not pointer.exists():
        return []
    report = ADAPTERS / pointer.read_text().strip() / "report.json"
    if not report.exists():
        return []
    preds = json.loads(report.read_text()).get("preds", {})
    labels = {r["id"]: r["target"] for r in read_jsonl(LABELS) if "target" in r}
    by_class: dict[int, list[dict]] = {}
    for m in read_jsonl(SCENARIOS):
        if m["split"] != "test" or m["id"] not in preds or m["id"] not in labels or len(m["message"]) > 520:
            continue
        t = labels[m["id"]]
        top = max(range(4), key=lambda i: t["triage"][i])
        if len(by_class.setdefault(top, [])) < n_per_class:
            by_class[top].append({"id": m["id"], "message": m["message"], "channel": m["channel"],
                                  "teacher": t, "student": {k: [round(p, 4) for p in v] for k, v in preds[m["id"]].items()}})
    return [s for i in range(n_per_class) for c in sorted(by_class) for s in by_class[c][i:i + 1]]


def data() -> dict:
    from . import voice

    test = [m for m in read_jsonl(SCENARIOS) if m["split"] == "test"]
    clips = [{"id": r["id"], "spoken": r["spoken"], "words": voice.words(r["spoken"]), "style": r["style"]}
             for r in read_jsonl(voice.SCRIPTS) if r["split"] == "test" and (voice.AUDIO / f"{r['id']}.wav").exists()]
    return {
        "questions": QUESTIONS,
        "options": {k: options(q) for k, q in QUESTIONS.items()},
        "slips": slips(),
        "inbox": [{"id": m["id"], "message": m["message"]} for m in test[:40]],
        "examples": [m["message"] for m in test[40:52]],
        "clips": clips[:8],
    }


def page() -> str:
    return PAGE.read_text().replace("/*LAB_DATA*/null", json.dumps(data(), ensure_ascii=False).replace("</", "<\\/"))


def audio(clip: str) -> Path | None:
    from . import voice

    if not CLIP_ID.fullmatch(clip):
        return None
    path = voice.AUDIO / f"{clip}.wav"
    return path if path.exists() else None
