"""Teach one student and judge it against the teacher on the test rows.

Runs anywhere (a Modal GPU by default, see modal_app.py, or this machine).
Every number is agreement with Gemini on scenarios the student never saw:

- agreement: the student's top option is Gemini's top option
- kl: how far the student's distribution is from Gemini's (0 is the same)
- auc (noul): how well P(yes) ranks Gemini's yes above its no
- mae: |student P(yes) - Gemini P(yes)| for a noul, |student level -
  Gemini level| (expected levels) for a score

Each is given for all test rows, the coverage set and the challenge set.
The test numbers are for the report only. Whether a student is kept (and
which one the decider serves) is decided on the val rows, so the test set
does not pick the winner.
The student is compared with its baseline: plain Gemma zero-shot for lora
(the decision model with no teaching), the prior (Gemini's average answer)
for embed, which cannot answer without teaching.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch


def auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def judge(preds: list[dict], rows: list[dict], questions: dict) -> dict:
    out = {}
    for group in ("all", "coverage", "challenge"):
        idx = [i for i, r in enumerate(rows) if group == "all" or r["set"] == group]
        if not idx:
            continue
        per = {}
        for name, q in questions.items():
            p = torch.tensor([preds[i][name] for i in idx]).clamp_min(1e-6)
            t = torch.tensor([rows[i]["target"][name] for i in idx]).clamp_min(1e-6)
            m = {"agreement": (p.argmax(-1) == t.argmax(-1)).float().mean().item(),
                 "kl": (t * (t.log() - p.log())).sum(-1).mean().item()}
            if q["type"] == "noul":
                m["auc"] = auc(p[:, 0].tolist(), (t[:, 0] >= 0.5).int().tolist())
                m["mae"] = (p[:, 0] - t[:, 0]).abs().mean().item()
            if q["type"] == "score":
                levels = torch.arange(p.shape[1], dtype=torch.float32)
                m["mae"] = ((p * levels).sum(-1) - (t * levels).sum(-1)).abs().mean().item()
            per[name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
        out[group] = {"n": len(idx), "questions": per,
                      "agreement": round(sum(m["agreement"] for m in per.values()) / len(per), 4),
                      "kl": round(sum(m["kl"] for m in per.values()) / len(per), 4)}
    return out


def prior(train: list[dict], test: list[dict], questions: dict) -> list[dict]:
    mean = {name: torch.tensor([r["target"][name] for r in train]).mean(0).tolist() for name in questions}
    return [dict(mean) for _ in test]


def better(a: dict, b: dict) -> bool:
    return a["agreement"] > b["agreement"] + 1e-9 or (math.isclose(a["agreement"], b["agreement"]) and a["kl"] < b["kl"])


def train_and_judge(kind: str, rows: list[dict], questions: dict, cfg: dict, out: Path, log=print) -> dict:
    from .students import make

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    texts = [r["message"] for r in test]
    vtexts = [r["message"] for r in val]
    log(f"  {kind}: {len(train)} train, {len(val)} val, {len(test)} test messages")
    started = time.time()
    student = make(kind, cfg.get("model"))
    report = {"kind": kind, "model": student.name, "device": student.device, "config": cfg,
              "split": {"train": len(train), "val": len(val), "test": len(test)},
              "prior": judge(prior(train, test, questions), test, questions)}
    if kind == "lora":
        report["baseline_name"] = "plain Gemma (zero-shot decision model)"
        report["baseline"] = judge(student.predict(texts, questions), test, questions)
        val_baseline = judge(student.predict(vtexts, questions), val, questions)
    else:
        report["baseline_name"] = "prior (Gemini's average answer)"
        report["baseline"] = report["prior"]
        val_baseline = judge(prior(train, val, questions), val, questions)
    log(f"  {report['baseline_name']}: agreement {report['baseline']['all']['agreement']:.3f}")

    t = time.time()
    student.fit(train, val, questions, log=log, **{k: v for k, v in cfg.items() if k != "model"})
    report["train_seconds"] = round(time.time() - t, 1)
    t = time.time()
    preds = student.predict(texts, questions)
    report["predict_ms_per_message"] = round((time.time() - t) * 1000 / max(1, len(texts)), 1)
    report["student"] = judge(preds, test, questions)
    # Every test answer, so a later report can be judged on the same messages.
    report["preds"] = {r["id"]: p for r, p in zip(test, preds)}
    report["val"] = {"ids": sorted(r["id"] for r in val), "baseline": val_baseline["all"],
                     "student": judge(student.predict(vtexts, questions), val, questions)["all"]}
    report["kept"] = better(report["val"]["student"], report["val"]["baseline"])
    report["size"] = student.size()
    report["seconds"] = round(time.time() - started, 1)
    student.save(out)
    log(f"  {kind}: agreement {report['student']['all']['agreement']:.3f} (baseline {report['baseline']['all']['agreement']:.3f}), "
        f"trained in {report['train_seconds']} s, {'kept' if report['kept'] else 'not better than the baseline'}")
    return report
