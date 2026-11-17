"""Fake scenarios, the book's way (section 1.4): dimensions, then tuples,
then one message per tuple, then quality control.

1. Tuples: every dimension value comes up about as often as the others
   (each dimension is a shuffled cycle), and rules in task.plausible drop
   combinations that make no sense.
2. Messages: the writer model turns a batch of tuples into messages, one
   per tuple, with made-up people. A second step and not one big prompt,
   so the messages vary.
3. Quality: gemini-decision-1 judges whether each message is realistic, and
   near-duplicates go. `runit review` prints a sample to read by hand.
4. Splits: train, val (to fit the temperature) and test, by a hash of the
   id, so a scenario stays in its split when more are added.
"""

from __future__ import annotations

import hashlib
import random
import re
from concurrent.futures import ThreadPoolExecutor

from . import gemini
from .paths import SCENARIOS, TUPLES, read_jsonl, write_jsonl
from .task import CHALLENGE, DIMENSIONS, DOMAIN, plausible

BATCH = 8
WORKERS = 8


def sample_tuples(n: int, seed: int = 0, start: int = 0) -> list[dict]:
    rng = random.Random(seed)
    cycles = {k: [] for k in DIMENSIONS}
    out = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        t = {}
        for k, values in DIMENSIONS.items():
            if not cycles[k]:
                cycles[k] = rng.sample(values, len(values))
            t[k] = cycles[k].pop()
        if plausible(t):
            t["id"] = f"s{start + len(out):05d}"
            t["set"] = "challenge" if t["clarity"] in CHALLENGE else "coverage"
            out.append(t)
    return out


PROMPT = """You write realistic test data for a clinic's message-triage system. Every person, name, date of birth, phone number, address and detail must be made up. Never use a real person.

Domain: {domain}.

Write exactly one message for each scenario below. Follow each scenario's fields:
- topic: what the message is about
- urgency: how urgent the real situation is (emergency, same_day, routine or admin). Do not say the urgency in the message; let the facts show it.
- sender: who writes or calls
- clarity: how clearly the writer gives the facts
- style: how the writer writes or talks
- channel: a patient-portal message, or a voicemail transcript (spoken words, no greeting line in brackets)

Make the messages different from each other in length, wording, names and details. Include made-up identifiers where real people would (a name, sometimes a date of birth or a phone number). Use real medication names where they fit.

Scenarios:
{tuples}

Answer with a JSON list, one object per scenario, in the same order: [{{"id": "...", "message": "..."}}]"""


def write_batch(batch: list[dict]) -> list[dict]:
    lines = "\n".join(
        f'- id {t["id"]}: topic={t["topic"]}; urgency={t["urgency"]}; sender={t["sender"]}; clarity={t["clarity"]}; style={t["style"]}; channel={t["channel"]}'
        for t in batch)
    try:
        out = gemini.write(PROMPT.format(domain=DOMAIN, tuples=lines))
    except Exception as e:  # one bad batch must not stop the run
        print(f"  ✗ batch {batch[0]['id']}: {e}")
        return []
    by_id = {str(o.get("id")): o.get("message", "").strip() for o in out if isinstance(o, dict)}
    return [{**t, "message": by_id[t["id"]]} for t in batch if by_id.get(t["id"])]


REALISTIC = {"realistic": {"type": "noul", "instructions": f"Is this a realistic message that a real person could send in this setting: {DOMAIN}? Answer no if it is awkward, generic, self-describing (it names its own urgency or style), or obviously written by an AI."}}


def realistic(row: dict) -> dict | None:
    """The row with Gemini's realism score; None when the call failed (the
    message is dropped, the others are kept)."""
    try:
        p = gemini.decide(row["message"], REALISTIC)["answers"]["realistic"]["noul"]
    except gemini.Blocked:
        p = 0.0
    except Exception as e:  # one failed call must not lose the batch
        print(f"  ✗ realism check of {row['id']}: {type(e).__name__}: {str(e)[:200]}")
        return None
    return {**row, "realistic": round(p, 4)}


def words(text: str) -> set[str]:
    w = re.findall(r"[a-z']+", text.lower())
    return {" ".join(w[i:i + 4]) for i in range(max(1, len(w) - 3))}


def dedupe(rows: list[dict], limit: float = 0.5) -> list[dict]:
    kept, seen = [], []
    for r in rows:
        s = words(r["message"])
        if all(len(s & o) / max(1, len(s | o)) < limit for o in seen):
            kept.append(r)
            seen.append(s)
    return kept


def split(scenario_id: str) -> str:
    h = int(hashlib.sha256(scenario_id.encode()).hexdigest(), 16) % 100
    return "test" if h < 20 else "val" if h < 30 else "train"


# 0.5 dropped real emergencies (a short, panicked overdose message scored
# 0.14), so the bar is lower: it still drops the stiff, AI-sounding ones.
def make(n: int, seed: int = 0, keep_above: float = 0.3) -> list[dict]:
    """Adds n new tuples and their messages to data/; returns all scenarios."""
    have = read_jsonl(SCENARIOS)
    old_tuples = read_jsonl(TUPLES)
    tuples = sample_tuples(n, seed=seed + len(old_tuples), start=len(old_tuples))
    write_jsonl(TUPLES, old_tuples + tuples)
    print(f"  {len(tuples)} tuples ({sum(t['set'] == 'challenge' for t in tuples)} challenge)")

    batches = [tuples[i:i + BATCH] for i in range(0, len(tuples), BATCH)]
    with ThreadPoolExecutor(WORKERS) as pool:
        written = [r for rows in pool.map(write_batch, batches) for r in rows]
    print(f"  {len(written)} messages written by {gemini.WRITER}")

    with ThreadPoolExecutor(WORKERS) as pool:
        judged = [r for r in pool.map(realistic, written) if r is not None]
    good = [r for r in judged if r["realistic"] >= keep_above]
    print(f"  {len(good)} judged realistic by {gemini.DECIDER} (dropped {len(judged) - len(good)})")

    rows = dedupe(have + good)
    new = [r for r in rows if r not in have]
    print(f"  {len(new)} kept after removing near-duplicates")
    for r in new:
        r["split"] = split(r["id"])
    write_jsonl(SCENARIOS, have + new)
    return have + new
