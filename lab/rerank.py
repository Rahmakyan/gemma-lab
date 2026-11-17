"""The reranker: put the inbox messages that match a staff member's search
first. The same distillation as the decisions, with other labels.

1. Queries: Gemini writes fake searches that a clinic's staff would type
   ("refill requests for blood pressure pills", "a parent worried about a
   rash"). Each query is in train, val or test by a hash of its text.
2. Teacher: gemini-decision-1 grades every message against every query with
   one noul question per query (50 per call), so P(match) is a graded
   relevance, not a 0/1 mark.
3. Students (same interface: fit(train, val), score(query, texts)):
   - cross: Gemma 3 1B + LoRA, a cross-encoder. It reads the query and the
     message together and P(yes) of "does this message match?" is the
     score: the decision model with the query in the question.
   - bi: EmbeddingGemma + LoRA, a bi-encoder (a retriever). The query and
     the message become vectors on their own; the score is
     sigmoid(a * cosine + b). Message vectors can be computed once, so a
     search over many messages is fast.
4. Judge: test queries over the test messages (neither seen in training).
   Val (best epoch, kept or not) is the val queries over the val and train
   messages.
   NDCG@10 with Gemini's P(match) as the gain, Spearman rank correlation,
   recall@10 of Gemini's matches, and the MAE of P(match).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from . import gemini
from .paths import DATA, SCENARIOS, read_jsonl, write_jsonl
from .task import DIMENSIONS, DOMAIN

QUERIES = DATA / "queries.jsonl"
RELEVANCE = DATA / "relevance.jsonl"
PER_CALL = 50
WORKERS = 8


def question(query: str) -> dict:
    """The noul question that grades one (query, message) pair. The cross
    student is taught and asked with exactly these words."""
    return {"type": "noul", "instructions": f'A clinic staff member searches the inbox for: "{query}". Does this message match the search?'}


def split(text: str) -> str:
    h = int(hashlib.sha256(text.lower().encode()).hexdigest(), 16) % 100
    return "test" if h < 20 else "val" if h < 30 else "train"


# --- data -----------------------------------------------------------------------

PROMPT = """You write test data for the search box of a clinic's message inbox. Domain: {domain}.

The inbox holds patient-portal messages and voicemail transcripts. Topics in it include: {topics}. Senders: {senders}.

Write {n} different searches that clinic staff (nurses, doctors, front desk, billing, pharmacy) would type to find messages. Mix:
- short keyword searches ("statin refill", "billing dispute")
- plain-language needs ("parents worried about a child's fever", "someone who ran out of insulin")
- searches with a condition ("refill requests that are not for controlled drugs", "messages from caregivers, not patients")
- searches about urgency, sender, feelings or channel ("angry voicemails about wait times", "anything that sounds like an emergency")
- specific ones that match few messages, and broad ones that match many
Do not repeat a search in other words. Do not number them.

Answer with a JSON list of strings."""


GROUNDED = """You write test data for the search box of a clinic's message inbox. Domain: {domain}.

Here are messages from the inbox:
{messages}

For each message, write 2 different searches that clinic staff would type to find that message and others like it. Make one short (keywords) and one in plain language or with a condition. Do not copy names, dates or numbers from the messages. Do not mention the message number.

Answer with a JSON list of strings."""


def grounded_queries(n: int, have: list[dict], seed: int = 0) -> list[dict]:
    """Searches written from train messages, so that each one has matches."""
    rng = random.Random(seed + len(have))
    train = [m for m in read_jsonl(SCENARIOS) if m["split"] == "train"]
    seen = {q["query"].lower() for q in have}
    out = []
    while len(out) < n:
        batch = rng.sample(train, 10)
        texts = gemini.write(GROUNDED.format(domain=DOMAIN, messages="\n".join(f"{i + 1}. {m['message'][:600]}" for i, m in enumerate(batch))))
        for t in texts:
            t = str(t).strip()
            if t and t.lower() not in seen and len(out) < n:
                seen.add(t.lower())
                out.append({"id": f"q{len(have) + len(out):03d}", "query": t, "split": split(t), "from": "messages"})
    return out


def make_queries(n: int = 100, grounded: int = 100) -> list[dict]:
    """n searches from the topics alone, then `grounded` searches written
    from train messages (the first kind often matches no message)."""
    have = read_jsonl(QUERIES)
    if have:
        if grounded and not any(q.get("from") == "messages" for q in have):
            new = grounded_queries(grounded, have)
            write_jsonl(QUERIES, have + new)
            print(f"  {len(new)} queries written from messages (train {sum(q['split'] == 'train' for q in new)}, "
                  f"val {sum(q['split'] == 'val' for q in new)}, test {sum(q['split'] == 'test' for q in new)})")
            return have + new
        print(f"  {len(have)} queries already written ({QUERIES.name})")
        return have
    out, seen = [], set()
    for i in range((n + 49) // 50):
        texts = gemini.write(PROMPT.format(domain=DOMAIN, topics=", ".join(DIMENSIONS["topic"]), senders=", ".join(DIMENSIONS["sender"]), n=50)
                             + ("" if not out else "\n\nDo not repeat these: " + "; ".join(q["query"] for q in out)))
        for t in texts:
            t = str(t).strip()
            if t and t.lower() not in seen and len(out) < n:
                seen.add(t.lower())
                out.append({"id": f"q{len(out):03d}", "query": t, "split": split(t)})
    out += grounded_queries(grounded, out) if grounded else []
    write_jsonl(QUERIES, out)
    print(f"  {len(out)} queries by {gemini.WRITER} (train {sum(q['split'] == 'train' for q in out)}, "
          f"val {sum(q['split'] == 'val' for q in out)}, test {sum(q['split'] == 'test' for q in out)})")
    return out


def grade(message: dict, queries: list[dict]) -> dict | None:
    """Gemini's P(match) of one message for every query; None if a call failed."""
    scores = {}
    try:
        for i in range(0, len(queries), PER_CALL):
            chunk = queries[i:i + PER_CALL]
            answers = gemini.decide(message["message"], {q["id"]: question(q["query"]) for q in chunk})["answers"]
            scores.update({q["id"]: round(answers[q["id"]]["noul"], 4) for q in chunk})
    except gemini.Blocked as e:
        return {"id": message["id"], "blocked": str(e)[:200]}
    except Exception as e:
        print(f"  ✗ {message['id']}: {type(e).__name__}: {str(e)[:200]}")
        return None
    return {"id": message["id"], "scores": scores}


def label() -> list[dict]:
    """Grades every message that has no grades yet (appended as they come)."""
    queries = read_jsonl(QUERIES)
    if not queries:
        raise SystemExit("no queries yet: run `runit rerank queries` first")
    have = {r["id"]: r for r in read_jsonl(RELEVANCE)}
    # Only the pairs without a grade: a message graded before gets the new queries.
    todo = []
    for m in read_jsonl(SCENARIOS):
        old = have.get(m["id"], {})
        if "blocked" in old:
            continue
        missing = [q for q in queries if q["id"] not in old.get("scores", {})]
        if missing:
            todo.append((m, missing, old.get("scores", {})))
    if not todo:
        print(f"  {len(have)} messages already graded against {len(queries)} queries")
        return list(have.values())
    new, failed = [], 0
    with ThreadPoolExecutor(WORKERS) as pool, RELEVANCE.open("a") as f:
        futures = {pool.submit(grade, m, missing): old for m, missing, old in todo}
        for fut in as_completed(futures):
            row = fut.result()
            if row is None:
                failed += 1
                continue
            if "scores" in row:
                row["scores"] = {**futures[fut], **row["scores"]}
            f.write(json.dumps(row) + "\n")
            f.flush()
            new.append(row)
    # A message graded again (new queries) has two rows: the last one wins.
    rows = {**have, **{r["id"]: r for r in new}}
    write_jsonl(RELEVANCE, list(rows.values()))
    print(f"  {len(new)} messages graded by {gemini.DECIDER} against {len(queries)} queries ({failed} failed)")
    return list(rows.values())


def dataset() -> dict:
    """{split: {"queries": [...], "messages": [...], "scores": {(qid, mid): p}}}:
    the queries and messages of one split, and Gemini's grades between them."""
    queries = read_jsonl(QUERIES)
    grades = {r["id"]: r["scores"] for r in read_jsonl(RELEVANCE) if "scores" in r}
    messages = [m for m in read_jsonl(SCENARIOS) if m["id"] in grades]
    if not queries or not messages:
        raise SystemExit("no graded queries yet: run `runit rerank data` first")
    out = {}
    for s in ("train", "val", "test"):
        qs = [q for q in queries if q["split"] == s]
        # Val queries also search the train messages: 68 val messages alone give
        # most val queries no match. The val queries are still unseen.
        pool = ("val", "train") if s == "val" else (s,)
        ms = [{"id": m["id"], "message": m["message"]} for m in messages if m["split"] in pool]
        out[s] = {"queries": qs, "messages": ms, "scores": {f"{q['id']}|{m['id']}": grades[m["id"]][q["id"]] for q in qs for m in ms}}
    return out


def pairs(part: dict, per_query: int | float, seed: int = 0) -> list[tuple[str, str, float]]:
    """(query, message, P) training pairs: for each query all of Gemini's
    matches, the near misses (0.2 to 0.5) and random others, up to per_query."""
    rng = random.Random(seed)
    text = {m["id"]: m["message"] for m in part["messages"]}
    out = []
    for q in part["queries"]:
        graded = [(mid, part["scores"][f"{q['id']}|{mid}"]) for mid in text]
        hits = [g for g in graded if g[1] >= 0.5]
        near = [g for g in graded if 0.2 <= g[1] < 0.5]
        rest = [g for g in graded if g[1] < 0.2]
        rng.shuffle(hits), rng.shuffle(near), rng.shuffle(rest)
        take = hits[: per_query // 2]
        take += near[: (per_query - len(take)) // 2]
        take += rest[: per_query - len(take)]
        out += [(q["query"], text[mid], p) for mid, p in take]
    rng.shuffle(out)
    return out


# --- students -------------------------------------------------------------------


class CrossReranker:
    """Gemma 3 1B + LoRA: P(yes) of the match question for each message."""

    kind = "cross"

    def __init__(self, model: str | None = None):
        from .students.lora import LoraStudent

        self.student = LoraStudent(model) if model else LoraStudent()
        self.name, self.device = self.student.name, self.student.device

    def fit(self, train: dict, val: dict, per_query: int = 24, log=print, **cfg):
        items = [(m, "rerank", question(q), [p, 1 - p]) for q, m, p in pairs(train, per_query)]
        vitems = [(m, "rerank", question(q), [p, 1 - p]) for q, m, p in pairs(val, per_query, seed=1)]
        log(f"    {len(items)} training pairs, {len(vitems)} val pairs")
        self.student.fit_items(items, vitems, log=log, **cfg)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [p["rerank"][0] for p in self.student.predict(texts, {"rerank": question(query)})]

    def size(self) -> dict:
        return self.student.size()

    def save(self, folder: Path):
        self.student.save(folder)
        meta = json.loads((folder / "student.json").read_text())
        (folder / "student.json").write_text(json.dumps({**meta, "kind": "cross", "base": "lora"}))

    @classmethod
    def load(cls, folder: Path):
        from peft import PeftModel

        meta = json.loads((folder / "student.json").read_text())
        r = cls(meta["model"])
        if (folder / "adapter_config.json").exists():
            r.student.model = PeftModel.from_pretrained(r.student.model, str(folder)).merge_and_unload().eval()
        return r


class BiReranker:
    """EmbeddingGemma (+ LoRA): sigmoid(a * cos(query, message) + b)."""

    kind = "bi"

    def __init__(self, model: str | None = None):
        from .students.common import pick_device
        from .students.embed import MODEL, load_st

        self.name = model or MODEL
        self.device, _ = pick_device()
        self.model = load_st(self.name, self.device)
        self.model.max_seq_length = 512
        self.prompts = self.model.prompts or {}
        self.a, self.b = 10.0, -5.0  # fitted in calibrate()
        self.tuned = None

    def texts(self, texts: list[str], role: str) -> list[str]:
        return [self.prompts.get(role, "") + t[:4000] for t in texts]

    def vectors(self, texts: list[str], role: str) -> torch.Tensor:
        self.model.eval()
        return torch.tensor(self.model.encode(self.texts(texts, role), normalize_embeddings=True, batch_size=16, convert_to_numpy=True))

    def vectors_with_grad(self, texts: list[str], role: str) -> torch.Tensor:
        features = self.model.preprocess(self.texts(texts, role))
        features = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in features.items()}
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            v = self.model(features)["sentence_embedding"]
        return torch.nn.functional.normalize(v.float(), dim=-1)

    def calibrate(self, part: dict, log=print):
        """a and b on the val grades: the cosines stay in order, only P moves."""
        cos = self.vectors([q["query"] for q in part["queries"]], "query") @ self.vectors([m["message"] for m in part["messages"]], "document").T
        y = torch.tensor([[part["scores"][f"{q['id']}|{m['id']}"] for m in part["messages"]] for q in part["queries"]])
        ab = torch.tensor([self.a, self.b], requires_grad=True)
        opt = torch.optim.LBFGS([ab], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

        def loss():
            opt.zero_grad()
            v = torch.nn.functional.binary_cross_entropy_with_logits(ab[0] * cos + ab[1], y)
            v.backward()
            return v

        opt.step(loss)
        self.a, self.b = (float(x) for x in ab.detach())
        log(f"    calibrated: P = sigmoid({self.a:.2f} * cos {self.b:+.2f})")

    def fit(self, train: dict, val: dict, tune_epochs: int = 0, per_query: int = 48, rank: int = 16, lr: float = 2e-4, batch: int = 32, seed: int = 0, log=print, **_):
        if tune_epochs:
            self.tune(train, val, tune_epochs, per_query, rank, lr, batch, seed, log)
        self.calibrate(val, log)

    def tune(self, train, val, epochs, per_query, rank, lr, batch, seed, log):
        """LoRA on EmbeddingGemma: soft BCE of sigmoid(a * cos + b) against
        Gemini's P(match), a and b learned along."""
        from peft import LoraConfig, get_peft_model

        from .students.embed import TARGETS

        torch.manual_seed(seed)
        for p in self.model.parameters():
            p.requires_grad_(False)
        enc = self.model[0]
        enc.model = get_peft_model(enc.model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=TARGETS))
        self.tuned = {"rank": rank, "epochs": epochs, "lr": lr, "targets": TARGETS}
        lora = [p for p in self.model.parameters() if p.requires_grad]
        ab = torch.tensor([self.a, self.b], device=self.device, requires_grad=True)
        opt = torch.optim.AdamW([{"params": lora, "lr": lr}, {"params": [ab], "lr": 1e-2, "weight_decay": 0.0}])
        rows = pairs(train, per_query, seed)
        vrows = pairs(val, per_query, seed + 1)
        steps = epochs * ((len(rows) + batch - 1) // batch)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
        rng = random.Random(seed)

        def loss_of(chunk):
            q = self.vectors_with_grad([c[0] for c in chunk], "query")
            m = self.vectors_with_grad([c[1] for c in chunk], "document")
            y = torch.tensor([c[2] for c in chunk], device=q.device)
            return torch.nn.functional.binary_cross_entropy_with_logits(ab[0] * (q * m).sum(-1) + ab[1], y)

        def val_loss():
            self.model.eval()
            with torch.no_grad():
                return sum(loss_of(vrows[i:i + 64]).item() * len(vrows[i:i + 64]) for i in range(0, len(vrows), 64)) / len(vrows)

        best = (val_loss(), None)
        log(f"    {len(rows)} training pairs, {len(vrows)} val pairs; {sum(p.numel() for p in lora):,} LoRA numbers; val loss before {best[0]:.4f}")
        for epoch in range(epochs):
            self.model.train()
            rng.shuffle(rows)
            total = 0.0
            for i in range(0, len(rows), batch):
                chunk = rows[i:i + batch]
                loss = loss_of(chunk)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                total += loss.item() * len(chunk)
            v = val_loss()
            log(f"    tune epoch {epoch + 1}/{epochs}: train loss {total / len(rows):.4f}, val loss {v:.4f}")
            if v < best[0]:
                best = (v, {k: t.detach().clone() for k, t in enc.model.state_dict().items() if "lora_" in k})
        if best[1] is not None:
            enc.model.load_state_dict(best[1], strict=False)
        else:
            for k, t in enc.model.state_dict().items():
                if "lora_B" in k:
                    t.zero_()
        self.model.eval()

    def score(self, query: str, texts: list[str]) -> list[float]:
        return self.score_many([query], texts)[0]

    def score_many(self, queries: list[str], texts: list[str]) -> list[list[float]]:
        """Each message is encoded once, for all the queries."""
        cos = self.vectors(queries, "query") @ self.vectors(texts, "document").T
        return torch.sigmoid(self.a * cos + self.b).tolist()

    def size(self) -> dict:
        lora = sum(p.numel() for n, p in self.model.named_parameters() if "lora_" in n)
        return {"parameters": sum(p.numel() for p in self.model.parameters()) - lora, "trained_parameters": lora + 2}

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        if self.tuned:
            self.model[0].model.save_pretrained(str(folder / "encoder"))
        (folder / "student.json").write_text(json.dumps({"kind": "bi", "model": self.name, "a": self.a, "b": self.b, "tuned": self.tuned}))

    @classmethod
    def load(cls, folder: Path):
        meta = json.loads((folder / "student.json").read_text())
        r = cls(meta["model"])
        if meta.get("tuned"):
            from peft import PeftModel

            r.model[0].model = PeftModel.from_pretrained(r.model[0].model, str(folder / "encoder")).merge_and_unload()
            r.tuned = meta["tuned"]
        r.a, r.b = meta["a"], meta["b"]
        return r


def make(kind: str, model: str | None = None):
    return {"cross": CrossReranker, "bi": BiReranker}[kind](model)


def load(folder: Path):
    kind = json.loads((folder / "student.json").read_text())["kind"]
    return {"cross": CrossReranker, "bi": BiReranker}[kind].load(folder)


# --- judge ----------------------------------------------------------------------


def ndcg(scores: list[float], gains: list[float], k: int = 10) -> float:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    dcg = sum(gains[i] / math.log2(r + 2) for r, i in enumerate(order))
    ideal = sum(g / math.log2(r + 2) for r, g in enumerate(sorted(gains, reverse=True)[:k]))
    return dcg / ideal if ideal > 0 else 1.0


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos
        return torch.tensor(r)

    ra, rb = ranks(a), ranks(b)
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def judge(score_lists: dict[str, list[float]], part: dict) -> dict:
    """Mean over the queries of NDCG@10, Spearman, recall@10 and MAE."""
    per = []
    for q in part["queries"]:
        s = score_lists[q["id"]]
        g = [part["scores"][f"{q['id']}|{m['id']}"] for m in part["messages"]]
        top = set(sorted(range(len(s)), key=lambda i: -s[i])[:10])
        hits = [i for i, x in enumerate(g) if x >= 0.5]
        per.append({"ndcg@10": ndcg(s, g), "spearman": spearman(s, g),
                    "recall@10": len(top & set(hits)) / min(10, len(hits)) if hits else None,
                    "mae": sum(abs(x - y) for x, y in zip(s, g)) / len(g)})
    out = {}
    for k in ("ndcg@10", "spearman", "recall@10", "mae"):
        vals = [p[k] for p in per if p[k] is not None]
        out[k] = round(sum(vals) / len(vals), 4) if vals else None
    out["queries"], out["messages"] = len(part["queries"]), len(part["messages"])
    return out


def run_scores(student, part: dict) -> tuple[dict[str, list[float]], float]:
    texts = [m["message"] for m in part["messages"]]
    t = time.time()
    if hasattr(student, "score_many"):
        out = dict(zip([q["id"] for q in part["queries"]], student.score_many([q["query"] for q in part["queries"]], texts)))
    else:
        out = {q["id"]: student.score(q["query"], texts) for q in part["queries"]}
    return out, (time.time() - t) * 1000 / max(1, len(part["queries"]) * len(texts))


def two_stage(bi_scores: dict, cross, part: dict, k: int = 30) -> dict[str, list[float]]:
    """The bi student keeps its top k messages, the cross student orders them."""
    texts = [m["message"] for m in part["messages"]]
    out = {}
    for q in part["queries"]:
        s = bi_scores[q["id"]]
        top = sorted(range(len(s)), key=lambda i: -s[i])[:k]
        c = cross.score(q["query"], [texts[i] for i in top])
        final = [-1.0 + x * 1e-3 for x in s]  # the rest stays below, in the bi order
        for i, x in zip(top, c):
            final[i] = 1.0 + x
        out[q["id"]] = final
    return out


def train_and_judge(kind: str, data: dict, cfg: dict, out: Path, log=print) -> dict:
    started = time.time()
    student = make(kind, cfg.get("model"))
    test = data["test"]
    log(f"  rerank {kind}: {len(data['train']['queries'])} train queries × {len(data['train']['messages'])} messages; "
        f"test {len(test['queries'])} queries × {len(test['messages'])} messages")
    report = {"task": "rerank", "kind": kind, "model": student.name, "config": cfg}
    base, _ = run_scores(student, test)
    v = "EmbeddingGemma 2" if "embeddinggemma-2" in student.name else "EmbeddingGemma"
    report["baseline_name"] = "plain Gemma 3 1B (zero-shot)" if kind == "cross" else f"plain {v} (cosine)"
    report["baseline"] = judge(base, test)
    report["val_baseline"] = judge(run_scores(student, data["val"])[0], data["val"])
    log(f"  {report['baseline_name']}: NDCG@10 {report['baseline']['ndcg@10']:.3f}")
    t = time.time()
    student.fit(data["train"], data["val"], log=log, **{k: v for k, v in cfg.items() if k != "model"})
    report["train_seconds"] = round(time.time() - t, 1)
    scores, ms = run_scores(student, test)
    report["ms_per_pair"] = round(ms, 2)
    report["student"] = judge(scores, test)
    report["val"] = judge(run_scores(student, data["val"])[0], data["val"])
    report["kept"] = report["val"]["ndcg@10"] > report["val_baseline"]["ndcg@10"]
    report["scores"] = scores
    report["message_ids"] = [m["id"] for m in test["messages"]]
    report["size"] = student.size()
    report["seconds"] = round(time.time() - started, 1)
    student.save(out)
    log(f"  rerank {kind}: NDCG@10 {report['student']['ndcg@10']:.3f} (baseline {report['baseline']['ndcg@10']:.3f}), trained in {report['train_seconds']} s")
    return report
