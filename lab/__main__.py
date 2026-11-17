"""python -m lab <command>: the steps of the pipeline. See `runit --help`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

# torch says "Redirects are currently not supported" on macOS at import; it
# does not matter here.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

from .paths import ADAPTERS, DATA, LABELS, REPORTS, SCENARIOS, read_jsonl
from .task import QUESTIONS


def rows() -> list[dict]:
    """Scenarios that have a teacher label, joined."""
    labels = {r["id"]: r for r in read_jsonl(LABELS) if "target" in r}
    out = [{**s, "target": labels[s["id"]]["target"]} for s in read_jsonl(SCENARIOS) if s["id"] in labels]
    if not out:
        raise SystemExit("no labelled scenarios yet: run `runit scenarios` and `runit label` first")
    return out


def keep(report: dict, files: dict[str, bytes]) -> Path:
    kind = "embed2" if "embeddinggemma-2" in report.get("model", "") else report["kind"]
    tag = "-frozen" if report["kind"] == "embed" and not report.get("config", {}).get("tune_epochs") else ""
    folder = ADAPTERS / f"{kind}{tag}-{time.strftime('%Y%m%d-%H%M%S')}"
    for name, data in files.items():
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
    (folder / "report.json").write_text(json.dumps(report, indent=2))
    return folder


def promote(folders: list[Path]) -> Path | None:
    """adapters/current points at the student with the best val agreement
    (then the lowest val KL), among every kept student in adapters/ that was
    judged on the same val messages as this run's students. Students judged
    on other data do not compare."""
    from .train import better

    def report(f: Path) -> dict | None:
        p = f / "report.json"
        return json.loads(p.read_text()) if p.exists() else None

    kept = [f for f in folders if report(f)["kept"]]
    if not kept:
        return None
    ids = report(kept[0])["val"]["ids"]
    candidates = [f for f in sorted(ADAPTERS.iterdir()) if f.is_dir() and (r := report(f)) and r.get("task", "decide") == "decide"
                  and r.get("kept") and r.get("val", {}).get("ids") == ids]
    best = candidates[0]
    for f in candidates[1:]:
        if better(report(f)["val"]["student"], report(best)["val"]["student"]):
            best = f
    (ADAPTERS / "current").write_text(best.name)
    return best


def train(jobs: list[tuple[str, dict]], local: bool) -> list[Path]:
    data = rows()
    if local:
        import tempfile

        from .train import train_and_judge

        results = []
        for kind, cfg in jobs:
            out = Path(tempfile.mkdtemp())
            report = train_and_judge(kind, data, QUESTIONS, cfg, out)
            results.append((report, {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}))
    else:
        from .modal_app import train_on_modal

        results = [r for r in train_on_modal(jobs, data, QUESTIONS) if r is not None]
        if not results:
            raise SystemExit("  ✗ no student finished: see the Modal errors above")
    folders = [keep(r, f) for r, f in results]
    for f in folders:
        print(f"  ✓ saved {f.relative_to(ADAPTERS.parent)}")
    if best := promote(folders):
        print(f"  ✓ adapters/current → {best.name} (the local decider serves it)")
    return folders


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def system_name(r: dict) -> str:
    cfg = r.get("config", {})
    if r["kind"] == "lora":
        return f"Gemma 3 1B + LoRA ({'q, v' if cfg.get('targets') == 'qv' else 'all linear layers'}, r {cfg.get('rank', 16)})"
    v = "EmbeddingGemma 2 (text)" if "embeddinggemma-2" in r.get("model", "") else "EmbeddingGemma"
    if cfg.get("tune_epochs"):
        extra = f" (lr {cfg['lr']:g}, {cfg['tune_epochs']} epochs)" if cfg.get("lr", 2e-4) != 2e-4 else ""
        return f"{v} + LoRA, tuned with the heads{extra}"
    return f"{v} (frozen) + logistic regression"


def table(reports: list[dict]) -> str:
    """The benchmark as a markdown table: every way of answering, on the same test messages."""
    systems = [("Prior (Gemini's average answer)", reports[0]["prior"], None)]
    for r in reports:
        if r["kind"] == "lora":
            systems.append(("Plain Gemma 3 1B (zero-shot decision model)", r["baseline"], None))
    for r in reports:
        systems.append((system_name(r), r["student"], r))
    names = list(QUESTIONS)
    head = ["System", "Agreement (all)", "Coverage", "Challenge", "KL", *[f"{n} ({QUESTIONS[n]['type']})" for n in names],
            "callback AUC", "severity MAE", "Trained params", "Train time", "ms/message"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, j, r in systems:
        q = j["all"]["questions"]
        cells = [label, pct(j["all"]["agreement"]), pct(j.get("coverage", {}).get("agreement")), pct(j.get("challenge", {}).get("agreement")),
                 f"{j['all']['kl']:.3f}", *[pct(q[n]["agreement"]) for n in names],
                 f"{q['callback_today'].get('auc') or 0:.3f}", f"{q['severity']['mae']:.2f}",
                 f"{r['size']['trained_parameters']:,}" if r else "0", f"{r['train_seconds']} s" if r else "–",
                 f"{r['predict_ms_per_message']}" if r else "–"]
        lines.append("| " + " | ".join(cells) + " |")
    r = reports[0]
    return (f"Test set: {r['split']['test']} fake messages never seen in training ({r['student']['all']['n']} labelled; "
            f"coverage {r['student'].get('coverage', {}).get('n', 0)}, challenge {r['student'].get('challenge', {}).get('n', 0)}). "
            f"Trained on {r['split']['train']}, temperature/penalty chosen on {r['split']['val']}. "
            f"Agreement = the top answer is Gemini's top answer, mean over the {len(names)} questions. "
            f"Speed on {r.get('gpu', r['device'])}.\n\n" + "\n".join(lines))


def rerank_name(r: dict) -> str:
    if r["kind"] == "cross":
        return "Gemma 3 1B + LoRA (cross-encoder)"
    v = "EmbeddingGemma 2" if "embeddinggemma-2" in r.get("model", "") else "EmbeddingGemma"
    return f"{v} + LoRA (bi-encoder)" if r["config"].get("tune_epochs") else f"{v} frozen, calibrated"


def rerank_table(reports: list[dict]) -> str:
    from .rerank import dataset, judge, two_stage

    test = dict(dataset()["test"])
    # The test messages of these reports (the data can grow after a run).
    ids = reports[0].get("message_ids")
    n = len(next(iter(reports[0]["scores"].values())))
    keep_ids = set(ids or [])
    test["messages"] = [m for m in test["messages"] if m["id"] in keep_ids] if ids else test["messages"][:n]
    head = ["System", "NDCG@10", "Spearman", "Recall@10", "P(match) MAE", "Trained params", "Train time", "ms/pair"]
    rows = []
    for r in reports:  # a raw cosine is not a probability, so it has no MAE
        rows.append([r["baseline_name"], {**r["baseline"], "mae": None} if "cosine" in r["baseline_name"] else r["baseline"], None])
    for r in reports:
        rows.append([rerank_name(r), r["student"], r])
    bi = max((r for r in reports if r["kind"] == "bi" and r["config"].get("tune_epochs")), key=lambda r: r["val"]["ndcg@10"], default=None)
    cross = next((r for r in reports if r["kind"] == "cross"), None)
    if bi and cross:
        class Saved:  # the cross student's test scores, already computed on Modal
            def score(self, query, texts):
                q = next(q for q in test["queries"] if q["query"] == query)
                by_text = dict(zip([m["message"] for m in test["messages"]], cross["scores"][q["id"]]))
                return [by_text[t] for t in texts]
        two = judge(two_stage(bi["scores"], Saved(), test), test)
        two["mae"] = None  # its scores only order the messages
        rows.append([f"Two stages: {rerank_name(bi)} top 30, then the cross-encoder", two, None])
    seen, lines = set(), ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, j, r in rows:
        if name in seen:
            continue
        seen.add(name)
        lines.append("| " + " | ".join([name, f"{j['ndcg@10']:.3f}", f"{j['spearman']:.3f}", pct(j["recall@10"]), "–" if j["mae"] is None else f"{j['mae']:.3f}",
                                        f"{r['size']['trained_parameters']:,}" if r else "0", f"{r['train_seconds']} s" if r else "–",
                                        f"{r['ms_per_pair']}" if r else "–"]) + " |")
    j = reports[0]["student"]
    return (f"Test: {j['queries']} fake searches × {j['messages']} fake messages, neither seen in training. "
            f"Gain = Gemini's P(match). Recall@10 = share of Gemini's matches (P ≥ 0.5) in the top 10. "
            f"Speed on {reports[0].get('gpu', '?')} (the bi student's message vectors can be stored, so a search costs one query vector).\n\n" + "\n".join(lines))


def rerank_cmd(a):
    from . import rerank

    if a.action == "queries":
        rerank.make_queries(a.n)
    elif a.action == "label":
        rerank.label()
    elif a.action == "data":
        rerank.make_queries(a.n)
        rerank.label()
    elif a.action == "bench":
        data = rerank.dataset()
        from .students.embed import MODEL2

        bi = {"tune_epochs": a.tune_epochs, "per_query": a.bi_pairs}
        jobs = [("cross", {"epochs": a.epochs, "targets": "all", "per_query": a.cross_pairs}), ("bi", bi),
                ("bi", {**bi, "model": MODEL2}), ("bi", {"tune_epochs": 0})]
        if a.local:
            import tempfile

            results = []
            for kind, cfg in jobs:
                out = Path(tempfile.mkdtemp())
                report = rerank.train_and_judge(kind, data, cfg, out)
                results.append((report, {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}))
        else:
            from .modal_app import rerank_on_modal

            results = [r for r in rerank_on_modal(jobs, data) if r is not None]
            if not results:
                raise SystemExit("  ✗ no reranker finished: see the Modal errors above")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        folders = []
        for report, files in results:
            tag = ("bi2" if "embeddinggemma-2" in report.get("model", "") else report["kind"]) + ("" if report["kind"] == "cross" or report["config"].get("tune_epochs") else "-frozen")
            folder = ADAPTERS / f"rerank-{tag}-{stamp}"
            for name, blob in files.items():
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_bytes(blob)
            (folder / "report.json").write_text(json.dumps(report, indent=2))
            folders.append(folder)
            print(f"  ✓ saved {folder.relative_to(ADAPTERS.parent)}")
        kept = [f for f, (r, _) in zip(folders, results) if r["kept"]]
        if kept:
            best = max(kept, key=lambda f: json.loads((f / "report.json").read_text())["val"]["ndcg@10"])
            (ADAPTERS / "rerank-current").write_text(best.name)
            print(f"  ✓ adapters/rerank-current → {best.name} (best val NDCG@10; /v1/rerank serves it)")
        write_rerank_report(stamp, [r for r, _ in results])
    elif a.action == "table":  # the table again, from the saved reranker folders of one run (query = the stamp)
        reports = [json.loads((f / "report.json").read_text()) for f in sorted(ADAPTERS.glob(f"rerank-*-{a.query}"))]
        if not a.query or not reports:
            raise SystemExit("no saved rerankers for that stamp: runit rerank table STAMP (see adapters/rerank-*-STAMP)")
        write_rerank_report(a.query, reports)
    elif a.action == "search":
        from .serve import current_reranker

        folder = current_reranker()
        if not folder:
            raise SystemExit("no reranker trained yet: run `runit rerank bench` first")
        r = rerank.load(folder)
        msgs = [m for m in read_jsonl(SCENARIOS) if m["split"] == "test"]
        scores = r.score(a.query, [m["message"] for m in msgs])
        print(f"{r.kind} reranker ({folder.name}), top {a.top} of {len(msgs)} test messages for: {a.query}")
        for p, m in sorted(zip(scores, msgs), key=lambda x: -x[0])[: a.top]:
            print(f"  {p:.2f}  [{m['id']}] {m['message'][:150]}")


def write_rerank_report(stamp: str, reports: list[dict]):
    text = rerank_table(reports)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"rerank-{stamp}.md").write_text(f"# Reranker benchmark {stamp}\n\n{text}\n")
    (REPORTS / f"rerank-{stamp}.json").write_text(json.dumps(reports, indent=2))
    print("\n" + text + f"\n\n  ✓ wrote data/reports/rerank-{stamp}.md")


def voice_cmd(a):
    from . import voice

    if a.action == "data":
        voice.audio()
        voice.bar()
    elif a.action == "bar":
        voice.bar()
    elif a.action == "smoke":  # the whole path on Modal, small: 300 LibriSpeech clips, 1 epoch
        from .modal_app import transcriber_on_modal

        texts = {s: [m["message"] for m in read_jsonl(SCENARIOS) if m["split"] == s][:300] for s in ("train", "val", "test")}
        cfg = {"epochs": 1, "libri_limit": 300, "libri_test": 10, "batch_seconds": 200, "workers": 4}
        report = transcriber_on_modal({"train": [], "val": [], "test": []}, voice.AUDIO, texts, cfg, "smoke", Path("tmp/transcriber-smoke"))
        print(json.dumps({k: report[k] for k in ("words", "reader_parameters", "train_seconds", "test", "punctuation")}, indent=1)[:3000])
    elif a.action == "train":
        from .modal_app import transcriber_on_modal

        splits = {s: [r for r in read_jsonl(voice.SCRIPTS) if r["split"] == s and (voice.AUDIO / f"{r['id']}.wav").exists()] for s in ("train", "val", "test")}
        if not splits["train"]:
            raise SystemExit("no voicemail clips yet: run `runit voice data` first")
        # Written texts for the punctuation model: every message of the split.
        texts = {s: [m["message"] for m in read_jsonl(SCENARIOS) if m["split"] == s] + [r["spoken"] for r in splits[s]] for s in ("train", "val", "test")}
        cfg = {"epochs": a.epochs, "libri": a.libri.split(","), "batch_seconds": 400, "lr": a.lr, "voicemail_repeat": 4,
               "time_mask": a.time_mask, "reader": {"layers": a.layers, "d": a.width, "mix": 12 if a.mix else 0}}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        folder = ADAPTERS / f"transcriber-{stamp}"
        report = transcriber_on_modal(splits, voice.AUDIO, texts, cfg, stamp, folder)
        report["bar"] = voice.bar()
        (folder / "report.json").write_text(json.dumps(report, indent=2))
        old = ADAPTERS / "transcriber-current"
        prev = ADAPTERS / old.read_text().strip() / "report.json" if old.exists() else None
        prev = json.loads(prev.read_text()) if prev is not None and prev.exists() else None
        if prev is None or val_wer(prev) is None or val_wer(prev) > val_wer(report):
            old.write_text(folder.name)
            print(f"  ✓ adapters/transcriber-current → {folder.name} (best val WER; /v1/transcribe serves it)")
        text = voice_table(transcriber_reports() or [report])
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / f"transcriber-{stamp}.md").write_text(f"# Transcriber {stamp}\n\n{text}\n")
        print("\n" + text + f"\n\n  ✓ wrote data/reports/transcriber-{stamp}.md")
    elif a.action == "lm":  # a tuned word LM for a saved transcriber (default: the current one)
        import zlib

        from .lm import WordLM, tune
        from .serve import current_transcriber
        from .transcriber import Transcriber

        folder = Path(a.file) if a.file else current_transcriber()
        if folder is None:
            raise SystemExit("no transcriber yet: run `runit voice train` first")
        text = DATA / "lm" / "librispeech-train-clean.txt"
        if not text.exists():
            from .modal_app import libri_text_on_modal

            print("  LibriSpeech train transcripts: reading them on Modal (text only)")
            text.parent.mkdir(parents=True, exist_ok=True)
            text.write_text("\n".join(libri_text_on_modal(["train.100", "train.360"])))
        t = Transcriber.load(folder)
        scripts = read_jsonl(voice.SCRIPTS)
        train_vm = [voice.words(r["spoken"]) for r in scripts if r["split"] == "train"]
        lm = WordLM(t.words).fit([l.split() for l in text.read_text().split("\n")] + train_vm * 20)
        print(f"  word LM: {len(lm.tri):,} trigrams, {len(lm.bi):,} bigrams")
        prep = lambda c: {"wav": voice.phone(voice.load(c["wav"]), zlib.crc32(c["id"].encode())), "words": c["words"]}
        result = tune(t, lm, [prep(c) for c in voice.clips("val")[:60]])
        test = [prep(c) for c in voice.clips("test")]
        scores = [t.scores(c["wav"]) for c in test]
        def test_wer():
            e_all = n_all = 0
            for s, c in zip(scores, test):
                e, n = voice.wer(c["words"], t.decode(s))
                e_all, n_all = e_all + e, n_all + n
            return round(e_all / max(1, n_all), 4)
        chosen_lm = t.lm
        t.lm = None
        result["test_greedy"] = test_wer()
        t.lm = chosen_lm
        result["test"] = test_wer()
        print(f"  test voicemails (phone line): greedy {result['test_greedy']:.1%}, with the LM {result['test']:.1%}")
        if t.lm is not None:
            t.lm.save(folder / "lm.npz")
            meta = json.loads((folder / "student.json").read_text())
            meta["lm"] = t.lm_cfg
            (folder / "student.json").write_text(json.dumps(meta))
            print(f"  ✓ saved the LM in {folder.name}; /v1/transcribe and `runit voice transcribe` use it")
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text())
        report["lm_added"] = result
        report_path.write_text(json.dumps(report, indent=2))
    elif a.action == "table":  # every saved run in one table
        reports = transcriber_reports()
        if not reports:
            raise SystemExit("no transcriber runs yet: run `runit voice train` first")
        text = voice_table(reports)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (REPORTS / f"transcriber-{stamp}.md").write_text(f"# Transcriber {stamp}\n\n{text}\n")
        print(text + f"\n\n  ✓ wrote data/reports/transcriber-{stamp}.md")
    elif a.action == "transcribe":
        from .serve import current_transcriber
        from .transcriber import Transcriber

        if not a.file:
            raise SystemExit("give an audio file: runit voice transcribe FILE")
        folder = current_transcriber()
        if not folder:
            raise SystemExit("no transcriber trained yet: run `runit voice train` first")
        t = Transcriber.load(folder)
        x = voice.load(a.file)
        if a.phone:
            x = voice.phone(x, 0)
        started = time.time()
        res = t.transcribe([x])[0]
        print(res["text"])
        print(f"  ({len(x) / 16000:.1f} s of audio in {time.time() - started:.1f} s, {folder.name})")


def val_wer(r: dict) -> float | None:
    """The WER on the tuning val set (60 voicemails + 100 LibriSpeech), at the chosen bonus."""
    b = r.get("bonus")
    if not b:
        return None
    return b["val_wer"].get(str(b["chosen"]), b["val_wer"].get(b["chosen"]))  # JSON makes the keys strings


def transcriber_reports() -> list[dict]:
    """Every saved transcriber run, oldest first."""
    return [json.loads((f / "report.json").read_text()) for f in sorted(ADAPTERS.glob("transcriber-2*")) if (f / "report.json").exists()]


def transcriber_name(r: dict) -> str:
    c = r["config"]
    hours = {"train.100": 100, "train.360": 360, "other/train.500": 500}
    h = sum(hours.get(x, 0) for x in c.get("libri", ["train.100"]))
    rd = c.get("reader", {})
    parts = [f"{h} h LibriSpeech", f"{rd.get('layers', 4)}-layer reader"]
    if rd.get("mix"):
        parts.append("layer mix")
    if c.get("time_mask"):
        parts.append(f"time mask {c['time_mask']}")
    if r.get("bonus"):
        parts.append(f"word bonus {r['bonus']['chosen']}")
    return "EmbeddingGemma 2 ear + reader (" + ", ".join(parts) + ")"


def voice_table(reports: list[dict]) -> str:
    """One row for each transcriber run; the last run gives the notes and samples."""
    from .voice import TTS

    r = reports[-1]
    vm, lib = r["test"]["voicemail"], r["test"]["librispeech"]
    pct = lambda part: f"{part['wer']:.1%}" if part else "–"
    lines = ["| System | Voicemails, phone line (WER) | LibriSpeech test-clean (WER) | LibriSpeech test-other (WER) | Trained numbers | Speed |", "|---|---|---|---|---|---|",
             f"| {r['bar']['model']} (the bar) | {r['bar']['wer']:.1%} | – | – | – | API |"]
    for x in reports:
        t, name = x["test"], transcriber_name(x)
        size, speed = f"{x['reader_parameters']:,} + punctuation", f"{t['voicemail']['real_time']}× real time on {x['gpu']}"
        if x.get("test_greedy"):  # trained with the LM: the greedy row first
            lines.append(f"| {name}, greedy | {pct(x['test_greedy']['voicemail'])} | – | – | {size} | {speed} |")
            name += " + word LM"
        lines.append(f"| {name} | {pct(t['voicemail'])} | {pct(t['librispeech'])} | {pct(t.get('librispeech_other'))} | {size} | {speed} |")
        if x.get("lm_added"):  # an LM added later (`runit voice lm`), with overlapping windows
            add = x["lm_added"]
            lines.append(f"| {name}, overlapping windows | {add['test_greedy']:.1%} | – | – | {size} | – |")
            if add.get("chosen"):
                lines.append(f"| {name}, overlapping windows + word LM (alpha {add['chosen']['alpha']}, beta {add['chosen']['beta']}) | {add['test']:.1%} | – | – | {size} + 7 MB LM | – |")
    lines.append("| For comparison: mlboydaisuke EG2-Transcribe v1 / v2 (100 h / 960 h, greedy, published) | – | 6.81% / 4.21% | 15.10% / 10.42% | 19.6M (v2) | – |")
    p = r["punctuation"]
    out = (f"Test: {vm['clips']} fake voicemails (spoken by {TTS}, phone-line effect) and {lib['clips']} LibriSpeech test-clean clips. "
           f"WER on lower-case words without punctuation. Reader trained {r['train_seconds'] / 60:.0f} min. "
           f"Punctuation model: capitals right {p['case_accuracy']:.1%}, punctuation right {p['punct_accuracy']:.1%} of words.\n\n" + "\n".join(lines))
    for name, part in (("voicemail", vm), ("LibriSpeech", lib)):
        out += f"\n\nSamples ({name}):\n" + "\n".join(f"- truth: {s['truth']}\n  heard: {s['heard']}" for s in part["samples"][:3])
    return out


def decision_report(folders: list[Path]) -> str:
    """One table for the decision students of folders (judged on the same
    test messages), and their progress on the first 137 test messages."""
    from .train import judge

    reports = [json.loads((f / "report.json").read_text()) for f in folders]
    text = table(reports)
    original = {m["id"] for m in read_jsonl(SCENARIOS)[:784]}  # the scenarios of the first rounds
    first = {r["id"]: r for r in rows() if r["split"] == "test" and r["id"] in original}
    old = [(stamp, json.loads((REPORTS / f"bench-{stamp}.json").read_text())) for stamp in ("20261114-194440", "20261114-202106") if (REPORTS / f"bench-{stamp}.json").exists()]
    lines = ["| Student | v1 (q, v LoRA; frozen embed) | v2 (all-layer LoRA; tuned embed) | v3 (2× the data) |", "|---|---|---|---|"]
    test = [first[i] for i in sorted(first)]
    for r in reports:
        if "preds" not in r:
            continue
        name = system_name(r)
        now = judge([r["preds"][t["id"]] for t in test], test, QUESTIONS)["all"]["agreement"]
        past = []
        for _, rs in old:
            same = [x for x in rs if x["kind"] == r["kind"] and ("embeddinggemma-2" in x.get("model", "")) == ("embeddinggemma-2" in r.get("model", ""))
                    and bool(x.get("config", {}).get("tune_epochs")) == bool(r.get("config", {}).get("tune_epochs"))]
            if r["kind"] == "embed" and r.get("config", {}).get("tune_epochs") and not same:
                same = [x for x in rs if x["kind"] == "embed" and "embeddinggemma-2" not in x.get("model", "") and "embeddinggemma-2" not in r.get("model", "")]
            past.append(pct(same[0]["student"]["all"]["agreement"]) if same else "–")
        past += ["–"] * (2 - len(past))
        lines.append(f"| {name} | {past[0]} | {past[1]} | {pct(now)} |")
    return text + f"\n\nProgress on the first {len(test)} test messages (the same messages in every round; agreement with Gemini):\n\n" + "\n".join(lines)


def main():
    p = argparse.ArgumentParser(prog="python -m lab")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scenarios")
    s.add_argument("--n", type=int, default=900)
    s.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("review")
    r.add_argument("--n", type=int, default=8)
    sub.add_parser("label")
    for name in ("train", "bench"):
        t = sub.add_parser(name)
        if name == "train":
            t.add_argument("--student", choices=["lora", "embed", "embed2"], default="lora")
        t.add_argument("--local", action="store_true")
        t.add_argument("--epochs", type=int, default=3)
        t.add_argument("--rank", type=int, default=16)
        t.add_argument("--targets", choices=["all", "qv"], default="all")
        t.add_argument("--tune-epochs", type=int, default=6)
        t.add_argument("--embed-lr", type=float, default=2e-4, help="LoRA learning rate of the embed students")
    d = sub.add_parser("decide")
    d.add_argument("text")
    d.add_argument("--teacher", action="store_true")
    sub.add_parser("status")
    tb = sub.add_parser("table")
    tb.add_argument("folders", nargs="+")
    rr = sub.add_parser("rerank")
    rr.add_argument("action", choices=["queries", "label", "data", "bench", "search", "table"])
    rr.add_argument("query", nargs="?", default="")
    rr.add_argument("--n", type=int, default=100)
    rr.add_argument("--top", type=int, default=8)
    rr.add_argument("--local", action="store_true")
    rr.add_argument("--epochs", type=int, default=3)
    rr.add_argument("--tune-epochs", type=int, default=6)
    rr.add_argument("--cross-pairs", type=int, default=24, help="training pairs per query for the cross-encoder")
    rr.add_argument("--bi-pairs", type=int, default=48, help="training pairs per query for the bi-encoders")
    v = sub.add_parser("voice")
    v.add_argument("action", choices=["data", "bar", "train", "smoke", "transcribe", "table", "lm"])
    v.add_argument("file", nargs="?", default="")
    v.add_argument("--phone", action="store_true")
    v.add_argument("--epochs", type=int, default=6)
    v.add_argument("--lr", type=float, default=1e-3)
    v.add_argument("--libri", default="train.100")
    v.add_argument("--layers", type=int, default=4)
    v.add_argument("--width", type=int, default=512)
    v.add_argument("--time-mask", type=float, default=0.0)
    v.add_argument("--mix", action="store_true", help="the reader mixes all 13 ear states, not only the last")
    a = p.parse_args()

    if a.cmd == "scenarios":
        from .scenarios import make

        rows_ = make(a.n, seed=a.seed)
        print(f"  ✓ {len(rows_)} scenarios in {SCENARIOS.relative_to(SCENARIOS.parent.parent)}")
    elif a.cmd == "review":
        scenarios = read_jsonl(SCENARIOS)
        for row in random.sample(scenarios, min(a.n, len(scenarios))):
            print(f"[{row['id']} {row['split']}/{row['set']}] {row['topic']} · {row['urgency']} · {row['clarity']} · {row['style']} · {row['channel']} (realistic {row['realistic']})")
            print(f"  {row['message']}\n")
    elif a.cmd == "label":
        from .teacher import run

        run()
    elif a.cmd in ("train", "bench"):
        from .students.embed import MODEL2

        # The model names go in the config: a Modal container does not see this shell's env.
        lora = {"epochs": a.epochs, "rank": a.rank, "targets": a.targets}
        tuned = {"tune_epochs": a.tune_epochs, "lr": a.embed_lr}
        if a.cmd == "train":
            jobs = {"lora": [("lora", lora)], "embed": [("embed", tuned)], "embed2": [("embed", {**tuned, "model": MODEL2})]}[a.student]
        else:  # the three students, and the frozen embed student to show what tuning adds
            jobs = [("lora", lora), ("embed", tuned), ("embed", {**tuned, "model": MODEL2}), ("embed", {"tune_epochs": 0})]
        for kind, cfg in jobs:
            if "model" not in cfg and os.environ.get(var := {"lora": "GEMMA_MODEL", "embed": "EMBED_MODEL"}[kind]):
                cfg["model"] = os.environ[var]
        folders = train(jobs, a.local)
        if a.cmd == "bench":
            reports = [json.loads((f / "report.json").read_text()) for f in folders]
            text = table(reports)
            REPORTS.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            (REPORTS / f"bench-{stamp}.md").write_text(f"# Benchmark {stamp}\n\n{text}\n")
            (REPORTS / f"bench-{stamp}.json").write_text(json.dumps(reports, indent=2))
            print("\n" + text + f"\n\n  ✓ wrote data/reports/bench-{stamp}.md")
    elif a.cmd == "decide":
        from .answers import answer
        from .serve import current
        from .students import load, make

        folder = current()
        s = load(folder) if folder else make("lora")
        probs = s.predict([a.text], QUESTIONS)[0]
        print(f"local {s.kind} ({folder.name if folder else 'plain Gemma'}):")
        print(json.dumps({n: answer(q, probs[n]) for n, q in QUESTIONS.items()}, indent=1))
        if a.teacher:
            from . import gemini

            print(f"teacher {gemini.DECIDER}:")
            print(json.dumps(gemini.decide(a.text, QUESTIONS)["answers"], indent=1))
    elif a.cmd == "table":
        folders = [ADAPTERS / f for f in a.folders]
        text = decision_report(folders)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (REPORTS / f"decisions-{stamp}.md").write_text(f"# Decision students {stamp}\n\n{text}\n")
        if best := promote(folders):
            print(f"  ✓ adapters/current → {best.name}")
        print(text + f"\n\n  ✓ wrote data/reports/decisions-{stamp}.md")
    elif a.cmd == "voice":
        voice_cmd(a)
    elif a.cmd == "rerank":
        rerank_cmd(a)
    elif a.cmd == "status":
        sc, lb = read_jsonl(SCENARIOS), read_jsonl(LABELS)
        print(f"scenarios {len(sc)} ({sum(r['set'] == 'challenge' for r in sc)} challenge; "
              f"train {sum(r['split'] == 'train' for r in sc)}, val {sum(r['split'] == 'val' for r in sc)}, test {sum(r['split'] == 'test' for r in sc)})")
        print(f"labels    {sum('target' in r for r in lb)} ({sum('blocked' in r for r in lb)} blocked)")
        pointer = ADAPTERS / "current"
        print(f"current   {pointer.read_text().strip() if pointer.exists() else 'none (plain Gemma)'}")
        from .rerank import QUERIES, RELEVANCE

        rp = ADAPTERS / "rerank-current"
        print(f"rerank    {len(read_jsonl(QUERIES))} queries, {len(read_jsonl(RELEVANCE))} messages graded; "
              f"current {rp.read_text().strip() if rp.exists() else 'none'}")


if __name__ == "__main__":
    main()
