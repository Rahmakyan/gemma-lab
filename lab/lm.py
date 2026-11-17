"""A word language model for the transcriber, and a CTC beam search over
words that uses it.

The reader gives, for each 40 ms slice, a probability for "blank" and for
each word. Greedy decoding takes the best one per slice. The beam search
keeps the best few word sequences instead, and adds the LM's opinion each
time a word starts:

    score = log P(audio | words) + alpha * log P_LM(words) + beta * len(words)

The LM is a word trigram with interpolated Kneser-Ney smoothing, counted
from LibriSpeech training transcripts and the train voicemail scripts (no
test text). Trigrams seen once are dropped, to keep it small.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

D = 0.75  # the Kneser-Ney discount
NEG = -1e30


def lae(a: float, b: float) -> float:
    """log(exp(a) + exp(b))."""
    if a < b:
        a, b = b, a
    return a if b <= NEG else a + math.log1p(math.exp(b - a))


class WordLM:
    def __init__(self, words: list[str]):
        self.words = words
        self.index = {w: i for i, w in enumerate(words)}
        self.V = len(words)

    # --- counting -------------------------------------------------------------

    def fit(self, lines: list[list[str]], min_trigram: int = 2) -> "WordLM":
        """lines: word lists. Words outside the word list are skipped."""
        V = self.V
        bi, tri = Counter(), Counter()
        for ws in lines:
            ids = [V, V] + [self.index[w] for w in ws if w in self.index]  # V = start of a line
            for i in range(2, len(ids)):
                a, b, c = ids[i - 2], ids[i - 1], ids[i]
                bi[(b, c)] += 1
                tri[(a, b, c)] += 1
        tri = {k: n for k, n in tri.items() if n >= min_trigram}
        self._build(bi, tri)
        return self

    def _build(self, bi: dict, tri: dict):
        # Continuation counts: how many different words come before w.
        cont = Counter(c for (_, c) in bi)
        self.cont_total = sum(cont.values())
        self.cont = cont
        # Bigram: count, and per history its total and its number of follow words.
        self.bi = bi
        self.bi_hist = Counter()
        self.bi_types = Counter()
        for (b, _), n in bi.items():
            self.bi_hist[b] += n
            self.bi_types[b] += 1
        self.tri = tri
        self.tri_hist = Counter()
        self.tri_types = Counter()
        for (a, b, _), n in tri.items():
            self.tri_hist[(a, b)] += n
            self.tri_types[(a, b)] += 1
        self.cache = {}

    # --- scores ---------------------------------------------------------------

    def p_uni(self, c: int) -> float:
        # Continuation probability, mixed with a uniform floor for unseen words.
        return 0.99 * self.cont.get(c, 0) / self.cont_total + 0.01 / self.V

    def p_bi(self, b: int, c: int) -> float:
        h = self.bi_hist.get(b, 0)
        if not h:
            return self.p_uni(c)
        return max(self.bi.get((b, c), 0) - D, 0) / h + D * self.bi_types[b] / h * self.p_uni(c)

    def logp(self, a: int, b: int, c: int) -> float:
        """log P(c | a b); a, b are word ids or V (the start of a line)."""
        key = (a, b, c)
        if key in self.cache:
            return self.cache[key]
        h = self.tri_hist.get((a, b), 0)
        if h:
            p = max(self.tri.get(key, 0) - D, 0) / h + D * self.tri_types[(a, b)] / h * self.p_bi(b, c)
        else:
            p = self.p_bi(b, c)
        out = math.log(p)
        if len(self.cache) < 2_000_000:
            self.cache[key] = out
        return out

    # --- files ----------------------------------------------------------------

    def save(self, path: Path):
        def arr(d: dict, width: int):
            keys = np.array(list(d.keys()), dtype=np.int32).reshape(-1, width)
            return keys, np.array(list(d.values()), dtype=np.int32)

        bk, bv = arr(self.bi, 2)
        tk, tv = arr(self.tri, 3)
        np.savez_compressed(path, bi_keys=bk, bi_vals=bv, tri_keys=tk, tri_vals=tv)

    @classmethod
    def load(cls, path: Path, words: list[str]) -> "WordLM":
        z = np.load(path)
        lm = cls(words)
        bi = {(int(a), int(b)): int(n) for (a, b), n in zip(z["bi_keys"], z["bi_vals"])}
        tri = {(int(a), int(b), int(c)): int(n) for (a, b, c), n in zip(z["tri_keys"], z["tri_vals"])}
        lm._build(bi, tri)
        return lm


def beam_search(blank_lp: np.ndarray, top_ids: np.ndarray, top_lp: np.ndarray, lm: WordLM,
                alpha: float = 0.5, beta: float = 1.0, beam: int = 12, skip_blank: float = 0.999) -> list[int]:
    """CTC prefix beam search over words. blank_lp [T]: log P(blank);
    top_ids [T, k]: the k best word ids (0-based) per slice; top_lp [T, k]:
    their log P. Returns the word ids of the best sequence.

    A sequence is a node in a tree (parent, word), so a long sequence is
    never copied or hashed whole."""
    V = lm.V
    parent, word, prev2 = [-1], [-1], [(V, V)]  # node 0: the empty sequence
    child = {}

    def extend(node: int, w: int) -> int:
        key = (node, w)
        if key not in child:
            child[key] = len(parent)
            parent.append(node)
            word.append(w)
            prev2.append((prev2[node][1], w))
        return child[key]

    beams = {0: (0.0, NEG)}  # node: (log P ending in blank, log P ending in its last word)
    sure_blank = math.log(skip_blank)
    for t in range(len(blank_lp)):
        b_t = float(blank_lp[t])
        if b_t > sure_blank:  # almost surely blank: every sequence just waits
            beams = {n: (lae(pb, pnb) + b_t, NEG) for n, (pb, pnb) in beams.items()}
            continue
        nxt = defaultdict(lambda: [NEG, NEG])
        for node, (pb, pnb) in beams.items():
            total = lae(pb, pnb)
            e = nxt[node]
            e[0] = lae(e[0], total + b_t)
            last = word[node]
            for w, lw in zip(top_ids[t].tolist(), top_lp[t].tolist()):
                if w == last:
                    e[1] = lae(e[1], pnb + lw)  # the same word goes on
                    src = pb  # a new copy of the same word needs a blank between
                else:
                    src = total
                if src <= NEG:
                    continue
                a, b = prev2[node]
                child_node = extend(node, w)
                f = nxt[child_node]
                f[1] = lae(f[1], src + lw + alpha * lm.logp(a, b, w) + beta)
        best = sorted(nxt.items(), key=lambda kv: lae(kv[1][0], kv[1][1]), reverse=True)[:beam]
        beams = {n: (pb, pnb) for n, (pb, pnb) in best}
    node = max(beams, key=lambda n: lae(*beams[n]))
    out = []
    while node > 0:
        out.append(word[node])
        node = parent[node]
    return out[::-1]


def tune(t, lm: WordLM, clips: list[dict], log=print, alphas=(0.1, 0.2, 0.3, 0.5), betas=(0.0, 1.0, 2.0, 3.0)) -> dict:
    """Chooses alpha and beta for the transcriber t on val clips
    [{wav: 16 kHz array, words}]. The slice scores are computed once."""
    from .voice import wer

    old_lm, old_cfg = t.lm, dict(t.lm_cfg)
    t.lm, t.lm_cfg["topk"] = lm, 8  # frame_scores keeps 8 words per slice
    scores = [t.scores(c["wav"]) for c in clips]

    def run():
        e_all = n_all = 0
        for s, c in zip(scores, clips):
            e, n = wer(c["words"], t.decode(s))
            e_all, n_all = e_all + e, n_all + n
        return e_all / max(1, n_all)

    t.lm = None
    results = {"greedy": round(run(), 4)}
    t.lm = lm
    best = None
    for a in alphas:
        for b in betas:
            t.lm_cfg.update(alpha=a, beta=b, beam=12)
            w = run()
            results[f"alpha {a} beta {b}"] = round(w, 4)
            if best is None or w < best[0]:
                best = (w, a, b)
    log(f"  word LM: val WER {best[0]:.1%} at alpha {best[1]}, beta {best[2]} (greedy {results['greedy']:.1%})")
    if best[0] < results["greedy"]:
        t.lm_cfg.update(alpha=best[1], beta=best[2], beam=12, topk=8)
    else:  # the LM does not help on val: keep what t had before
        t.lm, t.lm_cfg = old_lm, old_cfg
    return {"val": results, "chosen": {"alpha": best[1], "beta": best[2]} if t.lm is lm else None}
