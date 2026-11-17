"""EmbeddingGemma 2 as a transcriber, the eg2 way (eg2.haku.sh).

EmbeddingGemma 2 turns audio into vectors, but it never writes text. Here
it does, with only a small new part trained:

1. Ear (frozen): EmbeddingGemma 2's audio encoder, exactly as it is. It
   gives one vector of 1536 numbers per 40 ms of 16 kHz audio.
2. Reader (trained, about 14 M numbers): a small Transformer that reads the
   ear's vectors in both directions (the ear only looks back) and moves
   each 40 ms slice into EmbeddingGemma 2's text space (768 numbers), or
   says "blank" (no new word starts here).
3. Words (frozen): the text side of EmbeddingGemma 2 turns each word of a
   word list (about 200,000 English words, the LibriSpeech lexicon plus
   the clinic's words) into a vector. The closest word to a slice wins.
   Same words in a row become one, blanks go (CTC). The reader is trained
   with the CTC loss, with these word vectors as its fixed output layer.
   So it can also write a word it never heard in training, if the word is
   in the list.
4. Punctuation (trained, tiny): a 2-layer GRU over the word vectors adds
   capitals and . , ? to the words.

Data: open English speech (LibriSpeech train-clean-100, CC BY 4.0) and the
fake voicemails of voice.py, with a phone-line effect.
"""

from __future__ import annotations

import io
import json
import math
import os
import random
import re
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

ENCODER = os.environ.get("EAR_MODEL", "unsloth/embeddinggemma-2")  # google/embeddinggemma-2 without the login
LEXICON = "https://www.openslr.org/resources/11/librispeech-lexicon.txt"
LIBRI = "openslr/librispeech_asr"
FRAME_SECONDS = 0.04
CASE = ["lower", "title", "upper"]
PUNCT = ["", ",", ".", "?"]


# --- the ear --------------------------------------------------------------------


class Ear:
    """EmbeddingGemma 2's audio encoder, frozen."""

    def __init__(self, device: str, dtype=torch.bfloat16, all_layers: bool = False):
        from transformers import AutoFeatureExtractor, AutoModel

        self.device, self.dtype, self.all_layers = device, dtype, all_layers
        self.features = AutoFeatureExtractor.from_pretrained(ENCODER)
        model = AutoModel.from_pretrained(ENCODER, vision_config=None, dtype=dtype)
        self.tower = model.audio_tower.to(device).eval().requires_grad_(False)
        del model

    @torch.no_grad()
    def __call__(self, waves: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.prepare(waves)
        return self.encode(f)

    def prepare(self, waves: list[np.ndarray]) -> dict:
        """The log-mel features (CPU work: a DataLoader worker can do it)."""
        f = self.features(waves, sampling_rate=16000, return_tensors="pt", padding="longest")
        return {"input_features": f["input_features"], "mask": f["input_features_mask"]}

    @torch.no_grad()
    def encode(self, f: dict) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.tower(f["input_features"].to(self.device, self.dtype), f["mask"].to(self.device), output_hidden_states=self.all_layers)
        if not self.all_layers:
            return out.last_hidden_state, out.attention_mask
        # The 12 inner states (1024 numbers: the input and layers 1-11), then
        # the last state (layer 12 after the output projection, 1536), in a row.
        inner = torch.cat(out.hidden_states[:-1], -1)
        return torch.cat([inner, out.last_hidden_state], -1), out.attention_mask


# --- the words ------------------------------------------------------------------


def clean_words(text: str) -> list[str]:
    from .voice import words

    return words(text)


def word_list(extra_texts: list[str], cache: Path, extra_words: list[list[str]] = ()) -> list[str]:
    """The LibriSpeech lexicon (about 200,000 words), plus every word of the
    given texts and of the given word lists, a-z and the apostrophe only.
    Word lists are added as they are: clean_words a second time can change
    them (spelled letters "m m" become "mm", then "mm" is a filler)."""
    path = cache / "librispeech-lexicon.txt"
    if not path.exists():
        import urllib.request

        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LEXICON, path)
    words = {line.split()[0].lower() for line in path.read_text().splitlines() if line.strip()}
    for t in extra_texts:
        words.update(clean_words(t))
    for ws in extra_words:
        words.update(ws)
    return sorted(w for w in words if re.fullmatch(r"[a-z][a-z']*", w))


@torch.no_grad()
def word_vectors(words: list[str], device: str) -> torch.Tensor:
    """Each word through EmbeddingGemma 2's text side: a unit vector of 768."""
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(ENCODER, device=device, config_kwargs={"vision_config": None, "audio_config": None},
                             model_kwargs={"dtype": torch.bfloat16 if device == "cuda" else torch.float32})
    v = st.encode(words, batch_size=1024, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    del st
    return v.float().cpu()


# --- the reader -----------------------------------------------------------------


class Reader(nn.Module):
    def __init__(self, d_in: int = 1536, d: int = 512, layers: int = 4, heads: int = 8, d_out: int = 768, dropout: float = 0.1,
                 mix: int = 0, d_mix: int = 1024):
        """mix > 0: the input also has the ear's `mix` inner states (d_mix
        each) before the last state, and the reader learns one weight for
        each (the SUPERB way: inner layers keep more of the sound)."""
        super().__init__()
        self.cfg = {"d_in": d_in, "d": d, "layers": layers, "heads": heads, "d_out": d_out, "dropout": dropout, "mix": mix, "d_mix": d_mix}
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.Dropout(dropout))
        if mix:
            self.mix_weights = nn.Parameter(torch.zeros(mix))
            self.inp_mix = nn.Sequential(nn.Linear(d_mix, d), nn.Dropout(dropout))
        self.pos = nn.Conv1d(d, d, kernel_size=31, padding=15, groups=d)  # relative position, as in wav2vec 2
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, d_out)
        self.blank = nn.Linear(d, 1)
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mix, d_mix = self.cfg["mix"], self.cfg["d_mix"]
        if mix:
            inner = h[..., :mix * d_mix].float().unflatten(-1, (mix, d_mix))
            inner = nn.functional.layer_norm(inner, (d_mix,))
            x = self.inp(h[..., mix * d_mix:].float()) + self.inp_mix((inner * self.mix_weights.softmax(0)[:, None]).sum(-2))
        else:
            x = self.inp(h.float())
        x = x + self.pos(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(self.body(x, src_key_padding_mask=~mask.bool()))
        return nn.functional.normalize(self.out(x), dim=-1), self.blank(x).squeeze(-1)

    def logits(self, z: torch.Tensor, blank: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """[blank, word 1, word 2, …] per slice: the blank logit and the
        scaled cosine to each word vector."""
        return torch.cat([blank.unsqueeze(-1), self.log_scale.exp() * (z @ W.T)], -1)


class Punctuator(nn.Module):
    """Capitals and . , ? from the word vectors alone (no audio)."""

    def __init__(self, d_in: int = 768, d: int = 256):
        super().__init__()
        self.gru = nn.GRU(d_in, d, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.case = nn.Linear(2 * d, len(CASE))
        self.punct = nn.Linear(2 * d, len(PUNCT))

    def forward(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.gru(v)
        return self.case(x), self.punct(x)


def punct_labels(text: str) -> list[tuple[str, int, int]]:
    """(word, case, punctuation after) for each word of a written text."""
    out = []
    for tok in re.findall(r"[A-Za-z][A-Za-z']*|[,.?!;:]", text.replace("’", "'").replace("-", " ")):
        if tok[0].isalpha():
            case = 2 if tok.isupper() and len(tok) > 1 else 1 if tok[0].isupper() else 0
            out.append([tok.lower(), case, 0])
        elif out:
            out[-1][2] = {",": 1, ";": 1, ":": 1, ".": 2, "!": 2, "?": 3}[tok]
    return [tuple(x) for x in out]


def dress(words: list[str], case: list[int], punct: list[int]) -> str:
    out = []
    for w, c, p in zip(words, case, punct):
        w = w.upper() if c == 2 else w.capitalize() if c == 1 else w
        if w == "i" or w.startswith("i'"):
            w = "I" + w[1:]
        out.append(w + PUNCT[p])
    text = " ".join(out)
    return text[:1].upper() + text[1:]


# --- the whole transcriber ------------------------------------------------------


class Transcriber:
    def __init__(self, reader: Reader, words: list[str], W: torch.Tensor, punctuator: Punctuator | None, device: str, ear: Ear | None = None,
                 heard: list[str] = (), bonus: float = 0.0, lm=None, lm_cfg: dict | None = None):
        self.reader, self.words, self.W, self.punctuator, self.device = reader.to(device).eval(), words, W.to(device), punctuator, device
        self.index = {w: i for i, w in enumerate(words)}
        self.heard = [w for w in heard if w in self.index]
        self.set_bonus(bonus)
        self.lm, self.lm_cfg = lm, dict(lm_cfg or {"alpha": 0.5, "beta": 1.0, "beam": 12, "topk": 8})
        if punctuator is not None:
            punctuator.to(device).eval()
        self.ear = ear or Ear(device, torch.bfloat16 if device == "cuda" else torch.float32, all_layers=reader.cfg.get("mix", 0) > 0)

    def set_bonus(self, bonus: float):
        """A logit bonus for the words of the training audio. Without it, a
        rare or misspelled lexicon word ("speach") can win over the common
        word next to it; the bonus is tuned on val."""
        self.bonus = bonus
        self.prior = torch.zeros(len(self.words) + 1, device=self.device)
        if bonus and self.heard:
            self.prior[torch.tensor([self.index[w] + 1 for w in self.heard], device=self.device)] = bonus

    @torch.no_grad()
    def frame_scores(self, h: torch.Tensor, mask: torch.Tensor) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """For each slice: log P(blank), the k best word ids (0-based) and
        their log P (k = 1 without an LM), with the bonus."""
        k = self.lm_cfg["topk"] if self.lm is not None else 1
        z, blank = self.reader(h, mask)
        out = []
        for b in range(z.shape[0]):
            n = int(mask[b].sum())
            parts = []
            for i in range(0, n, 512):  # [slices, 200,000 words] in parts
                lp = (self.reader.logits(z[b, i:i + 512], blank[b, i:i + 512], self.W).float() + self.prior).log_softmax(-1)
                top = lp[:, 1:].topk(k, -1)
                parts.append((lp[:, 0], top.indices, top.values))
            out.append(tuple(torch.cat([p[j] for p in parts]).cpu().numpy() for j in range(3)))
        return out

    def decode(self, scores: tuple[np.ndarray, np.ndarray, np.ndarray]) -> list[str]:
        blank_lp, ids, lp = scores
        if self.lm is not None:
            from .lm import beam_search

            c = self.lm_cfg
            return [self.words[i] for i in beam_search(blank_lp, ids, lp, self.lm, c["alpha"], c["beta"], c["beam"])]
        # Greedy CTC: the best of each slice; same ids in a row become one, blanks go.
        words, last = [], -1
        for b, i, l in zip(blank_lp, ids[:, 0], lp[:, 0]):
            k = -1 if b >= l else int(i)
            if k != last and k != -1:
                words.append(self.words[k])
            last = k
        return words

    @torch.no_grad()
    def dress(self, words: list[str]) -> str:
        if not words:
            return ""
        if self.punctuator is None:
            return " ".join(words)
        v = self.W[[self.index[w] for w in words]].unsqueeze(0)
        case, punct = self.punctuator(v)
        return dress(words, case[0].argmax(-1).tolist(), punct[0].argmax(-1).tolist())

    def scores(self, w: np.ndarray, chunk_seconds: float = 30.0, overlap_seconds: float = 4.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The slice scores of one 16 kHz wave. The ear takes at most 30 s, so
        long audio goes in 30 s windows that overlap by 4 s; each window keeps
        its slices up to the middle of the overlap, so a word at a cut is read
        whole by one window."""
        win, step = int(chunk_seconds * 16000), int((chunk_seconds - overlap_seconds) * 16000)
        half = int(overlap_seconds / 2 / FRAME_SECONDS)
        kept, starts = [], list(range(0, max(1, len(w) - (win - step)), step)) or [0]
        for j, i in enumerate(starts):
            h, m = self.ear([w[i:i + win]])
            f = self.frame_scores(h, m)[0]
            n = len(f[0])
            a, b = (0 if j == 0 else half), (n if j == len(starts) - 1 else n - half)
            kept.append(tuple(x[a:b] for x in f))
        return tuple(np.concatenate([k[j] for k in kept]) for j in range(3))

    def transcribe(self, waves: list[np.ndarray]) -> list[dict]:
        """[{words, text}] per 16 kHz wave."""
        out = []
        for w in waves:
            words = self.decode(self.scores(w))
            out.append({"words": words, "text": self.dress(words)})
        return out

    def save(self, folder: Path, extra: dict):
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.reader.state_dict(), folder / "reader.pt")
        if self.punctuator is not None:
            torch.save(self.punctuator.state_dict(), folder / "punctuator.pt")
        torch.save(self.W.half().cpu(), folder / "word_vectors.pt")
        (folder / "words.txt").write_text("\n".join(self.words))
        (folder / "heard.txt").write_text("\n".join(self.heard))
        if self.lm is not None:
            self.lm.save(folder / "lm.npz")
        (folder / "student.json").write_text(json.dumps({"kind": "transcriber", "ear": ENCODER, "reader": self.reader.cfg, "bonus": self.bonus,
                                                         "lm": self.lm_cfg if self.lm is not None else None, **extra}))

    @classmethod
    def load(cls, folder: Path, device: str | None = None):
        from .students.common import pick_device

        device = device or pick_device()[0]
        meta = json.loads((folder / "student.json").read_text())
        reader = Reader(**meta["reader"])
        reader.load_state_dict(torch.load(folder / "reader.pt", map_location="cpu"))
        punct = None
        if (folder / "punctuator.pt").exists():
            punct = Punctuator()
            punct.load_state_dict(torch.load(folder / "punctuator.pt", map_location="cpu"))
        W = torch.load(folder / "word_vectors.pt", map_location="cpu").float()
        heard = (folder / "heard.txt").read_text().split("\n") if (folder / "heard.txt").exists() else []
        words = (folder / "words.txt").read_text().split("\n")
        lm = None
        if meta.get("lm") and (folder / "lm.npz").exists():
            from .lm import WordLM

            lm = WordLM.load(folder / "lm.npz", words)
        return cls(reader, words, W, punct, device, heard=heard, bonus=meta.get("bonus", 0.0), lm=lm, lm_cfg=meta.get("lm"))


# --- training (on Modal) --------------------------------------------------------


def libri(split: str, limit: int | None = None) -> list[dict]:
    """LibriSpeech clips from the Hugging Face parquet files: [{id, wav, words}]."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    folder = split if "/" in split else f"clean/{split}"  # "train.100" or "other/train.500"
    files = sorted(f for f in HfApi().list_repo_files(LIBRI, repo_type="dataset") if f.startswith(f"{folder}/") and f.endswith(".parquet"))
    out = []
    for f in files:
        table = pq.read_table(hf_hub_download(LIBRI, f, repo_type="dataset"), columns=["id", "audio", "text"])
        for row in table.to_pylist():
            out.append({"id": row["id"], "wav": row["audio"]["bytes"], "words": clean_words(row["text"]), "source": "libri"})
            if limit and len(out) >= limit:
                return out
    return out


class Clips(torch.utils.data.Dataset):
    """Decoded, perhaps phone-line, 16 kHz clips and their word ids."""

    def __init__(self, rows: list[dict], index: dict[str, int], train: bool, phone_share: float = 0.5):
        self.rows, self.index, self.train, self.phone_share = rows, index, train, phone_share

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from .voice import load, phone

        r = self.rows[i]
        x = load(r["wav"])
        if r["source"] == "voicemail":
            seed = random.randrange(1 << 30) if self.train else zlib.crc32(r["id"].encode())
            if not self.train or random.random() < self.phone_share:
                x = phone(x, seed)
        if self.train:  # loudness and a little noise
            x = x * random.uniform(0.5, 1.2) + np.random.normal(0, random.uniform(0, 0.003), len(x)).astype(np.float32)
        return x.astype(np.float32), [self.index[w] + 1 for w in r["words"]], r["id"]


def batches(rows: list[dict], seconds: float, seed: int) -> list[list[int]]:
    """Index batches of about `seconds` of audio, clips of a similar length together."""
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    out = []
    for i in range(0, len(order), 256):
        group = sorted(order[i:i + 256], key=lambda k: rows[k]["seconds"])
        cur, longest = [], 0.0
        for k in group:
            sec = rows[k]["seconds"]
            if cur and max(longest, sec) * (len(cur) + 1) > seconds:
                out.append(cur)
                cur, longest = [], 0.0
            cur.append(k)
            longest = max(longest, sec)
        if cur:
            out.append(cur)
    rng.shuffle(out)
    return out


def seconds_of(wav: bytes) -> float:
    import soundfile as sf

    info = sf.info(io.BytesIO(wav))
    return info.frames / info.samplerate


def evaluate(t: Transcriber, rows: list[dict], log=print, name: str = "") -> dict:
    from .voice import load, phone, wer

    edits = total = 0
    samples, started, audio_s = [], time.time(), 0.0
    for r in rows:
        x = load(r["wav"])
        if r["source"] == "voicemail":
            x = phone(x, zlib.crc32(r["id"].encode()))
        audio_s += len(x) / 16000
        res = t.transcribe([x])[0]
        e, n = wer(r["words"], res["words"])
        edits, total = edits + e, total + n
        if len(samples) < 4:
            samples.append({"id": r["id"], "truth": " ".join(r["words"]), "heard": res["text"]})
    out = {"clips": len(rows), "wer": round(edits / max(1, total), 4), "real_time": round(audio_s / max(1e-6, time.time() - started), 1), "samples": samples}
    log(f"    {name}: WER {out['wer']:.1%} on {len(rows)} clips, {out['real_time']}× real time")
    return out


def train_punctuator(texts: list[str], val_texts: list[str], W: torch.Tensor, index: dict[str, int], device: str, log=print, epochs: int = 8) -> tuple[Punctuator, dict]:
    def encode(ts):
        out = []
        for t in ts:
            lab = [x for x in punct_labels(t) if x[0] in index]
            if len(lab) >= 3:
                out.append((torch.tensor([index[x[0]] for x in lab]), torch.tensor([x[1] for x in lab]), torch.tensor([x[2] for x in lab])))
        return out

    train, val = encode(texts), encode(val_texts)
    model = Punctuator().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    Wd = W.to(device)

    def run(rows, learn):
        model.train(learn)
        right_c = right_p = n = 0
        for i in range(0, len(rows), 32):
            chunk = rows[i:i + 32]
            pad = nn.utils.rnn.pad_sequence
            ids, case, punct = (pad([c[k] for c in chunk], batch_first=True, padding_value=-1).to(device) for k in range(3))
            with torch.set_grad_enabled(learn):
                lc, lp = model(Wd[ids.clamp(min=0)])
                loss = nn.functional.cross_entropy(lc.transpose(1, 2), case, ignore_index=-1) + nn.functional.cross_entropy(lp.transpose(1, 2), punct, ignore_index=-1)
            if learn:
                loss.backward()
                opt.step()
                opt.zero_grad()
            keep = ids >= 0
            right_c += (lc.argmax(-1) == case)[keep].sum().item()
            right_p += (lp.argmax(-1) == punct)[keep].sum().item()
            n += keep.sum().item()
        return right_c / max(1, n), right_p / max(1, n)

    for epoch in range(epochs):
        random.shuffle(train)
        run(train, True)
    case_acc, punct_acc = run(val, False)
    log(f"    punctuation: {len(train)} texts; val: capitals right {case_acc:.1%}, punctuation right {punct_acc:.1%} (per word)")
    return model.eval(), {"case_accuracy": round(case_acc, 4), "punct_accuracy": round(punct_acc, 4), "texts": len(train)}


def train_and_judge(voice: dict, texts: dict, cfg: dict, out: Path, cache: Path, log=print) -> dict:
    """voice: {split: [{id, wav, words}]} fake voicemails; texts: {split: [written text]}
    for the punctuation; cfg: epochs, batch_seconds, lr, libri splits, reader size."""
    device = "cuda"
    started = time.time()
    torch.manual_seed(cfg.get("seed", 0))
    random.seed(cfg.get("seed", 0))
    report = {"task": "transcribe", "config": cfg, "ear": ENCODER}

    log("  LibriSpeech: loading")
    lib_train = [r for s in cfg.get("libri", ["train.100"]) for r in libri(s, limit=cfg.get("libri_limit"))]
    lib_test = libri("test", limit=cfg.get("libri_test", 400))
    lib_val = libri("validation", limit=200)
    vm = {s: [{**r, "source": "voicemail"} for r in rows] for s, rows in voice.items()}
    log(f"  {len(lib_train)} LibriSpeech clips + {len(vm['train'])} voicemails (×{cfg.get('voicemail_repeat', 4)}) to train")

    # The lexicon plus the words of the train and val audio. Not the raw
    # messages (they keep their typos, "nneed", "thx"), and not the test
    # words: a name that only the test says must not be in the list.
    words = word_list([], cache, [r["words"] for r in lib_train + lib_val + vm["train"] + vm["val"]])
    key = zlib.crc32("\n".join(words).encode())
    vpath = cache / f"word-vectors-{key:08x}.pt"
    if vpath.exists():
        W = torch.load(vpath)
    else:
        t = time.time()
        W = word_vectors(words, device)
        torch.save(W, vpath)
        log(f"  {len(words):,} word vectors in {time.time() - t:.0f} s")
    index = {w: i for i, w in enumerate(words)}
    report["words"] = len(words)

    # Training output layer: the words of the training texts and random others.
    train_rows = lib_train + vm["train"] * cfg.get("voicemail_repeat", 4)
    seen = sorted({w for r in train_rows for w in r["words"]})
    rng = random.Random(0)
    seen_set = set(seen)
    others = rng.sample([w for w in words if w not in seen_set], min(cfg.get("negatives", 20000), len(words) - len(seen)))
    train_words = seen + others
    tindex = {w: i for i, w in enumerate(train_words)}
    Wt = W[[index[w] for w in train_words]].to(device)
    log(f"  training output layer: {len(seen):,} heard words + {len(others):,} others; full list {len(words):,}")

    for r in train_rows:
        r.setdefault("seconds", seconds_of(r["wav"]))
    reader = Reader(**cfg.get("reader", {})).to(device)
    ear = Ear(device, all_layers=reader.cfg["mix"] > 0)
    report["reader_parameters"] = sum(p.numel() for p in reader.parameters())
    opt = torch.optim.AdamW(reader.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=0.01)
    epochs = cfg.get("epochs", 6)
    plan = [batches(train_rows, cfg.get("batch_seconds", 400), seed=e) for e in range(epochs)]
    steps = sum(len(p) for p in plan)
    warm = max(1, min(500, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    def collate(items):
        waves, targets, ids = zip(*items)
        f = ear.prepare(list(waves))
        return f, list(targets), list(ids)

    data = Clips(train_rows, tindex, train=True)
    step, t0 = 0, time.time()
    log(f"  reader: {report['reader_parameters']:,} numbers; {epochs} epochs, {steps} steps")
    for epoch in range(epochs):
        loader = torch.utils.data.DataLoader(data, batch_sampler=plan[epoch], collate_fn=collate, num_workers=cfg.get("workers", 12), prefetch_factor=4, persistent_workers=False)
        reader.train()
        total = n = 0.0
        for f, targets, _ in loader:
            h, mask = ear.encode(f)
            if cfg.get("time_mask", 0.0) > 0:  # SpecAugment-like: hide short runs of slices
                drop = torch.rand(h.shape[:2], device=h.device) < cfg["time_mask"] / 5
                drop = torch.nn.functional.max_pool1d(drop.float().unsqueeze(1), 5, 1, 2).squeeze(1).bool()
                h = h.masked_fill(drop.unsqueeze(-1), 0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z, blank = reader(h, mask)
                logits = reader.logits(z, blank, Wt)
            logp = logits.float().log_softmax(-1).transpose(0, 1)
            lengths = mask.sum(-1)
            flat = torch.tensor([k for t in targets for k in t], device=device)
            tl = torch.tensor([len(t) for t in targets], device=device)
            loss = nn.functional.ctc_loss(logp, flat, lengths, tl, blank=0, reduction="mean", zero_infinity=True)
            loss.backward()
            nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            total, n, step = total + loss.item(), n + 1, step + 1
            if step % 200 == 0:
                log(f"    step {step}/{steps}: CTC loss {total / n:.3f} ({(time.time() - t0) / step:.2f} s/step)")
        log(f"    epoch {epoch + 1}/{epochs}: CTC loss {total / max(1, n):.3f}")
        tr = Transcriber(reader, words, W, None, device, ear=ear)
        evaluate(tr, vm["val"][:60], log, "val voicemails")
        reader.train()
    report["train_seconds"] = round(time.time() - t0, 1)

    # The bonus for heard words, chosen on val (voicemails and LibriSpeech).
    tr = Transcriber(reader, words, W, None, device, ear=ear, heard=seen)
    tune = vm["val"][:60] + lib_val[:100]
    scores = {}
    for b in cfg.get("bonuses", [0.0, 0.25, 0.5, 1.0]):
        tr.set_bonus(b)
        scores[b] = evaluate(tr, tune, log, f"val, word bonus {b}")["wer"]
    bonus = min(scores, key=scores.get)
    report["bonus"] = {"chosen": bonus, "val_wer": scores}
    log(f"  word bonus {bonus} (val WER {scores[bonus]:.1%}, without {scores[min(scores)]:.1%})")

    # The word LM: LibriSpeech train transcripts and the train voicemail
    # scripts (×20: they are few, and they are the clinic's words); alpha
    # and beta chosen on the same val clips.
    from .lm import WordLM, tune as tune_lm
    from .voice import load, phone

    tr.set_bonus(bonus)
    t = time.time()
    lm = WordLM(words).fit([r["words"] for r in lib_train] + [r["words"] for r in vm["train"]] * 20)
    log(f"  word LM: {len(lm.tri):,} trigrams, {len(lm.bi):,} bigrams in {time.time() - t:.0f} s")
    clips = [{"wav": phone(load(r["wav"]), zlib.crc32(r["id"].encode())) if r["source"] == "voicemail" else load(r["wav"]), "words": r["words"]} for r in tune]
    report["lm"] = tune_lm(tr, lm, clips, log)

    punct, report["punctuation"] = train_punctuator(texts["train"], texts["val"], W, index, device, log)
    tr = Transcriber(reader, words, W, punct, device, ear=ear, heard=seen, bonus=bonus, lm=tr.lm, lm_cfg=tr.lm_cfg)
    if tr.lm is not None:  # the test without the LM too, to show what it adds
        with_lm = tr.lm
        tr.lm = None
        report["test_greedy"] = {"voicemail": evaluate(tr, vm["test"], log, "test voicemails (phone line), greedy")}
        tr.lm = with_lm
    report["test"] = {"voicemail": evaluate(tr, vm["test"], log, "test voicemails (phone line)"),
                      "librispeech": evaluate(tr, lib_test, log, "LibriSpeech test-clean"),
                      "librispeech_other": evaluate(tr, libri("other/test", limit=cfg.get("libri_test", 400)), log, "LibriSpeech test-other")}
    report["gpu"] = torch.cuda.get_device_name()
    report["seconds"] = round(time.time() - started, 1)
    tr.save(out, {"report": {k: report[k] for k in ("words", "reader_parameters")}})
    return report
