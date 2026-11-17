"""EmbeddingGemma vectors and a logistic regression per question.

EmbeddingGemma cannot answer a question: it only turns a text into a
vector (768 numbers). A small classifier on top learns the answers from
the teacher's labels: per question one linear layer, softmax over the
options, fitted to the teacher's full distribution (soft labels) with an
L2 penalty chosen on the val rows. A noul is two options (yes, no); a
score's levels are its options, and its score is the expected level.

Two ways to teach it (cfg tune_epochs):

- tune_epochs 0: the model is frozen; only the regressions learn (a few
  thousand numbers), in seconds.
- tune_epochs N: first the frozen regressions as above, then a LoRA adapter
  on EmbeddingGemma itself learns together with the heads (soft-label loss
  of all questions at once), so the vectors themselves move toward the
  teacher's distinctions. The epoch with the lowest val loss is kept, then
  the regressions are fitted again on the tuned vectors.

A question it was not trained on has no regression, so it cannot answer
that question.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .common import pick_device, soft_ce, spec

MODEL = os.environ.get("EMBED_MODEL", "unsloth/embeddinggemma-300m")  # google/embeddinggemma-300m without the login
MODEL2 = "unsloth/embeddinggemma-2"  # EmbeddingGemma 2: its text side only (270M), no image or audio encoder
PENALTIES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
TARGETS = r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
MAX_TOKENS = 512


def load_st(model: str, device: str):
    """A SentenceTransformer; for EmbeddingGemma 2 without its image and
    audio encoders (they are not needed for text)."""
    from sentence_transformers import SentenceTransformer

    extra = {"config_kwargs": {"vision_config": None, "audio_config": None}} if "embeddinggemma-2" in model else {}
    return SentenceTransformer(model, device=device, **extra)


class EmbedStudent:
    kind = "embed"

    def __init__(self, model: str = MODEL):
        self.name = model
        self.device, _ = pick_device()
        self.model = load_st(model, self.device)
        self.model.max_seq_length = MAX_TOKENS
        self.heads: dict[str, dict] = {}  # question -> {W, b, penalty}
        self.specs: dict[str, dict] = {}  # question -> the question it learned
        self.tuned = None  # the LoRA config when the model itself was tuned

    @property
    def prefix(self) -> str:
        return (self.model.prompts or {}).get("Classification", "")

    def vectors(self, texts: list[str]) -> torch.Tensor:
        self.model.eval()
        v = self.model.encode([self.prefix + t[:4000] for t in texts], normalize_embeddings=True, batch_size=16, convert_to_numpy=True)
        return torch.tensor(np.asarray(v), dtype=torch.float32)

    def vectors_with_grad(self, texts: list[str]) -> torch.Tensor:
        """The same vectors as vectors(), with gradients (for tuning)."""
        features = self.model.preprocess([self.prefix + t[:4000] for t in texts])
        features = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in features.items()}
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            out = self.model(features)["sentence_embedding"]
        return out.float()

    @staticmethod
    def regression(x: torch.Tensor, y: torch.Tensor, penalty: float) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.zeros(x.shape[1], y.shape[1], requires_grad=True)
        b = torch.zeros(y.shape[1], requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1, max_iter=500, line_search_fn="strong_wolfe")

        def loss():
            opt.zero_grad()
            value = soft_ce(x @ w + b, y) + penalty * (w ** 2).sum()
            value.backward()
            return value

        opt.step(loss)
        return w.detach(), b.detach()

    def fit(self, train: list[dict], val: list[dict], questions: dict, log=print, tune_epochs: int = 0, rank: int = 16,
            lr: float = 2e-4, head_lr: float = 2e-3, batch: int = 16, seed: int = 0, **_):
        self.fit_heads(train, val, questions, log)
        if tune_epochs:
            self.tune(train, val, questions, tune_epochs, rank, lr, head_lr, batch, seed, log)
            log("    heads again, on the tuned vectors:")
            self.fit_heads(train, val, questions, log)

    def fit_heads(self, train: list[dict], val: list[dict], questions: dict, log=print):
        xt, xv = self.vectors([r["message"] for r in train]), self.vectors([r["message"] for r in val])
        for name in questions:
            yt = torch.tensor([r["target"][name] for r in train])
            yv = torch.tensor([r["target"][name] for r in val])
            best = None
            for penalty in PENALTIES:
                w, b = self.regression(xt, yt, penalty)
                loss = soft_ce(xv @ w + b, yv).item()
                if best is None or loss < best[0]:
                    best = (loss, penalty, w, b)
            self.heads[name] = {"W": best[2], "b": best[3], "penalty": best[1]}
            self.specs[name] = spec(questions[name])
            log(f"    {name}: L2 penalty {best[1]:g}, val loss {best[0]:.4f}")

    def tune(self, train, val, questions, epochs, rank, lr, head_lr, batch, seed, log):
        """LoRA on EmbeddingGemma and the heads together, from the frozen fit."""
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(seed)
        self.tuned = {"rank": rank, "targets": TARGETS, "epochs": epochs, "lr": lr}
        enc = self.model[0]
        for p in self.model.parameters():  # only the adapter and the heads learn (not the Dense layers)
            p.requires_grad_(False)
        enc.model = get_peft_model(enc.model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=TARGETS))
        names = list(questions)
        heads = {n: [self.heads[n]["W"].clone().to(self.device).requires_grad_(), self.heads[n]["b"].clone().to(self.device).requires_grad_()] for n in names}
        lora = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW([{"params": lora, "lr": lr}, {"params": [p for h in heads.values() for p in h], "lr": head_lr, "weight_decay": 0.0}])
        steps = epochs * ((len(train) + batch - 1) // batch)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps))
        rng = random.Random(seed)
        rows = list(train)

        def loss_of(chunk):
            x = self.vectors_with_grad([r["message"] for r in chunk])
            return sum(soft_ce(x @ heads[n][0] + heads[n][1], torch.tensor([r["target"][n] for r in chunk], device=x.device)) for n in names) / len(names)

        def val_loss():
            self.model.eval()
            with torch.no_grad():
                total = sum(loss_of(val[i:i + 32]).item() * len(val[i:i + 32]) for i in range(0, len(val), 32))
            return total / len(val)

        best = (val_loss(), None)
        log(f"    tune: {sum(p.numel() for p in lora):,} LoRA numbers on EmbeddingGemma, val loss before {best[0]:.4f}")
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
        else:  # no epoch beat the frozen model: switch the adapter off
            for k, t in enc.model.state_dict().items():
                if "lora_B" in k:
                    t.zero_()
        self.model.eval()

    def predict(self, texts: list[str], questions: dict) -> list[dict]:
        # A head reads its own options in its own order: a question with the
        # same name but other options or words would get wrong labels.
        missing = [n for n, q in questions.items() if n not in self.heads or self.specs.get(n) != spec(q)]
        if missing:
            raise KeyError(f"the embed student was not trained on these questions as asked (name, type, instructions and criteria must match): {', '.join(missing)}")
        x = self.vectors(texts)
        out = [{} for _ in texts]
        for name in questions:
            h = self.heads[name]
            for row, p in zip(out, torch.softmax(x @ h["W"] + h["b"], -1).tolist()):
                row[name] = p
        return out

    def size(self) -> dict:
        lora = sum(p.numel() for n, p in self.model.named_parameters() if "lora_" in n)
        return {"parameters": sum(p.numel() for p in self.model.parameters()) - lora,
                "trained_parameters": lora + sum(h["W"].numel() + h["b"].numel() for h in self.heads.values())}

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({n: {"W": h["W"].cpu(), "b": h["b"].cpu()} for n, h in self.heads.items()}, folder / "heads.pt")
        if self.tuned:
            self.model[0].model.save_pretrained(str(folder / "encoder"))
        (folder / "student.json").write_text(json.dumps({"kind": self.kind, "model": self.name, "penalty": {n: h["penalty"] for n, h in self.heads.items()},
                                                         "questions": self.specs, "tuned": self.tuned}))

    @classmethod
    def load(cls, folder: Path):
        meta = json.loads((folder / "student.json").read_text())
        s = cls(meta["model"])
        if meta.get("tuned"):
            from peft import PeftModel

            # Merged into the weights: as fast as the plain model.
            s.model[0].model = PeftModel.from_pretrained(s.model[0].model, str(folder / "encoder")).merge_and_unload()
            s.tuned = meta["tuned"]
        heads = torch.load(folder / "heads.pt")
        s.heads = {n: {**h, "penalty": meta["penalty"].get(n)} for n, h in heads.items()}
        s.specs = meta.get("questions", {})
        return s
