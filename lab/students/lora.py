"""Gemma 3 1B as a decision model, taught with LoRA.

The decision model (nishtahir.com/build-your-own-decision-model): the prompt
ends where the answer starts; one forward pass; only the logits of the
option tokens are read ("Yes"/"No", or a letter per option or level) and
softmaxed over that small set. Nothing is generated.

Teaching: a LoRA adapter (on every linear layer by default, or only the
attention's q and v), with the loss on the option logits alone, against the
teacher's full distribution (soft labels). After each epoch the val loss is
measured and the best epoch is kept. Then a temperature per question,
fitted on the val rows.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch

from .common import fit_temperature, options, pick_device, soft_ce, spec

MODEL = os.environ.get("GEMMA_MODEL", "unsloth/gemma-3-1b-it")  # google/gemma-3-1b-it without the login
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_INPUT_TOKENS = 1024
# "all" (every linear layer of attention and MLP) learns more than "qv"
# (the q and v projections, the first version) for the same rank.
TARGETS = {"qv": r".*\.(q_proj|v_proj)$", "all": r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"}


class LoraStudent:
    kind = "lora"

    def __init__(self, model: str = MODEL):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model
        self.device, self.dtype = pick_device()
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=self.dtype).to(self.device).eval()
        self.peft = None
        self.temperature: dict[str, float] = {}
        self.specs: dict[str, dict] = {}  # question -> the question its temperature was fitted for
        self.yes_no = [self.first_token("Yes"), self.first_token("No")]
        self.letters = [self.first_token(x) for x in LETTERS]

    def first_token(self, text: str) -> int:
        return self.tok.encode(text, add_special_tokens=False)[0]

    @property
    def net(self):
        return self.peft if self.peft is not None else self.model

    # --- the decision model -------------------------------------------------

    def option_ids(self, q: dict) -> list[int]:
        return self.yes_no if q["type"] == "noul" else self.letters[: len(options(q))]

    def prompt(self, text: str, q: dict) -> list[int]:
        body = self.tok.decode(self.tok.encode(text, add_special_tokens=False)[:MAX_INPUT_TOKENS])
        user = f"Read the message, then answer the question about it.\n\n<message>\n{body}\n</message>\n\nQuestion: {q['instructions']}\n"
        if q["type"] == "noul":
            user += "Answer with Yes or No."
        else:
            items = q["criteria"].items() if isinstance(q["criteria"], dict) else enumerate(q["criteria"])
            lines = [f"{LETTERS[i]}) {k}: {v}" if q["type"] == "choice" else f"{LETTERS[i]}) {v}" for i, (k, v) in enumerate(items)]
            user += "Options:\n" + "\n".join(lines) + "\nAnswer with the letter only."
        rendered = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        return self.tok.encode(rendered, add_special_tokens=False)

    def batch(self, rows: list[list[int]]) -> dict:
        width = max(len(r) for r in rows)
        pad = self.tok.pad_token_id or 0
        ids = torch.tensor([[pad] * (width - len(r)) + r for r in rows], device=self.device)
        mask = torch.tensor([[0] * (width - len(r)) + [1] * len(r) for r in rows], device=self.device)
        return {"input_ids": ids, "attention_mask": mask, "position_ids": (mask.cumsum(-1) - 1).clamp(min=0)}

    def option_logits(self, prompts: list[list[int]], ids: list[list[int]]) -> list[torch.Tensor]:
        """One forward pass per prompt: the last position's logits at each
        prompt's own option tokens."""
        last = self.net(**self.batch(prompts), logits_to_keep=1).logits[:, -1].float()
        return [last[i, row] for i, row in enumerate(ids)]

    @torch.inference_mode()
    def raw_logits(self, texts: list[str], q: dict, size: int = 8) -> torch.Tensor:
        ids = self.option_ids(q)
        out = []
        for i in range(0, len(texts), size):
            prompts = [self.prompt(t, q) for t in texts[i : i + size]]
            out += [x.cpu() for x in self.option_logits(prompts, [ids] * len(prompts))]
        return torch.stack(out)

    def predict(self, texts: list[str], questions: dict) -> list[dict]:
        out = [{} for _ in texts]
        for name, q in questions.items():
            # A temperature holds only for the question it was fitted on; a
            # new or changed question is answered plainly (T = 1).
            t = self.temperature.get(name, 1.0) if self.specs.get(name) == spec(q) else 1.0
            probs = torch.softmax(self.raw_logits(texts, q) / t, -1)
            for row, p in zip(out, probs.tolist()):
                row[name] = p
        return out

    # --- teaching -------------------------------------------------------------

    @staticmethod
    def items(rows: list[dict], questions: dict) -> list[tuple[str, str, dict, list[float]]]:
        """(text, name, question, target): one training row per (message, question)."""
        return [(r["message"], name, q, r["target"][name]) for r in rows for name, q in questions.items()]

    @torch.inference_mode()
    def items_logits(self, items: list, size: int = 8) -> list[torch.Tensor]:
        out = []
        for i in range(0, len(items), size):
            chunk = items[i : i + size]
            out += [x.cpu() for x in self.option_logits([self.prompt(c[0], c[2]) for c in chunk], [self.option_ids(c[2]) for c in chunk])]
        return out

    def items_loss(self, items: list) -> float:
        logits = self.items_logits(items)
        return sum(soft_ce(l.unsqueeze(0), torch.tensor([c[3]])).item() for l, c in zip(logits, items)) / max(1, len(items))

    def fit(self, train: list[dict], val: list[dict], questions: dict, log=print, **cfg):
        """Teaches the questions on the rows, then a temperature per question."""
        self.fit_items(self.items(train, questions), self.items(val, questions), log=log, **cfg)
        texts = [r["message"] for r in val]
        for name, q in questions.items():
            target = torch.tensor([r["target"][name] for r in val])
            self.temperature[name] = fit_temperature(self.raw_logits(texts, q), target)
            self.specs[name] = spec(q)
        log(f"    temperatures {', '.join(f'{k} {v:.2f}' for k, v in self.temperature.items())}")

    def fit_items(self, train: list, val: list, epochs: int = 3, rank: int = 16, lr: float = 2e-4, batch: int = 8, seed: int = 0,
                  targets: str = "all", log=print):
        """The LoRA adapter on (text, name, question, target) rows; any mix of
        questions. The epoch with the lowest val loss is kept."""
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(seed)
        self.peft = get_peft_model(self.model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, target_modules=TARGETS[targets]))
        opt = torch.optim.AdamW([p for p in self.peft.parameters() if p.requires_grad], lr=lr)
        rows = [(self.prompt(text, q), self.option_ids(q), torch.tensor(target, dtype=torch.float32)) for text, _, q, target in train]
        steps = epochs * ((len(rows) + batch - 1) // batch)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
        rng = random.Random(seed)
        best = None
        for epoch in range(epochs):
            self.peft.train()
            rng.shuffle(rows)
            total = 0.0
            for i in range(0, len(rows), batch):
                chunk = rows[i : i + batch]
                logits = self.option_logits([c[0] for c in chunk], [c[1] for c in chunk])
                # Rows with different option counts: the loss row by row.
                loss = sum(soft_ce(l.unsqueeze(0), c[2].to(l.device).unsqueeze(0)) for l, c in zip(logits, chunk)) / len(chunk)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.peft.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                total += loss.item() * len(chunk)
            self.peft.eval()
            v = self.items_loss(val)
            log(f"    epoch {epoch + 1}/{epochs}: loss {total / len(rows):.4f}, val loss {v:.4f}")
            if best is None or v < best[0]:
                best = (v, epoch + 1, {k: t.detach().clone() for k, t in self.peft.state_dict().items() if "lora_" in k})
        self.peft.load_state_dict(best[2], strict=False)
        self.peft.eval()
        log(f"    kept epoch {best[1]} (val loss {best[0]:.4f})")

    def size(self) -> dict:
        trained = sum(p.numel() for p in self.net.parameters() if p.requires_grad) if self.peft is not None else 0
        # peft puts the LoRA weights into self.model: the base is the rest.
        return {"parameters": sum(p.numel() for p in self.model.parameters()) - trained, "trained_parameters": trained}

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        if self.peft is not None:
            self.peft.save_pretrained(str(folder))
        (folder / "student.json").write_text(json.dumps({"kind": self.kind, "model": self.name, "temperature": self.temperature, "questions": self.specs}))

    @classmethod
    def load(cls, folder: Path):
        from peft import PeftModel

        meta = json.loads((folder / "student.json").read_text())
        s = cls(meta["model"])
        if (folder / "adapter_config.json").exists():
            # Merged into the weights: as fast as plain Gemma (LoRA on every
            # layer, unmerged, is about 1.6 times slower).
            s.model = PeftModel.from_pretrained(s.model, str(folder)).merge_and_unload().eval()
        s.temperature = meta.get("temperature", {})
        s.specs = meta.get("questions", {})
        return s
