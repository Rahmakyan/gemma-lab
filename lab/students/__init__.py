"""Students: local models that learn the teacher's decisions.

Every student has the same interface, so one can replace the other:

    student = make("lora")                     # or "embed"
    student.fit(train_rows, val_rows, questions, **cfg)  # rows: {message, target: {question: [p per option]}}
    student.predict(texts, questions)          # -> [{question: [p per option]}], options in criteria order
    student.save(folder); load(folder)

- lora:  Gemma 3 1B as a decision model (one forward pass, option logits),
         taught with a LoRA adapter and a temperature per question.
         Answers any question, also without training (plain Gemma).
- embed: EmbeddingGemma vectors and one logistic regression per question.
         Answers only the questions it was trained on.
"""

from __future__ import annotations

import json
from pathlib import Path

KINDS = ("lora", "embed")


def make(kind: str, model: str | None = None):
    if kind == "lora":
        from .lora import LoraStudent

        return LoraStudent(model) if model else LoraStudent()
    if kind == "embed":
        from .embed import EmbedStudent

        return EmbedStudent(model) if model else EmbedStudent()
    raise SystemExit(f"unknown student {kind!r}: use one of {', '.join(KINDS)}")


def load(folder: Path):
    meta = json.loads((folder / "student.json").read_text())
    if meta["kind"] == "lora":
        from .lora import LoraStudent

        return LoraStudent.load(folder)
    from .embed import EmbedStudent

    return EmbedStudent.load(folder)
