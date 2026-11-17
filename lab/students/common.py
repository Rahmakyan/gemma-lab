from __future__ import annotations

import torch


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def options(q: dict) -> list[str]:
    """The option names in order: [yes, no] for a noul, the criteria keys
    for a choice, the level numbers for a score."""
    if q["type"] == "noul":
        return ["yes", "no"]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return list(q["criteria"])


def spec(q: dict) -> dict:
    """What makes two questions the same question: type, instructions and
    options in order. A student's learning for a name holds only for it.
    Criteria become a list of pairs: two dicts with the same keys in another
    order are equal in Python, but the order is the order of the options."""
    c = q.get("criteria")
    return {"type": q["type"], "instructions": q["instructions"], "criteria": [list(kv) for kv in c.items()] if isinstance(c, dict) else c}


def soft_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against the teacher's distribution: KL(teacher ||
    student) plus a constant, so lowering it copies the whole distribution,
    not only the winner."""
    return -(target * torch.log_softmax(logits, -1)).sum(-1).mean()


def fit_temperature(logits: torch.Tensor, target: torch.Tensor) -> float:
    """nishtahir's calibration step: one temperature T in softmax(logits / T),
    fitted to the teacher's probabilities on held-out rows."""
    # The logits come from inference mode, which autograd cannot use: a
    # clone made here is a normal tensor.
    logits, target = logits.detach().clone().float().cpu(), target.detach().clone().float().cpu()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def loss():
        opt.zero_grad()
        value = soft_ce(logits / log_t.exp(), target)
        value.backward()
        return value

    opt.step(loss)
    return float(log_t.exp().clamp(0.05, 20).item())
