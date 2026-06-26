from __future__ import annotations

import torch


def batch_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=-1)
    correct = (pred == target).float().mean().item()
    return float(correct)
