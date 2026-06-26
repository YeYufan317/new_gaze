from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from bicause_gaze.engine.metrics import batch_accuracy


def evaluate(loader, model, device: torch.device) -> Dict[str, float]:
    model.eval()
    meters = {"loss": 0.0, "acc": 0.0}
    count = 0

    with torch.no_grad():
        for batch in loader:
            left_eye = batch["left_eye"].to(device)
            right_eye = batch["right_eye"].to(device)
            label = batch["label"].to(device)

            out = model(left_eye, right_eye)
            logits = out["logits"]

            loss = F.cross_entropy(logits, label)
            acc = batch_accuracy(logits, label)

            meters["loss"] += float(loss.item())
            meters["acc"] += acc
            count += 1

    if count == 0:
        return {k: 0.0 for k in meters}
    return {k: v / count for k, v in meters.items()}
