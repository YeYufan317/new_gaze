from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from bicause_gaze.engine.metrics import batch_accuracy
from bicause_gaze.losses import (
    HeadToGaze,
    causal_consistency_loss,
    micro_dynamic_loss,
    physical_consistency_loss,
)


def train_one_epoch(
    loader,
    model,
    mapper: HeadToGaze,
    optimizer,
    device: torch.device,
    cfg: Dict,
) -> Dict[str, float]:
    model.train()
    mapper.train()

    meters = {"loss": 0.0, "cls": 0.0, "phy": 0.0, "causal": 0.0, "micro": 0.0, "acc": 0.0}
    count = 0

    for batch in loader:
        left_eye = batch["left_eye"].to(device)
        right_eye = batch["right_eye"].to(device)
        label = batch["label"].to(device)

        out = model(left_eye, right_eye)
        logits = out["logits"]
        pose = out["pose"]
        g_l = out["gaze_l"]
        g_r = out["gaze_r"]

        l_cls = F.cross_entropy(logits, label)
        l_phy = physical_consistency_loss(g_l, pose) + physical_consistency_loss(g_r, pose)
        l_causal = causal_consistency_loss(g_l, g_r, pose, mapper)
        l_micro = micro_dynamic_loss(g_l, g_r)

        loss = (
            cfg["w_cls"] * l_cls
            + cfg["w_phy"] * l_phy
            + cfg["w_causal"] * l_causal
            + cfg["w_micro"] * l_micro
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        acc = batch_accuracy(logits.detach(), label.detach())
        meters["loss"] += float(loss.item())
        meters["cls"] += float(l_cls.item())
        meters["phy"] += float(l_phy.item())
        meters["causal"] += float(l_causal.item())
        meters["micro"] += float(l_micro.item())
        meters["acc"] += acc
        count += 1

    if count == 0:
        return {k: 0.0 for k in meters}
    return {k: v / count for k, v in meters.items()}
