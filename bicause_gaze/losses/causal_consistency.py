from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadToGaze(nn.Module):
    """Learn f(head_pose)->gaze as structural prior."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(pose), p=2, dim=-1)


def causal_consistency_loss(
    gaze_left: torch.Tensor,
    gaze_right: torch.Tensor,
    pose: torch.Tensor,
    mapper: HeadToGaze,
) -> torch.Tensor:
    """
    Consistency of residuals under shared causal parent(head pose).
    Inputs: [B,T,3]
    """
    b, t, _ = pose.shape
    flat_pose = pose.reshape(b * t, 3)
    f_head = mapper(flat_pose).reshape(b, t, 3)

    eps_l = gaze_left - f_head
    eps_r = gaze_right - f_head
    return F.l1_loss(eps_l, eps_r)
