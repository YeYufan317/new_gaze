from __future__ import annotations

import torch
import torch.nn.functional as F


def euler_to_rotmat(ypr: torch.Tensor) -> torch.Tensor:
    """Convert yaw-pitch-roll (rad) to rotation matrix."""
    yaw, pitch, roll = ypr[..., 0], ypr[..., 1], ypr[..., 2]

    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    ry = torch.stack(
        [
            torch.stack([cy, torch.zeros_like(cy), sy], dim=-1),
            torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
            torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1),
        ],
        dim=-2,
    )

    rp = torch.stack(
        [
            torch.stack([torch.ones_like(cp), torch.zeros_like(cp), torch.zeros_like(cp)], dim=-1),
            torch.stack([torch.zeros_like(cp), cp, -sp], dim=-1),
            torch.stack([torch.zeros_like(cp), sp, cp], dim=-1),
        ],
        dim=-2,
    )

    rr = torch.stack(
        [
            torch.stack([cr, -sr, torch.zeros_like(cr)], dim=-1),
            torch.stack([sr, cr, torch.zeros_like(cr)], dim=-1),
            torch.stack([torch.zeros_like(cr), torch.zeros_like(cr), torch.ones_like(cr)], dim=-1),
        ],
        dim=-2,
    )

    return rr @ rp @ ry


def physical_consistency_loss(gaze: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """
    Enforce gaze vector consistency with expected direction from head pose.
    gaze: [B, T, 3], pose: [B, T, 3]
    """
    canonical = torch.tensor([0.0, 0.0, 1.0], device=gaze.device, dtype=gaze.dtype)
    g0 = canonical.view(1, 1, 3, 1).expand(gaze.size(0), gaze.size(1), 3, 1)

    rot = euler_to_rotmat(pose)
    g_exp = (rot @ g0).squeeze(-1)

    gaze_n = F.normalize(gaze, p=2, dim=-1)
    g_exp_n = F.normalize(g_exp, p=2, dim=-1)
    return F.smooth_l1_loss(gaze_n, g_exp_n)
