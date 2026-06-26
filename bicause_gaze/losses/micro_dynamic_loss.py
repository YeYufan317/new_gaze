from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_diff(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:] - x[:, :-1]


def micro_dynamic_loss(
    gaze_left: torch.Tensor,
    gaze_right: torch.Tensor,
    min_var: float = 1e-4,
    max_var: float = 5e-2,
) -> torch.Tensor:
    """
    Encourage realistic micro-eye dynamics:
    - not over-smooth (variance too low)
    - not unstable jitter (variance too high)
    - balanced frequency energy
    """
    dl = temporal_diff(gaze_left)
    dr = temporal_diff(gaze_right)
    d = torch.cat([dl, dr], dim=-1)  # [B, T-1, 6]

    var = d.var(dim=1).mean(dim=-1)
    low_penalty = F.relu(min_var - var).mean()
    high_penalty = F.relu(var - max_var).mean()

    spec = torch.fft.rfft(d, dim=1).abs().mean(dim=-1)
    if spec.shape[1] > 2:
        n = max(spec.shape[1] // 3, 1)
        low = spec[:, :n].mean()
        high = spec[:, -n:].mean()
        freq_penalty = torch.abs((high / (low + 1e-6)) - 0.35)
    else:
        freq_penalty = d.new_tensor(0.0)

    return low_penalty + high_penalty + 0.1 * freq_penalty
