from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EyeEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, feat_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feat_dim, 3, 2, 1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class HeadPoseNet(nn.Module):
    def __init__(self, in_dim: int = 512) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),  # yaw pitch roll
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.mlp(feat)


class Gaze3DNet(nn.Module):
    def __init__(self, in_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        g = self.mlp(feat)
        return F.normalize(g, p=2, dim=-1)


class BiCauseGazeDetector(nn.Module):
    def __init__(self, feat_dim: int = 256, num_classes: int = 2) -> None:
        super().__init__()
        self.left_eye_enc = EyeEncoder(feat_dim=feat_dim)
        self.right_eye_enc = EyeEncoder(feat_dim=feat_dim)

        self.head_pose = HeadPoseNet(in_dim=feat_dim * 2)
        self.gaze_left = Gaze3DNet(in_dim=feat_dim)
        self.gaze_right = Gaze3DNet(in_dim=feat_dim)

        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim * 2 + 9, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, left_eye: torch.Tensor, right_eye: torch.Tensor) -> dict:
        # left/right: [B,T,3,H,W]
        b, t, c, h, w = left_eye.shape
        l = left_eye.reshape(b * t, c, h, w)
        r = right_eye.reshape(b * t, c, h, w)

        fl = self.left_eye_enc(l)
        fr = self.right_eye_enc(r)

        pose = self.head_pose(torch.cat([fl, fr], dim=-1))
        g_l = self.gaze_left(fl)
        g_r = self.gaze_right(fr)

        z = torch.cat([fl, fr, pose, g_l, g_r], dim=-1)
        logits = self.cls_head(z).reshape(b, t, -1).mean(dim=1)

        return {
            "logits": logits,
            "pose": pose.reshape(b, t, 3),
            "gaze_l": g_l.reshape(b, t, 3),
            "gaze_r": g_r.reshape(b, t, 3),
        }
