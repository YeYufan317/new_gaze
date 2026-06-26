from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from bicause_gaze.datasets import VideoClipDataset
from bicause_gaze.engine import evaluate
from bicause_gaze.models import BiCauseGazeDetector
from bicause_gaze.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_set = VideoClipDataset(
        root=str(cfg.get("val_root", "")),
        clip_len=int(cfg["clip_len"]),
        image_size=int(cfg["image_size"]),
        synthetic_if_missing=bool(cfg.get("synthetic_if_missing", True)),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    model = BiCauseGazeDetector(num_classes=int(cfg.get("num_classes", 2))).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    metrics = evaluate(val_loader, model, device)
    print(f"Eval done. loss={metrics['loss']:.4f}, acc={metrics['acc']:.4f}")


if __name__ == "__main__":
    main()
