from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from bicause_gaze.datasets import VideoClipDataset
from bicause_gaze.engine import evaluate, train_one_epoch
from bicause_gaze.losses import HeadToGaze
from bicause_gaze.models import BiCauseGazeDetector
from bicause_gaze.utils.io import ensure_dir, load_yaml
from bicause_gaze.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="runs/bicause_gaze")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = VideoClipDataset(
        root=str(cfg.get("train_root", "")),
        clip_len=int(cfg["clip_len"]),
        image_size=int(cfg["image_size"]),
        synthetic_if_missing=bool(cfg.get("synthetic_if_missing", True)),
    )
    val_set = VideoClipDataset(
        root=str(cfg.get("val_root", "")),
        clip_len=int(cfg["clip_len"]),
        image_size=int(cfg["image_size"]),
        synthetic_if_missing=bool(cfg.get("synthetic_if_missing", True)),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    model = BiCauseGazeDetector(num_classes=int(cfg.get("num_classes", 2))).to(device)
    mapper = HeadToGaze().to(device)

    optimizer = Adam(list(model.parameters()) + list(mapper.parameters()), lr=float(cfg["lr"]))

    save_dir = ensure_dir(args.save_dir)
    best_acc = -1.0
    epochs = int(cfg["epochs"])

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(train_loader, model, mapper, optimizer, device, cfg)
        val_metrics = evaluate(val_loader, model, device)

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f}"
        )

        ckpt = {
            "model": model.state_dict(),
            "mapper": mapper.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "cfg": cfg,
        }
        torch.save(ckpt, Path(save_dir) / "last.pt")

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            torch.save(ckpt, Path(save_dir) / "best.pt")

    print(f"Training done. best_val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
