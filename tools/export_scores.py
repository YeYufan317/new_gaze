from __future__ import annotations

import argparse
import csv

import torch
from torch.utils.data import DataLoader

from bicause_gaze.datasets import VideoClipDataset
from bicause_gaze.models import BiCauseGazeDetector
from bicause_gaze.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="scores.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = VideoClipDataset(
        root=str(cfg.get("val_root", "")),
        clip_len=int(cfg["clip_len"]),
        image_size=int(cfg["image_size"]),
        synthetic_if_missing=bool(cfg.get("synthetic_if_missing", True)),
    )
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False)

    model = BiCauseGazeDetector(num_classes=int(cfg.get("num_classes", 2))).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    rows = [("index", "label", "fake_prob", "real_prob")]
    idx = 0
    with torch.no_grad():
        for batch in loader:
            left_eye = batch["left_eye"].to(device)
            right_eye = batch["right_eye"].to(device)
            label = batch["label"].cpu().tolist()

            logits = model(left_eye, right_eye)["logits"]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()

            for lb, pr in zip(label, probs):
                rows.append((idx, int(lb), float(pr[0]), float(pr[1])))
                idx += 1

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Saved scores to {args.out}, rows={len(rows) - 1}")


if __name__ == "__main__":
    main()
