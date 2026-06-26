from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VideoClipDataset(Dataset):
    """
    Minimal clip dataset for binocular deepfake detection.

    Expected directory layout (optional):
        root/
          fake/
            sample_x/
              left/000.jpg ...
              right/000.jpg ...
          real/
            sample_y/
              left/000.jpg ...
              right/000.jpg ...

    If root does not exist and synthetic_if_missing=True, the dataset returns
    random tensors for smoke training.
    """

    def __init__(
        self,
        root: str,
        clip_len: int = 16,
        image_size: int = 112,
        synthetic_if_missing: bool = True,
    ) -> None:
        self.root = Path(root) if root else None
        self.clip_len = clip_len
        self.image_size = image_size
        self.synthetic_if_missing = synthetic_if_missing

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.samples: List[Tuple[Path, int]] = []
        self._use_synthetic = False
        self._discover_samples()

    def _discover_samples(self) -> None:
        if self.root is None or not self.root.exists():
            self._use_synthetic = self.synthetic_if_missing
            return

        for class_name, label in [("fake", 0), ("real", 1)]:
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for sample_dir in sorted(class_dir.iterdir()):
                if sample_dir.is_dir():
                    self.samples.append((sample_dir, label))

        if len(self.samples) == 0:
            self._use_synthetic = self.synthetic_if_missing

    def __len__(self) -> int:
        if self._use_synthetic:
            return 64
        return len(self.samples)

    def _read_eye_frames(self, eye_dir: Path) -> List[Path]:
        frames = sorted([p for p in eye_dir.glob("*.jpg")])
        if len(frames) < self.clip_len:
            frames += frames[-1:] * (self.clip_len - len(frames)) if frames else []
        return frames[: self.clip_len]

    def _load_clip(self, frame_paths: List[Path]) -> torch.Tensor:
        clip: List[torch.Tensor] = []
        for p in frame_paths:
            img = Image.open(p).convert("RGB")
            clip.append(self.transform(img))
        return torch.stack(clip, dim=0)  # [T,3,H,W]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._use_synthetic:
            left_eye = torch.rand(self.clip_len, 3, self.image_size, self.image_size)
            right_eye = torch.rand(self.clip_len, 3, self.image_size, self.image_size)
            label = torch.tensor(random.randint(0, 1), dtype=torch.long)
            return {"left_eye": left_eye, "right_eye": right_eye, "label": label}

        sample_dir, label = self.samples[idx]
        left_paths = self._read_eye_frames(sample_dir / "left")
        right_paths = self._read_eye_frames(sample_dir / "right")

        if len(left_paths) == 0 or len(right_paths) == 0:
            # fallback when sample is malformed
            left_eye = torch.rand(self.clip_len, 3, self.image_size, self.image_size)
            right_eye = torch.rand(self.clip_len, 3, self.image_size, self.image_size)
        else:
            left_eye = self._load_clip(left_paths)
            right_eye = self._load_clip(right_paths)

        return {
            "left_eye": left_eye,
            "right_eye": right_eye,
            "label": torch.tensor(label, dtype=torch.long),
        }
