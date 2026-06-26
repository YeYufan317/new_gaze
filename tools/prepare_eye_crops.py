from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None

try:
    import dlib  # type: ignore
except Exception:
    dlib = None


IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class CropResult:
    total_frames: int = 0
    landmark_ok: int = 0
    fallback_used: int = 0
    skipped: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare left/right eye crops for BiCause-Gaze training.")
    parser.add_argument("--input-root", type=str, required=True, help="Input root of extracted face frames.")
    parser.add_argument("--output-root", type=str, required=True, help="Output root for eye crops.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mediapipe", "dlib"],
        default="mediapipe",
        help="Landmark backend. Use mediapipe by default; dlib requires --shape-predictor.",
    )
    parser.add_argument("--shape-predictor", type=str, default="", help="Path to dlib 68 landmarks model.")
    parser.add_argument("--eye-size", type=int, default=112, help="Output eye patch size.")
    parser.add_argument("--expand", type=float, default=1.8, help="Expand ratio around eye landmarks.")
    parser.add_argument("--min-frames", type=int, default=8, help="Minimum frames in a clip to keep it.")
    parser.add_argument("--max-clips", type=int, default=-1, help="Limit number of clips for debugging.")
    parser.add_argument("--save-report", type=str, default="", help="Optional path to save JSON report.")
    return parser.parse_args()


def list_frame_files(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_SUFFIXES])


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def eye_bbox(points: np.ndarray, expand: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    side = max((x_max - x_min), (y_max - y_min)) * expand
    side = max(side, 8.0)

    x1 = int(cx - side / 2.0)
    y1 = int(cy - side / 2.0)
    x2 = int(cx + side / 2.0)
    y2 = int(cy + side / 2.0)
    return clamp_bbox(x1, y1, x2, y2, img_w, img_h)


def center_fallback_boxes(img_w: int, img_h: int, eye_size_ratio: float = 0.2) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    side = int(min(img_w, img_h) * eye_size_ratio)
    side = max(side, 24)
    y = int(img_h * 0.35)

    left_cx = int(img_w * 0.35)
    right_cx = int(img_w * 0.65)

    left = clamp_bbox(left_cx - side // 2, y - side // 2, left_cx + side // 2, y + side // 2, img_w, img_h)
    right = clamp_bbox(right_cx - side // 2, y - side // 2, right_cx + side // 2, y + side // 2, img_w, img_h)
    return left, right


def crop_and_save(
    img_bgr: np.ndarray,
    left_box: Tuple[int, int, int, int],
    right_box: Tuple[int, int, int, int],
    out_left: Path,
    out_right: Path,
    eye_size: int,
) -> None:
    lx1, ly1, lx2, ly2 = left_box
    rx1, ry1, rx2, ry2 = right_box

    left_eye = img_bgr[ly1:ly2, lx1:lx2]
    right_eye = img_bgr[ry1:ry2, rx1:rx2]

    left_eye = cv2.resize(left_eye, (eye_size, eye_size), interpolation=cv2.INTER_AREA)
    right_eye = cv2.resize(right_eye, (eye_size, eye_size), interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(out_left), left_eye)
    cv2.imwrite(str(out_right), right_eye)


def discover_clips(root: Path) -> List[Tuple[str, Path]]:
    clips: List[Tuple[str, Path]] = []
    if not root.exists():
        return clips

    # expected: input_root/fake/<clip_dir>, input_root/real/<clip_dir>
    for class_name in ["fake", "real"]:
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        for clip_dir in sorted(class_dir.iterdir()):
            if clip_dir.is_dir():
                clips.append((class_name, clip_dir))
    return clips


def _init_backend(backend: str, shape_predictor_path: str) -> Dict[str, Any]:
    if backend == "mediapipe":
        if mp is None:
            raise RuntimeError("mediapipe is not installed. Please install: pip install mediapipe")
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        return {"backend": backend, "mesh": mesh}

    if backend == "dlib":
        if dlib is None:
            raise RuntimeError("dlib is not installed. Please install dlib or switch to --backend mediapipe")
        predictor_path = Path(shape_predictor_path)
        if not predictor_path.exists():
            raise FileNotFoundError(f"Shape predictor not found: {predictor_path}")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(predictor_path))
        return {"backend": backend, "detector": detector, "predictor": predictor}

    raise ValueError(f"Unsupported backend: {backend}")


def _extract_eye_boxes_dlib(img_bgr: np.ndarray, backend_ctx: Dict[str, Any], expand: float) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    faces = backend_ctx["detector"](gray, 1)
    if len(faces) == 0:
        return None, None

    face = max(faces, key=lambda r: r.width() * r.height())
    shape = backend_ctx["predictor"](gray, face)
    pts = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        pts[i] = (shape.part(i).x, shape.part(i).y)

    left_pts = pts[36:42]
    right_pts = pts[42:48]
    left_box = eye_bbox(left_pts, expand, w, h)
    right_box = eye_bbox(right_pts, expand, w, h)
    return left_box, right_box


def _extract_eye_boxes_mediapipe(img_bgr: np.ndarray, backend_ctx: Dict[str, Any], expand: float) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    result = backend_ctx["mesh"].process(img_rgb)
    if not result.multi_face_landmarks:
        return None, None

    # Face Mesh landmark indices around eyes
    left_idx = [33, 133, 159, 145, 160, 158, 153, 144]
    right_idx = [362, 263, 386, 374, 385, 387, 380, 373]

    face = result.multi_face_landmarks[0]
    left_pts = np.array(
        [[int(face.landmark[i].x * w), int(face.landmark[i].y * h)] for i in left_idx],
        dtype=np.int32,
    )
    right_pts = np.array(
        [[int(face.landmark[i].x * w), int(face.landmark[i].y * h)] for i in right_idx],
        dtype=np.int32,
    )

    left_box = eye_bbox(left_pts, expand, w, h)
    right_box = eye_bbox(right_pts, expand, w, h)
    return left_box, right_box


def extract_eye_boxes(img_bgr: np.ndarray, backend_ctx: Dict[str, Any], expand: float) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    if backend_ctx["backend"] == "dlib":
        return _extract_eye_boxes_dlib(img_bgr, backend_ctx, expand)
    return _extract_eye_boxes_mediapipe(img_bgr, backend_ctx, expand)


def process_clip(
    clip_dir: Path,
    out_clip_dir: Path,
    backend_ctx: Dict[str, Any],
    eye_size: int,
    expand: float,
    min_frames: int,
) -> CropResult:
    result = CropResult()
    frame_files = list_frame_files(clip_dir)
    if len(frame_files) < min_frames:
        result.skipped = len(frame_files)
        return result

    left_dir = out_clip_dir / "left"
    right_dir = out_clip_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    prev_left: Optional[Tuple[int, int, int, int]] = None
    prev_right: Optional[Tuple[int, int, int, int]] = None

    for idx, frame_path in enumerate(frame_files):
        result.total_frames += 1

        img = cv2.imread(str(frame_path))
        if img is None:
            result.skipped += 1
            continue

        h, w = img.shape[:2]
        left_box: Optional[Tuple[int, int, int, int]] = None
        right_box: Optional[Tuple[int, int, int, int]] = None

        left_box, right_box = extract_eye_boxes(img, backend_ctx, expand)
        if left_box is not None and right_box is not None:
            result.landmark_ok += 1
        else:
            result.fallback_used += 1

        if left_box is None or right_box is None:
            if prev_left is not None and prev_right is not None:
                left_box, right_box = prev_left, prev_right
            else:
                left_box, right_box = center_fallback_boxes(w, h)

        prev_left, prev_right = left_box, right_box

        out_name = f"{idx:05d}.jpg"
        crop_and_save(img, left_box, right_box, left_dir / out_name, right_dir / out_name, eye_size)

    return result


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    backend_ctx = _init_backend(args.backend, args.shape_predictor)

    clips = discover_clips(input_root)
    if args.max_clips > 0:
        clips = clips[: args.max_clips]

    if len(clips) == 0:
        raise RuntimeError(
            "No clips discovered. Expected structure like input_root/fake/<clip_dir> and input_root/real/<clip_dir>."
        )

    agg = CropResult()
    per_clip: Dict[str, Dict[str, int]] = {}

    for class_name, clip_dir in clips:
        out_clip = output_root / class_name / clip_dir.name
        res = process_clip(
            clip_dir=clip_dir,
            out_clip_dir=out_clip,
            backend_ctx=backend_ctx,
            eye_size=args.eye_size,
            expand=args.expand,
            min_frames=args.min_frames,
        )

        agg.total_frames += res.total_frames
        agg.landmark_ok += res.landmark_ok
        agg.fallback_used += res.fallback_used
        agg.skipped += res.skipped

        per_clip[f"{class_name}/{clip_dir.name}"] = {
            "total_frames": res.total_frames,
            "landmark_ok": res.landmark_ok,
            "fallback_used": res.fallback_used,
            "skipped": res.skipped,
        }

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_clips": len(clips),
        "total_frames": agg.total_frames,
        "landmark_ok": agg.landmark_ok,
        "fallback_used": agg.fallback_used,
        "skipped": agg.skipped,
        "landmark_ratio": (agg.landmark_ok / agg.total_frames) if agg.total_frames else 0.0,
    }

    print("=== Eye crop preparation done ===")
    print(json.dumps(summary, indent=2))
    print(f"Backend: {args.backend}")

    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_clip": per_clip}, f, indent=2)
        print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()
