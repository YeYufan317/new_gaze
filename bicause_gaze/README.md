# BiCause-Gaze (on top of DFGaze)

This sub-project provides a minimal runnable baseline for binocular inconsistency based deepfake detection with three research contributions:

1. Physical consistency between 3D gaze and head pose.
2. Binocular causal consistency conditioned on head pose.
3. Micro-dynamics modeling for gaze jitter (time + frequency).

## Quick start

Run from repository root:

- Train (synthetic fallback enabled by default):
  - `python bicause_gaze/tools/train.py --config bicause_gaze/configs/bicause_ffpp.yaml`

- Evaluate:
  - `python bicause_gaze/tools/test.py --config bicause_gaze/configs/bicause_ffpp.yaml --ckpt runs/bicause_gaze/best.pt`

## Data contract

Dataset should return a dict:
- `left_eye`: `[T, 3, H, W]`
- `right_eye`: `[T, 3, H, W]`
- `label`: scalar in `{0,1}`
- optional `meta`

If `train_root` / `val_root` is empty or missing and `synthetic_if_missing=true`, a synthetic dataset is used for smoke training.

## Prepare eye crops (recommended before training)

Use the script below to convert face-frame clips into binocular eye patches:

- Default backend is `mediapipe` (recommended on Windows, no dlib model file needed).
- Optional backend: `dlib` with `shape_predictor_68_face_landmarks.dat`.

- Input expected:
  - `input_root/fake/<clip_name>/*.jpg`
  - `input_root/real/<clip_name>/*.jpg`
- Output generated:
  - `output_root/fake/<clip_name>/left/*.jpg`
  - `output_root/fake/<clip_name>/right/*.jpg`
  - `output_root/real/<clip_name>/left/*.jpg`
  - `output_root/real/<clip_name>/right/*.jpg`

Example:

- `python bicause_gaze/tools/prepare_eye_crops.py --input-root D:/datasets/ffpp_frames/train --output-root D:/datasets/bicause_ffpp/train --backend mediapipe --eye-size 112 --expand 1.8 --save-report runs/bicause_gaze/preprocess_train_report.json`

If you want to use dlib backend:

- `python bicause_gaze/tools/prepare_eye_crops.py --input-root D:/datasets/ffpp_frames/train --output-root D:/datasets/bicause_ffpp/train --backend dlib --shape-predictor D:/models/shape_predictor_68_face_landmarks.dat --eye-size 112 --expand 1.8`

Then set `train_root` and `val_root` in `bicause_gaze/configs/bicause_ffpp.yaml` to your generated eye-crop directories.
