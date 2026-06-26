# BiCause-Gaze (Colab Ready)

This repository is now documented for the `bicause_gaze` pipeline only.

BiCause-Gaze is a binocular inconsistency based deepfake detector with three key ideas:

1. Physical consistency between 3D gaze and head pose.
2. Binocular causal consistency conditioned on head pose.
3. Micro-dynamics modeling for gaze jitter (time + frequency).

## Project entry points

- Train: `bicause_gaze/tools/train.py`
- Test: `bicause_gaze/tools/test.py`
- Export scores: `bicause_gaze/tools/export_scores.py`
- Eye crop preprocessing: `bicause_gaze/tools/prepare_eye_crops.py`

## Colab quick start

In Colab runtime:

1. Upload/unzip this project.
2. Install dependencies.
3. Prepare eye crops.
4. Train and evaluate.

Install dependencies:

- `pip install -r requirements.txt`

Train (smoke run with synthetic fallback):

- `python bicause_gaze/tools/train.py --config bicause_gaze/configs/bicause_ffpp.yaml`

Evaluate:

- `python bicause_gaze/tools/test.py --config bicause_gaze/configs/bicause_ffpp.yaml --ckpt runs/bicause_gaze/best.pt`

## Data format

Training dataset format for `bicause_gaze/datasets/video_clip_dataset.py`:

```text
train_root/
    fake/
        clip_0001/
            left/*.jpg
            right/*.jpg
    real/
        clip_0001/
            left/*.jpg
            right/*.jpg
```

`val_root/` uses the same layout.

## Eye crop preprocessing

Input face-frame clips:

```text
input_root/
    fake/<clip_name>/*.jpg
    real/<clip_name>/*.jpg
```

Output eye clips:

```text
output_root/
    fake/<clip_name>/left/*.jpg
    fake/<clip_name>/right/*.jpg
    real/<clip_name>/left/*.jpg
    real/<clip_name>/right/*.jpg
```

Recommended (no dlib required):

- `python bicause_gaze/tools/prepare_eye_crops.py --input-root /path/to/frames/train --output-root /path/to/eye/train --backend mediapipe --eye-size 112 --expand 1.8`

Optional dlib backend:

- `python bicause_gaze/tools/prepare_eye_crops.py --input-root /path/to/frames/train --output-root /path/to/eye/train --backend dlib --shape-predictor /path/to/shape_predictor_68_face_landmarks.dat`

## Notes

- If `train_root` / `val_root` is empty and `synthetic_if_missing=true`, training will use synthetic random data for smoke validation.
- To run on real data, update `train_root` and `val_root` in `bicause_gaze/configs/bicause_ffpp.yaml` (or your copied config).
