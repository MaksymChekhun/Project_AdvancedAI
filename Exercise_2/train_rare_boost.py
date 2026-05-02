#!/usr/bin/env python3
"""Extra rare-class fine-tuning for buses and trucks."""

from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_WEIGHTS = PROJECT_ROOT / "Exercise_2" / "runs" / "train" / "yolov8n_zhandong_v2" / "weights" / "best.pt"
DATA_YAML = PROJECT_ROOT / "Exercise_2" / "dataset_rare_boost.yaml"
RUNS_DIR = PROJECT_ROOT / "Exercise_2" / "runs" / "train"


def main() -> None:
    if not BASE_WEIGHTS.exists():
        raise SystemExit(f"Missing base weights: {BASE_WEIGHTS}")
    if not DATA_YAML.exists():
        raise SystemExit(f"Missing rare-boost dataset YAML: {DATA_YAML}")

    model = YOLO(str(BASE_WEIGHTS))
    model.train(
        data=str(DATA_YAML),
        epochs=25,
        imgsz=640,
        batch=8,
        device=0,
        patience=6,
        workers=0,
        seed=42,
        close_mosaic=5,
        degrees=5.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.08,
        hsv_h=0.015,
        hsv_s=0.40,
        hsv_v=0.25,
        project=str(RUNS_DIR),
        name="yolov8n_zhandong_v2_rare_boost_stable",
        exist_ok=True,
        plots=True,
        cache=False,
    )


if __name__ == "__main__":
    main()
