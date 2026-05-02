#!/usr/bin/env python3
"""Exercise 2 pipeline: pretrained baseline, YOLO fine-tuning, and evaluation.

The training command is intentionally not run by default. Run it on a GPU PC.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Data_2_v2"
DATASET_YAML = DATASET_DIR / "dataset.yaml"
EXERCISE_DIR = PROJECT_ROOT / "Exercise_2"
RUNTIME_DATASET_YAML = EXERCISE_DIR / "dataset_runtime.yaml"
RUNS_DIR = EXERCISE_DIR / "runs"
MANUAL_DIR = EXERCISE_DIR / "manual_annotation_10"
CLASS_NAMES = ["car", "bus", "truck"]
COCO_TO_PROJECT_CLASS = {2: 0, 5: 1, 7: 2}
IMAGE_SIZE = (1280, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-manual-subset", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate-finetuned", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--examples", action="store_true")
    parser.add_argument("--weights", default=str(PROJECT_ROOT / "Data" / "yolov8n.pt"))
    parser.add_argument("--finetuned-weights", default=str(RUNS_DIR / "train" / "yolov8n_zhandong_v2" / "weights" / "best.pt"))
    parser.add_argument("--examples-weights", default=None)
    parser.add_argument("--examples-name", default="finetuned_examples")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default=0, help="Use 0 for first CUDA GPU, 'cpu', or 'mps'.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install it on the training PC with:\n"
            "  pip install ultralytics pandas matplotlib seaborn pillow opencv-python\n"
        ) from exc
    return YOLO


def write_runtime_dataset_yaml() -> Path:
    """Write a portable YAML with the current machine's absolute dataset path."""
    yaml_text = "\n".join(
        [
            f"path: {DATASET_DIR.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            "  0: car",
            "  1: bus",
            "  2: truck",
            "",
        ]
    )
    RUNTIME_DATASET_YAML.write_text(yaml_text, encoding="utf-8")
    return RUNTIME_DATASET_YAML


def label_path_for_image(image_path: Path) -> Path:
    return DATASET_DIR / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def read_yolo_labels(label_path: Path) -> list[dict]:
    boxes = []
    if not label_path.exists():
        return boxes
    width, height = IMAGE_SIZE
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, xc, yc, bw, bh = line.split()
        cls_id = int(cls)
        xc = float(xc) * width
        yc = float(yc) * height
        bw = float(bw) * width
        bh = float(bh) * height
        boxes.append(
            {
                "class_id": cls_id,
                "xyxy": np.array([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], dtype=float),
            }
        )
    return boxes


def xyxy_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if len(recalls) == 0:
        return 0.0
    ap = 0.0
    for threshold in np.linspace(0, 1, 101):
        valid = precisions[recalls >= threshold]
        ap += valid.max() if valid.size else 0.0
    return ap / 101.0


def compute_metrics(ground_truths: dict, predictions: list[dict], iou_thresholds: list[float]) -> dict:
    totals_by_class = Counter()
    for image_gts in ground_truths.values():
        totals_by_class.update(gt["class_id"] for gt in image_gts)

    output = {"per_iou": {}, "classes": CLASS_NAMES}
    for iou_threshold in iou_thresholds:
        per_class = {}
        aps = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for cls_id, cls_name in enumerate(CLASS_NAMES):
            cls_predictions = sorted(
                [pred for pred in predictions if pred["class_id"] == cls_id],
                key=lambda pred: pred["confidence"],
                reverse=True,
            )
            matched = defaultdict(set)
            tp = []
            fp = []

            for pred in cls_predictions:
                image_id = pred["image_id"]
                candidates = [
                    (idx, gt)
                    for idx, gt in enumerate(ground_truths[image_id])
                    if gt["class_id"] == cls_id and idx not in matched[image_id]
                ]
                best_idx = None
                best_iou = 0.0
                for idx, gt in candidates:
                    iou = xyxy_iou(pred["xyxy"], gt["xyxy"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                if best_idx is not None and best_iou >= iou_threshold:
                    matched[image_id].add(best_idx)
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            gt_total = totals_by_class[cls_id]
            recalls = tp_cum / gt_total if gt_total else np.array([])
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
            ap = average_precision(recalls, precisions) if gt_total else 0.0
            final_tp = int(tp_cum[-1]) if len(tp_cum) else 0
            final_fp = int(fp_cum[-1]) if len(fp_cum) else 0
            final_fn = int(gt_total - final_tp)
            precision = final_tp / (final_tp + final_fp) if final_tp + final_fp else 0.0
            recall = final_tp / gt_total if gt_total else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

            per_class[cls_name] = {
                "ground_truth": int(gt_total),
                "predictions": len(cls_predictions),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap": ap,
            }
            aps.append(ap)
            total_tp += final_tp
            total_fp += final_fp
            total_fn += final_fn

        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output["per_iou"][str(iou_threshold)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mAP": float(np.mean(aps)) if aps else 0.0,
            "per_class": per_class,
        }

    map_50_95 = np.mean([output["per_iou"][str(t)]["mAP"] for t in iou_thresholds])
    output["summary"] = {
        "precision_iou50": output["per_iou"]["0.5"]["precision"],
        "recall_iou50": output["per_iou"]["0.5"]["recall"],
        "f1_iou50": output["per_iou"]["0.5"]["f1"],
        "mAP50": output["per_iou"]["0.5"]["mAP"],
        "mAP50_95": float(map_50_95),
    }
    return output


def collect_test_images() -> list[Path]:
    return sorted((DATASET_DIR / "images" / "test").glob("*.jpg"))


def evaluate_model(weights: str, output_name: str, imgsz: int, conf: float, device: str, remap_coco: bool) -> dict:
    YOLO = import_yolo()
    model = YOLO(weights)
    image_paths = collect_test_images()
    ground_truths = {
        str(image_path): read_yolo_labels(label_path_for_image(image_path)) for image_path in image_paths
    }
    predictions = []

    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
        stream=False,
    )
    for image_path, result in zip(image_paths, results):
        for box in result.boxes:
            model_cls = int(box.cls.item())
            if remap_coco:
                if model_cls not in COCO_TO_PROJECT_CLASS:
                    continue
                cls_id = COCO_TO_PROJECT_CLASS[model_cls]
            else:
                if model_cls not in [0, 1, 2]:
                    continue
                cls_id = model_cls
            predictions.append(
                {
                    "image_id": str(image_path),
                    "class_id": cls_id,
                    "confidence": float(box.conf.item()),
                    "xyxy": box.xyxy[0].cpu().numpy().astype(float),
                }
            )

    thresholds = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]
    metrics = compute_metrics(ground_truths, predictions, thresholds)
    metrics["weights"] = weights
    metrics["test_images"] = len(image_paths)
    metrics["predictions_after_class_filter"] = len(predictions)
    metrics["remapped_coco_classes"] = remap_coco

    output_path = EXERCISE_DIR / f"{output_name}_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["summary"], indent=2))
    print(f"Saved: {output_path}")
    return metrics


def prepare_manual_subset(seed: int) -> None:
    if MANUAL_DIR.exists():
        shutil.rmtree(MANUAL_DIR)
    (MANUAL_DIR / "images").mkdir(parents=True)
    (MANUAL_DIR / "labels").mkdir(parents=True)

    candidate_images = sorted((DATASET_DIR / "images" / "train").glob("*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(candidate_images)

    selected = []
    class_presence = Counter()
    for image_path in candidate_images:
        labels = read_yolo_labels(label_path_for_image(image_path))
        present = {label["class_id"] for label in labels}
        if len(selected) < 10:
            selected.append(image_path)
            class_presence.update(present)
        if len(selected) >= 10 and all(class_presence[idx] > 0 for idx in range(3)):
            break
    selected = selected[:10]

    preview_images = []
    for image_path in selected:
        label_path = label_path_for_image(image_path)
        shutil.copy2(image_path, MANUAL_DIR / "images" / image_path.name)
        shutil.copy2(label_path, MANUAL_DIR / "labels" / f"{image_path.stem}.txt")

        with Image.open(image_path).convert("RGB") as im:
            draw = ImageDraw.Draw(im)
            colors = {0: (0, 220, 80), 1: (255, 190, 0), 2: (255, 70, 70)}
            for label in read_yolo_labels(label_path):
                x1, y1, x2, y2 = label["xyxy"]
                draw.rectangle([x1, y1, x2, y2], outline=colors[label["class_id"]], width=2)
            draw.rectangle([0, 0, 460, 34], fill=(0, 0, 0))
            draw.text((8, 9), image_path.name, fill=(255, 255, 255))
            im.thumbnail((640, 360))
            preview_images.append(im.copy())

    rows = math.ceil(len(preview_images) / 2)
    preview = Image.new("RGB", (1280, rows * 360), (30, 30, 30))
    for idx, im in enumerate(preview_images):
        preview.paste(im, ((idx % 2) * 640, (idx // 2) * 360))
    preview.save(MANUAL_DIR / "manual_10_preview.jpg", quality=92)

    readme = """# Manual Annotation Subset

These 10 images are copied from `Data_2_v2/train` with prefilled YOLO labels.

To satisfy the Exercise 2 requirement truthfully, open these images in a labeling
tool such as LabelImg, CVAT, Roboflow, or Label Studio, inspect the boxes, and
manually correct/save the labels. The provided labels are useful assisted labels,
but your report should mention that you reviewed/corrected at least these 10.

Classes:

- 0 car
- 1 bus
- 2 truck
"""
    (MANUAL_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"Prepared manual subset: {MANUAL_DIR}")


def train_model(weights: str, epochs: int, imgsz: int, batch: int, device: str, seed: int, patience: int) -> None:
    YOLO = import_yolo()
    dataset_yaml = write_runtime_dataset_yaml()
    model = YOLO(weights if Path(weights).exists() else "yolov8n.pt")
    result = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        seed=seed,
        patience=patience,
        close_mosaic=10,
        degrees=5.0,
        translate=0.08,
        scale=0.35,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.05,
        hsv_h=0.015,
        hsv_s=0.40,
        hsv_v=0.25,
        dropout=0.0,
        project=str(RUNS_DIR / "train"),
        name="yolov8n_zhandong_v2",
        exist_ok=True,
        plots=True,
        cache=False,
    )
    print(result)


def compare_metrics() -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    baseline_path = EXERCISE_DIR / "pretrained_baseline_metrics.json"
    finetuned_path = EXERCISE_DIR / "finetuned_metrics.json"
    if not baseline_path.exists() or not finetuned_path.exists():
        raise SystemExit("Run --baseline and --evaluate-finetuned first.")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["summary"]
    finetuned = json.loads(finetuned_path.read_text(encoding="utf-8"))["summary"]
    rows = []
    for model_name, metrics in [("Pretrained YOLOv8n", baseline), ("Fine-tuned YOLOv8n", finetuned)]:
        rows.append({"model": model_name, **metrics})
    df = pd.DataFrame(rows)
    csv_path = EXERCISE_DIR / "comparison_summary.csv"
    df.to_csv(csv_path, index=False)

    ax = df.set_index("model")[["precision_iou50", "recall_iou50", "f1_iou50", "mAP50", "mAP50_95"]].plot(
        kind="bar", figsize=(11, 5), ylim=(0, 1), rot=0
    )
    ax.set_title("Pretrained vs Fine-tuned YOLOv8n on Data_2_v2 Test Set")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig_path = EXERCISE_DIR / "comparison_metrics.png"
    fig.savefig(fig_path, dpi=180)
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")


def save_example_detections(weights: str, output_name: str, imgsz: int, conf: float, device: str, seed: int) -> None:
    YOLO = import_yolo()
    model = YOLO(weights)
    image_paths = collect_test_images()
    random.Random(seed).shuffle(image_paths)
    selected = image_paths[:12]
    model.predict(
        source=[str(path) for path in selected],
        imgsz=imgsz,
        conf=conf,
        device=device,
        project=str(EXERCISE_DIR / "example_detections"),
        name=output_name,
        exist_ok=True,
        save=True,
        verbose=False,
    )
    print(f"Saved example detections to: {EXERCISE_DIR / 'example_detections' / output_name}")


def main() -> None:
    args = parse_args()
    EXERCISE_DIR.mkdir(exist_ok=True)
    if args.prepare_manual_subset:
        prepare_manual_subset(args.seed)
    if args.baseline:
        evaluate_model(args.weights, "pretrained_baseline", args.imgsz, args.conf, args.device, remap_coco=True)
    if args.train:
        train_model(args.weights, args.epochs, args.imgsz, args.batch, args.device, args.seed, args.patience)
    if args.evaluate_finetuned:
        evaluate_model(args.finetuned_weights, "finetuned", args.imgsz, args.conf, args.device, remap_coco=False)
    if args.compare:
        compare_metrics()
    if args.examples:
        weights = args.examples_weights or args.finetuned_weights
        save_example_detections(weights, args.examples_name, args.imgsz, args.conf, args.device, args.seed)


if __name__ == "__main__":
    main()
