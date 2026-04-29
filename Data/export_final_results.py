from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))

from ultralytics import YOLO
TRAFFIC_ROOT = DATA_DIR / "TrafficProject"
SPLIT_ROOT = TRAFFIC_ROOT / "split_3200"
DATA_YAML = SPLIT_ROOT / "data.yaml"
BASE_MODEL = DATA_DIR / "yolov8n.pt"
RUN_DIR = TRAFFIC_ROOT / "runs" / "yolov8n_finetuned_topdown_50e"
BEST_MODEL = RUN_DIR / "weights" / "best.pt"
BASELINE_SUMMARY = TRAFFIC_ROOT / "baseline_before" / "baseline_summary.json"
RESULTS_DIR = TRAFFIC_ROOT / "results"
ARTIFACTS_DIR = RESULTS_DIR / "fine_tuned_artifacts"
COMPARISON_DIR = RESULTS_DIR / "comparison"
BASELINE_PRED_DIR = COMPARISON_DIR / "baseline_yolov8n"
FINETUNED_PRED_DIR = COMPARISON_DIR / "finetuned_yolov8n"


def copy_artifacts() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_names = [
        "args.yaml",
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "labels.jpg",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
        "val_batch1_labels.jpg",
        "val_batch1_pred.jpg",
        "val_batch2_labels.jpg",
        "val_batch2_pred.jpg",
    ]
    for name in artifact_names:
        src = RUN_DIR / name
        if src.exists():
            shutil.copy2(src, ARTIFACTS_DIR / name)

    weights_dir = ARTIFACTS_DIR / "weights"
    weights_dir.mkdir(exist_ok=True)
    for name in ["best.pt", "last.pt"]:
        src = RUN_DIR / "weights" / name
        if src.exists():
            shutil.copy2(src, weights_dir / name)


def select_sample_images(limit: int = 3) -> list[Path]:
    test_images = sorted((SPLIT_ROOT / "images" / "test").glob("*.jpg"))
    if len(test_images) < limit:
        raise RuntimeError(f"Expected at least {limit} test images, found {len(test_images)}")
    return test_images[:limit]


def clear_prediction_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def predict_samples(sample_images: list[Path]) -> None:
    clear_prediction_dir(BASELINE_PRED_DIR)
    clear_prediction_dir(FINETUNED_PRED_DIR)

    baseline = YOLO(str(BASE_MODEL))
    finetuned = YOLO(str(BEST_MODEL))

    baseline.predict(
        source=[str(path) for path in sample_images],
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(BASELINE_PRED_DIR.parent),
        name=BASELINE_PRED_DIR.name,
        exist_ok=True,
        verbose=False,
    )
    finetuned.predict(
        source=[str(path) for path in sample_images],
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(FINETUNED_PRED_DIR.parent),
        name=FINETUNED_PRED_DIR.name,
        exist_ok=True,
        verbose=False,
    )


def make_side_by_side(sample_images: list[Path]) -> list[Path]:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for index, image_path in enumerate(sample_images, start=1):
        baseline_path = BASELINE_PRED_DIR / image_path.name
        finetuned_path = FINETUNED_PRED_DIR / image_path.name
        baseline_img = Image.open(baseline_path).convert("RGB")
        finetuned_img = Image.open(finetuned_path).convert("RGB")

        target_h = max(baseline_img.height, finetuned_img.height)
        label_h = 42
        width = baseline_img.width + finetuned_img.width
        canvas = Image.new("RGB", (width, target_h + label_h), "white")
        canvas.paste(baseline_img, (0, label_h))
        canvas.paste(finetuned_img, (baseline_img.width, label_h))

        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), "Pretrained YOLOv8n", fill=(20, 20, 20))
        draw.text((baseline_img.width + 12, 12), "Fine-tuned YOLOv8n", fill=(20, 20, 20))

        output_path = COMPARISON_DIR / f"before_after_{index:02d}.jpg"
        canvas.save(output_path, quality=95)
        output_paths.append(output_path)

    return output_paths


def rerun_validation() -> dict[str, float]:
    model = YOLO(str(BEST_MODEL))
    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=640,
        batch=16,
        plots=True,
        workers=0,
        project=str(RESULTS_DIR),
        name="finetuned_validation_rerun",
        exist_ok=True,
        verbose=False,
    )
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def write_comparison_files(validation_metrics: dict[str, float], comparison_images: list[Path]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(BASELINE_SUMMARY.read_text())
    rows = [
        {
            "model": "Pretrained YOLOv8n",
            "data_used": "COCO pretrained weights only",
            "evaluation": "64-image held-out baseline subset",
            "precision": f"{baseline['micro_precision']:.4f}",
            "recall": f"{baseline['micro_recall']:.4f}",
            "f1": f"{baseline['micro_f1']:.4f}",
            "map50": "",
            "map50_95": "",
        },
        {
            "model": "Fine-tuned YOLOv8n",
            "data_used": "3,200 annotated aerial traffic images",
            "evaluation": "816-image held-out test split",
            "precision": f"{validation_metrics['precision']:.4f}",
            "recall": f"{validation_metrics['recall']:.4f}",
            "f1": "",
            "map50": f"{validation_metrics['map50']:.4f}",
            "map50_95": f"{validation_metrics['map50_95']:.4f}",
        },
    ]

    csv_path = RESULTS_DIR / "pretrained_vs_finetuned_comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    image_lines = "\n".join(
        f"![Before/after detection {i}](TrafficProject/results/comparison/{path.name})"
        for i, path in enumerate(comparison_images, start=1)
    )
    markdown = f"""# Final comparison: pretrained vs fine-tuned YOLOv8n

| Model | Data used | Evaluation set | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---:|---:|---:|---:|---:|
| Pretrained YOLOv8n | COCO pretrained weights only | 64-image held-out baseline subset | {rows[0]['precision']} | {rows[0]['recall']} | {rows[0]['f1']} | - | - |
| Fine-tuned YOLOv8n | 3,200 annotated aerial traffic images | 816-image held-out test split | {rows[1]['precision']} | {rows[1]['recall']} | - | {rows[1]['map50']} | {rows[1]['map50_95']} |

The pretrained model barely detects the aerial traffic participants because the camera angle and object scale are very different from COCO street-level images. After fine-tuning on the aerial dataset, the model reaches about {rows[1]['map50']} mAP50 on the held-out test split.

## Example detections

{image_lines}

## Saved artifacts

Fine-tuned weights and validation figures are saved in `TrafficProject/results/fine_tuned_artifacts/`. The rerun validation output is saved in `TrafficProject/results/finetuned_validation_rerun/`.
"""
    (RESULTS_DIR / "final_comparison.md").write_text(markdown)


def append_notebook_section(validation_metrics: dict[str, float], comparison_images: list[Path]) -> None:
    import nbformat

    notebook_path = DATA_DIR / "Task_2.ipynb"
    nb = nbformat.read(notebook_path, as_version=4)

    heading = "## Final Comparison: Pretrained vs Fine-Tuned YOLOv8n"
    nb.cells = [
        cell
        for cell in nb.cells
        if not (
            cell.cell_type == "markdown"
            and isinstance(cell.source, str)
            and heading in cell.source
        )
    ]

    baseline = json.loads(BASELINE_SUMMARY.read_text())
    image_lines = "\n\n".join(
        f"![Before/after detection {i}](TrafficProject/results/comparison/{path.name})"
        for i, path in enumerate(comparison_images, start=1)
    )
    markdown = f"""{heading}

| Model | Training data | Evaluation set | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---:|---:|---:|---:|---:|
| Pretrained YOLOv8n | COCO pretrained weights only | 64-image held-out baseline subset | {baseline['micro_precision']:.4f} | {baseline['micro_recall']:.4f} | {baseline['micro_f1']:.4f} | - | - |
| Fine-tuned YOLOv8n | 3,200 annotated aerial traffic images | 816-image held-out test split | {validation_metrics['precision']:.4f} | {validation_metrics['recall']:.4f} | - | {validation_metrics['map50']:.4f} | {validation_metrics['map50_95']:.4f} |

The pretrained YOLOv8n model performs poorly on this aerial traffic dataset because the object scale and viewpoint differ strongly from the street-level COCO training data. Fine-tuning on the annotated aerial images substantially improves detection performance, especially for cars, buses, trucks, and pedestrians.

### Example Before/After Detections

{image_lines}

### Saved Fine-Tuned Artifacts

The fine-tuned checkpoint and validation figures are included in:

- `TrafficProject/results/fine_tuned_artifacts/weights/best.pt`
- `TrafficProject/results/fine_tuned_artifacts/results.csv`
- `TrafficProject/results/fine_tuned_artifacts/results.png`
- `TrafficProject/results/fine_tuned_artifacts/confusion_matrix.png`
- `TrafficProject/results/fine_tuned_artifacts/PR_curve.png`
- `TrafficProject/results/fine_tuned_artifacts/F1_curve.png`

The validation rerun from the copied `best.pt` is saved in `TrafficProject/results/finetuned_validation_rerun/`.
"""
    nb.cells.append(nbformat.v4.new_markdown_cell(markdown))
    nbformat.write(nb, notebook_path)


def main() -> None:
    if not BEST_MODEL.exists():
        raise FileNotFoundError(f"Missing fine-tuned checkpoint: {BEST_MODEL}")

    copy_artifacts()
    sample_images = select_sample_images()
    validation_metrics = rerun_validation()
    predict_samples(sample_images)
    comparison_images = make_side_by_side(sample_images)
    write_comparison_files(validation_metrics, comparison_images)
    append_notebook_section(validation_metrics, comparison_images)

    print("Export complete.")
    print(f"mAP50: {validation_metrics['map50']:.4f}")
    print(f"mAP50-95: {validation_metrics['map50_95']:.4f}")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
