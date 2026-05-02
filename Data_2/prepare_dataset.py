#!/usr/bin/env python3
"""Prepare a YOLO object-detection dataset from the Zhandong Road video.

The source CSV contains one row per object box per frame. This script samples
one annotated frame per second, resizes frames to 1280x720, clips imperfect
boxes to the image boundary, and writes YOLO label files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1080
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
CLASS_MAP = {
    "car": 0,
    "bus": 1,
    "trunk": 2,  # Source spelling. Dataset YAML exposes this as "truck".
}
CLASS_NAMES = ["car", "bus", "truck"]


@dataclass(frozen=True)
class Box:
    cls_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="/Users/maksymchekhun/Desktop/A. Zhandong Road1.MP4",
        help="Path to source MP4.",
    )
    parser.add_argument(
        "--csv",
        default="/Users/maksymchekhun/Desktop/A.Zhandong Road1.csv",
        help="Path to source annotation CSV.",
    )
    parser.add_argument(
        "--out",
        default="/Users/maksymchekhun/Desktop/Project_AdvancedAI/Data_2",
        help="Output dataset folder.",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=30,
        help="Frame interval. 30 is approximately one second for this video.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.10)
    return parser.parse_args()


def load_selected_labels(csv_path: Path, sample_every: int) -> tuple[dict[int, list[Box]], dict]:
    selected: dict[int, list[Box]] = defaultdict(list)
    source_counts = Counter()
    clean_counts = Counter()
    skipped_counts = Counter()
    selected_frames: set[int] | None = None
    min_frame = None
    max_frame = None
    total_rows = 0

    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            total_rows += 1
            try:
                frame = int(row["frame_num"])
            except (KeyError, TypeError, ValueError):
                skipped_counts["bad_frame"] += 1
                continue

            if min_frame is None:
                min_frame = frame
            max_frame = frame

            if selected_frames is None:
                selected_frames = set()

            # The first annotated frame is 6. Sampling 6, 36, 66... preserves
            # exact alignment with the CSV while giving approximately 1 fps.
            if (frame - min_frame) % sample_every != 0:
                continue

            name = row.get("name", "").strip()
            if name not in CLASS_MAP:
                skipped_counts["unknown_class"] += 1
                continue

            try:
                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])
            except (KeyError, TypeError, ValueError):
                skipped_counts["bad_box"] += 1
                continue

            source_counts[name] += 1
            xmin = min(max(xmin, 0.0), float(ORIGINAL_WIDTH))
            ymin = min(max(ymin, 0.0), float(ORIGINAL_HEIGHT))
            xmax = min(max(xmax, 0.0), float(ORIGINAL_WIDTH))
            ymax = min(max(ymax, 0.0), float(ORIGINAL_HEIGHT))
            width = xmax - xmin
            height = ymax - ymin

            if width < 2 or height < 2:
                skipped_counts["too_small_after_clip"] += 1
                continue

            x_center = ((xmin + xmax) / 2.0) / ORIGINAL_WIDTH
            y_center = ((ymin + ymax) / 2.0) / ORIGINAL_HEIGHT
            norm_width = width / ORIGINAL_WIDTH
            norm_height = height / ORIGINAL_HEIGHT

            selected[frame].append(
                Box(CLASS_MAP[name], x_center, y_center, norm_width, norm_height)
            )
            clean_counts[CLASS_NAMES[CLASS_MAP[name]]] += 1

    if min_frame is None or max_frame is None:
        raise ValueError("No annotation rows found in CSV.")

    expected_frames = list(range(min_frame, max_frame + 1, sample_every))
    selected = {frame: selected[frame] for frame in expected_frames if selected[frame]}
    stats = {
        "source_csv_rows": total_rows,
        "first_annotated_frame": min_frame,
        "last_annotated_frame": max_frame,
        "sample_every_frames": sample_every,
        "expected_sampled_frames": len(expected_frames),
        "labeled_sampled_frames": len(selected),
        "source_class_counts_on_sampled_frames": dict(source_counts),
        "clean_class_counts": dict(clean_counts),
        "skipped_counts": dict(skipped_counts),
    }
    return selected, stats


def run_ffmpeg_extract(video_path: Path, out_dir: Path, first_frame: int, sample_every: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH.")

    tmp_dir = out_dir / "_tmp_frames"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    zero_based_start = first_frame - 1
    filter_expr = (
        f"select='gte(n\\,{zero_based_start})*not(mod(n-{zero_based_start}\\,{sample_every}))',"
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        filter_expr,
        "-vsync",
        "vfr",
        "-q:v",
        "2",
        str(tmp_dir / "sample_%06d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def split_frames(frames: list[int], train: float, val: float, test: float, seed: int) -> dict[str, list[int]]:
    if not math.isclose(train + val + test, 1.0, abs_tol=1e-6):
        raise ValueError("train + val + test must equal 1.0")

    shuffled = frames[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = round(n * train)
    val_end = train_end + round(n * val)
    return {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }


def write_dataset(out_dir: Path, labels_by_frame: dict[int, list[Box]], splits: dict[str, list[int]]) -> dict:
    tmp_dir = out_dir / "_tmp_frames"
    frame_sequence = sorted(labels_by_frame)
    sequence_to_frame = {
        sequence_index: frame for sequence_index, frame in enumerate(frame_sequence, start=1)
    }
    frame_to_tmp = {
        frame: tmp_dir / f"sample_{sequence_index:06d}.jpg"
        for sequence_index, frame in sequence_to_frame.items()
    }

    for stale_dir in [out_dir / "images", out_dir / "labels"]:
        if stale_dir.exists():
            shutil.rmtree(stale_dir)

    for subset in splits:
        (out_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

    frame_to_subset = {
        frame: subset for subset, subset_frames in splits.items() for frame in subset_frames
    }
    missing_images = []
    subset_stats = {}

    for subset, subset_frames in splits.items():
        box_counter = Counter()
        for frame in subset_frames:
            image_name = f"frame_{frame:06d}.jpg"
            label_name = f"frame_{frame:06d}.txt"
            src = frame_to_tmp.get(frame)
            if not src or not src.exists():
                missing_images.append(frame)
                continue

            shutil.copy2(src, out_dir / "images" / subset / image_name)
            with (out_dir / "labels" / subset / label_name).open("w", encoding="utf-8") as fh:
                for box in labels_by_frame[frame]:
                    box_counter[CLASS_NAMES[box.cls_id]] += 1
                    fh.write(
                        f"{box.cls_id} {box.x_center:.6f} {box.y_center:.6f} "
                        f"{box.width:.6f} {box.height:.6f}\n"
                    )

        subset_stats[subset] = {
            "frames": len(subset_frames),
            "boxes": sum(box_counter.values()),
            "class_counts": dict(box_counter),
        }

    if missing_images:
        raise RuntimeError(f"Missing extracted images for frames: {missing_images[:10]}")

    shutil.rmtree(tmp_dir)
    return {
        "subset_stats": subset_stats,
        "frame_to_subset_sample": {
            str(frame): frame_to_subset[frame] for frame in sorted(frame_to_subset)[:10]
        },
    }


def write_clean_annotation_csv(
    out_dir: Path, labels_by_frame: dict[int, list[Box]], splits: dict[str, list[int]]
) -> None:
    frame_to_subset = {
        frame: subset for subset, subset_frames in splits.items() for frame in subset_frames
    }
    csv_path = out_dir / "annotations_sampled_clean.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "split",
                "frame_num",
                "image",
                "class_id",
                "class_name",
                "x_center",
                "y_center",
                "width",
                "height",
            ]
        )
        for frame in sorted(labels_by_frame):
            subset = frame_to_subset[frame]
            image_name = f"images/{subset}/frame_{frame:06d}.jpg"
            for box in labels_by_frame[frame]:
                writer.writerow(
                    [
                        subset,
                        frame,
                        image_name,
                        box.cls_id,
                        CLASS_NAMES[box.cls_id],
                        f"{box.x_center:.6f}",
                        f"{box.y_center:.6f}",
                        f"{box.width:.6f}",
                        f"{box.height:.6f}",
                    ]
                )


def write_metadata(out_dir: Path, stats: dict, split_stats: dict) -> None:
    dataset_yaml = "\n".join(
        [
            f"path: {out_dir}",
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
    (out_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")

    report = {
        **stats,
        **split_stats,
        "image_size": {"width": TARGET_WIDTH, "height": TARGET_HEIGHT},
        "annotation_format": "YOLO txt: class_id x_center y_center width height, normalized 0-1",
        "class_names": CLASS_NAMES,
        "note": "The source CSV class 'trunk' is normalized to dataset class 'truck'.",
    }
    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    readme = f"""# Data_2: Zhandong Road Object Detection Dataset

This folder contains a prepared YOLO-format dataset for Exercise 1.

## Preparation choices

- Source video: `/Users/maksymchekhun/Desktop/A. Zhandong Road1.MP4`
- Source annotations: `/Users/maksymchekhun/Desktop/A.Zhandong Road1.csv`
- Sampling: one annotated frame every {stats["sample_every_frames"]} frames, starting at frame {stats["first_annotated_frame"]}; this is approximately one image per second.
- Resize: frames are resized from 1920x1080 to {TARGET_WIDTH}x{TARGET_HEIGHT}.
- Cleaning: bounding boxes are clipped to the image boundary and boxes smaller than 2 pixels after clipping are removed.
- Annotation format: YOLO normalized labels (`class_id x_center y_center width height`).
- Classes: `car`, `bus`, `truck` (`truck` comes from the source CSV spelling `trunk`).

## Folder structure

```text
Data_2/
  dataset.yaml
  dataset_report.json
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  annotations_sampled_clean.csv
```

## Result

- Sampled labeled images: {stats["labeled_sampled_frames"]}
- Train images: {split_stats["subset_stats"]["train"]["frames"]}
- Validation images: {split_stats["subset_stats"]["val"]["frames"]}
- Test images: {split_stats["subset_stats"]["test"]["frames"]}
- Total cleaned boxes: {sum(s["boxes"] for s in split_stats["subset_stats"].values())}

Use `dataset.yaml` directly with YOLO training tools.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    labels_by_frame, stats = load_selected_labels(csv_path, args.sample_every)
    frames = sorted(labels_by_frame)
    splits = split_frames(frames, args.train, args.val, args.test, args.seed)

    run_ffmpeg_extract(video_path, out_dir, frames[0], args.sample_every)
    split_stats = write_dataset(out_dir, labels_by_frame, splits)
    write_clean_annotation_csv(out_dir, labels_by_frame, splits)
    write_metadata(out_dir, stats, split_stats)

    print(json.dumps({**stats, **split_stats}, indent=2))


if __name__ == "__main__":
    main()
