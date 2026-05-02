#!/usr/bin/env python3
"""Create Data_2_v2 by combining both Zhandong Road annotated videos."""

from __future__ import annotations

import csv
import json
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
SAMPLE_EVERY = 30
SEED = 42
CLASS_MAP = {"car": 0, "bus": 1, "trunk": 2}
CLASS_NAMES = ["car", "bus", "truck"]
SOURCES = [
    {
        "source_id": "zhandong_road1",
        "video": Path("/Users/maksymchekhun/Desktop/A. Zhandong Road1.MP4"),
        "csv": Path("/Users/maksymchekhun/Desktop/A.Zhandong Road1.csv"),
    },
    {
        "source_id": "zhandong_road2",
        "video": Path("/Users/maksymchekhun/Desktop/A. Zhandong Road2.MP4"),
        "csv": Path("/Users/maksymchekhun/Desktop/A.Zhandong Road2.csv"),
    },
]


@dataclass(frozen=True)
class Box:
    cls_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def load_selected_labels(csv_path: Path) -> tuple[dict[int, list[Box]], dict]:
    labels_by_frame: dict[int, list[Box]] = defaultdict(list)
    source_counts = Counter()
    clean_counts = Counter()
    skipped_counts = Counter()
    total_rows = 0
    min_frame = None
    max_frame = None

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

            if (frame - min_frame) % SAMPLE_EVERY != 0:
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

            cls_id = CLASS_MAP[name]
            labels_by_frame[frame].append(
                Box(
                    cls_id=cls_id,
                    x_center=((xmin + xmax) / 2.0) / ORIGINAL_WIDTH,
                    y_center=((ymin + ymax) / 2.0) / ORIGINAL_HEIGHT,
                    width=width / ORIGINAL_WIDTH,
                    height=height / ORIGINAL_HEIGHT,
                )
            )
            clean_counts[CLASS_NAMES[cls_id]] += 1

    if min_frame is None or max_frame is None:
        raise ValueError(f"No annotation rows found in {csv_path}")

    expected_frames = list(range(min_frame, max_frame + 1, SAMPLE_EVERY))
    labels_by_frame = {
        frame: labels_by_frame[frame] for frame in expected_frames if labels_by_frame[frame]
    }
    return labels_by_frame, {
        "source_csv_rows": total_rows,
        "first_annotated_frame": min_frame,
        "last_annotated_frame": max_frame,
        "expected_sampled_frames": len(expected_frames),
        "labeled_sampled_frames": len(labels_by_frame),
        "source_class_counts_on_sampled_frames": dict(source_counts),
        "clean_class_counts": dict(clean_counts),
        "skipped_counts": dict(skipped_counts),
    }


def extract_frames(video_path: Path, tmp_dir: Path, first_frame: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH.")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    zero_based_start = first_frame - 1
    filter_expr = (
        f"select='gte(n\\,{zero_based_start})*not(mod(n-{zero_based_start}\\,{SAMPLE_EVERY}))',"
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}"
    )
    subprocess.run(
        [
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
        ],
        check=True,
    )


def split_source_frames(items: list[tuple[str, int]], train: float = 0.70, val: float = 0.20) -> dict:
    by_source: dict[str, list[int]] = defaultdict(list)
    for source_id, frame in items:
        by_source[source_id].append(frame)

    splits = {"train": [], "val": [], "test": []}
    rng = random.Random(SEED)
    for source_id, frames in by_source.items():
        shuffled = frames[:]
        rng.shuffle(shuffled)
        train_end = round(len(shuffled) * train)
        val_end = train_end + round(len(shuffled) * val)
        splits["train"].extend((source_id, frame) for frame in shuffled[:train_end])
        splits["val"].extend((source_id, frame) for frame in shuffled[train_end:val_end])
        splits["test"].extend((source_id, frame) for frame in shuffled[val_end:])

    for subset in splits:
        splits[subset] = sorted(splits[subset])
    return splits


def write_dataset(out_dir: Path, all_labels: dict[str, dict[int, list[Box]]], splits: dict) -> dict:
    for stale_dir in [out_dir / "images", out_dir / "labels"]:
        if stale_dir.exists():
            shutil.rmtree(stale_dir)

    for subset in splits:
        (out_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

    subset_stats = {}
    annotation_rows = []
    sequence_maps = {
        source_id: {frame: idx for idx, frame in enumerate(sorted(labels), start=1)}
        for source_id, labels in all_labels.items()
    }

    for subset, items in splits.items():
        box_counter = Counter()
        for source_id, frame in items:
            sequence_index = sequence_maps[source_id][frame]
            src = out_dir / "_tmp_frames" / source_id / f"sample_{sequence_index:06d}.jpg"
            if not src.exists():
                raise RuntimeError(f"Missing extracted image: {src}")

            image_name = f"{source_id}_frame_{frame:06d}.jpg"
            label_name = f"{source_id}_frame_{frame:06d}.txt"
            shutil.copy2(src, out_dir / "images" / subset / image_name)

            with (out_dir / "labels" / subset / label_name).open("w", encoding="utf-8") as fh:
                for box in all_labels[source_id][frame]:
                    box_counter[CLASS_NAMES[box.cls_id]] += 1
                    label_line = (
                        f"{box.cls_id} {box.x_center:.6f} {box.y_center:.6f} "
                        f"{box.width:.6f} {box.height:.6f}"
                    )
                    fh.write(label_line + "\n")
                    annotation_rows.append(
                        [
                            subset,
                            source_id,
                            frame,
                            f"images/{subset}/{image_name}",
                            box.cls_id,
                            CLASS_NAMES[box.cls_id],
                            f"{box.x_center:.6f}",
                            f"{box.y_center:.6f}",
                            f"{box.width:.6f}",
                            f"{box.height:.6f}",
                        ]
                    )

        subset_stats[subset] = {
            "images": len(items),
            "boxes": sum(box_counter.values()),
            "class_counts": dict(box_counter),
        }

    with (out_dir / "annotations_sampled_clean.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "split",
                "source_id",
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
        writer.writerows(annotation_rows)

    shutil.rmtree(out_dir / "_tmp_frames")
    return subset_stats


def write_metadata(out_dir: Path, source_stats: dict, subset_stats: dict) -> None:
    (out_dir / "dataset.yaml").write_text(
        "\n".join(
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
        ),
        encoding="utf-8",
    )

    total_images = sum(stat["images"] for stat in subset_stats.values())
    total_boxes = sum(stat["boxes"] for stat in subset_stats.values())
    report = {
        "dataset_version": "Data_2_v2_combined",
        "sources": source_stats,
        "sample_every_frames": SAMPLE_EVERY,
        "image_size": {"width": TARGET_WIDTH, "height": TARGET_HEIGHT},
        "classes": CLASS_NAMES,
        "annotation_format": "YOLO txt: class_id x_center y_center width height, normalized 0-1",
        "subset_stats": subset_stats,
        "total_images": total_images,
        "total_boxes": total_boxes,
        "note": "The source CSV class 'trunk' is normalized to dataset class 'truck'.",
    }
    (out_dir / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    readme = f"""# Data_2_v2: Combined Zhandong Road Object Detection Dataset

This is version 2 of the Exercise 1 dataset. It combines both provided annotated videos using the same preparation choices as `Data_2`.

## Preparation choices

- Sources: `A. Zhandong Road1.MP4` + `A. Zhandong Road2.MP4`
- Sampling: one annotated frame every {SAMPLE_EVERY} frames, approximately one image per second.
- Resize: frames are resized from 1920x1080 to {TARGET_WIDTH}x{TARGET_HEIGHT}.
- Cleaning: bounding boxes are clipped to the image boundary and boxes smaller than 2 pixels after clipping are removed.
- Annotation format: YOLO normalized labels (`class_id x_center y_center width height`).
- Classes: `car`, `bus`, `truck` (`truck` comes from the source CSV spelling `trunk`).
- Split: 70% train, 20% validation, 10% test per source video.

## Result

- Total images: {total_images}
- Train images: {subset_stats["train"]["images"]}
- Validation images: {subset_stats["val"]["images"]}
- Test images: {subset_stats["test"]["images"]}
- Total cleaned boxes: {total_boxes}

Use `dataset.yaml` directly with YOLO training tools.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    out_dir = Path("/Users/maksymchekhun/Desktop/Project_AdvancedAI/Data_2_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_labels = {}
    source_stats = {}
    split_items = []

    for source in SOURCES:
        source_id = source["source_id"]
        labels, stats = load_selected_labels(source["csv"])
        all_labels[source_id] = labels
        source_stats[source_id] = stats
        split_items.extend((source_id, frame) for frame in labels)

        tmp_dir = out_dir / "_tmp_frames" / source_id
        extract_frames(source["video"], tmp_dir, min(labels))

    splits = split_source_frames(split_items)
    subset_stats = write_dataset(out_dir, all_labels, splits)
    write_metadata(out_dir, source_stats, subset_stats)
    print(json.dumps({"sources": source_stats, "subset_stats": subset_stats}, indent=2))


if __name__ == "__main__":
    main()
