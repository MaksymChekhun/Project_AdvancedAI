#!/usr/bin/env python3
"""Exercise 3: traffic flow time series and forecasting.

Run this after Exercise 2 has produced a fine-tuned YOLO checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Data_2_v2"
EXERCISE_DIR = PROJECT_ROOT / "Exercise_3"
RESULTS_DIR = EXERCISE_DIR / "results"
DEFAULT_WEIGHTS = PROJECT_ROOT / "Exercise_2" / "runs" / "train" / "yolov8n_zhandong_v2" / "weights" / "best.pt"
DEFAULT_LINES = EXERCISE_DIR / "counting_lines.json"
IMAGE_RE = re.compile(r"(?P<source>zhandong_road\d+)_frame_(?P<frame>\d+)\.jpg$", re.IGNORECASE)
CLASS_NAMES = {0: "car", 1: "bus", 2: "truck"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--lines", default=str(DEFAULT_LINES))
    parser.add_argument("--source", default="zhandong_road1", help="Use zhandong_road1, zhandong_road2, or all.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--bin-seconds", type=int, default=10)
    parser.add_argument("--forecast-steps", type=int, default=360, help="360 ten-second bins = 1 hour.")
    parser.add_argument("--max-match-distance", type=float, default=95.0)
    parser.add_argument("--max-missed-frames", type=int, default=2)
    parser.add_argument("--overlay-only", action="store_true", help="Only draw counting lines; do not run detector.")
    return parser.parse_args()


def import_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install Ultralytics first: pip install ultralytics") from exc
    return YOLO


def load_counting_lines(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    lines = {}
    for name, line in raw.items():
        lines[name] = {
            "p1": tuple(line["p1"]),
            "p2": tuple(line["p2"]),
            "axis": line["axis"],
            "direction": int(line["direction"]),
            "description": line.get("description", ""),
        }
    return lines


def collect_images(source: str) -> pd.DataFrame:
    rows = []
    for subset in ["train", "val", "test"]:
        for path in sorted((DATASET_DIR / "images" / subset).glob("*.jpg")):
            match = IMAGE_RE.match(path.name)
            if not match:
                continue
            source_id = match.group("source")
            frame = int(match.group("frame"))
            if source != "all" and source_id != source:
                continue
            rows.append({"source_id": source_id, "frame": frame, "subset": subset, "path": path})
    if not rows:
        raise SystemExit(f"No images found for source={source!r}")
    return pd.DataFrame(rows).sort_values(["source_id", "frame"]).reset_index(drop=True)


def draw_counting_lines(image_path: Path, lines: dict, output_path: Path) -> None:
    colors = {
        "west_in": (255, 90, 90),
        "east_in": (90, 180, 255),
        "north_in": (255, 220, 90),
        "south_in": (90, 230, 140),
    }
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        for name, line in lines.items():
            color = colors.get(name, (255, 255, 255))
            p1, p2 = line["p1"], line["p2"]
            draw.line([p1, p2], fill=color, width=5)
            draw.rectangle([p1[0], p1[1] - 20, p1[0] + 150, p1[1]], fill=(0, 0, 0))
            draw.text((p1[0] + 4, p1[1] - 17), name, fill=color)
            if line["axis"] == "x":
                y_mid = (p1[1] + p2[1]) // 2
                x1 = p1[0]
                x2 = x1 + 65 * line["direction"]
                draw.line([(x1, y_mid), (x2, y_mid)], fill=color, width=4)
            else:
                x_mid = (p1[0] + p2[0]) // 2
                y1 = p1[1]
                y2 = y1 + 65 * line["direction"]
                draw.line([(x_mid, y1), (x_mid, y2)], fill=color, width=4)
        image.save(output_path, quality=92)


def run_detector(weights: Path, image_df: pd.DataFrame, imgsz: int, conf: float, device: str, batch: int) -> pd.DataFrame:
    if not weights.exists():
        raise SystemExit(f"Missing detector weights: {weights}. Run Exercise 2 training first.")
    YOLO = import_yolo()
    model = YOLO(str(weights))
    detections = []
    paths = [str(path) for path in image_df["path"].tolist()]
    meta_by_path = {str(row.path): row for row in image_df.itertuples()}

    results = model.predict(
        source=paths,
        imgsz=imgsz,
        conf=conf,
        device=device,
        batch=batch,
        stream=False,
        verbose=False,
    )
    for result in results:
        row = meta_by_path[str(Path(result.path))]
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id not in CLASS_NAMES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
            detections.append(
                {
                    "source_id": row.source_id,
                    "frame": int(row.frame),
                    "image": Path(row.path).name,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "confidence": float(box.conf.item()),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                }
            )
    return pd.DataFrame(detections)


def crosses_line(previous_center: tuple[float, float], current_center: tuple[float, float], line: dict) -> bool:
    px, py = previous_center
    cx, cy = current_center
    p1, p2 = line["p1"], line["p2"]
    if line["axis"] == "x":
        line_x = p1[0]
        y_min, y_max = sorted([p1[1], p2[1]])
        crossed = px < line_x <= cx if line["direction"] > 0 else px > line_x >= cx
        return crossed and ((y_min <= py <= y_max) or (y_min <= cy <= y_max))
    line_y = p1[1]
    x_min, x_max = sorted([p1[0], p2[0]])
    crossed = py < line_y <= cy if line["direction"] > 0 else py > line_y >= cy
    return crossed and ((x_min <= px <= x_max) or (x_min <= cx <= x_max))


def track_and_count(
    image_df: pd.DataFrame,
    detections_df: pd.DataFrame,
    lines: dict,
    max_match_distance: float,
    max_missed_frames: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if detections_df.empty:
        raise SystemExit("Detector produced no usable detections.")

    active_tracks = {}
    next_track_id = 1
    flow_rows = []
    track_rows = []
    detections_by_key = {
        key: group.sort_values("confidence", ascending=False).to_dict("records")
        for key, group in detections_df.groupby(["source_id", "frame"])
    }

    for source_id, source_images in image_df.groupby("source_id"):
        active_tracks.clear()
        # The dataset keeps original video frame IDs, sampled every N frames
        # (for example 36, 66, 96...). Tracking should use the sequential image
        # index, not the raw frame-number gap, otherwise every vehicle track is
        # discarded between sampled frames.
        for frame_index, frame in enumerate(sorted(source_images["frame"].unique())):
            frame_dets = detections_by_key.get((source_id, int(frame)), [])
            counts = {"source_id": source_id, "frame": int(frame), **{name: 0 for name in lines}}
            assigned_tracks = set()
            assigned_detections = set()

            track_items = sorted(active_tracks.items(), key=lambda item: item[1]["last_frame"])
            for det_index, det in enumerate(frame_dets):
                center = np.array([det["cx"], det["cy"]], dtype=float)
                best_track_id = None
                best_distance = float("inf")
                for track_id, track in track_items:
                    if track_id in assigned_tracks:
                        continue
                    if track["class_id"] != det["class_id"]:
                        continue
                    if frame_index - track["last_index"] > max_missed_frames + 1:
                        continue
                    distance = float(np.linalg.norm(center - np.asarray(track["center"])))
                    if distance < best_distance:
                        best_distance = distance
                        best_track_id = track_id

                if best_track_id is None or best_distance > max_match_distance:
                    best_track_id = next_track_id
                    next_track_id += 1
                    previous_center = None
                else:
                    previous_center = active_tracks[best_track_id]["center"]

                if previous_center is not None:
                    for line_name, line in lines.items():
                        if crosses_line(previous_center, (det["cx"], det["cy"]), line):
                            already_counted = active_tracks[best_track_id].setdefault("counted_lines", set())
                            if line_name not in already_counted:
                                counts[line_name] += 1
                                already_counted.add(line_name)

                active_tracks[best_track_id] = {
                    "center": (det["cx"], det["cy"]),
                    "last_frame": int(frame),
                    "last_index": frame_index,
                    "class_id": det["class_id"],
                    "counted_lines": active_tracks.get(best_track_id, {}).get("counted_lines", set()),
                }
                assigned_tracks.add(best_track_id)
                assigned_detections.add(det_index)
                track_rows.append(
                    {
                        "source_id": source_id,
                        "frame": int(frame),
                        "track_id": best_track_id,
                        "class_name": det["class_name"],
                        "cx": det["cx"],
                        "cy": det["cy"],
                        "confidence": det["confidence"],
                    }
                )

            for track_id in list(active_tracks):
                if frame_index - active_tracks[track_id]["last_index"] > max_missed_frames:
                    del active_tracks[track_id]

            counts["total_flow"] = sum(counts[name] for name in lines)
            flow_rows.append(counts)

    return pd.DataFrame(flow_rows), pd.DataFrame(track_rows)


def aggregate_flow(flow_df: pd.DataFrame, line_names: list[str], bin_seconds: int) -> pd.DataFrame:
    outputs = []
    for source_id, group in flow_df.groupby("source_id"):
        group = group.sort_values("frame").reset_index(drop=True)
        group["time_second"] = np.arange(len(group))
        group["time_bin"] = group["time_second"] // bin_seconds
        agg = group.groupby("time_bin")[[*line_names, "total_flow"]].sum().reset_index()
        agg.insert(0, "source_id", source_id)
        outputs.append(agg)
    return pd.concat(outputs, ignore_index=True)


def naive_forecast(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    history = list(train)
    predictions = []
    for actual in test:
        predictions.append(history[-1] if history else 0.0)
        history.append(actual)
    return np.asarray(predictions, dtype=float)


def arima_forecast(train: np.ndarray, steps: int) -> tuple[np.ndarray, str]:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        return np.repeat(float(train[-1]) if len(train) else 0.0, steps), "fallback_no_statsmodels"

    for order in [(1, 0, 1), (1, 0, 0), (0, 0, 1)]:
        try:
            model = SARIMAX(
                train,
                order=order,
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False, maxiter=200)
            forecast = np.asarray(result.forecast(steps=steps), dtype=float)
            return np.maximum(forecast, 0.0), f"ARIMA{order}"
        except Exception:
            continue
    return np.repeat(float(train[-1]) if len(train) else 0.0, steps), "fallback_last_value"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"MAE": mae, "RMSE": rmse}


def forecast_and_plot(agg_df: pd.DataFrame, line_names: list[str], forecast_steps: int, output_dir: Path) -> pd.DataFrame:
    import matplotlib.pyplot as plt

    rows = []
    columns = [*line_names, "total_flow"]
    for source_id, source_df in agg_df.groupby("source_id"):
        source_dir = output_dir / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        for column in columns:
            series = source_df[column].to_numpy(dtype=float)
            train_size = max(3, int(len(series) * 0.8))
            train, test = series[:train_size], series[train_size:]
            if len(test) == 0:
                continue
            naive_test = naive_forecast(train, test)
            arima_test, arima_name = arima_forecast(train, len(test))
            naive_future = np.repeat(float(series[-1]) if len(series) else 0.0, forecast_steps)
            arima_future, _ = arima_forecast(series, forecast_steps)

            for model_name, pred in [("naive_persistence", naive_test), (arima_name, arima_test)]:
                row = {
                    "source_id": source_id,
                    "series": column,
                    "model": model_name,
                    **metrics(test, pred),
                }
                rows.append(row)

            x = np.arange(len(series))
            test_x = np.arange(train_size, len(series))
            future_x = np.arange(len(series), len(series) + forecast_steps)
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(x, series, label="observed", linewidth=1.4)
            ax.plot(test_x, naive_test, label="naive test", linestyle="--")
            ax.plot(test_x, arima_test, label=f"{arima_name} test")
            ax.plot(future_x, naive_future, label="naive future", linestyle="--")
            ax.plot(future_x, arima_future, label=f"{arima_name} future")
            ax.axvline(train_size, color="black", linestyle=":", linewidth=1)
            ax.axvline(len(series), color="gray", linestyle=":", linewidth=1)
            ax.set_title(f"{source_id} {column} forecast")
            ax.set_xlabel("time bin")
            ax.set_ylabel("vehicle crossings")
            ax.grid(alpha=0.25)
            ax.legend(ncols=2)
            fig.tight_layout()
            fig.savefig(source_dir / f"forecast_{column}.png", dpi=160)
            plt.close(fig)

    return pd.DataFrame(rows)


def plot_time_series(agg_df: pd.DataFrame, line_names: list[str], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    for source_id, source_df in agg_df.groupby("source_id"):
        columns = [*line_names, "total_flow"]
        fig, axes = plt.subplots(len(columns), 1, figsize=(13, 10), sharex=True)
        for ax, column in zip(axes, columns):
            ax.plot(source_df["time_bin"], source_df[column], linewidth=1.3)
            ax.set_title(column)
            ax.set_ylabel("crossings")
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("time bin")
        fig.tight_layout()
        source_dir = output_dir / source_id
        source_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(source_dir / "traffic_flow_timeseries.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = load_counting_lines(Path(args.lines))
    image_df = collect_images(args.source)

    overlay_source = image_df.iloc[len(image_df) // 2]["path"]
    overlay_path = RESULTS_DIR / f"counting_lines_overlay_{args.source}.jpg"
    draw_counting_lines(overlay_source, lines, overlay_path)
    print(f"Saved counting line overlay: {overlay_path}")

    if args.overlay_only:
        return

    detections_path = RESULTS_DIR / f"detections_{args.source}.csv"
    flow_path = RESULTS_DIR / f"flow_frame_level_{args.source}.csv"
    agg_path = RESULTS_DIR / f"flow_{args.bin_seconds}s_{args.source}.csv"
    metrics_path = RESULTS_DIR / f"forecast_metrics_{args.source}.csv"

    detections_df = run_detector(Path(args.weights), image_df, args.imgsz, args.conf, args.device, args.batch)
    detections_df.to_csv(detections_path, index=False)
    print(f"Saved detections: {detections_path}")

    flow_df, tracks_df = track_and_count(
        image_df,
        detections_df,
        lines,
        args.max_match_distance,
        args.max_missed_frames,
    )
    flow_df.to_csv(flow_path, index=False)
    tracks_df.to_csv(RESULTS_DIR / f"tracks_{args.source}.csv", index=False)
    print(f"Saved frame-level flow: {flow_path}")

    line_names = list(lines.keys())
    agg_df = aggregate_flow(flow_df, line_names, args.bin_seconds)
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved aggregated flow: {agg_path}")

    plot_time_series(agg_df, line_names, RESULTS_DIR)
    forecast_metrics = forecast_and_plot(agg_df, line_names, args.forecast_steps, RESULTS_DIR)
    forecast_metrics.to_csv(metrics_path, index=False)
    print(f"Saved forecast metrics: {metrics_path}")

    summary = {
        "source": args.source,
        "images": len(image_df),
        "detections": len(detections_df),
        "total_crossings": {name: int(flow_df[name].sum()) for name in line_names},
        "total_flow": int(flow_df["total_flow"].sum()),
        "bin_seconds": args.bin_seconds,
        "forecast_steps": args.forecast_steps,
        "outputs": {
            "overlay": str(overlay_path),
            "detections": str(detections_path),
            "flow_frame_level": str(flow_path),
            "flow_aggregated": str(agg_path),
            "forecast_metrics": str(metrics_path),
        },
    }
    summary_path = RESULTS_DIR / f"summary_{args.source}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
