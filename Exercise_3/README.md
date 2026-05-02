# Exercise 3 - Traffic Flow Time Series

This folder estimates traffic flow entering the junction from each direction using the fine-tuned detector from Exercise 2.

## What it does

1. Loads chronological frames from `Data_2_v2`.
2. Runs the fine-tuned YOLO detector from Exercise 2.
3. Tracks detected vehicles using a simple centroid tracker.
4. Counts vehicles crossing four incoming-road counting lines:
   - `west_in`
   - `east_in`
   - `north_in`
   - `south_in`
5. Aggregates crossings into time-series bins.
6. Forecasts traffic flow with ARIMA/SARIMAX and compares it with a naive persistence baseline.

## Run after Exercise 2

First make sure this file exists:

```text
Exercise_2/runs/train/yolov8n_zhandong_v2/weights/best.pt
```

Install the extra forecasting dependency if needed:

```bash
pip install statsmodels scikit-learn
```

## Step 1: check counting lines

This is fast and does not run the detector:

```bash
python Exercise_3/exercise3_flow_forecasting.py --overlay-only --source zhandong_road1
```

Open:

```text
Exercise_3/results/counting_lines_overlay_zhandong_road1.jpg
```

If the lines are not positioned well, edit:

```text
Exercise_3/counting_lines.json
```

The line coordinates are in the resized image space, `1280x720`.

## Step 2: run traffic-flow pipeline

For the longer video:

```bash
python Exercise_3/exercise3_flow_forecasting.py --source zhandong_road1 --device 0 --batch 16
```

For the second video:

```bash
python Exercise_3/exercise3_flow_forecasting.py --source zhandong_road2 --device 0 --batch 16
```

Or run both combined:

```bash
python Exercise_3/exercise3_flow_forecasting.py --source all --device 0 --batch 16
```

## Expected outputs

Outputs are saved in `Exercise_3/results/`.

- `counting_lines_overlay_*.jpg`
- `detections_*.csv`
- `tracks_*.csv`
- `flow_frame_level_*.csv`
- `flow_10s_*.csv`
- `forecast_metrics_*.csv`
- `summary_*.json`
- `zhandong_road1/traffic_flow_timeseries.png`
- `zhandong_road1/forecast_total_flow.png`
- forecast plots for each incoming direction

## Report wording note

The available video duration is short, so a one-hour forecast is an extrapolation. In the report, describe it as a demonstration of the forecasting workflow rather than a production-grade long-horizon traffic forecast.
