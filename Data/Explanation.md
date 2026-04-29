# Explanation of Exercise 2 and Exercise 3

## Exercise 2 - Traffic Object Detection

The goal of Exercise 2 was to train a computer vision model that can detect traffic participants in aerial images. For this task, I used YOLOv8n. I chose YOLO because it is a common object detection model, it is fast, and it works well for detecting many objects in one image. YOLOv8n is the small version of YOLOv8, so it is easier and faster to train while still giving useful results.

The dataset contains aerial/top-down traffic images. The relevant classes are:

- `Cars`
- `Bus_Truck`
- `Pedestrian`
- `Two_Wheeler`

The original annotations were in COCO format. Since YOLO needs labels in YOLO format, the annotations were converted. The prepared dataset was split into 3,200 training images and 816 held-out test images. The model was fine-tuned for 50 epochs using the pretrained `yolov8n.pt` weights.

I also evaluated the original pretrained YOLOv8n model before fine-tuning. This was important because the task asks to compare a pretrained model with a fine-tuned model. The pretrained model was trained mostly on normal street-level COCO images, not aerial traffic images. Because of this, it performed very poorly on the aerial dataset.

Pretrained YOLOv8n baseline:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Pretrained YOLOv8n | 0.0667 | 0.0039 | 0.0073 |

After fine-tuning, the model performed much better:

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Fine-tuned YOLOv8n | 0.8605 | 0.5503 | 0.5892 | 0.4932 |

These results show that fine-tuning was necessary. The pretrained model did not understand the aerial viewpoint well, but after training on aerial traffic images it learned to detect the objects much better. Cars were detected especially well because they are the most common class in the dataset. The two-wheeler class is weaker because there are very few two-wheeler examples, so the model has less data to learn from.

The final fine-tuned model is saved as:

`TrafficProject/results/fine_tuned_artifacts/weights/best.pt`

The results, plots, confusion matrix, and example detections are saved in:

`TrafficProject/results/fine_tuned_artifacts/`

## Exercise 3 - Traffic Flow Time Series

The goal of Exercise 3 was to use the output of the detection model to estimate traffic flow entering the junction from each direction, create time series data, and forecast future traffic.

For the final version of Exercise 3, I used the notebook:

`Task_3_v2.ipynb`

This version uses counting lines, which matches the task description more closely than simple region counting.

The source video was sampled at one frame per second. This means:

`1 frame = 1 second`

The dataset images are not full video frames. They are 640x640 crops from the original aerial video. The filename contains the frame number and crop position, for example:

`frame_0010_y440_x512...jpg`

This tells us that the crop belongs to frame 10 and starts at position `x=512`, `y=440` in the full image.

First, the fine-tuned YOLO model detects vehicles in each crop. Then each detection is converted back into the full 1920x1080 intersection coordinate system:

`global_x = crop_x + detected_x`

`global_y = crop_y + detected_y`

This is needed because the counting lines are drawn on the full intersection, not inside one crop.

Some crops overlap, so the same vehicle can be detected more than once. To reduce duplicate detections, I used non-maximum suppression with an IoU threshold of 0.5. In simple words, if two boxes overlap a lot and represent the same class in the same frame, only the best detection is kept.

After that, I used a simple centroid tracker. The tracker follows the center point of each detected vehicle from frame to frame. If a vehicle center crosses one of the counting lines in the correct direction, it is counted as traffic entering the junction from that road.

The four incoming directions are:

- `west_in`
- `east_in`
- `north_in`
- `south_in`

Only vehicle classes are counted:

- `Cars`
- `Bus_Truck`
- `Two_Wheeler`

Pedestrians are not included in traffic flow.

The line-crossing method counted the following total vehicle crossings in the available video:

| Direction | Crossings |
|---|---:|
| `west_in` | 74 |
| `east_in` | 122 |
| `north_in` | 89 |
| `south_in` | 59 |
| `total_flow` | 344 |

These numbers are not predictions. They are the actual counted crossings from the detection and tracking pipeline. The predictions come later in the forecasting step.

The line-crossing counts were converted into a time series. Since line crossings are sparse, I aggregated the data into 10-second bins. This makes the time series smoother and easier to forecast.

For forecasting, I used SARIMAX/ARIMA. I chose SARIMAX because it is suitable for short time series and is explicitly related to ARIMA, which is mentioned in the task. I did not use LSTM because the available data is too short for a neural network. After aggregation, there are only around 82 time points. An LSTM would likely overfit and learn noise instead of real traffic patterns. I also did not use Prophet because Prophet works best with longer time series that contain daily or weekly seasonality, which this short video does not have.

The SARIMAX model was compared with a simple baseline method called naive persistence. Naive persistence predicts that the next value will be the same as the last observed value. This baseline is important because it shows whether the more advanced forecasting model is actually better than a very simple method.

The forecast horizon was set to one hour:

`1 hour = 3600 seconds = 360 future 10-second bins`

The task does not specify the exact forecast length, so one hour is a reasonable choice. The available video is only about 13.6 minutes long, so longer forecasts would be very uncertain.

Forecast evaluation results:

| Series | Model | MAE | RMSE | MAPE |
|---|---|---:|---:|---:|
| west_in | SARIMAX | 0.77 | 0.87 | 2.01 |
| west_in | Naive persistence | 0.47 | 0.77 | 75.00 |
| east_in | SARIMAX | 1.34 | 1.45 | 47.49 |
| east_in | Naive persistence | 1.06 | 1.46 | 91.67 |
| north_in | SARIMAX | 1.06 | 1.33 | 30.08 |
| north_in | Naive persistence | 1.35 | 1.93 | 110.71 |
| south_in | SARIMAX | 0.81 | 0.86 | 24.89 |
| south_in | Naive persistence | 0.41 | 0.73 | 100.00 |
| total_flow | SARIMAX | 2.19 | 2.48 | 128.92 |
| total_flow | Naive persistence | 1.65 | 2.11 | 66.67 |

The results show that SARIMAX does not always beat the naive baseline. For total flow, the naive baseline has lower MAE and RMSE. This happens because the video is short and the crossing time series is sparse. Many 10-second bins have zero or only a few vehicle crossings, so simply repeating the last value can be surprisingly strong.

This does not mean the method failed. The task asks to compare the forecasting model with a simple baseline, and the comparison shows an important result: for this short dataset, a simple baseline is difficult to beat. SARIMAX still gives a proper time-series forecast, but the forecast should be interpreted as a rough projection, not a reliable long-term traffic prediction.

The main Exercise 3 v2 outputs are saved in:

`TrafficProject/results/task3_v2_line_crossing/`

Important files include:

- `task3_v2_counting_lines_overlay.jpg`
- `task3_v2_line_crossing_timeseries.csv`
- `task3_v2_line_crossing_timeseries_10sec.csv`
- `task3_v2_line_crossing_timeseries.png`
- `task3_v2_forecast_metrics.csv`
- `task3_v2_forecast_plot_total_flow.png`

Overall, Exercise 3 v2 satisfies the task because it uses the detection model output, counts vehicles crossing incoming-road lines, creates directional traffic-flow time series, forecasts future traffic, and compares the forecasting model with a simple baseline.
