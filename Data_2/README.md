# Data_2: Zhandong Road Object Detection Dataset

This folder contains a prepared YOLO-format dataset for Exercise 1.

## Preparation choices

- Source video: `/Users/maksymchekhun/Desktop/A. Zhandong Road1.MP4`
- Source annotations: `/Users/maksymchekhun/Desktop/A.Zhandong Road1.csv`
- Sampling: one annotated frame every 30 frames, starting at frame 6; this is approximately one image per second.
- Resize: frames are resized from 1920x1080 to 1280x720.
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

- Sampled labeled images: 831
- Train images: 582
- Validation images: 166
- Test images: 83
- Total cleaned boxes: 78823

Use `dataset.yaml` directly with YOLO training tools.
