# Data_2_v2: Combined Zhandong Road Object Detection Dataset

This is version 2 of the Exercise 1 dataset. It combines both provided annotated videos using the same preparation choices as `Data_2`.

## Preparation choices

- Sources: `A. Zhandong Road1.MP4` + `A. Zhandong Road2.MP4`
- Sampling: one annotated frame every 30 frames, approximately one image per second.
- Resize: frames are resized from 1920x1080 to 1280x720.
- Cleaning: bounding boxes are clipped to the image boundary and boxes smaller than 2 pixels after clipping are removed.
- Annotation format: YOLO normalized labels (`class_id x_center y_center width height`).
- Classes: `car`, `bus`, `truck` (`truck` comes from the source CSV spelling `trunk`).
- Split: 70% train, 20% validation, 10% test per source video.

## Result

- Total images: 1263
- Train images: 884
- Validation images: 252
- Test images: 127
- Total cleaned boxes: 115203

Use `dataset.yaml` directly with YOLO training tools.
