# Final comparison: pretrained vs fine-tuned YOLOv8n

| Model | Data used | Evaluation set | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---:|---:|---:|---:|---:|
| Pretrained YOLOv8n | COCO pretrained weights only | 64-image held-out baseline subset | 0.0667 | 0.0039 | 0.0073 | - | - |
| Fine-tuned YOLOv8n | 3,200 annotated aerial traffic images | 816-image held-out test split | 0.8605 | 0.5503 | - | 0.5892 | 0.4932 |

The pretrained model barely detects the aerial traffic participants because the camera angle and object scale are very different from COCO street-level images. After fine-tuning on the aerial dataset, the model reaches about 0.5892 mAP50 on the held-out test split.

## Example detections

![Before/after detection 1](TrafficProject/results/comparison/before_after_01.jpg)
![Before/after detection 2](TrafficProject/results/comparison/before_after_02.jpg)
![Before/after detection 3](TrafficProject/results/comparison/before_after_03.jpg)

## Saved artifacts

Fine-tuned weights and validation figures are saved in `TrafficProject/results/fine_tuned_artifacts/`. The rerun validation output is saved in `TrafficProject/results/finetuned_validation_rerun/`.
