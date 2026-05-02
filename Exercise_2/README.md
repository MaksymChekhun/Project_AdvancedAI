# Exercise 2 - Traffic Object Detection

This folder contains a ready-to-run YOLOv8 pipeline for Question 2 using the combined `Data_2_v2` dataset.

## Dataset

- Dataset folder: `../Data_2_v2`
- Classes: `car`, `bus`, `truck`
- Train/val/test images: `884 / 252 / 127`
- Total boxes: `115,203`

## Important note about the manual annotation requirement

The folder `manual_annotation_10` contains 10 images and prefilled YOLO labels. Use a labeling tool such as LabelImg, CVAT, Roboflow, or Label Studio to inspect and manually correct/save those 10 label files. The labels are prefilled from the CSV annotations, so this acts as assisted labeling, but you should still manually review them before writing the report.

## Install on the training PC

```bash
cd /path/to/Project_AdvancedAI
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install ultralytics pandas matplotlib seaborn pillow opencv-python
```

If your PC has an NVIDIA GPU, install the CUDA-enabled PyTorch build if Ultralytics does not detect CUDA automatically.

## Run order

Prepare/recreate the manual annotation subset:

```bash
python Exercise_2/exercise2_yolo_pipeline.py --prepare-manual-subset
```

Evaluate the pretrained COCO model. This remaps COCO classes `car`, `bus`, and `truck` to the project classes before computing metrics:

```bash
python Exercise_2/exercise2_yolo_pipeline.py --baseline --device 0
```

Train the fine-tuned model:

```bash
python Exercise_2/exercise2_yolo_pipeline.py --train --epochs 80 --imgsz 960 --batch 16 --device 0
```

This uses early stopping with `--patience 15`, so training stops if validation performance does not improve for 15 epochs. The script also enables moderate augmentation and closes mosaic augmentation for the final 10 epochs, which usually helps YOLO settle on cleaner validation performance.

Evaluate the fine-tuned model:

```bash
python Exercise_2/exercise2_yolo_pipeline.py --evaluate-finetuned --device 0
```

Create the comparison plot/table:

```bash
python Exercise_2/exercise2_yolo_pipeline.py --compare
```

Save example detections for the report:

```bash
python Exercise_2/exercise2_yolo_pipeline.py --examples --device 0
```

## Expected outputs

- `pretrained_baseline_metrics.json`
- `runs/train/yolov8n_zhandong_v2/weights/best.pt`
- `finetuned_metrics.json`
- `comparison_summary.csv`
- `comparison_metrics.png`
- `example_detections/finetuned_examples/`

## Rough training-time estimate

For `yolov8n`, `80` epochs, `imgsz=960`, `batch=16`, and 884 training images:

- RTX 4090 / 4080: about 20-45 minutes
- RTX 4070 / 4070 Ti: about 35-75 minutes
- RTX 3060 / 4060: about 60-120 minutes
- CPU-only: many hours, not recommended

If the GPU runs out of memory, reduce `--batch` to `8` or reduce `--imgsz` to `640`.

## Overfitting checks

After training, inspect:

- `Exercise_2/runs/train/yolov8n_zhandong_v2/results.png`
- `Exercise_2/runs/train/yolov8n_zhandong_v2/results.csv`

Healthy training: train loss and validation loss both decrease or flatten, while validation mAP improves and then plateaus.

Possible overfitting: train loss keeps falling but validation loss rises and validation mAP stops improving. In that case, keep the `best.pt` checkpoint, reduce epochs, or train with `--patience 10`.
