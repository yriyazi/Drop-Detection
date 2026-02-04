# YOLOv11 Droplet Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/Model-YOLOv11-orange)
![Status](https://img.shields.io/badge/Status-Active-green)

This directory contains the implementation of a deep learning-based detection pipeline utilizing **YOLOv11**. Designed for high-speed fluid dynamics research, this model provides robust localization of droplets against complex backgrounds, varying lighting conditions, and dynamic noise artifacts where traditional computer vision methods often degrade.

## 🚀 Overview

The pipeline consists of three core stages:
1.  **Synthetic Data Generation**: Creating a massive, diverse training set from limited experimental samples.
2.  **Model Training**: Fine-tuning YOLOv11 on the generated dataset.
3.  **Inference & Analysis**: Converting detections into experimental coordinates for contact angle measurement.

---

## 🏗️ Directory Structure

```plaintext
YOLO/
├── dataset/                # Dataset directory (Images & Labels)
│   ├── images/             # Training and validation images
│   ├── labels/             # YOLO format labels (.txt)
│   ├── classes.txt         # Class definition file
│   └── notes.json          # Dataset metadata
├── models/                 # Pre-trained and fine-tuned model checkpoints
│   ├── yolo11n.pt          # Nano base model
│   └── yolo12*.pt          # Trained weights
├── runs/                   # Training logs, metrics, and partial results
├── Droplet.yaml            # YOLO configuration file describing classes and paths
├── Phase1_Preprocess.py    # Step 1: Extracts droplet templates from raw frames
├── Phase2_augmantation.py  # Step 2: Generates synthetic scenes using templates
├── train.ipynb             # Training notebook
├── YOLO_test.ipynb         # Inference and evaluation notebook
└── YOLO_coordinates.py     # Utility to convert bounding boxes to Cartesian coordinates
```

---

## 🛠️ Workflow & Usage

### 1. Data Preparation
To overcome data scarcity, we employ a two-phase synthetic generation strategy.

#### **Phase 1: Droplet Extraction**
*   **Script**: `Phase1_Preprocess.py`
*   **Function**: Ingests raw experimental high-speed frames, segments the droplets via thresholding, and creates a library of isolated droplet templates.
*   **Output**: Normalized, cropped binary/grayscale templates stored for augmentation.

#### **Phase 2: Synthetic Augmentation**
*   **Script**: `Phase2_augmantation.py`
*   **Function**: Composites the templates from Phase 1 onto procedurally generated backgrounds.
    *   *Features*: Simulates sensor noise, surface reflections, and varying drop scales (1x - 3x).
    *   *Output*: A fully annotated dataset (images + YOLO text labels) ready for training.

### 2. Model Training
We utilize the Ultralytics YOLOv11 framework.

*   **Configuration**: Defined in `Droplet.yaml`.
    *   **Classes**: Single class detection (`Droplet`).
    *   **Paths**: Points to local `dataset/` directory.
*   **Execution**: Open `train.ipynb` to initiate training.
    ```python
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")  # Load pre-trained model
    model.train(data="Droplet.yaml", epochs=300, imgsz=640)
    ```

### 3. Inference & Evaluation
*   **Execution**: Use `YOLO_test.ipynb` for visual verification and testing.
*   **Coordinate Conversion**: Use `YOLO_coordinates.py` to transform model outputs (pixel, bounding boxes) into scientific metrics (contact points, substrate baseline).

---

## 📊 Performance Metrics

The model has been benchmarked on a validation set representing diverse experimental conditions.

| Metric | Score | Description |
| :--- | :--- | :--- |
| **mAP@0.5** | **99.5%** | Mean Average Precision at 0.5 IoU threshold. |
| **Precision** | **1.00** | Achieved at high confidence thresholds (~0.89). |
| **Recall** | **1.00** | No droplets missed at operational thresholds. |
| **Inference Time** | **~115 ms** | On standard laptop GPU (NVIDIA Ada 2000m). |

---

## 📦 Requirements

*   **Python**: 3.8+
*   **Libraries**:
    *   `ultralytics` (YOLOv v11/12 framework)
    *   `opencv-python`
    *   `numpy`
    *   `matplotlib`
    *   `tqdm`

Ensure all dependencies from the root `requirement.txt` are installed.
