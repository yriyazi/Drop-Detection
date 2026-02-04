# Drop Detection Algorithms for High-Speed Recordings

**Authors**: Yassin Riyazi, Sajjad Shumaly, Leonard Ries, Oleksandra Kukharenko, Gunter Auernhammer, Peter Stephan, Hans-Jurgen Butt, Rudiger Berger.  
*(Max Planck Institute for Polymer Research & TU Darmstadt)*

## Overview

This project implements and evaluates algorithms for accurate **droplet localization** in high-speed imaging, a critical prerequisite for analyzing interfacial dynamics and contact angles. The work compares classical computer vision techniques with modern deep learning approaches to balance real-time performance with analytical precision.

## Project Structure

*   [**DropDetection/**](DropDetection/README.md)
    *   Contains classical image processing algorithms: **Sum-Based** (fast, approx. 407μs) and **Difference-Based** (robust to motion). Also includes edge detection logic using Euclidean distance sorting.
    
*   [**YOLO/**](YOLO/README.md)
    *   Contains the Deep Learning pipeline using **YOLOv11**. Includes scripts for **synthetic data generation**, training notebooks, and inference tools. This model achieves ~99.5% mAP@0.5.

*   [**Sam/**](Sam/README.md)
    *   Helper tools utilizing **SAM 2 (Segment Anything Model)** for semi-automated data annotation and ground-truth generation.

*   [**Report/**](Report/README.md)
    *   LaTeX source files for the accompanying manuscript/report detailing the methodology and benchmarks.

*   [**testSet/**](testSet/README.md)
    *   Directory for test dataset images.
    
## Dependencies

See [requirement.txt](requirement.txt) for a full list of Python dependencies.

## Key Findings

| Algorithm | Type | Execution Time | Strengths |
|-----------|------|----------------|-----------|
| **Sum-Based** | Heuristic | ~407 μs | Extremely fast, low memory. Good for real-time embedded use. |
| **Difference-Based** | Temporal | Fast (GPU accel) | Robust to static noise, no reference frame needed. |
| **YOLO** | Deep Learning | ~115 ms | High accuracy (99.5% mAP), robust to complex backgrounds/lighting. |

## Usage

1.  **Classical Detection**: Run scripts in `DropDetection/` for lightweight processing.
2.  **Training YOLO**: Use `YOLO/train.ipynb` to train on your own data or the provided synthetic generation pipeline.
3.  **Annotation**: Use `Sam/dynamics.py` to bootstrap labeling for new high-speed videos.
