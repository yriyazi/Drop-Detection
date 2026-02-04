# Drop Detection Algorithms for High-Speed Recordings

**Authors**: Yassin Riyazi, Sajjad Shumaly, Leonard Ries, Oleksandra Kukharenko, Gunter Auernhammer, Peter Stephan, Hans-Jurgen Butt, Rudiger Berger.  
*(Max Planck Institute for Polymer Research & TU Darmstadt)*

## Overview

The analysis of interfacial dynamics, particularly wettability, relies heavily on contact angle (CA) measurements taken using high-speed imaging. Accurate localisation of the droplet within these image sequences is the fundamental first step.

This project implements and benchmarks three distinct methods: a **heuristic sum-based approach**, a **temporal difference-based technique**, and a **neural network-based YOLO pipeline**. These methods explicitly explore the trade-off between computational speed and spatial accuracy:
*   **Lightweight algorithms** (Sum/Difference) identify droplet location within a budget of ~2ms per frame, ideal for **real-time camera triggering systems**.
*   **Deep Learning** (YOLO) enables robust, high-precision detection essential for detailed **post-processing tasks** like contact angle measurement, despite higher computational cost.

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

## Acknowledgements

We would like to thank the **Max Planck Society** for funding this research. This work was conducted as part of a larger project.
