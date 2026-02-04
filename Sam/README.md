# SAM 2 Automated Annotation Tools

This directory contains tools utilizing the **Segment Anything Model 2 (SAM 2)** to assist in the annotation of high-speed video data.

## Purpose

Manual annotation of high-speed footage is labor-intensive. These scripts employ SAM 2 to generate "ground truth" labels for complex real-world scenarios where synthetic data is insufficient. This facilitates the semi-automated creation of high-quality training data for the YOLO model.

## File Description

*   **`dynamics.py`**: **Interactive SAM 2 Video Tracking Driver**
    *   Allows users to annotate an initial frame with prompts (points/boxes).
    *   Propagates these prompts through the video sequence using the SAM 2 video predictor.
    *   Exports masks and overlays for training data generation.

## Installation

Please refer to the installation instructions in the original [SAM 2 GitHub repository](https://github.com/facebookresearch/sam2?tab=readme-ov-file#installation).

## GUI Guide

Interactive prompt selection controls:

*   **Left click**: Add positive point
*   **Right click**: Add negative point
*   **'m'**: Toggle box mode (drag to draw box)
*   **'n'**: Move to next object
*   **'u'** or **Backspace**: Undo last point
*   **'q'**: Finish and close window

> [!TIP]
> **Performance Note**: SAM 2 is GPU-intensive. To handle long videos effectively, frames are flattened into patches within `dynamics.py`.

## Command Line Arguments

The `dynamics.py` script accepts the following arguments:

*   `--frame-root` (Required): Root directory containing frames or `batch_*` subfolders.
*   `--checkpoint`: Path to SAM 2 checkpoint (default: `checkpoints/sam2.1_hiera_large.pt`).
*   `--config`: Path to SAM 2 config (default: `configs/sam2.1/sam2.1_hiera_l.yaml`).
*   `--device`: Device to use (`auto`, `cuda`, `cpu`) (default: `auto`).
*   `--batch-pattern`: Glob pattern for batch subdirectories (default: `batch_*`).
*   `--interactive-frame-index`: Frame index to annotate (default: `0`).
*   `--max-frames`: Optional cap on total frames to process.
*   `--flatten-dir`: Optional directory to place flattened frames.
*   `--keep-flattened`: Keep flattened frames instead of deleting the temporary directory.
*   `--max-objects`: Optional limit on objects to annotate.
*   `--output-dir`: Directory to store tracking outputs.
*   `--save-overlays`: Save RGB overlays for each frame.
*   `--save-masks`: Save binary mask PNGs for each object and frame.
*   `--save-tracks`: Export centroid tracks as CSV files.
*   `--overlay-alpha`: Alpha value for mask overlays (default: `0.5`).
*   `--preview`: Preview masks on annotated frame before propagation.
*   `--supervision-stride`: Stride (in frames) for saving supervision overlays; set to 0 to disable (default: `20`).
*   `--chain-batches`: Process contiguous `batch_*` folders sequentially.
*   `--chain-tolerance`: Pixel padding applied to derived bounding boxes when chaining batches (default: `5`).
*   `--roi-file`: Optional ROI prompts JSON to load instead of opening the GUI.
*   `--roi-name`: Filename to search for ROI prompts under `frame-root`/`output-dir` (default: `roi_prompts.json`).