# Classical Drop Detection Algorithms

This directory contains implementations of classical computer vision algorithms for droplet detection in high-speed imaging, as described in the report. These methods are designed for computational efficiency and real-time processing.

## Files

*   **`DropDetection_Sum.py`**: Implements the **Sum-Based Drop Detection** algorithm.
    *   *Method*: Projects the 2D image into a 1D vertical intensity profile via column-wise summation. The profile is normalized, and horizontal boundaries are determined by scanning for significant deviations in the signal derivative (controlled by a slope threshold $\epsilon$).
    *   *Features*: Extremely fast (~407 μs), low memory usage.
    *   *Use Case*: Real-time embedded applications, initial presence detection.

*   **`DropDetection_Difference.py`**: Implements the **Difference-Based Drop Detection** algorithm.
    *   *Method*: Implements a **Relative Difference** (Temporal Differencing) approach. It computes the absolute difference between consecutive frames ($\Delta I = |I_t - I_{t-1}|$) to highlight motion (advancing/receding edges). A motion mask is generated via binary thresholding and refined using morphological operations (opening/dilation) to bridge gaps and remove noise.
    *   *Features*: Robust to static background noise, requires no reference frame.
    *   *Use Case*: High-speed footage with continuous motion.

*   **`edgeDetection.py`**: Core logic for boundary extraction and edge processing.
    *   Includes the **Euclidean Distance-Based Pixel Selection** strategy to mitigate bias in contact line extraction (as detailed in the report).
    *   Extracts advancing/receding angles and apex coordinates.

*   **`DropCoordinateSystem.py`**: Handles coordinate system transformations, establishing the substrate baseline ($y=0$) and normalizing drop positions.

*   **`LightSourceReflectionRemoving.py`**: Pre-processing utility to mitigate specular reflections and lighting artifacts common in high-speed experimental setups.
