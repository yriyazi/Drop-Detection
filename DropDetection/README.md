# Classical Drop Detection Algorithms

This directory contains implementations of classical computer vision algorithms for droplet detection in high-speed imaging, as described in the report. These methods are designed for computational efficiency and real-time processing.

## Files

*   **`DropDetection_Sum.py`**: Implements the **Sum-Based Drop Detection** algorithm.
    *   *Method*: Projects 2D image intensities into a 1D vertical profile and analyzes the slope to find horizontal boundaries.
    *   *Features*: Extremely fast (~407 μs), low memory usage.
    *   *Use Case*: Real-time embedded applications, initial presence detection.

*   **`DropDetection_Difference.py`**: Implements the **Difference-Based Drop Detection** algorithm.
    *   *Method*: Uses temporal frame differencing ($|I_t - I_{t-1}|$) to detect motion (droplet edges).
    *   *Features*: Robust to static background noise, requires no reference frame.
    *   *Use Case*: High-speed footage with continuous motion.

*   **`edgeDetection.py`**: Core logic for boundary extraction and edge processing.
    *   Includes the **Euclidean Distance-Based Pixel Selection** strategy to mitigate bias in contact line extraction (as detailed in the report).
    *   Extracts advancing/receding angles and apex coordinates.

*   **`DropCoordinateSystem.py`**: Handles coordinate system transformations, establishing the substrate baseline ($y=0$) and normalizing drop positions.

*   **`LightSourceReflectionRemoving.py`**: Pre-processing utility to mitigate specular reflections and lighting artifacts common in high-speed experimental setups.
