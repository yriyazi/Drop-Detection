"""
    Author:         Yassin Riyazi
    Date:           01.07.2025

    Description: 
        Detects drops in video frames using frame differencing and morphological operations.
"""

from email.mime import image
import cv2 
import matplotlib.pyplot as plt
import os
from typing import Tuple
from networkx import difference
import numpy as np
from numpy.typing import NDArray

class DropPreProcessor:
    """
    Preprocesses grayscale video frames to detect moving objects (e.g., drops) via frame differencing.
    
    This class supports both CPU and CUDA-accelerated pipelines using OpenCV.
    """

    def __init__(self, kernel_size: tuple[int, int] = (5, 5), 
                 threshold_val: int = 30, 
                 use_cuda: bool = False,
                 max_area: int = 500) -> None:
        """
        Initializes the preprocessor with morphological filters and CUDA setup.

        Args:
            kernel_size (tuple[int, int]): Size of the structuring element for morphological operations.
            threshold_val (int): Threshold value used to binarized the frame difference.
            use_cuda (bool): Whether to use CUDA acceleration if available.
            max_area (int): Minimum area of contours to consider as valid motion.
        """
        self.use_cuda       = use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.kernel         = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        self.threshold_val  = threshold_val
        self.max_area       = max_area

        if self.use_cuda:
            # Initialize CUDA-based morphological filters and GPU memory
            self.morph_open     = cv2.cuda.createMorphologyFilter(cv2.MORPH_OPEN, cv2.CV_8UC1, self.kernel)        # type: ignore
            self.morph_dilate   = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, self.kernel, anchor=(-1, -1), iterations=2)      # type: ignore
            self.gpu_prev       = cv2.cuda_GpuMat() # type: ignore  
            self.gpu_curr       = cv2.cuda_GpuMat() # type: ignore


    def process(self,
                prev_gray: cv2.Mat, 
                curr_gray: cv2.Mat) -> Tuple[int, int, int, int]:
        """
        Processes a pair of consecutive grayscale frames to extract contours of moving objects.

        Args:
            prev_gray (np.ndarray): Previous grayscale frame.
            curr_gray (np.ndarray): Current grayscale frame.

        Returns:
            Tuple[int, int, int, int]: Bounding box (x, y, w, h) of the largest detected motion.
        """
        if self.use_cuda:
            # Upload frames to GPU memory
            self.gpu_prev.upload(prev_gray) # type: ignore
            self.gpu_curr.upload(curr_gray) # type: ignore

            # Compute absolute difference and threshold on GPU
            diff_gpu = cv2.cuda.absdiff(self.gpu_prev, self.gpu_curr) # type: ignore
            _, thresh_gpu = cv2.cuda.threshold(diff_gpu, self.threshold_val, 255, cv2.THRESH_BINARY) # type: ignore

            # Apply morphological opening and dilation to reduce noise and connect components
            opened_gpu = self.morph_open.apply(thresh_gpu) # type: ignore
            dilated_gpu = self.morph_dilate.apply(opened_gpu) # type: ignore

            # Download the result back to CPU for contour detection
            dilated = dilated_gpu.download()# type: ignore
        else:
            # CPU fallback: frame differencing and morphology
            diff        = cv2.absdiff(prev_gray, curr_gray)
            _, thresh   = cv2.threshold(diff, self.threshold_val, 255, cv2.THRESH_BINARY)
            opened      = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
            dilated     = cv2.dilate(opened, self.kernel, iterations=2)

        # Contour detection using CPU
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine bounding boxes of all significant contours
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        found_any = False

        # sort contours by area and filter small ones
        contours = sorted([c for c in contours if cv2.contourArea(c) > self.max_area], key=cv2.contourArea, reverse=True)
        for cnt in contours:
            found_any = True
            bx, by, bw, bh = cv2.boundingRect(cnt)
            x_min = min(x_min, bx)
            y_min = min(y_min, by)
            x_max = max(x_max, bx + bw)
            y_max = max(y_max, by + bh)

        if found_any:
            x = int(x_min)
            y = int(y_min)
            w = int(x_max - x_min)
            h = int(y_max - y_min)
        else:
            x, y, w, h = 0, 0, 0, 0

        return x, y, w, h


def plot_rectangle(image: cv2.Mat, bbox: Tuple[int, int, int, int]) -> cv2.Mat:
    """
    Plots a bounding box on the given image.

    Args:
        image (cv2.Mat): The input image on which to draw the bounding box.
        bbox (Tuple[int, int, int, int]): The bounding box defined as (x, y, w, h).

    Returns:
        cv2.Mat: The image with the bounding box drawn.
    """
    x, y, w, h = bbox
    output_image = image.copy()
    cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return output_image

def perfMeasure(scaleDownFactor: int, 
                imageAddress1: str,
                imageAddress2: str)-> None:
    """
    Measure the performance of the drop detection algorithm.
    This function is a placeholder for performance measurement logic.
    args:
        scaleDownFactor (int): Factor to scale down the images for processing.
        imageAddress1 (str): Path to the first image file to be processed.
        imageAddress2 (str): Path to the second image file to be processed.
    returns:
        None: This function does not return any value.

    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/DropDetection/doc/draw_bounds_1754261882.9925468.png" alt="Italian Trulli" style="width: 800px; height: auto;">

    [detection_perf] Ran 1000 times
    [detection_perf] Avg execution time: 0.000442 seconds
    [detection_perf] Avg peak memory usage: 158.03 KiB
    """
    print(f"Performance Measurement for scaleDownFactor={scaleDownFactor}")

    from Performance_mesurements import average_performance,measure_performance # type: ignore

    difference = DropPreProcessor(kernel_size=(5, 5), threshold_val=2, use_cuda=False,max_area=1)
    @average_performance(runs=1_000)

    def detection_perf(image: cv2.Mat|NDArray[np.float64],scaleDownFactor: int):
        current_addr2 = imageAddress2
        max_attempts = 10
        
        for _ in range(max_attempts):
            image1 = cv2.imread(imageAddress1, cv2.IMREAD_GRAYSCALE)
            image2 = cv2.imread(current_addr2, cv2.IMREAD_GRAYSCALE)
            
            if image1 is None or image2 is None:
                break
                
            # Resize images
            if scaleDownFactor > 1:
                width = int(image1.shape[1]/ scaleDownFactor)
                height = int(image1.shape[0] / scaleDownFactor)
                new_size = (max(1, width), max(1, height))
                
                img1_small = cv2.resize(image1, new_size, interpolation=cv2.INTER_AREA)
                img2_small = cv2.resize(image2, new_size, interpolation=cv2.INTER_AREA)
                
                # Use a smaller kernel for smaller images to avoid eroding the object away
                # Or pass a proportionate kernel size to DropPreProcessor
                
                x, y, w, h = difference.process(img1_small, img2_small)
                
                # Scale coordinates back to original size
                if w > 0 and h > 0:
                    x = x * scaleDownFactor
                    y = y * scaleDownFactor
                    w = w * scaleDownFactor
                    h = h * scaleDownFactor
                    return (x, y, w, h)
            else:
                bbox = difference.process(image1, image2)
                if bbox != (0, 0, 0, 0):
                    return bbox

            # If no detection, move to next frame
            dirname, filename = os.path.split(current_addr2)
            name, ext = os.path.splitext(filename)
            if name.isdigit():
                next_num = int(name) + 1
                new_filename = f"{next_num:0{len(name)}d}{ext}"
                current_addr2 = os.path.join(dirname, new_filename)
                if not os.path.exists(current_addr2):
                    break
            else:
                break
                
        return (0, 0, 0, 0)
    
    # scaleDownFactor = 5
    # image_path      = "src/PyThon/ContactAngle/DropDetection/doc/Long Drop.jpg"
    image2           = cv2.imread(imageAddress2, cv2.IMREAD_GRAYSCALE)
    bbox = detection_perf(image2, scaleDownFactor)
    print(f"Measuring performance on image of shape: {bbox}")
    
    import matplotlib.patches as patches
    ax = plt.figure().add_subplot(111)                                          # type: ignore
    plt.imshow(image2, cmap='gray')                                             # type: ignore
    x, y, w, h = bbox
    # Create a Rectangle patch with red color
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.axis('on')  # or 'off' to hide axis
    ax.invert_yaxis()  # Invert y-axis to match the image orientation
    ax.set_ylim(image2.shape[0],image2.shape[0]-200)
    plt.savefig(os.path.join(os.path.dirname(__file__),f"draw_bounds_{scaleDownFactor}.png"), dpi=300, bbox_inches='tight')
    plt.show()

def extendLength(array: NDArray[np.float64],
                 new_length: int) -> NDArray[np.float64]:
    """
    Extend the length of the array to a new length by repeating the entire array.
    Args:
        array (np.ndarray): The input array to extend.
        new_length (int): The desired new length of the array.
    Returns:
        np.ndarray: The extended array with the new length.
    """
    # Example array
    x = len(array)

        # Repeat each element
    repeats = new_length // x  # Number of times to repeat each element (3)
    extended_arr = np.repeat(array, repeats)[:new_length]

    return extended_arr
def draw_bounds(image:cv2.Mat,
                start:int, end:int,
                scaleDownFactor:int, thickness:int=2,
                Array:NDArray[np.float64] | None = None,
                address: str | None = None) -> None:
    """
    Draw a rectangle on the image from start to end. 
    For testing purposes, it draws a rectangle on the image to visualize the detected drop.
    Args:
        image (cv2.Mat): Input image to draw the rectangle on.
        start (int): Starting index of the rectangle.
        end (int): Ending index of the rectangle.
        scaleDownFactor (int): Factor to scale down the image.
        thickness (int): Thickness of the rectangle border.
        address (str | None): Path to save the image with drawn bounds. If None, the image is not saved.
    Returns:
        None: Displays the image with the rectangle drawn.

    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/DropDetection/doc/DropDetection_Sum.png" alt="Italian Trulli" style="width: 800px; height: auto;">    
    os.path.relpath(os.path.join(os.path.dirname(__file__), 'doc',f"{__name__}.png"),'Main')
    """
    ax = plt.figure().add_subplot(111)                                          # type: ignore
    plt.imshow(image[::-1, :], alpha=0.5, cmap='gray')                          # type: ignore
    if Array is not None:
        _temp = Array*-255
        _temp = _temp - _temp.min()  # Normalize the sum of rows to have a minimum of 0
        _temp = extendLength(_temp, image.shape[1])
        ax.plot(_temp, color='blue', linewidth=thickness)                       # type: ignore
    ax.axis('on')  # or 'off' to hide axis
    ax.axvline(x=start*scaleDownFactor,    color='red', linewidth=thickness)    # type: ignore
    ax.axvline(x=end*scaleDownFactor,      color='red', linewidth=thickness)    # type: ignore
    ax.invert_yaxis()  # Invert y-axis to match the image orientation

if __name__ == "__main__":
    # difference = DropPreProcessor(kernel_size=(5, 5), threshold_val=2, use_cuda=False,
    #                               max_area=50)

    # image1 = cv2.imread("testSet/000003.jpg", cv2.IMREAD_GRAYSCALE)
    # image2 = cv2.imread("testSet/000004.jpg", cv2.IMREAD_GRAYSCALE)
    # bbox = difference.process(image1, image2)
    # output_image = plot_rectangle(image2, bbox)
    # cv2.imshow("Detected Droplet", output_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for i in [1,2,3,4 ]:
        perfMeasure(scaleDownFactor=i,
                    imageAddress1="testSet/000003.jpg",
                    imageAddress2="testSet/000006.jpg")