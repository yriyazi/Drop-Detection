"""
    Remove light source reflection from an image.
    Author: Yassin Riyazi
    Date: 20-08-2025
"""
import os
import colorama
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from typing import List, Tuple, Union
from tqdm import tqdm
from numpy.typing import NDArray

def DropBoundaryExtractor(imageAddress:str,
                         outputAddress:str,) -> None:
    """
    Returning the boundary of drop.
    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/Viscosity/lightSource/doc/DropBoundryExtractor.png" alt="Italian Trulli">
    
    Args:
        imageAddress (str): The file path to the input image.
        outputAddress (str): The file path to save the output image.
        
    Returns:
        None: none
    """
    # Load the image
    source_img = cv2.imread(imageAddress, cv2.IMREAD_GRAYSCALE)
    if source_img is None:
        raise ValueError("Image not found or unable to load.")

    # Invert the image
    inverted_img = cv2.bitwise_not(source_img)

    # Apply Gaussian blur to reduce noise
    blurred_img = cv2.GaussianBlur(inverted_img, (5, 5), 0)

    # Thresholding to create a binary mask
    _, binary_mask = cv2.threshold(blurred_img, 10, 250, cv2.THRESH_BINARY_INV)

    # Find contours of the light source reflection
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the largest contour
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(source_img)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Invert the mask to get the area without reflection
    mask_inv = cv2.bitwise_not(mask)

    # Apply the mask to the original image
    result_img = cv2.bitwise_and(source_img, source_img, mask=mask_inv)

    # Save or display the result
    cv2.imwrite(outputAddress, result_img)

def LightSourceReflectionRemover(source_img:NDArray[np.uint8],
                                 threshold_activation:int = 100) -> NDArray[np.uint8]:
    """
    Remove light source reflection from an image.
    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/Viscosity/lightSource/doc/LightSourceReflectionRemover.png" alt="Italian Trulli">
    
    Args:
        imageAddress (str): The file path to the input image.
        outputAddress (str): The file path to save the output image.
        threshold_activation (int, optional): The threshold value for reflection removal. Defaults to 100.

    Returns:
        None: none
    """
    assert isinstance(source_img, np.ndarray), colorama.Fore.RED + " Input must be a NumPy array " + colorama.Style.RESET_ALL
    assert source_img.ndim == 2, colorama.Fore.RED + " Input must be a grayscale image (2D array) " + colorama.Style.RESET_ALL
    assert source_img[0,:].sum() < 5, colorama.Fore.RED + " Probably not bitWised " + colorama.Style.RESET_ALL # It actual save my time in 05-09-2025
    # Invert the image
    # source_img = cv2.bitwise_not(source_img)

    # Apply Gaussian blur to reduce noise
    # source_img = cv2.GaussianBlur(source_img, (7,7), 0)
    kernel = np.ones((5,5),np.uint8)
    source_img = cv2.morphologyEx(source_img, cv2.MORPH_OPEN, kernel)

    # Thresholding to create a binary mask
    _, binary_mask = cv2.threshold(source_img, 10, 250, cv2.THRESH_BINARY_INV)

    # Find contours of the light source reflection
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the largest contour
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(source_img)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    inside = (mask <= threshold_activation).astype(np.uint8) * 255
    # inside = cv2.bitwise_not(inside)
    return inside

if __name__ == "__main__":
    img = cv2.imread('/media/Dont/Teflon-AVP/280/S3-SNr3.07_D/T105_11_79.813535314440/databases/frame_002781.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Original', img)

    vv = LightSourceReflectionRemover(img, threshold_activation=100)
    cv2.imshow('Result', vv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()