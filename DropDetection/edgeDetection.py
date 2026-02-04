"""
    Edited by: Yassin Riyazi
    Main Author: Sajjad Shumaly
    Date: 01-07-2025
    Description: This module provides functions for edge extraction from images,

    Changelog:
        1.  improve on the edge_extraction function, refer to the docstring for details.
"""
import  os
import  colorama
import  numpy                   as      np
import  matplotlib.pyplot       as      plt
from    typing                  import  Tuple, List # type: ignore
from    numpy.typing            import  NDArray


def edge_extraction(gray:NDArray[np.int8], 
                    thr:int=40
                    ) -> tuple[NDArray[np.int8], NDArray[np.int8]] | tuple[List[int], List[int]]:
    """
    Extract edge pixels from an upscaled image using a threshold.
    Caution:
        Images are supposed to be bitwise_not

    
    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/CaMeasurer/doc/edge_extraction_thr_10.png" alt="Italian Trulli" style="width: 800px; height: auto;">

    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/CaMeasurer/doc/edge_extraction_thr_100.png" alt="Italian Trulli" style="width: 800px; height: auto;">
    
    This function detects the first pixel above the threshold from the left, right,
    and top of the image to form a rough outline of detected objects. Duplicate points
    are removed.

    Parameters:
        upscaled_image (np.ndarray): Input BGR image (as NumPy array or PIL Image).
        thr (int): Threshold value for pixel intensity (0-255).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of (i_list, j_list) representing the x and y
                                     coordinates of edge points (with vertical flip on y).
    Author:
        - Yassin Riyazi (Using SIMD for speedup)
        - Sajjad Shumaly
    """

    assert isinstance(gray, np.ndarray), colorama.Fore.RED + " Input must be a NumPy array " + colorama.Style.RESET_ALL
    assert gray.ndim == 2, colorama.Fore.RED + " Input must be a grayscale image (2D array) " + colorama.Style.RESET_ALL
    assert gray[0,:].sum() < 5, colorama.Fore.RED + " Probably not bitWised " + colorama.Style.RESET_ALL # It actual save my time in 05-09-2025

    height, width = gray.shape

    # Mask where intensity is greater than threshold
    mask = gray > thr

    # Allocate edge pixel lists
    i_list: List[np.int8] = []
    j_list: List[np.int8] = []

    # External left edge (first hit in each row from the left)
    left_hits       = np.argmax(mask, axis=1)
    has_hit_left    = mask[np.arange(height), left_hits]
    rows_left       = np.where(has_hit_left)[0]

    i_list.extend(left_hits[rows_left])
    j_list.extend(rows_left)

    # External right edge (first hit in each row from the right)
    right_hits      = width - 1 - np.argmax(mask[:, ::-1], axis=1)
    has_hit_right   = mask[np.arange(height), right_hits]
    rows_right      = np.where(has_hit_right)[0]

    i_list.extend(right_hits[rows_right])
    j_list.extend(rows_right)

    # External top edge (first hit in each column from the top)
    top_hits = np.argmax(mask, axis=0)
    has_hit_top = mask[top_hits, np.arange(width)]
    cols_top = np.where(has_hit_top)[0]
    i_list.extend(cols_top)
    j_list.extend(top_hits[cols_top])

    # Remove duplicates and flip y-coordinates
    coords = set(zip(i_list, j_list))
    
    if not coords:
        return [], []

    i_list, j_list  = zip(*coords)
        
    j_list = [height - 1 - j for j in j_list]  # flip y-axis

    return np.array(i_list), np.array(j_list)

def Advancing_pixel_selection_Euclidean(i_array: List[int]|NDArray[np.int64],
                                        j_array: List[int]|NDArray[np.int64],
                                        left_number_of_pixels:int=150
                                        ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Selects pixels from the advancing (left) side of a droplet, sorted by 2D Euclidean distance
    from the leftmost point, returning specified number of pixels.

    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/CaMeasurer/doc/advancing_pixel_selection_Euclidean_advacingPoints10.png" alt="Italian Trulli">

    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/CaMeasurer/doc/advancing_pixel_selection_Euclidean_advacingPoints90.png" alt="Italian Trulli">
    
    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/CaMeasurer/doc/advancing_pixel_selection_Euclidean_advacingPoints150.png" alt="Italian Trulli">
    

    Args:
        i_array (List[int]|NDArray[np.int64]): x-coordinates (horizontal positions) of edge pixels.
        j_array (List[int]|NDArray[np.int64]): y-coordinates (vertical positions) of edge pixels.
        left_number_of_pixels (int): Number of pixels to return.

    Returns:
        Tuple[List[int], List[int]]: Selected advancing edge pixels (x, y).

    Author:
        - Yassin Riyazi (Norm2 based selection)
    """
    # Convert to numpy arrays once
    if not isinstance(i_array, np.ndarray):
        i_array = np.array(i_array, dtype=np.int64)
    if not isinstance(j_array, np.ndarray):
        j_array = np.array(j_array, dtype=np.int64)

    if i_array.size == 0:
        raise ValueError("Input arrays for advancing contact angle calculation are empty.")

    # Find origin (leftmost x-coordinate)
    origin_x = np.min(i_array)

    # Vectorized Euclidean distance calculation
    distances = np.sqrt(np.square(i_array - origin_x) + np.square(j_array))

    # Get indices of sorted distances
    sorted_indices = np.argsort(distances)[:left_number_of_pixels]

    # # Select pixels
    selected_i = i_array[sorted_indices]#.tolist()
    selected_j = j_array[sorted_indices]#.tolist()

    return selected_i, selected_j

def Receding_pixel_selection_Euclidean(i_array: List[int]|NDArray[np.int64],
                                      j_array: List[int]|NDArray[np.int64],
                                      right_number_of_pixels:int=150
                                      ) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    """
    Selects pixels from the receding (right) side of a droplet, sorted by 2D Euclidean distance
    from the leftmost point, returning specified number of pixels from both ends.

    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/CaMeasurer/doc/Receding_pixel_selection_Euclidean_advacingPoints10.png" alt="Italian Trulli">

    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/CaMeasurer/doc/Receding_pixel_selection_Euclidean_advacingPoints90.png" alt="Italian Trulli">

    <img src="https://raw.githubusercontent.com/YassinRiyazi/Main/refs/heads/main/src/PyThon/ContactAngle/CaMeasurer/doc/Receding_pixel_selection_Euclidean_advacingPoints150.png" alt="Italian Trulli">



    Args:
        i_list (List[int]): x-coordinates (horizontal positions) of edge pixels.
        j_list (List[int]): y-coordinates (vertical positions) of edge pixels.
        left_number_of_pixels (int): Number of pixels to return from each end (total 2*left_number_of_pixels).

    Returns:
        Tuple[List[int], List[int]]: Selected receding edge pixels (x, y).

    Author:
        - Yassin Riyazi (Norm2 based selection)
    """
    # Convert to numpy arrays once
    if not isinstance(i_array, np.ndarray):
        i_array = np.array(i_array, dtype=np.int64)
    if not isinstance(j_array, np.ndarray):
        j_array = np.array(j_array, dtype=np.int64)

    if i_array.size == 0:
        raise ValueError("Input arrays for receding contact angle calculation are empty.")
    
    # Find origin (leftmost x-coordinate)
    origin_x = np.max(i_array)

    # Vectorized Euclidean distance calculation
    distances = np.sqrt((i_array - origin_x)**2 + j_array**2)

    # Get indices of sorted distances
    sorted_indices = np.argsort(distances)[:right_number_of_pixels]

    # Select pixels
    selected_i = i_array[sorted_indices]#.tolist()
    selected_j = j_array[sorted_indices]#.tolist()

    return selected_i, selected_j


if __name__ == "__main__":
    pass