import  os
import  cv2
from    tqdm    import tqdm

import  matplotlib.pyplot as plt
from    multiprocessing import Pool, cpu_count

import natsort
import matplotlib.pyplot as plt
import numpy as np


def load_files(ad):
    valid_extensions = {"tiff", "tif", "png", "jpg", "jpeg", "bmp", "gif", "webp"}  # Common image formats
    FileNames = []
    for file in sorted(os.listdir(ad)):
        try:
            if file.split(".")[-1].lower() in valid_extensions:
                FileNames.append(file)
        except IndexError:
            pass
    return natsort.natsorted(FileNames)



def load_files(ad):
    valid_extensions = {"tiff", "tif", "png", "jpg", "jpeg", "bmp", "gif", "webp"}  # Common image formats
    FileNames = []
    for file in sorted(os.listdir(ad)):
        try:
            if file.split(".")[-1].lower() in valid_extensions:
                FileNames.append(file)
        except IndexError:
            pass
    return natsort.natsorted(FileNames)

def process_image(filename, input_dir, output_dir):
    adress = os.path.join(input_dir, filename)
    # Load image in grayscale
    image = cv2.imread(adress, cv2.IMREAD_GRAYSCALE)
    # Apply thresholding to segment the droplet
    _, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)

    x, y = np.where(binary == 0)
    x1, y1 = y[y.argmin()], x[x.argmin()]
    x2, y2 = y[y.argmax()], binary.shape[-1]
    # binary = binary[y1:y2, x1:x2]
    image = image[y1-2:y2, x1:x2+2]
    image = cv2.bitwise_not(image)

    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, image)

def process_images_parallel(input_dir, output_dir):
    filenames = load_files(input_dir)
    
    # Create a pool of workers to process the images in parallel
    with Pool(cpu_count()) as pool:
        # Use tqdm to show the progress bar while processing
        list(tqdm(pool.starmap(process_image, [(filename, input_dir, output_dir) for filename in filenames]), total=len(filenames)))

if __name__ == "__main__":
    input_directory = "4S-SROF"
    output_directory = "Phase1_4S-SROF"  # Specify your desired output directory here

    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    
    process_images_parallel(input_directory, output_directory)


# if __name__ == "__main__":
#     filenames = load_files("4S-SROF")

#     for filename in filenames:
#         adress = os.path.join("4S-SROF",filename)
#         # Load image in grayscale
#         image = cv2.imread(adress, cv2.IMREAD_GRAYSCALE)
#         # Apply thresholding to segment the droplet
#         _, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)

#         x,y = np.where(binary==0)
#         x1,y1 = y[y.argmin()],x[x.argmin()]
#         x2,y2 = y[y.argmax()],binary.shape[-1]
#         binary = binary[y1:y2, x1:x2]

#         cv2.imwrite(adress, binary)

