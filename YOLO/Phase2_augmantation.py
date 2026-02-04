import os
import cv2
import random
import natsort
import numpy as np
import matplotlib.pyplot as plt
import string
from PIL import Image
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def load_files(ad):
    valid_extensions = {"tiff", "tif", "png", "jpg", "jpeg", "bmp", "gif", "webp"}
    FileNames = [file for file in sorted(os.listdir(ad)) if file.split(".")[-1].lower() in valid_extensions]
    return natsort.natsorted(FileNames)

def drop_loader(droplets_path):
    magnification = (np.random.random(size=(1)) * 2 + 1).item()
    droplet_path = os.path.join("dataset/.BK/Phase1_4S-SROF", random.choice(droplets_path))
    droplet = cv2.imread(droplet_path, cv2.IMREAD_UNCHANGED)
    h, w = droplet.shape[:2]
    droplet = cv2.resize(droplet, (int(w * magnification), int(h * magnification)), interpolation=cv2.INTER_LINEAR)
    droplet = cv2.cvtColor(droplet, cv2.COLOR_GRAY2RGB)
    _, mask = cv2.threshold(droplet, 1, 255, cv2.THRESH_BINARY)
    return droplet, mask

def place_droplet(background, droplet, mask, x, y):
    h, w = droplet.shape[:2]
    roi = background[y:y+h, x:x+w]
    roi[np.where(mask == 255)] = droplet[np.where(mask == 255)]
    return background, (x, y, w, h)

def save_yolo_bbox(file_name, x, y, w, h, img_width, img_height):
    cx, cy = (x + w / 2) / img_width, (y + h / 2) / img_height
    w_norm, h_norm = w / img_width, h / img_height
    yolo_bbox = f"0 {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}\n"
    txt_file = file_name.replace(".png", ".txt")
    with open(txt_file, "w") as f:
        f.write(yolo_bbox)

def generate_image(args):
    i, droplets_path, output_size, h_level, num_droplets = args
    background = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8) * 255
    for _ in range(num_droplets):
        file_name = ''.join(random.choice(string.ascii_lowercase) for _ in range(16)) + ".png"
        background[-h_level-5:-h_level, :] = np.random.randint(low=230, high=255, size=background[-h_level-5:-h_level, :].shape)
        background[-h_level:, :] = np.random.randint(low=1, high=35, size=background[-h_level:, :].shape)
        droplet, mask = drop_loader(droplets_path)
        h, w = droplet.shape[:2]
        x = random.randint(0, output_size[0] - w)
        y = output_size[1] - h - h_level
        background, bbox = place_droplet(background, droplet, mask, x, y)
        save_yolo_bbox(os.path.join('dataset', 'labels', file_name), *bbox, output_size[0], output_size[1])
        cv2.imwrite(os.path.join('dataset', 'images', file_name), background)

def main():
    output_size = (imgsz=640)
    h_level = 15
    droplets_path = load_files("dataset/.BK/Phase1_4S-SROF")
    num_droplets = np.random.randint(low=1, high=2)
    num_processes = cpu_count()
    
    args_list = [(i, droplets_path, output_size, h_level, num_droplets) for i in range(60_000)]
    
    with Pool(num_processes) as pool, tqdm(total=len(args_list)) as pbar:
        for _ in pool.imap_unordered(generate_image, args_list):
            pbar.update(1)

if __name__ == "__main__":
    main()
