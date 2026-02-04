import  os
import  cv2
import  random
import  ultralytics
import  numpy                   as      np
import  matplotlib.pyplot       as      plt
from    PIL                     import  Image

def load_files(ad: str) -> list[str]:
    valid_extensions = {"tiff", "tif", "png", "jpg", "jpeg", "bmp", "gif", "webp"}  # Common image formats
    FileNames = []
    for file in sorted(os.listdir(ad)):
        try:
            if file.split(".")[-1].lower() in valid_extensions:
                FileNames.append(file)
        except IndexError:
            pass
    return sorted(FileNames)

def drop_loader(droplets_path: list[str], high: int) -> tuple[np.ndarray, np.ndarray]:
    magnification   = np.random.random(size=(1))*2+1

    droplet_path    = os.path.join("Phase1_4S-SROF",droplets_path[np.random.randint(low=0, high=high,)])
    droplet         = cv2.imread(droplet_path, cv2.IMREAD_UNCHANGED)
    # droplet = cv2.bitwise_not(droplet)
    h, w    = droplet.shape[:2]
    droplet = cv2.resize(droplet, (int(w*magnification),int(h*magnification))) 
    droplet = cv2.cvtColor(droplet, cv2.COLOR_GRAY2RGB)
    _, mask = cv2.threshold(droplet, 1, 255, cv2.THRESH_BINARY)
    return droplet, mask

def place_droplet(background, droplet, mask, x, y):
    h, w = droplet.shape[:2]
    roi = background[y:y+h, x:x+w]
    # Use mask to place droplet
    roi[np.where(mask == 255)] = droplet[np.where(mask == 255)]
    # Draw bounding box
    h, w = h-2, w-2
    # cv2.rectangle(background, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return background


if __name__ == "__main__":
# Define output image size
    output_size     = (1240, 1004)
    h_level         = 15


    droplets_path   = load_files("test")
    high            = len(droplets_path)


    background = np.ones((output_size[1], output_size[0], 3), dtype=np.uint8)*255

    num_droplets    = np.random.randint(low=1, high=2,)  # Number of droplets to place
    num_droplets_ll = np.zeros_like(num_droplets)

    for ii in range(num_droplets):
        background[-h_level-5:-h_level,:]   = np.random.randint(low=230, high=255,size=background[-h_level-5:-h_level,:].shape)
        background[-h_level:,:]             = np.random.randint(low=1, high=35,size=background[-h_level:,:].shape)
        droplet,mask                        = drop_loader(droplets_path, high)
        h, w                                = droplet.shape[:2]
        x                                   = random.randint(0, output_size[0] - w)
        y = output_size[1]-h- h_level#random.randint(0, output_size[1] - h)

        # TODO
            # add multi drop here

        background = place_droplet(background, droplet, mask, x, y)

    # Save and display the output
    output_path = "random_droplets.png"
    cv2.imwrite(output_path, background)
    print(f"Image saved to {output_path}")
