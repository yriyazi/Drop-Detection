import torch
import os 
import colorama
from ultralytics import YOLO

if not torch.cuda.is_available():
    print(colorama.Fore.RED + "CUDA is not available. Training will be performed on CPU, which may be significantly slower.\n" + colorama.Style.RESET_ALL)
if torch.backends.cudnn.enabled:
    print(colorama.Fore.GREEN + "cuDNN is enabled. This may improve training performance on compatible NVIDIA GPUs.\n" + colorama.Style.RESET_ALL)
else:
    print("cuDNN is not enabled. Training may be slower on NVIDIA GPUs without cuDNN support.")

root = os.path.dirname(os.path.abspath(__file__))
name = os.path.join(root, "models/yolo12n.pt")
model = YOLO(name)  # load a pretrained model (recommended for training)


# model = YOLO(name).load(name)  # build from YAML and transfer weights

model.train(data    = os.path.join(root, "Droplet.yaml"),
            epochs  = 3,
            workers = 8,
            batch   = 6,
            device  = 0 if torch.cuda.is_available() else "cpu",
            imgsz=(320, 1240), 
            # cache = False , #"ram",
            # resume=True
            )