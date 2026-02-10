from ultralytics import YOLO
import torch
torch.cuda.is_available()
# Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
import os 

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