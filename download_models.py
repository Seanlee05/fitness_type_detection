import torch
from ultralytics import YOLO
import timm

def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo_model = YOLO('yolov8n.pt')  # You may upload your own weights
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=4)
    vit_model.to(device).eval()

    return yolo_model, vit_model, device