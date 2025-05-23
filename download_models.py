import os
import torch
import timm
from ultralytics import YOLO
import requests

# ==== Hugging Face Model URLs ====
VIT_URL = "https://huggingface.co/Seanlee05/Fitness_Type_Detection/resolve/main/vit_fitness_type_cls.pth"
YOLO_URL = "https://huggingface.co/Seanlee05/Fitness_Type_Detection/resolve/main/yolov8s_teeth_detection.pt"

# ==== Download Directory ====
WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ==== Helper: Download File If Not Exists ====
def download_file(url, save_path):
    if not os.path.exists(save_path):
        print(f"📥 Downloading: {url}")
        response = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✅ Saved to: {save_path}")
    else:
        print(f"✅ Found existing: {save_path}")

# ==== Download and Load YOLOv8 Model ====
def load_yolo_model():
    yolo_path = os.path.join(WEIGHTS_DIR, "yolov8s_teeth_detection.pt")
    download_file(YOLO_URL, yolo_path)
    model = YOLO(yolo_path)
    return model

# ==== Download and Load ViT Model ====
def load_vit_model(device):
    vit_path = os.path.join(WEIGHTS_DIR, "vit_fitness_type_cls.pth")
    download_file(VIT_URL, vit_path)
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(vit_path, map_location=device))
    model.to(device).eval()
    return model
