import os
import urllib.request

def download_weights():
    os.makedirs("weights", exist_ok=True)
    
    yolov8_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    vit_url = "https://github.com/facebookresearch/deit/releases/download/vit_base_patch16_224.pth/vit_base_patch16_224.pth"

    yolov8_path = "weights/yolov8s.pt"
    vit_path = "weights/vit_base_patch16_224.pth"

    if not os.path.exists(yolov8_path):
        print("Downloading YOLOv8 weights...")
        urllib.request.urlretrieve(yolov8_url, yolov8_path)

    if not os.path.exists(vit_path):
        print("Downloading ViT weights...")
        urllib.request.urlretrieve(vit_url, vit_path)