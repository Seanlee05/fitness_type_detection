import torch
from PIL import Image, ImageDraw
from torchvision import transforms

# Placeholder for loading models
def load_yolov8_model():
    from ultralytics import YOLO
    return YOLO("weights/yolov8s.pt")

def load_vit_model():
    from transformers import ViTForImageClassification, ViTFeatureExtractor
    model = ViTForImageClassification.from_pretrained("facebook/vit-base-patch16-224")
    feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/vit-base-patch16-224")
    return model, feature_extractor

def run_yolov8_and_vit(image):
    # Run YOLOv8
    yolov8 = load_yolov8_model()
    results = yolov8(image)
    result_img = results[0].plot()

    # Run ViT
    vit_model, extractor = load_vit_model()
    inputs = extractor(images=image, return_tensors="pt")
    outputs = vit_model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

    return Image.fromarray(result_img), f"ViT Prediction: Class {predicted_class}"