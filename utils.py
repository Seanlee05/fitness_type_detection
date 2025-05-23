# utils.py

import cv2
import torch
from torchvision import transforms

class_names = ["fit", "type1", "type2", "unknown"]
short_labels = {"fit": "f", "type1": "1", "type2": "2", "unknown": "u"}
colors = {
    "fit": (255, 0, 0),       # Blue
    "type1": (0, 255, 0),     # Green
    "type2": (0, 0, 255),     # Red
    "unknown": (128, 0, 128)  # Purple
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def run_inference(image_bgr, image_rgb, yolo_model, vit_model, device):
    results = yolo_model(image_rgb)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cropped = image_rgb[y1:y2, x1:x2]

        if cropped.size == 0 or (x2 - x1) < 10 or (y2 - y1) < 10:
            continue

        input_tensor = transform(cropped).unsqueeze(0).to(device)
        with torch.no_grad():
            output = vit_model(input_tensor)
            pred_idx = output.argmax(1).item()
            label = class_names[pred_idx]

        short = short_labels[label]
        color = colors[label]

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 1)
        cv2.putText(image_bgr, short, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return image_bgr
