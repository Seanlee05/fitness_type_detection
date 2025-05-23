import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from download_models import load_models

class_names = ["fit", "type1", "type2", "unknown"]
short_labels = {"fit": "f", "type1": "1", "type2": "2", "unknown": "u"}
colors = {
    "fit": (255, 0, 0),
    "type1": (0, 255, 0),
    "type2": (0, 0, 255),
    "unknown": (128, 0, 128)
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

yolo_model, vit_model, device = load_models()

def run_detection_and_classification(image_pil):
    image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    image_rgb = np.array(image_pil)

    results = yolo_model(image_rgb)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cropped = image_rgb[y1:y2, x1:x2]
        if cropped.size == 0 or (x2 - x1) < 10 or (y2 - y1) < 10:
            continue

        cropped_pil = Image.fromarray(cropped)
        input_tensor = transform(cropped_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = vit_model(input_tensor)
            pred_idx = output.argmax(1).item()
            label = class_names[pred_idx]

        color = colors[label]
        short = short_labels[label]
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 1)
        cv2.putText(image_bgr, short, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))