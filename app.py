# app.py

import streamlit as st
import cv2
import numpy as np
import torch
from download_models import load_yolo_model, load_vit_model
from utils import run_inference  # We'll define this in utils.py

st.title("Fitness Type Detection with YOLOv8 + ViT")

device = torch.device("cpu")

@st.cache_resource(show_spinner=True)
def load_models():
    yolo = load_yolo_model()
    vit = load_vit_model(device)
    return yolo, vit

yolo_model, vit_model = load_models()

uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # === Resize if too big ===
    MAX_SIZE = 640
    if max(img_rgb.shape[:2]) > MAX_SIZE:
        scale = MAX_SIZE / max(img_rgb.shape[:2])
        img_rgb = cv2.resize(img_rgb, (0, 0), fx=scale, fy=scale)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)  # Update BGR version too
    
    st.image(img_rgb, caption='Uploaded Image', use_column_width=True)

    # Run detection and classification
    annotated_img = run_inference(img_bgr, img_rgb, yolo_model, vit_model, device)

    st.image(annotated_img, caption='Processed Image', use_column_width=True)

    del img_bgr, img_rgb
    import gc
    gc.collect()
