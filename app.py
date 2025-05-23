import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from download_models import load_models
from utils import run_detection_and_classification

st.set_page_config(page_title="Fitness Type Detection", layout="centered")

st.title("🏋️‍♂️ Fitness Type Detection (YOLOv8 + ViT)")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running detection and classification..."):
        annotated_image = run_detection_and_classification(image)
        st.image(annotated_image, caption="Output Image", use_column_width=True)