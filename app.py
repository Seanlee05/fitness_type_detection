import streamlit as st
from utils import run_yolov8_and_vit
from download_models import download_weights
from PIL import Image
import os

st.set_page_config(page_title="Fitness Type Detection", layout="centered")
st.title("🏋️‍♂️ Fitness Type Detection using YOLOv8 + ViT")

# Download models if not already downloaded
download_weights()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running detection..."):
        result_img, predictions = run_yolov8_and_vit(image)
        st.image(result_img, caption="Detection Result", use_column_width=True)
        st.write("Predictions:", predictions)