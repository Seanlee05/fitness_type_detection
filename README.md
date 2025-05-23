# Fitness Type Detection (YOLOv8 + ViT)

A Streamlit app that uses YOLOv8 for object detection and ViT for classification on fitness-related images.

## Setup

```bash
git clone https://github.com/yourusername/fitness_type_detection.git
cd fitness_type_detection
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Deploy on Render

1. Go to https://render.com
2. Click "New + > Web Service"
3. Connect your GitHub repo
4. Set the build and start commands:

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
streamlit run app.py --server.port $PORT
```

Enjoy your public Streamlit app!
