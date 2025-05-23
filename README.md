# Fitness Type Detection (YOLOv8 + ViT)

This is a ready-to-deploy Streamlit app for detecting fitness posture types using a YOLOv8 object detector and a ViT classifier.

## 🛠️ Setup

1. Clone or unzip the repo.
2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

## 📦 Deployment (Render.com)

- Push this repo to GitHub.
- Create a free Web Service on [Render](https://render.com)
- Set:
  - Build command: `pip install -r requirements.txt`
  - Start command: `streamlit run app.py --server.port $PORT`

Enjoy your public app! 🎉