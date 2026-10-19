import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import eval_tf, IMG_SIZE
from model import build_model
from gradcam import get_target_layer

LOW_CONFIDENCE = 0.80
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="Blood Cancer Detection", page_icon="🩸", layout="wide")


@st.cache_resource
def load_model(name):
    ckpt = torch.load(MODEL_DIR / f"{name}_best.pth", map_location="cpu", weights_only=False)
    model = build_model(name, len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"]


def analyse(name, img):
    model, classes = load_model(name)
    tensor = eval_tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), 1)[0].numpy()
    pred = int(probs.argmax())
    with GradCAM(model=model, target_layers=get_target_layer(name, model)) as cam:
        heat = cam(input_tensor=tensor, targets=[ClassifierOutputTarget(pred)])[0]
    rgb = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    overlay = show_cam_on_image(rgb, heat, use_rgb=True)
    return classes, probs, pred, overlay


st.title("🩸 Blood Cancer Detection using Machine Learning")
st.warning(
    "For academic and research use only. This is NOT a medical diagnosis tool. "
    "Results must never replace evaluation by a qualified professional."
)

available = [m for m in ["resnet50", "efficientnet_b0", "simple_cnn"]
             if (MODEL_DIR / f"{m}_best.pth").exists()]
if not available:
    st.error("No trained models found in the models/ folder.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", available)
    st.markdown(
        "**Classes**\n- Benign: no malignant cells\n- Early / Pre / Pro: "
        "stages of acute lymphoblastic leukemia (ALL)"
    )
    st.caption("Trained on the Blood Cells Cancer [ALL] dataset. "
               "Images from other labs or stains may give unreliable results.")

file = st.file_uploader("Upload a microscopic blood cell image", type=["jpg", "jpeg", "png", "bmp", "tif"])

if file is not None:
    img = Image.open(file).convert("RGB")
    with st.spinner("Analysing..."):
        classes, probs, pred, overlay = analyse(model_name, img)
    conf = float(probs[pred])

    c1, c2 = st.columns(2)
    c1.subheader("Uploaded image")
    c1.image(img, use_container_width=True)
    c2.subheader("Grad-CAM: where the model looked")
    c2.image(overlay, use_container_width=True)

    st.subheader(f"Prediction: {classes[pred]}  ({conf:.1%} confidence)")
    if conf < LOW_CONFIDENCE:
        st.warning("Low confidence. The image may be unlike the training data. Treat this result as unreliable.")
    if classes[pred] != "Benign":
        st.info("The model suggests possible malignant cells. This needs expert confirmation.")

    st.bar_chart(pd.DataFrame({"Probability": probs}, index=classes))
    st.caption("Heatmap is a visual aid showing influential regions. It does not prove the model reasons like a clinician.")
else:
    st.info("Upload an image to begin. You can try images from the test split in data/raw.")