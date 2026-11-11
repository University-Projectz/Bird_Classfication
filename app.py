from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DATASET_DIR = ROOT / "Dataset" / "Raw"
IMAGE_SIZE = (128, 128)
MODEL_FILES = {
    "SVM · HOG": ("svm_hog.pkl", "scaler_hog.pkl", "hog"),
    "Random Forest · HOG": ("rf_hog.pkl", "scaler_hog.pkl", "hog"),
    "SVM · LBP": ("svm_lbp.pkl", "scaler_lbp.pkl", "lbp"),
    "Random Forest · LBP": ("rf_lbp.pkl", "scaler_lbp.pkl", "lbp"),
}

SPECIES_INFO = {
    "AMERICAN GOLDFINCH": "A small North American songbird. Males are bright yellow with black wings and a black cap; it feeds mainly on seeds.",
    "BARN OWL": "A pale, nocturnal owl recognized by its heart-shaped face. It hunts small mammals and occurs across much of the world.",
    "CARMINE BEE-EATER": "A vivid red bird from sub-Saharan Africa. It catches bees and other flying insects, often from an exposed perch.",
    "DOWNY WOODPECKER": "A small black-and-white North American woodpecker. Males have a small red patch on the back of the head.",
    "EMPEROR PENGUIN": "The largest living penguin, native to Antarctica. It is adapted to sea ice and dives for fish, squid, and krill.",
    "FLAMINGO": "A long-legged wading bird whose pink color comes from pigments in its diet. It feeds by filtering small food items from water.",
}

st.set_page_config(page_title="Bird Species Explorer", page_icon="🪶", layout="wide")
st.markdown("""
<style>
    .stApp { background: #f5f7f3; color: #17241e; }
    [data-testid="stAppViewContainer"] .main,
    [data-testid="stAppViewContainer"] .main p,
    [data-testid="stAppViewContainer"] .main label,
    [data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] { color: #17241e; }
    [data-testid="stSidebar"] { background: #172c25; }
    [data-testid="stSidebar"] * { color: #f5f7f3; }
    .hero { padding: 1.8rem 2rem; border-radius: 22px; background: linear-gradient(120deg,#173c30,#39745a); color: white; margin-bottom: 1.2rem; }
    .hero h1 { color: white; margin: 0; font-size: clamp(1.65rem, 4vw, 2.4rem); }
    .hero p { color: #e8f1eb; margin: .45rem 0 0; font-size: 1.05rem; }
    div[data-testid="stMetric"] { background: white; color: #17241e; border: 1px solid #e4e9e2; padding: 1rem; border-radius: 16px; }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #17241e !important; }
    @media (max-width: 640px) {
        .hero { padding: 1.25rem; border-radius: 16px; }
        div[data-testid="stMetric"] { padding: .7rem; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    features_file = ROOT / "notebooks" / "features.npz"
    if not features_file.exists():
        features_file = ROOT / "features.npz"
        
    data = np.load(features_file, allow_pickle=True)
    labels = data["label_map"].item()
    models = {}
    for name, (model_file, scaler_file, feature_type) in MODEL_FILES.items():
        model_path, scaler_path = MODEL_DIR / model_file, MODEL_DIR / scaler_file
        if model_path.exists() and scaler_path.exists():
            models[name] = (joblib.load(model_path), joblib.load(scaler_path), feature_type)
            
    yolo_path = ROOT / "notebooks" / "yolov8n.pt"
    if not yolo_path.exists():
        yolo_path = ROOT / "yolov8n.pt"
        
    detector = YOLO(str(yolo_path))
    return labels, models, detector

def detect_and_preprocess(rgb_image, detector):
    """Detect the bird, return its box and preprocess its crop like the training pipeline."""
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    results = detector.predict(source=bgr_image, verbose=False, conf=0.20)
    box = None
    crop = bgr_image

    if results and results[0].boxes is not None:
        names = results[0].names
        bird_ids = {class_id for class_id, name in names.items() if str(name).lower() == "bird"}
        candidates = [item for item in results[0].boxes if int(item.cls[0]) in bird_ids]
        if candidates:
            best = max(candidates, key=lambda item: float(item.conf[0]))
            x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
            height, width = bgr_image.shape[:2]
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 > x1 and y2 > y1:
                box = (x1, y1, x2, y2, float(best.conf[0]))
                crop = bgr_image[y1:y2, x1:x2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    segmented = cv2.bitwise_and(gray, gray, mask=mask)
    return cv2.resize(segmented, IMAGE_SIZE), box

def extract_features(gray, feature_type):
    resized = cv2.resize(gray, IMAGE_SIZE)
    if feature_type == "hog":
        vector = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2-Hys")
        return vector.reshape(1, -1)
    lbp = local_binary_pattern(resized, 24, 3, method="uniform")
    bins, parts = 26, []
    cell_h, cell_w = resized.shape[0] // 4, resized.shape[1] // 4
    for row in range(4):
        for col in range(4):
            cell = lbp[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w]
            hist, _ = np.histogram(cell.ravel(), bins=bins, range=(0, bins), density=True)
            parts.extend(hist)
    return np.asarray(parts).reshape(1, -1)

st.markdown('<div class="hero"><h1>🪶 Bird Species Explorer</h1><p>Explore the collection and identify a bird from an image.</p></div>', unsafe_allow_html=True)
try:
    labels, models, detector = load_assets()
except Exception as exc:
    st.error(f"Could not load the trained models: {exc}. Please make sure you have trained models saved in 'models/' folder.")
    st.stop()

species = sorted(labels.values())
counts = {}
if DATASET_DIR.exists():
    for name in species:
        folder = DATASET_DIR / name
        counts[name] = sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}) if folder.exists() else 0
else:
    counts = {name: 0 for name in species}

with st.sidebar:
    st.header("Identify a bird")
    model_name = st.selectbox("Model", list(models) or list(MODEL_FILES))
    uploaded = st.file_uploader("Upload a bird image", type=["jpg", "jpeg", "png", "bmp"])
    st.caption("Choose a clear image showing one bird for the best result.")
    st.divider()
    st.caption("Models trained on the project’s HOG and LBP image features.")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Species", len(species))
col_b.metric("Training images", f"{sum(counts.values()):,}")
col_c.metric("Available models", len(models))

left, right = st.columns([1, 1.25], gap="large")
with left:
    st.subheader("Make a prediction")
    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        rgb = np.asarray(pil_image)
        processed, box = detect_and_preprocess(rgb, detector)
        displayed = rgb.copy()
        if box is not None:
            x1, y1, x2, y2, detector_confidence = box
            cv2.rectangle(displayed, (x1, y1), (x2, y2), (30, 210, 90), max(3, min(rgb.shape[:2]) // 180))
        st.image(displayed, use_container_width=True)
        if model_name not in models:
            st.warning("The selected model files are not available in the models folder.")
        else:
            model, scaler, feature_type = models[model_name]
            features = scaler.transform(extract_features(processed, feature_type))
            prediction = int(model.predict(features)[0])
            predicted_species = labels[prediction]
            probabilities = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None
            st.success(f"Predicted species: **{predicted_species}**")
            if box is not None:
                st.caption(f"Green box: YOLO bird detection · {detector_confidence:.0%} detector confidence")
            else:
                st.warning("YOLO did not find a bird, so the whole image was used for classification and no detection box is shown.")
            if probabilities is not None:
                ranked = sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True)[:3]
                st.caption("Top species matches")
                for class_id, probability in ranked:
                    st.write(f"{labels[int(class_id)]} · {probability:.1%}")
                    st.progress(float(probability))
            st.markdown("#### About this bird")
            st.write(SPECIES_INFO.get(predicted_species, "Species information is not available yet."))
            st.caption(f"Species folder contains {counts.get(predicted_species, 0)} training images.")
    else:
        st.info("Upload an image from the sidebar to see a species prediction.")
with right:
    st.subheader("Collection overview")
    st.write("Image counts in each species folder")
    chart_data = {name.title(): counts.get(name, 0) for name in species}
    st.bar_chart(chart_data, horizontal=True, color="#39745a")
    with st.expander("Browse species"):
        for name in species:
            st.write(f"**{name.title()}** — {counts.get(name, 0)} images")
