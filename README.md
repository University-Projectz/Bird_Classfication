# 🪶 Bird Species Classification — Computer Vision Pipeline

A Classical Computer Vision and Machine Learning pipeline for fine-grained bird species classification. The project demonstrates an end-to-end workflow consisting of object detection, foreground segmentation, handcrafted feature engineering (HOG & LBP), classical machine learning benchmarking (SVM & Random Forest), and an interactive Streamlit web application.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Data Overview](#-data-overview)
- [Directory Architecture](#-directory-architecture)
- [Pipeline Workflow](#-pipeline-workflow)
  - [Notebook 01: Preprocessing & Visual Inspection](#1-notebook-01_preprocessingipynb)
  - [Notebook 02: Feature Extraction (HOG & LBP)](#2-notebook-02_featureextractionipynb)
  - [Notebook 03: Training & Benchmarking](#3-notebook-03_trainandevaluateipynb)
  - [Interactive Explorer: Streamlit App](#4-interactive-web-app-apppy)
- [Dataset Summary](#-dataset-summary)
- [Installation & Quickstart](#-installation--quickstart)
- [Results & Evaluation](#-results--evaluation)
- [Authors & Supervision](#-authors--supervision)

---

## 🔭 Project Overview

Classifying bird species in natural scenes presents significant computer vision challenges: complex backgrounds (foliage, sky, water), varying lighting conditions, and subtle inter-class visual differences.

This project tackles these challenges through a modular pipeline:
1. **Object Localization:** Uses **YOLOv8n** to detect and tightly crop the bird, discarding distracting background context.
2. **Foreground Isolation:** Uses **Gaussian Filtering & Otsu's Thresholding** to create a binary mask that suppresses residual background noise.
3. **Shape & Texture Descriptors:** 
   - **HOG (Histogram of Oriented Gradients):** Captures bird silhouette, beak curvature, and wing outlines.
   - **LBP (Local Binary Patterns):** Captures micro-patterns and texture distributions of feathers.
4. **Machine Learning Classifiers:** Benchmarks **Support Vector Machines (SVM)** and **Random Forest** across both descriptor types.

---

## 📊 Data Overview

The dataset used in this project is the **[Bird Species Dataset](https://www.kaggle.com/datasets/rahmasleam/bird-speciees-dataset/data)** published on Kaggle by *Rahma Sleam*.

### 🔍 Key Dataset Highlights:
* **Source:** [Kaggle — Bird Species Dataset](https://www.kaggle.com/datasets/rahmasleam/bird-speciees-dataset/data)
* **Total Samples:** **811 high-resolution images** across **6 distinct avian classes**.
* **Diversity & Visual Challenges:**
  - Real-world natural settings featuring complex and cluttered backgrounds (dense forests, snow, open water, perches).
  - High intra-class variance due to different bird postures (flying, swimming, perched, wings spread).
  - Variable lighting, shadow occlusions, and scales across photographs.
* **Classes Included:**
  1. `AMERICAN GOLDFINCH` (130 images)
  2. `BARN OWL` (143 images)
  3. `CARMINE BEE-EATER` (131 images)
  4. `DOWNY WOODPECKER` (135 images)
  5. `EMPEROR PENGUIN` (140 images)
  6. `FLAMINGO` (132 images)

This rich variety provides an ideal benchmark for evaluating how effectively automated localization (YOLO) and foreground segmentation (Otsu) isolate the subjects before extracting discriminating shape (HOG) and texture (LBP) features.

---

## 📁 Directory Architecture

```text
Bird_Classfication/
├── Dataset/
│   ├── Raw/                  # Original bird images across 6 species
│   ├── Birds_crop/           # Stage 1 output: YOLO-detected bounding box crops
│   ├── Birds_segmented/      # Stage 2 output: Background removed via Otsu thresholding
│   └── Birds_preprocessed/   # Stage 3 output: Normalized 128x128 grayscale images
├── notebooks/
│   ├── 01_preprocessing.ipynb     # Step-by-step preprocessing + visual demos (5 samples)
│   ├── 02_featureExtraction.ipynb # HOG & LBP feature extraction -> features.npz
│   ├── 03_TrainAndEvaluate.ipynb  # Train SVM & RF models, confusion matrices -> models/
│   └── yolov8n.pt                 # Lightweight YOLOv8 detector weights
├── models/                        # Serialized models (.pkl) and standard scalers
│   ├── svm_hog.pkl
│   ├── rf_hog.pkl
│   ├── svm_lbp.pkl
│   ├── rf_lbp.pkl
│   ├── scaler_hog.pkl
│   └── scaler_lbp.pkl
├── app.py                         # Interactive Streamlit Web Explorer
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

---

## 🔄 Pipeline Workflow

### 1. `notebooks/01_preprocessing.ipynb`
Builds and visually validates the multi-stage image cleaning pipeline:
* **Stage 1 (Detection & Crop):** Passes raw images through YOLOv8n to extract bounding box coordinates $(x_1, y_1, x_2, y_2)$ with maximum confidence. Displays a 5-image visual comparison before saving to `Dataset/Birds_crop/`.
* **Stage 2 (Segmentation):** Converts crops to grayscale, applies a $5 \times 5$ Gaussian kernel, computes Otsu's optimal threshold, and isolates bird pixels using bitwise masking. Displays 5 visual comparisons before saving to `Dataset/Birds_segmented/`.
* **Stage 3 (Normalization):** Resizes all segmented outputs to a standard $128 \times 128$ resolution and saves them to `Dataset/Birds_preprocessed/`.

### 2. `notebooks/02_featureExtraction.ipynb`
Extracts handcrafted descriptors from `Dataset/Birds_preprocessed/`:
* **HOG Descriptor:**
  - `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, `transform_sqrt=True`, `block_norm='L2-Hys'`.
  - Generates structural shape vectors invariant to illumination changes.
* **LBP Descriptor:**
  - Circular sampling with `radius=3` and `n_points=24` using uniform pattern mapping (`method='uniform'`).
  - Evaluated over a $4 \times 4$ spatial grid to retain local texture positioning ($16 \times 26 = 416$ features).
* **Export:** Saves compact archive `features.npz` containing `X_hog`, `X_lbp`, `y`, and `label_map`.

### 3. `notebooks/03_TrainAndEvaluate.ipynb`
Loads `features.npz` and executes the training pipeline:
* **Stratified 80/20 Split:** Maintains species distribution balance between train and test sets.
* **Feature Normalization:** Fits `StandardScaler` on training sets and transforms test sets.
* **Model Benchmarking:** Evaluates 4 distinct combinations:
  1. `SVM (HOG)` — RBF kernel, $C=10.0$
  2. `Random Forest (HOG)` — 100 estimators
  3. `SVM (LBP)` — RBF kernel, $C=10.0$
  4. `Random Forest (LBP)` — 100 estimators
* **Performance Metrics:** Generates classification reports (Precision, Recall, F1-Score) and plots seaborn confusion matrices.
* **Model Serialization:** Exports all fitted estimators and scalers to `models/` via `joblib`.

### 4. Interactive Web App (`app.py`)
A user-friendly Streamlit web dashboard allowing users to:
* Upload any arbitrary bird image.
* View real-time YOLO bounding box detection overlaid in green.
* Select which model (SVM vs. Random Forest) and feature type (HOG vs. LBP) to test.
* View top species probability distribution bars and natural history details.

---

## 🦅 Dataset Summary

The dataset comprises **811 images** distributed across 6 distinct species:

| Species Name | Scientific / Common Context |
|--------------|-----------------------------|
| **American Goldfinch** | Small songbird with distinctive yellow and black plumage |
| **Barn Owl** | Pale nocturnal raptor with characteristic heart-shaped face |
| **Carmine Bee-Eater** | Striking sub-Saharan bird with vivid red and carmine coloration |
| **Downy Woodpecker** | Small black-and-white woodpecker with localized red highlights |
| **Emperor Penguin** | Largest living penguin species adapted to cold environments |
| **Flamingo** | Long-legged wading bird with distinct pink pigmentation |

---

## 🚀 Installation & Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/University-Projectz/Bird_Classfication
cd Bird_Classfication
```

### 2. Create a virtual environment & install dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the pipeline
Open and run the notebooks sequentially in Jupyter or VS Code:
1. `notebooks/01_preprocessing.ipynb` — Generates cropped, segmented, and resized images.
2. `notebooks/02_featureExtraction.ipynb` — Extracts HOG & LBP features and creates `features.npz`.
3. `notebooks/03_TrainAndEvaluate.ipynb` — Trains models and saves them to `models/`.

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test predictions interactively!

---

## 📊 Results & Evaluation

| Model Configuration | Feature Type | Primary Strength |
|---------------------|--------------|------------------|
| **SVM (HOG)** | Shape | Excellent high-dimensional boundary separation on contours and silhouettes. |
| **Random Forest (HOG)** | Shape | Robust non-linear decision thresholds for multi-class classification. |
| **SVM (LBP)** | Texture | High sensitivity to feather patterns (e.g., woodpecker stripes, owl speckles). |
| **Random Forest (LBP)** | Texture | Strong generalization across spatial histogram grids without overfitting. |

---

## 👥 Authors & Supervision

### 🧑‍💻 Developed by:
- **Lele Osama** — [@leleosabry-max](https://github.com/leleosabry-max)
- **Amr Alaa** — [@Amor296](https://github.com/Amor296)
- **Mina Wagih** — [@MinaWagihz](https://github.com/MinaWagihz)

### 👨‍🏫 Supervised by:
- **Eng. Mina Naser** [@Naser215](https://github.com/Naser215)


