# ⛰️ Ethiopia Soil Erosion Intelligence — ErosionNet

> **A Deep Learning Framework for RUSLE-Based Soil Erosion Risk Classification in the Ethiopian Highlands**

---

## 📌 Overview

This project delivers an end-to-end machine learning pipeline for **automated soil erosion susceptibility mapping** across five Woredas (administrative districts) of the Ethiopian Highlands. It combines the well-established **Revised Universal Soil Loss Equation (RUSLE)** with a custom **Deep Neural Network (ErosionNet)** trained on 75,000 geospatial data points, and integrates **Google Earth Engine (GEE)** for real-time environmental covariate retrieval and interactive visualization.

The pipeline goes from raw environmental data all the way through preprocessing, feature engineering, model training, evaluation, and deployment as an interactive Jupyter dashboard — enabling policy-relevant erosion risk predictions at the point-click level.

---

## 🌍 Study Area

| Attribute | Details |
|---|---|
| **Country** | Ethiopia |
| **Region** | North Shewa Zone, Amhara |
| **Study Woredas** | Ankober, Kewet, Merabete, Menjar, Menze Gera |
| **Coordinate Range** | Lat: 8.72°N – 10.55°N, Lon: 38.66°E – 40.10°E |
| **Elevation Range** | 973 m – 3,652 m a.s.l. |

---

## 📂 File Structure

```
soilErosion/
│
├── SoilErosion Advanced Final.ipynb   # Main analysis notebook
├── Merged_Woredas_75k.xlsx            # Raw dataset (75,000 samples)
├── Clean_Erosion_Dataset.csv          # Preprocessed & RUSLE-labeled dataset
├── erosionnet_model.pth               # Saved model weights (PyTorch)
├── feature_scaler.pkl                 # StandardScaler artifact (joblib)
├── feature_columns.pkl                # Ordered feature list artifact
└── README.md                          # This file
```

---

## 📊 Dataset

### Source & Size
- **75,000 geospatial samples** extracted from five Ethiopian Woredas
- Collected from GIS layers and remote sensing products

### Raw Features (19 columns)

| Category | Features |
|---|---|
| **Spatial** | Latitude, Longitude |
| **Terrain** | Elevation (m), Slope (°), Aspect (°), Plan Curvature, Profile Curvature, TPI, TRI, TWI |
| **Hydrology** | Drainage Density (m), SPI |
| **Vegetation** | NDVI Value |
| **Climate** | Rainfall (mm) |
| **Geology/Land** | Ferrous Materials, Geology Formation, Land Use, Soil Type |
| **Administrative** | Woreda |

### Class Distribution (RUSLE-Based Labels)

| Class | Threshold (t/ha/yr) | Count |
|---|---|---|
| **Low** | 0 – 30 | 16,518 |
| **Medium** | 30 – 100 | 13,376 |
| **High** | > 100 | 45,106 |

---

## 🔬 RUSLE Factor Computation

The target variable is derived from the **Revised Universal Soil Loss Equation**:

```
A = R × K × LS × C × P
```

| Factor | Description | Method |
|---|---|---|
| **R** | Rainfall Erosivity | Hurni (1985): R = −8.12 + 0.562 × Rainfall |
| **K** | Soil Erodibility | Soil-type lookup table (K = 0.15–0.30) |
| **LS** | Slope Length & Steepness | Wischmeier & Smith (1978); McCool et al. (1987) |
| **C** | Cover Management | Van der Knijff (2000) NDVI method: C = exp(−2 × NDVI/(1−NDVI)) |
| **P** | Support Practice | Land use & slope-based lookup (P = 0.10–1.0) |

---

## ⚙️ Data Preprocessing Pipeline

1. **Missing Value Imputation**
   - Terrain features (Elevation, Slope, TPI) → **KNN Imputer** (k=5)
   - Drainage Density → **Group-median** imputation by Woreda
   - Soil Type → **Spatial KNN classification** using lat/lon
   - Geology → Filled as `"Unknown"`

2. **Feature Engineering**
   - Aspect (circular) → `Aspect_sin`, `Aspect_cos` (avoids wrap-around discontinuity)
   - SPI dropped due to 22% missingness

3. **Outlier Capping**
   - Physics-constrained clipping (e.g., Slope ≤ 90°, TWI ∈ [−5, 20])
   - Per-split 5th–95th percentile clipping after train/test split

4. **Categorical Encoding**
   - `OneHotEncoder` with `drop='first'` for `Ferrous Materials` & `Land Use`

5. **Spatial Holdout Generalization Strategy**
   - **Test set**: entire Kewet Woreda (held out by geography, not random split)
   - **Train/Val**: remaining 4 Woredas, 80/20 stratified random split

6. **Feature Normalization**
   - `StandardScaler` fit on training set only; applied to val and test sets

---

## 🧠 Model Architecture — ErosionNet

A compact **Deep Neural Network** built in PyTorch:

```
Input (N features)
    ↓
Linear(input_dim → 32) → BatchNorm1d → ReLU → Dropout(0.3)
    ↓
Linear(32 → 16) → BatchNorm1d → ReLU → Dropout(0.2)
    ↓
Linear(16 → 3)   ← Output: [Low, Medium, High]
```

### Training Details

| Hyperparameter | Value |
|---|---|
| **Optimizer** | Adam (lr = 5e-4) |
| **Loss Function** | CrossEntropyLoss (class-weighted) |
| **Class Weighting** | Balanced (sklearn `compute_class_weight`) |
| **Batch Size** | Default DataLoader |
| **Max Epochs** | 50 |
| **Early Stopping** | Patience = 5 (monitor Val Accuracy) |
| **Random Seed** | 42 |

Training converged in **23 epochs** with early stopping triggered.

---

## 📈 Results

### Training Progress (Selected Epochs)

| Epoch | Train Loss | Train Acc | Val Acc |
|---|---|---|---|
| 01 | 0.9536 | 56.08% | 68.18% |
| 05 | 0.3416 | 89.34% | 91.38% |
| 10 | 0.2316 | 92.41% | 94.35% |
| 18 | 0.1906 | 93.63% | **96.20%** |
| 23 | — | — | *(early stop)* |

### Classification Report (Kewet Woreda — Unseen Geography)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **High** | 1.00 | 0.95 | 0.97 | 10,353 |
| **Low** | 0.95 | 0.96 | 0.95 | 2,073 |
| **Medium** | 0.79 | 0.94 | 0.86 | 2,301 |
| **Accuracy** | | | **0.95** | 14,727 |
| **Macro Avg** | 0.91 | 0.95 | 0.93 | 14,727 |
| **Weighted Avg** | 0.96 | 0.95 | 0.95 | 14,727 |

> ✅ **95% accuracy on a spatially held-out Woreda** demonstrates strong geographic generalization.

---

## 🗺️ Interactive Dashboard

The final notebook cell deploys a fully interactive **Erosion Intelligence Dashboard** using:

- [`ipywidgets`](https://ipywidgets.readthedocs.io/) — UI controls (dropdowns, sliders, buttons)
- [`geemap`](https://geemap.org/) — Interactive satellite map with boundary overlays
- [`Google Earth Engine`](https://earthengine.google.com/) — Real-time covariate retrieval (DEM, NDVI, CHIRPS rainfall, ESA WorldCover)
- `Plotly` — 3D decision-space visualization

### Dashboard Features

| Feature | Description |
|---|---|
| **Woreda Selector** | Dropdown auto-centers map and fills lat/lon |
| **Map Click** | Click anywhere on the map to set analysis coordinates |
| **GEE Covariate Pull** | Fetches Slope, NDVI, Rainfall, TPI, Land Use via GEE |
| **AI Susceptibility** | ErosionNet prediction with class probabilities |
| **RUSLE Baseline** | Physics-based cross-check with estimated soil loss (t/ha/yr) |
| **Uncertainty** | Monte Carlo Dropout (25 passes) → epistemic uncertainty % |
| **Scenario Lab** | Simulate NDVI change, rainfall shift, land-use change |
| **Policy Advice** | Automated intervention recommendations by risk class |
| **3D Visualization** | Multi-covariate decision space plot (Slope × Rainfall × NDVI) |

---

## 🛠️ Installation & Requirements

### Dependencies

```bash
pip install torch torchvision
pip install scikit-learn pandas numpy matplotlib seaborn
pip install statsmodels joblib
pip install earthengine-api geemap
pip install plotly ipywidgets
```

### Google Earth Engine Setup

The interactive dashboard requires a GEE account and authentication:

```python
import ee
ee.Authenticate()   # Run once
ee.Initialize()
```

---

## 🚀 Usage

1. **Clone / download** this repository.
2. **Place the dataset** `Merged_Woredas_75k.xlsx` in the project directory.
3. **Open** `SoilErosion Advanced Final.ipynb` in Jupyter Lab or Jupyter Notebook.
4. **Run all cells** sequentially (Cell → Run All).
5. The final cell launches the interactive dashboard — authenticate with GEE when prompted.

### Saved Artifacts (after first run)
| File | Description |
|---|---|
| `Clean_Erosion_Dataset.csv` | Preprocessed dataset with RUSLE labels |
| `erosionnet_model.pth` | Trained model weights |
| `feature_scaler.pkl` | Fitted StandardScaler |
| `feature_columns.pkl` | Ordered feature column list |

> On subsequent runs, the dashboard loads saved artifacts directly without retraining.

---

## 🔑 Key Design Decisions

### Why Spatial Holdout?
Using an entire Woreda (Kewet) as the test set simulates real-world deployment where the model must generalize to **previously unseen geographic areas** — a much harder and more realistic evaluation than random splitting.

### Why Drop Lat/Lon from Model Input?
Latitude and Longitude are used only for GEE covariate retrieval. They are **excluded from ErosionNet inputs** to ensure predictions are driven by physical erosion processes (slope, rainfall, NDVI, etc.) rather than spatial memorization. This improves transferability.

### Why Monte Carlo Dropout?
Standard neural networks produce point estimates. **MC Dropout** approximates Bayesian inference by running inference multiple times with dropout active, yielding a distribution of predictions and an **epistemic uncertainty score** — critical for policy decisions in high-stakes environments.

---

## 📚 References

- Hurni, H. (1985). *Erosion-Productivity-Conservation Systems in Ethiopia.*
- Wischmeier, W.H. & Smith, D.D. (1978). *Predicting Rainfall Erosion Losses.* USDA Agriculture Handbook No. 537.
- McCool, D.K. et al. (1987). *Revised Slope Steepness Factor for the Universal Soil Loss Equation.* Transactions of the ASAE.
- Van der Knijff, J.M. et al. (2000). *Soil Erosion Risk Assessment in Italy.* European Soil Bureau.
- Renard, K.G. et al. (1997). *Predicting Soil Erosion by Water: RUSLE.* USDA Agriculture Handbook No. 703.

---

## 👤 Author

**Bereket** — Data Science Student, Debre Berhan University

---

## 📄 License

This project is intended for academic and research purposes.

---

*Last updated: April 2026*
