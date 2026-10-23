# Research Notebooks & Exploratory Data Analysis (`notebooks/`)

This directory contains Jupyter notebooks for exploratory data analysis (EDA), feature embedding visualizations, and temporal anomaly score curve overlays.

---

## 📂 Notebooks Overview

| Notebook | Focus & Objective |
| :--- | :--- |
| **`01_ucf_crime_eda.ipynb`** | Class distribution visualization across all 13 anomaly categories, video duration histograms, and raw video frame sample inspection. |
| **`02_feature_visualization.ipynb`** | 2D/3D t-SNE and UMAP manifold projections comparing normal vs. anomalous I3D feature embeddings to inspect feature separation margins. |
| **`03_anomaly_score_curves.ipynb`** | Frame-by-frame temporal anomaly probability score curve plots overlaid on true ground-truth intervals $[start, end]$ for test videos. |

---

## 🚀 Launching Jupyter

```bash
jupyter notebook notebooks/
```

