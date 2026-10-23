# Data Pipeline & UCF-Crime Dataset (`data/`)

This directory houses the data ingestion, temporal sampling, feature caching, and split management infrastructure for the **UCF-Crime** benchmark.

---

## 📂 Data Directory Layout

```text
data/
├── dataset_documentation.md     # In-depth dataset audit, class stats & download mirrors
├── ucf_crime_loader.py          # PyTorch Dataset for weakly labeled training bags & test frames
├── extract_features.py          # Offline 3D CNN (I3D / X3D-M) feature extraction pipeline
├── temporal_sampler.py          # 32-segment bag partitioner for Multiple Instance Learning
├── transforms.py                # Spatial resizing (224x224), center crop, ImageNet normalization
├── features/                    # Precomputed feature cache (.npy / .pt)
│   └── i3d_ucf_crime/           # 2048-d feature vectors per 16-frame clip (~15 GB)
├── splits/                      # Official partition text files
│   ├── Anomaly_Train.txt        # 1,610 weakly labeled training videos (810 abnormal + 800 normal)
│   ├── Anomaly_Test.txt         # 290 test videos (140 abnormal + 150 normal)
│   └── Temporal_Anomaly_Annotation_for_Testing_Videos.txt # Frame-level intervals [start, end]
└── README.md                    # This document
```

---

## 📊 UCF-Crime Summary

- **Total Videos:** 1,900 untrimmed surveillance videos (128 hours total).
- **13 Crime Classes:** Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Road Accidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism, and Normal.
- **Training Protocol (Weak Supervision):** 1,610 videos labeled *only* at the video level ($Y \in \{0, 1\}$). No temporal start/end frame timestamps are provided.
- **Testing Protocol (Frame-Level Ground Truth):** 290 videos evaluated with frame-by-frame binary masks to compute Frame-level ROC-AUC.

---

## ⚙️ 32-Segment Bag Partitioning Logic

Under Multiple Instance Learning (MIL), each video of arbitrary length $N$ frames is partitioned into exactly $B=32$ non-overlapping or interpolated segment bags:

$$\text{Segment Length } L_s = \left\lfloor \frac{N}{32} \right\rfloor$$

For any segment $i \in \{1, 2, \dots, 32\}$, the temporal frame range is:
$$\text{Range}_i = \left[(i-1) \cdot L_s, \; i \cdot L_s\right)$$

If a video contains fewer than 32 frames ($N < 32$), the temporal sampler dynamically repeats frames via nearest-neighbor interpolation to guarantee consistent tensor dimensions `(32, feature_dim)`.

---

## 🚀 Precomputed Feature Setup

To avoid the 110 GB raw video storage and decoding overhead, download the official precomputed I3D features (~15 GB) into `data/features/i3d_ucf_crime/`:

```bash
# Example extraction script
python data/extract_features.py --video_dir data/raw/UCF_Crimes/ --output_dir data/features/i3d_ucf_crime/ --backbone I3D
```
