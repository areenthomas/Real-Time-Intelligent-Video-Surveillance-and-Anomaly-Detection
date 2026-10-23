# Neural Network Architectures & Objectives (`models/`)

This directory houses the modular neural network components for Video Anomaly Detection, structured into 3D spatial-temporal backbones, anomaly reasoning heads, and specialized loss functions.

---

## 📂 Models Directory Organization

```text
models/
├── backbones/                   # 3D Feature extractors
│   ├── i3d.py                   # Inflated 3D ConvNet (Kinetics-400, 2048-d)
│   ├── x3d.py                   # X3D-M lightweight architecture (>60 FPS, 512-d)
│   ├── c3d.py                   # Classic C3D architecture (4096-d fc6)
│   └── video_swin.py            # Video Swin Transformer (1024-d)
├── heads/                       # Anomaly classification & temporal reasoning
│   ├── mil_ranking_head.py      # Sultani 3-layer FC ranking network
│   ├── rtfm_magnitude_head.py   # Robust Temporal Feature Magnitude (Tian et al. ICCV'21)
│   ├── mist_attention_head.py   # Self-Guided Attention Boosted Encoder (Feng et al. CVPR'21)
│   └── ur_dmu_head.py           # Dual Memory Units with Uncertainty Regulation (AAAI'23)
├── losses/                      # Custom objective functions
│   ├── ranking_loss.py          # MIL hinge ranking loss + smoothness + sparsity
│   └── magnitude_loss.py        # Top-k feature magnitude margin separation loss
└── README.md                    # This document
```

---

## 📐 Mathematical Objective Formulations

### 1. Sultani MIL Ranking Loss with Regularizers (CVPR 2018)
The deep ranking objective separates the maximum anomaly score in an abnormal bag $\mathcal{B}_a$ from a normal bag $\mathcal{B}_n$ by a hinge margin of $1$:

$$\mathcal{L}_{\text{ranking}} = \max\left(0, 1 - \max_{i \in \mathcal{B}_a} f(v_a^i) + \max_{j \in \mathcal{B}_n} f(v_n^j)\right)$$

To ensure continuous event coherence and penalize flickering, two regularizers are added:
- **Temporal Smoothness Constraint:**
  $$\mathcal{L}_{\text{smooth}} = \sum_{i=1}^{m-1} \left(f(v_a^{i+1}) - f(v_a^i)\right)^2$$
- **Temporal Sparsity Constraint:**
  $$\mathcal{L}_{\text{sparse}} = \sum_{i=1}^m f(v_a^i)$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ranking}} + \lambda_1 \mathcal{L}_{\text{smooth}} + \lambda_2 \mathcal{L}_{\text{sparse}} + \lambda_3 \|\mathcal{W}\|_2$$

---

### 2. RTFM Feature Magnitude Margin Loss (ICCV 2021)
Rather than ranking scalar scores, RTFM forces anomalous video segments to generate larger $L_2$ activation norms in deep feature space:

$$\mathcal{L}_{\text{mag}} = \max\left(0, m - \frac{1}{k}\sum_{i=1}^k \|e_a^{(i)}\|_2 + \frac{1}{k}\sum_{j=1}^k \|e_n^{(j)}\|_2\right)$$

Where $e_a^{(i)}$ denotes the top-$k$ feature vectors with largest magnitude in the abnormal bag, and $e_n^{(j)}$ denotes the top-$k$ in the normal bag (default margin $m=100.0, k=3$).

---

## ⚡ Backbone Comparison & Real-Time Viability

| Architecture | Temporal Input | Output Feature Dim | Inference Speed (GPU) | Parameter Count | Edge Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C3D** | 16 frames | 4,096-d | ~15–20 FPS | ~60M | ❌ Deprecated / Slow |
| **I3D** | 16–32 frames | 2,048-d | ~30–35 FPS | ~25M | ✅ Standard Research Choice |
| **X3D-M** | 16 frames | 512-d | **~65–80 FPS** | **~3.8M** | 🚀 **Ideal for Edge CCTV Deployment** |
| **Video Swin** | 32 frames | 1,024-d | ~12–18 FPS | ~88M | ❌ High Compute / Latency |

