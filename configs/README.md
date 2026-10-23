# Configuration Files (`configs/`)

This directory houses centralized, version-controlled YAML configuration files for datasets, neural network models, and real-time streaming engines.

---

## 📂 Configuration Files Overview

| Configuration File | Scope & Purpose |
| :--- | :--- |
| **[`dataset_ucf_crime.yaml`](dataset_ucf_crime.yaml)** | Paths to UCF-Crime raw MP4s and precomputed features, 14 class names, train/test split definitions, and 32-segment bag parameters. |
| **[`model_rtfm_i3d.yaml`](model_rtfm_i3d.yaml)** | Backbone architecture settings (I3D / X3D), pyramid dilated convolution dilations, loss margin parameters ($m=100.0, k=3$), and training hyperparameters. |
| **[`streaming_pipeline.yaml`](streaming_pipeline.yaml)** | Real-time RTSP camera URLs, sliding window buffer parameters (16 frames, stride 8), inference FPS targets, and alert dispatcher thresholds. |

---

## 🛠️ Usage Example

```python
import yaml

with open("configs/model_rtfm_i3d.yaml", "r") as f:
    config = yaml.safe_load(f)

print(f"Loaded Backbone: {config['model']['backbone']}")
print(f"Loss Margin: {config['loss']['margin']}")
```

