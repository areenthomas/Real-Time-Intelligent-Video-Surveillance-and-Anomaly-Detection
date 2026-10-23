# Test Suite Documentation (`tests/`)

This directory contains the automated test harness for **Real-Time Intelligent Video Surveillance and Anomaly Detection**. The test architecture enforces 6 testing dimensions to guarantee mathematical correctness, streaming latency adherence, and robustness.

---

## 📂 Test Directory Organization

```text
tests/
├── conftest.py                  # Pytest fixtures: mock RTSP streams, synthetic tensors, GT masks
├── test_data_pipeline.py        # Unit tests: 32-segment partitioners, shape integrity (TC-DATA)
├── test_model_inference.py      # Unit tests: bounds [0, 1], ranking loss, smoothness loss (TC-MOD)
├── test_crime_categories.py     # Functional tests: all 13 UCF-Crime classes + normal (TC-CRIME)
├── test_real_time_streaming.py  # Latency, FPS throughput & buffer overflow resilience (TC-RT)
├── test_stress_robustness.py    # Environmental stress: night noise, camera shake, crowds (TC-STRESS)
├── test_metrics_evaluation.py   # Ground-truth alignment: ROC-AUC, PR-AUC, FAR, tIoU (TC-METRIC)
└── README.md                    # This document
```

---

## 🧪 Testing Dimensions Overview

For the exhaustive, specification-level test matrix with mathematical formulas and video-by-video pass criteria, refer to:
📄 **[`TEST_CASES.md`](../TEST_CASES.md)**

1. **`TC-DATA` (Data Pipeline):** Verifies that arbitrary-length untrimmed videos are cleanly segmented into exactly 32 bags with tensor dimensions `(B, 32, 2048)`.
2. **`TC-MOD` (Model & Loss Stability):** Ensures ranking loss symmetry, smoothness regularization penalties on flickering, and gradient backpropagation integrity.
3. **`TC-CRIME` (13 UCF-Crime Categories):** Validates detection profiles across Road Accidents, Explosions, Fighting, Robbery, Burglary, Shoplifting, Arson, Vandalism, Assault, Shooting, Abuse, Arrest, Stealing, and Normal baseline footage.
4. **`TC-RT` (Streaming & Throughput):** Verifies that the sliding window buffer sustains $\ge 25\text{ FPS}$ with end-to-end latency $\le 120\text{ ms}$.
5. **`TC-STRESS` (Environmental Perturbations):** Evaluates sensor noise, illumination flash immunity, camera wind vibration, and crowded plaza false alarm rates ($\text{FAR} \le 2.0\%$).
6. **`TC-METRIC` (Ground-Truth Alignment):** Validates trapezoidal ROC-AUC and False Alarm Rate calculations against synthetic edge-case masks.

---

## 🚀 Running the Tests

```bash
# Execute entire test suite
pytest tests/ -v

# Run specific category test module
pytest tests/test_crime_categories.py -v

# Run real-time latency & streaming tests
pytest tests/test_real_time_streaming.py -v

# Generate HTML coverage report
pytest --cov=models --cov=data --cov=streaming tests/ --cov-report=html
```

