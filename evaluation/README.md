# Evaluation Metrics & Benchmarking Protocols (`evaluation/`)

This directory houses the quantitative evaluation harness for measuring Video Anomaly Detection performance on the **UCF-Crime** test set.

---

## 📂 Evaluation Directory Layout

```text
evaluation/
├── roc_auc_calculator.py        # Frame-level Receiver Operating Characteristic & AUC
├── pr_auc_calculator.py         # Precision-Recall AUC (critical for imbalanced anomaly frames)
├── false_alarm_auditor.py       # False Alarm Rate (FAR) calculation on normal test videos
├── localization_evaluator.py    # Temporal Intersection-over-Union (tIoU) for incident boundaries
└── README.md                    # This document
```

---

## 📐 Mathematical Metric Definitions

### 1. Frame-Level Area Under the ROC Curve (Frame ROC-AUC)
The primary standard benchmark metric in Video Anomaly Detection literature (Sultani 2018, RTFM 2021, VadCLIP 2024).

Let $\hat{y}_t \in [0, 1]$ be the model prediction at frame $t$, and $y_t \in \{0, 1\}$ be the binary ground truth. The True Positive Rate (TPR) and False Positive Rate (FPR) are evaluated across varying discrimination thresholds $\tau \in [0, 1]$:

$$\text{TPR}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}, \quad \text{FPR}(\tau) = \frac{\text{FP}(\tau)}{\text{FP}(\tau) + \text{TN}(\tau)}$$

$$\text{ROC-AUC} = \int_{0}^1 \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$

---

### 2. False Alarm Rate on Normal Videos ($\text{FAR}_{\tau}$)
In real-world surveillance operations, high ROC-AUC can mask an unacceptable number of false alerts. $\text{FAR}_{\tau}$ measures the percentage of normal frames incorrectly classified as anomalous at operating threshold $\tau$:

$$\text{FAR}_{\tau} = \frac{\sum_{t \in \mathcal{V}_{\text{normal}}} \mathbb{I}(\hat{y}_t > \tau)}{|\mathcal{V}_{\text{normal}}|}$$

*(Standard acceptable commercial threshold: $\text{FAR}_{0.50} \le 2.0\%$).*

---

### 3. Precision-Recall AUC (PR-AUC)
Because anomalies represent less than 2% of total surveillance footage, PR-AUC evaluates the true precision-recall trade-off without being skewed by large True Negative counts:

$$\text{Precision}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FP}(\tau)}, \quad \text{Recall}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}$$

---

### 4. Temporal Localization Intersection-over-Union (tIoU)
Evaluates how accurately the predicted anomalous time window matches the ground-truth interval $[s_{\text{gt}}, e_{\text{gt}}]$:

$$\text{tIoU} = \frac{|[s_{\text{pred}}, e_{\text{pred}}] \cap [s_{\text{gt}}, e_{\text{gt}}]}{|[s_{\text{pred}}, e_{\text{pred}}] \cup [s_{\text{gt}}, e_{\text{gt}}]}$$

---

## 🚀 Running Evaluation on UCF-Crime Test Set

```bash
# Evaluate model checkpoint on UCF-Crime 290 test videos
python evaluation/roc_auc_calculator.py \
    --predictions_file logs/test_predictions.npy \
    --annotations_file data/splits/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
    --plot_curve \
    --output_dir logs/roc_plots/
```
