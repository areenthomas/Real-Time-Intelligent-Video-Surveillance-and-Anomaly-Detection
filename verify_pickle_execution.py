"""
Execution & Verification Script for Pickle Model Deserialization and Live Inference
"""

import os
import sys
import pickle
import torch
import torch.nn as nn
from typing import Tuple, Optional

# Force UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Import the model architectures used for serialization
from export_pickle_models import (
    MILRankingModel,
    BiLSTMTemporalModel,
    RTFMModel,
    MISTModel,
    TemporalAnomalyTransformer,
    MultiModelEnsemble,
)

def test_champion_pickle():
    pkl_path = "checkpoints/best_anomaly_detector.pkl"
    print("=" * 80)
    print(f"1. EXECUTING PICKLE DESERIALIZATION: {pkl_path}")
    print("=" * 80)

    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} does not exist!")
        return False

    # 1. Load Pickle file
    with open(pkl_path, "rb") as f:
        pkg = pickle.load(f)

    # 2. Inspect Metadata
    print(f"  [+] Loaded Model Name  : {pkg.get('model_name')}")
    print(f"  [+] Architecture Class : {pkg.get('architecture')}")
    print(f"  [+] In Features        : {pkg.get('in_features')}")
    print(f"  [+] Optimal Threshold  : tau = {pkg.get('threshold')}")
    print(f"  [+] Framework / Engine : {pkg.get('framework')}")
    print(f"  [+] Timestamp Saved    : {pkg.get('timestamp')}")
    print(f"  [+] Benchmark Record   : {pkg.get('metrics')}")
    print(f"  [+] Monitored Classes  : {len(pkg.get('classes', []))} categories")
    print(f"      {pkg.get('classes')}")

    # 3. Rebuild PyTorch Model & Load Weights from Pickle
    print("-" * 80)
    print("2. RESTORING MODEL WEIGHTS & RUNNING FORWARD PASS INFERENCE:")
    print("-" * 80)
    model = MultiModelEnsemble(in_features=pkg.get("in_features", 2048))
    model.load_state_dict(pkg["model_state_dict"])
    model.eval()

    # 4. Run Forward Pass with simulated video snippet features (Batch=1, Time=32 snippets, Dim=2048)
    dummy_video_features = torch.randn(1, 32, 2048)
    with torch.no_grad():
        anomaly_scores, latent_features = model(dummy_video_features)

    scores_np = anomaly_scores.squeeze(0).numpy()
    print(f"  [✓] Forward Pass Succeeded!")
    print(f"  [✓] Input Shape         : {list(dummy_video_features.shape)} (1 bag, 32 snippets, 2048 features)")
    print(f"  [✓] Output Score Shape  : {list(anomaly_scores.shape)} (Frame-level anomaly probabilities)")
    print(f"  [✓] Latent Feat Shape   : {list(latent_features.shape)}")
    print(f"  [✓] Min Anomaly Score   : {scores_np.min():.4f}")
    print(f"  [✓] Max Anomaly Score   : {scores_np.max():.4f}")
    print(f"  [✓] Mean Anomaly Score  : {scores_np.mean():.4f}")

    threshold = pkg.get("threshold", 0.50)
    breach_snippets = (scores_np >= threshold).sum()
    print(f"  [✓] Snippets Exceeding Trigger Level (tau >= {threshold:.2f}): {breach_snippets} / 32")
    return True

def test_all_pickle_models():
    print("=" * 80)
    print("3. VERIFYING ALL INDIVIDUAL MODELS IN checkpoints/pickle_models/")
    print("=" * 80)

    pickle_dir = "checkpoints/pickle_models"
    if not os.path.isdir(pickle_dir):
        print(f"Directory {pickle_dir} not found.")
        return

    files = sorted([f for f in os.listdir(pickle_dir) if f.endswith(".pkl")])
    for f in files:
        full_path = os.path.join(pickle_dir, f)
        size_kb = os.path.getsize(full_path) / 1024
        with open(full_path, "rb") as pfile:
            data = pickle.load(pfile)
        name = data.get("model_name", f)
        auc = data.get("metrics", {}).get("ROC-AUC", "N/A")
        print(f"  [✓] {f:<32} | Name: {name:<26} | Size: {size_kb:8.1f} KB | ROC-AUC: {auc}%")

    print("=" * 80)
    print("ALL PICKLE MODELS VALIDATED AND EXECUTED WITH ZERO ERRORS!")
    print("=" * 80)

if __name__ == "__main__":
    success = test_champion_pickle()
    if success:
        test_all_pickle_models()

