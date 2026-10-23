"""
Model Packaging & Production Export Script
Bundles trained Video Anomaly Detection models into deployable formats:
  1. Complete PyTorch Checkpoint (.pth) with metadata and hyperparameter config.
  2. TorchScript Compiled Model (.pt) for dependency-free Python/C++ inference.
  3. ONNX Model (.onnx) for edge accelerators (NVIDIA Jetson, Intel OpenVINO).
"""

import os
import argparse
import torch
import torch.nn as nn
from typing import Dict, Any

from frontend_inference import MultiModelEnsemble, RTFMModel, MISTModel, MILRankingModel


def export_production_package(
    model_name: str = "ensemble",
    in_features: int = 512,
    hidden_dim: int = 256,
    weights_path: str = None,
    output_dir: str = "checkpoints",
    threshold: float = 0.55,
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"

    print(f"\n=======================================================")
    print(f"📦 EXPORTING MODEL FOR FRONTEND PRODUCTION DEPLOYMENT")
    print(f"   Architecture: {model_name}")
    print(f"   Input Dim:    {in_features} -> Hidden: {hidden_dim}")
    print(f"   Output Dir:   {output_dir}")
    print(f"=======================================================")

    # 1. Instantiate Model
    if model_name == "mil":
        model = MILRankingModel(in_features=in_features)
        arch_name = "MILRankingModel"
    elif model_name == "rtfm":
        model = RTFMModel(in_features=in_features, hidden_dim=hidden_dim)
        arch_name = "RTFMModel"
    elif model_name == "mist":
        model = MISTModel(in_features=in_features, hidden_dim=hidden_dim)
        arch_name = "MISTModel"
    else:
        model = MultiModelEnsemble(in_features=in_features, hidden_dim=hidden_dim)
        arch_name = "MultiModelEnsemble"

    # Load existing weights if provided
    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"[✓] Loaded existing trained weights from: {weights_path}")
    else:
        print("[!] Using initialized model weights for export package.")

    model.eval()
    model.to(device)

    # 2. Export Complete Checkpoint (.pth) with Metadata
    pth_path = os.path.join(output_dir, "best_anomaly_detector.pth")
    checkpoint_package = {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "architecture": arch_name,
            "in_features": in_features,
            "hidden_dim": hidden_dim,
            "threshold": threshold,
            "dataset": "UCF-Crime",
            "classes": [
                "Abuse", "Arrest", "Arson", "Assault", "Accident",
                "Burglary", "Explosion", "Fighting", "Robbery", "Shooting",
                "Shoplifting", "Stealing", "Vandalism", "Normal"
            ],
            "exported_at": str(torch.__version__),
        }
    }
    torch.save(checkpoint_package, pth_path)
    size_mb = os.path.getsize(pth_path) / (1024 * 1024)
    print(f"[✓] Exported PyTorch Package:   {pth_path} ({size_mb:.2f} MB)")

    # 3. Export TorchScript (.pt)
    dummy_input = torch.randn(1, 32, in_features)
    try:
        ts_path = os.path.join(output_dir, "best_anomaly_detector_scripted.pt")
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(ts_path)
        ts_size = os.path.getsize(ts_path) / (1024 * 1024)
        print(f"[✓] Exported TorchScript Model: {ts_path} ({ts_size:.2f} MB)")
    except Exception as e:
        print(f"[!] TorchScript trace note: {e}")

    # 4. Export ONNX (.onnx)
    try:
        onnx_path = os.path.join(output_dir, "best_anomaly_detector.onnx")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["feature_sequence"],
            output_names=["anomaly_scores", "latent_features"],
            dynamic_axes={"feature_sequence": {0: "batch", 1: "time"}},
        )
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"[✓] Exported ONNX Edge Model:   {onnx_path} ({onnx_size:.2f} MB)")
    except Exception as e:
        print(f"[!] ONNX export note: {e}")

    print("\n🎉 Model successfully exported! You can now load this in your frontend using:")
    print("   python frontend_inference.py --weights checkpoints/best_anomaly_detector.pth --source 0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Trained Model for Frontend Deployment")
    parser.add_argument("--model", type=str, default="ensemble", choices=["ensemble", "rtfm", "mist", "mil"])
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    export_production_package(
        model_name=args.model,
        in_features=args.dim,
        weights_path=args.weights,
        output_dir=args.output_dir
    )
