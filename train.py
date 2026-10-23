import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import (
    MILRankingHead,
    RTFMMagnitudeHead,
    MISTAttentionHead,
    MultiModelEnsembleHead,
    MILRankingLoss,
    FeatureMagnitudeLoss,
)
from data.ucf_crime_loader import get_ucf_crime_loaders


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model Training on UCF-Crime")
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble",
        choices=["mil", "rtfm", "mist", "ensemble"],
        help="Model architecture to train",
    )
    parser.add_argument("--features_dir", type=str, default="data/features/i3d_ucf_crime/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--mock_data", action="store_true", help="Use simulated data for testing")
    return parser.parse_args()


def build_model(model_name: str, in_features: int = 2048) -> nn.Module:
    if model_name == "mil":
        return MILRankingHead(in_features=in_features)
    elif model_name == "rtfm":
        return RTFMMagnitudeHead(in_features=in_features)
    elif model_name == "mist":
        return MISTAttentionHead(in_features=in_features)
    elif model_name == "ensemble":
        return MultiModelEnsembleHead(in_features=in_features)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"🚀 Training Anomaly Detection Model: {args.model.upper()} on {device}")
    print(f"============================================================")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    dataloader = get_ucf_crime_loaders(
        features_dir=args.features_dir,
        batch_size=args.batch_size,
        mock_mode=args.mock_data or not os.path.exists(args.features_dir),
    )

    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ranking_criterion = MILRankingLoss()
    magnitude_criterion = FeatureMagnitudeLoss(margin=100.0, top_k=3)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{args.epochs:02d}")

        for feat_abnormal, feat_normal in pbar:
            feat_abnormal = feat_abnormal.to(device)
            feat_normal = feat_normal.to(device)

            optimizer.zero_grad()

            if args.model == "mil":
                scores_abn = model(feat_abnormal)
                scores_norm = model(feat_normal)
                loss = ranking_criterion(scores_abn, scores_norm)
            elif args.model == "rtfm":
                scores_abn, feats_abn_proj = model(feat_abnormal)
                scores_norm, feats_norm_proj = model(feat_normal)
                loss = ranking_criterion(scores_abn, scores_norm)
                loss += 0.001 * magnitude_criterion(feats_abn_proj, feats_norm_proj)
            elif args.model == "mist":
                scores_abn, _ = model(feat_abnormal)
                scores_norm, _ = model(feat_normal)
                loss = ranking_criterion(scores_abn, scores_norm)
            elif args.model == "ensemble":
                scores_abn, diag_abn = model(feat_abnormal)
                scores_norm, diag_norm = model(feat_normal)
                loss = ranking_criterion(scores_abn, scores_norm)
                loss += 0.001 * magnitude_criterion(diag_abn["rtfm_features"], diag_norm["rtfm_features"])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        mean_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch:02d} Complete - Mean Loss: {mean_loss:.4f}")

    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model}_model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n[✓] Model successfully trained and saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
