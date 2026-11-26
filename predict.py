"""学習済みモデルで評価用画像を予測し、提出用CSV（ヘッダなし、<相対パス>,<0|1>）を作る。

各画像を 90度単位の回転 × 反転 の8通りで予測して平均し（TTA）、train.py が決めた閾値以上を1にする。

    python predict.py
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import WallDataset, build_model


def eight_views(images):
    """回転4通り × 反転の有無2通り = 8通りの画像を作る。"""
    views = []
    for k in range(4):
        rotated = torch.rot90(images, k, dims=(2, 3))
        views += [rotated, rotated.flip(3)]
    return views


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/model"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model()
    model.load_state_dict(torch.load(args.model_dir / "model.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    threshold = json.loads((args.model_dir / "threshold.json").read_text())["threshold"]

    test_paths = (args.data_dir / "test.txt").read_text(encoding="utf-8").split()
    loader = DataLoader(WallDataset([(path, 0) for path in test_paths], args.data_dir / "SDNET2018"), batch_size=128)
    probabilities = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                views = [model(view).squeeze(1).float().sigmoid() for view in eight_views(images)]
            probabilities.append(torch.stack(views).mean(dim=0).cpu().numpy())
    predictions = (np.concatenate(probabilities) >= threshold).astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(zip(test_paths, predictions))
    print(f"threshold={threshold:.4f}, positive={predictions.sum()}/{len(predictions)} -> {args.output}")


if __name__ == "__main__":
    main()
