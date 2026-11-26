"""学習済みモデルで評価用画像を予測し、提出用CSV（ヘッダなし、<相対パス>,<0|1>）を作る。

1. 各画像を 90度単位の回転 × 反転 の8通りで予測して平均する（TTA）
2. ハフ変換で暗い円（Pコン跡の穴、暗さ80以上）が見つかった画像は、確率に +0.2 する
3. train.py が決めた閾値以上を1にする

    python predict.py
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from train import WallDataset, build_model

HOUGH_BONUS = 0.2       # 暗い円が見つかったときに確率へ足す値
DARKNESS_THRESHOLD = 80  # 円の外側と内側の明るさの差がこれ以上なら「暗い円（穴）」とみなす
PAD = 180               # 中心が画像の外にある円（端で切れた弧）も見つけるための余白


def hole_darkness(path):
    """ハフ変換で円を探し、「円の外側の輪の明るさ − 円の内側の明るさ」の最大値を返す。円がなければ0。

    Pコン跡の穴は内側が影で暗くなるため、この値が大きくなる。
    """
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    padded = cv2.copyMakeBorder(blurred, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=int(blurred.mean()))
    circles = cv2.HoughCircles(padded, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
                               param1=30, param2=30, minRadius=40, maxRadius=180)
    if circles is None:
        return 0.0
    yy, xx = np.mgrid[0:256, 0:256]
    best = 0.0
    for x, y, r in circles[0][:20]:
        distance = np.hypot(xx - (x - PAD), yy - (y - PAD))
        inside = distance < 0.9 * r
        ring = (distance > 1.1 * r) & (distance < 1.4 * r)
        # 円の内側と外側が画像内に十分写っているものだけ使う
        if inside.sum() >= 400 and ring.sum() >= 400:
            best = max(best, float(gray[ring].mean() - gray[inside].mean()))
    return best


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
    parser.add_argument("--hough-bonus", type=float, default=HOUGH_BONUS, help="暗い円があるときの加点（0で加点なし）")
    parser.add_argument("--darkness", type=float, default=DARKNESS_THRESHOLD, help="暗い円とみなす暗さの基準")
    parser.add_argument("--hough-cache", type=Path, default=Path("outputs/hough_darkness.csv"),
                        help="ハフ変換の結果（画像ごとの暗さ）の保存先。計算済みの画像は再計算しない")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info = json.loads((args.model_dir / "threshold.json").read_text())
    threshold, edges = info["threshold"], info.get("edges", False)
    model = build_model(edges, info.get("arch", "resnet18"))
    model.load_state_dict(torch.load(args.model_dir / "model.pt", map_location=device, weights_only=True))
    model.to(device).eval()

    test_paths = (args.data_dir / "test.txt").read_text(encoding="utf-8").split()
    dataset = WallDataset([(path, 0) for path in test_paths], args.data_dir / "SDNET2018", edges=edges)
    loader = DataLoader(dataset, batch_size=128)
    probabilities = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                views = [model(view).squeeze(1).float().sigmoid() for view in eight_views(images)]
            probabilities.append(torch.stack(views).mean(dim=0).cpu().numpy())
    probabilities = np.concatenate(probabilities)

    # 加点で判定が変わりうるのは「閾値 - 加点 <= 確率 < 閾値」の画像だけなので、それだけにハフ変換をかける。
    candidates = np.flatnonzero((probabilities >= threshold - args.hough_bonus) & (probabilities < threshold))
    # 暗さは画像とハフ変換の設定だけで決まる（モデルによらない）ので、計算結果をファイルに保存して使い回す。
    cache = {}
    if args.hough_cache.is_file():
        with args.hough_cache.open(encoding="utf-8") as handle:
            cache = {path: float(value) for path, value in csv.reader(handle)}
    darkness = np.zeros(len(test_paths))
    for i in candidates:
        if test_paths[i] not in cache:
            cache[test_paths[i]] = hole_darkness(args.data_dir / "SDNET2018" / test_paths[i])
        darkness[i] = cache[test_paths[i]]
    args.hough_cache.parent.mkdir(parents=True, exist_ok=True)
    with args.hough_cache.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(sorted(cache.items()))
    probabilities = probabilities + args.hough_bonus * (darkness >= args.darkness)
    predictions = (probabilities >= threshold).astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(zip(test_paths, predictions))
    print(f"threshold={threshold:.4f}, hough checked={len(candidates)}, "
          f"boosted={(darkness >= args.darkness).sum()}, positive={predictions.sum()}/{len(predictions)} -> {args.output}")


if __name__ == "__main__":
    main()
