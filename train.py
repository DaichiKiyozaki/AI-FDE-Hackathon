"""Pコン跡の有無を判定する ResNet-18 を学習する（事前学習なし）。

学習用の元写真40枚のうち、folds.csv で検証用（fold 0）に指定した8枚を検証に使い、残り32枚で学習する。
最後のepochのモデルで検証画像を予測し、F1が最大になる閾値を決めて保存する。

    python train.py
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18


class WallDataset(Dataset):
    """(画像の相対パス, ラベル) のリストから、画像テンソルとラベルを返す。"""

    def __init__(self, rows, image_dir, augment=False):
        self.rows = rows
        self.image_dir = Path(image_dir)
        self.augment = augment
        # 画素値を [-1, 1] に変換する（外部データの統計量は使わない）。
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label = self.rows[index]
        with Image.open(self.image_dir / path) as image:
            tensor = self.transform(image.convert("RGB"))
        if self.augment:
            # Pコン跡は円形なので、90度単位の回転・反転をしてもラベルは変わらない。
            # 90度単位なら画像の端が欠けず、端に一部だけ写ったPコン跡も残る。
            tensor = torch.rot90(tensor, int(torch.randint(4, ())), dims=(1, 2))
            if torch.rand(()) < 0.5:
                tensor = tensor.flip(2)
        return tensor, torch.tensor(label, dtype=torch.float32)


def build_model():
    """事前学習なしの ResNet-18。出力は「Pコン跡らしさ」のスコア1つ。"""
    return resnet18(weights=None, num_classes=1)


def seed_everything(seed):
    """乱数と計算方法を固定し、同じ環境なら同じ結果になるようにする。"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def best_threshold(labels, probabilities):
    """「確率 >= 閾値」を陽性としたときに F1 が最大になる (閾値, F1) を返す。"""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    index = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[index]), float(f1[index])


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    """1epoch分を処理する。optimizer を渡すと学習、渡さないと検証（予測確率も返す）。"""
    training = optimizer is not None
    model.train(training)
    total_loss, count, probabilities = 0.0, 0, []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                probabilities.append(logits.float().sigmoid().cpu().numpy())
            total_loss += loss.item() * len(labels)
            count += len(labels)
    return total_loss / count, (None if training else np.concatenate(probabilities))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--folds", type=Path, default=Path("folds.csv"))
    parser.add_argument("--valid-fold", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/model"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = args.data_dir / "SDNET2018"

    # 1. データを読み、元写真単位で学習用と検証用に分ける。
    #    同じ写真のタイルは見た目が似ているため、タイル単位で分けると検証が甘くなる。
    with (args.data_dir / "train_label.csv").open(encoding="utf-8") as handle:
        rows = [(path, int(label)) for path, label in csv.reader(handle)]
    with args.folds.open(encoding="utf-8") as handle:
        fold_of_photo = {row["photo_id"]: int(row["fold"]) for row in csv.DictReader(handle)}
    is_valid = [fold_of_photo[Path(path).stem.split("-")[0]] == args.valid_fold for path, _ in rows]
    train_rows = [row for row, valid in zip(rows, is_valid) if not valid]
    valid_rows = [row for row, valid in zip(rows, is_valid) if valid]
    train_labels = np.array([label for _, label in train_rows])
    valid_labels = np.array([label for _, label in valid_rows])
    print(f"train: {len(train_rows)} images ({train_labels.sum()} positive), "
          f"valid: {len(valid_rows)} images ({valid_labels.sum()} positive)")

    # 2. 陽性は約2%しかないため、学習時は陽性が約1割になるよう多めに抽選する。
    seed_everything(args.seed + args.valid_fold)
    counts = np.bincount(train_labels, minlength=2)
    weights = np.where(train_labels == 1, args.pos_fraction / counts[1], (1 - args.pos_fraction) / counts[0])
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(train_rows), replacement=True,
        generator=torch.Generator().manual_seed(args.seed + args.valid_fold),
    )
    train_loader = DataLoader(
        WallDataset(train_rows, image_dir, augment=True), batch_size=args.batch_size, sampler=sampler,
        generator=torch.Generator().manual_seed(args.seed + args.valid_fold), pin_memory=True,
    )
    valid_loader = DataLoader(WallDataset(valid_rows, image_dir), batch_size=args.batch_size, pin_memory=True)

    # 3. 学習する。学習率は cosine で徐々に下げ、最後のepochのモデルを安定させる。
    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        valid_loss, probabilities = run_epoch(model, valid_loader, device, criterion)
        scheduler.step()
        threshold, f1 = best_threshold(valid_labels, probabilities)
        ap = average_precision_score(valid_labels, probabilities)
        print(f"epoch {epoch:2d}: train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
              f"valid_F1={f1:.4f} valid_AP={ap:.4f}")

    # 4. 最後のepochのモデルと、検証画像でF1が最大になる閾値を保存する。
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output / "model.pt")
    (args.output / "threshold.json").write_text(
        json.dumps({"threshold": threshold, "valid_f1": f1, "valid_ap": float(ap)}, indent=2)
    )
    with (args.output / "valid_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "probability"])
        writer.writerows((path, label, float(p)) for (path, label), p in zip(valid_rows, probabilities))
    print(f"threshold={threshold:.4f}, valid F1={f1:.4f}, valid AP={ap:.4f} -> {args.output}")


if __name__ == "__main__":
    main()
