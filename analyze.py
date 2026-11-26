"""学習曲線と、検証画像の成功例・失敗例を並べた画像を作る（評価用画像は使わない）。

learning_curve.png: エポックごとの損失と検証 F1 / AP
analysis.png: 次の例を確率の順に並べる
    TP: 検出できた陽性 / FN: 見逃した陽性 / FP: 誤検出した陰性 / TN near threshold: 閾値に近かった陰性

    python analyze.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/model"))
    parser.add_argument("--per-row", type=int, default=8)
    args = parser.parse_args()

    with (args.model_dir / "history.csv").open(encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    epochs = [int(h["epoch"]) for h in history]
    fig, (loss_axis, score_axis) = plt.subplots(1, 2, figsize=(10, 3.5))
    for key in ["train_loss", "valid_loss"]:
        loss_axis.plot(epochs, [float(h[key]) for h in history], marker=".", label=key)
    for key in ["valid_f1", "valid_ap"]:
        score_axis.plot(epochs, [float(h[key]) for h in history], marker=".", label=key)
    loss_axis.set_title("loss")
    score_axis.set_title("validation F1 / AP")
    score_axis.set_ylim(0, 1)
    for axis in (loss_axis, score_axis):
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(args.model_dir / "learning_curve.png", dpi=90)

    threshold = json.loads((args.model_dir / "threshold.json").read_text())["threshold"]
    with (args.model_dir / "valid_predictions.csv").open(encoding="utf-8") as handle:
        rows = [(r["path"], int(r["label"]), float(r["probability"])) for r in csv.DictReader(handle)]

    def pick(label, detected, highest_first):
        members = [r for r in rows if r[1] == label and (r[2] >= threshold) == detected]
        return sorted(members, key=lambda r: r[2], reverse=highest_first)

    groups = [
        ("TP: detected", pick(1, True, True)),
        ("FN: missed", pick(1, False, False)),
        ("FP: false alarm", pick(0, True, True)),
        ("TN near threshold", pick(0, False, True)),
    ]
    fig, axes = plt.subplots(len(groups), args.per_row, figsize=(2.1 * args.per_row, 2.5 * len(groups)))
    for axes_row, (name, members) in zip(axes, groups):
        for axis in axes_row:
            axis.axis("off")
        for axis, (path, _, probability) in zip(axes_row, members):
            with Image.open(args.data_dir / "SDNET2018" / path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"p={probability:.3f}\n{Path(path).name}", fontsize=8)
        axes_row[0].text(-0.08, 0.5, f"{name}\n{len(members)} images", transform=axes_row[0].transAxes,
                         rotation=90, ha="right", va="center", fontsize=11, fontweight="bold")
    fig.suptitle(f"validation examples (threshold={threshold:.4f})")
    fig.tight_layout()
    fig.savefig(args.model_dir / "analysis.png", dpi=90)
    print(f"-> {args.model_dir / 'learning_curve.png'}, {args.model_dir / 'analysis.png'}")


if __name__ == "__main__":
    main()
