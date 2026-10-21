"""Pick a recall-first decision threshold on validation, then score the untouched test set."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from common import XrayDataset, build_model, eval_transform, load_split
from train import predict

BLUE, INK, MUTED, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"


def metrics_at(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0, 1]).ravel()
    return {"threshold": float(t), "recall": tp / (tp + fn), "specificity": tn / (tn + fp),
            "precision": tp / (tp + fp), "f1": 2 * tp / (2 * tp + fp + fn), "accuracy": (tp + tn) / len(y),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}


def recall_threshold(y, p, target):
    """Highest threshold whose validation recall is still >= target (fewest false alarms)."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr >= target)])


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(y, p, op, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    for ax, (xs, ys, xl, yl, title, pt) in zip(axes, [
        (fpr, tpr, "False positive rate (1 - specificity)", "Recall (sensitivity)",
         f"Test ROC curve, AUC {roc_auc_score(y, p):.3f}", (1 - op["specificity"], op["recall"])),
        (rec, prec, "Recall (sensitivity)", "Precision",
         f"Test precision-recall curve, AP {average_precision_score(y, p):.3f}", (op["recall"], op["precision"])),
    ]):
        style(ax)
        ax.plot(xs, ys, color=BLUE, linewidth=2)
        ax.plot(*pt, "o", color=BLUE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"chosen threshold {op['threshold']:.2f}", pt, xytext=(12, -18),
                    textcoords="offset points", color=INK, fontsize=9)
        ax.set(xlabel=xl, ylabel=yl, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_confusion(ops, path):
    fig, axes = plt.subplots(1, len(ops), figsize=(4.6 * len(ops), 4), facecolor=SURFACE)
    for ax, (name, m) in zip(axes, ops.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#e8f1fc", "#184f95"]))
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                        color="white" if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], ["Normal", "Pneumonia"])
        ax.set_yticks([0, 1], ["Normal", "Pneumonia"])
        ax.set(xlabel="Predicted", ylabel="Actual")
        ax.set_title(f"{name}\nrecall {m['recall']:.1%}, specificity {m['specificity']:.1%}",
                     color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--target-recall", type=float, default=0.98)
    args = ap.parse_args()

    val = np.load(f"{args.out}/val_probs.npz")
    t = recall_threshold(val["y"], val["probs"], args.target_recall)

    x, y, paths = load_split(args.cache, "test")
    model = build_model()
    model.load_state_dict(torch.load(f"{args.out}/model.pt", map_location="cpu"))
    p = predict(model, DataLoader(XrayDataset(x, y, eval_transform()), batch_size=64))
    np.savez(f"{args.out}/test_probs.npz", probs=p, y=y, paths=paths)

    ops = {"Default threshold 0.50": metrics_at(y, p, 0.5),
           f"Recall-tuned threshold {t:.2f}": metrics_at(y, p, t)}
    result = {"test_images": int(len(y)), "test_pneumonia": int(y.sum()), "test_normal": int(len(y) - y.sum()),
              "roc_auc": roc_auc_score(y, p), "average_precision": average_precision_score(y, p),
              "target_val_recall": args.target_recall,
              "val": {"roc_auc": roc_auc_score(val["y"], val["probs"]),
                      "at_tuned_threshold": metrics_at(val["y"], val["probs"], t)},
              "test_at_0.5": ops["Default threshold 0.50"],
              "test_at_tuned_threshold": ops[f"Recall-tuned threshold {t:.2f}"]}
    json.dump(result, open(f"{args.out}/metrics.json", "w"), indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))

    plot_curves(y, p, ops[f"Recall-tuned threshold {t:.2f}"], f"{args.out}/roc_pr_curves.png")
    plot_confusion(ops, f"{args.out}/confusion_matrices.png")


if __name__ == "__main__":
    main()
