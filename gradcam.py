"""Grad-CAM heatmaps on real test X-rays: which region drives the pneumonia score."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import IMG_SIZE, XrayDataset, build_model, eval_transform, load_split

INK, SURFACE = "#0b0b0b", "#fcfcfb"


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) on the last conv layer of EfficientNet-B0."""

    def __init__(self, model, layer):
        self.model = model.eval()
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x):
        logit = self.model(x).squeeze()
        self.model.zero_grad()
        logit.backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().numpy(), torch.sigmoid(logit).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--per-group", type=int, default=4)
    args = ap.parse_args()

    metrics = json.load(open(f"{args.out}/metrics.json"))
    t = metrics["test_at_tuned_threshold"]["threshold"]
    x, y, paths = load_split(args.cache, "test")
    probs = np.load(f"{args.out}/test_probs.npz")["probs"]

    model = build_model()
    model.load_state_dict(torch.load(f"{args.out}/model.pt", map_location="cpu"))
    cam = GradCAM(model, model.conv_head)
    ds = XrayDataset(x, y, eval_transform())

    pred = probs >= t
    groups = {
        "Pneumonia, caught (true positive)": np.where(pred & (y == 1))[0],
        "Pneumonia, missed (false negative)": np.where(~pred & (y == 1))[0],
        "Normal, false alarm (false positive)": np.where(pred & (y == 0))[0],
        "Normal, cleared (true negative)": np.where(~pred & (y == 0))[0],
    }
    rng = np.random.default_rng(0)
    n = args.per_group
    fig, axes = plt.subplots(len(groups), n, figsize=(2.6 * n, 3.1 * len(groups)), facecolor=SURFACE)
    off = (x.shape[1] - IMG_SIZE) // 2
    for row, (name, idx) in enumerate(groups.items()):
        chosen = rng.choice(idx, size=min(n, len(idx)), replace=False) if len(idx) else []
        for col in range(n):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(chosen):
                continue
            i = chosen[col]
            heat, p = cam(ds[i][0].unsqueeze(0))
            ax.imshow(x[i][off:off + IMG_SIZE, off:off + IMG_SIZE], cmap="gray")
            ax.imshow(heat, cmap="inferno", alpha=0.45)
            ax.set_title(f"{paths[i].split('/')[-1]}\np(pneumonia) = {p:.2f}", fontsize=8, color=INK)
        axes[row, 0].text(0, 1.2, f"{name}  ({len(idx)} in test set)", transform=axes[row, 0].transAxes,
                          fontsize=11, color=INK, fontweight="bold")
    fig.suptitle(f"Grad-CAM on real test X-rays (threshold {t:.2f})\nBrighter = more influence on the pneumonia score",
                 color=INK, fontsize=11, x=0.02, y=0.985, ha="left")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.01, wspace=0.08, hspace=0.42)
    fig.savefig(f"{args.out}/gradcam_grid.png", dpi=130)

    # Two pneumonia cases side by side for the README: one where the heatmap sits on the
    # opacity in the lung, one where it drifts to the neck and lines (a shortcut to watch for).
    examples = ["test/PNEUMONIA/person134_bacteria_644.jpeg", "test/PNEUMONIA/person100_bacteria_477.jpeg"]
    notes = ["Heatmap on the opacity in the lung", "Heatmap drifts to neck and lines"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8.4), facecolor=SURFACE)
    for row, (name, note) in enumerate(zip(examples, notes)):
        i = int(np.where(paths == name)[0][0])
        heat, p = cam(ds[i][0].unsqueeze(0))
        crop = x[i][off:off + IMG_SIZE, off:off + IMG_SIZE]
        axes[row, 0].imshow(crop, cmap="gray")
        axes[row, 1].imshow(crop, cmap="gray")
        axes[row, 1].imshow(heat, cmap="inferno", alpha=0.45)
        axes[row, 0].set_title(name, fontsize=9, color=INK)
        axes[row, 1].set_title(f"{note}, p(pneumonia) = {p:.2f}", fontsize=9, color=INK)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{args.out}/gradcam_example.png", dpi=130)


if __name__ == "__main__":
    main()
