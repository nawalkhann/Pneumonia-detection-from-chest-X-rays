"""Fine-tune an ImageNet-pretrained EfficientNet-B0 to detect pneumonia.

Kaggle's own val split has only 16 images, which is too few to pick a checkpoint or a
threshold. So the Kaggle train + val images are pooled and ~10% of patients are held out
as a validation set (split by patient, so no patient appears on both sides).
The test set is never touched here.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from common import XrayDataset, build_model, eval_transform, load_split, patient_id, train_transform


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs = [torch.sigmoid(model(x).squeeze(1)) for x, _ in loader]
    return torch.cat(probs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--weights", required=True, help="ImageNet EfficientNet-B0 weights (.pth)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    torch.manual_seed(0)

    xtr, ytr, ptr = load_split(args.cache, "train")
    xva, yva, pva = load_split(args.cache, "val")
    x, y, paths = np.concatenate([xtr, xva]), np.concatenate([ytr, yva]), np.concatenate([ptr, pva])
    groups = [patient_id(p) for p in paths]
    tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0).split(x, y, groups))
    assert not set(groups[i] for i in tr_idx) & set(groups[i] for i in va_idx)
    np.savez(f"{args.out}/split.npz", train=paths[tr_idx], val=paths[va_idx])
    print(f"train {len(tr_idx)} (pneumonia {y[tr_idx].sum()}), val {len(va_idx)} (pneumonia {y[va_idx].sum()})")

    train_dl = DataLoader(XrayDataset(x[tr_idx], y[tr_idx], train_transform()),
                          batch_size=args.batch_size, shuffle=True, num_workers=3, persistent_workers=True)
    val_dl = DataLoader(XrayDataset(x[va_idx], y[va_idx], eval_transform()), batch_size=64, num_workers=2)

    # Class weights inversely proportional to class frequency (sklearn's "balanced"), so the
    # 3:1 pneumonia majority doesn't swamp NORMAL. Recall is then pushed up by the threshold.
    n_pos, n_neg = y[tr_idx].sum(), len(tr_idx) - y[tr_idx].sum()
    w_pos, w_neg = len(tr_idx) / (2 * n_pos), len(tr_idx) / (2 * n_neg)

    model = build_model(args.weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl),
                                                pct_start=0.15)
    best_auc, history = -1, []
    for epoch in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for xb, yb in train_dl:
            logits = model(xb).squeeze(1)
            weight = torch.where(yb == 1, w_pos, w_neg)
            loss = F.binary_cross_entropy_with_logits(logits, yb, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(yb)
        p = predict(model, val_dl)
        auc = roc_auc_score(y[va_idx], p)
        recall = ((p >= 0.5) & (y[va_idx] == 1)).sum() / y[va_idx].sum()
        history.append({"epoch": epoch + 1, "train_loss": total / len(tr_idx), "val_auc": auc,
                        "val_recall@0.5": recall, "seconds": time.time() - t0})
        print(history[-1], flush=True)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"{args.out}/model.pt")
            np.savez(f"{args.out}/val_probs.npz", probs=p, y=y[va_idx])

    json.dump(history, open(f"{args.out}/history.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
