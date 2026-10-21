"""Index and cache the Kaggle Chest X-Ray Pneumonia dataset (paultimothymooney/chest-xray-pneumonia).

Expects the original folder layout:
    <root>/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg

Each image is decoded, converted to grayscale and resized to CACHE_SIZE x CACHE_SIZE once,
so training on CPU doesn't spend most of its time decoding large JPEGs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["NORMAL", "PNEUMONIA"]
CACHE_SIZE = 256
# Image counts of the official Kaggle release (v2, 5,856 images).
EXPECTED = {
    "train": {"NORMAL": 1341, "PNEUMONIA": 3875},
    "val": {"NORMAL": 8, "PNEUMONIA": 8},
    "test": {"NORMAL": 234, "PNEUMONIA": 390},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing train/ val/ test/")
    ap.add_argument("--out", default="cache")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for split in ["train", "val", "test"]:
        images, labels, paths = [], [], []
        for label, cls in enumerate(CLASSES):
            files = sorted(p for p in (root / split / cls).iterdir()
                           if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
            assert len(files) == EXPECTED[split][cls], \
                f"{split}/{cls}: found {len(files)}, expected {EXPECTED[split][cls]}"
            for p in files:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                assert img is not None, f"unreadable image {p}"
                images.append(cv2.resize(img, (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA))
                labels.append(label)
                paths.append(str(p.relative_to(root)))
        np.savez(out / f"{split}.npz", x=np.stack(images), y=np.array(labels), paths=np.array(paths))
        digest = hashlib.sha256("".join(paths).encode()).hexdigest()[:16]
        manifest[split] = {c: int(sum(1 for l in labels if l == i)) for i, c in enumerate(CLASSES)}
        manifest[split]["file_list_sha256"] = digest
        print(split, manifest[split])

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
