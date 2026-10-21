import re

import numpy as np
import timm
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def patient_id(path):
    """Patient id from a file name, so one patient's images never land in both train and val.

    PNEUMONIA: person1_bacteria_1.jpeg -> person1
    NORMAL:    IM-0115-0001.jpeg -> IM-0115, NORMAL2-IM-1427-0001.jpeg -> NORMAL2-IM-1427
    """
    name = path.split("/")[-1]
    m = re.match(r"(person\d+)_", name)
    if m:
        return m.group(1)
    m = re.match(r"((?:NORMAL2-)?IM-\d+)-", name)
    return m.group(1) if m else name


def build_model(weights_path=None):
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        state = {k: v for k, v in state.items() if not k.startswith("classifier.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert set(missing) == {"classifier.weight", "classifier.bias"} and not unexpected
    return model


def train_transform():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.Normalize(MEAN, STD),
    ])


def eval_transform():
    return T.Compose([T.CenterCrop(IMG_SIZE), T.Normalize(MEAN, STD)])


class XrayDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = torch.from_numpy(self.x[i]).float().div(255).unsqueeze(0).expand(3, -1, -1)
        return self.transform(img), torch.tensor(self.y[i], dtype=torch.float32)


def load_split(cache, split):
    d = np.load(f"{cache}/{split}.npz")
    return d["x"], d["y"], d["paths"]
