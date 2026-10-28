"""Data loading, subject-level and slice-level splits, and the PyTorch dataset."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2


def load_npz(path):
    """Load the arrays written by scripts/prepare_oasis1.py."""
    d = np.load(path, allow_pickle=False)
    data = {k: d[k] for k in d.files}
    data["images"] = torch.from_numpy(data["images"].astype(np.float32))  # (N, K, H, W)
    data["labels"] = data["labels"].astype(np.int64)
    return data


def subject_split(labels, seed, val_fraction=0.15, test_fraction=0.15):
    """Stratified split of SUBJECT indices into train / val / test."""
    idx = np.arange(len(labels))
    trval, test = train_test_split(idx, test_size=test_fraction, stratify=labels, random_state=seed)
    train, val = train_test_split(trval, test_size=val_fraction / (1 - test_fraction),
                                  stratify=labels[trval], random_state=seed)
    train, val, test = np.sort(train), np.sort(val), np.sort(test)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test)), "subject overlap"
    return train, val, test


def pairs_for_subjects(subjects, n_slices):
    """All (subject, slice) pairs of the given subjects."""
    return np.array([(s, k) for s in subjects for k in range(n_slices)], dtype=np.int64)


def slice_split(labels, n_slices, seed, val_fraction=0.15, test_fraction=0.15):
    """Stratified split of SLICES, ignoring subjects. Used only for the data leakage experiment."""
    pairs = pairs_for_subjects(np.arange(len(labels)), n_slices)
    y = labels[pairs[:, 0]]
    idx = np.arange(len(pairs))
    trval, test = train_test_split(idx, test_size=test_fraction, stratify=y, random_state=seed)
    train, val = train_test_split(trval, test_size=val_fraction / (1 - test_fraction),
                                  stratify=y[trval], random_state=seed)
    return pairs[train], pairs[val], pairs[test]


def train_stats(images, train_pairs):
    """Mean and standard deviation of all training-slice pixels (no information from val/test)."""
    x = images[train_pairs[:, 0], train_pairs[:, 1]]
    return float(x.mean()), float(x.std())


class SliceDataset(Dataset):
    def __init__(self, images, labels, pairs, mean, std, channels, image_size=224,
                 augment=False, aug_cfg=None):
        self.images = images
        self.labels = labels
        self.pairs = pairs
        self.mean, self.std = mean, std
        self.channels = channels
        self.size = image_size
        self.augment = augment
        self.cfg = aug_cfg or {}
        if augment:
            background = (0.0 - mean) / std
            self.geom = v2.Compose([
                v2.RandomAffine(degrees=self.cfg.get("rotation_degrees", 10),
                                translate=(self.cfg.get("translate", 0.05),) * 2,
                                fill=background),
                v2.RandomHorizontalFlip(p=self.cfg.get("hflip_p", 0.5)),
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        s, k = self.pairs[i]
        x = self.images[s, k][None]                      # (1, H, W)
        x = (x - self.mean) / self.std
        x = F.interpolate(x[None], size=(self.size, self.size), mode="bilinear",
                          align_corners=False)[0]
        if self.augment:
            x = self.geom(x)
            a = 1 + (torch.rand(1) * 2 - 1) * self.cfg.get("intensity_scale", 0.1)
            b = (torch.rand(1) * 2 - 1) * self.cfg.get("intensity_shift", 0.1)
            x = x * a + b
        if self.channels == 3:
            x = x.repeat(3, 1, 1)
        return x, int(self.labels[s]), int(s)
