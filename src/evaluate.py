"""Predictions, subject-level aggregation and evaluation metrics."""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)


@torch.no_grad()
def predict(model, loader, device):
    """AD probability for every slice, with its true label and subject index."""
    model.eval()
    probs, labels, subjects = [], [], []
    for x, y, s in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
        subjects.append(s.numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(subjects)


def to_subject_level(probs, labels, subjects):
    """Mean AD probability over the slices of each subject."""
    ids = np.unique(subjects)
    p = np.array([probs[subjects == i].mean() for i in ids])
    y = np.array([labels[subjects == i][0] for i in ids])
    return p, y, ids


def metrics(y_true, prob, threshold=0.5):
    y_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        sensitivity=tp / (tp + fn) if tp + fn else float("nan"),
        specificity=tn / (tn + fp) if tn + fp else float("nan"),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc=roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else float("nan"),
        tp=int(tp), fn=int(fn), tn=int(tn), fp=int(fp),
    )
