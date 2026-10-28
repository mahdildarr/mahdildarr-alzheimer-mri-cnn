"""Training loop with early stopping on validation loss."""
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def class_weights(labels, train_subject_ids, device):
    """Weights inversely proportional to class frequency in the training set."""
    y = labels[train_subject_ids]
    counts = np.bincount(y, minlength=2).astype(np.float32)
    w = counts.sum() / (2.0 * counts)
    return torch.tensor(w, dtype=torch.float32, device=device)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def train_model(model, train_loader, val_loader, lr, weights, cfg, device, on_first_epoch=None):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["scheduler_factor"], patience=cfg["scheduler_patience"])

    best_loss, best_state, best_epoch, wait, stopped_epoch = float("inf"), None, 0, 0, None
    rows = []
    for epoch in range(1, cfg["max_epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        rows.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss,
                         val_acc=va_acc, lr=optimizer.param_groups[0]["lr"], seconds=time.time() - t0))
        if epoch == 1 and on_first_epoch is not None:
            on_first_epoch(rows[-1]["seconds"])
        if va_loss < best_loss - 1e-6:
            best_loss, best_epoch, wait = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg["early_stopping_patience"]:
                stopped_epoch = epoch
                break

    model.load_state_dict(best_state)
    info = dict(best_epoch=best_epoch, stopped_epoch=stopped_epoch, epochs_run=len(rows),
                best_val_loss=best_loss)
    return model, pd.DataFrame(rows), info
