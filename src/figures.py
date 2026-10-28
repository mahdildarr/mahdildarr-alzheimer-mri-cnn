"""Figures for the report (PNG, 300 dpi)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DPI = 300
CLASS_NAMES = ["CN", "AD"]
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})


def _upsample(img, size):
    t = torch.as_tensor(img, dtype=torch.float32)[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def example_slices(images, labels, path, slice_index):
    cn = int(np.where(labels == 0)[0][0])
    ad = int(np.where(labels == 1)[0][0])
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    for ax, s, name in zip(axes, (cn, ad), ("Cognitively normal (CN)", "Alzheimer's disease (AD)")):
        ax.imshow(images[s, slice_index].numpy(), cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close(fig)


def learning_curves(history, best_epoch, model_label, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    e = history["epoch"]
    a1.plot(e, history["train_acc"], label="Training")
    a1.plot(e, history["val_acc"], label="Validation")
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Accuracy")
    a2.plot(e, history["train_loss"], label="Training")
    a2.plot(e, history["val_loss"], label="Validation")
    a2.set_xlabel("Epoch")
    a2.set_ylabel("Loss")
    for ax in (a1, a2):
        ax.axvline(best_epoch, color="gray", ls="--", lw=1, label=f"Best epoch ({best_epoch})")
        ax.grid(alpha=0.3)
        ax.legend()
    a1.set_title(f"{model_label}: accuracy")
    a2.set_title(f"{model_label}: loss")
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close(fig)


def confusion(m, path, title):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], [f"Predicted {c}" for c in CLASS_NAMES])
    ax.set_yticks([0, 1], [f"True {c}" for c in CLASS_NAMES])
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close(fig)


def example_predictions(images, test_pairs, probs, labels, path, seed=0, n=3):
    pred = (probs >= 0.5).astype(int)
    rng = np.random.default_rng(seed)
    correct = np.where(pred == labels)[0]
    wrong = np.where(pred != labels)[0]
    pick_c = rng.choice(correct, size=min(n, len(correct)), replace=False) if len(correct) else []
    pick_w = rng.choice(wrong, size=min(n, len(wrong)), replace=False) if len(wrong) else []
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4))
    for row, picks, name in ((0, pick_c, "Correct"), (1, pick_w, "Incorrect")):
        for col in range(n):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(picks):
                continue
            i = picks[col]
            s, k = test_pairs[i]
            ax.imshow(images[s, k].numpy(), cmap="gray")
            ax.set_title(f"{name}\nTrue: {CLASS_NAMES[labels[i]]}, P(AD) = {probs[i]:.2f}")
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close(fig)


def gradcam_grid(raw_slices, cams, probs, path):
    n = len(raw_slices)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.2))
    axes = np.array(axes).reshape(2, n)
    for j in range(n):
        img = _upsample(raw_slices[j], cams[j].shape[0])
        axes[0, j].imshow(img, cmap="gray")
        axes[0, j].set_title(f"AD subject {j + 1}, P(AD) = {probs[j]:.2f}")
        axes[1, j].imshow(img, cmap="gray")
        axes[1, j].imshow(cams[j], cmap="jet", alpha=0.4)
        axes[1, j].set_title("Grad-CAM")
        for r in range(2):
            axes[r, j].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close(fig)
