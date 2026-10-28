"""
Runs every experiment of the report and writes all tables and figures.

    python run_all.py --data oasis1_slices.npz

Outputs (outputs/):
    metrics/all_runs.csv, logs/*.csv, figures/*.png, results_summary.md
    splits/*.csv and checkpoints/ (contain subject IDs / weights; never commit them)
"""
import argparse
import os
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src import figures
from src.data import (SliceDataset, load_npz, pairs_for_subjects, slice_split, subject_split,
                      train_stats)
from src.evaluate import metrics, predict, to_subject_level
from src.gradcam import GradCAM
from src.models import build_model, count_parameters
from src.train import class_weights, train_model

MODEL_LABELS = {"baseline_cnn": "Baseline CNN", "resnet18": "ResNet-18 (TL)"}
METRIC_COLS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1", "auc"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(data, tr, va, te, mcfg, cfg, device):
    mean, std = train_stats(data["images"], tr)
    common = dict(images=data["images"], labels=data["labels"], mean=mean, std=std,
                  channels=mcfg["channels"], image_size=cfg["data"]["image_size"])
    tcfg = cfg["training"]
    kw = dict(batch_size=tcfg["batch_size"], num_workers=tcfg["num_workers"],
              pin_memory=device.type == "cuda")
    train = DataLoader(SliceDataset(pairs=tr, augment=True, aug_cfg=cfg["augmentation"], **common),
                       shuffle=True, **kw)
    val = DataLoader(SliceDataset(pairs=va, **common), shuffle=False, **kw)
    test = DataLoader(SliceDataset(pairs=te, **common), shuffle=False, **kw)
    return train, val, test, mean, std


def fmt(values, pct=True):
    v = np.asarray(values, dtype=float)
    m, s = np.nanmean(v), (np.nanstd(v, ddof=1) if len(v) > 1 else float("nan"))
    if pct:
        return f"{100 * m:.1f} ± {100 * s:.1f}"
    return f"{m:.2f} ± {s:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test (1 seed, 2 epochs)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.quick:
        cfg["seeds"] = cfg["seeds"][:2]
        cfg["training"]["max_epochs"] = 2
        cfg["training"]["num_workers"] = 0
        cfg["data"]["image_size"] = 64
        cfg["models"]["resnet18"]["pretrained"] = False
    out = cfg["output_dir"]
    for sub in ("metrics", "logs", "figures", "splits", "checkpoints"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    device = get_device()
    data = load_npz(args.data or cfg["data"]["path"])
    labels, images = data["labels"], data["images"]
    n_subj, n_slices = images.shape[:2]
    mid = n_slices // 2
    print(f"Device: {device}. Subjects: {n_subj} (CN {int((labels == 0).sum())}, AD {int((labels == 1).sum())}), "
          f"{n_slices} slices each, slice size {tuple(images.shape[2:])}")

    seeds = cfg["seeds"]
    tcfg = cfg["training"]
    n_runs = 3 * len(seeds)
    t_start = time.time()
    estimate_done = {"flag": False}

    def estimate(sec):
        if not estimate_done["flag"]:
            estimate_done["flag"] = True
            print(f"First epoch took {sec:.1f} s. Upper-bound estimate for all {n_runs} runs: "
                  f"{sec * tcfg['max_epochs'] * n_runs / 3600:.1f} h (early stopping usually shortens this)")

    rows, infos, keep = [], [], {}
    param_counts = {}

    # Runs 1 and 2: subject-level split
    for name in ("baseline_cnn", "resnet18"):
        mcfg = cfg["models"][name]
        for seed in seeds:
            set_seed(seed)
            tr_s, va_s, te_s = subject_split(labels, seed, cfg["data"]["val_fraction"], cfg["data"]["test_fraction"])
            split_file = os.path.join(out, "splits", f"split_seed{seed}.csv")
            if not os.path.exists(split_file):
                pd.DataFrame({"subject_id": data["subject_id"][np.r_[tr_s, va_s, te_s]],
                              "set": ["train"] * len(tr_s) + ["val"] * len(va_s) + ["test"] * len(te_s)}
                             ).to_csv(split_file, index=False)
            tr, va, te = (pairs_for_subjects(x, n_slices) for x in (tr_s, va_s, te_s))
            train_l, val_l, test_l, mean, std = make_loaders(data, tr, va, te, mcfg, cfg, device)
            model = build_model(name, mcfg)
            param_counts[name] = count_parameters(model)
            print(f"\n[{MODEL_LABELS[name]} | subject-level split | seed {seed}]")
            model, hist, info = train_model(model, train_l, val_l, mcfg["lr"], class_weights(labels, tr_s, device),
                                            tcfg, device, on_first_epoch=estimate)
            hist.to_csv(os.path.join(out, "logs", f"{name}_seed{seed}.csv"), index=False)
            p, y, s = predict(model, test_l, device)
            ps, ys, ids = to_subject_level(p, y, s)
            m_slice, m_subj = metrics(y, p), metrics(ys, ps)
            for level, m in (("slice", m_slice), ("subject", m_subj)):
                rows.append(dict(model=name, split="subject-level", seed=seed, level=level, **m))
            last = hist.iloc[-1]
            infos.append(dict(model=name, seed=seed, **info, final_train_loss=last["train_loss"],
                              final_val_loss=last["val_loss"],
                              val_loss_rose=bool(hist["val_loss"].iloc[-1] > info["best_val_loss"] + 1e-6
                                                 and hist["train_loss"].iloc[-1] < hist["train_loss"].iloc[info["best_epoch"] - 1])))
            print(f"  best epoch {info['best_epoch']}, subject-level balanced acc {m_subj['balanced_accuracy']:.3f}, AUC {m_subj['auc']:.3f}")
            if seed == seeds[0]:
                torch.save(model.state_dict(), os.path.join(out, "checkpoints", f"{name}_seed{seed}.pt"))
                keep[name] = dict(model=model, hist=hist, info=info, te=te, te_s=te_s, tr_s=tr_s, va_s=va_s,
                                  p=p, y=y, ps=ps, ys=ys, ids=ids, m_subj=m_subj, mean=mean, std=std)

    # Run 3: data leakage experiment (slice-level split), ResNet-18
    mcfg = cfg["models"]["resnet18"]
    for seed in seeds:
        set_seed(seed)
        tr, va, te = slice_split(labels, n_slices, seed, cfg["data"]["val_fraction"], cfg["data"]["test_fraction"])
        train_l, val_l, test_l, _, _ = make_loaders(data, tr, va, te, mcfg, cfg, device)
        model = build_model("resnet18", mcfg)
        print(f"\n[ResNet-18 | SLICE-level split (leakage) | seed {seed}]")
        model, hist, info = train_model(model, train_l, val_l, mcfg["lr"], class_weights(labels, tr[:, 0], device),
                                        tcfg, device, on_first_epoch=estimate)
        hist.to_csv(os.path.join(out, "logs", f"resnet18_sliceSplit_seed{seed}.csv"), index=False)
        p, y, _ = predict(model, test_l, device)
        m = metrics(y, p)
        rows.append(dict(model="resnet18", split="slice-level", seed=seed, level="slice", **m))
        print(f"  slice-level test accuracy {m['accuracy']:.3f}")

    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(out, "metrics", "all_runs.csv"), index=False)
    pd.DataFrame(infos).to_csv(os.path.join(out, "metrics", "training_info.csv"), index=False)
    total_time = time.time() - t_start

    # Figures
    fdir = os.path.join(out, "figures")
    figures.example_slices(images, labels, os.path.join(fdir, "fig1_example_slices.png"), mid)
    for name, fname in (("resnet18", "fig2_learning_curves_resnet18.png"),
                        ("baseline_cnn", "fig2b_learning_curves_baseline.png")):
        figures.learning_curves(keep[name]["hist"], keep[name]["info"]["best_epoch"], MODEL_LABELS[name],
                                os.path.join(fdir, fname))
    r = keep["resnet18"]
    figures.confusion(r["m_subj"], os.path.join(fdir, "fig3_confusion_matrix_resnet18.png"),
                      "ResNet-18, test subjects (split 1)")
    figures.example_predictions(images, r["te"], r["p"], r["y"], os.path.join(fdir, "fig4_example_predictions.png"))

    # Grad-CAM on correctly classified AD test subjects (middle slice)
    model = r["model"].to(device)
    cam = GradCAM(model, model.layer4)
    ad_ok = [int(i) for i, pp, yy in zip(r["ids"], r["ps"], r["ys"]) if yy == 1 and pp >= 0.5][:4]
    raws, cams, probs = [], [], []
    size = cfg["data"]["image_size"]
    for s in ad_ok:
        ds = SliceDataset(images, labels, np.array([[s, mid]]), r["mean"], r["std"], 3, size)
        x = ds[0][0][None].to(device)
        cams.append(cam(x, target_class=1))
        with torch.no_grad():
            probs.append(float(torch.softmax(model(x), 1)[0, 1]))
        raws.append(images[s, mid].numpy())
    cam.remove()
    if raws:
        figures.gradcam_grid(raws, cams, probs, os.path.join(fdir, "fig5_gradcam_resnet18.png"))

    write_summary(out, cfg, data, runs, pd.DataFrame(infos), keep, param_counts, device, total_time, n_slices)
    print(f"\nDone in {total_time / 60:.1f} min. See {out}/results_summary.md and {out}/figures/")


def write_summary(out, cfg, data, runs, infos, keep, param_counts, device, total_time, n_slices):
    labels, cdr = data["labels"], data["cdr"]
    L = ["# Results summary", ""]

    L += ["## 1. Cohort", f"- CN subjects (N_CN): {int((labels == 0).sum())}",
          f"- AD subjects: {int((labels == 1).sum())}", f"- Total: {len(labels)}",
          f"- Slices per subject: {n_slices}, slice size: {tuple(data['images'].shape[2:])} "
          f"(voxel size {data['voxel_size'].tolist() if 'voxel_size' in data else 'n/a'} mm)", ""]

    k = keep["resnet18"]
    L += ["## 2. Table 2 - data split (seed 0 = split 1)", "",
          "| Set | CN subjects | AD subjects | Total subjects | Slices |", "|---|---|---|---|---|"]
    tot = [0, 0, 0, 0]
    for name, ids in (("Training", k["tr_s"]), ("Validation", k["va_s"]), ("Test", k["te_s"])):
        cn, ad = int((labels[ids] == 0).sum()), int((labels[ids] == 1).sum())
        vals = [cn, ad, cn + ad, (cn + ad) * n_slices]
        tot = [a + b for a, b in zip(tot, vals)]
        L.append(f"| {name} | " + " | ".join(map(str, vals)) + " |")
    L += [f"| Total | " + " | ".join(map(str, tot)) + " |", ""]

    for level in ("subject", "slice"):
        L += [f"## 3. Table 3 - {level}-level test results (mean ± std over {len(cfg['seeds'])} seeds)", "",
              "| Model | Accuracy | Balanced acc. | Sensitivity | Specificity | F1-score | AUC |",
              "|---|---|---|---|---|---|---|"]
        for name in ("baseline_cnn", "resnet18"):
            d = runs[(runs.model == name) & (runs.split == "subject-level") & (runs.level == level)]
            cells = [fmt(d[c]) for c in METRIC_COLS[:-1]] + [fmt(d["auc"], pct=False)]
            L.append(f"| {MODEL_LABELS[name]} | " + " | ".join(cells) + " |")
        L.append("")

    a = runs[(runs.model == "resnet18") & (runs.split == "subject-level") & (runs.level == "slice")]["accuracy"]
    b = runs[(runs.split == "slice-level")]["accuracy"]
    L += ["## 4. Table 4 - data leakage experiment (ResNet-18, slice-level test accuracy)", "",
          "| Split strategy | Slice-level test accuracy |", "|---|---|",
          f"| Subject-level split (correct) | {fmt(a)} |", f"| Slice-level split (incorrect) | {fmt(b)} |", ""]

    L += ["## 5. Learning curves", "", "| Model | Seed | Best epoch | Early stop epoch | Epochs run | "
          "Final train loss | Final val loss | Val loss rose while train loss fell |", "|---|---|---|---|---|---|---|---|"]
    for _, r in infos.iterrows():
        stop = int(r.stopped_epoch) if pd.notna(r.stopped_epoch) else "not triggered"
        L.append(f"| {MODEL_LABELS[r.model]} | {r.seed} | {r.best_epoch} | {stop} | {r.epochs_run} | "
                 f"{r.final_train_loss:.3f} | {r.final_val_loss:.3f} | {'yes' if r.val_loss_rose else 'no'} |")
    L.append("")

    m = k["m_subj"]
    missed = [cdr[i] for i, pp, yy in zip(k["ids"], k["ps"], k["ys"]) if yy == 1 and pp < 0.5]
    L += ["## 6. Confusion matrix (ResNet-18, seed 0, subject level)",
          f"- TP (AD correct): {m['tp']}, FN (AD missed): {m['fn']}, TN (CN correct): {m['tn']}, FP (CN called AD): {m['fp']}",
          "- Missed AD subjects by CDR: " + (", ".join(f"CDR {v}: {missed.count(v)}" for v in sorted(set(missed))) or "none"), ""]

    L += ["## 7. Parameter counts"] + [f"- {MODEL_LABELS[n]}: {c:,}" for n, c in param_counts.items()] + [""]

    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "none"
    import torchvision
    L += ["## 8. Environment", f"- Device: {device.type}, GPU: {gpu}",
          f"- Python {platform.python_version()}, PyTorch {torch.__version__}, torchvision {torchvision.__version__}",
          f"- Total run time: {total_time / 60:.1f} min", ""]

    sb = runs[(runs.split == "subject-level") & (runs.level == "subject")]
    bb = sb[sb.model == "baseline_cnn"]["balanced_accuracy"]
    rr = sb[sb.model == "resnet18"]["balanced_accuracy"]
    diff = rr.mean() - bb.mean()
    larger = abs(diff) > max(bb.std(ddof=1), rr.std(ddof=1)) if len(bb) > 1 else "n/a (one seed)"
    L += ["## 9. Baseline vs ResNet-18", f"- Difference in mean subject-level balanced accuracy "
          f"(ResNet-18 minus baseline): {100 * diff:.1f} percentage points",
          f"- Larger than both standard deviations: {larger}", ""]

    with open(os.path.join(out, "results_summary.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
