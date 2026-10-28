"""
Prepare a compact OASIS-1 subset for the LEEN364 project (nilearn VBM version).

Input: the OASIS-1 data downloaded with nilearn (nilearn_data/oasis1), which contains
for each subject a grey matter map (mwrc1*.nii.gz) and a white matter map (mwrc2*.nii.gz),
plus oasis_cross-sectional.csv. Only the grey matter maps are used.

Runs locally. Nothing is uploaded. Only aggregate statistics are printed.

Usage:
    python prepare_oasis1.py --data_dir /Users/suleozdamar/nilearn_data/oasis1

Outputs (in the current folder):
    oasis1_slices.npz          16 coronal grey matter slices per subject + labels
    qc_coronal_slices.png      quality check image of one example subject

Requirements: pip install numpy nibabel pandas matplotlib
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

N_SLICES = 16          # coronal slices per subject, as in the report
CENTER_Y_MM = -24.0    # MNI y coordinate of the hippocampal body (centre of the slab)


def load_metadata(data_dir):
    path = os.path.join(data_dir, "oasis_cross-sectional.csv")
    if not os.path.exists(path):
        sys.exit(f"Metadata not found: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    for col in ("ID", "Age", "CDR"):
        if col not in df.columns:
            sys.exit(f"Metadata is missing column {col}. Found: {list(df.columns)}")
    return df


def select_cohort(df):
    total = len(df)
    df = df[~df["ID"].astype(str).str.endswith("_MR2")]
    df = df[(df["Age"] >= 60) & df["CDR"].notna()].copy()
    df["label"] = (df["CDR"] >= 0.5).astype(int)
    print(f"Metadata rows: {total}")
    print(f"Cohort (age >= 60, CDR available, no second visits): {len(df)} subjects")
    print(f"  CN (CDR 0):    {(df['label'] == 0).sum()}")
    print(f"  AD (CDR>=0.5): {(df['label'] == 1).sum()}")
    for v in sorted(df.loc[df["label"] == 1, "CDR"].unique()):
        print(f"    CDR {v}: {(df['CDR'] == v).sum()}")
    return df.reset_index(drop=True)


def find_gm(data_dir, sid):
    hits = glob.glob(os.path.join(data_dir, sid, f"mwrc1{sid}*.nii*"))
    return hits[0] if hits else None


def load_ras(path):
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(path))  # reorient to RAS: axis 1 = posterior -> anterior
    return np.squeeze(np.asarray(img.get_fdata(), dtype=np.float32)), img.affine


def slab_indices(affine, shape):
    inv = np.linalg.inv(affine)
    j = int(round((inv @ np.array([0.0, CENTER_Y_MM, 0.0, 1.0]))[1]))
    half = N_SLICES // 2
    idx = list(range(j - half, j + half))
    if idx[0] < 0 or idx[-1] >= shape[1]:
        sys.exit(f"Slab {idx[0]}..{idx[-1]} is outside the volume (axis 1 size {shape[1]})")
    return idx


def coronal(vol, j):
    return np.flipud(vol[:, j, :].T)  # rows: superior -> inferior, cols: left -> right (RAS x)


def save_qc(vol, affine, idx, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(2, 1, 1)
    sag = np.flipud(vol[vol.shape[0] // 2, :, :].T)
    ax.imshow(sag, cmap="gray")
    ax.axvspan(idx[0], idx[-1], color="red", alpha=0.25)
    ax.set_title(f"Mid-sagittal view (anterior to the right); red band = {N_SLICES} selected coronal slices")
    ax.axis("off")
    for k, j in enumerate(idx):
        a = fig.add_subplot(4, 8, 17 + k)
        a.imshow(coronal(vol, j), cmap="gray")
        y_mm = (affine @ np.array([0, j, 0, 1]))[1]
        a.set_title(f"y = {y_mm:.0f} mm", fontsize=8)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="folder that contains OAS1_xxxx_MR1 folders and the csv")
    p.add_argument("--out", default="oasis1_slices.npz")
    args = p.parse_args()

    df = select_cohort(load_metadata(args.data_dir))

    images, labels, cdr, age, ids = [], [], [], [], []
    idx, voxel = None, None
    missing = 0
    for k, row in df.iterrows():
        path = find_gm(args.data_dir, row["ID"])
        if path is None:
            missing += 1
            continue
        vol, affine = load_ras(path)
        if idx is None:
            idx = slab_indices(affine, vol.shape)
            voxel = np.abs(np.diag(affine)[:3])
            print(f"Volume shape (RAS): {vol.shape}, voxel size (mm): {np.round(voxel, 2).tolist()}")
            print(f"Coronal slab: axis-1 indices {idx[0]}..{idx[-1]} "
                  f"(MNI y {(affine @ [0, idx[0], 0, 1])[1]:.0f} to {(affine @ [0, idx[-1], 0, 1])[1]:.0f} mm)")
            save_qc(vol, affine, idx, "qc_coronal_slices.png")
            print("Saved qc_coronal_slices.png")
        images.append(np.stack([coronal(vol, j) for j in idx]).astype(np.float16))
        labels.append(int(row["label"]))
        cdr.append(float(row["CDR"]))
        age.append(float(row["Age"]))
        ids.append(str(row["ID"]))
        if (k + 1) % 25 == 0:
            print(f"  processed {k + 1}/{len(df)}")

    if not images:
        sys.exit("No grey matter maps (mwrc1*.nii.gz) found. Check --data_dir.")
    if missing:
        print(f"Warning: {missing} cohort subjects had no grey matter map and were skipped")

    np.savez_compressed(args.out, images=np.stack(images), labels=np.array(labels),
                        cdr=np.array(cdr), age=np.array(age), subject_id=np.array(ids),
                        slice_indices=np.array(idx), voxel_size=voxel)
    print(f"Saved {args.out}: {len(images)} subjects x {N_SLICES} slices of {images[0].shape[1:]} "
          f"pixels, {os.path.getsize(args.out) / 1e6:.1f} MB")
    print(f"Final cohort: CN {labels.count(0)}, AD {labels.count(1)}")


if __name__ == "__main__":
    main()
