# Results summary

## 1. Cohort
- CN subjects (N_CN): 98
- AD subjects: 100
- Total: 198
- Slices per subject: 16, slice size: (91, 91) (voxel size [2.0, 2.0, 2.0] mm)

## 2. Table 2 - data split (seed 0 = split 1)

| Set | CN subjects | AD subjects | Total subjects | Slices |
|---|---|---|---|---|
| Training | 68 | 70 | 138 | 2208 |
| Validation | 15 | 15 | 30 | 480 |
| Test | 15 | 15 | 30 | 480 |
| Total | 98 | 100 | 198 | 3168 |

## 3. Table 3 - subject-level test results (mean ± std over 5 seeds)

| Model | Accuracy | Balanced acc. | Sensitivity | Specificity | F1-score | AUC |
|---|---|---|---|---|---|---|
| Baseline CNN | 59.3 ± 8.3 | 59.3 ± 8.3 | 54.7 ± 18.5 | 64.0 ± 7.6 | 56.1 ± 14.1 | 0.65 ± 0.11 |
| ResNet-18 (TL) | 60.7 ± 5.5 | 60.7 ± 5.5 | 57.3 ± 10.1 | 64.0 ± 3.7 | 59.0 ± 7.3 | 0.70 ± 0.04 |

## 3. Table 3 - slice-level test results (mean ± std over 5 seeds)

| Model | Accuracy | Balanced acc. | Sensitivity | Specificity | F1-score | AUC |
|---|---|---|---|---|---|---|
| Baseline CNN | 59.7 ± 7.6 | 59.7 ± 7.6 | 55.9 ± 17.3 | 63.5 ± 8.1 | 57.1 ± 13.2 | 0.65 ± 0.10 |
| ResNet-18 (TL) | 62.7 ± 1.9 | 62.7 ± 1.9 | 59.8 ± 8.0 | 65.5 ± 7.0 | 61.4 ± 3.7 | 0.67 ± 0.04 |

## 4. Table 4 - data leakage experiment (ResNet-18, slice-level test accuracy)

| Split strategy | Slice-level test accuracy |
|---|---|
| Subject-level split (correct) | 62.7 ± 1.9 |
| Slice-level split (incorrect) | 89.5 ± 2.4 |

## 5. Learning curves

| Model | Seed | Best epoch | Early stop epoch | Epochs run | Final train loss | Final val loss | Val loss rose while train loss fell |
|---|---|---|---|---|---|---|---|
| Baseline CNN | 0 | 9 | 16 | 16 | 0.594 | 0.784 | yes |
| Baseline CNN | 1 | 26 | not triggered | 30 | 0.610 | 0.757 | no |
| Baseline CNN | 2 | 3 | 10 | 10 | 0.621 | 0.654 | yes |
| Baseline CNN | 3 | 15 | 22 | 22 | 0.630 | 0.615 | yes |
| Baseline CNN | 4 | 9 | 16 | 16 | 0.623 | 0.633 | yes |
| ResNet-18 (TL) | 0 | 1 | 8 | 8 | 0.229 | 1.452 | yes |
| ResNet-18 (TL) | 1 | 1 | 8 | 8 | 0.243 | 0.959 | yes |
| ResNet-18 (TL) | 2 | 1 | 8 | 8 | 0.234 | 1.222 | yes |
| ResNet-18 (TL) | 3 | 2 | 9 | 9 | 0.198 | 0.898 | yes |
| ResNet-18 (TL) | 4 | 2 | 9 | 9 | 0.186 | 1.435 | yes |

## 6. Confusion matrix (ResNet-18, seed 0, subject level)
- TP (AD correct): 11, FN (AD missed): 4, TN (CN correct): 10, FP (CN called AD): 5
- Missed AD subjects by CDR: CDR 0.5: 4

## 7. Parameter counts
- Baseline CNN: 388,834
- ResNet-18 (TL): 11,177,538

## 8. Environment
- Device: cuda, GPU: Tesla T4
- Python 3.13.16, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130
- Total run time: 32.9 min

## 9. Baseline vs ResNet-18
- Difference in mean subject-level balanced accuracy (ResNet-18 minus baseline): 1.3 percentage points
- Larger than both standard deviations: False
