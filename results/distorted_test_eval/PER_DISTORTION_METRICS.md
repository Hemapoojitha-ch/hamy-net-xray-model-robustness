# Per-Distortion Metrics — Every Model × Every Distortion

**Test set**: 241 studies (canonical v3 split). **Pneumonia prevalence**: 0.328.
**Threshold for Accuracy / Precision / Recall / F1**: **0.5** (consistent across all models and conditions).

Best value in each column is **bolded**. The confusion-matrix counts (TP / FP / TN / FN) are
included so you can re-derive any metric or build cost-weighted alternatives.

---

## 1. Clean (no distortion)

| Model    | AUROC      | Accuracy   | Precision  | Recall     | F1         |  TP |  FP |  TN | FN |
|----------|-----------:|-----------:|-----------:|-----------:|-----------:|----:|----:|----:|---:|
| v2       |     0.7121 |     0.7386 |     0.6379 |     0.4684 |     0.5401 |  37 |  21 | 141 | 42 |
| v3       |     0.7483 |     0.7386 |     0.7353 |     0.3165 |     0.4425 |  25 |   9 | 153 | 54 |
| v4_swa   |     0.7415 |     0.7510 |     0.6508 |     0.5190 |     0.5775 |  41 |  22 | 140 | 38 |
| v5       |     0.7415 |     0.6680 |     0.4951 |     0.6456 |     0.5604 |  51 |  52 | 110 | 28 |
| **v6**   | **0.9019** | **0.8548** | **0.7500** | **0.8354** | **0.7904** |  66 |  22 | 140 | 13 |

---

## 2. Contrast reduction (s2)

| Model    | AUROC      | Accuracy   | Precision  | Recall     | F1         |  TP |  FP |  TN | FN |
|----------|-----------:|-----------:|-----------:|-----------:|-----------:|----:|----:|----:|---:|
| v2       |     0.6760 |     0.6473 |     0.4712 |     0.6203 |     0.5355 |  49 |  55 | 107 | 30 |
| v3       |     0.7259 |     0.7303 |     0.7188 |     0.2911 |     0.4144 |  23 |   9 | 153 | 56 |
| v4_swa   |     0.7321 |     0.7552 |     0.6667 |     0.5063 |     0.5755 |  40 |  20 | 142 | 39 |
| v5       |     0.7193 |     0.6722 |     0.5000 |     0.5823 |     0.5380 |  46 |  46 | 116 | 33 |
| **v6**   | **0.8999** | **0.8465** | **0.7561** | **0.7848** | **0.7702** |  62 |  20 | 142 | 17 |

---

## 3. Motion blur (s2)

| Model    | AUROC      | Accuracy   | Precision  | Recall     | F1         |  TP |  FP |  TN | FN |
|----------|-----------:|-----------:|-----------:|-----------:|-----------:|----:|----:|----:|---:|
| v2       |     0.7064 |     0.7510 |     0.6667 |     0.4810 |     0.5588 |  38 |  19 | 143 | 41 |
| v3       |     0.7473 |     0.7427 |     0.7429 |     0.3291 |     0.4561 |  26 |   9 | 153 | 53 |
| v4_swa   |     0.7479 |     0.7552 |     0.6667 |     0.5063 |     0.5755 |  40 |  20 | 142 | 39 |
| v5       |     0.7411 |     0.6763 |     0.5050 |     0.6456 |     0.5667 |  51 |  50 | 112 | 28 |
| **v6**   | **0.9011** | **0.8548** | **0.7558** | **0.8228** | **0.7879** |  65 |  21 | 141 | 14 |

---

## 4. Gaussian noise (s2)

| Model    | AUROC      | Accuracy   | Precision  | Recall     | F1         |  TP |  FP |  TN | FN |
|----------|-----------:|-----------:|-----------:|-----------:|-----------:|----:|----:|----:|---:|
| v2       |     0.7074 |     0.7386 |     0.6379 |     0.4684 |     0.5401 |  37 |  21 | 141 | 42 |
| v3       |     0.7465 |     0.7427 |     0.7429 |     0.3291 |     0.4561 |  26 |   9 | 153 | 53 |
| v4_swa   |     0.7429 |     0.7510 |     0.6610 |     0.4937 |     0.5652 |  39 |  20 | 142 | 40 |
| v5       |     0.7345 |     0.6680 |     0.4952 |     0.6582 |     0.5652 |  52 |  53 | 109 | 27 |
| **v6**   | **0.9018** | **0.8465** | **0.7333** | **0.8354** | **0.7811** |  66 |  24 | 138 | 13 |

---

## 5. Mixed distortion

| Model    | AUROC      | Accuracy   | Precision  | Recall     | F1         |  TP |  FP |  TN | FN |
|----------|-----------:|-----------:|-----------:|-----------:|-----------:|----:|----:|----:|---:|
| v2       |     0.6997 |     0.7095 |     0.5616 |     0.5190 |     0.5395 |  41 |  32 | 130 | 38 |
| v3       |     0.7415 |     0.7427 |     0.7429 |     0.3291 |     0.4561 |  26 |   9 | 153 | 53 |
| v4_swa   |     0.7393 |     0.7676 |     0.6885 |     0.5316 |     0.6000 |  42 |  19 | 143 | 37 |
| v5       |     0.7350 |     0.6846 |     0.5155 |     0.6329 |     0.5682 |  50 |  47 | 115 | 29 |
| **v6**   | **0.8975** | **0.8423** | **0.7253** | **0.8354** | **0.7765** |  66 |  25 | 137 | 13 |

---

## 6. Quick observations across the 5 conditions

**v6 dominates every column in every condition.** AUROC stays in **0.8975 – 0.9019** across
all 5 conditions; F1 stays in **0.77 – 0.79**. Worst-case degradation is contrast reduction
(F1 drops 0.020), still leaving v6 ~0.20 F1 ahead of the next-best model.

**v3 is precision-heavy / recall-light at threshold 0.5.** Precision stays at **0.72 – 0.74**
across conditions but Recall is only **0.29 – 0.33** — it confidently flags pneumonia when it
predicts positive, but misses two-thirds of true cases at this threshold. If v3 needed to be
deployed it would benefit from threshold tuning down to ~0.30 – 0.35.

**v4_swa is the best image-only model on most distortions** by F1 — winning under
mixed (0.6000), contrast_reduction (0.5755), and tied with v3/v2 elsewhere. SWA's flat-minimum
fine-tune visibly buys robustness margin over plain v3.

**v5 has the highest Recall among image-only models** (0.58 – 0.66) but its lower Precision
caps F1 around v3/v4_swa levels. The TorchXRayVision DenseNet121 backbone gives it a more
sensitive operating point — useful if the cost of false negatives is high.

**v2 sits at the bottom** in nearly every column at every condition — as expected from a
standard ViT-base/224 baseline with no extra augmentation.

---

## 7. Files

```
results/distorted_test_eval/
├── PER_DISTORTION_METRICS.md                           ← this file
├── distortion_eval_per_distortion_all_metrics.csv      ← combined CSV: (distortion, model) × all metrics
└── per_distortion/
    ├── clean_metrics.csv
    ├── contrast_reduction_metrics.csv
    ├── motion_blur_metrics.csv
    ├── gaussian_noise_metrics.csv
    └── mixed_metrics.csv
```
