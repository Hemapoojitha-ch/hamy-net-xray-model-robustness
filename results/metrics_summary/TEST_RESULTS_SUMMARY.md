# Test Results Summary — All Models, All Conditions

**Test set**: 241 held-out studies (canonical split from `vit_v3_test_predictions.csv`)
**Pneumonia prevalence**: 0.328 (79 positives / 162 negatives)
**Classification threshold**: **0.5** (default; same threshold applied to all models for apples-to-apples comparison)

Metrics: **AUROC**, **Accuracy**, **Precision**, **Recall**, **F1**.
AUROC is threshold-free; the other four are computed at threshold = 0.5 unless noted otherwise.

---

## 1. Clean test set — full leaderboard

Source CSV: `results/metrics_summary/clean_test_metrics.csv`

| Model           | AUROC  | Accuracy | Precision | Recall | F1     |
|-----------------|-------:|---------:|----------:|-------:|-------:|
| v2              | 0.6996 |   0.6390 |    0.4643 | 0.6582 | 0.5445 |
| v3              | 0.7593 |   0.7676 |    0.7347 | 0.4557 | 0.5625 |
| v4_ensemble     | 0.7203 |   0.6846 |    0.5158 | 0.6203 | 0.5632 |
| v4_swa          | 0.7477 |   0.7261 |    0.5844 | 0.5696 | 0.5769 |
| v5_fusion       | 0.7333 |   0.6888 |    0.5244 | 0.5443 | 0.5342 |
| v5_stacker      | 0.7264 |   0.6846 |    0.5169 | 0.5823 | 0.5476 |
| **v6_fusion**   | **0.8875** | **0.8174** | 0.6923 | **0.7975** | **0.7412** |
| v6_stacker      | 0.8862 |   0.8382 |    0.7857 | 0.6962 | 0.7383 |
| xrv_prior_only  | 0.6865 |   0.6846 |    1.0000 | 0.0380 | 0.0732 |

### Observations

**v6_fusion is the headline model**: highest AUROC (0.888) and highest F1 (0.741), with balanced
precision/recall. It also achieves the best recall (0.798) — important for a screening setting
where missing pneumonia is more costly than over-flagging it.

**v6_stacker has higher precision than v6_fusion** (0.786 vs 0.692) but lower recall — a
slightly more conservative classifier; AUROC is essentially tied.

**v3 has the highest precision among image-only models** (0.735) but very low recall (0.456) —
it's confident when it predicts pneumonia but misses many true cases.

**xrv_prior_only is degenerate** at threshold 0.5 — it gives near-zero probabilities so almost
nothing crosses the threshold (Recall=0.038). Its AUROC of 0.687 is decent because the *ranking*
is informative, but it would need a much lower threshold to be operationally useful. This is a
good reminder that AUROC and F1 can disagree sharply for poorly-calibrated probabilities.

---

## 2. Distortion robustness — AUROC

Source CSV: `results/distorted_test_eval/distortion_eval_auroc_wide.csv`

| Distortion         | v2     | v3     | v4_swa | v5     | **v6**     |
|--------------------|-------:|-------:|-------:|-------:|-----------:|
| clean              | 0.7121 | 0.7483 | 0.7415 | 0.7415 | **0.9019** |
| contrast_reduction | 0.6760 | 0.7259 | 0.7321 | 0.7193 | **0.8999** |
| motion_blur        | 0.7064 | 0.7473 | 0.7479 | 0.7411 | **0.9011** |
| gaussian_noise     | 0.7074 | 0.7465 | 0.7429 | 0.7345 | **0.9018** |
| mixed              | 0.6997 | 0.7415 | 0.7393 | 0.7350 | **0.8975** |
| **worst-case Δ**   | **−0.0361** | −0.0224 | −0.0095 | −0.0223 | **−0.0044** |

> Note: v5/v6 here are **single-fusion-head retrains** on full train+val (no CV) to give one
> checkpoint per model. v6's clean AUROC = 0.9019 here vs 0.8875 in the main leaderboard
> because the single-model variant doesn't pay the OOF-averaging cost. v5's clean AUROC =
> 0.7415 vs 0.7333 in the main leaderboard for the same reason.

---

## 3. Distortion robustness — F1 (threshold = 0.5)

Source CSV: `results/distorted_test_eval/distortion_eval_f1_wide.csv`

| Distortion         | v2     | v3     | v4_swa | v5     | **v6**     |
|--------------------|-------:|-------:|-------:|-------:|-----------:|
| clean              | 0.5401 | 0.4425 | 0.5775 | 0.5604 | **0.7904** |
| contrast_reduction | 0.5355 | 0.4144 | 0.5755 | 0.5380 | **0.7702** |
| motion_blur        | 0.5588 | 0.4561 | 0.5755 | 0.5667 | **0.7879** |
| gaussian_noise     | 0.5401 | 0.4561 | 0.5652 | 0.5652 | **0.7811** |
| mixed              | 0.5395 | 0.4561 | 0.6000 | 0.5682 | **0.7765** |
| **worst-case Δ**   | −0.0046 | −0.0281 | −0.0123 | −0.0224 | **−0.0202** |

v6 dominates F1 across every condition by ~0.20–0.35 points. v3's lower F1 reflects its
precision-heavy / recall-light operating point at threshold 0.5 (visible in the clean table).

---

## 4. Distortion robustness — Accuracy / Precision / Recall

Per-metric wide tables are saved alongside the F1 table:
- `distortion_eval_accuracy_wide.csv`
- `distortion_eval_precision_wide.csv`
- `distortion_eval_recall_wide.csv`

The full long-format file with every metric **and** the `Δ vs clean` per metric is at
`distortion_eval_metrics_long.csv`.

---

## 5. Where everything is saved

```
results/
├── metrics_summary/
│   ├── clean_test_metrics.csv            ← leaderboard for clean test (9 model variants)
│   └── TEST_RESULTS_SUMMARY.md           ← this file
└── distorted_test_eval/
    ├── distortion_eval_metrics_long.csv  ← long format; 5 metrics + 5 Δ-metrics + confusion-matrix counts
    ├── distortion_eval_auroc_wide.csv    ← AUROC pivot table
    ├── distortion_eval_accuracy_wide.csv ← Accuracy pivot
    ├── distortion_eval_precision_wide.csv← Precision pivot
    ├── distortion_eval_recall_wide.csv   ← Recall pivot
    ├── distortion_eval_f1_wide.csv       ← F1 pivot
    ├── distortion_eval_predictions.csv   ← per-sample probabilities (unchanged)
    ├── distortion_eval_long.csv          ← original AUROC/AUPRC long file (kept for reference)
    └── ... distortion plots (PNG) ...
```

---

## 6. Caveat on threshold = 0.5

Because pneumonia prevalence in this cohort is 0.328 (below 0.5), threshold = 0.5 is a
conservative choice that tends to favour Precision over Recall. If a clinical use case
prioritises catching every pneumonia case (high Recall), the threshold should be tuned on
a separate validation set — typically using either:
- the prevalence (≈ 0.33) as the threshold, or
- the F1-optimal threshold computed on validation predictions.

For the current per-model headline reporting, threshold = 0.5 is fine and gives consistent
apples-to-apples comparisons across all 5 models and 5 conditions.
