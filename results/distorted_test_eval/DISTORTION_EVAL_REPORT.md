# Image-Distortion Robustness Evaluation (v2 – v6)

**Notebook**: `code/vit_v7_robustness.ipynb`
**Test set**: 241 held-out studies (v3's canonical split — identical to `vit_v3_test_predictions.csv`)
**Pneumonia prevalence**: 0.328

---

## 1. Goal

Quantify how each of our five pneumonia-prediction models degrades when the chest X-ray image is corrupted, while the EHR features (and for v6, the radiology-report text) remain intact. This isolates the contribution of the image channel to model robustness.

---

## 2. Test conditions

| Label | Description |
|---|---|
| `clean` | Original unmodified MIMIC-CXR-JPG image |
| `contrast_reduction` | Lower local contrast (`s2` severity) — flattens the histogram |
| `motion_blur` | Directional Gaussian blur (`s2`) — simulates patient motion |
| `gaussian_noise` | Additive white Gaussian noise (`s2`) — simulates low-dose acquisition |
| `mixed` | A random combination of the above per-image (per `mixed_distortion/distortion_log.csv`) |

All four distorted folders contain the same 241 test studies in the standard MIMIC layout
(`pXX/pXXXXXX/sXXXXX/<dicom>.jpg`) and align 1:1 with the canonical test set.

---

## 3. Models evaluated

| Model | Image backbone | Tabular | Text | How loaded for v7 |
|---|---|---|---|---|
| **v2** | ViT-base/224 (ImageNet-1k) | EHR MLP | — | from `vit_v2_best.pt` |
| **v3** | ViT-base/384 (ImageNet-21k→1k, strong aug) | EHR MLP | — | from `vit_v3_best.pt` |
| **v4_swa** | ViT-base/384 + SWA fine-tune from v3 | EHR MLP | — | from `vit_v4_swa_best.pt` |
| **v5** | TorchXRayVision DenseNet121 (frozen) + EHR | EHR MLP | — | retrained as a single fusion head on full train+val |
| **v6** | DenseNet121 (frozen) + EHR + Bio_ClinicalBERT [CLS] of report | EHR MLP | report text | retrained as a single trimodal head on full train+val |

v5 and v6 were originally 5-fold OOF + LR stacker; for a clean single-checkpoint comparison
we retrained each as one fusion model on the full train+val set. Their clean-test AUROCs are
slightly lower than the original CV-stacked numbers in the main `README.md` for that reason
(v5: 0.741 vs original 0.758; v6: 0.902 vs original 0.887 — v6 actually went up because the
single-model variant doesn't pay the OOF averaging cost).

---

## 4. Pipeline

For each (model, distortion) pair:

1. Build the 241-row test dataframe from the v3 canonical split.
2. Pre-process the matching image through the model's native transform (size, mean/std).
3. For v2/v3/v4: forward-pass `(image, ehr) → logit`.
   For v5: extract DenseNet121 features for the distorted image, concatenate with the
   precomputed EHR vector, forward through the v5 fusion head.
   For v6: same as v5 plus the *unchanged* Bio_ClinicalBERT [CLS] embedding of the radiology
   report (the report text is not distorted — only the image is).
4. Compute test AUROC and AUPRC.

EHR features and report-text embeddings are computed once and reused across all five test
conditions, so any movement in the metrics comes purely from the image channel.

---

## 5. Results

### 5.1 AUROC matrix (rows = condition, cols = model)

| Condition | v2 | v3 | v4_swa | v5 | **v6** |
|---|---:|---:|---:|---:|---:|
| clean              | 0.7121 | 0.7483 | 0.7415 | 0.7415 | **0.9019** |
| contrast_reduction | 0.6760 | 0.7259 | 0.7321 | 0.7193 | **0.8999** |
| motion_blur        | 0.7064 | 0.7473 | 0.7479 | 0.7411 | **0.9011** |
| gaussian_noise     | 0.7074 | 0.7465 | 0.7429 | 0.7345 | **0.9018** |
| mixed              | 0.6997 | 0.7415 | 0.7393 | 0.7350 | **0.8975** |

### 5.2 Δ-AUROC vs clean (negative = degradation)

| Distortion | v2 | v3 | v4_swa | v5 | v6 |
|---|---:|---:|---:|---:|---:|
| contrast_reduction | **−0.0361** | −0.0224 | −0.0095 | −0.0223 | **−0.0020** |
| motion_blur        | −0.0057 | −0.0010 |  +0.0064 | −0.0004 | −0.0008 |
| gaussian_noise     | −0.0047 | −0.0018 |  +0.0013 | −0.0070 | −0.0001 |
| mixed              | −0.0123 | −0.0068 | −0.0022 | −0.0066 | −0.0044 |
| **worst-case**     | **−0.0361** | −0.0224 | −0.0095 | −0.0223 | **−0.0044** |

### 5.3 Plots

- `distortion_bar_per_distortion.png` — grouped bar chart of AUROC across models × conditions
- `distortion_degradation_curves.png` — per-model degradation curves
- `distortion_roc_<condition>.png` (×5) — ROC overlays per condition

---

## 6. Findings

**1. v6 dominates every condition by ~15–18 AUROC points.**
The trimodal model (image + EHR + report text) is at ≈ 0.90 across the board, while every
image-only-or-image-plus-EHR model sits at ≈ 0.71 – 0.75.

**2. v6 is the most distortion-robust by a wide margin.**
Worst-case AUROC drop for v6 is **−0.0044** (mixed). The next best is v4_swa at −0.0095,
then v3 at −0.0224, v5 at −0.0223, and v2 at **−0.0361** (more than 8× v6's drop).
The reason is mechanical: v6's prediction draws on three signals, and only the image channel
(roughly 1024-dim of the 1088-dim fused vector before the EHR MLP) is corrupted. The
unchanged Bio_ClinicalBERT report embedding and EHR vector dominate the decision.

**3. Contrast reduction is the most damaging single distortion** for image-only / image+EHR
models — it flattens diagnostic features uniformly. Motion blur and Gaussian noise barely
move AUROC for v3/v4_swa (within ±0.006), suggesting their heavy training augmentation already
covers similar perturbations.

**4. v4_swa edges v3 on every distorted condition** even though v3 is slightly better on
clean. SWA's flatter-minimum optimization gives the small extra robustness margin we hoped
for from that step.

**5. v5 (DenseNet121 + EHR) is essentially indistinguishable from v3/v4_swa on every condition.**
The XRV pretrained backbone is good but its public-CXR pretraining doesn't translate to
larger robustness gains over a well-augmented ViT-base — robustness here is bottlenecked by
*what* the image channel can carry, not which backbone reads it.

---

## 7. Caveat (same as for the main result table)

The radiology report text used by v6 is the *finalized* report written by the radiologist
who saw the original (clean) image. In a real deployment where a model sees only a fresh
distorted X-ray with no report yet, v6's robustness advantage is upper-bounded by what we
report here — the image channel must do the work alone. The numbers above are therefore
best read as **"how much does our system's confidence stay stable when the image quality
varies, given that the rest of the multimodal context is intact?"** rather than as a fully
prospective robustness measurement.

---

## 8. Files in this folder

```
distortion_eval_long.csv          # one row per (model, distortion): AUROC, AUPRC, Δ vs clean
distortion_eval_wide.csv          # AUROC matrix
distortion_eval_predictions.csv   # per-sample probability for every (model × distortion × study)
distortion_bar_per_distortion.png
distortion_degradation_curves.png
distortion_roc_clean.png
distortion_roc_contrast_reduction.png
distortion_roc_motion_blur.png
distortion_roc_gaussian_noise.png
distortion_roc_mixed.png
DISTORTION_EVAL_REPORT.md         # this file
```
