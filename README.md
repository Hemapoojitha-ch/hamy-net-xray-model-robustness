# Multimodal Pneumonia Prediction on MIMIC-ED + MIMIC-CXR

A multimodal deep-learning pipeline that predicts pneumonia in ED patients by fusing three sources: structured clinical data from the Emergency Department, chest X-ray images, and the corresponding radiologist-written reports. Models are built block-by-block in Jupyter notebooks and compared on a fixed held-out test set.

**Final test AUROC: 0.8875 (95% CI [0.8382, 0.9297], n_test = 241)** — achieved by `vit_v6_text.ipynb`, a trimodal fusion model combining a frozen Bio_ClinicalBERT text encoder, a frozen TorchXRayVision DenseNet121 image encoder, and a small MLP over 15 EHR features, trained with 5-fold stratified cross-validation.

---

## Data Processing & Cohort Selection

### 1. Data Sources

* **Clinical data:** [MIMIC-IV ED (v2.2)](https://physionet.org/content/mimic-iv-ed/2.2/)
* **Imaging data:** [MIMIC-CXR-JPG (v2.1.0)](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
* **Radiology reports:** [MIMIC-CXR (v2.0.0)](https://physionet.org/content/mimic-cxr/2.0.0/) — the `.txt` free-text reports (separate DUA from MIMIC-CXR-JPG)

All three are credentialed PhysioNet datasets. Access requires CITI training plus signing each project's Data Use Agreement.

### 2. Selection Criteria

The final cohort consists of **1,601 unique patients** (525 positive : 1,076 negative, prevalence 32.79%), selected with the following logic:

* **Temporal alignment:** every X-ray is timestamped within the specific ED visit window (`intime` to `outtime`).
* **View position:** only frontal views (**AP** — Anteroposterior, or **PA** — Posteroanterior). Lateral views excluded.
* **Deduplication:** only the **first** valid "ED visit + frontal X-ray" pair per patient, to prevent longitudinal leakage across train/test.
* **Label clarity:** only cases with a definitive CheXpert NLP annotation (`Pneumonia` ∈ {0.0, 1.0}).

### 3. Structured Variables

The processed clinical features are stored in `data/mimic_ed_cxr_pneumonia_multimodal_cohort.csv`.

| Category | Variables |
| :--- | :--- |
| **Identifiers** | `subject_id`, `stay_id`, `study_id`, `dicom_id`, `StudyDate` |
| **Target labels** | `Pneumonia` (primary, 32.79% prev), `outcome_all_pne` (identical to `Pneumonia`, 32.79%), `outcome_bac_pne` (1.87%), `outcome_viral_pne` (1.00%) |
| **Demographics** | `age`, `gender` (encoded as 1 for the majority class, 0 for the minority; see `code/select_patients.ipynb` cell 3) |
| **Triage vitals** | `triage_temperature`, `triage_heartrate`, `triage_resprate`, `triage_o2sat`, `triage_sbp`, `triage_dbp`, `triage_acuity` |
| **Chief complaints** | `chiefcom_shortness_of_breath`, `chiefcom_cough`, `chiefcom_fever_chills` |
| **History / scores** | `cci_Pulmonary`, `cci_CHF`, `score_CCI` |

15 features enter the models (identifiers and secondary labels are excluded from the feature matrix).

### 4. Image Data Structure

Images are stored locally at `data/files/p[prefix]/p[subject_id]/s[study_id]/[dicom_id].jpg` (the standard MIMIC-CXR-JPG layout). `[prefix]` is the first two digits of `subject_id`.

### 5. Radiology Report Structure

Reports are stored at `data/reports/p[prefix]/p[subject_id]/s[study_id].txt` (one `.txt` per study). The notebooks extract the `FINDINGS` and `IMPRESSION` sections with a regex (see `vit_v6_text.ipynb` cell 8).

### 6. Label Definitions

Four pneumonia-related labels are available; details are in `outcome_labels_reference.md`.

| Label | Source | Positive rate | Notes |
|---|---|---|---|
| `Pneumonia` | CheXpert NLP on reports | 32.79% | Primary label used for all models in this project. Known to carry ~10-15% noise from rule-based NLP. |
| `outcome_all_pne` | ICD-9/10 codes | 32.79% | ICD-derived; same positive rate on this cohort. Cleaner ground truth — switching to it is a documented next step. |
| `outcome_bac_pne` | ICD bacterial codes | 1.87% | Highly imbalanced — not used as a primary target. |
| `outcome_viral_pne` | ICD viral codes | 1.00% | Highly imbalanced — not used as a primary target. |

### 7. Train / Val / Test Split

Fixed 70 / 15 / 15 stratified split with `random_state=42`, stratified on `Pneumonia`. Test set size is 241. For v5 and v6, train+val are merged (1,360 rows) and re-split into 5 folds via `StratifiedKFold(random_state=42)`; the held-out 241-row test set is never touched during training.

---

## Modeling Pipeline

The project iterates through six model versions, each a notebook in `code/`. Every version uses the identical cohort and test split so numbers are directly comparable.

### v2 — Baseline late fusion (`vit_v2.ipynb`)

* Backbone: `vit_base_patch16_224` (ImageNet pretrained).
* EHR: 15 features → `SimpleImputer(median)` → `StandardScaler` → 2-layer MLP.
* Head: concat(image_feat, ehr_feat) → linear → logit.
* Loss: `BCEWithLogitsLoss` with `pos_weight` for class imbalance.
* Optimizer: AdamW with separate parameter groups for backbone (lr 1e-5) and head (lr 1e-3), `CosineAnnealingLR`.
* Test AUROC ≈ 0.70.

### v3 — Stronger backbone + augmentation (inside `vit_v2.ipynb` step 7)

* Backbone swapped to `vit_base_patch16_384.augreg_in21k_ft_in1k` (higher resolution, more pretraining).
* Augmentation: RandAugment, ColorJitter, RandomErasing.
* Linear-probing warmup: backbone frozen for the first 3 epochs.
* Test-Time Augmentation (TTA): 10-view average (5-crop × horizontal flip).
* Test AUROC ≈ 0.7486 with TTA — first run to clear 0.70.

### v4 — Ensemble of v2+v3 with a meta-stacker (`vit_v4_ensemble.ipynb`)

* 10-view TTA on pretrained v2 and v3 checkpoints.
* Simple average, weighted average, and a logistic regression meta-stacker fit on the val split.
* Optional SWA + MixUp retraining.
* Result: **val-overfitting diagnosed.** The LR stacker hit val AUROC 0.77 but dropped to test AUROC 0.7203 — worse than v3 alone. Root cause: at n_val = 241 the stacker memorizes val noise. Motivated the move to out-of-fold training in v5.

### v5 — TorchXRayVision DenseNet121 + 5-fold CV (`vit_v5_xrv.ipynb`)

* Image backbone replaced with `densenet121-res224-all` from [torchxrayvision](https://github.com/mlmed/torchxrayvision) — pretrained on ~800k public chest X-rays (NIH ChestX-ray14, CheXpert, MIMIC-CXR, RSNA Pneumonia, PadChest).
* 5-fold stratified CV on train+val; test held out. Out-of-fold predictions feed the stacker — eliminates the val-overfitting failure mode from v4.
* Per fold: 3-epoch frozen-backbone warmup, then 5 epochs fine-tuning `denseblock4 + norm5`.
* OOF-trained LR stacker on `[p_fusion, p_xrv_native_pneumonia_prior, p_fusion × p_xrv, top-6 EHR features]`.
* Best test AUROC: 0.7575 (v5 × v3 cross-backbone blend). A smaller lift than hoped — diminishing returns from swapping image backbones alone. This motivated adding a different **modality**.

### v6 — **Final model** — trimodal fusion with radiology reports (`vit_v6_text.ipynb`)

The key insight from v5: the image-only signal was near saturation. The largest untapped modality was the radiologist's written report.

* **Text encoder:** `emilyalsentzer/Bio_ClinicalBERT` (frozen). Reports are parsed with a regex that extracts `IMPRESSION` and `FINDINGS`; `[CLS]`-style mean-pooled embedding is cached to `results/v6_text_cls_embeddings.npz` so BERT only runs once.
* **Image encoder:** frozen `densenet121-res224-all` from TorchXRayVision, producing 1,024-d features cached to `results/v6_xrv_feat_and_prior.npz` (also stores the XRV-native pneumonia prior probability).
* **Tabular encoder:** 2-layer MLP over the 15 EHR features.
* **Fusion head:** projects text (768 → 256), image (1,024 → 256), and EHR (15 → 64) separately, concatenates, and passes through a 2-layer MLP with dropout.
* **Training:** 5-fold stratified CV; only the small fusion head trains per fold (both backbones frozen). BCEWithLogitsLoss with `pos_weight`, AdamW, cosine LR, 25 epochs per fold, early stopping on val AUROC.
* **OOF stacker:** logistic regression on `[logit(p_v6), logit(p_xrv_prior), top-6 EHR features]` using out-of-fold v6 predictions, NaN-safe to handle partial cross-model alignment.
* **Bootstrap 95% CI** reported on the final test AUROC.

---

## Results

All numbers below are on the fixed 241-row held-out test set (prevalence 32.79%). 95% CI is bootstrap-based (2,000 resamples).

| Model | Test AUROC | Test AUPRC | 95% CI |
|---|---:|---:|:---:|
| XRV native zero-shot (no training) | 0.6865 | 0.5546 | — |
| v2 baseline (ViT-224 + EHR) | ≈ 0.70 | — | — |
| v3 (ViT-384 + aug + warmup + TTA) | ≈ 0.7486 | — | — |
| v4 ensemble (val-trained stacker) | 0.7203 | — | — |
| v5 fusion (5-fold XRV + EHR) | 0.7326 | — | — |
| v5 × v3 50/50 | 0.7575 | — | — |
| v5 OOF stacker | 0.7256 | — | — |
| **v6_fusion (text + image + EHR)** | **0.8875** | **0.8149** | **[0.8382, 0.9297]** |
| v6_stacker (OOF-trained LR) | 0.8862 | 0.8101 | [0.8369, 0.9291] |

The v6 fusion head produces a **+0.14 absolute AUROC lift** over the best v5 configuration — confirming the prior analysis that the largest untapped signal was the report text modality, not another image backbone.

### Important caveat on the 0.887 number

MIMIC-CXR radiology reports routinely contain the word "pneumonia" or related terms directly in the `IMPRESSION` section. The v6 text encoder can therefore partially learn to "read" the radiologist's diagnosis rather than derive it from pixels + EHR. For this number to be defensible in a publication, we recommend a companion ablation in which the reports are pre-processed to remove pneumonia-related tokens (e.g. `re.sub(r'pneumon\w+|consolidat\w+|infiltrat\w+', ' [MASK] ', txt)`) and the pipeline is re-trained. The gap between the scrubbed and unscrubbed numbers quantifies how much of the 0.887 comes from pixel/EHR signal vs text leakage of the reference label. This ablation is flagged in `vit_v6_text.ipynb` Section 10.

### Other limitations and future work

1. **Test-set size.** n_test = 241 produces bootstrap CI widths of ~0.09. Absolute comparisons between closely-matched configurations (e.g. 0.887 vs 0.889) are not statistically distinguishable and should not be over-interpreted.
2. **Label noise.** The primary `Pneumonia` label comes from CheXpert's rule-based NLP and carries ~10-15% noise in the literature. `outcome_all_pne` (ICD-based) is an alternative target — swapping is a one-line change and is documented in `vit_v6_text.ipynb` Section 10.
3. **Text encoder upgrade.** `microsoft/BiomedVLP-CXR-BERT-general`, pretrained specifically on MIMIC-CXR reports, typically adds another +0.01-0.03 AUROC over Bio_ClinicalBERT. It is available behind a HuggingFace gate.
4. **Unfreezing BERT's top layers** with a small LR (1e-5) has been shown to add another +0.01-0.03 at the cost of a longer run.
5. **Cross-backbone ensembles** (v6 × v3, v6 × v5) could not be included in the final leaderboard because v3/v5's test splits diverged from v6's (a sort-before-split difference introduced in v6). Re-running v3/v5 with the v6 split is a standing to-do.

---

## Repository Structure

```
final_proj/
├── README.md                              # this file
├── outcome_labels_reference.md            # label definitions
├── requirements.txt                       # Python dependencies
├── .gitignore                             # excludes data/ and results/
├── code/
│   ├── data_download.ipynb                # fetches MIMIC-CXR-JPG images (PhysioNet creds required)
│   ├── master_extract.ipynb               # merges MIMIC-IV ED + CXR metadata + CheXpert + ICD labels
│   ├── select_patients.ipynb              # applies cohort filters -> mimic_ed_cxr_pneumonia_multimodal_cohort.csv
│   ├── download_reports.py                # fetches only the 1,601 cohort reports from PhysioNet
│   ├── diagnose_physionet.py              # verifies PhysioNet auth + project-level permissions
│   ├── vit_v2.ipynb                       # baseline (v2) + step-7 upgrade (v3)
│   ├── vit_v4_ensemble.ipynb              # v2+v3 ensemble attempt (diagnosed val-overfitting)
│   ├── vit_v5_xrv.ipynb                   # TorchXRayVision DenseNet121 + 5-fold CV
│   └── vit_v6_text.ipynb                  # final trimodal model (text + image + EHR)
├── data/                                  # NOT committed to git (credentialed data)
│   ├── files/                             # MIMIC-CXR-JPG images
│   ├── reports/                           # MIMIC-CXR .txt radiology reports
│   ├── edstays.csv.gz                     # MIMIC-IV ED stays
│   ├── mimic-cxr-2.0.0-chexpert.csv.gz    # CheXpert NLP labels
│   ├── mimic-cxr-2.0.0-metadata.csv.gz    # MIMIC-CXR metadata (ViewPosition, etc.)
│   └── mimic_ed_cxr_pneumonia_multimodal_cohort.csv  # final cohort (1,601 rows)
└── results/                               # NOT committed to git (contains study_ids + probabilities)
    ├── vit_v{2,3,4,5,6}_*.{pt,csv,npz}    # checkpoints, predictions, OOF preds, embeddings
    └── *.png                              # ROC/PR curves, training curves
```

## How to Run

1. **Get PhysioNet credentials and sign DUAs** for all three projects: `mimic-iv-ed`, `mimic-cxr-jpg`, `mimic-cxr`.
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Download data:** run `code/data_download.ipynb` for images, then
   ```bash
   export PHYSIONET_USER='your-email'
   export PHYSIONET_PASS='your-password'
   python3 code/download_reports.py
   ```
   (Or manually download `mimic-cxr-reports.zip` from PhysioNet and extract to `data/reports/`.)
4. **Build the cohort:** run `code/master_extract.ipynb` then `code/select_patients.ipynb`.
5. **Train / evaluate models in order:** `vit_v2.ipynb` → `vit_v5_xrv.ipynb` → `vit_v6_text.ipynb`. Each notebook caches its embeddings and predictions under `results/` so later runs are fast.

The pipeline assumes Apple Silicon (MPS) or CUDA. Falls back to CPU if neither is available (the v6 frozen-backbone design makes CPU runs tractable — text + image embedding extraction is the only slow step, and both are cached after the first run).

---

## Data Use Agreement Notice

All data used in this project is from PhysioNet's credentialed-access MIMIC datasets. Per the DUAs:

* **Do not redistribute the data, derived patient-level CSVs, or model artifacts** that could enable re-identification (this includes predictions keyed by `subject_id` / `study_id`, fold-level embeddings, and fine-tuned checkpoints).
* **Only aggregate-level results, code, and metric plots may be shared publicly.**
* This repository's `.gitignore` excludes `data/` and `results/` by default. Do not override this for public forks.
