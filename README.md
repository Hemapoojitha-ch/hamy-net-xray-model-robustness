# hamy-net-xray-model-robustness
Deep learning models for chest X-ray analysis often perform well on clean, curated datasets, but their reliability under the kinds of imperfections common in
clinical imaging remains unclear. In real-world settings, chest X-ray images
can be affected by noise, blur, contrast and exposure variations, and compression artifacts, all of which may degrade model performance. In this work, we
evaluate the robustness of several deep learning architectures, including convolutional and transformer-based models, using controlled distortions applied to the
MIMIC-CXR dataset. We assess model performance across multiple distortion
types and severity levels and examine whether simple training strategies, such as
distortion-aware data augmentation, can improve robustness without reducing accuracy on clean images, providing insights toward developing more dependable
systems for clinical deployment.


## Data Processing & Cohort Selection

**1. Data Sources**
* **Clinical Data:** [MIMIC-IV ED (v2.2)](https://physionet.org/content/mimic-iv-ed/2.2/)
* **Imaging Data:** [MIMIC-CXR-JPG (v2.1.0)](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)

**2. Selection Criteria**
The final dataset consists of **1,601 unique patients** (1:2 ratio, 525 Positive : 1076 Negative) selected based on the following logic:
* **Temporal Alignment:** Every X-ray image is strictly timestamped within the specific Emergency Department (ED) visit window (`intime` to `outtime`).
* **View Position:** Only frontal views (**AP** - Anteroposterior or **PA** - Posteroanterior) are included. Lateral views were excluded to maintain anatomical consistency.
* **Deduplication:** We retained only the **first occurrence** of a valid "ED visit + Frontal X-ray" pair for each patient to prevent longitudinal data leakage.
* **Label Clarity:** Only cases with definitive "Pneumonia" labels (1.0 for Positive, 0.0 for Negative) from the CheXpert annotator are included.

**3. Structured Variables (CSV)**
The processed clinical features are stored in `../data/mimic_ed_cxr_pneumonia_multimodal_cohort.csv`.

| Category | Variables |
| :--- | :--- |
| **Identifiers** | `subject_id`, `stay_id`, `study_id`, `dicom_id`, `StudyDate` |
| **Target Labels** | `Pneumonia` (Primary, 32.79%), `outcome_all_pne` (Same with `Pneumonia`, 32.79%), `outcome_bac_pne`(1.87%) , `outcome_viral_pne` (1.00%) |
| **Demographics** | `age`, `gender` (F=1, M=0) |
| **Triage Vitals** | `triage_temperature`, `triage_heartrate`, `triage_resprate`, `triage_o2sat`, `triage_sbp`, `triage_dbp`, `triage_acuity` |
| **Chief Complaints** | `chiefcom_shortness_of_breath`, `chiefcom_cough`, `chiefcom_fever_chills`|
| **History/Scores** | `cci_Pulmonary`, `cci_CHF`, `score_CCI` |

**4. Image Data Structure**
Images are stored locally in `../data/files/p[prefix]/p[subject_id]/s[study_id]/[dicom_id].jpg`



------------------------------------------------------------------------------------------------------------------------

# X-ray Model Robustness

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
