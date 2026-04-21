"""
Download only the MIMIC-CXR reports referenced in our cohort (1601 studies).

Usage (from the project root, after rotating your PhysioNet password):

    export PHYSIONET_USER='jin00330@umn.edu'
    export PHYSIONET_PASS='<your NEW password>'
    python3 code/download_reports.py

Writes results/mimic_cxr_reports_cohort.csv with columns:
    subject_id, study_id, raw_report, sections

v6 notebook picks this file up automatically on the next run.
"""
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

# ---- paths ---------------------------------------------------------------
PROJ = Path(__file__).resolve().parent.parent
DATA = PROJ / "data"
RES = PROJ / "results"
RES.mkdir(parents=True, exist_ok=True)

COHORT_CSV = DATA / "mimic_ed_cxr_pneumonia_multimodal_cohort.csv"
CACHE_CSV = RES / "mimic_cxr_reports_cohort.csv"

# ---- credentials ---------------------------------------------------------
USER = os.environ.get("PHYSIONET_USER", "")
PASS = os.environ.get("PHYSIONET_PASS", "")
if not (USER and PASS):
    sys.exit(
        "Missing credentials. Set PHYSIONET_USER and PHYSIONET_PASS environment "
        "variables first."
    )

BASE = "https://physionet.org/files/mimic-cxr/2.0.0/files"

# ---- section extraction (same rules as v6 notebook) ---------------------
HEADER_RE = re.compile(
    r"^\s*(FINAL REPORT|FINDINGS|IMPRESSION|CONCLUSION|INDICATION|HISTORY|COMPARISON|"
    r"TECHNIQUE|EXAMINATION|REASON FOR EXAM|WET READ|CLINICAL HISTORY|NOTIFICATION)\s*:",
    re.IGNORECASE | re.MULTILINE,
)


def extract_sections(raw_text: str) -> str:
    if not isinstance(raw_text, str) or not raw_text.strip():
        return ""
    matches = list(HEADER_RE.finditer(raw_text))
    if not matches:
        return raw_text.strip()[:2000]
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        sections.setdefault(name, "")
        sections[name] += raw_text[start:end].strip() + "\n"
    ordered = []
    for key in ("IMPRESSION", "FINDINGS"):
        if key in sections:
            ordered.append(f"{key}: " + sections[key].strip())
    if ordered:
        return "\n".join(ordered)[:4000]
    return "\n".join(f"{k}: {v.strip()}" for k, v in sections.items())[:4000]


# ---- fetch loop ----------------------------------------------------------
def main() -> None:
    df = pd.read_csv(COHORT_CSV)
    # resume from existing cache if any
    already = set()
    existing_rows: list[dict] = []
    if CACHE_CSV.exists():
        old = pd.read_csv(CACHE_CSV)
        already = set(old["study_id"].astype(int))
        existing_rows = old.to_dict(orient="records")
        print(f"Resuming from cache: {len(already)} already downloaded")

    todo = df[~df["study_id"].astype(int).isin(already)].reset_index(drop=True)
    print(f"Need to fetch: {len(todo)} / {len(df)}")

    sess = requests.Session()
    sess.auth = HTTPBasicAuth(USER, PASS)
    sess.headers.update({"User-Agent": "mimic-cxr-pneumonia-research/0.1"})

    rows = existing_rows
    failures: list[tuple[int, int, int]] = []
    t0 = time.time()
    save_every = 100

    for i, r in todo.iterrows():
        pid, sid = int(r["subject_id"]), int(r["study_id"])
        url = f"{BASE}/p{str(pid)[:2]}/p{pid}/s{sid}.txt"
        try:
            resp = sess.get(url, timeout=20)
        except requests.RequestException as e:
            failures.append((pid, sid, -1))
            print(f"  [ERR] {sid}: {e}")
            continue
        if resp.status_code != 200:
            failures.append((pid, sid, resp.status_code))
            if resp.status_code == 401:
                sys.exit("401 unauthorized — check PHYSIONET_USER / PHYSIONET_PASS")
            continue
        rows.append(
            {
                "subject_id": pid,
                "study_id": sid,
                "raw_report": resp.text,
                "sections": extract_sections(resp.text),
            }
        )
        if (i + 1) % save_every == 0:
            pd.DataFrame(rows).to_csv(CACHE_CSV, index=False)
            rate = (i + 1) / max(time.time() - t0, 1e-3)
            eta = (len(todo) - i - 1) / max(rate, 1e-3)
            print(
                f"  {i+1}/{len(todo)}  rate={rate:.1f}/s  eta={eta/60:.1f} min  "
                f"failed so far={len(failures)}"
            )

    pd.DataFrame(rows).to_csv(CACHE_CSV, index=False)
    print(f"\nDone. wrote {CACHE_CSV}  ({len(rows)} reports, {len(failures)} failures)")
    if failures:
        fail_csv = RES / "mimic_cxr_reports_failed.csv"
        pd.DataFrame(failures, columns=["subject_id", "study_id", "http_status"]).to_csv(
            fail_csv, index=False
        )
        print(f"Failure list -> {fail_csv}")


if __name__ == "__main__":
    main()
