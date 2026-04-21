"""
Diagnose PhysioNet access: is the 403 about auth, or about project permissions?

Usage:
    export PHYSIONET_USER='your-email'
    export PHYSIONET_PASS='your-NEW-password'
    python3 code/diagnose_physionet.py
"""
import os
import sys

import requests
from requests.auth import HTTPBasicAuth

USER = os.environ.get("PHYSIONET_USER", "")
PASS = os.environ.get("PHYSIONET_PASS", "")
if not (USER and PASS):
    sys.exit("Set PHYSIONET_USER and PHYSIONET_PASS first.")

URLS = {
    # The JPG project you've already been using (should be 200)
    "mimic-cxr-jpg LICENSE": "https://physionet.org/files/mimic-cxr-jpg/2.1.0/LICENSE.txt",
    # The reports project — we want access here
    "mimic-cxr LICENSE":     "https://physionet.org/files/mimic-cxr/2.0.0/LICENSE.txt",
    # An actual report file
    "mimic-cxr report":      "https://physionet.org/files/mimic-cxr/2.0.0/files/p10/p10003019/s50543252.txt",
}

auth = HTTPBasicAuth(USER, PASS)
headers = {"User-Agent": "physionet-access-diagnostic/0.1"}

print(f"Logged in as: {USER}")
print("-" * 70)
for label, url in URLS.items():
    try:
        r = requests.get(url, auth=auth, headers=headers, timeout=20)
        print(f"[{r.status_code}] {label}")
        print(f"      {url}")
        if r.status_code == 200:
            print(f"      content length: {len(r.text)}")
        elif r.status_code == 401:
            print("      -> Bad credentials (auth failed)")
        elif r.status_code == 403:
            print("      -> Authenticated, but NOT authorized for this project.")
            print("         Need to sign DUA at https://physionet.org/content/<project>/")
    except requests.RequestException as e:
        print(f"[ERR] {label}: {e}")
    print()

print("Interpretation:")
print("  If 'mimic-cxr-jpg LICENSE' is 200 but 'mimic-cxr LICENSE' is 403,")
print("  that confirms you need to sign the DUA for the 'mimic-cxr' project")
print("  at https://physionet.org/content/mimic-cxr/2.0.0/")
