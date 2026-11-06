"""Offline evidence check for the freestyle submission (runs in CI: personal-evidence.yml).

No warehouse access and no retraining. It checks that:
  1. every receipt the paper and report rely on is committed;
  2. the capstone notebook executes top to bottom from receipts alone (in a temporary copy);
  3. the paper rebuilt from the receipts is byte-identical to work/paper/index.html,
     so no number on the page can drift from its receipt;
  4. the headline numbers quoted in work/capstone_report.md match the receipts;
  5. no pseudonymous client/content ID, and no `provider_used` value, appears in any committed
     receipt, notebook output, the paper or the report;
  6. every submission notebook has executed outputs, and submission/paper_url.txt is one https line.
Prints ALL CHECKS PASS or exits non-zero.
"""
import glob
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.abspath(HERE + "/..")
ROOT = os.path.dirname(W)
RECEIPTS = ["eda_ai_monthly.json", "eda_01_landscape.json", "eda_02_signals.json", "eda_03_inference.json", "eda_04_client_level.json",
            "panel_fe.json", "erosion_features.json", "erosion_frame_summary.json", "erosion_model_metrics.json",
            "erosion_model_metrics_sealed.json", "erosion_freeze.json", "erosion_leakage_checks.json", "label_design_log.json",
            "erosion_playbook_summary.json", "baseline_metrics.json", "signal_checks.json", "data_contract.json",
            "sealed_error_profile.json"]
NOTEBOOKS = ["w01_research_question", "w02_ml_task_framing", "w03_data_contract", "w03_feature_leakage_check", "w04_signal_audit",
             "w04_baseline_score", "w05_model", "w06_validation_audit", "w07_action_playbook", "capstone"]
ID_PATTERN = re.compile(r"\b(?:content|client)_[0-9a-f]{16}\b")
fails = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def rj(n):
    return json.load(open(f"{W}/outputs/{n}"))


# 1. receipts
for r in RECEIPTS:
    check(os.path.exists(f"{W}/outputs/{r}"), f"receipt present: {r}")

# 2. capstone executes offline
try:
    import nbformat
    from nbclient import NotebookClient
    with tempfile.TemporaryDirectory() as t:
        for d in ["outputs", "figures", "notebooks"]:
            os.makedirs(f"{t}/work/{d}")
        for f in glob.glob(f"{W}/outputs/*.json"):
            shutil.copy(f, f"{t}/work/outputs/")
        for f in glob.glob(f"{W}/figures/*.png"):
            shutil.copy(f, f"{t}/work/figures/")
        nb = nbformat.read(f"{W}/notebooks/capstone.ipynb", as_version=4)
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": f"{t}/work/notebooks"}}).execute()
        check(True, "capstone.ipynb executes offline from receipts")
except Exception as e:  # noqa: BLE001
    check(False, f"capstone.ipynb executes offline ({type(e).__name__}: {str(e)[:200]})")

# 3. paper == rebuild from receipts
sys.path.insert(0, HERE)
import make_paper  # noqa: E402

with tempfile.TemporaryDirectory() as t:
    for d in ["outputs", "figures", "paper"]:
        shutil.copytree(f"{W}/{d}", f"{t}/{d}")
    make_paper.W = t
    make_paper.build()
    rebuilt = open(f"{t}/paper/index.html", encoding="utf-8").read()
committed = open(f"{W}/paper/index.html", encoding="utf-8").read()
check(rebuilt == committed, "paper is byte-identical to its rebuild from the receipts")
check("flyrank.ai" in committed and "Acknowledgments" in committed, "paper carries the FlyRank data credit")
for sid in ["abstract", "problem", "data", "method", "results", "limits", "recs", "repro", "credit"]:
    check(f'id="{sid}"' in committed, f"paper section present: {sid}")

# 4. report numbers
rep = open(f"{W}/capstone_report.md", encoding="utf-8").read()
fe = rj("panel_fe.json")
top3 = next(r for r in fe["pos_1_3"]["effects"] if r["month"] == "2026-04")
s = rj("erosion_model_metrics_sealed.json")["origins"]["2026-05"]
dev = rj("erosion_model_metrics.json")["origins"]
c10 = next(r for r in rj("eda_04_client_level.json")["A_client_level"] if r["segment"] == "1-10")
lk = rj("erosion_leakage_checks.json")
claims = {
    f"{top3['ctr_multiplier']:.3f}": "top-3 fixed-position CTR multiplier",
    f"{top3['ci95'][0]} to {top3['ci95'][1]}": "its 95% CI",
    f"{c10['clients_down']} of {c10['clients']} clients": "clients with lower CTR",
    f"{s['metrics']['logreg']['p@50']:.2f}": "sealed logistic P@50",
    f"{s['metrics']['logreg']['ap']:.3f}": "sealed logistic PR-AUC",
    f"{s['base_rate']:.2f}": "sealed base rate",
    f"{lk['random_row_kfold']['ap_mean']:.3f}": "random-split PR-AUC",
    f"{lk['client_group_kfold']['ap_mean']:.3f}": "client-held-out PR-AUC",
}
for o, v in dev.items():
    claims[f"{v['metrics']['logreg']['p@50']:.2f}"] = f"dev logistic P@50 {o}"
for k, what in claims.items():
    check(k in rep, f"report quotes {what} = {k}")

# 5. privacy scan
targets = (glob.glob(f"{W}/outputs/*.json") + glob.glob(f"{W}/notebooks/*.ipynb") + [f"{W}/paper/index.html", f"{W}/capstone_report.md"]
           + glob.glob(f"{W}/archive/**/*.ipynb", recursive=True) + glob.glob(f"{W}/archive/**/*.json", recursive=True))
leaks = [os.path.relpath(p, ROOT) for p in targets if ID_PATTERN.search(open(p, encoding="utf-8", errors="ignore").read())]
check(not leaks, f"no pseudonymous IDs in committed outputs, notebooks, paper or report {leaks if leaks else ''}")
prov = [os.path.relpath(p, ROOT) for p in targets if re.search(r"---(openai|google|gemini)", open(p, encoding="utf-8", errors="ignore").read(), re.I)]
check(not prov, f"no provider_used values anywhere {prov if prov else ''}")

# 6. notebooks executed, paper URL
for n in NOTEBOOKS:
    nb = json.load(open(f"{W}/notebooks/{n}.ipynb", encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    errs = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    check(all(c.get("execution_count") for c in code) and not errs, f"{n}.ipynb executed with no errors")
url = open(f"{ROOT}/submission/paper_url.txt").read().strip().splitlines()
check(len(url) == 1 and url[0].startswith("https://"), "submission/paper_url.txt is one https line")

print("\nALL CHECKS PASS" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
