"""Offline evidence check for the freestyle submission, V2 edition (runs in CI: personal-evidence.yml).

No warehouse access and no retraining; it never re-runs a V2 pipeline step (the unseal step refuses a second run by design). It checks that:
  1. every receipt the paper, report and notebooks rely on is committed (V1 and V2);
  2. the capstone notebook executes top to bottom from receipts and figures alone (in a temporary copy);
  3. the paper and the report rebuilt from the receipts are byte-identical to the committed files,
     so no number in either can drift from its receipt;
  4. the headline numbers are quoted in the report and the READMEs, and the disclosure sentence appears verbatim where it must;
  5. the V2 frozen state is intact: freeze before test, test scored once, recorded hashes of the frozen code, spec and receipts still match;
  6. the receipts agree with each other (V2 features, selection decision, baselines identical to V1, split windows);
  7. no pseudonymous client/content ID, and no `provider_used` value, appears in any committed receipt, notebook, the paper, the report
     or the archive, and no dataset file is tracked by git;
  8. no em dash in my prose (READMEs, report, paper, notebook cells outside FlyRank's template text);
  9. every submission notebook has executed outputs and no errors, and submission/paper_url.txt is one https line.
Prints ALL CHECKS PASS or exits non-zero.
"""
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.abspath(HERE + "/..")
ROOT = os.path.dirname(W)
sys.path.insert(0, HERE)
import v2_view as VV  # noqa: E402

V1_RECEIPTS = ["eda_ai_monthly.json", "eda_01_landscape.json", "eda_02_signals.json", "eda_03_inference.json", "eda_04_client_level.json",
               "panel_fe.json", "erosion_features.json", "erosion_frame_summary.json", "erosion_model_metrics.json",
               "erosion_model_metrics_sealed.json", "erosion_freeze.json", "erosion_leakage_checks.json", "label_design_log.json",
               "erosion_playbook_summary.json", "baseline_metrics.json", "baseline_top20_review.json", "signal_checks.json", "data_contract.json",
               "sealed_error_profile.json"]
V2_RECEIPTS = ["v2_spec.json", "v2_spec_hash.json", "v2_features.json", "v2_selection.json", "v2_validation_folds.json", "v2_confirmation_apr.json",
               "v2_secondary_label.json", "v2_leakage_checks.json", "v2_freeze.json", "v2_test.json", "v2_playbook_summary.json", "v2_summary.json"]
RECEIPTS = V1_RECEIPTS + V2_RECEIPTS
NOTEBOOKS = ["w01_research_question", "w02_ml_task_framing", "w03_data_contract", "w03_feature_leakage_check", "w04_signal_audit",
             "w04_baseline_score", "w05_model", "w06_validation_audit", "w07_action_playbook", "capstone"]
DISCLOSURE_IN = ["w05_model", "w06_validation_audit", "capstone"]
ID_PATTERN = re.compile(r"\b(?:content|client)_[0-9a-f]{16}\b")
fails = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def rj(n):
    return json.load(open(f"{W}/outputs/{n}", encoding="utf-8"))


def sha_norm(path):
    """SHA-256 with CRLF normalised to LF (the V2 code hashes the same way, because git autocrlf rewrites line endings on Windows)."""
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def squash(s):
    return re.sub(r"\s+", " ", s)


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
        check(True, "capstone.ipynb executes offline from receipts and figures")
except Exception as e:  # noqa: BLE001
    check(False, f"capstone.ipynb executes offline ({type(e).__name__}: {str(e)[:200]})")

# 3. paper and report == rebuild from receipts
import make_paper  # noqa: E402
import make_report  # noqa: E402

with tempfile.TemporaryDirectory() as t:
    for d in ["outputs", "figures", "paper"]:
        shutil.copytree(f"{W}/{d}", f"{t}/{d}")
    make_paper.W = t
    make_paper.build()
    rebuilt = open(f"{t}/paper/index.html", encoding="utf-8").read()
    make_report.W = t
    rep_rebuilt = make_report.build()
committed = open(f"{W}/paper/index.html", encoding="utf-8").read()
rep = open(f"{W}/capstone_report.md", encoding="utf-8").read()
check(rebuilt == committed, "paper is byte-identical to its rebuild from the receipts")
check(rep_rebuilt == rep, "capstone_report.md is identical to its rebuild from the receipts")
check("flyrank.ai" in committed and "Acknowledgments" in committed, "paper carries the FlyRank data credit")
for sid in ["abstract", "problem", "data", "method", "results", "limits", "recs", "repro", "credit"]:
    check(f'id="{sid}"' in committed, f"paper section present: {sid}")
check('href="https://flyrank.ai"' in committed, "paper links https://flyrank.ai")
for h in ["## 0. Abstract", "## 1. Problem framing", "## 2. Data safety", "## 3. Baseline", "## 4. Model / analysis", "## 5. Evaluation",
          "## 6. Interpretation", "## 7. Recommendation", "## 8. Reproducibility", "## 9. Acknowledgments & data credit"]:
    check(h in rep, f"report section present: {h}")

# 4. headline numbers and the disclosure sentence
N = VV.view()
by, may, apr = N["by"], N["by"]["2026-05"], N["by"]["2026-04"]
fe = rj("panel_fe.json")
top3 = next(r for r in fe["pos_1_3"]["effects"] if r["month"] == "2026-04")
c10 = next(r for r in rj("eda_04_client_level.json")["A_client_level"] if r["segment"] == "1-10")
claims = {
    f"{top3['ctr_multiplier']:.3f}": "top-3 fixed-position CTR multiplier",
    f"{top3['ci95'][0]} to {top3['ci95'][1]}": "its 95% CI",
    f"{c10['clients_down']} of {c10['clients']} clients": "clients with lower CTR",
    f"{may['ap']:.3f}": "V2 test AP", f"{VV.ci3(may['ap_ci'])}": "V2 test AP CI", f"{may['p50']:.2f}": "V2 test P@50",
    f"{VV.ci2(may['p50_ci'])}": "V2 test P@50 CI", f"{may['base_rate']:.2f}": "test base rate", f"{may['p50_unm']:.2f}": "test P@50, unmeasurable = miss",
    f"{apr['p50_unm']:.2f}": "confirmation P@50, unmeasurable = miss", f"{may['best_ap']:.3f}": "test best-baseline AP",
    f"{may['v1_ap']:.3f}": "V1 test AP", f"{may['v1_p50']:.2f}": "V1 test P@50", f"{N['closest_lb']:+.4f}": "closest challenger lower bound",
    f"{N['ap_lead_months']} of 4": "months with AP above all baselines",
}
for m in N["months"]:
    claims[f"{m['ap']:.3f}"] = f"V2 AP {m['month']}"
    claims[f"{m['p50']:.2f}"] = f"V2 P@50 {m['month']}"
for k, what in claims.items():
    check(k in rep, f"report quotes {what} = {k}")
readme_root, readme_work = open(f"{ROOT}/README.md", encoding="utf-8").read(), open(f"{W}/README.md", encoding="utf-8").read()
for name, text in [("README.md", readme_root), ("work/README.md", readme_work)]:
    for k in [f"{top3['ctr_multiplier']:.3f}", f"{may['ap']:.3f}", f"{may['p50']:.2f}", f"{may['p50_unm']:.2f}", f"{may['base_rate']:.2f}"]:
        check(k in text, f"{name} quotes {k}")
    check(VV.DISCLOSURE in squash(text), f"{name} carries the disclosure sentence verbatim")
check("not for marking" in readme_work or "not part of the submission" in readme_work, "work/README.md keeps the reviewer note about the archive")
check(VV.DISCLOSURE in squash(committed), "paper carries the disclosure sentence verbatim")
check(squash(committed).count(VV.DISCLOSURE) >= 2, "paper carries the disclosure in Methodology and in Limitations")
check(VV.DISCLOSURE in squash(rep), "report carries the disclosure sentence verbatim")
for n in DISCLOSURE_IN:
    nbj = json.load(open(f"{W}/notebooks/{n}.ipynb", encoding="utf-8"))
    text = squash(" ".join("".join(c["source"]) for c in nbj["cells"] if c["cell_type"] == "markdown"))
    check(VV.DISCLOSURE in text, f"{n}.ipynb carries the disclosure sentence verbatim")
_last = "".join(json.load(open(f"{W}/notebooks/capstone.ipynb", encoding="utf-8"))["cells"][-1]["source"])
check("- [x] Committed to my repo" in _last, "capstone Self-check ticks 'Committed to my repo' (the work is on main)")
check(("- [x] My deployed paper has **all 9 sections**" in _last) or ("- [ ] My deployed paper has **all 9 sections**" in _last), "capstone Self-check carries the 'deployed paper has all 9 sections' box")
_rule_sent = "the editor's rule scores higher than the champion at 50 pages in every month; the champion's queue favours pages that later lose visibility."
check(all(m["rule"]["p@50_unmeasurable_as_miss"] > m["p50_unm"] for m in N["months"]), "under unmeasurable = miss the editor's rule scores higher than the champion in every month (receipts)")
check(_rule_sent in squash(committed) and _rule_sent in squash(rep) and _rule_sent in squash(readme_root) and _rule_sent in squash(readme_work), "the rule-versus-champion sentence is in the paper, report and both READMEs")
check("far less noisy than P@50" in squash(committed) and "far less noisy than P@50" in squash(rep) and "far less noisy than P@50" in squash(readme_root), "the reason AP selects the model is stated in the paper, report and README")
_mot = [m.start() for m in re.finditer("motivated by V1", committed + rep)]
check(all((committed + rep)[max(0, i - 20):i].endswith("not claimed to be ") for i in _mot), "the V2 features are never claimed to be motivated by V1 errors")

# 5. the frozen V2 state is intact (no V2 step is re-run here)
fz, ts = rj("v2_freeze.json"), rj("v2_test.json")
check(fz["frozen_at"] < ts["scored_at"], f"v2_freeze.json ({fz['frozen_at']}) was written before v2_test.json ({ts['scored_at']})")
check(ts["scored_once"] is True and fz["post_freeze_changes"] == [], "test scored once, and no change was logged after the freeze")
for f, h in fz["code_sha256"].items():
    check(sha_norm(f"{HERE}/{f}") == h, f"frozen code unchanged: {f}")
for f, h in fz["spec_sha256"].items():
    check(sha_norm(f"{ROOT}/{f}") == h, f"spec unchanged since the freeze: {f}")
for f, h in fz["receipt_sha256"].items():
    check(sha_norm(f"{W}/outputs/{f}") == h, f"pre-freeze receipt unchanged: {f}")
check(fz["spec_sha256_at_preregistration"] == rj("v2_spec_hash.json")["sha256_normalised_lf"], "pre-registered spec hashes match v2_spec_hash.json")

# 6. receipts agree with each other
feats = rj("v2_features.json")
check(feats["n_features"] == len(feats["features"]) == 36 == feats["n_v1_features"] - len(feats["removed"]) + len(feats["added"]), "36 features = 34 - 3 + 5")
sel = rj("v2_selection.json")
check(sel["selected_model"] == fz["selected_model"] == "LOGISTIC" and not any(v["passes"] for v in sel["challenger_vs_champion"].values()), "no challenger passed; logistic selected")
check(sel["n_challenger_comparisons"] == 3, "three challenger comparisons recorded")
check(N["rule_matches_v1"], "the editor's rule has identical AP and P@50 in the V1 and V2 receipts")
bm = rj("baseline_metrics.json")["per_origin"]
check(all(abs(bm[m["month"]]["p@50"] - m["rule"]["p@50"]) < 1e-9 and abs(bm[m["month"]]["ap"] - m["rule"]["ap"]) < 1e-9 for m in N["months"]),
      "baseline_metrics.json agrees with the V2 receipts for the editor's rule")
tl = rj("v2_leakage_checks.json")
check(tl["timeline_assertions_passed"] and all(v["latest_training_outcome_end_before_origin"] for v in tl["timeline"].values()),
      "every training outcome window ends before the scored origin begins")
check(rj("v2_playbook_summary.json")["queue_size"] == 50 and rj("v2_playbook_summary.json")["max_pages_one_client"] <= 10, "July queue: 50 pages, at most 10 per client")
check(N["ap_lead_months"] == 4, f"champion AP above all four baselines in {N['ap_lead_months']} of 4 months")

# 7. privacy and datasets
targets = (glob.glob(f"{W}/outputs/*.json") + glob.glob(f"{W}/notebooks/*.ipynb") + [f"{W}/paper/index.html", f"{W}/capstone_report.md", f"{W}/README.md", f"{ROOT}/README.md"]
           + glob.glob(f"{W}/archive/**/*.ipynb", recursive=True) + glob.glob(f"{W}/archive/**/*.json", recursive=True) + glob.glob(f"{W}/archive/**/*.md", recursive=True))
leaks = [os.path.relpath(p, ROOT) for p in targets if ID_PATTERN.search(open(p, encoding="utf-8", errors="ignore").read())]
check(not leaks, f"no pseudonymous IDs in committed outputs, notebooks, paper, report, READMEs or the archive {leaks if leaks else ''}")
try:  # widened scan: every tracked text file (the starter notebooks and outputs too), found with git ls-files
    _tracked = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    _text = [f for f in _tracked if f.endswith((".ipynb", ".json", ".md", ".html", ".py", ".txt", ".csv", ".svg", ".css", ".yml", ".toml", ".cfg", ".tex"))]
    _hits = [f for f in _text if os.path.exists(f"{ROOT}/{f}") and ID_PATTERN.search(open(f"{ROOT}/{f}", encoding="utf-8", errors="ignore").read())]
    check(not _hits, f"no pseudonymous IDs in any of {len(_text)} tracked text files {_hits if _hits else ''}")
except Exception:  # noqa: BLE001  (not a git checkout)
    print("SKIP tracked-text ID scan (not a git checkout)")
prov = [os.path.relpath(p, ROOT) for p in targets if re.search(r"---(openai|google|gemini)", open(p, encoding="utf-8", errors="ignore").read(), re.I)]
check(not prov, f"no provider_used values anywhere {prov if prov else ''}")
try:
    tracked = subprocess.run(["git", "-C", ROOT, "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    data = [f for f in tracked if re.search(r"\.(csv|parquet)$", f) and not f.startswith(("data/", "outputs/refresh_queue_sample"))]
    data = [f for f in data if f.startswith("work/")]
    check(not data, f"no CSV or parquet file under work/ is tracked by git {data if data else ''}")
except Exception:  # noqa: BLE001  (not a git checkout)
    print("SKIP tracked-dataset check (not a git checkout)")

# 8. no em dash in my prose
sys.path.insert(0, HERE)
from nb_template import TEMPLATE  # noqa: E402

DASH = (chr(0x2014), chr(0x2013))
frag = []
for t in TEMPLATE.values():
    frag += [p for p in t["title"].split("{BADGE}") if p.strip()] + list(t["sec"]) + [t["selfcheck"]] + t["selfcheck"].split("\n")
frag = sorted({f for f in frag if f.strip()}, key=len, reverse=True)
frag += [f.replace("- [ ]", "- [x]") for f in frag if "- [ ]" in f]
prose = {"README.md": open(f"{ROOT}/README.md", encoding="utf-8").read(), "work/README.md": readme_work, "work/capstone_report.md": rep, "work/paper/index.html": committed}
for n, text in prose.items():
    check(not any(d in text for d in DASH), f"no em or en dash in {n}")
for n in NOTEBOOKS:
    nbj = json.load(open(f"{W}/notebooks/{n}.ipynb", encoding="utf-8"))
    bad = 0
    for c in nbj["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            for f in frag:
                src = src.replace(f, "")
        bad += sum(src.count(d) for d in DASH)
    check(bad == 0, f"no em or en dash in my prose or code of {n}.ipynb outside FlyRank's template text")

# 9. notebooks executed, paper URL
for n in NOTEBOOKS:
    nbj = json.load(open(f"{W}/notebooks/{n}.ipynb", encoding="utf-8"))
    code = [c for c in nbj["cells"] if c["cell_type"] == "code"]
    errs = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    check(all(c.get("execution_count") for c in code) and not errs, f"{n}.ipynb executed with no errors")
url = open(f"{ROOT}/submission/paper_url.txt", encoding="utf-8").read().strip().splitlines()
check(len(url) == 1 and url[0].startswith("https://"), "submission/paper_url.txt is one https line")

print("\nALL CHECKS PASS" if not fails else f"\n{len(fails)} CHECK(S) FAILED")
sys.exit(1 if fails else 0)
