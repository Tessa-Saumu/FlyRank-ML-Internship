"""Opt-in reproduction of the V2 steps that come BEFORE the freeze, run in a scratch copy, never in the repository.

The notebooks call this only when RERUN_PIPELINE is set to True and the pinned warehouse snapshot (with the frame and rhythm caches
under ~/.cache/flyrank_freestyle) is on disk. It copies work/scripts, the JSON receipts and the spec to a temporary folder, runs

    v2_pipeline.py select | confirm | secondary,   v2_checks.py

there (about ten minutes), and compares the headline numbers with the committed receipts. The committed receipts are never touched,
so the freeze gate (which checks their hashes) stays valid.

It deliberately does NOT run `freeze`, `--unseal` or `v2_playbook.py`. The unseal step refuses a second run by design (the test month is
scored once), and the live queue is a frozen artefact identified by its SHA-256.

    python work/scripts/v2_rerun.py --dry-run     builds the scratch copy and prints the plan; runs nothing
    python work/scripts/v2_rerun.py               runs the four steps and prints the comparison
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.abspath(HERE + "/..")
STEPS = [["v2_pipeline.py", "select"], ["v2_pipeline.py", "confirm"], ["v2_pipeline.py", "secondary"], ["v2_checks.py"]]
FORBIDDEN = ("--unseal", "freeze", "v2_playbook.py")
assert not any(tok in step for step in STEPS for tok in FORBIDDEN), "the sealed and live steps are never re-run here"


def scratch_copy():
    tmp = tempfile.mkdtemp(prefix="v2_rerun_")
    shutil.copytree(f"{W}/scripts", f"{tmp}/work/scripts", ignore=shutil.ignore_patterns("__pycache__"))
    os.makedirs(f"{tmp}/work/outputs")
    for f in os.listdir(f"{W}/outputs"):
        if f.endswith(".json"):
            shutil.copy(f"{W}/outputs/{f}", f"{tmp}/work/outputs/{f}")
    shutil.copy(f"{W}/V2_SPEC.md", f"{tmp}/work/V2_SPEC.md")
    return tmp


def _rj(base, name):
    return json.load(open(f"{base}/outputs/{name}", encoding="utf-8"))


def compare(tmp):
    """Headline numbers of the rerun against the committed receipts (max absolute difference of every compared value)."""
    new, old = f"{tmp}/work", W
    s_new, s_old = _rj(new, "v2_selection.json"), _rj(old, "v2_selection.json")
    out = {"selected model equal": s_new["selected_model"] == s_old["selected_model"],
           "best configuration per family equal": all(s_new["best_config_per_family"][k]["cfg"] == s_old["best_config_per_family"][k]["cfg"]
                                                     for k in s_old["best_config_per_family"])}
    d = [abs(s_new["challenger_vs_champion"][k]["mean_gain"] - s_old["challenger_vs_champion"][k]["mean_gain"]) for k in s_old["challenger_vs_champion"]]
    d += [abs(s_new["best_config_per_family"][k]["mean_ap"] - s_old["best_config_per_family"][k]["mean_ap"]) for k in s_old["best_config_per_family"]]
    c_new, c_old = _rj(new, "v2_confirmation_apr.json"), _rj(old, "v2_confirmation_apr.json")
    d += [abs(c_new["evaluation"]["models"][k]["ap"] - c_old["evaluation"]["models"][k]["ap"]) for k in c_old["evaluation"]["models"]]
    out["max absolute difference, AP and gains"] = max(d)
    out["reproduces the committed receipts (within 1e-4)"] = bool(out["selected model equal"] and out["best configuration per family equal"] and max(d) < 1e-4)
    return out


def rerun(dry_run=False):
    tmp = scratch_copy()
    plan = [" ".join(s) for s in STEPS]
    if dry_run:
        return {"scratch copy": tmp, "would run": plan, "note": "nothing was run"}
    env = {**os.environ, "PYTHONUTF8": "1"}
    for step in STEPS:
        r = subprocess.run([sys.executable, step[0]] + step[1:], cwd=f"{tmp}/work/scripts", env=env, capture_output=True, text=True)
        if r.returncode != 0:
            return {"failed step": " ".join(step), "stderr tail": r.stderr[-800:]}
    res = compare(tmp)
    res["scratch copy"] = tmp
    return res


if __name__ == "__main__":
    print(json.dumps(rerun(dry_run="--dry-run" in sys.argv), indent=1, default=str))
