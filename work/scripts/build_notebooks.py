"""Writes the submission notebooks (template names, one per assignment card), V2 edition.

Every notebook keeps FlyRank's original template structure (title, numbered headings, closing Self-check; see nb_template.py).
Inside each section the analysis follows the same discipline:
  Question -> Why it matters (the FlyRank principle it serves) -> Method -> Result -> What it changes.
Numbers in prose are formatted from the committed receipts in work/outputs/*.json at build time (v2_view.py), so a number
in a markdown cell cannot drift from its receipt. Code cells read the same receipts and do light pandas work; the V2
pipeline itself (about 45 minutes) is never run here unless RERUN_PIPELINE is set to True (opt-in, scratch copy).
Run:  python work/scripts/build_notebooks.py   then execute with run_notebooks.py.
"""
import os

import nbformat as nbf

import v2_view as VV

HERE = os.path.dirname(os.path.abspath(__file__))
NB = HERE + "/../notebooks"
REPO = "Tessa-Saumu/FlyRank-ML-Internship"

# ------------------------------------------------------------------------------------------- numbers for the prose
N = VV.view()
rj = VV.load()
f2, f3, ci2, ci3 = VV.f2, VV.f3, VV.ci2, VV.ci3
DISC = VV.DISCLOSURE
feb, mar, apr, may = (N["by"][m] for m in ("2026-02", "2026-03", "2026-04", "2026-05"))
fe, e4, e3, e1, mon = rj("panel_fe.json"), rj("eda_04_client_level.json"), rj("eda_03_inference.json"), rj("eda_01_landscape.json"), rj("eda_ai_monthly.json")
frame = rj("erosion_frame_summary.json")
summ = rj("v2_summary.json")
top3 = next(r for r in fe["pos_1_3"]["effects"] if r["month"] == "2026-04")
c10 = next(r for r in e4["A_client_level"] if r["segment"] == "1-10")
sel, fz, pb = N["sel"], N["freeze"], N["playbook"]
v1s = N["v1"]["2026-05"]
nulls = [o["null_rate"] for o in frame["origins"] if o["null_rate"] is not None and o["null_rate"] == o["null_rate"]]


def names(pred):
    """Month names for which pred(month dict) is true, e.g. 'Feb, Mar and May'."""
    ms = [VV.MONTH_NAME[m["month"]] for m in N["months"] if pred(m)]
    return ", ".join(ms[:-1]) + " and " + ms[-1] if len(ms) > 1 else (ms[0] if ms else "none")


assert all(m["rule"]["p@50_unmeasurable_as_miss"] > m["p50_unm"] for m in N["months"])
RULE_SENT = ("Counted this way, the editor's rule scores higher than the champion at 50 pages in every month; "
             "the champion's queue favours pages that later lose visibility.")
RULE_UNM = (f"the editor's rule scores {f2(feb['rule']['p@50_unmeasurable_as_miss'])}, {f2(mar['rule']['p@50_unmeasurable_as_miss'])}, "
            f"{f2(apr['rule']['p@50_unmeasurable_as_miss'])} and {f2(may['rule']['p@50_unmeasurable_as_miss'])} in February to May against the champion's "
            f"{f2(feb['p50_unm'])}, {f2(mar['p50_unm'])}, {f2(apr['p50_unm'])} and {f2(may['p50_unm'])}")
AP_WHY = ("Model choice used AP because it scores the whole ranking and is far less noisy than P@50 at about 5 effective clients "
          "(P@50 intervals are roughly plus or minus 0.2); P@50 remains the reported operational metric.")
_c1 = next(r for r in sel["configs"]["LOGISTIC"] if r["cfg"] == {"C": 1.0, "penalty": "l2"})
_c100 = N["penalty_best"]["l2"]
GRID_CLAUSE = (f"The selected C = {_c100['C']:g} sits at the edge of the searched grid, but on a plateau (mean AP {_c1['mean_ap']:.3f} at C = 1 "
               f"against {_c100['mean_ap']:.3f} at C = {_c100['C']:g}), so the edge position is harmless.")
NULLS_ALL = nulls + [m["noise"] for m in N["months"]]
APRIL_CAVEAT = (f"the {names(lambda m: not m['d_ap_excl0']).replace('Apr', 'April')} gain interval includes 0, and this is whole-ranking AP, not P@50")
AP_SENT = (f"V2's logistic champion has the highest AP of all five rankers in {N['ap_lead_months']} of 4 months (Feb, Mar, Apr, May); "
           f"the 95% interval of its AP gain over the best baseline excludes 0 in {N['d_ap_excl0_months']} of 4 ({names(lambda m: m['d_ap_excl0'])}) "
           f"and includes 0 in {names(lambda m: not m['d_ap_excl0'])}")
P50_SENT = (f"Its P@50 is above all four baselines in {N['p50_lead_months']} of 4 months (not {names(lambda m: not m['p50_above_all'])}, where the "
            f"editor's rule scores {f2(mar['rule']['p@50'])} against {f2(mar['p50'])}); the interval of its P@50 gain over the best-AP baseline "
            f"excludes 0 in {N['d_p50_excl0_months']} of 4 months ({names(lambda m: m['d_p50_excl0'])}), so P@50 differences are not established")
UNM_SENT = (f"Counting pages that later lost visibility (unmeasurable pages) as misses, P@50 falls from {f2(may['p50'])} to {f2(may['p50_unm'])} on the "
            f"test month and from {f2(apr['p50'])} to {f2(apr['p50_unm'])} on the confirmation month "
            f"(validation: {f2(feb['p50'])} to {f2(feb['p50_unm'])} and {f2(mar['p50'])} to {f2(mar['p50_unm'])}). Under the same count {RULE_UNM}")
V1V2_SENT = (f"On the test month V2's AP is slightly lower than V1's ({f3(may['ap'])} against {f3(may['v1_ap'])}) and its P@50 higher "
             f"({f2(may['p50'])} against {f2(may['v1_p50'])}). V1's values lie inside V2's 95% intervals (AP {ci3(may['ap_ci'])}; P@50 {ci2(may['p50_ci'])}), "
             f"so the receipts do not show a difference in either direction. V2 is a process upgrade, not a better result")


def badge(name):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/work/notebooks/{name}?flush_cache=true)")


V2_PRELUDE = '''
# V2 receipts (aggregates only) and the helpers every notebook shares.
v2 = lambda n: receipt(f"v2_{n}.json")
BASE = ["rule_position_slipping", "mean_reversion_ctr_excess", "client_shock", "persistence_prev_ratio"]
BASE_LABEL = {"rule_position_slipping": "editor's rule", "mean_reversion_ctr_excess": "CTR above peers",
              "client_shock": "client shock", "persistence_prev_ratio": "persistence"}
def v2_months():
    """(month, role, evaluation block) for the four evaluated months. The logistic champion is in every block."""
    vf, cf, ts = v2("validation_folds"), v2("confirmation_apr"), v2("test")
    return [("2026-02", "validation", vf["folds"]["validation_1"]), ("2026-03", "validation", vf["folds"]["validation_2"]),
            ("2026-04", "confirmation", cf["evaluation"]), ("2026-05", "test", ts["evaluation"])]
def champ(ev):
    return ev["models"].get("LOGISTIC") or ev["models"]["SELECTED:LOGISTIC"]
def fold_table():
    """One row per evaluated month: the champion against the four frozen baselines, same rows, same metric."""
    rows = []
    for m, role, ev in v2_months():
        lg, best = champ(ev), ev["best_baseline_by_ap"]
        rows.append({"month": m, "role": role, "base rate": ev["base_rate"], "rule P@50": ev["models"][BASE[0]]["p@50"],
                     "best baseline (by AP)": BASE_LABEL[best], "best baseline AP": ev["models"][best]["ap"],
                     "V2 AP": lg["ap"], "V2 AP 95% CI": lg["ap_ci"], "V2 P@50": lg["p@50"], "V2 P@50 95% CI": lg["p@50_ci"],
                     "P@50, unmeasurable = miss": lg["p@50_unmeasurable_as_miss"],
                     "rule P@50, unmeasurable = miss": ev["models"][BASE[0]]["p@50_unmeasurable_as_miss"],
                     "rule share of click shortfall (top 50)": ev["models"][BASE[0]]["clicks_lost_captured"],
                     "V2 share of click shortfall (top 50)": lg["clicks_lost_captured"]})
    return pd.DataFrame(rows)'''

SETUP = '''# Setup: works locally and in Colab. Reads committed receipts; the warehouse is touched only by the opt-in rerun.
import importlib.util, json, os, sys
if any(importlib.util.find_spec(p) is None for p in ["duckdb", "sklearn", "scipy", "matplotlib"]):
    get_ipython().run_line_magic("pip", "install -q duckdb huggingface_hub pandas scikit-learn scipy matplotlib")
here = os.path.abspath(os.getcwd())
ROOT = next((p for p in [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
             if os.path.isdir(os.path.join(p, "work", "scripts"))), None)
if ROOT is None:  # fresh Colab runtime: fetch the code (no data lives in git)
    get_ipython().system("git clone -q https://github.com/Tessa-Saumu/FlyRank-ML-Internship.git")
    ROOT = os.path.abspath("FlyRank-ML-Internship")
sys.path.insert(0, os.path.join(ROOT, "work", "scripts"))
OUT, FIG = os.path.join(ROOT, "work", "outputs"), os.path.join(ROOT, "work", "figures")
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from IPython.display import Image, display, Markdown
import wh
RERUN_PIPELINE = False   # opt-in: True recomputes from the pinned snapshot (needs HF_TOKEN). V2 steps then run in a scratch copy, never in the repo.
DATA = wh.data_available(try_download=RERUN_PIPELINE)
print("pinned warehouse snapshot available:", DATA, "| revision", wh.REV[:8], "| RERUN_PIPELINE =", RERUN_PIPELINE)
receipt = lambda name: json.load(open(os.path.join(OUT, name)))
pd.set_option("display.width", 170); pd.set_option("display.max_columns", 30)''' + "\n" + V2_PRELUDE

SETUP_OFFLINE = '''# Offline setup: this notebook reads only committed receipts and figures, so it runs anywhere.
import json, os
here = os.path.abspath(os.getcwd())
ROOT = next(p for p in [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
            if os.path.isdir(os.path.join(p, "work", "outputs")))
OUT, FIG = os.path.join(ROOT, "work", "outputs"), os.path.join(ROOT, "work", "figures")
import pandas as pd
from IPython.display import Image, display
receipt = lambda name: json.load(open(os.path.join(OUT, name)))
pd.set_option("display.width", 170)''' + "\n" + V2_PRELUDE

NOTE = ("> **Freestyle submission, V2 edition.** This notebook belongs to my final, freestyle capstone: *which ranking pages are "
        "losing clicks to AI-answer search before it shows, and which should an editor protect first?* The pipeline was rebuilt as "
        "**V2**: the whole process was written down and hashed in `work/V2_SPEC.md` before any V2 code ran. V1 stays in the repository "
        "as the first process, and every V1 number shown here is labelled V1. My earlier Lane 2 (refresh) experiments are kept for the "
        "record in `work/archive/` and are not part of this submission.")


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, f"{NB}/{name}")


from nb_template import TEMPLATE  # noqa: E402  (FlyRank's original title, section headings and Self-check text)


def head(name, lead=""):
    """FlyRank's title cell (badge pointed at my repo), then my freestyle note and what the card asks."""
    t = TEMPLATE[name]["title"].replace("{BADGE}", badge(name + ".ipynb"))
    return md(t + "\n\n" + NOTE + ("\n\n" + lead.strip("\n") if lead else ""))


def sec(name, i, body=""):
    """FlyRank's numbered heading and italic prompt (verbatim), then my markdown for that section."""
    return md(TEMPLATE[name]["sec"][i - 1] + ("\n\n" + body.strip("\n") if body.strip() else ""))


def selfcheck(name, ticks=(True, True, True, True, True)):
    """FlyRank's Self-check cell, verbatim, with a box ticked only where it is true."""
    out, k = [], 0
    for line in TEMPLATE[name]["selfcheck"].split("\n"):
        if line.startswith("- [ ]"):
            line = ("- [x]" if ticks[k] else "- [ ]") + line[5:]
            k += 1
        out.append(line)
    assert k == len(ticks), (name, k)
    return md("\n".join(out))


# ------------------------------------------------------------------------------------------- ML-02
w01 = [head("w01_research_question", """
**What this card asks:** a question, the unit of analysis, the output, the action, the cost of a wrong call, why data or ML
helps, and two or three real numbers. **The principle it serves:** a strong capstone starts with a useful question, not a
model. The question has to connect the data, a decision, an action and the cost of getting it wrong.
"""), code(SETUP), sec("w01_research_question", 1, """
**Question:** which lane, and why that one? **Why it matters (FlyRank principle: start from a useful question):** a lane
only helps if it connects data, a decision and an action.

I declare the **freestyle** track (lane guide, section 9: bring your own question; same bar as any lane).

I spent July to October on Lane 2 (content refresh: will a page's impressions fall by 20% next month?). The archive
shows what that taught me. My target was largely implied by momentum that was already visible: a two-line acceleration
rule (precision@50 of 0.812) beat every model. More features cannot rescue a target that the past already answers. So
the first requirement for this study was a target that recent momentum does **not** give away (see ML-03 and ML-05).

The second reason is FlyRank's own direction. AI answers in search (Google AI Overviews and AI Mode, ChatGPT,
Perplexity, Gemini) are the new variable in FlyRank's business. FlyRank's *State of AI-Driven SEO 2026* report
measures AI referral sessions, but it never measures what AI answers do to the clicks a page earns from Google at the
rank it already holds. That gap is a question the warehouse can answer.

**Why there are two versions of the pipeline.** V1 compared two models, tuned them inside the training months and opened
one sealed month once. A rigorous pipeline needs more than that: several model families compared under a rule fixed in
advance, a three-way split (validation folds for decisions, a confirmation fold, a test month) and a receipt for every step.
So I wrote V2 down first (`work/V2_SPEC.md`, hashed before any V2 run) and then ran it. The V2 changes are changes of process.
They are not an attempt to raise the score, and the notebooks say so wherever V1 and V2 numbers meet.
"""), code('''
mon = receipt("eda_ai_monthly.json")
print(f"Freestyle track on the full warehouse: {mon['daily_rows']:,} daily rows, {mon['first_date']} to {mon['last_date']}.")
'''), sec("w01_research_question", 2, """
**Question:** which decision does this improve, who acts, and what does a wrong call cost? **Why it matters (FlyRank
principle: decision first):** an unactionable ranking is only a chart.

> **For** FlyRank's content team, **deciding** which ranking pages to work on before their clicks disappear, **I build**
> a monthly ranked queue of 50 top-10 pages from the daily Search Console warehouse, **scoring** the probability that a
> page earns at least 25% fewer clicks over the next two months than its current click-through rate and position
> predict. Success is **measured by** average precision (AP, the pre-registered selection criterion) and precision@50
> against the base rate and against frozen rules, on months the model never saw. **A wrong call costs** an editor's
> review (false alarm) or a silent loss of clicks on a page that still ranks (miss). **A plain rule isn't enough because**
> the obvious rules ("top positions are most exposed", "a slipping CTR keeps slipping") point the wrong way at page level
> (ML-07). **I will claim only** observed, measured, decision-support results: Search Console has no AI-answer flag, so the
> pattern is *consistent with* AI answers absorbing clicks, not proof of it.

| Element | My answer |
|---|---|
| Unit of analysis | one page at one monthly decision point (page x origin month) |
| Output | a ranked queue of 50 pages, each with one reason code and one action |
| Who acts | a content editor, with the account lead for client-wide cases |
| Action | make the page citable in AI answers, rewrite for the click an AI answer cannot satisfy, hold, or escalate |
| Cost of a wrong call | false alarm: about 20 minutes of review; miss: clicks lost on a page that still ranks |
| Why data / ML | erosion depends on rank, recent CTR, demand, intent and the client together, and the mix shifts by month |

**How the question is answered, in three layers.** Each layer must earn its place before the next one:
1. **Does the effect exist?** Inferential statistics with the client as the unit of evidence (ML-06).
2. **Can a page-level target be defined honestly?** An observed label that momentum does not give away, with a measured
   noise floor (ML-03, ML-04, ML-05).
3. **Does a learned ranking beat simple rules out of time?** Two validation folds for every decision, one confirmation fold
   that decides nothing, and a test month scored once (ML-07 to ML-09). Only then do I write a playbook (ML-10).
"""), code('''
pb = v2("playbook_summary")
print({k: pb[k] for k in ["queue_size", "queue_clients", "max_pages_one_client", "reason_codes"]})
'''), sec("w01_research_question", 3, f"""
**Why these three numbers:** each answers a doubt a reviewer should have. Is the problem real? Is it large? Why not
model AI referrals directly? All three come from the full warehouse ({mon['daily_rows'] / 1e6:.1f} million daily rows),
not the starter CSV.
"""), code('''
e4, fe, mon = receipt("eda_04_client_level.json"), receipt("panel_fe.json"), receipt("eda_ai_monthly.json")
top10 = next(r for r in e4["A_client_level"] if r["segment"] == "1-10")
apr = next(r for r in fe["pos_1_3"]["effects"] if r["month"] == "2026-04")
ai_total = sum(r["ai"] or 0 for r in mon["months"])
print(f"1. Same page, same position: CTR fell in {top10['clients_down']} of {top10['clients']} clients between Oct-Nov 2025 and "
      f"Mar-Apr 2026 (median client CTR ratio {top10['median_client_ctr_ratio']}, Wilcoxon p = {top10['wilcoxon_p']}).")
print(f"2. Fixed-effects estimate, top-3 positions: April 2026 CTR = {apr['ctr_multiplier']} x October 2025 "
      f"(95% CI {apr['ci95'][0]} to {apr['ci95'][1]}, {fe['pos_1_3']['clients']} clients, {fe['pos_1_3']['pages']:,} pages).")
print(f"3. AI-assistant referral sessions in the whole warehouse: {ai_total:,.0f}, all from Oct 2025 onward: too thin to be the target.")
'''), md("""
**What these mean for the study.** The phenomenon is real and large (numbers 1 and 2), so a decision tool is worth
building. The direct AI-referral signal is too thin and too concentrated to be the target (number 3; the lane guide warns
against a classifier on AI sessions alone). So the target is the click loss in Google itself, where the data is rich,
and AI referrals enter only as a feature and as a finding (ML-06).
"""), sec("w01_research_question", 4, f"""
**Question:** what may this work say, and what must it never say? **Why it matters (FlyRank principle: careful words):**
observational data supports decisions, not causal stories.

- **Observed:** at a fixed page and position, CTR fell after October 2025.
- **Measured:** on the V2 test month (May 2026, scored once) the champion's AP was {f3(may['ap'])} (95% CI {ci3(may['ap_ci'])}) against
  {f3(may['best_ap'])} for the best frozen baseline, and its P@50 was {f2(may['p50'])} against a base rate of {f2(may['base_rate'])}. With the
  effective number of clients near five, these are wide-interval results (ML-09 also gives the disclosure about the test month).
- **Directional:** transactional pages lost more CTR than informational pages in this panel.
- **Decision-support:** a ranked list that an editor reviews before acting.
- **Never claimed:** that AI answers caused the fall (Search Console has no AI-answer flag), that the model predicts
  Google, that V2 improved on V1, or that an action restores clicks (there is no hold-out yet).
"""), code('''
t = fold_table().set_index("month").loc["2026-05"]
print(f"Measured, V2 test month: AP {t['V2 AP']} {t['V2 AP 95% CI']} vs best baseline {t['best baseline AP']}; "
      f"P@50 {t['V2 P@50']} vs base rate {t['base rate']}.")
print(f"Observed: top-3 CTR in April 2026 = {apr['ctr_multiplier']} x October 2025 (95% CI {apr['ci95'][0]} to {apr['ci95'][1]}).")
'''), selfcheck("w01_research_question")]

# ------------------------------------------------------------------------------------------- ML-03
w02 = [head("w02_ml_task_framing", """
**What this card asks:** task type, target or proxy, success metric, the unit of analysis as a real dataframe, and why
ML beats a fixed rule. **The principle it serves:** decision first. Name the metric before training, and make the
target an observed outcome, never a rule's output.
"""), code(SETUP), sec("w02_ml_task_framing", 1, """
**Question:** which kind of ML task is this? **Why it matters (FlyRank principle: decision first):** the decision is
"which 50 pages first", so the task is a ranking, not a yes or no for every page.

| Item | Choice | Why |
|---|---|---|
| Task type | supervised ranking: binary label, pages ranked by predicted probability | the decision is "which 50 first", not "yes or no for every page" |
| Comparison | four frozen baselines, and four learned candidates compared under one rule fixed in advance (ML-08) | the model must beat simple rules, not just chance |
"""), code('''
summ = receipt("erosion_frame_summary.json")
spec = v2("spec")
print("task: supervised ranking | review budget K =", spec["metrics"]["headline_k"], "| candidates:", spec["candidates"], "| baselines:", len(spec["baselines"]))
'''), sec("w02_ml_task_framing", 2, f"""
**Question:** what is the target, and where does the label come from? **Why it matters (FlyRank principle: an observed
outcome, never a rule's output):** a label defined by a rule can be recovered by that rule, which is what ended my Lane 2
study.

| Item | Choice | Why |
|---|---|---|
| Target | `erode = 1` if clicks in months *o* and *o+1* < 0.75 x expected clicks | observed in a later window; nobody's rule defines it |
| Expected clicks | outcome impressions x current CTR (month *o-1*, shrunk) x position adjustment | compares the page with itself at its own rank |

The question, the unit, the eligibility rule and this label are unchanged from V1. V2 changes how features, models and the
final model are chosen, not what is predicted.

**The timeline (what is known when).**

```
decision on day 1 of month o
features : months o-2 and o-1 (plus o-3 for one trend feature)   -> closed before the decision
label    : months o and o+1                                         -> observed after the decision
training : only origins <= o-2, whose labels have matured by month o-1
```

**The V2 split, one row per role.** The outcome windows of training, validation and test never overlap in the way that
matters: every training label closes before the scored month begins, and the two validation windows end before the test
window opens.

| Role | Scored origin | Outcome window | Trained on origins | Used for decisions |
|---|---|---|---|---|
| Validation fold 1 | 2026-02 | Feb to Mar | 2025-12 | yes |
| Validation fold 2 | 2026-03 | Mar to Apr | 2025-12, 2026-01 | yes |
| Confirmation fold | 2026-04 | Apr to May | 2025-12 to 2026-02 | no, reported only |
| Test (sealed) | 2026-05 | May to Jun | 2025-12 to 2026-03 (refit) | no, scored once |
| Live | 2026-07 | none yet | 2025-12 to 2026-05 | queue only |

The confirmation fold's outcome window (Apr to May) overlaps the test window, which is exactly why it cannot drive any
choice: it is computed after the selection is written to disk and cannot change it.

**Three guards that make the label honest.**

| Guard | What it does | Why it is needed |
|---|---|---|
| Shrinkage (empirical Bayes, m = 1,000 impressions) | pulls month *o-1* CTR toward the client's position-adjusted CTR | at about 0.4% CTR, a page with 20 clicks has a noisy rate; shrinkage stops noise from posing as erosion |
| Fixed position curve (Oct-Nov 2025 only) | adjusts the expectation if rank moves | a page that drops rank loses clicks for a known reason; that is not AI-answer erosion |
| Poisson noise floor | simulates the label on pure sampling noise | shows how much of the base rate chance alone would produce |

The label anchors on the **latest** month, shrunk toward the **client**, not toward the page's own history. Two earlier
versions leaked momentum into the label. ML-05 (`w03_feature_leakage_check`) shows how I caught them.
"""), code('''
o = pd.DataFrame(summ["origins"]).dropna(subset=["base_rate"])
o["signal_over_noise"] = (o.base_rate / o.null_rate).round(1)
display(o[["origin", "measurable", "clients", "base_rate", "null_rate", "signal_over_noise"]])
tl = v2("leakage_checks")["timeline"]
display(pd.DataFrame([{"role": k, "scored origin": v["score_origin"], "trained on": ", ".join(v["train_outcome_windows_end"]),
                       "last training outcome day": max(v["train_outcome_windows_end"].values()), "origin first day": v["origin_first_day"],
                       "training ends before origin": v["latest_training_outcome_end_before_origin"]} for k, v in tl.items()]))
'''), md(f"""
**What it means.** Pure noise would label about 5% of pages as eroding ({min(NULLS_ALL):.3f} to {max(NULLS_ALL):.3f} across the training, validation, confirmation and test months). The observed
rate is {f2(N['base_rate_range'][0])} to {f2(N['base_rate_range'][1])} in the four evaluated months, several times higher, so most positives are real changes in click
share at a stable rank, not counting noise. The second table is the split above, read from the V2 leakage receipt: in every
row the last training outcome day is before the origin's first day.
"""), sec("w02_ml_task_framing", 3, f"""
**Question:** which one number means "good"? **Why it matters (FlyRank principle: name the metric before training):** a
metric chosen after seeing results can be bent to flatter the model.

| Item | Choice | Why |
|---|---|---|
| Selection criterion | average precision (AP) over the two validation folds, fixed in the spec before any run | one pre-declared number, so the winner is not picked by whichever metric flatters it |
| Headline, reported next to the base rate | precision@50, with client-bootstrap 95% intervals | 50 is the monthly review budget; P@50 is the operational metric |
| Also reported, never used to choose | P@20, P@100, AUC, lift, within-client P@10, clicks-lost captured, top-client share of the top 50 | P@50 alone has wide intervals (about +/-0.2 at K = 50) |
| Conservative variant | P@50 counting unmeasurable pages as misses | pages that lose almost all visibility cannot be labelled; counting them as misses is the cautious bound |
"""), code('''
b = pd.DataFrame([{"month": m, "role": r, "base rate": ev["base_rate"], "noise-only rate": ev["noise_only_rate"],
                   "eligible pages": ev["n_eligible"], "measurable": ev["n_measurable"], "lost-visibility share": ev["lost_visibility_share"],
                   "clients": ev["n_clients"], "effective clients": ev["effective_clients"]} for m, r, ev in v2_months()])
print("Every metric below is read next to the month's base rate, and with the effective number of clients:"); display(b)
'''), md(f"""
**What it means.** The effective number of clients (1 divided by the sum of squared row shares) is {f2(N['eff_range'][0])} to {f2(N['eff_range'][1])} in the
four evaluated months, although there are 20 to 30 clients. The top three clients hold {min(m['top3_share'] for m in N['months']) * 100:.0f}% to
{max(m['top3_share'] for m in N['months']) * 100:.0f}% of the rows. So every interval is built by resampling whole clients, and the
effective sample size is a limit on what can be claimed, not a detail.
"""), sec("w02_ml_task_framing", 4, """
**Question:** one row is one what? **Why it matters (FlyRank principle: one explicit grain):** a reviewer should see
exactly what one row is. IDs are pseudonymous and used only for grouping, so they are not shown. The frame is built by
`work/scripts/erosion_frame.py` from the cached page-month table.
"""), code('''
summ = receipt("erosion_frame_summary.json")
display(pd.DataFrame(summ["origins"]))
if DATA and os.path.exists(wh.CACHE + "/erosion_frame.parquet"):
    import pyarrow.parquet as pq   # read only the validation month: the sealed test month is never loaded here
    cols = ["origin", "imp_f", "clk_f", "pos_f", "pos_m1", "ctr_now", "imp_out", "clk_out", "pos_out", "exp_out", "erode"]
    f = pq.read_table(wh.CACHE + "/erosion_frame.parquet", columns=cols, filters=[("origin", "==", "2026-03")]).to_pandas()
    display(f.sample(6, random_state=1).round(4)); print("rows in the 2026-03 slice:", len(f))
'''), md("""
**Reading the table.** Each row is one top-10 page at one decision month, with at least 200 impressions and 20 clicks
in the two feature months. `exp_out` is what the page should earn in the next two months if its current CTR held at its
new position. `erode` compares what it actually earned against that expectation. The test origin (2026-05) and the
live origin (2026-07) have no base rate in the V1 frame receipt: the test month's outcomes are kept out of every V2 load
until the unseal step, and the live month has no outcome yet. The `lost_visibility` column is the share of eligible pages
that are unmeasurable (expected outcome clicks under 10), which matters again in ML-09.
"""), sec("w02_ml_task_framing", 5, f"""
**Question:** why not a fixed rule? **Why it matters (FlyRank principle: the solution hierarchy):** a model must earn its
place over a transparent rule.

- **Erosion is a combination, not one signal.** Rank, a recent CTR bump, CTR above what the position normally earns,
  demand swings and client-wide shifts all matter together (ML-06, ML-08).
- **The intuitive rules point the wrong way.** In the training months, top-3 pages eroded *less* than positions 7 to 10,
  and pages whose CTR had just risen eroded *more* than pages already slipping (ML-07).
- **No single rule holds every month.** The editor's rule scores P@50 of {f2(mar['rule']['p@50'])} in March and {f2(may['rule']['p@50'])} on the test month,
  and the best baseline by AP changes between months (table below).

The output feeds a real content action. ML-10 maps each reason code to an editor action, with human review before any
change.
"""), code('''
t = pd.DataFrame([{"month": m, "role": r, "best baseline by AP": BASE_LABEL[ev["best_baseline_by_ap"]],
                   **{f"{BASE_LABEL[b]} P@50": ev["models"][b]["p@50"] for b in BASE}} for m, r, ev in v2_months()])
display(t)
'''), selfcheck("w02_ml_task_framing")]


# ------------------------------------------------------------------------------------------- ML-04
snap = N["feats"]["snapshot_diagnostics_dim_content"]
upd = [d["share_content_updated_on_or_after_origin"] for d in snap.values()]
demo = summ["contract_leak_demo"]
f36 = N["feats"]["features"]
w03 = [head("w03_data_contract", """
**What this card asks:** five plain-word contract answers, exactly three verification queries on a mid-panel month, at
most five features each with "knowable at the decision moment because...", and a deliberate label-derived leak shown,
then removed. **The principle it serves:** data hygiene first. One explicit grain, null is not zero, and nothing from
the future in a feature.
"""), code(SETUP), sec("w03_data_contract", 1, """
**Question:** one row is one what, over which dates? **Why it matters (FlyRank principle: one explicit grain):** every
count, join and label below depends on it. **Method:** five plain-word answers, then queries to verify them in section 3.

1. **What is one row?** One top-10 page at one monthly decision point (page x origin month), built from the daily
   performance fact table (report_date x client x content).
2. **Which tables?** `fact_content_daily_performance` (Search Console and GA4, daily), `dim_content` (page
   metadata) and `dim_clients` (data start dates). Warehouse build v20260703, pinned revision `50cbf7c3`.
3. **Which window?** Features use the two closed months before the decision. The label uses the two months after it.
   Origins run December 2025 to May 2026, after Google's September 2025 change that removed deep-result impressions,
   plus a live origin in July 2026 that has no outcome yet.
4. **Label or proxy?** Observed: clicks in the outcome window below 75% of the clicks the page's current CTR predicts at
   its new position.
5. **One exclusion.** Pages whose expected outcome clicks are under 10 (4% to 16% per evaluated origin, mostly pages that lost
   visibility) are excluded because their label cannot be measured. This uses outcome-window impressions, so it is a
   disclosed population choice, and ML-09 reports what it hides.
"""), code('''
summ = receipt("erosion_frame_summary.json")
display(pd.DataFrame(summ["origins"])[["origin", "eligible", "measurable", "clients", "lost_visibility"]])
'''), sec("w03_data_contract", 2, f"""
**Question:** which fields do I touch, and in which bucket? **Why it matters (FlyRank principle: nothing from the future
in a feature):** a field in the wrong bucket is how leaks start.

| Bucket | Fields | Note |
|---|---|---|
| **Feature** | the five core features below, plus the {N['n_features']} features of the V2 model ({N['n_v1_features'] - 3} carried from V1, {N['n_new']} new date and rhythm features) | all computed from months before the decision; list exported to `v2_features.json` |
| **Label** | `erode` (primary); `erode_rel` (secondary, client-relative, never used to choose a model) | observed in the two months after the decision |
| **Context** | `origin` (decision month), `client_hash_id` (grouping and splits only), `content_hash_id` (joins and tie-breaks only) | pseudonymous; never displayed or used as a feature |
| **Excluded** | `clk_out`, `imp_out`, `pos_out`, `exp_out` (outcome-window columns, the label's own ingredients); `provider_used`; the 90-day query table; `model_used`; `last_optimized_date`; and three V1 features, `page_prev_ratio`, `client_prev_erode_rate`, `word_count` | why: see below |

**Why each exclusion.**
- **Outcome-window columns** are the label's ingredients, so as features they would be the answer (the leak demo in
  section 3 shows it).
- **`provider_used`** contains strings that look like client brand names in some rows. It is never selected anywhere in
  this project (privacy rule).
- **The 90-day query table** covers April to June 2026, which overlaps every outcome window.
- **`model_used`** records the latest regeneration of a page, not its author, so it is not a valid treatment.
- **`last_optimized_date`** covers 8.7% of pages, all dated after 24 April 2026, which is too little follow-up time.
- **`page_prev_ratio` and `client_prev_erode_rate`** (V1 features) are empty in the earliest training rows, because no origin
  exists four months back, and the label anti-persists. `page_prev_ratio` survives only as the input of the frozen persistence
  baseline.
- **`word_count`** comes from `dim_content`, a July 2026 snapshot. {min(upd) * 100:.0f}% to {max(upd) * 100:.1f}% of eligible pages were updated on or after
  each origin, so the word count may describe a later regeneration, not the page at the decision moment.

**Five core features, each knowable at the decision moment** (the model uses {N['n_features']}, ML-08):

| Feature | Meaning | Knowable at the decision moment because... |
|---|---|---|
| `l_imp_f` | log impressions over months *o-2* and *o-1* | both months close before day 1 of month *o* |
| `pos_f` | impression-weighted average position, same window | same window, computed from `gsc_sum_position / impressions` |
| `ctr_vs_ref` | log of shrunk CTR over the fixed position curve | the curve uses Oct-Nov 2025 only, before every outcome window |
| `ctr_growth` | log CTR change from month *o-2* to month *o-1* | both months are closed |
| `days_since_last_click` | days from the last click day in the window to the origin's first day (new in V2) | read from daily rows inside months *o-2* and *o-1* only; the latest daily date read for each origin is asserted to be before its first day |
"""), code('''
feats = v2("features")
names_ = [f["name"] for f in feats["features"]]
five = ["l_imp_f", "pos_f", "ctr_vs_ref", "ctr_growth", "days_since_last_click"]
print(len(names_), "features exported by the V2 code;", feats["n_v1_features"], "in V1,", len(feats["removed"]), "removed,", len(feats["added"]), "added")
print("the five core features are in the exported feature list:", {f: f in names_ for f in five})
print("label ingredients or removed V1 features present in the feature list (must be empty):",
      [c for c in ["imp_out", "clk_out", "pos_out", "exp_out", "erode", "log_ratio_out", "page_prev_ratio", "client_prev_erode_rate", "word_count"] if c in names_])
print("latest daily date read for the rhythm features, per origin (all before the origin's first day):")
display(pd.DataFrame(feats["timeline_check_rhythm"]).T[["max_daily_date_used", "origin_first_day"]])
'''), sec("w03_data_contract", 3, """
**Question:** does the contract hold on the real data? **Why it matters (FlyRank principle: a claim without a query is a
guess):** three checks prove the grain, the size and date span, and the availability handling before anything is built on
them. **Method:** exactly three queries on a mid-panel month, March 2026, away from the final months. With the pinned
snapshot on disk the queries run live and are compared with the committed receipt; without it the cell reads the receipt.
"""), code('''
contract = receipt("data_contract.json")
if DATA:
    c = wh.con()
    d = wh.DAILY_DIR + "/month=2026-03/*.parquet"
    q1 = c.sql(f\'\'\'SELECT report_date, content_hash_id, COUNT(*) n FROM read_parquet('{d}')
                   GROUP BY 1, 2 HAVING COUNT(*) > 1 LIMIT 5\'\'\').df()
    q2 = c.sql(f\'\'\'SELECT COUNT(*) rows_, MIN(report_date)::VARCHAR first_day, MAX(report_date)::VARCHAR last_day,
                   COUNT(DISTINCT client_hash_id) clients, COUNT(DISTINCT content_hash_id) pages FROM read_parquet('{d}')\'\'\').df()
    q3 = c.sql(f\'\'\'SELECT SUM((gsc_data_available IS TRUE)::INT) gsc_true, SUM((gsc_data_available IS NOT TRUE)::INT) gsc_not_true,
                   SUM((gsc_data_available IS NULL)::INT) gsc_null, SUM((ga4_data_available IS TRUE)::INT) ga4_true,
                   SUM((ga4_data_available IS NULL)::INT) ga4_null FROM read_parquet('{d}')\'\'\').df()
    live = {"q1_duplicate_grain_rows": len(q1), "q2": json.loads(q2.to_json(orient="records"))[0], "q3": json.loads(q3.to_json(orient="records"))[0]}
    print("queries re-run live on the pinned snapshot; identical to the committed receipt:", all(live[k] == contract[k] for k in live))
    contract.update(live)    # shown below, not written back
print("Q1 grain probe, duplicate (report_date, content) rows:", contract["q1_duplicate_grain_rows"], "(must be 0)")
print("Q2 size and span:", contract["q2"])
print("Q3 availability flags (three-valued, so filter with IS TRUE):", contract["q3"])
'''), md(f"""
**What it means.** The grain holds (zero duplicates), March has the expected one-month span, and the availability flags
contain NULLs as well as TRUE and FALSE. That is why every query in this project filters with `IS TRUE`, and why GA4
columns are summed only on days where `ga4_data_available IS TRUE` (earlier days are zero-filled, and zero there does
not mean "no engagement").

**A deliberate leak, shown and then removed** (a model check, not a fourth query). **Why:** to prove the evaluation can
see a leak. **Method:** train a logistic model on the five core features with the V2 split of validation fold 2 (train
December and January, score March). Then add the outcome-window clicks as a "feature" (the label's own ingredient) and
score the same month again. The receipt is `v2_summary.json`, written by `work/scripts/v2_summary.py`.
"""), code('''
if RERUN_PIPELINE and DATA:
    import v2_summary; v2_summary.main()     # about a minute; rewrites v2_summary.json only
demo = receipt("v2_summary.json")["contract_leak_demo"]
display(pd.DataFrame({k: demo[k] for k in ["five_core_features", "with_leak_clk_out"]}).T)
fold2 = v2("validation_folds")["folds"]["validation_2"]
print(f"base rate in {demo['score_origin']}: {demo['base_rate_2026-03']} | the full 36-feature champion on the same fold: "
      f"AP {fold2['models']['LOGISTIC']['ap']}, P@50 {fold2['models']['LOGISTIC']['p@50']}")
'''), md(f"""
**What it means.** With the outcome's own ingredient as a feature, the ranking becomes perfect (P@50 {demo['with_leak_clk_out']['p@50']:.2f}, AP
{demo['with_leak_clk_out']['pr_auc']:.2f}). That is the signature of a leak, not skill. With it removed, the five core features alone reach P@50
{demo['five_core_features']['p@50']:.2f} and AP {demo['five_core_features']['pr_auc']:.3f} against a base rate of {demo['base_rate_2026-03']:.3f}, while the
full {N['n_features']}-feature champion reaches AP {f3(mar['ap'])} and P@50 {f2(mar['p50'])} on the same fold. The core idea is real but thin on its
own, which is why the study engineers more features. The leaky column is never used again. Feature timing is
asserted in code: `erosion_frame.py` builds features only from months before the origin, and the V2 rhythm code asserts the latest daily
date it reads (table above).
"""), sec("w03_data_contract", 4, f"""
**Question:** what can this data never tell me? **Why it matters (FlyRank principle: careful words):** a limit stated up
front keeps a claim from outrunning its data.

- **No AI-answer flag.** Search Console does not record whether an AI answer appeared for a query. So this study
  measures click loss at a stable rank, which is **consistent with** AI answers absorbing clicks. Seasonality, changes
  to other search features and impression inflation are competing explanations. ML-06 tests the ones the data allows.
- **An unbalanced panel, and about five effective clients.** Search Console clients grow from 2 to 65 over 17 months, so any
  pooled trend mixes in onboarding. Within an evaluated month the top three clients hold {min(m['top3_share'] for m in N['months']) * 100:.0f}% to
  {max(m['top3_share'] for m in N['months']) * 100:.0f}% of the rows and the effective number of clients is {f2(N['eff_range'][0])} to {f2(N['eff_range'][1])}.
  Trends are estimated within page or within client, and every interval resamples clients.
- **GA4 starts late.** GA4 (and with it AI-referral tracking) starts in October 2025 with 4 clients. Earlier days are
  zero-filled, so zero there does not mean "no engagement".
- **A measurement change.** Google's September 2025 removal of deep-result impressions shifts impressions toward the top
  10, so every window starts in October 2025 or later.
- **A selected population.** Only top-10 pages with enough clicks, and only pages whose outcome is measurable ({f2(N['unm_range'][0] * 100)}% to
  {f2(N['unm_range'][1] * 100)}% dropped per evaluated month), so the results describe pages that still rank.
- **Snapshot attributes.** `search_volume`, `cpc`, `competition`, `main_intent`, `content_type` and the keyword attributes are as of
  July 2026, not as of each origin. `keyword_created_date` is checked in code to be on or before the origin (zero violations in
  every origin, `v2_features.json`).
- **One warehouse revision, and the data ends on 2026-06-30.** There is no more data to collect a fresh holdout from.
"""), code('''
mon = pd.DataFrame(receipt("eda_ai_monthly.json")["months"])
display(mon[["month", "gsc_clients", "ga4_clients"]])
snap = pd.DataFrame(v2("features")["snapshot_diagnostics_dim_content"]).T
display(snap[["eligible_pages", "share_content_updated_on_or_after_origin", "keyword_created_after_origin_count", "keyword_created_missing_share"]])
'''), selfcheck("w03_data_contract")]

# ------------------------------------------------------------------------------------------- ML-05
log = rj("label_design_log.json")
lk1 = rj("erosion_leakage_checks.json")
pl, rvg = N["planted"], N["rvg"]
w03b = [head("w03_feature_leakage_check", """
**The principle it serves:** attack your own model before anyone else does. A score that looks too good is
investigated, not celebrated. This notebook builds the feature vector, documents each feature, and records the attacks
on the label (V1) and on the V2 features and splits.
"""), code(SETUP), sec("w03_feature_leakage_check", 1, f"""
**Question:** what exactly goes into the model? **Why it matters (FlyRank principle: every feature must exist before the
moment I predict):** the vector is the only thing the model sees. **Method:** one function builds the {N['n_features']} V2 features per
page and origin from months *o-3* to *o-1* ({N['n_v1_features']} from V1, minus 3, plus {N['n_new']} date and rhythm features). The list, a
one-line meaning and a family per feature are exported from code (`v2_features.json`), so this notebook, the paper and the
model share it. The five new features come from a mentor's advice that a date column yields many derived features. They are
not claimed to be motivated by V1's errors.
"""), code('''
feats = v2("features")
t = pd.DataFrame(feats["features"])
print(f"{feats['n_features']} features exported by the V2 code | removed from V1: {feats['removed']} | added: {feats['added']}")
display(t.groupby("family").size().to_frame("features"))
display(t[["name", "family", "meaning", "new"]])
if RERUN_PIPELINE and DATA:
    import v2_pipeline as p
    D = p.Data("sealed")      # sealed mode: test-month outcomes dropped, live origin removed
    print("feature matrix:", D.X.shape, "| columns match the exported list:", list(D.X.columns) == t.name.tolist())
'''), sec("w03_feature_leakage_check", 2, """
**Question:** for each feature, what does it mean, how are missing values handled, which are categorical, and does it
exist before the decision? **Why it matters (FlyRank principle: null is not zero):** a silent fill can fake a signal.

- **Available when:** every feature is computed from months *o-3* to *o-1*. The five new features are read from the daily table
  strictly inside months *o-2* and *o-1*, one month at a time, and the latest daily date read is asserted to be before the
  origin's first day (table below). V1's two matured-label features are gone.
- **Missing values.** Missingness is structural (no GA4 for the client, no keyword date). The logistic model fills with the
  median and adds a missing-indicator column; gradient boosting handles missing values natively. `l_org_sess` is NaN, not
  zero, when no GA4 day exists, and `has_ga4` carries that fact. `imp_growth_prev` is set to missing for origin 2025-12 only,
  because its third month (September 2025) straddles Google's `num=100` change. A feature that is entirely missing in a fold's
  training rows is dropped for that fold and listed in the receipt (`imp_growth_prev` in validation fold 1).
- **Rhythm edge cases.** `longest_zero_click_run` is computed per month and combined by maximum, so a run that crosses the
  month boundary is not joined (a disclosed under-count). A page absent from one month gets that month's full length as its run.
- **Categorical.** Main intent is one-hot (`intent_informational`, `intent_transactional`, `intent_commercial`,
  `intent_navigational`), and content type is a flag (`is_feedly`).
- **Left out of boosted fits only.** scikit-learn 1.9's histogram binner raises an error on a column with fewer than two distinct
  non-missing values. In a boosted fit such a column is left out of that fit only (it cannot be split on anyway). This is the one
  logged implementation deviation in `work/V2_SPEC.md`, section 13.
"""), code('''
feats = v2("features")
share = pd.DataFrame(feats["non_missing_share_of_new_features_by_origin"]).T
print("share of eligible pages with a value, new features, by origin:"); display(share)
print("rows missing, new features, by origin:"); display(pd.DataFrame({o: v["n_missing"] for o, v in feats["timeline_check_rhythm"].items()}).T)
'''), sec("w03_feature_leakage_check", 3, f"""
**Question:** can the label be predicted from momentum that is already visible, and does my harness see a planted leak?
**Why it matters:** that failure ended my Lane 2 study (a two-line momentum rule beat every model). **Method:** four attacks.
The label attacks (1) were run in V1 on the development months and the label is unchanged in V2. The V2 attacks (2 to 4)
use the validation folds only.

**Attack 1 (V1 history): the label leaked twice, and the models looked better while it did.**
"""), code('''
log = receipt("label_design_log.json")
rows = []
for v, d in log["versions"].items():
    if "p50" in d:
        for m, vals in d["p50"].items():
            rows.append({"version": v, "ranker (V1)": m, **dict(zip(["2026-02", "2026-03", "2026-04"], vals))})
dev = receipt("erosion_model_metrics.json")["origins"]
for m in ["rule_position_slipping", "logreg", "hgb"]:
    rows.append({"version": "v3_final", "ranker (V1)": m, **{o: dev[o]["metrics"][m]["p@50"] for o in dev}})
display(pd.DataFrame(rows))
for v, d in log["versions"].items():
    print(f"{v}: {d.get('problem_found', 'final version')}")
'''), md("""
**What it means.** The leaky versions made the learned models look much better (gradient boosting reached 0.90 at 50
pages). After the fix, every model score dropped. That drop is the evidence that the leak was real, and the honest
lower label is the one V2 inherits.

**Attack 2 (V2): does the harness see a planted leak?** **Method:** on validation fold 2, add a noisy copy of the outcome
ratio (plus N(0, 0.3)) as a feature. If AP and P@50 do not jump, the harness itself is broken.
"""), code('''
lk = v2("leakage_checks")["planted_future_feature_fold_2"]
display(pd.DataFrame({k: v for k, v in lk.items() if isinstance(v, dict)}).T)
print({k: lk[k] for k in ["LOGISTIC_ap_lift", "LOGISTIC_p@50_lift", "HGB_CLF_ap_lift", "HGB_CLF_p@50_lift", "harness_detects_leak", "honest_models_not_leaky"]})
'''), md(f"""
**What it means.** The planted feature lifts the logistic AP from {pl['LOGISTIC_honest']['ap']:.3f} to {pl['LOGISTIC_planted_future_ratio']['ap']:.3f} (+{pl['LOGISTIC_ap_lift']:.2f}) and its P@50
from {pl['LOGISTIC_honest']['p@50']:.2f} to {pl['LOGISTIC_planted_future_ratio']['p@50']:.2f}. The boosted classifier moves the same way. So the harness can see a leak, and the honest models
sit far below the planted ones. The honest models have no such feature.

**Attack 3: does the split decide the answer?** **Question:** how much does a random row split flatter the model?
**Why it matters:** pages from one client share hidden character, so a random split lets the model recognise the client
instead of learning erosion. **Method:** the selected logistic configuration on the February and March rows,
5-fold with random rows against 5-fold with whole clients held out.
"""), code('''
rv = v2("leakage_checks")["random_vs_client_grouped_feb_mar"]
print({k: rv[k] for k in ["base_rate", "rows", "random_minus_grouped_ap"]}); print(rv["reading"])
display(pd.DataFrame({k: rv[k] for k in ["random_row_kfold", "client_group_kfold"]}).T)
v1 = receipt("erosion_leakage_checks.json")
print("V1 history, same idea on V1 features:", {k: v1[k]["ap_mean"] for k in ["random_row_kfold", "client_group_kfold"]})
display(Image(os.path.join(FIG, "fig3_validation_checks.png")))
'''), md(f"""
**What it means.** Random rows report AP {rvg['random_row_kfold']['ap_mean']:.3f}. Holding whole clients out gives {rvg['client_group_kfold']['ap_mean']:.3f}
(base rate {rvg['base_rate']:.3f}). The gap is {rvg['random_minus_grouped_ap']:.3f}, smaller than the V1 gap of
{lk1['random_row_kfold']['ap_mean'] - lk1['client_group_kfold']['ap_mean']:.3f} ({lk1['random_row_kfold']['ap_mean']:.3f} against {lk1['client_group_kfold']['ap_mean']:.3f}), but the two checks use different rows and features, so I do not
read the difference as an improvement. Both say the same thing: a random split flatters. Every claim in this submission
uses forward months and clients are resampled whole in every interval.

**Attack 4: timeline and snapshot assertions.** Every feature window ends before the origin's first day, and every training
origin's outcome window ends before the scored origin begins. Both are asserted in code and recorded for all five roles in
`v2_leakage_checks.json` (`timeline_assertions_passed`: {str(N['leak']['timeline_assertions_passed']).lower()}). The snapshot attributes are checked too:
`keyword_created_date` is never after the origin, and `word_count` was removed because {min(upd) * 100:.0f}% to {max(upd) * 100:.1f}% of pages were updated after each origin.

**Attack checklist.**

| Check | Status | Where |
|---|---|---|
| Timeline drawn: all features strictly before the label window | done; asserted in code for all five roles | ML-03, `v2_leakage_checks.json` |
| No label-derived or sibling columns in the features | done; planted-leak harness lifts AP by {pl['LOGISTIC_ap_lift']:.2f} | attack 2 here |
| No product flags or existing-system scores as features | done; the warehouse release has none, `provider_used` excluded | ML-04 |
| Snapshot attributes dated after the origin | removed `word_count`; keyword dates checked | attack 4 |
| Population selection checked for outcome-window information | disclosed: measurability uses outcome impressions; unmeasurable pages reported (ML-09) | ML-04, ML-09 |
| Split grouped by the repeating entity and by time | forward origins, clients resampled whole, random-vs-grouped check | ML-08, ML-09 |
| Base rate printed next to every metric | done | all V2 receipts |
| Tuning and the champion test use the same two folds | disclosed; the confirmation fold and the test month are the less-biased reads | ML-08 |
| Sealed holdout leaves receipts | `v2_freeze.json` before `v2_test.json`; scored once | ML-09 |
"""), sec("w03_feature_leakage_check", 4, """
**Question:** which fields did I refuse to use, and why? **Why it matters (FlyRank principle: privacy and leakage are
decided before modelling):** an exclusion with a reason can be audited.

| Excluded field | Why |
|---|---|
| `clk_out`, `imp_out`, `pos_out`, `exp_out` | outcome-window columns, the label's own ingredients (the ML-04 leak demo shows what they do) |
| `provider_used` | contains strings that look like client brand names; never selected |
| the 90-day query table | covers April to June 2026, which overlaps every outcome window |
| product flags and existing-system scores | the warehouse release has none |
| `model_used`, `last_optimized_date` | not a valid treatment (latest regeneration, not author) and too little follow-up time (8.7% of pages, all after 24 April 2026) |
| `page_prev_ratio`, `client_prev_erode_rate` (V1 features) | empty in the earliest training rows and the label anti-persists; `page_prev_ratio` stays only as the persistence baseline's input |
| `word_count` (V1 feature) | `dim_content` is a July 2026 snapshot; the count may describe a later regeneration |
| not added on purpose: outcome-window calendar, client tenure, plain month of year | with six origins they would let a model memorise the origin or the client instead of learning a pattern that transfers |
| not added on purpose: "require activity in both feature months" | it came from V1's error analysis of the test month, so adopting it would let test information shape V2 |
"""), code('''
names_ = [f["name"] for f in v2("features")["features"]]
banned = ["imp_out", "clk_out", "pos_out", "exp_out", "erode", "log_ratio_out", "provider_used", "model_used", "last_optimized_date",
          "page_prev_ratio", "client_prev_erode_rate", "word_count"]
print(f"excluded fields present in the {len(names_)}-feature list (must be empty):", [c for c in banned if c in names_])
'''), selfcheck("w03_feature_leakage_check")]

# ------------------------------------------------------------------------------------------- ML-06
rsa = summ["rhythm_signal_audit"]
_cc = summ["champion_coefficients"]
coef_top = {r["feature"]: r["std_coef"] for r in _cc["top_positive"] + _cc["top_negative"]}
coef_new = {r["feature"]: r["std_coef"] for r in _cc["new_features"]}
NL = chr(10)
rhythm_lines = NL.join(
    f"- **`{r['feature']}`:** within-client AUC {r['within_client_auc']:.3f} (95% CI {ci3(r['ci95'])}); by origin "
    f"{', '.join(f'{v:.3f}' for v in r['per_origin_auc'].values())}; erosion rate from the lowest to the highest quartile "
    f"{', '.join(str(b['erosion_rate']) for b in r['buckets'])} (base rate {r['base_rate']:.3f}). **{r['verdict']}.**" for r in rsa["features"])
rhythm_verdicts = "; ".join(f"`{r['feature']}` {r['verdict']}" for r in rsa["features"])
w04s = [head("w04_signal_audit", """
**Why this notebook is long, and why every section is here.** Each section answers one question that changes a design
decision later. Nothing here is description for its own sake. The pattern is always: question, why it matters, method,
result, and what it changes. The unit of evidence is the **client**, because pages of one client share shocks: thousands
of pages from one site are not thousands of independent proofs.
"""), code(SETUP), md("""
**Recompute or read.** Every EDA table below reads a committed receipt (`eda_*.json`, `panel_fe.json`). They were produced by the
V1 analysis scripts, which V2 does not change, because the question, the data and the label are the same. To recompute them,
run `eda_00_panel.py` to `eda_04_client_level.py` and `panel_fe.py` from `work/scripts/` (this overwrites the receipts at rounding
level, so the notebook does not do it silently). The one new analysis in this notebook, the test of the five V2 rhythm features,
reads `v2_summary.json`; with `RERUN_PIPELINE = True` and the snapshot on disk the cell in Test #4 recomputes it.
"""), code('''
print("receipts:", sorted(f for f in os.listdir(OUT) if f.startswith(("eda", "panel", "v2_summary"))))
'''), sec("w04_signal_audit", 1, """
**Look before deciding.** Three distributions shape every later design choice: who is in the panel and when, how
concentrated the AI referral signal is (the heavy tail), and how CTR varies with query length.

**Who is in the panel, and when?** **Question:** does the panel stay the same over time? **Why it matters:** if clients
join over time, any pooled trend mixes in onboarding. That decides whether trends must be estimated within client or
within page.
"""), code('''
mon = pd.DataFrame(receipt("eda_ai_monthly.json")["months"])
display(mon[["month", "daily_rows", "gsc_clients", "ga4_clients", "impressions", "clicks", "ai"]])
'''), md("""
**Result.** Search Console clients grow from 2 to 65 over 17 months. GA4 (and with it AI-referral tracking) starts in
October 2025 with 4 clients and reaches 48. **What it changes:** every trend below is within page or within client,
never a pooled monthly total, and the forward study starts in December 2025.

**How concentrated are AI referrals?** **Question:** are AI-assistant referral sessions rich enough to model directly?
**Why it matters:** it is the most direct AI signal, and FlyRank's free AI Visibility tool reports it.
"""), code('''
e1 = receipt("eda_01_landscape.json")
display(pd.DataFrame(e1["ai_concentration"]))
display(pd.DataFrame(e1["ai_by_client_top"]).head(3))
display(Image(os.path.join(FIG, "fig4_ai_referrals.png")))
'''), md("""
**Result.**
- **Volume.** There are 83,618 AI-assistant sessions, all from October 2025 onward.
- **Spread.** Only 13,270 of 205,933 GA4-tracked pages ever received one. The top 1% of those pages hold 40% of the
  sessions, and two clients hold 68%.
- **Platforms.** ChatGPT sends about 72%.

**What it changes:** AI referrals are not the target. The lane guide warns against a classifier on AI sessions alone,
and two clients would dominate any result. They enter the model as one feature (`has_ai_referral`) and the report as a
finding.

**Query length.** **Question:** do short or long queries lose more CTR at a stable rank? **Method:** the 90-day query
table (April to June 2026), pairs whose position moved by at most one place.
"""), code('''
e2 = receipt("eda_02_signals.json")
display(pd.DataFrame(e2["query_len_ctr"]))
'''), md("""
**Result.** One-word queries lose the most (0.571% to 0.457%), and queries of three or more words lose 3% to 6%.
**What it changes:** query mix would be a useful feature, but this table covers only April to June 2026. That overlaps
the outcome windows of every origin, so using it would leak. It is excluded from the model by design, and the exclusion
is documented.
"""), sec("w04_signal_audit", 2, """
Three tests, each with a question, why it matters, a bucket table, n and a verdict. The evidence unit is the client.

**Test #1: did CTR at a fixed position fall after October 2025?** **Why it matters:** if the change is an artefact of
Google's September 2025 removal of deep-result impressions (the `num=100` change), there is nothing to predict.

**Method, step A (rule out the measurement change).** A balanced panel of the 13 clients with impressions in every month
from July 2025 to June 2026, impression-weighted CTR by position band.
"""), code('''
display(pd.DataFrame(e2["ctr_by_pos_month_balanced"]))
'''), md("""
**Step A result.** The share of impressions in the top 10 jumps from 69% (August) to 85% (September) to 95% (October).
That is the `num=100` signature. CTR in positions 1 to 3 then keeps falling well after the change, from 0.586% in
October 2025 to 0.373% in March 2026, before a partial rebound in May and June. Every within-page test and every model
window therefore starts in October 2025 or later.

**Step B (how many clients show it).** Pages with at least 200 impressions in both Oct-Nov 2025 and Mar-Apr 2026 whose
position moved by at most one place. One CTR ratio per client, then a sign test and a Wilcoxon signed-rank test across
clients.
"""), code('''
display(pd.DataFrame(receipt("eda_04_client_level.json")["A_client_level"]))
'''), md("""
**Step B result.** In positions 1 to 10, CTR fell in 10 of 11 clients (median client ratio 0.664, Wilcoxon p = 0.007).

**Step C (how big, after removing page mix and rank changes).** A page fixed-effects regression, `log CTR ~ page +
month + position band`, weighted by impressions, with standard errors clustered by client (`work/scripts/panel_fe.py`).
The page effects remove page mix and the position bands remove rank changes.
"""), code('''
fe = receipt("panel_fe.json")
rows = []
for k in ["all_top20", "pos_1_3", "pos_4_10", "pos_11_20", "intent_transactional", "intent_commercial", "intent_informational"]:
    for r in fe[k]["effects"]:
        if r["month"] in ("2026-01", "2026-04", "2026-06"):
            rows.append({"segment": k, "clients": fe[k]["clients"], "pages": fe[k]["pages"], "month": r["month"],
                         "ctr_vs_oct_2025": r["ctr_multiplier"], "ci95": r["ci95"]})
display(pd.DataFrame(rows))
display(Image(os.path.join(FIG, "fig1_fixed_position_ctr.png")))
'''), md("""
**Step C result.** For the same page in the same position band, top-3 CTR in April 2026 was 0.642 times its October 2025
level (95% CI 0.615 to 0.669, 35 clients). December dips in every segment, which is the holiday season.

**n:** 11 clients in the client-level test (10 down), 107,295 pages in the fixed-effects model. **Verdict #1:
CONFIRMED.** **What it changes:** position band and intent go into the model as features, and the effect is broad, not
one client's story, so a model trained across clients has something general to learn.

**Test #2: does Google rank decide which pages AI assistants discover?** **Question:** within a client, do pages that
rank higher on Google get their first AI referral sooner? **Why it matters:** if rank decided it, a ranking tool would
double as an AI-visibility tool. **Method:** pages with no AI referral before month *m*; the outcome is a first AI
session in month *m*. For each single signal, compute a within-client AUC and weight it by the number of positives.
"""), code('''
e3 = receipt("eda_03_inference.json")
display(pd.DataFrame(e3["B_onset_signal_audit"]))
display(pd.DataFrame(e3["C_platform_rank_profile"]))
'''), md("""
**Result.** AI discovery follows exposure (prior impressions, within-client AUC 0.715) and not rank (Google position,
0.506, no better than chance). Perplexity referrals land on pages with a median prior Google position of 8.8, ChatGPT
13.1 and Gemini 17.7.

**n:** 84 client-months. **Verdict #2 (Google rank drives AI discovery): FALSE** (AUC 0.506). **What it changes:** this
is a finding for FlyRank's AI-visibility service: ranking well on Google is not what gets a page picked up by AI
assistants in this panel. It stays outside the model and goes into the recommendations.

**Test #3: do AI answers hit informational queries hardest?** **Question:** does the fall differ by search intent?
**Why it matters:** a popular industry belief says informational queries lose the most, and the playbook should not
chase the wrong group. **Method:** the same client-level test and fixed-effects model, split by main intent.
"""), code('''
a = receipt("eda_04_client_level.json")["A_client_level"]
display(pd.DataFrame([r for r in a if any(ch.isalpha() for ch in r["segment"])]))
display(pd.DataFrame([{"intent": k.replace("intent_", ""), "clients": fe[k]["clients"], "pages": fe[k]["pages"],
                       "ctr_vs_oct_2025_apr": next(r["ctr_multiplier"] for r in fe[k]["effects"] if r["month"] == "2026-04"),
                       "ci95": next(r["ci95"] for r in fe[k]["effects"] if r["month"] == "2026-04")}
                      for k in ["intent_transactional", "intent_commercial", "intent_informational"]]))
'''), md("""
**Result.** Transactional pages fell in 6 of 6 clients (p = 0.031). Informational pages are not significant (6 of 7,
p = 0.16). In the fixed-effects model, April 2026 CTR was 0.677 of October 2025 for transactional pages (95% CI 0.615
to 0.746), and 0.834 for informational pages, with an interval that includes no change.

**n:** 7 clients informational against 6 transactional. **Verdict #3: OPPOSITE** in this panel. **What it changes:**
transactional erosion becomes a reason worth naming in the playbook. The belief that AI answers hit informational
queries hardest is *not* what this panel shows, and this is stated as observed here, not as a general law.
"""), md(f"""
**Test #4 (added in V2): do the five date and rhythm features carry signal?** **Question:** does each new feature separate
eroding from non-eroding pages within the same client? **Why it matters (FlyRank principle: every feature earns its place with
evidence):** the spec added these five features from a mentor's advice that a date column yields many derived features. They
are not claimed to come from V1's errors, so they need their own test. **Method:** the labelled origins whose outcome windows
end before the test window opens (December to March, {rsa['features'][0]['n_rows']:,} rows, {rsa['features'][0]['n_clients']} clients). For each
feature, the within-client AUC (the chance that an eroding page has a higher value than a non-eroding page of the same client in
the same origin, pairs weighted), a client-cluster bootstrap 95% interval (B = {rsa['bootstrap_B']:,}), the AUC in each origin, and the
erosion rate by quartile (ties can leave a quartile empty, so some features show fewer than four buckets). The direction was written down before computing: higher value means more erosion for
`weekday_share`, `days_since_last_click` and `longest_zero_click_run`, less for `days_since_peak`, and no prior for
`keyword_to_content_gap_days`. Verdicts follow the template: CONFIRMED (interval excludes 0.5 and the sign holds in every origin,
in the expected direction), OPPOSITE (same, reverse direction), MIXED (the sign flips between origins), FALSE (interval includes
0.5). The confirmation origin (April) and the test month are not used. This audit was run after the V2 test was scored, and the
V2 feature list had been fixed in the spec before any run, so it changes no V2 decision.
"""), code('''
if RERUN_PIPELINE and DATA:
    import v2_summary; v2_summary.main()     # about a minute; rewrites v2_summary.json only
rsa = receipt("v2_summary.json")["rhythm_signal_audit"]
print("origins used:", rsa["origins"], "|", rsa["metric"])
display(pd.DataFrame([{k: r[k] for k in ["feature", "expected_direction", "within_client_auc", "ci95", "same_sign_in_all_origins", "verdict"]}
                      for r in rsa["features"]]))
for r in rsa["features"]:
    print(r["feature"], "| AUC by origin:", r["per_origin_auc"], "| base rate", r["base_rate"])
    display(pd.DataFrame(r["buckets"]))
'''), md(f"""
**Result.**
{rhythm_lines}

**n:** {rsa['features'][0]['n_rows']:,} page-origin rows from {rsa['features'][0]['n_clients']} clients ({rsa['features'][0]['n_client_origin_groups']} client-origin groups with at least 30 rows
and both classes). **Verdicts:** {rhythm_verdicts}. **What it changes:** nothing in the V2 pipeline, since the spec fixed the
features first. It changes how ML-08 is read. The champion's coefficients on these five features are small (the largest,
`longest_zero_click_run`, is {coef_new['longest_zero_click_run']:+.2f} on a standardised scale, against {coef_top['l_imp_f']:+.2f} for `l_imp_f`),
so V2's ranking still rests mostly on the same CTR-against-expectation signals as V1. A feature can carry a clear single-feature
signal and still add little to a model that already has related features: `click_day_share`, which is in the model, carries a
larger weight than any of the five. Within a client, pages that have gone quiet (a long click-free spell) are more likely to be
eroding pages in the next two months. This is an association in observational data, not a mechanism.
"""), sec("w04_signal_audit", 3, """
**Question:** FlyRank's product flags are not in the warehouse release, so I cannot test a flag's rule directly. **What I
did instead:** I linked the test to two claims in FlyRank's published AI report, which the warehouse can check. **Why it
matters:** a flag or a published claim is only useful if its assumption holds in the data.

**Claim 1 (FlyRank report):** pages with high AI visibility have about 2.6x the impressions of other pages. **Claim 2
(FlyRank report):** AI visits are more engaged than organic visits (1.37x).
"""), code('''
display(pd.DataFrame(e3["C_platform_rank_profile"])[["platform", "page_months", "clients", "median_prior_pos", "median_prior_imp"]])
display(pd.DataFrame(e1["engagement_pure_channel_days"]))
'''), md("""
**Result.** The median prior impressions of pages receiving AI referrals are 2,064 to 3,955, against 204 on all eligible
pages, so the report's direction holds and the gap is larger here (**CONFIRMED**). On page-days where every session came
from one channel, AI visits engage at 3.5% against 1.5% for organic, 2.3x, from 10,165 AI page-days and 605,640 organic
ones (**CONFIRMED** directionally). Both engagement levels are low for GA4, so I read only the ratio.

**What it changes:** the report's direction is supported, but the absolute levels differ, so I quote the direction and
not the multiples.
"""), code('''
print("Claim 1 median prior impressions by platform:", {r["platform"]: r["median_prior_imp"] for r in e3["C_platform_rank_profile"]})
print("Claim 2 engagement (pure-channel page-days):", {r["ch"]: (r["page_days"], r["engaged_pct"]) for r in e1["engagement_pure_channel_days"]})
'''), sec("w04_signal_audit", 4, f"""
A content team should read three things from this audit. At the same page and position, clicks per impression fell
after October 2025 in almost every client, so a page can lose clicks while its rank looks fine. Transactional and
commercial pages lost more than informational ones in this panel, so the first reviews belong there. Google rank does not
tell you which pages AI assistants pick up, so AI visibility needs its own check. All of this is observed in this panel
and is not a causal claim.

**Hypotheses I tested and rejected.**

| Hypothesis | Evidence | Decision |
|---|---|---|
| The LLM that wrote a page (`model_used`) drives its performance | pages created in June 2025 are tagged with a model released months later, so the field records the latest regeneration, not the author | not a valid treatment; not used |
| A refresh raises AI referrals | `last_optimized_date` covers 8.7% of pages, all dated after 24 April 2026 | too little follow-up time; not studied |
| AI referral traffic is growing per client | in the four clients with unbroken GA4 since December, the rate is flat to falling once the largest client is removed | pooled growth is onboarding; not claimed |

**Signal verdicts.**

| Signal or claim | n (unit) | Verdict |
|---|---|---|
| CTR at a fixed position fell after October 2025 | 11 clients (10 down), 107,295 pages in the FE model | **CONFIRMED** |
| AI answers hit informational queries hardest (industry claim) | 7 clients informational vs 6 transactional | **OPPOSITE** in this panel |
| Google rank drives AI discovery | 84 client-months | **FALSE** (AUC 0.506) |
| High-AI pages have about 2.6x the impressions (FlyRank report) | median prior impressions 2,064 to 3,955 on AI pages vs 204 on all eligible pages | **CONFIRMED** (larger here) |
| AI visits are more engaged (FlyRank report: 1.37x) | 10,165 AI page-days vs 605,640 organic | **CONFIRMED** directionally (2.3x), low absolute levels |
| AI referral traffic is growing per client | 4 clients with continuous GA4 | **MIXED**, not supported |
| `longest_zero_click_run` (V2): longer click-free spell, more erosion | {rsa['features'][2]['n_rows']:,} rows, {rsa['features'][2]['n_clients']} clients | **{rsa['features'][2]['verdict']}** (within-client AUC {rsa['features'][2]['within_client_auc']:.3f}) |
| `days_since_last_click` (V2): longer since the last click, more erosion | same rows | **{rsa['features'][1]['verdict']}**, weak (AUC {rsa['features'][1]['within_client_auc']:.3f}) |
| `weekday_share` (V2): higher weekday share, more erosion | same rows | **{rsa['features'][0]['verdict']}** (AUC {rsa['features'][0]['within_client_auc']:.3f}, interval includes 0.5) |
| `days_since_peak` (V2): a recent peak, more erosion | same rows | **{rsa['features'][3]['verdict']}** (AUC {rsa['features'][3]['within_client_auc']:.3f}, interval includes 0.5) |
| `keyword_to_content_gap_days` (V2, no prior) | {rsa['features'][4]['n_rows']:,} rows | **{rsa['features'][4]['verdict']}** (AUC {rsa['features'][4]['within_client_auc']:.3f}, not monotone) |
"""), code('''
auc = {r["feature"]: r["within_client_auc"] for r in e3["B_onset_signal_audit"]}
print("headline numbers behind the verdicts:")
print("  top-3 CTR, April 2026 vs October 2025:", next(r["ctr_multiplier"] for r in fe["pos_1_3"]["effects"] if r["month"] == "2026-04"))
print("  within-client AUC by single signal:", auc)
'''), selfcheck("w04_signal_audit")]


# ------------------------------------------------------------------------------------------- ML-07
PERS_DIFF = max(abs(N["v1"][m["month"]]["metrics"][b]["ap"] - m["base"][b]["ap"]) for m in N["months"] for b in VV.BASELINES)
sc1 = rj("signal_checks.json")
_t = {(r["era"], r["tier"]): r for r in sc1["check1_position_tier"]}
_c = {(r["era"], r["ctr_change"]): r for r in sc1["check2_recent_ctr_change"]}
bm1 = rj("baseline_metrics.json")
_per = {o: bm1["per_origin"][o] for o in ("2026-02", "2026-03", "2026-04", "2026-05")}
assert N["rule_matches_v1"], "the editor's rule must be identical in V1 and V2 receipts"
for _m in N["months"]:   # baseline_metrics.json (rule, per month) must agree with the V2 receipts
    assert abs(_per[_m["month"]]["p@50"] - _m["rule"]["p@50"]) < 1e-9 and abs(_per[_m["month"]]["ap"] - _m["rule"]["ap"]) < 1e-9
w04 = [head("w04_baseline_score", """
**What this card asks:** two signal checks with bucket tables, n and a verdict; ONE rule with a score, one reason code
and an action label, written to `work/outputs/baseline_action_score.csv`; and a review of the top 20. **The principle it
serves:** the solution hierarchy. A transparent rule comes first, and any model must beat it on the same rows.
"""), code(SETUP), sec("w04_baseline_score", 1, f"""
**Question:** what rule would an editor write today, and which reason codes can it output? **Why it matters (FlyRank
principle: the solution hierarchy):** a transparent rule is the bar any model must clear.

**Rule:** `score = position tier (top 3 = 2, positions 4-6 = 1, 7-10 = 0) + max(0, -CTR change from month -2 to -1)`,
with a deterministic hash tiebreak. It encodes two common intuitions: AI answers sit at the top, so top positions are
most exposed, and a CTR that has started slipping keeps slipping.

**Reason codes it can output:** `TOP3_SLIPPING` (top-3 position and CTR down more than 5% last month), `TOP3_EXPOSED`
(top-3 position, CTR not falling), `TOP10_SLIPPING` (positions 4 to 10, CTR down) and `TOP10_WATCH` (positions 4 to 10, CTR not
falling). The action label is `review now`, `review if budget allows` or `monitor`.

I froze the rule **before** looking at the checks, on purpose. It represents the status-quo heuristic, so it must not be
fitted to the data. The checks then test whether its two intuitions hold.

**Four frozen baselines, unchanged in V2.** V2 compares its learned candidates with the same four rules as V1: this editor's
rule, CTR above peers (a data-informed one-signal rule), client-wide shock and persistence of last period's ratio. The editor's
rule gives exactly the same AP and P@50 in the V1 and V2 receipts for all four months. The only implementation change is the
persistence baseline's fill value (the median over all eligible pages of the scored origin, knowable at decision time, instead of
over the measurable pages), which moves its AP by {PERS_DIFF:.4f} at most.

**Two signal checks (training-era months, December and January).**
"""), code('''
sc = receipt("signal_checks.json")
for k in sc:
    print(k); display(pd.DataFrame(sc[k]).pivot(index=list(sc[k][0])[1], columns="era", values=["n", "erosion_rate"]))
'''), md(f"""
**Check 1: are top positions more exposed?** Tied to the position tiers FlyRank's own report uses (CTR by tier: top 3,
positions 4-10). In the design months, top-3 pages eroded at {_t[('design_dec_jan', '1-3')]['erosion_rate']:.3f}, positions 4-6 at {_t[('design_dec_jan', '4-6')]['erosion_rate']:.3f} and positions 7-10 at
{_t[('design_dec_jan', '7-10')]['erosion_rate']:.3f} (n = {_t[('design_dec_jan', '1-3')]['n']:,} / {_t[('design_dec_jan', '4-6')]['n']:,} / {_t[('design_dec_jan', '7-10')]['n']:,}). **Verdict: OPPOSITE.** Over a two-month horizon, top pages
are *less* likely to lose further clicks. The large fall in top-3 CTR in ML-06 is a level shift that has already happened across
the panel, not a page-level warning sign.

**Check 2: does a slipping CTR keep slipping?** Pages whose CTR rose by more than 35% eroded at {_c[('design_dec_jan', 'rose >35%')]['erosion_rate']:.3f}. Pages whose
CTR fell by more than 26% eroded at {_c[('design_dec_jan', 'fell >26%')]['erosion_rate']:.3f} (n = {_c[('design_dec_jan', 'rose >35%')]['n']:,} and {_c[('design_dec_jan', 'fell >26%')]['n']:,}). **Verdict: OPPOSITE.** A recent bump reverts. In the
later months the falling group rises to {_c[('later_feb_apr', 'fell >26%')]['erosion_rate']:.3f} while bumps stay highest ({_c[('later_feb_apr', 'rose >35%')]['erosion_rate']:.3f}), so the pattern is not fully stable.

**What it means.** Both intuitions in the frozen rule fail the training-era data. I keep the rule as the baseline
anyway, because it is what an editor would plausibly do without a model. That is exactly the comparison the model has
to win. The other three baselines keep the comparison from being only against a weak strawman.
"""), sec("w04_baseline_score", 2, """
**Question:** what does the rule produce when it ranks every eligible page? **Why it matters:** a ranked queue with one
reason code per row is what an editor can act on. **Method:** the rule scores every July 2026 page, ranks it, assigns the reason
code and writes `work/outputs/baseline_action_score.csv` (row level, gitignored, so it exists only after the V1 playbook script
`erosion_playbook.py` has run on a machine with the snapshot). The rule reads only two features that V2 did not change
(`pos_f`, `ctr_growth`), so the file and `baseline_metrics.json` are the V1 artefacts, still valid. The notebook reads them and
does not rewrite them.
"""), code('''
bm = receipt("baseline_metrics.json")
print("rule:", bm["rule"]); print("reason codes in the July 2026 rule queue:", bm["live_queue_reason_codes"])
path = os.path.join(OUT, "baseline_action_score.csv")
if os.path.exists(path):
    bq = pd.read_csv(path, usecols=["rule_score", "reason_code", "action_label"])
    print("rows in work/outputs/baseline_action_score.csv (gitignored):", len(bq), "| distinct rule scores:", bq.rule_score.nunique())
    display(bq.groupby(["reason_code", "action_label"]).size().to_frame("pages"))
else:
    print("baseline_action_score.csv is gitignored and not on this machine; the receipt above holds its summary")
'''), md(f"""
**The rule against the V2 folds.** Same rows, same metrics as the learned candidates in ML-08. The base rate is next to every
row. The rule's P@50 is {f2(feb['rule']['p@50'])}, {f2(mar['rule']['p@50'])}, {f2(apr['rule']['p@50'])} and {f2(may['rule']['p@50'])} for February to May, and its AP is {f3(feb['rule']['ap'])}, {f3(mar['rule']['ap'])}, {f3(apr['rule']['ap'])} and {f3(may['rule']['ap'])}.
"""), code('''
rows = []
for m, role, ev in v2_months():
    for b in BASE:
        r = ev["models"][b]
        rows.append({"month": m, "role": role, "baseline": BASE_LABEL[b], "base rate": ev["base_rate"], "P@20": r["p@20"], "P@50": r["p@50"],
                     "P@50 95% CI": r["p@50_ci"], "AP": r["ap"], "AUC": r["auc"], "top-client share of top 50": r["top_client_share_top50"]})
display(pd.DataFrame(rows))
'''), sec("w04_baseline_score", 3, """
**Question:** do the rule's top 20 make sense to a person? **Why it matters (FlyRank principle: a ranking is only useful
if its top rows are explainable):** for each of the rule's top 20 pages at the July 2026 decision point, I record the
action, the reason code, why the rule picked it, and what would make the pick wrong. Pages are shown by rank, never by ID.
The review is the V1 artefact (`baseline_top20_review.json`), still valid because the rule and the two features it reads are unchanged.
"""), code('''
top20 = pd.read_json(os.path.join(OUT, "baseline_top20_review.json"))
display(top20)
print("reason codes:", top20.reason_code.value_counts().to_dict(), "| rows sharing identical position, CTR change and clicks with another row:",
      int(top20.duplicated(["position", "ctr_change_pct", "clicks_last_month"], keep=False).sum()))
'''), md("""
**What the review shows.** The rule's top 20 are all top-3 pages (positions 0.5 to 3.4) whose CTR fell 94% to 99% in one
month, and every one earns more than its client's peers at that position. A one-month collapse of that size is the kind
of change that tracking gaps or demand drops produce, so the last column sends the editor to impressions and season
before anything else. That is the failure pattern the model needs to avoid.
"""), sec("w04_baseline_score", 4, f"""
**Question:** which picks look wrong, and did any product flag or future window leak in? **Why it matters (FlyRank
principle: attack your own baseline):** a baseline that is quietly weak or leaky makes any model look better than it is.

**Weak picks.**
- **Likely tracking or demand events.** All 20 reviewed picks carry the note "a tracking gap or demand drop, not AI
  answers", because CTR fell by more than 50% in one month.
- **Repeated rows.** The last three of the 20 share identical values (position, CTR change, and last-month clicks), so
  I would review them together, not as three independent finds.
- **A tie problem.** In three of four evaluated months all 50 picks share one score, because the CTR-fall term is capped
  and many top-3 pages hit the cap (their CTR collapsed). The hash tiebreak then picks 50 of them, effectively at
  random, so the rule's precision is the erosion rate of that tied group. I keep the rule frozen as written, because
  changing it after seeing results would defeat its purpose. The distinct-score column below shows it.
"""), code('''
display(pd.DataFrame(receipt("baseline_metrics.json")["per_origin"]).T)
'''), md(f"""
**What it means.** The rule ranks above the base rate at 50 pages in all four months ({f2(may['rule']['p@50'])} to {f2(mar['rule']['p@50'])}), but its whole-ranking quality
is near or below the base rate in two of four months, and its AUC is {f3(may['rule']['auc'])} on the test month, worse than random order. This
is the baseline the model has to beat. It is also the strongest baseline on AP in {sum(m['best_ap_name'] == 'rule_position_slipping' for m in N['months'])} of the 4 months, so the comparison is not
against a strawman.

**Leakage check.** The rule reads only two features, the impression-weighted position and the log CTR change between
the two closed feature months. Neither contains an outcome-window value, a product flag (the release has none) or a
future month.
"""), code('''
feats = {f["name"]: f["meaning"] for f in v2("features")["features"]}
print({f: feats[f] for f in ["pos_f", "ctr_growth"]})
print("outcome-window or flag columns among the rule's inputs: none (the rule uses only the two features above)")
'''), selfcheck("w04_baseline_score")]


# ------------------------------------------------------------------------------------------- ML-08
ch, pen, mean2, bst = N["ch"], N["penalty_best"], N["mean2"], N["boost"]
hg, po, en = ch["HGB_CLF"], ch["POISSON_RATE"], ch["ENSEMBLE"]
nest = sel["nested_time_ordered_check"]
nev = nest["fold_2_evaluation"]
nest_ap = nev["models"]["NESTED_SELECTED"]["ap"]
nest_bb = nev["models"][nev["best_baseline_by_ap"]]["ap"]
po_its = bst["POISSON_RATE"]["test_iters"] + bst["POISSON_RATE"]["confirmation_iters"]
cand4 = ("LOGISTIC", "HGB_CLF", "POISSON_RATE", "ENSEMBLE")
apr_c = [apr["models"][n]["ap"] for n in cand4]
m2 = [mean2[n]["ap"] for n in cand4]
cc = summ["champion_coefficients"]
ctop = {r["feature"]: r["std_coef"] for r in cc["top_positive"] + cc["top_negative"]}
cfgs = sel["best_config_per_family"]
ES = sel["early_stopping_block"]
w05 = [head("w05_model", """
**What this card asks:** a method chosen from the menu and explained, compared with the Week-4 baseline on the same
data, split and metric, with features and errors interpreted, ending with a model-vs-baseline table. **The principle it
serves:** a complex model must earn its place over a frozen baseline under identical conditions. Complexity alone earns
nothing.
"""), code(SETUP), sec("w05_model", 1, f"""
**Question:** which methods from the toolkit fit this lane, and how many should I compare? **Why it matters (FlyRank principle:
a complex model must earn its place):** the choice should follow the data, not fashion, and the more candidates I compare on the
same rows, the more likely a lucky winner looks like skill.

**Four learned candidates, fixed in `work/V2_SPEC.md` before any run** (scikit-learn only, seed 42):

1. **LOGISTIC (the champion).** Median fill with missing indicators, standard scaling, logistic regression. Eighteen
   configurations: three penalties (L2, L1, elastic net with mixing 0.5) times six values of C (0.001 to 100). L1 and elastic net
   are **penalty options of this one candidate**, not extra candidates. They ask whether a sparse model does better.
2. **HGB_CLF.** Histogram gradient boosting classifier, up to 20 random configurations (learning rate, leaves, minimum leaf size, L2,
   feature subsampling). The number of trees is set by early stopping on a held-out block, then the model is refit on all training
   rows with that many trees, and predictions are averaged over five seeds.
3. **POISSON_RATE.** Histogram gradient boosting with a Poisson loss on the rate (clicks divided by expected clicks), weighted by
   expected clicks. A different loss that uses the exposure, so it can disagree with the classifier. Same search space and early stopping.
4. **ENSEMBLE.** An equal-weight rank average of the tuned logistic and tuned classifier scores. Declared in advance, not picked
   from combinations.

**Why four, and why these.** With about five effective clients per month, the AP of any one configuration on two validation folds
is noisy. Choosing the best of many candidates on the same rows that score them gives an optimistic estimate, and the optimism
grows with the number of candidates (Cawley and Talbot, 2010, *Journal of Machine Learning Research*). So the set is small, diverse and fixed
in advance. A single family would be an untested assumption: no model family wins on every tabular dataset (McElfresh et al.,
2023, NeurIPS Datasets and Benchmarks), so a linear model, a tree ensemble with two different losses and one blend is a fair test.
**Rejected:** random forest (unstable across months in my archived Lane 2 work, see `work/archive/`) and deep networks (about
{summ['champion_coefficients']['n_train_rows']:,} to {N['playbook']['train_rows']:,} training rows, and tree ensembles remain strong on medium-sized tabular data, Grinsztajn et al., 2022, NeurIPS Datasets and Benchmarks).

**Features ({N['n_features']}).** Exposure and trajectory, rank, CTR against expectation, content, GA4 channels, client context, and the
{N['n_new']} new date and rhythm features. The list is exported from code (`v2_features.json`), so the paper, this table and the model share it.
"""), code('''
spec = v2("spec")
print("candidates:", spec["candidates"], "| champion:", spec["champion"], "| challengers:", spec["challengers"])
lg = spec["models"]["LOGISTIC"]
print(f"LOGISTIC: {lg['n_configs']} configs = penalties {list(lg['penalties'])} x C {lg['C']}")
print("HGB_CLF / POISSON_RATE:", spec["models"]["HGB_CLF"]["n_configs"], "random configs each; trees by early stopping, cap", spec["models"]["HGB_CLF"]["max_iter_cap"])
f = pd.DataFrame(v2("features")["features"])
display(f.groupby("family").size().to_frame("features"))
display(f[["name", "family", "meaning", "new"]])
'''), md("""
Every feature is computed from months *o-3* to *o-1*. Two V1 features that depended on matured labels are gone, and so is
`word_count` (see ML-04). The five new ones are the date and rhythm features tested in ML-06.
"""), sec("w05_model", 2, f"""
**Question:** is the split honest for a "next month, all clients" decision? **Why it matters (FlyRank principle: the split must
match the deployment question):** a random split would let the model recognise clients it has seen, and a test that
leaks into the choices tells you nothing.

**The split, with no overlap between training, validation and test outcome windows.** For each scored month *o* the model trains
only on origins <= *o-2*, whose outcome windows have closed before *o* begins.

| Role | Scored month | Outcome window | Trains on origins | Decides anything? |
|---|---|---|---|---|
| Validation fold 1 | Feb 2026 | Feb to Mar | Dec | yes: tuning and the champion test |
| Validation fold 2 | Mar 2026 | Mar to Apr | Dec, Jan | yes: tuning and the champion test |
| Confirmation fold | Apr 2026 | Apr to May | Dec to Feb | **no**, reported only |
| Test (sealed) | May 2026 | May to Jun | Dec to Mar (refit) | **no**, scored once after the freeze |
| Live | Jul 2026 | none yet | Dec to May | queue only |

The two validation windows (Feb to Mar, Mar to Apr) end before the test window (May to Jun) opens. The confirmation fold's
window overlaps the test window, so it is computed after the selection is written to disk and cannot change it. Rolling-origin folds do reuse
earlier months by design (fold 2 trains on origin January, whose outcome window overlaps fold 1's validation window in
February), but within each fold every training window ends before the scored month begins.

**Tuning and selection (fixed before any run).** Each configuration is scored by its mean AP over validation folds 1 and 2, and the
best configuration per family is kept (ties within 0.0001 AP go to the simpler, more regularised one). AP is the only criterion.
The early-stopping block is a client-grouped 20% of clients in fold 1 ({ES['validation_1']['n_clients_held_out']} clients, {ES['validation_1']['row_share'] * 100:.0f}% of the rows, because only one training origin
exists) and the latest training origin in fold 2 (January). **Champion and challengers.** The logistic model is the champion. A challenger
replaces it only if **both** hold: the paired AP gain over the champion, pooled over the two folds, has a client-bootstrap 95% lower bound above 0
(B = 2000, clients resampled within each fold), and the mean gain stays above 0 when any single client is left out. There are
{N['n_comparisons']} challenger comparisons and **no multiplicity correction gates the decision**; a Bonferroni-adjusted bound is printed for
information. **A nested check** repeats the whole procedure on fold 1 only and scores the result on fold 2.

**Caveat stated up front.** Tuning and the champion test use the same two folds, so the validation-fold numbers of the tuned
models are optimistic. The confirmation fold and the test month are the less-biased reads. {DISC}
"""), code('''
spec = v2("spec"); tl = v2("leakage_checks")["timeline"]
split = pd.DataFrame([{"role": k, "scored origin": f["score"], "train origins": f["train"], "outcome window": f["outcome_window"],
                       "decides anything": f["used_for_decisions"]} for k, f in spec["folds"].items()])
display(split)
for k in ["validation_1", "validation_2", "confirmation", "test", "live"]:
    t = tl[{"validation_1": "validation_1", "validation_2": "validation_2", "confirmation": "confirmation", "test": "test", "live": "live"}[k]]
    assert max(t["train_outcome_windows_end"].values()) < t["origin_first_day"], k
print("every training outcome window ends before the scored origin begins: True")
print("selection settings:", {k: spec["selection"][k] for k in ["criterion", "tuning", "bootstrap_B", "pass_rule", "n_challenger_comparisons"]})
print("early-stopping blocks:", v2("selection")["early_stopping_block"])
'''), sec("w05_model", 3, f"""
**Question:** does the model beat the frozen baseline on the same data, split and metric, and which candidate should be the final
model? **Why it matters:** a model that cannot beat a transparent rule should not ship, and a winner picked on noise should not
be believed. **Method:** the selection table, the champion and challenger rule, then the model-versus-baseline tables on the validation
folds and the confirmation fold. **Why AP selects the model while P@50 is the operational metric.** {AP_WHY}

**1. The logistic penalty options (the selection table).** Best configuration per penalty by mean validation AP:

| Penalty | Best C | Mean AP (folds 1, 2) | Fold 1 AP | Fold 2 AP |
|---|---|---|---|---|
| L2 | {pen['l2']['C']:g} | {pen['l2']['mean_ap']:.4f} | {pen['l2']['fold_ap'][0]:.3f} | {pen['l2']['fold_ap'][1]:.3f} |
| L1 | {pen['l1']['C']:g} | {pen['l1']['mean_ap']:.4f} | {pen['l1']['fold_ap'][0]:.3f} | {pen['l1']['fold_ap'][1]:.3f} |
| Elastic net (0.5) | {pen['elasticnet']['C']:g} | {pen['elasticnet']['mean_ap']:.4f} | {pen['elasticnet']['fold_ap'][0]:.3f} | {pen['elasticnet']['fold_ap'][1]:.3f} |

The three penalties differ by {max(p['mean_ap'] for p in pen.values()) - min(p['mean_ap'] for p in pen.values()):.4f} AP at their best. Sparsity did not help: L1 and elastic net at their best C match L2, and strong
regularisation (small C) hurts all three. **Selected: L2 with C = {pen['l2']['C']:g}**, which is weak regularisation, so correlated features share weight (ML-08 section 4). {GRID_CLAUSE}
"""), code('''
sel = v2("selection")
lg = pd.DataFrame([{"penalty": r["cfg"]["penalty"], "C": r["cfg"]["C"], "fold 1 AP": r["fold_ap"][0], "fold 2 AP": r["fold_ap"][1],
                    "mean AP": r["mean_ap"], "fold P@50": r["fold_p@50"], "converged": not r["non_converged_folds"]} for r in sel["configs"]["LOGISTIC"]])
display(lg.pivot(index="C", columns="penalty", values="mean AP").round(4))
print("best configuration per family:"); display(pd.DataFrame({k: {"config": v["cfg"], "mean AP": v["mean_ap"]} for k, v in sel["best_config_per_family"].items()}).T)
print("non-converged logistic fits:", sel["non_converged_logistic_fits"], "| runtime of the selection step (s):", sel["runtime_seconds"])
display(Image(os.path.join(FIG, "fig5_selection.png")))
'''), md(f"""
**2. Champion against challengers.** Mean validation AP of the four candidates is {mean2['LOGISTIC']['ap']:.4f} (logistic), {mean2['HGB_CLF']['ap']:.4f} (boosted classifier),
{mean2['POISSON_RATE']['ap']:.4f} (Poisson-rate boosted) and {mean2['ENSEMBLE']['ap']:.4f} (rank-average ensemble), against {mean2['rule_position_slipping']['ap']:.4f} for the best baseline on average (the editor's rule).
The paired test asks whether any challenger's AP gain over the champion is distinguishable from zero.

| Challenger | Mean AP gain | 95% interval | Lower bound | Bonferroni-3 bound (information only) | Leave-one-client-out gain, min to max | Clients whose removal makes the gain <= 0 | Passes |
|---|---|---|---|---|---|---|---|
| Boosted classifier | {hg['mean_gain']:+.4f} | {hg['boot_95_ci'][0]:+.4f} to {hg['boot_95_ci'][1]:+.4f} | {hg['lower_bound']:+.4f} | {hg['bonferroni_3_lower_bound_info_only']:+.4f} | {hg['loco_min_mean_gain']:+.4f} to {hg['loco_max_mean_gain']:+.4f} | {hg['loco_clients_with_gain_le_0']} of {hg['loco_n_clients']} | no |
| Poisson-rate boosted | {po['mean_gain']:+.4f} | {po['boot_95_ci'][0]:+.4f} to {po['boot_95_ci'][1]:+.4f} | {po['lower_bound']:+.4f} | {po['bonferroni_3_lower_bound_info_only']:+.4f} | {po['loco_min_mean_gain']:+.4f} to {po['loco_max_mean_gain']:+.4f} | {po['loco_clients_with_gain_le_0']} of {po['loco_n_clients']} | no |
| Rank-average ensemble | {en['mean_gain']:+.4f} | {en['boot_95_ci'][0]:+.4f} to {en['boot_95_ci'][1]:+.4f} | {en['lower_bound']:+.4f} | {en['bonferroni_3_lower_bound_info_only']:+.4f} | {en['loco_min_mean_gain']:+.4f} to {en['loco_max_mean_gain']:+.4f} | {en['loco_clients_with_gain_le_0']} of {en['loco_n_clients']} | no |

**Decision: no challenger passed both parts of the rule, so the logistic champion stays.** The ensemble came closest: a mean gain of
{en['mean_gain']:+.4f} AP with a lower bound of {en['lower_bound']:+.4f} (just under 0) and a positive gain whichever single client is removed. That is not enough under a rule
written in advance. The challengers are statistically indistinguishable from the champion on these two folds. That is a statement about
what two folds and about five effective clients can show. It does not say the logistic model is better, only that nothing here justifies
a more complex model. **Multiple comparisons:** there were {N['n_comparisons']} comparisons and the rule does not correct for them. The Bonferroni
bounds are lower still, so a correction would not change the decision, but I do not claim more than the uncorrected rule allows.

**Nested check.** Repeating the whole procedure on fold 1 only selects the logistic model again (L1, C = {nest['configs_chosen_on_fold_1']['LOGISTIC']['C']:g}); scored on fold 2 it
reaches AP {nest_ap:.4f}, against {nev['models']['SELECTED_BOTH_FOLDS_INSAMPLE']['ap']:.4f} for the both-fold choice and {nest_bb:.4f} for the best baseline. The champion is not an artefact of tuning
on both folds.
"""), code('''
ch = v2("selection")["challenger_vs_champion"]
display(pd.DataFrame(ch).T[["per_fold_gain", "mean_gain", "boot_95_ci", "lower_bound", "bonferroni_3_lower_bound_info_only", "p_gain_le_0",
                            "loco_min_mean_gain", "loco_max_mean_gain", "loco_clients_with_gain_le_0", "loco_n_clients", "passes"]])
print("challengers passing:", v2("selection")["challengers_passing"], "| selected:", v2("selection")["selected_model"], "|", v2("selection")["decision"])
nest = v2("selection")["nested_time_ordered_check"]
print("nested check: selected on fold 1 only ->", nest["selected_on_fold_1"], nest["configs_chosen_on_fold_1"]["LOGISTIC"],
      "| fold-2 AP:", nest["fold_2_evaluation"]["models"]["NESTED_SELECTED"]["ap"])
vf = v2("validation_folds")["mean_over_two_folds"]
display(pd.DataFrame(vf).T.round(4))
'''), md(f"""
**3. Model against baseline, validation folds and confirmation fold.** Same rows and the same metrics for every ranker, with the base
rate. The validation numbers are tuned-on and therefore optimistic. The confirmation fold was run after the selection was
written, so it is the less-biased of the two.
"""), code('''
rows = []
for m, role, ev in v2_months()[:3]:
    for name in BASE + ["LOGISTIC", "HGB_CLF", "POISSON_RATE", "ENSEMBLE"]:
        r = ev["models"][name]
        rows.append({"month": m, "role": role, "ranker": BASE_LABEL.get(name, name), "base rate": ev["base_rate"], "AP": r["ap"], "AP 95% CI": r["ap_ci"],
                     "P@50": r["p@50"], "P@50 95% CI": r["p@50_ci"], "AUC": r["auc"], "P@50, unmeasurable = miss": r["p@50_unmeasurable_as_miss"]})
display(pd.DataFrame(rows))
'''), md(f"""
**Reading it.** On the validation folds every learned candidate has a higher AP than every frozen baseline, and the four
candidates' mean AP over the two folds sits within {max(m2) - min(m2):.3f} ({min(m2):.4f} to {max(m2):.4f}). {AP_SENT}. {P50_SENT}. On the confirmation fold (April) the four candidates are within
{max(apr_c) - min(apr_c):.3f} AP of one another ({f3(min(apr_c))} to {f3(max(apr_c))}); the ensemble is nominally highest, and
the champion had already been chosen before this fold was run, so this is reported and nothing more. **Boosted models stop early:** the selected boosted classifier stops
after {bst['HGB_CLF']['validation_median_iter_selected'][0]} and {bst['HGB_CLF']['validation_median_iter_selected'][1]} trees (median, validation folds 1 and 2), {min(bst['HGB_CLF']['confirmation_iters'])} to {max(bst['HGB_CLF']['confirmation_iters'])} on the confirmation fold and
{min(bst['HGB_CLF']['test_iters'])} to {max(bst['HGB_CLF']['test_iters'])} in the test refit (five seeds); the Poisson-rate model uses more ({min(po_its)} to {max(po_its)} trees across those two refits).
Very short ensembles suggest there is little structure beyond what a linear model captures, which agrees with the champion test.

The bootstrap intervals (resampling clients) are wide at 50 pages, so single-month P@50 differences are not decisive. Consistency
across months, and the whole-ranking AP, are the signal.
"""), sec("w05_model", 4, f"""
**Question:** what did the model learn, and does it make sense? **Why it matters (FlyRank principle: a score that looks too good
is investigated):** a top driver must be explained, not celebrated. **Method:** standardised coefficients of the champion
logistic (selected configuration) refit on the test-refit rows (origins December to March, {cc['n_train_rows']:,} rows). This uses
training labels only. The receipt is `v2_summary.json`.
"""), code('''
cc = receipt("v2_summary.json")["champion_coefficients"]
print("champion logistic, selected config", cc["config"], "| fitted on origins", cc["fitted_on_origins"], "| converged:", cc["converged"])
print(cc["note"])
display(pd.DataFrame(cc["top_positive"]).rename(columns={"std_coef": "largest positive"}))
display(pd.DataFrame(cc["top_negative"]).rename(columns={"std_coef": "largest negative"}))
print("the five new V2 features:"); display(pd.DataFrame(cc["new_features"]))
pbs = v2("playbook_summary")
print("live July queue, top driver of each of the 50 queued pages:", pbs["top_drivers"])
'''), md(f"""
**Interpretation.** The largest positive weights are `l_imp_f` ({ctop['l_imp_f']:+.2f}) with a matching negative `l_clk_f` ({ctop['l_clk_f']:+.2f}) (together: many impressions per
click), `ctr_vs_ref` ({ctop['ctr_vs_ref']:+.2f}, the page earns more clicks than its position normally does) and `ctr_growth` ({ctop['ctr_growth']:+.2f}, a recent CTR bump). Read
together: **the pages that lose clicks are the ones currently earning more than their rank normally earns, or riding a recent
bump.** That is the opposite of the editor's rule ("slipping keeps slipping"), it agrees with ML-07 check 2, and it is the same story V1 told. With
C = {cc['config']['C']:g} the model is only weakly regularised and several features are correlated, so the coefficients show where the model leans and are not
importances. The five new date and rhythm features carry small weights (largest `longest_zero_click_run`, {coef_new['longest_zero_click_run']:+.2f}),
so V2's gain in process is not a gain from new signal. In the live July queue the top driver is `ctr_vs_ref` for {pb['top_drivers']['ctr_vs_ref']} of the {pb['queue_size']} queued pages.

**Is this just measurement noise reverting?** No. The one-signal mean-reversion rule (CTR above peers) ranks at about the base
rate (AP {f3(feb['base']['mean_reversion_ctr_excess']['ap'])}, {f3(mar['base']['mean_reversion_ctr_excess']['ap'])}, {f3(apr['base']['mean_reversion_ctr_excess']['ap'])} and {f3(may['base']['mean_reversion_ctr_excess']['ap'])} against base rates of {f3(feb['base_rate'])}, {f3(mar['base_rate'])}, {f3(apr['base_rate'])} and {f3(may['base_rate'])}),
and the label already shrinks noisy CTRs. The model's edge comes from combining the signals, not from any one of them.

**Error notes.** Failure analysis with the unmeasurable pages is in ML-09. In short: the 50-page queue holds only about {min(m['clk'] for m in N['months']) * 100:.0f}% to {max(m['clk'] for m in N['months']) * 100:.0f}% of a month's total click
shortfall ("clicks-lost captured"), because the model ranks the probability of erosion and not its size; its top 50 come from {min(m['clients_top50'] for m in N['months'])} to {max(m['clients_top50'] for m in N['months'])} clients,
with the biggest client holding {min(m['top_client'] for m in N['months']) * 100:.0f}% to {max(m['top_client'] for m in N['months']) * 100:.0f}% of the queue; and {UNM_SENT}. {RULE_SENT}

**Disclosure.** {DISC}

**Model against baseline, the final table** (the champion with the four frozen baselines, all four evaluated months):
"""), code('''
t = fold_table(); display(t)
print("rows where the champion's AP exceeds every frozen baseline:", int(sum(champ(ev)["ap"] > max(ev["models"][b]["ap"] for b in BASE) for _, _, ev in v2_months())), "of 4")
print("rows where its P@50 exceeds every frozen baseline:", int(sum(champ(ev)["p@50"] > max(ev["models"][b]["p@50"] for b in BASE) for _, _, ev in v2_months())), "of 4")
'''), md("""
**Reproducing the selection (opt-in).** Set `RERUN_PIPELINE = True` in the setup cell and run the next cell to re-run the V2 steps that
come before the freeze (`select`, `confirm`, `secondary`, the leakage checks) in a **scratch copy**, never in the repository, and compare
the headline numbers with the committed receipts. It takes about ten minutes and needs the pinned snapshot. It deliberately never
runs the unseal or live steps: the unseal step refuses to run a second time, by design, because the test month is scored once.
"""), code('''
if RERUN_PIPELINE and DATA:
    import v2_rerun
    print(v2_rerun.rerun())
else:
    print("RERUN_PIPELINE is False: nothing recomputed. The V2 receipts in work/outputs are what this notebook reads.")
'''), selfcheck("w05_model")]


# ------------------------------------------------------------------------------------------- ML-09
tsm = N["ts"]
sec2 = N["test_secondary"]
tmod = N["test_models"]
bs = tmod["rule_position_slipping"]
csd = tmod["client_shock"]
w06 = [head("w06_validation_audit", """
**What this card asks:** two findings from FlyRank's research paper with a methodology question for each; my model
under a grouped or time-aware split with a before/after; a leakage audit; real failure examples; and a claim rewrite.
**The principle it serves:** the split must match the deployment question, and an honest, lower score beats an inflated
one.
"""), code(SETUP), sec("w06_validation_audit", 1, """
**Question:** does each finding's validation design carry its claim? **Why it matters (FlyRank principle: where the label
comes from decides what a claim can say):** I ask both questions in a constructive tone, and turn them on my own study.

**Finding A: the Freshness Multiplier.** Pages older than 365 days that were refreshed in the last 30 days show 3.2x the
health score and 57x the impressions.
*My question:* refreshed pages are chosen by people, often because they already matter. Without a comparison group of
similar pages that were not refreshed, chosen before the refresh, how much of the 57x is selection rather than the
refresh? My own study meets the same risk: the queue picks pages, editors act on them, and any later "improvement"
mixes the action with regression to the mean. That is why the playbook proposes a hold-out of queued pages (ML-10).

**Finding B: the Content Performance Curve.** Health peaks at 61 to 90 days of age, decays to 14 by day 271 to 365, and
rebounds after a year.
*My question:* each age bucket is a different set of pages measured at one moment, so can a single snapshot show a
lifecycle? Pages that survive past a year are survivors. My study avoids this by following the same page forward
(fixed effects in ML-06, forward origins here), but it inherits a milder version: the eligibility rule keeps pages
that still rank in the top 10, which favours survivors, and the label cannot be measured for pages that lose almost all of their visibility (the
unmeasurable pages, which come back in the failure analysis below). That is a stated limitation.
"""), code('''
summ = receipt("erosion_frame_summary.json")
t = pd.DataFrame(summ["origins"])[["origin", "eligible", "measurable"]].dropna()
t["share_dropped_as_unmeasurable"] = (1 - t.measurable / t.eligible).round(3)
print("survivor effect in my own eligibility rule (pages kept vs dropped per labelled origin):"); display(t)
'''), sec("w06_validation_audit", 2, f"""
**Question:** does the evaluation design change the answer? **Why it matters:** the deployment question is "next month,
for all clients", so only a forward split answers it. **Method:** the same model under three splits (before and after), then
the V1 and V2 results side by side, the test month, and a failure analysis.

**Before and after.** The selected logistic configuration, on the February and March rows with 5-fold splits: random rows
(before) against whole clients held out (after), then the forward March fold (train December and January, score March).
"""), code('''
lk = v2("leakage_checks"); rv = lk["random_vs_client_grouped_feb_mar"]; fold2 = v2("validation_folds")["folds"]["validation_2"]
t = pd.DataFrame([
    {"split": "random rows, 5-fold (before)", "AP": rv["random_row_kfold"]["ap_mean"], "AP std": rv["random_row_kfold"]["ap_std"], "base rate": rv["base_rate"], "answers": "re-describing clients it has seen"},
    {"split": "clients held out, 5-fold (after)", "AP": rv["client_group_kfold"]["ap_mean"], "AP std": rv["client_group_kfold"]["ap_std"], "base rate": rv["base_rate"], "answers": "a new client, same months"},
    {"split": "forward, March fold (after)", "AP": fold2["models"]["LOGISTIC"]["ap"], "AP std": None, "base rate": fold2["base_rate"], "answers": "next month, matured labels only"}])
display(t)
v1 = receipt("erosion_leakage_checks.json")
print("V1 history, V1 features:", {k: v1[k]["ap_mean"] for k in ["random_row_kfold", "client_group_kfold"]}, "gap", round(v1["random_row_kfold"]["ap_mean"] - v1["client_group_kfold"]["ap_mean"], 3))
'''), md(f"""
**Result.** Random rows report AP {rvg['random_row_kfold']['ap_mean']:.3f}; holding clients out gives {rvg['client_group_kfold']['ap_mean']:.3f}; the forward March fold is {f3(mar['ap'])}. The random split's extra {rvg['random_minus_grouped_ap']:.3f} is
client memorisation (V1 measured {lk1['random_row_kfold']['ap_mean'] - lk1['client_group_kfold']['ap_mean']:.3f} on its own features). Every claim in this submission uses the forward design, and every interval resamples whole clients.

**The honest split, one more time.** Training windows end before the scored month begins in every row.

| Role | Scored month | Outcome window | Trained on | Decides anything? |
|---|---|---|---|---|
| Validation fold 1 | Feb | Feb to Mar | Dec | yes |
| Validation fold 2 | Mar | Mar to Apr | Dec, Jan | yes |
| Confirmation fold | Apr | Apr to May | Dec to Feb | no, reported only |
| Test (sealed) | May | May to Jun | Dec to Mar (refit) | no, scored once |
| Live | Jul | none yet | Dec to May | queue only |

**V1 and V2 side by side.** Same months, V1 logistic (34 features, chosen in V1) against V2 logistic champion (36 features, chosen under the V2
rule). V1's February to April were its development months and its May was its sealed month, opened once in V1. V2's February and March
were selection folds, April a confirmation fold and May the test. Neither set is a pristine out-of-sample read.
"""), code('''
v1d, v1s = receipt("erosion_model_metrics.json")["origins"], receipt("erosion_model_metrics_sealed.json")["origins"]
v1all = {**v1d, **v1s}
rows = []
for m, role, ev in v2_months():
    a, b = v1all[m]["metrics"]["logreg"], champ(ev)
    rows.append({"month": m, "V2 role": role, "base rate": ev["base_rate"], "V1 AP": a["ap"], "V2 AP": b["ap"], "AP, V2 minus V1": round(b["ap"] - a["ap"], 4),
                 "V1 P@50": a["p@50"], "V2 P@50": b["p@50"], "P@50, V2 minus V1": round(b["p@50"] - a["p@50"], 2),
                 "V1 P@50 95% CI": v1all[m]["bootstrap"]["logreg"]["p@50_ci"], "V2 P@50 95% CI": b["p@50_ci"]})
display(pd.DataFrame(rows))
'''), md(f"""
**What it shows.** {V1V2_SENT}. Across the four months V2's AP is higher than V1's in February, March and April ({feb['ap'] - feb['v1_ap']:+.3f}, {mar['ap'] - mar['v1_ap']:+.3f}, {apr['ap'] - apr['v1_ap']:+.3f}) and
lower on the test month ({may['ap'] - may['v1_ap']:+.3f}); its P@50 is higher in February, March and the test month and lower in April ({apr['p50'] - apr['v1_p50']:+.2f}). Those are changes
of a few points inside intervals of about 0.2, and the V2 numbers for February and March were tuned on, so I do not read them as
improvement.

**The test month (scored once, after the freeze).** **Design.** Origin May 2026 (outcome May and June 2026, the panel's final months) was seen during V1.
The V2 code therefore enforces its protection: it drops the test origin's outcome columns from every load unless a freeze file
exists, its stored code and spec hashes match, and the test has not been scored before. I wrote `v2_freeze.json` (chosen model and configuration, spec and code
hashes) at {N['frozen_at']}, scored the test at {N['scored_at']}, and the code refuses a second run. {DISC}
"""), code('''
fz, ts = v2("freeze"), v2("test")
print("frozen at", fz["frozen_at"], "| test scored at", ts["scored_at"], "| freeze before test:", fz["frozen_at"] < ts["scored_at"], "| scored once:", ts["scored_once"])
print("selected model:", fz["selected_model"], "|", fz["selected_model_decision"]); print("post-freeze changes:", fz["post_freeze_changes"])
print("code hashes recorded at freeze (first 12 hex):", {k: v[:12] for k, v in fz["code_sha256"].items()})
print("spec hash, pre-registration vs freeze (first 12 hex):", {k: (fz["spec_sha256_at_preregistration"][k][:12], fz["spec_sha256"][k][:12]) for k in fz["spec_sha256"]})
ev = ts["evaluation"]
print(f"test origin {ts['score_origin']}: {ev['n_eligible']:,} eligible pages, {ev['n_measurable']:,} measurable, {ev['n_clients']} clients, effective clients {ev['effective_clients']}, "
      f"base rate {ev['base_rate']}, noise-only rate {ev['noise_only_rate']}, trained on {ts['train_origins']} ({ts['n_train_rows']:,} rows)")
rows = [{"ranker": BASE_LABEL.get(n, "V2 logistic (selected)"), "AP": r["ap"], "AP 95% CI": r["ap_ci"], "P@50": r["p@50"], "P@50 95% CI": r["p@50_ci"], "P@20": r["p@20"],
         "AUC": r["auc"], "lift": r["lift"], "within-client P@10": r["within_client_p@10"], "clients in top 50": r["clients_in_top50"],
         "P@50, unmeasurable = miss": r["p@50_unmeasurable_as_miss"], "AP gain over best baseline": r["delta_ap_vs_best_baseline"],
         "gain 95% CI": r["delta_ap_vs_best_baseline_ci"]} for n, r in ev["models"].items()]
display(pd.DataFrame(rows))
display(Image(os.path.join(FIG, "fig2_forward_validation.png")))
'''), md(f"""
**Result.**
- **V2 logistic (the only learned model scored on the test month):** AP {f3(may['ap'])} (95% CI {ci3(may['ap_ci'])}) against {f3(may['best_ap'])} for the best frozen baseline (the editor's rule). The AP gain
  is {may['d_ap']:+.3f} (95% CI {ci3(may['d_ap_ci'])}). P@50 is {f2(may['p50'])} (95% CI {ci2(may['p50_ci'])}) against a base rate of {f2(may['base_rate'])} and a noise-only rate of {f2(may['noise'])}, a lift of {f2(may['lift'])}. Within-client P@10 is {f2(may['wc10'])}
  and AUC {f3(may['auc'])}.
- **At exactly 50 pages the lead is not established.** The best other baseline by P@50 is client shock ({f2(may['best_p50'])}), whose 50 picks come from only {csd['clients_in_top50']} clients, and the editor's rule scores
  {f2(bs['p@50'])} with an AUC of {f3(bs['auc'])}, worse than random order. The 95% interval of the P@50 gain over the best-AP baseline is {ci2(may['d_p50_ci'])}, which includes 0.
- **Leave-one-client-out:** removing any single client leaves the AP gain between {may['loco']['min']:+.3f} and {may['loco']['max']:+.3f}, always above 0.
- **Under the conservative count the rule is ahead.** {RULE_SENT} The rule scores {f2(may['rule']['p@50_unmeasurable_as_miss'])} on the test month against the champion's {f2(may['p50_unm'])}, and {f2(apr['rule']['p@50_unmeasurable_as_miss'])} against {f2(apr['p50_unm'])} on the confirmation month.
- **Not a pristine holdout.** See the disclosure above; the test number may be optimistic.

**Failure analysis: unmeasurable pages, concentration and what the queue misses.**
"""), code('''
rows = []
for m, role, ev in v2_months():
    g = champ(ev)
    rows.append({"month": m, "role": role, "unmeasurable share of eligible pages": ev["lost_visibility_share"], "P@50 (measurable pages)": g["p@50"],
                 "P@50, unmeasurable = miss": g["p@50_unmeasurable_as_miss"], "editor's rule P@50, unmeasurable = miss": ev["models"][BASE[0]]["p@50_unmeasurable_as_miss"],
                 "editor's rule share of click shortfall (top 50)": ev["models"][BASE[0]]["clicks_lost_captured"], "P@50 after the 10-per-client cap (descriptive)": g["capped_p@50_descriptive"],
                 "share of month's click shortfall in the top 50": g["clicks_lost_captured"], "clients in top 50": g["clients_in_top50"],
                 "largest client's share of top 50": g["top_client_share_top50"], "effective clients": ev["effective_clients"]})
display(pd.DataFrame(rows))
sec_ = v2("secondary_label")
fr = {"2026-02": sec_["folds"]["validation_1"], "2026-03": sec_["folds"]["validation_2"], "2026-04": sec_["folds"]["confirmation"], "2026-05": sec_["test"]}
display(pd.DataFrame([{"month": m, "secondary AP": v["ap"], "secondary base rate": v["base_rate_secondary"], "secondary P@50": v["p@50"],
                       "top-client share": v["top_client_share_top50"], "clients in top 50": v["clients_in_top50"],
                       "top-50 pages shared with the primary top 50": v["overlap_top50_with_primary_top50"], "primary AP": v["primary_logistic_ap_on_primary_label"]}
                      for m, v in fr.items()]))
pe = receipt("sealed_error_profile.json")
print("V1 history only (V1 logistic, V1 top 50 on the test month; V2 did not use it, spec section 11):"); display(pd.DataFrame(pe).T)
'''), md(f"""
**What the failures show.**
- **Unmeasurable pages are the main caveat on the headline.** Between {f2(N['unm_range'][0] * 100)}% and {f2(N['unm_range'][1] * 100)}% of eligible pages in a month lose so much visibility that the label cannot be measured (their expected outcome clicks are under
  10), so every headline metric drops them. If the queue ranks such pages, the headline does not see them. {UNM_SENT}. The conservative reading counts them as misses;
  some of them may well have lost clicks, but this label cannot say. **This is why the playbook asks the editor to verify visibility before acting (ML-10).**
- **The queue captures only a sliver of the clicks at stake.** The test month's top 50 holds {may['clk'] * 100:.1f}% of that month's total click shortfall on the measurable pages ({min(m['clk'] for m in N['months']) * 100:.1f}% to {max(m['clk'] for m in N['months']) * 100:.1f}% across the four months), because
  the model ranks the probability of erosion and not its size.
- **Concentration and the effective sample.** The top 50 come from {min(m['clients_top50'] for m in N['months'])} to {max(m['clients_top50'] for m in N['months'])} clients, the biggest client holding {min(m['top_client'] for m in N['months']) * 100:.0f}% to {max(m['top_client'] for m in N['months']) * 100:.0f}% of it, while
  the effective number of clients is {f2(N['eff_range'][0])} to {f2(N['eff_range'][1])}.
- **Secondary label (limited scope).** The label made client-relative (a page erodes if its clicks-to-expected ratio is at least 25% below the median page of its own client that month) was evaluated only for the champion logistic
  with its selected configuration, and never used to choose anything. On the test month its AP is {sec2['ap']:.3f} (base rate {sec2['base_rate_secondary']:.3f}) and P@50 {sec2['p@50']:.2f}, with {sec2['overlap_top50_with_primary_top50']} of its top 50 shared with the primary top 50. The ranking does
  not depend on the primary label's client-level normalisation alone.
- **V1's view of the same month, for the record.** V1 profiled its own top 50 on the test month after unsealing it (last table above). V2 did not adopt any of it: the spec lists
  "require activity in both feature months" under deliberately not done, because it came from that look at the test month.
"""), sec("w06_validation_audit", 3, f"""
**Question:** which leakage routes did I test on the final feature set, and what did each show? **Why it matters
(FlyRank principle: attack your own model):** a leak found by a reviewer costs more than one found by me.

| Risk | Test | Result |
|---|---|---|
| Momentum inside the label | three label versions (ML-05, V1) | fixed; models' scores fell after the fix |
| Label ingredient as a feature | planted outcome feature on validation fold 2 | AP {pl['LOGISTIC_honest']['ap']:.3f} to {pl['LOGISTIC_planted_future_ratio']['ap']:.3f}, P@50 {pl['LOGISTIC_honest']['p@50']:.2f} to {pl['LOGISTIC_planted_future_ratio']['p@50']:.2f}, so the harness detects leaks |
| Client memorisation | random vs grouped split on the V2 rows | {rvg['random_row_kfold']['ap_mean']:.3f} vs {rvg['client_group_kfold']['ap_mean']:.3f} AP; claims use forward months and whole-client resampling |
| Future months in features | timeline assertions for all five roles | features end at month *o-1*; training outcome windows end before the scored month |
| New date features | latest daily date read per origin | before each origin's first day, `v2_features.json` |
| Snapshot attributes dated after the origin | keyword dates checked, `word_count` removed | zero keyword-date violations |
| Tuning on the test | the freeze gate in code | `v2_freeze.json` at {N['frozen_at']}, before `v2_test.json` at {N['scored_at']}; the unseal step refuses a second run |
| The test month was seen in V1 | disclosed | {DISC} |
| Confirmation fold overlaps the test window | reported only, computed after the selection was written | decided nothing |
| Query table overlaps outcomes | window check (ML-06, query length) | table excluded |
"""), code('''
lk = v2("leakage_checks")
print("timeline assertions passed:", lk["timeline_assertions_passed"], "| harness detects leak:", lk["planted_future_feature_fold_2"]["harness_detects_leak"],
      "| honest models not leaky:", lk["planted_future_feature_fold_2"]["honest_models_not_leaky"])
print("seed invariance check:", lk["seed_invariance"])
print("latest daily date read for rhythm features:", lk["max_daily_date_read_for_rhythm_features"])
'''), sec("w06_validation_audit", 4, f"""
**Question:** which of my sentences is boldest, and what does the evidence support? **Why it matters (FlyRank principle:
careful words):** a claim should be no stronger than its design.

| Before (too strong) | After (what the evidence supports) |
|---|---|
| "AI Overviews are stealing FlyRank clients' clicks." | "At a fixed page and position, CTR fell after October 2025 in {c10['clients_down']} of {c10['clients']} clients, a pattern consistent with AI answers absorbing clicks; Search Console has no AI-answer flag, so the cause is not measured." |
| "V2 improved the model." | "V2 changed the process: four candidates, a selection rule fixed in advance, a confirmation fold and a test scored once. On the test month its AP is {f3(may['ap'])} against V1's {f3(may['v1_ap'])} and its P@50 {f2(may['p50'])} against {f2(may['v1_p50'])}, both inside the noise." |
| "The model predicts which pages will lose traffic." | "In four evaluated months the logistic ranking had a higher AP than all four frozen baselines every time and a P@50 of {f2(N['p50_range'][0])} to {f2(N['p50_range'][1])} against base rates of {f2(N['base_rate_range'][0])} to {f2(N['base_rate_range'][1])}. The AP gain over the best baseline has an interval above 0 in {N['d_ap_excl0_months']} of 4 months. Decision support for review, not a forecast of Google." |
| "The queue is {f2(may['p50'])} precise." | "{f2(may['p50'])} of the 50 top measurable pages eroded on the test month. Counting pages that lost visibility as misses it is {f2(may['p50_unm'])}, and what happened to those pages cannot be measured with this label." |
| "The logistic model is the best model." | "Four candidates were not distinguishable on two validation folds. No challenger passed the rule written in advance (the ensemble's lower bound was {N['closest_lb']:+.3f}), so the simplest model stayed." |
| "The model beats the rule." | "The model was more consistent than any single rule on the whole ranking (AP). At exactly 50 pages its lead over the best baseline is within noise in {4 - N['d_p50_excl0_months']} of 4 months, and with unmeasurable pages counted as misses the editor's rule scores higher in every month." |
"""), code('''
t = fold_table()
print(f"logistic AP across the four evaluated months: {t['V2 AP'].min()} to {t['V2 AP'].max()}; P@50: {t['V2 P@50'].min()} to {t['V2 P@50'].max()}; base rates: {t['base rate'].min()} to {t['base rate'].max()}")
print("P@50 counting unmeasurable pages as misses:", dict(zip(t.month, t["P@50, unmeasurable = miss"])))
'''), selfcheck("w06_validation_audit")]


# ------------------------------------------------------------------------------------------- guards: prose claims that are written as words
assert N["ap_lead_months"] == 4 and N["spec_hash_changed"] and N["freeze_before_test"] and N["rule_matches_v1"]
assert N["closest"] == "ENSEMBLE" and not any(v["passes"] for v in N["ch"].values())
assert N["freeze"]["selected_model"] == "LOGISTIC"
_wd = next(r for r in summ["rhythm_signal_audit"]["features"] if r["feature"] == "weekday_share")
_dp = next(r for r in summ["rhythm_signal_audit"]["features"] if r["feature"] == "days_since_peak")
assert _wd["ci95"][0] < 0.5 < _wd["ci95"][1] and _dp["ci95"][0] < 0.5 < _dp["ci95"][1]
assert {r["feature"]: r["verdict"] for r in summ["rhythm_signal_audit"]["features"]}["longest_zero_click_run"] == "CONFIRMED"
assert N["by"]["2026-03"]["p50_above_all"] is False and N["p50_lead_months"] == 3 and N["d_p50_excl0_months"] == 1
assert N["d_ap_excl0_months"] == 3 and not N["by"]["2026-04"]["d_ap_excl0"]


# ------------------------------------------------------------------------------------------- ML-10
cap_cost = pb["cap_cost_descriptive"]
HG_ITS = bst["HGB_CLF"]["validation_median_iter_selected"] + bst["HGB_CLF"]["confirmation_iters"] + bst["HGB_CLF"]["test_iters"]
rc_all = pb["reason_codes_all_eligible"]
w07 = [head("w07_action_playbook", """
**What this card asks:** ranked actions with reason codes, an archetype-to-action map, the key decay insight, intended
use, limits, human review, cost and value, monitoring and retrain triggers, and what must not be automated. **The
principle it serves:** rules give the reason code, models give the order, and a person decides.
"""), code(SETUP), sec("w07_action_playbook", 1, f"""
**Question:** what should an editor do first, and why? **Why it matters (FlyRank principle: rules give the reason code,
models give the order):** the queue has to be explainable in words a human trusts.

**The ranked queue for the July 2026 decision.** The V2 selected model (the logistic champion, because no challenger passed the
rule), refit on every month whose label had matured (December to May, {pb['train_rows']:,} rows, {pb['n_features_used']} features), scores {pb['eligible_pages']:,} eligible pages.
Pages younger than 90 days are excluded ({pb['excluded_younger_than_90_days']:,}) and no client gets more than {pb['guardrails']['client_cap']} slots. The reason code is the largest positive standardised
contribution of the champion logistic fitted on the same rows, mapped to a plain-language family. Rows are shown by rank and an anonymous
client index. `rank` is the model's rank before the guardrails, so gaps are pages removed for being younger than 90 days or for exceeding a
client's cap. The full queue with IDs for the editor is written to a gitignored CSV; its SHA-256 is recorded in `v2_playbook_summary.json`, which
makes the queue a frozen artefact: any later change to the file changes the hash.
"""), code('''
pb = v2("playbook_summary")
print({k: pb[k] for k in ["decision_origin", "feature_window", "trained_on_origins", "train_rows", "eligible_pages", "excluded_younger_than_90_days",
                          "queue_size", "queue_clients", "max_pages_one_client", "reason_codes", "clicks_at_risk_2_months_if_25pct_lost"]})
print("reason codes over all eligible pages:", pb["reason_codes_all_eligible"])
print("queue pages also in the V1 queue:", pb["overlap_with_v1_queue_pages"], "of", pb["queue_size"])
'''), code('''
import hashlib
path = os.path.join(OUT, pb["queue_file"]["name"])
if os.path.exists(path):
    q = pd.read_csv(path)
    print("frozen queue file matches the recorded SHA-256:", hashlib.sha256(open(path, "rb").read()).hexdigest() == pb["queue_file"]["sha256"])
    view = q[q.in_queue].copy(); view["client"] = pd.factorize(view.client_hash_id)[0]
    display(view[["rank", "client", "position", "score", "top_driver", "reason_code"]].head(15).reset_index(drop=True))
else:
    print("the queue CSV is gitignored and not on this machine; recorded SHA-256:", pb["queue_file"]["sha256"][:16], "...")
'''), md(f"""
**Reason code to action.**

| Reason code (largest model contribution) | What it means | Action |
|---|---|---|
| `CTR_ABOVE_PEERS` | the page earns more clicks than its position normally does | **Protect:** check the live results page for an AI answer; make the page the citable source (answer-first summary, FAQ, structured data) before editing anything else |
| `DEMAND_SHIFT` | impressions swung sharply, or the peak was very recent (`days_since_peak`, new in V2) | **Check demand first:** seasonality or tracking may explain the change; no rewrite until confirmed |
| `CTR_ALREADY_SLIPPING` | a CTR fall has started | rewrite title and description toward the click an AI answer cannot satisfy (comparison, pricing, tool, example) |
| `STALE_ACTIVITY` (new in V2) | a long quiet spell (`days_since_last_click`, `longest_zero_click_run`) | check whether the page still earns clicks for its query; re-read the live results page and refresh the content |
| `WEEKDAY_PROFILE` (new in V2) | traffic is concentrated on working days (`weekday_share`) | check whether the query is a work-task question an AI answer can resolve; test an answer-first rewrite |
| `TOP_POSITION_EXPOSED` | rank alone drives the score | make the answer citable; monitor |
| `RANK_MOVING` | position is unstable | **hold:** re-score next month |
| `CLIENT_WIDE_DECLINE` | the whole site is losing CTR at a stable rank | escalate to the account lead; page edits alone will not fix it |
| `PAGE_PROFILE` | no single driver | editor judgement: check intent match and freshness |

**What the July queue looks like.** {pb['reason_codes']['CTR_ABOVE_PEERS']} of the {pb['queue_size']} queued pages are `CTR_ABOVE_PEERS` and {pb['reason_codes']['CTR_ALREADY_SLIPPING']} is `CTR_ALREADY_SLIPPING`; across all {pb['eligible_pages']:,} eligible
pages the codes are spread ({rc_all['PAGE_PROFILE']:,} `PAGE_PROFILE`, {rc_all['CTR_ABOVE_PEERS']:,} `CTR_ABOVE_PEERS`, {rc_all['CTR_ALREADY_SLIPPING']:,} `CTR_ALREADY_SLIPPING`, {rc_all['DEMAND_SHIFT']:,} `DEMAND_SHIFT`,
{rc_all['CLIENT_WIDE_DECLINE']} `CLIENT_WIDE_DECLINE`, {rc_all['STALE_ACTIVITY']} `STALE_ACTIVITY`). So inside the queue the order comes from the model, and the code mainly tells the editor where to look first.
{pb['overlap_with_v1_queue_pages']} of the {pb['queue_size']} queued pages are also in V1's July queue.

**The insight behind the playbook.** The pages that lose clicks next are not the ones already slipping. They are the
ones **currently earning more than their rank normally earns**. That excess click share is what an AI answer can take.
So the first action is protective: make the page the source an AI answer cites. It is not a rewrite after the loss.
"""), sec("w07_action_playbook", 2, f"""
**Question:** who uses this, for what, and where does it stop being valid? **Why it matters (FlyRank principle: a person
decides):** a tool used outside its intended use can mislead.

- **Who:** a FlyRank content editor, once a month, with a budget of 50 reviews.
- **When:** day 1 of the month, using the two closed months before it.
- **What it is:** a decision-support queue of top-10 pages most likely to lose at least 25% of their expected clicks at
  their current rank in the next two months, each with one reason code and one suggested action.
- **What it is not:** an automatic rewrite, a forecast of Google, or a causal claim that the action restores clicks.
- **Where it stops being valid:** pages outside the top 10, pages younger than 90 days, pages whose visibility has collapsed (the
  label cannot be measured for them), and any month whose base rate leaves the range seen so far (see the monitoring triggers).

**Cost and value.**
"""), code('''
pb = v2("playbook_summary"); ts = v2("test")["evaluation"]; g = ts["models"]["SELECTED:LOGISTIC"]
print(f"Test month: {g['p@50'] * 50:.0f} of 50 queued pages eroded with the model, {g['p@50_unmeasurable_as_miss'] * 50:.0f} if unmeasurable pages count as misses, against about {ts['base_rate'] * 50:.0f} for a random 50.")
print(f"Editor time: 50 reviews x about 20 minutes = about 17 hours per month.")
print(f"Clicks at stake in the July queue if each page lost 25% over two months: about {round(pb['clicks_at_risk_2_months_if_25pct_lost'], -3):,.0f}.")
display(pd.DataFrame(pb["cap_cost_descriptive"]).T)
print(pb["guardrails"])
'''), md(f"""
**Reading it.**
- **Value of the model:** on the test month, the queue holds about {may['p50'] * 50 - may['base_rate'] * 50:.0f} more eroding pages per 50 reviews than random selection ({may['p50'] * 50:.0f} against about {may['base_rate'] * 50:.0f}). Counting the pages
  that later lost visibility as misses, the gain shrinks to about {may['p50_unm'] * 50 - may['base_rate'] * 50:.0f}. That is the time saved, with a wide interval (P@50 95% CI {ci2(may['p50_ci'])}).
- **The 10-pages-per-client cap is a guardrail, not a result.** It was introduced in V1 after the sealed month had been opened, for fairness of the review budget, and it plays no part in any V2 selection or headline metric. Descriptively, it
  changes the champion's P@50 by {min(cap_cost[k]['capped_p@50'] - cap_cost[k]['uncapped_p@50'] for k in cap_cost):+.2f} to {max(cap_cost[k]['capped_p@50'] - cap_cost[k]['uncapped_p@50'] for k in cap_cost):+.2f} across the four evaluated months.
"""), sec("w07_action_playbook", 3, f"""
**Question:** what must a person check before acting, and what should never be automated? **Why it matters (FlyRank
principle: a person decides):** every false alarm costs an edit that can put a page's rank at risk.

**Human review (required before any action).**
1. Open the live results page for the page's main query and record whether an AI answer appears and whether the page is
   cited.
2. **Verify visibility.** Confirm the page still ranks and still earns impressions. A page whose visibility has collapsed cannot be scored by the label:
   {f2(N['unm_range'][0] * 100)}% to {f2(N['unm_range'][1] * 100)}% of eligible pages per month are unmeasurable, and counting such pages as misses lowers the test month's P@50 from {f2(may['p50'])} to {f2(may['p50_unm'])}. This is the main guardrail against acting on a
   page the model ranked for the wrong reason.
3. If impressions jumped threefold or more, or the page had data in only one month, confirm the demand is real before
   acting.
4. Respect a 30-day cooldown after any edit, and skip pages younger than 90 days.
5. The editor can override any row. Overrides are logged with a reason, and they become training signal.

**The no-go list (what must not be automated).**
- **Rewriting or publishing content from the queue.** A person reviews the live results page first.
- **Treating a false alarm as harmless.** Every edit risks the rank the page holds.
- **Claiming that an action restored clicks.** That needs a hold-out (for example, leave 10 of the 50 queued pages
  untouched each month and compare), and this study does not have one yet.
- **Client-level conclusions from page queues.** Client-wide declines go to the account lead.
"""), code('''
pb = v2("playbook_summary")
print(f"guardrails in the July queue: client cap = {pb['guardrails']['client_cap']} slots, pages younger than 90 days excluded = {pb['excluded_younger_than_90_days']:,}")
print("unmeasurable share of eligible pages by evaluated month:", {m: ev["lost_visibility_share"] for m, _, ev in v2_months()})
'''), sec("w07_action_playbook", 4, f"""
**Question:** what would tell me the recommendations went stale? **Why it matters (FlyRank principle: a model is a
process, not a file):** a ranking trained on past months drifts as Google and the panel change.

| Trigger | Threshold | Response |
|---|---|---|
| A new month's label matures | monthly | refit the frozen configuration on all matured months and re-run the forward check; a different model needs a new pre-registered spec |
| Precision of the last matured queue | below 1.3x that month's base rate (the four evaluated months run {f2(min(m['lift'] for m in N['months']))}x to {f2(max(m['lift'] for m in N['months']))}x) | pause the queue and investigate drift |
| Base rate moves | outside 0.15 to 0.45 (seen so far: {f2(N['base_rate_range'][0])} to {f2(N['base_rate_range'][1])}) | check for a Google change or a tracking break before trusting scores |
| Share of unmeasurable pages | above the highest seen so far ({f2(N['unm_range'][1] * 100)}%) | check for a tracking break; re-read the headline with unmeasurable pages counted as misses |
| Client concentration | one client above 10 slots before the cap, or effective clients below the {f2(N['eff_range'][0])} seen so far | review that client's data feed; widen the intervals before reading anything |
| AI-answer presence | editors record AI answers on reviewed pages | after 3 months, test whether recorded presence adds signal |

**Next iteration (to be pre-registered before it runs).** The data ends on 2026-06-30, so the July queue cannot be scored yet. It is frozen by its SHA-256,
and the one untouched test left is its own outcome (July and August 2026), once a later data release exists.
1. **Score the frozen July queue** against its matured outcome, once, with the same metrics.
2. **Log editors' AI-answer observations as a feature.**
3. **Run the 10-page hold-out to measure the effect of the actions.**
"""), code('''
print("precision-to-base-rate ratio (lift) of the champion in each evaluated month (the trigger fires below 1.3):")
display(pd.Series({m: round(champ(ev)["lift"], 2) for m, _, ev in v2_months()}).to_frame("P@50 / base rate"))
'''), sec("w07_action_playbook", 5, """
**Question:** which files does the paper build on? **Why it matters (FlyRank principle: every number traces to a
receipt):** the paper and the capstone notebook read these files, so they must exist and match.

`work/outputs/v2_queue_2026-07.csv` (gitignored, row level, frozen by its SHA-256), `v2_playbook_summary.json`, `v2_selection.json`,
`v2_validation_folds.json`, `v2_confirmation_apr.json`, `v2_test.json`, `v2_freeze.json`, `v2_leakage_checks.json`, `v2_secondary_label.json`,
`v2_features.json`, `v2_summary.json`, the V1 receipts it compares with, and the figures in `work/figures/` feed the paper.
"""), code('''
for f in ["v2_spec.json", "v2_spec_hash.json", "v2_features.json", "v2_selection.json", "v2_validation_folds.json", "v2_confirmation_apr.json", "v2_secondary_label.json",
          "v2_leakage_checks.json", "v2_freeze.json", "v2_test.json", "v2_playbook_summary.json", "v2_summary.json", "v2_queue_2026-07.csv"]:
    print(f"{f:34s} exists: {os.path.exists(os.path.join(OUT, f))}")
print("figures:", sorted(f for f in os.listdir(FIG) if f.endswith(".png")))
'''), selfcheck("w07_action_playbook")]

# ------------------------------------------------------------------------------------------- ML-11 + ML-12
cap = [head("capstone", f"""
**Paper title:** Same rank, fewer clicks: an early warning for click loss to AI answers (V2).

This notebook mirrors the paper section by section, and reads only committed receipts and figures, so it runs without
warehouse access (`work/scripts/verify_evidence.py` executes it offline). The analysis itself lives in the assignment
notebooks it links to.
"""), code(SETUP_OFFLINE), sec("capstone", 1, f"""
**Abstract.** AI answers in search are the new variable in FlyRank's business, so I asked which top-10 pages are about to lose clicks at the rank they already hold, so that an editor can protect 50 of them first each month.
I used FlyRank's pseudonymized internship warehouse ({mon['daily_rows'] / 1e6:.1f} million daily rows, January 2025 to June 2026), where page fixed effects with client-clustered errors put top-3 click-through in April 2026 at {top3['ctr_multiplier']:.3f} times its October 2025 level for the same page in the same position band (95% CI {top3['ci95'][0]} to {top3['ci95'][1]}).
The monthly ranking comes from a pipeline that was specified and hashed before it ran: a label that visible momentum cannot give away, {N['n_features']} features, four frozen baselines, four learned candidates compared under a selection rule fixed in advance, and a test month scored once.
The logistic champion had the highest average precision (AP) in {N['ap_lead_months']} of 4 evaluated months, although the {names(lambda m: not m['d_ap_excl0']).replace('Apr', 'April')} gain interval includes 0; on the test month its AP was {f3(may['ap'])} (95% CI {ci3(may['ap_ci'])}) against {f3(may['best_ap'])} for the best baseline, and its precision at 50 pages (P@50) was {f2(may['p50'])} against a base rate of {f2(may['base_rate'])}, which falls to {f2(may['p50_unm'])} when pages that later lost visibility count as misses, below the editor's rule at {f2(may['rule']['p@50_unmeasurable_as_miss'])}.
The output is a reviewed monthly queue for editors, a predictive and not causal decision aid whose intervals are wide at {N['eff_range'][0]:.1f} to {N['eff_range'][1]:.1f} effective clients and whose test month was seen during V1, so the result may be optimistic.

**Introduction.** AI answers in search are the new variable in FlyRank's work. FlyRank's own *State of AI-Driven SEO 2026*
report measures AI referral sessions but not what AI answers do to Google clicks at a given rank. My Lane 2 study
(archived) showed that a target answered by visible momentum lets a two-line rule beat any model. This study therefore
starts from a target that momentum does not give away, and from the decision an editor faces: which 50 pages to protect
this month. V1 of the pipeline did that with two models and one sealed month. V2 is a process upgrade: the same question, label and
unit, with a rule for choosing the model fixed in advance. It is not a way to raise the score, and the V1 numbers stay in the
repository, labelled as V1.
"""), code('''
fe = receipt("panel_fe.json")
r = next(e for e in fe["pos_1_3"]["effects"] if e["month"] == "2026-04")
t = fold_table().set_index("month")
print(f"Abstract numbers: top-3 CTR {r['ctr_multiplier']} x October 2025 (95% CI {r['ci95'][0]} to {r['ci95'][1]}, {fe['pos_1_3']['clients']} clients); "
      f"V2 logistic AP {t['V2 AP'].min()} to {t['V2 AP'].max()}, P@50 {t['V2 P@50'].min()} to {t['V2 P@50'].max()}, base rates {t['base rate'].min()} to {t['base rate'].max()}.")
print("test month:", t.loc["2026-05", ["V2 AP", "V2 AP 95% CI", "V2 P@50", "base rate", "P@50, unmeasurable = miss"]].to_dict())
'''), sec("capstone", 2, ""), code('''
mon, summ = receipt("eda_ai_monthly.json"), receipt("erosion_frame_summary.json")
print(f"Warehouse build v20260703, revision {summ['warehouse_revision'][:8]}: {mon['daily_rows']:,} daily rows, {mon['first_date']} to {mon['last_date']}.")
display(pd.DataFrame(summ["origins"])[["origin", "eligible", "measurable", "clients", "lost_visibility", "base_rate", "null_rate"]])
display(pd.DataFrame([{"month": m, "role": r, "eligible": ev["n_eligible"], "measurable": ev["n_measurable"], "clients": ev["n_clients"], "effective clients": ev["effective_clients"],
                       "top-3 clients' row share": ev["top3_client_row_share"], "base rate": ev["base_rate"], "noise-only rate": ev["noise_only_rate"]} for m, r, ev in v2_months()]))
'''), md(f"""
Tables used: the daily performance fact table, `dim_content` and `dim_clients`. Excluded: pages outside the top 10, pages
with fewer than 200 impressions or 20 clicks in the feature window, pages whose expected outcome clicks are under 10
(the unmeasurable pages, {f2(N['unm_range'][0] * 100)}% to {f2(N['unm_range'][1] * 100)}% of eligible pages per evaluated month), the 90-day query table (its window overlaps every outcome), the `provider_used` column (it contains brand-like
strings), and three V1 features (`page_prev_ratio`, `client_prev_erode_rate`, `word_count`). Rate columns are percentages, GA4 is used only on days flagged `IS TRUE`, and no data enters git.
The warehouse has one revision (the pinned one) and ends on 2026-06-30, so there is no more data to collect a fresh holdout from. {DISC}
"""), sec("capstone", 3, f"""
- **Unit:** page x monthly decision point. Features use the two closed months before the decision; the label uses the
  two months after.
- **Label:** clicks below 75% of what the page's current CTR (latest month, shrunk toward its client's position-adjusted
  CTR) predicts at its new position. A Poisson simulation puts the noise-only rate at {min(NULLS_ALL):.3f} to {max(NULLS_ALL):.3f} (training, validation, confirmation and test months) against base rates of {f2(N['base_rate_range'][0])} to {f2(N['base_rate_range'][1])} in the evaluated months.
- **Features ({N['n_features']}):** {N['n_v1_features']} from V1, minus three (`page_prev_ratio`, `client_prev_erode_rate`, `word_count`), plus five date and rhythm features
  (`weekday_share`, `days_since_last_click`, `longest_zero_click_run`, `days_since_peak`, `keyword_to_content_gap_days`), exported from code (`v2_features.json`).
- **Baselines (frozen):** the editor's rule (top position + slipping CTR), CTR above peers, client shock and persistence.
- **Candidates (four):** a logistic model (L2, L1 or elastic net penalty; 18 configurations; the champion), a boosted classifier, a Poisson-rate boosted regressor
  (20 random configurations each, trees by early stopping) and a pre-declared rank-average ensemble. Four is a deliberate limit: with about five effective clients, selecting
  among many candidates mostly selects noise (Cawley and Talbot, 2010).
- **Selection (fixed before any run):** mean AP over two validation folds, best configuration per family, then a champion and challenger rule (a challenger must have a client-bootstrap lower
  bound above 0 and a positive gain whichever client is left out; {N['n_comparisons']} comparisons, not corrected for multiplicity). {AP_WHY} {GRID_CLAUSE}
- **Split:** validation folds Feb (train Dec) and Mar (train Dec, Jan) decide; the Apr confirmation fold (train Dec to Feb) is reported only, because its outcome window overlaps the test window; May is the test month (refit Dec to Mar), scored
  once after `v2_freeze.json`; July is live (Dec to May). Training, validation and test outcome windows do not overlap.
- **Checks:** a planted-leak harness, a random-versus-grouped split comparison, timeline assertions for all five roles, and a secondary client-relative label for the champion only.
- **Disclosure.** {DISC}
"""), code('''
sel, fz = v2("selection"), v2("freeze")
print(len(v2("features")["features"]), "features; champion:", fz["champion"], "| selected:", fz["selected_model"], "| challengers passing:", sel["challengers_passing"])
print("frozen at", fz["frozen_at"], "| test scored at", v2("test")["scored_at"], "| scored once:", v2("test")["scored_once"])
print("noise-only label rate by origin:", {o["origin"]: o["null_rate"] for o in summ["origins"] if o.get("null_rate") is not None})
'''), sec("capstone", 4, ""), code('''
fe = receipt("panel_fe.json")
print("Fixed-position CTR, April 2026 vs October 2025:")
for k in ["pos_1_3", "pos_4_10", "intent_transactional", "intent_informational"]:
    r = next(e for e in fe[k]["effects"] if e["month"] == "2026-04")
    print(f"  {k:22s} {r['ctr_multiplier']:.3f}  (95% CI {r['ci95'][0]} to {r['ci95'][1]}; {fe[k]['clients']} clients)")
display(Image(os.path.join(FIG, "fig1_fixed_position_ctr.png")))
'''), code('''
t = fold_table(); display(t)
rows = []
for m, role, ev in v2_months():
    g = champ(ev)
    rows.append({"month": m, "role": role, "AP above all four baselines": g["ap"] > max(ev["models"][b]["ap"] for b in BASE),
                 "AP gain over best baseline": g["delta_ap_vs_best_baseline"], "gain 95% CI": g["delta_ap_vs_best_baseline_ci"],
                 "P@50 above all four baselines": g["p@50"] > max(ev["models"][b]["p@50"] for b in BASE), "P@50 gain 95% CI": g["delta_p@50_vs_best_baseline_ci"]})
display(pd.DataFrame(rows))
display(Image(os.path.join(FIG, "fig2_forward_validation.png")))
display(Image(os.path.join(FIG, "fig5_selection.png")))
display(Image(os.path.join(FIG, "fig3_validation_checks.png")))
v1d, v1s = receipt("erosion_model_metrics.json")["origins"], receipt("erosion_model_metrics_sealed.json")["origins"]
v1all = {**v1d, **v1s}
display(pd.DataFrame([{"month": m, "V2 role": r, "V1 AP": v1all[m]["metrics"]["logreg"]["ap"], "V2 AP": champ(ev)["ap"], "V1 P@50": v1all[m]["metrics"]["logreg"]["p@50"], "V2 P@50": champ(ev)["p@50"]}
                      for m, r, ev in v2_months()]))
'''), md(f"""
**Reading the results.** {AP_SENT}. {P50_SENT}. The editor's rule wins P@50 in March ({f2(mar['rule']['p@50'])}) and falls below random order on the test month (AUC {f3(may['rule']['auc'])}).

**Test month.** AP {f3(may['ap'])} (95% CI {ci3(may['ap_ci'])}) against {f3(may['best_ap'])} for the best baseline, a gain of {may['d_ap']:+.3f} (95% CI {ci3(may['d_ap_ci'])}); P@50 {f2(may['p50'])} (95% CI {ci2(may['p50_ci'])}) against a base rate of {f2(may['base_rate'])}, a lift of {f2(may['lift'])}.
At 50 pages the interval of the gain over the best-AP baseline is {ci2(may['d_p50_ci'])}, so a P@50 lead is not established.

**Unmeasurable pages.** {UNM_SENT}. Pages that lose most of their visibility cannot be scored by the label, so the headline metrics drop them. The conservative reading counts them as misses. **{RULE_SENT}**

**Reading the AP lead.** The champion led on AP in {N['ap_lead_months']} of 4 months, but {APRIL_CAVEAT}.

**Candidates.** No challenger passed the champion rule. The closest was the ensemble (lower bound {N['closest_lb']:+.3f}). The four candidates are statistically indistinguishable on two validation folds,
and the boosted classifier stops after {min(HG_ITS)} to {max(HG_ITS)} trees.

**V1 against V2.** {V1V2_SENT}. {DISC}
"""), sec("capstone", 5, f"""
- **Disclosure.** {DISC}
- **Predictive, not causal.** Search Console has no AI-answer flag, so the CTR fall is *consistent with* AI answers. It
  could also come from seasonality (December dips everywhere), other search-feature changes or impression inflation.
- **About five effective clients per month** ({f2(N['eff_range'][0])} to {f2(N['eff_range'][1])}): every interval resamples whole clients, and the intervals at 50 pages are wide (about +/-0.2). P@50 differences between rankers are not established.
- **Unmeasurable pages:** the headline P@50 does not see pages that lost visibility; counting them as misses lowers it to {f2(may['p50_unm'])} on the test month and {f2(apr['p50_unm'])} on the confirmation month, while the editor's rule scores {f2(may['rule']['p@50_unmeasurable_as_miss'])} and {f2(apr['rule']['p@50_unmeasurable_as_miss'])}. {RULE_SENT}
- **Tuning and the champion test share two folds,** so the validation numbers are optimistic; the confirmation fold and the test month are the less-biased reads. Three comparisons were made and not corrected for multiplicity.
- **Only four evaluated months,** and the V1 comparison is not a like-for-like improvement test.
- **Eligibility favours survivors:** pages still in the top 10 with enough clicks.
- **Snapshot attributes** (`dim_content`, keyword fields) are as of July 2026, not each origin.
- **The client cap and the 90-day rule are playbook guardrails.** The cap was introduced in V1 after the test month was opened and plays no part in any V2 selection or headline metric.
- **The new date features add little:** the champion gives them small weights, so V2's gain is in process, not in signal.
- **AI referral findings** rest on a thin, concentrated signal (two clients hold 68% of sessions).
"""), code('''
ts = v2("test")["evaluation"]
print("test month P@50 95% CI:", champ(ts)["p@50_ci"], "| n pages:", ts["n_measurable"], "| clients:", ts["n_clients"], "| effective clients:", ts["effective_clients"])
print("AI referral concentration:", receipt("eda_01_landscape.json")["ai_concentration"][0]["top1pct_share"], "of sessions on the top 1% of pages")
'''), sec("capstone", 6, f"""
1. **Use the queue for protection, not repair.** The pages to act on are those earning more clicks than their rank
   normally earns. Make them the source an AI answer cites before editing anything else (ML-10).
2. **Verify visibility before acting.** A page that has lost most of its impressions cannot be scored by the label; check the live results page first.
3. **Treat transactional and commercial pages as the more exposed group in this panel.** Informational pages did not
   show a significant fall.
4. **Do not use Google rank as a proxy for AI visibility.** AI discovery follows exposure (AUC {e3['B_onset_signal_audit'][0]['within_client_auc']}), not rank ({next(r['within_client_auc'] for r in e3['B_onset_signal_audit'] if r['feature'] == 'pos')}).
5. **Measure the action:** hold out 10 of each month's 50 pages to estimate what the actions are worth.
6. **Retrain monthly, and pause the queue** if a matured month's precision falls below 1.3x its base rate. Score the frozen July queue against its outcome once data exists.
"""), code('''
pb = v2("playbook_summary")
print("July 2026 queue:", pb["queue_size"], "pages |", pb["queue_clients"], "clients | client cap", pb["guardrails"]["client_cap"], "| reason codes", pb["reason_codes"])
print("queue file SHA-256 (frozen artefact):", pb["queue_file"]["sha256"])
'''), sec("capstone", 7, f"""
**Reproducibility.**
- **Code:** `work/scripts/` (one module per stage; `v2_*.py` is the V2 pipeline). Notebooks w01 to w07 read the receipts and, with `RERUN_PIPELINE = True`, recompute what the pinned snapshot allows. The V2 steps then run in a scratch copy, never in the repository.
- **Pre-registration:** `work/V2_SPEC.md` and `work/outputs/v2_spec.json` were committed before any V2 code ran (SHA-256 recorded in `v2_spec_hash.json`). One implementation deviation is logged in section 13 of the spec (a column with no information in a given fit is left out of that boosted fit only). That edit means the spec hash at the freeze differs from the pre-registered hash;
  `v2_freeze.json` records both ({N['spec_hash_pre']['work/V2_SPEC.md'][:10]} and {N['spec_hash_now']['work/V2_SPEC.md'][:10]} for the markdown spec).
- **Order of operations, in receipts:** spec, selection, confirmation, secondary label, checks, freeze ({N['frozen_at']}), test scored once ({N['scored_at']}), live queue. The unseal step refuses a second run.
- **Seeds and pins:** seed 42 everywhere, warehouse revision `50cbf7c3`. Fits are single-threaded, so results are bit-reproducible.
- **Environment:** `requirements.txt` plus scipy.
- **Receipts:** `work/outputs/*.json`; figures: `work/figures/`.
- **Offline checker:** `python work/scripts/verify_evidence.py` re-executes this notebook and checks every headline number in the paper and report against the receipts.

| Result | Notebook | Receipt |
|---|---|---|
| Fixed-position CTR fall | `w04_signal_audit` | `panel_fe.json`, `eda_04_client_level.json` |
| Label, noise floor, split | `w02_ml_task_framing` | `erosion_frame_summary.json`, `v2_spec.json` |
| Leak demo, contract queries | `w03_data_contract` | `data_contract.json`, `v2_summary.json` |
| Label leaks, planted leak, split gap | `w03_feature_leakage_check` | `label_design_log.json`, `v2_leakage_checks.json` |
| Rhythm features as signals | `w04_signal_audit` | `v2_summary.json` |
| Baselines and checks | `w04_baseline_score` | `baseline_metrics.json`, `signal_checks.json`, `v2_validation_folds.json` |
| Candidates, selection, champion | `w05_model` | `v2_selection.json`, `v2_features.json`, `v2_confirmation_apr.json` |
| Test month, V1 vs V2, failures | `w06_validation_audit` | `v2_freeze.json`, `v2_test.json`, `v2_secondary_label.json` |
| Playbook | `w07_action_playbook` | `v2_playbook_summary.json` |

**Acknowledgments and data credit.** Built on the FlyRank ML Internship dataset ([flyrank.ai](https://flyrank.ai)): the
pseudonymized warehouse release `FlyRank/internship-warehouse` (build v20260703), the starter repository and the
curriculum. Client identities are pseudonymized and none appear here. Errors and interpretations are mine.
"""), code('''
print("figures the paper embeds:", sorted(f for f in os.listdir(FIG) if f.startswith(("fig", "paper"))))
print("V2 receipts read by this notebook:", sorted(f for f in os.listdir(OUT) if f.startswith("v2_") and f.endswith(".json")))
'''), md(f"""
**ML-12 · Five-minute demo outline**

1. **Question (45 s).** Editors can review 50 pages a month. Which ranking pages are about to lose clicks to AI answers?
2. **Evidence the problem is real (60 s).** Figure 1: same page, same position, top-3 CTR at {top3['ctr_multiplier']:.2f}x in six months.
3. **Method (75 s).** An honest label (latest CTR, shrunk, position-adjusted, noise floor about 5%), four frozen rules, four candidates compared under a rule written in advance,
   a confirmation fold that decides nothing and a test month scored once.
4. **One honest result (60 s).** Figure 2: the logistic champion has the highest AP in all four evaluated months. At 50 pages it is about twice the base rate, but with intervals of about 0.2, and counting pages that lost visibility as misses the test month's P@50 falls from {f2(may['p50'])} to {f2(may['p50_unm'])}.
5. **One recommendation (60 s).** Protect the pages earning more than their rank normally earns, verify visibility first, and hold out 10 a month
   to measure it.
"""), md(f"""
**ML-12 · Social post (methodology cut)**

> I spent my FlyRank ML internship learning that the target matters more than the model, and then that process matters more than the score. My first study's label was
> mostly answered by momentum, so a two-line rule beat every model. For my final project I asked a newer question:
> which pages are losing Google clicks at the rank they already hold, the pattern AI answers would cause? I rebuilt the pipeline so that four model
> families compete under a rule I wrote down before running anything, with a test month scored once. No challenger beat the simple logistic model, the ranking
> still beat every rule on average precision, and the honest caveats are in the open: about five effective clients, and pages that lost visibility that the label cannot score.
> Predictive, not causal, and decision support for editors.
"""), md(f"""
**ML-12 · Employer summary (three sentences)**

I built a monthly early-warning ranking that tells a content team which of their top-ranking pages are about to lose
clicks to AI-answer search, on {mon['daily_rows'] / 1e6:.1f} million rows of pseudonymized Search Console and GA4 data across up to {max(r['gsc_clients'] for r in mon['months'])} clients
from FlyRank's internship warehouse. I showed with page fixed-effects regression that click-through at a fixed position
fell by about a third at the top three positions in six months, and I designed a leak-free label, frozen baselines, a pre-registered pipeline with four candidate models, a confirmation fold and a test month scored once.
The champion had a higher average precision than every frozen rule in all four evaluated months, but its intervals are wide and a conservative check that counts pages that lost visibility as misses lowers its precision, so I report it as decision support and turned it into a reviewed playbook with reason codes, guardrails and retrain triggers.
"""), selfcheck("capstone", (True, True, True, True, True, True, True))]

NOTEBOOKS = {"w01_research_question.ipynb": w01, "w02_ml_task_framing.ipynb": w02, "w03_data_contract.ipynb": w03,
             "w03_feature_leakage_check.ipynb": w03b, "w04_signal_audit.ipynb": w04s, "w04_baseline_score.ipynb": w04,
             "w05_model.ipynb": w05, "w06_validation_audit.ipynb": w06, "w07_action_playbook.ipynb": w07, "capstone.ipynb": cap}

if __name__ == "__main__":
    os.makedirs(NB, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        write(name, cells)
        print("wrote", name, len(cells), "cells")
