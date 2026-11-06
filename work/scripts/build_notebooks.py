"""Writes the submission notebooks (template names, one per assignment card).

Every notebook keeps FlyRank's original template structure (title, numbered headings, closing Self-check; see nb_template.py).
Inside each section the analysis follows the same discipline:
  Question -> Why it matters (the FlyRank principle it serves) -> Method -> Result -> What it means.
Numbers in prose are the committed receipts in work/outputs/*.json; code cells recompute them when
the pinned warehouse snapshot is available and otherwise read the receipts.
Run:  python work/scripts/build_notebooks.py   then execute with nbclient (see run_notebooks.py).
"""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
NB = HERE + "/../notebooks"
REPO = "Tessa-Saumu/FlyRank-ML-Internship"


def badge(name):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/{REPO}/blob/main/work/notebooks/{name}?flush_cache=true)")


SETUP = '''# Setup: works locally and in Colab. Installs only missing packages and never prints a token.
import importlib.util, json, os, sys
if any(importlib.util.find_spec(p) is None for p in ["duckdb", "huggingface_hub", "sklearn", "scipy", "matplotlib"]):
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
DATA = wh.data_available(try_download=True)   # needs HF_TOKEN as an env var or Colab secret
print("pinned warehouse snapshot available:", DATA, "| revision", wh.REV[:8])
receipt = lambda name: json.load(open(os.path.join(OUT, name)))
pd.set_option("display.width", 170); pd.set_option("display.max_columns", 30)'''

SETUP_OFFLINE = '''# Offline setup: this notebook reads only committed receipts and figures, so it runs anywhere.
import json, os
here = os.path.abspath(os.getcwd())
ROOT = next(p for p in [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
            if os.path.isdir(os.path.join(p, "work", "outputs")))
OUT, FIG = os.path.join(ROOT, "work", "outputs"), os.path.join(ROOT, "work", "figures")
import pandas as pd
from IPython.display import Image, display
receipt = lambda name: json.load(open(os.path.join(OUT, name)))
pd.set_option("display.width", 170)'''

NOTE = ("> **Freestyle submission.** This notebook belongs to my final, freestyle capstone: *which ranking pages are "
        "losing clicks to AI-answer search before it shows, and which should an editor protect first?* My earlier "
        "Lane 2 (refresh) experiments are kept for the record in `work/archive/` and are not part of this submission.")


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


def selfcheck(name, ticks=(True, True, True, True, False)):
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
"""), code('''
mon = receipt("eda_ai_monthly.json")
print(f"Freestyle track on the full warehouse: {mon['daily_rows']:,} daily rows, {mon['first_date']} to {mon['last_date']}.")
'''), sec("w01_research_question", 2, """
**Question:** which decision does this improve, who acts, and what does a wrong call cost? **Why it matters (FlyRank
principle: decision first):** an unactionable ranking is only a chart.

> **For** FlyRank's content team, **deciding** which ranking pages to work on before their clicks disappear, **I build**
> a monthly ranked queue of 50 top-10 pages from the daily Search Console warehouse, **scoring** the probability that a
> page earns at least 25% fewer clicks over the next two months than its current click-through rate and position
> predict. Success is **measured by** precision@50 against the base rate and against frozen rules, on months the model
> never saw. **A wrong call costs** an editor's review (false alarm) or a silent loss of clicks on a page that still
> ranks (miss). **A plain rule isn't enough because** the obvious rules ("top positions are most exposed", "a slipping
> CTR keeps slipping") point the wrong way at page level (ML-07). **I will claim only** observed, measured,
> decision-support results: Search Console has no AI-answer flag, so the pattern is *consistent with* AI answers
> absorbing clicks, not proof of it.

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
3. **Does a learned ranking beat simple rules out of time?** Forward validation with a sealed month (ML-07 to ML-09).
   Only then do I write a playbook (ML-10).
"""), code('''
pb = receipt("erosion_playbook_summary.json")
print({k: pb[k] for k in ["queue_size", "client_cap", "reason_codes"]})
'''), sec("w01_research_question", 3, """
**Why these three numbers:** each answers a doubt a reviewer should have. Is the problem real? Is it large? Why not
model AI referrals directly? All three come from the full warehouse (78.8 million daily rows), not the starter CSV.
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
"""), sec("w01_research_question", 4, """
**Question:** what may this work say, and what must it never say? **Why it matters (FlyRank principle: careful words):**
observational data supports decisions, not causal stories.

- **Observed:** at a fixed page and position, CTR fell after October 2025.
- **Measured:** the queue's precision against the base rate, on months the model never saw.
- **Directional:** transactional pages lost more CTR than informational pages in this panel.
- **Decision-support:** a ranked list that an editor reviews before acting.
- **Never claimed:** that AI answers caused the fall (Search Console has no AI-answer flag), that the model predicts
  Google, or that an action restores clicks (there is no hold-out yet).
"""), code('''
sealed = receipt("erosion_model_metrics_sealed.json")["origins"]["2026-05"]
print(f"Observed: top-3 CTR in April 2026 = {apr['ctr_multiplier']} x October 2025 (95% CI {apr['ci95'][0]} to {apr['ci95'][1]}).")
print(f"Measured: sealed month P@50 = {sealed['metrics']['logreg']['p@50']} against a base rate of {sealed['base_rate']}.")
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
| Comparison | four frozen baselines and random order | the model must beat simple rules, not just chance |
"""), code('''
summ = receipt("erosion_frame_summary.json")
print("task: supervised ranking | review budget K =", receipt("erosion_playbook_summary.json")["queue_size"], "| origins:", [o["origin"] for o in summ["origins"]])
'''), sec("w02_ml_task_framing", 2, """
**Question:** what is the target, and where does the label come from? **Why it matters (FlyRank principle: an observed
outcome, never a rule's output):** a label defined by a rule can be recovered by that rule, which is what ended my Lane 2
study.

| Item | Choice | Why |
|---|---|---|
| Target | `erode = 1` if clicks in months *o* and *o+1* < 0.75 x expected clicks | observed in a later window; nobody's rule defines it |
| Expected clicks | outcome impressions x current CTR (month *o-1*, shrunk) x position adjustment | compares the page with itself at its own rank |

**The timeline (what is known when).**

```
decision on day 1 of month o
features : months o-2 and o-1 (plus o-3 for one trend feature)   -> closed before the decision
label    : months o and o+1                                         -> observed after the decision
training : only origins <= o-2, whose labels have matured by month o-1
```

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
'''), md("""
**What it means.** Pure noise would label about 5% of pages as eroding. The observed rate is 23% to 34%, four to seven
times higher, so most positives are real changes in click share at a stable rank, not counting noise.
"""), sec("w02_ml_task_framing", 3, """
**Question:** which one number means "good"? **Why it matters (FlyRank principle: name the metric before training):** a
metric chosen after seeing results can be bent to flatter the model.

| Item | Choice | Why |
|---|---|---|
| Primary metric | precision@50 per decision month, next to the base rate | 50 is the monthly review budget |
| Secondary | PR-AUC (whole ranking), precision@20/100, within-client precision@10 | P@50 alone has wide intervals (about +/-0.2 at K = 50) |
"""), code('''
b = pd.DataFrame(summ["origins"]).dropna(subset=["base_rate"])[["origin", "base_rate"]]
print("Every P@50 below is read next to the month's base rate:"); display(b)
'''), sec("w02_ml_task_framing", 4, """
**Question:** one row is one what? **Why it matters (FlyRank principle: one explicit grain):** a reviewer should see
exactly what one row is. IDs are pseudonymous and used only for grouping, so they are not shown. The frame is built by
`work/scripts/erosion_frame.py` from the cached page-month table.
"""), code('''
summ = receipt("erosion_frame_summary.json")
display(pd.DataFrame(summ["origins"]))
if DATA:
    import erosion_frame
    if not os.path.exists(wh.CACHE + "/erosion_frame.parquet"):
        erosion_frame.build_frame()
    f = pd.read_parquet(wh.CACHE + "/erosion_frame.parquet")
    cols = ["origin", "imp_f", "clk_f", "pos_f", "pos_m1", "ctr_now", "imp_out", "clk_out", "pos_out", "exp_out", "erode"]
    display(f.loc[f.origin == "2026-03", cols].sample(6, random_state=1).round(4))
    print("rows per origin:", f.groupby("origin").size().to_dict())
'''), md("""
**Reading the table.** Each row is one top-10 page at one decision month, with at least 200 impressions and 20 clicks
in the two feature months. `exp_out` is what the page should earn in the next two months if its current CTR held at its
new position. `erode` compares what it actually earned against that expectation. The sealed origin (2026-05) and the
live origin (2026-07) have no base rate in the receipt: the first stays hidden until the end, the second has no outcome yet.
"""), sec("w02_ml_task_framing", 5, """
**Question:** why not a fixed rule? **Why it matters (FlyRank principle: the solution hierarchy):** a model must earn its
place over a transparent rule.

- **Erosion is a combination, not one signal.** Rank, a recent CTR bump, CTR above what the position normally earns,
  demand swings and client-wide shifts all matter together (ML-06, ML-08).
- **The intuitive rules point the wrong way.** In the training months, top-3 pages eroded *less* than positions 7 to 10,
  and pages whose CTR had just risen eroded *more* than pages already slipping (ML-07).
- **The best simple rule changes every month** (ML-08), so no single if-statement holds.

The output feeds a real content action. ML-10 maps each reason code to an editor action, with human review before any
change.
"""), code('''
dev = receipt("erosion_model_metrics.json")["origins"]
print("best frozen baseline per development month:", {o: v["best_baseline"] for o, v in dev.items()})
'''), selfcheck("w02_ml_task_framing")]

# ------------------------------------------------------------------------------------------- ML-04
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
   Origins run December 2025 to May 2026, after Google's September 2025 change that removed deep-result impressions.
4. **Label or proxy?** Observed: clicks in the outcome window below 75% of the clicks the page's current CTR predicts at
   its new position.
5. **One exclusion.** Pages whose expected outcome clicks are under 10 (4% to 13% per origin, mostly pages that lost
   visibility) are excluded because their label cannot be measured. This uses outcome-window impressions, so it is a
   disclosed population choice.
"""), code('''
summ = receipt("erosion_frame_summary.json")
display(pd.DataFrame(summ["origins"])[["origin", "eligible", "measurable", "clients"]])
'''), sec("w03_data_contract", 2, """
**Question:** which fields do I touch, and in which bucket? **Why it matters (FlyRank principle: nothing from the future
in a feature):** a field in the wrong bucket is how leaks start.

| Bucket | Fields | Note |
|---|---|---|
| **Feature** | the five core features below, plus the 34 engineered features of ML-08 | all computed from months before the decision |
| **Label** | `erode` | observed in the two months after the decision |
| **Context** | `origin` (decision month), `client_hash_id` (grouping and splits only), `content_hash_id` (joins and tie-breaks only) | pseudonymous; never displayed or used as a feature |
| **Excluded** | `clk_out`, `imp_out`, `pos_out`, `exp_out` (outcome-window columns, the label's own ingredients); `provider_used`; the 90-day query table; `model_used`; `last_optimized_date` | why: see below |

**Why each exclusion.**
- **Outcome-window columns** are the label's ingredients, so as features they would be the answer (the leak demo in
  section 3 shows it).
- **`provider_used`** contains strings that look like client brand names in some rows. It is never selected anywhere in
  this project (privacy rule).
- **The 90-day query table** covers April to June 2026, which overlaps every outcome window.
- **`model_used`** records the latest regeneration of a page, not its author, so it is not a valid treatment.
- **`last_optimized_date`** covers 8.7% of pages, all dated after 24 April 2026, which is too little follow-up time.

**Five core features, each knowable at the decision moment** (the full model uses 34 engineered features, ML-08):

| Feature | Meaning | Knowable at the decision moment because... |
|---|---|---|
| `l_imp_f` | log impressions over months *o-2* and *o-1* | both months close before day 1 of month *o* |
| `pos_f` | impression-weighted average position, same window | same window, computed from `gsc_sum_position / impressions` |
| `ctr_vs_ref` | log of shrunk CTR over the fixed position curve | the curve uses Oct-Nov 2025 only, before every outcome window |
| `ctr_growth` | log CTR change from month *o-2* to month *o-1* | both months are closed |
| `page_prev_ratio` | the page's own clicks-vs-expected two origins ago | that origin's outcome window ended at month *o-1* |
"""), code('''
feats = receipt("erosion_features.json")
five = ["l_imp_f", "pos_f", "ctr_vs_ref", "ctr_growth", "page_prev_ratio"]
print("the five core features are in the exported feature list:", {f: f in feats for f in five})
print("label ingredients present in the feature list (must be empty):", [c for c in ["imp_out", "clk_out", "pos_out", "exp_out", "erode"] if c in feats])
'''), sec("w03_data_contract", 3, """
**Question:** does the contract hold on the real data? **Why it matters (FlyRank principle: a claim without a query is a
guess):** three checks prove the grain, the size and date span, and the availability handling before anything is built on
them. **Method:** exactly three queries on a mid-panel month, March 2026, away from the sealed final months.
"""), code('''
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
    contract = {"q1_duplicate_grain_rows": len(q1), "q2": q2.iloc[0].to_dict(), "q3": {k: int(v) for k, v in q3.iloc[0].items()}}
    json.dump(contract, open(os.path.join(OUT, "data_contract.json"), "w"), indent=1, default=str)
contract = receipt("data_contract.json")
print("Q1 grain probe, duplicate (report_date, content) rows:", contract["q1_duplicate_grain_rows"], "(must be 0)")
print("Q2 size and span:", contract["q2"])
print("Q3 availability flags (three-valued, so filter with IS TRUE):", contract["q3"])
'''), md("""
**What it means.** The grain holds (zero duplicates), March has the expected one-month span, and the availability flags
contain NULLs as well as TRUE and FALSE. That is why every query in this project filters with `IS TRUE`, and why GA4
columns are summed only on days where `ga4_data_available IS TRUE` (earlier days are zero-filled, and zero there does
not mean "no engagement").

**A deliberate leak, shown and then removed** (a model check, not a fourth query). **Why:** to prove the evaluation can
see a leak. I add the outcome-window clicks as a "feature" (the label's own ingredient), train on December to January,
and score March.
"""), code('''
if DATA:
    from erosion_model import load, make_features, lr_model, p_at_k, tiebreak
    from sklearn.metrics import average_precision_score
    df, grid, vals = load(); df = df[df.measurable.fillna(False)].reset_index(drop=True)
    X5 = make_features(df, grid, vals)[["l_imp_f", "pos_f", "ctr_vs_ref", "ctr_growth", "page_prev_ratio"]]
    y = df.erode.to_numpy().astype(int)
    tr, te = df.origin.isin(["2025-12", "2026-01"]).to_numpy(), (df.origin == "2026-03").to_numpy()
    tb = tiebreak(df.content_hash_id[te]); res = {}
    for name, X in [("five_features", X5), ("with_leak_clk_out", X5.assign(leak_clk_out=np.log1p(df.clk_out.fillna(0)) - np.log1p(df.exp_out)))]:
        cols = [c for c in X.columns if X.loc[tr, c].notna().any()]
        s = lr_model(0.1).fit(X.loc[tr, cols], y[tr]).predict_proba(X.loc[te, cols])[:, 1]
        res[name] = {"p@50": round(p_at_k(y[te], s + tb, 50), 3), "pr_auc": round(float(average_precision_score(y[te], s)), 3)}
    res["base_rate_2026-03"] = round(float(y[te].mean()), 3)
    contract = receipt("data_contract.json"); contract["leak_demo"] = res
    json.dump(contract, open(os.path.join(OUT, "data_contract.json"), "w"), indent=1, default=str)
display(pd.DataFrame(receipt("data_contract.json")["leak_demo"]).T)
'''), md("""
**What it means.** With the outcome's own ingredient as a feature, the ranking becomes perfect. That is the signature
of a leak, not skill. With it removed, these five features alone rank only slightly above the base rate (P@50 0.36
against 0.338). The core idea is real but weak on its own, which is why ML-08 engineers 34 features: feature
engineering, not model choice, is where this study earns its signal. The leaky column is never used again. Feature timing is asserted in code (`erosion_frame.py` builds features only from months
before the origin).
"""), sec("w03_data_contract", 4, """
**Question:** what can this data never tell me? **Why it matters (FlyRank principle: careful words):** a limit stated up
front keeps a claim from outrunning its data.

- **No AI-answer flag.** Search Console does not record whether an AI answer appeared for a query. So this study
  measures click loss at a stable rank, which is **consistent with** AI answers absorbing clicks. Seasonality, changes
  to other search features and impression inflation are competing explanations. ML-06 tests the ones the data allows.
- **An unbalanced panel.** Search Console clients grow from 2 to 65 over 17 months, so any pooled trend mixes in
  onboarding. Trends are estimated within page or within client.
- **GA4 starts late.** GA4 (and with it AI-referral tracking) starts in October 2025 with 4 clients. Earlier days are
  zero-filled, so zero there does not mean "no engagement".
- **A measurement change.** Google's September 2025 removal of deep-result impressions shifts impressions toward the top
  10, so every window starts in October 2025 or later.
- **A selected population.** Only top-10 pages with enough clicks, and only pages whose outcome is measurable (4% to 13%
  per origin dropped), so the results describe pages that still rank.
"""), code('''
mon = pd.DataFrame(receipt("eda_ai_monthly.json")["months"])
display(mon[["month", "gsc_clients", "ga4_clients"]])
'''), selfcheck("w03_data_contract")]

# ------------------------------------------------------------------------------------------- ML-05
w03b = [head("w03_feature_leakage_check", """
**The principle it serves:** attack your own model before anyone else does. A score that looks too good is
investigated, not celebrated. This notebook builds the feature vector, documents each feature, and records the three
attacks that changed the study.
"""), code(SETUP), sec("w03_feature_leakage_check", 1, """
**Question:** what exactly goes into the model? **Why it matters (FlyRank principle: every feature must exist before the
moment I predict):** the vector is the only thing the model sees. **Method:** one function builds 34 features per page
and origin from months *o-3* to *o-1*; the list is exported from code so this notebook, the paper and the model share it.
"""), code('''
feats = receipt("erosion_features.json")
print(f"{len(feats)} features exported by the model code")
if DATA:
    from erosion_model import load, make_features
    df, grid, vals = load(); X = make_features(df, grid, vals)
    print("feature matrix:", X.shape, "| columns match the exported list:", set(X.columns) == set(feats))
'''), sec("w03_feature_leakage_check", 2, """
**Question:** for each feature, what does it mean, how are missing values handled, which are categorical, and does it
exist before the decision? **Why it matters (FlyRank principle: null is not zero):** a silent fill can fake a signal.

- **Available when:** every feature is computed from months *o-3* to *o-1*, or from labels whose outcome window closed
  by *o-1* (`client_prev_erode_rate`, `page_prev_ratio`). Nothing is read from month *o* or later.
- **Missing values.** Missingness is structural (no GA4 for the client, no matured history yet). The logistic model
  fills with the median and adds a missing-indicator column; gradient boosting handles missing values natively.
  `l_org_sess` is NaN, not zero, when no GA4 day exists, and `has_ga4` carries that fact.
- **Categorical.** Main intent is one-hot (`intent_informational`, `intent_transactional`, `intent_commercial`,
  `intent_navigational`), and content type is a flag (`is_feedly`).
"""), code('''
fam = {"l_imp_f": "exposure", "l_clk_f": "exposure", "imp_growth": "trajectory", "ctr_growth": "trajectory", "imp_growth_prev": "trajectory",
       "imp_l14_share": "trajectory", "ctr_recency": "trajectory", "click_day_share": "stability", "spikiness": "stability",
       "pos_f": "rank", "pos_delta": "rank", "pos_std": "rank", "ctr_vs_ref": "CTR vs expectation", "ctr_vs_client": "CTR vs expectation"}
t = pd.DataFrame([{"feature": k, "family": fam.get(k, "client context" if k.startswith(("c_", "client")) else "page history" if k.startswith("page_")
                   else "channels (GA4)" if k in ("has_ga4", "l_org_sess", "has_ai_referral") else "content"), "meaning": v} for k, v in feats.items()])
display(t)
'''), sec("w03_feature_leakage_check", 3, """
**Question:** can the label be predicted from momentum that is already visible, and does my harness see a planted leak?
**Why it matters:** that failure ended my Lane 2 study (a two-line momentum rule beat every model). **Method:** three
label versions, each run only on the three development months. The sealed month stayed closed.

**Attack 1: the label leaked twice, and the models looked better while it did.**
"""), code('''
log = receipt("label_design_log.json")
rows = []
for v, d in log["versions"].items():
    if "p50" in d:
        for m, vals in d["p50"].items():
            rows.append({"version": v, "ranker": m, **dict(zip(["2026-02", "2026-03", "2026-04"], vals))})
dev = receipt("erosion_model_metrics.json")["origins"]
for m in ["rule_position_slipping", "logreg", "hgb"]:
    rows.append({"version": "v3_final", "ranker": m, **{o: dev[o]["metrics"][m]["p@50"] for o in dev}})
display(pd.DataFrame(rows))
for v, d in log["versions"].items():
    print(f"{v}: {d.get('problem_found', 'final version')}")
'''), md("""
**What it means.** The leaky versions made the learned models look much better (gradient boosting reached 0.90 at 50
pages). After the fix, every model score dropped. That drop is the evidence that the leak was real, and the honest
lower numbers are the ones this submission claims.

**Attack 2: does the harness see a planted leak?** **Method:** add a noisy copy of the outcome ratio as a feature on the
April forward split. If precision does not jump toward 1.0, the harness itself is broken.
"""), code('''
lk = receipt("erosion_leakage_checks.json")
display(pd.DataFrame({k: v for k, v in lk.items() if k.startswith("forward")}).T)
'''), md("""
**What it means.** Precision@50 goes from 0.66 to 1.00 with the planted feature, so the evaluation can see a leak. The
honest model has no such feature.

**Attack 3: does the split decide the answer?** **Question:** how much does a random row split flatter the model?
**Why it matters:** pages from one client share hidden character, so a random split lets the model recognise the client
instead of learning erosion.
"""), code('''
split = pd.DataFrame({k: lk[k] for k in ["random_row_kfold", "client_group_kfold"]}).T
display(split)
display(pd.DataFrame(lk["unseen_client_within_origin"]).T.assign(lift=lambda d: (d.ap_mean / d.base_mean).round(2)))
'''), md("""
**What it means.** A random row split reports PR-AUC 0.538. Holding whole clients out gives 0.422. The 0.116 gap is
client memorisation, so every claim in this submission uses forward months and reports held-out-client checks. Within
every month, the model still ranks pages of clients it never saw above the base rate (lift column).

**Attack checklist.**

| Check | Status | Where |
|---|---|---|
| Timeline drawn: all features strictly before the label window | done | ML-03, `erosion_frame.py` |
| No label-derived or sibling columns in the features | done; planted-leak test above | ML-04, attack 2 here |
| No product flags or existing-system scores as features | done; the warehouse release has none, `provider_used` excluded | ML-04 |
| Population selection checked for outcome-window information | disclosed: measurability uses outcome impressions | ML-04 |
| Split grouped by the repeating entity and by time | forward origins plus held-out clients | ML-08, ML-09 |
| Base rate printed next to every metric | done | all receipts |
| Top feature importance sanity-checked | done; the top driver is explained, not celebrated | ML-08 |
| Metrics out of sample | forward only; tuning inside training months | ML-08 |
| Sealed holdout leaves receipts | `erosion_freeze.json`, `erosion_model_metrics_sealed.json` | ML-09 |
"""), sec("w03_feature_leakage_check", 4, """
**Question:** which fields did I refuse to use, and why? **Why it matters (FlyRank principle: privacy and leakage are
decided before modelling):** an exclusion with a reason can be audited.

| Excluded field | Why |
|---|---|
| `clk_out`, `imp_out`, `pos_out`, `exp_out` | outcome-window columns, the label's own ingredients (attack 2 shows what they do) |
| `provider_used` | contains strings that look like client brand names; never selected |
| the 90-day query table | covers April to June 2026, which overlaps every outcome window |
| product flags and existing-system scores | the warehouse release has none |
| `model_used`, `last_optimized_date` | not a valid treatment (latest regeneration, not author) and too little follow-up time (8.7% of pages, all after 24 April 2026) |
"""), code('''
banned = ["imp_out", "clk_out", "pos_out", "exp_out", "erode", "provider_used", "model_used", "last_optimized_date"]
print("excluded fields present in the 34-feature list (must be empty):", [c for c in banned if c in feats])
'''), selfcheck("w03_feature_leakage_check")]

# ------------------------------------------------------------------------------------------- ML-06
w04s = [head("w04_signal_audit", """
**Why this notebook is long, and why every section is here.** Each section answers one question that changes a design
decision later. Nothing here is description for its own sake. The pattern is always: question, why it matters, method,
result, and what it changes. The unit of evidence is the **client**, because pages of one client share shocks: thousands
of pages from one site are not thousands of independent proofs.
"""), code(SETUP), md("""
**Recompute or read.** With the snapshot available, the cell below rebuilds the cached page-month table if needed and
reruns the EDA scripts. Their receipts are committed, so the rest of the notebook reads the same numbers either way.
"""), code('''
if DATA:
    import glob, importlib
    if not glob.glob(wh.CACHE + "/page_month/*.parquet"):
        import build_page_month  # noqa: F401  (about 7 minutes once)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):   # the scripts print long tables; the receipts are read below
        for mod in ["eda_00_panel", "eda_01_landscape", "eda_02_signals", "eda_03_inference", "eda_04_client_level", "panel_fe"]:
            m = importlib.import_module(mod)
            if hasattr(m, "main"):
                m.main()
print("receipts:", sorted(f for f in os.listdir(OUT) if f.startswith(("eda", "panel"))))
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
'''), sec("w04_signal_audit", 4, """
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
"""), code('''
auc = {r["feature"]: r["within_client_auc"] for r in e3["B_onset_signal_audit"]}
print("headline numbers behind the verdicts:")
print("  top-3 CTR, April 2026 vs October 2025:", next(r["ctr_multiplier"] for r in fe["pos_1_3"]["effects"] if r["month"] == "2026-04"))
print("  within-client AUC by single signal:", auc)
'''), selfcheck("w04_signal_audit")]

# ------------------------------------------------------------------------------------------- ML-07
w04 = [head("w04_baseline_score", """
**What this card asks:** two signal checks with bucket tables, n and a verdict; ONE rule with a score, one reason code
and an action label, written to `work/outputs/baseline_action_score.csv`; and a review of the top 20. **The principle it
serves:** the solution hierarchy. A transparent rule comes first, and any model must beat it on the same rows.
"""), code(SETUP), sec("w04_baseline_score", 1, """
**Question:** what rule would an editor write today, and which reason codes can it output? **Why it matters (FlyRank
principle: the solution hierarchy):** a transparent rule is the bar any model must clear.

**Rule:** `score = position tier (top 3 = 2, positions 4-6 = 1, 7-10 = 0) + max(0, -CTR change from month -2 to -1)`,
with a deterministic hash tiebreak. It encodes two common intuitions: AI answers sit at the top, so top positions are
most exposed, and a CTR that has started slipping keeps slipping.

**Reason codes it can output:** `TOP3_SLIPPING` (top-3 position and CTR down more than 5% last month), `TOP3_EXPOSED`
(top-3 position, CTR not falling) and `TOP10_SLIPPING` (positions 4 to 10). The action label is `review now`.

I froze the rule **before** looking at the checks, on purpose. It represents the status-quo heuristic, so it must not be
fitted to the data. The checks then test whether its two intuitions hold.

**Two signal checks (training-era months, December and January).**
"""), code('''
if DATA:
    from erosion_model import load, make_features
    df, grid, vals = load(); df = df[df.measurable.fillna(False)].reset_index(drop=True)
    X = make_features(df, grid, vals)
    d = pd.DataFrame({"origin": df.origin, "y": df.erode, "pos": X.pos_f, "g": X.ctr_growth})
    d = d[d.origin != "2026-05"]
    d["era"] = np.where(d.origin.isin(["2025-12", "2026-01"]), "design_dec_jan", "later_feb_apr")
    d["tier"] = pd.cut(d.pos, [0, 3.5, 6.5, 10.5], labels=["1-3", "4-6", "7-10"])
    d["ctr_change"] = pd.cut(d.g, [-9, -0.3, -0.05, 0.05, 0.3, 9], labels=["fell >26%", "fell 5-26%", "flat", "rose 5-35%", "rose >35%"])
    out = {}
    for key, col in [("check1_position_tier", "tier"), ("check2_recent_ctr_change", "ctr_change")]:
        t = d.groupby(["era", col], observed=True).y.agg(n="size", erosion_rate="mean").round(3).reset_index()
        out[key] = json.loads(t.to_json(orient="records"))
    json.dump(out, open(os.path.join(OUT, "signal_checks.json"), "w"), indent=1)
sc = receipt("signal_checks.json")
for k in sc:
    print(k); display(pd.DataFrame(sc[k]).pivot(index=sc[k] and list(sc[k][0])[1], columns="era", values=["n", "erosion_rate"]))
'''), md("""
**Check 1: are top positions more exposed?** Tied to the position tiers FlyRank's own report uses (CTR by tier: top 3,
positions 4-10). In the design months, top-3 pages eroded at 0.266, positions 4-6 at 0.279 and positions 7-10 at 0.310
(n = 2,917 / 5,929 / 2,317). **Verdict: OPPOSITE.** Over a two-month horizon, top pages are *less* likely to lose
further clicks. The large fall in top-3 CTR in ML-06 is a level shift that has already happened across the panel, not a
page-level warning sign.

**Check 2: does a slipping CTR keep slipping?** Pages whose CTR rose by more than 35% eroded at 0.399. Pages whose CTR
fell by more than 26% eroded at 0.254 (n = 1,452 and 3,958). **Verdict: OPPOSITE.** A recent bump reverts. In the later
months the falling group rises to 0.307 while bumps stay highest (0.417), so the pattern is not fully stable.

**What it means.** Both intuitions in the frozen rule fail the training-era data. I keep the rule as the baseline
anyway, because it is what an editor would plausibly do without a model. That is exactly the comparison the model has
to win. ML-08 also compares a data-informed one-signal rule (CTR above peers), a client-shock rule and a persistence
rule, so the model is never only beating a weak strawman.
"""), sec("w04_baseline_score", 2, """
**Question:** what does the rule produce when it ranks every eligible page? **Why it matters:** a ranked queue with one
reason code per row is what an editor can act on. **Method:** score every July 2026 page, rank, assign the reason code
and write `work/outputs/baseline_action_score.csv` (row level, gitignored).
"""), code('''
if DATA:
    import erosion_playbook; erosion_playbook.main()
bq = pd.read_csv(os.path.join(OUT, "baseline_action_score.csv")) if os.path.exists(os.path.join(OUT, "baseline_action_score.csv")) else None
bm = receipt("baseline_metrics.json")
print("rule:", bm["rule"]); print("reason codes in the July 2026 rule queue:", bm["live_queue_reason_codes"])
if bq is not None:
    print("rows written to work/outputs/baseline_action_score.csv (gitignored):", len(bq))
'''), sec("w04_baseline_score", 3, """
**Question:** do the rule's top 20 make sense to a person? **Why it matters (FlyRank principle: a ranking is only useful
if its top rows are explainable):** for each of the rule's top 20 pages at the July 2026 decision point, I record the
action, the reason code, why the rule picked it, and what would make the pick wrong. Pages are shown by rank, never by ID.
"""), code('''
if DATA:
    from erosion_model import load, make_features, rule_score, tiebreak
    df, grid, vals = load(); X = make_features(df, grid, vals); live = df.origin == "2026-07"
    t = pd.DataFrame({"position": X.pos_f[live].round(1), "ctr_change_pct": ((np.exp(X.ctr_growth[live]) - 1) * 100).round(0),
                      "ctr_vs_peers_pct": ((np.exp(X.ctr_vs_client[live]) - 1) * 100).round(0),
                      "clicks_last_month": df.clk_m1[live].fillna(0).astype(int),
                      "score": rule_score(X[live]) + tiebreak(df.content_hash_id[live])}).sort_values("score", ascending=False).head(20)
    t.insert(0, "rank", range(1, 21))
    t["reason_code"] = np.where(t.position <= 3.5, np.where(t.ctr_change_pct < -5, "TOP3_SLIPPING", "TOP3_EXPOSED"), "TOP10_SLIPPING")
    t["action"] = "review now"
    t["why"] = [f"position {p}, CTR {c:+.0f}% last month" for p, c in zip(t.position, t.ctr_change_pct)]
    t["what_would_make_it_wrong"] = np.where(t.ctr_change_pct < -50, "a tracking gap or demand drop, not AI answers (check impressions and season)",
                                    np.where(t.ctr_vs_peers_pct < 0, "the page already earns less than its position; the fall may be over",
                                             "a recent bump that is now stable; ML-07 check 2 says bumps tend to revert"))
    t.drop(columns="score").to_json(os.path.join(OUT, "baseline_top20_review.json"), orient="records", indent=1)
top20 = pd.read_json(os.path.join(OUT, "baseline_top20_review.json"))
display(top20)
'''), md("""
**What the review shows.** The rule's top 20 are all top-3 pages (positions 0.5 to 3.4) whose CTR fell 94% to 99% in one
month, and every one earns more than its client's peers at that position. A one-month collapse of that size is the kind
of change that tracking gaps or demand drops produce, so the last column sends the editor to impressions and season
before anything else. That is the failure pattern the model needs to avoid.
"""), sec("w04_baseline_score", 4, """
**Question:** which picks look wrong, and did any product flag or future window leak in? **Why it matters (FlyRank
principle: attack your own baseline):** a baseline that is quietly weak or leaky makes any model look better than it is.

**Weak picks.**
- **Likely tracking or demand events.** All 20 reviewed picks carry the note "a tracking gap or demand drop, not AI
  answers", because CTR fell by more than 50% in one month.
- **Repeated rows.** The last three of the 20 share identical values (position, CTR change, and last-month clicks), so
  I would review them together, not as three independent finds.
- **A tie problem.** In three of four forward months all 50 picks share one score, because the CTR-fall term is capped
  and many top-3 pages hit the cap (their CTR collapsed). The hash tiebreak then picks 50 of them, effectively at
  random, so the rule's precision is the erosion rate of that tied group. I keep the rule frozen as written, because
  changing it after seeing results would defeat its purpose, and I report this as the same weakness my Lane 2 rule had.
  The distinct-score column below shows it.
"""), code('''
display(pd.DataFrame(bm["per_origin"]).T)
'''), md("""
**What it means.** The rule ranks above the base rate at 50 pages (0.38 to 0.76), but its whole-ranking quality is
near or below the base rate in two of four months, and on the sealed month its AUC is 0.45, worse than random order.
This is the baseline the model has to beat.

**Leakage check.** The rule reads only two features, the impression-weighted position and the log CTR change between
the two closed feature months. Neither contains an outcome-window value, a product flag (the release has none) or a
future month.
"""), code('''
feats = receipt("erosion_features.json")
print({f: feats[f] for f in ["pos_f", "ctr_growth"]})
print("outcome-window or flag columns among the rule's inputs: none (the rule uses only the two features above)")
'''), selfcheck("w04_baseline_score")]

# ------------------------------------------------------------------------------------------- ML-08
w05 = [head("w05_model", """
**What this card asks:** a method chosen from the menu and explained, compared with the Week-4 baseline on the same
data, split and metric, with features and errors interpreted, ending with a model-vs-baseline table. **The principle it
serves:** a complex model must earn its place over a frozen baseline under identical conditions. Complexity alone earns
nothing.
"""), code(SETUP), sec("w05_model", 1, """
**Question:** which method from the toolkit fits this lane? **Why it matters (FlyRank principle: a complex model must
earn its place):** the choice should follow the data, not fashion.

- **Regularised logistic regression.** Readable coefficients, stable on small training sets, and every coefficient is
  checkable against ML-06.
- **Histogram gradient boosting.** Captures interactions and handles missing values natively. Missingness here is
  structural (no GA4, no matured history yet), so that matters.
- **Rejected:** random forest (unstable across months in my archived Lane 2 work, see `work/archive/`) and deep
  models (about 50,000 rows, so no case for them).

**Features (34, engineered from the ML-06 findings).**
"""), code('''
feats = receipt("erosion_features.json")
fam = {"l_imp_f": "exposure", "l_clk_f": "exposure", "imp_growth": "trajectory", "ctr_growth": "trajectory", "imp_growth_prev": "trajectory",
       "imp_l14_share": "trajectory", "ctr_recency": "trajectory", "click_day_share": "stability", "spikiness": "stability",
       "pos_f": "rank", "pos_delta": "rank", "pos_std": "rank", "ctr_vs_ref": "CTR vs expectation", "ctr_vs_client": "CTR vs expectation"}
t = pd.DataFrame([{"feature": k, "family": fam.get(k, "client context" if k.startswith(("c_", "client")) else "page history" if k.startswith("page_")
                   else "channels (GA4)" if k in ("has_ga4", "l_org_sess", "has_ai_referral") else "content"), "meaning": v} for k, v in feats.items()])
display(t)
'''), md("""
Every feature is computed from months *o-3* to *o-1*, or from labels whose outcome window closed by *o-1*. The list is
exported from code (`erosion_features.json`), so the paper, this table and the model use the same list.
"""), sec("w05_model", 2, """
**Question:** is the split honest for a "next month, all clients" decision? **Why it matters (FlyRank principle: the
split must match the deployment question):** a random split would let the model recognise clients it has seen.

**Forward origins.** For each forward month *o*, the model trains only on origins <= *o-2*, whose labels have matured by
month *o-1*, and is tested on origin *o*. Intervals come from resampling clients.

**Tuning, nested and honest.** Tuning uses only training months, with an inner GroupKFold by client scored by PR-AUC.
The grid is five values of C for the logistic model and 16 random configurations for boosting. The test month is never
seen during tuning, and the May 2026 origin stays sealed until ML-09.
"""), code('''
if DATA and not os.path.exists(os.path.join(OUT, "erosion_model_metrics.json")):
    import erosion_model; erosion_model.run(erosion_model.DEV_ORIGINS, "")
dev = receipt("erosion_model_metrics.json")["origins"]
display(pd.DataFrame({o: {"n_test": v["n_test"], "test_clients": v["test_clients"], "base_rate": v["base_rate"]} for o, v in dev.items()}).T)
'''), sec("w05_model", 3, """
**Question:** does the model beat the frozen baseline on the same data, split and metric? **Why it matters:** a model
that cannot beat a transparent rule should not ship. **Method:** three development months, six rankers (two models, four
frozen baselines), P@50 with client-bootstrap intervals next to the base rate.
"""), code('''
names = ["logreg", "hgb", "rule_position_slipping", "mean_reversion_ctr_excess", "client_shock", "persistence_prev_ratio"]
rows = []
for o, v in dev.items():
    for n in names:
        m = v["metrics"][n]
        rows.append({"origin": o, "ranker": n, "base_rate": v["base_rate"], "p@50": m["p@50"], "p@50_ci95": v["bootstrap"][n]["p@50_ci"],
                     "pr_auc": m["ap"], "auc": round(m["auc"], 3), "within_client_p@10": round(m["within_client_p@10"], 3)})
display(pd.DataFrame(rows))
print({o: (v["best_baseline"], v["n_test"], v["test_clients"]) for o, v in dev.items()})
'''), md("""
**Reading it.** The base rate is 0.23 to 0.34.
- **Logistic model (P@50):** 0.52, 0.60 and 0.66, about twice the base rate every month.
- **"Top position + slipping" rule (P@50):** 0.46, 0.76 and 0.50. It wins March and loses February and April.
- **Best baseline:** a different one every month (the rule, the rule, then client shock).
- **Whole ranking (PR-AUC):** the logistic model beats every baseline in all three months (0.317, 0.451, 0.378).

The bootstrap intervals (resampling clients) are wide at 50 pages, so single-month P@50 differences are not decisive.
Consistency across months is the signal.

**Model vs baseline (development months).**
"""), code('''
tab = pd.DataFrame({o: {"base rate": v["base_rate"], "rule (frozen) P@50": v["metrics"]["rule_position_slipping"]["p@50"],
                       "best baseline P@50": max(v["metrics"][b]["p@50"] for b in names[2:]), "logistic P@50": v["metrics"]["logreg"]["p@50"],
                       "best baseline PR-AUC": max(v["metrics"][b]["ap"] for b in names[2:]), "logistic PR-AUC": v["metrics"]["logreg"]["ap"]}
                    for o, v in dev.items()}).T
tab.loc["mean"] = tab.mean().round(3)
display(tab)
'''), md("""
**Which model to freeze, and why.**
"""), code('''
fz = receipt("erosion_freeze.json")
print(fz["final_model"], "|", fz["reason"]); print("post-freeze change:", fz["post_freeze_changes"][0]["verification"])
'''), md("""
**Decision.** I froze the logistic model before opening the sealed month:
- it beats the best baseline on PR-AUC in 3 of 3 development months and on P@50 in 2 of 3;
- gradient boosting was unstable when trained on a single month (February);
- gradient boosting also turned out not to be bit-reproducible across reruns on this machine. Its scores moved
  between runs with identical parameters, while logistic regression and every baseline reproduced exactly.

The nested rule-plus-model blend chose a model weight of 1.0 in every month, so no blend is used.

**Said plainly:** gradient boosting scored higher than the logistic model at 50 pages in some months of the run above
(see the table above and the sealed receipt). Those numbers are reported, not claimed: a score that changes
between identical reruns cannot be frozen, and choosing the model with the best test-month score would be tuning on the
test.
"""), sec("w05_model", 4, """
**Question:** what did the model learn, and does it make sense? **Why it matters (FlyRank principle: a score that looks
too good is investigated):** a top driver must be explained, not celebrated.
"""), code('''
sealed = receipt("erosion_model_metrics_sealed.json")
print("logistic coefficients (standardised), model trained for the sealed month:"); display(pd.DataFrame(sealed["logreg_coef_last_origin"]))
print("permutation importance (PR-AUC drop), gradient boosting, sealed month:"); display(pd.Series(sealed["permutation_importance_last_origin"]).to_frame("pr_auc_drop"))
'''), md("""
**Interpretation.** The largest positive weights are `l_imp_f` with a matching negative `l_clk_f` (together: many
impressions per click), `ctr_vs_ref` (the page earns more clicks than its position normally does) and `ctr_growth` (a
recent CTR bump). Read together: **the pages that lose clicks are the ones currently earning more than their rank
normally earns, or riding a recent bump.** That is the opposite of the editor's rule ("slipping keeps slipping") and it
agrees with ML-07 check 2.

**Is this just measurement noise reverting?** No. The one-signal mean-reversion rule (CTR above peers) ranks at about
the base rate (PR-AUC 0.223 to 0.317), and the label already shrinks noisy CTRs. The model's edge comes from combining
the signals, not from any one of them.

**Error notes.** Detailed failure examples are in ML-09. In short, the top of the queue is dominated by pages with large
month-over-month impression swings. Some of these are real erosions, others are demand or tracking events, and the
human-review step in ML-10 exists for exactly those.
"""), selfcheck("w05_model")]

# ------------------------------------------------------------------------------------------- ML-09
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
that still rank in the top 10, which favours survivors. That is a stated limitation.
"""), code('''
summ = receipt("erosion_frame_summary.json")
t = pd.DataFrame(summ["origins"])[["origin", "eligible", "measurable"]]
t["share_dropped_as_unmeasurable"] = (1 - t.measurable / t.eligible).round(3)
print("survivor effect in my own eligibility rule (pages kept vs dropped per origin):"); display(t)
'''), sec("w06_validation_audit", 2, """
**Question:** does the evaluation design change the answer? **Why it matters:** the deployment question is "next month,
for all clients", so only a forward split answers it. **Method:** the same model under three splits.
"""), code('''
lk = receipt("erosion_leakage_checks.json"); dev = receipt("erosion_model_metrics.json")["origins"]
t = pd.DataFrame([
    {"split": "random rows, 5-fold (before)", "pr_auc": lk["random_row_kfold"]["ap_mean"], "base_rate": lk["random_row_kfold"]["base_rate"], "answers": "re-describing clients it has seen"},
    {"split": "clients held out, 5-fold", "pr_auc": lk["client_group_kfold"]["ap_mean"], "base_rate": lk["client_group_kfold"]["base_rate"], "answers": "a new client, same months"},
    {"split": "forward, April 2026 (after)", "pr_auc": lk["forward_2026-04_honest"]["ap"], "base_rate": dev["2026-04"]["base_rate"], "answers": "next month, matured labels only"}])
display(t)
'''), md("""
**Result.** A random row split reports PR-AUC 0.538. Holding clients out gives 0.422, and the forward April test lands
in the same range. The random split's extra 0.12 is client memorisation. Every claim in this submission uses the
forward design.

**The sealed month (scored once, after freezing).** **Design.** Origin May 2026 (outcome May and June 2026, the panel's
final months) was excluded from every development step. I wrote `erosion_freeze.json` (final model, reason, code
hashes), then scored the month once with the frozen logistic model trained on December to March.
"""), code('''
s = receipt("erosion_model_metrics_sealed.json")["origins"]["2026-05"]
rows = [{"ranker": n, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}, "p@50_ci95": s["bootstrap"][n]["p@50_ci"]}
        for n, m in s["metrics"].items() if n != "logreg_rule_blend"]
print(f"sealed origin 2026-05: {s['n_test']:,} pages, {s['test_clients']} clients, base rate {s['base_rate']}, noise floor {s['null_noise_rate']}")
display(pd.DataFrame(rows))
display(Image(os.path.join(FIG, "fig2_forward_validation.png")))
'''), md("""
**Result.**
- **Logistic model:** P@50 0.42 (95% CI 0.28 to 0.70), P@20 0.55, PR-AUC 0.348, AUC 0.658. Within-client precision at
  10 pages is 0.50.
- **Best baseline that month (client shock):** P@50 0.40, but its top 50 come from only 2 clients, and its AUC is 0.434.
- **The rule that won March:** 0.38, with AUC 0.453, worse than random order.

The model beats every baseline on PR-AUC, AUC and within-client precision. At exactly 50 pages its lead over the best
baseline is inside the noise. Gradient boosting, which was not frozen, scored higher at 50 pages on this run; it is
reported in the receipt but not claimed, because it did not reproduce across identical reruns. The sealed month's
noise-only rate is 0.071, so its base rate of 0.24 is about 3.4 times noise, a thinner margin than in the development
months.

**Reproducibility check.** With the snapshot available, the next cell refits the frozen model and asserts that its
P@50 equals the committed receipt.
"""), code('''
if DATA:
    import erosion_analysis
    scored = erosion_analysis.scored_origin("2026-05")
    print("frozen logistic model reproduces the sealed receipt: P@50 =", round(scored.head(50).y.mean(), 2))
    prof = erosion_analysis.error_profile(scored)
    prof.to_json(os.path.join(OUT, "sealed_error_profile.json"), orient="index", indent=1)
    ex = scored.head(50).assign(outcome=lambda d: np.where(d.y == 1, "eroded", "did not erode"))
    ex = ex[["client", "pos_f", "ctr_vs_client", "ctr_growth", "imp_growth", "outcome"]].round(2)
    pd.concat([ex[ex.outcome == "did not erode"].head(4), ex[ex.outcome == "eroded"].head(4)]).to_json(
        os.path.join(OUT, "sealed_error_examples.json"), orient="records", indent=1)
'''), md("""
**Real failure examples.**
"""), code('''
display(pd.read_json(os.path.join(OUT, "sealed_error_profile.json"), orient="index"))
display(pd.read_json(os.path.join(OUT, "sealed_error_examples.json")))
'''), md("""
**What the errors show.**
- **True positives beat their peers' CTR by more** (median ratio 1.17 against 0.69 for false alarms).
- **False alarms cluster in 7 clients; true positives come from 13.**
- **Most pages at the top had data in only one of the two feature months**, with impressions jumping threefold or more.
  Their CTR estimates are volatile.

**What it changes:** two guardrails go into the playbook, a per-client cap and human review of impression jumps. A
pre-registered rule for the next iteration is to require activity in both feature months. I did **not** apply that
rule to the sealed month after seeing it: that would be tuning on the test.
"""), sec("w06_validation_audit", 3, """
**Question:** which leakage routes did I test on the final feature set, and what did each show? **Why it matters
(FlyRank principle: attack your own model):** a leak found by a reviewer costs more than one found by me.

| Risk | Test | Result |
|---|---|---|
| Momentum inside the label | three label versions (ML-05) | fixed; models' scores fell after the fix |
| Label ingredient as a feature | planted outcome feature | P@50 0.66 to 1.00, so the harness detects leaks |
| Client memorisation | random vs grouped split | 0.538 vs 0.422 PR-AUC; claims use forward months |
| Future months in features | timeline assertion in `erosion_frame.py` | features end at month *o-1* |
| Tuning on the test | inner GroupKFold inside training months | sealed month opened once after `erosion_freeze.json` |
| Query table overlaps outcomes | window check (ML-06, query length) | table excluded |
"""), code('''
lk = receipt("erosion_leakage_checks.json")
print("planted-leak harness (April forward split):"); display(pd.DataFrame({k: v for k, v in lk.items() if k.startswith("forward")}).T)
print("random vs client-grouped split:"); display(pd.DataFrame({k: lk[k] for k in ["random_row_kfold", "client_group_kfold"]}).T)
'''), sec("w06_validation_audit", 4, """
**Question:** which of my sentences is boldest, and what does the evidence support? **Why it matters (FlyRank principle:
careful words):** a claim should be no stronger than its design.

| Before (too strong) | After (what the evidence supports) |
|---|---|
| "AI Overviews are stealing FlyRank clients' clicks." | "At a fixed page and position, CTR fell after October 2025 in 10 of 11 clients, a pattern consistent with AI answers absorbing clicks; Search Console has no AI-answer flag, so the cause is not measured." |
| "The model predicts which pages will lose traffic." | "In four forward months, a logistic ranking put 42% to 66% eroding pages in a 50-page queue against base rates of 23% to 34%, and led every baseline on PR-AUC. Decision support for review, not a forecast of Google." |
| "The model beats the rule." | "The model was more consistent than any single rule. At exactly 50 pages its lead over the best rule of the month is within noise." |
"""), code('''
allo = {**receipt("erosion_model_metrics.json")["origins"], **receipt("erosion_model_metrics_sealed.json")["origins"]}
p = [v["metrics"]["logreg"]["p@50"] for v in allo.values()]; b = [v["base_rate"] for v in allo.values()]
print(f"logistic P@50 across the four forward months: {min(p)} to {max(p)}; base rates: {min(b)} to {max(b)}")
'''), selfcheck("w06_validation_audit")]

# ------------------------------------------------------------------------------------------- ML-10
w07 = [head("w07_action_playbook", """
**What this card asks:** ranked actions with reason codes, an archetype-to-action map, the key decay insight, intended
use, limits, human review, cost and value, monitoring and retrain triggers, and what must not be automated. **The
principle it serves:** rules give the reason code, models give the order, and a person decides.
"""), code(SETUP), sec("w07_action_playbook", 1, """
**Question:** what should an editor do first, and why? **Why it matters (FlyRank principle: rules give the reason code,
models give the order):** the queue has to be explainable in words a human trusts.

**The ranked queue for the July 2026 decision.** The frozen logistic model, retrained on every month whose label had
matured (December to May), scores 14,100 eligible pages. Pages younger than 90 days are excluded and no client gets more
than 10 slots. Rows are shown by rank and an anonymous client index. `rank` is the model's rank before the guardrails,
so gaps (for example, no ranks 1 to 3) are pages removed for being younger than 90 days or for exceeding a client's cap.
The full queue with IDs for the editor is written to a gitignored CSV.
"""), code('''
if DATA:
    import erosion_playbook; erosion_playbook.main()
pb = receipt("erosion_playbook_summary.json")
print({k: pb[k] for k in ["decision_origin", "feature_window", "trained_on_origins", "eligible_pages", "excluded_younger_than_90_days",
                          "queue_size", "queue_clients", "client_cap", "reason_codes", "clicks_at_risk_2_months_if_25pct_lost"]})
'''), code('''
q = pd.read_csv(os.path.join(OUT, "erosion_queue_2026-07.csv")) if os.path.exists(os.path.join(OUT, "erosion_queue_2026-07.csv")) else None
if q is not None:
    import erosion_playbook as P
    view = P.capped_top(q[~q.excluded_young]).copy()
    view["client"] = pd.factorize(view.client_hash_id)[0]
    display(view[["rank", "client", "position", "score", "top_driver", "reason_code"]].head(15).reset_index(drop=True))
'''), md("""
**Reason code to action.**

| Reason code (largest model contribution) | What it means | Action |
|---|---|---|
| `CTR_ABOVE_PEERS` | the page earns more clicks than its position normally does | **Protect:** check the live results page for an AI answer; make the page the citable source (answer-first summary, FAQ, structured data) before editing anything else |
| `DEMAND_SHIFT` | impressions swung sharply in the last month | **Check demand first:** seasonality or tracking may explain the change; no rewrite until confirmed |
| `CTR_ALREADY_SLIPPING` | a CTR fall has started | rewrite title and description toward the click an AI answer cannot satisfy (comparison, pricing, tool, example) |
| `TOP_POSITION_EXPOSED` | rank alone drives the score | make the answer citable; monitor |
| `RANK_MOVING` | position is unstable | **hold:** re-score next month |
| `CLIENT_WIDE_DECLINE` | the whole site is losing CTR at a stable rank | escalate to the account lead; page edits alone will not fix it |

**The insight behind the playbook.** The pages that lose clicks next are not the ones already slipping. They are the
ones **currently earning more than their rank normally earns**. That excess click share is what an AI answer can take.
So the first action is protective: make the page the source an AI answer cites. It is not a rewrite after the loss.
"""), sec("w07_action_playbook", 2, """
**Question:** who uses this, for what, and where does it stop being valid? **Why it matters (FlyRank principle: a person
decides):** a tool used outside its intended use can mislead.

- **Who:** a FlyRank content editor, once a month, with a budget of 50 reviews.
- **When:** day 1 of the month, using the two closed months before it.
- **What it is:** a decision-support queue of top-10 pages most likely to lose at least 25% of their expected clicks at
  their current rank in the next two months, each with one reason code and one suggested action.
- **What it is not:** an automatic rewrite, a forecast of Google, or a causal claim that the action restores clicks.
- **Where it stops being valid:** pages outside the top 10, pages younger than 90 days, and any month whose base rate
  leaves the range seen so far (see the monitoring triggers).

**Cost and value.**
"""), code('''
sealed = receipt("erosion_model_metrics_sealed.json")["origins"]["2026-05"]
p_model, base = sealed["metrics"]["logreg"]["p@50"], sealed["base_rate"]
print(f"Sealed month: {p_model*50:.0f} of 50 queued pages eroded with the model vs about {base*50:.0f} for a random 50 pages.")
print(f"Editor time: 50 reviews x about 20 minutes = about 17 hours per month.")
print(f"Clicks at stake in the July queue if each page lost 25% over two months: about {pb['clicks_at_risk_2_months_if_25pct_lost']:,}.")
display(pd.DataFrame(pb["cap_cost_forward_origins"]).T)
'''), md("""
**Reading it.**
- **Value of the model:** on the sealed month, the queue holds about 9 more eroding pages per 50 reviews than random
  selection. That is the time saved.
- **Cost of the client cap:** nothing in the development months (+0.02, +0.02, 0.00). On the sealed month it would have
  raised P@50 from 0.42 to 0.54. I added the cap after unsealing, for fairness of the review budget, so that number is a
  post-hoc observation, not the sealed claim.
"""), sec("w07_action_playbook", 3, """
**Question:** what must a person check before acting, and what should never be automated? **Why it matters (FlyRank
principle: a person decides):** every false alarm costs an edit that can put a page's rank at risk.

**Human review (required before any action).**
1. Open the live results page for the page's main query and record whether an AI answer appears and whether the page is
   cited.
2. If impressions jumped threefold or more, or the page had data in only one month, confirm the demand is real before
   acting (the main false-alarm pattern in ML-09).
3. Respect a 30-day cooldown after any edit, and skip pages younger than 90 days.
4. The editor can override any row. Overrides are logged with a reason, and they become training signal.

**The no-go list (what must not be automated).**
- **Rewriting or publishing content from the queue.** A person reviews the live results page first.
- **Treating a false alarm as harmless.** Every edit risks the rank the page holds.
- **Claiming that an action restored clicks.** That needs a hold-out (for example, leave 10 of the 50 queued pages
  untouched each month and compare), and this study does not have one yet.
- **Client-level conclusions from page queues.** Client-wide declines go to the account lead.
"""), code('''
print(f"guardrails in the July queue: client cap = {pb['client_cap']} slots, pages younger than 90 days excluded = {pb['excluded_younger_than_90_days']:,}")
'''), sec("w07_action_playbook", 4, """
**Question:** what would tell me the recommendations went stale? **Why it matters (FlyRank principle: a model is a
process, not a file):** a ranking trained on past months drifts as Google and the panel change.

| Trigger | Threshold | Response |
|---|---|---|
| A new month's label matures | monthly | retrain on all matured months; re-run the forward check |
| Precision of the last matured queue | below 1.3x that month's base rate | pause the queue and investigate drift |
| Base rate moves | outside 0.15 to 0.45 | check for a Google change or a tracking break before trusting scores |
| Client concentration | one client above 10 slots before the cap | review that client's data feed |
| AI-answer presence | editors record AI answers on reviewed pages | after 3 months, test whether recorded presence adds signal |

**Next iteration (pre-registered).** From the ML-09 error analysis, I fix these before the next live month and test them
on it:
1. **Require activity in both feature months.**
2. **Log editors' AI-answer observations as a feature.**
3. **Run the 10-page hold-out to measure the effect of the actions.**
"""), code('''
allo = {**receipt("erosion_model_metrics.json")["origins"], **receipt("erosion_model_metrics_sealed.json")["origins"]}
print("precision-to-base-rate ratio of each forward month's queue (the trigger fires below 1.3):")
display(pd.Series({o: round(v["metrics"]["logreg"]["p@50"] / v["base_rate"], 2) for o, v in allo.items()}).to_frame("p@50 / base rate"))
'''), sec("w07_action_playbook", 5, """
**Question:** which files does the paper build on? **Why it matters (FlyRank principle: every number traces to a
receipt):** the paper and the capstone notebook read these files, so they must exist and match.

`work/outputs/erosion_queue_2026-07.csv` (gitignored, row level), `erosion_playbook_summary.json`,
`baseline_action_score.csv` (gitignored), `baseline_metrics.json` and the figures in `work/figures/` feed the paper.
"""), code('''
for f in ["erosion_playbook_summary.json", "baseline_metrics.json", "erosion_queue_2026-07.csv", "baseline_action_score.csv"]:
    print(f"{f:34s} exists: {os.path.exists(os.path.join(OUT, f))}")
print("figures:", sorted(f for f in os.listdir(FIG) if f.endswith(".png")))
'''), selfcheck("w07_action_playbook")]

# ------------------------------------------------------------------------------------------- ML-11 + ML-12
cap = [head("capstone", """
**Paper title:** Same rank, fewer clicks: an early warning for click loss to AI answers.

This notebook mirrors the paper section by section, and reads only committed receipts and figures, so it runs without
warehouse access (`work/scripts/verify_evidence.py` executes it offline). The analysis itself lives in the assignment
notebooks it links to.
"""), code(SETUP_OFFLINE), sec("capstone", 1, """
**Abstract.** FlyRank's content pages are losing clicks in Google at the rank they already hold. On the same page in the
same position band, top-3 click-through in April 2026 was 0.642 times its October 2025 level (95% CI 0.615 to 0.669, 35
clients, page fixed effects with client-clustered errors). I asked whether a model can tell an editor which top-10 pages
will lose at least a quarter of their expected clicks in the next two months, so they can be protected first. On four
forward months, including one sealed month scored once after freezing, a logistic ranking of 34 engineered features put
42% to 66% eroding pages into a 50-page queue against base rates of 23% to 34%. It led every frozen baseline on PR-AUC
in all four months, while the best simple rule changed from month to month. The results are observational and
decision-support only: Search Console has no AI-answer flag.

**Introduction.** AI answers in search are the new variable in FlyRank's work. FlyRank's own *State of AI-Driven SEO 2026*
report measures AI referral sessions but not what AI answers do to Google clicks at a given rank. My Lane 2 study
(archived) showed that a target answered by visible momentum lets a two-line rule beat any model. This study therefore
starts from a target that momentum does not give away, and from the decision an editor faces: which 50 pages to protect
this month.
"""), code('''
fe, allo = receipt("panel_fe.json"), {**receipt("erosion_model_metrics.json")["origins"], **receipt("erosion_model_metrics_sealed.json")["origins"]}
r = next(e for e in fe["pos_1_3"]["effects"] if e["month"] == "2026-04")
p = [v["metrics"]["logreg"]["p@50"] for v in allo.values()]; b = [v["base_rate"] for v in allo.values()]
print(f"Abstract numbers: top-3 CTR {r['ctr_multiplier']} x October 2025 (95% CI {r['ci95'][0]} to {r['ci95'][1]}, {fe['pos_1_3']['clients']} clients); "
      f"logistic P@50 {min(p)} to {max(p)} against base rates {min(b)} to {max(b)}.")
'''), sec("capstone", 2, ""), code('''
mon, summ = receipt("eda_ai_monthly.json"), receipt("erosion_frame_summary.json")
print(f"Warehouse build v20260703, revision {summ['warehouse_revision'][:8]}: {mon['daily_rows']:,} daily rows, {mon['first_date']} to {mon['last_date']}.")
display(pd.DataFrame(summ["origins"])[["origin", "eligible", "measurable", "clients", "base_rate", "null_rate", "sealed", "live_no_outcome"]])
'''), md("""
Tables used: the daily performance fact table, `dim_content` and `dim_clients`. Excluded: pages outside the top 10, pages
with fewer than 200 impressions or 20 clicks in the feature window, pages whose expected outcome clicks are under 10,
the 90-day query table (its window overlaps every outcome), and the `provider_used` column (it contains brand-like
strings). Rate columns are percentages, GA4 is used only on days flagged `IS TRUE`, and no data enters git.
"""), sec("capstone", 3, """
- **Unit:** page x monthly decision point. Features use the two closed months before the decision; the label uses the
  two months after.
- **Label:** clicks below 75% of what the page's current CTR (latest month, shrunk toward its client's position-adjusted
  CTR) predicts at its new position. A Poisson simulation puts the noise-only rate at about 5% in the development
  months (7% in the sealed month), against base rates of 23% to 34%.
- **Features:** 34, exported from code (`erosion_features.json`): exposure, trajectory, rank, CTR against expectation,
  content, GA4 channels, client context, page history.
- **Baselines (frozen):** the editor's rule (top position + slipping CTR), CTR above peers, client shock, persistence,
  and random order.
- **Models:** logistic regression and gradient boosting. Tuning uses an inner GroupKFold by client inside training
  months only.
- **Validation:** forward months (training only on origins whose labels matured), client-bootstrap intervals, a sealed
  month opened once after a freeze receipt, a random vs grouped split comparison, and a planted-leak harness.
"""), code('''
print(len(receipt("erosion_features.json")), "engineered features; frozen model:", receipt("erosion_freeze.json")["final_model"])
print("noise-only label rate by origin:", {o["origin"]: o["null_rate"] for o in summ["origins"] if o.get("null_rate") is not None})
'''), sec("capstone", 4, ""), code('''
fe = receipt("panel_fe.json")
print("Fixed-position CTR, April 2026 vs October 2025:")
for k in ["pos_1_3", "pos_4_10", "intent_transactional", "intent_informational"]:
    r = next(e for e in fe[k]["effects"] if e["month"] == "2026-04")
    print(f"  {k:22s} {r['ctr_multiplier']:.3f}  (95% CI {r['ci95'][0]} to {r['ci95'][1]}; {fe[k]['clients']} clients)")
display(Image(os.path.join(FIG, "fig1_fixed_position_ctr.png")))
'''), code('''
allo = {**receipt("erosion_model_metrics.json")["origins"], **receipt("erosion_model_metrics_sealed.json")["origins"]}
B = ["rule_position_slipping", "mean_reversion_ctr_excess", "client_shock", "persistence_prev_ratio"]
res = pd.DataFrame({o: {"base rate": v["base_rate"], "rule P@50": v["metrics"]["rule_position_slipping"]["p@50"],
                        "best baseline P@50": max(v["metrics"][b]["p@50"] for b in B), "logistic P@50": v["metrics"]["logreg"]["p@50"],
                        "logistic P@50 95% CI": v["bootstrap"]["logreg"]["p@50_ci"],
                        "best baseline PR-AUC": max(v["metrics"][b]["ap"] for b in B), "logistic PR-AUC": v["metrics"]["logreg"]["ap"],
                        "sealed": o == "2026-05"} for o, v in allo.items()}).T
display(res)
display(Image(os.path.join(FIG, "fig2_forward_validation.png")))
display(Image(os.path.join(FIG, "fig3_validation_checks.png")))
'''), md("""
The logistic ranking leads every baseline on PR-AUC in all four forward months. At 50 pages it is about twice the base
rate each month, and it beats the best baseline of that month in three of four, inside wide intervals. On the sealed
month: P@50 0.42 (CI 0.28 to 0.70) against 0.40 for the best baseline and 0.38 for the editor's rule, with PR-AUC 0.348
against 0.234. The editor's rule wins one month and falls below random order (AUC 0.453) on the sealed one. A random
row split would have reported PR-AUC 0.538 instead of 0.422 with clients held out.
"""), sec("capstone", 5, """
- **Predictive, not causal.** Search Console has no AI-answer flag, so the CTR fall is *consistent with* AI answers. It
  could also come from seasonality (December dips everywhere), other search-feature changes or impression inflation.
- **Only four forward months,** and the intervals at 50 pages are wide (about +/-0.2).
- **Eligibility favours survivors:** pages still in the top 10 with enough clicks.
- **Measurability uses outcome impressions:** 4% to 13% of pages are excluded per month.
- **The top of the queue is dominated by pages with sudden impression swings,** the main false-alarm source.
- **The client cap was added after the sealed month was opened.** Its sealed effect is reported only as an observation.
- **AI referral findings** rest on a thin, concentrated signal (two clients hold 68% of sessions).
"""), code('''
s = allo["2026-05"]
print("sealed month P@50 95% CI:", s["bootstrap"]["logreg"]["p@50_ci"], "| n pages:", s["n_test"], "| clients:", s["test_clients"])
print("AI referral concentration:", receipt("eda_01_landscape.json")["ai_concentration"][0]["top1pct_share"], "of sessions on the top 1% of pages")
'''), sec("capstone", 6, """
1. **Use the queue for protection, not repair.** The pages to act on are those earning more clicks than their rank
   normally earns. Make them the source an AI answer cites before editing anything else (ML-10).
2. **Treat transactional and commercial pages as the more exposed group in this panel.** Informational pages did not
   show a significant fall.
3. **Do not use Google rank as a proxy for AI visibility.** AI discovery follows exposure (AUC 0.715), not rank (0.506).
4. **Measure the action:** hold out 10 of each month's 50 pages to estimate what the actions are worth.
5. **Retrain monthly, and pause the queue** if a matured month's precision falls below 1.3x its base rate.
"""), code('''
pb = receipt("erosion_playbook_summary.json")
print("July 2026 queue:", pb["queue_size"], "pages | client cap", pb["client_cap"], "| reason codes", pb["reason_codes"])
'''), sec("capstone", 7, """
**Reproducibility.**
- **Code:** `work/scripts/` (one module per stage). Notebooks w01 to w07 recompute every receipt when the pinned
  snapshot is available.
- **Seeds and pins:** seed 42 everywhere, warehouse revision `50cbf7c3`.
- **Environment:** `requirements.txt` plus scipy.
- **Receipts:** `work/outputs/*.json`; figures: `work/figures/`.
- **Offline checker:** `python work/scripts/verify_evidence.py` re-executes this notebook and checks every headline
  number in the paper and report against the receipts.

| Result | Notebook | Receipt |
|---|---|---|
| Fixed-position CTR fall | `w04_signal_audit` | `panel_fe.json`, `eda_04_client_level.json` |
| Label, noise floor | `w02_ml_task_framing` | `erosion_frame_summary.json` |
| Leak demo, contract queries | `w03_data_contract` | `data_contract.json` |
| Label leaks, split gap | `w03_feature_leakage_check` | `label_design_log.json`, `erosion_leakage_checks.json` |
| Baseline and checks | `w04_baseline_score` | `baseline_metrics.json`, `signal_checks.json` |
| Model vs baselines | `w05_model` | `erosion_model_metrics.json`, `erosion_freeze.json` |
| Sealed month, errors | `w06_validation_audit` | `erosion_model_metrics_sealed.json`, `sealed_error_profile.json` |
| Playbook | `w07_action_playbook` | `erosion_playbook_summary.json` |

**Acknowledgments and data credit.** Built on the FlyRank ML Internship dataset ([flyrank.ai](https://flyrank.ai)): the
pseudonymized warehouse release `FlyRank/internship-warehouse` (build v20260703), the starter repository and the
curriculum. Client identities are pseudonymized and none appear here. Errors and interpretations are mine.
"""), code('''
print("figures the paper embeds:", sorted(f for f in os.listdir(FIG) if f.startswith(("fig", "paper"))))
print("receipts read by this notebook:", sorted(f for f in os.listdir(OUT) if f.endswith(".json")))
'''), md("""
**ML-12 · Five-minute demo outline**

1. **Question (45 s).** Editors can review 50 pages a month. Which ranking pages are about to lose clicks to AI answers?
2. **Evidence the problem is real (60 s).** Figure 1: same page, same position, top-3 CTR at 0.64x in six months.
3. **Method (75 s).** An honest label (latest CTR, shrunk, position-adjusted, noise floor 5%), four frozen rules, and
   forward months with a sealed one.
4. **One honest result (60 s).** Figure 2: the model leads every rule on whole-ranking quality in all four months; at 50
   pages it is about twice the base rate, and the best rule changes every month.
5. **One recommendation (60 s).** Protect the pages earning more than their rank normally earns, and hold out 10 a month
   to measure it.
"""), md("""
**ML-12 · Social post (methodology cut)**

> I spent my FlyRank ML internship learning that the target matters more than the model. My first study's label was
> mostly answered by momentum, so a two-line rule beat every model. For my final project I asked a newer question:
> which pages are losing Google clicks at the rank they already hold, the pattern AI answers would cause? The label had
> to be rebuilt three times to stop momentum leaking in, and each fix *lowered* my model's score. The honest version
> still ranked eroding pages at about twice the base rate across four forward months, including one I sealed until the
> end. Predictive, not causal, and decision support for editors.
"""), md("""
**ML-12 · Employer summary (three sentences)**

I built a monthly early-warning ranking that tells a content team which of their top-ranking pages are about to lose
clicks to AI-answer search, on 78.8 million rows of pseudonymized Search Console and GA4 data across up to 66 clients
from FlyRank's internship warehouse. I showed with page fixed-effects regression that click-through at a fixed position
fell by about a third at the top three positions in six months, and designed a leak-free label, frozen baselines, nested tuning and forward
validation with a sealed month. The final model ranked eroding pages at about twice the base rate in every forward month
and led every rule on whole-ranking quality, and I turned it into a reviewed playbook with reason codes, guardrails and
retrain triggers.
"""), selfcheck("capstone", (True, True, True, True, False, False, True))]

NOTEBOOKS = {"w01_research_question.ipynb": w01, "w02_ml_task_framing.ipynb": w02, "w03_data_contract.ipynb": w03,
             "w03_feature_leakage_check.ipynb": w03b, "w04_signal_audit.ipynb": w04s, "w04_baseline_score.ipynb": w04,
             "w05_model.ipynb": w05, "w06_validation_audit.ipynb": w06, "w07_action_playbook.ipynb": w07, "capstone.ipynb": cap}

if __name__ == "__main__":
    os.makedirs(NB, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        write(name, cells)
        print("wrote", name, len(cells), "cells")
