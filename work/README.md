# work/: my capstone submission (freestyle track)

## For the reviewer: what to mark

**Mark the freestyle study only:** the notebooks in `work/notebooks/` (template names, one per assignment card), the
paper (`work/paper/index.html`, deployed URL in `submission/paper_url.txt`) and `work/capstone_report.md`.

**`work/archive/` is not part of the submission.** It keeps my two earlier Lane 2 (refresh) experiments, so the
process that led to this question is on record. Its README explains each one in a few lines.

> **Question.** Which top-ranking pages are losing clicks in Google at the rank they already hold (the pattern AI
> answers would cause), and which 50 should an editor protect first each month?

| | |
|---|---|
| Track | freestyle (lane guide, section 9) |
| Data | FlyRank internship warehouse, build v20260703, revision `50cbf7c3`: 78.8 M daily rows, Jan 2025 to Jun 2026 |
| Finding | same page, same position: top-3 CTR in April 2026 at 0.642x October 2025 (95% CI 0.615 to 0.669, 35 clients) |
| Pipeline | **V2**, pre-registered in `V2_SPEC.md` before any V2 run: 36 features, four frozen baselines, four learned candidates (a logistic model with L2, L1 and elastic-net options, two boosted models, a rank-average ensemble) selected on two validation folds, a confirmation fold that decides nothing, a test month scored once. V1 stays in the repository as the first process |
| Result | no challenger beat the logistic champion; its AP is above all four baselines in 4 of 4 months. Test month: AP 0.333 (95% CI 0.267 to 0.424) against 0.234 for the best baseline; P@50 0.52 against a base rate of 0.24, and 0.34 if pages that later lost visibility count as misses, against 0.38 for the editor's rule (April: 0.16 against 0.50). Counted this way, the editor's rule scores higher than the champion at 50 pages in every month; the champion's queue favours pages that later lose visibility. The April AP gain interval includes 0, and the AP lead is whole-ranking, not P@50 |
| Honest reading | V2 is a process upgrade, not a better result than V1 (test AP 0.333 against V1's 0.348, P@50 0.52 against 0.42, both inside the noise). About five effective clients per month; P@50 differences between rankers are not established. Model choice used AP because it scores the whole ranking and is far less noisy than P@50 at about 5 effective clients (P@50 intervals are roughly plus or minus 0.2); P@50 remains the reported operational metric. |
| Output | a monthly 50-page review queue with reason codes, guardrails and retrain triggers; the July 2026 queue is frozen by its SHA-256 |

**Disclosure.** The test month (May 2026) was seen during V1, so it is not a pristine holdout. No V2 decision (features, models, hyperparameters, selection rule) used it; all were made on validation folds whose outcome windows end before the test window opens. The V2 test result may therefore be optimistic.

## Assignment index

| Notebook | Card | Open | Status |
|---|---|---|---|
| `notebooks/w01_research_question.ipynb` | ML-02 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w01_research_question.ipynb?flush_cache=true) | ☑ |
| `notebooks/w02_ml_task_framing.ipynb` | ML-03 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w02_ml_task_framing.ipynb?flush_cache=true) | ☑ |
| `notebooks/w03_data_contract.ipynb` | ML-04 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w03_data_contract.ipynb?flush_cache=true) | ☑ |
| `notebooks/w03_feature_leakage_check.ipynb` | ML-05, optional stretch: attacking my own label, features and splits | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w03_feature_leakage_check.ipynb?flush_cache=true) | ☑ |
| `notebooks/w04_signal_audit.ipynb` | ML-06, optional stretch: the exploratory and inferential analysis, plus the five new V2 features as signals | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w04_signal_audit.ipynb?flush_cache=true) | ☑ |
| `notebooks/w04_baseline_score.ipynb` | ML-07 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w04_baseline_score.ipynb?flush_cache=true) | ☑ |
| `notebooks/w05_model.ipynb` | ML-08 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w05_model.ipynb?flush_cache=true) | ☑ |
| `notebooks/w06_validation_audit.ipynb` | ML-09 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w06_validation_audit.ipynb?flush_cache=true) | ☑ |
| `notebooks/w07_action_playbook.ipynb` | ML-10 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w07_action_playbook.ipynb?flush_cache=true) | ☑ |
| `notebooks/capstone.ipynb` | ML-11 (the paper mirrors it) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/capstone.ipynb?flush_cache=true) | ☑ |
| closing section of `notebooks/capstone.ipynb` | ML-12: 5-minute demo outline, social-post cut, employer summary | (same notebook, last cells) | ☑ |

Every analysis section in these notebooks follows one pattern: **question, why it matters (the FlyRank principle it
serves), method, result, what it changes.** The notebooks read the committed receipts and run in seconds. Set
`RERUN_PIPELINE = True` in a notebook's setup cell to recompute from the pinned snapshot; the V2 steps then run in a scratch copy, never in the repository.

## Layout

```text
work/
  V2_SPEC.md            the V2 pre-registration (committed and hashed before any V2 run); v2_spec.json is its machine-readable copy
  notebooks/            the submission, one notebook per card (executed, outputs committed)
  scripts/              the pipeline, one module per stage (see below)
  outputs/              JSON receipts every number traces back to (committed); row-level CSVs are gitignored
  figures/              charts for the paper and notebooks (drawn from receipts only)
  paper/index.html      the research paper (deployed to GitHub Pages)
  capstone_report.md    the capstone write-up (from the template, generated from the receipts)
  archive/              earlier Lane 2 experiments, not for marking
```

| Script | Stage |
|---|---|
| `wh.py` | DuckDB views over the pinned, locally cached warehouse snapshot; downloads it once with an `HF_TOKEN` |
| `build_page_month.py` | page x month cache (row level, written to `~/.cache/flyrank_freestyle`, never to git) |
| `eda_00_panel.py` to `eda_04_client_level.py`, `panel_fe.py` | exploratory and inferential analysis (ML-06) |
| `erosion_frame.py` | forward-origin frame, honest label, noise floor (shared by V1 and V2) |
| `erosion_model.py`, `erosion_analysis.py`, `erosion_playbook.py` | the V1 process: features, frozen baselines, nested tuning, sealed evaluation, V1 queue |
| `v2_rhythm.py` | the five new date and rhythm features, computed month by month inside the feature window |
| `v2_models.py` | the logistic (three penalties), boosted classifier and Poisson-rate boosted candidates; single-threaded and seeded |
| `v2_pipeline.py` | V2 selection on the validation folds, champion and challenger rule, confirmation fold, secondary label, freeze, the sealed test step (refuses a second run) |
| `v2_checks.py` | planted-leak harness, random-versus-grouped split check, timeline assertions |
| `v2_playbook.py` | the live July 2026 queue (refuses until the test has been scored); writes the frozen CSV and `v2_playbook_summary.json` |
| `v2_summary.py` | descriptive receipt: the five new features as signals, the ML-04 leak demo, champion coefficients (run after the test; decides nothing) |
| `v2_view.py` | read-only view of the V1 and V2 receipts; every document takes its numbers from here |
| `v2_rerun.py` | opt-in reproduction of the pre-freeze V2 steps in a scratch copy; never runs the unseal or live steps |
| `make_figures.py`, `make_paper.py`, `make_report.py`, `deploy_paper.py` | figures, paper, report, publishing |
| `build_notebooks.py`, `nb_template.py`, `run_notebooks.py` | generate the notebooks on FlyRank's template headings, then execute them |
| `verify_evidence.py` | offline check: runs the capstone notebook, rebuilds the paper and report, matches headline numbers to receipts, confirms the frozen V2 state |

## Reproduce

1. `pip install -r requirements.txt scipy`
2. Read the receipts: `python work/scripts/run_notebooks.py` executes the ten notebooks in card order in under a minute (no data needed).
3. `python work/scripts/verify_evidence.py` checks the receipts, the figures, the paper and the report offline. It never re-runs a V2 step.
4. To recompute from data, set `HF_TOKEN` (a read token) as an environment variable or Colab secret, never typed into a cell. The first run downloads the pinned snapshot (about 1.2 GB) and builds the page-month cache (about 7 minutes).
   Then set `RERUN_PIPELINE = True`, or run `python work/scripts/v2_rerun.py` (about ten minutes, scratch copy). The unseal step is not repeated: it refuses a second run by design.

**Seeds:** 42 everywhere. **Revision:** `50cbf7c3909d07be4d1b5906b4d09e882e5acbf2`.

**Privacy.** Pseudonymous client and content IDs are redacted (`client_[redacted]`, `content_[redacted]`) in every tracked text file, including the saved outputs of the starter notebooks in `notebooks/`; `verify_evidence.py` scans all tracked text files for them.

**One logged deviation.** `V2_SPEC.md` was edited once after it was committed, to add its deviation log (section 13: a column that carries no information in a given boosted fit is left out of that fit only), before any V2 result was used. So its current hash differs from the pre-registered one, and `outputs/v2_freeze.json` records both.

## Notes for FlyRank

- **Data-release issue.** `dim_content.provider_used` carries strings that look like client brand names in some rows. It is never selected in this project, no value appears in any committed file, and it is listed here so the release owner can review it.
- **CI.** GitHub Actions is not running on this repository because of a billing lock on my GitHub account, unrelated to the code. `python work/scripts/verify_evidence.py` is the verification of record (it passed locally on 2026-11-06).

## Rules of the road (from the template, kept)

1. **Copy, don't edit.** `scripts/` at the repository root is the pristine reference pipeline and is untouched.
2. **No datasets in git.** Row-level outputs are gitignored. The JSON receipts are committed.
3. **Stay reproducible.** Fixed seeds, a pinned revision, receipts for every number.
4. **Public-safety language.** Observed, measured, directional, decision-support. No client-identifying details, and no
   causal claims without a design.
