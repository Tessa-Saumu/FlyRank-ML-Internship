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
| Model | logistic ranking on 34 engineered features, frozen before a sealed month |
| Result | P@50 0.42 to 0.66 against base rates of 0.23 to 0.34 over four forward months; leads every frozen baseline on PR-AUC in all four |
| Output | a monthly 50-page review queue with reason codes, guardrails and retrain triggers |

## Assignment index

| Notebook | Card | Open | Status |
|---|---|---|---|
| `notebooks/w01_research_question.ipynb` | ML-02 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w01_research_question.ipynb?flush_cache=true) | ☑ |
| `notebooks/w02_ml_task_framing.ipynb` | ML-03 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w02_ml_task_framing.ipynb?flush_cache=true) | ☑ |
| `notebooks/w03_data_contract.ipynb` | ML-04 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w03_data_contract.ipynb?flush_cache=true) | ☑ |
| `notebooks/w03_feature_leakage_check.ipynb` | ML-05, optional stretch: attacking my own label and splits | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w03_feature_leakage_check.ipynb?flush_cache=true) | ☑ |
| `notebooks/w04_signal_audit.ipynb` | ML-06, optional stretch: the exploratory and inferential analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w04_signal_audit.ipynb?flush_cache=true) | ☑ |
| `notebooks/w04_baseline_score.ipynb` | ML-07 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w04_baseline_score.ipynb?flush_cache=true) | ☑ |
| `notebooks/w05_model.ipynb` | ML-08 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w05_model.ipynb?flush_cache=true) | ☑ |
| `notebooks/w06_validation_audit.ipynb` | ML-09 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w06_validation_audit.ipynb?flush_cache=true) | ☑ |
| `notebooks/w07_action_playbook.ipynb` | ML-10 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/w07_action_playbook.ipynb?flush_cache=true) | ☑ |
| `notebooks/capstone.ipynb` | ML-11 (the paper mirrors it) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tessa-Saumu/FlyRank-ML-Internship/blob/main/work/notebooks/capstone.ipynb?flush_cache=true) | ☑ |
| closing section of `notebooks/capstone.ipynb` | ML-12: 5-minute demo outline, social-post cut, employer summary | (same notebook, last cells) | ☑ |

Every analysis section in these notebooks follows one pattern: **question, why it matters (the FlyRank principle it
serves), method, result, what it changes.**

## Layout

```text
work/
  notebooks/            the submission, one notebook per card (executed, outputs committed)
  scripts/              the pipeline, one module per stage (see below)
  outputs/              JSON receipts every number traces back to (committed); row-level CSVs are gitignored
  figures/              charts for the paper and notebooks (drawn from receipts only)
  paper/index.html      the research paper (deployed to GitHub Pages)
  capstone_report.md    the capstone write-up (from the template)
  archive/              earlier Lane 2 experiments, not for marking
```

| Script | Stage |
|---|---|
| `wh.py` | DuckDB views over the pinned, locally cached warehouse snapshot; downloads it once with an `HF_TOKEN` |
| `build_page_month.py` | page x month cache (row level, written to `~/.cache/flyrank_freestyle`, never to git) |
| `eda_00_panel.py` to `eda_04_client_level.py`, `panel_fe.py` | exploratory and inferential analysis (ML-06) |
| `erosion_frame.py` | forward-origin frame, honest label, noise floor |
| `erosion_model.py` | features, frozen baselines, nested tuning, forward and sealed evaluation, leakage checks |
| `erosion_analysis.py` | reproduces the sealed result and profiles its errors |
| `erosion_playbook.py` | ML-07 rule queue and the ML-10 live queue with reason codes |
| `make_figures.py`, `make_paper.py`, `deploy_paper.py` | figures, paper, publishing |
| `verify_evidence.py` | offline check: runs the capstone notebook and matches headline numbers to receipts |

## Reproduce

1. `pip install -r requirements.txt scipy`
2. Set `HF_TOKEN` (a read token) as an environment variable or Colab secret. Never type it into a cell.
3. Run the notebooks in card order. The first data notebook downloads the pinned snapshot (about 1.2 GB) and builds the
   page-month cache (about 7 minutes). After that, every query runs locally.
4. `python work/scripts/verify_evidence.py` checks the receipts, the figures and the paper offline.

**Seeds:** 42 everywhere. **Revision:** `50cbf7c3909d07be4d1b5906b4d09e882e5acbf2`.

## Rules of the road (from the template, kept)

1. **Copy, don't edit.** `scripts/` at the repository root is the pristine reference pipeline and is untouched.
2. **No datasets in git.** Row-level outputs are gitignored. The JSON receipts are committed.
3. **Stay reproducible.** Fixed seeds, a pinned revision, receipts for every number.
4. **Public-safety language.** Observed, measured, directional, decision-support. No client-identifying details, and no
   causal claims without a design.
