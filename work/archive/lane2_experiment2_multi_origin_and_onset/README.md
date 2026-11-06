# Lane 2, experiment 2: multi-origin pipeline and onset screening (archived)

**Not part of the submission. Kept for the record of my process.**

- **Track:** Lane 2, Refresh / Content Opportunity (same question as experiment 1).
- **Snapshot of:** branch `cff39ae8-flyrank-ml-internship` at commit `2ecacd7`, plus three onset receipts that were
  never committed (`outputs/onset_*.json`, `outputs/onset_headline.md`).
- **Contents:** the notebooks as revised for this experiment, the warehouse pipeline (`scripts/warehouse_feature_experiments.py`,
  `final_submission_pipeline.py`, `onset_screening.py`, `diagnose_ranking.py`), `REVIEW_pipeline_verdict.md`,
  `PLAN_b_onset_screening.md`, receipts under `outputs/` (including `final_pipeline/` and `diagnostics/`), and the
  manifest of the cached frames. The cached parquet frames themselves are row-level data and are gitignored.

## What was done

- Rebuilt the frames on the full warehouse with a 1,000-impression floor and five forward monthly origins (December
  2025 to April 2026), each trained on the latest month whose label had matured.
- Added three momentum features and diagnosed why the hand rule did not rank.
- Tested an "onset" screen: only pages that were still flat or growing.

## What it showed

| Ranker (mean P@50 over five origins, base 0.364) | P@50 |
|---|---|
| Hand rule | 0.372 |
| Five-feature logistic | 0.496 |
| Eight-feature logistic | 0.704 |
| Eight-feature random forest | 0.760 |
| Two-line acceleration rule | 0.812 |

Onset screen: logistic 0.508 vs hand rule 0.212 (pool base 0.257). Bootstrap intervals excluded zero in only 3 of 5
months, and January carried the mean.

## Why it was superseded

The label (next 30 days below 0.8 x the last 30) shares its recent window with the features. A page already falling is
labelled "declining" without any forecast, so a momentum rule is close to unbeatable. That lesson is the starting
point of the freestyle study.

To rerun: `git checkout 2ecacd7` and follow `REVIEW_pipeline_verdict.md`.
