# Content-review ranking: personal capstone report

Author: Tessa-Saumu. Lane: content refresh decision support.
Evidence reconciliation: 13 October 2026; historical metrics are unchanged.

## 0. Abstract

This capstone asks whether month-end page signals can prioritize a human editor's
50-page review budget before next-month impressions decline. Retained March and
April frames contain 95,810 and 99,279 pages. A transparent rule is compared with
logistic regression and random forest under grouped and forward validation.
Grouped LR P@50 is 0.636 versus rule 0.512, but forward LR 0.54 is below the
April base rate of about 0.56. The output supports cautious investigation, not
automatic refreshing or a claim of traffic recovery.

## 1. Problem framing

The unit is a page-month. Rank pages for limited editorial review and attach
investigation actions. Incorrect priorities consume editor time; observed
decline is not proof that a refresh would help.

## 2. Data safety

FlyRank supplied a pseudonymized research warehouse. Client/page IDs define
joins and grouped evaluation, not model inputs. Label-derived trend fields
are excluded. No warehouse dataset or page-level ranked queue is distributed
here; follow [data-use terms](../DATA_USE.md).

## 3. Baseline

The personal rule combines visibility, staleness and recent change signals;
see [rule notebook](notebooks/w04_baseline_score.ipynb) for the exact formula.
Its entry floor (100 impressions) differs from its visibility cut (500).
Grouped mean P@50 is 0.512; forward P@50 is 0.44 on the same retained frame
and budget used by LR. Root starter scripts/results are separate reference work.

## 4. Model / analysis

LR and RF use five month-end signals; their exact feature construction is
in [model notebook](notebooks/w05_model.ipynb). The target is an observed
next-month impressions drop greater than 20%. Simple models permit interpretable
comparisons; no new tuning was performed for this portfolio update.

## 5. Evaluation

Five client-grouped splits have zero client overlap. LR beats the rule on four
folds and loses one; fold populations differ and splits are correlated. Mean
P@50: LR 0.636, RF 0.608, rule 0.512 (March base 0.491). Fit-March/score-April
P@50: LR 0.54, rule 0.44, base about 0.56. Random-split and planted-leak
results are negative controls, not portfolio headline performance.

## 6. Interpretation

The grouped advantage does not establish future-month head lift. Signal drift,
client concentration and tracking filters constrain conclusions. Outcome
tracking excludes 2.6% of March and 3.6% of April candidates. Receipts document
survivorship and leakage diagnostics; no causal inference is supported.

## 7. Recommendation

Use the playbook's gates to investigate quiet risk and unusual changes, with
human review and fallback clocks. The historical top-50 queue spans ten clients
and its largest-client share is 48%; its observed decline rate equals the April
base rate. No editor adoption, A/B result or revenue impact was measured.

## 8. Reproducibility

From a fresh checkout, Python 3.12:

```bash
python -m pip install -r work/requirements-evidence.txt
python work/scripts/verify_evidence.py
```

This executes every capstone cell against committed receipts, checks 31 headline
numbers and regenerates figures in a temporary directory. It does not retrain
models, query the warehouse, or overwrite the paper. Historical seed 42 and
library versions are in the receipts. A full rebuild requires authorized
warehouse access and the w03–w07 notebook sequence; it has not been rerun here.
Offline check environment is deliberately distinct from the historical model
environment. Hosted CI must still run after publication.

## 9. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai). FlyRank supplied
curriculum, warehouse and starter pipeline. Personal framing, experiment
notebooks, validation, paper and playbook are under `work/`; AI assistance was
used in implementation/documentation. Template code is not claimed as personal.
