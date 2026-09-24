# Capstone Report: which pages to review first

- **Author:** Theresia Saumu
- **Lane:** refresh prioritization. Rank pages by the chance that next-month search impressions fall by 20% or more, so an editor can spend a 50-page review budget on the right pages.
- **Repo:** https://github.com/Tessa-Saumu/FlyRank-ML-Internship
- **Date:** 24 September 2026
- **Deployed paper:** https://tessa-saumu.github.io/FlyRank-ML-Internship/

This report is the written version of the deployed paper. The paper's source is `work/paper/index.html`. Every headline number below comes from a committed receipt in `work/outputs/` or from a table an executed notebook printed. Where an earlier notebook sentence disagrees with that notebook's printed table, this report follows the printed table.

## 0. Abstract

When a content team can review only 50 pages a month, the practical question is which 50 should go first. We built one table per month from FlyRank's anonymized internship warehouse, about 79 million daily rows, and the March working table holds 95,810 pages across 40 clients, each with five numbers known at month-end and a label for whether next-month search impressions fell by 20% or more. On those five numbers we trained logistic regression and a random forest, and we compared both with a hand-written rule under a client-grouped split, a one-month-forward test, and a planted leak. Measured across five client-grouped folds, mean precision@50 was 0.636 and 0.608 versus 0.512 for the rule, against a 0.49 base rate; one month forward the edge mostly disappeared (0.54 versus the rule's 0.44 and a 0.56 base rate; random forest 0.14), and a random split that reuses clients scored 0.98, which we treat as leakage, not skill. The shipped output is a 50-row monthly review order for a human editor, with a plain reason per row and a fall-back to the rule: decision support for which pages to inspect, with no claim that reviewing or changing a page changes its traffic.

## 1. Problem framing

The decision is the order of a monthly review queue, not a yes or no flag on every page. The budget is fixed at 50 reviews, so the metric is precision@50: of the 50 highest-ranked pages, how many really declined.

- **Unit:** one page on one client site. Clients are used only to group the train and test splits. They are never model features.
- **Output:** a ranked 50-row queue. Each row has a model probability (the order) and a hand-rule reason code (the explanation).
- **Actor:** a content lead or editor running a monthly refresh sprint.
- **Action:** work down the queue. Check the page before changing anything.
- **Cost of a wrong call:** a wasted review uses about an hour of a 50-hour monthly budget. A missed decline leaves a page to decay for another month. Neither is catastrophic, which is why this is decision support, not automation.

A hand rule already exists. It is blunt: 24,462 March pages tie at its maximum score, and that band's decline rate is 50.7%, barely above the month's 49.1% base rate. The model's job is to order that tie. A model helps here because the useful pattern is not one threshold. Age risk rises and then falls. Client pages move together. A random split hides that and produces a fake 0.98.

## 2. Data safety

**Release.** FlyRank ML Internship warehouse, `FlyRank/internship-warehouse` on Hugging Face, gated. About 79 million daily rows, late 2025 through 30 June 2026, read with DuckDB, never downloaded in bulk. No client names, domains, URLs, page titles, or search queries are in the release. This report adds none.

**Tables used.** `fact_content_daily_performance`, `dim_content`, `dim_clients`. The query-level table is excluded because its fixed 90-day window can overlap the outcome month.

**Two impression cuts. Do not mix them.**

| Cut | Number | Job |
|---|---|---|
| Entry floor (Week 3 data contract) | at least 100 impressions and 14 days of data in the feature month, plus 14 days in the outcome month | Who is allowed into the modeling table. SQL in every later notebook: `recent30_impressions >= 100`. |
| Visibility cut (Week 4 hand rule) | at least 500 impressions | Who the rule calls visible. The stale and low-CTR checks apply only above this cut. Pages with 100 to 499 impressions stay in the table and are flagged `low_visibility`. |

The 500 number is the one used in the rule, the signal audit, and the "511 barely clears the floor" examples. The 100 number is the contract for who enters the table. The paper says 100 in the data section because that is the entry floor. It still uses 500 as the rule's visibility cut. Replacing 100 with 500 would rebuild the 95,810-page frame. That experiment was not run. Week 5 left the 500 cut untouched on purpose.

**Windows.** March development frame: features 2 Mar to 31 Mar, labels 1 Apr to 30 Apr, 95,810 pages, 40 clients, base rate 49.1%, largest client 22.1%. April scoring frame: features 1 Apr to 30 Apr, labels 1 May to 30 May, 99,279 pages, 44 clients, base rate 56.0%. June partitions were sealed and never read.

**Excluded on purpose.** Future-window fields, especially `future30_impressions` (they are the label). Label-derived trend fields. `provider_used` and `model_used`. Fixed-window query signals. Client and content hash IDs (grouping and joins only). The rule score (a baseline to beat, never a feature).

**Survivorship, from the receipt.** March kept 95,810 of 98,398 candidates (2.6% dropped: 391 absent, 2,197 thin). April kept 99,279 of 102,943 (3.6% dropped). Pages whose tracking went quiet are excluded, so the results describe pages that could still be measured.

**Label.** Declining if next-30-day impressions fall below 80% of last-30-day impressions, with at least 100 recent impressions. The printed Week 3 sensitivity cell shows 0.402 at a 70% line, 0.491 at 80%, and 0.571 at 90%. An earlier sentence in that notebook quoted 0.426, 0.491, and 0.556. This report follows the printed cell. 0.80 was kept as the contracted line.

Nothing client-identifying appears in `work/`.

## 3. Baseline

The Week 4 rule is a weighted sum of three yes/no conditions, with no fitted weights:

- visible = impressions at least 500
- low CTR in the top 10 = visible, position 1 to 10, CTR under 1%
- stale = visible and at least 91 days old

Score = 0.40 times visible + 0.35 times low-CTR-in-top-10 + 0.25 times stale.

Reason codes, one per page: `stale_low_ctr_top10`, `stale`, `low_ctr_top10`, `visible`, `low_visibility`. The label is not used to build any part of the score.

The rule is a fair comparison because it is scored on the same rows, the same folds, the same metric (precision@50), and the same tie-break (impressions, highest first) as the models.

Printed Week 4 signal tables, which this report follows:

- Pages 91 days or older declined at 52.8% (65,503 pages) versus 41.1% for newer pages (30,307 pages).
- Among visible top-10 pages, CTR under 1% declined at 48.0% (35,710 pages) versus 17.1% for healthier top-10 CTR (1,921 pages).
- Volume at the 500 cut was mixed. High-volume pages did not decline more often. The 500 cut is an editorial choice, not evidence that more impressions means more risk.

An earlier verdict sentence quoted 55.0% versus 42.5%, and 50.3% versus 18.6%. Those sentences do not match the printed tables.

On the five client-grouped folds, the rule's mean precision@50 is 0.512 (spread 0.201), against a 0.49 base rate. Its top 50 is a draw from a 24,462-page tie at the maximum score. That band's decline rate is 50.7%.

## 4. Model / analysis

Logistic regression and a random forest. Library defaults, no tuning, seed 42. Logistic regression standardizes features inside the training pipeline, so test-fold statistics do not leak into training. Both output a probability. The probability is the ranking score. The deployed queue is ordered by logistic regression, because it was the model that stayed above the rule one month forward.

**Feature list, the same one everywhere:** `log_recent30_impressions`, `recent30_ctr_pct`, `recent30_avg_position`, `recent30_active_days`, `content_age_days`.

**Target, one sentence:** a page is declining if its next 30 days of impressions are below 80% of its last 30 days, and it had at least 100 impressions in that last window.

Permutation importance on a held-out fold: CTR 0.080, active days 0.028, content age 0.021, log impressions 0.012, position 0.009. Logistic regression's age coefficient is negative (-0.15) even though staleness is a confirmed signal, because decline risk peaks at 90 to 180 days and then falls (April buckets: 0.38, 0.61, 0.66, 0.58, 0.36). A straight line cannot represent that shape.

## 5. Evaluation

**Split.** Five client-grouped draws (`GroupShuffleSplit`, test size 0.25, seed 42). Zero clients shared per fold. Test sides are unequal: fold 3's test side is 67,657 rows, 70.6% of the frame. The fold mean is not five equal experiments. That imbalance is disclosed, and the row counts are in `model_metrics.json`.

Within a frame the design is already time-honest: features end the day before the label window opens. Cross-month drift is a separate test.

**The fair-test contract.** Same rows, same folds, same metric, same tie-break, for the model and the rule.

| Test | Rule | Logistic regression | Random forest | Base rate |
|---|---:|---:|---:|---:|
| A. Random split (34 of 40 clients on both sides) | 0.44 | 0.74 | 0.98 | 0.49 |
| B. Client-grouped, five-fold mean | 0.512 | 0.636 | 0.608 | 0.49 |
| C. Fit March, score April | 0.44 | 0.54 | 0.14 | 0.56 |

Fold-level precision@50, from the printed Week 5 table:

| Fold | Rule | LR | RF |
|---|---:|---:|---:|
| 1 | 0.84 | 0.78 | 0.84 |
| 2 | 0.32 | 0.48 | 0.38 |
| 3 | 0.38 | 0.40 | 0.38 |
| 4 | 0.50 | 0.66 | 0.60 |
| 5 | 0.52 | 0.86 | 0.84 |

Logistic regression is above the rule on 4 folds and below it on fold 1. Random forest is above on 3 folds and tied on folds 1 and 3. An earlier sentence said neither model loses a fold. That sentence counted ties as wins and does not match this table. The five-fold means are still above the rule. The paired RF-minus-LR difference is -0.028 (spread 0.059), inside fold noise.

**One month forward.** Logistic regression 0.54 versus the rule 0.44, but the April base rate is 0.56, so the head of the queue does not beat picking at random. Random forest falls to 0.14. Logistic regression keeps some whole-list order (average precision 0.624 versus the rule's 0.559).

**Walk-forward**, fixed panel of 20 clients, same April test rows (base rate 0.547): rule stays at 0.38. Logistic regression reads 0.56, 0.68, 0.70, 0.66 as history grows from one to four frames. Random forest reads 0.28, 0.46, 0.40, 0.48. The rule is 0.38 here and 0.44 in Test C because the test rows are not the same set. More history helps briefly, then stops.

**Planted leak.** Average precision 0.761 with the honest five features, 0.993 with the label column planted in. The column was removed. Every reported number uses the honest five.

**Errors.** The confident mistakes repeat one pattern: rank 1 or 2, CTR 0.07% to 0.18%, thousands of impressions, often one client. That is usually a results page that answers the search itself, or a brand-name search. A page rewrite cannot fix it. It is now a gate, not a monthly surprise.

## 6. Interpretation

CTR is the feature the forest leans on. Active days and age follow. Position is last. The linear model gets age's direction wrong because the age pattern is a hill, not a line: risk peaks at 3 to 6 months, then falls. Age is a hint about what kind of review a page needs, not a reason to put the oldest page first.

The negative result that matters: one month forward, the head of the queue no longer beats the base rate. A random split's 0.98 is also a result. It measures leakage, not skill. Volume at 500 impressions was mixed, so "more visible" is not the same as "more likely to decline."

Some "declines" are measurement gaps. Pages with 14 to 20 days of outcome-month tracking declined at 96.2% (3,485 pages), versus 39.8% for pages covered all 30 days (70,253 pages).

## 7. Recommendation

Use the queue as a review order. Do not use it as a forecast, and do not automate edits from it.

This cycle's queue, fit on March and scored on April, is entirely `low_visibility`: under the 500 cut, but inside the 100-impression table. The printed head is small pages, about 100 to 180 impressions, with zero measured clicks. After gates, the top 50 declined at 0.56, the same as the April base rate. It is a diagnostic list: 49 quiet-risk checks and 1 tracking check, 49.5 editor-hours, about 4,800 impressions on those rows.

Ranked actions:

1. Cap one client's share of the queue before handing it out. One client holds 48% against a 35% limit. Until the cap is set, the list is not ready.
2. Review with a checklist: search terms, results-page layout, technical state, seasonality, business value, and sensitive-topic routing. Reviewer marks (`act`, `defer`, `not_actionable`, `escalate`) are the labels a later model should learn from.
3. Refit monthly. Retire any queue older than its feature month.
4. If precision@50 falls below the rule for two labelled months, order by the rule until a refit beats it. This cycle the fall-back is not triggered (0.54 versus 0.44). The stricter check, clearing the base rate by 0.05, did trigger (0.54 versus 0.61) and is reported.
5. Watch drift before release. Population stability index (a check that a feature's distribution did not shift too far): position 0.108 and content age 0.179 are in the watch band, under the 0.25 act line. Schema and base-rate drift passed.
6. Add an edit-history table so the "recently edited" lockout can move from proposed to implemented.

**Gates in code:** new pages (under 30 days or no prior month) excluded; risers (1.5 times) excluded as a one-month test; rank 1 or 2 with CTR under 0.20% get a senior look; jumps over 20 times get a tracking check; the 100-impression floor is applied upstream. A recently-edited lockout is proposed only.

**Never automated:** deletions, redirects, de-indexing, merges, unreviewed AI rewrites, edits to health, money, legal, or safety pages, bulk template changes, client-facing promises, acting on rising pages, or rewriting rank-1 or rank-2 near-zero-CTR pages beyond checking which searches surface them.

Confidence is limited to one cycle, on pages that could still be measured, as a review order. No causal claim.

## 8. Reproducibility

From a fresh clone:

```bash
pip install -r requirements.txt
# request access to FlyRank/internship-warehouse; do not hardcode a token
# run, in order:
#   work/notebooks/w03_data_contract.ipynb
#   work/notebooks/w04_baseline_score.ipynb
#   work/notebooks/w05_model.ipynb
#   work/notebooks/w06_validation_audit.ipynb
#   work/notebooks/w07_action_playbook.ipynb
python work/scripts/make_paper_figures.py
python work/scripts/deploy_paper.py
```

`capstone.ipynb` does not need warehouse access. It reloads the receipts and checks the headline numbers. Seed 42. Versions on the committed run: pandas 3.0.3, numpy 2.5.1, scikit-learn 1.9.0, duckdb 1.5.4.

Receipts, committed:

- `work/outputs/baseline_metrics.json`
- `work/outputs/model_metrics.json`
- `work/outputs/validation_audit_metrics.json`
- `work/outputs/action_playbook_summary.json`

There is no sealed holdout claimed as already scored. June 2026 partitions were never read. The cell that would build a later frame is the same frame builder in the Week 3 to Week 7 notebooks. It has not been run on June.

## 9. Acknowledgments and data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai). The warehouse is a gated, pseudonymized release of real production search data, prepared by the FlyRank team. Code in this repo is MIT-licensed. The dataset is governed by its own release terms and by `DATA_USE.md`.

---

Claims on this page use observed, measured, directional, and decision-support language. Precision@50 is always next to its base rate. No causal claim. No claim that this work predicted Google's ranking algorithm. No client-identifying details. Numbers match the committed receipts and the printed notebook tables named above.
