# V2 pipeline: pre-registered specification

Spec version: v2.0. Written and committed BEFORE any V2 model code was written or run.
Machine-readable copy: `work/outputs/v2_spec.json` (the code asserts against it). Hashes of both files:
`work/outputs/v2_spec_hash.json`.

## 1. Purpose and framing

V2 is a process-rigour upgrade of the V1 CTR-erosion early-warning pipeline. The question, the unit
(page x origin), the eligibility rule and the primary label (`erode`) are unchanged from V1. What changes is how
features, models and the final model are chosen: every choice is made on validation folds under a rule fixed
in this document, the sealed test month is scored once, and every step leaves a receipt.

**Disclosure (verbatim).** The test month (May 2026) was seen during V1, so it is not a pristine holdout. No V2 decision (features, models, hyperparameters, selection rule) used it; all were made on validation folds whose outcome windows end before the test window opens. The V2 test result may therefore be optimistic.

The new date and rhythm features come from a mentor's advice that a date column yields many derived features.
They are not claimed to be motivated by V1 errors.

Facts that bound every claim: the warehouse has one revision (50cbf7c3, the pinned one); the data ends
2026-06-30; there is no more data. The effective number of clients per origin is about 5 (the top 3 clients hold
61 to 73 percent of rows), so all uncertainty is reported at the client-cluster level and wide intervals are expected.

## 2. Timeline and splits

For origin month o: features come from calendar months o-2 and o-1 (one trend feature also uses o-3); the label
comes from months o and o+1. A model scoring origin o may train only on origins <= o-2 (the training label must
have matured before o begins).

| Role | Score origin | Feature months | Outcome window | Train origins (their outcome windows end) | Used for decisions |
|---|---|---|---|---|---|
| Validation fold 1 | 2026-02 | Dec 2025, Jan 2026 | Feb-Mar 2026 | 2025-12 (Jan 31) | yes |
| Validation fold 2 | 2026-03 | Jan, Feb 2026 | Mar-Apr 2026 | 2025-12, 2026-01 (Feb 28) | yes |
| Confirmation fold | 2026-04 | Feb, Mar 2026 | Apr-May 2026 | 2025-12 to 2026-02 (Mar 31) | NO, reported only |
| TEST (sealed) | 2026-05 | Mar, Apr 2026 | May-Jun 2026 | 2025-12 to 2026-03 (Apr 30) | scored once, no decision after |
| LIVE | 2026-07 | May, Jun 2026 | none yet | 2025-12 to 2026-05 (Jun 30) | queue only |

Only validation folds 1 and 2 are used for selection. Their outcome windows (Feb-Mar, Mar-Apr) end before the
test window (May-Jun) opens. The confirmation fold's outcome window (Apr-May) overlaps the test window, which is
exactly why it must not drive any choice: it is computed after the selection is written to disk and cannot
change it. Origin 2026-04's label has not matured when the test month is scored, so the test refit uses origins
2025-12 to 2026-03 only. Rolling-origin folds reuse earlier months by design (fold 2 trains on origin 2026-01,
whose outcome window overlaps fold 1's validation window in February); within each fold, training windows end
before the scored origin begins.

## 3. Features

Start from the 34 V1 features (`erosion_model.FEATURES`).

**Removed (3):**
- `page_prev_ratio` and `client_prev_erode_rate`: all-missing in the training rows of folds 1 and 2 (no origin
  o-4 exists), and the label anti-persists. `page_prev_ratio` stays available as the input of the frozen baseline
  `persistence_prev_ratio` only; it is not a model feature.
- `word_count`: `dim_content` is a July-2026 snapshot and 87 to 99.8 percent of pages were updated after each
  origin, so word count may reflect later regeneration.

**Set to missing (1):** `imp_growth_prev` for origin 2025-12 only (its o-3 month is September 2025, which
straddles Google's num=100 change).

**Added (5), each computed strictly inside months o-2 and o-1** from the daily table (per page-month cache in
`~/.cache/flyrank_freestyle/page_month_rhythm/`, one month at a time, never in the repo):
- `weekday_share`: share of window impressions on Monday to Friday (missing if no impressions).
- `days_since_last_click`: days from the last day with a click in the window to the origin's first day (missing
  if no click).
- `longest_zero_click_run`: longest run of consecutive calendar days with no click inside the two-month window;
  days with no row count as no-click days. Computed per month and combined by maximum, so a run that crosses the
  month boundary is not joined (disclosed under-count). A page absent from one of the two months gets that
  month's full length as its run for that month.
- `days_since_peak`: days from the single day with maximum impressions in the window to the origin's first day
  (ties resolve to the most recent such day).
- `keyword_to_content_gap_days`: `content_created_date` minus `keyword_created_date` in days (missing if either
  is missing).

**Not added:** an outcome-window calendar, client tenure, plain month-of-year (see section 11).

**Result:** 34 - 3 + 5 = **36 features** (the code asserts this count and exports the list, a one-line meaning and a
family per feature to `work/outputs/v2_features.json`). Features that are entirely missing in a fold's training
rows (for example `imp_growth_prev` in fold 1) are dropped for that fold and listed in the receipt.

**Snapshot disclosure.** `search_volume`, `cpc`, `competition`, `main_intent`, `content_type` and the keyword
attributes are also as of the July-2026 snapshot; `keyword_created_date` is checked in code to be <= the origin
(violations, if any, are reported in the receipt). All other features come from the daily table inside the
feature window.

## 4. Models (exactly four learned candidates; scikit-learn only)

Common: seed 42; training rows are the measurable rows of the training origins; scores are tie-broken with the V1
hash tiebreak; all fits run single-threaded so results are bit-reproducible.

1. **LOGISTIC (champion).** Median imputation with missing indicators, StandardScaler, `LogisticRegression`.
   Grid: C in {1e-3, 1e-2, 1e-1, 1, 10, 100} x penalty in {l2, l1, elasticnet with l1_ratio 0.5}
   (scikit-learn 1.9 parametrisation: l1_ratio 0, 1, 0.5); solver lbfgs for l2 (max_iter 5000), saga for l1 and
   elasticnet (max_iter 20000, random_state 42). Any ConvergenceWarning is recorded per fit. 18 configs.
2. **HGB_CLF.** `HistGradientBoostingClassifier`, at most 20 random configs from a seeded sampler
   (`numpy.random.default_rng(42)`): learning_rate log-uniform [0.02, 0.2]; max_leaf_nodes in {7, 15, 31};
   min_samples_leaf = max(20, round(f x rows in the fit)) with f in {0.005, 0.015, 0.04}; l2_regularization in
   {0, 1, 5, 20}; max_features in {0.5, 0.8, 1.0}.
   Number of trees: cap 600; chosen by early stopping on a validation block, then the model is refit on all
   training rows with that number of trees fixed. Validation block = the latest training origin (fit on the earlier
   training origins, time-ordered); when only one training origin exists (fold 1) it is a client-grouped 20 percent
   of clients (GroupShuffleSplit, first seed from 42 upwards whose holdout holds 10 to 40 percent of the rows).
   The number of trees is the argmin of the validation log loss over the staged predictions. Predictions are
   averaged over 5 seeds (42 to 46; the seed affects max_features subsampling, so configs with max_features 1.0 are
   seed-invariant and are fitted once, asserted in code).
3. **POISSON_RATE.** `HistGradientBoostingRegressor(loss="poisson")`, target `clk_out / exp_out`, sample weight
   `exp_out`, trained on measurable rows; same 20 configs, early stopping (weighted Poisson deviance on the
   validation block), refit and seed averaging as HGB_CLF. Score = minus the predicted ratio (a lower predicted
   ratio means more erosion).
4. **ENSEMBLE (pre-declared, not selected).** Equal-weight rank average (percentile ranks within the scored origin)
   of the tuned LOGISTIC score and the tuned HGB_CLF score.

**Frozen baselines (unchanged from V1):** `rule_position_slipping` (the editor's rule), `mean_reversion_ctr_excess`,
`client_shock`, `persistence_prev_ratio`. The persistence baseline's fill value is the median of
`page_prev_ratio` over all eligible pages of the scored origin (knowable at decision time) rather than over the
measurable pages, so it can differ slightly from V1. If `page_prev_ratio` is all-missing for a fold the baseline is
reported as unavailable, not as zero.

## 5. Selection rule (fixed before running)

- **Tuning.** Each config is scored by its mean AP (average precision on the measurable rows, `erode` label) over
  validation folds 1 and 2. The best config per family is kept. Ties (mean AP within 1e-4) go to the simpler,
  more regularised config (LOGISTIC: smaller C, then l1 over elasticnet over l2; HGB: fewer leaves, larger
  min_samples_leaf fraction, larger l2, smaller learning rate, then lower index).
- **Champion and challengers.** LOGISTIC is the champion. A challenger (HGB_CLF, POISSON_RATE, ENSEMBLE) replaces
  it only if BOTH hold: (a) the paired AP gain over the champion, pooled over the two folds, has a client-cluster
  bootstrap 95 percent lower bound (2.5th percentile) above 0 (B = 2000; clients are resampled within each fold;
  the AP of challenger and champion is computed on the same resampled rows; the two folds' deltas are averaged;
  one set of resamples with seed 42 is shared by the three comparisons); and (b) the mean gain stays above 0 when
  any single client is left out (leave-one-client-out: the client's rows are removed from every fold it appears
  in). The number of challenger comparisons (3) is reported next to the result; no multiplicity correction gates
  the decision (a Bonferroni-adjusted bound is given for information). If more than one challenger passes, the one
  with the larger mean gain is selected.
- **Criterion.** AP only. P@20, P@50, P@100 and the rest are reported, not used for selection.
- **Nested time-ordered check.** The whole procedure (tuning plus champion/challenger test) is repeated using fold 1
  only; the model it selects is refit on fold 2's training rows and scored on fold 2, and compared with the
  best baseline and with the both-fold selection.
- All decisions are recorded in `work/outputs/v2_selection.json`.

Caveat stated up front: tuning and the champion/challenger comparison use the same two folds, so the
validation-fold numbers of the tuned models are optimistic; the confirmation fold and the test month are the
less-biased reads.

## 6. Metrics (every validation, confirmation and test evaluation)

P@20, P@50, P@100, AP, AUC, within-client P@10 (clients with at least 30 rows), lift (P@50 divided by the base
rate), clicks-lost captured (share of the month's total shortfall sum(max(0, exp_out - clk_out)) over measurable
rows that falls in the top 50), top-client share of the top 50, number of clients in the top 50, effective clients
(1 / HHI of row shares) for the month, client-cluster bootstrap 95 percent CI (B = 2000) for P@50 and AP, the
leave-one-client-out min and max of (model minus best baseline) AP, and P@50 counting unmeasurable pages as misses
(score all eligible pages, label unmeasurable pages 0). Base rate and the Poisson noise-only rate are printed next
to every month. The best baseline is the one with the highest AP in that month (conservative), and the editor's
rule is also reported as the pre-declared comparator. The headline test metrics are AP and P@50 of the selected
model with their CIs against the base rate, the noise-only rate, the editor's rule and the best-AP baseline.
Descriptive only: P@50 after the 10-pages-per-client cap, and the top-3 client row share.

## 7. Secondary label (limited scope, never used for selection)

`erode_rel = 1` if `log((clk_out + 0.5) / (exp_out + 0.5))` minus the median of that quantity over the client's
measurable pages in the same origin is <= ln(0.75). Evaluated only for the champion LOGISTIC with its selected
configuration, the same features and folds (validation folds, confirmation fold, and test only at the unseal
step): AP, P@50, top-client share of the top 50 and the overlap of its top 50 with the primary top 50.
Labelled secondary everywhere. Receipt `work/outputs/v2_secondary_label.json`.

## 8. Leakage and split checks (validation folds only)

- Planted-future-feature harness on fold 2: a noisy copy of `log_ratio_out` (plus N(0, 0.3)) added as a feature
  must lift AP and P@50 strongly; the honest model must not look that good.
- Random-row KFold against client-GroupKFold AP on the Feb + Mar rows (selected logistic config).
- Timeline assertions in code: every feature window ends before the origin's first day (dates read from the
  daily table are asserted), and every training origin's outcome window ends before the scored origin's first day.
- Receipt `work/outputs/v2_leakage_checks.json`.

## 9. Sealed-month protocol (enforced in code)

Row loading drops the outcome columns (`erode`, `clk_out`, `imp_out`, `pos_out`, `exp_out`, `log_ratio_out`,
`null_erode`, `measurable`, and anything derived, including the matured-label columns of origin 2026-07) for the
test origin 2026-05 and removes origin 2026-07 rows altogether, unless an explicit `--unseal` path runs AND
`work/outputs/v2_freeze.json` exists AND its stored code and spec hashes match the current files. The unseal path
writes `v2_test.json` only if it is absent and refuses to run a second time. The rhythm cache for months 2026-05
and 2026-06 (the test outcome months) is built only after the test has been scored. Order of operations:
(1) write this spec, (2) commit it, (3) implement, (4) run selection on the validation folds, then the
confirmation fold, (5) write `v2_freeze.json` (chosen model and config, hashes of spec and code, timestamp),
(6) run the unseal step once, (7) build the live July queue. No May-origin outcome is looked at before step 6.

## 10. Playbook (July 2026 live queue)

Rules give the reason code, the model gives the order. Order = the selected model's score (refit on origins
2025-12 to 2026-05). Reason code = the largest positive standardised contribution (standardised value times
coefficient) of the champion LOGISTIC fitted on the same rows with its selected config, mapped to the families in
`erosion_playbook.FAMILIES` / `ACTIONS` plus: `DEMAND_SHIFT` also covers `days_since_peak`; new `STALE_ACTIVITY`
(`days_since_last_click`, `longest_zero_click_run`) and `WEEKDAY_PROFILE` (`weekday_share`). Guardrails as in V1:
pages younger than 90 days are excluded and at most 10 pages per client enter the 50-page queue. The 10-per-client
cap was introduced in V1 after the sealed month was opened; it is a playbook guardrail only and plays no part in
any V2 selection or headline metric. Outputs: `v2_queue_2026-07.csv` (gitignored, contains IDs),
`v2_playbook_summary.json` (no IDs) with the SHA-256 of the CSV, which is the frozen artefact.

## 11. Deliberately NOT done

- No outcome-window calendar, client tenure or plain month-of-year feature: with six origins they would let a
  model memorise the origin or the client rather than learn a pattern that transfers.
- No "require activity in both feature months" rule: it came from the V1 error analysis of the sealed month, so
  adopting it would let test information shape V2.
- No claim that the new features were motivated by V1 errors.
- No new data and no new packages (no xgboost, lightgbm, deep nets).
- No decision from the confirmation fold or the test month; no re-tuning after unsealing.
- No multiplicity correction in the decision rule (the comparison count is reported).
- No calibration step and no use of the per-client cap in any selection or headline metric.
- Per-client results are never reported with client names or IDs; receipts contain aggregates only.

## 12. Known limitations

About 5 effective clients per origin; the test month is not pristine; snapshot attributes may reflect later
changes; the inner early-stopping split (fit on earlier origins, validate on the latest) has outcome windows that
overlap by one month between adjacent origins, so the tree count may be slightly high (it never touches the
scored origin's labels); tuning and the champion/challenger test share the same two folds; the label definition
was designed in V1 on the development origins.

## 13. Deviation log

- 2026-11-06, before any V2 result was used (implementation note, not a methodological change). scikit-learn 1.9's
  histogram binner raises an error on a column with fewer than two distinct non-missing values. In the boosted fits,
  a column that carries no information in a given fit (all missing or constant; for example `imp_growth_prev` in the
  early-stopping fit that uses origin 2025-12 alone, or `intent_navigational` in the early origins) is left out of
  that fit only. Such a column cannot be split on, so the trees are the ones the spec describes; the number of
  columns left out is recorded per fit in the receipts.
