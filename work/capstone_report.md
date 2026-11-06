# Capstone Report: Freestyle · Same rank, fewer clicks

- **Author:** Theresia Saumu
- **Lane:** Freestyle (lane guide, section 9). My earlier Lane 2 (refresh) work is archived in `work/archive/` and is not part of this submission.
- **Repo:** https://github.com/Tessa-Saumu/FlyRank-ML-Internship
- **Date:** November 2026

Every number below is read from a committed receipt in `work/outputs/`, and `work/scripts/verify_evidence.py` checks
the headline ones against those receipts.

## 0. Abstract

AI answers in search are the new variable in FlyRank's business, yet nobody had measured what they do to the clicks a
page earns from Google at the rank it already holds. On FlyRank's pseudonymized internship warehouse (78.8 million daily
rows, January 2025 to June 2026), page fixed effects with client-clustered errors put top-3 click-through in April 2026
at 0.642 times its October 2025 level for the same page in the same position band (95% CI 0.615 to 0.669). I built a
monthly ranking of top-10 pages likely to lose at least a quarter of their expected clicks in the next two months: an
observed label designed so that visible momentum cannot give it away, 34 engineered features, four frozen baselines,
nested tuning, and forward validation with a sealed month opened once. Across four forward months the logistic ranking
reached precision@50 of 0.52, 0.60, 0.66 and, on the sealed month, 0.42, against base rates of 0.23 to 0.34, and it led
every baseline on PR-AUC in all four. The output is a reviewed monthly queue for editors: decision support, predictive
and not causal.

## 1. Problem framing

- **Decision:** which 50 top-10 pages an editor should protect this month before their clicks disappear.
- **Unit:** one page at one monthly decision point.
- **Output:** a ranked queue of 50 pages, each with one reason code and one action.
- **Who acts:** a content editor, with the account lead for client-wide declines.
- **Cost of a wrong call:** a false alarm costs about 20 minutes of review plus the risk of editing a healthy page; a miss
  costs clicks on a page that still ranks.
- **Why ML:** erosion depends on rank, recent CTR, CTR against what the position normally earns, demand swings and the
  client together. The intuitive one-signal rules point the wrong way (section 6), and the best simple rule changes from
  month to month.

The question exists because of a measured effect: at a fixed page and position, CTR fell in 10 of 11 clients between
October and November 2025 and March and April 2026 (Wilcoxon p = 0.007). FlyRank's own *State of AI-Driven SEO 2026*
report does not measure this.

## 2. Data safety

- **Data:** the warehouse release `FlyRank/internship-warehouse` (build v20260703, pinned revision `50cbf7c3`): the daily
  performance fact table, `dim_content` and `dim_clients`. Cached locally outside the repository; no data in git.
- **Excluded on purpose:**
  - `provider_used`, which contains brand-like strings (never selected; flagged to my mentor);
  - the 90-day query table, whose window overlaps every outcome window;
  - `model_used` as a treatment, because it records the latest regeneration, not the author;
  - AI-referral sessions as a target, because they are too thin (83,618 sessions, two clients hold 68%).
- **Leakage considered:**
  - Momentum inside the label: found twice and fixed (`label_design_log.json`).
  - Label ingredients as features: a planted outcome feature lifts precision@50 from 0.66 to 1.00, which proves the
    harness detects leaks.
  - Future windows: features end at the month before the decision.
  - Tuning on the test: tuning happens inside training months only.
  - IDs: pseudonymous and used only for grouping. They never appear in a receipt, notebook output, the paper or this
    report; `verify_evidence.py` scans for them.

## 3. Baseline

The editor's rule, frozen before any check: position tier (top 3 = 2, positions 4-6 = 1, 7-10 = 0) plus how much CTR
fell last month, with a hash tiebreak. One reason code, one action label, and the queue is written to
`work/outputs/baseline_action_score.csv`.

Three more frozen baselines keep the comparison fair:
- CTR above peers (a data-informed one-signal rule);
- client-wide CTR shock;
- persistence of last period's ratio.

Rule precision@50 by forward month: 0.46, 0.76, 0.50 and 0.38 (sealed). On the sealed month its AUC was 0.453, worse
than random order.

## 4. Model / analysis

- **Target, in one sentence:** a page erodes if its clicks in the two months after the decision fall below 75% of what
  its latest-month CTR, shrunk toward its client's position-adjusted CTR, predicts at its new position.
- **Noise floor:** a Poisson simulation gives 0.045 to 0.058 on the non-sealed months and 0.071 on the sealed month, against base rates of 0.23 to 0.34.
- **Features (34, listed in `erosion_features.json`):** exposure, trajectory, stability, rank, CTR against expectation,
  content metadata, GA4 channels (including any AI referral), client context and matured page history.
- **Left out on purpose:** query-mix features (window overlap) and anything from the outcome months.
- **Models:** regularised logistic regression and histogram gradient boosting, tuned with an inner GroupKFold by client
  inside training months.
- **Frozen choice:** the logistic model, recorded in `erosion_freeze.json` with code hashes before the sealed month was
  opened. It was the more consistent of the two, and gradient boosting was not bit-reproducible across reruns.

## 5. Evaluation

- **Split:** forward months, so each decision month trains only on months whose labels had matured. Clients are held
  out in a secondary check.
- **Why this split:** the deployment question is "next month, for every client".
- **Before and after:** a random row split reports PR-AUC 0.538. Holding clients out gives 0.422. The gap is client
  memorisation.

| Decision month | Base rate | Rule P@50 | Best baseline P@50 | Model P@50 | Best baseline PR-AUC | Model PR-AUC |
|---|---|---|---|---|---|---|
| 2026-02 | 0.227 | 0.46 | 0.46 | 0.52 | 0.234 | 0.317 |
| 2026-03 | 0.338 | 0.76 | 0.76 | 0.60 | 0.385 | 0.451 |
| 2026-04 | 0.308 | 0.50 | 0.56 | 0.66 | 0.371 | 0.378 |
| 2026-05 (sealed) | 0.24 | 0.38 | 0.40 | 0.42 | 0.234 | 0.348 |

- **At 50 pages:** the model is about twice the base rate every month and beats the best baseline of that month in three
  of four. The client-bootstrap intervals are wide (sealed: 0.28 to 0.70), so single-month differences at 50 pages are
  not decisive.
- **On the whole ranking (PR-AUC):** the model leads every baseline in all four months.

**Errors.** In the sealed top 50:
- false alarms cluster in 7 clients, while true positives come from 13;
- most top pages had data in only one of the two feature months, with impressions jumping threefold or more;
- true positives beat their peers' CTR by more (median ratio 1.17 against 0.69).

## 6. Interpretation

- **What the model learned:** the pages losing clicks next are the ones currently earning more than their rank
  normally earns, or riding a recent CTR bump. That excess click share is what an AI answer can take.
- **This contradicts the editor's intuition.** In the training months, top-3 pages eroded *less* often than positions
  7 to 10 (0.266 against 0.310), and pages whose CTR had just risen eroded most (0.399). This explains why the rule
  collapses out of sample.
- **It is not noise reverting:** the one-signal "CTR above peers" rule ranks at about the base rate, so the edge comes
  from combining signals.
- **Negative and surprising results:**
  - Informational pages did not lose CTR significantly; transactional pages did (0.677).
  - Within a client, AI-assistant discovery follows exposure (AUC 0.715), not Google rank (0.506).
  - AI-referral growth for clients is not supported once onboarding is removed.

## 7. Recommendation

1. **Each month, review the 50-page queue** (at most 10 pages per client; pages at least 90 days old). Start with pages
   flagged `CTR_ABOVE_PEERS`: make them the source an AI answer cites before rewriting anything.
2. **Before acting, open the live results page** and record whether an AI answer appears. Confirm demand when
   impressions jumped or the page has one month of data.
3. **Treat transactional and commercial pages as the more exposed group** in this panel.
4. **Hold out 10 of each month's 50 pages**, so the value of the actions can be measured.
5. **Retrain monthly**, and pause the queue if a matured month's precision drops below 1.3x its base rate.

**Confidence:** moderate. The effect is consistent across clients and months, but there are only four forward months,
the intervals at 50 pages are wide, and the cause is not measured.

## 8. Reproducibility

```bash
pip install -r requirements.txt scipy
export HF_TOKEN=...          # read token; or a Colab secret. Never typed into a cell.
python work/scripts/run_notebooks.py           # executes w01 ... capstone in order (downloads the pinned snapshot once)
python work/scripts/verify_evidence.py         # offline: receipts, capstone, paper rebuild, report numbers, privacy
```

- **Seeds and pins:** seed 42 everywhere; warehouse revision `50cbf7c3909d07be4d1b5906b4d09e882e5acbf2`.
- **Python:** 3.12, with pandas, numpy, scikit-learn, duckdb, scipy and matplotlib.
- **Sealed evaluation:**
  - frame builder: `work/scripts/erosion_frame.py`, with `SEALED_ORIGIN = "2026-05"`;
  - freeze receipt: `work/outputs/erosion_freeze.json`;
  - sealed metrics: `work/outputs/erosion_model_metrics_sealed.json`, produced by `erosion_model.py --unseal`;
  - reproduced, with an assertion, by `erosion_analysis.py` in `w06_validation_audit.ipynb`.

## 9. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai): the pseudonymized warehouse release, the starter
repository and the curriculum. Errors and interpretations are mine.

---

> Claims checklist: observed / measured / directional / decision-support language throughout; no causal claims; no
> "predicted Google's algorithm"; no client-identifying details; base rate next to every precision figure; numbers
> checked against a fresh rerun by `verify_evidence.py`.
