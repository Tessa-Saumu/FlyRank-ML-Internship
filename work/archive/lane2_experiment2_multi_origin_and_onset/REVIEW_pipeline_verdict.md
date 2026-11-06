# Submission review: the new pipeline, the real lift, and what to do before you ship

**Scope:** `main` (the published capstone) versus `submission/warehouse-temporal-pipeline`
(`work/scripts/warehouse_feature_experiments.py`, `work/scripts/final_submission_pipeline.py`,
`work/outputs/final_pipeline/*.json`, `work/outputs/warehouse_feature_experiments/experiment_results.json`).
I read both notebook sets, the scripts, the receipts, the paper and the starter pipeline
(`scripts/`, `outputs/model_report.md`). No warehouse access was needed: every number below is
already committed, and I re-derived the comparisons from the receipts.

**Bottom line.** (Revised after the audit ran — see §12 for the measured result, which upgrades this
verdict from "no lift" to "lift on the onset population".) The new experimental pipeline is a real
methodological upgrade and its lift numbers are real, but you are reading the lift against the wrong
object. The frozen rule never ranked at all
(its top-50 is a lottery draw from a 44–71% tie band), so "0.372 → 0.704" is lift over a lottery, not
over a rule. And the 0.81–0.94 head precision that looks like the win is mostly a two-number momentum
signal that is close to a restatement of the label. The useful news: your model's lift over the base
rate (+34 to +40 points) is roughly *double* the starter pipeline's headline lift (+20 points), and
the 25/75 target you have been chasing comes from a starter baseline that scores **below its own base
rate**. You are not short of a result. You are short of a comparator that means something, and of one
label that a model can beat a rule on.

---

## 0. What is actually on the branch, and whether it is trustworthy

**Verification boundary:** I could not read the gated warehouse from this environment, so D2–D4 below
have not been run on the real frames. They are implemented and self-tested on synthetic data; every
number quoted in §1–§5 comes from a committed receipt. Where an argument depends on the frames, it is
labelled as a hypothesis with the exact check that confirms or kills it.

The receipts are traceable. `final_pipeline/baselines.json` records
`pipeline_sha256 = f49f5cca…` and `query_sha256 = b301169a…`. The committed scripts hash to
`eb534e4…` / `f355398d…` with LF endings and to exactly the recorded values with CRLF endings, which
is what a Windows run produces. So the numbers were produced by the committed code, not invented.
Two weaker points: the receipts are not wired into `verify_evidence.py`, and the notebooks' own
interpretation cells still say **"PENDING LOCAL WAREHOUSE EXECUTION"** while the same notebooks
contain populated outputs. Fix both before submitting; a reviewer will notice.

Population difference (important, do not merge the two tables):

| | `main` (published) | `submission/warehouse-temporal-pipeline` |
|---|---|---|
| Eligibility floor | 100 impressions (rule visibility cut 500) | **1,000** impressions for everything |
| Months | March + April 2026 | Oct 2025 – Apr 2026, five origins (Dec–Apr) |
| March/April rows | 95,810 / 99,279 | 41,931 / 45,664 |
| Validation | 5 client-grouped CV + one month forward | 5 forward origins, latest-mature-month training |
| Base rate | 0.491 / 0.560 | 0.290, 0.195, 0.218, 0.473, 0.644 (mean **0.364**) |

The two pipelines are therefore not "before and after" on one experiment. They are two different
populations. That is fine, but the write-up has to say it in the same sentence as any number that
moves.

---

## 1. The numbers, side by side

**Published (`main`), P@50, conservative labels:**

| Evaluation | Base | Static rule | LR | RF |
|---|---:|---:|---:|---:|
| 5 client-grouped folds, March | 0.491 | 0.512 | **0.636** | 0.608 |
| March → April forward | 0.560 | 0.44 | 0.54 | 0.14 |

**New branch, mean of five forward origins, conservative (unknowns counted as non-declines):**

| Ranker | Mean P@50 | Worst origin | Mean P@20 | Mean P@100 | Mean AP | Mean AUC |
|---|---:|---:|---:|---:|---:|---:|
| Uniform random | 0.352 | 0.18 | 0.45 | 0.386 | 0.365 | 0.502 |
| **Frozen static rule** | **0.372** | 0.16 | 0.35 | 0.350 | 0.370 | 0.514 |
| Control LR (rule's 5 inputs, continuous) | 0.496 | 0.16 | 0.53 | 0.486 | 0.412 | 0.548 |
| Control RF | 0.468 | 0.10 | 0.51 | 0.446 | 0.425 | 0.562 |
| Momentum LR (8 inputs) | 0.704 | 0.56 | 0.77 | 0.718 | 0.593 | 0.689 |
| Momentum RF (8 inputs) | 0.760 | 0.36 | 0.74 | 0.734 | 0.559 | 0.685 |
| Acceleration rule (1 ratio) | 0.812 | 0.54 | 0.80 | 0.864 | 0.631 | 0.705 |
| Coverage-adjusted acceleration rule | 0.824 | 0.60 | 0.77 | 0.870 | 0.628 | 0.700 |

Base rate 0.364. Paired against the frozen rule: control LR **+0.124** (4/5 origins positive),
control RF +0.096 (3/5), momentum LR **+0.332** (5/5), momentum RF **+0.388** (5/5). The declared
hurdle (+5 cases per 50, positive in 4/5) passes for three of the four — but the hurdle is measured
against the broken comparator, so passing it is not the same as showing value.

Unknown outcomes move the picture more than the paper currently admits. Averaged over the origins:

| Ranker | Lower bound | Measurable only | Upper bound |
|---|---:|---:|---:|
| Static rule | 0.372 | 0.375 | 0.376 |
| Momentum LR | 0.704 | 0.768 | 0.784 |
| Momentum RF | 0.760 | 0.765 | 0.768 |
| Acceleration rule | 0.812 | **0.908** | 0.924 |

March is the origin that drives the rule's "worst origin 0.54": 15 of its 50 picks have no
14-day outcome window. Lower 0.54, measurable 0.771, upper 0.84. One origin, three stories. Pick one
convention and report all three in the same row, or the number will look cherry-picked.

---

## 2. Finding 1 — the frozen rule was never ranking. This is now measurable.

In the new receipts every static-rule row carries `static_rule_tie_expectation`: the precision you
would get by drawing 50 pages at random from the top score band.

| Origin | Candidates | Rows sharing the top score | Share of population | Queue draws | Order-independent expectation | Observed P@50 |
|---|---:|---:|---:|---:|---:|---:|
| 2025-12 | 19,073 | 12,373 | 64.9% | 0.40% | 0.265 | 0.340 |
| 2026-01 | 22,340 | 15,839 | 70.9% | 0.32% | 0.208 | 0.160 |
| 2026-02 | 30,275 | 18,020 | 59.5% | 0.28% | 0.258 | 0.200 |
| 2026-03 | 41,931 | 19,458 | 46.4% | 0.26% | 0.502 | 0.480 |
| 2026-04 | 45,664 | 19,967 | 43.7% | 0.25% | 0.604 | 0.680 |
| **Mean** | | | | | **0.368** | **0.372** |

Observed minus expected: **+0.004**, i.e. **0.2 cases out of 50**. The rule's realised performance is
its own tie lottery. The published paper already admitted the mechanism ("24,462 pages tie at the
maximum score… the band's decline rate is 50.7%"), but the new pipeline turns that admission into a
per-row number, which is strictly better and should be kept.

Two consequences you have to internalise:

1. **"Model beats baseline" has meant "model breaks ties".** In the published grouped table the rule's
   0.512 mean sits essentially on its band's decline rate (50.7% on the March frame); LR's 0.636 is
   real ordering, but
   +12.4 points over the rule is not +12.4 points over a ranking, because the rule never ranked.
   Against the base rate it is +14.5 points, and forward it is −2 points.
2. **Raising the floor to 1,000 impressions made the baseline worse.** At ≥1,000 impressions every
   page passes the rule's `visible` predicate, so one of its three conditions becomes constant and the
   score can take only four values; the top band then swallows 44–71% of the population. The new
   pipeline did not fix the degenerate baseline; it amplified it and then measured it.

**Answer to "did we overcome the hurdle of the baseline queue just reporting all the same action?"**
No. The pipeline *diagnosed* it and then compared models to it anyway. The 1,000-impression floor
must not be the primary population unless the baseline is given resolution (a deterministic secondary
sort inside the band, or a continuous version of the same score). The 100-impression population is
also degenerate but at least comparable to the published work; report both as a floor sensitivity and
say plainly that the rule is a filter, not a ranker.

---

## 3. Finding 2 — the label partly records a step-down that has already happened

> **Corrected in §12.** This section was written as a hypothesis before the audit ran. The measurement
> confirms the mechanism for the acceleration rule's queue (100% of its top-50 is persistence-implied)
> but **refutes the strong version** for the population: persistence explains only 33–52% of observed
> declines, and its calibration flips sign across months. The circularity is real but partial, and the
> gap it leaves is exactly where the model wins. Read this section as the mechanism, §12 as the
> verdict.

The label is

```
decline = future30_impressions < 0.80 × recent30_impressions
```

and `recent30` ends seven days before month-end, i.e. it *contains* the recent collapse. Let
`a = (recent7 / 7) / (recent30 / 30)` be the acceleration ratio. A page that simply holds its last
seven days' level lands at `future30 = 30 × recent7/7 = recent30 × a`. So, for any page whose new
level persists, the label is equivalent to

```
a < 0.80
```

That is exactly the score of the acceleration rule (`rank by a`), and it is one of the eight features
of the "momentum" models. You are not measuring whether a model can forecast decay; you are measuring
whether it can find pages that already stepped down, where the trailing-30 denominator still remembers
the higher level. Two mechanical consequences:

- A page that fell last week and then stays flat is labelled "declining" for the next 30 days even
  though nothing further is happening. Late detection is rewarded; early warning is not.
- Any ranker that sorts by recent momentum is close to sorting by the label. Precision at the head
  near 0.94 against a 0.29 base rate is the expected signature — and so is a *modest* global AUC
  (0.68–0.76). High head precision, ordinary discrimination: a threshold effect, not a forecasting
  breakthrough.

The seven-day reporting gap between the feature cutoff and the outcome window does not remove this.
It means the rule is forecasting 7–37 days ahead, so persistence has to hold. Measured (§12), it holds
for about half of the declines: persistence-implied recall is 0.33–0.52, and the realised/persistence
ratio runs 1.24 (December, traffic rebounds) down to 0.71 (April, traffic falls further than the trend
implied). So momentum is a lagging indicator with unstable calibration, not a restatement of the label
for the whole population. It is a restatement for the *queue*, which is why that queue scores 0.82–0.94.

This explains, in one stroke: why no model beats the acceleration rule; why the control models (no
momentum) only add 10–13 points; and why the "combined" model in the earlier run hit 100% at P@50 —
that result was flagged as unexplained and now has a mechanism. (March's 15 unknown outcomes are a
separate coverage problem, not part of this argument.)

Before you write a single word about momentum in the paper, run the check. It is implemented as D2 in
`work/scripts/diagnose_ranking.py`. What you want to see is whether the persistence-implied label
agrees with the observed label on ~85–95% of measurable pages and whether
`future30 / (30 × recent7/7)` has a median near 1.0. If it does, the honest sentence is:

> The momentum signal is close to a restatement of the label: a page that keeps its last seven days'
> run rate falls below the 30-day reference by construction. The high head precision measures
> detection of an existing step-down, not prediction of a new one.

That sentence is a better contribution than a 0.76 P@50, and it is defensible under any reviewer.

---

## 4. Finding 3 — the honest headline is a ladder, not a single lift

Read the table in §1 as a ladder. This is the result I would publish:

| Rung | Mean P@50 | Δ vs base (0.364) | What it tells you |
|---|---:|---:|---|
| Uniform random | 0.352 | −0.012 | sanity floor |
| Frozen static rule | 0.372 | +0.008 | the rule is a tie lottery |
| Continuous model on the rule's own 5 inputs | 0.496 | **+0.132** | the rule's *binarisation*, not its features, destroys the signal |
| Momentum model (8 inputs) | 0.704–0.760 | **+0.340 / +0.396** | momentum is the dominant signal |
| Two-number momentum rule | 0.812–0.824 | **+0.448 / +0.460** | the best ranker in the study is not a model |

Three conclusions that a reviewer will accept:

1. **ML earns a place relative to the hand rule.** Same five inputs, continuous instead of thresholded,
   plus 13 points. That is a clean, interpretable result about hand-written thresholds.
2. **ML does not earn a place relative to a momentum rule.** On the mean it loses on P@20, P@50,
   P@100, AP and AUC. It wins individual origins (December AUC 0.775 vs 0.692; March RF 0.92 vs 0.54)
   but not the comparison. Say so.
3. **The models are less stable than the rule.** Momentum RF ranges 0.36–0.98 across origins;
   the rule 0.54–0.94. A 5-origin mean hides that. The conservative summary is the right one, and it
   is the reason the rule wins the table.

Caveat that must travel with the table: the five origins overlap in outcome windows (December's label
window is January, January's is February, …) and repeat pages and clients. They are five development
slices, not five independent tests. "5/5 origins" is a directional statement, not a p-value.

---

## 5. Finding 4 — the queue can still be operationally useless at high P@50

The published playbook is the clearest example: top-50 after gates declined at 0.56, which is exactly
the April base rate, and **49 of 50 rows carried the same action** (`investigate_quiet_risk`) with a
48% single-client share. The new pipeline computes concentration and leave-one-client-out reranking
(good), but does not act on it; `w07` still emits the static-rule queue and marks manual review
`PENDING`.

The queue-level question is not "what is P@50" but "can an editor act on these 50 rows". Three cheap
tests, all in the diagnostic script:

- **Client cap.** Greedy top-50 with at most 5 rows per client. If P@50 holds while the largest-client
  share falls, the uncapped number was a concentration artefact.
- **Reason mix.** Count the reason codes on the queue. If one code is >70%, the queue is one action.
- **Acceleration band mix.** Split the queue into `<0.3` (already collapsed), `0.3–0.8` (sliding),
  `≥0.8` (healthy). A queue that is all `<0.3` is a post-mortem list, not a repair list.

---

## 6. Direct answers

**Is the new pipeline an improvement?**
On machinery, yes, clearly: tie expectation per row, unknown-outcome bounds, daily-rate sensitivity,
mature-label discipline, concentration and leave-one-client-out, calendar and partition assertions,
provenance hashes, and a synthetic fixture that executes all eight notebooks. On the headline claim,
no: the baseline is more degenerate and the best model is beaten by a rule.

**Is there a more obvious lift from baseline to model?**
Yes, and it is real: +33 to +39 points over the frozen rule, +34 to +40 over the base rate, positive in
all five origins. It is a much more obvious lift than the published +12.4. It is also mostly momentum,
so the lift is "the model found the signal the rule ignores" — which is a fine result — while the same
signal in a two-line rule does better.

**Did we overcome the "baseline queue reports one action" hurdle?**
Diagnosed, not overcome. The static rule still has a 44–71% tie band and still produces one action.
The acceleration comparator does rank, but it ranks by the label's twin. The real fix is to give the
baseline resolution and to make the queue heterogeneous by action, client and trajectory stage.

**Should the methodology go into the main pipeline?**
The methodology: yes. The notebooks: no. The branch replaced 17–27-cell, sectioned, executed notebooks
with 4-cell wrappers that call one function and leave the interpretation placeholder. That is a
structural regression and it will cost you rubric points. Port the pieces listed in §7 into the
verified notebooks; do not merge the branch wholesale, and do not merge its notebook files.

**About the 25/75 target.**
It comes from the supplied starter `outputs/model_report.md`: `baseline_rules` P@50 = 0.240, RF = 0.740,
label rate 0.542. The starter baseline is **30 points below its own base rate** — it is anti-predictive.
"25 vs 75" was never a scientific benchmark; it is a broken comparator in a different dataset with a
different label (`is_declining_label`, derived from `trend_pct`). Measured against the base rate, the
starter model adds 20 points; your momentum model adds 34–40. On the only comparison that matters you
are already ahead of the reference implementation.

---

## 7. What to port, exactly, without touching your verified structure

Keep `main`'s notebooks, section order and voice. Add cells; do not rewrite them.

| Where | What to add | Why | Effort |
|---|---|---|---|
| `w04`, after §1 | **"1b. Does my rule actually rank?"** — one code cell computing distinct scores, top-band size and share, band decline rate, and order-independent P@50 vs observed. Use `tie_expectation` from the branch. | Converts the paper's existing tie-band admission into a receipt. This is the single most valuable cell on the branch. | 30 min |
| `w04`, §2 | Floor sensitivity: same rule at 100 and 1,000 impressions, tie share on both. | Shows the floor change *is* the degeneracy, and prevents the "you moved the goalposts" critique. | 1 h |
| `w05`, §3 | **Control-continuous model**: LR on the rule's same five features. Keep the existing LR/RF comparison below it. | Adds the missing rung that proves the rule's thresholds, not its features, are the problem. | 1 h |
| `w05`, §3 | Momentum challenger, clearly labelled *diagnostic*: same split, same metric, plus the persistence check from D2. | Lets you report the lift and its caveat in one place. | 2 h |
| `w06`, §2 | **Multi-origin paired table**: five forward origins, paired deltas vs the frozen rule *and* vs the base rate, with lower/measurable/upper precision, unknown counts, largest-client share. | Replaces "5/5 origins" hand-waving with one auditable table. Reuse `final_submission_pipeline.validation()` and point it at the base rate as well as the rule. | 3 h |
| `w06`, §2 | Client-capped rerun and leave-largest-client-out. | Tests whether the lift is concentration. | 1 h |
| `w07`, §1 | Client cap (≤5/client), reason-mix report, acceleration-band mix, and split the queue into 3 action lanes. | Turns a 50-row list into something an editor can run. | 3 h |
| `capstone` §4 + paper | The ladder table from §4 above, plus the revised abstract/limits sentences from §8. | This is the new headline. | 3 h |

Copy only `warehouse_feature_experiments.py`, `final_submission_pipeline.py` and the aggregate receipts
into `main`. Leave the branch's notebooks where they are.

**Do not port:** the 4-cell wrapper notebooks, the 1,000-impression floor as the *primary* population,
acceleration as the primary comparator (it is a diagnostic), the `development_hurdle_met` flag (it is
measured against a degenerate baseline), or any "PENDING" placeholder left next to populated output.

**Do port as-is:** `warehouse_feature_experiments.py`'s `_feature_window`, mature-label filter,
partition date-bound assertions, unknown-outcome conventions, daily-rate target, tie keys and the
synthetic fixture. Those are correct and better than what `main` has.

---

## 8. How to make the model earn its place (three experiments, ordered)

The current label cannot answer "does ML add value" because the answer is bundled into the label. Three
ways out, cheapest first. All three run on the cached frames; no warehouse rebuild.

**E1 — Onset target (the one that decides it).** Restrict to pages whose current run rate is healthy
(`a ≥ 1.0`, at least 3 covered days in the last 7). On that population the label is *not* implied by
momentum. Report the base rate and P@50 for: static rule, acceleration rule, LR, RF, refit on healthy
training rows. If the model beats the static rule by a clear margin here, ML has earned its place on
the question editors actually care about ("which healthy page is about to slip?"). If it does not,
that is the honest negative result. Implemented as **D4**. A defensible target refinement: label
`future30 < 0.8 × (30 × recent7/7)`, i.e. decline *beyond* the current trajectory.

**E2 — Expected-loss ranking (the one where ML structurally wins).** Rank by
`P(decline) × recent30_impressions` instead of rank by probability. A rule cannot price magnitude; a
calibrated model can. Evaluate with a utility metric — captured impressions at risk, or mean
impressions-at-stake in the top 50 — and report P@50 alongside so the trade-off is visible. This is
also the better editorial product: 50 reviews should go where the traffic is. Not implemented yet;
`diagnose_ranking.py` is the place to add it (the scoring and ranking helpers are already in place).

**E3 — Within-momentum reordering (the cheap, honest margin).** Take the top decile by acceleration
and ask whether the model orders *inside* that pool better than the rule. Implemented as **D3**. If the
model wins in-pool, the right product is a two-tier queue: rule for tier 1, model for tier 2. If it
does not, stop trying to beat the rule on this label.

Do **not** spend time on monotone constraints, gradient boosting, or hyper-parameter search to rescue
the current comparison. If the label is the constraint, a better model cannot fix it; it will just
approximate the same ratio more expensively.

---

## 9. Submission plan

**Option A — ship this week (3 working days).** Port §7. Publish the ladder plus the structural
findings, with the negative result stated. Replace the headline numbers in the abstract, §3, §5 and
the README, keep the structure and the voice, and move the old grouped/forward table into a labelled
historical appendix rather than deleting it. This is a stronger paper than the current one because it replaces a fragile +12.4 claim with a
measured structural finding and a +40 ladder.

Suggested headline block:

> The hand-written rule does not rank: 44–71% of candidates share its top score and its top-50 is a
> draw from that tie band (observed 0.372 vs order-independent 0.368). A continuous model on the same
> five inputs lifts P@50 to 0.496; adding traffic momentum lifts it to 0.704–0.760 against a 0.364
> base rate, positive in all five development origins (paired +0.33 / +0.39 over the rule). A
> two-number momentum rule reaches 0.812. The lift is real, but it is momentum, not model complexity,
> and the label rewards detecting a step-down that has already occurred. A prospective test on a
> non-overlapping month and an onset-style target are the next step, not another model.

**Option B — ship the stronger version (7–9 days).** Option A plus E1 and E2. Headline becomes:
"momentum finds pages that are already sliding; among pages that are still healthy, the model adds X
points." That is the version where ML earns its place, and it is the version I would aim for if the
deadline allows.

Fix before either: remove the `PENDING`/`INPUT AFTER RUN` text from executed cells, wire
`final_pipeline/*.json` into `verify_evidence.py` (or add a second checker), update
`submission/paper_url.txt` if the URL changes, and rerun every notebook top to bottom.

---

## 10. How to run the audit

```bash
# in the branch where the frames exist (or with FLYRANK_FINAL_CACHE pointing at them)
python work/scripts/diagnose_ranking.py
```

It reads `work/local_final_pipeline/*.parquet`, writes
`work/outputs/diagnostics/ranking_diagnosis.json`, and prints five tables:

- **D1** top-score band size/share, order-independent vs observed P@50 per ranker;
- **D2** agreement between the persistence-implied label and the observed label, the
  realised/persistence median, and the share of the acceleration queue already implied by momentum;
- **D3** model vs rule inside the top momentum decile;
- **D4** the onset target (healthy pages only), base rate and model vs rule;
- **D5** client concentration, client-capped P@50, reason mix and acceleration-band mix.

`--selftest` runs the same code on synthetic frames with no cache and no warehouse, and was verified
to execute end to end. What to look for: D1 observed ≈ expected (rule is a lottery); D2 agreement
high and realised/persistence ≈ 1 (label is momentum); D3/D4 decide whether the model has a question
left to win.

---

## 11. What is actually good here (keep it)

- Five origins with a declared reporting-lag scenario instead of one forward month.
- `_mature_training_frames` — training labels must be available at the origin date. That is proper
  time-discipline and `main` does not do it.
- Unknown-outcome lower/measurable/upper reporting, and the rule that unknowns stay in the queue.
- Daily-rate target sensitivity against raw-total coverage artefacts.
- Client concentration and leave-largest-client-out reranking.
- Calendar edge-case assertion and per-partition date-bound verification.
- Provenance hashes and a synthetic fixture that executes all eight notebooks.
- The decision to keep `latest_only` training: the experiment's own history interaction table shows no
  consistent gain from longer history (positive in some origins, negative in others), so the choice is
  defensible and should be stated.

The work is not mediocre. It is a good experiment sitting behind a comparator that cannot lose or win,
and a label that answers a narrower question than the one written on the paper. Fix those two things
and the submission is stronger than what is currently published.

---

## 12. What the audit returned on the real frames

Run on the cached frames on 29 October 2026; receipts in `work/outputs/diagnostics/ranking_diagnosis.json`.
Five development origins, December 2025–April 2026, K = 50, conservative labels (unknown outcomes count
as non-declines).

### D1 — the rule does not rank. Confirmed, beyond argument.

| Origin | Rows sharing the top score | Share of population | Band decline rate | Order-independent P@50 | Observed P@50 | Distinct scores |
|---|---:|---:|---:|---:|---:|---:|
| 2025-12 | 12,373 | 64.9% | 0.266 | 0.266 | 0.340 | 4 |
| 2026-01 | 15,839 | 70.9% | 0.208 | 0.208 | 0.160 | 4 |
| 2026-02 | 18,020 | 59.5% | 0.262 | 0.262 | 0.200 | 4 |
| 2026-03 | 19,458 | 46.4% | 0.503 | 0.503 | 0.480 | 4 |
| 2026-04 | 19,967 | 43.7% | 0.614 | 0.614 | 0.680 | 4 |
| **Mean** | | | | **0.371** | **0.372** | |

Four distinct values, 44–71% of the population in one band, observed = expected to within one case.
The baseline is a filter with a lottery attached. Every "model beats baseline" sentence in the current
paper has to be re-read with that in mind.

### D2 — the label is momentum-correlated, not momentum-defined. I was half wrong, usefully.

| Origin | Base | Implied | Agreement | Precision | Recall | Realised ÷ persistence (median) | Accel queue already implied |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2025-12 | 0.289 | 0.304 | 0.705 | 0.490 | 0.516 | 1.237 | 1.000 |
| 2026-01 | 0.195 | 0.150 | 0.845 | 0.631 | 0.488 | 1.163 | 1.000 |
| 2026-02 | 0.218 | 0.141 | 0.785 | 0.513 | 0.331 | 1.058 | 1.000 |
| 2026-03 | 0.473 | 0.288 | 0.646 | 0.706 | 0.430 | 0.876 | 1.000 |
| 2026-04 | 0.644 | 0.330 | 0.592 | 0.857 | 0.440 | 0.706 | 1.000 |

Three things this settles:

1. **For the acceleration rule's queue, the circularity is total.** 100% of its top-50 is composed of
   pages whose own last-seven-day rate already implies a decline. Its 0.812 is detection of a collapse
   in progress, not forecasting. That claim is now measured, not inferred.
2. **For the population, it is not.** Persistence catches only 33–52% of the pages that actually
   decline, and it misses 15–41% of the labels outright. Roughly half of the decline population is
   invisible to momentum — and that half is the addressable question.
3. **The calibration flips sign across months.** Median `realised ÷ persistence` runs 1.24 in December
   (traffic recovers relative to the recent run rate, so momentum over-flags) down to 0.71 in April
   (traffic falls further than the trend implied, so momentum under-flags). This is why the momentum
   models swing between 0.36 and 0.98 across origins: they inherit a signal whose bias changes sign
   every quarter. Any pipeline that ships momentum needs that disclosure.

### D3 — no meaningful lift inside the momentum pool. Do not build tier-2 on this.

| Origin | Pool base | Acceleration rule | Momentum LR | Momentum RF |
|---|---:|---:|---:|---:|
| 2025-12 | 0.821 | 0.940 | 0.920 | 0.920 |
| 2026-01 | 0.763 | 0.920 | 0.780 | 0.600 |
| 2026-02 | 0.572 | 0.820 | 0.900 | 0.680 |
| 2026-03 | 0.815 | 0.540 | 0.880 | 0.920 |
| 2026-04 | 0.937 | 0.840 | 0.960 | 0.980 |

LR wins 3/5 for a mean of +0.076; RF wins 2/5 for +0.008, and the wins concentrate in March, where the
rule itself fell to 0.54 on 15 unknown outcomes. Verdict: no reordering margin worth a product.

### D4 — the onset population is where ML earns its place. This is the finding.

Restricted to pages that are *not* already collapsing (acceleration ratio ≥ 1.0, at least 3 covered days
in the last 7). Within this pool the label requires traffic to fall 20% below a currently healthy level,
so it is genuinely forward-looking and momentum has no structural edge — confirmed by the acceleration
rule landing at the base rate.

| Origin | n | Base | Static rule | Accel rule | Momentum LR | Momentum RF | LR − rule | LR − base |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2025-12 | 8,395 | 0.215 | 0.140 | 0.000 | 0.260 | 0.340 | +0.120 | +0.045 |
| 2026-01 | 15,138 | 0.093 | 0.120 | 0.140 | **0.780** | 0.020 | +0.660 | +0.687 |
| 2026-02 | 19,781 | 0.140 | 0.140 | 0.140 | 0.280 | 0.180 | +0.140 | +0.140 |
| 2026-03 | 20,398 | 0.319 | 0.300 | 0.480 | 0.500 | 0.620 | +0.200 | +0.181 |
| 2026-04 | 19,850 | 0.517 | 0.360 | 0.640 | 0.720 | 0.480 | +0.360 | +0.203 |
| **Mean** | | 0.257 | 0.212 | 0.280 | **0.508** | 0.328 | **+0.296** | **+0.251** |
| **Median** | | | | | | | **+0.200** | **+0.181** |

- **LR beats the frozen rule in 5/5 origins.** Mean +29.6 points, median +20.0, i.e. roughly 15 and 10
  extra correct picks out of 50. Leave-one-out means range 0.205–0.340, so it is not one month.
- **LR beats the base rate in 5/5 origins** — the test the published pipeline failed forward.
- **RF is unstable** (0.020 in January, 0.620 in March): report it as unstable rather than averaging it
  into a claim.
- **Internal validity check passes:** the acceleration rule scores 0.000–0.640 here against a 0.257 mean
  base rate, i.e. momentum's structural edge is gone once you exclude collapsed pages. The comparison is
  therefore not contaminated by the label twin.

### D5 — concentration is real and must be published with the queue.

| Ranker | Mean P@50 | Mean client-capped P@50 | Change | Mean largest-client share |
|---|---:|---:|---:|---:|
| Static rule | 0.372 | 0.440 | **+0.068** | 0.35 |
| Acceleration rule | 0.812 | 0.740 | −0.072 | 0.40 (0.70 in December) |
| Momentum LR | 0.704 | 0.644 | −0.060 | 0.38 |
| Momentum RF | 0.760 | 0.684 | −0.076 | 0.38 |

Capping at five rows per client *improves* the rule and *reduces* every stronger ranker, which is the
honest reading: part of the momentum head precision is a small number of clients, and part of the rule's
weakness is that it spreads thin. Report both numbers side by side, and add the cap to the shipped queue.

### Revised verdict

The project is no longer "no lift". It is two results on two populations:

1. **Triage (any trajectory): the model adds nothing over a two-line momentum rule.** Mean P@50 0.704–0.760
   vs 0.812–0.824, and the momentum queue is 100% persistence-implied. Negative result, worth publishing.
2. **Onset screening (healthy pages): the model is the best ranker in the study**, +29.6 points mean and
   +20 median over the frozen rule, 5/5 origins, above the base rate in 5/5. It wins precisely where the
   label stops being a momentum restatement.

That is a coherent thesis: *momentum detects decline already in progress; the learned model is what finds
the next one.* It is also the version the rubric rewards, because the ML is doing work no rule in the
study can do.

### Before that becomes the headline, four audits

1. **The January anomaly.** Base 0.093 with LR at 0.780 (39/50 vs 4.7 expected) while RF sits at 0.020 is
   an 8× lift with a model that fails on the same rows. It must be explained: coefficient signs, permutation
   importance, and whether the top-50 survives excluding the largest client and thin-coverage pages. The
   likely benign story — a page in a 30-day downtrend that ticks up in the last seven days (velocity < 1,
   acceleration ≥ 1) tends to resume falling — is testable and would itself be the paper's mechanism.
2. **Threshold sensitivity.** 1.0 is the natural cut ("flat or growing"), not a tuned one, so pre-register it
   and show the lift at 0.9 and 1.2, and with 3/5/7 covered days required.
3. **Unknown outcomes in the onset queue**, reported per origin as lower/measurable/upper like everything else.
4. **Bootstrap confidence intervals on the paired per-origin deltas**, resampling pages within origin, so the
   +0.20 median arrives with an interval rather than a point.
