# Content-review ranking: personal capstone report

Author: Tessa-Saumu. Lane: content refresh decision support.
Evidence reconciliation: 29 October 2026. The earlier grouped and forward results are kept as
historical evidence and are labelled as such wherever they appear.

## 0. Abstract

This capstone asks whether learned ranking earns its place against a transparent hand rule when a
content team can review only fifty pages a month before their impressions fall. Five monthly origins
from the FlyRank warehouse release, at a shared floor of 1,000 recent impressions, show two different
answers depending on the question asked. On the full candidate population the hand rule does not rank
at all: 43 to 71 percent of candidates share its top score, and its observed precision@50 of 0.372
matches the order-independent expectation of 0.368, a gap of two tenths of one case in fifty. A two-line momentum rule reaches 0.812 there and
no learned model beats it, because the label compares the next 30 days with a trailing window that
already contains the fall. Restricted to pages still flat or growing, where that shortcut cannot
apply, logistic regression beats the hand rule in all five origins (mean 0.508 against 0.212, pool
base rate 0.257) and beats the population rate forward at both 50 and 20 slots. The output is a
monthly review order in two lanes, one for traffic already falling and one for healthy pages at risk,
with no claim that an edit changes the outcome.

## 1. Problem framing

The unit is a page-month. Rank pages for a limited editorial review budget and attach an
investigation action. Incorrect priorities consume editor time; observed decline is not proof that a
refresh would help.

## 2. Data safety

FlyRank supplied a pseudonymized research warehouse. Client and page IDs define joins and grouped
evaluation, never model inputs. Label-derived trend fields are excluded. No warehouse dataset or
page-level ranked queue is distributed here; follow [data-use terms](../DATA_USE.md). Five feature
columns are known at the cutoff: logged recent impressions, click-through rate, average position,
active days and content age.

## 3. Baseline

The personal rule combines visibility, staleness and recent change signals; see the
[rule notebook](notebooks/w04_baseline_score.ipynb) for the exact formula. Its first job is to be a
fair comparator, and the audit found that it is not one. Because each condition is a threshold, the
score takes four values, and at the 1,000-impression floor between 43 and 71 percent of candidates
share the top value. The top fifty is therefore a draw from that band rather than an ordering, and
its precision moves with row order. Measured across the five origins, observed precision@50 is 0.372
against an order-independent expectation of 0.368, a gap of two tenths of one case in fifty. The diagnostic run, which counts the boundary at the band rate rather than as a lower bound, puts the same expectation at 0.371, and the conclusion does not depend on which of the two is used. The rule
behaves as a filter for visible, stale, low-CTR pages, not as a ranker. Notebook section 1b carries
the per-origin numbers.

The published weeks used a 100-impression entry floor, where the March frame held 95,810 pages and
24,462 of them shared the top score. The revised pipeline uses 1,000, which every page clears at the
visibility test, so the band grows and the lottery gets larger. The two floors are separate
populations and are never compared row for row.

## 4. Model / analysis

Logistic regression and random forest use the five contracted features; a second challenger adds
three momentum columns (30-day against prior-30-day traffic, last seven days against the 30-day rate,
and a prior-coverage flag). Feature construction is in the [model notebook](notebooks/w05_model.ipynb).
The target is an observed next-month impressions drop greater than 20 percent. Training uses only
months whose labels had matured at the origin date, so no frame trains on a label that did not yet
exist. Simple models keep the comparison readable; no tuning search was run over these months.

## 5. Evaluation

Five monthly origins with equal weight, each trained on the latest mature month, all candidates
paired against the rule on identical rows. On the full population the ladder is: random 0.352, hand
rule 0.372, five-feature logistic regression 0.496, eight-feature logistic regression 0.704, eight-feature
random forest 0.760, momentum rule 0.812, coverage-adjusted momentum rule 0.824, against a mean base
rate of 0.364. The five-feature model improves on the rule by 12 points, which isolates the rule's
thresholds rather than its features as the problem. The momentum models improve by 33 to 39 points,
and the two-line rule still beats them.

That last result is a property of the label, not of the models. Decline means the next 30 days fall
below 80 percent of the last 30, and the last 30 includes the last seven, so a page that keeps its
recent run rate lands under the line by construction. Measured directly: every page in the momentum
rule's top fifty has a decline implied by its own recent rate, and persistence-implied labels agree
with observed labels on 59 to 85 percent of measurable pages. Momentum is therefore a lagging
detector with a calibration that shifts with the month, and it explains only 33 to 52 percent of the
declines that actually occurred.

Restricting to pages whose last-seven-day rate is at or above their 30-day rate removes that
shortcut, because those pages must fall further to be labelled. In that pool the mean base rate is
0.257, the momentum rule sits at 0.280 and the hand rule at 0.212, so neither has a structural
advantage. Logistic regression reaches 0.508, beating the hand rule in five of five origins (mean
difference +0.296, median +0.200, leave-one-out means 0.205 to 0.340) and beating the pool base rate
in five of five. At the co-primary budget of 20 slots it reaches 0.530 against 0.190 for the rule.
Random forest is unstable across origins (0.020 to 0.620) and is reported as unstable rather than
averaged into a claim.

Three limits travel with those numbers. The origins overlap in outcome windows and repeat pages and
clients, so five origins are development slices, not five independent tests. Removing the largest
client and restricting to complete coverage leaves the onset result standing but narrows the margin
per origin. The earlier published evaluation, five client-grouped folds inside March (LR 0.636, rule
0.512, base 0.491) and one month forward (0.54, 0.44, 0.56), remains in the receipts as historical
evidence for the 100-impression population and is not blended into the results above.

## 6. Interpretation

The learned models do not predict decline better than a two-line momentum rule on the full
population, and saying so is part of the result. What they do is order pages that are not already
falling: the pattern the model appears to find is a page in a 30-day downtrend whose last week ticked
up, which then resumes falling. Two momentum features alone recover most of the eight-feature score,
which is why the finding is stated as narrow rather than general. Client concentration matters in the
other direction too: capping the queue at five rows per client raises the hand rule's precision and
lowers every stronger ranker, so part of the momentum advantage is a small number of clients.

## 7. Recommendation

Hand an editor two lists instead of one. Lane 1 holds pages already falling, ordered by the momentum
rule, and its first action is to check tracking and indexing before any content work. Lane 2 holds
pages still flat or growing, ordered by the model, as a watch list for the following month. Both
lanes cap rows per client and carry the rule flag as a reason code. The earlier historical queue
spanned ten clients with a 48 percent largest-client share and one action on 49 of 50 rows; the
two-lane version removes both failures by construction and asserts them in the notebook. No editor
adoption, controlled test or revenue effect was measured.

## 8. Reproducibility

From a fresh checkout, Python 3.12:

```bash
python -m pip install -r work/requirements-evidence.txt
python work/scripts/verify_evidence.py
```

That runs the capstone notebook against the committed receipts, checks the 31 headline numbers and
the onset checks, and regenerates the figures in a temporary directory. It does not query the
warehouse or retrain.

A full rebuild needs authorized warehouse access and runs in this order: `w03` data contract and
leakage check, `w04` rule and ranking audit, `w05` models and the onset comparison, `w06` validation
audit, `w07` two-lane playbook, then `capstone`. The onset comparison is also available as a single
script:

```bash
python work/scripts/onset_screening.py            # uses the cached monthly frames
python work/scripts/onset_screening.py --warehouse # rebuilds them from the release at floor 1000
python work/scripts/onset_screening.py --selftest  # synthetic frames, no data access
```

Seeds are 42 everywhere. Library versions are recorded in every receipt. The revised pipeline's
receipts live in `work/outputs/final_pipeline/`, and the onset audit writes
`work/outputs/onset_screening.json` plus a generated sentence block in
`work/outputs/onset_headline.md`.

## 9. Acknowledgments & data credit

Built on the [FlyRank ML Internship dataset](https://flyrank.ai). FlyRank supplied curriculum,
warehouse and starter pipeline. Personal framing, experiment notebooks, validation, paper and
playbook are under `work/`; AI assistance was used in implementation and documentation. Template
code is not claimed as personal.
