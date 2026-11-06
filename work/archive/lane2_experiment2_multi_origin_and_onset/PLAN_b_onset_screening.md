# Plan B — ship the onset-screening result

**Status:** ready to execute; waiting for your go-ahead. Nothing below has been started.
**Basis:** `work/outputs/diagnostics/ranking_diagnosis.json` (run 29 Oct 2026) and §12 of
`work/REVIEW_pipeline_verdict.md`.

---

## The thesis we would be shipping

> Momentum identifies pages whose decline is already under way: the acceleration queue is 100%
> persistence-implied and reaches 0.81–0.94 P@50 while no model beats it. Restricted to pages that are
> still healthy, where decline is not implied by their own recent run rate, a learned model beats the
> frozen rule in all five development origins (mean +29.6 points, median +20.0, i.e. about 15 and 10 extra
> correct picks per 50) and beats the population decline rate in all five. The ML earns its place on the
> question an editor actually has: which healthy page is about to slip — not on which collapsed page is
> already collapsing.

Why this is a better paper than the current one: it states a negative result (models lose to a two-line
momentum rule at triage), a positive result (models win at onset screening, 5/5), and a structural
finding (the hand rule never ranked) — three claims, each with a receipt, instead of one fragile +12.4.

---

## Work packages

### WP1 — Audit the January result before it can become a headline *(blocking)*
`work/scripts/onset_screening.py`, new. Reads the cached frames; writes
`work/outputs/diagnostics/onset_screening.json`.

- LR coefficient table per origin, standardised units, with signs.
- Permutation importance (AP scoring, 10 repeats, seed 42) per origin, on the test side only.
- Explicit test of the mechanism hypothesis: is the signal "velocity < 1 with acceleration ≥ 1" (a bounce
  inside a downtrend, which resumes falling)? Fit LR on those two features alone and compare with the
  8-feature model. If two features recover most of 0.78, that is the paper's mechanism and it is a good one.
- Robustness of 2026-01: drop the largest client; drop pages with < 7 covered days; recompute.
- Report every number in all three unknown-outcome conventions.

**Acceptance:** the January lift survives dropping the largest client and the coverage filter, and at least
one feature-level explanation is documented. If it does not survive, we report January as unstable and lean
the headline on the four-origin median (+20.0) instead of the five-origin mean (+29.6).

### WP2 — Sensitivity and intervals
Same script.

- Pool threshold at acceleration ratio ≥ 0.9 / 1.0 / 1.2; covered-days requirement 3 / 5 / 7.
- Bootstrap (2,000 resamples of pages within origin, seed 42) for the paired LR − rule delta per origin and
  for the pooled median. Report intervals, not just points.
- Label variant: current (`future < 0.8 × recent30`) vs decline-beyond-trajectory
  (`future < 0.8 × 30 × recent7/7`). If the ranking is stable across both, the result is not a label artefact.

**Acceptance:** lift positive at the pre-registered 1.0 cut and at both neighbours, intervals exclude zero
on at least 4/5 origins, and the label variant changes nothing material.

### WP3 — Notebook cells in the verified notebooks *(additive only)*
Your headings, order and voice stay exactly as they are. New cells, inserted; nothing rewritten.

| File | Section | New cell | Content |
|---|---|---|---|
| `w04_baseline_score.ipynb` | after §1 | **1b. Does my rule actually rank?** | distinct scores, top-band size and share, band decline rate, order-independent vs observed P@50 (the D1 table). This is the single best cell of the whole project. |
| `w04_baseline_score.ipynb` | §2 | 2b. Floor sensitivity | same rule at the 100 and 1,000-impression floors, with tie share on both. Answers "you moved the goalposts" before it is asked. |
| `w05_model.ipynb` | §3 | 3b. Where the model is asked the wrong question | the momentum comparison, labelled diagnostic, with the 100%-persistence-implied queue fact. |
| `w05_model.ipynb` | §4 | 4b. Onset screening | the D4 table plus the mechanism test from WP1. This is the new headline cell. |
| `w06_validation_audit.ipynb` | §2 | 2b. Five origins, paired | per-origin deltas vs rule *and* base rate, lower/measurable/upper, unknowns, largest-client share, bootstrap interval, client-capped rerun. |
| `w07_action_playbook.ipynb` | §1 | two lanes | Lane 1 triage (rule/momentum, collapsed pages, "verify and repair"), Lane 2 onset (model, healthy pages, "watch and prepare"). Client cap ≤5, reason codes per lane, unknown disclosure. |
| `capstone.ipynb` | §4 | results | the ladder table plus the onset table. |

**Acceptance:** `Run All` clean on each notebook, no placeholder text next to populated output, and the
committed receipts regenerate from the cells.

### WP4 — Claims and documents
`capstone_report.md`, `work/paper/index.html`, `README.md`, `work/README.md`, `work/FINAL_PIPELINE.md`.

Abstract/§5/§6/§9 rewritten around the thesis above; the old grouped and forward tables move to a labelled
historical appendix rather than disappearing. Every number in the paper traces to a JSON in `work/outputs/`.

**Acceptance:** `verify_evidence.py` extended to check the new onset receipts (it currently checks only the
`main` receipts), and it passes.

### WP5 — Figures
Two new figures: (a) the ladder, P@50 by ranker with the base-rate line; (b) onset screening per origin,
model vs rule vs base. Existing five figures kept.

### WP6 — Final read-through
Rerun every notebook top to bottom, regenerate the paper, redeploy, update `submission/paper_url.txt`,
confirm no client identifiers anywhere, confirm no `PENDING` remains.

---

## Timeline

| Day | Work |
|---|---|
| 1 | WP1 audit script + January explanation (blocking) |
| 2 | WP2 sensitivity, intervals, label variant |
| 3–4 | WP3 cells for w04, w05 |
| 5 | WP3 cells for w06, w07 |
| 6 | WP4 claims + `verify_evidence.py` extension |
| 7 | WP5 figures, paper rebuild, redeploy |
| 8 | WP6 final read-through and rerun |

Days 1–2 are the honest-risk part. Days 3–8 are mechanical once the numbers hold.

---

## Draft headline (numbers to be confirmed by WP1–WP2)

> Across five development origins, the hand-written rule never ranked: 44–71% of candidates shared its top
> score and its top-50 was a draw from that band (mean observed 0.372 vs order-independent 0.371). Sorting
> the same pages by the last seven days' run rate reaches 0.812 and no learned model beats it — that signal
> is a restatement of the label for the pages it selects, all 50 of which are already collapsing. Restricted
> to pages whose traffic is still flat or growing, where decline is not implied by their own recent rate,
> logistic regression beats the rule in five of five origins (mean 0.508 vs 0.212; base rate 0.257) and is
> the only ranker in the study that beats the population rate forward. Measured decision support for
> choosing which healthy page to inspect next month; no claim that a refresh changes the outcome.

---

## What I need from you before starting

1. **Go-ahead on the order** — WP1 and WP2 first, or start the notebook cells in parallel (faster, but the
   cells would need editing again if January fails a robustness check).
2. **One decision:** primary K. Everything to date is K = 50. Keeping 50 keeps continuity with the published
   paper; adding K = 20 as a co-primary shows the result is not a wide-queue artefact. I recommend both.
3. **One constraint check:** are you able to rerun the notebooks against the warehouse, or is the local cache
   (already committed manifest, ignored frames) the only environment for the next two weeks? That decides
   whether WP3 reuses `work/local_final_pipeline/` or rebuilds.

Nothing in this plan touches your headings, voice or section order. It adds cells and replaces the headline
numbers, and it makes the paper claim a result the ML can actually defend.

## Handover: the local run that closes the loop

No number in the paper, report or notebook copy about the onset pool is treated as measured
until this runs against the warehouse. Everything else in the shipped copy comes from receipts
that are already committed.

```bash
# 1. from the repository root, with the warehouse token already configured
python work/scripts/onset_screening.py --warehouse      # builds the frames once, then audits them
#    later runs can use the cached frames:  python work/scripts/onset_screening.py

# 2. the notebooks, in order
#    w04_baseline_score  (section 1b reads the committed revised receipts)
#    w05_model           (section 3b reads them, section 3c writes the onset receipt)
#    w06_validation_audit (section 2b reads the onset receipt and adds intervals and robustness)
#    w07_action_playbook  (section 1b builds the two lanes)
#    capstone             (section 4b re-checks the quoted numbers)

# 3. the offline check, which now requires the onset receipt
python work/scripts/verify_evidence.py
```

Two things fixed after the first attempt at step 1: the dataset revision is now read from
`work/outputs/final_pipeline/contract.json` (or the manifest, or `--revision`, or
`FLYRANK_HF_REVISION`) instead of from `work/outputs/warehouse_feature_experiments/experiment_results.json`,
which is a local artifact that is not part of the repository; and the frames land in
`work/local_final_pipeline_floor1000/`, which the notebooks now find on their own when the default
cache is empty.

What to read when it finishes:

- `work/outputs/onset_headline.md` holds the sentences with this run's numbers filled in.
  Compare them with Table 6 in `work/paper/index.html` and with the evaluation section of
  `work/capstone_report.md`. A gap wider than 0.02 on any single figure means the copy has to be
  adjusted to the run, not the other way round.
- `cross_check` inside `work/outputs/onset_screening.json` compares this script with the
  diagnostic receipt. `verdict` should read `matches`; `REVIEW` means the frames or a metric
  definition moved and the difference has to be explained before submission.
- `sensitivity` holds the threshold and impression-floor grid, the label variant and the
  bootstrap intervals. If the sign of the paired difference flips anywhere in that grid, the
  claim in the paper has to be softened to the cells where it holds.
- The `training population` line in w06 section 2b shows whether fitting on every training row
  instead of onset-eligible rows changes the result. `stable=False` is reportable, not fatal.

If the numbers land where the audit expects, nothing else changes. If they do not, the copy
follows the run.
