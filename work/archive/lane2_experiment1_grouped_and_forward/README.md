# Lane 2, experiment 1: grouped and forward validation (archived)

**Not part of the submission. Kept for the record of my process.**

- **Track:** Lane 2, Refresh / Content Opportunity.
- **Question:** which 50 pages should an editor review this month because their search impressions will fall by more
  than 20% next month?
- **Snapshot of:** `origin/main` at commit `c4f5853` (the version the live paper was built from in October 2026).
- **Contents:** the template notebooks w01 to w07 and capstone, the receipts in `outputs/`, the figures, the paper
  source and the scripts as they were.

## What was done

- Label: next-30-day impressions below 0.8 x the last 30 days (100-impression floor).
- Five month-end features: log impressions, CTR, average position, active days, content age.
- A three-condition hand rule versus logistic regression and random forest.
- Validation: five client-grouped splits inside March 2026, then one March-to-April forward test.

## What it showed

| Test | Base rate | Rule | Logistic | Random forest |
|---|---|---|---|---|
| March, clients held out (P@50) | 0.491 | 0.512 | 0.636 | 0.608 |
| March to April forward (P@50) | 0.560 | 0.44 | 0.54 | 0.14 |

## Why it was superseded

The forward test fell below the base rate for both models, the random forest collapsed, and the hand rule had only four
distinct score values, so it barely ranked. Experiment 2 investigated why.

To rerun: `git checkout c4f5853` and follow that commit's `work/README.md`.
