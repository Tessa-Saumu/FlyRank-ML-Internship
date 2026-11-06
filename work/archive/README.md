# Archive: earlier Lane 2 experiments (not part of the submission)

**Reviewers: you do not need to mark anything in this folder.** The submission is the freestyle study in
`work/notebooks/` (see `work/README.md`). This folder keeps my earlier work so the path that led to the final question
stays visible and auditable.

Both experiments were on **Lane 2: Refresh / Content Opportunity**, with the question: *which 50 pages per month should
an editor review because their search impressions will fall by more than 20% next month?*

| Folder | What it was | Headline (honest reading) | Why I moved on |
|---|---|---|---|
| `lane2_experiment1_grouped_and_forward/` | The first full submission (live paper as of October 2026): five features, a hand rule, logistic regression and random forest; client-grouped CV inside March 2026 plus one March-to-April forward test | grouped CV: logistic P@50 0.636 vs rule 0.512 (base 0.491); forward: 0.54 vs 0.44 against a 0.56 base rate | the forward test showed the edge did not hold up over time, and the rule never really ranked (its scores were mostly ties) |
| `lane2_experiment2_multi_origin_and_onset/` | The rebuilt pipeline on the full warehouse: five forward monthly origins, momentum features, a ranking diagnosis and an "onset" screening of pages not yet declining | a two-line acceleration rule (P@50 0.812) beat every model; the onset screen won (0.508 vs 0.212) but was carried by one month | the label was largely implied by momentum already visible in the features, so no model could earn its place on it |

**The lesson that shaped the final study:** check whether the label is answered by what is already visible before
modelling anything. The freestyle study builds its label so that momentum does not give it away, and documents the two
times it did during development (`work/notebooks/w03_feature_leakage_check.ipynb`).

These folders are frozen records. Their notebooks and scripts reference the repository layout they were written in, so
they may not rerun from here. To rerun one, check out the commit named in its README.

**Redaction note.** Pseudonymous page and client IDs that appeared in saved notebook outputs and receipts here were
replaced with `content_[redacted]` / `client_[redacted]` on 2026-11-05. Nothing else was changed.
