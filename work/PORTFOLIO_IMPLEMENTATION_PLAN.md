# Bounded portfolio implementation — 13 October 2026

Updates the earlier central project evidence; no prior per-project plan was
found in this remote tree. Completed locally: report/index/badges, fold prose
corrections, maintained capstone figure path, isolated offline verifier, pins,
and personal-evidence CI definition. Source metrics and starter code unchanged.

1. Review and publish the local diff; rebuild/deploy the paper so corrected fold
   counts replace the live overclaim. Verify paper renders and links resolve.
   Acceptance: main and gh-pages carry the same corrected prose and figures.
2. Inspect GitHub's Actions failure annotations/account runner availability.
   Failed jobs inspected here have zero executed steps and unavailable logs;
   do not rewrite model code speculatively. Run the new personal-evidence job
   after delivery. Acceptance: hosted 31-check evidence pass, or a clearly
   recorded external runner/access blocker without a green-CI claim.
3. Keep full warehouse rebuild as an authorized-access reproducibility tier;
   receipt checking remains a narrower, supported tier. No new model/tuning,
   raw ranked queue publication, or reference-pipeline overhaul is required.

Local evidence command: `python work/scripts/verify_evidence.py`. Environment
record: `work/requirements-evidence.txt`, distinct from historical model versions
in receipts. Do not optimize for perfect modelling: absence of future-month
base-rate lift is the finding. Preserve it in the README, paper and CV language.
