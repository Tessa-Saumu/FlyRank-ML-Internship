# My revised submission pipeline

I keep the original W03â€“W07 notebooks byte-for-byte in `work/archive/pre-warehouse-temporal/notebooks/`. Their old metrics, capstone and published paper describe my earlier analysis. They are not the revised pipeline results.

The completed development experiment is in `work/notebooks/warehouse_feature_experiments.ipynb`, its runner in `work/scripts/warehouse_feature_experiments.py`, and its aggregate receipt in `work/outputs/warehouse_feature_experiments/experiment_results.json`. The original static visibility/CTR/age rule remains the frozen primary baseline. LR/RF control and momentum models are compared against it; acceleration rules are additional comparators. The previous static-rule mean conservative P@50 was 36.4%, not a promised 20–30%. No score is forced into a target range. Model lift against the static rule does not establish superiority over acceleration or an editorial impact.

## Local run

Use Python 3.12 in a dedicated environment from the repository root:

```powershell
python -m venv .venv-final
.\.venv-final\Scripts\python.exe -m pip install -r work/requirements-final-pipeline.txt
.\.venv-final\Scripts\python.exe -m ipykernel install --user --name flyrank-final --display-name "FlyRank final"
```

Select **FlyRank final** in the notebook editor. Supply an authorized Hugging Face READ token through `HF_TOKEN` or the existing local `.env` mechanism; never commit a token. Keep enough disk space for the ten required source shards and local aggregated frames. The first W03 run may take time. It uses the exact revision recorded in the completed experiment, August 2025â€“May 2026; June is excluded.

Run each notebook top to bottom, in order:

1. `w03_data_contract.ipynb`: build/reuse private local frames and verify the contract.
2. `w03_feature_leakage_check.ipynb`: verify feature exclusions and label maturity.
3. `w04_signal_audit.ipynb`: inspect score buckets with sample sizes.
4. `w04_baseline_score.ipynb`: evaluate fixed rules and seeded random reference.
5. `w05_model.ipynb`: fit control/momentum LR and RF, latest mature month only.
6. `w06_validation_audit.ipynb`: compare identical candidate populations and paired differences.
7. `w07_action_playbook.ipynb`: create the private historical review queue; review at least 20 entries.
8. `capstone_revised.ipynb`: assemble the new receipt tables and complete the final interpretation/demo placeholders.

Frames and the private queue are in ignored `work/local_final_pipeline/`. Cache provenance includes the pinned dataset revision, configuration and source hashes; changed definitions require `pipeline.contract(rebuild=True)` in W03. Public aggregate receipts are written to `work/outputs/final_pipeline/`. Do not commit local frames, queues, tokens or identifier-bearing notebook outputs. The notebooks print aggregates only.

The old full experiment is preserved for history and sensitivity evidence. Running it again is not required to run W03â€“W07. The new W05 still fits 20 models, so allow time; keep its full output and resolve any warning before making claims.

## My results to complete

- INPUT AFTER W03: candidates, clients, label coverage and date bounds for each origin.
- INPUT AFTER W04: signal verdict across months, mean/worst P@50, P@20/P@100 and unknown outcomes.
- INPUT AFTER W05â€“W06: paired model-versus-rule deltas, wins, base rates, coverage sensitivity and client concentration.
- INPUT AFTER W07: top-20 manual review, weak picks, three wrong cases and proposed investigation steps, without identifiers.
- INPUT BEFORE PUBLICATION: update my capstone report, capstone notebook and paper with checked new receipts; replace old headline numbers only after the local rerun. Current published results remain historical.

My main claim will remain review prioritization from observed data. The Decemberâ€“April months influenced selection and are development comparisons. Seven days is a reporting-delay scenario. A fresh prospective test is needed before promoting a model. No causal traffic gain is measured.

## Software verification

`python work/scripts/check_final_pipeline_synthetic.py` builds temporary synthetic Parquet partitions, exercises the real query and executes all eight notebooks in temporary kernels. All test receipts and queues are temporary. Passing this check does not establish real warehouse performance. The committed working notebooks deliberately retain empty outputs and descriptive result placeholders.

Verification on 17 October 2026: all eight notebooks executed top to bottom against the synthetic warehouse fixture, including eligible candidates with unknown future outcomes. Experiment receipt invariants, archive-content comparison, Python compilation and git whitespace checks passed. The revised pipeline has not yet been rerun against the real warehouse.

Baseline correction: W06 paired deltas now use the original static rule. W07 writes that baseline queue for inspection, without automatically selecting a winner. Rebuild W03 if a previous local cache exists, then rerun the downstream notebooks because source provenance has changed. The raw-total target is unchanged; daily-rate and complete-coverage sensitivities must be reviewed for tracking artifacts.

Impressions threshold: the active pipeline requires at least **1,000 recent-30-day GSC impressions** for every candidate, including mature training frames. Rules and models use the same eligible population. The historical static rule retains its original weights and visibility predicate; all eligible pages now exceed that predicate. Rebuild W03 with `pipeline.contract(rebuild=True)`, then rerun W04–W07 and the revised capstone. Earlier 100-impression experiment scores do not predict results at the new floor.
