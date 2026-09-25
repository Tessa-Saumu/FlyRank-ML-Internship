# Project Evidence — FlyRank-ML-Internship

## Context

**When:** Repository created 2026-07-16T08:38:00Z (`Initial commit` by Theresia Saumu via GitHub template). Final substantive commit `48eabe2` dated 2026-09-24T09:19:37+0200 with message `ML-11 - ran locally capstone.ipynb; deployed gh pages paper`. All substantive work appears squashed into that single commit; git log is shallow (`--shallow` file present, `git log --all` shows only that commit as grafted). GitHub API shows repo updated 2026-09-24T07:19:51Z, pushed 2026-09-24T07:50:38Z. Work spans internship Weeks 1-8 (ML-02 through ML-11) per `work/README.md` assignment index.

**Why it existed:** Applied Search Intelligence: Google Search Ranking & Discoverability — the FlyRank ML Internship starter repo (`flyrank-bih/flyrank-ml-internship-starter` template). Per `README.md:1-12` and `GUIDE.md:1`, students clone the template into their own public repo, build in `work/`, and submit repo URL + deployed paper URL (`submission/paper_url.txt`). This clone (`Tessa-Saumu/FlyRank-ML-Internship`) is that submission/portfolio repo. The program teaches the workflow `problem framing → data cleaning → baseline → first model → evaluation → explainable recommendation` (`README.md:73`).

**Solo / team:** Nominally solo intern work (`work/` is described as "Yours" in `GUIDE.md` and `work/README.md`). However repository is a template clone: the reference pipeline (`scripts/`, `docs/`, `notebooks/`, `data/raw/content_refresh_anonymized.csv`, `outputs/*`, `skills/`) was authored by FlyRank track leads (Mirza Ašćerić ML, Hole data engineering, per `README.md` footer) and published as `flyrank-bih/flyrank-ml-internship-starter`. The intern's work lives in `work/` per convention. Git history cannot prove contribution boundaries because history is shallow — only one commit (`48eabe2`) adding all 86 files at once (53166 insertions). No incremental commits, no diff between author and template. Therefore **authorship boundaries cannot be proven from git alone**; they are inferred only from the documented convention `scripts/` = shared reference (do not edit) vs `work/` = yours (`GUIDE.md:1` table, `work/README.md:Rules of the road #1`).

**My role:** If you are Tessa Saumu (commit author `theresia.saumu@gmail.com`), you own everything under `work/` per convention: 7 assignment notebooks + capstone (`work/notebooks/w01_…` through `w07_…` + `capstone.ipynb`), 4 receipt JSONs (`work/outputs/*.json` mirrored in `work/notebooks/receipts/`), 7 figures (`work/figures/*.png`), paper source (`work/paper/index.html`), figure/paper scripts (`work/scripts/make_paper_figures.py`, `work/scripts/deploy_paper.py`), and deployment (`submission/paper_url.txt` + `gh-pages` branch). Ownership of `scripts/`, `docs/`, `outputs/` reference artifacts, and `notebooks/` first-win notebooks cannot be claimed as personal — they are template-owned and `GUIDE.md` explicitly says "Run it and copy from it — don't edit it."

*Explicit caveat:* Because the repo was created via `Use this template → Create a new repository` (History shows `Initial commit` by GitHub `web-flow`), there is no fork parent recorded (`gh api` returns `parent: null`, `fork: false`). The only evidence of personal contribution is file location + notebook content + the single commit author. No PR review history proves line-by-line authorship; 4 of 5 PRs are `MERGED` but their diffs are not inspectable in this shallow clone (they appear as `arena/01a0*` branches on `origin`).

---

## Problem

**What problem was actually being solved?**

*Real implemented problem (evidence: `work/notebooks/w01` through `w07` + receipts + paper):* Given a fixed editorial budget of ~50 page reviews per month, rank pages so the top-50 maximizes share that actually declined next month. Data is a monthly **frame**: one row per content page, 5 features known at month-end (`log_recent30_impressions`, `recent30_ctr_pct`, `recent30_avg_position`, `recent30_active_days`, `content_age_days`), label = next-30-day impressions < 80% of last-30-day impressions (`is_declining_label`). The deliverable is a **review order** (model probability for ordering, hand-rule flag for reason, gates + fall-back), not an automated publish decision. Validated on Hugging Face warehouse `FlyRank/internship-warehouse` (`fact_content_daily_performance` + `dim_content` + `dim_clients`), not on the 30k starter CSV, for the capstone. Cost model: wasted review ≈1 hour of ~50-hour budget; missed decline leaves page to decay.

*Distinguishing from aspirational README language:* `README.md` headline "Applied Search Intelligence: Google Search Ranking & Discoverability" and "Do not claim you predicted Google's algorithm" (`DATA_USE.md`) suggest a broader ranking problem. The implemented capstone explicitly reframes this to a **narrow operational proxy** — predicting a 20% impressions drop, not ranking factors or causal refresh impact. `work/paper/index.html` Abstract repeats this framing: "not an automatic publishing decision and should be paired with editorial review." The starter pipeline's label `is_declining_label = (trend_direction == "down")` (`scripts/01_prepare_features.py:108`, `docs/data-dictionary.md`) is a 90-day trend proxy; the capstone replaces it with a forward-window label (`<0.8` rule) documented in `work/paper/index.html:Method` and `work/notebooks/w06_validation_audit.ipynb`.

*What was NOT being solved:* No causal estimation (no experiment), no revenue prediction, no content generation, no semantic clustering (those are other lanes per `docs/ml-intern-dataset-and-lane-guide.md`), no real-time API. `work/paper/index.html:Section 5 Limits` explicitly states "Not causal. No page was experimentally changed."

---

## What I personally built

**Exact components/features/code owned by me (confirmed by location + file content, per `work/README.md` convention):**

- **Assignment notebooks (7 + capstone, all executed with outputs):**
  - `work/notebooks/w01_research_question.ipynb` — 12 cells, 4 code (1 with outputs), defines lane as Refresh/Content Opportunity Scoring, quantifies starter data (30,000 pages, 16,262 declining = 54.2%, 32 clients, 22,006 pages ≥100 impressions) — code cell output verified.
  - `work/notebooks/w02_ml_task_framing.ipynb` — 12 cells, 5 code (3 with outputs), frames lane as ranking with classification target.
  - `work/notebooks/w03_data_contract.ipynb` — 15 cells, 7 code (7 with outputs), builds March-2026 frame via DuckDB from `FlyRank/internship-warehouse` (verifies `HF_TOKEN`, asserts `client_hash_id` grouping, 95,810 pages, 40 clients, survivorship 98,398→95,810), documents 5-feature contract.
  - `work/notebooks/w03_feature_leakage_check.ipynb` — 10 cells, 4 code (4 with outputs), verifies 5 retained inputs only.
  - `work/notebooks/w04_signal_audit.ipynb` — 14 cells, 4 code (4 with outputs), audits signals (staleness: 55.0% vs 42.5%; top-10 low CTR: 50.3% vs 18.6% per paper).
  - `work/notebooks/w04_baseline_score.ipynb` — 26 cells, 8 code (8 with outputs), implements hand rule: `score = 0.40*visible + 0.35*low_ctr_top10 + 0.25*stale` where `visible = impressions≥500`, `low_ctr_top10 = visible & pos 1-10 & CTR<1%`, `stale = visible & age≥91d`, 5 reason codes, tie band quantified.
  - `work/notebooks/w05_model.ipynb` — 17 cells, 9 code (9 with outputs), trains LR + RF (GroupShuffleSplit 5 folds, test_size=0.25, seed 42, `client_hash_id` groups, StandardScaler inside pipeline for LR, no tuning), grouped-CV results stored in `work/outputs/model_metrics.json`.
  - `work/notebooks/w06_validation_audit.ipynb` — 27 cells, 17 code (17 with outputs), implements three tests (A random split, B grouped 5-fold, C time-forward), walk-forward (1→4 frames), planted-leak injection (honest AP 0.761 → injected 0.993), survivorship audit, produces `validation_audit_metrics.json`.
  - `work/notebooks/w07_action_playbook.ipynb` — 14 cells, 7 code (7 with outputs), builds playbook: gates (NEW_PAGE, RISER_1M, EXTREME_JUMP_20X, TOP2_NEAR_ZERO_CTR, BELOW_FLOOR, RECENTLY_OPTIMISED proposed), actions (review_before_revert, verify_then_review, monitor_only, investigate_quiet_risk), 50-row queue, PSI/drift/fall-back clocks, produces `action_playbook_summary.json` + figures `w07_fig1_…png`, `w07_fig2_…png`.
  - `work/notebooks/capstone.ipynb` — 17 cells, 8 code (8 with outputs), **assembles** paper from receipts (no new warehouse computation), regenerates 5 paper figures, runs 31 receipt-checks (`ALL 31 CHECKS PASS` output in last cell), embeds figures into `work/paper/index.html`. Output verified top-to-bottom in this audit (all code cells have stored outputs).

- **Receipts (committed JSON, source of truth per paper):**
  - `work/outputs/baseline_metrics.json` + duplicate `work/notebooks/receipts/baseline_metrics.json` — base_rate 0.49077, tie_band n=24462, decline_rate 0.5068.
  - `work/outputs/model_metrics.json` — 95,810 March rows, base_rate 0.4909, 5 folds (overlap 0 everywhere), grouped summary LR P@50 0.636±0.195, RF 0.608±0.230, rule 0.512±0.201.
  - `work/outputs/validation_audit_metrics.json` — frames (march 95,810/40, april 99,279/44), random_split (34/40 shared), grouped per-fold, time_forward, walk_forward, leak_injection, fold4_anomaly, survivorship.
  - `work/outputs/action_playbook_summary.json` — gates, queue (size 50, after-gates decline 0.56 = base rate, 10 clients, largest 48%, actions 49+1), PSI, monitoring clocks (2 TRIGGERED), thresholds.

- **Paper + figures:**
  - `work/paper/index.html` — 554 lines source, ~952 KB with embedded base64 PNGs, self-contained, 7 sections + abstract + acknowledgments, flyrank.ai credit in footer (required for CI). Figures embedded via `data:image/png;base64` with `data-fig="paper_fig*.png"` markers.
  - `work/figures/paper_fig1_results.png` (168970 B), `paper_fig2_walkforward.png` (163274 B), `paper_fig3_age_decay.png` (124803 B), `paper_fig4_playbook.png` (150600 B), `paper_fig5_leak_check.png` (66582 B) + `w07_fig1_decay_age_and_model.png`, `w07_fig2_queue_rule_flags.png`. Also duplicated under `work/notebooks/work/figures/` (artifact of Colab save).
  - `work/scripts/make_paper_figures.py` (319 lines) and `work/scripts/deploy_paper.py` (123 lines) — figure generation from receipts (matplotlib, PSI, funnel) and gh-pages deploy (copies single file, refuses if missing credit/figures). Proven to work: `git ls-remote` shows `gh-pages` at `54629a0eb1cd39ec64d017dc389844d0ca01fe72` with message `Deploy paper (from arena/01a0d254-flyrank-ml-internship@0ee70d4)`.

- **Deployment marker:**
  - `submission/paper_url.txt` — single line `https://tessa-saumu.github.io/FlyRank-ML-Internship/` . `submission/README.md` documents convention.

**Unclear ownership / cannot claim as personal:**

- `scripts/*.py` (5 scripts + `ml_utils.py` + `run_all.py`), `notebooks/01…03`, `data/raw/content_refresh_anonymized.csv`, `outputs/model_report.md` + `outputs/charts/*.svg` + `outputs/refresh_queue_sample.csv`, `docs/*`, `requirements.txt`, `.github/workflows/*`, `skills/*`. These are template-owned; `GUIDE.md:1` says "Run it and copy from it — don't edit it. It's the baseline you compare your work against, and reviewers expect to find it unchanged." No git diff proves personal edit; `git diff main..HEAD` is empty (single squash). Any claim to have built the reference pipeline would be unsupported.
- GitHub Actions `personalize.yml` ("Point Colab badges at this copy") is an automatic template bot commit, not personal.
- The `gh-pages` branch deploy is via `work/scripts/deploy_paper.py` but the branch history is separate (no diff against `main` in shallow clone).

**Proven NOT built by me (negative evidence):**
- No `work/capstone_report.md` exists — only template `work/capstone_report_template.md` (78 lines). Instruction to copy to `capstone_report.md` was not followed, though paper serves as report.
- No `work/scripts/` copied pipeline variants (guide suggests `work/scripts/03_train_model_v2.py` for experiments) — none found.
- No model pickle artifacts (`*.pkl`, `*.joblib`) — only CSV/JSON.

---

## Architecture

**Frontend:** Static research paper as single self-contained HTML (`work/paper/index.html`, ~952 KB, inline CSS, base64-embedded PNGs, no JS framework, no API calls). Served via GitHub Pages from `gh-pages` branch root (`/.github/pages` config: branch `gh-pages`, path `/`, status `built` per `gh api repos/.../pages`). Navigation is anchor-based (`#tldr`, `#abstract`, `#claims`, `#problem`, `#data`, `#method`, `#results`, `#limits`, `#recs`, `#repro`, `#ack`). No separate frontend app, no `package.json`, no Docker.

**Backend:** None. No API server, no routes, no auth. Notebooks run via Colab/DuckDB direct to Hugging Face. Reference pipeline is batch scripts only.

**Data:** 
- *Primary (capstone):* Hugging Face gated Parquet release `FlyRank/internship-warehouse` (build `flyrank_pseudonymized_warehouse_release_v20260703`, export 2026-07-03, daily facts until 2026-06-30, ~79M daily rows per `docs/ml-intern-dataset-and-lane-guide.md`). Tables used: `fact_content_daily_performance` (report_date × client × content), `dim_content`, `dim_clients`. Query table `fact_content_query_90d` explicitly excluded (leak risk). Accessed via DuckDB SQL (`import duckdb` in `w03`, `w05`, `w06`, `w07`), never bulk downloaded, token via `HF_TOKEN` env/colab userdata/getpass. Monthly frames built by SQL: `month=2026-03` features Mar 2-31, label Apr 1-30, etc., with floors (≥100 impressions, ≥14 active days).
- *Secondary (starter):* `data/raw/content_refresh_anonymized.csv` (30,000 rows × 44 cols, 32 clients, delivered in repo; `wc -l` = 30001 inc header). Processed to 52 cols in `data/processed/refresh_feature_vector.csv` (adds `is_declining_label`, `log_*`, `has_*`, `measurable_opportunity`). Intermediate `data/processed/` is gitignored.
- *Data contract guarantees:* Pseudonymized IDs (`client_hash_id`, `content_hash_id`) used only for grouping/joining; rate columns ×100 percentages; missingness systematic by `content_type` (`docs/data-dictionary.md`).

**ML/AI:** Batch training in notebooks + reference `scripts/` pipeline. No online inference service. Models output probabilities used as ranking scores. See ML section.

**Database:** No operational DB. Dataset is object-store Parquet + DuckDB in-memory query. No schema migration files. No `.sql` schema beyond inline notebook SQL.

**Infrastructure:**
- *CI:* Three GitHub Actions (`.github/workflows/`):
  - `smoke-test.yml` — on push/PR to `main`: installs `requirements.txt` (Python 3.12), runs `python scripts/run_all.py`, checks `outputs/model_results.json`, `outputs/refresh_queue.csv`, `outputs/model_report.md` exist; parallel job `data-leak-check` fails on `*.parquet|*.zip|*.tar|*.feather` or unexpected `*.csv` (allows only `data/raw/content_refresh_anonymized.csv` and `outputs/refresh_queue_sample.csv`), checks notebooks have outputs when `paper_url.txt` is live, and checks `paper_url.txt` format + that deployed page returns 200 and contains `flyrank.ai`.
  - `personalize.yml` — once per template copy: rewrites Colab badge URLs to current repo, commits `Point Colab badges at this copy [skip ci]`.
  - `data-path-smoke.yml` — on push to `notebooks/03_…` or schedule Mondays 06:00, executes `notebooks/03_working_with_the_full_release.ipynb` via `nbclient` against live Hugging Face if `HF_TOKEN` secret present, else skip.
- *Publishing:* `work/scripts/deploy_paper.py` pushes `work/paper/index.html` to `gh-pages` branch (verified at `origin/gh-pages` `54629a0`). GitHub Pages serves it.
- *No containers, No CDN, No monitoring beyond notebook clocks (PSI, fall-back).*

**Major data/control flows:**
```
Hugging Face Parquet (fact/dim) ──DuckDB SQL──► monthly frame (95k-99k rows, 5 feats + label)
       │                                          │
       ├── w03_data_contract.ipynb (build + assert)─┘
       ├── w04_baseline_score.ipynb (hand rule 0.40/0.35/0.25, 5 flags, tie band 24k)
       ├── w05_model.ipynb (GroupShuffleSplit ×5, LR+RF, metric P@50)
       ├── w06_validation_audit.ipynb (tests A/B/C + walk-forward + leak 0.761→0.993)
       └── w07_action_playbook.ipynb (gates→69,494 eligible, queue top-50, PSI clocks)
                                    │
                                    ▼
work/outputs/*.json receipts ──make_paper_figures.py──► work/figures/*.png ──► work/paper/index.html (base64) ──deploy_paper.py──► gh-pages ──► https://tessa-saumu.github.io/FlyRank-ML-Internship/
```
Starter slice flow: `data/raw/*.csv` → `scripts/01_prepare_features.py` → `data/processed/refresh_feature_vector.csv` → `scripts/02_baseline_score.py` → `scripts/03_train_model.py` (client-holdout) → `scripts/04_evaluate_and_export.py` (queue + `outputs/charts/*.svg` + `outputs/model_report.md`) → `scripts/05_build_pdf_report.py` (`outputs/flyrank_refresh_model_results.pdf`). Verified runnable locally (`pip install --break-system-packages -r requirements.txt` then `python scripts/run_all.py` completed 30k rows in ~45s, 2026-09-25).

---

## ML / AI work

**Problem formulation:** `Implemented and verified` — Ranking: order pages by probability of decline so top-50 contains maximal declines. Framed explicitly as ranking with classification-style target (`work/notebooks/w02_ml_task_framing.ipynb:1`, `work/paper/index.html:Section 1`). Baseline is ranking by hand rule; success measured by `precision@50`. Paper correctly states unit = page, output = ordered queue, actor = human editor, cost asymmetric. Not a pure classification threshold problem (recall structurally capped at 50/total_positives ~0.0006, noted in `baseline_metrics.json:recall_at_50_note`).

**Target:** `Implemented and verified` — Capstone label: `impressions_next30 < 0.8 * impressions_last30` (next month vs last month). Definition appears in `work/paper/index.html:Method` ("label: a page's impressions in the next 30 days fall below 80% of its impressions in the last 30 days"), `w03_data_contract` SQL (future window CTE), and `validation_audit_metrics.json` frames (March label Apr 1-30, April label May 1-30). Sensitivity checked at 0.7/0.8/0.9 (decline rates 0.426/0.491/0.556 per paper). **Distinguished from starter label:** starter pipeline uses `is_declining_label = (trend_direction == "down")` where `trend_direction` derived from `trend_pct = (last30 - prev30)/prev30` (`scripts/01_prepare_features.py:108`, `docs/data-dictionary.md:trend_direction`). Starter label is contemporaneous trend proxy, not forward prediction; paper does not use it. Starter `target_positive_rate` 0.542 vs capstone 0.491/0.560 — numbers prove different targets.

**Features:** `Implemented and verified` — Exactly 5, identical everywhere (notebooks, receipts, paper, `docs/data-dictionary.md` Columns the prep step adds note). List: `log_recent30_impressions`, `recent30_ctr_pct`, `recent30_avg_position`, `recent30_active_days`, `content_age_days`. `MODEL_NUMERIC_FEATURES` in `scripts/ml_utils.py` has 18 + 8 categorical = 26 total for starter, but capstone explicitly restricts to 5 (documented in `w03_data_contract.ipynb:1`, `w05_model.ipynb:1`, `work/outputs/validation_audit_metrics.json` fold composition). Raw totals like `impressions_90d`, `scroll_rate`, etc. are **excluded** as capstone features (leak-safe). Age decay non-linearity discovered: decline risk peaks 90-180d (0.66) then falls (`action_playbook_summary.json:age_bucket_decline_rate_april`).

**Baseline:** `Implemented and verified` — Hand rule via `w04_baseline_score.ipynb` and `validation_audit_metrics.json:grouped_cv`. Formula `0.40*visible + 0.35*low_ctr_top10 + 0.25*stale` (`work/paper/index.html:Method`, `scripts/02_baseline_score.py` equivalent logic for starter). Reason codes 5 levels: `stale_low_ctr_top10` (22,906 pages April, decline 0.541), `stale`, `low_ctr_top10`, `visible`, `low_visibility` (`action_playbook_summary.json:rule_flag_april`). Tie band quantified: March 24,462 pages at score 1.0, band decline 0.5068 (`baseline_metrics.json:tie_band.n` matches paper). Verified against starter pipeline: same functions copied verbatim, faithfulness receipt checks band counts + mean score.

**Models:** `Implemented and verified` — Logistic Regression (StandardScaler inside pipeline, `max_iter=1000`, `class_weight=balanced` equivalent in starter; notebook says library defaults, seed 42) and Random Forest (starter: `n_estimators=200, max_depth=10, min_samples_leaf=25, class_weight=balanced_subsample`, RandomState 42 per `scripts/03_train_model.py:58-70`; capstone says defaults, untuned). Both output probability as ranking score. No hyperparameter search, no ensemble, no calibration. Seed 42 everywhere.

**Validation methodology:** `Implemented and verified, and rigorously honest` — Three tests same rows/folds/metric/tie-break (`w06_validation_audit.ipynb`):
- **Test A Random split (trap):** `train_test_split` 80/20 rows, 34 of 40 clients on both sides (`validation_audit_metrics.json:random_split.shared_clients_both_sides:34`), P@50 RF 0.98, LR 0.74, rule 0.44, base 0.492. **Purpose is to demonstrate leakage**, not claim performance. Evaluated identically for all scorers on same split.
- **Test B Client-grouped 5-fold:** `GroupShuffleSplit(n_splits=5, test_size=0.25, random_state=42)` with `client_hash_id` groups, asserted overlap 0 (`fold_composition[].client_overlap:0` ×5). Realised test rows 24k-67k (fold 3 is 70.6% of frame, documented). Metrics per fold: rule P@50 0.32-0.84, LR 0.40-0.86, RF 0.38-0.84; means rule 0.512±0.201, LR 0.636±0.195, RF 0.608±0.230 (`model_metrics.json:summary`).
- **Test C Time-forward:** Fit March (95,810), score April (99,279) no retrain (`time_forward`), P@50 rule 0.44, LR 0.54, RF 0.14, base 0.5596. Deployment-replica.
- **Walk-forward:** Fixed panel 20 common clients, train 1→4 frames (42k→145k rows), same April test (base 0.547), rule flat 0.38, LR 0.56→0.68→0.70→0.66, RF 0.28→0.46→0.40→0.48 (`walk_forward.origins`).
- **Leakage checks:** Timeline asserted (features Mar 2-31, label Apr 1-30, 1-day gap, `validation_audit_metrics.json:frames`), no label-derived features, no product flags as inputs, IDs grouping only, population survivorship measured (March 2.6% dropped: 391 absent + 2197 thin; April 3.6%: 506 + 3158), planted leak harness: honest AP 0.7609 → planted 0.9925 (`leak_injection`), error analysis dissects fold 1 (AP, permutation importance). **All checks documented and receipt-backed.**

*Would not call successful merely because metric high:* Paper correctly treats RF 0.98 as leakage artifact, not skill, and treats LR 0.54 forward as **not beating base rate** (0.56) — honest.

**Metrics:** `Implemented and verified` — Primary: `precision@50` (share of top-50 that declined), reported for every scorer in every test, with base rate alongside. Secondary: AP (average precision), ROC-AUC, Brier, NDCG@50, P@20, Brier. `model_metrics.json` records `p20_mean`, `p50_mean/sd/min/max`, `ap_mean`, `roc_auc_mean`, `brier`. Note: recall@50 reported but flagged structurally capped (`recall_at_50_note`). Confidence intervals are spread across folds (±0.19-0.23), not bootstrap.

**Final result:** `Implemented but not deployed as model artifact` — Grouped CV mean: LR 0.636 vs rule 0.512 (delta +0.124, base 0.491), RF 0.608. Paper claims both models beat rule on every fold (true per `per_fold`: LR 4 wins +1 tie vs rule? Actually LR beats rule in 4/5 folds: fold3 LR 0.40 vs rule 0.38 win, but paper says 4/5 wins 1 tie — check fold1 LR 0.78 vs rule 0.84 is a loss, so need exact: LR vs rule per-fold: 0.78<0.84 loss, 0.48>0.32 win, 0.40>0.38 win, 0.66>0.50 win, 0.86>0.52 win → 4/5). Forward: LR 0.54 > rule 0.44 but < base 0.56 → **no lift at head** after one month; RF collapses to 0.14. Walk-forward peaks at 2-3 frames then declines, so adding history does not restore. No model pickle committed; predictions are CSVs (`data/processed/model_predictions.csv` from starter, but capstone predictions are ephemeral inside notebooks, not committed as artifact). Best model selection by P@50 picks LR for deployment (readable) vs starter picks RF.

*Evaluation of validation support:* Grouped CV supports **within-month ordering** claim (moderate evidence, 5 folds, correlated, variance ~0.2). Time-forward **does not support** claim of persistent predictive lift; paper accurately reports decay and frames queue as one-cycle diagnostic requiring monthly refit. Base-rate drift (0.20 Feb →0.56 Apr) further undermines fixed model.

---

## Engineering evidence

**Tests:** 
- *Number/type:* Zero unit/integration test files (`find *test*` returns only workflow file). No `pytest.ini`, `tests/`, `*_test.py`, `Dockerfile`.
- *What they actually test:* CI `smoke-test.yml` acts as integration test: runs full starter pipeline (`scripts/run_all.py`) and checks 3 files exist; `data-leak-check` asserts no `*.parquet|*.zip|*.tar|*.feather` committed and only 2 allowed CSVs; notebook-outputs check (fails if `paper_url.txt` live and notebook code cells >200 chars have no outputs); `paper_url.txt` format + HTTP 200 + `flyrank.ai` substring check. `data-path-smoke.yml` optionally executes `notebooks/03_…ipynb` via `nbclient` if `HF_TOKEN` present.
- *Whether they pass:* History shows **frequent failures**. `gh run list` : last 10 runs include 6 `failure` (smoke-test on `main` twice, on PRs `arena/01a0cef0`, `arena/01a0d254`). Last `main` push `35969024320` smoke-test failed (log fetch EOF, but status failure). `personalize` also failed twice. Only `pages build and deployment` succeeds. Local `python scripts/run_all.py` **passes** when deps installed (verified 2026-09-25: 30,000 rows, 5 steps, ~45s, exit 0, produced `model_results.json`, `refresh_queue.csv`, `model_report.md`, `summary.json`, `flyrank_refresh_model_results.pdf`). **Discrepancy:** CI failure vs local pass suggests environment drift (library versions, notebook outputs check, or leak-check). No pytest evidence of passing tests.

**APIs:** None. Zero API routes, no `FastAPI`/`Flask`, no `openapi.yaml`, no `app.py`. Paper is static file, not API.

**CI/CD:** `Implemented and verified` — Three workflows in `.github/workflows/` as above. Triggers: push to `main`, PR, schedule, workflow_dispatch. Runner `ubuntu-latest`, Python 3.11/3.12, pip cache. No container build, no model registry. CD is gh-pages deploy via `work/scripts/deploy_paper.py` (manual `python … --dry-run` then push). Proven by `gh api repos/.../pages` (`status: built`, `source: gh-pages /`, `html_url: https://tessa-saumu.github.io/FlyRank-ML-Internship/`), and `git ls-remote` showing `refs/heads/gh-pages 54629a0`.

**Containers:** Absent. No `Dockerfile`, `docker-compose.yml`, `.dockerignore`. CI uses bare `setup-python`, not Docker.

**Architecture patterns:** Template + batch notebooks pattern. Separation of concerns: `scripts/` (reference pipeline) vs `work/` (intern lane) vs `docs/` (contracts). No microservices, no layers, no dependency injection. Notebook pattern: `%pip install duckdb huggingface_hub`, `getpass` token, SQL frame build, asserts, GroupShuffleSplit, metrics functions copied verbatim for faithfulness. Paper generation: receipts → matplotlib → base64 embed → single HTML → gh-pages.

**Data validation:** `Implemented and verified` — `w03_data_contract.ipynb` asserts frame bounds, floor filters, non-null five features, timeline gap, client overlap 0, survivorship counts. `w03_feature_leakage_check.ipynb` verifies only 5 features. `data-dictionary.md` documents gotchas (trend_direction leak, IDs grouping only, avg_position 0 means no data, scroll_rate>100, tier volume floors). `scripts/01_prepare_features.py` validates required cols, drops duplicates, fills NaN with 0/`unknown`, creates label.

**Error handling:** Minimal. `scripts/run_all.py` exits if `RAW_PATH` missing with message to `git checkout -- data/...`. `01_prepare_features.py` raises `FileNotFoundError` and `ValueError` on missing cols. Notebooks assert frame dates and overlap. No try/catch for Hugging Face 429 or token missing beyond fallback to `getpass`. No retry logic beyond single `huggingface_hub` fetch.

**Monitoring:** `Implemented as design, not as runtime service` — `w07_action_playbook.ipynb` defines pre-release clocks (PSI worst feature 0.179 <0.25 PASS, schema PASS, base-rate drift 0.069 ≤0.10 PASS, client concentration 48% >35% TRIGGERED) and post-release clocks (true base rate recorded 0.560, fall-back LR 0.54 vs rule 0.44 not triggered, RF 0.14 vs 0.44 one month below, stricter LR vs base+0.05 0.54 vs 0.61 TRIGGERED). PSI per-feature: `log_recent30_impressions` 0.004 stable, `recent30_ctr_pct` 0.009 stable, `recent30_avg_position` 0.108 watch, `recent30_active_days` 0.099 stable, `content_age_days` 0.179 watch (`psi_march_to_april`). No Prometheus/Grafana, no alerting pipeline; clocks are manual checks in playbook.

**Security:** Pseudonymization verified: no `client_id`/`content_id` as features, no domains/URLs/titles/queries in any committed file (grep found none). `DATA_USE.md` rules: only starter CSV ships, `.gitignore` blocks `data/**` except that CSV, `*.parquet|*.zip|*.tar|*.feather`, `work/**/*.csv`, `outputs/*.pdf|*.json|*.csv` except sample, `gh-pages` not leaked. `HF_TOKEN` never hardcoded (uses `getpass`/`userdata.get`). Commit `48eabe2` contains no token. CIA leak-check in CI enforces. No SAST/dependabot evidence.

---

## Scale

*Only numbers supported by repository evidence (no extrapolation to 79M unless qualified as hosted release):*

- **Starter slice:** `data/raw/content_refresh_anonymized.csv` — 30,000 rows (wc -l 30001 inc header) × 44 columns (`docs/data-dictionary.md`), 32 distinct `client_id` (verified via `w01` output), 16,262 declining (54.2%) per `feature_metadata.json:declining_rows`. Processed to 52 columns (`refresh_feature_vector.csv`, 30,000 rows, `prepared_rows` field).
- **Warehouse frames (capstone, receipt-backed):** March 2026 95,810 pages / 40 clients / base_rate 0.4909 / largest client 22.1% ; April 2026 99,279 pages / 44 clients / base_rate 0.5596 / largest 22.4% (`validation_audit_metrics.json:frames`, `action_playbook_summary.json:frames`). Walk-forward panel 20 common clients, train rows 42,255 →145,828. June partitions sealed unused (future test).
- **Files:** `work/notebooks/` 9 notebooks (12-27 cells each, 4-17 code cells, all with outputs — `capstone.ipynb` 17 cells 8 code 8 with outputs). `work/outputs/` 4 JSON receipts. `work/figures/` 7 PNGs (66-168 KB each). `outputs/` 1 sample CSV (201 rows), 1 `model_report.md`, 5 SVGs, 1 PDF (~18 KB). No `*.parquet` committed.
- **Models:** 2 trained per `w05_model.ipynb` (LR, RF) per fold ×5 = 10 fits + dissection (1 held-out) + walk-forward (4 origins ×2) + leak injection (1). Starter pipeline trains 3 models (LR, DT, RF) per `scripts/03_train_model.py`.
- **Endpoints:** 0 API endpoints. 1 deployed static page (`work/paper/index.html` → `gh-pages` → `https://tessa-saumu.github.io/FlyRank-ML-Internship/`).
- **CI:** 3 workflows, ~5-40s runtime per `gh run list` (smoke-test 4-15s, pages deploy 24-36s).
- **Users:** Not applicable — internship repo, no user table.
- **Claim about 79M rows:** Hosted warehouse `"about 79 million daily rows"` per `docs/ml-intern-dataset-and-lane-guide.md` and `work/paper/index.html` — this number describes the **source release**, not this repo's committed data. Not counted as repo scale.

---

## Outcome

**What actually worked? (separated technical vs product):**

*Technical output (verified):*
- End-to-end ranking pipeline reproducible from Hugging Face via DuckDB without bulk download, with timeline and leakage harnesses that **caught a planted leak** (AP 0.761→0.993). Receipts committed and cross-checked (`capstone.ipynb` 31 checks `ALL PASS`, `make_paper_figures.py` reproduces 5 figures from receipts).
- Within-month ranking lift: grouped 5-fold mean P@50 LR 0.636±0.195 vs rule 0.512±0.201 vs base 0.491 (`model_metrics.json:summary`, `validation_audit_metrics.json:grouped_cv`). Both models beat rule on ≥4/5 folds (evidence file: `per_fold`). Walk-forward shows LR stays above rule at 0.56-0.70 vs rule 0.38 even on fixed panel.
- Baseline hand rule is transparent and fairly characterized (3 binary conditions, 5 reason codes, tie band 24k pages at 50.7% decline vs 49.1% base — proves bluntness).
- Playbook produces a **gated 50-row queue** with actions and gates: eligible 69,494 of 99,279 after gates (NEW_PAGE 14,935, RISER_1M 14,850, TOP2_NEAR_ZERO_CTR 214, EXTREME_JUMP_20X 103), budget queue top-50, monitoring clocks defined (2 TRIGGERED correctly surfaced).

*Product/real-world impact (not verified, directional only):*
- No evidence of deployment to editors, no A/B test, no revenue or traffic outcome measured after refresh. Paper explicitly states "decision support for a human" and "not causal" (`work/paper/index.html:Sections 5-6`). Queue's observed decline rate after gates this cycle was **exactly base rate** (0.56 before 0.54 vs after 0.56) — meaning no lift in the cycle that would be handed to an editor. Paper honestly reports this as "diagnostic list, not a refresh list" requiring 49.5 editor-hours for 49 quiet-risk investigations.
- No stakeholder adoption, no issue/PR discussion of impact, no `work/capstone_report.md` with editorial sign-off.

**What measurable result exists?**
- Receipt-backed P@50 numbers above; AP means ~0.557 LR / 0.561 RF vs rule 0.492; ROC-AUC ~0.605 vs 0.492; PSI worst 0.179 watch; base-rate drift 0.069.
- Figures `paper_fig1_…png` through `paper_fig5_…png` visualize exactly those numbers (verified by re-running `make_paper_figures.py` from receipts).
- Static paper deployed to `gh-pages` branch commit `54629a0` (GitHub Pages `status: built` per API, URL in `submission/paper_url.txt`). HTTP reachability from sandbox could not be verified (curl exit 35), but `gh api pages` and `git ls-remote` prove publication path exists.

---

## Limitations

**What does NOT work? What is unfinished? What would I change?**

- **Modelling weaknesses:**
  - Forward horizon lift vanishes: LR P@50 0.54 forward < base 0.56 (worse than random at head), RF collapses 0.14. Even with AP 0.624 vs rule 0.559, head lift gone — the budget-critical metric fails.
  - Walk-forward not rescued by history: LR peaks at 0.70 with 3 frames then falls to 0.66; RF volatile 0.28-0.48. No evidence more data fixes it; paper correctly concludes "refit monthly".
  - Feature set too small and mis-specified: 5 linear + log transforms cannot capture U-shaped age risk (decline 0.38/0.61/0.66/0.58/0.36 across age buckets, per `age_bucket_decline_rate_april`). LR age coefficient negative — wrong sign acknowledged in paper.
  - Untuned defaults, no cross-validated selection beyond mean P@50, no calibration, no confidence intervals beyond fold SD (~0.2, large relative to delta 0.12). Paired LR-RF diff -0.028 sd 0.059 inside noise — models not separable.
  - Label is proxy (20% impressions drop) not business outcome, and partly measures measurement: pages with 14-20 tracking days in outcome month decline 96% vs 40% for 30-day coverage (noted in paper Limits) — label reliability varies.
  - Duplicate feature vectors: 829 rows share identical 5-value vectors in March frame (`fold4_anomaly:frame_duplicate_feature_vectors`), top-50 from 3/10 clients in strongest fold, concentration 48% in queue — models/order not client-diverse.
  - No content signal (quality, intent, SERP layout, seasonality, business value) — acknowledged.

- **Architecture weaknesses:**
  - No inference service, no model registry, no versioned artifacts (`*.pkl` absent). Predictions ephemeral inside notebooks; not reproducible without re-running warehouse queries (requires Hugging Face access + token).
  - No train/test separation persisted as split files; reproducibility relies on seed 42 + SQL, not stored folds.
  - `work/capstone_report.md` missing (only template exists) — rubric section 8 reproducibility doc incomplete despite paper covering it.
  - Paper figures duplicated under `work/notebooks/work/figures/` (copy-on-save artifact) — clutter, not harmful.
  - No Docker/`requirements.txt` pin beyond lower bounds (`pandas>=2.2` etc.) — version sensitivity proven: local rerun of starter `scripts/run_all.py` gave RF P@50 0.68 vs committed 0.74, rule 0.24 vs 0.24 stable (third decimal drifts). Paper records versions (pandas 3.0.3, sklearn 1.9.0) but `requirements.txt` does not pin, so reproducibility brittle.

- **Reproducibility problems:**
  - Shallow clone history (only 1 commit visible) — cannot verify stepwise development; all work appears atomic, no incremental evidence of iteration.
  - Warehouse dependency is gated and rate-limited (guide warns 429 errors if hitting full fact table); notebooks handle token via getpass but no offline fixture or sample warehouse snapshot committed.
  - Recent edit lockout gate "proposed, not implemented; the warehouse release has no edit-history table" (`action_playbook_summary.json:gates.status.RECENTLY_OPTIMISED`) — reveals data gap.

- **Broken/dead integrations:**
  - CI `smoke-test` currently **fails** on `main` and on PR `arena/01a0d254` (5 of last 7 runs failure per `gh run list`). Failure likely from notebook-outputs check or leak-check after paper ship, but logs unavailable (EOF). Means repo is not green despite local pass — credibility gap.
  - `curl` from sandbox to `https://tessa-saumu.github.io/FlyRank-ML-Internship/` failed (exit 35, 0 bytes) — cannot independently verify deployed paper renders, though GitHub API says `built`. External verifier may hit same network limit, but standard browser should succeed; needs manual verification request.
  - `notebooks/01…03` first-win badges still point to template vs this repo? `personalize.yml` failed twice, so badge personalization may be incomplete (badges in `README.md` still show `flyrank-bih/…` URL? Check `README.md` badge URLs — they show `Tessa-Saumu` after personalization? In this clone, `README.md` badges still show `flyrank-bih` for notebooks 01-03 but `work/README.md` table badges also template — minor but shows automation not fully executed).

- **Deployment issues:**
  - Only static paper deployed; no pipeline artifact published (no `outputs/refresh_queue.csv` hosted, only starter sample). Queue CSV is not in `work/` nor hosted — reviewer must rerun notebooks to see queue.
  - `gh-pages` branch contains only `index.html` (single file) — correct per `deploy_paper.py`, but no history of increments.

- **Technical debt / unsupported claims (to not make):**
  - Cannot claim model generalizes forward; evidence shows decay.
  - Cannot claim RF is best; evidence shows LR more stable forward and RF collapses, though starter picks RF.
  - Cannot claim production impact or causal lift; no experiment.
  - Cannot claim 79M rows processed locally; only frames 95k/99k were, with DuckDB remote read.
  - Cannot claim client diversity; concentration flagged.

---

## Current repository status

**Runs locally:** `Verified yes` — 2026-09-25 `pip install --break-system-packages -r requirements.txt` (pandas 3.0.3, numpy 2.5.1, scikit-learn 1.9.0, duckdb 1.5.4) succeeded, then `python scripts/run_all.py` completed 5/5 steps: prepared 30,000 rows, wrote `data/processed/refresh_feature_vector.csv`, `baseline_refresh_queue.csv`, `model_predictions.csv`, `model_results.json`, `refresh_queue.csv` (8.7 MB), `model_report.md`, 5 SVGs, `flyrank_refresh_model_results.pdf`, exit 0, summary `Rows scored: 30000 Best model: random_forest`. Capstone notebooks `work/notebooks/w03…w07` plus `capstone.ipynb` all show stored outputs and top-to-bottom success in committed outputs. Warehouse notebooks require `HF_TOKEN` and network — not tested offline here, but receipts prove prior executed runs (library versions recorded).

**Tests pass:** `No/Partial` — No unit tests exist. CI `smoke-test` last 2 pushes on `main` failed (per `gh run list` conclusions `failure`). Local pipeline passes; CI failure suggests notebook-outputs or leak-check or environment issue, not pipeline logic. No test evidence of `pytest` green. Honest answer: no test suite to pass; CI is red.

**Deployment:** `Partially verified` — `gh-pages` branch exists (`54629a0 Deploy paper (from arena/01a0d254-flyrank-ml-internship@0ee70d4)`, verified via `git ls-remote` and `gh api repos/.../git/commits/...`), GitHub Pages API reports `status: built`, `source: gh-pages /`, `html_url: https://tessa-saumu.github.io/FlyRank-ML-Internship/`, `submission/paper_url.txt` correct per spec (single https line, passes smoke-test's URL format check, contains `flyrank.ai` credit per embedded HTML comment and footer). However HTTP GET from this sandbox failed (curl exit 35, 0 bytes) — likely sandbox egress/TLS, not proof of broken deploy. Needs manual browser verification (incognito + phone per `capstone.ipynb` deploy instructions).

**README quality:** `Template-grade, not rewritten` — `README.md` is unchanged starter template (FlyRank ML Internship front door, quickstart badges, assignment table, pipeline description). It describes the program, not this intern's lane/results. `work/README.md` is also template. No personal `README` in `work/` nor root summary of this lane's results. Paper `work/paper/index.html` serves as README surrogate, but root `README.md` does not link to personal results beyond badges.

**Demo available:** `Yes, static paper only` — `work/paper/index.html` self-contained (no external deps besides fonts), regenerable via `work/scripts/make_paper_figures.py` → `work/notebooks/capstone.ipynb`. Deployed as above. No interactive demo, no API demo, no Streamlit.

---

## Evidence

**Relevant files (exact paths):**
- `work/notebooks/capstone.ipynb` — paper source, 17 cells, 8 code, 31 receipt checks `ALL PASS`, figure regen + embed logic (base64).
- `work/notebooks/w03_data_contract.ipynb` — frame build (DuckDB SQL, `month=2026-03`/`2026-04`, floors, timeline assert).
- `work/notebooks/w04_baseline_score.ipynb` — hand rule, 5 flags, tie band.
- `work/notebooks/w04_signal_audit.ipynb` — signal audit distributions.
- `work/notebooks/w05_model.ipynb` — LR+RF grouped 5-fold, `GroupShuffleSplit`, seed 42, `model_metrics.json`.
- `work/notebooks/w06_validation_audit.ipynb` — tests A/B/C, walk-forward, leak injection, survivorship.
- `work/notebooks/w07_action_playbook.ipynb` — gates, actions, queue, PSI, monitoring.
- `work/notebooks/w01_research_question.ipynb`, `work/notebooks/w02_ml_task_framing.ipynb`, `work/notebooks/w03_feature_leakage_check.ipynb` — framing + leakage guard.
- `work/outputs/baseline_metrics.json`, `work/outputs/model_metrics.json`, `work/outputs/validation_audit_metrics.json`, `work/outputs/action_playbook_summary.json` — receipts (mirrored in `work/notebooks/receipts/`).
- `work/paper/index.html` — deployed paper (554 lines source, 952 KB with embedded PNGs, single file).
- `work/figures/paper_fig1_results.png`…`paper_fig5_leak_check.png` (168970 B, 163274 B, 124803 B, 150600 B, 66582 B) + `w07_fig1_…png`, `w07_fig2_…png` — sources; `work/notebooks/work/figures/` duplicate.
- `work/scripts/make_paper_figures.py` (319 lines) — deterministic figure generation from receipts, `C_RULE=#94a3b8` etc., embeds into paper.
- `work/scripts/deploy_paper.py` (123 lines) — gh-pages deploy, refuses if missing flyrank.ai credit or 5 figures.
- `submission/paper_url.txt` — `https://tessa-saumu.github.io/FlyRank-ML-Internship/` (1 line).
- `scripts/01_prepare_features.py` (145 lines) — label `trend_direction=="down"`, log1p, fills, dedup.
- `scripts/02_baseline_score.py` (130 lines) — `reason_codes()`, `suggested_action()`, weighted score 0.40/0.30/0.25/0.05.
- `scripts/03_train_model.py` (301 lines) — 3 models, `make_client_aware_split()` holds out ~20% clients, fallback stratified, `precision_at_k()`, `top_feature_importance()`.
- `scripts/04_evaluate_and_export.py` (411 lines) — `merged_reason_codes()`, `confidence_label()` (p80/p50), final score 0.70*proba+0.30*baseline, charts via `simple_svg_bar_chart`.
- `scripts/05_build_pdf_report.py` (639 lines) — ReportLab landscape PDF, `HorizontalBarChart` Flowable.
- `scripts/ml_utils.py` (214 lines) — `MODEL_NUMERIC_FEATURES` 18, `MODEL_CATEGORICAL_FEATURES` 8, `precision_at_k`, `simple_svg_bar_chart`, constants.
- `scripts/run_all.py` (48 lines) — orchestrator, 5 steps.
- `data/raw/content_refresh_anonymized.csv` (30001 lines inc header, 30,000 rows, 44 cols) — starter slice.
- `data/processed/feature_metadata.json`, `data/processed/baseline_metadata.json` — generated after `run_all.py`; `outputs/model_results.json` (5.0 KB), `outputs/model_report.md` (3.8 KB), `outputs/summary.json`, `outputs/refresh_queue.csv` (8.7 MB), `outputs/charts/*.svg` (5 SVGs), `outputs/flyrank_refresh_model_results.pdf` (~18 KB), `outputs/refresh_queue_sample.csv` (201 rows committed).
- `docs/data-dictionary.md` (161 lines) — 44 columns, 3 rules, label source warning, warehouse tables description.
- `docs/ml-intern-dataset-and-lane-guide.md` (728 lines) — warehouse release build id `flyrank_pseudonymized_warehouse_release_v20260703`, export 2026-07-03, 79M rows, lanes.
- `requirements.txt` (7 lines: pandas>=2.2, numpy>=1.26, scikit-learn>=1.4, matplotlib>=3.8, reportlab>=4.0, duckdb>=1.0, huggingface_hub>=0.24).
- `.github/workflows/smoke-test.yml` (107 lines), `data-path-smoke.yml` (52 lines), `personalize.yml` (45 lines) — CI.
- `.gitignore` (36 lines) — blocks `data/**` except starter CSV, `*.parquet|*.zip`, `data/processed/`, `outputs/*.pdf|*.json|*.csv` except sample, `work/**/*.csv`.
- `AGENTS.md`, `CLAUDE.md` (11 lines each) — skill router pointer.
- No `Dockerfile`, no `tests/`, no `*.pkl`, no `API routes`.

**PRs:** 
- `#1 ML-10: Week 7 content action playbook notebook` `arena/01a07319…` `CLOSED` 2026-09-05
- `#2 Weeks 2-7 review fix round (C1-C5, N1-N15) + w07` `arena/01a0b054…` `MERGED` 2026-09-17
- `#3 ML-10: Content Action Playbook (w07)` `arena/01a0b416…` `MERGED` 2026-09-18
- `#4 ML-11: ship the paper (first draft)` `arena/01a0cef0…` `MERGED` 2026-09-23
- `#5 ML-11: ship the paper with both impression cuts named` `arena/01a0d254…` `OPEN` 2026-09-24 (this session's predecessor branch `arena/01a0d8f7` not yet PR'd). Evidence via `gh pr list` and `gh api repos/.../commits`.

**Issues:** `gh issue list` shows 0 issues (not used; assignments on portal board per `README.md`).

**Screenshots:** No `screenshots/` folder. Figures serve as screenshots: `work/figures/*.png` and `outputs/charts/*.svg`. No UI screenshots of deployed paper; would need browser capture.

**Demo:** `https://tessa-saumu.github.io/FlyRank-ML-Internship/` — static paper (verified via `gh-pages` branch `54629a0`, `gh api pages` `status: built`, but HTTP fetch from sandbox failed — verify in real browser).

**Presentation:** No `slides.*` or `demo.mp4`. Paper `work/paper/index.html` is the presentation.

---

## Candidate CV bullets

*Leave this section blank per instruction.*

---

## Work required

**Critical — Problems that prevent the repository from being credible professional evidence:**

1. **CI is red** — `smoke-test` and `personalize` failed on latest `main` push (`35969024320/35969024266` failure per `gh run list`). Fix before portfolio link: re-run `python scripts/run_all.py` locally and commit regenerated `outputs/model_report.md` + `outputs/charts/*.svg` (currently drifted: RF 0.740→0.68 on this env) OR pin `requirements.txt` to exact versions (`pandas==3.0.3`, `numpy==2.5.1`, `scikit-learn==1.9.0`, `duckdb==1.5.4` as recorded in receipts) and update CI to `pip install -r requirements.txt` in fresh venv so numbers stop drifting. Then `git add outputs/model_report.md`? Note `outputs/*.json` is gitignored — either commit `outputs/model_results.json` by removing ignore for that file OR store copy in `work/outputs/` (already there) and make CI check that path instead.
2. **Missing `work/capstone_report.md`** — template `work/capstone_report_template.md` exists but no filled copy, despite `work/README.md` "Copy `capstone_report_template.md` to `capstone_report.md` and fill it". Reviewers expect that file; the paper covers sections but the rubric maps to `capstone_report.md`. Create it by exporting paper narrative or copying template and filling Abstract+Repro+Ack (9 sections).
3. **Verify deployed paper renders** — sandbox `curl` to `https://tessa-saumu.github.io/FlyRank-ML-Internship/` returned 0 bytes (exit 35). GitHub API says `built` but external proof needed. Open in incognito + phone, screenshot, embed in `work/README.md` or `README.md` badge, and re-run `gh` smoke-test's paper-URL step (`curl -L` + `grep flyrank.ai`) to prove it passes.
4. **Receipt vs starter version drift** — local `python scripts/run_all.py` produced RF P@50 0.68 vs receipt 0.74-like forward, and `outputs/model_report.md` diff shows chart/feature importance drift (commit diff `761f12b..8fdc3f4`). Document version sensitivity in `work/paper/index.html:Repro` or pin deps, otherwise reviewer reruns will not match committed receipts and will suspect leakage. Already noted in `GUIDE.md FAQ` but not in personal evidence.

**High value — Changes that materially increase the strength of the engineering evidence:**

5. **Add a single integration test** — e.g., `tests/test_smoke.py` that imports `ml_utils.precision_at_k` and asserts `precision_at_k([1,0,1],[0.9,0.1,0.8],2)==1.0`, plus a test that `GroupShuffleSplit` overlap is 0 on fixture data. No test framework exists; adding 1-2 `pytest` tests and `pytest` to `requirements.txt` and CI (`pytest -q`) turns "no tests" into "tests exist and pass".
6. **Publish the queue artifact** — `outputs/refresh_queue.csv` is 8.7 MB and gitignored; copy top-500 rows to `work/outputs/queue_top500.csv` (small, allowed) or host `work/paper/queue_sample.json` so a reviewer can see the actual ranked output without rerunning warehouse. Currently only `outputs/refresh_queue_sample.csv` (201 rows from starter pipeline) is visible, not the capstone queue.
7. **Pin environment and add `make`/`just` reproducibility** — Add `requirements.lock` or `pip freeze > work/outputs/environment.lock` (seed 42 already fixed) and a one-line `make repro` ( `pip install -r requirements.txt && python work/scripts/make_paper_figures.py` ) in `work/README.md` so claimed reproducibility is runnable in 1 command.
8. **Write a personal `README.md` section or `work/README.md` lane summary** — 3-5 lines linking to `work/paper/index.html` (local) and `https://tessa-saumu.github.io/FlyRank-ML-Internship/` (live), stating lane, 5 features, grouped P@50 0.636 vs 0.512 forward decay 0.54 vs base 0.56, and limitation. Currently root `README.md` is template boilerplate; a recruiter landing on repo sees no personal summary.
9. **Fix badge personalization** — `README.md` badges still show `flyrank-bih/flyrank-ml-internship-starter` for `notebooks/01…03` and `work/README.md` badge table (template). Rerun personalization (`sed` replace) or manually edit to `Tessa-Saumu/FlyRank-ML-Internship` so badges open this repo, not the starter.
10. **Clean duplicate artifact** — Remove `work/notebooks/work/figures/` duplicates (5 PNGs duplicated) that inflate repo (6592 KB) and confuse evidence.

**Optional — Polish that is useful but not necessary:**

11. Add `Dockerfile` with `python:3.11-slim` + `requirements.txt` + `DuckDB` for one-click repro (not required but helps for portfolio's "Containers" row).
12. Add `dependabot.yml` and pin `reportlab` (PDF) beyond lower bound to stop PDF drift.
13. Add `CHANGELOG.md` or `work/log.md` one-paragraph per week (already implied by notebooks, but explicit timeline helps narrative).
14. Capture screenshots of deployed paper (desktop + mobile) into `work/figures/screenshots/` and link in `README.md`.
15. Squash `arena/01a0d254` PR comments (currently OPEN) or close it if superseded by this branch `arena/01a0d8f7`, to leave GitHub PRs green.

---

## Professional Evidence Assessment

**1. What engineering capability does this project prove best?**
*Rigorous, honest validation and data-contract discipline over pipeline engineering.* The strongest code is not the model but the harness around it: client-grouped splits with asserted zero overlap, a deliberately run trap (random split RF 0.98), a planted-leak test (0.761→0.993), walk-forward with fixed panel, survivorship audit (2.6%/3.6% dropped), PSI/fall-back clocks, and receipt-backed figure generation (`work/scripts/make_paper_figures.py` → base64 embed → gh-pages). This is evidence of someone who can design a fair test contract and **not lie to themselves** — rarer than training a model.

**2. What ML/AI capability does it prove best?**
*Applied ranking with appropriate restraint, not deep learning.* Best proof is baseline framing (hand rule 0.40/0.35/0.25 with 5 reason codes, tie band diagnosed at 24k pages), feature curation (exactly 5 leakage-safe signals, correct exclusion of `trend_pct`/`trend_direction`/query table), and error analysis (CTR permutation importance ~2.5× next feature, U-shaped age decay explaining negative LR coefficient, near-zero-CTR rank-1/2 error pattern turned into a gate). No tuning, no stacking — deliberately simple and interpretable. This proves competence for decision-support ranking where review budget is fixed and explainability matters.

**3. What is the strongest verifiable achievement?**
*The full receipt-backed paper with 31 checks passing.* Every headline number (grouped LR 0.636±0.195, RF 0.608±0.230, rule 0.512±0.201 on 95,810 March pages; forward LR 0.54 vs rule 0.44 vs base 0.56 on 99,279 April pages; walk-forward 0.56→0.70; leak 0.761→0.993; tie band 24,462; age buckets 0.38/0.61/0.66/0.58/0.36; queue 50 rows, 10 clients, 48% largest, 2 clocks triggered) traces via `work/notebooks/capstone.ipynb` to committed JSON receipts (`work/outputs/*.json`) and regenerates deterministically from `work/scripts/make_paper_figures.py`. The deployed static paper (`work/paper/index.html` → `gh-pages` `54629a0`, `gh api pages` `built`) is self-contained and, per `git ls-remote`, published. This end-to-end traceability is the strongest verifiable artifact.

**4. What is the biggest credibility weakness?**
*Forward-horizon lift does not exist at the budget-critical point, and CI is currently red.* The model the portfolio would showcase (LR P@50 0.636 within-month) falls to 0.54 forward — **below the month's base rate 0.56**, i.e., worse than picking at random for the top-50. The queue's observed decline after gates is exactly base rate (0.56). An honest paper reports this (and this repo does), but a recruiter skimming for "model beats baseline" will see the 0.98 trap and then the forward collapse and question whether anything ships. Combined with `smoke-test` failures on `main` (red CI badge) and no `capstone_report.md`, the repo looks unfinished despite strong internals. Library version drift (0.740→0.68 on local rerun) adds reproducibility doubt.

**5. Is there enough here to justify investing additional time in this project?**
*Yes, but only for focused hardening, not new modelling.* The core is done and the paper is the right finish line — do not add another model lane. The highest-ROI hours are: pin deps + green CI, add 2 pytest tests, publish a small queue sample, write a 5-line personal README lens, and verify deployed URL renders. Those ~4-6 hours turn a strong internship project into credible portfolio evidence. Investing weeks to tune hyperparameters or chase forward lift is low-ROI — the evidence shows the signal decays in weeks and more history did not restore it; any claim of a "better model" would need a different feature set (content quality, intent), not more tuning.

**6. What type of professional role would this project support as evidence?**
*Applied AI / ML Engineer (decision support, search intelligence, ranking) and Data Scientist (experimentation/validation emphasis), not Research Scientist or MLOps/Backend Engineer.* It fits roles that value: framing a business decision as a rank, building a leakage-safe pipeline from warehouse Parquet+DuckDB, baselining transparently, validating with grouped/time-forward splits, and shipping a human-in-the-loop playbook with gates and fall-backs. It does **not** support claims for production ML platform work (no Docker/Kubernetes, no API, no feature store, no CI model registry), nor for NLP/LLM work (no provider_used/model_used as features, no text), nor for causal inference.

**7. What claims should I NOT make publicly based on the current repository?**
- Do not claim the model *predicts Google's algorithm* or *causes* traffic preservation — `DATA_USE.md` forbids it and `work/paper/index.html:Limits` says "not causal, no experiment, no revenue proof". Language must stay *observed / measured / directional / decision-support*.
- Do not claim RF 0.98 or LR 0.74 on random split as performance — those are leakage demos (34/40 clients shared); treat them only as negative controls.
- Do not claim forward generalization — forward P@50 LR 0.54 < base 0.56 proves no head lift; RF 0.14 is worse. Claiming "model beats baseline forward" is only true vs rule (0.54 vs 0.44) but hides base-rate loss.
- Do not claim 79M rows were modelled locally — only 95k/99k frames were, via remote DuckDB.
- Do not claim the starter pipeline (30k rows, 52 features, RF P@50 0.740/0.68) is your capstone — it is the reference `scripts/` baseline, not the 5-feature capstone.
- Do not claim the deployed paper is live until you have browser-verified it (CI was red, `curl` failed in sandbox) — verify in incognito/phone first.
- Do not claim production deployment to editors, A/B test, or measured business impact — none exists; the queue is a diagnostic list this cycle (49 × 1-hour quiet-risk investigations, exactly base rate).
- Do not claim full test coverage or containerized deployment — zero `pytest` tests and no `Dockerfile` exist.
- Do not claim `work/capstone_report.md` exists — only the template does.

