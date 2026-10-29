"""Onset-screening audit: WP1 and WP2 of the submission plan.

The question this script answers: once we remove the pages whose decline is already
implied by their own last-seven-day run rate, can a learned model order the rest better
than the frozen hand rule?

It reads the private monthly frames (built once by ``final_submission_pipeline.contract``)
or rebuilds them from the warehouse at a chosen impression floor, then runs:

WP1  January audit      coefficients, permutation importance, a two-feature mechanism
                        model, and robustness to the largest client and thin coverage.
WP2  Sensitivity        onset thresholds, coverage requirements, impression floors and a
                        decline-beyond-trajectory label variant, plus bootstrap intervals
                        on the paired per-origin deltas at K=20 and K=50 (co-primary).

Outputs
-------
``work/outputs/onset_screening.json``   every table and the claim checks
``work/outputs/onset_headline.md``      the sentence block, numbers filled in

Usage
-----
    python work/scripts/onset_screening.py                  # use the existing frame cache
    python work/scripts/onset_screening.py --warehouse      # rebuild frames from the release
    python work/scripts/onset_screening.py --selftest       # synthetic frames, no data needed

The definitions of the rule, the features and the label are imported from
``warehouse_feature_experiments`` so this audit cannot drift away from the pipeline
that produced the committed receipts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import warehouse_feature_experiments as study  # noqa: E402

TARGET = study.TARGET
K_VALUES = (20, 50)          # co-primary budgets
DEFAULT_ORIGINS = ("2025-12", "2026-01", "2026-02", "2026-03", "2026-04")
DEFAULT_K = 50
DEFAULT_THRESHOLD = 1.0      # onset pool: last-7-day rate at or above the 30-day rate
DEFAULT_MIN_DAYS = 3         # covered days required inside the last-seven-day window
SEED = 42

RULES = ("static_rule", "acceleration_rule", "coverage_adjusted_acceleration_rule")
MODELS = {
    "control:logistic_regression": "control",
    "control:random_forest": "control",
    "momentum:logistic_regression": "momentum",
    "momentum:random_forest": "momentum",
}


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #
def load_cached_frames(cache: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    manifest_path = cache / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No frame cache at {cache}. Build it with final_submission_pipeline.contract(), "
            "pass --cache, or run with --warehouse to rebuild from the release."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("provenance", {}).get("config", manifest.get("config", {}))
    months = config.get("feature_months") or sorted(
        path.stem for path in cache.glob("*.parquet")
    )
    frames = {month: pd.read_parquet(cache / f"{month}.parquet") for month in months}
    return frames, {"source": "cache", "cache": str(cache), "config": config}


def build_frames_from_warehouse(
    floor: int, cache: Path, revision: str | None = None, lag_days: int = 7
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Rebuild the monthly frames at a chosen impression floor.

    Mirrors ``final_submission_pipeline.contract`` (same query layer, same revision,
    same June guard) but parameterises the eligibility floor so WP2 can show the
    baseline's degeneracy at 100 versus 1,000 impressions.
    """
    from huggingface_hub import hf_hub_download

    config = study.ExperimentConfig(minimum_impressions=floor,
                                    reporting_lag_sensitivity_days=())
    months = study.months_between(
        study._feature_window(
            study._month_end(config.feature_months[0]) - study.timedelta(days=7)
        )["prior_start"],
        study._month_end(config.feature_months[-1]) + study.timedelta(days=30),
    )
    assert max(months) <= "2026-05" and "2026-06" not in months, "June must not be read"

    if revision is None:
        experiment = ROOT / "work/outputs/warehouse_feature_experiments/experiment_results.json"
        revision = json.loads(experiment.read_text(encoding="utf-8"))["source"]["huggingface_revision"]

    token = study.resolve_token()

    def download(name: str) -> str:
        return hf_hub_download(
            "FlyRank/internship-warehouse", name, repo_type="dataset",
            revision=revision, token=token,
        )

    paths = {
        month: download(f"fact_content_daily_performance/month={month}/data_0.parquet")
        for month in months
    }
    dim = download("dim_content.parquet")
    con = study.duckdb.connect()
    try:
        con.execute("SET memory_limit='2GB'")
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=false")
        built = study.load_origin_frames(
            con, config.feature_months, config, (lag_days,), paths, dim
        )[lag_days]
    finally:
        con.close()

    cache.mkdir(parents=True, exist_ok=True)
    for month, frame in built.items():
        frame.to_parquet(cache / f"{month}.parquet", index=False)
    manifest = {
        "provenance": {
            "revision": revision,
            "config": {**study.asdict(config), "feature_months": list(config.feature_months)},
            "source": "onset_screening.build_frames_from_warehouse",
        }
    }
    (cache / "manifest.json").write_text(json.dumps(manifest, default=str), encoding="utf-8")
    return built, {"source": "warehouse", "revision": revision, "floor": floor, "config": manifest["provenance"]["config"]}


def resolve_frames(
    cache: Path | str | None = None,
    floor: int | None = None,
    revision: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Return the monthly frames, from the local cache when present.

    The notebooks call this so a reader never has to remember which directory holds the
    frames. If the cache is absent the frames are rebuilt from the release at the
    submitted revision, using the recorded impression floor.
    """
    cache_path = Path(cache or os.environ.get(
        "FLYRANK_FINAL_CACHE", ROOT / "work/local_final_pipeline"))
    if (cache_path / "manifest.json").exists():
        return load_cached_frames(cache_path)
    if revision is None:
        contract = ROOT / "work/outputs/final_pipeline/contract.json"
        if contract.exists():
            revision = json.loads(contract.read_text(encoding="utf-8"))["provenance"]["revision"]
    if revision is None:
        raise SystemExit(
            "No frame cache and no recorded dataset revision. Run "
            "final_submission_pipeline.contract() once, or pass revision= explicitly."
        )
    return build_frames_from_warehouse(floor or 1000, cache_path, revision)


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def _rank(frame: pd.DataFrame, scores: np.ndarray, k: int) -> np.ndarray:
    return study._rank_indices(frame.reset_index(drop=True), np.asarray(scores, float), k)


def _labels(frame: pd.DataFrame) -> np.ndarray:
    return frame[TARGET]


def precision_block(frame: pd.DataFrame, scores: np.ndarray, k: int, mask=None) -> dict[str, Any]:
    """Precision@k under all three unknown-outcome conventions."""
    frame = frame.reset_index(drop=True)
    scores = np.asarray(scores, float)
    if mask is not None:
        mask = np.asarray(mask, bool)
        frame = frame.loc[mask].reset_index(drop=True)
        scores = scores[mask]
    if len(frame) < k:
        return {"rows": int(len(frame)), "available": False}
    idx = _rank(frame, scores, k)
    labels = _labels(frame).iloc[idx]
    confirmed = int(labels.eq(1).sum())
    unknown = int(labels.isna().sum())
    measurable = k - unknown
    return {
        "rows": int(len(frame)),
        "available": True,
        "precision_lower": confirmed / k,
        "precision_upper": (confirmed + unknown) / k,
        "precision_measurable": confirmed / measurable if measurable else None,
        "unknown": unknown,
        "base_rate": float(_labels(frame).mean()),
    }


def onset_pool(frame: pd.DataFrame, threshold: float, min_days: int) -> np.ndarray:
    """Pages whose recent run rate is not already collapsed."""
    rules = study._rule_scores(frame)
    ratio = -np.asarray(rules["acceleration_rule"], dtype=float)   # recent7 / (recent30 * 7/30)
    covered = np.asarray(frame["recent7_days"].ge(min_days), dtype=bool)
    elevated = np.asarray(ratio >= threshold, dtype=bool)
    return covered & elevated


def score_all(train: pd.DataFrame, test: pd.DataFrame, rules_test: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    scores = {name: rules_test[name] for name in RULES}
    for label, feature_set in MODELS.items():
        scores[label] = study._fit_score(train, test, feature_set, label.split(":")[1])
    return scores


def _design(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median")
    train_x = imputer.fit_transform(train[columns])
    test_x = imputer.transform(test[columns])
    scaler = StandardScaler().fit(train_x)
    return scaler.transform(train_x), scaler.transform(test_x)


def fit_custom(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Logistic regression on an explicit column list (mechanism tests only)."""
    from sklearn.linear_model import LogisticRegression

    known = train.loc[train[TARGET].notna()]
    train_x, test_x = _design(known, test, columns)
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(train_x, known[TARGET].astype(int))
    return model.predict_proba(test_x)[:, 1]


def with_momentum(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the pipeline's momentum columns and the two mechanism flags."""
    out = study._momentum(frame).copy()
    days = out["recent7_days"].replace(0, np.nan)
    out["run_rate_ratio_7d"] = (
        out["recent7_impressions"] / days * 30.0
        / out["recent30_impressions"].replace(0, np.nan)
    )
    out["bounce_in_downtrend"] = (
        out["traffic_velocity_30d"].lt(1.0) & out["traffic_acceleration_7d"].ge(1.0)
    ).astype(float)
    return out


# --------------------------------------------------------------------------- #
# WP1: within-origin diagnostics
# --------------------------------------------------------------------------- #
def coefficients_and_importance(
    train: pd.DataFrame, test: pd.DataFrame, seed: int = SEED
) -> dict[str, Any]:
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score

    known = train.loc[train[TARGET].notna()]
    x_train, x_test, numeric, categorical = study._design_matrices(known, test, "momentum")
    y_train = known[TARGET].astype(int)
    y_test = test[TARGET]
    measurable = y_test.notna().to_numpy()

    model = study.build_model("logistic_regression", numeric, categorical)
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]

    coefs = model.named_steps["model"].coef_[0]
    table = sorted(
        ({"feature": name, "coefficient_std_units": float(value)}
         for name, value in zip(numeric, coefs, strict=True)),
        key=lambda row: -abs(row["coefficient_std_units"]),
    )

    tree = study.build_model("random_forest", numeric, categorical)
    tree.fit(x_train, y_train)
    importance = permutation_importance(
        tree, x_test.loc[measurable], y_test[measurable].astype(int),
        scoring="average_precision", n_repeats=10, random_state=seed, n_jobs=4,
    )
    importance_table = sorted(
        ({"feature": name, "ap_drop_mean": float(mean), "ap_drop_std": float(sd)}
         for name, mean, sd in zip(
             numeric, importance.importances_mean, importance.importances_std, strict=True
         )),
        key=lambda row: -row["ap_drop_mean"],
    )

    return {
        "coefficients": table,
        "permutation_importance": importance_table,
        "pool_ap_logistic_regression": float(
            average_precision_score(y_test[measurable].astype(int), probability[measurable])
        ),
    }


def mechanism_test(train: pd.DataFrame, test: pd.DataFrame, k: int) -> dict[str, Any]:
    """Can two momentum features recover the eight-feature model on the onset pool?

    The hypothesis under audit: the model is finding pages in a 30-day downtrend whose
    last seven days ticked up, which then resume falling.
    """
    two_feature = ["traffic_velocity_30d", "traffic_acceleration_7d"]
    with_flag = two_feature + ["bounce_in_downtrend"]

    out: dict[str, Any] = {}
    for name, columns in (("mechanism_2f", two_feature), ("mechanism_2f_plus_flag", with_flag)):
        scores = fit_custom(train, test, columns)
        out[name] = precision_block(test, scores, k)
    full = study._fit_score(train, test, "momentum", "logistic_regression")
    out["momentum:logistic_regression"] = precision_block(test, full, k)
    out["bounce_share_of_pool"] = float(test["bounce_in_downtrend"].mean())
    out["bounce_pool_base_rate"] = float(
        test.loc[test["bounce_in_downtrend"].eq(1.0), TARGET].mean()
    )
    out["non_bounce_pool_base_rate"] = float(
        test.loc[test["bounce_in_downtrend"].eq(0.0), TARGET].mean()
    )
    return out


def robustness(pool: pd.DataFrame, scores: dict[str, np.ndarray], k: int) -> dict[str, Any]:
    """Does the pool lift survive dropping the biggest client and thin coverage?"""
    if pool.empty:
        return {}
    counts = pool["client_hash_id"].value_counts()
    keep_client = pool["client_hash_id"].ne(counts.index[0]).to_numpy()
    keep_coverage = (pool["recent30_days"].ge(30) & pool["recent7_days"].ge(7)).to_numpy()
    watched = ("static_rule", "acceleration_rule", "momentum:logistic_regression",
               "momentum:random_forest")
    return {
        "largest_client_share_of_pool": float(counts.max() / len(pool)),
        "full": {name: precision_block(pool, scores[name], k) for name in watched},
        "without_largest_client": {
            name: precision_block(pool, scores[name], k, mask=keep_client) for name in watched
        },
        "complete_coverage_only": {
            name: precision_block(pool, scores[name], k, mask=keep_coverage) for name in watched
        },
    }


# --------------------------------------------------------------------------- #
# WP2: bootstrap and sensitivity
# --------------------------------------------------------------------------- #
def bootstrap_delta(
    frame: pd.DataFrame,
    score_a: np.ndarray,
    score_b: np.ndarray,
    k: int,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Percentile interval for precision@k(a) - precision@k(b) on one origin.

    Pages are resampled with replacement inside the origin, which is the unit the lift is
    claimed on. Unknown outcomes stay in the queue as non-declines, matching the
    conservative convention used everywhere else.
    """
    labels = frame[TARGET].fillna(0).astype(int).to_numpy()
    ties = study._tie_keys(frame)
    a = np.asarray(score_a, float)
    b = np.asarray(score_b, float)
    n = len(frame)
    if n < k:
        return {"available": False}

    deltas = np.empty(reps, dtype=float)
    for rep in range(reps):
        idx = rng.integers(0, n, n)
        order_a = np.lexsort((ties[idx], -a[idx]))[:k]
        order_b = np.lexsort((ties[idx], -b[idx]))[:k]
        deltas[rep] = labels[idx][order_a].mean() - labels[idx][order_b].mean()

    observed = (
        labels[np.lexsort((ties, -a))[:k]].mean() - labels[np.lexsort((ties, -b))[:k]].mean()
    )
    return {
        "available": True,
        "observed_delta": float(observed),
        "mean": float(deltas.mean()),
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "share_above_zero": float((deltas > 0).mean()),
        "reps": int(reps),
    }


def sensitivity(
    frames: dict[str, pd.DataFrame],
    origin: str,
    k: int,
    reps: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Threshold, coverage and label-variant sensitivity for one origin."""
    rows: list[dict[str, Any]] = []
    train_all, train = study._mature_training_frames(frames, origin, 7)["latest_only"]
    full_test = with_momentum(frames[origin])

    for metric, name in (("raw", "threshold"), ("coverage", "coverage_threshold")):
        for threshold in (0.9, 1.0, 1.2):
            for min_days in (3, 5, 7):
                if metric == "raw":
                    rules = study._rule_scores(full_test)
                    ratio = -rules["acceleration_rule"]
                else:
                    days30 = full_test["recent30_days"].replace(0, np.nan)
                    days7 = full_test["recent7_days"].replace(0, np.nan)
                    ratio = (
                        (full_test["recent7_impressions"] / days7)
                        / (full_test["recent30_impressions"] / days30)
                    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                covered = np.asarray(full_test["recent7_days"].ge(min_days), dtype=bool)
                elevated = np.asarray(np.asarray(ratio, dtype=float) >= threshold, dtype=bool)
                mask = covered & elevated
                pool = full_test.loc[mask].reset_index(drop=True)
                if len(pool) < k or pool[TARGET].nunique() < 2:
                    continue
                train_pool = with_momentum(train)
                if train_pool[TARGET].nunique() < 2:
                    continue
                rules_pool = study._rule_scores(pool)
                lr = study._fit_score(train_pool, pool, "momentum", "logistic_regression")
                rows.append({
                    "pool_metric": metric,
                    "threshold": threshold,
                    "min_days": min_days,
                    "pool_rows": int(len(pool)),
                    "pool_base_rate": float(pool[TARGET].mean()),
                    "static_rule": precision_block(pool, rules_pool["static_rule"], k)["precision_lower"],
                    "acceleration_rule": precision_block(
                        pool, rules_pool["acceleration_rule"], k
                    )["precision_lower"],
                    "momentum:logistic_regression": precision_block(pool, lr, k)["precision_lower"],
                })

    # Label variant: decline measured against the page's own last-seven-day run rate.
    variant = frames[origin].copy()
    days = variant["recent7_days"].replace(0, np.nan)
    trajectory = days * (variant["recent7_impressions"] / days)
    measurable = variant["future30_days"].ge(14) & days.ge(3)
    variant[TARGET] = np.nan
    variant.loc[measurable, TARGET] = (
        variant.loc[measurable, "future30_impressions"]
        < 0.8 * trajectory[measurable]
    ).astype(int)
    variant = with_momentum(variant)
    mask = onset_pool(variant, DEFAULT_THRESHOLD, DEFAULT_MIN_DAYS)
    pool = variant.loc[mask].reset_index(drop=True)
    if len(pool) >= k and pool[TARGET].nunique() == 2:
        train_variant = with_momentum(train)
        rules_pool = study._rule_scores(pool)
        lr = study._fit_score(train_variant, pool, "momentum", "logistic_regression")
        rows.append({
            "pool_metric": "label_variant_beyond_trajectory",
            "threshold": DEFAULT_THRESHOLD,
            "min_days": DEFAULT_MIN_DAYS,
            "pool_rows": int(len(pool)),
            "pool_base_rate": float(pool[TARGET].mean()),
            "static_rule": precision_block(pool, rules_pool["static_rule"], k)["precision_lower"],
            "acceleration_rule": precision_block(
                pool, rules_pool["acceleration_rule"], k
            )["precision_lower"],
            "momentum:logistic_regression": precision_block(pool, lr, k)["precision_lower"],
        })
    return rows


def load_or_run(
    out_dir: Path | str,
    frames: dict[str, pd.DataFrame] | None = None,
    origins: list[str] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], bool]:
    """Read the audit receipt when it exists, otherwise run the audit."""
    path = Path(out_dir) / "onset_screening.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), True
    if frames is None:
        raise SystemExit(
            f"No receipt at {path}. Run Week 5 section 3c, or run "
            "work/scripts/onset_screening.py, before this cell."
        )
    return run(frames, origins or list(DEFAULT_ORIGINS), out_dir=Path(out_dir), **kwargs), False


# --------------------------------------------------------------------------- #
# Queue construction
# --------------------------------------------------------------------------- #
REASON_FOR_ACTION = {
    "stale_low_ctr_top10": "review_before_revert",
    "low_ctr_top10": "review_before_revert",
    "stale_visible": "review_before_revert",
    "visible_only": "monitor_only",
    "below_visibility_floor": "verify_then_review",
}


def reason_codes(queue: pd.DataFrame) -> np.ndarray:
    """The Week-4 rule flags, kept as the human-readable reason on each row."""
    visible = queue["recent30_impressions"].ge(500).to_numpy()
    ctr_flag = (
        queue["recent30_avg_position"].between(1, 10) & queue["recent30_ctr_pct"].lt(1.0)
    ).to_numpy()
    stale = queue["content_age_days"].ge(91).to_numpy()
    return np.select(
        [visible & ctr_flag & stale, visible & ctr_flag, visible & stale, visible],
        ["stale_low_ctr_top10", "low_ctr_top10", "stale_visible", "visible_only"],
        default="below_visibility_floor",
    )


def capped_order(frame: pd.DataFrame, scores: np.ndarray, k: int, cap: int | None) -> np.ndarray:
    """Top-k positions, optionally with at most ``cap`` rows per client."""
    order = study._rank_indices(frame.reset_index(drop=True), np.asarray(scores, float), len(frame))
    if cap is None:
        return order[:k]
    used: dict[str, int] = {}
    picked: list[int] = []
    for position in order:
        client = frame["client_hash_id"].iloc[position]
        if used.get(client, 0) >= cap:
            continue
        used[client] = used.get(client, 0) + 1
        picked.append(int(position))
        if len(picked) == k:
            break
    return np.asarray(picked, dtype=int)


def _triage_actions(selected: pd.DataFrame) -> np.ndarray:
    """Lane 1: a collapse is a measurement question before it is an editorial one."""
    out = np.asarray([REASON_FOR_ACTION[reason] for reason in selected["reason"]], dtype=object)
    thin = selected["recent30_days"].lt(21).to_numpy() | selected["recent7_days"].lt(5).to_numpy()
    collapsed = selected["run_rate_ratio_7d"].lt(0.30).to_numpy()
    out[collapsed] = "investigate_collapse"
    out[thin] = "verify_then_review"
    return out


def _onset_actions(selected: pd.DataFrame) -> np.ndarray:
    """Lane 2: separate a data problem, a click problem and a drift problem."""
    low_ctr = selected["reason"].isin(["stale_low_ctr_top10", "low_ctr_top10"]).to_numpy()
    thin = selected["reason"].eq("below_visibility_floor").to_numpy()
    drift = selected["bounce_in_downtrend"].eq(1.0).to_numpy()
    return np.select(
        [thin, low_ctr, drift],
        ["verify_then_review", "review_before_revert", "watch_for_resume"],
        default="monitor_only",
    )


def _lane(queue: pd.DataFrame, scores: np.ndarray, k: int, cap: int | None, lane: str) -> dict[str, Any]:
    positions = capped_order(queue.reset_index(drop=True), scores, k, cap)
    selected = queue.reset_index(drop=True).iloc[positions].copy()
    reasons = reason_codes(selected)
    selected["lane"] = lane
    selected["reason"] = reasons
    selected["action"] = _triage_actions(selected) if lane == "triage" else _onset_actions(selected)
    counts = selected["client_hash_id"].value_counts()
    return {
        "rows": selected,
        "summary": {
            "lane": lane,
            "selected": int(len(selected)),
            "decline_rate_lower": float(selected[TARGET].fillna(0).mean()),
            "unknown": int(selected[TARGET].isna().sum()),
            "clients": int(counts.size),
            "largest_client_share": float(counts.max() / len(selected)) if len(selected) else None,
            "actions": {str(k2): int(v) for k2, v in selected["action"].value_counts().items()},
            "reasons": {str(k2): int(v) for k2, v in pd.Series(reasons).value_counts().items()},
            "run_rate_bands": {
                str(k2): int(v) for k2, v in pd.cut(
                    selected["run_rate_ratio_7d"],
                    [0, 0.3, 0.5, 0.8, 1.0, np.inf],
                    labels=["<0.3 collapsed", "0.3-0.5", "0.5-0.8", "0.8-1.0", ">=1.0 healthy"],
                    include_lowest=True,
                ).value_counts(sort=False).items()
            },
        },
    }


def build_queues(
    frames: dict[str, pd.DataFrame],
    origin: str | None = None,
    k: int = 50,
    cap: int | None = 5,
) -> dict[str, Any]:
    """Two lanes for one origin: triage (already sliding) and onset (still healthy).

    Lane 1 ranks by the momentum rule and is the honest place for near-term repair work.
    Lane 2 ranks the still-healthy pool by the model, which is the only setting in this
    study where the learned ranking beats the hand rule. Neither lane claims that an edit
    changes the outcome.
    """
    origin = origin or DEFAULT_ORIGINS[-1]
    test = with_momentum(frames[origin])
    train_months, train = study._mature_training_frames(frames, origin, 7)["latest_only"]
    train = with_momentum(train)

    rules = study._rule_scores(test)
    triage = _lane(test, rules["coverage_adjusted_acceleration_rule"], k, cap, "triage")

    mask = onset_pool(test, DEFAULT_THRESHOLD, DEFAULT_MIN_DAYS)
    pool = test.loc[mask].reset_index(drop=True)
    train_pool = train.loc[onset_pool(train, DEFAULT_THRESHOLD, DEFAULT_MIN_DAYS)].reset_index(drop=True)
    model_scores = study._fit_score(train_pool, pool, "momentum", "logistic_regression")
    onset = _lane(pool, model_scores, k, cap, "onset")

    return {
        "origin": origin,
        "training_months": list(train_months),
        "client_cap": cap,
        "triage": triage,
        "onset": onset,
        "scope": (
            "Development ranking for human review. Lane 1 marks traffic already sliding; "
            "lane 2 marks still-healthy pages the model expects to slip. No causal claim."
        ),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(
    frames: dict[str, pd.DataFrame],
    origins: list[str],
    *,
    k_values: tuple[int, ...] = K_VALUES,
    threshold: float = DEFAULT_THRESHOLD,
    min_days: int = DEFAULT_MIN_DAYS,
    reps: int = 2000,
    seed: int = SEED,
    out_dir: Path | None = None,
    provenance: dict[str, Any] | None = None,
    do_sensitivity: bool = True,
    cross_check_receipt: bool = True,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    per_origin: dict[str, Any] = {}

    for origin in origins:
        if origin not in frames:
            continue
        test = with_momentum(frames[origin])
        train_months, train = study._mature_training_frames(frames, origin, 7)["latest_only"]
        train = with_momentum(train)

        mask = onset_pool(test, threshold, min_days)
        pool = test.loc[mask].reset_index(drop=True)
        rules_pool = study._rule_scores(pool)

        # Primary training policy: the model is fitted on onset-eligible rows only, so the
        # population it learns from is the population it ranks. Fitting on every row would
        # teach it how a collapse looks, which is the signal this design exists to exclude.
        train_pool = train.loc[onset_pool(train, threshold, min_days)].reset_index(drop=True)
        if train_pool[TARGET].nunique() != 2 or len(train_pool) < 100:
            raise SystemExit(
                f"{origin}: only {len(train_pool)} onset-eligible training rows with both "
                "classes. Use the 1,000-impression frame or a later origin."
            )
        scores = score_all(train_pool, pool, rules_pool)
        scores_all_rows = score_all(train, pool, rules_pool)

        entry: dict[str, Any] = {
            "origin": origin,
            "training_months": list(train_months),
            "training_rows": int(len(train)),
            "candidates": int(len(test)),
            "pool_rows": int(len(pool)),
            "pool_share_of_candidates": float(mask.mean()),
            "pool_base_rate": float(pool[TARGET].mean()),
            "pool_unknown_share": float(pool[TARGET].isna().mean()),
            "coverage": {
                "recent7_days_median": float(pool["recent7_days"].median()),
                "recent30_days_median": float(pool["recent30_days"].median()),
                "bounce_share": float(pool["bounce_in_downtrend"].mean()),
            },
            "metrics": {
                name: {f"p_at_{k}": precision_block(pool, score, k) for k in k_values}
                for name, score in scores.items()
            },
            "metrics_all_training_rows": {
                name: {f"p_at_{k}": precision_block(pool, score, k) for k in k_values}
                for name, score in scores_all_rows.items()
            },
            "bootstrap": {
                f"p_at_{k}": {
                    "momentum:logistic_regression_minus_static_rule": bootstrap_delta(
                        pool, scores["momentum:logistic_regression"], scores["static_rule"], k, reps, rng
                    ),
                    "momentum:random_forest_minus_static_rule": bootstrap_delta(
                        pool, scores["momentum:random_forest"], scores["static_rule"], k, reps, rng
                    ),
                    "momentum:logistic_regression_minus_acceleration_rule": bootstrap_delta(
                        pool, scores["momentum:logistic_regression"],
                        scores["acceleration_rule"], k, reps, rng,
                    ),
                }
                for k in k_values
            },
            "mechanism": mechanism_test(train_pool, pool, DEFAULT_K),
            "robustness": robustness(pool, scores, DEFAULT_K),
            "diagnostics": coefficients_and_importance(train_pool, pool, seed),
        }
        per_origin[origin] = entry
        print(f"  origin {origin}: pool {len(pool):,} of {len(test):,} "
              f"({mask.mean():.0%}), base rate {entry['pool_base_rate']:.3f}", flush=True)

    summary = summarise(per_origin, k_values)
    checks = claim_checks(per_origin, summary, k_values)
    real_frames = (provenance or {}).get("source") in {"cache", "warehouse"}
    cross_check = (
        cross_check_diagnosis(per_origin, DEFAULT_K)
        if cross_check_receipt and real_frames
        else {"available": False, "note": "synthetic run: no receipt comparison"}
    )

    payload: dict[str, Any] = {
        "notebook": "onset_screening",
        "seed": seed,
        "provenance": provenance or {},
        "configuration": {
            "k_values": list(k_values),
            "onset_threshold": threshold,
            "onset_min_days": min_days,
            "bootstrap_reps": reps,
            "decline_threshold": 0.8,
            "label_min_coverage_days": 14,
            "origins": origins,
        },
        "origin_results": per_origin,
        "summary": summary,
        "claim_checks": checks,
        "cross_check_vs_ranking_diagnosis": cross_check,
        "scope": (
            "Five development origins, latest-mature-month training, paired within origin. "
            "Outcome windows overlap and pages and clients repeat, so these are development "
            "slices, not independent tests. The onset pool excludes pages whose own last-seven-day "
            "rate already implies a decline; that is the whole point of the comparison."
        ),
    }
    if do_sensitivity and per_origin:
        payload["sensitivity"] = {
            origin: sensitivity(frames, origin, DEFAULT_K, max(200, reps // 4), seed)
            for origin in per_origin
        }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "onset_screening.json").write_text(
            json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
        )
        (out_dir / "onset_headline.md").write_text(headline(payload), encoding="utf-8")
    return payload


def summarise(per_origin: dict[str, Any], k_values: tuple[int, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    candidates = sorted({name for entry in per_origin.values() for name in entry["metrics"]})
    for k in k_values:
        rows = {name: [] for name in candidates}
        deltas = {"momentum:logistic_regression": [], "momentum:random_forest": []}
        for entry in per_origin.values():
            for name in candidates:
                block = entry["metrics"][name][f"p_at_{k}"]
                if block["available"]:
                    rows[name].append(block["precision_lower"])
            static = entry["metrics"]["static_rule"][f"p_at_{k}"]["precision_lower"]
            for name in deltas:
                block = entry["metrics"][name][f"p_at_{k}"]
                if block["available"]:
                    deltas[name].append(block["precision_lower"] - static)
        table = []
        for name in candidates:
            values = rows[name]
            if not values:
                continue
            table.append({
                "candidate": name,
                "mean_precision": float(np.mean(values)),
                "median_precision": float(np.median(values)),
                "worst": float(np.min(values)),
                "best": float(np.max(values)),
                "origins": len(values),
            })
        out[f"p_at_{k}"] = {
            "table": table,
            "paired_vs_static_rule": {
                name: {
                    "mean_delta": float(np.mean(values)) if values else None,
                    "median_delta": float(np.median(values)) if values else None,
                    "positive_origins": int(sum(value > 0 for value in values)),
                    "origins": len(values),
                    "leave_one_out_means": [
                        float((sum(values) - value) / (len(values) - 1))
                        for value in values
                    ] if len(values) > 1 else [],
                }
                for name, values in deltas.items()
            },
            "base_rate_mean": float(np.mean([e["pool_base_rate"] for e in per_origin.values()])),
        }
    return out


def claim_checks(per_origin: dict[str, Any], summary: dict[str, Any], k_values: tuple[int, ...]) -> dict[str, Any]:
    """Every sentence the write-up will claim, checked against this run."""
    checks: dict[str, Any] = {}

    for k in k_values:
        block = summary.get(f"p_at_{k}", {})
        paired = block.get("paired_vs_static_rule", {}).get("momentum:logistic_regression", {})
        table = {row["candidate"]: row for row in block.get("table", [])}
        lr = table.get("momentum:logistic_regression", {}).get("mean_precision")
        static = table.get("static_rule", {}).get("mean_precision")
        accel = table.get("acceleration_rule", {}).get("mean_precision")
        base = block.get("base_rate_mean")
        checks[f"p_at_{k}"] = {
            "logistic_beats_static_in_at_least_4_of_5": bool(
                paired.get("positive_origins", 0) >= 4
            ),
            "logistic_beats_base_rate_on_mean": bool(
                lr is not None and base is not None and lr > base
            ),
            "momentum_rule_has_no_structural_edge_in_pool": bool(
                accel is not None and base is not None and accel <= base + 0.05
            ),
            "values": {
                "logistic_mean": lr, "static_mean": static,
                "acceleration_mean": accel, "base_rate_mean": base,
            },
        }

    spreads = []
    for entry in per_origin.values():
        full = entry["metrics"].get("momentum:logistic_regression", {}).get("p_at_50", {})
        other = entry.get("metrics_all_training_rows", {}).get(
            "momentum:logistic_regression", {}).get("p_at_50", {})
        if full.get("available") and other.get("available"):
            spreads.append(abs(full["precision_lower"] - other["precision_lower"]))
    if spreads:
        checks["training_population"] = {
            "max_abs_difference_pool_vs_all_training": float(max(spreads)),
            "stable_across_training_population": bool(max(spreads) <= 0.10),
        }

    january = per_origin.get("2026-01", {})
    robustness_block = january.get("robustness", {})
    if robustness_block:
        full = robustness_block["full"].get("momentum:logistic_regression", {})
        dropped = robustness_block["without_largest_client"].get("momentum:logistic_regression", {})
        complete = robustness_block["complete_coverage_only"].get("momentum:logistic_regression", {})
        checks["january_robustness"] = {
            "full_precision": full.get("precision_lower"),
            "without_largest_client": dropped.get("precision_lower"),
            "complete_coverage_only": complete.get("precision_lower"),
            "survives_dropping_largest_client": bool(
                dropped.get("precision_lower") is not None
                and dropped["precision_lower"] > 0.5
            ),
        }
    return checks


def cross_check_diagnosis(per_origin: dict[str, Any], k: int) -> dict[str, Any]:
    """Compare with the earlier diagnostic receipt, when it is present.

    Both scripts call the same ranking and metric definitions, so agreement to a few
    thousandths is expected. A larger gap means the frames or the definitions moved.
    """
    path = ROOT / "work/outputs/diagnostics/ranking_diagnosis.json"
    if not path.exists():
        return {"available": False, "note": "ranking_diagnosis.json not found"}
    prior = json.loads(path.read_text(encoding="utf-8")).get("origins", {})
    watched = ("static_rule", "acceleration_rule", "momentum:logistic_regression",
               "momentum:random_forest")
    rows = []
    for origin, entry in per_origin.items():
        previous = prior.get(origin, {}).get("D4_onset")
        if not previous:
            continue
        source = entry["metrics"]
        for name in watched:
            current = source.get(name, {}).get(f"p_at_{k}", {})
            old = previous.get(name, {}).get("p_at_k")
            if current.get("available") and old is not None:
                rows.append({
                    "origin": origin,
                    "candidate": name,
                    "diagnosis_p_at_k": float(old),
                    "onset_screening_p_at_k": float(current["precision_lower"]),
                    "difference": float(current["precision_lower"] - old),
                })
    if not rows:
        return {"available": False, "note": "no comparable rows in the diagnostic receipt"}
    worst = max(rows, key=lambda row: abs(row["difference"]))
    return {
        "available": True,
        "rows": rows,
        "max_abs_difference": abs(worst["difference"]),
        "worst_row": worst,
        "verdict": "matches" if abs(worst["difference"]) <= 0.02 else "REVIEW: frames or definitions differ",
    }


def headline(payload: dict[str, Any]) -> str:
    """The sentence block for the write-up, with this run's numbers filled in."""
    def table(k: int) -> dict[str, dict[str, Any]]:
        return {row["candidate"]: row for row in payload["summary"].get(f"p_at_{k}", {}).get("table", [])}

    def value(rows: dict[str, dict[str, Any]], name: str, field: str = "mean_precision") -> str:
        row = rows.get(name)
        return "n/a" if row is None else f"{row[field]:.3f}"

    rows50, rows20 = table(50), table(20)
    block50 = payload["summary"].get("p_at_50", {})
    base = block50.get("base_rate_mean")
    paired = block50.get("paired_vs_static_rule", {}).get("momentum:logistic_regression", {})
    paired_rf = block50.get("paired_vs_static_rule", {}).get("momentum:random_forest", {})
    n_origins = len(payload["origin_results"])
    base_text = "n/a" if base is None else f"{base:.3f}"

    lines = [
        "# Onset screening: generated headline block",
        "",
        "Generated by `work/scripts/onset_screening.py` from",
        "`work/outputs/diagnostics/onset_screening.json`. Do not edit by hand; rerun instead.",
        "",
        f"Origins: {n_origins}. Pool: pages whose last-seven-day impression rate is at or above",
        f"their 30-day rate, with at least {payload['configuration']['onset_min_days']} covered days",
        f"in that window. Pool base rate {base_text}.",
        "",
        "Mean precision@50 by ranker:",
        f"- frozen hand rule {value(rows50, 'static_rule')}",
        f"- coverage-adjusted acceleration rule {value(rows50, 'coverage_adjusted_acceleration_rule')}",
        f"- momentum logistic regression {value(rows50, 'momentum:logistic_regression')}",
        f"- momentum random forest {value(rows50, 'momentum:random_forest')}",
        "",
        f"Logistic regression minus frozen rule at 50 slots: mean"
        f" {paired.get('mean_delta', float('nan')):+.3f}, median"
        f" {paired.get('median_delta', float('nan')):+.3f}, positive in"
        f" {paired.get('positive_origins', 0)} of {paired.get('origins', 0)} origins.",
        f"Random forest minus frozen rule: mean {paired_rf.get('mean_delta', float('nan')):+.3f},"
        f" positive in {paired_rf.get('positive_origins', 0)} of {paired_rf.get('origins', 0)} origins.",
        "",
        f"At the co-primary budget of 20 slots: frozen rule {value(rows20, 'static_rule')},",
        f"logistic regression {value(rows20, 'momentum:logistic_regression')},",
        f"random forest {value(rows20, 'momentum:random_forest')}.",
        "",
    ]
    return "\n".join(lines)


def synthetic_frames(seed: int = 11) -> dict[str, pd.DataFrame]:
    """Small frames that exercise every path, including a bounce-in-downtrend pattern."""
    rng = np.random.default_rng(seed)
    months = ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]
    frames: dict[str, pd.DataFrame] = {}
    for offset, month in enumerate(months):
        n = 900
        client = np.array([f"c{i:02d}" for i in range(15)])[rng.integers(0, 15, n)]
        recent30 = rng.lognormal(7.2, 0.8, n).round()
        trend = rng.lognormal(0.0, 0.5, n)              # 30-day vs prior-30 momentum
        momentum = rng.lognormal(0.0, 0.4, n)           # last-7 vs 30-day momentum
        recent7 = np.maximum((recent30 * momentum * 7 / 30).round(), 0)
        bounce = ((trend < 1.0) & (momentum >= 1.0)).astype(float)
        future_rate = momentum * np.where(bounce > 0, 0.55, 1.0) * rng.lognormal(0, 0.3, n)
        future30 = (recent30 * future_rate).round()
        ctr = np.clip(rng.gamma(2, 0.5, n), 0, 20)
        position = rng.integers(0, 40, n).astype(float)
        frame = pd.DataFrame({
            "client_hash_id": client,
            "content_hash_id": [f"p{offset}_{i:05d}" for i in range(n)],
            "feature_month": month,
            "feature_cutoff_date": pd.Timestamp("2025-10-31") + pd.DateOffset(days=30 * offset),
            "outcome_start": pd.Timestamp("2025-11-01") + pd.DateOffset(days=30 * offset),
            "outcome_end": pd.Timestamp("2025-11-30") + pd.DateOffset(days=30 * offset),
            "recent30_impressions": recent30,
            "recent30_clicks": (recent30 * ctr / 100).round(),
            "recent30_days": 30,
            "recent30_active_days": rng.integers(14, 31, n),
            "recent30_avg_position": position,
            "recent30_ctr_pct": ctr,
            "prev30_impressions": (recent30 / trend).round(),
            "prior30_days": 30,
            "recent7_impressions": recent7,
            "recent7_days": rng.choice([7, 7, 7, 5, 2], n),
            "future30_impressions": future30,
            "future30_days": rng.choice([30, 30, 30, 30, 12], n),
            "content_age_days": rng.integers(10, 900, n),
            "content_created_date": pd.Timestamp("2023-01-01"),
        })
        frame["log_recent30_impressions"] = np.log1p(frame["recent30_impressions"])
        measurable = frame["future30_days"].ge(14)
        frame[TARGET] = np.nan
        frame.loc[measurable, TARGET] = (
            frame.loc[measurable, "future30_impressions"]
            < 0.8 * frame.loc[measurable, "recent30_impressions"]
        ).astype(int)
        frames[month] = frame
    return frames


def selftest() -> None:
    import tempfile

    frames = synthetic_frames()
    out = Path(tempfile.mkdtemp(prefix="flyrank_onset_"))
    payload = run(frames, ["2025-12", "2026-01"], reps=100, out_dir=out,
                  do_sensitivity=True, cross_check_receipt=False)
    assert payload["claim_checks"]["p_at_50"]["values"]["logistic_mean"] is not None
    assert (out / "onset_screening.json").exists() and (out / "onset_headline.md").exists()
    assert all(entry["pool_rows"] > 50 for entry in payload["origin_results"].values())

    # Frames round-trip through the cache loader the notebooks will use.
    cache = out / "cache"
    cache.mkdir()
    for month, frame in frames.items():
        frame.to_parquet(cache / f"{month}.parquet", index=False)
    (cache / "manifest.json").write_text(
        json.dumps({"provenance": {"config": {"feature_months": list(frames)}}}),
        encoding="utf-8",
    )
    loaded, provenance = load_cached_frames(cache)
    assert set(loaded) == set(frames) and provenance["source"] == "cache"

    queues = build_queues(loaded, "2026-02", k=25, cap=3)
    for lane in ("triage", "onset"):
        summary = queues[lane]["summary"]
        assert summary["selected"] == 25
        assert summary["largest_client_share"] <= 3 / 25 + 1e-9
        assert set(summary["actions"]) <= {
            "review_before_revert", "monitor_only", "verify_then_review",
            "watch_for_resume", "investigate_collapse",
        }
    print("queues:", queues["triage"]["summary"]["actions"], queues["onset"]["summary"]["actions"])
    print("selftest: PASS")
    print(json.dumps(payload["claim_checks"], indent=2, default=str)[:1200])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--cache", default=None, help="frame cache directory")
    parser.add_argument("--warehouse", action="store_true", help="rebuild frames from the release")
    parser.add_argument("--floor", type=int, default=1000, help="impression floor when rebuilding")
    parser.add_argument("--out", default=None)
    parser.add_argument("--origins", default=None, help="comma-separated origins")
    parser.add_argument("--k", default="20,50")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-days", type=int, default=DEFAULT_MIN_DAYS)
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    if args.warehouse:
        cache = Path(args.cache or ROOT / f"work/local_final_pipeline_floor{args.floor}")
        frames, provenance = build_frames_from_warehouse(args.floor, cache)
    else:
        cache = Path(args.cache or os.environ.get(
            "FLYRANK_FINAL_CACHE", ROOT / "work/local_final_pipeline"))
        frames, provenance = load_cached_frames(cache)

    origins = args.origins.split(",") if args.origins else list(DEFAULT_ORIGINS)
    out = Path(args.out or ROOT / "work/outputs")
    k_values = tuple(int(value) for value in args.k.split(","))

    payload = run(
        frames, origins,
        k_values=k_values,
        threshold=args.threshold,
        min_days=args.min_days,
        reps=args.reps,
        out_dir=out,
        provenance=provenance,
        do_sensitivity=not args.skip_sensitivity,
    )
    print("\n" + headline(payload))
    print(f"Receipts: {out / 'onset_screening.json'}")
    print(f"Headline: {out / 'onset_headline.md'}")


if __name__ == "__main__":
    main()
