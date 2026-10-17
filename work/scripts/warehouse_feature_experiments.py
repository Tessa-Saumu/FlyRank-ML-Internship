from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
from calendar import monthrange
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
RELEASE = "hf://datasets/FlyRank/internship-warehouse"
FACT_ROOT = f"{RELEASE}/fact_content_daily_performance"
DIM_CONTENT = f"{RELEASE}/dim_content.parquet"
SEED = 42
N_SPLITS = 5
TARGET = "is_declining_next30"
K_VALUES = (20, 50, 100)
CONTROL_FEATURES = [
    "log_recent30_impressions",
    "recent30_ctr_pct",
    "recent30_avg_position",
    "recent30_active_days",
    "content_age_days",
]
FEATURE_SETS: dict[str, list[str]] = {
    "control": [],
    "traffic_momentum": [
        "traffic_velocity_30d",
        "traffic_acceleration_7d",
        "has_prior30_gsc_coverage",
    ],
    "ctr_peer_gap": ["ctr_opportunity_gap"],
    "ga4_engagement": [
        "ga4_engagement_rate",
        "ga4_scroll_rate",
        "ga4_sessions_per_1000_impressions",
        "ga4_available_days",
    ],
    "nonlinear_age": ["age_band"],
    "combined": [
        "traffic_velocity_30d",
        "traffic_acceleration_7d",
        "has_prior30_gsc_coverage",
        "ctr_opportunity_gap",
        "ga4_engagement_rate",
        "ga4_scroll_rate",
        "ga4_sessions_per_1000_impressions",
        "ga4_available_days",
        "age_band",
    ],
}
MODELS = ("logistic_regression", "random_forest")
DEVELOPMENT_ORIGINS = (
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
)
FRAME_MONTHS = (
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
)
HISTORY_POLICIES = ("latest_only", "latest_three", "expanding")
REPORTING_LAG_DAYS = 7


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = SEED
    primary_reporting_lag_days: int = REPORTING_LAG_DAYS
    reporting_lag_sensitivity_days: tuple[int, ...] = (0,)
    top_k: tuple[int, ...] = K_VALUES
    minimum_recent_coverage_days: int = 14
    minimum_impressions: int = 100
    minimum_label_coverage_days: int = 14
    decline_threshold: float = 0.80
    practical_hurdle_cases_per_50: int = 5
    practical_hurdle_origins: int = 4
    feature_months: tuple[str, ...] = FRAME_MONTHS
    prediction_origins: tuple[str, ...] = DEVELOPMENT_ORIGINS
    output_dir: str = "work/outputs/warehouse_feature_experiments"


DEFAULT_CONFIG = ExperimentConfig()


def resolve_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        env_path = ROOT / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key, value = stripped.split("=", 1)
                    if key.strip() == "HF_TOKEN":
                        token = value.strip().strip("'\"")
                        break
    if not token:
        from huggingface_hub import get_token

        token = get_token()
    if not token:
        raise RuntimeError(
            "No Hugging Face READ token was found. Set HF_TOKEN in the environment "
            "or add it to the repository-local .env; never paste it into this notebook."
        )
    return token


def quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def months_between(start: date, end: date) -> list[str]:
    months: list[str] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def parquet_source(
    months: list[str],
    partition_files: dict[str, str] | None = None,
) -> str:
    # The release starts in January 2025; earlier days in the first available
    # prior window simply have no panel rows and are tracked by prev30_days.
    months = sorted(month for month in months if month >= "2025-01")
    if not months:
        raise ValueError("Requested dates fall before the warehouse release starts.")
    if partition_files is None:
        source_files = [
            f"{FACT_ROOT}/month={month}/*.parquet" for month in months
        ]
    else:
        missing = set(months) - partition_files.keys()
        if missing:
            raise ValueError(
                f"HF cache is missing required partitions: {sorted(missing)}"
            )
        source_files = [partition_files[month] for month in months]
    paths = ", ".join(quote_sql_string(path) for path in source_files)
    return f"read_parquet([{paths}])"


def cache_hf_warehouse_files(
    months: list[str], token: str
) -> tuple[str, dict[str, str], str]:
    """Fetch the required immutable HF release shards into the Hub-managed cache."""
    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi(token=token).dataset_info(
        "FlyRank/internship-warehouse"
    ).sha
    partition_files = {
        month: hf_hub_download(
            repo_id="FlyRank/internship-warehouse",
            filename=(
                f"fact_content_daily_performance/month={month}/data_0.parquet"
            ),
            repo_type="dataset",
            revision=revision,
            token=token,
        )
        for month in months
    }
    dimension_file = hf_hub_download(
        repo_id="FlyRank/internship-warehouse",
        filename="dim_content.parquet",
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    return revision, partition_files, dimension_file


def precision_at_k(
    y_true: pd.Series,
    scores: np.ndarray,
    k: int,
    tie_breakers: pd.Series,
) -> float:
    count = min(k, len(y_true))
    # Use a stable, label-independent page hash only to break equal-score ties.
    stable_keys = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(
                    str(value).encode("utf-8"),
                    digest_size=8,
                    person=b"FlyRankP50",
                ).digest(),
                byteorder="big",
            )
            for value in tie_breakers
        ),
        dtype=np.uint64,
        count=len(tie_breakers),
    )
    order = np.lexsort((stable_keys, -scores))[:count]
    return float(y_true.iloc[order].mean()) if count else float("nan")


def baseline_scores(frame: pd.DataFrame) -> np.ndarray:
    visible = (frame["recent30_impressions"] >= 500).astype(int)
    top10 = (
        frame["recent30_avg_position"].gt(0)
        & frame["recent30_avg_position"].le(10)
    ).astype(int)
    low_ctr = (frame["recent30_ctr_pct"] < 1.0).astype(int)
    stale = (frame["content_age_days"] >= 91).astype(int)
    return (
        0.40 * visible
        + 0.35 * visible * top10 * low_ctr
        + 0.25 * visible * stale
    ).to_numpy(dtype=float)


def metric_record(
    y_true: pd.Series,
    scores: np.ndarray,
    tie_breakers: pd.Series,
    *,
    experiment: str,
    model: str,
    fold: int | str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "experiment": experiment,
        "model": model,
        "fold": fold,
        "rows": len(y_true),
        "base_rate": float(y_true.mean()),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }
    for k in K_VALUES:
        record[f"precision_at_{k}"] = precision_at_k(
            y_true, scores, k, tie_breakers
        )
    return record


def build_frame(con: duckdb.DuckDBPyConnection, feature_month: str) -> pd.DataFrame:
    if not re.fullmatch(r"\d{4}-\d{2}", feature_month):
        raise ValueError(f"Invalid feature month: {feature_month}")
    return load_origin_frames(con, (feature_month,))[REPORTING_LAG_DAYS][feature_month]


def grouped_test_client_sets(frame: pd.DataFrame) -> list[set[str]]:
    splitter = GroupShuffleSplit(
        n_splits=N_SPLITS, test_size=0.25, random_state=SEED
    )
    groups = frame["client_hash_id"].fillna("unknown").astype(str)
    return [
        set(groups.iloc[test_indices])
        for _, test_indices in splitter.split(frame, groups=groups)
    ]


def indices_for_test_clients(
    frame: pd.DataFrame, test_clients: set[str]
) -> tuple[np.ndarray, np.ndarray]:
    clients = frame["client_hash_id"].fillna("unknown").astype(str)
    test_mask = clients.isin(test_clients).to_numpy()
    train_indices = np.flatnonzero(~test_mask)
    test_indices = np.flatnonzero(test_mask)
    y_train = frame.iloc[train_indices][TARGET]
    y_test = frame.iloc[test_indices][TARGET]
    if not len(train_indices) or not len(test_indices):
        raise ValueError("Grouped split produced an empty train or test fold")
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "A client-grouped split has only one target class; do not silently "
            "fall back to a row-level split."
        )
    return train_indices, test_indices


def position_bands(values: pd.Series) -> pd.Series:
    position = pd.to_numeric(values, errors="coerce")
    bands = pd.cut(
        position,
        bins=[-np.inf, 3, 10, 20, 50, np.inf],
        labels=["top3", "4to10", "11to20", "21to50", "over50"],
    ).astype("object")
    return bands.where(position.gt(0), "no_position").fillna("no_position")


def feature_frames(
    train: pd.DataFrame, test: pd.DataFrame, experiment: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train_x = train[CONTROL_FEATURES].copy()
    test_x = test[CONTROL_FEATURES].copy()
    numeric = CONTROL_FEATURES.copy()
    categorical: list[str] = []

    additions = FEATURE_SETS[experiment]
    if experiment in {"traffic_momentum", "combined"}:
        for original, destination in ((train, train_x), (test, test_x)):
            recent = pd.to_numeric(
                original["recent30_impressions"], errors="coerce"
            ).clip(lower=0)
            previous = pd.to_numeric(
                original["prev30_impressions"], errors="coerce"
            ).clip(lower=0)
            prior_covered = original["prev30_days"].ge(14)
            velocity = recent / (previous + 1.0)
            velocity = velocity.where(prior_covered)
            acceleration = original["recent7_impressions"] / (
                (recent / 30.0 * 7.0) + 1.0
            )
            destination["traffic_velocity_30d"] = velocity
            destination["traffic_acceleration_7d"] = acceleration
            destination["has_prior30_gsc_coverage"] = prior_covered.astype(float)
        numeric.extend(
            [
                "traffic_velocity_30d",
                "traffic_acceleration_7d",
                "has_prior30_gsc_coverage",
            ]
        )

    if experiment in {"ctr_peer_gap", "combined"}:
        train_bands = position_bands(train["recent30_avg_position"])
        test_bands = position_bands(test["recent30_avg_position"])
        expected_by_band = train["recent30_ctr_pct"].groupby(
            train_bands, observed=True
        ).mean()
        global_expected = float(train["recent30_ctr_pct"].mean())
        train_x["ctr_opportunity_gap"] = (
            train_bands.map(expected_by_band).fillna(global_expected)
            - train["recent30_ctr_pct"]
        )
        test_x["ctr_opportunity_gap"] = (
            test_bands.map(expected_by_band).fillna(global_expected)
            - test["recent30_ctr_pct"]
        )
        numeric.append("ctr_opportunity_gap")

    if experiment in {"ga4_engagement", "combined"}:
        for original, destination in ((train, train_x), (test, test_x)):
            sessions = pd.to_numeric(original["ga4_sessions"], errors="coerce")
            pageviews = pd.to_numeric(original["ga4_pageviews"], errors="coerce")
            engaged = pd.to_numeric(
                original["ga4_engaged_sessions"], errors="coerce"
            )
            scroll = pd.to_numeric(original["scroll_events"], errors="coerce")
            available = pd.to_numeric(
                original["ga4_available_days"], errors="coerce"
            ).fillna(0)
            destination["ga4_engagement_rate"] = (
                100 * engaged / sessions.replace(0, np.nan)
            )
            destination["ga4_scroll_rate"] = (
                100 * scroll / pageviews.replace(0, np.nan)
            )
            destination["ga4_sessions_per_1000_impressions"] = (
                1000
                * sessions
                / original["recent30_impressions"].replace(0, np.nan)
            )
            destination["ga4_available_days"] = available
        numeric.extend(
            [
                "ga4_engagement_rate",
                "ga4_scroll_rate",
                "ga4_sessions_per_1000_impressions",
                "ga4_available_days",
            ]
        )

    if experiment in {"nonlinear_age", "combined"}:
        age_bins = [0, 90, 180, 365, np.inf]
        age_labels = ["up_to_90d", "91_to_180d", "181_to_365d", "over_365d"]
        for original, destination in ((train, train_x), (test, test_x)):
            destination["age_band"] = pd.cut(
                pd.to_numeric(original["content_age_days"], errors="coerce"),
                bins=age_bins,
                labels=age_labels,
                include_lowest=True,
            ).astype("object")
        train_x = train_x.drop(columns=["content_age_days"])
        test_x = test_x.drop(columns=["content_age_days"])
        numeric.remove("content_age_days")
        categorical.append("age_band")

    if additions:
        expected = set(additions)
        actual = set(numeric + categorical)
        if not expected.issubset(actual):
            raise AssertionError(
                f"Missing candidate features for {experiment}: {expected - actual}"
            )
    forbidden = {
        "client_hash_id",
        "content_hash_id",
        "future30_impressions",
        "future30_days",
        TARGET,
        "feature_month",
        "cutoff_date",
    }
    if forbidden.intersection(numeric + categorical):
        raise AssertionError("An identifier, outcome field, or date entered predictors.")
    return train_x, test_x, numeric, categorical


def build_model(
    model_name: str, numeric: list[str], categorical: list[str]
) -> Pipeline:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if model_name == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value="unknown"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    preprocess = ColumnTransformer(transformers, remainder="drop")

    if model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000, random_state=SEED)
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            class_weight="balanced_subsample",
            max_depth=10,
            min_samples_leaf=25,
            n_estimators=200,
            n_jobs=4,
            random_state=SEED,
        )
    elif model_name == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=SEED,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def evaluate_grouped(
    frame: pd.DataFrame,
    test_client_sets: list[set[str]],
    experiments: list[str],
    models: tuple[str, ...] = MODELS,
) -> list[dict[str, Any]]:
    results = []
    for experiment in experiments:
        for fold_number, test_clients in enumerate(test_client_sets, start=1):
            train_indices, test_indices = indices_for_test_clients(frame, test_clients)
            train = frame.iloc[train_indices].copy()
            test = frame.iloc[test_indices].copy()
            train_x, test_x, numeric, categorical = feature_frames(
                train, test, experiment
            )
            y_train = train[TARGET]
            y_test = test[TARGET]
            for model_name in models:
                model = build_model(model_name, numeric, categorical)
                model.fit(train_x, y_train)
                scores = model.predict_proba(test_x)[:, 1]
                results.append(
                    metric_record(
                        y_test,
                        scores,
                        test["content_hash_id"],
                        experiment=experiment,
                        model=model_name,
                        fold=fold_number,
                    )
                )
            if experiment == "control":
                results.append(
                    metric_record(
                        y_test,
                        baseline_scores(test),
                        test["content_hash_id"],
                        experiment="current_rule_baseline",
                        model="heuristic",
                        fold=fold_number,
                    )
                )
    return results


def summarize_metrics(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "base_rate",
        "precision_at_20",
        "precision_at_50",
        "precision_at_100",
        "average_precision",
        "roc_auc",
    ]
    return (
        results.groupby(["experiment", "model"], sort=False)[columns]
        .agg(["mean", "std"])
        .round(3)
    )


def paired_decisions(results: pd.DataFrame) -> pd.DataFrame:
    model_rows = results[results["model"] != "heuristic"]
    controls = model_rows[model_rows["experiment"] == "control"].set_index(
        ["model", "fold"]
    )
    rows = []
    for (experiment, model), candidate in model_rows[
        model_rows["experiment"] != "control"
    ].groupby(["experiment", "model"], sort=False):
        candidate = candidate.set_index("fold")
        control = controls.xs(model, level="model").reindex(candidate.index)
        delta = candidate["precision_at_50"] - control["precision_at_50"]
        wins = int(delta.gt(0).sum())
        mean_delta = float(delta.mean())
        if mean_delta >= 0.02 and wins >= 4:
            verdict = "directional keep"
        elif mean_delta <= -0.02 and wins <= 1:
            verdict = "directional reject"
        else:
            verdict = "inconclusive"
        rows.append(
            {
                "experiment": experiment,
                "model": model,
                "mean_delta_p50": mean_delta,
                "folds_better_at_p50": f"{wins}/{len(delta)}",
                "mean_delta_ap": float(
                    (
                        candidate["average_precision"]
                        - control["average_precision"]
                    ).mean()
                ),
                "mean_delta_auc": float(
                    (candidate["roc_auc"] - control["roc_auc"]).mean()
                ),
                "decision": verdict,
            }
        )
    return pd.DataFrame(rows).round(
        {
            "mean_delta_p50": 3,
            "mean_delta_ap": 3,
            "mean_delta_auc": 3,
        }
    )


def evaluate_temporal(
    train: pd.DataFrame, test: pd.DataFrame, experiment: str, model_name: str
) -> dict[str, Any]:
    train_x, test_x, numeric, categorical = feature_frames(
        train, test, experiment
    )
    model = build_model(model_name, numeric, categorical)
    model.fit(train_x, train[TARGET])
    scores = model.predict_proba(test_x)[:, 1]
    return metric_record(
        test[TARGET],
        scores,
        test["content_hash_id"],
        experiment=experiment,
        model=model_name,
        fold="temporal",
    )


def print_floor_sensitivity(
    frame: pd.DataFrame, test_client_sets: list[set[str]]
) -> None:
    rows = []
    for floor in (100, 500, 1000):
        subset = frame[frame["recent30_impressions"] >= floor].copy()
        for fold_number, test_clients in enumerate(test_client_sets, start=1):
            train_indices, test_indices = indices_for_test_clients(
                subset, test_clients
            )
            train = subset.iloc[train_indices]
            test = subset.iloc[test_indices]
            train_x, test_x, numeric, categorical = feature_frames(
                train, test, "control"
            )
            for model_name in MODELS:
                model = build_model(model_name, numeric, categorical)
                model.fit(train_x, train[TARGET])
                scores = model.predict_proba(test_x)[:, 1]
                rows.append(
                    {
                        "floor": floor,
                        "fold": fold_number,
                        "model": model_name,
                        "rows": len(subset),
                        "base_rate": float(test[TARGET].mean()),
                        "precision_at_50": precision_at_k(
                            test[TARGET], scores, 50, test["content_hash_id"]
                        ),
                    }
                )
            rows.append(
                {
                    "floor": floor,
                    "fold": fold_number,
                    "model": "heuristic",
                    "rows": len(subset),
                    "base_rate": float(test[TARGET].mean()),
                    "precision_at_50": precision_at_k(
                        test[TARGET],
                        baseline_scores(test),
                        50,
                        test["content_hash_id"],
                    ),
                }
            )
    summary = (
        pd.DataFrame(rows)
        .groupby(["floor", "model"], sort=True)
        .agg(
            rows=("rows", "first"),
            base_rate_mean=("base_rate", "mean"),
            p50_mean=("precision_at_50", "mean"),
            p50_std=("precision_at_50", "std"),
        )
        .round(3)
    )
    print("\nPopulation sensitivity (same held-out client groups):")
    print(summary.to_string())


def print_coverage_sensitivity(
    frame: pd.DataFrame, test_client_sets: list[set[str]]
) -> None:
    results = []
    for coverage, subset in (
        ("at_least_14d", frame),
        ("full_30d", frame[frame["future30_days"] == 30].copy()),
    ):
        for fold_number, test_clients in enumerate(test_client_sets, start=1):
            train_indices, test_indices = indices_for_test_clients(
                subset, test_clients
            )
            train, test = subset.iloc[train_indices], subset.iloc[test_indices]
            train_x, test_x, numeric, categorical = feature_frames(
                train, test, "control"
            )
            for model_name in MODELS:
                model = build_model(model_name, numeric, categorical)
                model.fit(train_x, train[TARGET])
                scores = model.predict_proba(test_x)[:, 1]
                results.append(
                    {
                        "coverage": coverage,
                        "fold": fold_number,
                        "model": model_name,
                        "rows": len(subset),
                        "base_rate": float(test[TARGET].mean()),
                        "p50": precision_at_k(
                            test[TARGET], scores, 50, test["content_hash_id"]
                        ),
                    }
                )
    print("\nOutcome-coverage sensitivity (post-outcome diagnostic only):")
    print(
        pd.DataFrame(results)
        .groupby(["coverage", "model"], sort=False)
        .agg(
            rows=("rows", "first"),
            base_rate_mean=("base_rate", "mean"),
            p50_mean=("p50", "mean"),
            p50_std=("p50", "std"),
        )
        .round(3)
        .to_string()
    )


def _month_end(month: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError(f"Invalid month: {month}")
    year, month_number = map(int, month.split("-"))
    return date(year, month_number, monthrange(year, month_number)[1])


def _origin_column(month: str, name: str) -> str:
    return f"o_{month.replace('-', '_')}_{name}"


def _feature_window(cutoff: date) -> dict[str, date]:
    return {
        "prior_start": cutoff - timedelta(days=59),
        "prior_end": cutoff - timedelta(days=30),
        "recent_start": cutoff - timedelta(days=29),
        "recent_end": cutoff,
        "recent7_start": cutoff - timedelta(days=6),
        "recent7_end": cutoff,
        "outcome_start": cutoff + timedelta(days=1),
        "outcome_end": cutoff + timedelta(days=30),
    }


def _sum_expr(
    month: str,
    name: str,
    start: date,
    end: date,
    value: str,
    condition: str = "TRUE",
) -> str:
    alias = _origin_column(month, name)
    return (
        f"SUM(CASE WHEN report_date BETWEEN DATE '{start}' AND DATE '{end}' "
        f"AND {condition} THEN {value} ELSE 0 END) AS {alias}"
    )


def _count_expr(
    month: str,
    name: str,
    start: date,
    end: date,
    condition: str,
) -> str:
    alias = _origin_column(month, name)
    return (
        f"SUM(CASE WHEN report_date BETWEEN DATE '{start}' AND DATE '{end}' "
        f"AND {condition} THEN 1 ELSE 0 END) AS {alias}"
    )


def load_origin_frames(
    con: duckdb.DuckDBPyConnection,
    feature_months: tuple[str, ...],
    config: ExperimentConfig = DEFAULT_CONFIG,
    reporting_lags: tuple[int, ...] = (REPORTING_LAG_DAYS, 0),
    partition_files: dict[str, str] | None = None,
    dimension_file: str | None = None,
) -> dict[int, dict[str, pd.DataFrame]]:
    """Read candidate-filtered temporal frames from the Hugging Face release."""
    if not feature_months:
        raise ValueError("At least one feature month is required.")
    if not reporting_lags or any(lag < 0 for lag in reporting_lags):
        raise ValueError("Reporting-lag scenarios must be nonnegative.")
    if len(set(reporting_lags)) != len(reporting_lags):
        raise ValueError("Reporting-lag scenarios must be unique.")
    cutoffs = {month: _month_end(month) for month in feature_months}
    windows: dict[str, dict[str, date]] = {}
    scenarios: dict[str, tuple[str, int, date, date]] = {}
    for lag_days in reporting_lags:
        for month, prediction_date in cutoffs.items():
            scenario = f"{month}__lag{lag_days}"
            feature_cutoff = prediction_date - timedelta(days=lag_days)
            window = _feature_window(feature_cutoff)
            window["outcome_start"] = prediction_date + timedelta(days=1)
            window["outcome_end"] = prediction_date + timedelta(days=30)
            windows[scenario] = window
            scenarios[scenario] = (
                month, lag_days, prediction_date, feature_cutoff
            )
    first_date = min(window["prior_start"] for window in windows.values())
    last_date = max(window["outcome_end"] for window in windows.values())
    months = months_between(first_date, last_date)
    if "2026-06" in months or any(month > "2026-05" for month in months):
        raise AssertionError("Development scans must not read June 2026.")

    partition_bounds = {}
    for month in months:
        partition = parquet_source([month], partition_files)
        min_date, max_date = con.execute(
            f"SELECT MIN(report_date), MAX(report_date) FROM {partition}"
        ).fetchone()
        if min_date is None or max_date is None:
            raise ValueError(f"Required warehouse partition is unavailable: {month}")
        partition_bounds[month] = {
            "min_report_date": str(min_date),
            "max_report_date": str(max_date),
        }

    frames: dict[int, dict[str, pd.DataFrame]] = {
        lag: {} for lag in reporting_lags
    }
    for scenario in sorted(windows):
        window = windows[scenario]
        month, lag_days, prediction_date, feature_cutoff = scenarios[scenario]
        scenario_first_date = window["prior_start"]
        scenario_last_date = window["outcome_end"]
        scenario_months = months_between(scenario_first_date, scenario_last_date)
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE eligible_candidates AS
            SELECT client_hash_id, content_hash_id
            FROM {parquet_source(scenario_months, partition_files)}
            WHERE report_date BETWEEN DATE '{window["recent_start"]}'
                AND DATE '{window["recent_end"]}'
            GROUP BY client_hash_id, content_hash_id
            HAVING COUNT(DISTINCT CASE WHEN gsc_data_available IS TRUE
                THEN report_date END) >= {config.minimum_recent_coverage_days}
                AND SUM(CASE WHEN gsc_data_available IS TRUE
                    THEN COALESCE(gsc_impressions, 0) ELSE 0 END)
                    >= {config.minimum_impressions}
            """
        )
        select_exprs = [
            "client_hash_id",
            "content_hash_id",
            "CASE WHEN COUNT(*) > COUNT(DISTINCT report_date) "
            "THEN 1 ELSE 0 END AS daily_grain_violation",
        ]
        select_exprs.extend(
            [
                _sum_expr(
                    scenario, "recent30_impressions", window["recent_start"],
                    window["recent_end"], "gsc_impressions",
                    "gsc_available = 1",
                ),
                _sum_expr(
                    scenario, "recent30_clicks", window["recent_start"],
                    window["recent_end"], "gsc_clicks",
                    "gsc_available = 1",
                ),
                _sum_expr(
                    scenario, "prev30_impressions", window["prior_start"],
                    window["prior_end"], "gsc_impressions",
                    "gsc_available = 1",
                ),
                _sum_expr(
                    scenario, "recent7_impressions", window["recent7_start"],
                    window["recent7_end"], "gsc_impressions",
                    "gsc_available = 1",
                ),
                _sum_expr(
                    scenario, "future30_impressions", window["outcome_start"],
                    window["outcome_end"], "gsc_impressions",
                    "gsc_available = 1",
                ),
                _count_expr(
                    scenario, "recent30_days", window["recent_start"],
                    window["recent_end"], "gsc_available = 1",
                ),
                _count_expr(
                    scenario, "prior30_days", window["prior_start"],
                    window["prior_end"], "gsc_available = 1",
                ),
                _count_expr(
                    scenario, "recent7_days", window["recent7_start"],
                    window["recent7_end"], "gsc_available = 1",
                ),
                _count_expr(
                    scenario, "future30_days", window["outcome_start"],
                    window["outcome_end"], "gsc_available = 1",
                ),
                _count_expr(
                    scenario, "recent30_active_days", window["recent_start"],
                    window["recent_end"],
                    "gsc_available = 1 AND gsc_impressions > 0",
                ),
                (
                    "AVG(CASE WHEN report_date BETWEEN "
                    f"DATE '{window['recent_start']}' AND DATE '{window['recent_end']}' "
                    "AND gsc_available = 1 THEN NULLIF(gsc_avg_position, 0) END) "
                    f"AS {_origin_column(scenario, 'recent30_avg_position')}"
                ),
                _sum_expr(
                    scenario, "ga4_sessions", window["recent_start"],
                    window["recent_end"], "ga4_sessions",
                    "ga4_available = 1",
                ),
                _sum_expr(
                    scenario, "ga4_pageviews", window["recent_start"],
                    window["recent_end"], "ga4_pageviews",
                    "ga4_available = 1",
                ),
                _sum_expr(
                    scenario, "ga4_engaged_sessions", window["recent_start"],
                    window["recent_end"], "ga4_engaged_sessions",
                    "ga4_available = 1",
                ),
                _sum_expr(
                    scenario, "scroll_events", window["recent_start"],
                    window["recent_end"], "scroll_events",
                    "ga4_available = 1",
                ),
                _count_expr(
                    scenario, "ga4_available_days", window["recent_start"],
                    window["recent_end"], "ga4_available = 1",
                ),
            ]
        )
        sql = f"""
            WITH fact_rows AS (
                SELECT
                    report_date,
                    client_hash_id,
                    content_hash_id,
                    CASE WHEN gsc_data_available IS TRUE
                        THEN COALESCE(gsc_impressions, 0) ELSE 0 END AS gsc_impressions,
                    CASE WHEN gsc_data_available IS TRUE
                        THEN COALESCE(gsc_clicks, 0) ELSE 0 END AS gsc_clicks,
                    CASE WHEN gsc_data_available IS TRUE
                        THEN NULLIF(gsc_avg_position, 0) END AS gsc_avg_position,
                    CASE WHEN gsc_data_available IS TRUE THEN 1 ELSE 0 END
                        AS gsc_available,
                    CASE WHEN ga4_data_available IS TRUE
                        THEN COALESCE(ga4_sessions, 0) ELSE 0 END AS ga4_sessions,
                    CASE WHEN ga4_data_available IS TRUE
                        THEN COALESCE(ga4_pageviews, 0) ELSE 0 END AS ga4_pageviews,
                    CASE WHEN ga4_data_available IS TRUE
                        THEN COALESCE(ga4_engaged_sessions, 0) ELSE 0 END
                        AS ga4_engaged_sessions,
                    CASE WHEN ga4_data_available IS TRUE
                        THEN COALESCE(scroll_events, 0) ELSE 0 END AS scroll_events,
                    CASE WHEN ga4_data_available IS TRUE THEN 1 ELSE 0 END
                        AS ga4_available
                FROM {parquet_source(scenario_months, partition_files)} fact
                INNER JOIN eligible_candidates candidates
                    USING (client_hash_id, content_hash_id)
                WHERE report_date BETWEEN DATE '{scenario_first_date}'
                    AND DATE '{scenario_last_date}'
            ),
            wide AS (
                SELECT {", ".join(select_exprs)}
                FROM fact_rows
                GROUP BY client_hash_id, content_hash_id
            )
            SELECT wide.*, c.content_created_date
            FROM wide
            LEFT JOIN (
                SELECT client_hash_id, content_hash_id, content_created_date
                FROM read_parquet(
                    {quote_sql_string(dimension_file or DIM_CONTENT)}
                )
            ) c USING (client_hash_id, content_hash_id)
        """
        aggregated = con.execute(sql).df()
        if aggregated.empty:
            raise ValueError(f"No warehouse rows were returned for {month}.")
        if aggregated["daily_grain_violation"].fillna(0).gt(0).any():
            raise ValueError(
                f"Daily client-page grain check failed for {month}."
            )
        if aggregated.duplicated(["client_hash_id", "content_hash_id"]).any():
            raise ValueError(
                f"Dimension join created duplicate client-page rows for {month}."
            )
        con.execute("DROP TABLE eligible_candidates")
        prefix = f"o_{scenario.replace('-', '_')}_"
        scenario_columns = [
            column for column in aggregated.columns if column.startswith(prefix)
        ]
        common_columns = [
            "client_hash_id", "content_hash_id", "daily_grain_violation",
            "content_created_date",
        ]
        rename = {
            column: column.removeprefix(prefix)
            for column in scenario_columns
        }
        frame = aggregated[common_columns + scenario_columns].rename(
            columns=rename
        ).copy()
        frame = frame.drop(columns=["daily_grain_violation"])
        frame["cutoff_date"] = prediction_date
        frame["feature_cutoff_date"] = feature_cutoff
        frame["reporting_lag_days"] = lag_days
        frame["feature_month"] = month
        frame["outcome_start"] = prediction_date + timedelta(days=1)
        frame["outcome_end"] = prediction_date + timedelta(days=30)
        frame["label_available_date"] = (
            prediction_date + timedelta(days=30 + lag_days)
        )
        frame["log_recent30_impressions"] = np.log1p(
            frame["recent30_impressions"].clip(lower=0)
        )
        frame["recent30_ctr_pct"] = (
            100.0 * frame["recent30_clicks"]
            / frame["recent30_impressions"].replace(0, np.nan)
        )
        frame["content_age_days"] = (
            feature_cutoff - pd.to_datetime(frame["content_created_date"]).dt.date
        ).map(lambda value: value.days if pd.notna(value) else np.nan)
        feature_window = windows[scenario]
        assert feature_window["recent_end"] <= prediction_date
        assert feature_cutoff + timedelta(days=lag_days) <= prediction_date
        assert feature_window["prior_end"] < feature_window["recent_start"]
        assert feature_window["recent7_end"] <= feature_cutoff
        assert frame["outcome_start"].iloc[0] > prediction_date
        eligible = (
            frame["recent30_days"].ge(config.minimum_recent_coverage_days)
            & frame["recent30_impressions"].ge(config.minimum_impressions)
        )
        frame = frame.loc[eligible].copy()
        measurable = frame["future30_days"].ge(
            config.minimum_label_coverage_days
        )
        frame[TARGET] = np.nan
        frame.loc[measurable, TARGET] = (
            frame.loc[measurable, "future30_impressions"]
            < config.decline_threshold
            * frame.loc[measurable, "recent30_impressions"]
        ).astype(int)
        frame["daily_rate_decline"] = np.nan
        rate_measurable = measurable & frame["recent30_days"].gt(0)
        frame.loc[rate_measurable, "daily_rate_decline"] = (
            frame.loc[rate_measurable, "future30_impressions"]
            / frame.loc[rate_measurable, "future30_days"]
            < config.decline_threshold
            * frame.loc[rate_measurable, "recent30_impressions"]
            / frame.loc[rate_measurable, "recent30_days"]
        ).astype(int)
        if frame.empty:
            raise ValueError(f"No prediction-time candidates for {month}.")
        if frame["content_age_days"].dropna().lt(0).any():
            raise ValueError(f"Negative content age found for {month}.")
        if frame.duplicated(["client_hash_id", "content_hash_id"]).any():
            raise ValueError(f"Duplicate page-client-cutoff rows found for {month}.")
        frames[lag_days][month] = frame.sort_values(
            ["client_hash_id", "content_hash_id"]
        ).reset_index(drop=True)
        frames[lag_days][month].attrs["source_partition_bounds"] = partition_bounds
        del aggregated
    return frames


def _position_bands(values: pd.Series) -> pd.Series:
    position = pd.to_numeric(values, errors="coerce")
    bands = pd.cut(
        position,
        [-np.inf, 3, 10, 20, 50, np.inf],
        labels=["top3", "4to10", "11to20", "21to50", "over50"],
    ).astype("object")
    return bands.where(position.gt(0), "no_position").fillna("no_position")


def _momentum(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    recent = pd.to_numeric(frame["recent30_impressions"], errors="coerce").clip(lower=0)
    prior = pd.to_numeric(frame["prev30_impressions"], errors="coerce").clip(lower=0)
    recent7 = pd.to_numeric(frame["recent7_impressions"], errors="coerce").clip(lower=0)
    prior_covered = frame["prior30_days"].ge(14)
    result["traffic_velocity_30d"] = (recent / (prior + 1.0)).where(prior_covered)
    result["traffic_acceleration_7d"] = recent7 / (recent / 30.0 * 7.0 + 1.0)
    result["has_prior30_gsc_coverage"] = prior_covered.astype(float)
    return result


def _design_matrices(
    train: pd.DataFrame, test: pd.DataFrame, feature_set: str
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train_all, test_all = _momentum(train), _momentum(test)
    numeric = CONTROL_FEATURES.copy()
    categorical: list[str] = []
    train_x, test_x = train_all[CONTROL_FEATURES].copy(), test_all[CONTROL_FEATURES].copy()
    if feature_set in {
        "momentum", "combined", "velocity_only", "acceleration_only",
        "coverage_only", "velocity_acceleration", "momentum_plus_ctr",
        "momentum_plus_ga4", "momentum_plus_age",
    }:
        selected = {
            "momentum": ["traffic_velocity_30d", "traffic_acceleration_7d", "has_prior30_gsc_coverage"],
            "combined": ["traffic_velocity_30d", "traffic_acceleration_7d", "has_prior30_gsc_coverage"],
            "velocity_only": ["traffic_velocity_30d"],
            "acceleration_only": ["traffic_acceleration_7d"],
            "coverage_only": ["has_prior30_gsc_coverage"],
            "velocity_acceleration": ["traffic_velocity_30d", "traffic_acceleration_7d"],
            "momentum_plus_ctr": ["traffic_velocity_30d", "traffic_acceleration_7d", "has_prior30_gsc_coverage"],
            "momentum_plus_ga4": ["traffic_velocity_30d", "traffic_acceleration_7d", "has_prior30_gsc_coverage"],
            "momentum_plus_age": ["traffic_velocity_30d", "traffic_acceleration_7d", "has_prior30_gsc_coverage"],
        }[feature_set]
        train_x = pd.concat([train_x, train_all[selected]], axis=1)
        test_x = pd.concat([test_x, test_all[selected]], axis=1)
        numeric.extend(selected)

    if feature_set in {"combined", "momentum_plus_ctr"}:
        train_bands = _position_bands(train["recent30_avg_position"])
        test_bands = _position_bands(test["recent30_avg_position"])
        expected = train["recent30_ctr_pct"].groupby(train_bands, observed=True).mean()
        global_mean = float(train["recent30_ctr_pct"].mean())
        train_x["ctr_opportunity_gap"] = (
            train_bands.map(expected).fillna(global_mean) - train["recent30_ctr_pct"]
        )
        test_x["ctr_opportunity_gap"] = (
            test_bands.map(expected).fillna(global_mean) - test["recent30_ctr_pct"]
        )
        numeric.append("ctr_opportunity_gap")

    if feature_set in {"combined", "momentum_plus_ga4"}:
        for frame, design in ((train, train_x), (test, test_x)):
            sessions = pd.to_numeric(frame["ga4_sessions"], errors="coerce")
            pageviews = pd.to_numeric(frame["ga4_pageviews"], errors="coerce")
            engaged = pd.to_numeric(frame["ga4_engaged_sessions"], errors="coerce")
            scroll = pd.to_numeric(frame["scroll_events"], errors="coerce")
            design["ga4_engagement_rate"] = 100 * engaged / sessions.replace(0, np.nan)
            design["ga4_scroll_rate"] = 100 * scroll / pageviews.replace(0, np.nan)
            design["ga4_sessions_per_1000_impressions"] = (
                1000 * sessions / frame["recent30_impressions"].replace(0, np.nan)
            )
            design["ga4_available_days"] = frame["ga4_available_days"]
        numeric.extend(
            [
                "ga4_engagement_rate", "ga4_scroll_rate",
                "ga4_sessions_per_1000_impressions", "ga4_available_days",
            ]
        )

    if feature_set in {"combined", "momentum_plus_age"}:
        labels = ["up_to_90d", "91_to_180d", "181_to_365d", "over_365d"]
        for frame, design in ((train, train_x), (test, test_x)):
            design["age_band"] = pd.cut(
                pd.to_numeric(frame["content_age_days"], errors="coerce"),
                [0, 90, 180, 365, np.inf],
                labels=labels,
                include_lowest=True,
            ).astype("object")
        train_x = train_x.drop(columns=["content_age_days"])
        test_x = test_x.drop(columns=["content_age_days"])
        numeric.remove("content_age_days")
        categorical.append("age_band")
    return train_x, test_x, numeric, categorical


def _tie_keys(frame: pd.DataFrame) -> np.ndarray:
    return np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(
                    f"{client}:{page}".encode("utf-8"),
                    digest_size=8,
                    person=b"FlyRankP50",
                ).digest(),
                "big",
            )
            for client, page in zip(
                frame["client_hash_id"], frame["content_hash_id"], strict=True
            )
        ),
        dtype=np.uint64,
        count=len(frame),
    )


def _rank_indices(frame: pd.DataFrame, scores: np.ndarray, k: int) -> np.ndarray:
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("Ranking scores must be finite and align with candidates.")
    return np.lexsort((_tie_keys(frame), -scores))[: min(k, len(frame))]


def _coverage_bucket(days: pd.Series) -> dict[str, int]:
    bins = [-1, 6, 13, 20, 29, 30]
    labels = ["0-6", "7-13", "14-20", "21-29", "30"]
    return (
        pd.cut(days.fillna(0), bins=bins, labels=labels, include_lowest=True)
        .value_counts(sort=False)
        .astype(int)
        .to_dict()
    )


def _queue_diagnostics(frame: pd.DataFrame, scores: np.ndarray, k: int = 50) -> dict[str, Any]:
    indices = _rank_indices(frame, scores, k)
    queue = frame.iloc[indices]
    counts = queue["client_hash_id"].value_counts()
    traffic = queue["recent30_impressions"]
    labels = queue[TARGET]
    positive = int(labels.eq(1).sum())
    unknown = int(labels.isna().sum())
    selected_count = len(queue)
    complete = queue["recent30_days"].eq(30) & queue["future30_days"].eq(30)
    complete_labels = labels.loc[complete].dropna()
    largest_client = counts.index[0] if len(counts) else None
    excluding_largest = frame["client_hash_id"].ne(largest_client)
    leave_largest = frame.loc[excluding_largest]
    leave_scores = scores[excluding_largest.to_numpy()]
    leave_indices = _rank_indices(leave_largest, leave_scores, k)
    leave_queue = leave_largest.iloc[leave_indices]
    leave_labels = leave_queue[TARGET]
    leave_positive = int(leave_labels.eq(1).sum())
    leave_unknown = int(leave_labels.isna().sum())
    return {
        "selected": selected_count,
        "confirmed_positive": positive,
        "unknown_outcomes": unknown,
        "precision_lower_bound": positive / selected_count if selected_count else None,
        "precision_upper_bound": (
            (positive + unknown) / selected_count if selected_count else None
        ),
        "measurable_selected": int(labels.notna().sum()),
        "measurable_precision": (
            positive / int(labels.notna().sum()) if labels.notna().any() else None
        ),
        "recent_coverage_days": _coverage_bucket(queue["recent30_days"]),
        "prior_coverage_days": _coverage_bucket(queue["prior30_days"]),
        "future_coverage_days": _coverage_bucket(queue["future30_days"]),
        "insufficient_future_coverage_fraction": (
            unknown / selected_count if selected_count else None
        ),
        "represented_clients": int(counts.size),
        "largest_client_share": (
            float(counts.max() / selected_count) if selected_count else None
        ),
        "recent_impression_quantiles": {
            "p25": float(traffic.quantile(0.25)),
            "p50": float(traffic.quantile(0.50)),
            "p75": float(traffic.quantile(0.75)),
        },
        "complete_recent_and_future_coverage_same_queue": {
            "selected": int(complete.sum()),
            "confirmed_positive": int(complete_labels.eq(1).sum()),
            "measurable": int(len(complete_labels)),
            "measurable_precision": (
                float(complete_labels.mean()) if len(complete_labels) else None
            ),
        },
        "reranked_after_excluding_largest_top50_client": {
            "selected": len(leave_queue),
            "confirmed_positive": leave_positive,
            "unknown_outcomes": leave_unknown,
            "precision_lower_bound": (
                leave_positive / len(leave_queue) if len(leave_queue) else None
            ),
            "precision_upper_bound": (
                (leave_positive + leave_unknown) / len(leave_queue)
                if len(leave_queue) else None
            ),
        },
    }


def _tie_expectation(frame: pd.DataFrame, scores: np.ndarray, k: int) -> dict[str, float]:
    boundary = np.sort(scores)[-min(k, len(scores))]
    above = scores > boundary
    tied = scores == boundary
    slots = max(0, min(k, len(scores)) - int(above.sum()))
    known_positive = frame[TARGET].eq(1).to_numpy()
    unknown = frame[TARGET].isna().to_numpy()
    tied_count = int(tied.sum())
    tie_fraction = slots / tied_count if tied_count else 0.0
    expected_positive = float(known_positive[above].sum())
    expected_unknown = float(unknown[above].sum())
    if tied_count:
        expected_positive += slots * float(known_positive[tied].mean())
        expected_unknown += slots * float(unknown[tied].mean())
    selected = min(k, len(scores))
    return {
        "expected_precision_lower_bound": expected_positive / selected,
        "expected_precision_upper_bound": (
            expected_positive + expected_unknown
        ) / selected,
        "boundary_tie_size": tied_count,
        "fraction_of_boundary_tie_selected": tie_fraction,
    }


def _metric_row(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    origin: str,
    candidate: str,
    history_policy: str,
    training_months: tuple[str, ...] = (),
    reporting_lag_days: int | None = None,
    model: str = "rule",
) -> tuple[dict[str, Any], np.ndarray]:
    frame = frame.reset_index(drop=True)
    scores = np.asarray(scores, dtype=float)
    queue50 = _rank_indices(frame, scores, 50)
    queue20 = _rank_indices(frame, scores, 20)
    queue100 = _rank_indices(frame, scores, 100)
    measured = frame[TARGET].notna().to_numpy()
    y = frame[TARGET].fillna(0).astype(int).to_numpy()
    top50 = y[queue50]
    unknown50 = int(frame[TARGET].iloc[queue50].isna().sum())
    positive50 = int(top50.sum())
    positive20 = int(y[queue20].sum())
    positive100 = int(y[queue100].sum())
    unknown20 = int(frame[TARGET].iloc[queue20].isna().sum())
    unknown100 = int(frame[TARGET].iloc[queue100].isna().sum())
    measured_test = frame.loc[measured, TARGET].astype(int)
    measured_scores = scores[measured]
    if measured_test.nunique() < 2:
        raise ValueError(f"Both target classes are required at origin {origin}.")
    record: dict[str, Any] = {
        "origin": origin,
        "candidate": candidate,
        "model": model,
        "history_policy": history_policy,
        "training_months": list(training_months),
        "reporting_lag_days": reporting_lag_days,
        "candidate_count": len(frame),
        "labeled_count": int(measured.sum()),
        "label_coverage": float(measured.mean()),
        "prevalence_measurable": float(measured_test.mean()),
        "p_at_20_confirmed_lower": positive20 / len(queue20),
        "p_at_20_confirmed_upper": (positive20 + unknown20) / len(queue20),
        "p_at_20_measurable": (
            positive20 / (len(queue20) - unknown20)
            if len(queue20) > unknown20 else None
        ),
        "p_at_50_confirmed_lower": positive50 / len(queue50),
        "p_at_50_measurable": (
            positive50 / (len(queue50) - unknown50)
            if len(queue50) > unknown50 else None
        ),
        "p_at_50_confirmed_upper": (positive50 + unknown50) / len(queue50),
        "top50_confirmed_positive": positive50,
        "top50_unknown": unknown50,
        "top50_measurable": len(queue50) - unknown50,
        "top50_precision_lower": positive50 / len(queue50),
        "top50_precision_upper": (positive50 + unknown50) / len(queue50),
        "p_at_100_confirmed_lower": positive100 / len(queue100),
        "p_at_100_confirmed_upper": (positive100 + unknown100) / len(queue100),
        "p_at_100_measurable": (
            positive100 / (len(queue100) - unknown100)
            if len(queue100) > unknown100 else None
        ),
        "average_precision_labeled": float(
            average_precision_score(measured_test, measured_scores)
        ),
        "roc_auc_labeled": float(roc_auc_score(measured_test, measured_scores)),
        "queue_diagnostics": _queue_diagnostics(frame, scores),
        "static_rule_tie_expectation": _tie_expectation(
            frame, baseline_scores(frame), 50
        ) if candidate == "static_rule" else None,
        "daily_rate_target_top50": {
            "confirmed_positive": int(
                frame["daily_rate_decline"].iloc[queue50].eq(1).sum()
            ),
            "unknown": int(
                frame["daily_rate_decline"].iloc[queue50].isna().sum()
            ),
            "measurable": int(
                frame["daily_rate_decline"].iloc[queue50].notna().sum()
            ),
            "precision_lower_bound": (
                float(frame["daily_rate_decline"].iloc[queue50].eq(1).sum())
                / len(queue50)
            ),
            "precision_upper_bound": (
                float(
                    frame["daily_rate_decline"].iloc[queue50].eq(1).sum()
                    + frame["daily_rate_decline"].iloc[queue50].isna().sum()
                )
                / len(queue50)
            ),
        },
    }
    return record, queue50


def _rule_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    recent = frame["recent30_impressions"].astype(float).clip(lower=0)
    prior = frame["prev30_impressions"].astype(float).clip(lower=0)
    recent_days = frame["recent30_days"].astype(float)
    prior_days = frame["prior30_days"].astype(float)
    recent7 = frame["recent7_impressions"].astype(float).clip(lower=0)
    recent7_days = frame["recent7_days"].astype(float)
    raw_velocity_ratio = recent / (prior + 1.0)
    raw_velocity_ratio = raw_velocity_ratio.where(prior_days.ge(14), 1.0)
    raw_acceleration_ratio = (
        recent7 / (recent / 30.0 * 7.0 + 1.0)
    ).where(recent7_days.ge(3), 1.0)
    adjusted_velocity = (
        (recent / recent_days.replace(0, np.nan))
        / (prior / prior_days.replace(0, np.nan))
    ).replace([np.inf, -np.inf], np.nan).where(prior_days.ge(14), 1.0).fillna(1.0)
    adjusted_acceleration = (
        (recent7 / recent7_days.replace(0, np.nan))
        / (recent / recent_days.replace(0, np.nan))
    ).replace([np.inf, -np.inf], np.nan).where(recent7_days.ge(3), 1.0).fillna(1.0)
    return {
        "static_rule": baseline_scores(frame),
        "velocity_rule": -raw_velocity_ratio.to_numpy(dtype=float),
        "acceleration_rule": -raw_acceleration_ratio.to_numpy(dtype=float),
        "coverage_adjusted_velocity_rule": -adjusted_velocity.to_numpy(dtype=float),
        "coverage_adjusted_acceleration_rule": -adjusted_acceleration.to_numpy(dtype=float),
    }


def _mature_training_frames(
    frames: dict[str, pd.DataFrame],
    origin_month: str,
    reporting_lag_days: int,
) -> dict[str, tuple[tuple[str, ...], pd.DataFrame]]:
    origin_date = _month_end(origin_month)
    mature_months = []
    for month in sorted(frames):
        cutoff = _month_end(month)
        label_end = cutoff + timedelta(days=30)
        available_date = label_end + timedelta(days=reporting_lag_days)
        if cutoff < origin_date and available_date <= origin_date:
            mature_months.append(month)
    if not mature_months:
        return {}
    choices = {
        "latest_only": tuple(mature_months[-1:]),
        "latest_three": tuple(mature_months[-3:]),
        "expanding": tuple(mature_months),
    }
    result = {}
    for policy, selected in choices.items():
        training = pd.concat([frames[month] for month in selected], ignore_index=True)
        training = training.loc[training[TARGET].notna()].copy()
        if training[TARGET].nunique() != 2:
            raise ValueError(
                f"Training set for {origin_month}/{policy} lacks both target classes."
            )
        result[policy] = (selected, training)
    return result


def _fit_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
    model_name: str,
) -> np.ndarray:
    known_train = train.loc[train[TARGET].notna()].copy()
    train_x, test_x, numeric, categorical = _design_matrices(
        known_train, test, feature_set
    )
    model = build_model(model_name, numeric, categorical)
    model.fit(train_x, known_train[TARGET].astype(int))
    return model.predict_proba(test_x)[:, 1]


def _summary_by_candidate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if "training_set_unique" in df:
        df = df.loc[df["training_set_unique"].isna() | df["training_set_unique"]]
    grouped = (
        df.groupby(["reporting_lag_days", "candidate", "model", "history_policy"])
        ["p_at_50_confirmed_lower"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_p_at_50",
                "std": "origin_sd_not_ci",
                "min": "worst_origin_p_at_50",
                "max": "best_origin_p_at_50",
                "count": "origin_count",
            }
        )
    )
    return df_round(grouped).to_dict(orient="records")


def df_round(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=["number"]).columns
    result = frame.copy()
    result[numeric] = result[numeric].round(4)
    return result


def _source_hashes() -> dict[str, str]:
    digest = hashlib.sha256
    paths = {
        "script": Path(__file__).resolve(),
        "notebook": ROOT / "work" / "notebooks" / "warehouse_feature_experiments.ipynb",
    }
    return {
        name: digest(path.read_bytes()).hexdigest()
        for name, path in paths.items()
        if path.is_file()
    }


def _package_versions() -> dict[str, str]:
    packages = ("duckdb", "numpy", "pandas", "scikit-learn", "huggingface-hub")
    return {
        name: importlib.metadata.version(name)
        for name in packages
    }


def _record_queue_overlaps(
    rows: list[dict[str, Any]],
    queue_sets: dict[tuple[str, str, int], set[tuple[str, str]]],
) -> None:
    for row in rows:
        key = (row["origin"], row["_queue_id"], row["reporting_lag_days"])
        current = queue_sets[key]
        row["queue_overlap_at_50"] = {
            name: len(current.intersection(other))
            for (origin, name, lag), other in queue_sets.items()
            if origin == row["origin"]
            and lag == row["reporting_lag_days"]
            and name != row["_queue_id"]
        }
        row.pop("_queue_id")


def _register_queue(
    frame: pd.DataFrame, scores: np.ndarray, candidate: str, origin: str,
    reporting_lag_days: int,
    queue_sets: dict[tuple[str, str, int], set[tuple[str, str]]],
) -> None:
    selected = _rank_indices(frame, scores, 50)
    queue_sets[(origin, candidate, reporting_lag_days)] = set(
        zip(
            frame.iloc[selected]["client_hash_id"].astype(str),
            frame.iloc[selected]["content_hash_id"].astype(str),
            strict=True,
        )
    )


def _run_one_origin(
    *,
    origin_month: str,
    test: pd.DataFrame,
    histories: dict[str, tuple[tuple[str, ...], pd.DataFrame]],
    reporting_lag_days: int,
    rows: list[dict[str, Any]],
    queue_sets: dict[tuple[str, str, int], set[tuple[str, str]]],
    cache: dict[tuple[Any, ...], np.ndarray],
    feature_sets: tuple[str, ...] = ("control", "momentum"),
    history_policies: tuple[str, ...] = HISTORY_POLICIES,
    model_names: tuple[str, ...] = MODELS,
    include_rules: bool = True,
) -> None:
    if include_rules:
        for candidate, scores in _rule_scores(test).items():
            row, _ = _metric_row(
                test, scores, origin=origin_month, candidate=candidate,
                history_policy="not_applicable",
                reporting_lag_days=reporting_lag_days, model="rule",
            )
            row["_queue_id"] = candidate
            rows.append(row)
            _register_queue(
                test, scores, candidate, origin_month, reporting_lag_days, queue_sets
            )
        rng = np.random.default_rng(SEED + int(origin_month.replace("-", "")))
        random_scores = np.empty(len(test), dtype=float)
        random_scores[rng.permutation(len(test))] = np.arange(len(test), dtype=float)
        random_row, _ = _metric_row(
            test, random_scores, origin=origin_month,
            candidate="random_reference", history_policy="not_applicable",
            reporting_lag_days=reporting_lag_days, model="uniform_random",
        )
        target = test[TARGET].to_numpy(dtype=float)
        positive_pool = int(np.nansum(target))
        unknown_pool = int(np.isnan(target).sum())
        negative_pool = len(target) - positive_pool - unknown_pool
        sampled_positive_count = rng.hypergeometric(
            positive_pool, len(target) - positive_pool, 50, size=1000
        )
        sampled_unknown_count = np.zeros(1000, dtype=int)
        for positive_count in np.unique(sampled_positive_count):
            sample_indices = np.flatnonzero(sampled_positive_count == positive_count)
            sampled_unknown_count[sample_indices] = rng.hypergeometric(
                unknown_pool,
                negative_pool,
                50 - int(positive_count),
                size=len(sample_indices),
            )
        sampled_positive = sampled_positive_count / 50
        sampled_unknown = sampled_unknown_count / 50
        sampled_measurable = 50 - sampled_unknown_count
        sampled_measured_precision = np.divide(
            sampled_positive_count,
            sampled_measurable,
            out=np.full(len(sampled_positive_count), np.nan),
            where=sampled_measurable > 0,
        )
        random_row["random_sampling_distribution"] = {
            "expected_measurable_precision": float(
                np.nanmean(sampled_measured_precision)
            ),
            "expected_confirmed_lower_bound": float(sampled_positive.mean()),
            "confirmed_lower_bound_q05_q95": [
                float(np.quantile(sampled_positive, 0.05)),
                float(np.quantile(sampled_positive, 0.95)),
            ],
            "unknown_upper_bound_increment_q05_q95": [
                float(np.quantile(sampled_unknown, 0.05)),
                float(np.quantile(sampled_unknown, 0.95)),
            ],
            "simulations": 1000,
        }
        random_row["_queue_id"] = "random_reference"
        rows.append(random_row)
        _register_queue(
            test, random_scores, "random_reference", origin_month,
            reporting_lag_days, queue_sets,
        )

    seen_history: dict[tuple[str, ...], str] = {}
    for policy in history_policies:
        training_months, train = histories[policy]
        same_as = seen_history.get(training_months)
        seen_history.setdefault(training_months, policy)
        for feature_set in feature_sets:
            for model_name in model_names:
                key = (
                    origin_month,
                    reporting_lag_days,
                    training_months,
                    feature_set,
                    model_name,
                )
                if key not in cache:
                    cache[key] = _fit_score(train, test, feature_set, model_name)
                scores = cache[key]
                candidate = feature_set
                row, _ = _metric_row(
                    test, scores, origin=origin_month, candidate=candidate,
                    history_policy=policy, training_months=training_months,
                    reporting_lag_days=reporting_lag_days, model=model_name,
                )
                row["training_set_unique"] = same_as is None
                row["same_training_set_as"] = same_as
                queue_id = f"{candidate}:{model_name}:{policy}"
                row["_queue_id"] = queue_id
                rows.append(row)
                _register_queue(
                    test,
                    scores,
                    queue_id,
                    origin_month,
                    reporting_lag_days,
                    queue_sets,
                )


def run_experiment(config: ExperimentConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Run the declared temporal study and persist aggregate-only receipts."""
    if config.decline_threshold != 0.80:
        raise ValueError("The primary target threshold is frozen at 0.80.")
    if config.top_k != K_VALUES:
        raise ValueError("The primary queue sizes are frozen at P@20/50/100.")
    if config.primary_reporting_lag_days < 0:
        raise ValueError("Reporting lag cannot be negative.")
    if config.primary_reporting_lag_days in config.reporting_lag_sensitivity_days:
        raise ValueError("The primary lag cannot be repeated as a sensitivity.")
    if not set(config.prediction_origins).issubset(config.feature_months):
        raise ValueError("Each prediction origin must have a feature frame.")

    edge_label_end = date(2026, 1, 31) + timedelta(days=30)
    february_prediction = date(2026, 2, 28)
    february_window_start = _feature_window(february_prediction)["recent_start"]
    if (
        edge_label_end != date(2026, 3, 2)
        or edge_label_end <= february_prediction
        or february_window_start != date(2026, 1, 30)
    ):
        raise AssertionError("January 31 maturity edge case was not handled correctly.")

    source_months = months_between(
        min(
            _feature_window(_month_end(month) - timedelta(days=lag))["prior_start"]
            for month in config.feature_months
            for lag in (
                config.primary_reporting_lag_days,
                *config.reporting_lag_sensitivity_days,
            )
        ),
        max(
            _month_end(month) + timedelta(days=30)
            for month in config.feature_months
        ),
    )
    if "2026-06" in source_months or any(
        month > "2026-05" for month in source_months
    ):
        raise AssertionError("Development scans must not read June 2026.")

    print("Source: gated FlyRank/internship-warehouse from the Hugging Face Hub.")
    print(
        f"Downloading {len(source_months)} required monthly Parquet shards and "
        "dim_content into the Hugging Face-managed cache for DuckDB."
    )
    print("June partitions are excluded; feature frames are held in memory only.")
    print(f"Declared reporting-lag scenario: {config.primary_reporting_lag_days} days; "
          f"sensitivity: {config.reporting_lag_sensitivity_days} day(s).")
    token = resolve_token()
    warehouse_revision, partition_files, dimension_file = (
        cache_hf_warehouse_files(source_months, token)
    )
    del token
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=false")
    try:
        frames_by_lag = load_origin_frames(
            con,
            config.feature_months,
            config,
            (config.primary_reporting_lag_days, *config.reporting_lag_sensitivity_days),
            partition_files,
            dimension_file,
        )
    finally:
        con.close()

    data_audit = []
    for lag_days, lag_frames in frames_by_lag.items():
        for month, frame in lag_frames.items():
            prediction_date = _month_end(month)
            feature_cutoff = prediction_date - timedelta(days=lag_days)
            window = _feature_window(feature_cutoff)
            data_audit.append(
                {
                    "feature_month": month,
                    "reporting_lag_days": lag_days,
                    "prediction_date": str(prediction_date),
                    "feature_cutoff": str(feature_cutoff),
                    "feature_recent_start": str(window["recent_start"]),
                    "feature_recent_end": str(window["recent_end"]),
                    "feature_prior_start": str(window["prior_start"]),
                    "feature_prior_end": str(window["prior_end"]),
                    "feature_recent7_start": str(window["recent7_start"]),
                    "feature_recent7_end": str(window["recent7_end"]),
                    "outcome_start": str(prediction_date + timedelta(days=1)),
                    "outcome_end": str(prediction_date + timedelta(days=30)),
                    "candidate_count_prediction_time": len(frame),
                    "labeled_count": int(frame[TARGET].notna().sum()),
                    "label_coverage": float(frame[TARGET].notna().mean()),
                    "measurable_prevalence": float(frame[TARGET].mean()),
                    "complete_recent_and_future_count": int(
                        (frame["recent30_days"].eq(30) & frame["future30_days"].eq(30)).sum()
                    ),
                }
            )
    frames = frames_by_lag[config.primary_reporting_lag_days]
    result_rows: list[dict[str, Any]] = []
    queue_sets: dict[tuple[str, str, int], set[tuple[str, str]]] = {}
    score_cache: dict[tuple[Any, ...], np.ndarray] = {}
    maturity_audit = []
    for lag_days, lag_frames in frames_by_lag.items():
        previous_sets: dict[tuple[str, ...], str] = {}
        for origin in config.prediction_origins:
            histories = _mature_training_frames(lag_frames, origin, lag_days)
            for policy, (months, training) in histories.items():
                label_ends = [
                    _month_end(month) + timedelta(days=30) for month in months
                ]
                available_dates = [
                    end_date + timedelta(days=lag_days) for end_date in label_ends
                ]
                maturity_audit.append(
                    {
                        "origin": origin,
                        "origin_date": str(_month_end(origin)),
                        "reporting_lag_days": lag_days,
                        "history_policy": policy,
                        "training_months": list(months),
                        "training_set_unique": months not in previous_sets,
                        "same_training_set_as": previous_sets.get(months),
                        "training_rows_labeled": int(len(training)),
                        "training_label_end_dates": [str(value) for value in label_ends],
                        "training_label_available_dates": [
                            str(value) for value in available_dates
                        ],
                    }
                )
                previous_sets.setdefault(months, policy)

    for origin in config.prediction_origins:
        histories = _mature_training_frames(
            frames, origin, config.primary_reporting_lag_days
        )
        if not histories:
            raise ValueError(f"No mature training history exists for {origin}.")
        test = frames[origin]
        _run_one_origin(
            origin_month=origin, test=test, histories=histories,
            reporting_lag_days=config.primary_reporting_lag_days,
            rows=result_rows, queue_sets=queue_sets, cache=score_cache,
        )
        _run_one_origin(
            origin_month=origin, test=test, histories=histories,
            reporting_lag_days=config.primary_reporting_lag_days,
            rows=result_rows, queue_sets=queue_sets, cache=score_cache,
            feature_sets=("combined",),
            history_policies=("latest_only", "expanding"),
            model_names=("random_forest",),
            include_rules=False,
        )

    # A second declared zero-lag sensitivity uses latest-only and expanding history.
    for lag_days in config.reporting_lag_sensitivity_days:
        for origin in config.prediction_origins:
            lag_frames = frames_by_lag[lag_days]
            histories = _mature_training_frames(lag_frames, origin, lag_days)
            _run_one_origin(
                origin_month=origin,
                test=frames_by_lag[lag_days][origin],
                histories=histories,
                reporting_lag_days=lag_days, rows=result_rows,
                queue_sets=queue_sets, cache=score_cache,
                history_policies=("latest_only", "expanding"),
                include_rules=False,
            )

    main_df = pd.DataFrame(
        row for row in result_rows
        if row["reporting_lag_days"] == config.primary_reporting_lag_days
    )
    momentum_rf = main_df[
        main_df["candidate"].eq("momentum")
        & main_df["model"].eq("random_forest")
        & main_df["history_policy"].eq("latest_only")
    ].set_index("origin")
    combined_rf = main_df[
        main_df["candidate"].eq("combined")
        & main_df["model"].eq("random_forest")
        & main_df["history_policy"].eq("latest_only")
    ].set_index("origin")
    momentum_component_ran = (
        not momentum_rf.empty
        and float(
            (
                momentum_rf["p_at_50_confirmed_lower"]
                - main_df[
                    main_df["candidate"].eq("control")
                    & main_df["model"].eq("random_forest")
                    & main_df["history_policy"].eq("latest_only")
                ].set_index("origin")["p_at_50_confirmed_lower"]
            ).mean()
        ) > 0
        and int(
            (
                momentum_rf["p_at_50_confirmed_lower"]
                > main_df[
                    main_df["candidate"].eq("control")
                    & main_df["model"].eq("random_forest")
                    & main_df["history_policy"].eq("latest_only")
                ].set_index("origin")["p_at_50_confirmed_lower"]
            ).sum()
        ) >= 3
    )
    if momentum_component_ran:
        for origin in config.prediction_origins:
            histories = _mature_training_frames(
                frames, origin, config.primary_reporting_lag_days
            )
            _run_one_origin(
                origin_month=origin, test=frames[origin], histories=histories,
                reporting_lag_days=config.primary_reporting_lag_days,
                rows=result_rows, queue_sets=queue_sets, cache=score_cache,
                feature_sets=(
                    "velocity_only", "acceleration_only", "coverage_only",
                    "velocity_acceleration",
                ),
                history_policies=("latest_only", "expanding"),
                model_names=("random_forest",),
                include_rules=False,
            )

    combined_additions_ran = False
    if not combined_rf.empty and not momentum_rf.empty:
        combined_delta = (
            combined_rf["p_at_50_confirmed_lower"]
            - momentum_rf["p_at_50_confirmed_lower"]
        )
        combined_additions_ran = (
            float(combined_delta.mean()) > 0 and int(combined_delta.gt(0).sum()) >= 3
        )
    if combined_additions_ran:
        for origin in config.prediction_origins:
            histories = _mature_training_frames(
                frames, origin, config.primary_reporting_lag_days
            )
            _run_one_origin(
                origin_month=origin, test=frames[origin], histories=histories,
                reporting_lag_days=config.primary_reporting_lag_days,
                rows=result_rows, queue_sets=queue_sets, cache=score_cache,
                feature_sets=(
                    "momentum_plus_ctr", "momentum_plus_ga4", "momentum_plus_age",
                ),
                history_policies=("latest_only", "expanding"),
                model_names=("random_forest",),
                include_rules=False,
            )

    _record_queue_overlaps(result_rows, queue_sets)
    primary_rows = [
        row for row in result_rows
        if row["reporting_lag_days"] == config.primary_reporting_lag_days
    ]
    per_origin = pd.DataFrame(primary_rows)
    model_rows = per_origin[
        per_origin["model"].isin(MODELS)
        & per_origin["candidate"].isin(("control", "momentum"))
        & per_origin["history_policy"].isin(HISTORY_POLICIES)
    ]
    interaction_rows = []
    for origin in config.prediction_origins:
        for policy in HISTORY_POLICIES:
            for model_name in MODELS:
                current = model_rows[
                    model_rows["origin"].eq(origin)
                    & model_rows["history_policy"].eq(policy)
                    & model_rows["model"].eq(model_name)
                ].set_index("candidate")
                latest = model_rows[
                    model_rows["origin"].eq(origin)
                    & model_rows["history_policy"].eq("latest_only")
                    & model_rows["model"].eq(model_name)
                ].set_index("candidate")
                if (
                    {"control", "momentum"}.issubset(current.index)
                    and {"control", "momentum"}.issubset(latest.index)
                ):
                    history_control = float(
                        float(current.loc["control", "p_at_50_confirmed_lower"])
                        - float(latest.loc["control", "p_at_50_confirmed_lower"])
                    )
                    history_momentum = float(
                        float(current.loc["momentum", "p_at_50_confirmed_lower"])
                        - float(latest.loc["momentum", "p_at_50_confirmed_lower"])
                    )
                    interaction_rows.append(
                        {
                            "origin": origin,
                            "model": model_name,
                            "history_policy": policy,
                            "momentum_benefit_p50": float(
                                float(current.loc["momentum", "p_at_50_confirmed_lower"])
                                - float(current.loc["control", "p_at_50_confirmed_lower"])
                            ),
                            "history_benefit_control_p50": history_control,
                            "history_benefit_momentum_p50": history_momentum,
                            "history_feature_interaction_p50": (
                                history_momentum - history_control
                            ),
                            "rf_increment_over_lr_p50": (
                                float(
                                    model_rows[
                                        model_rows["origin"].eq(origin)
                                        & model_rows["history_policy"].eq(policy)
                                        & model_rows["candidate"].eq("control")
                                        & model_rows["model"].eq("random_forest")
                                    ]["p_at_50_confirmed_lower"].iloc[0]
                                )
                                - float(
                                    model_rows[
                                        model_rows["origin"].eq(origin)
                                        & model_rows["history_policy"].eq(policy)
                                        & model_rows["candidate"].eq("control")
                                        & model_rows["model"].eq("logistic_regression")
                                    ]["p_at_50_confirmed_lower"].iloc[0]
                                )
                            ),
                            "rf_increment_over_lr_momentum_p50": (
                                float(
                                    model_rows[
                                        model_rows["origin"].eq(origin)
                                        & model_rows["history_policy"].eq(policy)
                                        & model_rows["candidate"].eq("momentum")
                                        & model_rows["model"].eq("random_forest")
                                    ]["p_at_50_confirmed_lower"].iloc[0]
                                )
                                - float(
                                    model_rows[
                                        model_rows["origin"].eq(origin)
                                        & model_rows["history_policy"].eq(policy)
                                        & model_rows["candidate"].eq("momentum")
                                        & model_rows["model"].eq("logistic_regression")
                                    ]["p_at_50_confirmed_lower"].iloc[0]
                                )
                            ),
                            "training_months": current.loc["momentum", "training_months"],
                            "training_set_unique": current.loc["momentum", "training_set_unique"],
                            "same_training_set_as": current.loc["momentum", "same_training_set_as"],
                        }
                    )
    summaries = _summary_by_candidate(primary_rows)
    rules = per_origin[
        per_origin["candidate"].isin(
            [
                "static_rule", "velocity_rule", "acceleration_rule",
                "coverage_adjusted_velocity_rule",
                "coverage_adjusted_acceleration_rule",
            ]
        )
    ]
    rule_means = rules.groupby("candidate")["p_at_50_confirmed_lower"].mean()
    strongest_simple_rule = str(rule_means.idxmax())
    paired_comparisons = []
    for candidate_row in primary_rows:
        if candidate_row["model"] not in MODELS:
            continue
        for comparator in (
            "random_reference",
            "static_rule",
            "velocity_rule",
            "acceleration_rule",
            "coverage_adjusted_velocity_rule",
            "coverage_adjusted_acceleration_rule",
        ):
            baseline_row = next(
                row for row in primary_rows
                if row["origin"] == candidate_row["origin"]
                and row["candidate"] == comparator
                and row["reporting_lag_days"]
                == candidate_row["reporting_lag_days"]
            )
            candidate_diag = candidate_row["queue_diagnostics"]
            baseline_diag = baseline_row["queue_diagnostics"]
            paired_comparisons.append(
                {
                    "origin": candidate_row["origin"],
                    "candidate": candidate_row["candidate"],
                    "model": candidate_row["model"],
                    "history_policy": candidate_row["history_policy"],
                    "training_months": candidate_row["training_months"],
                    "training_set_unique": candidate_row.get("training_set_unique"),
                    "comparator": comparator,
                    "delta_p_at_50_confirmed_lower": (
                        candidate_row["p_at_50_confirmed_lower"]
                        - baseline_row["p_at_50_confirmed_lower"]
                    ),
                    "delta_p_at_20_confirmed_lower": (
                        candidate_row["p_at_20_confirmed_lower"]
                        - baseline_row["p_at_20_confirmed_lower"]
                    ),
                    "delta_p_at_100_confirmed_lower": (
                        candidate_row["p_at_100_confirmed_lower"]
                        - baseline_row["p_at_100_confirmed_lower"]
                    ),
                    "delta_unknown_fraction_top50": (
                        candidate_diag["insufficient_future_coverage_fraction"]
                        - baseline_diag["insufficient_future_coverage_fraction"]
                    ),
                    "delta_largest_client_share": (
                        candidate_diag["largest_client_share"]
                        - baseline_diag["largest_client_share"]
                    ),
                }
            )
    paired_df = pd.DataFrame(paired_comparisons)
    fixed_rule_pairs = paired_df[
        paired_df["comparator"].eq(strongest_simple_rule)
    ]
    decision_rows = []
    for (candidate, model_name, policy), group in fixed_rule_pairs.groupby(
        ["candidate", "model", "history_policy"], sort=False
    ):
        mean_delta = float(group["delta_p_at_50_confirmed_lower"].mean())
        wins = int(group["delta_p_at_50_confirmed_lower"].gt(0).sum())
        p20_gain = float(group["delta_p_at_20_confirmed_lower"].mean())
        p100_gain = float(group["delta_p_at_100_confirmed_lower"].mean())
        unknown_delta = float(group["delta_unknown_fraction_top50"].mean())
        concentration_delta = float(group["delta_largest_client_share"].mean())
        hurdle = (
            len(group) == len(config.prediction_origins)
            and mean_delta >= config.practical_hurdle_cases_per_50 / 50
            and wins >= config.practical_hurdle_origins
            and p20_gain > 0
            and p100_gain > 0
            and unknown_delta <= 0
            and concentration_delta <= 0
        )
        decision_rows.append(
            {
                "candidate": candidate,
                "model": model_name,
                "history_policy": policy,
                "fixed_simple_competitor": strongest_simple_rule,
                "origins": len(group),
                "mean_delta_p_at_50": mean_delta,
                "positive_origins": wins,
                "mean_delta_p_at_20": p20_gain,
                "mean_delta_p_at_100": p100_gain,
                "mean_delta_unknown_fraction": unknown_delta,
                "mean_delta_largest_client_share": concentration_delta,
                "practical_hurdle_met": hurdle,
                "adoption_status": (
                    "provisional hurdle met; independent final test still required"
                    if hurdle else "not demonstrated; retain as unresolved/reject for adoption"
                ),
            }
        )
    march_zero_lag = next(
        row for row in data_audit
        if row["feature_month"] == "2026-03" and row["reporting_lag_days"] == 0
    )
    march_primary_lag = next(
        row for row in data_audit
        if row["feature_month"] == "2026-03"
        and row["reporting_lag_days"] == config.primary_reporting_lag_days
    )
    result = {
        "configuration": asdict(config),
        "source": {
            "release": RELEASE,
            "release_build": "flyrank_pseudonymized_warehouse_release_v20260703",
            "table": "fact_content_daily_performance",
            "query_engine": "DuckDB",
            "read_uri_scheme": "Hugging Face Hub snapshot cache read by DuckDB",
            "huggingface_revision": warehouse_revision,
            "fact_partition_paths": {
                month: partition_files[month] for month in source_months
            },
            "dimension_path": dimension_file,
            "fact_partitions": list(
                months_between(
                    min(
                        _feature_window(
                            _month_end(m) - timedelta(days=lag)
                        )["prior_start"]
                        for m in config.feature_months
                        for lag in (
                            config.primary_reporting_lag_days,
                            *config.reporting_lag_sensitivity_days,
                        )
                    ),
                            max(
                                _month_end(m) + timedelta(days=30)
                                for m in config.feature_months
                            ),
                )
            ),
            "june_read": False,
            "raw_rows_saved": False,
            "identifiers_saved": False,
            "partition_date_bounds": next(iter(frames.values())).attrs[
                "source_partition_bounds"
            ],
        },
        "reporting_delay_model": {
            "primary_days": config.primary_reporting_lag_days,
            "feature_observations_end": (
                "prediction date minus reporting lag"
            ),
            "training_label_usable_when": (
                "nominal 30-day label end plus reporting lag is on or before prediction date"
            ),
            "actual_warehouse_reporting_sla_verified": False,
            "lag_sensitivity_days": list(config.reporting_lag_sensitivity_days),
        },
        "rule_definitions": {
            "static_rule": "unchanged 0.40 visibility + 0.35 visible/top-10/CTR<1% + 0.25 visible/age>=91d",
            "velocity_rule": "rank ascending recent30_impressions/(prior30_impressions+1); <14 prior GSC days uses neutral ratio 1",
            "acceleration_rule": "rank ascending recent7_impressions/(recent30_impressions*7/30+1); <3 recent7 GSC days uses neutral ratio 1",
            "coverage_adjusted_velocity_rule": "rank ascending recent30 impressions per observed day divided by prior30 impressions per observed day; insufficient prior coverage uses neutral ratio 1",
            "coverage_adjusted_acceleration_rule": "rank ascending recent7 impressions per observed day divided by recent30 impressions per observed day; <3 recent7 GSC days uses neutral ratio 1",
        },
        "data_window_audit": data_audit,
        "training_maturity_audit": maturity_audit,
        "calendar_edge_case": {
            "january_31_label_end": str(edge_label_end),
            "february_28_prediction": str(february_prediction),
            "february_28_zero_lag_trailing30_start": str(
                _feature_window(date(2026, 2, 28))["recent_start"]
            ),
            "february_trailing30_includes_january_30_31": (
                _feature_window(date(2026, 2, 28))["recent_start"]
                == date(2026, 1, 30)
            ),
            "january_31_excluded_for_february_28": True,
            "assertion_passed": True,
        },
        "results_per_origin": result_rows,
        "equal_weight_origin_summaries": summaries,
        "feature_history_interaction_by_origin": interaction_rows,
        "paired_comparisons_by_origin": paired_comparisons,
        "feature_and_model_decisions": decision_rows,
        "conditional_followups": {
            "momentum_component_ablations_run": momentum_component_ran,
            "combined_feature_group_ablations_run": combined_additions_ran,
            "combined_rf_vs_momentum_rf_positive_origins": (
                int(combined_delta.gt(0).sum())
                if not combined_rf.empty and not momentum_rf.empty else 0
            ),
        },
        "previous_artifact_comparison": {
            "march_rows_in_saved_notebook_output": 95810,
            "new_march_zero_lag_candidate_count": march_zero_lag[
                "candidate_count_prediction_time"
            ],
            "new_march_zero_lag_labeled_count": march_zero_lag["labeled_count"],
            "new_march_zero_lag_label_count_delta_vs_saved_rows": (
                march_zero_lag["labeled_count"] - 95810
            ),
            "new_march_primary_lag_candidate_count": march_primary_lag[
                "candidate_count_prediction_time"
            ],
            "note": (
                "The previous 95,810 March rows were after an inner future-label join. "
                "Zero-lag counts compare the same nominal feature cutoff; the primary "
                "scenario additionally withholds the latest seven feature days."
            ),
            "duplicate_heuristic_rows_removed_by_rebuilding_results": True,
            "narrative_tables_generated_from_structured_results": True,
        },
        "prior_run_context": {
            "reported_forward_rf_control_p_at_50": 0.14,
            "reported_forward_rf_control_with_february_plus_march_training": 0.72,
            "reported_forward_combined_rf_p_at_50_march_only": 1.0,
            "reported_forward_combined_rf_p_at_50_february_plus_march": 0.98,
            "interpretation": (
                "These were one April-feature/May-outcome period on the prior future-filtered "
                "population. The combined score did not isolate momentum, and the 100% "
                "result requires the new coverage/concentration diagnostics; its cause is unproven."
            ),
        },
        "caveats": [
            "Development origins were previously inspected and are not untouched tests.",
            "The 7-day primary reporting delay is an explicit scenario, not a verified warehouse SLA.",
            "Zero-day reporting lag is reported as sensitivity only.",
            "The five development periods include overlapping outcomes and repeated pages/clients; origin SD is descriptive, not a confidence interval.",
            "June was not read; provenance for all earlier human or external feature-selection decisions is not independently certified, so no untouched final holdout is claimed.",
            "Decline prediction does not establish that refreshing a page improves its outcome.",
        ],
        "package_versions": _package_versions(),
        "python_version": platform.python_version(),
        "source_hashes_sha256": _source_hashes(),
    }
    output_dir = ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "experiment_results.json"
    output_path.write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print_report(result, output_path)
    return result


def print_report(result: dict[str, Any], output_path: Path | None = None) -> None:
    print("\nDATA/WINDOW AUDIT (prediction-time candidates; outcomes left unknown)")
    audit = pd.DataFrame(result["data_window_audit"])
    cols = [
        "feature_month", "reporting_lag_days", "feature_cutoff",
        "candidate_count_prediction_time", "labeled_count",
        "label_coverage", "measurable_prevalence", "complete_recent_and_future_count",
    ]
    print(df_round(audit[cols]).to_string(index=False))
    print("\nSUCCESSIVE DEVELOPMENT RESULTS (P@50 conservative lower bound)")
    rows = result["results_per_origin"]
    table = pd.DataFrame(rows)
    table = table[
        table["reporting_lag_days"].eq(result["configuration"]["primary_reporting_lag_days"])
        & table["candidate"].isin(
            ["random_reference", "static_rule", "velocity_rule", "acceleration_rule",
             "coverage_adjusted_velocity_rule", "coverage_adjusted_acceleration_rule",
             "control", "momentum", "combined"]
        )
    ]
    print(
        df_round(
            table[
                ["origin", "candidate", "model", "history_policy",
                 "training_months", "p_at_20_confirmed_lower",
                 "p_at_50_confirmed_lower", "p_at_50_confirmed_upper",
                 "p_at_50_measurable", "p_at_100_confirmed_lower",
                 "average_precision_labeled", "roc_auc_labeled",
                 "label_coverage"]
            ]
        ).to_string(index=False)
    )
    print("\nFEATURE × HISTORY INTERACTION BY ORIGIN")
    print(pd.DataFrame(result["feature_history_interaction_by_origin"]).to_string(index=False))
    print("\nPAIRED DELTAS OVER RANDOM AND FIXED SIMPLE RULES")
    paired = pd.DataFrame(result["paired_comparisons_by_origin"])
    if not paired.empty:
        paired = paired[
            paired["candidate"].isin(["control", "momentum", "combined"])
            & paired["history_policy"].isin(["latest_only", "expanding"])
            & paired["comparator"].isin(
                [
                    "random_reference",
                    result["feature_and_model_decisions"][0]["fixed_simple_competitor"]
                    if result["feature_and_model_decisions"] else "static_rule",
                ]
            )
        ]
        print(
            df_round(
                paired[
                    [
                        "origin", "candidate", "model", "history_policy",
                        "comparator", "delta_p_at_50_confirmed_lower",
                        "delta_p_at_20_confirmed_lower",
                        "delta_p_at_100_confirmed_lower",
                        "delta_unknown_fraction_top50",
                        "delta_largest_client_share",
                    ]
                ]
            ).to_string(index=False)
        )
    print("\nEQUAL-WEIGHT ORIGIN SUMMARIES (SD is descriptive, not a CI)")
    print(pd.DataFrame(result["equal_weight_origin_summaries"]).to_string(index=False))
    print("\nPREDECLARED ADOPTION SCREEN")
    print(pd.DataFrame(result["feature_and_model_decisions"]).to_string(index=False))
    april_combined = pd.DataFrame(result["results_per_origin"])
    april_combined = april_combined[
        april_combined["origin"].eq("2026-04")
        & april_combined["candidate"].eq("combined")
        & april_combined["model"].eq("random_forest")
        & april_combined["reporting_lag_days"].eq(
            result["configuration"]["primary_reporting_lag_days"]
        )
    ]
    if not april_combined.empty:
        print("\nAPRIL COMBINED RF TOP-50 DIAGNOSTICS (prior perfect-score check)")
        print(
            april_combined[
                [
                    "history_policy", "p_at_50_confirmed_lower",
                    "p_at_50_confirmed_upper", "queue_diagnostics",
                ]
            ].to_string(index=False)
        )
    print(
        "\nPrior artifact context: "
        f"RF control 14% (one forward month); combined RF 100% March-only / "
        f"98% with added training history. See the structured diagnostics above; "
        "these prior values are not generalization evidence."
    )
    print("\nEvidence boundary: no June data was read and no pipeline adoption is automatic.")
    if output_path:
        print(f"Aggregate-only results saved to: {output_path}")


def main() -> None:
    run_experiment(ExperimentConfig())


if __name__ == "__main__":
    main()
