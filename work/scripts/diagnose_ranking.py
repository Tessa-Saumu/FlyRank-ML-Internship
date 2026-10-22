"""Diagnose the submission pipeline's ranking claims on cached frames.

Reads the private frames produced by ``final_submission_pipeline.contract()``
(``work/local_final_pipeline/*.parquet``). It does not touch the warehouse and
does not rewrite any committed receipt.

It answers five questions that decide whether the model earns its place:

D1  Baseline resolution: is the frozen static rule actually ranking, or is its
    top-50 a lottery draw from a huge tie band?
D2  Label contamination: is the loss-of-traffic label largely implied by the
    last-seven-day run rate, i.e. is ``traffic_acceleration_7d`` an algebraic
    twin of the label rather than a forecast of it?
D3  Momentum-matched lift: inside a pool that momentum already ranks well, does
    a model still order pages better than the rule?
D4  Onset target: on pages whose recent run rate is still healthy, where the
    label is NOT implied by momentum, what is the base rate and does the model
    beat the rules?
D5  Queue shape: client concentration, unknown outcomes and action diversity of
    the 50-row queue, plus a client-capped alternative.

Run:
    python work/scripts/diagnose_ranking.py
    python work/scripts/diagnose_ranking.py --selftest   # synthetic, no cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

try:  # the new pipeline lives on submission/warehouse-temporal-pipeline
    import warehouse_feature_experiments as study  # noqa: E402
    SOURCE = "warehouse_feature_experiments"
except ModuleNotFoundError:  # pragma: no cover - exercised on main
    study = None
    SOURCE = "local fallback (copied definitions)"

TARGET = "is_declining_next30"
K = 50

CONTROL_FEATURES = [
    "log_recent30_impressions",
    "recent30_ctr_pct",
    "recent30_avg_position",
    "recent30_active_days",
    "content_age_days",
]
MOMENTUM_FEATURES = [
    "traffic_velocity_30d",
    "traffic_acceleration_7d",
    "has_prior30_gsc_coverage",
]


# --------------------------------------------------------------------------- #
# Minimal copy of the pipeline definitions, so this audit runs on any branch.
# These are byte-for-byte equivalent to warehouse_feature_experiments.py.
# --------------------------------------------------------------------------- #
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


def _design_matrices(train, test, feature_set):
    train_all, test_all = _momentum(train), _momentum(test)
    numeric = CONTROL_FEATURES.copy()
    categorical: list[str] = []
    train_x = train_all[CONTROL_FEATURES].copy()
    test_x = test_all[CONTROL_FEATURES].copy()
    if feature_set in {"control", "momentum"}:
        selected = [] if feature_set == "control" else MOMENTUM_FEATURES
        if selected:
            train_x = pd.concat([train_x, train_all[selected]], axis=1)
            test_x = pd.concat([test_x, test_all[selected]], axis=1)
            numeric.extend(selected)
    return train_x, test_x, numeric, categorical


def _tie_keys(frame: pd.DataFrame) -> np.ndarray:
    import hashlib

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
    return np.lexsort((_tie_keys(frame), -np.asarray(scores, dtype=float)))[: min(k, len(frame))]


def _rule_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    recent = frame["recent30_impressions"].astype(float).clip(lower=0)
    prior = frame["prev30_impressions"].astype(float).clip(lower=0)
    recent_days = frame["recent30_days"].astype(float)
    prior_days = frame["prior30_days"].astype(float)
    recent7 = frame["recent7_impressions"].astype(float).clip(lower=0)
    recent7_days = frame["recent7_days"].astype(float)
    raw_velocity = (recent / (prior + 1.0)).where(prior_days.ge(14), 1.0)
    raw_acceleration = (recent7 / (recent / 30.0 * 7.0 + 1.0)).where(recent7_days.ge(3), 1.0)
    adjusted_velocity = (
        (recent / recent_days.replace(0, np.nan))
        / (prior / prior_days.replace(0, np.nan))
    ).replace([np.inf, -np.inf], np.nan).where(prior_days.ge(14), 1.0).fillna(1.0)
    adjusted_acceleration = (
        (recent7 / recent7_days.replace(0, np.nan))
        / (recent / recent_days.replace(0, np.nan))
    ).replace([np.inf, -np.inf], np.nan).where(recent7_days.ge(3), 1.0).fillna(1.0)
    visible = (frame["recent30_impressions"] >= 500).astype(int)
    top10 = (frame["recent30_avg_position"].gt(0) & frame["recent30_avg_position"].le(10)).astype(int)
    low_ctr = (frame["recent30_ctr_pct"] < 1.0).astype(int)
    stale = (frame["content_age_days"] >= 91).astype(int)
    static = (0.40 * visible + 0.35 * visible * top10 * low_ctr + 0.25 * visible * stale).to_numpy(float)
    return {
        "static_rule": static,
        "velocity_rule": -raw_velocity.to_numpy(float),
        "acceleration_rule": -raw_acceleration.to_numpy(float),
        "coverage_adjusted_velocity_rule": -adjusted_velocity.to_numpy(float),
        "coverage_adjusted_acceleration_rule": -adjusted_acceleration.to_numpy(float),
    }


def _mature_training_frames(frames, origin_month, reporting_lag_days):
    from calendar import monthrange
    from datetime import date, timedelta

    year, month = map(int, origin_month.split("-"))
    origin_date = date(year, month, monthrange(year, month)[1])
    mature = []
    for name in sorted(frames):
        y, m = map(int, name.split("-"))
        cutoff = date(y, m, monthrange(y, m)[1])
        available = cutoff + timedelta(days=30 + reporting_lag_days)
        if cutoff < origin_date and available <= origin_date:
            mature.append(name)
    if not mature:
        return {}
    train = pd.concat([frames[m] for m in mature[-1:]], ignore_index=True)
    train = train.loc[train[TARGET].notna()]
    return {"latest_only": (tuple(mature[-1:]), train)}


def _fit_score(train, test, feature_set, model_name):
    known = train.loc[train[TARGET].notna()]
    x_train, x_test, numeric, categorical = _design_matrices(known, test, feature_set)
    model = build_model(model_name, numeric, categorical)
    model.fit(x_train, known[TARGET].astype(int))
    return model.predict_proba(x_test)[:, 1]


def build_model(model_name, numeric, categorical):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
    preprocess = ColumnTransformer(
        [("numeric", Pipeline(steps), numeric)], remainder="drop"
    )
    classifier = (
        LogisticRegression(max_iter=1000, random_state=42)
        if model_name == "logistic_regression"
        else RandomForestClassifier(
            class_weight="balanced_subsample",
            max_depth=10,
            min_samples_leaf=25,
            n_estimators=200,
            n_jobs=4,
            random_state=42,
        )
    )
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


if study is None:  # provide the same surface the audit calls
    from types import SimpleNamespace

    study = SimpleNamespace(
        TARGET=TARGET,
        MODELS=("logistic_regression", "random_forest"),
        _momentum=_momentum,
        _design_matrices=_design_matrices,
        _tie_keys=_tie_keys,
        _rank_indices=_rank_indices,
        _rule_scores=_rule_scores,
        _mature_training_frames=_mature_training_frames,
        _fit_score=_fit_score,
        build_model=build_model,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _pos(labels: pd.Series) -> np.ndarray:
    """1/0 for confirmed outcomes, 0 for unknown (confirmed-lower convention)."""
    return labels.fillna(0).astype(int).to_numpy()


def topk_indices(frame: pd.DataFrame, scores: np.ndarray, k: int = K) -> np.ndarray:
    return study._rank_indices(frame.reset_index(drop=True), np.asarray(scores, float), k)


def precision_block(frame: pd.DataFrame, scores: np.ndarray, k: int = K, mask=None) -> dict:
    """P@k on a (optionally masked) frame, with unknown outcomes disclosed."""
    frame = frame.reset_index(drop=True)
    if mask is not None:
        frame = frame.loc[mask].reset_index(drop=True)
        scores = np.asarray(scores, float)[np.asarray(mask)]
    if len(frame) < k:
        return {"n": int(len(frame)), "p_at_k": None, "unknown": None}
    idx = topk_indices(frame, scores, k)
    labels = frame[TARGET].iloc[idx]
    return {
        "n": int(len(frame)),
        "p_at_k": float(_pos(labels).mean()),
        "unknown": int(labels.isna().sum()),
    }


def tie_band(frame: pd.DataFrame, scores: np.ndarray, k: int = K) -> dict:
    """Order-independent precision@k for a score with a boundary tie."""
    scores = np.asarray(scores, float)
    labels = frame[TARGET]
    measurable = labels.notna().to_numpy()
    y1 = labels.eq(1).to_numpy()
    boundary = np.sort(scores)[-min(k, len(scores))]
    above = scores > boundary
    tied = (scores == boundary) & measurable
    slots = max(0, min(k, len(scores)) - int(above.sum()))
    band_rate = float(y1[tied].mean()) if tied.sum() else float("nan")
    expected = (int(y1[above].sum()) + slots * band_rate) / min(k, len(scores))
    return {
        "top_score_band_size": int((scores == scores.max()).sum()),
        "top_score_band_share": float((scores == scores.max()).mean()),
        "boundary_tie_size": int((scores == boundary).sum()),
        "slots_drawn_from_tie": int(slots),
        "share_of_tie_band_selected": float(slots / max(1, (scores == boundary).sum())),
        "band_decline_rate": band_rate,
        "expected_p_at_k": float(expected),
        "distinct_score_values": int(pd.Series(scores).nunique()),
    }


def build_model_scores(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> dict:
    out = {}
    for model_name in study.MODELS:
        out[f"{feature_set}:{model_name}"] = study._fit_score(
            train, test, feature_set, model_name
        )
    return out


def _persistence_label(frame: pd.DataFrame) -> pd.Series:
    """Would the page decline if the last seven days simply continued?

    Uses the observed daily rate (dividing by days actually covered), so a page
    with partial tracking is not penalised twice.
    """
    days = frame["recent7_days"].replace(0, np.nan)
    return (frame["recent7_impressions"] / days * 30.0) < 0.80 * frame["recent30_impressions"]


def _coverage_ok(frame: pd.DataFrame) -> pd.Series:
    return frame["recent7_days"].ge(3)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def d1_resolution(frame: pd.DataFrame, scores: dict) -> dict:
    rows = {}
    for name, score in scores.items():
        block = tie_band(frame, score)
        observed = precision_block(frame, score)
        block.update(
            {
                "observed_p_at_k": observed["p_at_k"],
                "unknown_in_queue": observed["unknown"],
                "gap_vs_expected": (
                    None if observed["p_at_k"] is None
                    else float(observed["p_at_k"] - block["expected_p_at_k"])
                ),
            }
        )
        rows[name] = block
    return rows


def d2_contamination(frame: pd.DataFrame, scores: dict) -> dict:
    measurable = frame[TARGET].notna()
    covered = _coverage_ok(frame)
    keep = measurable & covered
    implied = _persistence_label(frame)
    actual = frame[TARGET].eq(1)
    agree = float((implied[keep] == actual[keep]).mean())
    implied_pos = implied[keep]
    precision = float(actual[keep][implied_pos].mean()) if implied_pos.any() else None
    recall = float(implied[keep][actual[keep]].mean()) if actual[keep].any() else None
    days = frame["recent7_days"].replace(0, np.nan)
    rate = frame["future30_impressions"] / (
        (frame["recent7_impressions"] / days * 30.0).replace(0, np.nan)
    )
    rate = rate[measurable & covered]
    out = {
        "rows_checked": int(keep.sum()),
        "agreement_with_persistence": agree,
        "implied_prevalence": float(implied[keep].mean()),
        "actual_prevalence": float(actual[keep].mean()),
        "implied_precision": precision,
        "implied_recall": recall,
        "realised_over_persistence_median": float(rate.median()),
        "realised_over_persistence_p25": float(rate.quantile(0.25)),
        "realised_over_persistence_p75": float(rate.quantile(0.75)),
    }
    if "acceleration_rule" in scores:
        idx = topk_indices(frame, scores["acceleration_rule"])
        q = frame.iloc[idx]
        out["acceleration_queue_implied_share"] = float(
            _persistence_label(q)[_coverage_ok(q)].mean()
        )
        out["acceleration_queue_prevalence"] = float(q[TARGET].mean())
        out["acceleration_queue_unknown"] = int(q[TARGET].isna().sum())
    return out


def d3_momentum_matched(frame: pd.DataFrame, model_scores: dict) -> dict:
    """Pool = the top decile by momentum. Does the model add ordering inside it?"""
    if "acceleration_rule" not in model_scores:
        return {}
    accel = model_scores["acceleration_rule"]
    cut = np.quantile(accel, 0.90)
    pool_mask = accel >= cut  # higher score = lower acceleration
    pool = frame.loc[pool_mask]
    out = {
        "pool_rows": int(pool_mask.sum()),
        "pool_base_rate": float(pool[TARGET].mean()),
        "rule_in_pool": precision_block(frame, accel, mask=pool_mask),
    }
    for name in ("momentum:logistic_regression", "momentum:random_forest",
                 "control:logistic_regression", "control:random_forest"):
        if name in model_scores:
            out[name] = precision_block(frame, model_scores[name], mask=pool_mask)
    return out


def _fit_on_subset(train: pd.DataFrame, test: pd.DataFrame, feature_set: str) -> dict:
    out = {}
    for model_name in study.MODELS:
        known = train.loc[train[TARGET].notna()]
        x_train, x_test, numeric, categorical = study._design_matrices(
            known, test, feature_set
        )
        model = study.build_model(model_name, numeric, categorical)
        model.fit(x_train, known[TARGET].astype(int))
        out[f"{feature_set}:{model_name}"] = model.predict_proba(x_test)[:, 1]
    return out


def d4_onset(train: pd.DataFrame, test: pd.DataFrame, rules_train: dict, rules_test: dict) -> dict:
    """Restrict to pages whose current run rate is healthy (acceleration >= 1.0).

    The label is then no longer implied by momentum, so this is the question a
    model can actually win: which currently healthy page is about to slip?
    """
    healthy_train = rules_train["acceleration_rule"] <= -1.0  # score = -ratio
    healthy_test = rules_test["acceleration_rule"] <= -1.0
    healthy_train = healthy_train & _coverage_ok(train)
    healthy_test = healthy_test & _coverage_ok(test)
    tr, te = train.loc[healthy_train], test.loc[healthy_test].reset_index(drop=True)
    out = {
        "healthy_train_rows": int(healthy_train.sum()),
        "healthy_test_rows": int(healthy_test.sum()),
        "healthy_base_rate": float(te[TARGET].mean()),
    }
    for name in ("static_rule", "acceleration_rule"):
        out[name] = precision_block(
            test, rules_test[name], mask=healthy_test.to_numpy()
        )
    scores = _fit_on_subset(tr, te, "momentum")
    for name, s in scores.items():
        out[name] = precision_block(te, s)
    return out


def _reason_mix(queue: pd.DataFrame) -> dict:
    visible = queue["recent30_impressions"].ge(1000)
    ctr_flag = queue["recent30_avg_position"].between(1, 10) & queue["recent30_ctr_pct"].lt(1)
    stale = queue["content_age_days"].ge(91)
    reason = np.select(
        [visible & ctr_flag & stale, visible & ctr_flag, visible & stale, visible],
        ["stale_low_ctr_top10", "low_ctr_top10", "stale_visible", "visible_only"],
        default="below_visibility_floor",
    )
    return {k: int(v) for k, v in pd.Series(reason).value_counts().items()}


def _acceleration_bands(queue: pd.DataFrame) -> dict:
    days = queue["recent7_days"].replace(0, np.nan)
    ratio = queue["recent7_impressions"] / days * 30.0 / queue["recent30_impressions"].replace(0, np.nan)
    bands = pd.cut(ratio, [0, 0.3, 0.5, 0.8, 1.0, np.inf],
                   labels=["<0.3 collapsed", "0.3-0.5", "0.5-0.8", "0.8-1.0", ">=1.0 healthy"],
                   include_lowest=True)
    return {str(k): int(v) for k, v in bands.value_counts(sort=False).items()}


def d5_queue(frame: pd.DataFrame, scores: dict, cap: int = 5) -> dict:
    rows = {}
    for name, score in scores.items():
        if name not in ("static_rule", "acceleration_rule", "momentum:logistic_regression",
                        "momentum:random_forest"):
            continue
        idx = topk_indices(frame, score)
        q = frame.iloc[idx]
        counts = q["client_hash_id"].value_counts()
        entry = {
            "p_at_k": float(_pos(q[TARGET]).mean()),
            "unknown": int(q[TARGET].isna().sum()),
            "clients": int(counts.size),
            "largest_client_share": float(counts.max() / len(q)),
            "reason_mix": _reason_mix(q),
            "acceleration_bands": _acceleration_bands(q),
        }
        entry["capped"] = _capped_queue(frame, score, cap)
        rows[name] = entry
    # overlaps with the frozen rule
    base = set(frame.iloc[topk_indices(frame, scores["static_rule"])].index)
    for name in rows:
        if name == "static_rule":
            continue
        other = set(frame.iloc[topk_indices(frame, scores[name])].index)
        rows[name]["overlap_with_static_rule"] = len(base & other)
    return rows


def _capped_queue(frame: pd.DataFrame, score: np.ndarray, cap: int) -> dict:
    """Greedy top-k with a per-client cap, to stop one client eating the queue."""
    order = np.lexsort((study._tie_keys(frame), -np.asarray(score, float)))
    used: dict = {}
    picked = []
    for i in order:
        client = frame["client_hash_id"].iloc[i]
        if used.get(client, 0) >= cap:
            continue
        used[client] = used.get(client, 0) + 1
        picked.append(i)
        if len(picked) == K:
            break
    q = frame.iloc[picked]
    return {
        "p_at_k": float(_pos(q[TARGET]).mean()),
        "unknown": int(q[TARGET].isna().sum()),
        "clients": int(q["client_hash_id"].nunique()),
        "largest_client_share": float(q["client_hash_id"].value_counts().max() / len(q)),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_frames(cache: Path) -> dict[str, pd.DataFrame]:
    manifest = cache / "manifest.json"
    if not manifest.exists():
        raise SystemExit(
            f"No cache at {cache}. Run w03_data_contract.ipynb first, or pass "
            "FLYRANK_FINAL_CACHE. Use --selftest to check the script without data."
        )
    config = json.loads(manifest.read_text(encoding="utf-8"))["provenance"]["config"]
    months = config["feature_months"]
    return {m: pd.read_parquet(cache / f"{m}.parquet") for m in months}


def run(frames: dict[str, pd.DataFrame], origins: list[str], out_dir: Path) -> dict:
    results = {}
    for origin in origins:
        if origin not in frames:
            continue
        test = frames[origin]
        mature = study._mature_training_frames(frames, origin, 7).get("latest_only")
        if mature is None:
            continue
        training_months, train = mature
        rules_test = study._rule_scores(test)
        rules_train = study._rule_scores(train)
        models_test = build_model_scores(train, test, "momentum")
        all_scores = {**rules_test, **models_test}

        results[origin] = {
            "n": int(len(test)),
            "base_rate": float(test[TARGET].mean()),
            "training_months": list(training_months),
            "training_rows": int(len(train)),
            "D1_resolution": d1_resolution(test, all_scores),
            "D2_label": d2_contamination(test, all_scores),
            "D3_momentum_matched": d3_momentum_matched(test, all_scores),
            "D4_onset": d4_onset(train, test, rules_train, rules_test),
            "D5_queue": d5_queue(test, all_scores),
        }
        print(f"  scored {origin}", flush=True)

    payload = {
        "k": K,
        "origins": results,
        "reading": {
            "D1": "If top_band_share_of_population is large and observed_p_at_k ~ expected_p_at_k, "
                  "the rule is not ranking; its top-K is a draw from a tie.",
            "D2": "If agreement_with_persistence is high and realised_over_persistence_median ~ 1, "
                  "the label is largely the continuation of the last seven days.",
            "D3": "If a model beats acceleration_rule inside the momentum pool, models add ordering "
                  "on top of momentum; otherwise they only re-derive it.",
            "D4": "The honest ML question. Compare model P@50 with healthy_base_rate and static_rule.",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ranking_diagnosis.json").write_text(
        json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8"
    )
    print_report(payload)
    return payload


def print_report(payload: dict) -> None:
    print("\n=== D1  Is the frozen rule ranking? ===")
    print(f"{'origin':8s} {'candidate':36s} {'top band':>9s} {'% pop':>7s} {'tie@50':>7s} "
          f"{'band':>6s} {'exp P@50':>9s} {'obs P@50':>9s} {'distinct':>9s}")
    for origin, r in payload["origins"].items():
        for name, b in r["D1_resolution"].items():
            print(f"{origin:8s} {name:36s} {b['top_score_band_size']:9d} "
                  f"{100*b['top_score_band_share']:6.1f}% {b['boundary_tie_size']:7d} "
                  f"{b['band_decline_rate']:6.3f} "
                  f"{b['expected_p_at_k']:9.3f} {(b['observed_p_at_k'] or 0):9.3f} "
                  f"{b['distinct_score_values']:9d}")

    print("\n=== D2  Is the label a momentum twin? ===")
    print(f"{'origin':8s} {'base':>6s} {'implied':>8s} {'agree':>7s} {'prec':>6s} {'rec':>6s} "
          f"{'F/persist':>10s} {'queue implied':>14s}")
    for origin, r in payload["origins"].items():
        d = r["D2_label"]
        print(f"{origin:8s} {d['actual_prevalence']:6.3f} {d['implied_prevalence']:8.3f} "
              f"{d['agreement_with_persistence']:7.3f} {d['implied_precision']:6.3f} "
              f"{d['implied_recall']:6.3f} {d['realised_over_persistence_median']:10.3f} "
              f"{d.get('acceleration_queue_implied_share', float('nan')):14.3f}")

    print("\n=== D3  Model lift inside the momentum pool (top decile by acceleration) ===")
    for origin, r in payload["origins"].items():
        d = r["D3_momentum_matched"]
        if not d:
            continue
        print(f"{origin}  pool base {d['pool_base_rate']:.3f}  rule {d['rule_in_pool']['p_at_k']:.3f}")
        for name in ("momentum:logistic_regression", "momentum:random_forest"):
            if name in d:
                print(f"          {name:36s} {d[name]['p_at_k']:.3f}")

    print("\n=== D4  Onset target: healthy pages only (acceleration >= 1.0) ===")
    print(f"{'origin':8s} {'healthy n':>10s} {'base':>6s} {'static':>7s} {'accel':>7s} "
          f"{'LR':>6s} {'RF':>6s}")
    for origin, r in payload["origins"].items():
        d = r["D4_onset"]
        lr = d.get("momentum:logistic_regression", {}).get("p_at_k")
        rf = d.get("momentum:random_forest", {}).get("p_at_k")
        print(f"{origin:8s} {d['healthy_test_rows']:10d} {d['healthy_base_rate']:6.3f} "
              f"{d['static_rule']['p_at_k']:7.3f} {d['acceleration_rule']['p_at_k']:7.3f} "
              f"{(lr if lr is not None else float('nan')):6.3f} "
              f"{(rf if rf is not None else float('nan')):6.3f}")

    print("\n=== D5  Queue shape (top 50) ===")
    for origin, r in payload["origins"].items():
        print(f"{origin}")
        for name, q in r["D5_queue"].items():
            top_reason = max(q["reason_mix"], key=q["reason_mix"].get)
            top_band = max(q["acceleration_bands"], key=q["acceleration_bands"].get)
            print(f"   {name:36s} P@50 {q['p_at_k']:.3f} unknown {q['unknown']:2d} "
                  f"clients {q['clients']:2d} largest {q['largest_client_share']:.2f} "
                  f"| capped P@50 {q['capped']['p_at_k']:.3f} clients {q['capped']['clients']:2d} "
                  f"| top reason {top_reason} | mostly {top_band}")


# --------------------------------------------------------------------------- #
# Synthetic self-test
# --------------------------------------------------------------------------- #
def synthetic_frames(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    months = ["2025-10", "2025-11", "2025-12"]
    frames = {}
    for offset, month in enumerate(months):
        n = 1200
        clients = np.array([f"c{i:02d}" for i in range(20)])
        client = rng.choice(clients, n)
        recent30 = rng.lognormal(7.0, 0.9, n).round()
        momentum = rng.lognormal(0.0, 0.45, n)          # last-7 rate / 30-day rate
        recent7 = (recent30 * momentum * 7 / 30).round()
        future_rate = momentum * rng.lognormal(0, 0.35, n)
        future30 = (recent30 * future_rate).round()
        age = rng.integers(10, 900, n)
        ctr = np.clip(rng.gamma(2, 0.5, n), 0, 20)
        position = rng.integers(0, 40, n).astype(float)
        df = pd.DataFrame(
            {
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
                "prev30_impressions": (recent30 / rng.lognormal(0, 0.2, n)).round(),
                "prior30_days": 30,
                "recent7_impressions": recent7,
                "recent7_days": 7,
                "future30_impressions": future30,
                "future30_days": 30,
                "content_age_days": age,
                "content_created_date": pd.Timestamp("2023-01-01"),
            }
        )
        df["log_recent30_impressions"] = np.log1p(df["recent30_impressions"])
        df[TARGET] = (df["future30_impressions"] < 0.8 * df["recent30_impressions"]).astype(int)
        df.loc[rng.random(n) < 0.03, TARGET] = np.nan
        frames[month] = study._momentum(df)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="run on synthetic frames")
    parser.add_argument("--cache", default=None, help="frame cache directory")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--origins", default=None, help="comma-separated origins")
    args = parser.parse_args()

    if args.selftest:
        frames = synthetic_frames()
        origins = list(frames)[-1:]
        import tempfile
        out = Path(args.out or tempfile.mkdtemp(prefix="flyrank_diag_"))
        run(frames, origins, out)
        print(f"\nSelf-test receipts: {out / 'ranking_diagnosis.json'}")
        return

    cache = Path(args.cache or os.environ.get("FLYRANK_FINAL_CACHE",
                                              ROOT / "work/local_final_pipeline"))
    frames = load_frames(cache)
    origins = args.origins.split(",") if args.origins else sorted(frames)[1:]
    out = Path(args.out or os.environ.get("FLYRANK_DIAG_OUTPUT",
                                         ROOT / "work/outputs/diagnostics"))
    run(frames, origins, out)
    print(f"\nReceipts: {out / 'ranking_diagnosis.json'}")


if __name__ == "__main__":
    main()
