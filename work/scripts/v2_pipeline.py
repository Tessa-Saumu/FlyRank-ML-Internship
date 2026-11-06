"""V2 pipeline: four learned candidates, rolling selection on validation folds, sealed test scored once.

The decisions are pre-registered in work/V2_SPEC.md / work/outputs/v2_spec.json (committed before this code
existed); this module asserts against the machine-readable spec so it cannot silently drift.

Commands (run in this order):
  python v2_pipeline.py select       tune on validation folds 1+2, champion/challenger rule, nested check
  python v2_pipeline.py confirm      confirmation fold (reported only; reads the written selection, cannot change it)
  python v2_pipeline.py secondary    secondary label (champion LOGISTIC only) on validation and confirmation folds
  python v2_checks.py                leakage and split checks (validation folds only)
  python v2_pipeline.py freeze       write v2_freeze.json (chosen model/config, spec and code hashes)
  python v2_pipeline.py --unseal     score the sealed test month ONCE (refuses without a valid freeze)
  python v2_playbook.py              live July queue (refuses until the test has been scored)

Sealed-month protocol: load_frame() drops the outcome columns of the test origin and removes the live origin's
rows unless the unseal gate passes (freeze file present, its hashes match the current files, test not yet scored).
"""
import v2_models as mm            # sets single-thread environment variables before numpy / scikit-learn load

import datetime as dt
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

import erosion_model as v1
import v2_rhythm as rh
from erosion_frame import month_add
from wh import CACHE

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(HERE + "/../outputs")
SPEC = json.load(open(OUT + "/v2_spec.json", encoding="utf-8"))

# ------------------------------------------------------------------ assert the code against the spec
assert SPEC["spec_version"] == "v2.0"
assert SPEC["seed"] == mm.SEED == v1.SEED == 42
assert SPEC["candidates"] == ["LOGISTIC", "HGB_CLF", "POISSON_RATE", "ENSEMBLE"]
assert SPEC["models"]["LOGISTIC"]["C"] == mm.LOGISTIC_C and SPEC["models"]["LOGISTIC"]["n_configs"] == len(mm.logistic_grid())
assert SPEC["models"]["LOGISTIC"]["penalties"] == mm.PENALTY_L1_RATIO
assert SPEC["models"]["LOGISTIC"]["max_iter"] == mm.MAX_ITER
_h = SPEC["models"]["HGB_CLF"]
assert (_h["max_leaf_nodes"], _h["min_samples_leaf_fraction"], _h["l2_regularization"], _h["max_features"]) == \
    (mm.HGB_LEAVES, mm.HGB_MSL_FRAC, mm.HGB_L2, mm.HGB_MAXF)
assert _h["max_iter_cap"] == mm.HGB_ITER_CAP and _h["seeds"] == mm.HGB_SEEDS and _h["sampler_seed"] == mm.SEED
assert _h["min_samples_leaf_floor"] == mm.HGB_MSL_FLOOR and _h["grouped_holdout_row_share_range"] == list(mm.GROUPED_HOLDOUT_SHARE)
assert SPEC["baselines"] == list(v1.BASELINES)
assert SPEC["metrics"]["headline_k"] == v1.K == 50
B = SPEC["selection"]["bootstrap_B"]
assert SPEC["selection"]["n_challenger_comparisons"] == len(SPEC["challengers"]) == 3
FOLDS = SPEC["folds"]
SEALED = SPEC["origins"]["sealed_test"]
LIVE = SPEC["origins"]["live"]
OUTCOME_COLS = SPEC["sealed_protocol"]["drop_outcome_columns"]
PREV_COLS = ["page_prev_erode", "page_prev_ratio", "client_prev_erode_rate"]
REMOVED = SPEC["features"]["removed"]
ADDED = SPEC["features"]["added"]
assert ADDED == rh.NEW_FEATURES
FEATURE_LIST = [f for f in v1.FEATURES if f not in REMOVED] + ADDED
assert len(FEATURE_LIST) == SPEC["features"]["n_total"] == 36
assert not (set(FEATURE_LIST) & (set(OUTCOME_COLS) | set(PREV_COLS)))

CODE_FILES = ["wh.py", "erosion_frame.py", "erosion_model.py", "v2_rhythm.py", "v2_models.py", "v2_pipeline.py",
              "v2_checks.py", "v2_playbook.py"]
SPEC_FILES = {"work/V2_SPEC.md": HERE + "/../V2_SPEC.md", "work/outputs/v2_spec.json": OUT + "/v2_spec.json"}
PRE_FREEZE_RECEIPTS = ["v2_selection.json", "v2_validation_folds.json", "v2_confirmation_apr.json", "v2_leakage_checks.json"]
FAMILY_ORDER = ["LOGISTIC", "HGB_CLF", "POISSON_RATE"]
CAP = SPEC["playbook"]["client_cap"]


def sha_norm(path):
    """SHA-256 of the file with CRLF normalised to LF (git autocrlf on Windows rewrites line endings)."""
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def now():
    return dt.datetime.now().isoformat(timespec="seconds")


def dump(obj, name):
    with open(f"{OUT}/{name}", "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=_jd)
        f.write("\n")


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (dt.date, pd.Timestamp)):
        return str(o)
    raise TypeError(type(o))


def rnd(v, k=5):
    return [round(x, k) for x in v] if isinstance(v, (list, tuple)) else round(v, k)


def first_day(m):
    return rh.month_bounds(m)[0]


def last_day(m):
    return rh.month_bounds(m)[1]


# ------------------------------------------------------------------ freeze gate and sealed loading
def freeze_problems():
    p = OUT + "/v2_freeze.json"
    if not os.path.exists(p):
        return None, ["v2_freeze.json does not exist"]
    fz = json.load(open(p, encoding="utf-8"))
    bad = []
    for f, h in fz["code_sha256"].items():
        if sha_norm(f"{HERE}/{f}") != h:
            bad.append(f"code changed since freeze: {f}")
    for f, h in fz["spec_sha256"].items():
        if sha_norm(f"{HERE}/../../{f}") != h:
            bad.append(f"spec changed since freeze: {f}")
    for f, h in fz["receipt_sha256"].items():
        if sha_norm(f"{OUT}/{f}") != h:
            bad.append(f"receipt changed since freeze: {f}")
    return fz, bad


def gate(kind):
    """kind 'test': freeze valid and the test not yet scored. kind 'live': freeze valid and the test already scored."""
    fz, bad = freeze_problems()
    if bad:
        raise PermissionError("sealed-month gate closed: " + "; ".join(bad))
    done = os.path.exists(OUT + "/v2_test.json")
    if kind == "test" and done:
        raise PermissionError("the sealed test month was already scored once; v2_test.json exists; refusing to run again")
    if kind == "live" and not done:
        raise PermissionError("the live step needs the test to have been scored (v2_test.json missing)")
    return fz


def load_frame(mode="sealed"):
    """mode 'sealed': outcome columns of the test origin dropped, live origin removed (default for every step).
    mode 'test': test-origin outcomes present, live origin still removed (needs the unseal gate).
    mode 'live': everything (needs the live gate)."""
    path = CACHE + "/erosion_frame.parquet"
    names = pq.ParquetFile(path).schema.names
    if mode == "test":
        gate("test")
    elif mode == "live":
        gate("live")
    if mode == "sealed":
        a = pq.read_table(path, filters=[("origin", "not in", [SEALED, LIVE])]).to_pandas()
        safe = [c for c in names if c not in OUTCOME_COLS]
        b = pq.read_table(path, columns=safe, filters=[("origin", "==", SEALED)]).to_pandas()
        df = pd.concat([a, b], ignore_index=True)
        assert not [c for c in OUTCOME_COLS if c in b.columns]
        assert LIVE not in set(df.origin)
        for c in OUTCOME_COLS:
            assert df.loc[df.origin == SEALED, c].isna().all(), f"sealed outcome column {c} is populated"
    elif mode == "test":
        df = pq.read_table(path, filters=[("origin", "!=", LIVE)]).to_pandas()
    else:
        df = pq.read_table(path).to_pandas()
    rc = json.load(open(OUT + "/erosion_frame_summary.json"))["reference_curve"]
    return df, np.array(rc["pos"]), np.array(rc["ctr"])


class Data:
    """Frame plus V2 features and the row-aligned arrays every step uses."""

    def __init__(self, mode="sealed"):
        df, grid, vals = load_frame(mode)
        self.mode = mode
        self.df = df.reset_index(drop=True)
        d = self.df
        d["measurable"] = d["measurable"].eq(True) if "measurable" in d else False
        self.meas = d["measurable"].to_numpy(bool)
        X1 = v1.make_features(d, grid, vals)                    # the 34 V1 columns (kept for the frozen baselines)
        self.X1 = X1
        X = X1.drop(columns=REMOVED)
        X.loc[d.origin == "2025-12", "imp_growth_prev"] = np.nan  # its o-3 month straddles the num=100 change
        R, self.rhythm_diag = rh.rhythm_features(d, allow_sealed_months=(mode == "live"))
        self.X = pd.concat([X, R[ADDED]], axis=1)
        assert list(self.X.columns) == FEATURE_LIST
        self.origin = d.origin.to_numpy()
        self.g = d.client_hash_id.to_numpy()
        self.tb = v1.tiebreak(d.content_hash_id)
        self.y = d["erode"].fillna(0).to_numpy(float) if "erode" in d else np.zeros(len(d))
        exp = d["exp_out"].to_numpy(float) if "exp_out" in d else np.full(len(d), np.nan)
        clk = d["clk_out"].fillna(0).to_numpy(float) if "clk_out" in d else np.full(len(d), np.nan)
        self.exp = exp
        self.ratio = np.where(self.meas, clk / np.where(exp > 0, exp, 1.0), np.nan)
        self.shortfall = np.where(self.meas, np.maximum(0.0, exp - clk), 0.0)
        self.null = d["null_erode"].to_numpy(float) if "null_erode" in d else np.full(len(d), np.nan)
        lr = d["log_ratio_out"].to_numpy(float) if "log_ratio_out" in d else np.full(len(d), np.nan)
        self.log_ratio = lr
        # client-relative secondary label (needs outcomes; NaN where the outcome is not loaded)
        q = pd.Series(lr).where(self.meas)
        med = q.groupby([d.origin, d.client_hash_id]).transform("median")
        self.erode_rel = np.where(self.meas, ((q - med) <= np.log(0.75)).astype(float), np.nan)

    def idx(self, origins):
        return np.flatnonzero(np.isin(self.origin, origins))


def assert_timeline(train, score):
    """Every training origin's outcome window (months t, t+1) ends before the scored origin's first day."""
    for t in train:
        end = last_day(month_add(t, 1))
        assert end < first_day(score), f"training origin {t} outcome window ends {end}, not before {score}"
        assert t <= month_add(score, -2)
    assert score not in train


# ------------------------------------------------------------------ metrics
def ap_fast(y_sorted):
    tp = np.cumsum(y_sorted)
    n = tp[-1]
    return float((y_sorted * tp / np.arange(1, len(y_sorted) + 1)).sum() / n) if n > 0 else float("nan")


def ap_of(y, s):
    return ap_fast(y[np.argsort(-s, kind="stable")])


def capped_top_idx(order, g, k=50, cap=CAP):
    taken, out = {}, []
    for i in order:
        c = g[i]
        if taken.get(c, 0) < cap:
            out.append(i)
            taken[c] = taken.get(c, 0) + 1
        if len(out) == k:
            break
    return np.array(out)


def rank_pct(s):
    return pd.Series(s).rank(pct=True).to_numpy()


def point_metrics(ym, sm, gm, short_m, base_rate):
    """Metrics on the measurable rows. sm already includes the tiebreak."""
    order = np.argsort(-sm, kind="stable")
    top = order[:50]
    cnt = pd.Series(gm[top]).value_counts()
    p50 = float(ym[top].mean())
    ct = capped_top_idx(order, gm)
    r = {"p@20": float(ym[order[:20]].mean()), "p@50": p50, "p@100": float(ym[order[:100]].mean()),
         "ap": float(average_precision_score(ym, sm)), "auc": float(roc_auc_score(ym, sm)),
         "within_client_p@10": v1.within_client_p(ym, sm, gm),
         "lift": p50 / base_rate if base_rate > 0 else float("nan"),
         "clicks_lost_captured": float(short_m[top].sum() / short_m.sum()) if short_m.sum() > 0 else float("nan"),
         "top_client_share_top50": float(cnt.iloc[0] / len(top)), "clients_in_top50": int(len(cnt)),
         "capped_p@50_descriptive": float(ym[ct].mean())}
    return r


def boot_models(ym, gm, S, ref, B=B, seed=mm.SEED):
    """Client-cluster bootstrap CIs of P@50 and AP for every score in S and of their gap to S[ref]."""
    rng = np.random.default_rng(seed)
    clients = np.unique(gm)
    idx_by = {c: np.flatnonzero(gm == c) for c in clients}
    p50 = {n: np.empty(B) for n in S}
    ap = {n: np.empty(B) for n in S}
    for b in range(B):
        idx = np.concatenate([idx_by[c] for c in rng.choice(clients, len(clients))])
        yb = ym[idx]
        for n, s in S.items():
            yy = yb[np.argsort(-s[idx], kind="stable")]
            p50[n][b] = yy[:50].mean()
            ap[n][b] = ap_fast(yy)
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]
    out = {}
    for n in S:
        d_ap, d_p = ap[n] - ap[ref], p50[n] - p50[ref]
        out[n] = {"p@50_ci": ci(p50[n]), "ap_ci": ci(ap[n]), "delta_ap_vs_best_baseline_ci": ci(d_ap),
                  "delta_p@50_vs_best_baseline_ci": ci(d_p), "p_delta_ap_le_0": round(float(np.mean(d_ap <= 0)), 4)}
    return out


def loco_deltas(ym, gm, S, ref):
    out = {}
    for n, s in S.items():
        vals = []
        for c in np.unique(gm):
            m = gm != c
            vals.append(ap_of(ym[m], s[m]) - ap_of(ym[m], S[ref][m]))
        out[n] = {"min": round(float(np.min(vals)), 4), "max": round(float(np.max(vals)), 4)}
    return out


def evaluate_set(D, te_idx, scores, train_origins=None, with_ci=True, ref=None):
    """Full metric block for one scored origin. scores: name -> array over ALL eligible rows of that origin (or None)."""
    meas = D.meas[te_idx]
    ym, gm, tbm = D.y[te_idx][meas], D.g[te_idx][meas], D.tb[te_idx][meas]
    short_m = D.shortfall[te_idx][meas]
    base = float(ym.mean())
    counts = pd.Series(gm).value_counts()
    share = counts / counts.sum()
    res = {"n_eligible": int(len(te_idx)), "n_measurable": int(meas.sum()),
           "lost_visibility_share": round(1 - float(meas.mean()), 4), "n_clients": int(len(counts)),
           "effective_clients": round(float(1 / (share ** 2).sum()), 2),
           "top3_client_row_share": round(float(share.iloc[:3].sum()), 4),
           "base_rate": round(base, 4), "noise_only_rate": round(float(np.nanmean(D.null[te_idx][meas])), 4)}
    if train_origins is not None:
        res["train_origins"] = list(train_origins)
    avail = {n: s for n, s in scores.items() if s is not None and not np.isnan(s).any()}
    S_m = {n: s[meas] + tbm for n, s in avail.items()}
    models = {}
    for n in scores:
        if n not in avail:
            models[n] = {"available": False}
            continue
        m = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in point_metrics(ym, S_m[n], gm, short_m, base).items()}
        full = avail[n] + D.tb[te_idx]
        m["p@50_unmeasurable_as_miss"] = round(float(D.y[te_idx][np.argsort(-full, kind="stable")[:50]].mean()), 4)
        models[n] = m
    bases = [n for n in v1.BASELINES if n in avail]
    best = max(bases, key=lambda n: models[n]["ap"]) if bases else None
    res["best_baseline_by_ap"] = best
    if with_ci and best is not None:
        ci = boot_models(ym, gm, S_m, best if ref is None else ref)
        lo = loco_deltas(ym, gm, S_m, best)
        for n in avail:
            models[n].update(ci[n])
            models[n]["loco_delta_ap_vs_best_baseline"] = lo[n]
            models[n]["delta_ap_vs_best_baseline"] = round(models[n]["ap"] - models[best]["ap"], 4)
    res["models"] = models
    return res


def baseline_scores(D, te_idx):
    out = {}
    X1 = D.X1.iloc[te_idx]
    for n, f in v1.BASELINES.items():
        if n == "persistence_prev_ratio" and X1.page_prev_ratio.notna().sum() == 0:
            out[n] = None                      # reported as unavailable, never as zero
        else:
            out[n] = np.asarray(f(X1), float)
    return out


# ------------------------------------------------------------------ fold preparation and fitting
def fold_payload(D, fold):
    f = FOLDS[fold]
    train, score = f["train"], f["score"]
    assert_timeline(train, score)
    trm = np.isin(D.origin, train) & D.meas
    tr = np.flatnonzero(trm)
    te = np.flatnonzero(D.origin == score)
    Xall_tr = D.X.iloc[tr]
    cols = [c for c in D.X.columns if Xall_tr[c].notna().any()]
    Xtr, Xte = Xall_tr[cols].reset_index(drop=True), D.X.iloc[te][cols].reset_index(drop=True)
    if len(train) >= 2:
        val_mask = D.origin[tr] == train[-1]
        hold = {"kind": "time_ordered_latest_training_origin", "validation_origin": train[-1]}
    else:
        val_mask, info = mm.grouped_holdout_mask(D.g[tr])
        hold = {"kind": "client_grouped_20pct_of_clients", **info}
    return {"fold": fold, "train": train, "score": score, "tr": tr, "te": te, "cols": cols,
            "dropped_all_missing": [c for c in D.X.columns if c not in cols], "Xtr": Xtr, "Xte": Xte,
            "ytr": D.y[tr].astype(int), "rtr": D.ratio[tr], "wtr": D.exp[tr], "val_mask": val_mask, "holdout": hold,
            "gtr": D.g[tr]}


def fit_score_family(family, cfg, P):
    """Fit one family on the fold's training rows with a config; scores over all eligible rows of the scored origin."""
    if family == "LOGISTIC":
        m, info = mm.fit_logistic(P["Xtr"], P["ytr"], cfg)
        return mm.score_logistic(m, P["Xte"]), info
    if family == "HGB_CLF":
        s, info = mm.fit_boosted("clf", cfg, P["Xtr"], P["ytr"], None, P["val_mask"], P["Xte"])
        return s, info
    if family == "POISSON_RATE":
        s, info = mm.fit_boosted("reg", cfg, P["Xtr"], P["rtr"], P["wtr"], P["val_mask"], P["Xte"])
        return -s, info                       # lower predicted ratio = more erosion
    raise KeyError(family)


def ensemble_score(s_log, s_hgb):
    return 0.5 * (rank_pct(s_log) + rank_pct(s_hgb))


# ------------------------------------------------------------------ paired champion/challenger test
def paired_gain(Dsets, S_champ, S_chal, B=B, seed=mm.SEED):
    """Dsets: list of (ym, gm) per fold; S_champ: list of score arrays per fold; S_chal: {name: list per fold}.
    Returns per challenger: mean gain (AP chal - AP champ averaged over folds), bootstrap 95 percent interval,
    leave-one-client-out result."""
    rng = np.random.default_rng(seed)
    nf = len(Dsets)
    clients = [np.unique(g) for _, g in Dsets]
    idx_by = [{c: np.flatnonzero(g == c) for c in cl} for (_, g), cl in zip(Dsets, clients)]
    names = list(S_chal)
    boot = {n: np.empty(B) for n in names}
    for b in range(B):
        d = {n: 0.0 for n in names}
        for f in range(nf):
            idx = np.concatenate([idx_by[f][c] for c in rng.choice(clients[f], len(clients[f]))])
            yb = Dsets[f][0][idx]
            a0 = ap_fast(yb[np.argsort(-S_champ[f][idx], kind="stable")])
            for n in names:
                d[n] += (ap_fast(yb[np.argsort(-S_chal[n][f][idx], kind="stable")]) - a0) / nf
        for n in names:
            boot[n][b] = d[n]
    allc = np.unique(np.concatenate(clients))
    out = {}
    for n in names:
        gains = []
        for f in range(nf):
            y, g = Dsets[f]
            gains.append(ap_of(y, S_chal[n][f]) - ap_of(y, S_champ[f]))
        mean_gain = float(np.mean(gains))
        loo = []
        for c in allc:
            gs = []
            for f in range(nf):
                y, g = Dsets[f]
                m = g != c
                gs.append(ap_of(y[m], S_chal[n][f][m]) - ap_of(y[m], S_champ[f][m]))
            loo.append(float(np.mean(gs)))
        lb, ub = float(np.percentile(boot[n], 2.5)), float(np.percentile(boot[n], 97.5))
        out[n] = {"per_fold_gain": [round(x, 4) for x in gains], "mean_gain": round(mean_gain, 4),
                  "boot_95_ci": [round(lb, 4), round(ub, 4)], "lower_bound": round(lb, 4),
                  "bonferroni_3_lower_bound_info_only": round(float(np.percentile(boot[n], 100 * 0.025 / 3)), 4),
                  "p_gain_le_0": round(float(np.mean(boot[n] <= 0)), 4),
                  "loco_min_mean_gain": round(float(min(loo)), 4), "loco_max_mean_gain": round(float(max(loo)), 4),
                  "loco_clients_with_gain_le_0": int(sum(x <= 0 for x in loo)), "loco_n_clients": int(len(loo)),
                  "passes": bool(lb > 0 and min(loo) > 0)}
    return out


def decide(res):
    passing = [n for n, r in res.items() if r["passes"]]
    if not passing:
        return "LOGISTIC", passing
    return max(passing, key=lambda n: res[n]["mean_gain"]), passing


# ------------------------------------------------------------------ tie-aware best config
def logistic_rank_key(cfg):
    return (cfg["C"], {"l1": 0, "elasticnet": 1, "l2": 2}[cfg["penalty"]])


def hgb_rank_key(cfg):
    return (cfg["max_leaf_nodes"], -cfg["msl_frac"], -cfg["l2"], cfg["learning_rate"])


def best_config(rows, family):
    """rows: list of dicts with cfg, mean_ap, in grid order. Ties (within tolerance) go to the simpler config."""
    tol = SPEC["selection"]["tie_tolerance_ap"]
    top = max(r["mean_ap"] for r in rows)
    tied = [i for i, r in enumerate(rows) if r["mean_ap"] >= top - tol]
    kf = logistic_rank_key if family == "LOGISTIC" else hgb_rank_key
    return min(tied, key=lambda i: (kf(rows[i]["cfg"]), i)), len(tied)


# ------------------------------------------------------------------ select
def tune_all(D, fold_names, tag=""):
    """Score every config of every family on the given folds. Returns payloads, per-config scores and rows."""
    P = {f: fold_payload(D, f) for f in fold_names}
    grids = {"LOGISTIC": mm.logistic_grid(), "HGB_CLF": mm.sample_hgb_configs(), "POISSON_RATE": mm.sample_hgb_configs()}
    scores = {fam: [dict() for _ in grids[fam]] for fam in FAMILY_ORDER}
    infos = {fam: [dict() for _ in grids[fam]] for fam in FAMILY_ORDER}
    t0 = time.time()
    for ci, cfg in enumerate(grids["LOGISTIC"]):
        for f in fold_names:
            s, info = fit_score_family("LOGISTIC", cfg, P[f])
            scores["LOGISTIC"][ci][f], infos["LOGISTIC"][ci][f] = s, info
    print("logistic grid done", round(time.time() - t0, 1), "s", flush=True)
    for fam, kind in (("HGB_CLF", "clf"), ("POISSON_RATE", "reg")):
        def payload(f):
            p = P[f]
            y = p["ytr"] if kind == "clf" else p["rtr"]
            w = None if kind == "clf" else p["wtr"]
            return dict(Xtr=p["Xtr"], ytr=y, wtr=w, val_mask=p["val_mask"], Xte=p["Xte"])
        pl = [payload(f) for f in fold_names]
        res = Parallel(n_jobs=min(10, os.cpu_count() or 4), backend="loky")(
            delayed(mm.tune_boosted_config)(kind, cfg, pl) for cfg in grids[fam])
        for ci, r in enumerate(res):
            for j, f in enumerate(fold_names):
                s, info = r[j]
                scores[fam][ci][f] = s if kind == "clf" else -s
                infos[fam][ci][f] = info
        print(fam, "done", round(time.time() - t0, 1), "s", flush=True)
    return P, grids, scores, infos


def config_rows(D, P, grids, scores, infos, fold_names):
    rows = {}
    for fam in FAMILY_ORDER:
        rr = []
        for ci, cfg in enumerate(grids[fam]):
            aps, p50s = [], []
            for f in fold_names:
                te = P[f]["te"]
                meas = D.meas[te]
                ym, sm = D.y[te][meas], scores[fam][ci][f][meas] + D.tb[te][meas]
                aps.append(float(average_precision_score(ym, sm)))
                p50s.append(float(ym[np.argsort(-sm, kind="stable")[:50]].mean()))
            row = {"key": mm.logistic_key(cfg) if fam == "LOGISTIC" else mm.hgb_key(cfg), "cfg": cfg,
                   "fold_ap": aps, "mean_ap": float(np.mean(aps)), "fold_p@50": p50s}
            inf = infos[fam][ci]
            if fam == "LOGISTIC":
                row["non_converged_folds"] = [f for f in fold_names if not inf[f]["converged"]]
                row["n_iter"] = [inf[f]["n_iter"] for f in fold_names]
            else:
                row["median_best_iter"] = [int(np.median(inf[f]["best_iters"])) for f in fold_names]
                row["fits_at_iteration_cap"] = [inf[f]["hit_cap"] for f in fold_names]
                row["seeds_fitted"] = [inf[f]["n_seeds_fitted"] for f in fold_names]
            rr.append(row)
        rows[fam] = rr
    return rows


def tuned_fold_scores(P, scores, best_idx, fold_names):
    out = {}
    for f in fold_names:
        s = {fam: scores[fam][best_idx[fam]][f] for fam in FAMILY_ORDER}
        s["ENSEMBLE"] = ensemble_score(s["LOGISTIC"], s["HGB_CLF"])
        out[f] = s
    return out


def champion_challenger(D, P, tuned, fold_names):
    Dsets, champ, chal = [], [], {n: [] for n in SPEC["challengers"]}
    for f in fold_names:
        te = P[f]["te"]
        meas = D.meas[te]
        tbm = D.tb[te][meas]
        Dsets.append((D.y[te][meas], D.g[te][meas]))
        champ.append(tuned[f]["LOGISTIC"][meas] + tbm)
        for n in chal:
            chal[n].append(tuned[f][n][meas] + tbm)
    res = paired_gain(Dsets, champ, chal)
    sel, passing = decide(res)
    return res, sel, passing


def run_select():
    t0 = time.time()
    D = Data("sealed")
    fold_names = ["validation_1", "validation_2"]
    P, grids, scores, infos = tune_all(D, fold_names)
    rows = config_rows(D, P, grids, scores, infos, fold_names)
    best_idx, tie_counts = {}, {}
    for fam in FAMILY_ORDER:
        best_idx[fam], tie_counts[fam] = best_config(rows[fam], fam)
    tuned = tuned_fold_scores(P, scores, best_idx, fold_names)
    res, selected, passing = champion_challenger(D, P, tuned, fold_names)
    print("selection:", selected, {n: (r["mean_gain"], r["lower_bound"], r["loco_min_mean_gain"]) for n, r in res.items()}, flush=True)

    # ---- nested time-ordered check: tune and decide on fold 1 only, score on fold 2
    n_best, n_tie = {}, {}
    rows1 = {fam: [dict(r, mean_ap=r["fold_ap"][0]) for r in rows[fam]] for fam in FAMILY_ORDER}
    for fam in FAMILY_ORDER:
        n_best[fam], n_tie[fam] = best_config(rows1[fam], fam)
    tuned1 = tuned_fold_scores(P, scores, n_best, ["validation_1"])
    res1, sel1, pass1 = champion_challenger(D, P, tuned1, ["validation_1"])
    f2 = "validation_2"
    nested_sc = {fam: scores[fam][n_best[fam]][f2] for fam in FAMILY_ORDER}
    nested_sc["ENSEMBLE"] = ensemble_score(nested_sc["LOGISTIC"], nested_sc["HGB_CLF"])
    bs = baseline_scores(D, P[f2]["te"])
    ev = evaluate_set(D, P[f2]["te"], {**bs, "NESTED_SELECTED": nested_sc[sel1], "NESTED_CHAMPION_LOGISTIC": nested_sc["LOGISTIC"],
                                       "SELECTED_BOTH_FOLDS_INSAMPLE": tuned[f2][selected]},
                      with_ci=True)
    oracle = {fam: round(max(r["fold_ap"][1] for r in rows[fam]), 4) for fam in FAMILY_ORDER}
    nested = {"procedure": "tune and decide on validation fold 1 only; refit the chosen config on fold 2's training rows and score fold 2",
              "selected_on_fold_1": sel1, "challenger_results_fold_1": res1,
              "configs_chosen_on_fold_1": {fam: rows[fam][n_best[fam]]["cfg"] for fam in FAMILY_ORDER},
              "fold_2_evaluation": ev,
              "fold_2_best_possible_config_ap_per_family_info_only": oracle}

    sel = {"spec_version": SPEC["spec_version"], "written_at": now(), "criterion": "mean AP over validation folds 1 and 2",
           "folds": {f: {"train": FOLDS[f]["train"], "score": FOLDS[f]["score"], "outcome_window": FOLDS[f]["outcome_window"]} for f in fold_names},
           "n_features": len(FEATURE_LIST),
           "dropped_all_missing_in_training": {f: P[f]["dropped_all_missing"] for f in fold_names},
           "early_stopping_block": {f: P[f]["holdout"] for f in fold_names},
           "configs": {fam: [{k: (rnd(v) if k in ("mean_ap", "fold_ap", "fold_p@50") else v) for k, v in r.items()} for r in rows[fam]] for fam in FAMILY_ORDER},
           "best_config_per_family": {fam: {"key": rows[fam][best_idx[fam]]["key"], "cfg": rows[fam][best_idx[fam]]["cfg"],
                                            "mean_ap": round(rows[fam][best_idx[fam]]["mean_ap"], 4),
                                            "n_configs_tied_within_tolerance": tie_counts[fam]} for fam in FAMILY_ORDER},
           "champion": "LOGISTIC", "n_challenger_comparisons": SPEC["selection"]["n_challenger_comparisons"],
           "challenger_vs_champion": res, "challengers_passing": passing, "selected_model": selected,
           "decision": (f"{selected} replaces the LOGISTIC champion" if selected != "LOGISTIC"
                        else "no challenger passed both the bootstrap lower bound and the leave-one-client-out test; LOGISTIC stays"),
           "nested_time_ordered_check": nested,
           "non_converged_logistic_fits": [r["key"] for r in rows["LOGISTIC"] if r["non_converged_folds"]],
           "runtime_seconds": round(time.time() - t0)}
    dump(sel, "v2_selection.json")

    # ---- validation-fold receipt: all candidates (tuned on these very folds) and baselines
    vf = {"spec_version": SPEC["spec_version"], "written_at": now(),
          "note": "Tuned on these two folds: the model numbers are optimistic by construction. Baselines have no tuning.",
          "folds": {}}
    for f in fold_names:
        te = P[f]["te"]
        sc = {**baseline_scores(D, te), **tuned[f]}
        vf["folds"][f] = evaluate_set(D, te, sc, train_origins=FOLDS[f]["train"])
        vf["folds"][f]["score_origin"] = FOLDS[f]["score"]
        print(f, "base", vf["folds"][f]["base_rate"], {n: m.get("p@50") for n, m in vf["folds"][f]["models"].items()}, flush=True)
    pooled = {}
    for n in vf["folds"][fold_names[0]]["models"]:
        ms = [vf["folds"][f]["models"][n] for f in fold_names]
        if all(m.get("available", True) for m in ms):
            pooled[n] = {k: round(float(np.mean([m[k] for m in ms])), 4) for k in ("ap", "p@50", "p@20", "p@100", "auc")}
    vf["mean_over_two_folds"] = pooled
    dump(vf, "v2_validation_folds.json")
    print("select done", round(time.time() - t0), "s", flush=True)


# ------------------------------------------------------------------ confirmation
def selected_cfgs():
    sel = json.load(open(OUT + "/v2_selection.json"))
    return sel, {fam: sel["best_config_per_family"][fam]["cfg"] for fam in FAMILY_ORDER}


def fit_all_candidates(P, cfgs):
    s, info = {}, {}
    for fam in FAMILY_ORDER:
        s[fam], info[fam] = fit_score_family(fam, cfgs[fam], P)
    s["ENSEMBLE"] = ensemble_score(s["LOGISTIC"], s["HGB_CLF"])
    return s, info


def run_confirm():
    t0 = time.time()
    sel, cfgs = selected_cfgs()                       # the decision is already on disk and is not touched here
    D = Data("sealed")
    P = fold_payload(D, "confirmation")
    s, info = fit_all_candidates(P, cfgs)
    ev = evaluate_set(D, P["te"], {**baseline_scores(D, P["te"]), **s}, train_origins=FOLDS["confirmation"]["train"])
    rec = {"spec_version": SPEC["spec_version"], "written_at": now(),
           "note": "REPORTED ONLY, not used for any decision. Origin 2026-04, outcome window Apr-May 2026 overlaps the test window. "
                   "Configs are those chosen on validation folds 1 and 2; trees and fits are redone on this fold's training rows.",
           "selected_model_from_validation": sel["selected_model"], "configs_used": cfgs, "score_origin": FOLDS["confirmation"]["score"],
           "dropped_all_missing_in_training": P["dropped_all_missing"], "early_stopping_block": P["holdout"],
           "evaluation": ev,
           "fit_info": {fam: ({k: v for k, v in i.items()} if fam != "LOGISTIC" else i) for fam, i in info.items()}}
    dump(rec, "v2_confirmation_apr.json")
    print("confirmation", ev["base_rate"], {n: m.get("p@50") for n, m in ev["models"].items()}, round(time.time() - t0), "s", flush=True)


# ------------------------------------------------------------------ secondary label (champion LOGISTIC only)
def secondary_fold(D, fold, cfg):
    f = FOLDS[fold]
    train, score = f["train"], f["score"]
    assert_timeline(train, score)
    tr = np.flatnonzero(np.isin(D.origin, train) & D.meas)
    te = np.flatnonzero(D.origin == score)
    Xall_tr = D.X.iloc[tr]
    cols = [c for c in D.X.columns if Xall_tr[c].notna().any()]
    Xtr, Xte = Xall_tr[cols], D.X.iloc[te][cols]
    m1, _ = mm.fit_logistic(Xtr, D.y[tr].astype(int), cfg)
    m2, info2 = mm.fit_logistic(Xtr, D.erode_rel[tr].astype(int), cfg)
    s1, s2 = mm.score_logistic(m1, Xte), mm.score_logistic(m2, Xte)
    meas = D.meas[te]
    tbm = D.tb[te][meas]
    yr, gm = np.nan_to_num(D.erode_rel[te][meas]), D.g[te][meas]
    a, b = s1[meas] + tbm, s2[meas] + tbm
    top_p, top_s = np.argsort(-a, kind="stable")[:50], np.argsort(-b, kind="stable")[:50]
    cnt = pd.Series(gm[top_s]).value_counts()
    return {"score_origin": score, "label": "erode_rel (SECONDARY, client-relative)", "base_rate_secondary": round(float(yr.mean()), 4),
            "base_rate_primary": round(float(D.y[te][meas].mean()), 4),
            "ap": round(float(average_precision_score(yr, b)), 4), "p@50": round(float(yr[top_s].mean()), 4),
            "top_client_share_top50": round(float(cnt.iloc[0] / 50), 4), "clients_in_top50": int(len(cnt)),
            "overlap_top50_with_primary_top50": int(len(set(top_p) & set(top_s))),
            "primary_logistic_ap_on_primary_label": round(float(average_precision_score(D.y[te][meas], a)), 4),
            "primary_logistic_p@50_on_primary_label": round(float(D.y[te][meas][top_p].mean()), 4),
            "primary_top50_p@50_on_secondary_label": round(float(yr[top_p].mean()), 4),
            "secondary_fit_converged": info2["converged"]}


def run_secondary():
    sel, cfgs = selected_cfgs()
    D = Data("sealed")
    rec = {"spec_version": SPEC["spec_version"], "written_at": now(),
           "label_status": "SECONDARY. Never used for selection. Champion LOGISTIC only, with the config selected for the primary label.",
           "logistic_config": cfgs["LOGISTIC"], "folds": {}}
    for fold in ("validation_1", "validation_2", "confirmation"):
        rec["folds"][fold] = secondary_fold(D, fold, cfgs["LOGISTIC"])
        print("secondary", fold, {k: v for k, v in rec["folds"][fold].items() if k in ("ap", "p@50", "overlap_top50_with_primary_top50")}, flush=True)
    rec["test"] = "filled at the unseal step"
    dump(rec, "v2_secondary_label.json")


# ------------------------------------------------------------------ features receipt
FAMILY = {}
for _f in ["l_imp_f", "l_clk_f", "imp_growth", "ctr_growth", "imp_growth_prev", "imp_l14_share", "ctr_recency", "click_day_share", "spikiness"]:
    FAMILY[_f] = "exposure and trajectory"
for _f in ["pos_f", "pos_delta", "pos_std"]:
    FAMILY[_f] = "rank"
for _f in ["ctr_vs_ref", "ctr_vs_client"]:
    FAMILY[_f] = "CTR vs expectation"
for _f in ["l_age", "months_active", "keyword_token_count", "l_search_volume", "cpc", "competition", "is_feedly"] + [f"intent_{i}" for i in v1.INTENTS]:
    FAMILY[_f] = "content"
for _f in ["has_ga4", "l_org_sess", "has_ai_referral"]:
    FAMILY[_f] = "channels"
for _f in ["c_ctr_growth", "c_l_imp", "c_l_pages"]:
    FAMILY[_f] = "client context"
for _f in ADDED:
    FAMILY[_f] = "date and rhythm (new in V2)"
MEANING_NEW = {
    "weekday_share": "share of window impressions on Monday to Friday",
    "days_since_last_click": "days from the last click day in the feature window to the origin's first day",
    "longest_zero_click_run": "longest run of consecutive no-click days inside one feature month (max of the two months)",
    "days_since_peak": "days from the single max-impression day in the feature window to the origin's first day",
    "keyword_to_content_gap_days": "content creation date minus keyword creation date, days",
}


def run_features():
    D = Data("sealed")
    d = D.df
    cov = {o: {c: round(float(D.X.loc[d.origin == o, c].notna().mean()), 3) for c in ADDED} for o in sorted(set(d.origin))}
    rec = {"spec_version": SPEC["spec_version"], "written_at": now(), "n_features": len(FEATURE_LIST),
           "n_v1_features": len(v1.FEATURES), "removed": REMOVED, "added": ADDED,
           "set_missing": SPEC["features"]["set_missing"],
           "features": [{"name": f, "family": FAMILY[f], "meaning": MEANING_NEW.get(f) or v1.FEATURES[f], "new": f in ADDED} for f in FEATURE_LIST],
           "non_missing_share_of_new_features_by_origin": cov,
           "timeline_check_rhythm": D.rhythm_diag,
           "snapshot_diagnostics_dim_content": rh.snapshot_diagnostics(d),
           "note": "page_prev_ratio is not a model feature; it feeds only the frozen persistence baseline."}
    dump(rec, "v2_features.json")
    print("features", len(FEATURE_LIST), flush=True)


# ------------------------------------------------------------------ freeze
def run_freeze():
    for f in PRE_FREEZE_RECEIPTS + ["v2_secondary_label.json", "v2_features.json"]:
        assert os.path.exists(f"{OUT}/{f}"), f"missing receipt before freeze: {f}"
    assert not os.path.exists(OUT + "/v2_freeze.json"), "already frozen"
    assert not os.path.exists(OUT + "/v2_test.json")
    sel = json.load(open(OUT + "/v2_selection.json"))
    selected = sel["selected_model"]
    spec_hash = json.load(open(OUT + "/v2_spec_hash.json"))
    fz = {"frozen_at": now(), "spec_version": SPEC["spec_version"], "selected_model": selected,
          "selected_model_decision": sel["decision"],
          "selected_configs": {fam: sel["best_config_per_family"][fam]["cfg"] for fam in FAMILY_ORDER},
          "champion": "LOGISTIC", "test_origin": SEALED, "test_train_origins": FOLDS["test"]["train"],
          "live_origin": LIVE, "live_train_origins": FOLDS["live"]["train"],
          "spec_sha256_at_preregistration": spec_hash["sha256_normalised_lf"],
          "spec_sha256": {k: sha_norm(v) for k, v in SPEC_FILES.items()},
          "code_sha256": {f: sha_norm(f"{HERE}/{f}") for f in CODE_FILES},
          "receipt_sha256": {f: sha_norm(f"{OUT}/{f}") for f in PRE_FREEZE_RECEIPTS},
          "info_sha256_secondary_before_test": sha_norm(OUT + "/v2_secondary_label.json"),
          "order": ["spec committed", "select", "confirm", "secondary", "checks", "freeze", "unseal (once)", "live queue"],
          "post_freeze_changes": []}
    dump(fz, "v2_freeze.json")
    print("frozen:", selected, flush=True)


# ------------------------------------------------------------------ unseal: the test month, once
def run_unseal():
    fz = gate("test")                                  # raises unless the freeze is intact and the test was not scored
    t0 = time.time()
    D = Data("test")
    selected, cfgs = fz["selected_model"], fz["selected_configs"]
    P = fold_payload(D, "test")
    train = FOLDS["test"]["train"]
    assert P["tr"].size and not set(D.origin[P["tr"]]) - set(train)
    te = P["te"]
    assert D.meas[te].sum() > 0, "test outcomes are not loaded"
    s, info = fit_all_candidates(P, cfgs)
    shown = {"SELECTED:" + selected: s[selected], "CHAMPION:LOGISTIC": s["LOGISTIC"]} if selected != "LOGISTIC" else {"SELECTED:LOGISTIC": s["LOGISTIC"]}
    ev = evaluate_set(D, te, {**baseline_scores(D, te), **shown}, train_origins=train)
    sec = secondary_fold(D, "test", cfgs["LOGISTIC"])
    rec = {"spec_version": SPEC["spec_version"], "scored_at": now(), "scored_once": True,
           "disclosure": SPEC["disclosure"], "score_origin": SEALED, "outcome_window": FOLDS["test"]["outcome_window"],
           "selected_model": selected, "configs_used": cfgs, "train_origins": train, "n_train_rows": int(P["tr"].size),
           "dropped_all_missing_in_training": P["dropped_all_missing"], "early_stopping_block": P["holdout"],
           "evaluation": ev, "secondary_label": sec,
           "fit_info": info,
           "headline": {"metric": "AP and P@50 of the selected model against base rate, noise-only rate, editor's rule and best-AP baseline",
                        "selected_model_ap": ev["models"]["SELECTED:" + selected]["ap"] if ("SELECTED:" + selected) in ev["models"] else None,
                        "selected_model_p@50": ev["models"]["SELECTED:" + selected]["p@50"] if ("SELECTED:" + selected) in ev["models"] else None,
                        "base_rate": ev["base_rate"], "noise_only_rate": ev["noise_only_rate"],
                        "best_baseline_by_ap": ev["best_baseline_by_ap"]}}
    with open(OUT + "/v2_test.json", "x", encoding="utf-8", newline="\n") as f:     # exclusive: never overwritten
        json.dump(rec, f, indent=1, default=_jd)
        f.write("\n")
    sl = json.load(open(OUT + "/v2_secondary_label.json"))
    sl["test"] = sec
    dump(sl, "v2_secondary_label.json")
    h = rec["headline"]
    print("TEST scored once. base", h["base_rate"], "noise", h["noise_only_rate"], "selected", selected,
          "AP", h["selected_model_ap"], "P@50", h["selected_model_p@50"], "best baseline", h["best_baseline_by_ap"],
          round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if "--unseal" in sys.argv:
        run_unseal()
    elif cmd == "select":
        run_select()
    elif cmd == "confirm":
        run_confirm()
    elif cmd == "secondary":
        run_secondary()
    elif cmd == "features":
        run_features()
    elif cmd == "freeze":
        run_freeze()
    else:
        print(__doc__)
