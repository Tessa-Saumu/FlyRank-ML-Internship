"""V2 model fits: logistic (champion), boosted classifier, boosted Poisson rate. scikit-learn only.

Everything here is single-threaded and seeded so results are bit-reproducible; parallelism, where used,
is across processes (one config per task), never inside a fit. Pure functions, no file or warehouse access.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"          # must be set before numpy / scikit-learn are imported

import warnings

import numpy as np
from scipy.special import xlogy
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
LOGISTIC_C = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
PENALTY_L1_RATIO = {"l2": 0.0, "l1": 1.0, "elasticnet": 0.5}
SOLVER = {"l2": "lbfgs", "l1": "saga", "elasticnet": "saga"}
MAX_ITER = {"lbfgs": 5000, "saga": 20000}
HGB_LEAVES = [7, 15, 31]
HGB_MSL_FRAC = [0.005, 0.015, 0.04]
HGB_L2 = [0.0, 1.0, 5.0, 20.0]
HGB_MAXF = [0.5, 0.8, 1.0]
HGB_ITER_CAP = 600
HGB_MSL_FLOOR = 20
HGB_SEEDS = [42, 43, 44, 45, 46]
GROUPED_HOLDOUT_SHARE = (0.10, 0.40)


# ------------------------------------------------------------------ logistic
def logistic_grid():
    return [{"C": C, "penalty": p} for p in ("l2", "l1", "elasticnet") for C in LOGISTIC_C]


def logistic_key(cfg):
    return f"{cfg['penalty']}_C{cfg['C']:g}"


def make_logistic(cfg):
    solver = SOLVER[cfg["penalty"]]
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
        LogisticRegression(C=cfg["C"], l1_ratio=PENALTY_L1_RATIO[cfg["penalty"]], solver=solver,
                           max_iter=MAX_ITER[solver], tol=1e-4, random_state=SEED))


def fit_logistic(Xtr, ytr, cfg):
    """Returns (fitted pipeline, info). info records non-convergence and the iteration count."""
    m = make_logistic(cfg)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.fit(Xtr, ytr)
    return m, {"converged": not any(issubclass(x.category, ConvergenceWarning) for x in w),
               "n_iter": int(np.max(m[-1].n_iter_))}


def score_logistic(m, X):
    return m.predict_proba(X)[:, 1]


def logistic_contributions(m, X):
    """Standardised contribution per original feature: scaled value times coefficient (indicator columns excluded)."""
    n = X.shape[1]
    Z = m[:-1].transform(X)[:, :n]
    return Z * m[-1].coef_[0][:n]


# ------------------------------------------------------------------ boosted models
def sample_hgb_configs(n=20, seed=SEED):
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        cfg = {"learning_rate": round(float(np.exp(rng.uniform(np.log(0.02), np.log(0.2)))), 5),
               "max_leaf_nodes": int(rng.choice(HGB_LEAVES)), "msl_frac": float(rng.choice(HGB_MSL_FRAC)),
               "l2": float(rng.choice(HGB_L2)), "max_features": float(rng.choice(HGB_MAXF))}
        key = tuple(cfg.values())
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def hgb_key(cfg):
    return f"lr{cfg['learning_rate']:g}_leaves{cfg['max_leaf_nodes']}_msl{cfg['msl_frac']:g}_l2{cfg['l2']:g}_mf{cfg['max_features']:g}"


def _make_hgb(kind, cfg, n_rows, n_iter, seed):
    kw = dict(learning_rate=cfg["learning_rate"], max_leaf_nodes=cfg["max_leaf_nodes"],
              min_samples_leaf=max(HGB_MSL_FLOOR, int(round(cfg["msl_frac"] * n_rows))),
              l2_regularization=cfg["l2"], max_features=cfg["max_features"], max_iter=n_iter,
              early_stopping=False, random_state=seed)
    if kind == "clf":
        return HistGradientBoostingClassifier(**kw)
    return HistGradientBoostingRegressor(loss="poisson", **kw)


def _stage_losses(kind, est, Xv, yv, wv):
    """Validation loss after every boosting iteration: log loss (clf) or weighted Poisson deviance (reg)."""
    out = []
    if kind == "clf":
        for p in est.staged_predict_proba(Xv):
            p1 = np.clip(p[:, 1], 1e-15, 1 - 1e-15)
            out.append(-np.mean(yv * np.log(p1) + (1 - yv) * np.log1p(-p1)))
    else:
        sw = wv.sum()
        for mu in est.staged_predict(Xv):
            mu = np.clip(mu, 1e-12, None)
            out.append(2.0 * np.sum(wv * (xlogy(yv, yv / mu) - yv + mu)) / sw)
    return np.asarray(out)


def grouped_holdout_mask(groups, start_seed=SEED, share=GROUPED_HOLDOUT_SHARE):
    """Client-grouped 20 percent of clients as the early-stopping block (first seed whose row share is in range)."""
    groups = np.asarray(groups)
    dummy = np.zeros(len(groups))
    for sd in range(start_seed, start_seed + 200):
        _, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=sd).split(dummy, groups=groups))
        r = len(va) / len(groups)
        if share[0] <= r <= share[1]:
            m = np.zeros(len(groups), bool)
            m[va] = True
            return m, {"seed": sd, "row_share": round(float(r), 4), "n_clients_held_out": int(len(np.unique(groups[va])))}
    raise RuntimeError("no grouped holdout with an admissible row share")


def usable_columns(X):
    """Columns with at least two distinct non-missing values (scikit-learn 1.9's histogram binner raises on fewer)."""
    return [c for c in X.columns if X[c].nunique(dropna=True) >= 2]


def fit_boosted(kind, cfg, Xtr, ytr, wtr, val_mask, Xte, seeds=HGB_SEEDS):
    """Early-stop on the validation block (fit on the rest), refit on all rows with that many trees, average seeds.

    kind 'clf': ytr in {0,1}, returns mean predicted probability.
    kind 'reg': ytr = clicks / expected clicks, wtr = expected clicks, returns mean predicted ratio.
    max_features == 1.0 makes the seed irrelevant, so those configs are fitted once.
    Columns that carry no information in a given fit (all missing or constant) are left out of that fit only.
    """
    use = list(seeds) if cfg["max_features"] < 1.0 else [seeds[0]]
    fit_m = ~val_mask
    Xfit, Xval = Xtr[fit_m], Xtr[val_mask]
    cols_es, cols_fin = usable_columns(Xfit), usable_columns(Xtr)
    preds, iters = [], []
    for sd in use:
        w_fit = None if wtr is None else wtr[fit_m]
        est = _make_hgb(kind, cfg, int(fit_m.sum()), HGB_ITER_CAP, sd)
        est.fit(Xfit[cols_es], ytr[fit_m], sample_weight=w_fit)
        loss = _stage_losses(kind, est, Xval[cols_es], ytr[val_mask], None if wtr is None else wtr[val_mask])
        best = int(np.argmin(loss)) + 1
        iters.append(best)
        fin = _make_hgb(kind, cfg, len(Xtr), best, sd).fit(Xtr[cols_fin], ytr, sample_weight=wtr)
        Xt = Xte[cols_fin]
        preds.append(fin.predict_proba(Xt)[:, 1] if kind == "clf" else fin.predict(Xt))
    return np.mean(preds, axis=0), {"n_seeds_fitted": len(use), "best_iters": iters,
                                    "hit_cap": int(sum(i >= HGB_ITER_CAP for i in iters)),
                                    "columns_left_out_of_early_stopping_fit": int(Xtr.shape[1] - len(cols_es)),
                                    "columns_left_out_of_final_fit": int(Xtr.shape[1] - len(cols_fin))}


def tune_boosted_config(kind, cfg, payloads):
    """One task for the process pool: one config fitted on every fold payload (dicts with Xtr, ytr, wtr, val_mask, Xte)."""
    return [fit_boosted(kind, cfg, p["Xtr"], p["ytr"], p["wtr"], p["val_mask"], p["Xte"]) for p in payloads]
