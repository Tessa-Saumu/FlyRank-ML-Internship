"""CTR-erosion early warning: features, frozen baselines, tuned models, forward validation.

Design (fixed before any test-origin result was looked at):
  * dev test origins 2026-02, 2026-03, 2026-04; each trains only on origins <= o-2 (label maturity)
  * sealed origin 2026-05 (outcome May-Jun 2026) is scored only with --unseal, once, after freezing
  * tuning: inner GroupKFold by client over the TRAINING origins only, scored by average precision
  * primary metric: P@50 per origin (review budget 50 pages/month), base rate and baselines alongside;
    secondary: P@20, P@100, PR-AUC (AP), ROC-AUC, mean within-client P@10
  * uncertainty: client-cluster bootstrap (B=1000) of P@50 and of the model-minus-baseline delta
Outputs: work/freestyle/outputs/erosion_model_metrics.json (+ erosion_features.json)
"""
import hashlib
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from erosion_frame import ORIGINS, SEALED_ORIGIN, f_pos, month_add
from wh import CACHE

OUT = os.path.dirname(os.path.abspath(__file__)) + "/../outputs"
DEV_ORIGINS = ["2026-02", "2026-03", "2026-04"]
K = 50
SEED = 42
INTENTS = ["informational", "transactional", "commercial", "navigational"]

FEATURES = {
    # exposure and trajectory
    "l_imp_f": "log impressions, feature window (2 months)",
    "l_clk_f": "log clicks, feature window",
    "imp_growth": "log impressions month -1 vs month -2",
    "ctr_growth": "log CTR month -1 vs month -2 (clicks growth minus impressions growth)",
    "imp_growth_prev": "log impressions month -2 vs month -3",
    "imp_l14_share": "share of month -1 impressions in its last 14 days",
    "ctr_recency": "log CTR last 14 days vs whole feature window",
    "click_day_share": "days with a click / days with an impression",
    "spikiness": "max single-day impressions / window impressions",
    # rank
    "pos_f": "impression-weighted position, feature window",
    "pos_delta": "position month -1 minus month -2",
    "pos_std": "impression-weighted std of daily position",
    # CTR relative to expectations
    "ctr_vs_ref": "log shrunk CTR vs reference CTR-at-position curve (Oct-Nov 2025)",
    "ctr_vs_client": "log shrunk CTR vs client's position-adjusted CTR",
    # content
    "l_age": "log page age in days at origin",
    "months_active": "months with impressions before origin",
    "word_count": "word count",
    "keyword_token_count": "target keyword length in tokens",
    "l_search_volume": "log target keyword search volume",
    "cpc": "target keyword CPC",
    "competition": "target keyword competition",
    "is_feedly": "content type = feedly article",
    **{f"intent_{i}": f"main intent = {i}" for i in INTENTS},
    # channels (GA4, availability-gated)
    "has_ga4": "any GA4-available day in feature window",
    "l_org_sess": "log organic sessions, feature window",
    "has_ai_referral": "any AI-assistant session in feature window",
    # client context (past only)
    "c_ctr_growth": "client-wide log CTR change month -1 vs -2 (top-10 pages)",
    "c_l_imp": "client log impressions on top-10 pages, feature window",
    "c_l_pages": "client log count of top-10 pages",
    "client_prev_erode_rate": "client erosion rate at origin -2 (matured label)",
    # page history (matured)
    "page_prev_ratio": "page log(clicks/expected) at origin -2 (matured)",
}


def load():
    df = pd.read_parquet(CACHE + "/erosion_frame.parquet")
    rc = json.load(open(OUT + "/erosion_frame_summary.json"))["reference_curve"]
    grid, vals = np.array(rc["pos"]), np.array(rc["ctr"])
    return df, grid, vals


def make_features(df, grid, vals):
    X = pd.DataFrame(index=df.index)
    lg = lambda a, b: np.log((a.fillna(0) + 1) / (b.fillna(0) + 1))
    X["l_imp_f"] = np.log1p(df.imp_f)
    X["l_clk_f"] = np.log1p(df.clk_f)
    X["imp_growth"] = lg(df.imp_m1, df.imp_m2)
    X["ctr_growth"] = lg(df.clk_m1, df.clk_m2) - X.imp_growth
    X["imp_growth_prev"] = np.where(df.imp_m3.notna(), lg(df.imp_m2, df.imp_m3), np.nan)
    X["imp_l14_share"] = df.imp_l14 / df.imp_m1.replace(0, np.nan)
    X["ctr_recency"] = np.log(((df.clk_l14.fillna(0) + 0.5) / (df.imp_l14.fillna(0) + 1)) / ((df.clk_f + 0.5) / (df.imp_f + 1)))
    X["click_day_share"] = df.click_days_f / df.imp_days_f.replace(0, np.nan)
    X["spikiness"] = df.max_day_imp / df.imp_f
    X["pos_f"] = df.pos_f
    X["pos_delta"] = df.pos_m1 - df.pos_m2
    X["pos_std"] = np.sqrt(np.clip(df.s2 / df.imp_f - df.pos_f ** 2, 0, None))
    ffeat = f_pos(df.pos_f, grid, vals)
    X["ctr_vs_ref"] = np.log(df.ctr_shrunk / ffeat)
    X["ctr_vs_client"] = np.log(df.ctr_shrunk / (df.c_ctr_f * ffeat / f_pos(df.c_pos_f, grid, vals)))
    X["l_age"] = np.log1p(df.age_days.clip(lower=0))
    X["months_active"] = df.months_active
    X["word_count"] = df.word_count
    X["keyword_token_count"] = df.keyword_token_count
    X["l_search_volume"] = np.log1p(df.search_volume)
    X["cpc"] = df.cpc
    X["competition"] = df.competition
    X["is_feedly"] = (df.content_type == "feedly article").astype(float)
    for i in INTENTS:
        X[f"intent_{i}"] = (df.main_intent == i).astype(float)
    X["has_ga4"] = (df.ga4_days_f.fillna(0) > 0).astype(float)
    X["l_org_sess"] = np.where(X.has_ga4 > 0, np.log1p(df.org_f.fillna(0)), np.nan)
    X["has_ai_referral"] = (df.ai_f.fillna(0) > 0).astype(float)
    X["c_ctr_growth"] = lg(df.c_clk_m1, df.c_imp_m1) - lg(df.c_clk_m2, df.c_imp_m2)
    X["c_l_imp"] = np.log1p(df.c_imp_m1.fillna(0) + df.c_imp_m2.fillna(0))
    X["c_l_pages"] = np.log1p(df.c_pages)
    X["client_prev_erode_rate"] = df.client_prev_erode_rate
    X["page_prev_ratio"] = df.page_prev_ratio
    assert list(X.columns) == list(FEATURES), "feature list drifted from FEATURES"
    return X.astype(float)


def tiebreak(ids):
    """Deterministic, seed-free tiebreak: tiny value from a hash of the page id."""
    return np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 2**32 * 1e-9 for s in ids])


# ---------------- frozen baselines (each returns a score; higher = more likely to erode)
def rule_score(X):
    """Business rule: top positions are most exposed to AI answers, and a CTR already slipping
    month over month is the warning sign. Reason codes: TOP3_SLIPPING / TOP10_SLIPPING / TOP3_EXPOSED."""
    tier = np.select([X.pos_f <= 3.5, X.pos_f <= 6.5], [2.0, 1.0], 0.0)
    return tier + np.clip(-X.ctr_growth, 0, 3)


BASELINES = {
    "rule_position_slipping": rule_score,
    "mean_reversion_ctr_excess": lambda X: X.ctr_vs_client.fillna(0),
    "client_shock": lambda X: -X.c_ctr_growth.fillna(0),
    "persistence_prev_ratio": lambda X: -X.page_prev_ratio.fillna(X.page_prev_ratio.median()),
}


def p_at_k(y, s, k):
    idx = np.argsort(-s, kind="stable")[:k]
    return float(np.mean(y[idx]))


def within_client_p(y, s, g, k=10, min_rows=30):
    vals = []
    for c in np.unique(g):
        m = g == c
        if m.sum() >= min_rows:
            vals.append(p_at_k(y[m], s[m], k))
    return float(np.mean(vals)) if vals else np.nan


def evaluate(y, s, g, k=K):
    return {"p@20": p_at_k(y, s, 20), "p@50": p_at_k(y, s, k), "p@100": p_at_k(y, s, 100),
            "ap": float(average_precision_score(y, s)), "auc": float(roc_auc_score(y, s)),
            "within_client_p@10": within_client_p(y, s, g), "clients_in_top50": int(len(set(g[np.argsort(-s, kind='stable')[:k]])))}


def cluster_boot(y, scores: dict, g, ref, B=1000, k=K, seed=SEED):
    """Client-cluster bootstrap of P@k for each score and of (score - ref)."""
    rng = np.random.default_rng(seed)
    clients = np.unique(g)
    idx_by = {c: np.flatnonzero(g == c) for c in clients}
    res = {n: [] for n in scores}
    for _ in range(B):
        idx = np.concatenate([idx_by[c] for c in rng.choice(clients, len(clients))])
        for n, s in scores.items():
            res[n].append(p_at_k(y[idx], s[idx] + tiebreak_cache[idx], k))
    out = {}
    for n, v in res.items():
        v = np.array(v)
        d = v - np.array(res[ref])
        out[n] = {"p@50_ci": [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)],
                  "delta_vs_" + ref + "_ci": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
                  "p_delta_le_0": round(float(np.mean(d <= 0)), 4)}
    return out


def lr_model(C):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


HGB_SPACE = dict(learning_rate=[0.03, 0.06, 0.1], max_leaf_nodes=[7, 15, 31], min_samples_leaf=[50, 150, 400],
                 l2_regularization=[0.0, 1.0, 5.0], max_features=[0.5, 0.8, 1.0])


def hgb_model(p):
    return HistGradientBoostingClassifier(max_iter=400, early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=25, random_state=SEED, **p)


def tune(Xtr, ytr, gtr, n_hgb=16):
    """Inner GroupKFold(3) by client over training origins; returns best LR C and HGB params by mean AP."""
    gkf = GroupKFold(n_splits=3)
    folds = list(gkf.split(Xtr, ytr, gtr))
    def cv_ap(make):
        aps = []
        for tr, va in folds:
            m = make().fit(Xtr.iloc[tr], ytr[tr])
            aps.append(average_precision_score(ytr[va], m.predict_proba(Xtr.iloc[va])[:, 1]))
        return float(np.mean(aps))
    lr_res = {C: cv_ap(lambda C=C: lr_model(C)) for C in [0.003, 0.01, 0.03, 0.1, 1.0]}
    rng = np.random.default_rng(SEED)
    cands = [{k: v[rng.integers(len(v))] for k, v in HGB_SPACE.items()} for _ in range(n_hgb)]
    hgb_res = [(cv_ap(lambda p=p: hgb_model(p)), p) for p in cands]
    best_C = max(lr_res, key=lr_res.get)
    best_ap, best_p = max(hgb_res, key=lambda t: t[0])
    return best_C, best_p, {"lr_inner_ap": {str(k): round(v, 4) for k, v in lr_res.items()},
                            "hgb_inner_best_ap": round(best_ap, 4), "hgb_trials": len(cands)}


def rank_blend(model_score, rule, w):
    """Weighted average of percentile ranks: w on the model, 1-w on the rule."""
    r = lambda s: pd.Series(s).rank(pct=True).to_numpy()
    return w * r(model_score) + (1 - w) * r(rule)


def choose_blend_weight(Xm_tr, X_tr, ytr, gtr, C, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Out-of-fold (GroupKFold by client) LR scores on the training origins; pick w by AP."""
    oof = np.zeros(len(ytr))
    for a, b in GroupKFold(n_splits=3).split(Xm_tr, ytr, gtr):
        oof[b] = lr_model(C).fit(Xm_tr.iloc[a], ytr[a]).predict_proba(Xm_tr.iloc[b])[:, 1]
    rule = rule_score(X_tr)
    return max(grid, key=lambda w: average_precision_score(ytr, rank_blend(oof, rule, w)))


def run(test_origins, tag):
    global tiebreak_cache
    df, grid, vals = load()
    df = df[df.measurable].reset_index(drop=True)
    X = make_features(df, grid, vals)
    y = df.erode.to_numpy().astype(int)
    g = df.client_hash_id.to_numpy()
    tb = tiebreak(df.content_hash_id)
    report = {"design": {"dev_origins": DEV_ORIGINS, "sealed_origin": SEALED_ORIGIN, "k": K, "features": len(FEATURES)},
              "origins": {}}
    for o in test_origins:
        tr = df.origin.isin([m for m in ORIGINS if m <= month_add(o, -2)]).to_numpy()
        te = (df.origin == o).to_numpy()
        # matured-history features are all-missing when no origin o-4 exists: drop them for this fold
        cols = [c for c in X.columns if X.loc[tr, c].notna().any()]
        Xm = X[cols]
        C, p, tuning = tune(Xm[tr], y[tr], g[tr])
        lr = lr_model(C).fit(Xm[tr], y[tr])
        hgb = hgb_model(p).fit(Xm[tr], y[tr])
        yt, gt, tbt = y[te], g[te], tb[te]
        tiebreak_cache = tbt
        scores = {n: np.asarray(f(X[te]), float) for n, f in BASELINES.items()}
        scores["logreg"] = lr.predict_proba(Xm[te])[:, 1]
        scores["hgb"] = hgb.predict_proba(Xm[te])[:, 1]
        # rule + model rank blend; weight chosen on out-of-fold training predictions only
        w = choose_blend_weight(Xm[tr], X[tr], y[tr], g[tr], C)
        scores["logreg_rule_blend"] = rank_blend(scores["logreg"], scores["rule_position_slipping"], w)
        tuning["blend_weight_on_model"] = w
        res = {n: {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in evaluate(yt, s + tbt, gt).items()}
               for n, s in scores.items()}
        best_base = max(BASELINES, key=lambda n: res[n]["p@50"])
        boot = cluster_boot(yt, scores, gt, ref=best_base)
        # rule ties: expected P@50 under random order within tied scores
        report["origins"][o] = {
            "train_origins": sorted(df.origin[tr].unique().tolist()), "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "test_clients": int(len(np.unique(gt))), "base_rate": round(float(yt.mean()), 4),
            "null_noise_rate": round(float(df.null_erode[te].mean()), 4),
            "rule_distinct_scores_top50": int(len(np.unique(np.round(scores["rule_position_slipping"][np.argsort(-(scores["rule_position_slipping"] + tbt))[:K]], 6)))),
            "best_baseline": best_base, "features_used": len(cols), "features_dropped_all_missing": [c for c in X.columns if c not in cols],
            "tuning": {"lr_C": C, "hgb_params": p, **tuning},
            "metrics": res, "bootstrap": boot}
        print(o, "base", round(yt.mean(), 3), {n: res[n]["p@50"] for n in res}, "best_base", best_base, flush=True)
        if o == test_origins[-1]:
            last = (Xm[te], yt, hgb, lr, scores, df[te])
    # ---- permutation importance (last test origin, AP drop), HGB
    Xt, yt, hgb, lr, scores, dft = last
    pi = permutation_importance(hgb, Xt, yt, scoring="average_precision", n_repeats=5, random_state=SEED)
    imp = pd.Series(pi.importances_mean, index=Xt.columns).sort_values(ascending=False)
    report["permutation_importance_last_origin"] = imp.round(5).head(15).to_dict()
    coef = pd.Series(lr[-1].coef_[0][:Xt.shape[1]], index=Xt.columns).sort_values()
    report["logreg_coef_last_origin"] = {"most_negative": coef.head(6).round(3).to_dict(), "most_positive": coef.tail(6).round(3).to_dict()}
    # ---- error profile: top-50 of HGB, true vs false positives
    top = np.argsort(-scores["hgb"], kind="stable")[:K]
    prof = Xt.iloc[top].assign(y=yt[top]).groupby("y")[["pos_f", "ctr_vs_client", "ctr_growth", "c_ctr_growth", "l_imp_f", "page_prev_ratio"]].median()
    report["top50_error_profile_last_origin"] = json.loads(prof.round(3).to_json(orient="index"))
    json.dump(report, open(f"{OUT}/erosion_model_metrics{tag}.json", "w"), indent=1, default=str)
    return report


def fit_score(p, Xa, ya, Xb):
    """Fit HGB on Xa (dropping columns that are entirely missing there) and score Xb."""
    cols = [c for c in Xa.columns if Xa[c].notna().any()]
    return hgb_model(p).fit(Xa[cols], ya).predict_proba(Xb[cols])[:, 1]


def leakage_and_split_checks():
    """(1) planted leak must push P@50 toward 1; (2) random-row vs forward gap; (3) unseen-client CV per origin."""
    df, grid, vals = load()
    df = df[df.measurable & (df.origin != SEALED_ORIGIN)].reset_index(drop=True)
    X = make_features(df, grid, vals)
    y = df.erode.to_numpy().astype(int)
    g = df.client_hash_id.to_numpy()
    p = {"learning_rate": 0.06, "max_leaf_nodes": 15, "min_samples_leaf": 150, "l2_regularization": 1.0, "max_features": 0.8}
    out = {}
    # planted leak on the 2026-04 forward split
    tr = df.origin.isin(["2025-12", "2026-01", "2026-02"]).to_numpy(); te = (df.origin == "2026-04").to_numpy()
    rng = np.random.default_rng(SEED)
    XL = X.assign(leak=df.log_ratio_out + rng.normal(0, 0.3, len(df)))
    for name, XX in [("honest", X), ("planted_future_ratio", XL)]:
        s = fit_score(p, XX[tr], y[tr], XX[te])
        out[f"forward_2026-04_{name}"] = {"p@50": round(p_at_k(y[te], s, K), 3), "ap": round(float(average_precision_score(y[te], s)), 3)}
    # random-row KFold vs client-grouped KFold over all dev rows (same model)
    for name, cv in [("random_row_kfold", KFold(5, shuffle=True, random_state=SEED)), ("client_group_kfold", GroupKFold(5))]:
        aps = []
        for a, b in cv.split(X, y, g):
            aps.append(average_precision_score(y[b], fit_score(p, X.iloc[a], y[a], X.iloc[b])))
        out[name] = {"ap_mean": round(float(np.mean(aps)), 4), "ap_std": round(float(np.std(aps)), 4), "base_rate": round(float(y.mean()), 4)}
    # unseen-client check inside each origin (same time, new clients)
    per = {}
    for o, d in df.groupby("origin"):
        ii = d.index.to_numpy(); aps = []; bases = []
        for a, b in GroupKFold(5).split(ii, y[ii], g[ii]):
            aps.append(average_precision_score(y[ii[b]], fit_score(p, X.iloc[ii[a]], y[ii[a]], X.iloc[ii[b]]))); bases.append(y[ii[b]].mean())
        per[o] = {"ap_mean": round(float(np.mean(aps)), 4), "base_mean": round(float(np.mean(bases)), 4)}
    out["unseen_client_within_origin"] = per
    json.dump(out, open(f"{OUT}/erosion_leakage_checks.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    json.dump(FEATURES, open(OUT + "/erosion_features.json", "w"), indent=1)
    if "--unseal" in sys.argv:
        run([SEALED_ORIGIN], "_sealed")
    elif "--checks" in sys.argv:
        leakage_and_split_checks()
    else:
        run(DEV_ORIGINS, "")
