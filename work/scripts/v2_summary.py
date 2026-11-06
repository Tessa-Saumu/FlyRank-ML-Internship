"""V2 summary receipt: three small descriptive analyses that the V2 narrative needs and no earlier receipt holds.

Writes work/outputs/v2_summary.json (aggregates only: no IDs, no test-month outcomes). It decides nothing:
the V2 selection, freeze and test were written before this file existed and none of them reads it.

  1. rhythm_signal_audit   the five date and rhythm features added in V2, tested as single signals on the labelled
                           origins whose outcome windows end before the test window opens (2025-12 to 2026-03).
                           Within-client AUC (pairs weighted), client-cluster bootstrap, a verdict per feature.
  2. contract_leak_demo    the ML-04 leak demo with the five core contract features, run on validation fold 2
                           (train 2025-12 and 2026-01, score 2026-03), with and without the outcome ingredient.
  3. champion_coefficients standardised coefficients of the champion logistic (its selected configuration) refitted
                           on the test refit rows (origins 2025-12 to 2026-03). Uses training labels only.

Loads the frame in the default sealed mode (test-origin outcomes are dropped), so it cannot see the test month.
Run:  python work/scripts/v2_summary.py      (about two minutes; needs the cached frame and rhythm cache)
"""
import v2_pipeline as p            # first: sets single-thread environment variables before numpy loads
import v2_models as mm

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

AUDIT_ORIGINS = ["2025-12", "2026-01", "2026-02", "2026-03"]      # outcome windows end 2026-04-30 at the latest
B = 2000
MIN_ROWS = 30
CORE_FIVE = ["l_imp_f", "pos_f", "ctr_vs_ref", "ctr_growth", "days_since_last_click"]
DEMO_CFG = {"C": 0.1, "penalty": "l2"}                            # fixed in advance, never tuned
# Directions written down before the numbers were computed (higher value -> more erosion is +1).
EXPECTED = {
    "weekday_share": (+1, "pages read on working days sit in transactional and commercial searches, the intents that lost most CTR in ML-06"),
    "days_since_last_click": (+1, "a page whose last click is long ago is already fading"),
    "longest_zero_click_run": (+1, "a long run of click-free days is a fading page"),
    "days_since_peak": (-1, "a peak just before the decision is a spike that tends to revert (ML-07 check 2: bumps revert)"),
    "keyword_to_content_gap_days": (0, "no prior direction; reported as descriptive only"),
}


def group_stats(x, y, groups):
    """Per group: Mann-Whitney U and the number of positive-negative pairs (ties count one half)."""
    out = {}
    for g in np.unique(groups):
        m = groups == g
        n = int(m.sum())
        yy = y[m]
        npos = int(yy.sum())
        nneg = n - npos
        if n < MIN_ROWS or npos == 0 or nneg == 0:
            continue
        r = rankdata(x[m])
        u = float(r[yy == 1].sum() - npos * (npos + 1) / 2)
        out[g] = (u, float(npos * nneg))
    return out


def audit_feature(D, f):
    idx = np.flatnonzero(np.isin(D.origin, AUDIT_ORIGINS) & D.meas & D.X[f].notna().to_numpy())
    x, y = D.X[f].to_numpy(float)[idx], D.y[idx].astype(int)
    o, g = D.origin[idx], D.g[idx]
    key = np.array([f"{a}|{b}" for a, b in zip(o, g)])
    st = group_stats(x, y, key)
    clients = np.unique(g)
    # per (origin, client) pieces, indexed for the bootstrap
    rows = [(k.split("|")[0], k.split("|")[1], u, pairs) for k, (u, pairs) in st.items()]
    df = pd.DataFrame(rows, columns=["origin", "client", "u", "pairs"])
    pooled = float(df.u.sum() / df.pairs.sum())
    per_origin = {oo: float(d.u.sum() / d.pairs.sum()) for oo, d in df.groupby("origin")}
    by_client = df.groupby("client")[["u", "pairs"]].sum()
    cl = by_client.index.to_numpy()
    U, P = by_client.u.to_numpy(), by_client.pairs.to_numpy()
    rng = np.random.default_rng(mm.SEED)
    boot = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, len(cl), len(cl)), minlength=len(cl))
        boot[b] = (w * U).sum() / (w * P).sum()
    ci = [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]
    # percentile buckets within origin, pooled erosion rate per bucket
    pct = pd.Series(x).groupby(o).rank(pct=True, method="average").to_numpy()
    q = np.digitize(pct, [0.25, 0.5, 0.75], right=True)        # 0..3
    buckets = [{"bucket": f"Q{k + 1}", "n": int((q == k).sum()), "erosion_rate": round(float(y[q == k].mean()), 4),
                "feature_range": [round(float(x[q == k].min()), 2), round(float(x[q == k].max()), 2)]}
               for k in range(4) if (q == k).any()]            # ties can leave a bucket empty; it is then omitted
    exp, why = EXPECTED[f]
    sig = ci[0] > 0.5 or ci[1] < 0.5
    direction = 1 if pooled > 0.5 else -1
    consistent = all((a - 0.5) * direction > 0 for a in per_origin.values())
    if not sig:
        verdict = "FALSE" if exp != 0 else "NO SIGNAL"
    elif not consistent:
        verdict = "MIXED"
    elif exp == 0:
        verdict = "SIGNAL (descriptive)"
    else:
        verdict = "CONFIRMED" if direction == exp else "OPPOSITE"
    return {"feature": f, "expected_direction": {1: "higher -> more erosion", -1: "higher -> less erosion", 0: "none"}[exp],
            "why_expected": why, "within_client_auc": round(pooled, 4), "ci95": ci,
            "per_origin_auc": {k: round(v, 4) for k, v in per_origin.items()},
            "same_sign_in_all_origins": bool(consistent), "n_rows": int(len(idx)), "n_clients": int(len(clients)),
            "n_client_origin_groups": int(len(df)), "base_rate": round(float(y.mean()), 4), "buckets": buckets, "verdict": verdict}


def contract_demo(D):
    tr = np.flatnonzero(np.isin(D.origin, ["2025-12", "2026-01"]) & D.meas)
    te = np.flatnonzero((D.origin == "2026-03") & D.meas)
    d = D.df
    leak = np.log1p(d.clk_out.fillna(0).to_numpy(float)) - np.log1p(d.exp_out.to_numpy(float))
    res = {}
    for name, extra in [("five_core_features", False), ("with_leak_clk_out", True)]:
        X = D.X[CORE_FIVE].copy()
        if extra:
            X["leak_clk_out"] = leak
        cols = [c for c in X.columns if X.iloc[tr][c].notna().any()]
        m, _ = mm.fit_logistic(X.iloc[tr][cols].reset_index(drop=True), D.y[tr].astype(int), DEMO_CFG)
        s = mm.score_logistic(m, X.iloc[te][cols].reset_index(drop=True)) + D.tb[te]
        yt = D.y[te]
        res[name] = {"p@50": round(float(yt[np.argsort(-s, kind="stable")[:50]].mean()), 3),
                     "pr_auc": round(float(average_precision_score(yt, s)), 3)}
    res["base_rate_2026-03"] = round(float(D.y[te].mean()), 3)
    res["train_origins"] = ["2025-12", "2026-01"]
    res["score_origin"] = "2026-03"
    res["config"] = DEMO_CFG
    res["features"] = CORE_FIVE
    return res


def champion_coefficients(D):
    P = p.fold_payload(D, "test")
    sel, cfgs = p.selected_cfgs()
    assert sel["selected_model"] == "LOGISTIC"
    m, info = mm.fit_logistic(P["Xtr"], P["ytr"], cfgs["LOGISTIC"])
    coef = m[-1].coef_[0][:len(P["cols"])]
    t = pd.DataFrame({"feature": P["cols"], "std_coef": np.round(coef, 4)})
    new = set(p.ADDED)
    t["new_in_v2"] = t.feature.isin(new)
    t = t.sort_values("std_coef", ascending=False).reset_index(drop=True)
    return {"fitted_on_origins": P["train"], "n_train_rows": int(len(P["Xtr"])), "config": cfgs["LOGISTIC"], "converged": info["converged"],
            "note": "standardised coefficients; the configuration is weakly regularised (C = 100), so correlated features share weight",
            "top_positive": t.head(8).to_dict("records"), "top_negative": t.tail(8).iloc[::-1].to_dict("records"),
            "new_features": t[t.new_in_v2].to_dict("records"), "n_features": int(len(t))}


def main():
    D = p.Data("sealed")
    assert not set(AUDIT_ORIGINS) & {p.SEALED, p.LIVE, "2026-04"}
    audit = [audit_feature(D, f) for f in p.ADDED]
    out = {"spec_version": p.SPEC["spec_version"], "written_at": p.now(),
           "note": "Descriptive. Written after the V2 test was scored; nothing in the V2 selection, freeze or test reads it.",
           "rhythm_signal_audit": {"origins": AUDIT_ORIGINS, "metric": "within-client AUC, pairs weighted; client-cluster bootstrap",
                                   "bootstrap_B": B, "min_rows_per_client_origin": MIN_ROWS, "features": audit},
           "contract_leak_demo": contract_demo(D),
           "champion_coefficients": champion_coefficients(D)}
    p.dump(out, "v2_summary.json")
    for r in audit:
        print(r["feature"], r["within_client_auc"], r["ci95"], r["verdict"])
    print(out["contract_leak_demo"])


if __name__ == "__main__":
    main()
