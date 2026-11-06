"""Row-level re-scoring of a forward origin with the FROZEN logistic model, for error analysis.

Re-fits exactly what erosion_model.run() fits for the logistic model (same rows, same inner
tuning, same seed), asserts that its P@50 reproduces the committed receipt, and returns the
scored test frame. Nothing here is tuned; it only explains results already committed.
"""
import json

import numpy as np
import pandas as pd

from erosion_frame import ORIGINS, month_add
from erosion_model import K, OUT, load, lr_model, make_features, p_at_k, rule_score, tiebreak, tune


def scored_origin(o, receipt="erosion_model_metrics_sealed.json"):
    df, grid, vals = load()
    df = df[df.measurable.fillna(False)].reset_index(drop=True)
    X = make_features(df, grid, vals)
    y = df.erode.to_numpy().astype(int)
    g = df.client_hash_id.to_numpy()
    tr = df.origin.isin([m for m in ORIGINS if m <= month_add(o, -2)]).to_numpy()
    te = (df.origin == o).to_numpy()
    cols = [c for c in X.columns if X.loc[tr, c].notna().any()]
    C, _, _ = tune(X.loc[tr, cols], y[tr], g[tr], n_hgb=1)
    s = lr_model(C).fit(X.loc[tr, cols], y[tr]).predict_proba(X.loc[te, cols])[:, 1]
    tb = tiebreak(df.content_hash_id[te])
    got = p_at_k(y[te], s + tb, K)
    want = json.load(open(f"{OUT}/{receipt}"))["origins"][o]["metrics"]["logreg"]["p@50"]
    assert abs(got - want) < 1e-9, f"frozen logreg did not reproduce: {got} vs receipt {want}"
    t = X[te].copy()
    t["y"], t["score"], t["rule"] = y[te], s + tb, rule_score(X[te]).to_numpy()
    t["client"] = pd.factorize(g[te])[0]          # anonymous 0..n index, never the hash
    t["imp_m1"], t["imp_m2"], t["clk_m1"], t["clk_m2"] = (df.loc[te, c].to_numpy() for c in ["imp_m1", "imp_m2", "clk_m1", "clk_m2"])
    return t.sort_values("score", ascending=False)


def error_profile(t, k=K):
    top = t.head(k).copy()
    top["outcome"] = np.where(top.y == 1, "true positive", "false positive")
    top["imp_jump_x3"] = (np.abs(top.imp_growth) > np.log(3)).astype(int)
    top["one_month_only"] = ((top.imp_m1.fillna(0) == 0) | (top.imp_m2.fillna(0) == 0)).astype(int)
    agg = top.groupby("outcome").agg(pages=("y", "size"), median_position=("pos_f", "median"),
                                     median_ctr_vs_peers=("ctr_vs_client", "median"),
                                     share_impressions_jumped_3x=("imp_jump_x3", "mean"),
                                     share_active_one_month_only=("one_month_only", "mean"),
                                     clients=("client", "nunique"))
    return agg.round(3)


if __name__ == "__main__":
    t = scored_origin("2026-05")
    print(error_profile(t).to_string())
