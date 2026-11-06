"""ML-07 baseline artefacts and the ML-10 live review queue (decision origin 2026-07).

Principle: rules give the reason code, the model gives the order. The frozen logistic model
(retrained on every origin whose label has matured by July 2026, i.e. origins <= 2026-05, with
the same inner client-grouped tuning) orders the pages; each page's largest positive model
contribution, mapped to a plain-language family, becomes its reason code and action.

Outputs
  work/outputs/baseline_action_score.csv    (gitignored row-level queue, rule only)
  work/outputs/baseline_metrics.json        (rule receipts per forward origin)
  work/outputs/erosion_queue_2026-07.csv    (gitignored row-level queue, model + reason code)
  work/outputs/erosion_playbook_summary.json
"""
import json

import numpy as np
import pandas as pd

from erosion_frame import LIVE_ORIGIN, ORIGINS, month_add
from erosion_model import (FEATURES, K, OUT, load, lr_model, make_features, rule_score, tiebreak, tune)

MIN_AGE_DAYS = 90
CLIENT_CAP = 10  # max pages per client in one month's queue (budget fairness)
FAMILIES = {
    "TOP_POSITION_EXPOSED": ["pos_f"],
    "CTR_ALREADY_SLIPPING": ["ctr_growth", "ctr_recency", "page_prev_ratio"],
    "RANK_MOVING": ["pos_delta", "pos_std"],
    "DEMAND_SHIFT": ["imp_growth", "imp_growth_prev", "imp_l14_share", "spikiness"],
    "CTR_ABOVE_PEERS": ["ctr_vs_client", "ctr_vs_ref"],
    "CLIENT_WIDE_DECLINE": ["c_ctr_growth", "client_prev_erode_rate", "c_l_imp", "c_l_pages"],
}
ACTIONS = {
    "TOP_POSITION_EXPOSED": "Make the answer citable: answer-first summary, FAQ block and structured data, so the page is the source an AI answer quotes.",
    "CTR_ALREADY_SLIPPING": "Check the live results page for an AI answer; rewrite title and description toward the click an AI answer cannot satisfy (comparison, pricing, tool, example).",
    "RANK_MOVING": "Hold: do not rewrite while rank is still moving; re-score next month.",
    "DEMAND_SHIFT": "Check seasonality and demand first; the loss may be demand, not AI answers.",
    "CTR_ABOVE_PEERS": "Protect: the page out-performs its position; review recent snippet and AI-answer changes before editing.",
    "CLIENT_WIDE_DECLINE": "Escalate to the account lead: the whole site is losing clicks at stable rank; page edits alone will not fix it.",
    "PAGE_PROFILE": "Editor judgement: no single driver; check intent match and freshness.",
}
RULE_CODES = {(True, True): "TOP3_SLIPPING", (False, True): "TOP10_SLIPPING", (True, False): "TOP3_EXPOSED", (False, False): "TOP10_WATCH"}
RULE_ACTIONS = {"TOP3_SLIPPING": "review now", "TOP10_SLIPPING": "review now", "TOP3_EXPOSED": "review if budget allows", "TOP10_WATCH": "monitor"}


def family_of(feature):
    for fam, cols in FAMILIES.items():
        if feature in cols:
            return fam
    return "PAGE_PROFILE"


def capped_top(order_df, k=K, cap=CLIENT_CAP):
    """Walk the ranking, skipping a page once its client already holds `cap` slots."""
    taken, out = {}, []
    for i, c in zip(order_df.index, order_df.client_hash_id):
        if taken.get(c, 0) < cap:
            out.append(i); taken[c] = taken.get(c, 0) + 1
        if len(out) == k:
            break
    return order_df.loc[out]


def cap_cost(df, X, y, g):
    """P@50 uncapped vs capped for the frozen logreg on each forward origin (dev + sealed)."""
    res = {}
    for o in ["2026-02", "2026-03", "2026-04", "2026-05"]:
        tr = (df.measurable.fillna(False) & df.origin.isin([m for m in ORIGINS if m <= month_add(o, -2)])).to_numpy()
        te = (df.measurable.fillna(False) & (df.origin == o)).to_numpy()
        cols = [c for c in X.columns if X.loc[tr, c].notna().any()]
        C, _, _ = tune(X.loc[tr, cols], y[tr], g[tr], n_hgb=1)
        s = lr_model(C).fit(X.loc[tr, cols], y[tr]).predict_proba(X.loc[te, cols])[:, 1]
        d = pd.DataFrame({"y": y[te], "client_hash_id": g[te], "s": s}).sort_values("s", ascending=False)
        res[o] = {"uncapped_p@50": round(float(d.head(K).y.mean()), 3), "capped_p@50": round(float(capped_top(d).y.mean()), 3),
                  "uncapped_clients": int(d.head(K).client_hash_id.nunique()), "capped_clients": int(capped_top(d).client_hash_id.nunique())}
    return res


def main():
    df, grid, vals = load()
    X = make_features(df, grid, vals)
    live = (df.origin == LIVE_ORIGIN).to_numpy()
    young = (df.age_days < MIN_AGE_DAYS).to_numpy()

    # ---------------- ML-07: the frozen rule as a queue with one reason code and an action label
    Xl = X[live]
    rs = rule_score(Xl) + tiebreak(df.content_hash_id[live])
    code = [RULE_CODES[(p <= 3.5, g < -0.05)] for p, g in zip(Xl.pos_f, Xl.ctr_growth)]
    bq = pd.DataFrame({"content_hash_id": df.content_hash_id[live].values, "origin": LIVE_ORIGIN,
                       "rule_score": np.round(rs, 4), "reason_code": code,
                       "action_label": [RULE_ACTIONS[c] for c in code]}).sort_values("rule_score", ascending=False)
    bq.to_csv(f"{OUT}/baseline_action_score.csv", index=False)
    dev = json.load(open(f"{OUT}/erosion_model_metrics.json"))["origins"]
    sealed = json.load(open(f"{OUT}/erosion_model_metrics_sealed.json"))["origins"]
    allo = {**dev, **sealed}
    json.dump({"rule": "position tier (top 3 = 2, 4-6 = 1, 7-10 = 0) + max(0, -CTR change month -1 vs -2), hash tiebreak",
               "per_origin": {o: {"base_rate": v["base_rate"], **{k: v["metrics"]["rule_position_slipping"][k] for k in ["p@20", "p@50", "p@100", "ap", "auc"]},
                                  "distinct_scores_in_top50": v["rule_distinct_scores_top50"]} for o, v in allo.items()},
               "live_queue_reason_codes": pd.Series(code).value_counts().to_dict()},
              open(f"{OUT}/baseline_metrics.json", "w"), indent=1)

    # ---------------- ML-10: frozen model retrained on every matured origin, scored at the live origin
    m = df.measurable.fillna(False).to_numpy() & df.origin.isin([o for o in ORIGINS if o <= month_add(LIVE_ORIGIN, -2)]).to_numpy()
    y = df.erode.fillna(0).to_numpy().astype(int)
    g = df.client_hash_id.to_numpy()
    C, _, tuning = tune(X[m], y[m], g[m], n_hgb=1)
    lr = lr_model(C).fit(X[m], y[m])
    s = lr.predict_proba(X[live])[:, 1]
    Z = lr[:-1].transform(X[live])[:, :len(FEATURES)]
    contrib = Z * lr[-1].coef_[0][:len(FEATURES)]
    top_feat = np.array(FEATURES and list(FEATURES))[np.argmax(contrib, axis=1)]
    fam = [family_of(f) for f in top_feat]
    q = pd.DataFrame({"content_hash_id": df.content_hash_id[live].values, "client_hash_id": g[live],
                      "score": np.round(s, 4), "top_driver": top_feat, "reason_code": fam,
                      "action": [ACTIONS[f] for f in fam], "position": np.round(Xl.pos_f.values, 1),
                      "clicks_last_month": df.clk_m1[live].fillna(0).values, "excluded_young": young[live]})
    q = q.sort_values("score", ascending=False)
    q["rank"] = np.arange(1, len(q) + 1)
    queue = capped_top(q[~q.excluded_young])
    q.to_csv(f"{OUT}/erosion_queue_{LIVE_ORIGIN}.csv", index=False)
    clicks_at_risk = float(queue.clicks_last_month.sum() * 2 * 0.25)
    summary = {
        "decision_origin": LIVE_ORIGIN, "feature_window": [month_add(LIVE_ORIGIN, -2), month_add(LIVE_ORIGIN, -1)],
        "trained_on_origins": sorted(df.origin[m].unique().tolist()), "train_rows": int(m.sum()), "lr_C": C,
        "eligible_pages": int(live.sum()), "excluded_younger_than_90_days": int((live & young).sum()),
        "queue_size": int(len(queue)), "queue_clients": int(queue.client_hash_id.nunique()),
        "max_pages_one_client": int(queue.client_hash_id.value_counts().max()),
        "reason_codes": queue.reason_code.value_counts().to_dict(),
        "queue_median_position": float(queue.position.median()),
        "clicks_at_risk_2_months_if_25pct_lost": round(clicks_at_risk),
        "score_quantiles_eligible": {k: round(float(v), 4) for k, v in pd.Series(s).quantile([.5, .9, .99]).items()},
        "queue_min_score": float(queue.score.min()),
    }
    summary["client_cap"] = CLIENT_CAP
    summary["cap_cost_forward_origins"] = cap_cost(df, X, y, g)
    json.dump(summary, open(f"{OUT}/erosion_playbook_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
