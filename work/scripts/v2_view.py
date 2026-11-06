"""Read-only view of the V1 and V2 receipts: every number the V2 documents quote is computed here, once.

Used by make_figures.py, make_paper.py, make_report.py and verify_evidence.py, so the figures, the paper, the report and the
checker cannot disagree. Pure Python and json: no warehouse, no model, no scikit-learn. Writes nothing.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
W0 = os.path.abspath(HERE + "/..")

DISCLOSURE = ("The test month (May 2026) was seen during V1, so it is not a pristine holdout. No V2 decision (features, models, "
              "hyperparameters, selection rule) used it; all were made on validation folds whose outcome windows end before the "
              "test window opens. The V2 test result may therefore be optimistic.")
BASELINES = ["rule_position_slipping", "mean_reversion_ctr_excess", "client_shock", "persistence_prev_ratio"]
BASELINE_LABEL = {"rule_position_slipping": "editor's rule (top position + slipping CTR)", "mean_reversion_ctr_excess": "CTR above peers",
                  "client_shock": "client-wide shock", "persistence_prev_ratio": "persistence"}
ROLE = {"2026-02": "validation", "2026-03": "validation", "2026-04": "confirmation", "2026-05": "test"}
ROLE_NOTE = {"validation": "tuned on", "confirmation": "reported only", "test": "scored once"}
MONTH_NAME = {"2026-02": "Feb", "2026-03": "Mar", "2026-04": "Apr", "2026-05": "May"}


def load(w=None):
    w = w or W0
    return lambda n: json.load(open(f"{w}/outputs/{n}", encoding="utf-8"))


def view(w=None):
    """All headline numbers as one dict of plain values. `w` is the work/ directory (default: this repo)."""
    rj = load(w)
    vf, cf, ts = rj("v2_validation_folds.json"), rj("v2_confirmation_apr.json"), rj("v2_test.json")
    sel, fz, lk, feats = rj("v2_selection.json"), rj("v2_freeze.json"), rj("v2_leakage_checks.json"), rj("v2_features.json")
    sec, pb, spec = rj("v2_secondary_label.json"), rj("v2_playbook_summary.json"), rj("v2_spec.json")
    v1d, v1s = rj("erosion_model_metrics.json")["origins"], rj("erosion_model_metrics_sealed.json")["origins"]
    v1 = {**v1d, **v1s}
    evals = {"2026-02": vf["folds"]["validation_1"], "2026-03": vf["folds"]["validation_2"], "2026-04": cf["evaluation"], "2026-05": ts["evaluation"]}
    months = []
    for m, ev in evals.items():
        mod = ev["models"]
        lg = mod["LOGISTIC"] if "LOGISTIC" in mod else mod["SELECTED:LOGISTIC"]
        base = {b: mod[b] for b in BASELINES}
        best_ap = max(base, key=lambda b: base[b]["ap"])
        best_p50 = max(base, key=lambda b: base[b]["p@50"])
        v1m = v1[m]["metrics"]["logreg"]
        months.append({
            "month": m, "role": ROLE[m], "ev": ev, "lg": lg, "base": base, "models": mod,
            "base_rate": ev["base_rate"], "noise": ev["noise_only_rate"], "n_eligible": ev["n_eligible"], "n_measurable": ev["n_measurable"],
            "lost_visibility": ev["lost_visibility_share"], "n_clients": ev["n_clients"], "eff_clients": ev["effective_clients"],
            "top3_share": ev["top3_client_row_share"], "best_ap_name": best_ap, "best_ap": base[best_ap]["ap"],
            "best_p50_name": best_p50, "best_p50": base[best_p50]["p@50"], "rule": base["rule_position_slipping"],
            "ap": lg["ap"], "ap_ci": lg["ap_ci"], "p50": lg["p@50"], "p50_ci": lg["p@50_ci"], "p20": lg["p@20"], "p100": lg["p@100"],
            "auc": lg["auc"], "lift": lg["lift"], "wc10": lg["within_client_p@10"], "clk": lg["clicks_lost_captured"],
            "top_client": lg["top_client_share_top50"], "clients_top50": lg["clients_in_top50"], "capped": lg["capped_p@50_descriptive"],
            "p50_unm": lg["p@50_unmeasurable_as_miss"], "d_ap": lg["delta_ap_vs_best_baseline"], "d_ap_ci": lg["delta_ap_vs_best_baseline_ci"],
            "d_p50_ci": lg["delta_p@50_vs_best_baseline_ci"], "loco": lg["loco_delta_ap_vs_best_baseline"],
            "ap_above_all": lg["ap"] > max(b["ap"] for b in base.values()),
            "p50_above_all": lg["p@50"] > max(b["p@50"] for b in base.values()),
            "d_ap_excl0": lg["delta_ap_vs_best_baseline_ci"][0] > 0, "d_p50_excl0": lg["delta_p@50_vs_best_baseline_ci"][0] > 0,
            "v1_ap": v1m["ap"], "v1_p50": v1m["p@50"], "v1_p50_ci": v1[m]["bootstrap"]["logreg"]["p@50_ci"],
            "v1_base_rate": v1[m]["base_rate"]})
    by = {m["month"]: m for m in months}
    ch = sel["challenger_vs_champion"]
    pen = {}
    for r in sel["configs"]["LOGISTIC"]:
        pn = r["cfg"]["penalty"]
        if pn not in pen or r["mean_ap"] > pen[pn]["mean_ap"]:
            pen[pn] = {"key": r["key"], "C": r["cfg"]["C"], "mean_ap": r["mean_ap"], "fold_ap": r["fold_ap"], "fold_p50": r["fold_p@50"]}
    t = ts["evaluation"]["models"]["SELECTED:LOGISTIC"]
    test = by["2026-05"]
    mean2 = vf["mean_over_two_folds"]
    tt = test["models"]
    boost = {}
    for fam in ("HGB_CLF", "POISSON_RATE"):
        its = [x for r in sel["configs"][fam] for x in r["median_best_iter"]]
        bestkey = sel["best_config_per_family"][fam]["key"]
        bestrow = next(r for r in sel["configs"][fam] if r["key"] == bestkey)
        boost[fam] = {"validation_median_iter_selected": bestrow["median_best_iter"], "validation_iter_range_all_configs": [min(its), max(its)],
                      "confirmation_iters": cf["fit_info"][fam]["best_iters"], "test_iters": ts["fit_info"][fam]["best_iters"]}
    eff = [m["eff_clients"] for m in months]
    out = {
        "months": months, "by": by, "test": test, "v1": v1, "sel": sel, "freeze": fz, "leak": lk, "feats": feats, "secondary": sec,
        "playbook": pb, "spec": spec, "vf": vf, "cf": cf, "ts": ts,
        "penalty_best": pen, "ch": ch, "boost": boost, "mean2": mean2,
        "n_features": feats["n_features"], "n_v1_features": feats["n_v1_features"], "n_new": len(feats["added"]),
        "eff_range": [min(eff), max(eff)], "base_rate_range": [min(m["base_rate"] for m in months), max(m["base_rate"] for m in months)],
        "ap_lead_months": sum(m["ap_above_all"] for m in months), "p50_lead_months": sum(m["p50_above_all"] for m in months),
        "d_ap_excl0_months": sum(m["d_ap_excl0"] for m in months), "d_p50_excl0_months": sum(m["d_p50_excl0"] for m in months),
        "p50_range": [min(m["p50"] for m in months), max(m["p50"] for m in months)],
        "ap_range": [min(m["ap"] for m in months), max(m["ap"] for m in months)],
        "unm_range": [min(m["lost_visibility"] for m in months), max(m["lost_visibility"] for m in months)],
        "closest": max(ch, key=lambda n: ch[n]["lower_bound"]),
        "n_comparisons": sel["n_challenger_comparisons"],
        "planted": lk["planted_future_feature_fold_2"], "rvg": lk["random_vs_client_grouped_feb_mar"],
        "spec_hash_pre": fz["spec_sha256_at_preregistration"], "spec_hash_now": fz["spec_sha256"],
        "spec_hash_changed": fz["spec_sha256_at_preregistration"] != fz["spec_sha256"],
        "freeze_before_test": fz["frozen_at"] < ts["scored_at"], "frozen_at": fz["frozen_at"], "scored_at": ts["scored_at"],
        "test_secondary": sec["test"], "test_models": tt,
    }
    out["closest_lb"] = ch[out["closest"]]["lower_bound"]
    out["rule_matches_v1"] = all(
        abs(by[m]["rule"]["ap"] - v1[m]["metrics"]["rule_position_slipping"]["ap"]) < 1e-9
        and abs(by[m]["rule"]["p@50"] - v1[m]["metrics"]["rule_position_slipping"]["p@50"]) < 1e-9 for m in by)
    return out


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def ci2(c):
    return f"{c[0]:.2f} to {c[1]:.2f}"


def ci3(c):
    return f"{c[0]:.3f} to {c[1]:.3f}"
