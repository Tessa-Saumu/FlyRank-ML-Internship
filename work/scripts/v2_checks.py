"""V2 leakage and split checks. Validation folds only (never the confirmation or test outcomes).

  1. timeline assertions: feature windows end before the origin; every training origin's outcome window ends
     before the scored origin's first day (for all five roles, which needs no outcome data)
  2. planted-future-feature harness on validation fold 2 (a noisy copy of the outcome ratio must lift AP / P@50)
  3. random-row KFold against client-grouped GroupKFold AP on the Feb + Mar rows (selected logistic config)
  4. seed invariance of the boosted models at max_features = 1.0 (justifies fitting those configs once)
Receipt: work/outputs/v2_leakage_checks.json
"""
import v2_pipeline as p
import v2_models as mm

import time

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold, KFold

SEED = mm.SEED


def p50_ap(D, te, s):
    meas = D.meas[te]
    ym, sm = D.y[te][meas], s[meas] + D.tb[te][meas]
    return {"ap": round(float(average_precision_score(ym, sm)), 4),
            "p@50": round(float(ym[np.argsort(-sm, kind="stable")[:50]].mean()), 4),
            "base_rate": round(float(ym.mean()), 4)}


def main():
    t0 = time.time()
    D = p.Data("sealed")
    sel, cfgs = p.selected_cfgs()
    out = {"spec_version": p.SPEC["spec_version"], "written_at": p.now(), "scope": "validation folds 1 and 2 only"}

    # ---- 1. timeline
    tl = {}
    for role, f in p.FOLDS.items():
        p.assert_timeline(f["train"], f["score"])
        o = f["score"]
        tl[role] = {"score_origin": o, "origin_first_day": str(p.first_day(o)),
                    "feature_months": [p.month_add(o, -2), p.month_add(o, -1)],
                    "feature_window_ends": str(p.last_day(p.month_add(o, -1))),
                    "train_outcome_windows_end": {t: str(p.last_day(p.month_add(t, 1))) for t in f["train"]},
                    "latest_training_outcome_end_before_origin": bool(p.last_day(p.month_add(f["train"][-1], 1)) < p.first_day(o))}
        assert p.last_day(p.month_add(o, -1)) < p.first_day(o)
    # sealed origins carry no outcome values at this stage, and no feature is an outcome column
    assert not D.meas[D.origin == p.SEALED].any()
    assert not (set(D.X.columns) & set(p.OUTCOME_COLS + p.PREV_COLS))
    out["timeline"] = tl
    out["timeline_assertions_passed"] = True
    out["max_daily_date_read_for_rhythm_features"] = {o: v["max_daily_date_used"] for o, v in D.rhythm_diag.items()}

    # ---- 2. planted future feature on fold 2
    P = p.fold_payload(D, "validation_2")
    rng = np.random.default_rng(SEED)
    leak = D.log_ratio + rng.normal(0, 0.3, len(D.df))
    leak = np.where(D.meas, leak, np.nan)
    Xtr_l, Xte_l = P["Xtr"].assign(planted=leak[P["tr"]]), P["Xte"].assign(planted=leak[P["te"]])
    plant = {}
    cfg_l, cfg_h = cfgs["LOGISTIC"], cfgs["HGB_CLF"]
    for name, (Xa, Xb) in {"honest": (P["Xtr"], P["Xte"]), "planted_future_ratio": (Xtr_l, Xte_l)}.items():
        m, _ = mm.fit_logistic(Xa, P["ytr"], cfg_l)
        plant[f"LOGISTIC_{name}"] = p50_ap(D, P["te"], mm.score_logistic(m, Xb))
        s, _ = mm.fit_boosted("clf", cfg_h, Xa, P["ytr"], None, P["val_mask"], Xb)
        plant[f"HGB_CLF_{name}"] = p50_ap(D, P["te"], s)
    for fam in ("LOGISTIC", "HGB_CLF"):
        hon, pl = plant[f"{fam}_honest"], plant[f"{fam}_planted_future_ratio"]
        plant[f"{fam}_ap_lift"] = round(pl["ap"] - hon["ap"], 4)
        plant[f"{fam}_p@50_lift"] = round(pl["p@50"] - hon["p@50"], 4)
    plant["criterion"] = "harness works if the planted copy lifts AP and P@50 by at least 0.15 and the honest model stays below 0.8"
    plant["harness_detects_leak"] = bool(all(plant[f"{f}_ap_lift"] >= 0.15 and plant[f"{f}_p@50_lift"] >= 0.15 for f in ("LOGISTIC", "HGB_CLF")))
    plant["honest_models_not_leaky"] = bool(all(plant[f"{f}_honest"]["p@50"] < 0.8 and plant[f"{f}_honest"]["ap"] < 0.8 for f in ("LOGISTIC", "HGB_CLF")))
    out["planted_future_feature_fold_2"] = plant

    # ---- 3. random-row KFold vs client-grouped KFold on the Feb + Mar rows
    rows = np.flatnonzero(np.isin(D.origin, ["2026-02", "2026-03"]) & D.meas)
    Xr = D.X.iloc[rows]
    Xr = Xr[[c for c in Xr.columns if Xr[c].notna().any()]].reset_index(drop=True)
    yr, gr = D.y[rows].astype(int), D.g[rows]
    cv = {}
    for name, splitter in {"random_row_kfold": KFold(5, shuffle=True, random_state=SEED), "client_group_kfold": GroupKFold(5)}.items():
        aps = []
        for a, b in splitter.split(Xr, yr, gr):
            m, _ = mm.fit_logistic(Xr.iloc[a], yr[a], cfg_l)
            aps.append(float(average_precision_score(yr[b], mm.score_logistic(m, Xr.iloc[b]))))
        cv[name] = {"ap_mean": round(float(np.mean(aps)), 4), "ap_std": round(float(np.std(aps)), 4), "folds": 5}
    cv["base_rate"] = round(float(yr.mean()), 4)
    cv["rows"] = int(len(rows))
    cv["random_minus_grouped_ap"] = round(cv["random_row_kfold"]["ap_mean"] - cv["client_group_kfold"]["ap_mean"], 4)
    cv["reading"] = "random rows put the same page, in adjacent origins, and the same client on both sides of the split; the grouped number is the honest one"
    out["random_vs_client_grouped_feb_mar"] = cv

    # ---- 4. seed invariance
    cfg1 = dict(mm.sample_hgb_configs()[0])
    cfg1["max_features"] = 1.0
    a, _ = mm.fit_boosted("clf", cfg1, P["Xtr"], P["ytr"], None, P["val_mask"], P["Xte"], seeds=[42])
    b, _ = mm.fit_boosted("clf", cfg1, P["Xtr"], P["ytr"], None, P["val_mask"], P["Xte"], seeds=[43])
    cfg2 = dict(cfg1, max_features=0.5)
    c, _ = mm.fit_boosted("clf", cfg2, P["Xtr"], P["ytr"], None, P["val_mask"], P["Xte"], seeds=[42])
    d, _ = mm.fit_boosted("clf", cfg2, P["Xtr"], P["ytr"], None, P["val_mask"], P["Xte"], seeds=[43])
    out["seed_invariance"] = {"max_features_1.0_identical_across_seeds": bool(np.array_equal(a, b)),
                              "max_features_0.5_differs_across_seeds": bool(not np.array_equal(c, d))}
    assert out["seed_invariance"]["max_features_1.0_identical_across_seeds"]

    out["runtime_seconds"] = round(time.time() - t0)
    p.dump(out, "v2_leakage_checks.json")
    print(p.json.dumps({k: out[k] for k in ("planted_future_feature_fold_2", "random_vs_client_grouped_feb_mar", "seed_invariance")}, indent=1, default=p._jd))


if __name__ == "__main__":
    main()
