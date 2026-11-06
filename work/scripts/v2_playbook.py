"""V2 live July 2026 queue: the selected model orders, the champion logistic gives the reason code.

Runs only after the sealed test month has been scored (the loader's live gate refuses otherwise).
Rules give the reason code, the model gives the order. Guardrails as in V1: pages younger than 90 days are
excluded and at most 10 pages per client enter the 50-page queue. The per-client cap was introduced in V1 after the
sealed month was opened; it is a playbook guardrail only and plays no part in any V2 selection or headline metric.

Outputs
  work/outputs/v2_queue_2026-07.csv        row-level, gitignored (contains IDs); its SHA-256 is the frozen artefact
  work/outputs/v2_playbook_summary.json    aggregates only, no IDs

  python v2_playbook.py            live queue (needs the freeze and v2_test.json)
  python v2_playbook.py --dry-run  code-path check on the features of the test origin, no outcomes, writes nothing
"""
import v2_pipeline as p
import v2_models as mm

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

import erosion_playbook as v1pb

MIN_AGE_DAYS = p.SPEC["playbook"]["min_age_days"]
QUEUE_K = p.SPEC["playbook"]["queue_size"]
FAMILIES = {k: list(v) for k, v in v1pb.FAMILIES.items()}
for _k, _v in p.SPEC["playbook"]["new_families"].items():
    FAMILIES[_k] = FAMILIES.get(_k, []) + [f for f in _v if f not in FAMILIES.get(_k, [])]
FAMILIES = {k: [f for f in v if f in p.FEATURE_LIST] for k, v in FAMILIES.items()}   # V2 model features only
ACTIONS = dict(v1pb.ACTIONS)
ACTIONS["STALE_ACTIVITY"] = ("Check whether the page still earns clicks for its query: a long quiet spell usually means lost "
                             "visibility or a shift in intent; re-read the live results page and refresh the content.")
ACTIONS["WEEKDAY_PROFILE"] = ("Traffic is concentrated on working days: check whether the query is a work-task question that an "
                              "AI answer can resolve, then test an answer-first rewrite.")


def family_of(feature):
    for fam, cols in FAMILIES.items():
        if feature in cols:
            return fam
    return "PAGE_PROFILE"


def build_queue(D, fold, selected, cfgs):
    P = p.fold_payload(D, fold)
    te = P["te"]
    need = {"LOGISTIC"} | ({"HGB_CLF"} if selected == "ENSEMBLE" else {selected})
    s = {}
    for fam in need:
        s[fam], _ = p.fit_score_family(fam, cfgs[fam], P)
    score = p.ensemble_score(s["LOGISTIC"], s["HGB_CLF"]) if selected == "ENSEMBLE" else s[selected]
    lg, _ = mm.fit_logistic(P["Xtr"], P["ytr"], cfgs["LOGISTIC"])      # champion logistic, same training rows
    contrib = mm.logistic_contributions(lg, P["Xte"])
    cols = np.array(P["cols"])
    top_i = contrib.argmax(axis=1)
    top_feat = np.where(contrib.max(axis=1) > 0, cols[top_i], "none_positive")
    fam = [family_of(f) if f != "none_positive" else "PAGE_PROFILE" for f in top_feat]
    d = D.df.iloc[te]
    young = (d.age_days < MIN_AGE_DAYS).to_numpy()
    full = score + D.tb[te]
    order = np.argsort(-full, kind="stable")
    q = pd.DataFrame({"rank": np.arange(1, len(order) + 1),
                      "content_hash_id": d.content_hash_id.to_numpy()[order], "client_hash_id": d.client_hash_id.to_numpy()[order],
                      "score": np.round(score[order], 6), "top_driver": top_feat[order],
                      "reason_code": np.array(fam)[order], "action": [ACTIONS[f] for f in np.array(fam)[order]],
                      "position": np.round(D.X.iloc[te].pos_f.to_numpy()[order], 1),
                      "clicks_last_month": d.clk_m1.fillna(0).to_numpy()[order], "age_days": d.age_days.to_numpy()[order],
                      "excluded_young": young[order]})
    keep = np.flatnonzero(~q.excluded_young.to_numpy())
    pick = keep[p.capped_top_idx(np.arange(len(keep)), q.client_hash_id.to_numpy()[keep], k=QUEUE_K, cap=p.CAP)]
    q["in_queue"] = False
    q.loc[pick, "in_queue"] = True
    return q, P, score


def v1_queue_overlap(q):
    f = p.OUT + "/erosion_queue_2026-07.csv"
    if not os.path.exists(f):
        return None
    old = pd.read_csv(f)
    old = old.sort_values("score", ascending=False)
    old_q = v1pb.capped_top(old[~old.excluded_young])
    return int(len(set(old_q.content_hash_id) & set(q.content_hash_id[q.in_queue])))


def cap_cost_table(selected):
    """Descriptive: uncapped vs 10-per-client-capped P@50 of the selected model on the evaluated months."""
    out = {}
    for name, f, key in [("validation_1", "v2_validation_folds.json", None), ("validation_2", "v2_validation_folds.json", None),
                         ("confirmation", "v2_confirmation_apr.json", None), ("test", "v2_test.json", None)]:
        j = json.load(open(f"{p.OUT}/{f}"))
        if name.startswith("validation"):
            m = j["folds"][name]["models"][selected]
        elif name == "confirmation":
            m = j["evaluation"]["models"][selected]
        else:
            m = j["evaluation"]["models"].get("SELECTED:" + selected) or j["evaluation"]["models"]["SELECTED:LOGISTIC"]
        out[name] = {"uncapped_p@50": m["p@50"], "capped_p@50": m["capped_p@50_descriptive"]}
    return out


def main(dry_run=False):
    if dry_run:
        D = p.Data("sealed")
        sel, cfgs = p.selected_cfgs()
        selected, fold = sel["selected_model"], "test"      # features of the test origin only; no outcomes are loaded
    else:
        fz = p.gate("live")
        D = p.Data("live")
        selected, cfgs, fold = fz["selected_model"], fz["selected_configs"], "live"
    q, P, score = build_queue(D, fold, selected, cfgs)
    queue = q[q.in_queue]
    print("pages", len(q), "excluded_young", int(q.excluded_young.sum()), "queue", len(queue), "clients", queue.client_hash_id.nunique(),
          "max per client", int(queue.client_hash_id.value_counts().max()), "reasons", queue.reason_code.value_counts().to_dict())
    if dry_run:
        return
    path = f"{p.OUT}/v2_queue_2026-07.csv"
    q.to_csv(path, index=False, lineterminator="\n")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    summary = {
        "spec_version": p.SPEC["spec_version"], "written_at": p.now(), "decision_origin": p.LIVE,
        "feature_window": [p.month_add(p.LIVE, -2), p.month_add(p.LIVE, -1)],
        "selected_model": selected, "selected_configs_used": cfgs,
        "ordering": "selected model score" + (" (rank average of LOGISTIC and HGB_CLF)" if selected == "ENSEMBLE" else ""),
        "reason_code_source": "largest positive standardised contribution of the champion LOGISTIC fitted on the same rows",
        "trained_on_origins": P["train"], "train_rows": int(len(P["Xtr"])), "n_features_used": len(P["cols"]),
        "dropped_all_missing_in_training": P["dropped_all_missing"],
        "eligible_pages": int(len(q)), "excluded_younger_than_90_days": int(q.excluded_young.sum()),
        "queue_size": int(len(queue)), "queue_clients": int(queue.client_hash_id.nunique()),
        "max_pages_one_client": int(queue.client_hash_id.value_counts().max()),
        "reason_codes": queue.reason_code.value_counts().to_dict(),
        "top_drivers": queue.top_driver.value_counts().to_dict(),
        "reason_codes_all_eligible": q.reason_code.value_counts().to_dict(),
        "queue_median_position": float(queue.position.median()),
        "clicks_last_month_on_queue_pages": float(queue.clicks_last_month.sum()),
        "clicks_at_risk_2_months_if_25pct_lost": round(float(queue.clicks_last_month.sum() * 2 * 0.25)),
        "score_quantiles_eligible": {k: round(float(v), 4) for k, v in pd.Series(score).quantile([.5, .9, .99]).items()},
        "queue_min_score": float(queue.score.min()),
        "guardrails": {"min_age_days": MIN_AGE_DAYS, "client_cap": p.CAP, "queue_size": QUEUE_K,
                       "cap_in_selection_or_headline": False,
                       "note": "the cap was introduced in V1 after the sealed month was opened; descriptive only here"},
        "cap_cost_descriptive": cap_cost_table(selected),
        "overlap_with_v1_queue_pages": v1_queue_overlap(q),
        "families": {k: v for k, v in FAMILIES.items()},
        "queue_file": {"name": "v2_queue_2026-07.csv", "rows": int(len(q)), "bytes": os.path.getsize(path),
                       "sha256": digest, "gitignored": True,
                       "status": "frozen artefact: any later change to this file changes the hash"}}
    p.dump(summary, "v2_playbook_summary.json")
    print("queue sha256", digest, flush=True)


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
