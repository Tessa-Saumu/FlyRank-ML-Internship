"""Inferential core: how much did CTR at a FIXED position change month by month?

Model (page-month panel, Oct 2025 - Jun 2026, i.e. after Google's Sept-2025 num=100 change):
    log((clicks + 0.5) / impressions) = page FE + month FE + position-bin FE + error
estimated by weighted within-page demeaning (weights = impressions), with standard errors
clustered by CLIENT (the unit that shares shocks). exp(month coefficient) is the CTR multiplier
relative to October 2025 for the same page at the same position bin.
Why: it answers "is the decoupling real, or composition?" before any model is trained; page FE
remove page mix, position bins remove rank changes, client clustering stops thousands of pages
from one client masquerading as independent evidence.
Receipt: work/outputs/panel_fe.json
"""
import json
import os

import numpy as np
import pandas as pd

from wh import con

OUT = os.path.dirname(os.path.abspath(__file__)) + "/../outputs"
MONTHS = ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
POS_EDGES = [0, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20.5]


def load_panel():
    c = con()
    return c.sql(f"""SELECT pm.client_hash_id, pm.content_hash_id, pm.month, pm.imp, pm.clicks, pm.sum_pos*1.0/pm.imp pos,
          ct.main_intent FROM pm JOIN content ct USING (content_hash_id)
        WHERE pm.month BETWEEN '{MONTHS[0]}' AND '{MONTHS[-1]}' AND pm.imp >= 100 AND pm.sum_pos*1.0/pm.imp <= 20.5""").df()


def fe_month_effects(d):
    d = d[d.groupby("content_hash_id").month.transform("nunique") >= 2].copy()
    y = np.log((d.clicks + 0.5) / d.imp).to_numpy()
    w = d.imp.to_numpy().astype(float)
    M = pd.get_dummies(pd.Categorical(d.month, categories=MONTHS)).iloc[:, 1:].to_numpy(float)
    P = pd.get_dummies(pd.cut(d.pos, POS_EDGES)).iloc[:, 1:].to_numpy(float)
    X = np.hstack([M, P])
    keep = X.std(0) > 0
    X = X[:, keep]
    g = d.content_hash_id.to_numpy()
    codes, inv = np.unique(g, return_inverse=True)
    def demean(A):
        sw = np.bincount(inv, weights=w)
        if A.ndim == 1:
            return A - (np.bincount(inv, weights=w * A) / sw)[inv]
        return A - np.column_stack([np.bincount(inv, weights=w * A[:, j]) / sw for j in range(A.shape[1])])[inv]
    yd, Xd = demean(y), demean(X)
    XtW = Xd.T * w
    A = np.linalg.pinv(XtW @ Xd)
    beta = A @ (XtW @ yd)
    e = yd - Xd @ beta
    cl = d.client_hash_id.to_numpy()
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(cl):
        m = cl == c
        s = (Xd[m].T * w[m]) @ e[m]
        meat += np.outer(s, s)
    G = len(np.unique(cl))
    V = A @ meat @ A * G / (G - 1)
    se = np.sqrt(np.diag(V))
    nm = len(MONTHS) - 1
    rows = [{"month": MONTHS[0], "ctr_multiplier": 1.0, "ci95": [1.0, 1.0]}]
    for j in range(nm):
        b, s = beta[j], se[j]
        rows.append({"month": MONTHS[j + 1], "ctr_multiplier": round(float(np.exp(b)), 3),
                     "ci95": [round(float(np.exp(b - 1.96 * s)), 3), round(float(np.exp(b + 1.96 * s)), 3)]})
    return {"page_months": int(len(d)), "pages": int(len(codes)), "clients": int(G), "effects": rows}


def main():
    d = load_panel()
    res = {"spec": "log CTR ~ page FE + month FE + position-bin FE; impression weights; SE clustered by client",
           "all_top20": fe_month_effects(d)}
    tier = pd.cut(d.pos, [0, 3.5, 10.5, 20.5], labels=["pos_1_3", "pos_4_10", "pos_11_20"])
    for t in ["pos_1_3", "pos_4_10", "pos_11_20"]:
        res[t] = fe_month_effects(d[tier == t])
    for i in ["informational", "transactional", "commercial"]:
        res[f"intent_{i}"] = fe_month_effects(d[d.main_intent == i])
    json.dump(res, open(OUT + "/panel_fe.json", "w"), indent=1)
    for k, v in res.items():
        if isinstance(v, dict):
            e = {r["month"]: r["ctr_multiplier"] for r in v["effects"]}
            print(k, v["clients"], "clients", v["pages"], "pages |", e)
    return res


if __name__ == "__main__":
    main()
