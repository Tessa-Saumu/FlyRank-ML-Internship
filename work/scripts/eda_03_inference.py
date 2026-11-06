"""EDA 3: inferential checks with client-clustered bootstrap; AI-onset population and within-client signal audit."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from wh import con
c = con(); R = {}; rng = np.random.default_rng(42)
def boot_med(df, col, B=2000):
    """median of col; 95% CI resampling CLIENTS (cluster bootstrap)."""
    g = {k: v[col].to_numpy() for k, v in df.groupby("client_hash_id")}; keys = list(g)
    st = [np.median(np.concatenate([g[k] for k in rng.choice(keys, len(keys))])) for _ in range(B)]
    return [round(float(np.median(df[col])), 4), round(float(np.percentile(st, 2.5)), 4), round(float(np.percentile(st, 97.5)), 4)]
# ---- A) within-page CTR change at stable position, post-num=100 window: Oct-Nov 2025 vs Mar-Apr 2026
w = c.sql("""WITH a AS (SELECT content_hash_id, client_hash_id, SUM(imp) imp, SUM(clicks) clk, SUM(sum_pos)*1.0/SUM(imp) pos FROM pm
   WHERE month IN ('2025-10','2025-11') GROUP BY ALL HAVING SUM(imp)>=200),
  b AS (SELECT content_hash_id, SUM(imp) imp, SUM(clicks) clk, SUM(sum_pos)*1.0/SUM(imp) pos FROM pm
   WHERE month IN ('2026-03','2026-04') GROUP BY ALL HAVING SUM(imp)>=200)
 SELECT a.client_hash_id, a.content_hash_id, ct.main_intent, a.pos pos_a, b.pos pos_b, a.imp imp_a, b.imp imp_b,
  a.clk*100.0/a.imp ctr_a, b.clk*100.0/b.imp ctr_b
 FROM a JOIN b USING (content_hash_id) JOIN content ct USING (content_hash_id) WHERE ABS(a.pos-b.pos)<=1""").df()
w["bucket"] = pd.cut(w.pos_a, [0, 3.5, 10.5, 20.5, 1e9], labels=["1-3", "4-10", "11-20", "21+"])
w["d_ctr"] = w.ctr_b - w.ctr_a
w["rel"] = np.log((w.ctr_b + 0.05) / (w.ctr_a + 0.05))
out = []
for b, g in w.groupby("bucket", observed=True):
    out.append({"bucket": b, "pages": len(g), "clients": g.client_hash_id.nunique(), "med_ctr_a": round(g.ctr_a.median(), 3),
                "med_ctr_b": round(g.ctr_b.median(), 3), "share_down": round((g.d_ctr < 0).mean(), 3),
                "med_log_ratio[ci]": boot_med(g, "rel")})
R["A_within_page_post_change"] = out; print(pd.DataFrame(out).to_string())
top = w[w.bucket.isin(["1-3", "4-10"])].copy(); top["intent"] = top.main_intent.fillna("unknown")
out = []
for (b, i), g in top.groupby(["bucket", "intent"], observed=True):
    if len(g) >= 80 and g.client_hash_id.nunique() >= 3:
        out.append({"bucket": b, "intent": i, "pages": len(g), "clients": g.client_hash_id.nunique(), "share_down": round((g.d_ctr < 0).mean(), 3), "med_log_ratio[ci]": boot_med(g, "rel")})
R["A_by_intent"] = out; print(pd.DataFrame(out).to_string())
# erosion label prevalence: CTR at stable pos falls by >=25% (log ratio <= ln .75)
R["A_erosion_rate_top10"] = round(float((top.rel <= np.log(0.75)).mean()), 3); print("erosion>=25% share top10:", R["A_erosion_rate_top10"], "n", len(top))
# ---- B) AI onset: client-level GA4 availability (client has GA4 on >=25 distinct days in both months)
c.execute("""CREATE TEMP TABLE cga AS SELECT client_hash_id, month, COUNT(DISTINCT report_date) d FROM daily
  WHERE month>='2025-11' AND ga4_data_available IS TRUE GROUP BY ALL""")
b = c.sql("""WITH m AS (SELECT pm.* FROM pm JOIN cga USING (client_hash_id, month) WHERE cga.d>=25),
 hist AS (SELECT content_hash_id, month, SUM(s_ai) OVER (PARTITION BY content_hash_id ORDER BY month ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) ai_hist FROM pm WHERE month>='2025-10'),
 q AS (SELECT content_hash_id, COUNT(*) nq, AVG(query_token_count) qtok FROM q90 GROUP BY 1)
 SELECT cur.client_hash_id, cur.content_hash_id, cur.month, (cur.s_ai>0)::INT y, COALESCE(h.ai_hist,0)>0 had_ai,
  cur.ai_gpt>0 y_gpt, cur.ai_gem>0 y_gem, cur.ai_ppx>0 y_ppx,
  prv.imp, prv.clicks, prv.sum_pos*1.0/NULLIF(prv.imp,0) pos, prv.s_org, prv.sess, prv.imp_days,
  ct.word_count, ct.keyword_token_count, ct.search_volume, ct.main_intent, DATE_DIFF('day', ct.content_created_date::DATE, (cur.month||'-01')::DATE) age
 FROM m cur JOIN m prv ON prv.content_hash_id=cur.content_hash_id AND (prv.month||'-01')::DATE=(cur.month||'-01')::DATE - INTERVAL 1 MONTH
 JOIN content ct ON ct.content_hash_id=cur.content_hash_id
 LEFT JOIN hist h ON h.content_hash_id=cur.content_hash_id AND h.month=cur.month
 WHERE prv.imp>0""").df()
b.to_parquet(__import__("wh").CACHE + "/ai_onset_frame.parquet")
base = b.groupby("month").agg(n=("y", "size"), clients=("client_hash_id", "nunique"), pos=("y", "sum"), base=("y", "mean"),
                              onset_n=("had_ai", lambda s: int((~s).sum()))).reset_index()
base["onset_pos"] = b[~b.had_ai].groupby("month").y.sum().values
base["onset_base"] = (base.onset_pos / base.onset_n).round(4)
R["B_population"] = json.loads(base.round(4).to_json(orient="records")); print(base.round(4).to_string())
on = b[~b.had_ai].copy()
on["ctr"] = on.clicks / on.imp; on["l_imp"] = np.log1p(on.imp); on["info"] = (on.main_intent == "informational").astype(int)
def within_client_auc(df, f):
    aucs, ws = [], []
    for _, g in df.groupby(["client_hash_id", "month"]):
        g = g.dropna(subset=[f])
        if g.y.nunique() == 2 and g.y.sum() >= 5:
            aucs.append(roc_auc_score(g.y, g[f])); ws.append(g.y.sum())
    return round(float(np.average(aucs, weights=ws)), 3) if aucs else None, len(aucs)
sig = []
for f in ["l_imp", "clicks", "imp_days", "s_org", "sess", "pos", "ctr", "word_count", "keyword_token_count", "search_volume", "age", "info"]:
    a, k = within_client_auc(on, f); sig.append({"feature": f, "within_client_auc": a, "client_months": k})
R["B_onset_signal_audit"] = sig; print(pd.DataFrame(sig).to_string())
# ---- C) platform vs Google rank: among page-months that got AI referrals, prior Google position profile by platform
plat = []
for p in ["y_gpt", "y_gem", "y_ppx"]:
    g = b[b[p] == True]
    plat.append({"platform": p[2:], "page_months": len(g), "clients": g.client_hash_id.nunique(),
                 "share_prior_top10": round(float((g.pos <= 10.5).mean()), 3), "median_prior_pos": round(float(g.pos.median()), 1),
                 "median_prior_imp": float(g.imp.median())})
plat.append({"platform": "all_eligible", "page_months": len(b), "share_prior_top10": round(float((b.pos <= 10.5).mean()), 3),
             "median_prior_pos": round(float(b.pos.median()), 1), "median_prior_imp": float(b.imp.median())})
R["C_platform_rank_profile"] = plat; print(pd.DataFrame(plat).to_string())
json.dump(R, open("../outputs/eda_03_inference.json", "w"), indent=1, default=str)
