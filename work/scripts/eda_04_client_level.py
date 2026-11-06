"""EDA 4: client-level robustness for decoupling (impression-weighted CTR ratio per client, stable-position pages)."""
import json
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon
from wh import con
c = con(); R = {}
df = c.sql("""WITH a AS (SELECT content_hash_id, client_hash_id, SUM(imp) imp, SUM(clicks) clk, SUM(sum_pos)*1.0/SUM(imp) pos FROM pm
   WHERE month IN ('2025-10','2025-11') GROUP BY ALL HAVING SUM(imp)>=200),
  b AS (SELECT content_hash_id, SUM(imp) imp, SUM(clicks) clk, SUM(sum_pos)*1.0/SUM(imp) pos FROM pm
   WHERE month IN ('2026-03','2026-04') GROUP BY ALL HAVING SUM(imp)>=200)
 SELECT a.client_hash_id, CASE WHEN a.pos<3.5 THEN '1-3' WHEN a.pos<10.5 THEN '4-10' ELSE '11+' END bucket, ct.main_intent intent,
   a.imp imp_a, a.clk clk_a, b.imp imp_b, b.clk clk_b
 FROM a JOIN b USING (content_hash_id) JOIN content ct USING (content_hash_id) WHERE ABS(a.pos-b.pos)<=1""").df()
rows = []
for key, sub in [("1-3", df[df.bucket == "1-3"]), ("4-10", df[df.bucket == "4-10"]), ("1-10", df[df.bucket != "11+"]),
                 ("1-10 informational", df[(df.bucket != "11+") & (df.intent == "informational")]),
                 ("1-10 transactional", df[(df.bucket != "11+") & (df.intent == "transactional")]),
                 ("1-10 commercial", df[(df.bucket != "11+") & (df.intent == "commercial")])]:
    g = sub.groupby("client_hash_id").agg(n=("imp_a", "size"), ia=("imp_a", "sum"), ca=("clk_a", "sum"), ib=("imp_b", "sum"), cb=("clk_b", "sum"))
    g = g[(g.n >= 20) & (g.ca >= 20)]
    r = np.log((g.cb / g.ib) / (g.ca / g.ia))
    pooled = (sub.clk_b.sum() / sub.imp_b.sum()) / (sub.clk_a.sum() / sub.imp_a.sum())
    rows.append({"segment": key, "clients": len(g), "clients_down": int((r < 0).sum()),
                 "sign_test_p": round(binomtest(int((r < 0).sum()), len(g)).pvalue, 4) if len(g) else None,
                 "wilcoxon_p": round(wilcoxon(r).pvalue, 4) if len(g) >= 6 else None,
                 "median_client_ctr_ratio": round(float(np.exp(np.median(r))), 3) if len(g) else None,
                 "pooled_ctr_ratio": round(float(pooled), 3)})
out = pd.DataFrame(rows); R["A_client_level"] = rows; print(out.to_string())
json.dump(R, open("../outputs/eda_04_client_level.json", "w"), indent=1, default=str)
