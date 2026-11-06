"""EDA 2: decoupling at fixed position, generating-LLM comparison, AI-referral signal audit."""
import json
from wh import con
c = con(); R = {}
def show(name, sql):
    df = c.sql(sql).df(); R[name] = json.loads(df.to_json(orient="records", date_format="iso")); print(f"\n## {name}\n" + df.to_string(max_colwidth=26))
# 1) CTR at fixed position bucket, balanced client panel (clients with GSC impressions every month 2025-07..2026-06)
c.execute("""CREATE TEMP TABLE bal AS SELECT client_hash_id FROM pm WHERE month BETWEEN '2025-07' AND '2026-06' AND imp>0
  GROUP BY 1 HAVING COUNT(DISTINCT month)=12""")
show("balanced_panel_size", "SELECT COUNT(*) clients FROM bal")
show("ctr_by_pos_month_balanced", """WITH x AS (SELECT month, sum_pos*1.0/imp pos, imp, clicks FROM pm JOIN bal USING (client_hash_id)
  WHERE month BETWEEN '2025-07' AND '2026-06' AND imp>=20)
 SELECT month, SUM(imp)/1e6 imp_m,
  ROUND(100.0*SUM(clicks) FILTER (WHERE pos<3.5)/SUM(imp) FILTER (WHERE pos<3.5),3) ctr_p1_3,
  ROUND(100.0*SUM(clicks) FILTER (WHERE pos>=3.5 AND pos<10.5)/SUM(imp) FILTER (WHERE pos>=3.5 AND pos<10.5),3) ctr_p4_10,
  ROUND(100.0*SUM(clicks) FILTER (WHERE pos>=10.5 AND pos<20.5)/SUM(imp) FILTER (WHERE pos>=10.5 AND pos<20.5),3) ctr_p11_20,
  ROUND(100.0*SUM(imp) FILTER (WHERE pos<10.5)/SUM(imp),1) pct_imp_top10
 FROM x GROUP BY 1 ORDER BY 1""")
# within-page paired: same page, similar position (|dpos|<=1), early vs late
show("within_page_ctr_change", """WITH a AS (SELECT content_hash_id, client_hash_id, SUM(imp) imp, SUM(clicks) clk, SUM(sum_pos)*1.0/SUM(imp) pos FROM pm
   WHERE month BETWEEN '2025-10' AND '2025-12' GROUP BY ALL HAVING SUM(imp)>=200),
  b AS (SELECT content_hash_id, SUM(imp) imp, SUM(clicks) clk, SUM(sum_pos)*1.0/SUM(imp) pos FROM pm
   WHERE month BETWEEN '2026-04' AND '2026-06' GROUP BY ALL HAVING SUM(imp)>=200),
  j AS (SELECT a.client_hash_id, CASE WHEN a.pos<3.5 THEN '1-3' WHEN a.pos<10.5 THEN '4-10' WHEN a.pos<20.5 THEN '11-20' ELSE '21+' END bucket,
   a.clk*100.0/a.imp ctr_a, b.clk*100.0/b.imp ctr_b FROM a JOIN b USING (content_hash_id) WHERE ABS(a.pos-b.pos)<=1)
 SELECT bucket, COUNT(*) pages, COUNT(DISTINCT client_hash_id) clients, ROUND(MEDIAN(ctr_a),3) med_ctr_q4_25, ROUND(MEDIAN(ctr_b),3) med_ctr_q2_26,
  ROUND(AVG((ctr_b<ctr_a)::INT),3) share_pages_ctr_down, ROUND(MEDIAN(ctr_b-ctr_a),3) med_delta FROM j GROUP BY 1 ORDER BY 1""")
# 2) query length vs CTR at fixed position (90d query table): last30 vs prev30
show("query_len_ctr", """SELECT LEAST(query_token_count,8) qtok, COUNT(*) n, SUM(impressions_90d)/1e6 imp_m,
  ROUND(100.0*SUM(clicks_90d) FILTER (WHERE avg_position_90d<10.5)/SUM(impressions_90d) FILTER (WHERE avg_position_90d<10.5),3) ctr_top10,
  ROUND(100.0*SUM(clicks_prev30) FILTER (WHERE avg_position_prev30<10.5 AND ABS(avg_position_last30-avg_position_prev30)<=1)/NULLIF(SUM(impressions_prev30) FILTER (WHERE avg_position_prev30<10.5 AND ABS(avg_position_last30-avg_position_prev30)<=1),0),3) ctr_prev30_stablepos,
  ROUND(100.0*SUM(clicks_last30) FILTER (WHERE avg_position_prev30<10.5 AND ABS(avg_position_last30-avg_position_prev30)<=1)/NULLIF(SUM(impressions_last30) FILTER (WHERE avg_position_prev30<10.5 AND ABS(avg_position_last30-avg_position_prev30)<=1),0),3) ctr_last30_stablepos
 FROM q90 GROUP BY 1 ORDER BY 1""")
show("q90_window", "SELECT MIN(window_start)::DATE ws, MAX(window_end)::DATE we, COUNT(DISTINCT content_hash_id) pages, COUNT(DISTINCT client_hash_id) clients FROM q90")
# 3) generating LLM: within client x creation-month cells that used >=2 models
c.execute("""CREATE TEMP TABLE pg AS SELECT ct.client_hash_id, ct.content_hash_id, ct.model_used, ct.content_type, ct.main_intent, ct.word_count,
  ct.keyword_token_count, ct.search_volume, DATE_TRUNC('month', ct.content_created_date) cmonth, ct.content_created_date cdate
  FROM content ct WHERE ct.is_published IS TRUE AND ct.is_deleted IS NOT TRUE""")
show("model_time_overlap", """SELECT strftime(cmonth,'%Y-%m') cm, model_used, COUNT(*) n FROM pg WHERE cmonth>=DATE '2025-06-01' AND model_used IS NOT NULL
  GROUP BY ALL HAVING COUNT(*)>=500 ORDER BY 1,3 DESC""")
# outcome: impressions/clicks in days 31-90 after creation (from daily), for pages created 2025-06..2026-03
c.execute("""CREATE TEMP TABLE early AS SELECT d.content_hash_id, SUM(gsc_impressions) imp, SUM(gsc_clicks) clk,
   SUM(gsc_sum_position)*1.0/NULLIF(SUM(gsc_impressions),0) pos, SUM((gsc_data_available IS TRUE)::INT) gsc_days
  FROM daily d JOIN pg USING (content_hash_id)
  WHERE pg.cdate BETWEEN '2025-06-01' AND '2026-03-31' AND d.report_date BETWEEN pg.cdate + INTERVAL 31 DAY AND pg.cdate + INTERVAL 90 DAY
  GROUP BY 1""")
show("model_within_cell", """WITH x AS (SELECT pg.*, COALESCE(e.imp,0) imp, COALESCE(e.clk,0) clk, e.gsc_days FROM pg LEFT JOIN early e USING (content_hash_id)
   WHERE pg.cdate BETWEEN '2025-06-01' AND '2026-03-31' AND model_used IN ('gpt-4o-mini','gpt-5-mini','gemini-2.5-flash','gemini-3-flash-preview')
     AND e.gsc_days>=50),
  cell AS (SELECT client_hash_id, cmonth FROM x GROUP BY ALL HAVING COUNT(DISTINCT model_used)>=2),
  y AS (SELECT x.*, MEDIAN(LN(1+imp)) OVER (PARTITION BY x.client_hash_id, x.cmonth) cell_med FROM x JOIN cell USING (client_hash_id, cmonth))
 SELECT model_used, COUNT(*) pages, COUNT(DISTINCT client_hash_id) clients, COUNT(DISTINCT client_hash_id||cmonth::VARCHAR) cells,
  ROUND(MEDIAN(imp),0) med_imp_d31_90, ROUND(AVG((imp>0)::INT),3) share_any_imp, ROUND(AVG(LN(1+imp)-cell_med),3) mean_log_imp_vs_cell, MEDIAN(word_count) med_words
 FROM y GROUP BY 1 ORDER BY 2 DESC""")
# 4) AI-referral signal audit at page level (Feb-Jun 2026, GA4 clients): features from month m-1, outcome any AI in month m
c.execute("""CREATE TEMP TABLE aud AS WITH m AS (SELECT * FROM pm WHERE month>='2026-01'),
  q AS (SELECT content_hash_id, COUNT(*) nq, AVG(query_token_count) qtok, AVG((query_token_count>=6)::INT) long_share,
         ANY_VALUE(rare_impressions_share) rare_share FROM q90 GROUP BY 1)
  SELECT cur.client_hash_id, cur.content_hash_id, cur.month, (cur.s_ai>0)::INT y, (prv.s_ai>0)::INT prev_ai,
   LN(1+prv.imp) l_imp, LN(1+prv.clicks) l_clk, prv.sum_pos*1.0/NULLIF(prv.imp,0) pos, prv.clicks*100.0/NULLIF(prv.imp,0) ctr,
   LN(1+prv.s_org) l_org, LN(1+prv.sess) l_sess, pg.word_count, pg.keyword_token_count, LN(1+pg.search_volume) l_sv,
   DATE_DIFF('day', pg.cdate, (cur.month||'-01')::DATE) age, (pg.main_intent='informational')::INT info, pg.model_used,
   q.nq, q.qtok, q.long_share
  FROM m cur JOIN m prv ON prv.content_hash_id=cur.content_hash_id AND (prv.month||'-01')::DATE = (cur.month||'-01')::DATE - INTERVAL 1 MONTH
  JOIN pg ON pg.content_hash_id=cur.content_hash_id LEFT JOIN q ON q.content_hash_id=cur.content_hash_id
  WHERE cur.ga4_days>=20 AND prv.ga4_days>=20""")
show("audit_base", "SELECT month, COUNT(*) n, SUM(y) pos, ROUND(AVG(y),4) base, ROUND(AVG(y) FILTER (WHERE prev_ai=0),4) base_onset, COUNT(DISTINCT client_hash_id) clients FROM aud GROUP BY 1 ORDER BY 1")
show("audit_persistence", "SELECT prev_ai, COUNT(*) n, ROUND(AVG(y),4) p_ai FROM aud GROUP BY 1")
json.dump(R, open("../outputs/eda_02_signals.json", "w"), indent=1, default=str)
c.execute(f"COPY aud TO '{__import__('wh').CACHE}/audit_frame.parquet' (FORMAT parquet)")
