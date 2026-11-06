"""EDA 1: AI-referral landscape - concentration, cohort-held trend, engagement, content metadata."""
import json
from wh import con
c = con()
R = {}
def show(name, sql):
    df = c.sql(sql).df(); R[name] = json.loads(df.to_json(orient="records", date_format="iso")); print(f"\n## {name}\n" + df.to_string(max_colwidth=28))
# provider_used is NEVER selected: it embeds client brand names in some rows (privacy).
show("content_model_mix", """SELECT model_used, COUNT(*) n, COUNT(DISTINCT client_hash_id) clients,
  MIN(content_created_date)::DATE first_created, MAX(content_created_date)::DATE last_created FROM content GROUP BY ALL ORDER BY n DESC LIMIT 15""")
show("content_type_intent", "SELECT content_type, main_intent, COUNT(*) n FROM content GROUP BY ALL ORDER BY n DESC LIMIT 12")
show("content_fields", """SELECT COUNT(*) n, MEDIAN(word_count) med_words, AVG(keyword_token_count) kw_tokens, MEDIAN(search_volume) med_sv,
  AVG((last_optimized_date IS NOT NULL)::INT) share_optimized, MIN(last_optimized_date)::DATE opt_min, MAX(last_optimized_date)::DATE opt_max FROM content""")
# --- AI concentration over the AI-observable window (Oct 2025+)
show("ai_concentration", """WITH p AS (SELECT content_hash_id, client_hash_id, SUM(s_ai) ai, SUM(sess) sess FROM pm WHERE month>='2025-10' GROUP BY ALL),
 a AS (SELECT *, ROW_NUMBER() OVER (ORDER BY ai DESC) rk, COUNT(*) FILTER (WHERE ai>0) OVER () n_ai, SUM(ai) OVER () tot FROM p)
 SELECT COUNT(*) FILTER (WHERE sess>0) pages_with_ga4_sessions, MAX(n_ai) pages_with_any_ai, MAX(tot) total_ai,
  SUM(ai) FILTER (WHERE rk<=10)/MAX(tot) top10_pages_share, SUM(ai) FILTER (WHERE rk<=n_ai*0.01)/MAX(tot) top1pct_share,
  SUM(ai) FILTER (WHERE rk<=n_ai*0.10)/MAX(tot) top10pct_share, COUNT(*) FILTER (WHERE ai=1) pages_exactly_1 FROM a""")
show("ai_by_client_top", """WITH x AS (SELECT client_hash_id, SUM(s_ai) ai, SUM(sess) sess FROM pm WHERE month>='2025-10' GROUP BY 1)
 SELECT ROW_NUMBER() OVER (ORDER BY ai DESC) client_rank, ai, ai/SUM(ai) OVER () ai_share, ai*1000.0/NULLIF(sess,0) ai_per_1k_sess FROM x ORDER BY ai DESC LIMIT 8""")
show("june_spike_top_pages_share", """WITH x AS (SELECT content_hash_id, SUM(ai_gpt) g FROM pm WHERE month='2026-06' GROUP BY 1)
 SELECT SUM(g) total_gpt, (SELECT SUM(g) FROM (SELECT g FROM x ORDER BY g DESC LIMIT 1)) top1, (SELECT SUM(g) FROM (SELECT g FROM x ORDER BY g DESC LIMIT 5)) top5 FROM x""")
# --- fixed cohort: clients with GA4 on >=20 days in 2025-12 and in every month to 2026-06
c.execute("""CREATE TEMP TABLE cohort AS SELECT client_hash_id FROM (
  SELECT client_hash_id, month, MAX(ga4_days) d FROM pm WHERE month BETWEEN '2025-12' AND '2026-06' GROUP BY ALL)
  WHERE d>=20 GROUP BY 1 HAVING COUNT(*)=7""")
show("cohort_trend", """SELECT month, COUNT(DISTINCT client_hash_id) clients, SUM(sess) sess, SUM(s_org) org, SUM(s_ai) ai,
  ROUND(SUM(s_ai)*1000.0/SUM(sess),2) ai_per_1k_sess, ROUND(SUM(s_ai)*1000.0/SUM(s_org),2) ai_per_1k_org,
  COUNT(DISTINCT CASE WHEN s_ai>0 THEN content_hash_id END) ai_pages,
  ROUND(SUM(ai_gpt)*100.0/SUM(s_ai),1) gpt_pct, ROUND(SUM(ai_gem)*100.0/SUM(s_ai),1) gem_pct, ROUND(SUM(ai_ppx)*100.0/SUM(s_ai),1) ppx_pct,
  ROUND(SUM(clicks)*100.0/SUM(imp),3) ctr_pct, ROUND(SUM(sum_pos)*1.0/SUM(imp),1) pos
  FROM pm JOIN cohort USING (client_hash_id) WHERE month>='2025-12' GROUP BY 1 ORDER BY 1""")
show("cohort_trend_ex_top_client", """WITH top AS (SELECT client_hash_id FROM pm WHERE month>='2025-10' GROUP BY 1 ORDER BY SUM(s_ai) DESC LIMIT 1)
 SELECT month, SUM(s_ai) ai, ROUND(SUM(s_ai)*1000.0/SUM(sess),2) ai_per_1k_sess FROM pm JOIN cohort USING (client_hash_id)
 WHERE month>='2025-12' AND client_hash_id NOT IN (SELECT * FROM top) GROUP BY 1 ORDER BY 1""")
# --- engagement: page-days whose sessions are ALL from one channel (attribution-clean)
show("engagement_pure_channel_days", """SELECT ch, COUNT(*) page_days, SUM(ga4_sessions) sess,
  ROUND(SUM(ga4_engaged_sessions)*100.0/SUM(ga4_sessions),1) engaged_pct, ROUND(SUM(ga4_total_engagement_sec)*1.0/SUM(ga4_sessions),1) sec_per_sess,
  ROUND(SUM(ga4_pageviews)*1.0/SUM(ga4_sessions),2) pv_per_sess FROM (
  SELECT *, CASE WHEN sessions_ai=ga4_sessions THEN 'ai' WHEN sessions_organic=ga4_sessions THEN 'organic'
    WHEN sessions_direct=ga4_sessions THEN 'direct' WHEN sessions_referral=ga4_sessions THEN 'referral' END ch
  FROM daily WHERE month>='2025-10' AND ga4_data_available IS TRUE AND ga4_sessions>0) WHERE ch IS NOT NULL GROUP BY 1 ORDER BY 1""")
show("engagement_pure_ai_by_platform", """SELECT CASE WHEN ai_chatgpt=sessions_ai THEN 'chatgpt' WHEN ai_gemini=sessions_ai THEN 'gemini'
  WHEN ai_perplexity=sessions_ai THEN 'perplexity' WHEN ai_copilot=sessions_ai THEN 'copilot' ELSE 'mixed/other' END plat,
  COUNT(*) page_days, SUM(ga4_sessions) sess, ROUND(SUM(ga4_engaged_sessions)*100.0/SUM(ga4_sessions),1) engaged_pct,
  ROUND(SUM(ga4_total_engagement_sec)*1.0/SUM(ga4_sessions),1) sec_per_sess
  FROM daily WHERE month>='2025-10' AND ga4_data_available IS TRUE AND ga4_sessions>0 AND sessions_ai=ga4_sessions GROUP BY 1 ORDER BY 2 DESC""")
json.dump(R, open("../outputs/eda_01_landscape.json", "w"), indent=1, default=str)
