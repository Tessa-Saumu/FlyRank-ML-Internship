"""Build the page x month cache (all 17 months), one month partition at a time.

Output: ~/.cache/flyrank_freestyle/page_month/<YYYY-MM>.parquet (row-level, never committed).
GA4 columns are summed only on days where ga4_data_available IS TRUE (zero-fill rule).
"""
import glob
import os
import time

from wh import CACHE, DAILY_DIR, con

OUT = CACHE + "/page_month"
os.makedirs(OUT, exist_ok=True)
c = con()
t = time.time()
for d in sorted(glob.glob(DAILY_DIR + "/month=*")):
    d = d.replace("\\", "/")
    m = d.split("month=")[1]
    c.execute(f"""COPY (
    SELECT client_hash_id, content_hash_id, '{m}' AS month,
      COUNT(*) AS n_days, SUM(gsc_impressions) imp, SUM(gsc_clicks) clicks, SUM(gsc_sum_position) sum_pos,
      SUM((gsc_impressions > 0)::INT) imp_days,
      SUM((ga4_data_available IS TRUE)::INT) ga4_days,
      SUM(CASE WHEN ga4_data_available IS TRUE THEN ga4_sessions END) sess,
      SUM(CASE WHEN ga4_data_available IS TRUE THEN ga4_engaged_sessions END) eng_sess,
      SUM(CASE WHEN ga4_data_available IS TRUE THEN ga4_total_engagement_sec END) eng_sec,
      SUM(sessions_organic) s_org, SUM(sessions_direct) s_dir, SUM(sessions_referral) s_ref,
      SUM(sessions_social) s_soc, SUM(sessions_paid) s_paid, SUM(sessions_ai) s_ai,
      SUM(ai_chatgpt) ai_gpt, SUM(ai_perplexity) ai_ppx, SUM(ai_gemini) ai_gem,
      SUM(ai_copilot) ai_cop, SUM(ai_claude) ai_cla,
      SUM((sessions_ai > 0)::INT) ai_days, SUM(scroll_events) scroll_ev
    FROM read_parquet('{d}/*.parquet') GROUP BY ALL) TO '{OUT}/{m}.parquet' (FORMAT parquet)""")
    print(m, round(time.time() - t, 1), "s", flush=True)
print(c.sql(f"SELECT COUNT(*), SUM(s_ai), SUM(imp) FROM read_parquet('{OUT}/*.parquet')").fetchall())
# is avg_position = sum_position / impressions ?
print(c.sql("""SELECT corr(gsc_avg_position, gsc_sum_position*1.0/gsc_impressions),
  AVG(ABS(gsc_avg_position - gsc_sum_position*1.0/gsc_impressions))
  FROM daily WHERE month='2026-03' AND gsc_impressions>0""").fetchall())
