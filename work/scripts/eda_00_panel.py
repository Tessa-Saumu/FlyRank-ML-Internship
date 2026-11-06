"""EDA 0: panel structure by month - who is in the data, and when AI-referral tracking exists.
Why first: every later window and population choice depends on when clients enter the panel."""
import json
import os

from wh import con

OUT = os.path.dirname(os.path.abspath(__file__)) + "/../outputs"


def main():
    c = con()
    df = c.sql("""SELECT month, COUNT(*) daily_rows, COUNT(DISTINCT client_hash_id) gsc_clients,
      COUNT(DISTINCT CASE WHEN ga4_data_available IS TRUE THEN client_hash_id END) ga4_clients,
      SUM(gsc_impressions) impressions, SUM(gsc_clicks) clicks,
      SUM(CASE WHEN ga4_data_available IS TRUE THEN ga4_sessions END) ga4_sessions,
      SUM(sessions_ai) ai, SUM(ai_chatgpt) gpt, SUM(ai_gemini) gem, SUM(ai_perplexity) ppx, SUM(ai_copilot) cop, SUM(ai_claude) cla,
      COUNT(DISTINCT CASE WHEN sessions_ai > 0 THEN content_hash_id END) ai_pages
      FROM daily GROUP BY 1 ORDER BY 1""").df()
    tot = c.sql("SELECT COUNT(*) n, MIN(report_date)::VARCHAR d0, MAX(report_date)::VARCHAR d1 FROM daily").fetchone()
    rec = {"daily_rows": int(tot[0]), "first_date": tot[1], "last_date": tot[2],
           "months": json.loads(df.to_json(orient="records"))}
    json.dump(rec, open(f"{OUT}/eda_ai_monthly.json", "w"), indent=1)
    return df


if __name__ == "__main__":
    print(main().to_string())
