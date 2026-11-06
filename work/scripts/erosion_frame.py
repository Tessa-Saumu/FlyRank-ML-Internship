"""Forward-origin frame for the CTR-erosion study (Option 1).

Timeline for origin month o (decision made on day 1 of o):
    feature window  = calendar months o-2, o-1          (plus o-3 for trend features)
    outcome window  = calendar months o, o+1            (label matures at end of o+1)
    a model scored at origin o may only train on origins <= o-2 (label maturity)

Unit: page x origin.  Eligible (knowable at decision time): impression-weighted position <= 10.5
over the feature window, >= 200 impressions and >= 20 clicks in it, client has GSC in it.
Label (observed, outcome window):
    expected_clicks = imp_out * ctr_now * f(pos_out) / f(pos_m1)
    erode = clicks_out < 0.75 * expected_clicks
where ctr_now is month o-1's CTR shrunk toward the client's position-adjusted CTR (empirical
Bayes, m = 1000 impressions),
and f() is a fixed reference CTR-by-position curve estimated on Oct-Nov 2025 only (before every
outcome window).
Label measurability needs expected_clicks >= 10, which uses outcome-window impressions: that
population choice is disclosed and the excluded "lost visibility" rows are counted.

Outputs (row-level, outside the repo): ~/.cache/flyrank_freestyle/erosion_frame.parquet
Receipt (aggregates only): work/freestyle/outputs/erosion_frame_summary.json
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from wh import CACHE, DAILY_DIR, con

ORIGINS = ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]
SEALED_ORIGIN = "2026-05"          # outcome window May-Jun 2026 includes the sealed final month
LIVE_ORIGIN = "2026-07"            # the next real decision: features May-Jun 2026, no outcome yet
REF_MONTHS = ("2025-10", "2025-11")
SHRINK_M = 1000.0
ERODE_RATIO = 0.75
MIN_EXPECTED = 10.0
OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../outputs"


def month_add(m: str, k: int) -> str:
    y, mm = map(int, m.split("-"))
    t = y * 12 + (mm - 1) + k
    return f"{t // 12:04d}-{t % 12 + 1:02d}"


def build_daily_extras(c, months):
    """Per page-month extras that need daily rows: position dispersion, last-14-day activity."""
    out = CACHE + "/page_month_extra"
    os.makedirs(out, exist_ok=True)
    for m in months:
        f = f"{out}/{m}.parquet"
        if os.path.exists(f):
            continue
        d = f"{DAILY_DIR}/month={m}"
        c.execute(f"""COPY (
          SELECT content_hash_id, '{m}' AS month,
            SUM(gsc_impressions * gsc_avg_position * gsc_avg_position) FILTER (WHERE gsc_impressions > 0) s_imp_pos2,
            SUM(gsc_impressions) FILTER (WHERE report_date > LAST_DAY(report_date) - INTERVAL 14 DAY) imp_l14,
            SUM(gsc_clicks) FILTER (WHERE report_date > LAST_DAY(report_date) - INTERVAL 14 DAY) clk_l14,
            SUM((gsc_clicks > 0)::INT) click_days,
            MAX(gsc_impressions) max_day_imp
          FROM read_parquet('{d}/*.parquet') WHERE gsc_impressions > 0 GROUP BY ALL) TO '{f}' (FORMAT parquet)""")
        print("extras", m, flush=True)


def reference_curve(c):
    """Impression-weighted CTR by 1-position bin on Oct-Nov 2025, made monotone (isotonic)."""
    df = c.sql(f"""SELECT LEAST(FLOOR(sum_pos*1.0/imp), 30) AS b, SUM(clicks) clk, SUM(imp) imp FROM pm
        WHERE month IN {REF_MONTHS} AND imp > 0 GROUP BY 1 ORDER BY 1""").df()
    df = df[df.b >= 1]
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(df.b + 0.5, df.clk / df.imp, sample_weight=df.imp)
    grid = np.arange(1.0, 31.0, 0.5)
    vals = np.clip(iso.predict(grid), 1e-5, None)
    return grid, vals


def f_pos(pos, grid, vals):
    return np.exp(np.interp(np.clip(pos, grid[0], grid[-1]), grid, np.log(vals)))


def build_frame():
    t0 = time.time()
    c = con()
    months = sorted({month_add(o, k) for o in ORIGINS for k in (-3, -2, -1, 0, 1)})
    months = [m for m in months if glob.glob(f"{DAILY_DIR}/month={m}")]
    build_daily_extras(c, months)
    c.execute(f"CREATE VIEW px AS SELECT * FROM read_parquet('{CACHE}/page_month_extra/*.parquet')")
    grid, vals = reference_curve(c)
    frames = []
    for o in ORIGINS + [LIVE_ORIGIN]:
        m3, m2, m1, y1, y2 = (month_add(o, k) for k in (-3, -2, -1, 0, 1))
        df = c.sql(f"""
        WITH p AS (SELECT pm.*, px.s_imp_pos2, px.imp_l14, px.clk_l14, px.click_days, px.max_day_imp
                   FROM pm LEFT JOIN px USING (content_hash_id, month)),
        f AS (SELECT content_hash_id, ANY_VALUE(client_hash_id) client_hash_id,
                SUM(imp) imp_f, SUM(clicks) clk_f, SUM(sum_pos)*1.0/SUM(imp) pos_f,
                SUM(s_imp_pos2) s2, SUM(imp_days) imp_days_f, SUM(click_days) click_days_f, MAX(max_day_imp) max_day_imp,
                SUM(imp) FILTER (WHERE month='{m1}') imp_m1, SUM(imp) FILTER (WHERE month='{m2}') imp_m2,
                SUM(clicks) FILTER (WHERE month='{m1}') clk_m1, SUM(clicks) FILTER (WHERE month='{m2}') clk_m2,
                SUM(sum_pos) FILTER (WHERE month='{m1}')*1.0/NULLIF(SUM(imp) FILTER (WHERE month='{m1}'),0) pos_m1,
                SUM(sum_pos) FILTER (WHERE month='{m2}')*1.0/NULLIF(SUM(imp) FILTER (WHERE month='{m2}'),0) pos_m2,
                SUM(imp_l14) FILTER (WHERE month='{m1}') imp_l14, SUM(clk_l14) FILTER (WHERE month='{m1}') clk_l14,
                SUM(ga4_days) ga4_days_f, SUM(s_org) org_f, SUM(sess) sess_f, SUM(s_ai) ai_f
              FROM p WHERE month IN ('{m2}','{m1}') GROUP BY 1),
        h AS (SELECT content_hash_id, SUM(imp) imp_m3, SUM(clicks) clk_m3 FROM pm WHERE month='{m3}' GROUP BY 1),
        a AS (SELECT content_hash_id, COUNT(*) FILTER (WHERE imp>0) months_active FROM pm WHERE month < '{o}' GROUP BY 1),
        y AS (SELECT content_hash_id, SUM(imp) imp_out, SUM(clicks) clk_out, SUM(sum_pos)*1.0/NULLIF(SUM(imp),0) pos_out
              FROM pm WHERE month IN ('{y1}','{y2}') GROUP BY 1),
        cl AS (SELECT client_hash_id, SUM(imp) FILTER (WHERE month='{m1}') c_imp_m1, SUM(imp) FILTER (WHERE month='{m2}') c_imp_m2,
                 SUM(clicks) FILTER (WHERE month='{m1}') c_clk_m1, SUM(clicks) FILTER (WHERE month='{m2}') c_clk_m2,
                 SUM(clicks)*1.0/SUM(imp) c_ctr_f, SUM(sum_pos)*1.0/SUM(imp) c_pos_f, COUNT(DISTINCT content_hash_id) c_pages
               FROM pm WHERE month IN ('{m2}','{m1}') AND imp>0 AND sum_pos*1.0/imp <= 10.5 GROUP BY 1)
        SELECT '{o}' AS origin, f.*, h.imp_m3, h.clk_m3, a.months_active, y.imp_out, y.clk_out, y.pos_out, cl.*  EXCLUDE (client_hash_id),
          ct.word_count, ct.keyword_token_count, ct.search_volume, ct.cpc, ct.competition, ct.main_intent, ct.content_type,
          DATE_DIFF('day', ct.content_created_date::DATE, DATE '{o}-01') age_days
        FROM f LEFT JOIN h USING (content_hash_id) LEFT JOIN a USING (content_hash_id) LEFT JOIN y USING (content_hash_id)
        LEFT JOIN cl USING (client_hash_id) LEFT JOIN content ct USING (content_hash_id)
        WHERE f.imp_f >= 200 AND f.clk_f >= 20 AND f.pos_f <= 10.5""").df()
        frames.append(df)
        print("origin", o, len(df), round(time.time() - t0, 1), "s", flush=True)
    df = pd.concat(frames, ignore_index=True)

    # --- shrunk feature-window CTR and position-adjusted expectation (no outcome info in ctr_shrunk)
    f_feat = f_pos(df.pos_f, grid, vals)
    prior = df.c_ctr_f * f_feat / f_pos(df.c_pos_f, grid, vals)
    df["ctr_shrunk"] = (df.clk_f + SHRINK_M * prior) / (df.imp_f + SHRINK_M)
    # The expectation is anchored on the LATEST month (month o-1), shrunk toward the page's
    # two-month rate. Anchoring on the two-month average would label a page "eroding" merely
    # because its CTR had already fallen in month o-1 (momentum leaking into the label; the
    # failure mode of the earlier lane study). The label therefore means: further erosion
    # beyond the level the page is at when the decision is made.
    # The shrinkage target is the client's position-adjusted CTR, not the page's own two-month
    # rate: a declining page's own history is higher than its present, which would bias the
    # expectation upward and re-introduce the momentum leak through the back door.
    pos_m1 = df.pos_m1.fillna(df.pos_f)
    prior_m1 = df.c_ctr_f * f_pos(pos_m1, grid, vals) / f_pos(df.c_pos_f, grid, vals)
    df["ctr_now"] = (df.clk_m1.fillna(0) + SHRINK_M * prior_m1) / (df.imp_m1.fillna(0) + SHRINK_M)
    df["exp_out"] = (df.imp_out.fillna(0) * df.ctr_now
                     * f_pos(df.pos_out.fillna(pos_m1), grid, vals) / f_pos(pos_m1, grid, vals))
    df["measurable"] = df.exp_out >= MIN_EXPECTED
    df["erode"] = np.where(df.measurable, (df.clk_out.fillna(0) < ERODE_RATIO * df.exp_out).astype(float), np.nan)
    df["log_ratio_out"] = np.where(df.measurable, np.log((df.clk_out.fillna(0) + 0.5) / (df.exp_out + 0.5)), np.nan)
    rng = np.random.default_rng(7)
    sim = rng.poisson(df.exp_out.clip(lower=0))
    df["null_erode"] = np.where(df.measurable, (sim < ERODE_RATIO * df.exp_out).astype(float), np.nan)

    # --- lagged (matured) page and client labels: origin o-2's outcome ends at month o-1 (< o)
    lag = df[["origin", "content_hash_id", "erode", "log_ratio_out"]].copy()
    lag["origin"] = lag.origin.map(lambda m: month_add(m, 2))
    df = df.merge(lag.rename(columns={"erode": "page_prev_erode", "log_ratio_out": "page_prev_ratio"}),
                  on=["origin", "content_hash_id"], how="left")
    cl_lag = df.groupby(["origin", "client_hash_id"]).erode.mean().reset_index()
    cl_lag["origin"] = cl_lag.origin.map(lambda m: month_add(m, 2))
    df = df.merge(cl_lag.rename(columns={"erode": "client_prev_erode_rate"}), on=["origin", "client_hash_id"], how="left")

    df.to_parquet(CACHE + "/erosion_frame.parquet")
    summ = (df.groupby("origin").agg(eligible=("erode", "size"), measurable=("measurable", "sum"),
                                     clients=("client_hash_id", "nunique"), base_rate=("erode", "mean"),
                                     null_rate=("null_erode", "mean"),
                                     lost_visibility=("measurable", lambda s: 1 - s.mean()))
            .round(4).reset_index())
    summ["sealed"] = summ.origin == SEALED_ORIGIN
    summ["live_no_outcome"] = summ.origin == LIVE_ORIGIN
    summ.loc[summ.live_no_outcome, ["base_rate", "null_rate", "lost_visibility", "measurable"]] = np.nan
    # the sealed origin's label/base rate stays out of the receipt until unsealing
    summ.loc[summ.sealed, ["base_rate", "null_rate", "lost_visibility", "measurable"]] = np.nan
    receipt = {"origins": json.loads(summ.to_json(orient="records")),
               "reference_curve": {"months": REF_MONTHS, "pos": grid[::2].tolist(), "ctr": np.round(vals[::2], 6).tolist()},
               "params": {"shrink_m": SHRINK_M, "erode_ratio": ERODE_RATIO, "min_expected_clicks": MIN_EXPECTED,
                          "eligibility": "pos_f<=10.5, imp_f>=200, clk_f>=20"},
               "warehouse_revision": "50cbf7c3909d07be4d1b5906b4d09e882e5acbf2"}
    json.dump(receipt, open(OUT_DIR + "/erosion_frame_summary.json", "w"), indent=1)
    print(summ.to_string())
    return df


if __name__ == "__main__":
    build_frame()
