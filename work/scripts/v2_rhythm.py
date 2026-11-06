"""V2 date and rhythm features, built from the daily table one calendar month at a time.

Five features per page x origin, each computed strictly inside calendar months o-2 and o-1:
  weekday_share               share of window impressions on Monday to Friday
  days_since_last_click       days from the last click day in the window to the origin's first day
  longest_zero_click_run      longest run of consecutive no-click calendar days, per month, combined by max
                              (a run across the month boundary is NOT joined; days with no row count as no-click)
  days_since_peak             days from the single max-impression day in the window to the origin's first day
                              (ties resolve to the most recent such day)
  keyword_to_content_gap_days content_created_date minus keyword_created_date (dim_content), days

Row-level caches live in ~/.cache/flyrank_freestyle (never in the repo):
  page_month_rhythm/<YYYY-MM>.parquet   one row per page that has any daily row in the month
  v2_rhythm_feat/<origin>.parquet       the five features per frame page of that origin
The cache for the test outcome months (2026-05, 2026-06) may only be built after the test has been scored
(`allow_sealed_months=True`, passed by the live step).
"""
import calendar
import datetime as dt
import os

import numpy as np
import pandas as pd

from erosion_frame import month_add
from wh import CACHE, DAILY_DIR, con

RH_DIR = CACHE + "/page_month_rhythm"
FEAT_DIR = CACHE + "/v2_rhythm_feat"
NEW_FEATURES = ["weekday_share", "days_since_last_click", "longest_zero_click_run", "days_since_peak",
                "keyword_to_content_gap_days"]
SEALED_OUTCOME_MONTHS = ("2026-05", "2026-06")   # outcome months of the sealed test origin 2026-05


def month_bounds(m):
    y, mm = map(int, m.split("-"))
    return dt.date(y, mm, 1), dt.date(y, mm, calendar.monthrange(y, mm)[1])


def build_month(c, m, allow_sealed_months=False):
    """One page-month rhythm table from the daily partition of month m."""
    if m in SEALED_OUTCOME_MONTHS and not allow_sealed_months:
        raise PermissionError(f"month {m} is an outcome month of the sealed test origin; build it after the test is scored")
    os.makedirs(RH_DIR, exist_ok=True)
    f = f"{RH_DIR}/{m}.parquet"
    if os.path.exists(f):
        return f
    first, last = month_bounds(m)
    d = f"{DAILY_DIR}/month={m}"
    c.execute(f"""COPY (
      WITH pd AS (
        SELECT content_hash_id, report_date, SUM(COALESCE(gsc_impressions, 0)) imp, SUM(COALESCE(gsc_clicks, 0)) clk
        FROM read_parquet('{d}/*.parquet') GROUP BY 1, 2),
      mx AS (SELECT content_hash_id, MAX(imp) peak_imp FROM pd GROUP BY 1),
      pk AS (SELECT pd.content_hash_id, MAX(pd.report_date) peak_date FROM pd JOIN mx USING (content_hash_id)
             WHERE pd.imp = mx.peak_imp AND mx.peak_imp > 0 GROUP BY 1),
      cd AS (SELECT content_hash_id, report_date,
               LAG(report_date) OVER (PARTITION BY content_hash_id ORDER BY report_date) prev_d
             FROM pd WHERE clk > 0),
      gp AS (SELECT content_hash_id,
               MAX(CASE WHEN prev_d IS NULL THEN DATE_DIFF('day', DATE '{first}', report_date)
                        ELSE DATE_DIFF('day', prev_d, report_date) - 1 END) head_mid_gap,
               MAX(report_date) last_click_date FROM cd GROUP BY 1),
      ag AS (SELECT content_hash_id, SUM(imp) imp_total,
               SUM(imp) FILTER (WHERE ISODOW(report_date) <= 5) imp_wd,
               MIN(report_date) min_date, MAX(report_date) max_date FROM pd GROUP BY 1)
      SELECT ag.content_hash_id, '{m}' AS month, ag.imp_total, COALESCE(ag.imp_wd, 0) imp_wd,
             CASE WHEN mx.peak_imp > 0 THEN mx.peak_imp END peak_imp, pk.peak_date, gp.last_click_date,
             CASE WHEN gp.last_click_date IS NULL THEN {last.day}
                  ELSE GREATEST(gp.head_mid_gap, DATE_DIFF('day', gp.last_click_date, DATE '{last}')) END zero_run,
             {last.day} days_in_month, ag.min_date, ag.max_date
      FROM ag JOIN mx USING (content_hash_id) LEFT JOIN pk USING (content_hash_id) LEFT JOIN gp USING (content_hash_id)
    ) TO '{f}' (FORMAT parquet)""")
    t = pd.read_parquet(f, columns=["min_date", "max_date"])
    t = t.apply(pd.to_datetime)
    assert t.min_date.min() >= pd.Timestamp(first) and t.max_date.max() <= pd.Timestamp(last), \
        f"daily rows outside month {m}"
    return f


def ensure_months(months, allow_sealed_months=False):
    c = None
    for m in months:
        if not os.path.exists(f"{RH_DIR}/{m}.parquet"):
            c = c or con()
            build_month(c, m, allow_sealed_months)
            print("rhythm month", m, flush=True)


def _month_table(m, ids):
    t = pd.read_parquet(f"{RH_DIR}/{m}.parquet")
    for c in ("peak_date", "last_click_date", "min_date", "max_date"):
        t[c] = pd.to_datetime(t[c]).astype("datetime64[ns]")
    return t.set_index("content_hash_id").reindex(ids)


def rhythm_for_origin(o, ids, allow_sealed_months=False):
    """The five rhythm features for the frame pages `ids` of origin o, with timeline assertions."""
    m2, m1 = month_add(o, -2), month_add(o, -1)
    ensure_months([m2, m1], allow_sealed_months)
    first_o = pd.Timestamp(month_bounds(o)[0])
    w_start = pd.Timestamp(month_bounds(m2)[0])
    t2, t1 = _month_table(m2, ids), _month_table(m1, ids)
    imp = t2.imp_total.fillna(0).to_numpy() + t1.imp_total.fillna(0).to_numpy()
    wd = t2.imp_wd.fillna(0).to_numpy() + t1.imp_wd.fillna(0).to_numpy()
    out = pd.DataFrame(index=range(len(ids)))
    out["weekday_share"] = np.where(imp > 0, wd / np.where(imp > 0, imp, 1), np.nan)
    lc = pd.concat([t2.last_click_date.reset_index(drop=True), t1.last_click_date.reset_index(drop=True)], axis=1).max(axis=1)
    out["days_since_last_click"] = (first_o - lc).dt.days.astype(float)
    # peak: larger peak_imp wins; ties go to the more recent day
    p2i, p1i = t2.peak_imp.fillna(0).to_numpy(), t1.peak_imp.fillna(0).to_numpy()
    use1 = p1i >= p2i
    pdate = pd.Series(np.where(use1, t1.peak_date.to_numpy(), t2.peak_date.to_numpy()), dtype="datetime64[ns]")
    pdate[(p1i == 0) & (p2i == 0)] = pd.NaT
    out["days_since_peak"] = (first_o - pdate).dt.days.astype(float)
    r2 = np.where(np.isnan(t2.zero_run.to_numpy()), month_bounds(m2)[1].day, t2.zero_run.to_numpy())
    r1 = np.where(np.isnan(t1.zero_run.to_numpy()), month_bounds(m1)[1].day, t1.zero_run.to_numpy())
    out["longest_zero_click_run"] = np.maximum(r1, r2).astype(float)
    # timeline assertions: every date used lies inside [first day of o-2, last day of o-1], i.e. before the origin
    for name, s in [("last click", lc), ("peak day", pdate)]:
        s = s.dropna()
        if len(s):
            assert s.max() < first_o, f"{name} on or after the origin {o}"
            assert s.min() >= w_start, f"{name} before the feature window of origin {o}"
    for t in (t2, t1):
        mx = t.max_date.dropna()
        assert mx.empty or mx.max() < first_o, f"daily rows on or after origin {o}"
    out["_max_date_used"] = pd.concat([t2.max_date.reset_index(drop=True), t1.max_date.reset_index(drop=True)], axis=1).max(axis=1)
    return out


def kw_gap(ids):
    c = con()
    t = c.sql("SELECT content_hash_id, content_created_date, keyword_created_date, content_updated_date FROM content").df()
    t = t.set_index("content_hash_id").reindex(ids)
    gap = (t.content_created_date - t.keyword_created_date).dt.days.astype(float)
    return gap.to_numpy(), t


def rhythm_features(df, allow_sealed_months=False):
    """Five rhythm features aligned to df's index; diagnostics dict (aggregates only)."""
    os.makedirs(FEAT_DIR, exist_ok=True)
    res = pd.DataFrame(index=df.index, columns=NEW_FEATURES, dtype=float)
    diag = {}
    for o, d in df.groupby("origin"):
        f = f"{FEAT_DIR}/{o}.parquet"
        ids = d.content_hash_id.tolist()
        if os.path.exists(f):
            r = pd.read_parquet(f)
            assert r.content_hash_id.tolist() == ids, f"rhythm cache for {o} is stale"
            r = r.drop(columns="content_hash_id")
        else:
            r = rhythm_for_origin(o, ids, allow_sealed_months)
            gap, _ = kw_gap(ids)
            r["keyword_to_content_gap_days"] = gap
            r.insert(0, "content_hash_id", ids)
            r.to_parquet(f)
            r = r.drop(columns="content_hash_id")
        first_o = pd.Timestamp(month_bounds(o)[0])
        mdu = pd.to_datetime(r["_max_date_used"]) if "_max_date_used" in r else None
        diag[o] = {"max_daily_date_used": str(mdu.max().date()) if mdu is not None and mdu.notna().any() else None,
                   "origin_first_day": str(first_o.date()),
                   "n_missing": {k: int(r[k].isna().sum()) for k in NEW_FEATURES}}
        res.loc[d.index, NEW_FEATURES] = r[NEW_FEATURES].to_numpy()
    return res.astype(float), diag


def snapshot_diagnostics(df):
    """dim_content is a July-2026 snapshot: how many pages changed after each origin, and keyword dates vs origin."""
    out = {}
    for o, d in df.groupby("origin"):
        _, t = kw_gap(d.content_hash_id.tolist())
        first_o = pd.Timestamp(month_bounds(o)[0])
        out[o] = {"eligible_pages": int(len(d)),
                  "share_content_updated_on_or_after_origin": round(float((t.content_updated_date >= first_o).mean()), 4),
                  "keyword_created_after_origin_count": int((t.keyword_created_date > first_o).sum()),
                  "keyword_created_missing_share": round(float(t.keyword_created_date.isna().mean()), 4)}
    return out
