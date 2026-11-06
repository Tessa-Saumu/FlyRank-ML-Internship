"""Shared DuckDB access to the locally cached FlyRank warehouse snapshot (pinned revision).

Cache once with huggingface_hub.snapshot_download (see README); every query then runs locally,
with no HTTP 429s. Row-level caches live in ~/.cache/flyrank_freestyle, never in the repo.
"""
import glob
import os

import duckdb

REV = "50cbf7c3909d07be4d1b5906b4d09e882e5acbf2"
ROOT = (os.environ.get("FLYRANK_WAREHOUSE_DIR") or os.path.expanduser(
    f"~/.cache/huggingface/hub/datasets--FlyRank--internship-warehouse/snapshots/{REV}")).replace("\\", "/")
DAILY_DIR = ROOT + "/fact_content_daily_performance"
CACHE = os.path.expanduser("~/.cache/flyrank_freestyle").replace("\\", "/")
os.makedirs(CACHE, exist_ok=True)


def _token():
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:  # Colab Secrets panel
            from google.colab import userdata  # type: ignore
            tok = userdata.get("HF_TOKEN")
        except Exception:
            tok = None
    return tok


def data_available(try_download: bool = False) -> bool:
    """True when the pinned warehouse snapshot is on disk. With try_download and an HF_TOKEN
    (environment variable or Colab secret, never typed into a cell) it downloads it once (~1.2 GB)."""
    if glob.glob(DAILY_DIR + "/month=*"):
        return True
    tok = _token()
    if try_download and tok:
        from huggingface_hub import snapshot_download
        snapshot_download("FlyRank/internship-warehouse", repo_type="dataset", revision=REV, token=tok,
                          ignore_patterns=["fact_content_daily_performance_sample.parquet"])
        return bool(glob.glob(DAILY_DIR + "/month=*"))
    return False


def con(threads: int = 8) -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    tmp = CACHE + "/duckdb_tmp"
    os.makedirs(tmp, exist_ok=True)
    c.execute(f"SET threads={threads}")
    c.execute(f"SET temp_directory='{tmp}'")  # never spill into the OneDrive-synced repo
    c.execute("SET preserve_insertion_order=false")
    c.execute(f"CREATE VIEW daily AS SELECT * FROM read_parquet('{DAILY_DIR}/*/*.parquet', hive_partitioning=true)")
    c.execute(f"CREATE VIEW content AS SELECT * FROM read_parquet('{ROOT}/dim_content.parquet')")
    c.execute(f"CREATE VIEW clients AS SELECT * FROM read_parquet('{ROOT}/dim_clients.parquet')")
    c.execute(f"CREATE VIEW q90 AS SELECT * FROM read_parquet('{ROOT}/fact_content_query_90d.parquet')")
    if glob.glob(CACHE + "/page_month/*.parquet"):
        c.execute(f"CREATE VIEW pm AS SELECT * FROM read_parquet('{CACHE}/page_month/*.parquet')")
    return c
