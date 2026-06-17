"""
LakeDB — thin wrapper around DuckDB for the GMMIE data lake.

DuckDB chosen over Postgres/SQLite for three reasons:
  1. Columnar engine — analytical SELECTs over 50M rows are sub-second on a laptop.
  2. Zero-config single file. Backup is `cp data/lake/gmmie.duckdb`.
  3. Native Arrow / Pandas integration — research notebooks read with zero copy.

API is deliberately small. Direct SQL is preferred over ORM ceremony.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import duckdb

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def lake_path() -> Path:
    p = Path(os.environ.get("GMMIE_LAKE_PATH",
                            _REPO_ROOT / "data" / "lake" / "gmmie.duckdb"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class LakeDB:
    """
    Thread-safe-ish DuckDB handle.

    DuckDB allows one writer per process. We serialise writes through a lock;
    concurrent reads share the same connection (DuckDB releases the GIL on
    most ops, but holding a lock around writes is the safe default).
    """

    _lock = threading.RLock()

    def __init__(self, path: Optional[Path] = None, read_only: bool = False):
        self.path = Path(path) if path else lake_path()
        self._con: Optional[duckdb.DuckDBPyConnection] = None
        self.read_only = read_only

    # ── context manager ──────────────────────────────────────────────────────
    def __enter__(self) -> "LakeDB":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def open(self) -> None:
        if self._con is not None:
            return
        self._con = duckdb.connect(str(self.path), read_only=self.read_only)
        # Idempotent schema migration
        if not self.read_only:
            with self._lock:
                self._con.execute(_SCHEMA_PATH.read_text())
        # Performance pragmas — fine on a laptop, no harm in production
        self._con.execute("PRAGMA threads=4")
        self._con.execute("PRAGMA memory_limit='4GB'")

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    # ── execution ───────────────────────────────────────────────────────────
    def execute(self, sql: str, params: Optional[Sequence] = None):
        assert self._con is not None, "LakeDB not opened"
        with self._lock:
            return self._con.execute(sql, params or ())

    def executemany(self, sql: str, rows: Iterable[Sequence]) -> None:
        assert self._con is not None, "LakeDB not opened"
        rows = list(rows)
        if not rows:
            return
        with self._lock:
            self._con.executemany(sql, rows)

    def df(self, sql: str, params: Optional[Sequence] = None):
        """SQL → pandas DataFrame (zero-copy via Arrow)."""
        return self.execute(sql, params).fetchdf()

    def scalar(self, sql: str, params: Optional[Sequence] = None):
        row = self.execute(sql, params).fetchone()
        return row[0] if row else None

    # ── upsert helpers ───────────────────────────────────────────────────────
    def upsert(self, table: str, columns: Sequence[str],
               rows: Iterable[Sequence],
               conflict_key: Sequence[str]) -> int:
        """
        Idempotent INSERT … ON CONFLICT DO UPDATE, wrapped in a single
        transaction (much faster than per-row autocommit).

        Returns the number of rows attempted (not necessarily distinct rows
        actually inserted/updated; that's a DuckDB engine detail).
        """
        rows = list(rows)
        if not rows:
            return 0
        cols = ",".join(columns)
        placeholders = ",".join(["?"] * len(columns))
        update = ",".join(
            f"{c}=excluded.{c}" for c in columns if c not in conflict_key
        )
        on_conflict = ",".join(conflict_key)
        sql = (
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT ({on_conflict}) DO UPDATE SET {update}"
            if update
            else
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT ({on_conflict}) DO NOTHING"
        )
        assert self._con is not None
        with self._lock:
            self._con.begin()
            try:
                self._con.executemany(sql, rows)
                self._con.commit()
            except Exception:
                self._con.rollback()
                raise
        return len(rows)

    # ── Arrow bulk path ──────────────────────────────────────────────────────
    def bulk_upsert_df(self, table: str, df,
                       conflict_key: Sequence[str]) -> int:
        """
        Arrow-fast bulk upsert from a pandas DataFrame.

        Strategy:
          1. Register the DataFrame as a temporary view (zero-copy via Arrow).
          2. INSERT … SELECT … FROM tmp ON CONFLICT DO UPDATE.

        ~10–100× faster than executemany for >10k rows; equivalent for tiny
        batches.
        """
        if df is None or len(df) == 0:
            return 0
        assert self._con is not None
        # Stable temp-view name per call to avoid race conditions
        tmp = f"_tmp_bulk_{uuid.uuid4().hex[:8]}"
        cols = list(df.columns)
        col_list = ",".join(cols)
        update = ",".join(
            f"{c}=excluded.{c}" for c in cols if c not in conflict_key
        )
        on_conflict = ",".join(conflict_key)
        sql = (
            f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM {tmp} "
            + (f"ON CONFLICT ({on_conflict}) DO UPDATE SET {update}"
               if update
               else f"ON CONFLICT ({on_conflict}) DO NOTHING")
        )
        with self._lock:
            self._con.register(tmp, df)
            try:
                self._con.begin()
                try:
                    self._con.execute(sql)
                    self._con.commit()
                except Exception:
                    self._con.rollback()
                    raise
            finally:
                self._con.unregister(tmp)
        return len(df)

    # ── run provenance ───────────────────────────────────────────────────────
    @contextlib.contextmanager
    def run(self, source: str, notes: str = "") -> Iterator[str]:
        """Context manager: open a `lake_runs` row, close on exit with row count."""
        import datetime as dt
        run_id = uuid.uuid4().hex
        started = dt.datetime.utcnow()
        self.execute(
            "INSERT INTO lake_runs (run_id, started_at, source, notes) VALUES (?, ?, ?, ?)",
            [run_id, started, source, notes],
        )
        # rough row-counter via post-hoc COUNT(*) delta is unreliable across tables;
        # callers should record `rows_added` themselves before yielding.
        try:
            yield run_id
        finally:
            finished = dt.datetime.utcnow()
            self.execute(
                "UPDATE lake_runs SET finished_at = ? WHERE run_id = ?",
                [finished, run_id],
            )

    def record_rows(self, run_id: str, n: int) -> None:
        self.execute(
            "UPDATE lake_runs SET rows_added = COALESCE(rows_added, 0) + ? WHERE run_id = ?",
            [n, run_id],
        )
