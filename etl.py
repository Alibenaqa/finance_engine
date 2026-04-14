"""
Layer 2 — ETL (DuckDB)
======================
Consumes the canonical DataFrame produced by Layer 1 (ingester.py),
enriches it with derived columns, deduplicates by transaction_id, and
loads it into a persistent DuckDB database.

Enriched columns added on top of the ingester canonical schema
--------------------------------------------------------------
tx_type     : str  — inferred type: card_payment | transfer_in |
                     transfer_out | topup | fee | interest | atm |
                     refund | exchange | savings_move | pocket_move
is_internal : bool — True for intra-account moves (savings ↔ current,
                     pocket transfers) that should be excluded from
                     spend/income analysis
year        : int
month       : int  — 1-12
week        : int  — ISO week number
day_of_week : int  — 0 = Monday … 6 = Sunday

Persistent views in DuckDB
---------------------------
v_real          — real transactions only (is_internal = FALSE)
v_expenses      — real outflows   (amount < 0, is_internal = FALSE)
v_income        — real inflows    (amount > 0, is_internal = FALSE)
v_monthly       — monthly revenue / expense / net per account
v_top_merchants — total spend aggregated by merchant (description)

Usage
-----
    from ingester import CSVIngester
    from etl import FinanceETL

    df = CSVIngester().ingest("data/raw/revolut.csv", profile="revolut",
                              account="Revolut").dataframe
    etl = FinanceETL()
    etl.load(df)

    summary = etl.get_summary()
    monthly = etl.get_monthly()
    top     = etl.get_top_merchants(n=15)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal-transfer patterns
# ---------------------------------------------------------------------------

# Descriptions that represent moves between sub-accounts within the same
# institution — not real income or expenses.
_INTERNAL_PATTERNS: list[str] = [
    # Revolut savings / pocket
    r"^To Instant Access Savings$",
    r"^From Instant Access Savings$",
    r"^Pocket Withdrawal$",
    r"^To pocket EUR.*",
    r"^Closing transaction$",
    # Generic self-transfers (extend as needed)
    r"^Transfer to savings",
    r"^From savings",
]

_INTERNAL_RE = re.compile(
    "|".join(f"(?:{p})" for p in _INTERNAL_PATTERNS),
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Transaction-type inference rules  (evaluated top-to-bottom, first match wins)
# ---------------------------------------------------------------------------

_TYPE_RULES: list[tuple[str, re.Pattern]] = [
    ("interest",     re.compile(r"^Interest earned", re.I)),
    ("fee",          re.compile(r"fee|frais", re.I)),
    ("refund",       re.compile(r"refund|remboursement|avoir|cashback", re.I)),
    ("atm",          re.compile(r"^cash withdrawal|^retrait", re.I)),
    ("exchange",     re.compile(r"^exchange|^transfer to revolut digital assets", re.I)),
    ("savings_move", re.compile(r"instant access savings|closing transaction", re.I)),
    ("pocket_move",  re.compile(r"pocket withdrawal|to pocket eur", re.I)),
    ("topup",        re.compile(r"^top.up|^apple pay top|payment from |^payment from", re.I)),
    ("transfer_in",  re.compile(r"^transfer from |^from ", re.I)),
    ("transfer_out", re.compile(r"^transfer to |^to |^payment$|^beauty |^dkchi|^il reste", re.I)),
    ("card_payment", re.compile(r".", re.I)),  # catch-all
]


def _infer_tx_type(description: str, amount: float) -> str:
    for tx_type, pattern in _TYPE_RULES:
        if pattern.search(description):
            return tx_type
    return "card_payment"


def _is_internal(description: str) -> bool:
    return bool(_INTERNAL_RE.match(description))


# ---------------------------------------------------------------------------
# ETL class
# ---------------------------------------------------------------------------

class FinanceETL:
    """
    Loads, enriches, and stores canonical finance transactions in DuckDB.

    Parameters
    ----------
    db_path : path to the DuckDB file (created if it doesn't exist).
              Use ":memory:" for an in-memory database (testing only).
    """

    # Canonical columns we expect from ingester.py
    _REQUIRED_COLS = {
        "transaction_id", "date", "description", "raw_description",
        "amount", "currency", "category", "account", "source_file",
        "ingested_at",
    }

    def __init__(self, db_path: str | Path = "db/finance.duckdb") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.db_path))
        self._init_schema()
        self._create_views()
        logger.info("FinanceETL ready — db: %s", self.db_path)

    # ------------------------------------------------------------------
    # Public API — loading
    # ------------------------------------------------------------------

    def load(self, df: pd.DataFrame) -> int:
        """
        Enrich and upsert a canonical DataFrame into the transactions table.

        Returns the number of NEW rows inserted (duplicates are ignored).
        """
        missing = self._REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        enriched = self._enrich(df.copy())

        rows_before = self._count()
        # DuckDB INSERT OR IGNORE uses the PRIMARY KEY (transaction_id) to skip
        # rows that are already present.
        self._con.execute(
            "INSERT OR IGNORE INTO transactions SELECT * FROM enriched"
        )
        rows_after = self._count()
        inserted = rows_after - rows_before
        logger.info("Loaded %d new rows (%d already present)", inserted,
                    len(enriched) - inserted)
        return inserted

    def load_file(self, path: str | Path, *, profile: Optional[str] = None,
                  account: Optional[str] = None) -> int:
        """Convenience: ingest a CSV with Layer 1 then load into DuckDB."""
        from ingester import CSVIngester
        result = CSVIngester().ingest(path, profile=profile, account=account)
        return self.load(result.dataframe)

    # ------------------------------------------------------------------
    # Public API — querying
    # ------------------------------------------------------------------

    def query(self, sql: str) -> pd.DataFrame:
        """Run arbitrary SQL against the DuckDB database."""
        return self._con.execute(sql).df()

    def get_summary(self) -> dict:
        """High-level statistics dict."""
        row = self._con.execute("""
            SELECT
                COUNT(*)                                         AS total_tx,
                COUNT(*) FILTER (WHERE NOT is_internal)         AS real_tx,
                MIN(date)                                        AS first_date,
                MAX(date)                                        AS last_date,
                ROUND(SUM(amount) FILTER (
                    WHERE amount > 0 AND NOT is_internal), 2)    AS total_income,
                ROUND(SUM(amount) FILTER (
                    WHERE amount < 0 AND NOT is_internal), 2)    AS total_expenses,
                ROUND(SUM(amount) FILTER (
                    WHERE NOT is_internal), 2)                   AS net_balance,
                COUNT(DISTINCT account)                          AS accounts
            FROM transactions
        """).fetchone()

        keys = ["total_tx", "real_tx", "first_date", "last_date",
                "total_income", "total_expenses", "net_balance", "accounts"]
        return dict(zip(keys, row))

    def get_monthly(self, *, exclude_internal: bool = True) -> pd.DataFrame:
        """Monthly revenue / expenses / net per account."""
        where = "WHERE NOT is_internal" if exclude_internal else ""
        return self._con.execute(f"""
            SELECT
                account,
                year,
                month,
                PRINTF('%04d-%02d', year, month)            AS period,
                ROUND(SUM(amount) FILTER (WHERE amount > 0), 2) AS income,
                ROUND(SUM(amount) FILTER (WHERE amount < 0), 2) AS expenses,
                ROUND(SUM(amount), 2)                           AS net,
                COUNT(*)                                        AS tx_count
            FROM transactions
            {where}
            GROUP BY account, year, month
            ORDER BY account, year, month
        """).df()

    def get_top_merchants(self, n: int = 20, *,
                          expenses_only: bool = True) -> pd.DataFrame:
        """Top merchants by total spend."""
        amount_filter = "AND amount < 0" if expenses_only else ""
        return self._con.execute(f"""
            SELECT
                description                          AS merchant,
                COUNT(*)                             AS tx_count,
                ROUND(SUM(amount), 2)                AS total,
                ROUND(AVG(amount), 2)                AS avg_tx,
                MIN(date)                            AS first_seen,
                MAX(date)                            AS last_seen
            FROM transactions
            WHERE NOT is_internal {amount_filter}
            GROUP BY description
            ORDER BY total ASC
            LIMIT {n}
        """).df()

    def get_expenses(self, *, month: Optional[str] = None) -> pd.DataFrame:
        """Real expenses, optionally filtered to a YYYY-MM month string."""
        where_month = ""
        if month:
            y, m = month.split("-")
            where_month = f"AND year = {y} AND month = {m}"
        return self._con.execute(f"""
            SELECT transaction_id, date, description, amount, currency,
                   category, tx_type, account
            FROM v_expenses
            WHERE 1=1 {where_month}
            ORDER BY date DESC
        """).df()

    def get_income(self, *, month: Optional[str] = None) -> pd.DataFrame:
        """Real income, optionally filtered to a YYYY-MM month string."""
        where_month = ""
        if month:
            y, m = month.split("-")
            where_month = f"AND year = {y} AND month = {m}"
        return self._con.execute(f"""
            SELECT transaction_id, date, description, amount, currency,
                   category, tx_type, account
            FROM v_income
            WHERE 1=1 {where_month}
            ORDER BY date DESC
        """).df()

    def get_tx_types(self) -> pd.DataFrame:
        """Breakdown of transaction types."""
        return self._con.execute("""
            SELECT
                tx_type,
                COUNT(*)             AS tx_count,
                ROUND(SUM(amount), 2) AS total
            FROM transactions
            WHERE NOT is_internal
            GROUP BY tx_type
            ORDER BY tx_count DESC
        """).df()

    def get_daily_balance(self) -> pd.DataFrame:
        """Cumulative net balance over time (real transactions only)."""
        return self._con.execute("""
            SELECT
                date,
                ROUND(SUM(amount), 2)                        AS daily_net,
                ROUND(SUM(SUM(amount)) OVER (ORDER BY date), 2) AS cumulative
            FROM transactions
            WHERE NOT is_internal
            GROUP BY date
            ORDER BY date
        """).df()

    def close(self) -> None:
        self._con.close()

    # ------------------------------------------------------------------
    # Schema and views
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                -- from ingester canonical schema
                transaction_id  VARCHAR PRIMARY KEY,
                date            DATE        NOT NULL,
                description     VARCHAR     NOT NULL,
                raw_description VARCHAR,
                amount          DOUBLE      NOT NULL,
                currency        VARCHAR(3),
                category        VARCHAR,
                account         VARCHAR,
                source_file     VARCHAR,
                ingested_at     TIMESTAMP,
                -- enriched by ETL
                tx_type         VARCHAR,
                is_internal     BOOLEAN     DEFAULT FALSE,
                year            INTEGER,
                month           INTEGER,
                week            INTEGER,
                day_of_week     INTEGER
            )
        """)

    def _create_views(self) -> None:
        self._con.execute("""
            CREATE OR REPLACE VIEW v_real AS
            SELECT * FROM transactions WHERE NOT is_internal
        """)
        self._con.execute("""
            CREATE OR REPLACE VIEW v_expenses AS
            SELECT * FROM v_real WHERE amount < 0
        """)
        self._con.execute("""
            CREATE OR REPLACE VIEW v_income AS
            SELECT * FROM v_real WHERE amount > 0
        """)
        self._con.execute("""
            CREATE OR REPLACE VIEW v_monthly AS
            SELECT
                account,
                PRINTF('%04d-%02d', year, month)            AS period,
                year,
                month,
                ROUND(SUM(amount) FILTER (WHERE amount > 0), 2) AS income,
                ROUND(SUM(amount) FILTER (WHERE amount < 0), 2) AS expenses,
                ROUND(SUM(amount), 2)                           AS net,
                COUNT(*)                                        AS tx_count
            FROM v_real
            GROUP BY account, year, month
        """)
        self._con.execute("""
            CREATE OR REPLACE VIEW v_top_merchants AS
            SELECT
                description                     AS merchant,
                COUNT(*)                        AS tx_count,
                ROUND(SUM(amount), 2)           AS total,
                ROUND(AVG(amount), 2)           AS avg_tx
            FROM v_expenses
            GROUP BY description
            ORDER BY total ASC
        """)

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns to a canonical DataFrame."""
        df["date"] = pd.to_datetime(df["date"])

        df["is_internal"] = df["raw_description"].apply(_is_internal)
        df["tx_type"] = df.apply(
            lambda r: _infer_tx_type(r["raw_description"], r["amount"]), axis=1
        )
        df["year"]        = df["date"].dt.year
        df["month"]       = df["date"].dt.month
        df["week"]        = df["date"].dt.isocalendar().week.astype(int)
        df["day_of_week"] = df["date"].dt.dayofweek

        # Reorder to match the CREATE TABLE column order
        return df[[
            "transaction_id", "date", "description", "raw_description",
            "amount", "currency", "category", "account", "source_file",
            "ingested_at", "tx_type", "is_internal",
            "year", "month", "week", "day_of_week",
        ]]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count(self) -> int:
        return self._con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    def __repr__(self) -> str:
        n = self._count()
        return f"<FinanceETL db={self.db_path} rows={n}>"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Finance Engine — ETL (DuckDB)")
    parser.add_argument("csv", nargs="?", help="CSV file to load (optional)")
    parser.add_argument("--profile", "-p", default=None)
    parser.add_argument("--account", "-a", default=None)
    parser.add_argument("--db",     default="db/finance.duckdb")
    parser.add_argument("--sql",    default=None, help="Run a SQL query and print result")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--monthly", action="store_true")
    parser.add_argument("--top",    type=int, default=0, metavar="N",
                        help="Top N merchants by spend")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    etl = FinanceETL(db_path=args.db)

    if args.csv:
        n = etl.load_file(args.csv, profile=args.profile, account=args.account)
        print(f"Inserted {n} new rows.")

    if args.summary:
        s = etl.get_summary()
        print("\n--- Summary ---")
        for k, v in s.items():
            print(f"  {k:<18}: {v}")

    if args.monthly:
        print("\n--- Monthly ---")
        print(etl.get_monthly().to_string(index=False))

    if args.top:
        print(f"\n--- Top {args.top} merchants ---")
        print(etl.get_top_merchants(args.top).to_string(index=False))

    if args.sql:
        print(etl.query(args.sql).to_string(index=False))
