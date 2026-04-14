"""
Layer 1 — CSV Ingestion
=======================
Reads bank/account CSV exports from various institutions, normalises them
into a single canonical schema, deduplicates by content-hash, and returns
a clean pandas DataFrame ready for the ETL layer (etl.py → DuckDB).

Canonical output columns
------------------------
transaction_id  : str   — SHA-256(date + description + amount + account)
date            : date  — transaction date
description     : str   — cleaned merchant / narrative
raw_description : str   — original text before cleaning
amount          : float — negative = expense, positive = income (EUR/USD)
currency        : str   — 3-letter ISO code, default "EUR"
category        : str   — label from source CSV, or "Uncategorized"
account         : str   — logical account name (caller-supplied or filename)
source_file     : str   — absolute path of the ingested CSV
ingested_at     : datetime — UTC timestamp of this ingestion run

Supported built-in profiles
----------------------------
"generic"       — auto-detect from headers (best-effort)
"bnp"           — BNP Paribas (France) export format
"societe_generale" — Société Générale (France)
"credit_agricole"  — Crédit Agricole (France)
"revolut"       — Revolut (multi-currency)
"n26"           — N26 export
"chase"         — JPMorgan Chase (US)
"bofa"          — Bank of America (US)

Custom profiles can be registered via CSVIngester.register_profile().
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema profile — describes how a specific bank CSV is structured
# ---------------------------------------------------------------------------

@dataclass
class SchemaProfile:
    """Column-mapping + parsing hints for one bank/source format."""

    name: str

    # Required column mappings (CSV header → canonical field)
    date_col: str = "date"
    description_col: str = "description"

    # Amount: either a single signed column OR separate debit/credit columns
    amount_col: Optional[str] = None       # signed: positive = income
    debit_col: Optional[str] = None        # unsigned positive = money out
    credit_col: Optional[str] = None       # unsigned positive = money in

    # Optional columns
    category_col: Optional[str] = None
    currency_col: Optional[str] = None
    balance_col: Optional[str] = None      # ignored in output, kept for ref

    # Parsing hints
    date_formats: list[str] = field(default_factory=lambda: [
        "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y",
        "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d",
        "%d/%m/%y", "%m/%d/%y",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",  # datetime with time
        "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
    ])
    decimal_sep: str = "."
    thousands_sep: str = ","
    currency_default: str = "EUR"
    encoding: str = "utf-8"
    skiprows: int = 0
    csv_sep: str = ","

    # Header aliases: maps alternative spellings to the canonical col names
    # above.  Resolved before any mapping is applied.
    aliases: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

_BUILTIN_PROFILES: dict[str, SchemaProfile] = {
    # ------------------------------------------------------------------
    "generic": SchemaProfile(
        name="generic",
        date_col="date",
        description_col="description",
        amount_col="amount",
        category_col="category",
        currency_col="currency",
    ),
    # ------------------------------------------------------------------
    "bnp": SchemaProfile(
        name="bnp",
        date_col="Date",
        description_col="Libellé",
        debit_col="Débit",
        credit_col="Crédit",
        csv_sep=";",
        decimal_sep=",",
        thousands_sep=" ",
        encoding="latin-1",
        skiprows=0,
        aliases={
            "libelle": "Libellé",
            "debit": "Débit",
            "credit": "Crédit",
        },
    ),
    # ------------------------------------------------------------------
    "societe_generale": SchemaProfile(
        name="societe_generale",
        date_col="Date de comptabilisation",
        description_col="Libellé",
        debit_col="Débit",
        credit_col="Crédit",
        csv_sep=";",
        decimal_sep=",",
        thousands_sep=" ",
        encoding="latin-1",
    ),
    # ------------------------------------------------------------------
    "credit_agricole": SchemaProfile(
        name="credit_agricole",
        date_col="Date",
        description_col="Libellé opération",
        amount_col="Montant",
        csv_sep=";",
        decimal_sep=",",
        thousands_sep=" ",
        encoding="latin-1",
    ),
    # ------------------------------------------------------------------
    "revolut": SchemaProfile(
        name="revolut",
        date_col="Started Date",
        description_col="Description",
        amount_col="Amount",
        category_col="Category",
        currency_col="Currency",
        csv_sep=",",
        decimal_sep=".",
        thousands_sep="",
        encoding="utf-8",
        aliases={
            "started date": "Started Date",
            "completed date": "Started Date",  # fallback
        },
    ),
    # ------------------------------------------------------------------
    "n26": SchemaProfile(
        name="n26",
        date_col="Date",
        description_col="Payee",
        amount_col="Amount (EUR)",
        category_col="Category",
        csv_sep=",",
        decimal_sep=".",
        thousands_sep="",
        encoding="utf-8",
        aliases={
            "amount (eur)": "Amount (EUR)",
            "payee": "Payee",
        },
    ),
    # ------------------------------------------------------------------
    "chase": SchemaProfile(
        name="chase",
        date_col="Transaction Date",
        description_col="Description",
        amount_col="Amount",
        category_col="Category",
        csv_sep=",",
        decimal_sep=".",
        thousands_sep=",",
        currency_default="USD",
        encoding="utf-8",
    ),
    # ------------------------------------------------------------------
    "bofa": SchemaProfile(
        name="bofa",
        date_col="Date",
        description_col="Description",
        amount_col="Amount",
        csv_sep=",",
        decimal_sep=".",
        thousands_sep=",",
        currency_default="USD",
        encoding="utf-8",
        skiprows=6,  # BofA exports have 6 header lines before the CSV table
    ),
}


# ---------------------------------------------------------------------------
# Ingestion result
# ---------------------------------------------------------------------------

@dataclass
class IngestionResult:
    """Returned by CSVIngester.ingest()."""
    dataframe: pd.DataFrame
    source_file: Path
    profile_used: str
    total_rows_read: int
    rows_after_dedup: int
    rows_dropped: int          # validation failures
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main ingester
# ---------------------------------------------------------------------------

class CSVIngester:
    """
    Ingests one or many CSV bank-statement files into a canonical DataFrame.

    Usage
    -----
    ingester = CSVIngester()
    result = ingester.ingest("data/raw/bnp_jan2025.csv", profile="bnp", account="BNP Courant")
    df = result.dataframe          # canonical pandas DataFrame
    """

    def __init__(self) -> None:
        self._profiles: dict[str, SchemaProfile] = dict(_BUILTIN_PROFILES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_profile(self, profile: SchemaProfile) -> None:
        """Add or override a named schema profile."""
        self._profiles[profile.name] = profile
        logger.debug("Registered profile '%s'", profile.name)

    def list_profiles(self) -> list[str]:
        return sorted(self._profiles)

    def ingest(
        self,
        path: str | Path,
        *,
        profile: Optional[str] = None,
        account: Optional[str] = None,
    ) -> IngestionResult:
        """
        Ingest a single CSV file.

        Parameters
        ----------
        path    : path to the CSV file
        profile : profile name (see list_profiles()).  If None, auto-detect.
        account : logical account label. Defaults to the file stem.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        account = account or path.stem
        schema = self._resolve_profile(path, profile)
        logger.info("Ingesting '%s' using profile '%s'", path.name, schema.name)

        raw_df = self._read_csv(path, schema)
        logger.debug("Read %d raw rows from '%s'", len(raw_df), path.name)

        normalised, warnings, dropped = self._normalise(raw_df, schema, account, path)
        before_dedup = len(normalised)
        normalised = self._deduplicate(normalised)
        after_dedup = len(normalised)

        if before_dedup - after_dedup:
            logger.info(
                "Removed %d duplicate rows from '%s'",
                before_dedup - after_dedup,
                path.name,
            )

        return IngestionResult(
            dataframe=normalised.reset_index(drop=True),
            source_file=path,
            profile_used=schema.name,
            total_rows_read=len(raw_df),
            rows_after_dedup=after_dedup,
            rows_dropped=dropped,
            warnings=warnings,
        )

    def ingest_directory(
        self,
        directory: str | Path,
        *,
        profile: Optional[str] = None,
        account_map: Optional[dict[str, str]] = None,
        glob: str = "*.csv",
    ) -> pd.DataFrame:
        """
        Ingest all CSVs in *directory* and return a single concatenated DataFrame.

        Parameters
        ----------
        directory   : folder to scan
        profile     : profile name to apply to every file (None = auto-detect per file)
        account_map : {filename_stem: account_label} override map
        glob        : file pattern (default "*.csv")
        """
        directory = Path(directory).resolve()
        files = sorted(directory.glob(glob))
        if not files:
            raise FileNotFoundError(f"No files matching '{glob}' in {directory}")

        frames: list[pd.DataFrame] = []
        for f in files:
            acct = (account_map or {}).get(f.stem, f.stem)
            try:
                result = self.ingest(f, profile=profile, account=acct)
                frames.append(result.dataframe)
                logger.info(
                    "  ✓ %s — %d transactions (profile: %s)",
                    f.name, result.rows_after_dedup, result.profile_used,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("  ✗ %s — skipped: %s", f.name, exc)

        if not frames:
            return _empty_canonical_df()

        combined = pd.concat(frames, ignore_index=True)
        combined = self._deduplicate(combined)
        logger.info("Directory ingestion complete: %d transactions total", len(combined))
        return combined

    # ------------------------------------------------------------------
    # Profile resolution
    # ------------------------------------------------------------------

    def _resolve_profile(self, path: Path, name: Optional[str]) -> SchemaProfile:
        if name:
            if name not in self._profiles:
                raise ValueError(
                    f"Unknown profile '{name}'. Available: {self.list_profiles()}"
                )
            return self._profiles[name]
        return self._autodetect_profile(path)

    def _autodetect_profile(self, path: Path) -> SchemaProfile:
        """
        Sniff CSV headers and return the best-matching built-in profile.
        Falls back to "generic" if nothing matches.
        """
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                sniff = pd.read_csv(
                    path, nrows=0, sep=None, engine="python", encoding=enc
                )
                headers = {c.strip().lower() for c in sniff.columns}
                break
            except Exception:  # noqa: BLE001
                continue
        else:
            logger.warning("Could not sniff headers for '%s', using generic", path.name)
            return self._profiles["generic"]

        scores: dict[str, int] = {}
        for pname, prof in self._profiles.items():
            if pname == "generic":
                continue
            candidate_cols = {
                c.lower()
                for c in [
                    prof.date_col,
                    prof.description_col,
                    prof.amount_col or "",
                    prof.debit_col or "",
                    prof.credit_col or "",
                ]
                if c
            }
            scores[pname] = len(candidate_cols & headers)

        best = max(scores, key=lambda k: scores[k], default="generic")
        if scores.get(best, 0) >= 2:
            logger.debug("Auto-detected profile '%s' (score=%d)", best, scores[best])
            return self._profiles[best]

        logger.debug("No strong profile match, using 'generic'")
        return self._profiles["generic"]

    # ------------------------------------------------------------------
    # CSV reading
    # ------------------------------------------------------------------

    def _read_csv(self, path: Path, schema: SchemaProfile) -> pd.DataFrame:
        base_kwargs: dict = dict(
            sep=schema.csv_sep,
            skiprows=schema.skiprows,
            dtype=str,        # read everything as str — we parse manually
            keep_default_na=False,
        )
        # Required column names we expect to find (lowercased for matching).
        required_lower = {
            c.lower() for c in [schema.date_col, schema.description_col] if c
        }

        # Try the profile's encoding first, then common fallbacks.
        # latin-1 never raises UnicodeDecodeError (it accepts any byte), so
        # we also validate that required columns are actually present after
        # reading — if not, the encoding produced garbage and we try the next.
        encodings = list(dict.fromkeys([schema.encoding, "utf-8", "latin-1", "cp1252"]))
        last_exc: Exception = RuntimeError("No encodings tried")
        df: pd.DataFrame | None = None

        for enc in encodings:
            kwargs = {**base_kwargs, "encoding": enc}
            try:
                candidate = pd.read_csv(path, **kwargs)
            except UnicodeDecodeError:
                last_exc = RuntimeError(f"encoding '{enc}' failed")
                continue
            except Exception:
                # Separator mismatch — try Python engine with auto-detection
                kwargs["sep"] = None
                kwargs["engine"] = "python"
                try:
                    candidate = pd.read_csv(path, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    continue

            candidate.columns = [c.strip() for c in candidate.columns]
            found_lower = {c.lower() for c in candidate.columns}
            if required_lower.issubset(found_lower):
                if enc != schema.encoding:
                    logger.debug("Fell back to encoding '%s' for '%s'", enc, path.name)
                df = candidate
                break
            # Required columns missing → try next encoding
            last_exc = RuntimeError(
                f"Encoding '{enc}' produced column names that don't match profile "
                f"(got {list(candidate.columns)[:5]})"
            )

        if df is None:
            raise RuntimeError(
                f"Could not read '{path.name}' with any encoding: {last_exc}"
            )

        # Apply aliases (case-insensitive).
        # Only rename a column when the canonical target does NOT already exist
        # in the DataFrame — avoids creating duplicate columns.
        col_lower = {c.lower(): c for c in df.columns}
        existing = {c.lower() for c in df.columns}
        rename_map: dict[str, str] = {}
        for alias_lower, canonical in schema.aliases.items():
            if canonical.lower() in existing:
                continue  # target already present, skip alias
            if alias_lower in col_lower and col_lower[alias_lower] != canonical:
                rename_map[col_lower[alias_lower]] = canonical
        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    # ------------------------------------------------------------------
    # Normalisation pipeline
    # ------------------------------------------------------------------

    def _normalise(
        self,
        df: pd.DataFrame,
        schema: SchemaProfile,
        account: str,
        source_file: Path,
    ) -> tuple[pd.DataFrame, list[str], int]:
        """
        Returns (canonical_df, warnings, n_dropped).
        """
        warnings: list[str] = []
        records: list[dict] = []
        dropped = 0
        now = datetime.utcnow()

        for idx, row in df.iterrows():
            try:
                parsed_date = self._parse_date(
                    row.get(schema.date_col, ""), schema
                )
                description = str(row.get(schema.description_col, "")).strip()
                amount = self._parse_amount(row, schema)
                currency = self._parse_currency(row, schema)
                category = str(row.get(schema.category_col or "", "")).strip() or "Uncategorized"
                raw_description = description
                description = _clean_description(description)

                if not description:
                    warnings.append(f"Row {idx}: empty description, using 'Unknown'")
                    description = "Unknown"

                txn_id = _make_id(parsed_date, raw_description, amount, account)

                records.append({
                    "transaction_id": txn_id,
                    "date": parsed_date,
                    "description": description,
                    "raw_description": raw_description,
                    "amount": amount,
                    "currency": currency,
                    "category": category,
                    "account": account,
                    "source_file": str(source_file),
                    "ingested_at": now,
                })
            except _RowSkipError as exc:
                warnings.append(f"Row {idx}: skipped — {exc}")
                dropped += 1

        canonical = pd.DataFrame(records, columns=_CANONICAL_COLUMNS)
        canonical["date"] = pd.to_datetime(canonical["date"])
        canonical["amount"] = canonical["amount"].astype(float)
        return canonical, warnings, dropped

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        """Keep the first occurrence of each transaction_id."""
        return df.drop_duplicates(subset=["transaction_id"], keep="first")

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_date(self, raw: str, schema: SchemaProfile) -> date:
        raw = str(raw).strip()
        if not raw:
            raise _RowSkipError("missing date")
        for fmt in schema.date_formats:
            try:
                return datetime.strptime(raw, fmt).date()
            except ValueError:
                continue
        raise _RowSkipError(f"unparseable date '{raw}'")

    def _parse_amount(self, row: pd.Series, schema: SchemaProfile) -> float:
        if schema.amount_col:
            raw = str(row.get(schema.amount_col, "")).strip()
            return _to_float(raw, schema)

        debit_raw = str(row.get(schema.debit_col or "", "")).strip()
        credit_raw = str(row.get(schema.credit_col or "", "")).strip()

        debit = _to_float(debit_raw, schema) if debit_raw else 0.0
        credit = _to_float(credit_raw, schema) if credit_raw else 0.0

        if debit == 0.0 and credit == 0.0:
            raise _RowSkipError("both debit and credit are empty/zero")

        # Convention: expenses are negative
        return credit - debit

    @staticmethod
    def _parse_currency(row: pd.Series, schema: SchemaProfile) -> str:
        if schema.currency_col:
            val = str(row.get(schema.currency_col, "")).strip().upper()
            if len(val) == 3:
                return val
        return schema.currency_default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _RowSkipError(Exception):
    """Raised when a row cannot be parsed and must be dropped."""


_CANONICAL_COLUMNS = [
    "transaction_id",
    "date",
    "description",
    "raw_description",
    "amount",
    "currency",
    "category",
    "account",
    "source_file",
    "ingested_at",
]


def _empty_canonical_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_CANONICAL_COLUMNS)


def _to_float(raw: str, schema: SchemaProfile) -> float:
    """Parse a locale-aware numeric string to float."""
    if not raw or raw in ("-", "–", "n/a", "N/A", ""):
        return 0.0
    # Remove currency symbols and surrounding whitespace
    cleaned = re.sub(r"[€$£¥\s]", "", raw)
    # Normalise thousands / decimal separators
    if schema.decimal_sep == ",":
        cleaned = cleaned.replace(schema.thousands_sep, "").replace(",", ".")
    else:
        cleaned = cleaned.replace(schema.thousands_sep, "")
    # Handle parentheses for negatives: (1 234,56) → -1234.56
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    try:
        return float(cleaned)
    except ValueError:
        raise _RowSkipError(f"unparseable amount '{raw}'")


def _clean_description(text: str) -> str:
    """
    Normalise a bank transaction narrative:
    - Remove excessive whitespace / control characters
    - Collapse repeated spaces
    - Strip leading reference codes like "VIR SEPA ", "CB *", "PAIEMENT CB"
    - Preserve unicode (merchant names in non-ASCII scripts kept intact)
    """
    # Unicode normalisation
    text = unicodedata.normalize("NFKC", text)
    # Remove control characters
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    # Strip common bank prefixes (non-destructive: only if text has more content)
    prefixes = (
        r"^(VIR(EMENT)?\s+(SEPA\s+)?|VIREMENT\s+|"
        r"PAIEMENT\s+(CB|PAR CARTE)?\s*|CB\s*\*?\s*|"
        r"PRELEVEMENT\s+(SEPA\s+)?|RETRAIT\s+(DAB\s+)?|"
        r"ACHAT\s+(CB\s+)?|AVOIR\s+)"
    )
    stripped = re.sub(prefixes, "", text, flags=re.IGNORECASE).strip()
    result = stripped if stripped else text
    # Collapse whitespace
    return re.sub(r"\s+", " ", result).strip()


def _make_id(txn_date: date, description: str, amount: float, account: str) -> str:
    """Deterministic SHA-256 transaction ID."""
    payload = f"{txn_date}|{description.lower()}|{amount:.6f}|{account}"
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Finance Engine — CSV Ingester")
    parser.add_argument("path", help="CSV file or directory to ingest")
    parser.add_argument("--profile", "-p", default=None, help="Schema profile name")
    parser.add_argument("--account", "-a", default=None, help="Account label")
    parser.add_argument("--out", "-o", default=None, help="Output parquet path")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    ingester = CSVIngester()

    target = Path(args.path)
    if target.is_dir():
        df = ingester.ingest_directory(target, profile=args.profile)
    else:
        result = ingester.ingest(target, profile=args.profile, account=args.account)
        df = result.dataframe
        print(f"\nProfile used : {result.profile_used}")
        print(f"Rows read    : {result.total_rows_read}")
        print(f"After dedup  : {result.rows_after_dedup}")
        print(f"Dropped      : {result.rows_dropped}")
        if result.warnings:
            print(f"Warnings ({len(result.warnings)}):")
            for w in result.warnings[:10]:
                print(f"  • {w}")

    print(f"\n{df.shape[0]} transactions\n")
    print(df.head(10).to_string(index=False))

    if args.out:
        out = Path(args.out)
        df.to_parquet(out, index=False)
        print(f"\nSaved → {out}")
