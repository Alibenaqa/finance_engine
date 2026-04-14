"""
watcher.py — Automatic pipeline trigger
========================================
Watches data/inbox/ for new CSV files and automatically runs the full pipeline:
  1. Auto-detect bank profile from filename or column headers
  2. Ingest with Layer 1 (CSVIngester)
  3. Load into DuckDB with Layer 2 (FinanceETL)
  4. Re-run ML models (classifier + anomaly detector)
  5. Move processed file to data/processed/

Usage
-----
    python watcher.py              # watches data/inbox/ (default)
    python watcher.py --inbox /path/to/folder
    python watcher.py --once       # process existing files and exit (no watch)

Drop any CSV in data/inbox/ and the pipeline runs automatically.
The Streamlit dashboard will pick up the changes on next refresh (auto every 30s).
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Profile auto-detection ────────────────────────────────────────────────────
# Maps filename substrings (lowercase) → ingester profile name
_FILENAME_PROFILE_MAP: list[tuple[str, str]] = [
    ("revolut",         "revolut"),
    ("bnp",             "bnp"),
    ("societe",         "societe_generale"),
    ("sg_",             "societe_generale"),
    ("credit_agricole", "credit_agricole"),
    ("ca_",             "credit_agricole"),
    ("n26",             "n26"),
    ("chase",           "chase"),
    ("bofa",            "bofa"),
    ("bank_of_america", "bofa"),
]

# Column-header signatures for when filename gives no hint
_COLUMN_PROFILE_MAP: list[tuple[frozenset, str]] = [
    (frozenset({"type", "product", "started date", "completed date",
                "description", "amount", "fee", "currency", "state", "balance"}),
     "revolut"),
    (frozenset({"date", "libellé", "débit euros", "crédit euros"}), "bnp"),
    (frozenset({"date", "description", "amount", "type"}),          "chase"),
]


def _detect_profile(path: Path) -> str:
    """Return best-guess ingester profile for this CSV file."""
    name_lower = path.stem.lower()
    for keyword, profile in _FILENAME_PROFILE_MAP:
        if keyword in name_lower:
            logger.info("Profile detected from filename: %s → %s", path.name, profile)
            return profile

    # Fall back to column inspection
    try:
        import pandas as pd
        header = pd.read_csv(path, nrows=0, encoding="utf-8",
                             sep=None, engine="python")
        cols = frozenset(c.strip().lower() for c in header.columns)
        best_match, best_score = "generic", 0
        for signature, profile in _COLUMN_PROFILE_MAP:
            score = len(signature & cols)
            if score > best_score:
                best_match, best_score = profile, score
        logger.info("Profile detected from columns: %s → %s (score %d)",
                    path.name, best_match, best_score)
        return best_match
    except Exception:
        logger.warning("Could not inspect columns for %s — using 'generic'", path.name)
        return "generic"


def _derive_account(path: Path, profile: str) -> str:
    """Derive a human-readable account label from filename + profile."""
    stem = path.stem
    # Strip common date suffixes like _2025-08_2026-04
    import re
    clean = re.sub(r"[_-]?\d{4}[-_]\d{2}.*$", "", stem).replace("_", " ").title()
    return clean or profile.title()


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(csv_path: Path, inbox: Path, processed_dir: Path) -> bool:
    """
    Run the full ingestion + ETL + ML pipeline for one CSV file.
    Returns True on success, False on error.
    """
    logger.info("=" * 60)
    logger.info("New file detected: %s", csv_path.name)

    profile = _detect_profile(csv_path)
    account = _derive_account(csv_path, profile)
    logger.info("Profile: %s  |  Account: %s", profile, account)

    # ── Layer 1: ingest ──────────────────────────────────────────────────────
    try:
        from ingester import CSVIngester
        result = CSVIngester().ingest(csv_path, profile=profile, account=account)
        df = result.dataframe
        logger.info("Ingested %d rows from %s", len(df), csv_path.name)
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        return False

    # ── Layer 2: ETL ─────────────────────────────────────────────────────────
    try:
        from etl import FinanceETL
        etl = FinanceETL("db/finance.duckdb")
        inserted = etl.load(df)
        logger.info("ETL: %d new rows inserted into DuckDB", inserted)
        summary = etl.get_summary()
        logger.info(
            "DB now has %d transactions (%s → %s)",
            summary["real_tx"], summary["first_date"], summary["last_date"],
        )
    except Exception as e:
        logger.error("ETL failed: %s", e)
        return False

    # ── Layer 3: ML ───────────────────────────────────────────────────────────
    if inserted == 0:
        logger.info("No new rows — skipping ML re-run (data already up to date)")
    else:
        _run_ml(etl)

    # ── Move to processed ────────────────────────────────────────────────────
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest = processed_dir / csv_path.name
    # Avoid overwriting if same filename processed before
    if dest.exists():
        dest = processed_dir / f"{csv_path.stem}_{int(time.time())}{csv_path.suffix}"
    shutil.move(str(csv_path), dest)
    logger.info("Moved to processed: %s", dest)
    logger.info("Pipeline complete for %s", csv_path.name)
    return True


def _run_ml(etl) -> None:
    """Re-run classifier and anomaly detector, log results."""
    # Classifier
    try:
        from ml.classifier import TransactionClassifier
        df = etl.query("SELECT * FROM transactions")
        clf = TransactionClassifier()
        tagged = clf.fit_predict(df)
        top_cats = (
            tagged[tagged["amount"] < 0]
            .groupby("predicted_category")["amount"]
            .sum()
            .sort_values()
            .head(5)
        )
        logger.info("Top spending categories:\n%s", top_cats.to_string())
    except Exception as e:
        logger.warning("Classifier failed (non-fatal): %s", e)

    # Anomaly detector
    try:
        from ml.anomaly import AnomalyDetector
        df_real = etl.query("SELECT * FROM transactions WHERE NOT is_internal")
        det = AnomalyDetector(contamination=0.05)
        results = det.fit_predict(df_real)
        n_flagged = results["anomaly_flag"].sum()
        logger.info("Anomaly detector: %d transactions flagged", n_flagged)
        if n_flagged:
            top = results[results["anomaly_flag"]].nsmallest(3, "anomaly_score")
            for _, row in top.iterrows():
                logger.info(
                    "  [rank %d] %s  %s  %.2f€",
                    row["anomaly_rank"],
                    str(row.get("date", ""))[:10],
                    str(row.get("description", ""))[:30],
                    float(row.get("amount", 0)),
                )
    except Exception as e:
        logger.warning("Anomaly detector failed (non-fatal): %s", e)


# ── File event handler ────────────────────────────────────────────────────────

def _make_handler(inbox: Path, processed_dir: Path):
    from watchdog.events import FileSystemEventHandler

    class _Handler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() != ".csv":
                return
            # Small delay to ensure the file is fully written
            time.sleep(1)
            run_pipeline(path, inbox, processed_dir)

        def on_moved(self, event):
            """Also catch files moved/copied into the inbox."""
            if event.is_directory:
                return
            path = Path(event.dest_path)
            if path.suffix.lower() != ".csv":
                return
            time.sleep(1)
            run_pipeline(path, inbox, processed_dir)

    return _Handler()


# ── Batch mode: process existing files ───────────────────────────────────────

def process_existing(inbox: Path, processed_dir: Path) -> int:
    """Process all CSV files already present in inbox. Returns count processed."""
    csvs = sorted(inbox.glob("*.csv"))
    if not csvs:
        logger.info("No CSV files found in %s", inbox)
        return 0
    logger.info("Found %d CSV file(s) to process", len(csvs))
    ok = 0
    for path in csvs:
        if run_pipeline(path, inbox, processed_dir):
            ok += 1
    return ok


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finance Engine — automatic pipeline watcher"
    )
    parser.add_argument(
        "--inbox", default="data/inbox",
        help="Folder to watch for new CSV files (default: data/inbox)",
    )
    parser.add_argument(
        "--processed", default="data/processed",
        help="Folder for processed files (default: data/processed)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Process existing files in inbox then exit (no continuous watch)",
    )
    args = parser.parse_args()

    inbox     = Path(args.inbox)
    processed = Path(args.processed)
    inbox.mkdir(parents=True, exist_ok=True)

    # Always process files already waiting in inbox
    process_existing(inbox, processed)

    if args.once:
        logger.info("--once mode: done.")
        return

    # Continuous watch
    try:
        from watchdog.observers import Observer
    except ImportError:
        logger.error(
            "watchdog is not installed. Run: pip install watchdog\n"
            "Or use --once to process existing files without watching."
        )
        return

    handler  = _make_handler(inbox, processed)
    observer = Observer()
    observer.schedule(handler, str(inbox), recursive=False)
    observer.start()

    logger.info("Watching %s for new CSV files …  (Ctrl+C to stop)", inbox.resolve())
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        logger.info("Stopping watcher …")
        observer.stop()
    observer.join()
    logger.info("Watcher stopped.")


if __name__ == "__main__":
    main()
