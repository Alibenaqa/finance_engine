# Smart Personal Finance Engine

A fully local, privacy-first personal finance pipeline built in Python. No cloud, no third-party API — your bank data never leaves your machine.

---

## Architecture

The project is structured as 4 independent layers that compose into a complete pipeline:

```
CSV files
    │
    ▼
┌─────────────────────────────────────┐
│  Layer 1 — ingester.py              │
│  Multi-bank CSV normalisation        │
│  → canonical DataFrame               │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Layer 2 — etl.py                   │
│  DuckDB persistence + enrichment     │
│  → typed, deduplicated SQL store     │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Layer 3 — ml/                      │
│  ├── classifier.py   (XGBoost)      │
│  ├── forecaster.py   (Prophet)      │
│  └── anomaly.py      (IsoForest)    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Layer 4 — dashboard.py             │
│  Streamlit interactive dashboard     │
└─────────────────────────────────────┘
```

---

## Features

### Layer 1 — Multi-bank CSV Ingestion (`ingester.py`)

- Auto-detects bank format from column names
- Built-in profiles: **Revolut**, **BNP Paribas**, **Société Générale**, **Crédit Agricole**, **N26**, **Chase**, **Bank of America**, generic fallback
- Handles debit/credit split columns (BNP-style) and unified amount columns (Revolut-style)
- Multi-encoding fallback: UTF-8 → latin-1 → cp1252 with column-name validation
- Description cleaning: strips bank noise prefixes (`VIR SEPA`, `PAIEMENT CB`, `PRELEVEMENT`, etc.)
- SHA-256 content-hash `transaction_id` for deduplication across re-ingestions
- Canonical output schema:

| Column | Type | Description |
|---|---|---|
| `transaction_id` | str | SHA-256 hash of date\|description\|amount\|account |
| `date` | date | Parsed transaction date |
| `description` | str | Cleaned merchant name |
| `raw_description` | str | Original bank description |
| `amount` | float | Signed amount (negative = expense) |
| `currency` | str | ISO 4217 currency code |
| `category` | str | Bank-provided category (if any) |
| `account` | str | Account label |
| `source_file` | str | Source CSV filename |
| `ingested_at` | datetime | Ingestion timestamp |

### Layer 2 — ETL & DuckDB Storage (`etl.py`)

- Persists canonical transactions into a local **DuckDB** file (`db/finance.duckdb`)
- `INSERT OR IGNORE` deduplication on `transaction_id` primary key
- Enrichment columns added on load:

| Column | Description |
|---|---|
| `tx_type` | Inferred type: `card_payment`, `transfer_in`, `transfer_out`, `topup`, `fee`, `interest`, `atm`, `refund`, `exchange`, `savings_move`, `pocket_move` |
| `is_internal` | `True` for intra-account moves (savings ↔ current, pocket transfers) |
| `year`, `month`, `week`, `day_of_week` | Temporal breakdown |

- Persistent SQL views: `v_real`, `v_expenses`, `v_income`, `v_monthly`, `v_top_merchants`
- Query API: `get_summary()`, `get_monthly()`, `get_top_merchants()`, `get_expenses()`, `get_income()`, `get_daily_balance()`

### Layer 3 — ML Models (`ml/`)

#### Transaction Classifier (`ml/classifier.py`)
- Two-step approach: regex rule cascade → XGBoost generalisation
- **11 categories**: Alimentation, Restauration, Transport, Abonnements, Shopping, Beauté, Loisirs, Loyer, Virements, Revenus, Autre
- Features: TF-IDF char-ngrams (2–4) on merchant name + `[log|amount|, sign(amount), day_of_week]`
- Adds `predicted_category` column to any transactions DataFrame
- Save/load via pickle (`clf.save("models/classifier.pkl")`)

#### Spending Forecaster (`ml/forecaster.py`)
- Fits **Facebook Prophet** on historical daily expenses
- Weekly + monthly seasonality, French public holidays
- Multiplicative seasonality mode, 80% prediction interval
- Falls back to Ridge linear regression if Prophet is not installed
- Returns `ForecastResult` with `.summary`, `.monthly`, `.plot()` (matplotlib Figure)

#### Anomaly Detector (`ml/anomaly.py`)
- **Isolation Forest** (200 trees, contamination=5%)
- 8 engineered features: log-amount, sign, day-of-week, hour, is-weekend, merchant frequency, amount z-score vs merchant history, month
- Adds `anomaly_score`, `anomaly_flag`, `anomaly_rank` columns
- `top_anomalies(df, n=20)` returns the most suspicious transactions with context

### Layer 4 — Streamlit Dashboard (`dashboard.py`)

5-tab interactive dashboard:

| Tab | Content |
|---|---|
| **Vue d'ensemble** | KPI cards (income / expenses / net / tx count), monthly bar chart, cumulative balance line |
| **Catégories** | Pie chart + ranked bar of spend by ML-predicted category, per-category transaction list |
| **Marchands** | Top merchants by total spend, sortable table |
| **Anomalies** | Anomaly score scatter plot, flagged transaction table with rank |
| **Prévisions** | Prophet/linear forecast line with confidence band, monthly actual vs forecast bar chart |

Sidebar features:
- Date range filter (applied to all tabs)
- Forecast horizon slider (7–180 days)
- CSV upload with bank profile selector (processes and reloads in place)

---

## Setup

### Requirements

- Python 3.10+
- macOS: `brew install libomp` (required for XGBoost)

### Install

```bash
git clone <repo-url>
cd finance_engine

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Load your bank data

```bash
# Ingest a Revolut export
python etl.py data/raw/your_revolut_export.csv --profile revolut --account "Revolut"

# Ingest a BNP export
python etl.py data/raw/your_bnp_export.csv --profile bnp --account "BNP"

# Check what's loaded
python etl.py --summary
```

Supported `--profile` values: `revolut`, `bnp`, `societe_generale`, `credit_agricole`, `n26`, `chase`, `bofa`, `generic`

### Run the dashboard

```bash
streamlit run dashboard.py
```

Then open [http://localhost:8501](http://localhost:8501).

---

## Project Structure

```
finance_engine/
├── ingester.py          # Layer 1 — CSV ingestion
├── etl.py               # Layer 2 — DuckDB ETL
├── ml/
│   ├── __init__.py
│   ├── classifier.py    # XGBoost transaction categorizer
│   ├── forecaster.py    # Prophet spending forecaster
│   └── anomaly.py       # Isolation Forest anomaly detector
├── dashboard.py         # Layer 4 — Streamlit dashboard
├── requirements.txt
├── data/
│   └── samples/         # Sample CSVs for testing (real exports in .gitignore)
└── db/                  # DuckDB database (in .gitignore)
```

---

## Sample Data

The `data/samples/` directory contains anonymised sample exports for testing:

| File | Bank | Format |
|---|---|---|
| `revolut_sample.csv` | Revolut | Unified amount, ISO dates |
| `bnp_sample.csv` | BNP Paribas | Debit/Credit split, French dates |
| `chase_sample.csv` | Chase (US) | Debit/Credit split, US dates |

```bash
# Quick test with samples
python etl.py data/samples/revolut_sample.csv --profile revolut --account "Revolut Demo"
streamlit run dashboard.py
```

---

## Privacy

- All data is processed and stored **locally** — no network requests, no external APIs (Prophet uses local model fitting only)
- `data/raw/` and `db/` are excluded from git via `.gitignore`
- The DuckDB file is a single binary at `db/finance.duckdb` — back it up or delete it at will
