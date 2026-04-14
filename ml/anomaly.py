"""
ml/anomaly.py — Isolation Forest Anomaly Detector
==================================================
Detects unusual transactions using sklearn's IsolationForest.

Features used
-------------
  log_amount       — log(|amount|) captures magnitude across scales
  day_of_week      — 0-6 (Monday-Sunday)
  hour             — hour extracted from ingested_at timestamp
  merchant_freq    — how often this exact merchant appears (rare = suspicious)
  amount_zscore    — Z-score of |amount| vs same merchant's history
  is_weekend       — binary flag

What counts as anomalous?
-------------------------
  • Unusually large transaction at a known merchant  (e.g. €400 at a grocery)
  • Transaction at a completely new merchant with a large amount
  • Payment at an unusual hour (3 AM card payment)
  • Rare merchants that appear only once with a large amount

Output columns added to the DataFrame
--------------------------------------
  anomaly_score    : float  — IsolationForest raw score (lower = more anomalous)
  anomaly_flag     : bool   — True if score < threshold (top ~5% outliers)
  anomaly_rank     : int    — 1 = most anomalous

Usage
-----
    from etl import FinanceETL
    from ml.anomaly import AnomalyDetector

    etl = FinanceETL()
    df  = etl.query("SELECT * FROM transactions WHERE NOT is_internal")

    det     = AnomalyDetector()
    results = det.fit_predict(df)

    flagged = results[results["anomaly_flag"]].sort_values("anomaly_rank")
    print(flagged[["date", "description", "amount", "anomaly_score"]].head(20))
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Fits an IsolationForest on transaction features and flags outliers.

    Parameters
    ----------
    contamination : expected fraction of anomalies (default 0.05 = 5%)
    n_estimators  : number of trees (default 200)
    random_state  : reproducibility seed
    """

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self._forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._scaler  = StandardScaler()
        self._fitted  = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """Fit the model on df (should be real, non-internal transactions)."""
        X = self._build_features(df)
        X_scaled = self._scaler.fit_transform(X)
        self._forest.fit(X_scaled)
        self._fitted = True
        logger.info("AnomalyDetector fitted on %d transactions", len(df))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return df with anomaly_score, anomaly_flag, anomaly_rank columns added.
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        X = self._build_features(df)
        X_scaled = self._scaler.transform(X)

        # decision_function: higher = more normal, lower = more anomalous
        scores = self._forest.decision_function(X_scaled)
        labels = self._forest.predict(X_scaled)     # -1 = anomaly, 1 = normal

        out = df.copy()
        out["anomaly_score"] = scores.round(4)
        out["anomaly_flag"]  = labels == -1
        # Rank: 1 = most anomalous
        out["anomaly_rank"]  = pd.Series(-scores, index=df.index).rank(method="min").astype(int)
        return out

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and predict in one call."""
        return self.fit(df).predict(df)

    def top_anomalies(
        self, df: pd.DataFrame, n: int = 20
    ) -> pd.DataFrame:
        """
        Return the n most anomalous transactions with explanatory context.
        """
        result = self.fit_predict(df)
        flagged = result[result["anomaly_flag"]].copy()

        cols = [
            c for c in [
                "anomaly_rank", "date", "description", "amount",
                "currency", "tx_type", "account", "anomaly_score",
            ]
            if c in flagged.columns
        ]
        return flagged[cols].sort_values("anomaly_rank").head(n)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix from a transactions DataFrame."""
        amounts = df["amount"].astype(float)

        # 1. Log absolute amount
        log_amount = np.log1p(amounts.abs()).values

        # 2. Amount sign (expense vs income)
        sign = np.sign(amounts).values

        # 3. Day of week (use column if enriched, else derive from date)
        if "day_of_week" in df.columns:
            dow = df["day_of_week"].astype(float).values
        else:
            dow = pd.to_datetime(df["date"]).dt.dayofweek.values.astype(float)

        # 4. Hour of transaction (from ingested_at or default 12)
        if "ingested_at" in df.columns:
            hour = pd.to_datetime(df["ingested_at"]).dt.hour.astype(float).values
        else:
            hour = np.full(len(df), 12.0)

        # 5. Is weekend
        is_weekend = (dow >= 5).astype(float)

        # 6. Merchant frequency (rarer merchant → higher suspicion signal)
        freq_map   = df["description"].value_counts().to_dict()
        merch_freq = df["description"].map(freq_map).astype(float).values
        log_freq   = np.log1p(merch_freq)

        # 7. Z-score of |amount| within the same merchant
        # (how unusual is this amount for this merchant?)
        abs_amount = amounts.abs()
        merch_mean = df.groupby("description")["amount"].transform(
            lambda x: x.abs().mean()
        ).values
        merch_std  = df.groupby("description")["amount"].transform(
            lambda x: x.abs().std() if len(x) > 1 else 0.0
        ).fillna(0.0).values
        # Avoid division by zero
        safe_std   = np.where(merch_std < 1e-6, 1.0, merch_std)
        z_score    = ((abs_amount.values - merch_mean) / safe_std).clip(-5, 5)

        # 8. Month (captures seasonal patterns)
        if "month" in df.columns:
            month = df["month"].astype(float).values
        else:
            month = pd.to_datetime(df["date"]).dt.month.astype(float).values

        X = np.column_stack([
            log_amount,   # magnitude
            sign,         # direction
            dow,          # day of week
            hour,         # time of day
            is_weekend,   # weekend flag
            log_freq,     # merchant frequency
            z_score,      # amount vs merchant history
            month,        # month
        ])
        return X
