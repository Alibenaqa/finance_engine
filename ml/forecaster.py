"""
ml/forecaster.py — Prophet Spending Forecaster
===============================================
Aggregates daily expenses from DuckDB and fits a Facebook Prophet model
to predict future spending over a configurable horizon.

Features used by Prophet
------------------------
  • Daily expense totals as the target series (y)
  • Weekly seasonality  (higher spend on weekends)
  • Monthly seasonality (pay-cycle patterns)
  • French public holidays as special events (optional)

Output
------
  ForecastResult
    .forecast   : pd.DataFrame   — Prophet output (ds, yhat, yhat_lower,
                                   yhat_upper + all components)
    .summary    : pd.DataFrame   — clean daily forecast for the horizon
    .monthly    : pd.DataFrame   — aggregated monthly forecast
    .plot()                      — returns a matplotlib Figure

Usage
-----
    from etl import FinanceETL
    from ml.forecaster import SpendingForecaster

    etl = FinanceETL()
    fc  = SpendingForecaster(etl)
    res = fc.predict(horizon_days=60)

    print(res.summary.tail(10))
    res.plot().savefig("forecast.png")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False
    logger.warning("Prophet not available — falling back to linear trend model")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    summary: pd.DataFrame
    monthly: pd.DataFrame
    horizon_days: int
    model_type: str                     # "prophet" | "linear"
    _model: object = field(repr=False, default=None)
    _history: pd.DataFrame = field(repr=False, default=None)

    def plot(self):
        """Return a matplotlib Figure with history + forecast."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # ── Daily forecast ───────────────────────────────────────────────────
        ax = axes[0]
        hist = self._history
        fc   = self.summary

        ax.fill_between(fc["date"], fc["lower"], fc["upper"],
                        alpha=0.2, color="steelblue", label="Intervalle 80%")
        ax.plot(fc["date"], fc["forecast"], color="steelblue",
                linewidth=2, label="Prévision")
        ax.plot(hist["ds"], hist["y"], color="black",
                alpha=0.6, linewidth=1, label="Historique")
        ax.axvline(hist["ds"].max(), color="red", linestyle="--",
                   alpha=0.5, label="Aujourd'hui")
        ax.set_title("Prévision des dépenses journalières")
        ax.set_ylabel("Dépenses (€)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Monthly forecast ─────────────────────────────────────────────────
        ax2 = axes[1]
        m   = self.monthly
        colors = ["#d62728" if f else "#1f77b4"
                  for f in m["is_forecast"]]
        bars = ax2.bar(m["period"], m["total"], color=colors, width=0.6)
        ax2.set_title("Dépenses mensuelles (réel vs prévu)")
        ax2.set_ylabel("Total (€)")
        ax2.set_xlabel("")
        ax2.tick_params(axis="x", rotation=45)

        from matplotlib.patches import Patch
        ax2.legend(handles=[
            Patch(color="#1f77b4", label="Réel"),
            Patch(color="#d62728", label="Prévu"),
        ], fontsize=8)
        ax2.grid(True, axis="y", alpha=0.3)

        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

class SpendingForecaster:
    """
    Fits a time-series model on historical daily expenses and forecasts
    future spending.

    Parameters
    ----------
    etl            : FinanceETL instance (used to pull daily expense data)
    account        : filter to a specific account (None = all accounts)
    exclude_cats   : categories to exclude from the expense series
                     (e.g. rent distorts the day-to-day pattern)
    """

    def __init__(
        self,
        etl,
        *,
        account: Optional[str] = None,
        exclude_categories: Optional[list[str]] = None,
    ) -> None:
        self._etl = etl
        self._account = account
        self._exclude_cats = exclude_categories or ["Loyer", "Virements"]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def predict(self, horizon_days: int = 30) -> ForecastResult:
        """
        Fit on all available expense history and forecast `horizon_days` ahead.
        """
        history = self._get_daily_series()

        if len(history) < 14:
            raise ValueError(
                f"Not enough history ({len(history)} days). Need at least 14."
            )

        if _PROPHET_AVAILABLE:
            return self._prophet_forecast(history, horizon_days)
        return self._linear_forecast(history, horizon_days)

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _get_daily_series(self) -> pd.DataFrame:
        """Pull daily absolute expenses from DuckDB as a Prophet-ready df."""
        where_clauses = ["amount < 0", "NOT is_internal"]
        if self._account:
            where_clauses.append(f"account = '{self._account}'")
        if self._exclude_cats:
            # If classifier has been run, exclude these categories
            # Otherwise skip (column may not exist yet)
            pass

        where = " AND ".join(where_clauses)
        df = self._etl.query(f"""
            SELECT
                date                    AS ds,
                ABS(SUM(amount))        AS y
            FROM transactions
            WHERE {where}
            GROUP BY date
            ORDER BY date
        """)
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"]  = df["y"].astype(float)

        # Fill missing dates with 0 (no spending on that day)
        full_range = pd.DataFrame({
            "ds": pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
        })
        df = full_range.merge(df, on="ds", how="left").fillna({"y": 0.0})
        return df

    # ------------------------------------------------------------------
    # Prophet forecasting
    # ------------------------------------------------------------------

    def _prophet_forecast(
        self, history: pd.DataFrame, horizon_days: int
    ) -> ForecastResult:
        logger.info("Fitting Prophet on %d days of history …", len(history))

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            seasonality_mode="multiplicative",
            interval_width=0.80,
            changepoint_prior_scale=0.15,
        )
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

        # French public holidays
        try:
            model.add_country_holidays(country_name="FR")
        except Exception:
            pass

        model.fit(history)

        future   = model.make_future_dataframe(periods=horizon_days)
        forecast = model.predict(future)

        # Clip negative predictions to 0 (can't spend negative)
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            forecast[col] = forecast[col].clip(lower=0)

        summary  = self._build_summary(forecast, history, horizon_days)
        monthly  = self._build_monthly(history, forecast, horizon_days)

        logger.info("Prophet forecast ready — horizon %d days", horizon_days)
        return ForecastResult(
            forecast=forecast,
            summary=summary,
            monthly=monthly,
            horizon_days=horizon_days,
            model_type="prophet",
            _model=model,
            _history=history,
        )

    # ------------------------------------------------------------------
    # Linear fallback
    # ------------------------------------------------------------------

    def _linear_forecast(
        self, history: pd.DataFrame, horizon_days: int
    ) -> ForecastResult:
        """Simple linear regression fallback when Prophet is unavailable."""
        from sklearn.linear_model import Ridge
        import numpy as np

        X = np.arange(len(history)).reshape(-1, 1)
        y = history["y"].values

        model = Ridge()
        model.fit(X, y)

        last_idx = len(history)
        future_X = np.arange(last_idx, last_idx + horizon_days).reshape(-1, 1)
        preds    = model.predict(future_X).clip(0)
        std      = y.std()

        last_date = history["ds"].max()
        future_dates = pd.date_range(
            last_date + pd.Timedelta(days=1), periods=horizon_days
        )

        forecast_df = pd.DataFrame({
            "ds":         future_dates,
            "yhat":       preds,
            "yhat_lower": (preds - std).clip(0),
            "yhat_upper": preds + std,
        })
        full_forecast = pd.concat([
            history.rename(columns={"y": "yhat"}).assign(
                yhat_lower=lambda d: d["yhat"],
                yhat_upper=lambda d: d["yhat"],
            ),
            forecast_df,
        ], ignore_index=True)

        summary = self._build_summary(full_forecast, history, horizon_days)
        monthly = self._build_monthly(history, full_forecast, horizon_days)

        return ForecastResult(
            forecast=full_forecast,
            summary=summary,
            monthly=monthly,
            horizon_days=horizon_days,
            model_type="linear",
            _model=model,
            _history=history,
        )

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        forecast: pd.DataFrame,
        history: pd.DataFrame,
        horizon_days: int,
    ) -> pd.DataFrame:
        cutoff = history["ds"].max()
        future = forecast[forecast["ds"] > cutoff].copy()
        return pd.DataFrame({
            "date":     future["ds"].dt.date,
            "forecast": future["yhat"].round(2),
            "lower":    future["yhat_lower"].round(2),
            "upper":    future["yhat_upper"].round(2),
        }).reset_index(drop=True)

    @staticmethod
    def _build_monthly(
        history: pd.DataFrame,
        forecast: pd.DataFrame,
        horizon_days: int,
    ) -> pd.DataFrame:
        cutoff = history["ds"].max()

        # Historical monthly totals
        hist_m = (
            history.assign(period=history["ds"].dt.to_period("M"))
            .groupby("period")["y"]
            .sum()
            .reset_index()
            .rename(columns={"y": "total"})
            .assign(is_forecast=False)
        )

        # Forecast monthly totals
        fc_df = forecast[forecast["ds"] > cutoff].copy()
        fc_m = (
            fc_df.assign(period=fc_df["ds"].dt.to_period("M"))
            .groupby("period")["yhat"]
            .sum()
            .reset_index()
            .rename(columns={"yhat": "total"})
            .assign(is_forecast=True)
        )

        combined = pd.concat([hist_m, fc_m], ignore_index=True)
        combined["period"] = combined["period"].astype(str)
        combined["total"]  = combined["total"].round(2)
        return combined.sort_values("period").reset_index(drop=True)
