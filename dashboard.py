"""
Layer 4 — Streamlit Dashboard
==============================
Run with:
    streamlit run dashboard.py

5 tabs
------
  Vue d'ensemble  — KPIs, monthly income/expense bar chart, cumulative balance
  Catégories      — XGBoost spending breakdown (pie + bar)
  Marchands       — Top merchants ranked by spend
  Anomalies       — Isolation Forest flagged transactions
  Prévisions      — Prophet 60-day spending forecast
"""

import warnings
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Finance Engine",
    page_icon="💶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette ───────────────────────────────────────────────────────────────────
_INCOME_COLOR   = "#2ecc71"
_EXPENSE_COLOR  = "#e74c3c"
_NEUTRAL_COLOR  = "#3498db"
_FORECAST_COLOR = "#9b59b6"

_CATEGORY_COLORS = {
    "Alimentation":  "#f39c12",
    "Restauration":  "#e67e22",
    "Transport":     "#3498db",
    "Abonnements":   "#9b59b6",
    "Shopping":      "#1abc9c",
    "Beauté":        "#e91e63",
    "Loisirs":       "#00bcd4",
    "Loyer":         "#e74c3c",
    "Virements":     "#95a5a6",
    "Revenus":       "#2ecc71",
    "Autre":         "#bdc3c7",
}


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def _get_etl():
    from etl import FinanceETL
    return FinanceETL("db/finance.duckdb")


@st.cache_data(show_spinner="Classification XGBoost …")
def _run_classifier(_etl):
    from ml.classifier import TransactionClassifier
    df  = _etl.query("SELECT * FROM transactions")
    clf = TransactionClassifier()
    return clf.fit_predict(df)


@st.cache_data(show_spinner="Détection d'anomalies …")
def _run_anomaly(_etl):
    from ml.anomaly import AnomalyDetector
    df   = _etl.query("SELECT * FROM transactions WHERE NOT is_internal")
    det  = AnomalyDetector(contamination=0.05)
    return det.fit_predict(df)


@st.cache_data(show_spinner="Prévision Prophet …")
def _run_forecast(_etl, horizon: int):
    from ml.forecaster import SpendingForecaster
    fc = SpendingForecaster(_etl)
    return fc.predict(horizon_days=horizon)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_eur(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,.2f} €".replace(",", " ")


def _apply_date_filter(df: pd.DataFrame, start, end) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    return df[mask]


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar(etl):
    st.sidebar.title("💶 Finance Engine")
    st.sidebar.markdown("---")

    # Date range
    summary = etl.get_summary()
    d_min   = pd.to_datetime(summary["first_date"]).date()
    d_max   = pd.to_datetime(summary["last_date"]).date()

    st.sidebar.subheader("Période")
    start = st.sidebar.date_input("Début",  value=d_min, min_value=d_min, max_value=d_max)
    end   = st.sidebar.date_input("Fin",    value=d_max, min_value=d_min, max_value=d_max)

    # Forecast horizon
    st.sidebar.subheader("Prévision")
    horizon = st.sidebar.slider("Horizon (jours)", 14, 90, 30, step=7)

    # File upload
    st.sidebar.markdown("---")
    st.sidebar.subheader("Importer un relevé")
    uploaded = st.sidebar.file_uploader(
        "CSV bancaire", type=["csv"],
        help="Formats supportés : Revolut, BNP, SG, N26, Chase, BofA …"
    )
    profile = st.sidebar.selectbox(
        "Profil", ["(auto)", "revolut", "bnp", "societe_generale",
                   "credit_agricole", "n26", "chase", "bofa"]
    )
    account_name = st.sidebar.text_input("Nom du compte", "")

    if uploaded and st.sidebar.button("Ingérer", type="primary"):
        _ingest_upload(etl, uploaded, profile, account_name)

    return start, end, horizon


def _ingest_upload(etl, uploaded, profile, account_name):
    import tempfile
    from ingester import CSVIngester

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        ing   = CSVIngester()
        acct  = account_name or Path(uploaded.name).stem
        prof  = None if profile == "(auto)" else profile
        result = ing.ingest(tmp_path, profile=prof, account=acct)
        n = etl.load(result.dataframe)
        st.sidebar.success(f"✓ {n} nouvelles transactions chargées")
        # Clear caches so next render uses fresh data
        _run_classifier.clear()
        _run_anomaly.clear()
        _run_forecast.clear()
        st.rerun()
    except Exception as exc:
        st.sidebar.error(f"Erreur : {exc}")


# ── Tab 1 — Vue d'ensemble ────────────────────────────────────────────────────

def _tab_overview(etl, start, end):
    summary = etl.get_summary()

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenus totaux",   _fmt_eur(summary["total_income"] or 0))
    c2.metric("Dépenses totales", _fmt_eur(summary["total_expenses"] or 0))
    c3.metric("Solde net",        _fmt_eur(summary["net_balance"] or 0))
    c4.metric("Transactions",     f"{summary['real_tx']:,}")

    st.markdown("---")

    # Monthly income vs expenses
    monthly = etl.get_monthly()
    monthly = monthly[
        (monthly["period"] >= str(start)[:7]) &
        (monthly["period"] <= str(end)[:7])
    ]

    fig_monthly = go.Figure()
    fig_monthly.add_bar(
        x=monthly["period"], y=monthly["income"],
        name="Revenus", marker_color=_INCOME_COLOR,
    )
    fig_monthly.add_bar(
        x=monthly["period"], y=monthly["expenses"].abs(),
        name="Dépenses", marker_color=_EXPENSE_COLOR,
    )
    fig_monthly.add_scatter(
        x=monthly["period"], y=monthly["net"],
        mode="lines+markers", name="Solde net",
        line=dict(color=_NEUTRAL_COLOR, width=2, dash="dot"),
        yaxis="y",
    )
    fig_monthly.update_layout(
        barmode="group",
        title="Revenus vs Dépenses par mois",
        xaxis_title="", yaxis_title="€",
        legend=dict(orientation="h", y=1.1),
        height=380,
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Cumulative balance
    daily = etl.get_daily_balance()
    daily = _apply_date_filter(daily, start, end)

    fig_balance = px.area(
        daily, x="date", y="cumulative",
        title="Solde cumulé (dépenses réelles)",
        labels={"cumulative": "€", "date": ""},
        color_discrete_sequence=[_NEUTRAL_COLOR],
    )
    fig_balance.add_hline(y=0, line_dash="dot", line_color="red", opacity=0.5)
    fig_balance.update_layout(height=300)
    st.plotly_chart(fig_balance, use_container_width=True)


# ── Tab 2 — Catégories ────────────────────────────────────────────────────────

def _tab_categories(etl, start, end):
    with st.spinner("Classification XGBoost …"):
        df = _run_classifier(etl)

    df = _apply_date_filter(df, start, end)
    exp = df[(df["amount"] < 0) & (~df["is_internal"])].copy()
    exp["amount_abs"] = exp["amount"].abs()

    by_cat = (
        exp.groupby("predicted_category")["amount_abs"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "total", "count": "nb_tx"})
        .sort_values("total", ascending=False)
        .reset_index()
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_pie = px.pie(
            by_cat, values="total", names="predicted_category",
            title="Répartition des dépenses",
            color="predicted_category",
            color_discrete_map=_CATEGORY_COLORS,
            hole=0.4,
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            by_cat, x="total", y="predicted_category",
            orientation="h", title="Total par catégorie (€)",
            color="predicted_category",
            color_discrete_map=_CATEGORY_COLORS,
            text=by_cat["total"].map(lambda v: f"{v:,.0f} €"),
            labels={"total": "€", "predicted_category": ""},
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(showlegend=False, height=400,
                               yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, use_container_width=True)

    # Monthly category heatmap
    pivot = (
        exp.assign(period=exp["date"].dt.to_period("M").astype(str))
        .groupby(["period", "predicted_category"])["amount_abs"]
        .sum()
        .unstack(fill_value=0)
        .round(0)
    )
    fig_heat = px.imshow(
        pivot.T,
        title="Dépenses par catégorie × mois (€)",
        labels={"x": "Mois", "y": "Catégorie", "color": "€"},
        color_continuous_scale="Reds",
        aspect="auto",
        text_auto=True,
    )
    fig_heat.update_layout(height=380)
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("Détail par transaction"):
        st.dataframe(
            exp[["date", "description", "amount", "predicted_category"]]
            .sort_values("date", ascending=False)
            .rename(columns={
                "date": "Date", "description": "Marchand",
                "amount": "Montant (€)", "predicted_category": "Catégorie",
            }),
            use_container_width=True,
            height=350,
        )


# ── Tab 3 — Marchands ─────────────────────────────────────────────────────────

def _tab_merchants(etl, start, end):
    df = etl.get_expenses()
    df = _apply_date_filter(df, start, end)

    n_top = st.slider("Nombre de marchands", 10, 40, 20)

    by_merch = (
        df.groupby("description")["amount"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "total", "count": "nb_tx", "mean": "moyenne"})
        .sort_values("total")
        .head(n_top)
        .reset_index()
        .rename(columns={"description": "marchand"})
    )
    by_merch["total_abs"]  = by_merch["total"].abs().round(2)
    by_merch["moyenne_abs"] = by_merch["moyenne"].abs().round(2)

    fig = px.bar(
        by_merch, x="total_abs", y="marchand",
        orientation="h",
        title=f"Top {n_top} marchands par dépense totale",
        text=by_merch["total_abs"].map(lambda v: f"{v:,.0f} €"),
        color="total_abs",
        color_continuous_scale="Reds",
        labels={"total_abs": "Total (€)", "marchand": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False, coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
        height=max(400, n_top * 22),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        by_merch[["marchand", "nb_tx", "total_abs", "moyenne_abs"]]
        .rename(columns={
            "marchand": "Marchand", "nb_tx": "Nb tx",
            "total_abs": "Total (€)", "moyenne_abs": "Panier moyen (€)",
        }),
        use_container_width=True,
        height=350,
    )


# ── Tab 4 — Anomalies ─────────────────────────────────────────────────────────

def _tab_anomalies(etl, start, end):
    with st.spinner("Isolation Forest …"):
        df = _run_anomaly(etl)

    df = _apply_date_filter(df, start, end)
    flagged = df[df["anomaly_flag"]].sort_values("anomaly_rank")

    st.metric("Transactions suspectes détectées",
              f"{len(flagged)} / {len(df)}",
              help="Seuil de contamination : 5%")

    st.markdown("---")

    # Scatter amount vs date, highlight anomalies
    fig = go.Figure()
    normal = df[~df["anomaly_flag"]]
    fig.add_scatter(
        x=normal["date"], y=normal["amount"],
        mode="markers", name="Normal",
        marker=dict(color=_NEUTRAL_COLOR, size=4, opacity=0.5),
    )
    fig.add_scatter(
        x=flagged["date"], y=flagged["amount"],
        mode="markers", name="Anomalie",
        marker=dict(color=_EXPENSE_COLOR, size=9, symbol="x",
                    line=dict(width=1.5, color="darkred")),
        text=flagged["description"],
        hovertemplate="%{text}<br>%{y:.2f} €<extra></extra>",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4)
    fig.update_layout(
        title="Transactions — anomalies mises en évidence",
        xaxis_title="", yaxis_title="Montant (€)",
        height=380, legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly table
    cols_show = [c for c in
                 ["anomaly_rank", "date", "description", "amount",
                  "tx_type", "anomaly_score"]
                 if c in flagged.columns]
    st.dataframe(
        flagged[cols_show]
        .rename(columns={
            "anomaly_rank": "Rang", "date": "Date",
            "description": "Description", "amount": "Montant (€)",
            "tx_type": "Type", "anomaly_score": "Score",
        }),
        use_container_width=True,
        height=400,
    )


# ── Tab 5 — Prévisions ────────────────────────────────────────────────────────

def _tab_forecast(etl, horizon):
    with st.spinner(f"Prophet — prévision sur {horizon} jours …"):
        res = _run_forecast(etl, horizon)

    st.info(f"Modèle : **{res.model_type}** · Horizon : **{horizon} jours**",
            icon="📈")

    # ── Forecast line chart ──────────────────────────────────────────────────
    hist    = res._history
    summary = res.summary

    fig = go.Figure()
    fig.add_scatter(
        x=hist["ds"], y=hist["y"],
        mode="lines", name="Historique",
        line=dict(color=_EXPENSE_COLOR, width=1.5),
    )
    fig.add_scatter(
        x=pd.to_datetime(summary["date"]),
        y=summary["upper"],
        mode="lines", line=dict(width=0),
        showlegend=False, name="upper",
    )
    fig.add_scatter(
        x=pd.to_datetime(summary["date"]),
        y=summary["lower"],
        fill="tonexty", mode="lines", line=dict(width=0),
        fillcolor="rgba(155,89,182,0.2)",
        name="Intervalle 80%",
    )
    fig.add_scatter(
        x=pd.to_datetime(summary["date"]),
        y=summary["forecast"],
        mode="lines", name="Prévision",
        line=dict(color=_FORECAST_COLOR, width=2.5, dash="dash"),
    )
    today_x = str(hist["ds"].max().date())
    fig.add_shape(
        type="line",
        x0=today_x, x1=today_x,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(dash="dot", color="gray", width=1),
    )
    fig.add_annotation(
        x=today_x, y=1, xref="x", yref="paper",
        text="Aujourd'hui", showarrow=False,
        yanchor="bottom", font=dict(size=11, color="gray"),
    )
    fig.update_layout(
        title="Prévision des dépenses quotidiennes",
        xaxis_title="", yaxis_title="€/jour",
        height=400, legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Monthly forecast bar chart ───────────────────────────────────────────
    monthly = res.monthly
    colors  = [_FORECAST_COLOR if f else _EXPENSE_COLOR
               for f in monthly["is_forecast"]]

    fig2 = go.Figure(go.Bar(
        x=monthly["period"], y=monthly["total"],
        marker_color=colors,
        text=monthly["total"].map(lambda v: f"{v:,.0f} €"),
        textposition="outside",
    ))
    fig2.update_layout(
        title="Dépenses mensuelles — réel vs prévu",
        xaxis_title="", yaxis_title="€",
        height=350,
    )
    fig2.add_annotation(
        text="■ Réel  ■ Prévu",
        xref="paper", yref="paper", x=0.01, y=1.08,
        showarrow=False, font_size=11,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Forecast summary table ───────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prévision quotidienne")
        st.dataframe(
            summary.rename(columns={
                "date": "Date", "forecast": "Prévu (€)",
                "lower": "Min (€)", "upper": "Max (€)",
            }),
            use_container_width=True, height=350,
        )
    with col2:
        st.subheader("Par mois")
        st.dataframe(
            monthly.rename(columns={
                "period": "Période", "total": "Total (€)",
                "is_forecast": "Prévu",
            }),
            use_container_width=True, height=350,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    etl           = _get_etl()
    start, end, horizon = _sidebar(etl)

    st.title("💶 Smart Finance Engine")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Vue d'ensemble",
        "🏷️ Catégories",
        "🏪 Marchands",
        "🚨 Anomalies",
        "🔮 Prévisions",
    ])

    with tab1:
        _tab_overview(etl, start, end)
    with tab2:
        _tab_categories(etl, start, end)
    with tab3:
        _tab_merchants(etl, start, end)
    with tab4:
        _tab_anomalies(etl, start, end)
    with tab5:
        _tab_forecast(etl, horizon)


if __name__ == "__main__":
    main()
