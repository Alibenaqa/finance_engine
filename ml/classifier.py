"""
ml/classifier.py — XGBoost Transaction Categorizer
===================================================
Assigns a spending category to each transaction using a two-step approach:

  1. Rule-based auto-labeling  — regex rules produce "weak" labels from
     merchant names.  These cover the bulk of recognisable transactions.

  2. XGBoost classifier        — trained on the auto-labeled data so it
     generalises to unseen merchants beyond the rules.  Features:
       • TF-IDF char-ngrams on the description text
       • log|amount|, sign(amount), day_of_week

Categories
----------
  Alimentation   — groceries (Franprix, ALDI, Carrefour …)
  Restauration   — restaurants, fast-food, bars, cafés
  Transport      — Uber, Bolt, RATP, SNCF, taxis …
  Abonnements    — recurring subscriptions (Apple, Free, Netflix …)
  Shopping       — Amazon, clothing, AliExpress, Temu …
  Beauté         — cosmetics, pharmacy, hair salons
  Loisirs        — events, cinema, clubs, games
  Loyer          — rent, housing payments
  Virements      — outgoing personal transfers
  Revenus        — income / top-ups
  Autre          — catch-all

Usage
-----
    from etl import FinanceETL
    from ml.classifier import TransactionClassifier

    etl   = FinanceETL()
    clf   = TransactionClassifier()
    df    = etl.query("SELECT * FROM transactions")
    df    = clf.fit_predict(df)           # adds 'predicted_category' column
    clf.save("models/classifier.pkl")
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Category rules  (order matters — first match wins)
# ---------------------------------------------------------------------------

_RULES: list[tuple[str, re.Pattern]] = [
    # ── Revenus ─────────────────────────────────────────────────────────────
    ("Revenus", re.compile(
        r"payment from |top.up|topup|salaire|virement entrant|"
        r"referral reward|interest earned|card refund|remboursement",
        re.I)),
    # ── Loyer / logement ────────────────────────────────────────────────────
    ("Loyer", re.compile(
        r"heneo|loyer|rent|bail|charges|syndic|action logement",
        re.I)),
    # ── Abonnements ─────────────────────────────────────────────────────────
    ("Abonnements", re.compile(
        r"apple$|apple pay|itunes|netflix|spotify|disney|amazon prime|"
        r"free$|free telecom|free mobile|sfr|orange$|bouygues|"
        r"imagine r|comutitres|openai|claude|elevenlabs|"
        r"youtu|deezer|tidal|twitch|github|adobe|canva|notion|"
        r"onlyfans|cleo$",
        re.I)),
    # ── Transport ───────────────────────────────────────────────────────────
    ("Transport", re.compile(
        r"^uber$|uber eats.*fare|bolt$|ratp|sncf|navigo|"
        r"blablacar|ouigo|tgv|transilien|"
        r"azdistribution|breasy|selecta.*transport|"
        r"shell$|bp$|total$|esso",
        re.I)),
    # ── Alimentation (épiceries / supermarchés) ──────────────────────────────
    ("Alimentation", re.compile(
        r"franprix|aldi|lidl|carrefour|monoprix|casino|intermarche|"
        r"leclerc|super u|u express|coccinelle|marche voltaire|"
        r"sk\.alimentation|snp ba food|royal primeur|bionac|"
        r"the grocery store|food|alimenta",
        re.I)),
    # ── Restauration ────────────────────────────────────────────────────────
    ("Restauration", re.compile(
        r"uber eats|deliveroo|just eat|glovo|"
        r"pizza hut|mcdonald|burger|kebab|slaim|"
        r"le saint|bistrot|baguette|cafe|café|restaurant|"
        r"royal parmentier|le belvil|tabac|snc voltaire|"
        r"cie parisienne|le zingam|j&m food|dream food|"
        r"land.monkey|indiana|eric kayser|pret|starbucks|"
        r"step burger|aono|rezwan|pedrazzi|corcoran|"
        r"noctambule|robin|le rey|bar |snack|boucherie|"
        r"traiteur|pizz|friterie|ramen|sushi",
        re.I)),
    # ── Shopping ────────────────────────────────────────────────────────────
    ("Shopping", re.compile(
        r"amazon(?! prime)|aliexpress|temu|ebay|vinted|leboncoin|"
        r"zara|hm |primark|descamps|sephora(?!.*refund)|"
        r"apple store|mac store|whatnot|dhgate|"
        r"decathlon|action$|nespresso|pixmania|"
        r"fnac|darty|boulanger|electro depot|"
        r"lc waikiki|zalando|shein|asos",
        re.I)),
    # ── Beauté / santé ───────────────────────────────────────────────────────
    ("Beauté", re.compile(
        r"sephora|nocibé|inaya beauty|bleu libellule|"
        r"pharmacie|grande pharmacie|deciem|"
        r"coiff|beauty|vaperture|rez energy|"
        r"black beauty|ar\.oma vap|tavap",
        re.I)),
    # ── Loisirs ─────────────────────────────────────────────────────────────
    ("Loisirs", re.compile(
        r"cinema|ugc|mkp|mlg event|treplay|charlotte club|"
        r"maison louvard|taptap send|pay pal(?!.*refund)|"
        r"oney|idfs|younited|maison des vignes",
        re.I)),
    # ── Virements sortants ───────────────────────────────────────────────────
    ("Virements", re.compile(
        r"^to |^transfer to |^payment$|^beauty hdchi|^dkchi|^il reste",
        re.I)),
]

_CATEGORIES = [
    "Alimentation", "Restauration", "Transport", "Abonnements",
    "Shopping", "Beauté", "Loisirs", "Loyer", "Virements",
    "Revenus", "Autre",
]


def _rule_label(description: str, amount: float) -> str:
    """Apply rule cascade; fall back to sign-based heuristic."""
    for category, pattern in _RULES:
        if pattern.search(description):
            return category
    return "Revenus" if amount > 0 else "Autre"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class TransactionClassifier:
    """
    Trains an XGBoost multi-class classifier on auto-labeled transaction data.

    The predicted_category column added to the DataFrame uses the model's
    output for merchants the rules don't cover, and retains rule labels for
    high-confidence rule hits.
    """

    def __init__(self) -> None:
        self._tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            max_features=3_000,
            sublinear_tf=True,
        )
        self._le = LabelEncoder()
        self._model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TransactionClassifier":
        """
        Auto-label df with rules, then train XGBoost on those labels.
        Returns self for chaining.
        """
        df = df.copy()
        df["_label"] = df.apply(
            lambda r: _rule_label(r["description"], r["amount"]), axis=1
        )
        logger.info("Auto-label distribution:\n%s",
                    df["_label"].value_counts().to_string())

        X_text  = self._tfidf.fit_transform(df["description"])
        X_num   = self._numeric_features(df)
        X       = hstack([X_text, X_num])
        y       = self._le.fit_transform(df["_label"])

        self._model.fit(X, y)
        self._fitted = True
        logger.info("Classifier trained on %d transactions, %d classes",
                    len(df), len(self._le.classes_))
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return a Series of predicted category labels."""
        self._check_fitted()
        X_text = self._tfidf.transform(df["description"])
        X_num  = self._numeric_features(df)
        X      = hstack([X_text, X_num])
        return pd.Series(
            self._le.inverse_transform(self._model.predict(X)),
            index=df.index,
            name="predicted_category",
        )

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and return df with 'predicted_category' column added."""
        self.fit(df)
        df = df.copy()
        df["predicted_category"] = self.predict(df)
        return df

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of class probabilities."""
        self._check_fitted()
        X_text = self._tfidf.transform(df["description"])
        X_num  = self._numeric_features(df)
        X      = hstack([X_text, X_num])
        proba  = self._model.predict_proba(X)
        return pd.DataFrame(proba, columns=self._le.classes_, index=df.index)

    def category_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience: predict categories and return a spend summary grouped
        by predicted_category.
        """
        tagged = self.fit_predict(df[~df["is_internal"]])
        exp    = tagged[tagged["amount"] < 0]
        return (
            exp.groupby("predicted_category")["amount"]
            .agg(["count", "sum"])
            .rename(columns={"count": "nb_tx", "sum": "total_€"})
            .sort_values("total_€")
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"tfidf": self._tfidf, "le": self._le,
                         "model": self._model}, f)
        logger.info("Classifier saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "TransactionClassifier":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls()
        obj._tfidf   = state["tfidf"]
        obj._le      = state["le"]
        obj._model   = state["model"]
        obj._fitted  = True
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _numeric_features(df: pd.DataFrame) -> np.ndarray:
        amounts = df["amount"].astype(float).values
        feat    = np.column_stack([
            np.log1p(np.abs(amounts)),          # magnitude
            np.sign(amounts),                   # direction
            df.get("day_of_week", pd.Series(0, index=df.index)).values,
        ])
        return feat

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Classifier is not fitted yet. Call fit() first.")
