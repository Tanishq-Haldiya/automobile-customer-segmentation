"""Feature preparation for clustering.

Encoding choices:
  * Spending_Score is genuinely ordinal -> integer-coded Low(0) < Average(1)
    < High(2) rather than one-hot, so "one step apart" stays one step apart in
    the distance function instead of becoming a constant.
  * Nominal categoricals are imputed with an explicit "Missing" LEVEL rather
    than the mode. 18.6% of customers are missing at least one field, and the
    missing fields co-occur (a customer missing Family_Size is far more likely
    to also be missing Work_Experience), which points at one upstream cause
    rather than random dropout. Overwriting that with a mode would invent
    customers who never existed and would pull prototypes toward the mode.
  * Numerics are median-imputed and z-scored. Scaling matters here in a way it
    does not for tree models: every distance in this project is computed on
    these columns, so raw units would let Age (range 71) dominate Family_Size
    (range 8) purely by scale.

Nothing in this module reads a label, and there is no train/test split to leak
across -- clustering is fitted once on the whole existing-market population.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config as C


def build_preprocessor() -> ColumnTransformer:
    """Dense one-hot + scaled-numeric encoder, for K-Means and PCA baselines."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(
            categories=[C.ORDINAL_LEVELS[c] for c in C.ORDINAL],
            handle_unknown="use_encoded_value", unknown_value=-1)),
        ("scale", StandardScaler()),
    ])
    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=C.MISSING_TOKEN)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 min_frequency=10)),
    ])
    return ColumnTransformer(
        [("num", numeric_pipe, list(C.NUMERIC)),
         ("ord", ordinal_pipe, list(C.ORDINAL)),
         ("nom", nominal_pipe, list(C.NOMINAL))],
        remainder="drop")


def build_onehot_matrix(X: pd.DataFrame, pre=None):
    """Dense matrix for K-Means / PCA. Pass a fitted `pre` to reuse its encoding."""
    if pre is None:
        pre = build_preprocessor()
        M = pre.fit_transform(X)
    else:
        M = pre.transform(X)
    return np.asarray(M, dtype=float), pre


# --------------------------------------------------------------------------- #
# mixed-type frame -- the primary representation
# --------------------------------------------------------------------------- #
class MixedFrameEncoder:
    """Impute + scale a mixed-type frame, reusably.

    Fitted on the existing-market customers; the same medians, means and scales
    are then applied to the new-market prospects so both populations live in one
    coordinate system. Re-fitting on the prospects would silently move the
    origin and make every assignment wrong.
    """

    def __init__(self):
        self.medians_ = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: pd.DataFrame):
        X = X[C.FEATURES]
        self.medians_ = {c: float(X[c].median()) for c in C.NUMERIC}
        filled = X[C.NUMERIC].fillna(self.medians_)
        self.mean_ = filled.mean()
        self.scale_ = filled.std(ddof=0).replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame):
        """Return (scaled_frame, raw_imputed_frame)."""
        if self.medians_ is None:
            raise RuntimeError("MixedFrameEncoder.fit must be called first")
        X = X[C.FEATURES].copy()
        for col in C.NUMERIC:
            X[col] = X[col].fillna(self.medians_[col])
        for col in C.CATEGORICAL:
            X[col] = X[col].astype(object).where(X[col].notna(),
                                                 C.MISSING_TOKEN).astype(str)
        scaled = X.copy()
        scaled[C.NUMERIC] = (X[C.NUMERIC] - self.mean_) / self.scale_
        return scaled, X

    def fit_transform(self, X: pd.DataFrame):
        return self.fit(X).transform(X)


def column_indices():
    """Positional indices of numeric and categorical columns in a FEATURES frame."""
    num = [C.FEATURES.index(c) for c in C.NUMERIC]
    cat = [C.FEATURES.index(c) for c in C.CATEGORICAL]
    return num, cat


def build_cluster_frame(X: pd.DataFrame, encoder: MixedFrameEncoder = None):
    """Convenience wrapper: (scaled, raw_imputed, num_idx, cat_idx, encoder)."""
    if encoder is None:
        encoder = MixedFrameEncoder().fit(X)
    scaled, raw = encoder.transform(X)
    num_idx, cat_idx = column_indices()
    return scaled, raw, num_idx, cat_idx, encoder
