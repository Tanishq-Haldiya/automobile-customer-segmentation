"""Loading, cleaning and integrity auditing.

Unsupervised throughout. `load_customers()` and `load_prospects()` both return
the 9 feature columns only -- the loader drops `ID` and the raw file's legacy
sales-team column and never exposes either to the rest of the pipeline.

Design rules enforced here:
  * `ID` is dropped -- it is not a stable customer key across the two files
    (see data/raw/DATA_SOURCE.md, caveat 2). The prospects' raw IDs are
    re-attached to the output CSV after assignment, never used as an input.
  * The clustering population is de-duplicated on the 9 features. Identical
    customers would otherwise pull a prototype toward whatever the data happens
    to repeat, which is a property of the export and not of the market.
  * The prospect file is NOT de-duplicated: every prospect needs a row in the
    deliverable, even if another prospect looks identical.
  * No imputation happens here. Imputation belongs to the feature pipeline.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _read(path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_customers_raw() -> pd.DataFrame:
    return _read(C.CUSTOMERS_CSV)


def load_prospects_raw() -> pd.DataFrame:
    return _read(C.PROSPECTS_CSV)


def _strip_objects(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def clean(df: pd.DataFrame, deduplicate: bool = True) -> pd.DataFrame:
    """Return the 9 feature columns only. Nothing else survives this function."""
    df = _strip_objects(df.copy())
    missing = [c for c in C.FEATURES if c not in df.columns]
    if missing:
        raise ValueError("input is missing feature columns: %s" % missing)
    df = df[C.FEATURES]
    if deduplicate:
        df = df.drop_duplicates(subset=C.FEATURES, keep="first")
    return df.reset_index(drop=True)


def load_customers() -> pd.DataFrame:
    """The existing-market population the segments are learned from."""
    return clean(load_customers_raw(), deduplicate=True)


def load_prospects() -> pd.DataFrame:
    """New-market prospects, one row each, in original file order."""
    return clean(load_prospects_raw(), deduplicate=False)


# --------------------------------------------------------------------------- #
# validation and auditing
# --------------------------------------------------------------------------- #
def validate_schema(df: pd.DataFrame) -> list:
    """Return a list of human-readable schema problems (empty list == clean)."""
    problems = []
    missing = [c for c in C.FEATURES if c not in df.columns]
    if missing:
        problems.append("missing columns: %s" % missing)

    allowed = {
        "Gender": {"Male", "Female"},
        "Ever_Married": {"Yes", "No"},
        "Graduated": {"Yes", "No"},
        "Spending_Score": set(C.ORDINAL_LEVELS["Spending_Score"]),
        "Var_1": set("Cat_%d" % i for i in range(1, 8)),
    }
    for col, ok in allowed.items():
        if col in df.columns:
            bad = set(df[col].dropna().unique()) - ok
            if bad:
                problems.append("%s: unexpected values %s" % (col, sorted(bad)))

    for col in C.NUMERIC:
        if col in df.columns and (df[col].dropna() < 0).any():
            problems.append("%s: negative values present" % col)

    leaked = [c for c in (C.LEGACY_LABEL, C.ID_COL) if c in df.columns]
    if leaked:
        problems.append("non-feature columns leaked into the frame: %s" % leaked)
    return problems


def audit(df: pd.DataFrame) -> dict:
    """Summary statistics used by the EDA notebook and the final report."""
    return {
        "n_rows": int(len(df)),
        "n_features": int(df.shape[1]),
        "schema_problems": validate_schema(df),
        "missing_counts": {k: int(v) for k, v in df.isna().sum().items()},
        "missing_pct": {k: round(float(v) * 100, 2)
                        for k, v in df.isna().mean().items()},
        "n_complete_rows": int(df.notna().all(axis=1).sum()),
        "pct_complete_rows": round(float(df.notna().all(axis=1).mean()) * 100, 2),
        "cardinality": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
    }


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Which fields are missing, and do they go missing together?

    Without a label there is no way to test whether missingness is predictive,
    so the useful question becomes co-occurrence: fields that go missing on the
    same customers point at one upstream cause (a skipped form section) rather
    than nine independent ones.
    """
    rows = []
    flags = df[C.FEATURES].isna()
    any_missing = flags.any(axis=1)
    for col in C.FEATURES:
        n = int(flags[col].sum())
        if n == 0:
            continue
        others = flags.loc[flags[col], [c for c in C.FEATURES if c != col]]
        rows.append({
            "column": col,
            "n_missing": n,
            "pct_missing": round(float(flags[col].mean()) * 100, 2),
            "pct_also_missing_elsewhere": round(
                float(others.any(axis=1).mean()) * 100, 2),
            "most_co_missing_with": (others.sum().idxmax()
                                     if others.sum().max() > 0 else "-"),
        })
    out = pd.DataFrame(rows).sort_values("pct_missing", ascending=False)
    out.attrs["pct_rows_with_any_missing"] = round(float(any_missing.mean()) * 100, 2)
    return out.reset_index(drop=True)


def duplicate_report(raw: pd.DataFrame) -> dict:
    """How much of the raw file is repeated rows."""
    feat = raw[C.FEATURES]
    return {
        "raw_rows": int(len(raw)),
        "distinct_feature_rows": int(len(feat.drop_duplicates())),
        "duplicate_rows_removed": int(len(feat) - len(feat.drop_duplicates())),
        "largest_repeat_group": int(feat.groupby(C.FEATURES, dropna=False)
                                    .size().max()),
    }


def overlap_with_prospects() -> dict:
    """Do any prospects look exactly like an existing customer?"""
    cust = load_customers()
    pros = load_prospects()
    key = lambda d: d[C.FEATURES].astype(str).agg("|".join, axis=1)
    kc, kp = set(key(cust)), key(pros)
    return {
        "n_prospects": int(len(pros)),
        "n_prospects_matching_an_existing_customer": int(kp.isin(kc).sum()),
        "pct": round(float(kp.isin(kc).mean()) * 100, 2),
    }
