"""Exploratory analysis of an unlabelled customer base.

With no target column the questions change. There is nothing to compute an
association *with*, so the useful questions become:

  * What is the shape of each feature, and does anything look implausible?
  * Which features carry redundant information? Two near-duplicate features get
    double weight in a distance function, so collinearity is a modelling problem
    here, not a footnote.
  * Which features are so concentrated on one value that they cannot separate
    anyone?
  * Where is the data missing, and do the gaps co-occur?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from . import config as C
from . import data as D
from . import viz
from .viz import plt


def _as_str_cat(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), C.MISSING_TOKEN).astype(str)


def _binned(df: pd.DataFrame) -> dict:
    """Every feature as a string categorical, numerics quartile-binned."""
    out = {}
    for col in C.FEATURES:
        s = df[col]
        if col in C.NUMERIC:
            s = pd.qcut(s, 4, duplicates="drop")
        out[col] = _as_str_cat(s)
    return out


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V (Bergsma 2013) for two categorical series."""
    tab = pd.crosstab(x, y)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(tab)[0]
    n = tab.to_numpy().sum()
    phi2 = chi2 / n
    r, k = tab.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else np.nan


def association_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Feature-to-feature association. No target row -- there is no target."""
    prepped = _binned(df)
    M = pd.DataFrame(index=C.FEATURES, columns=C.FEATURES, dtype=float)
    for a in C.FEATURES:
        for b in C.FEATURES:
            M.loc[a, b] = 1.0 if a == b else cramers_v(prepped[a], prepped[b])
    return M.astype(float).round(3)


def redundancy_report(df: pd.DataFrame) -> pd.DataFrame:
    """Feature pairs that largely repeat each other, strongest first."""
    M = association_matrix(df)
    rows = []
    for i, a in enumerate(C.FEATURES):
        for b in C.FEATURES[i + 1:]:
            rows.append({"feature_a": a, "feature_b": b,
                         "cramers_v": round(float(M.loc[a, b]), 3)})
    return (pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
            .reset_index(drop=True))


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: spread, and how much separating power it can offer.

    `concentration` is the share held by the single most common value. A feature
    at 0.9 splits the population 90/10 at best and will barely move a distance
    calculation, whatever its other properties.
    """
    rows = []
    for col in C.FEATURES:
        s = df[col]
        kind = "numeric" if col in C.NUMERIC else (
            "ordinal" if col in C.ORDINAL else "nominal")
        top = _as_str_cat(s).value_counts(normalize=True)
        row = {
            "feature": col, "type": kind,
            "missing_pct": round(float(s.isna().mean()) * 100, 2),
            "n_levels": int(s.nunique(dropna=True)),
            "most_common": str(top.index[0]),
            "concentration": round(float(top.iloc[0]), 3),
        }
        if col in C.NUMERIC:
            row.update({"median": float(s.median()),
                        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
                        "min": float(s.min()), "max": float(s.max())})
        rows.append(row)
    return pd.DataFrame(rows)


def population_shift(customers: pd.DataFrame, prospects: pd.DataFrame) -> pd.DataFrame:
    """Do the new-market prospects look like the existing market?

    With no labels anywhere, this is the only check available on whether segments
    learned in one market can be applied to the other. Numerics are compared with
    a standardised mean difference, categoricals with total variation distance.
    """
    rows = []
    for col in C.FEATURES:
        if col in C.NUMERIC:
            a, b = customers[col].dropna(), prospects[col].dropna()
            sd = np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2) or 1.0
            stat = abs(a.mean() - b.mean()) / sd
            metric = "std. mean diff"
        else:
            a = _as_str_cat(customers[col]).value_counts(normalize=True)
            b = _as_str_cat(prospects[col]).value_counts(normalize=True)
            lv = sorted(set(a.index) | set(b.index))
            stat = 0.5 * sum(abs(a.get(l, 0.0) - b.get(l, 0.0)) for l in lv)
            metric = "total variation"
        rows.append({"feature": col, "metric": metric,
                     "difference": round(float(stat), 4),
                     "verdict": "material" if stat >= 0.10 else
                                "minor" if stat >= 0.05 else "negligible"})
    return pd.DataFrame(rows).sort_values("difference", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_feature_overview(df: pd.DataFrame) -> str:
    fig = plt.figure(figsize=(10.6, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.28)

    for i, col in enumerate(C.NUMERIC):
        ax = fig.add_subplot(gs[0, i])
        vals = df[col].dropna()
        ax.hist(vals, bins=min(30, max(8, int(vals.nunique()))),
                color=C.SEQUENTIAL_HUE, edgecolor=C.SURFACE, linewidth=1.1)
        med = float(vals.median())
        ax.axvline(med, color=C.INK, linestyle=(0, (4, 3)), linewidth=1.4)
        ax.annotate("median %g" % med, (med, ax.get_ylim()[1] * 0.92),
                    xytext=(5, 0), textcoords="offset points", fontsize=8.5,
                    color=C.INK)
        ax.set_title(col.replace("_", " "))
        ax.set_ylabel("Customers" if i == 0 else "")
        viz.despine(ax)

    cats = ["Spending_Score", "Profession", "Var_1"]
    for i, col in enumerate(cats):
        ax = fig.add_subplot(gs[1, i])
        s = _as_str_cat(df[col]).value_counts(normalize=True)
        if col == "Spending_Score":
            s = s.reindex([l for l in C.ORDINAL_LEVELS[col] if l in s.index]
                          + [l for l in s.index
                             if l not in C.ORDINAL_LEVELS[col]]).dropna()
        s = s.sort_values()
        bars = ax.barh(list(s.index), s.to_numpy() * 100, height=0.6,
                       color=C.SEQUENTIAL_HUE, edgecolor=C.SURFACE, linewidth=1.6)
        for b, v in zip(bars, s.to_numpy() * 100):
            ax.annotate("%.0f%%" % v, (v, b.get_y() + b.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=8.3, color=C.INK_SECONDARY)
        ax.set_title(col.replace("_", " "))
        ax.set_xlim(0, s.max() * 100 * 1.28)
        ax.set_xlabel("% of customers")
        ax.grid(axis="y", visible=False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    fig.suptitle("The raw material: what each feature looks like",
                 x=0.005, ha="left", fontsize=12, weight="bold", color=C.INK)
    viz.note(fig, "n = %s de-duplicated existing-market customers. "
                  "'Missing' is shown as its own level because the pipeline "
                  "treats it as one." % f"{len(df):,}", y=-0.02)
    return viz.save(fig, "01_feature_overview.png")


def fig_missingness(df: pd.DataFrame) -> str:
    rep = D.missingness_report(df)
    flags = df[C.FEATURES].isna()
    cols = [c for c in C.FEATURES if flags[c].any()]
    J = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            if a == b:
                J.loc[a, b] = 1.0
            else:
                union = (flags[a] | flags[b]).sum()
                J.loc[a, b] = (flags[a] & flags[b]).sum() / union if union else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(wspace=0.30)
    ax = axes[0]
    d = rep.sort_values("pct_missing")
    bars = ax.barh(d["column"], d["pct_missing"], height=0.6,
                   color=C.SEQUENTIAL_HUE, edgecolor=C.SURFACE, linewidth=2.0)
    for b, v in zip(bars, d["pct_missing"]):
        ax.annotate("%.1f%%" % v, (v, b.get_y() + b.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=9, color=C.INK_SECONDARY)
    ax.set_title("How much is missing")
    ax.set_xlabel("% of customers")
    ax.set_xlim(0, d["pct_missing"].max() * 1.3)
    ax.grid(axis="y", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    viz.heatmap(axes[1], (J.astype(float) * 100).round(0), fmt="{:.0f}",
                cbar_label="% overlap (Jaccard)", vmin=0, vmax=100)
    axes[1].set_title("Do the gaps happen to the same customers?")
    axes[1].tick_params(axis="x", rotation=30)
    for lbl in axes[1].get_xticklabels():
        lbl.set_ha("right")
    viz.note(fig, "%.1f%% of customers are missing at least one field. The "
                  "right panel is the Jaccard overlap between two fields being "
                  "absent: Work_Experience and Family_Size go missing together "
                  "far more than chance, which points at one upstream cause "
                  "rather than nine independent ones. That is why missingness "
                  "is encoded as its own level instead of imputed away."
             % rep.attrs["pct_rows_with_any_missing"], y=-0.05)
    return viz.save(fig, "02_missingness.png")


def fig_association(df: pd.DataFrame) -> str:
    M = association_matrix(df)
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    viz.heatmap(ax, M, fmt="{:.2f}", cbar_label="Cramer's V", vmin=0, vmax=1)
    ax.set_title("Feature redundancy (bias-corrected Cramer's V)")
    ax.tick_params(axis="x", rotation=40)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    viz.note(fig, "Numerics are quartile-binned so one statistic covers every "
                  "column. This matters more without a label than with one: two "
                  "highly associated features effectively vote twice in a "
                  "distance calculation, quietly reweighting the segmentation.")
    return viz.save(fig, "03_association_matrix.png")


def fig_population_shift(shift: pd.DataFrame) -> str:
    d = shift.sort_values("difference")
    fig, ax = plt.subplots(figsize=(7.8, 0.44 * len(d) + 2.2))
    colors = [C.CLUSTER_COLORS[1] if v >= 0.10 else C.SEQUENTIAL_HUE
              for v in d["difference"]]
    bars = ax.barh([f.replace("_", " ") for f in d["feature"]], d["difference"],
                   height=0.55, color=colors, edgecolor=C.SURFACE, linewidth=2.0)
    for b, v, m in zip(bars, d["difference"], d["metric"]):
        ax.annotate("%.3f  (%s)" % (v, m), (v, b.get_y() + b.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.6, color=C.INK_SECONDARY)
    ax.axvline(0.10, color=C.INK_MUTED, linewidth=1.0, linestyle=(0, (3, 3)))
    ax.annotate("material", (0.10, len(d) - 0.4), xytext=(4, 0),
                textcoords="offset points", fontsize=8.4, color=C.INK_MUTED)
    ax.set_title("Do new-market prospects resemble the existing market?")
    ax.set_xlabel("Difference between the two populations")
    ax.set_xlim(0, max(0.2, d["difference"].max() * 1.5))
    ax.grid(axis="y", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    viz.note(fig, "Numerics use a standardised mean difference, categoricals "
                  "total variation distance; both are 0 when the populations "
                  "match. With no labels anywhere this is the only available "
                  "check on whether segments learned in one market transfer to "
                  "the other.")
    return viz.save(fig, "10_population_shift.png")


def run(customers: pd.DataFrame, prospects: pd.DataFrame) -> dict:
    viz.set_style()
    shift = population_shift(customers, prospects)
    return {
        "figures": {
            "feature_overview": fig_feature_overview(customers),
            "missingness": fig_missingness(customers),
            "association": fig_association(customers),
            "population_shift": fig_population_shift(shift),
        },
        "feature_summary": feature_summary(customers),
        "redundancy": redundancy_report(customers),
        "population_shift": shift,
        "association_matrix": association_matrix(customers),
    }
