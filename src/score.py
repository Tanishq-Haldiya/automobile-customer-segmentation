"""Assign new-market prospects to the discovered segments.

The prospects are *assigned*, not clustered. Re-clustering them on their own
would produce segments that are internally sensible but numbered and shaped
differently from the existing market's, so no campaign could be carried across.
Instead the fitted encoder and prototypes are applied unchanged, which puts both
populations in one coordinate system.

Every assignment carries a `Separation` score -- how much closer the winning
prototype is than the runner-up. A customer sitting midway between two segments
is a real and common case in a continuum, and saying so is more useful than
emitting a confident-looking segment id.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import cluster as CL
from . import config as C
from . import data as D

# Separation thresholds. Fixed rather than data-derived so the tiers mean the
# same thing across re-runs and across the two populations.
CLEAR, MODERATE = 0.45, 0.20
TIERS = ["Borderline - blend or hold back", "Moderate - standard outreach",
         "Clear - automate"]


def tier(separation) -> pd.Categorical:
    s = np.asarray(separation, dtype=float)
    out = np.where(s >= CLEAR, TIERS[2], np.where(s >= MODERATE, TIERS[1], TIERS[0]))
    return pd.Categorical(out, categories=TIERS, ordered=True)


def assign_prospects(encoder, protos: dict, gamma: float,
                     personas: dict = None) -> pd.DataFrame:
    """Score every prospect. Row order matches the raw file, so IDs line up."""
    raw = D.load_prospects_raw()
    X = D.load_prospects()
    if len(raw) != len(X):
        raise RuntimeError("prospect row alignment broken")

    scaled, _ = encoder.transform(X)
    labels, conf = CL.assign(scaled, protos, gamma)
    dists, ks = CL.prototype_distances(scaled, protos, gamma)

    out = pd.DataFrame({C.ID_COL: raw[C.ID_COL].to_numpy()})
    for col in C.FEATURES:
        out[col] = X[col].to_numpy()
    out["Segment"] = [C.cluster_name(c) for c in labels]
    out["Segment_Id"] = labels
    if personas:
        out["Persona"] = [personas.get(int(c), "") for c in labels]
    out["Distance"] = conf["Distance"].to_numpy()
    out["Runner_Up"] = [C.cluster_name(c) for c in conf["Runner_Up"]]
    out["Separation"] = conf["Separation"].to_numpy()
    out["Assignment_Tier"] = tier(conf["Separation"])
    for i, k in enumerate(ks):
        out["Dist_to_%d" % k] = dists[:, i].round(4)
    return out


def assignment_summary(scored: pd.DataFrame) -> pd.DataFrame:
    work = scored.assign(_clear=scored["Separation"].ge(CLEAR))
    g = work.groupby("Segment", observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "share_pct": (g.size() / len(work) * 100).round(1),
        "mean_separation": g["Separation"].mean().round(3),
        "pct_clear": (g["_clear"].mean() * 100).round(1),
        "median_age": g["Age"].median(),
    })


def compare_to_reference(scored: pd.DataFrame, reference_profile: pd.DataFrame
                         ) -> pd.DataFrame:
    """Does the prospect pool split the same way the existing market does?

    A large gap would mean the new market is differently composed even though it
    is described by the same segments -- useful for sizing campaigns.
    """
    ref = (reference_profile["share_pct"] / reference_profile["share_pct"].sum()
           * 100).round(1)
    got = (scored.groupby("Segment_Id").size() / len(scored) * 100).round(1)
    rows = []
    for k in sorted(ref.index):
        rows.append({
            "segment": C.cluster_name(k),
            "existing_market_pct": float(ref.loc[k]),
            "prospect_pool_pct": float(got.get(k, 0.0)),
            "difference_pp": round(float(got.get(k, 0.0) - ref.loc[k]), 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def fig_assignment(scored: pd.DataFrame, comparison: pd.DataFrame) -> str:
    from . import viz
    from .viz import plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(wspace=0.28)

    ax = axes[0]
    ax.hist(scored["Separation"], bins=np.linspace(0, 1, 26),
            color=C.SEQUENTIAL_HUE, edgecolor=C.SURFACE, linewidth=1.2)
    for x, lab in [(MODERATE, "moderate"), (CLEAR, "clear")]:
        ax.axvline(x, color=C.INK, linestyle=(0, (4, 3)), linewidth=1.3)
        ax.annotate(lab, (x, ax.get_ylim()[1] * 0.93), xytext=(4, 0),
                    textcoords="offset points", fontsize=8.6, color=C.INK)
    ax.set_title("How cleanly each prospect lands in one segment")
    ax.set_xlabel("Separation (0 = midway between two segments, 1 = on the prototype)")
    ax.set_ylabel("Prospects")
    viz.despine(ax)

    ax = axes[1]
    y = np.arange(len(comparison))
    h = 0.36
    ax.barh(y + h / 2, comparison["existing_market_pct"], height=h,
            color=C.SEQUENTIAL_HUE, edgecolor=C.SURFACE, linewidth=1.6,
            label="Existing market")
    ax.barh(y - h / 2, comparison["prospect_pool_pct"], height=h,
            color=C.CLUSTER_COLORS[1], edgecolor=C.SURFACE, linewidth=1.6,
            label="Prospect pool")
    for i, r in comparison.iterrows():
        ax.annotate("%.0f%%" % r["existing_market_pct"],
                    (r["existing_market_pct"], i + h / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.4,
                    color=C.INK_SECONDARY)
        ax.annotate("%.0f%%" % r["prospect_pool_pct"],
                    (r["prospect_pool_pct"], i - h / 2), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.4,
                    color=C.INK_SECONDARY)
    ax.set_yticks(y, comparison["segment"])
    ax.set_title("Segment mix: new market vs existing", pad=20)
    ax.set_xlabel("% of population")
    ax.set_xlim(0, max(comparison[["existing_market_pct",
                                   "prospect_pool_pct"]].to_numpy().max() * 1.28, 10))
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncols=2,
                    handlelength=1.1, columnspacing=1.2)
    for t in leg.get_texts():
        t.set_color(C.INK_SECONDARY)
    ax.grid(axis="y", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    clear = float(scored["Separation"].ge(CLEAR).mean()) * 100
    border = float(scored["Separation"].lt(MODERATE).mean()) * 100
    viz.note(fig, "%.0f%% of prospects land clearly in one segment and %.0f%% "
                  "sit close to the boundary between two. In a continuum that "
                  "is the expected shape, and it is why the deliverable ships a "
                  "tier rather than a bare segment id." % (clear, border),
             y=-0.05)
    return viz.save(fig, "11_assignment.png")
