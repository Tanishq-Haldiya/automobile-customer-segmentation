"""Unsupervised segmentation of the existing-market customers.

The data is mixed-type (3 numeric, 6 categorical), so K-Prototypes is the
primary algorithm: it minimises squared Euclidean distance on the numeric part
and a matching-dissimilarity count on the categorical part, in one joint
objective. K-Means on one-hot columns is the usual default and is subtly wrong
here -- one-hot makes the distance between any two different professions a
constant, so it cannot express that Doctor is nearer Healthcare than Artist, and
a 9-level column outweighs a binary one purely by column count.

With no label there is no external answer key, so cluster quality is established
three ways, none of which needs one:

  1. **Internal validity** -- a Gower silhouette across k.
  2. **Resampling stability** -- bootstrap each k, recluster, and measure how
     often the same customers stay together (Hennig 2007). This is the load-
     bearing check: a partition that dissolves under resampling is an artefact
     of the sample, whatever its silhouette says.
  3. **Cross-algorithm agreement** -- structurally different algorithms should
     recover the same partition if it is real. Adjusted Rand Index is used to
     compare two *clusterings*; no label is involved.
"""
from __future__ import annotations

import warnings

import gower
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from . import config as C
from . import features as F
from . import viz
from .viz import plt


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def fit_kprototypes(scaled: pd.DataFrame, cat_idx, k: int,
                    seed: int = C.RANDOM_STATE, n_init: int = 5):
    model = KPrototypes(n_clusters=k, init="Cao", n_init=n_init, max_iter=60,
                        random_state=seed, verbose=0, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = model.fit_predict(scaled.to_numpy(dtype=object),
                                   categorical=cat_idx)
    return model, np.asarray(labels)


def _gower_matrix(raw: pd.DataFrame, n: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(raw), size=min(n, len(raw)), replace=False)
    D = np.asarray(gower.gower_matrix(raw.iloc[idx].reset_index(drop=True)),
                   dtype=float)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D, idx


# --------------------------------------------------------------------------- #
# prototypes and assignment
# --------------------------------------------------------------------------- #
def extract_prototypes(scaled: pd.DataFrame, labels) -> dict:
    """Numeric mean + categorical mode per cluster, in the scaled coordinate system."""
    df = scaled.copy()
    df["_k"] = np.asarray(labels)
    protos = {}
    for k, g in df.groupby("_k"):
        num = g[C.NUMERIC].astype(float).mean().to_dict()
        cat = {c: g[c].mode().iat[0] for c in C.CATEGORICAL}
        protos[int(k)] = {"numeric": num, "categorical": cat, "n": int(len(g))}
    return protos


def prototype_distances(scaled: pd.DataFrame, protos: dict, gamma: float):
    """K-Prototypes dissimilarity from every row to every prototype.

    d(x, p) = sum_num (x_j - p_j)^2  +  gamma * count of mismatched categoricals
    -- the same objective K-Prototypes minimises, so argmin reproduces its own
    assignment. `test_assignment_matches_kprototypes` asserts exactly that.
    """
    ks = sorted(protos)
    num = scaled[C.NUMERIC].to_numpy(dtype=float)
    out = np.zeros((len(scaled), len(ks)), dtype=float)
    for col_i, k in enumerate(ks):
        p = protos[k]
        centre = np.array([p["numeric"][c] for c in C.NUMERIC], dtype=float)
        d = ((num - centre) ** 2).sum(axis=1)
        mism = np.zeros(len(scaled), dtype=float)
        for c in C.CATEGORICAL:
            mism += (scaled[c].to_numpy() != p["categorical"][c]).astype(float)
        out[:, col_i] = d + gamma * mism
    return out, ks


def assign(scaled: pd.DataFrame, protos: dict, gamma: float):
    """Assign rows to the nearest prototype; return labels and a confidence frame."""
    D, ks = prototype_distances(scaled, protos, gamma)
    order = np.argsort(D, axis=1)
    best, second = order[:, 0], order[:, 1]
    d_best = D[np.arange(len(D)), best]
    d_second = D[np.arange(len(D)), second]
    labels = np.array(ks)[best]
    # Relative margin: how much closer the winner is than the runner-up.
    # 0 = a coin flip between two clusters, 1 = sits exactly on its prototype.
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.where(d_second > 0, (d_second - d_best) / d_second, 0.0)
    return labels, pd.DataFrame({
        "Cluster": labels,
        "Distance": d_best.round(4),
        "Runner_Up": np.array(ks)[second],
        "Separation": np.clip(margin, 0, 1).round(4),
    })


def gamma_of(model) -> float:
    g = getattr(model, "gamma", None)
    return float(g) if g else 0.5


# --------------------------------------------------------------------------- #
# model selection
# --------------------------------------------------------------------------- #
def select_k(scaled, raw, cat_idx, seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    """Cost elbow + Gower silhouette across C.K_RANGE."""
    Dg, sub = _gower_matrix(raw, C.GOWER_SAMPLE, seed)
    rows = []
    for k in C.K_RANGE:
        model, labels = fit_kprototypes(scaled, cat_idx, k, seed)
        sub_labels = labels[sub]
        sil = (silhouette_score(Dg, sub_labels, metric="precomputed")
               if len(np.unique(sub_labels)) > 1 else np.nan)
        sizes = pd.Series(labels).value_counts()
        rows.append({
            "k": k,
            "cost": round(float(model.cost_), 1),
            "gower_silhouette": round(float(sil), 4),
            "smallest_cluster_pct": round(float(sizes.min() / len(labels)) * 100, 1),
        })
    out = pd.DataFrame(rows)
    out["cost_drop_pct"] = (-out["cost"].pct_change() * 100).round(1)
    return out


# --------------------------------------------------------------------------- #
# bootstrap stability -- the primary validation
# --------------------------------------------------------------------------- #
def bootstrap_stability(scaled, cat_idx, k: int, B: int = C.BOOTSTRAP_B,
                        seed: int = C.RANDOM_STATE):
    """Hennig-style cluster-wise stability.

    For each resample: recluster, then match every original cluster to its most
    similar bootstrap cluster by Jaccard index over the original row ids that
    the resample happens to contain. A cluster that keeps scoring above ~0.75 is
    a real, reproducible group; below ~0.60 it is an artefact of this particular
    sample.
    """
    rng = np.random.default_rng(seed)
    n = len(scaled)
    _, base = fit_kprototypes(scaled, cat_idx, k, seed, n_init=5)
    per_cluster = {int(c): [] for c in np.unique(base)}

    for b in range(B):
        idx = rng.choice(n, size=n, replace=True)
        boot = scaled.iloc[idx].reset_index(drop=True)
        try:
            _, bl = fit_kprototypes(boot, cat_idx, k, seed + b + 1, n_init=1)
        except Exception:
            continue
        # first bootstrap label seen for each distinct original row
        first = {}
        for pos, orig in enumerate(idx):
            if orig not in first:
                first[orig] = bl[pos]
        present = np.fromiter(first.keys(), dtype=int)
        blab = np.fromiter((first[i] for i in present), dtype=int)

        for c in per_cluster:
            A = present[base[present] == c]
            if len(A) == 0:
                per_cluster[c].append(0.0)
                continue
            setA = set(A.tolist())
            best = 0.0
            for j in np.unique(blab):
                setB = set(present[blab == j].tolist())
                inter = len(setA & setB)
                if inter:
                    best = max(best, inter / len(setA | setB))
            per_cluster[c].append(best)

    rows = []
    for c, vals in sorted(per_cluster.items()):
        v = np.array(vals, dtype=float)
        rows.append({
            "k": k, "cluster": c, "n": int((base == c).sum()),
            "mean_jaccard": round(float(v.mean()), 4),
            "sd_jaccard": round(float(v.std()), 4),
            "min_jaccard": round(float(v.min()), 4),
            "verdict": ("stable" if v.mean() >= C.STABLE
                        else "pattern" if v.mean() >= C.PATTERN else "unstable"),
        })
    return pd.DataFrame(rows), base


def stability_across_k(scaled, cat_idx, ks=None, B: int = C.BOOTSTRAP_B,
                       seed: int = C.RANDOM_STATE):
    frames = []
    for k in (ks or C.K_RANGE):
        df, _ = bootstrap_stability(scaled, cat_idx, k, B=B, seed=seed)
        frames.append(df)
    allk = pd.concat(frames, ignore_index=True)
    summary = (allk.groupby("k")
               .apply(lambda g: pd.Series({
                   "mean_jaccard": round(float(g["mean_jaccard"].mean()), 4),
                   "worst_cluster": round(float(g["mean_jaccard"].min()), 4),
                   "n_stable": int((g["mean_jaccard"] >= C.STABLE).sum()),
                   "n_unstable": int((g["mean_jaccard"] < C.PATTERN).sum()),
               }), include_groups=False)
               .reset_index())
    return allk, summary


# --------------------------------------------------------------------------- #
# cross-checks
# --------------------------------------------------------------------------- #
def kmeans_labels(X: pd.DataFrame, k: int, seed: int = C.RANDOM_STATE):
    M, pre = F.build_onehot_matrix(X)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(M), M


def hierarchical_labels(raw: pd.DataFrame, k: int, seed: int = C.RANDOM_STATE):
    Dg, idx = _gower_matrix(raw, C.GOWER_SAMPLE, seed)
    Z = linkage(squareform(Dg, checks=False), method="average")
    return fcluster(Z, t=k, criterion="maxclust") - 1, idx


def method_agreement(kp, km, hc, hc_idx) -> dict:
    """ARI between pairs of *clusterings*. No label is involved anywhere here."""
    return {
        "K-Prototypes vs K-Means": round(float(adjusted_rand_score(kp, km)), 4),
        "K-Prototypes vs Ward/Gower": round(
            float(adjusted_rand_score(kp[hc_idx], hc)), 4),
        "K-Means vs Ward/Gower": round(
            float(adjusted_rand_score(km[hc_idx], hc)), 4),
    }


# --------------------------------------------------------------------------- #
# profiling
# --------------------------------------------------------------------------- #
def profile_clusters(X: pd.DataFrame, labels) -> pd.DataFrame:
    df = X.copy()
    df["cluster"] = np.asarray(labels)
    rows = []
    for cl, g in df.groupby("cluster"):
        rows.append({
            "cluster": int(cl),
            "n": len(g),
            "share_pct": round(len(g) / len(df) * 100, 1),
            "median_age": g["Age"].median(),
            "pct_married": round(g["Ever_Married"].eq("Yes").mean() * 100, 1),
            "pct_graduated": round(g["Graduated"].eq("Yes").mean() * 100, 1),
            "pct_spend_low": round(g["Spending_Score"].eq("Low").mean() * 100, 1),
            "pct_spend_high": round(g["Spending_Score"].eq("High").mean() * 100, 1),
            "median_family_size": g["Family_Size"].median(),
            "median_work_exp": g["Work_Experience"].median(),
            "top_profession": (g["Profession"].mode().iat[0]
                               if g["Profession"].notna().any() else "-"),
            "pct_top_profession": round(
                g["Profession"].eq(g["Profession"].mode().iat[0]).mean() * 100, 1)
            if g["Profession"].notna().any() else 0.0,
        })
    return pd.DataFrame(rows).set_index("cluster")


def distinguishing_features(X: pd.DataFrame, labels, top: int = 3) -> pd.DataFrame:
    """For each cluster, the features that deviate most from the population.

    Numerics are reported as a standardised difference of means; categoricals as
    the level whose share inside the cluster most exceeds its overall share.
    """
    df = X.copy()
    df["cluster"] = np.asarray(labels)
    rows = []
    for cl, g in df.groupby("cluster"):
        scores = []
        for c in C.NUMERIC:
            sd = X[c].std(ddof=0)
            if sd and not np.isnan(sd):
                z = (g[c].mean() - X[c].mean()) / sd
                scores.append((abs(z), "%s %s (z=%+.2f)" % (
                    c.replace("_", " "), "high" if z > 0 else "low", z)))
        for c in C.CATEGORICAL:
            base = X[c].value_counts(normalize=True)
            here = g[c].value_counts(normalize=True)
            for lvl in here.index:
                lift = here[lvl] - base.get(lvl, 0.0)
                scores.append((abs(lift) * 2.5,
                               "%s=%s (%.0f%% vs %.0f%%)" % (
                                   c.replace("_", " "), lvl, here[lvl] * 100,
                                   base.get(lvl, 0.0) * 100)))
        scores.sort(reverse=True)
        rows.append({"cluster": int(cl), "n": len(g),
                     "defining_traits": "; ".join(s for _, s in scores[:top])})
    return pd.DataFrame(rows).set_index("cluster")


def famd_coords(raw: pd.DataFrame, seed: int = C.RANDOM_STATE):
    """2-D projection of the mixed-type data, for visual inspection only."""
    try:
        import prince
        famd = prince.FAMD(n_components=2, n_iter=5, random_state=seed).fit(raw)
        coords = np.asarray(famd.row_coordinates(raw))
        ev = famd.eigenvalues_summary
        explained = [float(str(v).strip("%")) for v in ev["% of variance"][:2]]
        return coords, explained, "FAMD"
    except Exception:
        from sklearn.decomposition import PCA
        M, _ = F.build_onehot_matrix(raw)
        p = PCA(n_components=2, random_state=seed)
        return p.fit_transform(M), list(p.explained_variance_ratio_ * 100), \
            "PCA (one-hot)"


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_selection(sel: pd.DataFrame, chosen_k: int) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
    ax = axes[0]
    ax.plot(sel["k"], sel["cost"], color=C.SEQUENTIAL_HUE, marker="o",
            markersize=8, markeredgecolor=C.SURFACE, markeredgewidth=2.0)
    ax.set_title("K-Prototypes cost")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Total cost")
    hit = sel.loc[sel["k"] == chosen_k].iloc[0]
    ax.annotate("k = %d" % chosen_k, (hit["k"], hit["cost"]), xytext=(12, 14),
                textcoords="offset points", fontsize=9.5, weight="bold",
                color=C.INK, arrowprops=dict(arrowstyle="-", color=C.INK_MUTED,
                                             linewidth=1.2))
    viz.despine(ax)

    ax = axes[1]
    bars = ax.bar(sel["k"], sel["gower_silhouette"], width=0.6,
                  color=[C.SEQUENTIAL_HUE if k == chosen_k else "#bcd7ec"
                         for k in sel["k"]],
                  edgecolor=C.SURFACE, linewidth=2.0)
    viz.bar_labels(ax, bars, fmt="{:.3f}")
    ax.set_title("Gower silhouette")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette")
    ax.set_ylim(0, max(0.2, sel["gower_silhouette"].max() * 1.3))
    viz.despine(ax)
    viz.note(fig, "Silhouette on a %d-row Gower distance subsample. Every value "
                  "is low, which says the customer base is a gradient rather "
                  "than a set of separated clumps -- so k is chosen on "
                  "stability and usefulness, not on a peak that does not exist."
             % C.GOWER_SAMPLE, y=-0.06)
    return viz.save(fig, "04_cluster_selection.png")


def fig_stability(summary: pd.DataFrame, per_cluster: pd.DataFrame,
                  chosen_k: int) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8),
                             gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    ax.plot(summary["k"], summary["mean_jaccard"], color=C.SEQUENTIAL_HUE,
            marker="o", markersize=8, markeredgecolor=C.SURFACE,
            markeredgewidth=2.0, label="mean over clusters")
    ax.plot(summary["k"], summary["worst_cluster"], color=C.CLUSTER_COLORS[1],
            marker="s", markersize=7, markeredgecolor=C.SURFACE,
            markeredgewidth=2.0, linestyle=(0, (4, 2)), label="worst cluster")
    for y, lab in [(C.STABLE, "stable"), (C.PATTERN, "pattern")]:
        ax.axhline(y, color=C.INK_MUTED, linewidth=0.9, linestyle=(0, (2, 3)))
        ax.annotate(lab, (summary["k"].max(), y), xytext=(4, 2),
                    textcoords="offset points", fontsize=8, color=C.INK_MUTED)
    ax.set_title("Bootstrap stability by k")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean Jaccard over %d resamples" % C.BOOTSTRAP_B)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    viz.despine(ax)

    ax = axes[1]
    d = per_cluster[per_cluster["k"] == chosen_k].sort_values("mean_jaccard")
    colors = [C.CLUSTER_COLORS[int(c) % len(C.CLUSTER_COLORS)] for c in d["cluster"]]
    bars = ax.barh(["Cluster %d\n(n=%s)" % (c, f"{n:,}")
                    for c, n in zip(d["cluster"], d["n"])],
                   d["mean_jaccard"], height=0.55, color=colors,
                   edgecolor=C.SURFACE, linewidth=2.0)
    ax.errorbar(d["mean_jaccard"], np.arange(len(d)), xerr=d["sd_jaccard"],
                fmt="none", ecolor=C.INK_MUTED, elinewidth=1.3, capsize=3)
    for b, v, verdict in zip(bars, d["mean_jaccard"], d["verdict"]):
        ax.annotate("%.2f  %s" % (v, verdict),
                    (v, b.get_y() + b.get_height() / 2), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8.8,
                    color=C.INK_SECONDARY)
    for x in (C.PATTERN, C.STABLE):
        ax.axvline(x, color=C.INK_MUTED, linewidth=0.9, linestyle=(0, (2, 3)))
    ax.set_title("Per-cluster stability at k = %d" % chosen_k)
    ax.set_xlabel("Mean Jaccard (1 = the same customers every time)")
    ax.set_xlim(0, 1.25)
    ax.grid(axis="y", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    viz.note(fig, "Each resample draws n customers with replacement, re-runs "
                  "K-Prototypes and matches every original cluster to its "
                  "closest bootstrap counterpart. Above %.2f the same customers "
                  "group together every time; below %.2f the cluster is an "
                  "artefact of this particular sample."
             % (C.STABLE, C.PATTERN), y=-0.06)
    return viz.save(fig, "05_cluster_stability.png")


def fig_sizes(prof: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    colors = [C.CLUSTER_COLORS[int(c) % len(C.CLUSTER_COLORS)] for c in prof.index]
    bars = ax.bar([C.cluster_name(c) for c in prof.index], prof["n"], width=0.6,
                  color=colors, edgecolor=C.SURFACE, linewidth=2.0)
    for b, n, s in zip(bars, prof["n"], prof["share_pct"]):
        ax.annotate("%s\n%.1f%%" % (f"{n:,}", s),
                    (b.get_x() + b.get_width() / 2, n), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=C.INK_SECONDARY)
    ax.set_title("How the existing market divides")
    ax.set_ylabel("Customers")
    ax.set_ylim(0, prof["n"].max() * 1.28)
    viz.despine(ax)
    viz.note(fig, "Cluster sizes on the %s de-duplicated existing-market "
                  "customers." % f"{int(prof['n'].sum()):,}")
    return viz.save(fig, "06_cluster_sizes.png")


def fig_personas(prof: pd.DataFrame) -> str:
    cols = ["median_age", "pct_married", "pct_graduated", "pct_spend_low",
            "pct_spend_high", "median_family_size", "median_work_exp"]
    M = prof[cols].astype(float)
    Z = (M - M.mean()) / M.std(ddof=0).replace(0, 1)
    Z.index = ["Cluster %d\n(n=%s)" % (i, f"{prof.loc[i, 'n']:,}") for i in prof.index]
    Z.columns = [c.replace("median_", "med ").replace("pct_", "% ").replace("_", " ")
                 for c in cols]
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(Z) + 2.4))
    viz.heatmap(ax, Z.round(2), cmap=viz.DIV_CMAP, fmt="{:+.1f}", vmin=-2, vmax=2,
                cbar_label="z-score across clusters", center_text_threshold=1.1)
    ax.set_title("What distinguishes each segment")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    viz.note(fig, "Each column is z-scored across clusters, so +2 means 'high "
                  "for this variable relative to the other segments', not high "
                  "in absolute terms. Raw values are in the persona table.",
             y=-0.06)
    return viz.save(fig, "07_cluster_personas.png")


def fig_projection(coords, labels, explained, method: str) -> str:
    labs = np.asarray(labels)
    uniq = sorted(np.unique(labs))
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for cl in uniq:
        m = labs == cl
        color = C.CLUSTER_COLORS[int(cl) % len(C.CLUSTER_COLORS)]
        ax.scatter(coords[m, 0], coords[m, 1], s=9, color=color, alpha=0.55,
                   linewidths=0, label=C.cluster_name(cl))
        ax.annotate(str(cl), (coords[m, 0].mean(), coords[m, 1].mean()),
                    fontsize=12, weight="bold", color=C.INK, ha="center",
                    va="center",
                    bbox=dict(boxstyle="circle,pad=0.28", facecolor=C.SURFACE,
                              edgecolor=color, linewidth=2.0))
    ax.set_title("%s projection of the discovered segments" % method)
    ax.set_xlabel("Component 1 (%.1f%%)" % explained[0])
    ax.set_ylabel("Component 2 (%.1f%%)" % explained[1])
    ax.legend(loc="best", ncols=2)
    viz.despine(ax)
    viz.note(fig, "Each segment is direct-labelled at its centroid as well as "
                  "colour-coded. Two components capture only a small share of "
                  "the variance, so visual overlap here is expected and is not "
                  "evidence either way.")
    return viz.save(fig, "08_cluster_projection.png")


def fig_method_agreement(pairs: dict) -> str:
    names, vals = list(pairs.keys()), list(pairs.values())
    order = np.argsort(vals)
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(names) + 2.2))
    bars = ax.barh([names[i] for i in order], [vals[i] for i in order],
                   height=0.5, color=C.SEQUENTIAL_HUE, edgecolor=C.SURFACE,
                   linewidth=2.0)
    for b, v in zip(bars, [vals[i] for i in order]):
        ax.annotate("%.3f" % v, (v, b.get_y() + b.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=C.INK_SECONDARY)
    ax.set_title("Do structurally different algorithms find the same segments?")
    ax.set_xlabel("Adjusted Rand Index between two clusterings "
                  "(0 = chance, 1 = identical)")
    ax.set_xlim(0, max(0.6, max(vals) * 1.35))
    ax.grid(axis="y", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    viz.note(fig, "This compares clusterings with each other -- no label is "
                  "involved. Low agreement means the partition depends on the "
                  "algorithm, which is corroborating evidence that the data has "
                  "no single natural structure.")
    return viz.save(fig, "09_method_agreement.png")
