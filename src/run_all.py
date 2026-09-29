"""End-to-end unsupervised pipeline. Run with:  python -m src.run_all [--force]

No label is read at any stage. Cluster count is chosen from internal validity,
bootstrap stability and interpretability; cluster quality is established by
resampling and by cross-algorithm agreement.

Writes figures to docs/figures/, tables to docs/tables/, the assigned prospect
list to outputs/, and a machine-readable docs/results.json the report is built
from. Expensive steps are cached in data/processed/; --force recomputes.
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from . import cluster as CL, config as C, data as D, eda, features as F
from . import score, viz

CACHE = C.PROCESSED
TABLES = C.TABLES


def _cached(name: str, fn, force: bool):
    path = CACHE / name
    if path.exists() and not force:
        print("  [cache] %s" % name)
        return joblib.load(path)
    t = time.time()
    val = fn()
    joblib.dump(val, path)
    print("  [computed in %.1fs] %s" % (time.time() - t, name))
    return val


def _table(df: pd.DataFrame, name: str, index=True) -> str:
    p = TABLES / (name + ".csv")
    df.to_csv(p, index=index)
    return str(p.relative_to(C.ROOT)).replace("\\", "/")


K_TOLERANCE = 0.05


def choose_k(sel: pd.DataFrame, stab: pd.DataFrame) -> tuple:
    """Pick k by worst-cluster reproducibility, breaking ties toward more segments.

    Stability leads because it answers the only question available without a
    label: would these same customers group together again on a different sample
    of the market? The *worst* cluster is the criterion rather than the mean --
    a solution is only as good as its flimsiest segment, since that is the one a
    campaign would be built on and the one that would evaporate.

    More segments are more actionable, so among k values within `K_TOLERANCE` of
    the best worst-cluster score the largest k wins. The tolerance is deliberately
    tight: trading a large drop in reproducibility for extra granularity buys
    segments that look specific and are not real.
    """
    m = sel.merge(stab, on="k")
    best_worst = float(m["worst_cluster"].max())
    near = m[m["worst_cluster"] >= best_worst - K_TOLERANCE]
    row = near.sort_values("k", ascending=False).iloc[0]
    k = int(row["k"])

    reason = ["k=%d has the most reproducible weakest segment "
              "(Jaccard %.2f, mean %.2f)" % (k, row["worst_cluster"],
                                             row["mean_jaccard"])]
    bigger = m[m["k"] > k]
    if len(bigger):
        alt = bigger.sort_values("worst_cluster", ascending=False).iloc[0]
        reason.append("the best larger alternative, k=%d, drops its weakest "
                      "segment to %.2f, which is not worth the extra granularity"
                      % (int(alt["k"]), alt["worst_cluster"]))
    if row["worst_cluster"] < C.STABLE:
        reason.append("note: below the %.2f 'fully stable' bar" % C.STABLE)
    return k, "; ".join(reason), m


def main(force: bool = False) -> dict:
    viz.set_style()
    R: dict = {"random_state": C.RANDOM_STATE, "supervised": False}

    # ---------------------------------------------------------------- data --
    print("[1/5] data")
    raw = D.load_customers_raw()
    customers = D.load_customers()
    prospects = D.load_prospects()
    R["duplicates"] = D.duplicate_report(raw)
    R["n_customers"] = int(len(customers))
    R["n_prospects"] = int(len(prospects))
    R["audit"] = D.audit(customers)
    R["overlap"] = D.overlap_with_prospects()
    miss = D.missingness_report(customers)
    R["pct_rows_with_any_missing"] = miss.attrs["pct_rows_with_any_missing"]
    R["missingness"] = miss.to_dict("records")
    _table(miss, "missingness", index=False)

    # ----------------------------------------------------------------- eda --
    print("[2/5] exploratory analysis")
    e = eda.run(customers, prospects)
    R["figures"] = dict(e["figures"])
    R["feature_summary"] = e["feature_summary"].to_dict("records")
    R["redundancy"] = e["redundancy"].to_dict("records")
    R["population_shift"] = e["population_shift"].to_dict("records")
    _table(e["feature_summary"], "feature_summary", index=False)
    _table(e["redundancy"], "feature_redundancy", index=False)
    _table(e["population_shift"], "population_shift", index=False)
    _table(e["association_matrix"], "association_matrix")

    # ---------------------------------------------------------- clustering --
    print("[3/5] segmentation")
    scaled, rawf, num_idx, cat_idx, encoder = F.build_cluster_frame(customers)

    sel = _cached("cluster_selection.pkl",
                  lambda: CL.select_k(scaled, rawf, cat_idx), force)
    R["cluster_selection"] = sel.to_dict("records")
    _table(sel, "cluster_selection", index=False)

    per_cluster, summary = _cached(
        "stability.pkl",
        lambda: CL.stability_across_k(scaled, cat_idx), force)
    R["stability_by_cluster"] = per_cluster.to_dict("records")
    R["stability_summary"] = summary.to_dict("records")
    _table(per_cluster, "stability_by_cluster", index=False)
    _table(summary, "stability_summary", index=False)

    K, why, merged = choose_k(sel, summary)
    R["chosen_k"] = K
    R["chosen_k_reason"] = why
    R["selection_table"] = merged.to_dict("records")
    _table(merged, "k_selection_combined", index=False)
    print("  chosen k = %d (%s)" % (K, why))

    model, labels = _cached(
        "kproto_labels.pkl",
        lambda: CL.fit_kprototypes(scaled, cat_idx, K), force)
    gamma = CL.gamma_of(model)
    protos = CL.extract_prototypes(scaled, labels)
    R["gamma"] = round(gamma, 4)

    prof = CL.profile_clusters(customers, labels)
    traits = CL.distinguishing_features(customers, labels)
    prof = prof.join(traits["defining_traits"])
    R["cluster_profiles"] = prof.reset_index().to_dict("records")
    _table(prof, "cluster_profiles")

    km_labels, _ = _cached("kmeans_labels.pkl",
                           lambda: CL.kmeans_labels(customers, K), force)
    hc_labels, hc_idx = _cached("hier_labels.pkl",
                                lambda: CL.hierarchical_labels(rawf, K), force)
    R["method_agreement"] = CL.method_agreement(labels, km_labels, hc_labels, hc_idx)

    coords, explained, method = _cached("famd.pkl",
                                        lambda: CL.famd_coords(rawf), force)
    R["projection_method"] = method
    R["projection_explained"] = [round(float(v), 2) for v in explained]

    R["figures"]["cluster_selection"] = CL.fig_selection(sel, K)
    R["figures"]["cluster_stability"] = CL.fig_stability(summary, per_cluster, K)
    R["figures"]["cluster_sizes"] = CL.fig_sizes(prof)
    R["figures"]["cluster_personas"] = CL.fig_personas(prof)
    R["figures"]["cluster_projection"] = CL.fig_projection(
        coords, labels, explained, method)
    R["figures"]["method_agreement"] = CL.fig_method_agreement(R["method_agreement"])

    # ------------------------------------------------------------ assign --
    print("[4/5] assigning prospects")
    personas = {int(i): str(prof.loc[i, "defining_traits"]) for i in prof.index}
    scored = score.assign_prospects(encoder, protos, gamma, personas=personas)
    out_csv = C.OUTPUTS / "segmented_new_customers.csv"
    scored.to_csv(out_csv, index=False)
    R["scored_csv"] = str(out_csv.relative_to(C.ROOT)).replace("\\", "/")
    R["n_scored"] = int(len(scored))

    summ = score.assignment_summary(scored)
    comparison = score.compare_to_reference(scored, prof)
    _table(summ, "assignment_summary")
    _table(comparison, "segment_mix_comparison", index=False)
    R["assignment_summary"] = summ.reset_index().to_dict("records")
    R["segment_mix_comparison"] = comparison.to_dict("records")
    R["assignment_tiers"] = {str(k): int(v) for k, v in
                             scored["Assignment_Tier"].value_counts().items()}
    R["pct_clear"] = round(float(
        scored["Separation"].ge(score.CLEAR).mean()) * 100, 1)
    R["pct_borderline"] = round(float(
        scored["Separation"].lt(score.MODERATE).mean()) * 100, 1)
    R["figures"]["assignment"] = score.fig_assignment(scored, comparison)

    joblib.dump({"encoder": encoder, "prototypes": protos, "gamma": gamma,
                 "k": K, "features": C.FEATURES},
                C.OUTPUTS / "segmentation_model.joblib")

    # ------------------------------------------------------------- finish --
    print("[5/5] writing results")
    with open(C.RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, default=str)
    print("\nDone. k=%d; %d figures; %s" % (K, len(R["figures"]), out_csv.name))
    return R


if __name__ == "__main__":
    main(force="--force" in sys.argv)
