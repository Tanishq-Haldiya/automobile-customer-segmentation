"""Generate the three thin notebooks from a spec.

The notebooks narrate and display; every computation lives in `src/`.
Regenerate with:  python tools/make_notebooks.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
NB.mkdir(exist_ok=True)

BOOT = """import sys, warnings
sys.path.insert(0, "..")
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, joblib
from IPython.display import Image, display
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

from src import config as C, data as D, features as F, viz
viz.set_style()"""

SPECS = {
    "01_eda.ipynb": [
        ("md", "# 1 - Exploring an unlabelled customer base\n\n"
               "This project is **unsupervised end to end**. `D.load_customers()` "
               "returns the 9 feature columns and nothing else - the raw CSV's "
               "legacy sales-team column is never read, because a real "
               "market-entry segmentation would not have one.\n\n"
               "Every computation lives in `src/`; this notebook narrates and "
               "displays. **Run `python -m src.run_all` first.**"),
        ("code", BOOT),
        ("md", "## Load\n\n"
               "The clustering population is de-duplicated on the 9 features: "
               "identical customers would otherwise drag a prototype toward "
               "whatever the export happens to repeat. The prospect file is "
               "*not* de-duplicated - every prospect needs an output row."),
        ("code", 'raw = D.load_customers_raw()\ncustomers = D.load_customers()\n'
                 'prospects = D.load_prospects()\n'
                 'print("existing market:", customers.shape, "| prospects:", prospects.shape)\n'
                 'print("columns:", list(customers.columns))\ncustomers.head()'),
        ("code", 'import json\nprint(json.dumps(D.duplicate_report(raw), indent=1))'),
        ("md", "## Integrity audit\n\n"
               "`validate_schema` also fails loudly if a non-feature column ever "
               "leaks into the frame - that is the guard that keeps this project "
               "unsupervised."),
        ("code", 'print(json.dumps(D.audit(customers), indent=1))'),
        ("md", "## What each feature can contribute\n\n"
               "`concentration` is the share held by the single most common "
               "value. A feature near 1.0 splits the population barely at all "
               "and cannot do much work in a distance calculation."),
        ("code", 'from src import eda\neda.feature_summary(customers)'),
        ("md", "## Redundancy between features\n\n"
               "This matters more without a label than with one. Two strongly "
               "associated features effectively vote twice in the distance "
               "function, quietly reweighting the segmentation toward whatever "
               "they happen to share."),
        ("code", 'eda.redundancy_report(customers).head(10)'),
        ("md", "## Missingness\n\n"
               "With no label there is no way to ask whether missingness is "
               "*predictive*. The answerable question is whether the gaps "
               "co-occur - which would point at one upstream cause."),
        ("code", 'm = D.missingness_report(customers)\n'
                 'print("rows missing at least one field: %.1f%%" % '
                 'm.attrs["pct_rows_with_any_missing"])\nm'),
        ("md", "## Do the new-market prospects resemble the existing market?\n\n"
               "With no labels anywhere, this is the only available check on "
               "whether segments learned in one market can be applied to the "
               "other."),
        ("code", 'eda.population_shift(customers, prospects)'),
        ("md", "## Figures"),
        ("code", 'for name in ["01_feature_overview", "02_missingness",\n'
                 '             "03_association_matrix", "10_population_shift"]:\n'
                 '    display(Image(filename=str(C.FIGURES / (name + ".png"))))'),
    ],
    "02_segmentation.ipynb": [
        ("md", "# 2 - Finding the segments\n\n"
               "K-Prototypes is the primary algorithm: the data is 6/9 "
               "categorical, and K-Means on one-hot columns makes the distance "
               "between any two professions a constant, so it cannot express "
               "that Doctor is nearer Healthcare than Artist.\n\n"
               "With no label there is no answer key, so quality is established "
               "three ways that need none: internal validity (Gower "
               "silhouette), **resampling stability**, and agreement between "
               "structurally different algorithms."),
        ("code", BOOT + "\n\nfrom src import cluster as CL"),
        ("code", 'customers = D.load_customers()\n'
                 'scaled, raw, num_idx, cat_idx, encoder = F.build_cluster_frame(customers)\n'
                 'print(scaled.shape, "| categorical column indices:", cat_idx)'),
        ("md", "## Step 1 - internal validity across k\n\n"
               "Cached by `run_all` because K-Prototypes over the whole k range "
               "takes several minutes."),
        ("code", 'sel = joblib.load(C.PROCESSED / "cluster_selection.pkl")\nsel'),
        ("md", "Every silhouette is low. That is the finding, not a failure: the "
               "customer base is a **gradient**, so no k produces well-separated "
               "clumps and the silhouette alone cannot choose one."),
        ("md", "## Step 2 - bootstrap stability (the decisive check)\n\n"
               "Resample the customers with replacement, recluster, and measure "
               "how often the same people stay together (Hennig 2007). A "
               "partition that dissolves under resampling is an artefact of this "
               "particular sample, whatever its silhouette says."),
        ("code", 'per_cluster, summary = joblib.load(C.PROCESSED / "stability.pkl")\n'
                 'summary'),
        ("code", 'per_cluster'),
        ("md", "## Step 3 - the chosen k"),
        ("code", 'R = json.load(open(C.RESULTS_JSON, encoding="utf-8")) '
                 'if False else None\n'
                 'import json\n'
                 'R = json.load(open(C.RESULTS_JSON, encoding="utf-8"))\n'
                 'print("k =", R["chosen_k"])\nprint(R["chosen_k_reason"])'),
        ("code", 'for name in ["04_cluster_selection", "05_cluster_stability"]:\n'
                 '    display(Image(filename=str(C.FIGURES / (name + ".png"))))'),
        ("md", "## The segments"),
        ("code", 'model, labels = joblib.load(C.PROCESSED / "kproto_labels.pkl")\n'
                 'prof = CL.profile_clusters(customers, labels)\n'
                 'prof.join(CL.distinguishing_features(customers, labels)["defining_traits"])'),
        ("code", 'for name in ["06_cluster_sizes", "07_cluster_personas",\n'
                 '             "08_cluster_projection"]:\n'
                 '    display(Image(filename=str(C.FIGURES / (name + ".png"))))'),
        ("md", "## Step 4 - do different algorithms agree?\n\n"
               "Adjusted Rand Index here compares two **clusterings** with each "
               "other. No label is involved. If the structure were strong, "
               "structurally different algorithms would recover it."),
        ("code", 'print(json.dumps(R["method_agreement"], indent=1))\n'
                 'display(Image(filename=str(C.FIGURES / "09_method_agreement.png")))'),
    ],
    "03_assign_and_deliver.ipynb": [
        ("md", "# 3 - Assigning the new market\n\n"
               "Prospects are **assigned**, not re-clustered. Re-clustering them "
               "would produce segments numbered and shaped differently from the "
               "existing market's, so no campaign could carry across. The fitted "
               "encoder and prototypes are applied unchanged."),
        ("code", BOOT + "\n\nfrom src import cluster as CL, score\nimport json"),
        ("code", 'R = json.load(open(C.RESULTS_JSON, encoding="utf-8"))\n'
                 'art = joblib.load(C.OUTPUTS / "segmentation_model.joblib")\n'
                 'print("k =", art["k"], "| gamma = %.3f" % art["gamma"])\n'
                 'print("prototype for segment 0:"); art["prototypes"][0]'),
        ("md", "## The deliverable"),
        ("code", 'scored = pd.read_csv(C.OUTPUTS / "segmented_new_customers.csv")\n'
                 'print(scored.shape)\nscored.head()'),
        ("code", 'score.assignment_summary(scored)'),
        ("md", "## How confident is each assignment?\n\n"
               "`Separation` is how much closer the winning prototype is than "
               "the runner-up: 0 means the customer sits midway between two "
               "segments, 1 means they sit on the prototype. In a continuum a "
               "large borderline group is expected, and saying so is more useful "
               "than emitting a confident-looking segment id."),
        ("code", 'scored["Assignment_Tier"].value_counts()'),
        ("md", "## Does the new market have the same shape as the old one?"),
        ("code", 'pd.DataFrame(R["segment_mix_comparison"])'),
        ("code", 'display(Image(filename=str(C.FIGURES / "11_assignment.png")))'),
        ("md", "## Using the saved model on future prospects\n\n"
               "The artefact carries the encoder, the prototypes and gamma - "
               "everything needed to assign a new customer without re-running "
               "the clustering."),
        ("code", 'new = D.load_prospects().head(5)\n'
                 'scaled, _ = art["encoder"].transform(new)\n'
                 'labels, conf = CL.assign(scaled, art["prototypes"], art["gamma"])\n'
                 'conf.assign(Tier=score.tier(conf["Separation"]))'),
    ],
}


def cell(kind, src):
    lines = src.split("\n")
    body = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": body}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": body}


def main():
    for name, spec in SPECS.items():
        nb = {
            "cells": [cell(k, s) for k, s in spec],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python",
                               "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"},
            },
            "nbformat": 4, "nbformat_minor": 5,
        }
        (NB / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote notebooks/%s (%d cells)" % (name, len(spec)))


if __name__ == "__main__":
    main()
