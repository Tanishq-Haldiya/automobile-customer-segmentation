"""Guards for the invariants the README and report claim.

Run:  python -m pytest -q
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import cluster as CL
from src import config as C
from src import data as D
from src import features as F

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(scope="module")
def customers():
    return D.load_customers()


@pytest.fixture(scope="module")
def prospects():
    return D.load_prospects()


# --------------------------------------------------------------------------- #
# the project is unsupervised -- these are the load-bearing tests
# --------------------------------------------------------------------------- #
def test_loaders_return_features_only(customers, prospects):
    assert list(customers.columns) == C.FEATURES
    assert list(prospects.columns) == C.FEATURES
    for frame in (customers, prospects):
        assert C.LEGACY_LABEL not in frame.columns
        assert C.ID_COL not in frame.columns


def test_no_module_consumes_the_legacy_label():
    """Only the loader and config may even name it, and only to exclude it."""
    allowed = {"config.py", "data.py"}
    offenders = []
    for path in SRC.glob("*.py"):
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        code = "\n".join(
            l for l in text.split("\n") if not l.strip().startswith("#"))
        code = re.sub(r'"""].*?["]{3}', "", code, flags=re.S)
        if C.LEGACY_LABEL in code:
            offenders.append(path.name)
    assert not offenders, "modules reference the label: %s" % offenders


def test_label_column_never_survives_clean():
    raw = D.load_customers_raw()
    assert C.LEGACY_LABEL in raw.columns, "fixture assumption: raw file has it"
    assert C.LEGACY_LABEL not in D.clean(raw).columns


# --------------------------------------------------------------------------- #
# data hygiene
# --------------------------------------------------------------------------- #
def test_raw_files_have_expected_shape():
    assert D.load_customers_raw().shape == (8068, 11)
    assert D.load_prospects_raw().shape == (2627, 10)


def test_clustering_population_is_deduplicated(customers):
    assert customers.duplicated(subset=C.FEATURES).sum() == 0


def test_prospects_are_not_deduplicated_and_stay_aligned(prospects):
    raw = D.load_prospects_raw()
    assert len(prospects) == len(raw), "every prospect needs an output row"
    assert prospects["Age"].tolist() == raw["Age"].tolist()


def test_schema_is_clean(customers, prospects):
    assert D.validate_schema(customers) == []
    assert D.validate_schema(prospects) == []


def test_validate_schema_flags_a_leaked_label(customers):
    bad = customers.copy()
    bad[C.LEGACY_LABEL] = "A"
    assert any("leaked" in p for p in D.validate_schema(bad))


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
def test_encoder_is_fitted_once_and_reused(customers, prospects):
    """Prospects must be transformed by the customers' encoder, never re-fitted."""
    enc = F.MixedFrameEncoder().fit(customers)
    a, _ = enc.transform(customers)
    b, _ = enc.transform(prospects)
    enc2 = F.MixedFrameEncoder().fit(prospects)
    c, _ = enc2.transform(prospects)
    assert not np.allclose(b[C.NUMERIC].mean().to_numpy(),
                           c[C.NUMERIC].mean().to_numpy()), \
        "re-fitting on prospects should move the origin -- that is why we do not"
    assert abs(float(a[C.NUMERIC].mean().mean())) < 1e-9


def test_no_nans_survive_the_cluster_frame(customers):
    scaled, raw, _, _, _ = F.build_cluster_frame(customers)
    assert not scaled.isna().any().any()
    assert not raw.isna().any().any()


def test_missing_becomes_its_own_level(customers):
    _, raw, _, _, _ = F.build_cluster_frame(customers)
    present = {lvl for c in C.CATEGORICAL for lvl in raw[c].unique()}
    assert C.MISSING_TOKEN in present


def test_spending_score_is_encoded_ordinally():
    X = pd.DataFrame({
        "Age": [30, 30, 30], "Work_Experience": [1.0, 1.0, 1.0],
        "Family_Size": [2.0, 2.0, 2.0],
        "Spending_Score": ["Low", "Average", "High"],
        "Gender": ["Male"] * 3, "Ever_Married": ["Yes"] * 3,
        "Graduated": ["Yes"] * 3, "Profession": ["Artist"] * 3,
        "Var_1": ["Cat_6"] * 3,
    })[C.FEATURES]
    pre = F.build_preprocessor().fit(X)
    names = [str(n) for n in pre.get_feature_names_out()]
    col = names.index([n for n in names if "Spending_Score" in n][0])
    vals = np.asarray(pre.transform(X), dtype=float)[:, col]
    assert vals[0] < vals[1] < vals[2], "Low < Average < High ordering lost"


# --------------------------------------------------------------------------- #
# clustering and assignment
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def fitted(customers):
    scaled, raw, _, cat_idx, enc = F.build_cluster_frame(customers)
    model, labels = CL.fit_kprototypes(scaled, cat_idx, 3, n_init=1)
    return scaled, cat_idx, enc, model, labels


def test_assignment_matches_kprototypes(fitted):
    """Our prototype distance must reproduce K-Prototypes' own assignment.

    The deliverable assigns prospects with this function, so if it diverged from
    the algorithm that built the segments, every assignment would be subtly wrong.
    """
    scaled, _, _, model, labels = fitted
    protos = CL.extract_prototypes(scaled, labels)
    assigned, _ = CL.assign(scaled, protos, CL.gamma_of(model))
    assert (assigned == labels).mean() == 1.0


def test_separation_is_a_unit_interval(fitted):
    scaled, _, _, model, labels = fitted
    protos = CL.extract_prototypes(scaled, labels)
    _, conf = CL.assign(scaled, protos, CL.gamma_of(model))
    assert conf["Separation"].between(0, 1).all()
    assert (conf["Runner_Up"] != conf["Cluster"]).all()


def test_every_cluster_is_non_empty(fitted):
    _, _, _, _, labels = fitted
    assert len(np.unique(labels)) == 3
    assert pd.Series(labels).value_counts().min() > 0


def test_method_agreement_needs_no_label(fitted, customers):
    """ARI here compares two clusterings, not a clustering against an answer key."""
    _, _, _, _, labels = fitted
    km, _ = CL.kmeans_labels(customers, 3)
    out = CL.method_agreement(labels, km, labels[:10], np.arange(10))
    assert set(out) == {"K-Prototypes vs K-Means", "K-Prototypes vs Ward/Gower",
                        "K-Means vs Ward/Gower"}
    assert all(-1.0 <= v <= 1.0 for v in out.values())


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #
def test_palette_passes_the_colour_checks():
    from src.palette_check import validate
    r = validate(C.CLUSTER_COLORS, mode="light", pairs="adjacent")
    assert r["ok"], r["failures"]
