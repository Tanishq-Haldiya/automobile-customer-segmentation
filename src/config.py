"""Central paths and column definitions.

This project is deliberately **unsupervised end to end**. No label is read at any
point: `data.load_customers()` returns the 9 feature columns and nothing else.
The raw CSV happens to carry a legacy sales-team column, but a real market-entry
segmentation has no such column, so the pipeline behaves as though it does not
exist. Cluster quality is therefore judged by internal validity and by
resampling stability, never by agreement with an external answer key.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

# Everything a human reads lives under docs/; everything a machine consumes
# lives under outputs/ or data/.
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
TABLES = DOCS / "tables"
RESULTS_JSON = DOCS / "results.json"

for _p in (PROCESSED, OUTPUTS, DOCS, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

CUSTOMERS_CSV = RAW / "train.csv"                 # existing-market customers
PROSPECTS_CSV = RAW / "test_new_customers.csv"    # new-market prospects

ID_COL = "ID"
# Present in the raw file, never loaded. Named here only so the loader can drop
# it explicitly and the test suite can assert it never reaches a model.
LEGACY_LABEL = "Segmentation"

NUMERIC = ["Age", "Work_Experience", "Family_Size"]
ORDINAL = ["Spending_Score"]
ORDINAL_LEVELS = {"Spending_Score": ["Low", "Average", "High"]}
NOMINAL = ["Gender", "Ever_Married", "Graduated", "Profession", "Var_1"]

FEATURES = NUMERIC + ORDINAL + NOMINAL
CATEGORICAL = ORDINAL + NOMINAL

MISSING_TOKEN = "Missing"
RANDOM_STATE = 42

# Cluster-count search and validation budgets.
K_RANGE = range(2, 9)
GOWER_SAMPLE = 2500       # rows used for the Gower silhouette
BOOTSTRAP_B = 20          # resamples per k for stability
STABLE = 0.75             # mean Jaccard above this = a stable cluster
PATTERN = 0.60            # 0.60-0.75 = a real but fuzzy pattern

# --------------------------------------------------------------------------- #
# Chart palettes, verified with `python -m src.palette_check`
# (OKLCH lightness band, chroma floor, Machado-2009 protan/deutan separation,
# normal-vision floor, WCAG contrast against the #fcfcfb chart surface).
#
#   CLUSTER_COLORS -- PASS, --pairs adjacent. One contrast WARN on slot 4,
#                     covered by the rule that every cluster-coloured chart
#                     also carries a direct label or an adjacent table.
#
# Hues are assigned in fixed order and never cycled. A 9th cluster would fold
# into "Other" rather than getting a generated hue.
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
CLUSTER_COLORS = ["#0072B2", "#B37400", "#00996E", "#A8478C",
                  "#56B4E9", "#D55E00", "#4A3AA7", "#7D7D24"]
PALETTE = CLUSTER_COLORS

# Recessive ink tokens -- text never wears a series colour.
INK = "#1a1a19"
INK_SECONDARY = "#55554f"
INK_MUTED = "#8a8a80"
GRID = "#e4e4df"
SEQUENTIAL_HUE = "#0072B2"                     # single hue, light -> dark
DIVERGING = ("#B37400", "#f2f2ee", "#0072B2")  # warm / neutral / cool


def cluster_name(i: int) -> str:
    return "Cluster %d" % i
