# Automobile Customer Segmentation

Unsupervised segmentation of an automobile company's customer base, and
assignment of new-market prospects to the segments that were found.

The company sells five products (P1–P5) in an existing market and is entering a
new one. It holds records on 8,068 existing customers and has identified 2,627
prospects. It needs to know what kinds of customers it serves, and which kind
each prospect is.

## Approach

No label is used at any stage — a company entering a new market does not have
one, and creating segments is the point of the exercise. Clustering runs on nine
mixed numeric and categorical features with **K-Prototypes**, which handles both
types in one distance function.

Because there is no ground truth, quality is judged by **bootstrap stability**:
resample the customers, re-cluster, and measure how often the same people group
together again. Ward-on-Gower and K-Means provide independent cross-checks.

## Results

Three segments, each highly reproducible (bootstrap Jaccard 0.97–0.98):

| | **0 — Established Households** | **1 — Young Large Households** | **2 — Mid-Career Professionals** |
|---|---|---|---|
| Size | 3,594 (49.0%) | 2,183 (29.8%) | 1,557 (21.2%) |
| Median age | 51 | 29 | 37 |
| % married | 77.2 | 30.0 | 51.8 |
| % graduated | 74.0 | 35.0 | 67.2 |
| % low spenders | 49.9 | 75.5 | 65.3 |
| Median family size | 2 | 4 | 2 |
| Median work experience | 1 yr | 1 yr | **8 yrs** |
| Top profession | Artist (39%) | Healthcare (37%) | Artist (33%) |

Three was chosen because stability collapses beyond it — the weakest segment
falls from 0.97 at k=3 to 0.59 at k=4.

Segment 2 is defined almost entirely by **work experience**, a median of 8 years
against 1 in the other two. A segmentation built only on the obvious
demographics would have missed it.

The new market closely resembles the existing one: across all nine features the
largest difference is 0.068, and the prospect pool splits into the same segments
within 2.4 percentage points.

## The deliverable

[`outputs/segmented_new_customers.csv`](outputs/segmented_new_customers.csv) —
2,627 prospects with their assigned segment, persona, distance to each
prototype, and a `Separation` score measuring how much closer the winning
segment is than the runner-up.

| Tier | Prospects | Handling |
|---|---|---|
| Clear | 1,841 (70.1%) | Automate segment-specific outreach |
| Moderate | 499 (19.0%) | Standard outreach, monitor response |
| Borderline | 287 (10.9%) | Blend two segments' messaging, or hold back |

## Quick start

```bash
pip install -r requirements.txt
python -m src.run_all
```

Regenerates every figure, table, the assigned prospect list and
`docs/results.json`. Expensive steps are cached under `data/processed/`; pass
`--force` to recompute. A cold run takes about 12 minutes.

```bash
python -m pytest -q tests/
```

The notebooks narrate and display; all computation lives in `src/`. Run the
pipeline first, then open them.

## Layout

```
data/raw/          train.csv, test_new_customers.csv, DATA_SOURCE.md
src/               the pipeline — data, features, clustering, scoring
notebooks/         01 EDA, 02 segmentation, 03 assign & deliver
tests/             17 invariant tests
outputs/           the deliverable CSV and the fitted model
docs/              generated locally: report, figures, tables, results.json
```

## Data

The **Customer Segmentation** dataset, originally an Analytics Vidhya JanataHack
problem and mirrored on Kaggle. See
[`data/raw/DATA_SOURCE.md`](data/raw/DATA_SOURCE.md) for provenance, schema,
integrity verification and caveats.
