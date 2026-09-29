# Data source & provenance

## Dataset
**Customer Segmentation** — an automobile company entering new markets with its
existing product line (P1–P5). In its current market the sales team has manually
classified every customer into one of 4 segments (A, B, C, D) and run segmented
outreach per segment. The company has now identified 2,627 potential customers in
the new market and wants them assigned to the same 4 segments.

Originally released as an Analytics Vidhya JanataHack hackathon problem; mirrored
on Kaggle as `vetrirah/customer` and `kaushiksuresh147/customer-segmentation`.

## Files
| File | Rows | Cols | Description |
|---|---|---|---|
| `train.csv` | 8,068 | 11 | Existing-market customers. The project reads 9 feature columns from this file; the 11th (`Segmentation`) is ignored. |
| `test_new_customers.csv` | 2,627 | 10 | New-market prospects to be assigned to the discovered segments. |

## Schema
| Column | Type | Notes |
|---|---|---|
| `ID` | int | Row identifier. **Not** a stable customer key — see caveat 2 |
| `Gender` | cat | Male / Female |
| `Ever_Married` | cat | Yes / No — 140 missing in train |
| `Age` | int | 18–89, mean 43.5 |
| `Graduated` | cat | Yes / No — 78 missing |
| `Profession` | cat | 9 values (Artist, Doctor, Engineer, Entertainment, Executive, Healthcare, Homemaker, Lawyer, Marketing) — 124 missing |
| `Work_Experience` | float | 0–14 years — 829 missing (10.3%) |
| `Spending_Score` | ord | Low / Average / High |
| `Family_Size` | float | 1–9 — 335 missing |
| `Var_1` | cat | Cat_1 … Cat_7, anonymised — 76 missing |
| `Segmentation` | *(unused)* | A legacy sales-team label, A / B / C / D. **Never read by this project** — see "How this project uses the data" below. |

## Retrieval
Downloaded 2026-09-21 from public GitHub mirrors (Kaggle requires an authenticated
account, so mirrors were used and cross-verified):

- `train.csv` ← https://raw.githubusercontent.com/trevortnguyen/Customer-Segmentation-Classification/master/Train.csv
- `test_new_customers.csv` ← https://raw.githubusercontent.com/jebas-py/kaggle_customer_segmentation/main/test.csv

## Integrity verification performed
1. `train.csv` was downloaded from **three independent** repositories
   (trevortnguyen, snaffisah, jebas-py). All three are **byte-identical**
   (md5 `2af30f5dda88ee0836783183fe3f27d3`), which rules out one-off tampering
   by any single uploader.
2. The feature columns of the three test-set copies are identical.
3. Row counts (8,068 / 2,627), the 4-way label balance and the column schema all
   match the published dataset description.

SHA-256 of the files as shipped here: see [`SHA256SUMS.txt`](SHA256SUMS.txt).

## How this project uses the data

**This project is unsupervised end to end. The `Segmentation` column is never
read.**

[`src/data.py`](../../src/data.py) returns the 9 feature columns and nothing else; `validate_schema()`
raises if a non-feature column ever leaks into a modelling frame, and
[`tests/test_pipeline.py`](../../tests/test_pipeline.py) asserts that no module outside `config.py` and `data.py`
so much as names the label. The column is documented below for provenance only.

The reasoning: a company entering a genuinely new market has no segment labels
for it, and usually none for its existing market either -- the whole point of a
segmentation exercise is to create them. Building against a label that would not
exist in production makes the result undeployable. It also removes the temptation
to quietly tune an "unsupervised" method until it reproduces a known answer,
which is not unsupervised learning at all.

Consequently caveat 1 below is of historical interest only: this project does not
use the test set's labels because it does not use *any* labels.

## Caveats found during verification (important)

**1. Do not trust "labelled" copies of the test set.**
One mirror (`trevortnguyen/.../Test.csv`) ships a `Segmentation` column for the
2,627 test rows. Those labels are **not genuine**:
- A gradient-boosting model scoring **50.8%** 5-fold CV accuracy on `train.csv`
  scores only **31.9%** against them — barely above the 28.1% majority-class
  baseline and the 25% random baseline.
- Feature→label relationships are flat in that file. In `train.csv`,
  `Spending_Score = Average` maps to segment C 45.7% / D 7.0% of the time; in that
  test file the same value maps to A 31% / B 24% / C 22% / D 23% — i.e. ~uniform.
Conclusion: those labels are noise. The prospect file shipped here is therefore
deliberately **unlabelled** — which costs this project nothing, since it reads no
labels from either file. The finding is kept as a caution: data can arrive
looking complete and still be fabricated, so verify before trusting.

**2. `ID` is not a cross-file customer key.**
2,332 of the 2,627 test IDs also appear in `train.csv`, but only 136 of those
pairs have identical feature values. The two files reuse the same ID range for
different people. Drop `ID` from the feature set; never join the files on it.
This project drops it at load and re-attaches the prospects' IDs to the output
only after assignment.

**3. Duplicate rows exist.** 734 rows in `train.csv` are exact duplicates of
another row across all 9 feature columns. They are removed from the clustering
population: a cluster prototype is a mean and a mode, so repeated identical
customers pull it toward whatever the export happens to repeat — a property of
the file, not of the market. The prospect file is deliberately *not*
de-duplicated, because every prospect needs a row in the deliverable.

**4. The 4 segments were a business labelling, not ground truth.** They were
assigned manually by a sales team and carry real human inconsistency. This is
part of why the project ignores them: a segmentation built to reproduce one
sales team's historical opinions inherits every flaw in those opinions without
inheriting anyone's accountability for them.

## Licence
The Kaggle mirrors are published under CC0 / public domain. Free for educational
and portfolio use.
