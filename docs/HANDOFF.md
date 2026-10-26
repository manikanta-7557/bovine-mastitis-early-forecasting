# Bovine Mastitis ANFIS — Handoff for collaborators

Interpretable detection of (clinical) bovine mastitis with an ANFIS (Sugeno, subtractive clustering, hybrid learning).
Course: Soft Computing (M.Tech). Problem statement: SIH26109. Dataset: Mendeley "Clinical Mastitis in Cows based on Udder Parameter using IoT" (DOI 10.17632/kbvcdw5b4m; Version 1 = `.1`, 22 Oct 2020, CSV 413 KB; Version 2 = `.2`, 20 Nov 2020, 1.48 GB zip). **Our working CSV is NOT the Version 1 CSV; per A it comes from the Version 2 zip (SHA256 check pending, issue 2b).**

## 1. Read this first (5 minutes)

- Goal of the project: show a **rule-based, explainable** model can detect mastitis from udder sensor readings, compared with black-box ML (MasPA, Ghafoor & Sitkowska 2021).
- Rule for the whole project: **no invented numbers, no invented citations.** Every metric in a slide or paper comes from a script output file.
- Review II rubric (30 marks): lit survey 3 | system design 4 | data prep 3 | implementation 8 (needs 50-70% done) | prelim results 4 | article draft 4 | SC justification 2 | presentation/viva 2.

## 2. Repo layout

```
data/raw/clinical_mastitis_cows.csv     (6600 rows x 18 cols; NOT committed if large/private)
data/processed/                         train.csv, test.csv, full_processed.csv, data_prep_summary.json
reports/figures/                        fig1..fig8 PNGs (slide-ready)
reports/baseline_results.csv            output of baselines.py
data_prep.py   metrics.py   baselines.py                              (Modules 1-2: run on real data)
subtractive_clustering.py  anfis.py  explain.py  train_anfis.py       (Modules 3-5: written, unit-tested,
                                                                       dry-run on synthetic data; FIRST real run pending)
test_modules.py                         (python test_modules.py -> 4 tests, all should print OK)
models/  reports/rules_*.txt  reports/example_explanations.txt  reports/comparison.csv   (outputs of train_anfis.py)
app.py                                  (TODO: Streamlit demo, minimal for Review II)
```

## 3. Setup and run order

```bash
pip install pandas numpy scikit-learn matplotlib streamlit
python verify_dataset_version.py data/raw/clinical_mastitis_cows.csv   # fingerprint (SHA256) of our data file
python data_prep.py        # Module 1 -> data/processed + figures
python baselines.py        # Module 2 -> reports/baseline_results.csv + fig8
python test_modules.py     # sanity: clustering, gradient check, LSE, hybrid learning
python train_anfis.py      # Modules 3-5 -> reports/anfis_results.csv, comparison.csv, rules, figures 9-10
```

## 4. What Module 1 does and why (the facts you must be able to defend)

| Decision | Why |
|---|---|
| Dropped `Milk_visibility` | Identical to `class1` (r = 1.0, crosstab off-diagonals are 0). It IS the target. Evidence saved in `data_prep_summary.json` and fig2. |
| Cow-grouped split (`StratifiedGroupKFold`, fold 0 = test) | Each cow has 6 near-identical rows. Random split puts a cow's Day 2 in train and Day 5 in test, so the model memorises the cow. Result: 880 train cows / 220 test cows, zero overlap. |
| Scaler fitted on train only | Fitting on all rows leaks test min/max into training. 0 test values needed clipping. |
| Udder Swelling Index = mean of 8 sensors | One number for overall udder fullness. |
| Udder Asymmetry Index = (max-min)/mean of 4 quadrant totals (I* + E* = inhale limit + exhale limit) | Mastitis usually hits one quarter, so the spread between quarters jumps. Scale-free. This formula is OUR design choice, own it in the viva. |
| Two feature sets | **Early-Warning** (swelling, asymmetry, temperature, months_pp, breed): sensors only. **Full-Clinical** = Early + Hardness + Pain (needs a person to palpate the cow). |
| `Previous_Mastits_status` excluded from features | ~0.4% positive, a membership function cannot learn from it. Kept in CSVs. |
| Breed `'hostlene'` normalised to Holstein | Spelling in source file. Jersey = 0, Holstein = 1. |

Real-data numbers so far: 1100 cows x 6 days; split train mastitis fraction 0.3941, test 0.4227; single-rule accuracy: Hardness 0.8718, Pain 0.7729, prev_mastitis 0.6038.

## 5. Open issues (decide together, do not ignore)

1. **Forecasting vs detection.** Label is constant for 1099 of 1100 cows across Day 1-6. So the data supports *detection* of existing clinical mastitis, not time-ahead forecasting and not "early"/"subclinical" claims (the dataset page says the data is for detecting *clinical* mastitis). The 1 cow whose label changes is cow10 (Day 6: Hardness 1->0 and label 1->0); decision: keep, log in JOURNAL.
2. **Meaning of I* / E* sensor columns** (IUFL, EUFL, ...): the dataset page lists, for each of the 4 quarters, an "inhale limit" and an "exhale limit" (8 columns) but does not name the columns. I* = inhale, E* = exhale is very likely (letters match); write "likely" until a source confirms. Our earlier "inner/external" wording was wrong.
2b. **Dataset version.** The page has Version 1 and Version 2 with an identical description. Cite the DOI of the version we used. **Finding 2026-10-25:** the Version 1 CSV and our working file differ (Version 1: 16.8% mastitis rows, label almost equal to Hardness and Pain, no cow with a changing label; our file: 40% mastitis rows, cow10 changes). On Version 1 all models tie (row F1 0.972 Early-Warning, 0.958 Full-Clinical). Per A the working file is from the Version 2 zip; confirm by comparing SHA256 with the CSV inside the zip (`verify_dataset_version.py`), and run `compare_versions_deep.py` to describe the differences (1,529 labels differ, all 0 -> 1; sensor values also changed; cause undocumented, so never call it synthetic). SHA256 must also match between partners. Decision D13: Version 2 main, Version 1 as sensitivity check. Choose the dataset by provenance, never by which file flatters ANFIS.
2c. **Label rule undocumented.** The page says "normal cows 0, mastitis cows 1" and "milk quality 0 normal / 1 abnormal", but not how class1 was assigned; in our file class1 equals the milk-quality flag on every row. State this in the report's limitations.
3. **Hardness alone = 87% accurate.** Full-Clinical results will look great and prove little. The Early-Warning set is the real contribution.
4. One test fold (220 cows) is noisy. Final Review: grouped 5-fold CV using the existing `fold` column.

## 6. Module interfaces (implemented; keep stable if you edit)

- `subtractive_clustering.py`: `initial_rule_params(X, y, ra, use_target=True, max_rules=12) -> (centers, sigmas)`. Chiu 1994, clusters on [inputs, label] of TRAIN data only; sigma = ra / sqrt(8).
- `anfis.py`: class `ANFIS(centers, sigmas)` with `fit(X, y, X_val, y_val)`, `predict_score(X)`, `predict(X)`. 5 layers: fuzzify, firing strength (product), normalise, Sugeno consequent (linear), sum. Hybrid learning: LSE for consequents (forward), gradient descent for MF centres/widths (backward).
- `explain.py`: `describe_rules`, `explain_row` (fired-rule trace), `fuzzy_risk` / `risk_tier` (Low / Watch / High), `rank_cows`.
- All models report through `metrics.evaluate(y_true, y_prob, cow_ids)` so results are comparable.

## 7. Working agreement (two people)

- Git: `main` is always runnable. Work on branches `feat/<module>-<name>`, merge by PR/self-review, small commits (`data: ...`, `anfis: ...`). Tag `review2` at submission.
- Research journal: append to `JOURNAL.md` every session (date, what you did, what you learned, what is next). Evidence for the viva.
- Nobody merges code they cannot explain. Before merging, explain it to the other person for 5 minutes.
- Numbers go into slides only from `reports/*.csv` / `data_prep_summary.json`.

## 8. Known caveats for the viva

- ANFIS scores can fall outside [0, 1] (linear consequents extrapolate). Threshold stays 0.5; risk tiers clip the score. Say so before they ask.
- The "risk ~ x" printed per rule is that rule's output at its own prototype point, not at the animal being explained.
- `ra` is chosen on a validation fold of TRAIN cows; the test fold is scored once. Do not tune on the test fold.
- Synthetic-data dry runs only prove the code runs. Real numbers come from your own run of `train_anfis.py`.

## Changelog
- 2026-10-25: data_prep comments corrected (I*/E* = inhale/exhale, very likely); dataset versions added; wording changed from "early detection" to "detection of clinical mastitis"; label-rule and version provenance added to open issues; cow10 decision recorded.
- 2026-10-24: first version (modules 1-5, caveats).
- 2026-10-25 (later): working CSV found to differ from the Version 1 CSV; provenance is now the blocker (issue 2b).
- 2026-10-25 (evening): provenance resolved per A's account (Version 2 zip, hash check pending); `compare_versions_deep.py` added; external PDF reviewed (see roadmap section 2c).
- 2026-10-26: 12 papers read (`docs/LITERATURE.md`). MasPA (Ghafoor & Sitkowska 2021) and Kumar et al. 2023 use our dataset (3,961 / 2,639) and report about 99% with oversample-before-split and row-level splits. `leakage_replication.py` measures the inflation. Earlier statement "MasPA uses AMS data" was wrong.
