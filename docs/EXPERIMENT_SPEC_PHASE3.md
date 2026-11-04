# Phase 3 experiment specification: feature-set comparison

**Status:** proposed protocol; not executed. Freeze this specification and configuration
before training. It does not change current features, labels, folds or reported results.

## Research question and feature sets

Compare the same model families on the same cow-grouped observations:

1. **Raw-Sensors + context:** the eight raw udder sensor columns (`IUFL`, `EUFL`, `IUFR`,
   `EUFR`, `IURL`, `EURL`, `IURR`, `EURR`), plus raw `Temperature`, `Months after giving
   birth`, and `Breed` encoded with the documented Jersey/Holstein mapping. Fit the same
   training-fold-only MinMax scaling for continuous numeric inputs, while leaving the
   binary breed encoding unchanged. This matches the raw-input comparison described in the
   project roadmap.
2. **Sensor-only engineered:** `swelling_index`, `asymmetry_index`, `temperature_n`,
   `months_pp_n`, `breed_enc`.
3. **Full-Clinical:** Sensor-only engineered plus `hardness` and `pain`.

The target is `class1`; grouping is by `Cow_ID`. Exclude `Milk_visibility`,
`Previous_Mastits_status`, identifiers and the target from predictors. Keep `cow10` and all
its rows. The raw-input variant is a planned comparison, not a result currently available.

## Evaluation and preprocessing

- Use the same five outer cow-grouped folds for all feature sets and models. Reuse the
  existing `fold` assignment only after confirming every cow appears in exactly one outer
  fold and retaining a fold manifest. Do not compare results across different assignments.
- Within each outer training partition, perform hyperparameter selection with three
  `StratifiedGroupKFold` inner folds, grouping on cow. No outer-test labels or scores may
  inform transformations, tuning, thresholds or model selection.
- Fit every learned preprocessing step on the relevant inner/outer training partition only.
  Apply its fitted transformation to validation/test data. Compute deterministic row-wise
  indices using the documented formula; fit any scaler on training rows only. Encode breed
  with the documented mapping. Recompute indices and scaled values from the raw CSV within
  each split; do not use the already globally scaled columns in `data/processed/*.csv` for
  inner-fold evaluation. Never use `Milk_visibility`.
- Use one fixed, preregistered model roster for all three feature sets: majority-class
  reference, Logistic Regression, depth-4 Decision Tree, Random Forest and ANFIS. Add any
  optional estimator only as a separately documented extension.
- For tunable model families, preregister an equal maximum of 20 candidate configurations
  per feature set, with each candidate evaluated on the same three inner folds. The
  majority-class reference has no tuning. Record each search space, candidate list, stopping
  rules, seeds and actual fit count before launching; do not silently exceed the budget.
  The existing single-split defaults are starting evidence, not proof that this budget was
  previously used. Use validation mean ROC-AUC for selection, with a documented tie-break;
  keep the decision threshold at 0.5 for the primary comparison.
- Save every outer-fold prediction to `reports/oof_predictions.csv`, keyed by cow/day/fold
  and including feature set, model, true label, continuous score, threshold, seed and
  configuration fingerprint. Save fold-level metrics and per-fold selected settings to
  `reports/cv_results.csv` and a separate tuning record. Assert zero cow overlap for every
  inner and outer split.
- For every model and feature set, report the same row-level and cow-level metrics:
  accuracy, precision, recall/sensitivity, specificity, F1 and ROC-AUC, plus confusion
  counts at the fixed 0.5 threshold. Aggregate cow scores by the mean daily score and use
  the project's majority-label rule for cow labels. Report fold-wise metrics and pooled
  out-of-fold estimates.
- Quantify uncertainty with a paired cow bootstrap over pooled out-of-fold predictions
  (at least 2,000 resamples; resample cows, preserve all rows and pair predictions across
  models). Report confidence intervals for primary metrics and paired differences. Do not
  treat the five outer folds as independent samples for significance testing.

## Required outputs and interpretation

Store fold definitions, exact configuration/search spaces, package versions, source revision,
dataset hash, run timestamp/status, and commands in the run log. Preserve existing
single-split files; write this experiment to new, clearly named files. No experiment result
is valid until all folds complete and all provenance/overlap assertions pass.

Interpret results as evaluation of current-label clinical mastitis detection on this dataset.
Do not describe them as future-onset prediction, early/subclinical detection, a diagnosis,
or evidence of generalization to other farms.
