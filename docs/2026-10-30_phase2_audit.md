# Phase 2 audit: artifacts, provenance and protocol

Audit date: 2026-10-30 (local time, +05:30). This report records the repository state
observed before adding the Phase 2 governance documents. No training or data-preparation
command was run as part of this audit. The outputs already present are inventoried below;
they were not copied. The preceding session had already run the pipeline and generated
untracked outputs, so this audit cannot recover their earlier contents or timestamp.

Documentation note: this audit and the run log were moved into `docs/` during the
2026-11-01 documentation cleanup; historical paths below describe the locations observed
at the time of the audit.

## Git and repository inventory

- HEAD at audit: `deda8694d6632b03d0a0b8c40e65ce2f53dddd43`.
- Before edits, `git status --short` showed no tracked modifications. It showed these
  untracked paths: `COPILOT_PROJECT_CONTEXT.md`; `data/processed/data_prep_summary.json`;
  both `models/anfis_*.npz`; `reports/anfis_ra_grid.csv`, `anfis_results.csv`,
  `baseline_results.csv`, `comparison.csv`, `example_explanations.txt`,
  `rules_Early-Warning.txt`, `rules_Full-Clinical.txt`,
  `top10_risk_cows_Early-Warning.csv`, `top10_risk_cows_Full-Clinical.csv`; and figures
  `fig1` through `fig10` in `reports/figures/` (fig10 for both feature sets).
- Tracked changes before edits: none. Raw CSVs and processed CSVs are ignored by
  `.gitignore`; they are present locally.
- Raw data: `data/raw/clinical_mastitis_cows.csv` and
  `data/raw/clinical_mastitis_cows_version1.csv`.
- Processed data: `train.csv`, `test.csv`, `full_processed.csv`,
  `data_prep_summary.json`.
- Models: `anfis_Early-Warning.npz`, `anfis_Full-Clinical.npz`.
- Reports: baseline, ANFIS radius/results, comparison, rules, explanations, top-risk lists,
  and figures 1–10 as named above.
- Notebook files: none found. Dashboard source/files: no `app.py` or dashboard found.
- No local `.zip`, `.7z`, `.rar`, or file named for Version 2 was found. A Version 2
  archive is therefore not available in the inspected repository tree.
- Before this audit, `config.json`, `docs/CLAIMS.md`, `reports/RUNLOG.md` and this audit
  directory did not exist.

## Preserved artifact fingerprints

SHA-256 uses the standard SHA-256 digest of each file's bytes. Sizes are bytes. No artifacts
were copied or altered for this inventory.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `data/raw/clinical_mastitis_cows.csv` | 416118 | `e4166166bf83d91a63a62d1ab5e8bd66f9f7ce6687233a67abc4d99918f532d8` |
| `data/raw/clinical_mastitis_cows_version1.csv` | 422719 | `11e1646bda161d234b2be4017ea66b0e1fab54f4dd56ae454eb8ffdb8b718b92` |
| `data/processed/train.csv` | 846052 | `c2b9d5750b8ca07bbad2d130d66d1e71101a66a1684c61151796a6cee45e69b4` |
| `data/processed/test.csv` | 211876 | `e2f01142c06715123e4ea40d040c33714b9e5b49b0b5ed8de0b8cf93742a4259` |
| `data/processed/full_processed.csv` | 1057703 | `1fe65ed736017446ef333399fe1e536873eb7efc65f914421b555a7295919bfb` |
| `data/processed/data_prep_summary.json` | 2335 | `cffb59bd766f37ae6ae340417ebccacd613c2e6a08eaf06d344122d537257bc8` |
| `models/anfis_Early-Warning.npz` | 2012 | `708904dbf5cbb34b322fabaf94a297b5c8c8cc0f77eab892fa118dde2fb3b2ca` |
| `models/anfis_Full-Clinical.npz` | 2844 | `36548a3b0c7031027506ef02393ba2f42255ef501a45da277fbeacd76ab27669` |
| `reports/comparison.csv` | 951 | `640d0ce84f0c82ba37dee7271b23f048fda27049b3394e38f3e53cf768ebeeb4` |
| `reports/baseline_results.csv` | 2377 | `6ae9c136fdb109952e1e1d6b18d9e9657fd68fd51b272eefab0d67f8a5023863` |
| `reports/anfis_results.csv` | 933 | `a2bbe7c5a519921ad357a58b295a580bfcf64fa3f222a0ed8a3f60680ad6398c` |
| `reports/anfis_ra_grid.csv` | 656 | `651e75dcb07347ed627019b802cdf5d24a632d9691fdc1561bbde8c2195a621f` |
| `reports/rules_Early-Warning.txt` | 1511 | `b26d7f82d706319843292cfee87aba0f334b57daa8c5c2b2813ad69a681fcae9` |
| `reports/rules_Full-Clinical.txt` | 2232 | `88272fe0cc22d827cc4ff418024fa4dd1796992d387fb06ab4d2d19531820761` |
| `reports/example_explanations.txt` | 4086 | `27a940e3758e7e225d3805135d163ece8bb2cfb31d463c4558b15dc0d8f05950` |

**Manifest count:** 15 artifact files are fingerprinted in the table above.

Note: these are local fingerprints and do not establish dataset provenance. The Version 1
digest above is recorded from the actual output of `verify_dataset_version.py`; the utility
output printed its own exact digest. The working and Version 1 file hashes differ.

## Dataset provenance comparison

`verify_dataset_version.py` loads both CSVs, strips column-name whitespace, checks shape and
column equality, then aligns by `Cow_ID` and `Day` and compares common column values.
`compare_versions_deep.py` performs a validated one-to-one cow/day merge and summarizes
label transitions and per-column differences. The runs reported:

- Both files contain 6,600 rows and 18 columns with 1,100 cows and six rows per cow.
- Working file SHA-256: `e4166166bf83d91a63a62d1ab5e8bd66f9f7ce6687233a67abc4d99918f532d8`,
  size 416,118 bytes; label counts 3,961 class 0 and 2,639 class 1.
- Version 1 SHA-256:
  `11e1646bda161d234b2be4017ea66b0e1fab54f4dd56ae454eb8ffdb8b718b92`,
  size 422,719 bytes; label counts 5,490 class 0 and 1,110 class 1.
- Aligned comparison: 1,529 labels differ, all from 0 in Version 1 to 1 in the working
  file; 16,152 differing cells across the common columns. 254 cows had all six labels
  changed, and cow10 had five changed rows.
- No authentic Version 2 archive or extracted CSV was located. **Conclusion:** the working
  CSV is demonstrably not the available Version 1 CSV; whether it is the official Version 2
  CSV remains unresolved. The exact artifact needed is the authentic
  `Clinical_Mastitis_cows_version2.zip` obtained from the official Version 2 dataset record,
  plus provenance for that download. Do not infer identity from shape/counts.

## Anomalous-cow audit

`python audit_odd_cow.py data/raw/clinical_mastitis_cows.csv` identified `cow10`. A
read-only raw-data inspection found the sequence `class1 = 1,1,1,1,1,0` for Days 1–6.
The six records have temperatures 42, 43, 43, 43, 43, 43; Hardness 1, 1, 1, 1, 1, 0;
Pain is 0 on all days. Breed is Jersey and months after birth is 1 throughout. The eight
sensor readings for Days 1–5 are:

| Day | IUFL | EUFL | IUFR | EUFR | IURL | EURL | IURR | EURR | Milk_visibility | class1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 230 | 270 | 230 | 271 | 230 | 271 | 230 | 272 | 1 | 1 |
| 2 | 235 | 275 | 232 | 275 | 230 | 277 | 236 | 277 | 1 | 1 |
| 3 | 240 | 276 | 236 | 275 | 231 | 280 | 240 | 278 | 1 | 1 |
| 4 | 241 | 277 | 231 | 278 | 236 | 280 | 240 | 280 | 1 | 1 |
| 5 | 242 | 280 | 238 | 280 | 240 | 288 | 242 | 281 | 1 | 1 |
| 6 | 152 | 187 | 157 | 188 | 155 | 188 | 152 | 186 | 0 | 0 |

There are zero exact duplicate rows and zero duplicate `Cow_ID` + `Day` keys in the working
CSV. This rules out those two simple duplication anomalies, not broader measurement or
labeling errors. The last row's sensor readings, Hardness and target all change together;
the file alone cannot establish whether this is a genuine status change, a source correction
or a recording issue. `class1` agrees with `Milk_visibility` on these rows, but this is not
independent validation: those columns duplicate each other in the working CSV, and
`Milk_visibility` is the leaked target copy. Do not infer from this agreement that cow10's
label is correct or incorrect. Keep the cow and labels unchanged. A separately logged
sensitivity analysis could assess influence, but must not replace the primary result.

## Evaluation protocol audit

- Target/group: source code uses processed `label` (from raw `class1`) and `cow_id` (from
  `Cow_ID`) respectively. `Milk_visibility` is audited and dropped; `prev_mastitis` is not
  included in either defined feature set.
- Current feature order in `data_prep.py`:
  - Early-Warning / sensor-only: `swelling_index`, `asymmetry_index`, `temperature_n`,
    `months_pp_n`, `breed_enc`.
  - Full-Clinical: those five in the same order, then `hardness`, `pain`.
- Preparation uses `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`;
  fold 0 is test. Saved summary reports 5,280 training rows / 880 cows and 1,320 test rows
  / 220 cows, zero overlap. The summary records train/test label fractions 0.3998 / 0.4.
- `scale_features` fits `MinMaxScaler` on rows whose fold is not the held-out test fold,
  then transforms and clips outside-unit-interval values. The saved summary records seven
  test values clipped. The engineered index formula is deterministic row-level arithmetic.
- `baselines.py` fits the listed fixed estimators on `train.csv` and scores `test.csv`.
  Its listed settings are majority prior; Logistic Regression `max_iter=1000`, seed 42;
  depth-4 tree, seed 42; and 300-tree Random Forest, seed 42, `n_jobs=-1`. It has no
  hyperparameter search. It consumes preprocessed saved files and does not refit a
  preprocessing pipeline itself.
- `train_anfis.py` reserves fold 1 from the fold-0-excluded training rows for validation.
  It evaluates radii 0.3, 0.4, 0.5, 0.6, 0.7 and chooses the highest validation row AUC.
  Subtractive clustering uses training inputs and labels to initialize premises, is capped
  at 12 rules, and the model uses up to 100 epochs with learning rate 0.01. It refits on
  all fold-0-excluded training rows using the selected radius/epoch count. The held-out
  fold-0 test data are scored after selection; source shows no test-based tuning.
- `metrics.py` uses threshold 0.5. Row accuracy, precision, recall/sensitivity, specificity
  (`TN/(TN+FP)`), F1, and ROC-AUC are computed on row predictions/scores. Cow metrics first
  average scores per cow and assign the cow label as positive when the mean row label is at
  least 0.5. Current confidence intervals are 1,000 whole-cow bootstrap resamples for
  **row-level** F1 and AUC, with seed 42; they are not intervals for cow-level metrics.
- The final saved ANFIS rule counts are taken from `n_rules` in `reports/anfis_results.csv`
  and are consistent with model artifacts: 10 Early-Warning rules and 12 Full-Clinical
  rules. The radius-validation grid's `n_rules` is the candidate model count before the
  final refit.
- Current saved test metrics are single-split preliminary results, not nested CV. No
  training command was executed in this Phase 2 audit.

## Next experiment plan

See `docs/EXPERIMENT_SPEC_PHASE3.md` for the proposed three-way feature-set comparison,
identical grouped folds, training-fold-only preprocessing, tuning budget, OOF predictions,
fold metrics and paired cow-bootstrap analysis. It is a specification only; it has not run.

## Files and commands

Required project docs inspected: `docs/COPILOT_PROJECT_CONTEXT.md`, `README.md`, `docs/WORKFLOW.md`,
`docs/HANDOFF.md`, `docs/PROJECT_STATE_AND_ROADMAP.md`, `docs/TASKS.md`, `docs/JOURNAL.md`, and
`docs/LITERATURE.md`. Relevant source reviewed: `data_prep.py`, `baselines.py`, `train_anfis.py`,
`metrics.py`, `audit_odd_cow.py`, `verify_dataset_version.py`,
`compare_versions_deep.py`; relevant current outputs read: processed summary, baseline,
ANFIS, comparison and radius-grid CSVs.

Commands executed during this Phase 2 audit (no network access; no training or report
generation):

- `git status --short; git diff --stat; git ls-files` — exit 0; captured pre-edit status.
- Read-only recursive file inventory and SHA-256 calculation — completed.
- `python audit_odd_cow.py data/raw/clinical_mastitis_cows.csv` — exit 0.
- Read-only Python duplicate and full cow10-row inspection — exit 0.
- `git rev-parse HEAD` and Python/package version query — exit 0.
- `python verify_dataset_version.py data/raw/clinical_mastitis_cows.csv data/raw/clinical_mastitis_cows_version1.csv` — exit 0.
- `python compare_versions_deep.py data/raw/clinical_mastitis_cows_version1.csv data/raw/clinical_mastitis_cows.csv` — exit 0.

The earlier pipeline command outputs are recorded retrospectively in `docs/RUNLOG.md`
(originally created under `reports/`).
No model, split, label or existing result artifact was changed as part of this audit.
