# COPILOT PROJECT CONTEXT — INTERPRETABLE ANFIS FOR BOVINE CLINICAL MASTITIS DETECTION
**Purpose:** Give GitHub Copilot in VS Code a durable, detailed understanding of this project's objective, current codebase, data, methodology, evidence, constraints, user expectations, and implementation roadmap.

**How to use this file**
1. Keep this file at `docs/COPILOT_PROJECT_CONTEXT.md` in the `mastitis-anfis` repository.
2. Open the repository root in VS Code.
3. Open Copilot Chat in Agent mode if available.
4. First ask Copilot to read this file plus `README.md`, `docs/HANDOFF.md`, `docs/WORKFLOW.md`, `docs/PROJECT_STATE_AND_ROADMAP.md`, `docs/TASKS.md`, `docs/JOURNAL.md`, `docs/LITERATURE.md`, and `0 Evaluation Criteria.docx` if that rubric file is present locally.
5. Ask Copilot to audit and report before asking it to edit code. Then proceed with the phased prompts near the end of this document.
6. Do not paste secrets or commit sensitive data. Keep raw data and local model artifacts out of public repositories unless there is a clear, permitted reason to share them.

---

# 1. Project identity and purpose

## 1.1 Working title
**Interpretable Adaptive Neuro-Fuzzy Inference System (ANFIS) for Bovine Clinical Mastitis Detection Using Udder-Related Measurements**

The repository README currently says “Interpretable ANFIS for Early Bovine Mastitis Detection.” For scientific reporting and the dashboard, prefer **clinical mastitis detection** and the feature-set name **Sensor-only**. Avoid “early-warning” unless a future-onset prediction task is specifically defined and validated.

## 1.2 Academic context
- Course/project area: Soft Computing.
- Programme context: Integrated M.Tech Software Engineering, VIT-AP University.
- Problem statement reference recorded in README: `SIH26109`.
- Main goal: build and evaluate an explainable ANFIS classifier for the dataset's existing clinical mastitis label, compare it with conventional ML baselines, and expose the model and its explanations through a research-prototype UI.
- Expected outputs: reproducible Python code, trained artifacts, evaluation reports, fuzzy-rule explanations, a Streamlit dashboard, research article/report, presentation and viva preparation.

## 1.3 Problem statement
Investigate whether udder-related sensor readings and optional clinical observations can classify the dataset's existing cow-level clinical mastitis status. The project specifically investigates a neuro-fuzzy method because it combines data-driven learning with human-readable fuzzy rules.

This is **not currently a validated veterinary diagnostic product**. It is not a validated future-onset forecasting model, and the project has not established subclinical mastitis detection or clinical early-warning capability. All UI and documentation must say so clearly.

## 1.4 Objectives
1. Audit the supplied dataset, target, features and provenance.
2. Prevent target leakage and cow-level train/test contamination.
3. Engineer and document a small number of udder-related features.
4. Implement subtractive clustering to derive fuzzy rules.
5. Implement first-order Sugeno-style ANFIS and hybrid learning.
6. Compare ANFIS with appropriate baseline classifiers.
7. Evaluate using a defensible cow-grouped protocol.
8. Explain predictions with the actual learned rules and membership degrees where supported.
9. Build a Streamlit research dashboard using the **same model and preprocessing pipeline** as the evaluated experiments.
10. Record all experiments and support reproducibility.
11. Prepare a defensible research article and presentation.

## 1.5 What success means
Success is not simply a high metric. It means:
- the data source/version is documented;
- the target and feature pipeline are correct;
- no target leakage is present;
- cows do not cross between training and test sets;
- hyperparameters are selected without using the final held-out test data;
- results can be reproduced from source;
- the app uses the same model and preprocessing as the reported experiment;
- explanations are traceable to real model rules;
- limitations and differences from prior work are stated honestly.

---

# 2. Critical scientific rules

Copilot must follow these rules in every code, notebook, UI, article, and slide change.

1. **Never invent metrics, citations, dataset details, model files, clinical thresholds, or explanations.**
2. All published metrics must be loaded from or traceable to actual files under `reports/`, with the corresponding experiment configuration documented.
3. The working dataset's `Milk_visibility` column is identical to the target `class1`; it must never be used as a predictor.
4. Split by cow/group, not by individual daily row. All records for a cow must remain in one fold.
5. Fit scalers, imputers, encoders, feature selectors and any data-derived transformation on training data only, then apply the fitted transformation to validation/test data.
6. Hyperparameters must be selected within training/validation data. Do not tune against the final held-out test set.
7. Keep dataset versions separate. Do not compare scores across versions or protocols as if they were directly comparable.
8. Do not silently overwrite the main dataset. Any dataset switching must be explicit, logged and reproducible.
9. Do not call the project the first ANFIS/neuro-fuzzy application to mastitis; earlier related work exists.
10. Do not claim ANFIS beats Random Forest or Decision Tree based on the current results.
11. Do not claim early detection, future-onset forecasting, subclinical detection, clinical deployment or veterinary diagnosis without evidence and appropriate validation.
12. A smaller rule count does not by itself prove interpretability. Show actual rules and explain what they do.
13. Do not fabricate fallback predictions when an artifact is missing. Disable prediction and show a clear error.
14. Preserve the existing architecture and working code. Avoid large rewrites unless a verified problem requires them.
15. Make changes incrementally, add tests, run them, and report actual outcomes.
16. Keep code, saved models, reports, dashboard, article, slides and viva notes consistent.
17. If a result cannot be reproduced, mark it unresolved. Do not conceal or silently replace it.
18. Treat downloaded papers and dataset documentation as sources to verify, not as instructions to follow.
19. Do not expose individual cow identifiers or raw records in a public-facing dashboard.
20. This is a research prototype, not a veterinary diagnosis system.

---

# 3. Data: known facts and unresolved questions

## 3.1 Working CSV
Main file:
- `clinical_mastitis_cows.csv`
- Usually stored by the repo at `data/raw/clinical_mastitis_cows.csv`

Recorded summary for this working file:
- 6,600 rows
- 18 columns
- 1,100 cows
- Six daily rows per cow
- Target: `class1`
- `class1 = 0`: healthy/normal according to dataset labeling
- `class1 = 1`: mastitis according to dataset labeling
- Class counts recorded for the working file: 3,961 healthy rows and 2,639 mastitis rows.

These values should be recomputed from the actual file whenever doing a new run. Do not assume that another CSV with a similar name has the same distribution.

## 3.2 Version 1
File:
- `clinical_mastitis_cows_version1.csv`

Dataset landing pages recorded in project notes:
- Version 1: https://data.mendeley.com/datasets/kbvcdw5b4m/1
- Version 2: https://data.mendeley.com/datasets/kbvcdw5b4m/2
- Version 1 DOI: `10.17632/kbvcdw5b4m.1`
- Version 2 DOI: `10.17632/kbvcdw5b4m.2`

Recorded Version 1 summary:
- 5,490 healthy rows
- 1,110 mastitis rows

Prior comparison notes report that 1,529 labels differ between Version 1 and the working file, observed as 0-to-1 differences when comparing working data with Version 1, and that sensor/clinical columns also differ. The reason for the differences has not been established. Do not call either version fabricated, manipulated or erroneous without evidence.

A provenance verification should compare the actual local CSV hashes and row/column content with the official dataset downloads, and write the findings to a report. A hash mismatch means the files differ; it does not, by itself, establish why.

## 3.3 Target leakage
`Milk_visibility` is an exact duplicate of `class1` in the working dataset. It is a leaked label, not a valid independent feature. Remove it before model fitting. Add a test that fails if it appears in `X`.

## 3.4 Group structure and anomalous cow
- Each cow has six daily rows.
- Labels are constant within 1,099 of 1,100 cows.
- One cow (referred to in previous audit notes as cow 10) has a label that changes on Day 6.
- The anomaly must be investigated and documented. Do not silently change its label or delete the cow. If a sensitivity analysis excludes it, retain the primary analysis and explain the alternative.

## 3.5 Possible meaning of sensor columns
Project notes say `I*` and `E*` likely represent inhale/exhale limits for udder-quarter readings, but this exact interpretation should be checked against primary dataset documentation or the original paper before being stated as definitive.

## 3.6 Dataset source and provenance
Prior file fingerprints recorded in the audit notes:
- Working CSV: 416,118 bytes; SHA-256 `e4166166bf83d91a63a62d1ab5e8bd66f9f7ce6687233a67abc4d99918f532d8`
- Version 1 CSV: 422,719 bytes; SHA-256 `11e1646bda161d234b2be4017ea66b0e1fab54f4dd56ae454eb8b718b92`

**Important:** the second fingerprint must be checked against the actual file; do not use copied fingerprints as truth if a new local file produces a different hash. Always recompute and record the current values. The working CSV's match to a specific official Version 2 archive was pending in earlier notes.

## 3.7 Dataset handling requirements
- Preserve raw files unchanged.
- Put processed outputs under `data/processed/`.
- Use explicit paths and never overwrite a source CSV in place.
- Add a provenance report with file name, byte size, SHA-256, shape, target counts, column names, and comparison results.
- Document whether a run uses the working CSV or Version 1.
- Every report should record dataset version/hash and split configuration.

---

# 4. Feature engineering and feature sets

The current data-preparation pipeline engineers two features.

## 4.1 Udder Swelling Index
Recorded formula: mean of eight udder-related sensor readings. Verify the exact input column names and formula in `data_prep.py`; do not infer them from column prefixes alone.

## 4.2 Udder Asymmetry Index
Recorded formula: a measure of differences across four udder quarters, currently based on the maximum and minimum quarter readings relative to their average. Verify the exact implementation in `data_prep.py` and document it mathematically.

This is a project-engineered feature, not an established veterinary diagnostic threshold. Future work should compare it with a coefficient-of-variation alternative or another justified asymmetry measure using the same grouped evaluation protocol.

## 4.3 Current feature sets
The repository internally names the sensor-only set “Early-Warning.” This is a legacy name and should be changed in UI and scientific presentation text to **Sensor-only**. If renaming a model artifact would break code, implement a backward-compatible mapping and document it.

### Sensor-only
Recorded features:
- `swelling_index`
- `asymmetry_index`
- `temperature_n`
- `months_pp_n`
- `breed_enc`

This is five features.

### Full-Clinical
The Sensor-only features plus:
- `hardness`
- `pain`

This is seven features.

Verify the exact transformations, encoding categories, scaling, and order from `data_prep.py`, processed CSVs, saved artifacts, and the training scripts. The dashboard must use exactly the same feature order and transformations.

## 4.4 Raw sensor feature set
A subsequent audit reported a promising lead:
- Random Forest using the raw eight sensors reportedly achieved F1 around 0.983 under cow-grouped cross-validation, versus around 0.914 using engineered indices.
- ANFIS on raw sensors reportedly achieved ROC-AUC around 0.98 with about four rules, versus around 0.92 with ten rules on the engineered Sensor-only set.

These are **reported exploratory findings only** until the corresponding code, folds, predictions and outputs are located and reproduced. Do not put them into the main results table as final results yet. The next experimental priority is a fair comparison of raw sensors vs engineered features under identical grouped folds and comparable tuning budgets.

---

# 5. Train/test design and leakage controls

The current preparation notes specify:
- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
- Fold 0 used as a held-out test fold in the current headline reports.
- Training: 5,280 rows from 880 cows.
- Test: 1,320 rows from 220 cows.
- Zero cow overlap.
- Scaler fitted on training data only.
- Reported mastitis proportions: approximately 0.3941 in training and 0.4227 in test.

Recompute these from the actual source and generated files before reporting. The final study should use all five grouped folds, not only fold 0.

## 5.1 Required checks
1. Assert no cow ID occurs in both train and test.
2. Assert the target is not in the feature matrix.
3. Assert `Milk_visibility` is not in the feature matrix.
4. Assert every cow is assigned to one fold only.
5. Confirm scaler/imputer/encoder fit only on training data.
6. Confirm class distribution and group counts per fold.
7. Save fold assignments and run metadata.
8. Tune model settings only within training/validation data.
9. Compute final metrics from held-out predictions, not training predictions.
10. Save per-fold results, pooled out-of-fold predictions and configuration metadata.

## 5.2 Planned final evaluation
- Stratified group 5-fold CV, grouping on cow ID.
- Compare the same model families and feature sets across identical folds.
- Report mean and standard deviation across folds, plus suitable uncertainty intervals.
- Consider a cow-level bootstrap confidence interval that resamples cows, not individual daily rows.
- Use appropriate paired statistical comparisons. For binary predictions on the same independent test units, McNemar's test may be suitable; ensure unit-of-analysis assumptions are respected. Do not treat five folds as five independent samples for an inappropriate significance test.
- Report F1, ROC-AUC, precision, recall, specificity, accuracy, confusion matrices, and any other metrics only if correctly computed and relevant.
- Document decision thresholds and how they were selected.
- Include class prevalence and a simple baseline where useful.

---

# 6. ANFIS and soft-computing method

## 6.1 Why ANFIS
ANFIS combines fuzzy inference with adaptive learning. Fuzzy rules and membership functions can make parts of the decision process inspectable, while learning estimates parameters from data. It is relevant to a Soft Computing project because it combines approximate reasoning and data-driven optimization.

Do not argue that interpretability automatically makes ANFIS more accurate or more useful than every baseline. Evaluate the performance/interpretability trade-off.

## 6.2 Subtractive clustering
`subtractive_clustering.py` identifies representative data-space regions. Those cluster centers are used to initialize fuzzy rules and membership functions.

Explain:
- what a cluster center represents;
- how the radius parameter controls neighborhood influence and rule count;
- the relationship between clustering and initial rule construction;
- any normalization and numerical safeguards;
- how rule counts change with radius;
- what is and is not guaranteed by the method.

The clustering radius must be selected using training/validation data, not the held-out test set.

## 6.3 ANFIS model
`anfis.py` contains the ANFIS model implementation. Inspect its actual API and equations before modifying it. The intended structure is a first-order Sugeno-style fuzzy inference system:
1. Inputs are mapped to membership degrees.
2. Rule firing strengths are calculated.
3. Firing strengths are normalized.
4. Rule consequents produce outputs.
5. Normalized rule contributions are combined into the final output.

Confirm the exact membership function family and parameterization in the code; do not assume a Gaussian or bell shape without checking.

## 6.4 Hybrid learning
The implementation is described as using hybrid learning:
- consequent parameters are estimated using least squares;
- premise/membership parameters are updated with gradient-based learning.

Verify the training loop, loss, gradient formulas, parameter constraints, convergence behavior and numerical stability. The finite-difference gradient test is important and must remain.

## 6.5 Output interpretation
Verify whether the ANFIS output is a continuous score, a probability-like value, or a raw regression output. Do not label it a calibrated probability unless calibration has been demonstrated. Determine the threshold from the documented experiment. The UI must display exactly the interpretation supported by the training/evaluation code.

## 6.6 Explanation generation
`explain.py` and rule reports should be used to expose actual rules, membership degrees and rule contributions. Do not generate plausible-sounding natural-language explanations unless they can be traced to the model's actual computations.

A readable explanation should distinguish:
- input values;
- membership degrees;
- fired rule(s);
- normalized firing strengths;
- consequent values/contributions if available;
- final model output and threshold used.

---

# 7. Baselines and preliminary reported results

The repository has baseline code in `baselines.py` and metrics code in `metrics.py`. Current comparison models include:
- Logistic Regression
- Decision Tree
- Random Forest
- ANFIS

Check whether a majority-class baseline is implemented or only proposed. Do not claim it was evaluated unless an output confirms it.

The currently reported headline results from `reports/comparison.csv` are:

| Feature set | Model | F1-score | ROC-AUC | Rule count |
|---|---|---:|---:|---:|
| Sensor-only | ANFIS | 0.8694 | 0.9169 | 10 |
| Sensor-only | Random Forest | 0.9301 | 0.9834 | N/A |
| Sensor-only | Decision Tree | 0.8987 | 0.9470 | N/A |
| Full-Clinical | ANFIS | 0.9076 | 0.9380 | 12 |
| Full-Clinical | Random Forest | 0.9277 | 0.9761 | N/A |
| Full-Clinical | Decision Tree | 0.9211 | 0.9455 | N/A |

These are **preliminary held-out-fold results**, not final 5-fold CV results. Re-open the actual CSV and verify exact rows before using these values in any deliverable.

Interpretation:
- Adding clinical features improves ANFIS F1-score in the current reported experiment.
- Random Forest outperforms ANFIS in the current comparison.
- The project should explore whether ANFIS offers a useful rule-level interpretability trade-off, not claim it is the most accurate model.
- Do not overstate the significance of a single fold.

## 7.1 Notebook cell 64 and cell 107 issue
The supplied notebook `ANFIS_PROJECT_MASTITIS_FORECASTING.ipynb` was audited in a previous pass. The reported issue:
- Cell 64 copies Version 1 over the working CSV, so subsequent cells use Version 1.
- Cell 107 creates a comparison table with manually entered rounded mean values rather than computing them in that cell.
- The audit reported that recomputing the Version 1 results matched the rounded values, but they are not directly comparable with the repository's working-dataset, held-out-fold table.

Do not delete or conceal this. Make the notebook explicit about dataset version and experiment protocol. Prefer separate versioned input paths and saved outputs; never overwrite the working CSV as a side effect of running a notebook.

## 7.2 Leakage replication result
A previous audit reported that `leakage_replication.py` produced a Random Forest accuracy difference of roughly one percentage point (about 0.993 vs 0.984) between a paper-style recipe and a cow-grouped protocol. This is a reported audit finding; verify the script, settings and output before reusing the values. The evidence does **not** support saying that all 99% results are caused by leakage or accusing paper authors of misconduct.

---

# 8. Existing codebase: file-by-file guide

The repository root is named `mastitis-anfis` in the archive. The actual files present in the supplied ZIP include the following.

## Main modules
- `data_prep.py`: loads raw CSV, checks columns, removes leakage, engineers indices, performs group-aware split, fits train-only preprocessing, writes processed files and summary metadata. Verify each step.
- `baselines.py`: trains baseline classifiers and writes results. Check exact models, hyperparameters, split reuse and metrics.
- `metrics.py`: metrics and related evaluation helpers. Verify unit-of-analysis, threshold handling and confidence interval implementation.
- `subtractive_clustering.py`: clustering logic used to initialize ANFIS rules.
- `anfis.py`: ANFIS model, inference and learning code.
- `explain.py`: rule extraction and per-example explanation generation.
- `train_anfis.py`: training, validation-based clustering-radius selection, final fit, evaluation, model saving, rules and explanation outputs. Verify that test data is not used for model selection.
- `test_modules.py`: four unit tests for clustering, analytic gradients, least-squares fitting and hybrid learning on XOR.
- `verify_dataset_version.py`: dataset/version verification helper; inspect expected CLI arguments and ensure it does not overwrite files.
- `compare_versions_deep.py`: detailed comparison between CSV versions.
- `audit_odd_cow.py`: investigate the cow whose label changes across days.
- `leakage_replication.py`: compare paper-style row-level protocol and grouped protocol; verify exact behavior and ensure outputs are labelled correctly.

## Project documentation (`docs/`)
- `README.md` (repository root): setup, quick start, pipeline and team workflow.
- `docs/WORKFLOW.md`: project operating and validation instructions.
- `docs/TASKS.md`: task assignments and Review II/Final Review plan; dates in it may be stale.
- `docs/JOURNAL.md`: research log. Add an entry for each meaningful work session.
- `docs/PROJECT_STATE_AND_ROADMAP.md`: current state, decisions and future plan.
- `docs/LITERATURE.md`: literature survey notes; verify citations against original papers before final use.
- `docs/HANDOFF.md`: design decisions, open issues and collaborator context.
- Other project notes, reports, audit records and run logs are kept in `docs/`.
- `.github/pull_request_template.md`: contribution workflow template.
- `.gitignore`: files intended to stay out of Git.

## Data and reports
- `data/raw/clinical_mastitis_cows.csv`
- `data/raw/clinical_mastitis_cows_version1.csv`
- `data/processed/full_processed.csv`
- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/processed/data_prep_summary.json`
- `models/anfis_Early-Warning.npz` (legacy model name; verify metadata and feature order)
- `models/anfis_Full-Clinical.npz`
- `reports/anfis_ra_grid.csv`
- `reports/anfis_results.csv`
- `reports/baseline_results.csv`
- `reports/comparison.csv`
- `reports/example_explanations.txt`
- `reports/rules_Early-Warning.txt`
- `reports/rules_Full-Clinical.txt`
- `reports/top10_risk_cows_Early-Warning.csv`
- `reports/top10_risk_cows_Full-Clinical.csv`
- `reports/figures/fig1_class_distribution.png`
- `reports/figures/fig2_leakage_milk_visibility.png`
- `reports/figures/fig3_feature_label_correlation.png`
- `reports/figures/fig4_index_distributions.png`
- `reports/figures/fig5_swelling_vs_asymmetry.png`
- `reports/figures/fig6_split_sanity.png`
- `reports/figures/fig7_feature_correlation_heatmap.png`
- `reports/figures/fig8_baseline_roc.png`
- `reports/figures/fig9_anfis_learning_curve.png`
- `reports/figures/fig10_membership_Early-Warning.png`
- `reports/figures/fig10_membership_Full-Clinical.png`

Some generated artifacts contain cow-level examples or identifiers. Do not publish those raw examples in the public dashboard. Review privacy and data-sharing rules before committing large raw datasets or personal/animal identifiers.

## Dependencies
The supplied `requirements.txt` currently contains:
- `numpy>=1.23`
- `pandas>=1.5`
- `scikit-learn>=1.1`
- `matplotlib>=3.5`
- Streamlit is commented as a future demo dependency.

If the app is added, check actual Python compatibility and add the minimum needed dependencies. Do not add a library unless it is needed and tested. Record tested Python version and dependency versions.

## Git and workflow
README says:
- work on a branch `feat/<what>-<name>`;
- do not commit straight to `main`;
- the other teammate reviews and merges;
- raw data should not be committed;
- all reported numbers must be traceable to report files.
Follow this workflow if the local repo is under Git.

---

# 9. Research literature and contribution framing

The project has PDFs and `docs/LITERATURE.md` for:
- machine-learning mastitis detection;
- automatic milking system sensor data;
- prediction/forecasting studies;
- clinical and subclinical mastitis work;
- explainability methods;
- ANFIS and fuzzy systems.

Literature points previously noted include:
- MasPA: Ghafoor & Sitkowska (2021), DOI `10.3390/agriengineering3030037`.
- A 2023 paper associated with DOI `10.52865/DQFZ6016`.
- Jang (1993) as a foundational ANFIS reference.
- Earlier mastitis-related neuro-fuzzy studies attributed in notes to Mammadova & Keskin (2015) and Kohli (2011).
- A possible companion paper titled “Comparison of KNN and SVM Algorithms to Detect Clinical Mastitis in Cows Using Internet of Animal Health Things,” associated in notes with DOI `10.1007/978-981-15-5679-1_6`; its exact details and relation to this dataset must be verified in the original publication.

Do not use this list as a final bibliography without checking the original papers. Confirm title, authors, publication venue/year, DOI, dataset, split strategy and metrics.

## Recommended cautious gap statement
“Within the literature reviewed, this project investigates subtractive-clustering ANFIS for udder-measurement-based clinical mastitis classification under cow-level validation, with explicit inspection of learned fuzzy-rule explanations and comparison against conventional machine-learning baselines.”

This is deliberately qualified. Do not say “first,” “novel worldwide,” or “proven superior” without a defensible prior-art review.

## Article artifact
At minimum, develop:
1. Title.
2. Abstract with only verified results.
3. Introduction/problem definition.
4. Related work and carefully worded gap.
5. Dataset and provenance.
6. Preprocessing and leakage controls.
7. Feature engineering and feature-set definitions.
8. ANFIS and clustering methodology.
9. Experimental protocol and baselines.
10. Results tables and figures generated from reports.
11. Discussion including why RF currently performs better.
12. Limitations and future work.
13. References checked against original papers.

Potential venues must not be invented or portrayed as selected without team agreement. Identify a plausible target venue and confirm its scope and author instructions before final submission.

---

# 10. Streamlit application requirements

A separate Streamlit application was created by a teammate. The available exported webpage and asset ZIP are not the same as the original Python source. The teammate's actual `app.py`, model artifacts and preprocessing dependencies must be obtained or located before claiming the UI is integrated.

Previous audit notes state:
- the teammate's model files contain eight rules and a scaler;
- those artifacts differ from the repository's current model artifacts (10 rules for Sensor-only and 12 for Full-Clinical in the current reported run);
- a previous audit reported the teammate's sensor-only model F1 as 0.695 on the working-file test fold versus 0.869 for the repository model;
- the teammate's app feature formulas were reported to match its own training formulas;
- the app's temperature default was 38.5 while the working dataset temperature values were recorded as ranging from 24 to 58;
- “Early-Warning” wording remained in the UI.
These are prior audit claims and must be confirmed from the actual source and artifacts. The rule-count difference alone does not establish which dataset trained the teammate's model.

## 10.1 Dashboard sections
1. Overview: purpose, selected feature set, artifact version, validation status and brief project summary.
2. Prediction workspace: Sensor-only or Full-Clinical inputs, input validation, inference result and clear research-prototype warning.
3. Explainable ANFIS: actual rules, membership degrees, firing strengths and rule contributions where the implementation supports them.
4. Model evaluation: load metrics from `reports/`; never hard-code metrics.
5. About/limitations: dataset source, feature definitions, validation method and caveats.

## 10.2 UI integrity
- Do not claim the UI is a clinical diagnostic device.
- Do not show an “early warning” result for a classifier trained to detect current labels.
- Do not call raw ANFIS outputs calibrated probabilities unless calibration is evaluated.
- Do not create fabricated explanations or dummy predictions.
- Never silently use a fallback model if the correct artifact is missing.
- Use the same feature order and preprocessing used during training.
- Add a smoke test with known input vectors and compare app predictions against direct model inference.
- Display the selected model/configuration and validation status.
- Avoid exposing individual cow identifiers or private raw records.
- If the app cannot load the correct artifacts, show an error and disable inference.

## 10.3 App launch
The intended local command is:
```bash
streamlit run app.py
```
This command is valid only after `app.py` and dependencies exist. Do not claim it has been tested unless it has actually launched successfully.

---

# 11. Evaluation rubric: use as a project checklist

The supplied `0 Evaluation Criteria.docx` states:

## Review II — 30 marks
- Literature survey and refinement of research gap: 3
- Detailed methodology/system design: 4
- Implementation progress: 8
- Dataset/data preparation: 3
- Preliminary results and analysis: 4
- Article/patent/startup proposal development: 4
- Presentation and viva: 2
- Soft Computing technique justification: 2

Review II expects preferably 50–70% implementation completion and preliminary experimental results.

The “outcome” evidence guidance lists:
- Research article: title, abstract, methodology, preliminary results, target journal/conference.
- Patent/IPR: novel idea, inventive aspect, prior-art search, proposed claims.
- Startup proposal: problem, target users, proposed product, USP, market/application.
- Prototype/product: working prototype, application, TRL/proof of concept.

## Final Review — 40 marks
- Problem formulation and achievement of objectives: 3
- Methodology and technical correctness: 4
- Complete implementation: 8
- Experimental evaluation: 7
- Results, comparison and discussion: 5
- Application/novelty and effective use of Soft Computing: 5
- Presentation and demonstration: 3
- Viva-voce: 5

## Project report — 20 marks
- Abstract, introduction and problem definition: 2
- Literature survey and research gap: 3
- Proposed methodology/system design: 3
- Implementation: 2
- Results and analysis: 4
- Discussion, limitations and future scope: 2
- Article/patent/startup proposal documentation: 2
- References and citation quality: 1
- Organization, formatting and presentation: 1

Use this rubric to track deliverables. Do not invent marks or state that faculty has approved anything.

---

# 12. Planned work: priority order

## Phase 0 — Audit before edits
1. Read the existing docs and inspect actual source files.
2. Create a clean status report: confirmed, preliminary, unresolved, next actions.
3. Check Git status and create a branch before modifying anything.
4. Confirm data paths and current artifact hashes.
5. Run `python test_modules.py`.
6. Re-run data preparation and compare outputs to existing reports.
7. Inspect the notebook for dataset overwrite and manually entered metrics.
8. Verify the app source and model artifact provenance if available.

## Phase 1 — Reproducible baseline
1. Run data preparation.
2. Verify no cow overlap and no `Milk_visibility` feature.
3. Run baseline models.
4. Train the current ANFIS models.
5. Compare new outputs with existing `reports/comparison.csv`.
6. Record differences; never overwrite old results without preserving an experiment record.
7. Save run metadata, random seeds, package versions, dataset hash and configuration.

## Phase 2 — Feature-set study
Compare:
- raw eight sensor inputs;
- engineered Sensor-only features;
- Full-Clinical features.

Use identical grouped folds, consistent baselines and a clearly documented tuning budget. Do not rely on the reported exploratory raw-sensor results until reproduced.

## Phase 3 — Final validation
1. Grouped 5-fold cross-validation.
2. Store out-of-fold predictions and fold metrics.
3. Calculate confidence intervals using a method consistent with the grouping.
4. Perform paired comparisons on the same evaluation units where justified.
5. Report thresholds, confusion matrices and class balance.
6. Include a leakage/provenance analysis as a separate experiment.
7. Clearly distinguish exploratory results from final results.

## Phase 4 — Tuning and ablations
1. Establish a simple search baseline (grid or random search).
2. Explore GA tuning for subtractive-clustering radius parameters only after the evaluation pipeline is stable.
3. Compare with the baseline search using the same validation protocol.
4. Ablate the Asymmetry Index and compare with an alternative such as coefficient of variation.
5. Analyze rule count, rule stability, interpretability and robustness.
6. Do not add GA just to claim the use of an optimizer; show that the experiment is fair and useful.

## Phase 5 — App integration
1. Obtain actual app source.
2. Trace model loading, preprocessing and inference.
3. Match artifacts to the selected final model.
4. Test known examples against direct inference.
5. Build a clean, understandable Streamlit UI.
6. Read evaluation values from report files.
7. Add app smoke tests and error handling.
8. Record limitations.

## Phase 6 — Research deliverables
1. Complete article draft.
2. Create system architecture and data-flow diagrams.
3. Generate figures and tables directly from reports.
4. Create a slide deck aligned with the rubric.
5. Prepare speaking script and viva answers.
6. Create a final report with limitations and future work.
7. Keep all claims synchronized with the final results.

---

# 13. Copilot working protocol

Copilot should work as a careful implementation collaborator, not an autonomous code generator that rewrites everything.

For each task:
1. State what it inspected.
2. Identify the file(s) to change and why.
3. Propose the smallest useful change.
4. Implement only that change.
5. Add or update tests.
6. Run the relevant tests/commands.
7. Report exact commands and actual outcomes.
8. Summarize files changed, outputs generated and remaining risks.
9. Update `docs/JOURNAL.md` and relevant docs when a meaningful decision/result changes.
10. Ask for user approval before destructive data changes, dependency upgrades, large refactors, changing labels, excluding cows, replacing model artifacts, or rewriting evaluation logic.

Do not:
- make broad changes in one enormous pass;
- change model equations without tests;
- overwrite raw CSVs;
- hide failing tests;
- fabricate app screenshots or results;
- invent source references;
- use the held-out test fold to tune the model;
- rename files without updating every reference;
- delete old results without preserving their provenance;
- claim that the app runs unless it was launched;
- claim that experiments passed unless the commands were actually executed.

## Commands commonly used
Run from the repository root and use the existing virtual environment.

```bash
python test_modules.py
python data_prep.py
python baselines.py
python train_anfis.py
```

Inspect the scripts before running if they might overwrite outputs or consume substantial time. Confirm that `data/raw/clinical_mastitis_cows.csv` exists. Do not assume the current working directory or paths.

If Streamlit is implemented:
```bash
streamlit run app.py
```

Do not assume Streamlit is installed; verify the environment and install only if permitted.

---

# 14. Phased prompts for Copilot

Use these in order. Do not ask Copilot to execute all phases simultaneously.

## Prompt A — Audit only; do not edit
“Read `docs/COPILOT_PROJECT_CONTEXT.md` and all relevant project docs. Inspect the actual code, reports and data paths. Do not modify any file. Produce a status table with: confirmed working, existing but not verified, broken/missing, scientific risks, and next three actions. Cite exact filenames and functions. Identify any discrepancies between documentation and code. Do not claim a command passed unless you run it.”

## Prompt B — Reproducibility audit
“Check the current Git status, environment, data hashes, dataset shape, target counts, group counts and leakage columns. Run the existing test script. Inspect the preparation and training pipeline for group separation and train-only preprocessing. Do not overwrite raw data or existing results. Save an audit report with commands, outputs and unresolved issues.”

## Prompt C — Reproduce current results
“Reproduce the current data preparation, baselines and ANFIS results from the existing working CSV. Before execution, inspect scripts for output overwrites. Compare the generated output with the existing reports. Preserve prior reports and write this run to a timestamped or named experiment directory. Include dataset SHA-256, package versions, seeds, fold configuration, metrics and rule count. Do not tune against the test set.”

## Prompt D — Fair raw-vs-engineered comparison
“Implement or run a fair comparison of raw sensor features, engineered Sensor-only features and Full-Clinical features. Use identical stratified group folds by cow and consistent baselines. Fit preprocessing inside each training fold. Save out-of-fold predictions and per-fold metrics. Do not use `Milk_visibility`. Do not report the exploratory raw-sensor metrics as verified until this experiment completes. Add tests for fold separation.”

## Prompt E — Five-fold evaluation and statistics
“Implement grouped 5-fold CV with a documented configuration. Save fold metrics, out-of-fold predictions, confusion matrices and model settings. Compute appropriate confidence intervals at the cow/group level. Use paired tests only where their assumptions and evaluation units are appropriate. Do not treat folds as independent biological samples. Write an experiment report and update the journal.”

## Prompt F — Dashboard integration
“Inspect the actual Streamlit source and trained model artifacts. Trace preprocessing and inference. Integrate the model produced by the verified pipeline without mixing incompatible scalers or feature order. Implement a professional Streamlit research dashboard with Overview, Prediction, Explainability, Evaluation and Limitations sections. Load metrics from saved reports. Never fabricate predictions or explanations. If artifacts do not match or are missing, disable inference and explain the issue. Add smoke tests comparing app predictions with direct model inference.”

## Prompt G — Article and slides
“Using only verified reports, create an article draft and Review/Final Review slide outline aligned with the supplied evaluation rubric. Generate every metrics table from actual report files. Clearly distinguish current held-out-fold results from future grouped 5-fold experiments. Explain why Random Forest currently performs better and frame ANFIS as an interpretable alternative under investigation. Verify all citations against source PDFs. Do not claim first-ever novelty or forecasting capability.”

## Prompt H — Final release audit
“Review code, tests, dataset provenance, saved artifacts, dashboard consistency, report tables, citations, limitations and Git diff. Produce a release checklist and list every unverified claim. Confirm the application launches only if you actually launch it. Do not mark the project complete while critical tests or artifact-alignment checks are failing.”

---

# 15. Definition of done

A phase is not done because code was written. It is done only when the corresponding evidence exists.

## Code
- Tests run and pass.
- Inputs/outputs and assumptions documented.
- No raw data overwritten.
- Reproducible command documented.
- Git diff reviewed.

## Dataset
- Source/version and SHA-256 documented.
- `Milk_visibility` excluded.
- Group separation verified.
- Preprocessing fit only on training data.
- Anomalous cow documented.
- Version differences reported neutrally.

## Experiments
- Configuration and seeds saved.
- Validation protocol documented.
- Metrics generated from predictions.
- Baselines use comparable folds.
- Confidence intervals/tests are appropriate.
- Reports trace to actual runs.
- No results copied manually into a table without source.

## Dashboard
- Correct model and scaler loaded.
- Feature order matches training.
- App prediction agrees with direct inference.
- Explanations are model-derived.
- Metrics are loaded from reports.
- Missing artifacts cause an explicit error.
- Research-prototype disclaimer is visible.
- Launch has been tested.

## Article/presentation
- Every number traces to a report.
- Dataset version and split protocol are explicit.
- Prior work and citations verified.
- Current limitations are stated.
- No unsupported novelty, clinical, forecasting or superiority claims.
- Slides and app match the same experiment.

---

# 16. Current known status summary

Based on the files and audit notes available at the time this context was written:

**Implemented in the repository**
- Data preparation and feature engineering.
- Cow-grouped held-out split.
- Baseline model code.
- Subtractive clustering.
- ANFIS implementation and hybrid learning.
- Explanation/rule generation.
- Four module tests.
- Saved models, comparison CSV, rules, figures and example explanations.

**Reported as reproducible in the previous audit**
- Four tests pass.
- Processed data regenerates identically.
- Baseline and ANFIS results match `comparison.csv`.
- Notebook Version 1 rounded values were reproduced by the audit.
- The tested leakage-replication recipe changed Random Forest accuracy by approximately one percentage point.

These are prior audit reports. Re-run the relevant checks before declaring the current local environment reproducible.

**Unresolved or requiring fresh verification**
- Official provenance of the working CSV against the exact Version 2 archive.
- Why the dataset versions differ.
- The anomalous cow's cause and treatment.
- Original-paper verification of all literature claims.
- Reproduction of raw-sensor exploratory results.
- Final grouped 5-fold results and statistical comparisons.
- Genetic-algorithm tuning.
- Exact source and provenance of the teammate's app artifacts.
- App/model/scaler alignment.
- Whether Streamlit launches and predictions match the main pipeline.
- Completion and submission readiness of the article and final report.

**The next action is the audit-only prompt (Prompt A), not a rewrite.** After Copilot reports the actual local state, execute one phase at a time and preserve the evidence.

---

# 17. User communication preferences

The project owner wants:
- straightforward, beginner-friendly explanations;
- commands explained before they are run;
- one implementation step at a time rather than a large list of commands with no guidance;
- honest reporting of progress and failures;
- no unsupported claims;
- preservation of existing work;
- exact file names and locations for edits;
- explanations they can confidently repeat during faculty review.

When explaining an implementation, define unfamiliar terms in plain language and connect each code change to the research objective. The owner must be able to explain the code, not merely have Copilot generate it.
