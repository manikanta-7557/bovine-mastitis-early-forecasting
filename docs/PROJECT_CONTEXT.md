# Project context for AI assistants (Copilot, Claude, others). Updated 2 Nov 2026.

## 1. Goal and deadlines
Deliver the Final Review (40 marks), project report (20) and an article draft by Wednesday 4 Nov 2026. Grading file: "0 Evaluation Criteria.docx". Final Review items: problem formulation and objectives (3), methodology and technical correctness (4), complete implementation (8), experimental evaluation (7), results, comparison and discussion (5), application/novelty and use of soft computing (5), presentation and demo (3), viva (5). Report: abstract/intro (2), literature and gap (3), methodology (3), implementation (2), results (4), discussion/limits (2), article documentation (2), references (1), formatting (1).
People: Manikanta Peddi (A, reg. 23MIS7081), Madhur Sathya (B, GitHub AlphaUser2025, built the first Streamlit prototype). VIT-AP University.

## 2. Data (verified)
- Working file data/raw/clinical_mastitis_cows.csv: SHA256 e4166166bf83d91a63a62d1ab5e8bd66f9f7ce6687233a67abc4d99918f532d8, 416,118 bytes, 6,600 rows, 1,100 cows x 6 days, 3,961 healthy / 2,639 mastitis (target class1). Match with the official Version 2 zip is still to be verified.
- Version 1 (data/raw/clinical_mastitis_cows_version1.csv, 5,490 / 1,110) only as a sensitivity check. 1,529 labels differ; cause undocumented.
- Columns: 8 sensors IUFL EUFL IUFR EUFR IURL EURL IURR EURR (probably inhale/exhale limits of four quarters; not stated explicitly), Temperature (unit undocumented), Hardness, Pain, months postpartum, breed.

## 3. Pipeline and files
data_prep.py -> data/processed/{train,test,full_processed}.csv (fold column, fold 0 = test; StratifiedGroupKFold 5, seed 42, grouped by cow).
baselines.py, train_anfis.py (Review II models, "Early-Warning"/"Full-Clinical" = sensor-only/full-clinical engineered features), anfis.py, subtractive_clustering.py, explain.py, metrics.py.
New for Final Review: final_experiments.py (grouped 5-fold OOF predictions, 8 models x 4 feature sets), analyze_final.py (pooled F1/AUC, cow-bootstrap CI, cow-level McNemar), train_final_models.py (raw-sensor deployment models + rules in raw units), ga_tuning.py (GA vs radius grid), app_core.py + app.py (dashboard), test_app_core.py.
Feature sets: S5 engineered sensor-only, F7 engineered full-clinical, R8 raw 8 sensors, R13 raw 8 + temperature + months + breed + hardness + pain.

## 4. Verified results (run 1 Nov 2026; pooled out-of-fold over 1,100 cows; F1 / AUC)
R8: ANFIS 0.919/0.980 (4.2 rules), RF 0.983/0.999, kNN 0.961/0.990, GradBoost 0.922/0.984, SVM 0.898/0.959, DT(d4) 0.890/0.909, LogReg 0.858/0.895.
R13: ANFIS 0.937/0.977 (7.0 rules), RF 0.975/0.997, GB 0.939/0.988, kNN 0.933/0.974, DT 0.929/0.929, SVM 0.910/0.943, LogReg 0.898/0.942.
S5: ANFIS 0.865/0.919, RF 0.914/0.978. F7: ANFIS 0.916/0.941, RF 0.928/0.978.
ANFIS vs GB on R8: McNemar p=0.74 (tie); ANFIS better than LogReg, DT, SVM; below RF and kNN. Tests uncorrected for multiple comparisons.
Fold-0 deployment models: R8 F1 0.936 / AUC 0.983 (5 rules); R13 0.937 / 0.961 (5 rules).
GA tuning (R8): F1 0.913 vs grid 0.919, no gain (58 vs 5 evaluations per fold).
Leakage replication (RF, 8 sensors + temperature, 5 seeds): paper-style 0.993 vs cow-grouped 0.984 accuracy (about 1 point); 28% of test rows are exact copies of training rows even under cow grouping (unexplained).
Main finding: feature design (two engineered indices vs eight raw sensors) matters more than the split protocol.

## 5. Known issues and limits
- Notebook ANFIS_PROJECT_MASTITIS_FORECASTING.ipynb cell 64 overwrites the raw CSV with Version 1; cell 107 numbers are hand-typed Version 1 values. Do not use.
- Old Streamlit model files from Madhur's repo were trained on something else (probably Version 1); replaced by models/anfis_R8.npz and anfis_R13.npz.
- Dashboard maths tested (test_app_core.py), but the UI itself must be run by hand with streamlit.
- explain.py prints each rule's output at its centre, not for the entered cow (first-order consequents).
- Unread papers: Chiu 1994, Ebrahimi 2019, Ankitha & Manjaiah, Mammadova & Keskin 2015, Kohli 2011. Do not cite as read.
- A second unrelated dataset (qPCR pathogen study, 33 cows) was evaluated and dropped.

## 6. Task list (do in order, one at a time)
1. pip install -r requirements.txt; data_prep; run both test files.
2. Reproduce final_experiments + analyze_final into reports/final_repro; compare with reports/final.
3. train_final_models.py; ga_tuning.py; confirm fold-0 numbers above.
4. Provenance: SHA256 of the CSV inside the official Version 2 zip vs the working file; log in JOURNAL.md.
5. Run streamlit run app.py; test healthy and mastitis examples in both modes; screenshots.
6. README (summary, install, run order, results table from reports/final/RESULTS.md, limitations), .gitignore for data/raw/*.csv, reports/RUNLOG.md, JOURNAL.md.
7. Plot script for figures: pooled F1 with CI, AUC with CI, rules vs F1, R8 membership functions.
8. Report from docs/REPORT_DRAFT.md; article from its outline; slides.
9. Final checks: tests pass on a fresh clone; every number traceable; no forbidden wording; git tag final-review.

## 7. Defense guide in one paragraph
Group by cow because six rows per cow are correlated. ANFIS is chosen for readable rules, not accuracy: 4 to 5 rules in raw sensor units versus 3^5 = 243 for a grid partition. Hybrid learning: least squares solves the linear consequents exactly, Adam tunes centres and widths. Subtractive clustering picks rule count from data density. We report a negative GA result and that random forest is better. Rejected: random row splits, oversampling before splitting, Wilcoxon over 5 folds (minimum p 0.0625), SHAP-only explanation.
