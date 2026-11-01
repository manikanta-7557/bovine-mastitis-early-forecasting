# Copilot task list: finish the project by Wednesday 4 Nov 2026
Give Copilot (Agent mode, repo open) ONE task at a time. After each task: run the check, commit. Tell me if a check fails and paste the output.
Rule for every prompt: "Do not change existing metrics or models. Do not invent numbers. Never write forecasting, early warning or subclinical. Never claim ANFIS beats random forest."

## Timeline
| When | Goal |
|---|---|
| Sun night / Mon morning | Tasks 1 to 4: install pack, rerun everything on your machine, tests pass, results match mine |
| Mon | Tasks 5 to 7: app, repo hygiene, journal and run log |
| Tue | Tasks 8 to 10: report, paper, slides filled from `reports/final` |
| Wed | Task 11: rehearse, freeze, tag, submit |

## Task 1. Install the final pack (5 min)
Unzip `final_pack.zip`. Copy into the repo root (overwrite `requirements.txt`): `final_experiments.py analyze_final.py train_final_models.py ga_tuning.py app_core.py app.py test_app_core.py requirements.txt`; copy `models/*` into `models/`; copy `reports/final/` into `reports/`.
Prompt: "Run `pip install -r requirements.txt`. Then run `python test_modules.py` and `python test_app_core.py`. Show me the output."
Check: both print ALL ... PASSED.

## Task 2. Reproduce the main experiment (5 min run)
Prompt: "Run `python final_experiments.py --root . --out reports/final_repro` then `python analyze_final.py --root . --out reports/final_repro`. Compare `reports/final_repro/summary_pooled.csv` with `reports/final/summary_pooled.csv` and print any F1 or AUC that differs by more than 0.002."
Check: no differences (numpy/sklearn versions may cause tiny ones). If large differences appear, stop and tell me.

## Task 3. Rebuild the final models and GA comparison
Prompt: "Run `python train_final_models.py --root .` and `python ga_tuning.py --root . --set R8 --out reports/final_repro`. Show `reports/final_repro/fold0_results.csv` and `ga_summary_R8.csv`."
Expected: R8 fold-0 F1 about 0.936, AUC 0.9825, 5 rules; GA F1 0.913 vs grid 0.919 (GA does not win; that is the honest result).

## Task 4. Verify the dataset provenance (needs you)
Download the official Version 2 zip from Mendeley (doi 10.17632/kbvcdw5b4m.2), extract the CSV, then:
Prompt: "Write `verify_provenance.py` that prints the SHA256 and size of two files given as arguments and says IDENTICAL or DIFFERENT. Run it on data/raw/clinical_mastitis_cows.csv and the CSV from the Version 2 zip."
Write the result in docs/JOURNAL.md. If IDENTICAL, change the "pending" wording in the report to "verified".

## Task 5. Dashboard
Prompt: "Make sure `streamlit run app.py` starts. The app uses `app_core.py`; do not change its maths. Fix only import or layout errors. Then add a sidebar link text 'Research prototype, not a diagnosis'."
Check by hand: click 'Typical healthy cow' (expect Low) and 'Typical mastitis cow' (expect High) in both modes. Take two screenshots for the slides.
Note: I could not run Streamlit myself; I tested the maths (`test_app_core.py`) and checked the UI file compiles.

## Task 6. Repo hygiene
Prompt: "Update README.md: project summary (detection of existing clinical mastitis, not forecasting), install, run order (data_prep, baselines, train_anfis, final_experiments, analyze_final, train_final_models, ga_tuning, streamlit run app.py), results table copied from reports/final/RESULTS.md, limitations. Rename the feature-set label 'Early-Warning' to 'Sensor-only' in README and comments only (do not rename files or model keys). Add a `.gitignore` entry for data/raw/*.csv."
Then: `git add -A && git commit -m "Final Review experiments, raw-sensor ANFIS, dashboard"`.

## Task 7. Journal and run log
Prompt: "Append to docs/JOURNAL.md and docs/RUNLOG.md: for each script run (date, command, git commit hash, output file, one-line result). Use only outputs that exist."

## Task 8. Report (Tuesday)
Open `docs/REPORT_DRAFT.md`. Prompt per section: "Turn section N of docs/REPORT_DRAFT.md into Word-ready text for our project report. Keep every number exactly as written. Do not add citations that are not in the reference list." Export as .docx via Word (Copilot cannot render figures: insert figures from `reports/figures` and `reports/final` yourself).
Figures to produce (ask Copilot to write a plotting script reading the CSVs): (1) bar chart pooled F1 with CI per model for R8; (2) AUC same; (3) number of rules vs F1 for the four feature sets; (4) membership-function plot of the R8 model.

## Task 9. Article
Use the abstract and outline at the end of `docs/REPORT_DRAFT.md`. Pick a venue (any suitable conference or journal on AI in agriculture or veterinary informatics); put its name in the draft. Do not write "submitted" unless it is.

## Task 10. Slides
I will update the slide deck on request; send me "update deck". Until then use `docs/REPORT_DRAFT.md` section 5 for the results slides.

## Task 11. Final checks (Wednesday)
- [ ] `python test_modules.py` and `python test_app_core.py` pass from a fresh clone.
- [ ] Every number in the report appears in `reports/final/RESULTS.md` or `fold0_results.csv`.
- [ ] No text says forecasting, early warning, subclinical, first-ever, or that ANFIS beats RF.
- [ ] Notebook cell 107 not shown anywhere.
- [ ] `git tag final-review`, push.

## Defense guide for this plan
- Why raw sensors as the main model? Under the same cow-grouped folds they beat the engineered indices (ANFIS F1 0.919 vs 0.865), and rules in raw units are readable.
- Why keep the engineered-index models? They are the Review II design; keeping them shows the feature study honestly (ablation).
- Why not oversample? Oversampling before a split copies rows across train and test; oversampling only the training part changed nothing in our replication.
