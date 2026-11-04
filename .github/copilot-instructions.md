# Copilot instructions for this repository (auto-loaded by GitHub Copilot)

## Project
Interpretable ANFIS (first-order Sugeno, subtractive clustering, hybrid learning) for DETECTING EXISTING clinical bovine mastitis from udder-sensor data. M.Tech Soft Computing project, Smart India Hackathon SIH26109. Dataset: Mendeley Data kbvcdw5b4m (1,100 cows x 6 daily rows). Read docs/PROJECT_CONTEXT.md before any task.

## Hard rules (never break)
1. Never invent numbers, citations or results. Every metric must come from a file in reports/ and be reproducible by a script in this repo.
2. Call the task "detection of existing clinical mastitis". Never write forecasting, early warning, early detection or subclinical. Name the feature set "sensor-only" (the internal key "Early-Warning" stays in old code and file names; do not rename files or model keys).
3. Never claim ANFIS beats random forest, kNN or gradient boosting. Never claim "first ANFIS for mastitis".
4. Never call dataset Version 2 synthetic, fabricated or manipulated. Say the versions "differ".
5. Never use Milk_visibility as an input (it equals the label).
6. Keep all rows of a cow in one fold. Fit scalers on training data only. Tune on validation data, never on the held-out fold.
7. Do not accuse earlier authors of error. Say "we replicated the recipe".
8. Never overwrite data/raw/clinical_mastitis_cows.csv with Version 1.
9. Do not change existing metrics, models or results files unless the task says so. Write new outputs to new files.
10. Do not show or reuse notebook cell 107 numbers (hand-typed Version 1 values).

## Working style
- One task at a time. Before changing code, say what you will change and why in plain language. After changing, run the relevant test and show the output.
- Explain like to a beginner first, then add the technical detail. Add a short "Defense guide" (why this choice, what alternatives were rejected) to any new module docstring.
- Prefer small commits with clear messages. Do not refactor working code. Do not add dependencies other than numpy, pandas, scikit-learn, scipy, matplotlib, streamlit.
- Log every substantive run in reports/RUNLOG.md (date, command, git commit, output file, one-line result) and add a JOURNAL.md entry.

## Commands
python data_prep.py --input data/raw/clinical_mastitis_cows.csv
python test_modules.py ; python test_app_core.py
python final_experiments.py --root . --out reports/final_repro
python analyze_final.py --root . --out reports/final_repro
python train_final_models.py --root .
python ga_tuning.py --root . --set R8 --out reports/final_repro
streamlit run app.py
