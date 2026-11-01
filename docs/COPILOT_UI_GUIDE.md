# Dashboard guide for VS Code Copilot (Streamlit)
Goal: a clean research dashboard on top of the existing ANFIS code. It already exists as `app.py` in Madhur's repo; this guide fixes the models first, then upgrades the UI.

## Step 0. Fix the models (do this before any UI work)
1. Unzip `app_fix.zip` and copy into the repo, overwriting:
   - `models/anfis_Early-Warning.npz`
   - `models/anfis_Full-Clinical.npz`
   - `data/processed/data_prep_summary.json`
2. Why: the old model files were not trained on the working dataset (his sensor-only model scores F1 0.695 on our test fold, ours 0.869). The new ones are exactly the models behind the Review II table.
3. Run `pip install streamlit numpy pandas scikit-learn matplotlib` then `streamlit run app.py`. Check: a healthy row from `data/processed/test.csv` should give a Low tier, a mastitis row a High tier.

## Step 1. Paste this prompt into Copilot Chat (Agent mode), with `app.py`, `anfis.py`, `explain.py` and `data_prep.py` open
```
You are editing a Streamlit research dashboard for an interpretable ANFIS model that
detects EXISTING clinical mastitis in dairy cows from udder-sensor readings.
Keep all existing prediction logic in app.py (feature formulas, scaler from
data/processed/data_prep_summary.json, ANFIS.load, explain_row) exactly as is.
Only change layout, wording and add the features below.

Rules for wording (important):
- Call the task "detection of existing clinical mastitis". Never say forecasting,
  early warning or subclinical.
- Rename the model options "Sensor-only" and "Full-clinical" in the UI
  (keep the internal keys "Early-Warning" and "Full-Clinical" in code).
- Say "risk score", never "diagnosis". Keep the disclaimer: research prototype,
  not a veterinary diagnosis.

Add:
1. Page layout: title, one-line description, sidebar with model choice and a
   "How it works" expander (5 ANFIS layers in plain words).
2. Three example buttons that fill the form: "Typical healthy cow",
   "Typical mastitis cow", "Borderline cow" (take example values from rows of
   data/processed/test.csv; load the file if present, otherwise hide the buttons).
3. Input validation: sensors between 100 and 400 (show a warning outside the
   training range min/max stored in data_prep_summary.json), months between 1 and 6,
   temperature range 24 to 58 with a caption "dataset scale, unit not documented".
   Warn when a value is clipped to the training range.
4. Result panel: big risk-tier badge (Low / Watch / High), the clipped score as a
   progress bar, and the raw score in small text.
5. Explanation panel: replace st.json with a table of the top 3 fired rules
   (rule number, firing share as a percentage, rule text) and a bar chart of firing shares.
6. A "Feature values used" table showing swelling index, asymmetry index and the
   four quarter totals, so the user sees how raw sensors become model inputs.
7. A "Model facts" expander showing number of rules, training data
   (working file, 3,961 healthy / 2,639 mastitis rows), and the test-fold
   results F1 and AUC read from reports/comparison.csv if it exists.
8. Footer: "Research prototype. Not a diagnostic tool. Results are preliminary."
Do not add new dependencies other than streamlit, numpy, pandas, matplotlib.
After editing, run `python -m py_compile app.py` and fix errors.
```

## Step 2. Follow-up prompts (one at a time)
- "Add a second tab 'Cow ranking' that loads data/processed/test.csv, scores every row with the selected model, averages the six days per cow with explain.rank_cows, and shows the top 10 cows."
- "Add a tab 'Rules' that prints the full rule list from explain.describe_rules for the selected model, with a note that the number after 'risk ~' is the rule's output at its centre."
- "Pin versions in requirements.txt to the versions installed here (run pip freeze for streamlit, numpy, pandas, scikit-learn, matplotlib)."

## Step 3. Check before you present (5 minutes)
- [ ] Healthy example gives Low, mastitis example gives High, in both modes.
- [ ] No text says forecasting, early warning, diagnosis.
- [ ] Rule text and firing shares change when you change inputs.
- [ ] Take one screenshot of each mode for slide 13.
- [ ] Commit: `git add -A && git commit -m "Update dashboard and retrained models"`.

## Defense guide (why the dashboard is built this way)
- **Why Streamlit?** Pure Python, so the same ANFIS modules run unchanged; React or Java would need a separate API and add risk without scientific value. Rejected for Review II.
- **Why keep the scaler in a JSON file?** The app must scale exactly as training did; reading the saved training min/max prevents train/serve mismatch, which we verified numerically.
- **Why show rules?** It is the project's contribution: a forest gives a score only, ANFIS says which rules fired.
- **Known limits to state if asked:** the rule text shows each rule's output at its centre, not at the entered cow; inputs outside the training range are clipped; temperature unit is not documented; the model is not clinically validated.
