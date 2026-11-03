# Interpretable ANFIS for Detection of Existing Clinical Bovine Mastitis

Soft Computing project (M.Tech, VIT-AP). Problem statement SIH26109.
Goal: detect existing clinical mastitis from udder sensor readings with a **rule-based, explainable** ANFIS and compare it with other ML methods.

## Quick start (new collaborator: 15 minutes)

```bash
git clone <repo-url> && cd mastitis-anfis
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# get clinical_mastitis_cows.csv from your teammate (or the Mendeley page) and put it in data/raw/
python test_modules.py      # 4 x OK expected
python data_prep.py         # Module 1  -> data/processed/, reports/figures/fig1-7
python baselines.py         # Module 2  -> reports/baseline_results.csv
python train_anfis.py       # Modules 3-5 -> reports/anfis_results.csv, comparison.csv, rules, fig9-10
```

Same random seeds everywhere, so **you should get exactly the same numbers as your teammate.** If not, tell them: that is a bug worth finding.

## Pipeline

```
raw CSV -> data_prep.py (drop leakage, cow-grouped split, 2 indices, train-only scaling)
        -> baselines.py (LogReg / Tree / RandomForest)            \
        -> subtractive_clustering.py (rules from data) -> anfis.py (hybrid learning) -> explain.py
                                                                   /  compared in reports/comparison.csv
```

## Synthetic onset scenario

The app also includes a separate seven-day onset demonstration using the computer-generated data under `data/external/Synthetic_Data_for_Mastitis_Disease/`. This is not real-animal or farm evidence and does not change the project's primary task: detection of existing clinical mastitis.

```bash
python test_app_core.py
python test_explain_report.py
python test_forecast.py
streamlit run app.py
```

Choose **Synthetic onset study** in the app to inspect held-out study animals, the per-day score plot, rule explanation, and the synthetic-study results. The scenario modules and model use separate `forecast_*` files and `FC7` artifacts; they do not replace the R8 or R13 detection models. `requirements.txt` already includes the UI and PDF dependencies.

## Where to read next

1. `docs/HANDOFF.md` - every design decision, its reason, and open issues (read before touching code).
2. `docs/TASKS.md` - who is doing what, with dates.
3. `docs/JOURNAL.md` - research journal; add an entry every session.

## Team workflow

- `main` always runs. Never commit straight to `main`; use a branch `feat/<what>-<name>` and a pull request.
- The OTHER person reviews and merges. Reviewing means you can explain the change in 5 minutes.
- Raw data is not committed (`.gitignore`); share it by Drive/WhatsApp.
- Numbers on slides or in the paper must come from files in `reports/`. No invented numbers, no invented citations.
