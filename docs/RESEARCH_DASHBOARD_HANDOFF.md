# Research dashboard handoff

## What is in this branch

- The earlier UI v3 integration added an isolated computer-generated seven-day onset-study demonstration, its separate FC7 model/data/results, explanation support, and auxiliary metrics. The clinical detection dataset and R8/R13 models remain separate.
- The latest dashboard update organizes the app into Overview, Synthetic onset study, Explainable reasoning, Temporal analysis, Data explorer, Model laboratory, Evaluation, and About & methodology.
- The data explorer provides filtering, pagination, missing-value summaries, distributions, and downloads without animal/farm identifiers. The model and evaluation pages read saved artifacts; they do not train models or overwrite results.
- The clinical score explanation now states the evaluation threshold direction correctly. Navigation and main controls are covered by `test_dashboard_ui.py`.

## Evidence and limits

The primary project task remains detection of existing clinical mastitis. The onset-study pages use computer-generated data only; their results are not evidence of performance on real animals or farms. This dashboard work does not establish clinical validation.

## Verification

On 2026-11-03, the following passed:

- `python -m compileall -q app.py explain_report.py plots.py test_explain_report.py test_dashboard_ui.py`
- `python test_modules.py`
- `python test_app_core.py`
- `python test_explain_report.py`
- `python test_forecast.py`
- `python test_dashboard_ui.py`

The Streamlit CLI was not on `PATH`; `python -m streamlit run app.py --server.headless true --server.port 8502` started successfully and its health endpoint returned HTTP 200. Browser inspection covered the eight workspace sections. The temporary server was stopped afterward.

## Git

The dashboard update is based on integration checkpoint `591de9da2efbcadb29ec0944baa69735b3f21795` (`feat: integrate UI v3 with synthetic onset study`). Commit and push status is reflected in the branch history and remote.

## React/FastAPI application update

The branch now also contains a React/TypeScript/Vite interface and a FastAPI adapter over the existing Python source of truth. The seven screens are Overview, Clinical detection, Fuzzy reasoning, Synthetic onset study, Model evaluation, Reports, and Methodology. The Streamlit application remains in place for compatibility.

`api.py` exposes health, model schema, prediction, explanation, synthetic-study, saved-results, and PDF-report endpoints. Clinical calls use `app_core.py`; study calls use `forecast_core.py` and `forecast_study.py`; explanations and PDFs use `explain_report.py`. Evaluation panels load saved CSVs and do not embed metric values. The front end only charts observed eligible generated-study scores and reports the limitation that these data are not real farm validation.

`python serve.py` builds the frontend if needed and serves it with the API at `http://127.0.0.1:8000`. `docs/UI_DESIGN_AUDIT.md` records that the named Alpha screen sources were unavailable and distinguishes general design patterns from rejected model/data reuse. Detailed run commands and actual validation output are in `reports/RUNLOG.md` after the implementation test run.
