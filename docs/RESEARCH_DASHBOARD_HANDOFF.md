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
