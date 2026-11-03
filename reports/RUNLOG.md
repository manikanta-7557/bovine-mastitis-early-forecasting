# Application integration run log

| Date | Command / check | Git commit (base before changes) | Output file or endpoint | Result |
|---|---|---|---|---|
| 2026-11-03 | `npm install` (from `frontend`) | `3b553ef914472b8c6b7c462c8400867d2cbec0d1` | `frontend/package-lock.json` | Installed the declared packages; npm reported 0 vulnerabilities. |
| 2026-11-03 | `python -m compileall -q .` | `3b553ef914472b8c6b7c462c8400867d2cbec0d1` | No output file (syntax check) | Passed. |
| 2026-11-03 | `python test_modules.py` | `3b553ef914472b8c6b7c462c8400867d2cbec0d1` | Test output only | Passed all clustering, gradient, least-squares, and hybrid-learning checks. |
| 2026-11-03 | `python test_app_core.py` | `3b553ef914472b8c6b7c462c8400867d2cbec0d1` | Test output only | Passed R8 and R13 inference parity checks against saved model artifacts. |
| 2026-11-03 | `python test_explain_report.py` | `3b553ef914472b8c6b7c462c8400867d2cbec0d1` | Test output only | Passed explanation, report, range-clipping, navigation, and result-file checks. |
| 2026-11-03 | `python test_forecast.py` | `3b553ef914472b8c7c462c8400867d2cbec0d1` | Test output only | Passed leakage, chronological feature, cow-fold, saved-model, and report checks. |
| 2026-11-03 | `python test_dashboard_ui.py` | `3b553ef914472b8c7c462c8400867d2cbec0d1` | Test output only | Passed all eight retained Streamlit dashboard section checks. |
| 2026-11-03 | `python test_api.py` | `3b553ef914472b8c7c462c8400867d2cbec0d1` | Test output only | Passed 10 API checks, including R8/R13, validation, range clipping, FC7, saved results, PDF output, and missing assets. |
| 2026-11-03 | `npm run build` (from `frontend`) | `3b553ef914472b8c7c462c8400867d2cbec0d1` | `frontend/dist/index.html`, `frontend/dist/assets/` | TypeScript checks and Vite production build passed. |
| 2026-11-03 | `python serve.py` | `3b553ef914472b8c7c462c8400867d2cbec0d1` | `http://127.0.0.1:8000/`, `/api/health`, `/api/results` | Application served the frontend and API; UI and health returned HTTP 200, R8/R13/FC7 were available, and saved evaluation records loaded. |
| 2026-11-03 | Local browser review of all seven screens and real-input workflows | `3b553ef914472b8c7c462c8400867d2cbec0d1` | Browser session (no file generated) | Verified navigation, an actual held-out clinical input and rule explanation, an eligible FC7 observation, evaluation metrics, and the PDF success state. |

| 2026-11-03 | FC7 selected-day and rule-detail browser check | `3b553ef914472b8c6b7c462c8400867d2cbec0d1` | Browser session (no file generated) | Day 89 showed 14 actual temporal-history rows through the selected day, no future observations, and five expandable model-derived IF–THEN rules. |

These commands ran against the implementation worktree based on the recorded base commit; no dataset, trained-model, or existing numerical result files were regenerated or overwritten.
