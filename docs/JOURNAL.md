# Research journal

Add an entry every session: date, who, what you did, what you learned, what is next. Newest at the bottom. This is viva evidence.

## 2026-10-07 - A
- Ran `data_prep.py` on the real CSV (6600 rows x 18 cols, 1100 cows x 6 days).
- Found: `Milk_visibility` is identical to `class1` (r = 1.0), so it is the target; dropped.
- Cow-grouped split: 880 train cows / 220 test cows, zero overlap; mastitis fraction 0.3941 train / 0.4227 test.
- Label is constant within 1099 of 1100 cows. One cow changes. Single-rule accuracy: Hardness 0.8718, Pain 0.7729.
- Next: baselines.

## 2026-10-14 - A
- Received `baselines.py` and `metrics.py` (row-level and cow-level metrics, cow-bootstrap CIs). Results not yet recorded.

## 2026-10-22 - A
- Received Modules 3-5 (`subtractive_clustering.py`, `anfis.py`, `explain.py`, `train_anfis.py`, `test_modules.py`); unit tests and a synthetic-data dry run passed. Not yet run on real data.

## 2026-10-24 - A
- Set up the shared repo with B. Review II is on Tuesday 2026-10-27.
- Next: run all scripts on both machines and compare numbers.

## 2026-10-30 - Copilot-assisted audit
- Inventoried the current worktree and recorded SHA-256/byte sizes for raw, processed, model and headline report artifacts in `docs/2026-10-30_phase2_audit.md`; no existing artifacts were copied or overwritten in this audit.
- Compared the working CSV with the available Version 1 CSV: they differ; no authentic Version 2 archive was found locally, so provenance remains unresolved.
- Audited cow10 without changing/removing it; found no duplicate full rows or duplicate cow/day keys.
- Added descriptive verified-defaults config, claims register, run log and a proposed grouped-CV feature-set experiment specification. The existing scripts do not yet read `config.json`; the next action is to obtain and authenticate the official Version 2 archive before starting new experiments.

## 2026-11-01 - Copilot-assisted documentation cleanup
- Consolidated project Markdown documentation, audit notes and run log into `docs/`; retained root `README.md` and `.github/pull_request_template.md` in their GitHub-required locations.
- Updated documentation and helper references to the new paths. No data, model or experiment outputs were changed.

## 2026-11-02 - Copilot-assisted dashboard refresh
- Refined the Streamlit dashboard with a responsive research header, cleaner surfaces, consistent green styling and clearer feature-set guidance.
- Kept the prediction, explanation and model workflows unchanged; confirmed the page loads and a healthy example reaches its expected presentation category.
- Ran Python syntax checks, dashboard tests and report tests. Next: review the page during the final presentation run.

## 2026-11-02 - Copilot-assisted emoji removal
- Removed emoji from the browser title icon, sidebar heading and section navigation.
- Confirmed the live page renders the plain-text labels and scanned visible app modules for remaining emoji.
- Python syntax checks and dashboard/report tests passed.

## 2026-11-02 - Copilot-assisted demo visuals
- Added a visual project workflow and a five-layer ANFIS architecture at the beginning of the guided demo.
- Updated demo navigation and tests for 15 slides, and corrected the later internal-working slide to distinguish preprocessing from the five ANFIS layers.
- Verified both new visuals in the browser; syntax, demo/report and dashboard tests passed.

## 2026-11-03 - Copilot-assisted UI v3 integration
- Added the isolated computer-generated seven-day onset study, saved model, and supporting data without changing the existing R8/R13 detection models or clinical data.
- Integrated the new app section, per-day plot, explanation/report support, and auxiliary detection metrics while retaining the 15-page demo and current visual styling.
- Ran the model, dashboard, report, and scenario tests; all passed. Generated auxiliary detection metrics and verified the detection example, synthetic study section, and supplemental metrics panel in the browser.

## 2026-11-03 - Copilot-assisted React and FastAPI application
- Added a Vite/React/TypeScript interface and FastAPI endpoints that call the existing R8/R13 inference, explanation, FC7 study, report, and saved-results modules rather than reimplementing model logic.
- Built seven connected screens, dynamic model feature/range forms, actual-rule explanations, a trajectory over observed eligible generated-data rows, report download, and methodology/limitations views. Kept the Streamlit source and existing model/data/result artifacts.
- Added API tests for both clinical configurations, range clipping, validation, explanations, missing artifacts, generated-study scoring, saved results, and PDF creation. The complete model, dashboard, report, study, and API test suite passed; the TypeScript production build passed.
- Visually checked the seven navigation screens and an end-to-end clinical input, explanation, generated-data observation, evaluation table, and PDF action in the local browser.
- Follow-up browser check confirmed the selected Day 89 FC7 result contains only its 14 latest observed history rows through that day, shows no future observations, and expands five model-derived IF–THEN rules.
- Next: teammate review of `feature/ui-v3-integration`.
- Next: review the integrated branch and commit.

## 2026-11-03 - Copilot-assisted research dashboard redesign
- Reorganized the Streamlit workspace into Overview, Synthetic onset study, Explainable reasoning, Temporal analysis, Data explorer, Model laboratory, Evaluation, and About & methodology. Kept clinical detection as the primary task and labeled all generated-data results as demonstration-only.
- Added filtered, paginated dataset previews and identifier-redacted downloads; added time-series, grouped-evaluation, rule-membership, and training-evidence views without retraining or replacing saved artifacts.
- Corrected a contradictory sentence in the clinical score explanation and adjusted rule plots to avoid layout warnings.
- Ran compile checks, all existing tests, and the new `test_dashboard_ui.py`; all passed. Launched the local Streamlit app, confirmed HTTP 200, and visually inspected all eight workspace pages.
- Next: review the uncommitted UI changes; no commit or push was made.

## 2026-11-04 - Copilot-assisted veterinary reports and rotating examples
- Updated the clinical sample buttons to cycle through distinct measured records whose saved-model scores fall inside the requested category bands. Held-out rows are preferred; Full-clinical borderline samples use clearly identified training-fold examples because no matching held-out row exists.
- Generate a varied plain-English veterinary report on each estimate from that record's actual measurements, score and learned-rule contributions; retain the PDF download and mandatory disclaimer.
- Repeated all three sample buttons in both model modes and verified distinct inputs, in-band scores, matching report text and valid PDF bytes. `test_app_core.py`, `test_explain_report.py`, `test_forecast.py`, and `test_api.py` passed. No model weights, experiment results, or dashboard styling changed.
- Next: review the updated sample wording and veterinary report.

## 2026-11-04 - Temperature unit label clarification
- Updated the Streamlit input, React interface, and veterinary report measurement label to degrees Celsius after the project owner confirmed the unit.
- Updated project context and UI guidance to match. Kept stored temperature measurements, model scaling, and inference unchanged.
- Passed `test_app_core.py`, `test_explain_report.py`, the frontend production build, and a Streamlit check of the Full-clinical input label.
