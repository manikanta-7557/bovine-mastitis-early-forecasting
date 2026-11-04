# UI design-source audit

## What was available

- The supplied `ui_v3/ui_v3` directory contains the Streamlit application, model/report helpers, and FC7 study files.
- The repository contains the existing Streamlit research dashboard and its saved research artifacts.
- No Alpha React screen archive or individual Alpha screens (LandingPage, PredictionForm, PredictionResultView, UdderQuadrantView, ModelComparison, or related files) were present in the inspected repository/workspace locations.

## Reused as design direction

- Persistent research navigation and a clear project identity.
- A landing view that distinguishes the clinical detection workflow from the computer-generated study.
- A structured measurement form, result summary, rule-level explanation, evaluation tables, methodology, and report workflow.
- A trajectory visualization, used only for actual eligible FC7 study rows returned by the Python backend.
- A calm green research-tool palette, readable cards, responsive layouts, and explicit provenance/limitation notices.

These are general interaction patterns from the requested screen inventory and existing project dashboard; no Alpha source code or assets were available to copy or attribute to an individual screen.

## Rejected

- Alpha model weights, inference logic, hard-coded performance claims, and demonstration cow records.
- Decorative charts, invented measurements, static rule examples presented as actual output, and future observations not present in the dataset.
- Any interpretation of computer-generated records as real animal or farm evidence.

The React interface calls the existing Python model and explanation modules. Values in evaluation and forecasting views are read from repository artifacts or API responses; missing artifacts are identified rather than replaced with illustrative metrics.
