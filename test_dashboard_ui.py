"""Smoke-test Streamlit navigation and the dashboard's main interactive paths.

DEFENSE GUIDE
  Why use AppTest? It runs the actual Streamlit script, so route and widget wiring are
    checked without relying only on pure model tests or a manual browser session.
  What does this not prove? It does not certify visual appearance or clinical validity;
    browser review and the existing model/report tests cover those separate concerns.
"""
from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parent
SECTIONS = [
    "Overview",
    "Synthetic onset study",
    "Explainable reasoning",
    "Temporal analysis",
    "Data explorer",
    "Model laboratory",
    "Evaluation",
    "About & methodology",
]


def _assert_no_app_error(app: AppTest, where: str) -> None:
    errors = [item.message for item in app.exception]
    assert not errors, f"{where}: {errors}"


def main() -> None:
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=60)
    _assert_no_app_error(app, "initial page")
    for section in SECTIONS:
        app.sidebar.radio(key="section").set_value(section).run(timeout=60)
        _assert_no_app_error(app, section)
        print(f"OK  dashboard section: {section}")

    app.sidebar.radio(key="model_mode").set_value("Full-clinical").run(timeout=60)
    _assert_no_app_error(app, "Full-clinical model selection")
    app.sidebar.radio(key="section").set_value("Explainable reasoning").run(timeout=60)
    next(button for button in app.button if button.label == "Healthy cow (median score)").click()
    app.run(timeout=60)
    _assert_no_app_error(app, "held-out example selection")
    next(button for button in app.button if button.label == "Estimate risk").click()
    app.run(timeout=60)
    _assert_no_app_error(app, "Full-clinical score explanation")

    app.sidebar.radio(key="section").set_value("Data explorer").run(timeout=60)
    app.selectbox(key="explorer_dataset").set_value("Computer-generated daily study").run(timeout=60)
    _assert_no_app_error(app, "computer-generated dataset selection")
    app.sidebar.radio(key="section").set_value("Model laboratory").run(timeout=60)
    app.selectbox(key="laboratory_feature").set_value("IUFL").run(timeout=60)
    _assert_no_app_error(app, "membership-curve selection")
    print("ALL DASHBOARD UI TESTS PASSED")


if __name__ == "__main__":
    main()
