"""FastAPI interface for the saved clinical and synthetic-study models.

DEFENSE GUIDE
  Why keep inference in Python? The API calls app_core and forecast_core so the
    application uses the same preprocessing, trained artifacts, and explanations
    as the tested research pipeline.
  Alternatives rejected: JavaScript inference, hand-entered metrics, and treating
    the computer-generated onset study as clinical evidence.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from fastapi.staticfiles import StaticFiles

import app_core
import explain_report as er
import forecast_core as fc
import forecast_study as fs

ROOT = Path(__file__).resolve().parent
app = FastAPI(
    title="Bovine Mastitis Research API",
    description="Research prototype for existing clinical mastitis detection and a separate synthetic-data onset study.",
    version="1.0.0",
)
FRONTEND_DIST = ROOT / "frontend" / "dist"


class ClinicalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["Sensor-only", "Full-clinical"]
    measurements: dict[str, float]
    identifier: str | None = Field(default=None, max_length=80)


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cow_id: str = Field(min_length=1, max_length=80)
    day_index: int = Field(ge=0)


def _clinical_bundle(mode: str) -> dict:
    try:
        return app_core.load(mode, ROOT)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail=f"Clinical model '{mode}' is unavailable: {exc}") from exc


def _validated_measurements(bundle: dict, measurements: dict[str, float]) -> dict[str, float]:
    required = list(bundle["cols"])
    missing = [name for name in required if name not in measurements]
    extra = [name for name in measurements if name not in required]
    if missing or extra:
        message = []
        if missing:
            message.append(f"Missing required measurements: {', '.join(missing)}.")
        if extra:
            message.append(f"Unexpected measurements: {', '.join(extra)}.")
        raise HTTPException(status_code=422, detail=" ".join(message))
    values = {name: float(measurements[name]) for name in required}
    if not all(math.isfinite(value) for value in values.values()):
        raise HTTPException(status_code=422, detail="All measurements must be finite numbers.")
    return values


def _range_details(bundle: dict, values: dict[str, float]) -> list[dict]:
    result = []
    for name, low, high in zip(bundle["cols"], bundle["lo"], bundle["hi"]):
        value = values[name]
        adjusted = float(min(max(value, float(low)), float(high)))
        result.append(
            {
                "feature": name,
                "value": value,
                "low": float(low),
                "high": float(high),
                "inside_learned_range": bool(value >= low and value <= high),
                "adjusted_value": adjusted,
            }
        )
    return result


def _explanation_payload(bundle: dict, values: dict[str, float], prediction: dict) -> dict:
    analysis = er.analyse(bundle, values, out=prediction)
    return {
        "dominant_rule": analysis["dominant"],
        "active_rules": analysis["active"],
        "measurements": analysis["measurements"],
        "narrative": analysis["narrative"],
        "rules": [
            {
                "rule": rule["rule"],
                "raw_firing_strength": rule["raw_firing"],
                "firing_share": rule["firing_share"],
                "rule_output_for_this_input": rule["rule_output_for_this_cow"],
                "weighted_contribution": rule["weighted_contribution"],
                "conditions": rule["conditions"],
            }
            for rule in analysis["rules"]
        ],
    }


@app.get("/api/health")
def health() -> dict:
    models = {}
    for key in app_core.MODES.values():
        paths = (ROOT / f"models/anfis_{key}.npz", ROOT / f"models/scaler_{key}.json")
        models[key] = {"available": all(path.is_file() for path in paths)}
    forecast_paths = (
        ROOT / "models/anfis_FC7.npz",
        ROOT / "models/scaler_FC7.json",
        ROOT / "models/forecast_heldout_cows.json",
        ROOT / fc.DATA_DIR / "daily_cow_data.csv",
    )
    models["FC7"] = {"available": all(path.is_file() for path in forecast_paths), "data_type": "computer-generated"}
    return {"status": "ok", "project": "Interpretable ANFIS for Detection of Existing Clinical Bovine Mastitis", "models": models}


@app.get("/api/model/schema")
def model_schema(mode: Literal["Sensor-only", "Full-clinical"]) -> dict:
    bundle = _clinical_bundle(mode)
    return {
        "configuration": bundle["mode"],
        "key": bundle["key"],
        "features": bundle["cols"],
        "ranges": {
            name: {"low": float(low), "high": float(high)}
            for name, low, high in zip(bundle["cols"], bundle["lo"], bundle["hi"])
        },
    }


@app.post("/api/predict")
def predict(request: ClinicalRequest) -> dict:
    bundle = _clinical_bundle(request.model)
    values = _validated_measurements(bundle, request.measurements)
    try:
        result = app_core.predict(bundle, values)
    except (ValueError, KeyError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=f"Prediction could not be calculated: {exc}") from exc
    return {
        "task": "detection of existing clinical mastitis",
        "identifier": request.identifier,
        "model": {"configuration": bundle["mode"], "key": bundle["key"], "rules": result["n_rules"]},
        "prediction": {
            "score": result["score"],
            "score_unrounded": result["score_unrounded"],
            "displayed_score": result["score_clipped"],
            "presentation_category": result["tier"],
            "is_calibrated_probability": False,
        },
        "warnings": result["warnings"],
        "ranges": _range_details(bundle, values),
        "rule_outputs": result["all_rules"],
        "measurements": values,
    }


@app.post("/api/explain")
def explain(request: ClinicalRequest) -> dict:
    bundle = _clinical_bundle(request.model)
    values = _validated_measurements(bundle, request.measurements)
    try:
        prediction = app_core.predict(bundle, values)
        explanation = _explanation_payload(bundle, values, prediction)
    except (ValueError, KeyError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=f"Explanation could not be calculated: {exc}") from exc
    return {"model": bundle["key"], "prediction": prediction, "explanation": explanation}


def _forecast_data() -> tuple[dict, pd.DataFrame, list[str]]:
    try:
        bundle = fc.load_forecast(ROOT)
        daily = fc.load_daily(ROOT)
        cows = fc.heldout_cows(ROOT)
    except (FileNotFoundError, OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=503, detail=f"Synthetic study model or dataset is unavailable: {exc}") from exc
    return bundle, daily, cows


@app.get("/api/forecast/cows")
def forecast_cows() -> dict:
    bundle, daily, cows = _forecast_data()
    options = []
    eligible = daily[(daily.forecast_eligible == 1) & (daily.n_prior >= fs.MIN_PRIOR)]
    for cow in cows:
        rows = eligible[eligible.cow_id.astype(str) == str(cow)]
        if not rows.empty:
            options.append({"cow_id": str(cow), "days": [int(day) for day in rows.day_index.tolist()]})
    return {
        "data_notice": "Experimental forecasting using computer-generated data — not validated on real farm records.",
        "model": bundle["key"],
        "horizon_days": bundle["horizon"],
        "cows": options,
    }


@app.post("/api/forecast")
def forecast(request: ForecastRequest) -> dict:
    bundle, daily, cows = _forecast_data()
    if request.cow_id not in {str(cow) for cow in cows}:
        raise HTTPException(status_code=422, detail="Select a cow from the study's held-out synthetic-data set.")
    observations_to_date = daily[
        (daily.cow_id.astype(str) == request.cow_id)
        & (daily.day_index <= request.day_index)
    ]
    try:
        trajectory = fc.trajectory(bundle, observations_to_date, request.cow_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Study trajectory could not be calculated: {exc}") from exc
    eligible = trajectory[
        (trajectory.forecast_eligible == 1)
        & (trajectory.n_prior >= fs.MIN_PRIOR)
    ].copy()
    selected = eligible[eligible.day_index == request.day_index]
    if selected.empty:
        raise HTTPException(status_code=422, detail="The selected observation is not eligible for the saved study model.")
    scores = [
        {"day_index": int(row.day_index), "score": float(row.score), "alert": bool(row.alert)}
        for row in eligible.itertuples()
    ]
    temporal_history = [
        {
            "day_index": int(row.day_index),
            "inputs": {
                name: 0.0 if pd.isna(row[name]) else float(row[name])
                for name in bundle["cols"]
            },
        }
        for _, row in eligible.iterrows()
    ]
    try:
        values = fc.vals_at(bundle, trajectory, request.day_index)
        prediction = app_core.predict(bundle, values)
        explanation = _explanation_payload(bundle, values, prediction)
    except (IndexError, KeyError, ValueError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=f"Study explanation could not be calculated: {exc}") from exc
    selected_row = selected.iloc[0]
    return {
        "task": "experimental synthetic-data seven-day onset study",
        "data_notice": "Experimental forecasting using computer-generated data — not validated on real farm records.",
        "real_data_limitation": "The real clinical dataset does not establish future-onset forecasting.",
        "cow_id": request.cow_id,
        "observation_day": request.day_index,
        "horizon_days": bundle["horizon"],
        "model": bundle["key"],
        "model_score": float(selected_row.score),
        "alert_threshold": bundle["alert_threshold"],
        "alert": bool(selected_row.alert),
        "is_calibrated_probability": False,
        "temporal_inputs": values,
        "trajectory": scores,
        "temporal_history": temporal_history,
        "explanation": explanation,
    }


def _read_result(path: Path) -> list[dict] | None:
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    return json.loads(frame.to_json(orient="records", double_precision=8))


@app.get("/api/results")
def results() -> dict:
    detection = {
        "pooled_grouped_five_fold": ROOT / "reports/final/summary_pooled.csv",
        "additional_sensor_only_metrics": ROOT / "reports/final/extra_metrics_R8.csv",
    }
    study = {"synthetic_onset_results": ROOT / "reports/forecast/forecast_summary.csv"}
    matrix_files = []
    for directory in (ROOT / "reports/final", ROOT / "reports/forecast"):
        if directory.is_dir():
            matrix_files.extend(path for path in directory.rglob("*confusion*") if path.is_file())
    return {
        "detection": {
            name: {"available": path.is_file(), "records": _read_result(path)}
            for name, path in detection.items()
        },
        "synthetic_study": {
            name: {"available": path.is_file(), "records": _read_result(path)}
            for name, path in study.items()
        },
        "confusion_matrices": [
            {
                "file": str(path.relative_to(ROOT)),
                "records": _read_result(path) if path.suffix.lower() == ".csv" else None,
            }
            for path in matrix_files
        ],
        "provenance_note": "Detection and computer-generated onset-study results are separate evidence streams.",
    }


@app.post("/api/report")
def report(request: ClinicalRequest) -> Response:
    bundle = _clinical_bundle(request.model)
    values = _validated_measurements(bundle, request.measurements)
    try:
        prediction = app_core.predict(bundle, values)
        analysis = er.analyse(bundle, values, out=prediction)
        blocks = er.build_report(analysis, {"case": request.identifier or "User-entered measurements"})
        pdf = er.report_pdf(blocks)
    except (ValueError, KeyError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=f"PDF report could not be generated: {exc}") from exc
    if pdf is None:
        raise HTTPException(status_code=503, detail="PDF generation is unavailable because ReportLab is not installed.")
    filename = re.sub(r"[^A-Za-z0-9._-]+", "-", request.identifier or "clinical-detection").strip("-")
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename or "clinical-detection"}.pdf"'},
    )


if (FRONTEND_DIST / "index.html").is_file():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
