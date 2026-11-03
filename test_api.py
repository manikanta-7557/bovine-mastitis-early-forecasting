"""HTTP tests for the FastAPI adapter over the saved ANFIS research pipeline.

DEFENSE GUIDE
  These tests use rows already present in the repository's held-out clinical
  split and synthetic study data; they do not fabricate measurements or model
  outputs. They check that the web layer preserves the Python model results.
  Alternatives rejected: mocking inference in endpoint tests or asserting on
  manually entered performance values.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

import api
import app_core
import forecast_core as fc
import forecast_study as fs


ROOT = Path(__file__).resolve().parent


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(api.app)
        cls.bundle = app_core.load("Sensor-only", ROOT)
        clinical = pd.read_csv(ROOT / "data/processed/full_processed.csv")
        heldout = clinical[clinical.fold == 0].iloc[0]
        cls.clinical_values = {name: float(heldout[name]) for name in cls.bundle["cols"]}
        cls.clinical_body = {
            "model": "Sensor-only",
            "identifier": "test-case",
            "measurements": cls.clinical_values,
        }

    def test_health_and_schema_report_actual_artifact_state(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["models"]["R8"]["available"])
        schema = self.client.get("/api/model/schema", params={"mode": "Sensor-only"})
        self.assertEqual(schema.status_code, 200)
        self.assertEqual(schema.json()["features"], self.bundle["cols"])

    def test_clinical_prediction_matches_saved_model_and_reports_ranges(self) -> None:
        response = self.client.post("/api/predict", json=self.clinical_body)
        self.assertEqual(response.status_code, 200, response.text)
        expected = app_core.predict(self.bundle, self.clinical_values)
        body = response.json()
        self.assertAlmostEqual(body["prediction"]["score"], expected["score"], places=3)
        self.assertEqual(body["model"]["key"], "R8")
        self.assertTrue(all("inside_learned_range" in item for item in body["ranges"]))
        self.assertFalse(body["prediction"]["is_calibrated_probability"])

    def test_extended_clinical_model_uses_its_saved_feature_order(self) -> None:
        bundle = app_core.load("Full-clinical", ROOT)
        frame = pd.read_csv(ROOT / "data/processed/full_processed.csv")
        row = frame[frame.fold == 0].iloc[0]
        values = {name: float(row[name]) for name in bundle["cols"]}
        response = self.client.post(
            "/api/predict",
            json={"model": "Full-clinical", "measurements": values},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["model"]["key"], "R13")
        self.assertEqual(list(response.json()["measurements"]), bundle["cols"])

    def test_out_of_range_input_is_clipped_and_original_is_retained(self) -> None:
        values = dict(self.clinical_values, IUFL=9999.0)
        response = self.client.post(
            "/api/predict",
            json=dict(self.clinical_body, measurements=values),
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        item = next(entry for entry in body["ranges"] if entry["feature"] == "IUFL")
        self.assertFalse(item["inside_learned_range"])
        self.assertEqual(item["value"], 9999.0)
        self.assertEqual(item["adjusted_value"], item["high"])
        self.assertTrue(body["warnings"])

    def test_explanation_uses_model_rules_and_contributions(self) -> None:
        response = self.client.post("/api/explain", json=self.clinical_body)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        rules = body["explanation"]["rules"]
        self.assertEqual(len(rules), self.bundle["model"].R)
        self.assertAlmostEqual(
            sum(rule["weighted_contribution"] for rule in rules),
            body["prediction"]["score_unrounded"],
            places=10,
        )

    def test_prediction_rejects_missing_extra_and_nonfinite_inputs(self) -> None:
        missing = dict(self.clinical_body, measurements={})
        self.assertEqual(self.client.post("/api/predict", json=missing).status_code, 422)
        extra_values = dict(self.clinical_values, invented=1.0)
        extra = dict(self.clinical_body, measurements=extra_values)
        self.assertEqual(self.client.post("/api/predict", json=extra).status_code, 422)
        nonfinite = dict(self.clinical_values, IUFL=float("nan"))
        invalid = dict(self.clinical_body, measurements=nonfinite)
        response = self.client.post(
            "/api/predict",
            content=json.dumps(invalid),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422)

    def test_forecast_endpoint_uses_eligible_heldout_study_row(self) -> None:
        options = self.client.get("/api/forecast/cows")
        self.assertEqual(options.status_code, 200, options.text)
        cow = options.json()["cows"][0]
        day = cow["days"][len(cow["days"]) // 2]
        response = self.client.post("/api/forecast", json={"cow_id": cow["cow_id"], "day_index": day})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("computer-generated data", body["data_notice"])
        self.assertFalse(body["is_calibrated_probability"])
        self.assertEqual(body["observation_day"], day)
        self.assertTrue(body["trajectory"])
        self.assertTrue(body["temporal_history"])
        self.assertTrue(all(row["day_index"] <= day for row in body["trajectory"]))
        self.assertTrue(all(row["day_index"] <= day for row in body["temporal_history"]))
        self.assertTrue(body["explanation"]["rules"])
        bundle = fc.load_forecast(ROOT)
        daily = fs.make_features(pd.read_csv(ROOT / fc.DATA_DIR / "daily_cow_data.csv"))
        expected = fc.trajectory(bundle, daily, cow["cow_id"])
        expected = expected[expected.day_index == day].iloc[0].score
        self.assertAlmostEqual(body["model_score"], float(expected), places=10)

    def test_results_are_loaded_from_report_files(self) -> None:
        response = self.client.get("/api/results")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["detection"]["pooled_grouped_five_fold"]["available"])
        self.assertTrue(body["synthetic_study"]["synthetic_onset_results"]["available"])
        self.assertTrue(body["detection"]["pooled_grouped_five_fold"]["records"])

    def test_pdf_report_download_contains_pdf_bytes(self) -> None:
        response = self.client.post("/api/report", json=self.clinical_body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content[:5], b"%PDF-")
        self.assertIn("application/pdf", response.headers["content-type"])

    def test_missing_model_and_study_data_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(api, "ROOT", Path(temporary)):
                missing_model = self.client.post("/api/predict", json=self.clinical_body)
                self.assertEqual(missing_model.status_code, 503)
                missing_study = self.client.get("/api/forecast/cows")
                self.assertEqual(missing_study.status_code, 503)
                health = self.client.get("/api/health")
                self.assertFalse(health.json()["models"]["R8"]["available"])


if __name__ == "__main__":
    unittest.main()
