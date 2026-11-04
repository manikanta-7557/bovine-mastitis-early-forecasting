"""Build per-day ANFIS scores for the isolated synthetic 7-day onset scenario.

The inputs use only the selected cow's current measurements and prior 14-day baseline.
Every score and onset label here comes from the computer-generated study data, not the
clinical detection dataset.

DEFENSE GUIDE
  Why separate this module? It keeps scenario-specific data paths and scoring out of
    app_core, so the existing clinical detection workflow remains unchanged.
  Why retain the held-out animals? It enables a repeatable demonstration without
    presenting the synthetic rows as real-animal evidence.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from anfis import ANFIS
import forecast_study as fs

ROOT = Path(__file__).resolve().parent
DATA_DIR = "data/external/Synthetic_Data_for_Mastitis_Disease"


def load_forecast(root: Path = ROOT) -> dict:
    sc = json.loads((root / "models/scaler_FC7.json").read_text())
    return dict(mode="Synthetic onset scenario", key="FC7", task="forecast", model=ANFIS.load(root / "models/anfis_FC7.npz"), cols=sc["features"],
                lo=np.array(sc["data_min"], float), hi=np.array(sc["data_max"], float), alert_threshold=float(sc["alert_threshold"]), horizon=int(sc["horizon_days"]))


def heldout_cows(root: Path = ROOT) -> list:
    return json.loads((root / "models/forecast_heldout_cows.json").read_text())


def load_daily(root: Path = ROOT) -> pd.DataFrame:
    return fs.make_features(pd.read_csv(root / DATA_DIR / "daily_cow_data.csv"))


def _scaled(bundle, df):
    span = np.where(bundle["hi"] - bundle["lo"] == 0, 1.0, bundle["hi"] - bundle["lo"])
    return np.clip((df[bundle["cols"]].fillna(0).values - bundle["lo"]) / span, 0, 1)


def trajectory(bundle: dict, daily: pd.DataFrame, cow: str) -> pd.DataFrame:
    g = daily[(daily.cow_id == cow) & (daily.n_prior >= fs.MIN_PRIOR)].copy()
    g["score"] = np.asarray(bundle["model"].predict_score(_scaled(bundle, g))).ravel()
    g["alert"] = g.score >= bundle["alert_threshold"]
    onset = daily[(daily.cow_id == cow) & (daily.mastitis_status == 2)].day_index.min()
    g["onset_day"] = onset
    return g


def vals_at(bundle: dict, traj: pd.DataFrame, day: int) -> dict:
    r = traj[traj.day_index == day].iloc[0]
    return {c: float(0.0 if pd.isna(r[c]) else r[c]) for c in bundle["cols"]}
