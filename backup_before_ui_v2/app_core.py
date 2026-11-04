"""
app_core.py -- the prediction logic of the dashboard, with NO Streamlit code (so it can be unit-tested)
======================================================================================================
load(mode)            -> loads models/anfis_<R8|R13>.npz and models/scaler_<..>.json
predict(bundle, vals) -> score, risk tier, warnings (values outside the training range) and the top rules
                         for THIS cow, each rule written in raw sensor units with the cow's own value next to it.
MODES  "Sensor-only"  = R8  (8 raw udder sensors)
       "Full-clinical" = R13 (8 sensors + temperature, months postpartum, breed, hardness, pain)
DEFENSE GUIDE
  Why separate from app.py? The maths must be testable without a browser; app.py only draws widgets.
  Why clip to [0,1]? ANFIS memberships were learned inside the training range; outside it the model extrapolates
  blindly, so we clip AND warn.
  Tier thresholds come from explain.risk_tier (fuzzy Low/Watch/High on the 0-1 score); they are a presentation
  choice, not a clinically validated cut-off.
"""
import json
from pathlib import Path
import numpy as np
from anfis import ANFIS
from explain import risk_tier, fuzzy_risk

ROOT = Path(__file__).resolve().parent
MODES = {"Sensor-only": "R8", "Full-clinical": "R13"}
BINARY = {"breed_enc", "hardness", "pain"}


def load(mode: str, root: Path = ROOT) -> dict:
    key = MODES[mode]
    sc = json.loads((root / f"models/scaler_{key}.json").read_text())
    return dict(mode=mode, key=key, model=ANFIS.load(root / f"models/anfis_{key}.npz"), cols=sc["features"],
                lo=np.array(sc["data_min"], float), hi=np.array(sc["data_max"], float))


def scale(bundle: dict, vals: dict):
    raw = np.array([float(vals[c]) for c in bundle["cols"]])
    span = np.where(bundle["hi"] - bundle["lo"] == 0, 1.0, bundle["hi"] - bundle["lo"])
    z = (raw - bundle["lo"]) / span
    warns = [f"{c} = {v:g} is outside the training range [{lo:g}, {hi:g}]; clipped" for c, v, lo, hi, zz in
             zip(bundle["cols"], raw, bundle["lo"], bundle["hi"], z) if zz < 0 or zz > 1]
    return np.clip(z, 0, 1), warns


def rule_view(bundle: dict, i: int, vals: dict) -> list:
    m, cols = bundle["model"], bundle["cols"]; out = []
    for j, c in enumerate(cols):
        span = bundle["hi"][j] - bundle["lo"][j]
        out.append(dict(feature=c, centre=round(bundle["lo"][j] + m.c[i, j] * span, 1), width=round(m.s[i, j] * span, 1),
                        your_value=float(vals[c])))
    return out


def predict(bundle: dict, vals: dict, top: int = 3) -> dict:
    x, warns = scale(bundle, vals); m = bundle["model"]
    wbar = m._wbar(x[None, :])[0]; f = m._rule_outputs(x[None, :])[0]
    score = float((wbar * f).sum()); clipped = float(np.clip(score, 0, 1))
    order = np.argsort(-wbar)[:top]
    return dict(score=round(score, 3), score_clipped=round(clipped, 3), tier=risk_tier(clipped), tier_memberships=fuzzy_risk(clipped),
                warnings=warns, n_rules=int(m.R),
                top_rules=[dict(rule=int(i) + 1, firing_share=round(float(wbar[i]), 3), rule_output_for_this_cow=round(float(f[i]), 3),
                                conditions=rule_view(bundle, int(i), vals)) for i in order])
