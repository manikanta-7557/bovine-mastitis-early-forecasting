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


LOW_MATCH = 1e-5   # heuristic: below this, no rule matches the cow well (8% of fold-0 test rows); NOT a clinical cut-off


def predict(bundle: dict, vals: dict, top: int = 3) -> dict:
    """Score = sum_i wbar_i * f_i. Every rule is returned (`all_rules`) with its UNROUNDED weighted contribution
    wbar_i * f_i, so the contributions add up to the raw score exactly. `top_rules` is the same list cut to `top`."""
    x, warns = scale(bundle, vals); m = bundle["model"]
    logw = m._logw(x[None, :])[0]; raw_w = np.exp(logw)
    wbar = m._wbar(x[None, :])[0]; f = m._rule_outputs(x[None, :])[0]
    contrib = wbar * f
    score = float(contrib.sum()); clipped = float(np.clip(score, 0, 1))
    order = np.argsort(-wbar)
    rules = [dict(rule=int(i) + 1, raw_firing=float(raw_w[i]), firing_share=float(wbar[i]), rule_output_for_this_cow=float(f[i]),
                  weighted_contribution=float(contrib[i]), conditions=rule_view(bundle, int(i), vals)) for i in order]
    max_raw = float(raw_w.max())
    if max_raw < LOW_MATCH:
        warns = warns + [f"Weak match: the best rule's raw firing strength is {max_raw:.1e}. No rule fits this cow well, "
                         "so the normalised shares can look confident while the evidence is thin."]
    return dict(score=round(score, 3), score_unrounded=score, score_clipped=round(clipped, 3), clipped=bool(score != clipped),
                tier=risk_tier(clipped), tier_memberships=fuzzy_risk(clipped), warnings=warns, n_rules=int(m.R),
                max_raw_firing=max_raw, all_rules=rules, top_rules=rules[:top])


def pick_examples(bundle: dict, df) -> dict:
    """Honest example rows from the held-out fold: the healthy and mastitis rows whose score is nearest each class median,
    plus the row whose score is nearest 0.5. Returns {name: row}. (Not 'typical' in any clinical sense.)"""
    t = df[df.fold == 0].reset_index(drop=True); cols = bundle["cols"]
    x = np.clip((t[cols].values - bundle["lo"]) / np.where(bundle["hi"] - bundle["lo"] == 0, 1.0, bundle["hi"] - bundle["lo"]), 0, 1)
    s = np.asarray(bundle["model"].predict_score(x)).ravel()
    def near(mask, target):
        idx = np.where(mask)[0]; return t.iloc[idx[np.argmin(np.abs(s[idx] - target))]]
    h, ms = t.label.values == 0, t.label.values == 1
    return {"Healthy cow (median score)": near(h, np.median(s[h])), "Mastitis cow (median score)": near(ms, np.median(s[ms])),
            "Borderline cow (score near 0.5)": near(h | ms, 0.5)}
