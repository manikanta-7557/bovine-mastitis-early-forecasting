"""
extra_metrics.py -- PR-AUC, Brier score and score calibration from the SAVED out-of-fold predictions (no retraining)
=====================================================================================================================
Run:  python extra_metrics.py --root . --set R8
Reads reports/final/oof_predictions.csv (written by final_experiments.py) and writes
  reports/final/extra_metrics_<set>.csv      model, PR_AUC, Brier (each with 95% cow-level bootstrap CI)
  reports/final/calibration_ANFIS_<set>.csv  score bin, rows, mean score, observed mastitis rate
WHY  A review of the literature (2026 notes) recommends PR-AUC (minority-class view) and Brier (calibration) next to ROC-AUC.
     PR-AUC: area under the precision-recall curve (average precision).  Brier: mean (p - y)^2, lower is better.
     ANFIS scores are NOT probabilities; for Brier they are clipped to [0, 1], and the calibration table shows how far
     from a probability they are.  Brier of the other models uses their predict_proba output.
DEFENSE GUIDE
  Why cow-level bootstrap? Rows of one cow are correlated; resampling rows would give CIs that are too narrow.
  Why no isotonic/Platt calibration for ANFIS? It would require held-out calibration data and would change the reported
    score; we only MEASURE calibration here (the app keeps saying 'not a probability').
  Class balance here is 40% mastitis, so PR-AUC is less dramatic than in a 10-15% incidence herd; say so.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--set", default="R8"); ap.add_argument("--boot", type=int, default=1000)
a = ap.parse_args(); root = Path(a.root); rep = root / "reports/final"
d = pd.read_csv(rep / "oof_predictions.csv"); d = d[d.feature_set == a.set]
rng = np.random.default_rng(42)
cows = np.sort(d.cow_id.unique()); idx = {c: np.where(d.cow_id.values == c)[0] for c in cows}  # rows per cow (same order for all models after sort)
rows = []
for model, g in d.groupby("model"):
    g = g.sort_values(["cow_id", "day"]).reset_index(drop=True)
    y = g.label.values; s = g.score.values; p = np.clip(s, 0, 1)
    pos = {c: np.where(g.cow_id.values == c)[0] for c in cows}
    ap_, br = average_precision_score(y, s), float(np.mean((p - y) ** 2))
    bs = []
    for _ in range(a.boot):
        pick = np.concatenate([pos[c] for c in rng.choice(cows, len(cows))])
        bs.append((average_precision_score(y[pick], s[pick]), float(np.mean((p[pick] - y[pick]) ** 2))))
    bs = np.array(bs)
    rows.append(dict(model=model, PR_AUC=ap_, PR_lo=np.percentile(bs[:, 0], 2.5), PR_hi=np.percentile(bs[:, 0], 97.5),
                     Brier=br, Brier_lo=np.percentile(bs[:, 1], 2.5), Brier_hi=np.percentile(bs[:, 1], 97.5)))
out = pd.DataFrame(rows).round(4).sort_values("PR_AUC", ascending=False)
out.to_csv(rep / f"extra_metrics_{a.set}.csv", index=False); print(out.to_string(index=False))
g = d[d.model == "ANFIS"].copy(); g["bin"] = pd.cut(g.score.clip(-0.2, 1.2), [-0.2, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, 1.2], include_lowest=True)
cal = g.groupby("bin", observed=True).agg(rows=("label", "size"), mean_score=("score", "mean"), observed_rate=("label", "mean")).reset_index()
cal["bin"] = cal["bin"].astype(str); cal.round(4).to_csv(rep / f"calibration_ANFIS_{a.set}.csv", index=False); print(cal.round(3).to_string(index=False))
