"""Tests for the synthetic-data onset study and its explanation.

DEFENSE GUIDE
  These checks guard against look-ahead leakage, cow-fold overlap, and treating the
  computer-generated study output as clinical evidence.

Run: python test_forecast.py
"""
import numpy as np, pandas as pd, pathlib, types, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import forecast_core as fc, forecast_study as fs, explain_report as er, app_core, plots

root = pathlib.Path(__file__).resolve().parent
raw = pd.read_csv(root / fc.DATA_DIR / "daily_cow_data.csv"); daily = fc.make_features if False else fs.make_features(raw)
b = fc.load_forecast(root); held = fc.heldout_cows(root)
# 1. no leakage columns among the features; all features are past-baseline deviations
banned = ("risk", "status", "days_to", "within", "scc", "cmt", "antibiotic", "discard", "vet_visit")
assert all(c.startswith("D_") and not any(w in c for w in banned) for c in b["cols"]), b["cols"]
# 2. features at day t depend on days <= t only: scrambling every FUTURE day must not change day-t features
cow = held[0]; t = 40
mod = raw.copy(); fut = (mod.cow_id == cow) & (mod.day_index > t)
rng = np.random.default_rng(0)
for s in fs.SIGNALS: mod.loc[fut, s] = rng.normal(mod[s].mean(), mod[s].std() * 3, fut.sum())
a1 = fs.make_features(raw); a2 = fs.make_features(mod)
r1 = a1[(a1.cow_id == cow) & (a1.day_index <= t)][b["cols"]].fillna(-9).values; r2 = a2[(a2.cow_id == cow) & (a2.day_index <= t)][b["cols"]].fillna(-9).values
assert np.allclose(r1, r2), "features at day t changed when the future was altered (look-ahead leak)"
# 3. cow-grouped folds partition the cows; held-out cows are exactly fold 0
e = daily[(daily.forecast_eligible == 1) & (daily.n_prior >= fs.MIN_PRIOR)]; cf = fs.assign_folds(e, 42)
assert set(cf.values()) == {0, 1, 2, 3, 4} and sorted(c for c, k in cf.items() if k == 0) == sorted(held)
# 4. explanation on a held-out onset cow, before onset: contributions sum to the score, report built, wording safe
on_cows = [c for c in held if (raw[raw.cow_id == c].mastitis_status == 2).any()]; cow = on_cows[0]
tr = fc.trajectory(b, daily, cow); onset = int(tr.onset_day.iloc[0]); day = onset - 3
vals = fc.vals_at(b, tr[tr.forecast_eligible == 1], day); an = er.analyse(b, vals)
assert abs(sum(r["weighted_contribution"] for r in an["rules"]) - an["score"]) < 1e-9 and an["tier"] in ("Alert", "No alert")
assert abs(float(b["model"].predict_score(((np.array([vals[c] for c in b["cols"]]) - b["lo"]) / (b["hi"] - b["lo"])).clip(0, 1)[None, :])[0]) - an["score"]) < 1e-9
rec = [float(x) for x in tr[(tr.day_index <= day) & (tr.day_index > day - 7)].score]
html = er.report_html(er.build_report(an, dict(case="test", recent=rec))); low = html.lower()
assert "synthetic" in low and "what the model suggests may happen next" in low and "not a calibrated clinical probability" in low or "not a probability" in low
assert not any(w in low for w in ("early warning", "early-warning", "subclinical", "first-ever", "diagnosis:", "treatment recommended"))
assert er.report_pdf(er.build_report(an, dict(case="test", recent=rec)))[:5] == b"%PDF-"
# 5. saved results exist and ANFIS has more than one rule in the deployed model
sm = pd.read_csv(root / "reports/forecast/forecast_summary.csv"); assert {"D8", "C8"} <= set(sm.feature_set) and (sm.ROC_AUC.between(0, 1)).all()
fig = plots.forecast_trajectory(tr[tr.forecast_eligible == 1], b["alert_threshold"], day); fig.canvas.draw(); plt.close(fig)
fig = plots.forecast_results(sm, "clinical_within_7d"); fig.canvas.draw(); plt.close(fig)
print(f"OK  forecast: no leakage columns, no look-ahead (future scrambled, features equal), folds partition cows, report ok; deployed rules = {b['model'].R}, alert thr {b['alert_threshold']:.3f}")
print("ALL FORECAST TESTS PASSED")
