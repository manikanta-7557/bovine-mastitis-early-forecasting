"""Unit tests for app_core.py: the dashboard maths must equal the training/evaluation path. Run: python test_app_core.py"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import app_core
from final_experiments import SETS

df = pd.read_csv("data/processed/full_processed.csv"); test = df[df.fold == 0]
for mode, key in app_core.MODES.items():
    b = app_core.load(mode); cols = SETS[key]; assert b["cols"] == cols, "feature order differs from training"
    sub = test.sample(300, random_state=1); lo, hi = b["lo"], b["hi"]
    ref = np.asarray(b["model"].predict_score(np.clip((sub[cols].values - lo) / (hi - lo), 0, 1))).ravel()
    got = np.array([app_core.predict(b, r)["score"] for r in sub[cols].to_dict("records")])
    assert np.abs(ref - got).max() < 1e-3, f"{mode}: app path != model path"
    auc = roc_auc_score(sub.label, got); assert auc > 0.9, f"{mode}: AUC {auc:.3f} too low"
    out = app_core.predict(b, sub[cols].iloc[0].to_dict()); assert len(out["top_rules"]) == 3 and out["tier"] in ("Low", "Watch", "High")
    r = app_core.predict(b, sub[cols].iloc[0].to_dict()); assert len(r["all_rules"]) == b["model"].R
    assert abs(sum(x["weighted_contribution"] for x in r["all_rules"]) - r["score_unrounded"]) < 1e-9, "contributions must sum to the score"
    assert abs(sum(x["firing_share"] for x in r["all_rules"]) - 1) < 1e-9
    ex = app_core.pick_examples(b, df); assert len(ex) == 3 and all(k in ex[n] for n in ex for k in cols)
    far = sub[cols].iloc[0].to_dict(); far[cols[0]] = 9999; assert app_core.predict(b, far)["warnings"], "out-of-range value must warn"
    print(f"OK  {mode}: app path == model path (max diff {np.abs(ref-got).max():.1e}), AUC on 300 test rows {auc:.3f}, rules {b['model'].R}")
print("ALL APP TESTS PASSED")
