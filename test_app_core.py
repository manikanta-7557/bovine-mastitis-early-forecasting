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
    pools = app_core.pick_example_pools(b, df)
    assert set(pools) == set(app_core.EXAMPLE_SCORE_RANGES)
    heldout = df[df.fold == 0]
    heldout_x = np.clip((heldout[cols].values - b["lo"]) / (b["hi"] - b["lo"]), 0, 1)
    heldout_scores = np.asarray(b["model"].predict_score(heldout_x)).ravel()
    for name, (low, high) in app_core.EXAMPLE_SCORE_RANGES.items():
        pool = pools[name]
        assert pool, f"{mode}: no real records for {name}"
        vectors = [tuple(sample["row"][c] for c in cols) for sample in pool]
        assert len(vectors) == len(set(vectors)), f"{mode}: duplicate measurements in {name} pool"
        assert all(low <= sample["score"] <= high for sample in pool), f"{mode}: out-of-band {name} sample"
        heldout_match_exists = bool(((heldout_scores >= low) & (heldout_scores <= high)).any())
        assert all(sample["held_out"] == heldout_match_exists for sample in pool), (
            f"{mode}: use held-out rows when available and training rows only as a fallback"
        )
        for sample in pool[:3]:
            actual = app_core.predict(b, sample["row"])
            assert abs(actual["score_unrounded"] - sample["score"]) < 1e-9
        cursor = 0
        first_vectors = []
        for _ in range(len(pool)):
            sample, cursor = app_core.next_example(pool, cursor)
            first_vectors.append(tuple(sample["row"][c] for c in cols))
        assert len(first_vectors) == len(set(first_vectors))
        assert cursor == 0
        cycled, cursor = app_core.next_example(pool, cursor)
        assert tuple(cycled["row"][c] for c in cols) == vectors[0]
    far = sub[cols].iloc[0].to_dict(); far[cols[0]] = 9999; assert app_core.predict(b, far)["warnings"], "out-of-range value must warn"
    print(f"OK  {mode}: app path == model path (max diff {np.abs(ref-got).max():.1e}), AUC on 300 test rows {auc:.3f}, rules {b['model'].R}")
print("ALL APP TESTS PASSED")
