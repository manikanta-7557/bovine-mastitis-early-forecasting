"""facts.py -- every number the Demo shows is read here from the repo's own files (no hand-typed metrics).
Fallback constants (rows/cows/classes) are the verified dataset counts, used only if data/processed/full_processed.csv is absent."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FALLBACK = dict(rows=6600, cows=1100, healthy=3961, mastitis=2639)
MODEL_ORDER = ["ANFIS", "RandomForest", "kNN", "GradBoost", "SVM", "DecisionTree", "LogReg", "Majority"]


def _csv(p: Path):
    return pd.read_csv(p) if p.exists() else None


def load_facts(root: Path) -> dict:
    f = {}
    df = _csv(root / "data/processed/full_processed.csv")
    if df is not None:
        f.update(rows=len(df), cows=int(df.cow_id.nunique()), healthy=int((df.label == 0).sum()), mastitis=int((df.label == 1).sum()), source="computed from data/processed/full_processed.csv")
        f["folds"] = df.groupby("fold").agg(cows=("cow_id", "nunique"), rows=("label", "size"), mastitis_rows=("label", "sum")).reset_index()
    else:
        f.update(FALLBACK, source="verified constants (processed file not found)", folds=None)
    f["df"] = df
    p = _csv(root / "reports/final/summary_pooled.csv"); f["pooled"] = p
    f["pooled_all"] = p
    f["r8"] = None if p is None else p[p.feature_set == "R8"].set_index("model")
    f["s5_anfis"] = None if p is None else p[(p.feature_set == "S5") & (p.model == "ANFIS")].iloc[0].to_dict()
    r = _csv(root / "reports/final/anfis_rules_per_fold.csv"); f["rules_folds"] = None if r is None else r[r.feature_set == "R8"].reset_index(drop=True)
    v = _csv(root / "reports/final/anfis_vs_baselines.csv"); f["vs"] = None if v is None else v[v.feature_set == "R8"].set_index("vs_model")
    z = _csv(root / "reports/final/fold0_results.csv"); f["fold0"] = None if z is None or "R8" not in set(z.feature_set) else z[z.feature_set == "R8"].iloc[0].to_dict()
    return f


def verdict(vs_row) -> str:
    """Direction of ANFIS minus other model on F1 from the bootstrap CI (positive = ANFIS higher)."""
    lo, hi = vs_row["dF1_lo"], vs_row["dF1_hi"]
    return "ANFIS higher" if lo > 0 else ("ANFIS lower" if hi < 0 else "no clear difference")
