"""Train the isolated synthetic seven-day onset demonstration model.

The model uses prior-baseline D8 features from ``forecast_study.py``. Training and
validation animals are kept separate from the held-out fold; scaling, radius selection,
and the display threshold use training/validation data only. Outputs stay in the
scenario-specific model and report paths; detection artifacts are not overwritten.

DEFENSE GUIDE
  Why is this model separate? It keeps synthetic study outputs from replacing the
    trained models used for detection of existing clinical mastitis.
  Why validate by cow? Daily records from one animal must not cross data folds.

Run: ``python train_forecast_model.py --data data/external/Synthetic_Data_for_Mastitis_Disease``
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import forecast_study as fs

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--root", default="."); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--min-rules", type=int, default=3, help="deployed model must have at least this many rules (explanation demo)")
a = ap.parse_args(); root = Path(a.root); TARGET = "clinical_within_7d"; cols = fs.sets()["D8"]
d = fs.make_features(pd.read_csv(Path(a.data) / "daily_cow_data.csv")); d = d[(d.forecast_eligible == 1) & (d.n_prior >= fs.MIN_PRIOR)].copy()
cf = fs.assign_folds(d, a.seed); d["fold"] = d.cow_id.map(cf)
tr, te = d[d.fold != 0], d[d.fold == 0]
inner = sorted(tr.fold.unique())[0]; mi = (tr.fold != inner).values; itr, iva = np.where(mi)[0], np.where(~mi)[0]
sc = MinMaxScaler().fit(tr.iloc[itr][cols].fillna(0).values)
f = lambda x: np.clip(sc.transform(x[cols].fillna(0).values), 0, 1)
Xa, ya, Xv, yv = f(tr.iloc[itr]), tr.iloc[itr][TARGET].values, f(tr.iloc[iva]), tr.iloc[iva][TARGET].values
m, ra = fs.anfis_fit(Xa, ya, tr.iloc[itr].cow_id.values, np.random.default_rng(a.seed), Xv, yv, min_rules=a.min_rules)
thr = fs.best_f1_threshold(yv, np.asarray(m.predict_score(Xv)).ravel())
st = np.asarray(m.predict_score(f(te))).ravel(); y = te[TARGET].values; pred = st >= thr
tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
res = dict(min_rules_constraint=a.min_rules, target=TARGET, feature_set="D8", ra=ra, rules=int(m.R), alert_threshold=thr, test_rows=len(te), test_cows=int(te.cow_id.nunique()), positives=int(y.sum()),
           ROC_AUC=roc_auc_score(y, st), PR_AUC=average_precision_score(y, st), precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1))
Path(root / "models").mkdir(exist_ok=True); Path(root / "reports/forecast").mkdir(parents=True, exist_ok=True)
m.save(root / "models/anfis_FC7.npz")
json.dump(dict(features=cols, data_min=sc.data_min_.tolist(), data_max=sc.data_max_.tolist(), alert_threshold=thr, task="forecast", horizon_days=7, synthetic=True, target=TARGET, baseline_window_days=fs.BASE_WIN),
          open(root / "models/scaler_FC7.json", "w"), indent=2)
json.dump(sorted(te.cow_id.unique().tolist()), open(root / "models/forecast_heldout_cows.json", "w"))
pd.DataFrame([res]).round(4).to_csv(root / "reports/forecast/fold0_forecast.csv", index=False); print(pd.DataFrame([res]).round(3).T.to_string())
