"""
train_final_models.py -- the deployment models (raw-sensor ANFIS) + readable rules in real units
==============================================================================================
TWO MODELS (both trained on cows in folds 1-4, tested ONCE on fold 0 = 220 unseen cows, same protocol as Review II):
  R8   raw 8 udder sensors                                            -> app mode "Sensor-only"
  R13  raw 8 sensors + temperature + months postpartum + breed + hardness + pain   -> app mode "Full-clinical"
STEPS  MinMax scaler fitted on training cows -> radius `ra` chosen on inner validation fold 1 (training cows only)
       -> refit on all training cows -> score fold 0 once.
OUT    models/anfis_R8.npz, models/anfis_R13.npz, models/scaler_R8.json, models/scaler_R13.json
       reports/final/fold0_results.csv, rules_R8.txt, rules_R13.txt (rules written in RAW sensor units), example_explanations_raw.txt
RUN    python train_final_models.py --root .
DEFENSE GUIDE
  Why report rules in raw units? A vet cannot read "IUFL = 0.41"; "IUFL about 232" is checkable against the sensor.
  Why not retrain on all 1,100 cows for the app? Then no cow is left to test. The app model equals the model reported on fold 0.
  The sigma printed per feature is the width of the Gaussian membership: about 68% of the rule's strength lies within +-sigma.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from anfis import ANFIS
from metrics import evaluate
from train_anfis import RA_GRID, EPOCHS, LR, build
from final_experiments import SETS

NAMES = {"R8": "Sensor-only (raw 8 sensors)", "R13": "Full-clinical (raw 13 inputs)"}


def level(v):
    return "LOW" if v < 0.33 else ("MEDIUM" if v < 0.66 else "HIGH")


def rules_text(model, cols, sc):
    lo, hi = sc.data_min_, sc.data_max_
    out = []
    for i in range(model.R):
        parts = []
        for j, c in enumerate(cols):
            raw_c = lo[j] + model.c[i, j] * (hi[j] - lo[j]); raw_s = model.s[i, j] * (hi[j] - lo[j])
            if c in ("breed_enc", "hardness", "pain"):
                parts.append(f"{c} is {'YES' if model.c[i, j] >= .5 else 'NO'}")
            else:
                parts.append(f"{c} is {level(model.c[i, j])} (about {raw_c:.1f}, width {raw_s:.1f})")
        proto = float(np.append(model.c[i], 1.0) @ model.theta[i])
        out.append(f"Rule {i+1}: IF " + " AND ".join(parts) + f" THEN output at rule centre = {proto:.2f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); a = ap.parse_args(); root = Path(a.root)
    (root / "models").mkdir(exist_ok=True); (root / "reports/final").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "data/processed/full_processed.csv")
    train, test = df[df.fold != 0], df[df.fold == 0]
    res, ex = [], []
    for fs in ("R8", "R13"):
        cols = SETS[fs]; sc = MinMaxScaler().fit(train[cols])
        Xtr, Xte = np.clip(sc.transform(train[cols]), 0, 1), np.clip(sc.transform(test[cols]), 0, 1)
        ytr, yte = train.label.values, test.label.values
        mi = (train.fold != 1).values; best = None
        for ra in RA_GRID:
            m = build(Xtr[mi], ytr[mi], ra).fit(Xtr[mi], ytr[mi], Xtr[~mi], ytr[~mi], epochs=EPOCHS, lr=LR)
            auc = evaluate(ytr[~mi], m.predict_score(Xtr[~mi]), train.cow_id.values[~mi], n_boot=0)["row_auc"]
            if best is None or auc > best[0]: best = (auc, ra, m.best_epoch_)
        _, ra, ne = best
        model = build(Xtr, ytr, ra).fit(Xtr, ytr, epochs=max(ne, 5), lr=LR)
        s = np.asarray(model.predict_score(Xte)).ravel()
        r = evaluate(yte, s, test.cow_id.values); res.append(dict(feature_set=fs, ra=ra, rules=model.R, epochs=max(ne, 5), **r))
        model.save(root / f"models/anfis_{fs}.npz")
        json.dump(dict(features=cols, data_min=sc.data_min_.tolist(), data_max=sc.data_max_.tolist(), note="MinMax fitted on training cows (folds 1-4); inputs clipped to [0,1]"),
                  open(root / f"models/scaler_{fs}.json", "w"), indent=2)
        (root / f"reports/final/rules_{fs}.txt").write_text(rules_text(model, cols, sc))
        hi, lo = int(np.argmax(s)), int(np.argmin(s))
        for tag, i in (("highest-score test row", hi), ("lowest-score test row", lo)):
            wbar = model._wbar(Xte[i:i+1])[0]; top = np.argsort(-wbar)[:3]
            ex.append(f"[{fs}] {tag}: cow {test.cow_id.values[i]}, true label {int(yte[i])}, score {s[i]:.3f}; top rules (firing share): " +
                      ", ".join(f"rule {k+1} ({wbar[k]:.2f})" for k in top))
        print(fs, "rules", model.R, "ra", ra, {k: round(v, 4) for k, v in r.items() if k in ("row_f1", "row_auc", "cow_f1", "cow_auc", "row_accuracy")})
    pd.DataFrame(res).to_csv(root / "reports/final/fold0_results.csv", index=False)
    (root / "reports/final/example_explanations_raw.txt").write_text("\n".join(ex))


if __name__ == "__main__":
    main()
