"""
final_experiments.py -- Final Review experiment runner (grouped 5-fold, 4 feature sets, 8 models)
===========================================================================================
WHAT IT DOES (plain words)
  For every cow-grouped fold (the 'fold' column made by data_prep.py) it trains each model on the other
  4 folds and predicts the held-out fold. All held-out predictions ("out-of-fold", OOF) are saved, so
  analyze_final.py can compute means, confidence intervals and significance tests from them.
  ANFIS picks its cluster radius `ra` on an INNER validation split of the training cows only
  (never on the held-out fold), then refits on all training cows.
FEATURE SETS
  S5   engineered sensor-only (swelling, asymmetry, temperature, months, breed)       <- Review II "Early-Warning"
  F7   engineered full-clinical (S5 + hardness + pain)                                  <- Review II "Full-Clinical"
  R8   raw 8 sensors
  R13  raw 8 sensors + temperature + months + breed + hardness + pain
MODELS  Majority, LogReg, DecisionTree(d4), RandomForest(300), kNN(5), SVM(RBF), GradientBoosting, ANFIS
RUN     python final_experiments.py --root . --out reports/final
OUT     reports/final/oof_predictions.csv, reports/final/anfis_rules_per_fold.csv, reports/final/RUN_INFO.json
DEFENSE GUIDE
  Why OOF over cow-grouped folds? Every row is predicted by a model that never saw that cow. Pooling the
  five folds gives 1,100 cows of evidence instead of 220, which narrows the confidence intervals.
  Why scale raw features? kNN, SVM, LogReg and the ANFIS Gaussian memberships are scale-sensitive;
  MinMax is fitted on training cows only (no leakage). Trees/forests do not care.
  Why these extra baselines? kNN/SVM/GB are the models the published papers on this dataset used.
"""
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from metrics import evaluate
from train_anfis import RA_GRID, EPOCHS, LR, build

SENS = ["IUFL", "EUFL", "IUFR", "EUFR", "IURL", "EURL", "IURR", "EURR"]
SETS = {
    "S5": ["swelling_index", "asymmetry_index", "temperature_n", "months_pp_n", "breed_enc"],
    "F7": ["swelling_index", "asymmetry_index", "temperature_n", "months_pp_n", "breed_enc", "hardness", "pain"],
    "R8": SENS,
    "R13": SENS + ["temperature", "months_pp", "breed_enc", "hardness", "pain"],
}
ENGINEERED = {"S5", "F7"}  # already scaled by data_prep.py (train-fold-0 scaler); we rescale per fold anyway


def baselines(seed):
    return {
        "Majority": DummyClassifier(strategy="prior"),
        "LogReg": LogisticRegression(max_iter=3000),
        "DecisionTree": DecisionTreeClassifier(max_depth=4, random_state=seed),
        "RandomForest": RandomForestClassifier(300, random_state=seed, n_jobs=2),
        "kNN": KNeighborsClassifier(5),
        "SVM": SVC(kernel="rbf", probability=True, random_state=seed),
        "GradBoost": GradientBoostingClassifier(random_state=seed),
    }


def anfis_fit_predict(Xtr, ytr, cows_tr, folds_tr, Xte):
    inner = np.sort(np.unique(folds_tr))[0]            # one training fold = inner validation (whole cows)
    mi = folds_tr != inner
    best = None
    for ra in RA_GRID:
        m = build(Xtr[mi], ytr[mi], ra).fit(Xtr[mi], ytr[mi], Xtr[~mi], ytr[~mi], epochs=EPOCHS, lr=LR)
        auc = evaluate(ytr[~mi], m.predict_score(Xtr[~mi]), cows_tr[~mi], n_boot=0)["row_auc"]
        if best is None or auc > best[0]:
            best = (auc, ra, m.best_epoch_)
    fm = build(Xtr, ytr, best[1]).fit(Xtr, ytr, epochs=max(best[2], 5), lr=LR)
    return np.asarray(fm.predict_score(Xte)).ravel(), dict(ra=best[1], epochs=max(best[2], 5), rules=fm.R, val_auc=best[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="."); ap.add_argument("--out", default="reports/final")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--sets", default="S5,F7,R8,R13")
    a = ap.parse_args(); root = Path(a.root); out = root / a.out; out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "data/processed/full_processed.csv")
    rows, rules = [], []
    t0 = time.time()
    for fs in a.sets.split(","):
        cols = SETS[fs]
        for f in sorted(df.fold.unique()):
            tr, te = df[df.fold != f], df[df.fold == f]
            sc = MinMaxScaler().fit(tr[cols])
            Xtr, Xte = np.clip(sc.transform(tr[cols]), 0, 1), np.clip(sc.transform(te[cols]), 0, 1)
            ytr = tr.label.values
            preds = {}
            for name, m in baselines(a.seed).items():
                m.fit(Xtr, ytr); preds[name] = m.predict_proba(Xte)[:, 1]
            preds["ANFIS"], info = anfis_fit_predict(Xtr, ytr, tr.cow_id.values, tr.fold.values, Xte)
            rules.append(dict(feature_set=fs, fold=int(f), **info))
            for name, s in preds.items():
                rows.append(pd.DataFrame(dict(feature_set=fs, fold=int(f), model=name, cow_id=te.cow_id.values,
                                              day=te.day.values, label=te.label.values, score=s)))
            print(f"[{time.time()-t0:6.0f}s] {fs} fold {f} done (ANFIS ra={info['ra']}, rules={info['rules']})", flush=True)
    pd.concat(rows).to_csv(out / "oof_predictions.csv", index=False)
    pd.DataFrame(rules).to_csv(out / "anfis_rules_per_fold.csv", index=False)
    json.dump(dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                   seed=a.seed, ra_grid=RA_GRID, epochs=EPOCHS, lr=LR, seconds=round(time.time() - t0)), open(out / "RUN_INFO.json", "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
