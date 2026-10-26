"""
leakage_replication.py - how much of the "99% accuracy" in earlier papers is leakage?

Usage:
    python leakage_replication.py data/raw/clinical_mastitis_cows.csv
    python leakage_replication.py data/raw/clinical_mastitis_cows.csv --seeds 5 --out reports/leakage_replication.csv

Why this exists (plain English)
-------------------------------
Two published papers use this same dataset (8 udder sensor columns + Temperature) and report ~99% accuracy:
  * MasPA (Ghafoor & Sitkowska 2021, AgriEngineering 3(3):575-583): random forest, 99.12% on a random 80/20
    split of the OVERSAMPLED data (3,961 healthy / 2,639 mastitis -> 3,961 / 3,961), 10-fold CV 98.10%.
  * Kumar, Sindhwani & Sachdeva (2023): same recipe, KNN/RF about 99.4%.
Two things in that recipe let the test set "see" the training set:
  1. OVERSAMPLE BEFORE SPLITTING: oversampling copies mastitis rows; after a random split, copies of the
     same row sit in both train and test, so the model is tested on rows it has already seen.
  2. ROW-LEVEL SPLIT: each cow has 6 daily rows with similar sensor values; a random split puts some of a cow's
     rows in train and others in test, so the model can recognise the cow instead of the disease.
Our protocol splits by COW and (if it balances at all) balances the TRAINING part only.

What this script does: runs the same 9 features and the same models under four protocols and reports the
accuracy / F1 / AUC of each, so the gap between "paper-style" and "cow-grouped" is a measured number.

  P1 paper-style      : oversample everything -> random 80/20 row split          (MasPA / Kumar recipe)
  P2 row split        : random 80/20 row split, no oversampling
  P3 cow-grouped      : cow-grouped 80/20 split, no oversampling                 (our protocol)
  P4 cow-grouped + OS : cow-grouped split, oversample the TRAIN part only        (correct way to balance)

Defense guide
-------------
Q: Why do you report lower numbers than the literature on the same data?
A: Because the literature's number comes from a test set that overlaps the training set (copies of rows and
   other days of the same cow). P1 vs P3 in this table measures that overlap. We do not claim the papers are
   wrong about the sensors being informative; we claim their number does not measure detection of an UNSEEN cow.
Q: Is P3 the "true" performance?
A: It is the performance for a new cow from the same farm/dataset. It does not tell us about other farms (C10).
Q: Why not just use P4?
A: P3 vs P4 shows whether balancing helps at all. The label is 40% mastitis, which is not extreme.
Rejected alternatives: SMOTE (also synthesises near-copies; same leakage if done before the split);
class weights (fine, but changes nothing about the split problem).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SENSORS = ["IUFL", "EUFL", "IUFR", "EUFR", "IURL", "EURL", "IURR", "EURR"]
FEATURES = SENSORS + ["Temperature"]          # the MasPA / Kumar feature set (no Hardness, Pain, Milk_visibility)
LABEL, COW = "class1", "Cow_ID"


def models(seed):
    # RF parameters follow MasPA's Table 2 (entropy, 100 trees); the others are scikit-learn defaults
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=seed, n_jobs=-1),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "KNN(k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }


def oversample_idx(y, rng):
    """Indices that duplicate minority rows (sampling with replacement) until both classes are equal."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    pos, neg = idx[y == 1], idx[y == 0]
    small, big = (pos, neg) if len(pos) < len(neg) else (neg, pos)
    extra = rng.choice(small, size=len(big) - len(small), replace=True)
    return np.concatenate([idx, extra])


def dup_fraction(Xtr, Xte):
    """Share of test rows whose exact 9-number feature vector also appears in the training rows."""
    train_keys = set(map(tuple, np.round(Xtr, 6)))
    return float(np.mean([tuple(r) in train_keys for r in np.round(Xte, 6)]))


def evaluate(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    return accuracy_score(yte, pred), f1_score(yte, pred), roc_auc_score(yte, p)


def split_row(X, y, seed):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
    return tr, te


def split_cow(y, cows, seed):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, te = next(sgkf.split(np.zeros(len(y)), y, cows))
    return tr, te


def run(df, seeds):
    X = df[FEATURES].to_numpy(float)
    y = df[LABEL].to_numpy(int)
    cows = df[COW].to_numpy()
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        # P1: oversample the WHOLE data first, then split randomly (the published recipe)
        oi = oversample_idx(y, rng)
        Xo, yo, co = X[oi], y[oi], cows[oi]
        tr, te = split_row(Xo, yo, seed)
        leak1 = (dup_fraction(Xo[tr], Xo[te]), float(np.isin(co[te], co[tr]).mean()))
        protocols = {"P1 paper-style (oversample, then random row split)": (Xo, yo, tr, te, leak1)}
        # P2: random row split
        tr, te = split_row(X, y, seed)
        protocols["P2 random row split, no oversampling"] = (X, y, tr, te, (dup_fraction(X[tr], X[te]), float(np.isin(cows[te], cows[tr]).mean())))
        # P3 / P4: cow-grouped
        tr, te = split_cow(y, cows, seed)
        leak3 = (dup_fraction(X[tr], X[te]), float(np.isin(cows[te], cows[tr]).mean()))
        protocols["P3 cow-grouped split, no oversampling (ours)"] = (X, y, tr, te, leak3)
        trb = tr[oversample_idx(y[tr], rng)]
        protocols["P4 cow-grouped, oversample TRAIN only"] = (X, y, trb, te, leak3)
        for pname, (XX, yy, tr_i, te_i, leak) in protocols.items():
            for mname, m in models(seed).items():
                acc, f1, auc = evaluate(m, XX[tr_i], yy[tr_i], XX[te_i], yy[te_i])
                rows.append(dict(protocol=pname, model=mname, seed=seed, accuracy=acc, f1=f1, auc=auc,
                                 test_rows_exactly_in_train=leak[0], test_rows_from_seen_cows=leak[1]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="reports/leakage_replication.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    df.columns = df.columns.str.strip()
    missing = [c for c in FEATURES + [LABEL, COW] if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns: {missing}")
    print(f"{len(df)} rows, {df[COW].nunique()} cows, class1 counts {df[LABEL].value_counts().sort_index().to_dict()}")
    res = run(df, list(range(42, 42 + a.seeds)))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res.to_csv(a.out, index=False)
    g = res.groupby(["protocol", "model"])[["accuracy", "f1", "auc"]].agg(["mean", "std"]).round(3)
    print("\nMean and std over", a.seeds, "seeds (features = 8 sensors + Temperature)\n")
    print(g.to_string())
    leak = res.groupby("protocol")[["test_rows_exactly_in_train", "test_rows_from_seen_cows"]].mean().round(3)
    print("\nLeakage diagnostics (fraction of TEST rows)\n")
    print(leak.to_string())
    best = res[res.model == "RandomForest"].groupby("protocol")["accuracy"].mean()
    p1 = best.filter(like="P1").iloc[0]
    p3 = best.filter(like="P3").iloc[0]
    print(f"\nRandomForest accuracy: paper-style {p1:.3f} vs cow-grouped {p3:.3f} -> inflation of {100 * (p1 - p3):.1f} points")
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
