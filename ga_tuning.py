"""
ga_tuning.py -- Genetic-algorithm tuning of ANFIS hyper-parameters vs the plain radius grid (Final Review item)
=============================================================================================================
GENES  ra in [0.25, 0.80] (cluster radius), rb_factor in [1.2, 2.0] (how strongly a chosen centre suppresses its
       neighbours), lr in [0.003, 0.03] (Adam step size).
FITNESS validation AUC on ONE inner fold of the TRAINING cows only (never the held-out fold).
GA     population 10, 6 generations, tournament selection (size 3), blend crossover, Gaussian mutation, elitism 2.
COMPARE per outer fold, ANFIS-GA vs ANFIS-grid (the ra grid used everywhere else) on the same held-out cows.
RUN    python ga_tuning.py --root . --set R8 --out reports/final
OUT    reports/final/ga_oof_<set>.csv (OOF predictions of both variants), ga_summary_<set>.csv, ga_best_params_<set>.csv
DEFENSE GUIDE
  Why a GA? The fitness surface is noisy and non-differentiable (clustering is discrete); GA needs no gradients.
  Why compare with grid? If GA does not beat a 5-value grid, say so: extra search cost is not justified.
  Why 60 evaluations only? Each evaluation is a full ANFIS training; the budget is stated so the comparison is fair and reproducible.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from metrics import evaluate
from train_anfis import RA_GRID, EPOCHS, LR, build
from final_experiments import SETS
from subtractive_clustering import initial_rule_params
from anfis import ANFIS

LO = np.array([0.25, 1.2, 0.003]); HI = np.array([0.80, 2.0, 0.03])


def fit_with(g, Xa, ya, Xv=None, yv=None, epochs=EPOCHS):
    C, S = initial_rule_params(Xa, ya, ra=float(g[0]), use_target=True, max_rules=12, rb_factor=float(g[1]))
    return ANFIS(C, S).fit(Xa, ya, Xv, yv, epochs=epochs, lr=float(g[2]))


def ga(Xtr, ytr, cows, rng, pop=10, gens=6):
    P = rng.uniform(LO, HI, (pop, 3)); cache = {}
    def fit(g):
        k = tuple(np.round(g, 4))
        if k not in cache:
            try:
                m = fit_with(g, Xtr[mi], ytr[mi], Xtr[~mi], ytr[~mi])
                cache[k] = (evaluate(ytr[~mi], m.predict_score(Xtr[~mi]), cows[~mi], n_boot=0)["row_auc"], m.best_epoch_)
            except Exception:
                cache[k] = (0.0, 5)
        return cache[k]
    for _ in range(gens):
        F = np.array([fit(g)[0] for g in P]); order = np.argsort(-F); new = [P[i].copy() for i in order[:2]]
        while len(new) < pop:
            a = max(rng.choice(pop, 3, replace=False), key=lambda i: F[i]); b = max(rng.choice(pop, 3, replace=False), key=lambda i: F[i])
            w = rng.uniform(-.25, 1.25, 3); child = w * P[a] + (1 - w) * P[b] + rng.normal(0, .05, 3) * (HI - LO)
            new.append(np.clip(child, LO, HI))
        P = np.array(new)
    F = np.array([fit(g)[0] for g in P]); b = int(np.argmax(F)); return P[b], fit(P[b])[1], F[b], len(cache)


def main():
    global mi
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--set", default="R8"); ap.add_argument("--out", default="reports/final")
    a = ap.parse_args(); root = Path(a.root); out = root / a.out
    df = pd.read_csv(root / "data/processed/full_processed.csv"); cols = SETS[a.set]; rng = np.random.default_rng(42)
    rows, best_rows = [], []
    for f in sorted(df.fold.unique()):
        tr, te = df[df.fold != f], df[df.fold == f]; sc = MinMaxScaler().fit(tr[cols])
        Xtr, Xte = np.clip(sc.transform(tr[cols]), 0, 1), np.clip(sc.transform(te[cols]), 0, 1); ytr = tr.label.values
        inner = np.sort(tr.fold.unique())[0]; mi = (tr.fold != inner).values
        # grid
        gb = None
        for ra in RA_GRID:
            m = build(Xtr[mi], ytr[mi], ra).fit(Xtr[mi], ytr[mi], Xtr[~mi], ytr[~mi], epochs=EPOCHS, lr=LR)
            auc = evaluate(ytr[~mi], m.predict_score(Xtr[~mi]), tr.cow_id.values[~mi], n_boot=0)["row_auc"]
            if gb is None or auc > gb[0]: gb = (auc, ra, m.best_epoch_)
        fm = build(Xtr, ytr, gb[1]).fit(Xtr, ytr, epochs=max(gb[2], 5), lr=LR); sg = np.asarray(fm.predict_score(Xte)).ravel()
        # GA
        g, ne, fit_auc, nev = ga(Xtr, ytr, tr.cow_id.values, rng)
        gm = fit_with(g, Xtr, ytr, epochs=max(ne, 5)); ss = np.asarray(gm.predict_score(Xte)).ravel()
        for name, s, R in (("ANFIS-grid", sg, fm.R), ("ANFIS-GA", ss, gm.R)):
            rows.append(pd.DataFrame(dict(fold=f, model=name, cow_id=te.cow_id.values, day=te.day.values, label=te.label.values, score=s)))
        best_rows.append(dict(fold=f, grid_ra=gb[1], grid_val_auc=round(gb[0], 4), grid_rules=fm.R, ga_ra=round(g[0], 3), ga_rb=round(g[1], 3), ga_lr=round(g[2], 4),
                              ga_val_auc=round(fit_auc, 4), ga_rules=gm.R, ga_evaluations=nev, grid_evaluations=len(RA_GRID)))
        print("fold", f, best_rows[-1], flush=True)
    d = pd.concat(rows); d.to_csv(out / f"ga_oof_{a.set}.csv", index=False); pd.DataFrame(best_rows).to_csv(out / f"ga_best_params_{a.set}.csv", index=False)
    s = []
    for (m, f), g in d.groupby(["model", "fold"]):
        s.append((m, f, f1_score(g.label, g.score >= .5), roc_auc_score(g.label, g.score)))
    s = pd.DataFrame(s, columns=["model", "fold", "F1", "AUC"]).groupby("model").agg(F1=("F1", "mean"), F1_sd=("F1", "std"), AUC=("AUC", "mean"), AUC_sd=("AUC", "std")).round(4)
    s.to_csv(out / f"ga_summary_{a.set}.csv"); print(s)


if __name__ == "__main__":
    main()
