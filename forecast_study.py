"""Study the synthetic seven- and fourteen-day clinical-onset targets.

Run: ``python forecast_study.py --data data/external/Synthetic_Data_for_Mastitis_Disease --out reports/forecast``.
The supplied computer-generated dataset contains 300 animals observed for 90 days.
Its results are an isolated demonstration and are not real-animal or farm evidence.

Features use each cow's prior 14-day baseline (D8) or current sensor values only (C8).
Rows are eligible only before clinical onset. Risk labels, post-onset actions, sparse
laboratory fields, and other outcome-derived fields are excluded to avoid leakage.
Evaluation uses five cow-grouped folds, training-only scaling, and validation-only
parameter and threshold selection. All model and summary outputs are written under
``reports/forecast``; the working clinical detection data and artifacts are not used.

DEFENSE GUIDE
  Why keep this study separate? It demonstrates the code path on computer-generated data
    without changing the project's verified task: detection of existing clinical mastitis.
  Why are the results limited? The data generator includes signals before its generated
    onset labels, so results do not establish performance on real animals or farms.
  Why report PR-AUC? The generated targets are imbalanced, making this a useful
    complement to ROC-AUC for this demonstration.
  Why group folds by cow? Multiple daily rows from the same animal must remain together.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from train_anfis import build, EPOCHS, LR

SIGNALS = ["milk_conductivity_ms_cm", "milk_ph", "milk_yield_l", "milk_lactose_pct", "body_temp_c", "udder_surface_temp_c", "rumination_min", "activity_steps"]
BASE_WIN, MIN_PRIOR = 14, 3
RA_GRID = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5]   # first grid [0.4..0.7] hit its lower edge (1 rule = linear model), so it was extended downwards
NEG_PER_POS = 5


def make_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["cow_id", "day_index"]).reset_index(drop=True)
    g = d.groupby("cow_id")
    for s in SIGNALS:
        base = g[s].transform(lambda x: x.shift(1).rolling(BASE_WIN, min_periods=MIN_PRIOR).mean())   # PAST days only
        d["D_" + s] = d[s] - base
        d["C_" + s] = d[s]
    d["n_prior"] = g.cumcount()
    return d


def sets():
    return {"D8": ["D_" + s for s in SIGNALS], "C8": ["C_" + s for s in SIGNALS]}


def baselines(seed):
    return {"LogReg": LogisticRegression(max_iter=3000, class_weight="balanced"), "DecisionTree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=seed),
            "RandomForest": RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=3, random_state=seed, n_jobs=2),
            "kNN": KNeighborsClassifier(5), "GradBoost": GradientBoostingClassifier(random_state=seed)}


def assign_folds(d, seed=42):
    """cow -> fold (0..4), StratifiedGroupKFold by cow, stratified on 'cow ever has onset within 14 days'."""
    cow_ever = d.groupby("cow_id").clinical_within_14d.max(); cows = cow_ever.index.values; cow_fold = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(cows, cow_ever.values, groups=cows)):
        for c in cows[te]: cow_fold[c] = k
    return cow_fold


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[np.argmax(f)]) if len(t) else 0.5


def anfis_fit(Xtr, ytr, cows_tr, rng, Xv, yv, min_rules=1):
    """Fit one ANFIS per radius, pick the best validation PR-AUC. min_rules > 1 restricts the choice to models with at least that
    many rules (used for the deployed explanation demo, where a 1-rule model is just a linear formula); falls back if none qualifies."""
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    keep = np.concatenate([pos, rng.choice(neg, min(len(neg), NEG_PER_POS * len(pos)), replace=False)])
    cands = []
    for ra in RA_GRID:
        m = build(Xtr[keep], ytr[keep], ra).fit(Xtr[keep], ytr[keep], Xv, yv, epochs=EPOCHS, lr=LR)
        cands.append((average_precision_score(yv, m.predict_score(Xv)), ra, m))
    ok = [c for c in cands if c[2].R >= min_rules] or cands
    best = max(ok, key=lambda c: c[0])
    return best[2], best[1]


def cow_bootstrap(df, metric, n=500, seed=42):
    rng = np.random.default_rng(seed); cows = df.cow_id.unique(); idx = {c: np.where(df.cow_id.values == c)[0] for c in cows}
    y, s = df.label.values, df.score.values; v = []
    for _ in range(n):
        pick = np.concatenate([idx[c] for c in rng.choice(cows, len(cows))])
        if y[pick].min() != y[pick].max():
            v.append(metric(y[pick], s[pick]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def alert_stats(df, thr, horizon, onset_day):
    """Per-cow warning metrics. A 'timely alert' = any alert on a day 1..horizon days before onset."""
    out = dict(onset_cows=0, alerted=0, leads=[], healthy_days=0, false_alert_days=0)
    for cow, g in df.groupby("cow_id"):
        if cow in onset_day:
            out["onset_cows"] += 1
            w = g[(g.day_index >= onset_day[cow] - horizon) & (g.day_index < onset_day[cow])]
            hit = w[w.score >= thr]
            if len(hit):
                out["alerted"] += 1; out["leads"].append(int(onset_day[cow] - hit.day_index.min()))
        else:
            out["healthy_days"] += len(g); out["false_alert_days"] += int((g.score >= thr).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="reports/forecast"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models-dir", default="models"); a = ap.parse_args()
    t0 = time.time(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = make_features(pd.read_csv(Path(a.data) / "daily_cow_data.csv"))
    d = d[(d.forecast_eligible == 1) & (d.n_prior >= MIN_PRIOR)].copy()
    full = pd.read_csv(Path(a.data) / "daily_cow_data.csv")
    onset_day = full[full.mastitis_status == 2].groupby("cow_id").day_index.min().to_dict()          # first clinical day per cow
    cow_fold = assign_folds(d, a.seed)
    d["fold"] = d.cow_id.map(cow_fold)
    rows, summ, alerts, rules = [], [], [], []
    for target, horizon in (("clinical_within_7d", 7), ("clinical_within_14d", 14)):
        for fs, cols in sets().items():
            parts = []
            for k in range(5):
                tr, te = d[d.fold != k], d[d.fold == k]
                inner = sorted(tr.fold.unique())[0]; mi = (tr.fold != inner).values
                sc = MinMaxScaler().fit(tr.loc[mi, cols].fillna(0).values)                       # train cows only
                f = lambda x: np.clip(sc.transform(x[cols].fillna(0).values), 0, 1)
                itr, iva = np.where(mi)[0], np.where(~mi)[0]
                Xa, ya = f(tr.iloc[itr]), tr.iloc[itr][target].values
                Xv, yv = f(tr.iloc[iva]), tr.iloc[iva][target].values
                Xte = f(te); rng = np.random.default_rng(a.seed + k)
                preds = {}
                for name, mdl in baselines(a.seed).items():
                    mdl.fit(Xa, ya); preds[name] = (mdl.predict_proba(Xv)[:, 1], mdl.predict_proba(Xte)[:, 1])
                m, ra = anfis_fit(Xa, ya, tr.iloc[itr].cow_id.values, rng, Xv, yv)
                preds["ANFIS"] = (np.asarray(m.predict_score(Xv)).ravel(), np.asarray(m.predict_score(Xte)).ravel())
                rules.append(dict(target=target, feature_set=fs, fold=k, ra=ra, rules=int(m.R)))
                for name, (sv, st) in preds.items():
                    thr = best_f1_threshold(yv, sv)
                    parts.append(pd.DataFrame(dict(target=target, feature_set=fs, fold=k, model=name, cow_id=te.cow_id.values, day_index=te.day_index.values,
                                                   label=te[target].values, score=st, thr=thr)))
            P = pd.concat(parts); rows.append(P)
            for name, g in P.groupby("model"):
                y, s = g.label.values, g.score.values; thr_row = g.thr.values; pred = s >= thr_row
                tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
                prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                al = alert_stats(g.assign(score=s - thr_row + 0.5), 0.5, horizon, onset_day)     # per-fold thresholds aligned to 0.5
                prlo, prhi = cow_bootstrap(g, average_precision_score); aulo, auhi = cow_bootstrap(g, roc_auc_score)
                summ.append(dict(target=target, feature_set=fs, model=name, rows=len(g), positives=int(y.sum()), ROC_AUC=roc_auc_score(y, s), AUC_lo=aulo, AUC_hi=auhi,
                                 PR_AUC=average_precision_score(y, s), PR_lo=prlo, PR_hi=prhi, precision=prec, recall=rec, F1=2 * prec * rec / max(prec + rec, 1e-12),
                                 onset_cows=al["onset_cows"], onset_cows_alerted=al["alerted"], median_lead_days=float(np.median(al["leads"])) if al["leads"] else np.nan,
                                 false_alert_days_per_100=100 * al["false_alert_days"] / max(al["healthy_days"], 1)))
            print(f"{target} {fs} done  {time.time() - t0:.0f}s", flush=True)
    pd.concat(rows).to_csv(out / "oof_forecast.csv", index=False)
    pd.DataFrame(summ).round(4).to_csv(out / "forecast_summary.csv", index=False); pd.DataFrame(rules).to_csv(out / "forecast_anfis_rules.csv", index=False)
    json.dump(dict(seed=a.seed, base_window=BASE_WIN, min_prior=MIN_PRIOR, ra_grid=RA_GRID, neg_per_pos=NEG_PER_POS, signals=SIGNALS, rows_eligible=len(d),
                   cows=int(d.cow_id.nunique()), onset_cows=len(onset_day), seconds=round(time.time() - t0), synthetic=True), open(out / "RUN_INFO.json", "w"), indent=2)
    print(pd.DataFrame(summ).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
