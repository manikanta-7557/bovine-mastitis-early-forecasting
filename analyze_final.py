"""
analyze_final.py -- turns out-of-fold predictions into the Final Review tables
==========================================================================
IN   reports/final/oof_predictions.csv (from final_experiments.py)
OUT  reports/final/summary_by_fold.csv      mean +- sd of F1/AUC over the 5 folds
     reports/final/summary_pooled.csv       pooled F1/AUC with 95% cow-bootstrap CI (1,000 resamples of cows)
     reports/final/anfis_vs_baselines.csv   paired differences ANFIS - model (cow-bootstrap CI) + exact McNemar (cow level)
     reports/final/RESULTS.md               the same tables as markdown, ready for the report
RUN  python analyze_final.py --root . --out reports/final
DEFENSE GUIDE
  Why bootstrap CIs over cows? Rows of one cow are not independent; resampling whole cows respects that.
  Why McNemar at cow level? A cow is called positive if its mean score over the 6 days >= 0.5; each cow is one
  independent test case, and McNemar only looks at cows where ANFIS and the other model disagree.
  Why not a t-test/Wilcoxon over 5 folds? Only 5 numbers; Wilcoxon's smallest possible p is 0.0625.
  Multiple comparisons: many tests are run; treat p-values as descriptive, not as proof (state this in the report).
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, roc_auc_score

B = 1000
rng = np.random.default_rng(42)
NAMES = {"S5": "Sensor-only (engineered 5)", "F7": "Full-clinical (engineered 7)", "R8": "Raw 8 sensors", "R13": "Raw 8 sensors + clinical (13)"}


def f1auc(y, s):
    return f1_score(y, (s >= .5).astype(int)), roc_auc_score(y, s)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--out", default="reports/final")
    a = ap.parse_args(); out = Path(a.root) / a.out
    d = pd.read_csv(out / "oof_predictions.csv")
    # ---- per-fold mean +- sd
    pf = []
    for (fs, m, f), g in d.groupby(["feature_set", "model", "fold"]):
        F, A = f1auc(g.label.values, g.score.values); pf.append((fs, m, f, F, A))
    pf = pd.DataFrame(pf, columns=["feature_set", "model", "fold", "F1", "AUC"])
    sb = pf.groupby(["feature_set", "model"]).agg(F1_mean=("F1", "mean"), F1_sd=("F1", "std"), AUC_mean=("AUC", "mean"), AUC_sd=("AUC", "std")).round(4).reset_index()
    sb.to_csv(out / "summary_by_fold.csv", index=False)
    # ---- pooled + bootstrap over cows
    cows = d.cow_id.unique(); idx = {c: np.where(d.cow_id.values == c)[0] for c in cows}  # rows of each cow across all (fs, model)
    draws = [rng.choice(len(cows), len(cows), replace=True) for _ in range(B)]
    pooled, diffs, mcn = [], [], []
    for fs, gfs in d.groupby("feature_set"):
        piv = {m: g.sort_values(["cow_id", "day"]).reset_index(drop=True) for m, g in gfs.groupby("model")}
        ref = piv["ANFIS"]; cw = ref.cow_id.values; y = ref.label.values
        ucow = pd.unique(cw); pos = {c: np.where(cw == c)[0] for c in ucow}
        boots = {}
        for m, g in piv.items():
            s = g.score.values; vals = []
            for dr in draws:
                ii = np.concatenate([pos[ucow[k]] for k in dr])
                if len(np.unique(y[ii])) < 2: continue
                vals.append(f1auc(y[ii], s[ii]))
            boots[m] = np.array(vals); F, A = f1auc(y, s)
            pooled.append((fs, m, round(F, 4), *np.percentile(boots[m][:, 0], [2.5, 97.5]).round(4), round(A, 4), *np.percentile(boots[m][:, 1], [2.5, 97.5]).round(4)))
        cowscore = lambda g: g.groupby("cow_id").score.mean()
        cl = ref.groupby("cow_id").label.max(); a_pred = (cowscore(ref) >= .5).astype(int)
        for m, g in piv.items():
            if m == "ANFIS": continue
            n = min(len(boots["ANFIS"]), len(boots[m])); dF = boots["ANFIS"][:n, 0] - boots[m][:n, 0]; dA = boots["ANFIS"][:n, 1] - boots[m][:n, 1]
            b_pred = (cowscore(g) >= .5).astype(int); ok_a = (a_pred == cl).values; ok_b = (b_pred == cl).values
            n01 = int((ok_a & ~ok_b).sum()); n10 = int((~ok_a & ok_b).sum())
            p = 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, .5).pvalue
            diffs.append((fs, m, round((f1auc(y, ref.score.values)[0] - f1auc(y, g.score.values)[0]), 4), *np.percentile(dF, [2.5, 97.5]).round(4),
                          round((f1auc(y, ref.score.values)[1] - f1auc(y, g.score.values)[1]), 4), *np.percentile(dA, [2.5, 97.5]).round(4), n01, n10, round(p, 4)))
    P = pd.DataFrame(pooled, columns=["feature_set", "model", "F1", "F1_lo", "F1_hi", "AUC", "AUC_lo", "AUC_hi"]); P.to_csv(out / "summary_pooled.csv", index=False)
    D = pd.DataFrame(diffs, columns=["feature_set", "vs_model", "dF1", "dF1_lo", "dF1_hi", "dAUC", "dAUC_lo", "dAUC_hi", "cows_ANFIS_right_other_wrong", "cows_other_right_ANFIS_wrong", "mcnemar_p"])
    D.to_csv(out / "anfis_vs_baselines.csv", index=False)
    ru = pd.read_csv(out / "anfis_rules_per_fold.csv").groupby("feature_set").rules.agg(["mean", "min", "max"]).round(1)
    with open(out / "RESULTS.md", "w") as fh:
        fh.write("# Final Review results (grouped 5-fold, pooled out-of-fold, 1,100 cows)\n\n")
        for fs in NAMES:
            if fs not in P.feature_set.values: continue
            fh.write(f"## {NAMES[fs]}\n\n| Model | F1 (95% CI) | AUC (95% CI) | F1 per-fold mean +- sd |\n|---|---|---|---|\n")
            for _, r in P[P.feature_set == fs].iterrows():
                s = sb[(sb.feature_set == fs) & (sb.model == r.model)].iloc[0]
                fh.write(f"| {r.model} | {r.F1:.3f} ({r.F1_lo:.3f}-{r.F1_hi:.3f}) | {r.AUC:.3f} ({r.AUC_lo:.3f}-{r.AUC_hi:.3f}) | {s.F1_mean:.3f} +- {s.F1_sd:.3f} |\n")
            fh.write(f"\nANFIS rules per fold: mean {ru.loc[fs,'mean']}, min {ru.loc[fs,'min']}, max {ru.loc[fs,'max']}.\n\n")
            fh.write("ANFIS minus other model (positive = ANFIS better); McNemar at cow level:\n\n| vs | dF1 (95% CI) | dAUC (95% CI) | ANFIS-only right / other-only right | p |\n|---|---|---|---|---|\n")
            for _, r in D[D.feature_set == fs].iterrows():
                fh.write(f"| {r.vs_model} | {r.dF1:+.3f} ({r.dF1_lo:+.3f}, {r.dF1_hi:+.3f}) | {r.dAUC:+.3f} ({r.dAUC_lo:+.3f}, {r.dAUC_hi:+.3f}) | {r.cows_ANFIS_right_other_wrong} / {r.cows_other_right_ANFIS_wrong} | {r.mcnemar_p:.4f} |\n")
            fh.write("\n")
        fh.write("Notes: CIs are cow-level bootstrap (1,000 resamples). p-values are not corrected for multiple comparisons; treat as descriptive.\n")
    print((out / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
