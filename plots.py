"""plots.py -- matplotlib figures for the dashboard (return Figure objects; no Streamlit here)."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

G, T, GREY, AMB = "#1f6f54", "#0e7c86", "#b8c4bd", "#b7791f"


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e5ece8", lw=.8); ax.set_axisbelow(True)


def results_bars(r8, metric="F1", models=None):
    """r8: pooled table indexed by model with F1,F1_lo,F1_hi,AUC,AUC_lo,AUC_hi. Majority excluded (F1 = 0 by construction)."""
    d = r8.drop(index="Majority", errors="ignore").sort_values(metric)
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    cols = [G if m == "ANFIS" else GREY for m in d.index]
    ax.barh(d.index, d[metric], color=cols, xerr=[d[metric] - d[metric + "_lo"], d[metric + "_hi"] - d[metric]], error_kw=dict(ecolor="#33433b", lw=1, capsize=3))
    for i, v in enumerate(d[metric]):
        ax.text(1.02, i, f"{v:.3f}", va="center", fontsize=8, fontweight="bold" if d.index[i] == "ANFIS" else "normal")
    ax.set_xlim(max(0.8 if metric == "AUC" else 0.8, d[metric + "_lo"].min() - 0.03), 1.07)
    ax.set_title(f"{metric} (pooled out-of-fold, 95% CI)", fontsize=10, loc="left"); _clean(ax); fig.tight_layout()
    return fig


def contributions_bar(rules, score):
    rs = sorted(rules, key=lambda r: r["rule"])
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    v = [r["weighted_contribution"] for r in rs]
    ax.bar([f"Rule {r['rule']}" for r in rs], v, color=[G if x >= 0 else AMB for x in v])
    for i, x in enumerate(v):
        ax.text(i, x + (0.02 if x >= 0 else -0.06), f"{x:+.3f}", ha="center", fontsize=8)
    ax.axhline(0, color="#33433b", lw=.8)
    ax.set_title(f"Weighted contribution per rule (sum = raw score {score:.3f})", fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); return fig


def membership_curves(bundle, j, entered=None):
    m = bundle["model"]; lo, hi = bundle["lo"][j], bundle["hi"][j]; span = hi - lo
    xs = np.linspace(lo, hi, 300); xn = (xs - lo) / span
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    pal = ["#1f6f54", "#0e7c86", "#b7791f", "#7a5195", "#c0504d", "#4f6d7a"]
    for i in range(m.R):
        ax.plot(xs, np.exp(-0.5 * ((xn - m.c[i, j]) / m.s[i, j]) ** 2), color=pal[i % len(pal)], lw=1.8, label=f"Rule {i + 1}")
    if entered is not None:
        ax.axvline(entered, color="#33433b", ls="--", lw=1); ax.text(entered, 1.03, "entered", fontsize=8, ha="center")
    ax.set_ylim(0, 1.12); ax.set_xlabel(f"{bundle['cols'][j]} (raw units)"); ax.set_ylabel("membership")
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, -0.28))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); return fig
