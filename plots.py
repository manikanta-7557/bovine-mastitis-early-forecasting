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
    fig, ax = plt.subplots(figsize=(6, 3.2))
    v = [r["weighted_contribution"] for r in rs]
    ax.bar([f"Rule {r['rule']}" for r in rs], v, color=[G if x >= 0 else AMB for x in v])
    for i, x in enumerate(v):
        ax.text(i, x + (0.02 if x >= 0 else -0.06), f"{x:+.3f}", ha="center", fontsize=8)
    ax.axhline(0, color="#33433b", lw=.8)
    ax.set_title(f"Weighted contribution per rule (sum = raw score {score:.3f})", fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=.1, right=.98, top=.82, bottom=.2)
    return fig


def membership_curves(bundle, j, entered=None):
    m = bundle["model"]; lo, hi = bundle["lo"][j], bundle["hi"][j]; span = hi - lo
    xs = np.linspace(lo, hi, 300); xn = (xs - lo) / span
    fig, ax = plt.subplots(figsize=(6, 3.4))
    pal = ["#1f6f54", "#0e7c86", "#b7791f", "#7a5195", "#c0504d", "#4f6d7a"]
    for i in range(m.R):
        ax.plot(xs, np.exp(-0.5 * ((xn - m.c[i, j]) / m.s[i, j]) ** 2), color=pal[i % len(pal)], lw=1.8, label=f"Rule {i + 1}")
    if entered is not None:
        ax.axvline(entered, color="#33433b", ls="--", lw=1); ax.text(entered, 1.03, "entered", fontsize=8, ha="center")
    ax.set_ylim(0, 1.12); ax.set_xlabel(f"{bundle['cols'][j]} (raw units)"); ax.set_ylabel("membership")
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, -0.27))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=.13, right=.98, top=.95, bottom=.34)
    return fig


def forecast_trajectory(traj, thr, day=None):
    """Plot daily synthetic scenario scores, threshold, generated onset, and selected day."""
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.plot(traj.day_index, traj.score, color=G, lw=1.8, label="scenario score (7-day onset)")
    ax.axhline(thr, color=AMB, ls="--", lw=1, label=f"alert threshold {thr:.2f}")
    al = traj[traj.alert]; ax.scatter(al.day_index, al.score, color=AMB, s=14, zorder=3, label="alert")
    on = traj.onset_day.iloc[0] if len(traj) else np.nan
    if not np.isnan(on):
        ax.axvline(on, color="#c0504d", lw=1.4); ax.text(on, ax.get_ylim()[1], " generated clinical onset", color="#c0504d", fontsize=8, va="top", ha="right")
    if day is not None:
        ax.axvline(day, color="#33433b", ls=":", lw=1); ax.text(day, ax.get_ylim()[0], "selected", fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("study day"); ax.set_ylabel("score"); ax.legend(fontsize=7, frameon=False, loc="upper left", ncol=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); return fig


def forecast_results(summ, target, metric="PR_AUC"):
    """Plot synthetic study metrics for history-based D8 and current-value C8 features."""
    p = summ[summ.target == target].pivot(index="model", columns="feature_set", values=metric).sort_values("D8")
    fig, ax = plt.subplots(figsize=(5.2, 3.2)); y = np.arange(len(p))
    ax.barh(y + .2, p["D8"], .38, color=G, label="with own 14-day history (D8)"); ax.barh(y - .2, p["C8"], .38, color=GREY, label="today's values only (C8)")
    ax.set_yticks(y); ax.set_yticklabels(p.index); ax.set_title(f"{metric.replace('_', '-')} on synthetic data, {target.split('_')[-1]} target", fontsize=10, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right"); _clean(ax); fig.tight_layout(); return fig
