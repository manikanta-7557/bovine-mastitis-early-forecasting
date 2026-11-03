"""
demo.py -- the guided 15-page "Project Demo" (presentation inside the app)
=========================================================================
render(ctx)   draws the current page + the pill navigation.  ctx = dict(bundle=R8 bundle, facts=facts.load_facts(), df=processed df)
Navigation state lives in st.session_state["demo_page"] (0-based); nav_step() is pure and unit-tested.
The Demo ALWAYS uses the Sensor-only (R8) model, independent of the sidebar model choice.
Every number is read from reports/final/*.csv, the saved model, or the processed data: nothing is typed by hand.
DEFENSE GUIDE
  Why a Demo tab?  Faculty can follow the whole project without typing cow data; every page is generated from the repo's real files.
  Why callbacks for the buttons?  The page index changes BEFORE the script re-renders, so the page and the counter never disagree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import inspect

import streamlit as st
import streamlit.components.v1 as components

import app_core
import explain_report as er
import facts as F
import plots
import ui_style as ui
from explain import _label

TOTAL = 15


def nav_step(page: int, delta: int, total: int = TOTAL) -> int:
    """Pure navigation rule: clamp to [0, total-1]."""
    return max(0, min(total - 1, page + delta))


def _stretch(fn):
    """Full-width kwarg that works on old and new Streamlit (use_container_width was replaced by width='stretch')."""
    p = inspect.signature(fn).parameters
    return {"use_container_width": True} if "use_container_width" in p and "width" not in p else ({"width": "stretch"} if "width" in p else {})


def _go(delta=None, to=None):
    st.session_state["demo_page"] = 0 if to is not None else nav_step(st.session_state.get("demo_page", 0), delta)


def _fig(f):
    st.pyplot(f, clear_figure=True)


def _fmt_ci(r, m):
    return f"{r[m]:.3f} ({r[m + '_lo']:.3f} to {r[m + '_hi']:.3f})"


# ------------------------------------------------------------------------------------------------ pages
def p1(c):
    f = c["facts"]
    ui.hero("Interpretable detection of clinical bovine mastitis", "A neuro-fuzzy (ANFIS) research prototype that shows its reasoning, rule by rule")
    a, b, d = st.columns(3)
    with a: ui.kpi("Task", "Detection", "of EXISTING clinical mastitis")
    with b: ui.kpi("Data", f"{f['cows']:,} cows", f"{f['rows']:,} daily records")
    with d: ui.kpi("Model", f"{c['bundle']['model'].R} fuzzy rules", "first-order Sugeno ANFIS")
    st.markdown("**Problem.** Mastitis is inflammation of the cow's udder. Clinical mastitis shows visible signs and is one of the common health problems in dairy farming. "
                "Finding affected cows reliably, and explaining why a cow was flagged, supports the veterinarian's decision.")
    st.markdown("**Objective.** Build a model that takes eight udder-sensor measurements and reports (1) a detection score and (2) the fuzzy rules behind it, so a human can check the reasoning.")
    st.markdown("**Source of the problem.** Smart India Hackathon problem statement SIH26109.")
    ui.callout("<b>Detection, not forecasting.</b> The model recognises existing clinical mastitis in the measurements it is given. It does not predict whether a cow will become ill later.")


def p_flow(c):
    st.header("Project workflow")
    st.markdown("**From udder measurements to an explained risk score**")
    ui.flow([
        ("1. Cow measurements", "Eight raw udder-sensor readings", ""),
        ("2. Validate and scale", "Check ranges; scale with training limits", "alt"),
        ("3. Five-layer ANFIS", "Fuzzy rules combine the scaled readings", ""),
        ("4. Explain the result", "Risk score, rule contributions and report", "amb"),
    ], 2, 245)
    st.caption("Follow the numbered steps left to right, then continue on the next row.")
    a, b = st.columns(2)
    with a:
        ui.card("During model evaluation", "<p>All daily records from a cow stay together in one fold. Scaling limits are learned from training cows only; held-out cows are not used to fit the model.</p>")
    with b:
        ui.card("What the dashboard shows", "<p>A model risk score with its presentation category, the rules that contributed to it, and a structured explanation. The score is not a calibrated clinical probability.</p>")


def p_arch(c):
    m = c["bundle"]["model"]
    st.header("Five-layer ANFIS architecture")
    st.markdown("**Inside the model:** each layer transforms the same cow's scaled sensor values until the rule outputs combine into one score.")
    ui.flow([
        ("Layer 1 · Fuzzify", "Gaussian membership for each input and rule", ""),
        ("Layer 2 · Fire rules", "Multiply memberships within each rule", "alt"),
        ("Layer 3 · Normalize", "Convert rule strengths to shares", ""),
        ("Layer 4 · Consequents", "Compute one linear output per rule", "alt"),
        ("Layer 5 · Output", "Sum each output weighted by its share", "amb"),
    ], 3, 300)
    st.caption("Follow the layer numbers left to right, then continue on the next row.")
    st.caption(f"Shown for the Sensor-only model: {m.d} inputs and {m.R} learned rules. The saved model supplies the rule centres, widths and linear coefficients.")
    ui.callout("Input validation and training-range scaling happen <b>before</b> Layer 1; they are part of preprocessing, not extra ANFIS layers.")
    st.latex(r"\mu_{ij}\ \longrightarrow\ w_i=\prod_j\mu_{ij}\ \longrightarrow\ \bar w_i=\frac{w_i}{\sum_k w_k}\ \longrightarrow\ f_i(x)\ \longrightarrow\ \text{score}=\sum_i\bar w_i f_i(x)")


def p2(c):
    st.header("Problem and motivation")
    a, b = st.columns(2)
    with a:
        ui.card("Why manual observation is not enough", "<ul><li>It depends on frequent, skilled observation of every cow.</li><li>Signs can be missed or recorded inconsistently between people and shifts.</li>"
                "<li>Large herds make individual checks time-consuming.</li></ul>")
        ui.card("Why udder measurements", "<p>The dataset records the size of each of the four quarters at two limits (eight sensor values). Swelling of a quarter changes these sizes, "
                "so they carry information about the udder's condition.</p>")
    with b:
        ui.card("Why interpretable machine learning", "<p>A veterinarian will not act on a bare number. A rule such as 'front-left quarter large AND rear-right quarter large' "
                "can be checked against the animal. ANFIS keeps this rule form while still learning from data.</p>")
        ui.card("Our trade-off, stated honestly", "<p>Black-box models such as random forest score higher in our experiments (page 9). We accept lower accuracy in exchange for readable rules, "
                "and we report the gap instead of hiding it.</p>")


def p3(c):
    f = c["facts"]
    st.header("Dataset overview")
    a, b, d, e = st.columns(4)
    with a: ui.kpi("Cows", f"{f['cows']:,}")
    with b: ui.kpi("Records", f"{f['rows']:,}", "6 days per cow")
    with d: ui.kpi("Model inputs", "8", "udder sensors (Sensor-only)")
    with e: ui.kpi("Target", "class1", "1 = clinical mastitis")
    l, r = st.columns([1, 1])
    with l:
        st.markdown(f"**Class distribution** (records): healthy {f['healthy']:,} ({f['healthy'] / f['rows']:.1%}), mastitis {f['mastitis']:,} ({f['mastitis'] / f['rows']:.1%}).")
        st.bar_chart(pd.DataFrame({"records": [f["healthy"], f["mastitis"]]}, index=["healthy (0)", "mastitis (1)"]), color="#1f6f54")
    with r:
        rows = [[k, v] for k, v in er.SENSOR_MEANING.items() if k.startswith(("I", "E"))]
        st.markdown("**The eight sensor measurements**")
        st.dataframe(pd.DataFrame(rows, columns=["Column", "Meaning (very likely)"]), hide_index=True)
    ui.callout("Source: Mendeley Data record kbvcdw5b4m. The project uses its working clinical mastitis file. <b>Version 2 provenance is pending archive-level verification</b>; "
               "we do not claim it is the officially authenticated Version 2 until the file hash has been compared with the archive.", warn=True)
    st.markdown("**Limitations of the data.** The dataset page does not define the sensor columns (inhale/exhale limit is the likely reading), their unit, or healthy ranges. "
                "Version 1 and the working file differ in labels and readings, and the reason is undocumented. "
                "The same cow contributes six correlated rows, which is why evaluation is grouped by cow. The label comes from the dataset; we did not re-label it.")
    st.caption(f"Counts: {f['source']}.")


def p4(c):
    st.header("System architecture")
    ui.flow([("1. Cow measurements", "eight udder-sensor values entered or loaded", ""), ("2. Input validation", "numbers only; out-of-range values flagged", "alt"),
             ("3. Preprocessing", "min-max scaling with TRAINING limits, clipped to 0-1", ""), ("4. ANFIS model", "five layers, saved trained weights", "alt"),
             ("5. Fuzzy rule activation", "firing strength of every rule", ""), ("6. Rule-level explanation", "share x output = contribution", "alt"),
             ("7. Detection result", "raw score and presentation category", ""), ("8. Veterinary report", "structured, printable explanation", "amb")], 4, 290)
    st.dataframe(pd.DataFrame([
        ["Input validation", "app.py number inputs + app_core.scale", "Rejects non-numbers; warns when a value is outside the training range."],
        ["Preprocessing", "app_core.scale", "(value - training min) / (training max - training min), clipped to [0, 1]. Limits come from training data only."],
        ["ANFIS model", "anfis.py, models/anfis_R8.npz", "Computes memberships, firing strengths, rule outputs and the score."],
        ["Rule activation", "app_core.predict", "Raw firing strength and normalised share for each rule."],
        ["Explanation", "explain_report.analyse", "Per-measurement comparison with rule centres; per-rule contribution; plain-language text built from the numbers."],
        ["Result", "explain.risk_tier", "Low / Watch / High are presentation categories, not clinical thresholds."],
        ["Report", "explain_report.build_report", "Twelve-section report, HTML (print) and PDF."]], columns=["Component", "Code", "Role"]), hide_index=True)


def p5(c):
    st.header("Why ANFIS?")
    a, b, d = st.columns(3)
    with a: ui.card("Fuzzy inference system", "<p>Rules like <i>IF size is LARGE THEN risk is HIGH</i>. 'Large' is soft: a value can be 0.7 large. Readable, but someone must write the rules and tune the curves by hand.</p>")
    with b: ui.card("Neural network", "<p>Learns parameters from data with gradient-based training. Accurate, but its internal numbers have no direct meaning for a veterinarian.</p>")
    with d: ui.card("ANFIS = both", "<p>The structure is a fuzzy rule base, and its curves and outputs are <b>learned from data</b>. After training you can still read the rules.</p>")
    st.markdown("**Why interpretability matters here.** A farm or clinic needs to know <i>why</i> a cow is flagged so the result can be checked against the animal. "
                "Each of this model's rules is written in sensor units and can be inspected one by one.")
    st.markdown("**The variant we use: first-order Sugeno** (Takagi-Sugeno, 1985; ANFIS, Jang, 1993). Each rule's conclusion is a linear formula of the inputs, "
                "so the model can express a sloping relationship with few rules. Price: a rule's output is not limited to 0-1 and is not a probability.")
    st.markdown("**How rules are found:** subtractive clustering (Chiu, 1994) proposes the number and place of rules from the data's density; hybrid learning then fits the model: "
                "least squares for the linear conclusions and Adam gradient steps for the curves' centres and widths.")


def p6(c):
    m = c["bundle"]["model"]; R, d = m.R, m.d
    st.header("ANFIS internal working")
    st.caption("Model configuration from the saved file: "
               f"{R} rules, {d} inputs, {m.c.size} centres + {m.s.size} widths + {m.theta.size} linear coefficients = {m.c.size + m.s.size + m.theta.size} learned numbers.")
    st.markdown("Input scaling happens before the five ANFIS layers; each sensor is put on 0-1 using the training minimum and maximum:")
    st.latex(r"x_j=\mathrm{clip}\!\left(\frac{v_j-\min_j}{\max_j-\min_j},\,0,\,1\right)")
    st.markdown("**Layer 1 — Fuzzification.** Membership says how close value *j* is to rule *i*'s centre *c* (width *s*). 1 = at the centre, towards 0 = far:")
    st.latex(r"\mu_{ij}=\exp\!\left(-\tfrac12\left(\frac{x_j-c_{ij}}{s_{ij}}\right)^2\right)")
    st.markdown("**Layer 2 — Rule firing.** A rule fires strongly only if ALL its conditions match (product):")
    st.latex(r"w_i=\prod_{j=1}^{d}\mu_{ij}")
    st.markdown("**Layer 3 — Normalisation.** Compare each rule's strength with all the others:")
    st.latex(r"\bar w_i=\frac{w_i}{\sum_k w_k}")
    st.markdown("**Layer 4 — Sugeno consequent.** Each rule computes its own linear estimate for this cow:")
    st.latex(r"f_i(x)=\sum_{j=1}^{d}p_{ij}x_j+r_i")
    st.markdown("**Layer 5 — Output.** Add the rule estimates using their normalised firing shares:")
    st.latex(r"\text{score}=\sum_i \bar w_i\,f_i(x)")
    ui.callout("Because the shares are normalised they always add to 1, even when every raw firing strength is tiny. "
               "That is why the app shows both the raw strength and the share.")


def p7(c):
    b = c["bundle"]; m = b["model"]; cols = b["cols"]; lo, hi = b["lo"], b["hi"]; span = hi - lo
    st.header("Fuzzy rules and interpretability")
    st.markdown("These are the **actual rules learned by the saved Sensor-only model** (read from the model file, not written by hand). "
                "Centres and widths are converted back to raw sensor units.")
    cen = pd.DataFrame(np.round(lo + m.c * span, 1), columns=cols, index=[f"Rule {i + 1}" for i in range(m.R)])
    wid = pd.DataFrame(np.round(m.s * span, 1), columns=cols, index=cen.index)
    t1, t2 = st.tabs(["Rule centres", "Rule widths"])
    with t1: st.dataframe(cen)
    with t2: st.dataframe(wid)
    out0 = np.array([np.append(m.c[i], 1.0) @ m.theta[i] for i in range(m.R)])
    for i in range(m.R):
        with st.expander(f"Rule {i + 1} written as IF-THEN"):
            cond = " AND ".join(f"{n} is {_label(n, m.c[i, j])} (about {cen.iloc[i, j]:g}, width {wid.iloc[i, j]:g})" for j, n in enumerate(cols))
            st.write(f"IF {cond}")
            st.caption(f"THEN output = a linear formula of the eight inputs. At the rule's own centre it equals {out0[i]:.2f}; "
                       "for a real cow the output is different (see page 10). LOW / MEDIUM / HIGH = centre in the lower / middle / upper third of the training range: a reading aid, not a clinical grade.")
    j = st.selectbox("Membership curves for measurement", cols, key="demo_mf_feat")
    _fig(plots.membership_curves(b, cols.index(j)))
    st.markdown("**How rules combine:** each rule votes with its own output; its vote is weighted by its firing share. Rules far from the cow get a share near 0 and do not change the score.")


def p8(c):
    f = c["facts"]
    st.header("Experimental methodology")
    st.markdown("**Cow-grouped five-fold cross-validation.** Each cow has six daily rows that look alike. If rows of one cow land in both training and testing, the model is partly tested on cows it has already seen. "
                "We therefore assign all six rows of a cow to the same fold (StratifiedGroupKFold, 5 folds, seed 42).")
    if f["folds"] is not None:
        fo = f["folds"].rename(columns={"fold": "Fold", "cows": "Cows", "rows": "Rows", "mastitis_rows": "Mastitis rows"})
        st.dataframe(fo, hide_index=True)
    st.markdown("- **Training-only preprocessing:** scaling limits are computed from training cows only; the test fold never influences them.\n"
                "- **Tuning on validation data only:** the rule-radius is chosen on an inner validation fold, never on the test fold.\n"
                "- **Metrics:** F1 (balance of precision and recall at score >= 0.5) and ROC-AUC (ranking quality, threshold-free), pooled over all out-of-fold predictions, "
                "with 95% confidence intervals from 1,000 cow-level bootstrap resamples.\n"
                "- **Baselines:** majority class, logistic regression, decision tree (depth 4), random forest (300 trees), kNN (k = 5), SVM (RBF), gradient boosting - on the same folds.\n"
                "- **Significance:** cow-level exact McNemar test (a cow counts as positive if its mean daily score is at least 0.5). p-values are uncorrected and descriptive.")
    ui.callout("<b>Two different kinds of result.</b> Page 9 shows <i>cross-validation</i> results (five models, every cow tested once). "
               "The deployed app model was trained on folds 1-4 and tested once on fold 0 (220 cows); that single-fold number is preliminary and should not replace the cross-validation figure.")


def p9(c):
    f = c["facts"]; r8 = f["r8"]
    st.header("Experimental results (Sensor-only, raw 8 sensors)")
    if r8 is None:
        st.warning("reports/final/summary_pooled.csv not found."); return
    a, b, d, e = st.columns(4)
    an = r8.loc["ANFIS"]
    with a: ui.kpi("ANFIS F1", f"{an.F1:.3f}", f"95% CI {an.F1_lo:.3f} to {an.F1_hi:.3f}")
    with b: ui.kpi("ANFIS ROC-AUC", f"{an.AUC:.3f}", f"95% CI {an.AUC_lo:.3f} to {an.AUC_hi:.3f}")
    rf = f["rules_folds"]
    with d: ui.kpi("Rules per fold", f"{rf.rules.mean():.1f}", f"min {rf.rules.min()}, max {rf.rules.max()}")
    z = f["fold0"]
    with e: ui.kpi("Deployed model, fold 0", f"{z['row_f1']:.3f}" if z else "n/a", f"AUC {z['row_auc']:.3f}; {int(z['rules'])} rules" if z else "")
    l, r = st.columns(2)
    with l: _fig(plots.results_bars(r8, "F1"))
    with r: _fig(plots.results_bars(r8, "AUC"))
    tab = r8.drop(index="Majority").loc[[m for m in F.MODEL_ORDER if m in r8.index and m != "Majority"]]
    t = pd.DataFrame({"Model": tab.index, "F1 (95% CI)": [_fmt_ci(x, "F1") for _, x in tab.iterrows()], "AUC (95% CI)": [_fmt_ci(x, "AUC") for _, x in tab.iterrows()]})
    st.dataframe(t, hide_index=True)
    v = f["vs"]
    if v is not None:
        vt = pd.DataFrame([[k, f"{x.dF1:+.3f} ({x.dF1_lo:+.3f}, {x.dF1_hi:+.3f})", f"{x.mcnemar_p:.4f}", F.verdict(x)] for k, x in v.iterrows() if k != "Majority"],
                          columns=["ANFIS vs", "F1 difference (ANFIS minus other)", "McNemar p (cows)", "Reading"])
        st.markdown("**ANFIS compared with each baseline (same folds)**"); st.dataframe(vt, hide_index=True)
    order = list(r8.drop(index="Majority").sort_values("F1", ascending=False).index)
    st.markdown(f"**Findings.** By F1, ANFIS ranks {order.index('ANFIS') + 1} of {len(order)} classifiers. It is **not** the best classifier: random forest and kNN are higher. "
                "The contribution of ANFIS is a rule base of "
                f"about {rf.rules.mean():.0f} readable rules at a cost in accuracy that the table quantifies.")
    s5 = f["s5_anfis"]
    if s5:
        st.markdown(f"**Feature design matters.** With two engineered indices instead of the eight raw sensors, ANFIS F1 falls to {s5['F1']:.3f} (AUC {s5['AUC']:.3f}), so the app uses the raw sensors.")
    st.markdown("**Limitations.** One dataset; no external farm validation; p-values uncorrected; 28% of test rows are exact copies of training rows even with cow-grouping (unexplained).")
    st.caption("Source: reports/final/summary_pooled.csv, anfis_vs_baselines.csv, anfis_rules_per_fold.csv, fold0_results.csv.")


def _example(c, key):
    ex = c["examples"]
    names = list(ex)
    k = st.radio("Held-out example", names, key=key, horizontal=True)
    row = ex[k]
    return k, row


def p10(c):
    b = c["bundle"]; cols = b["cols"]
    st.header("Live model demonstration")
    st.markdown("A veterinary user enters the eight measurements and presses *Estimate risk* in the Predict section. Here, one of the **held-out test rows** is loaded instead (a genuine row, not typed by hand).")
    k, row = _example(c, "demo_ex10")
    vals = {x: float(row[x]) for x in cols}
    st.caption(f"Row: {row['cow_id']}, day {int(row['day'])}; recorded label: {'mastitis' if row['label'] == 1 else 'healthy'} (shown for the audience; the model never sees it).")
    st.dataframe(pd.DataFrame([vals]), hide_index=True)
    out = app_core.predict(b, vals); an = er.analyse(b, vals, out)
    a, d, e = st.columns(3)
    with a: ui.kpi("Raw score", f"{out['score']:.3f}")
    with d: ui.kpi("Category", out["tier"], "presentation category")
    with e: ui.kpi("Max raw firing", f"{out['max_raw_firing']:.1e}", "weak match" if an["weak"] else "above weak-match level")
    for w in out["warnings"]: st.warning(w)
    l, r = st.columns([3, 2])
    with l:
        st.dataframe(pd.DataFrame({"Rule": [x["rule"] for x in out["all_rules"]], "Raw firing": [f"{x['raw_firing']:.2e}" for x in out["all_rules"]],
                                   "Share": [f"{x['firing_share']:.1%}" for x in out["all_rules"]], "Rule output": [round(x["rule_output_for_this_cow"], 3) for x in out["all_rules"]],
                                   "Contribution": [round(x["weighted_contribution"], 3) for x in out["all_rules"]]}), hide_index=True)
    with r: _fig(plots.contributions_bar(out["all_rules"], out["score_unrounded"]))
    st.markdown("The score is the sum of the contributions: **score = sum( share x rule output )**. A rule with a large output but near-zero share changes almost nothing.")


def p11(c):
    b = c["bundle"]; cols = b["cols"]
    st.header("Veterinary explanation")
    st.markdown("After a prediction, the app turns the same arithmetic into a structured report in plain language. Example for a held-out row:")
    k, row = _example(c, "demo_ex11")
    vals = {x: float(row[x]) for x in cols}
    an = er.analyse(b, vals)
    blocks = er.build_report(an, dict(case=f"{k} - {row['cow_id']}, day {int(row['day'])} (held-out test row)"))
    components.html(er.report_html(blocks), height=620, scrolling=True)
    st.markdown("**How the translation works:** every sentence is generated from the computed numbers (firing shares, outputs, memberships, training ranges). "
                "The report never states a diagnosis, a clinical range or a treatment; its examination points are general reminders for professional verification.")


def p12(c):
    st.header("Limitations and future scope")
    a, b = st.columns(2)
    with a:
        ui.card("Limitations", "<ul><li><b>Dataset:</b> one research dataset of 1,100 cows; sensor units and meaning are not documented.</li>"
                "<li><b>Provenance:</b> Version 2 provenance is pending archive-level verification.</li><li><b>No external farm validation.</b></li>"
                "<li><b>Scores are not probabilities.</b> They are uncalibrated model outputs; Low / Watch / High are presentation categories.</li>"
                "<li><b>Rules:</b> they describe patterns in this data, are first-order (outputs can leave 0-1), and no veterinary expert has validated them.</li>"
                "<li><b>Accuracy:</b> below random forest and kNN on the same data.</li></ul>")
    with b:
        ui.card("Future scope", "<ul><li>Verify the dataset hash against the archive and contact the dataset authors about versions and duplicated rows.</li>"
                "<li>Validate on independent farm data.</li><li>Veterinary review of the learned rules and of the report wording.</li>"
                "<li>Calibrate scores if probabilities are ever to be reported.</li><li>Compare other interpretable models on the same folds.</li></ul>")


def p13(c):
    f = c["facts"]; r8 = f["r8"]
    st.header("Conclusion")
    an = r8.loc["ANFIS"] if r8 is not None else None
    ui.card("What was built", "<p>A first-order Sugeno ANFIS trained on eight raw udder-sensor measurements, a cow-grouped evaluation pipeline with eight models, and a dashboard that explains each prediction rule by rule.</p>")
    if an is not None:
        ui.card("What the experiments showed", f"<p>ANFIS reached F1 {an.F1:.3f} and ROC-AUC {an.AUC:.3f} in cow-grouped cross-validation with about {f['rules_folds'].rules.mean():.0f} rules. "
                "It is competitive with gradient boosting, better than logistic regression, decision tree and SVM, and below random forest and kNN. Feature design influenced results more than the split protocol.</p>")
    ui.card("Why interpretability matters", "<p>Every output can be traced to named rules in sensor units, so a veterinarian can see which rules drove the score and how well they matched the cow.</p>")
    ui.card("Contribution", "<p>A transparent, reproducible comparison of an interpretable neuro-fuzzy detector against standard classifiers under leakage-aware evaluation, plus a working explanation interface. "
            "It is a research prototype, not a diagnostic tool.</p>")


PAGES = [("Project introduction", p1), ("Project workflow", p_flow), ("Five-layer ANFIS architecture", p_arch),
         ("Problem and motivation", p2), ("Dataset overview", p3), ("Application components", p4), ("Why ANFIS?", p5),
         ("ANFIS internal working", p6), ("Fuzzy rules", p7), ("Methodology", p8), ("Results", p9), ("Live demonstration", p10),
         ("Veterinary explanation", p11), ("Limitations and future scope", p12), ("Conclusion", p13)]
assert len(PAGES) == TOTAL


def nav_bar():
    page = st.session_state.get("demo_page", 0)
    try:
        box = st.container(key="demo_nav")
    except TypeError:               # very old Streamlit without keyed containers: still works, just unstyled
        box = st.container()
    with box:
        a, b, d = st.columns([1, 1, 1])
        a.button("‹  Previous", key="demo_prev", on_click=_go, kwargs=dict(delta=-1), disabled=page == 0, **_stretch(st.button))
        b.markdown(f'<div class="cnt">{page + 1:02d} / {TOTAL:02d}</div>', unsafe_allow_html=True)
        d.button("Next  ›", key="demo_next", on_click=_go, kwargs=dict(delta=1), disabled=page == TOTAL - 1, **_stretch(st.button))
    st.markdown(f'<div class="dprog"><div style="width:{(page + 1) / TOTAL:.1%}"></div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([5, 1])
    c2.button("Restart demo", key="demo_restart", on_click=_go, kwargs=dict(to=0))


def render(ctx):
    st.session_state.setdefault("demo_page", 0)
    page = st.session_state["demo_page"]
    st.markdown(f'<div class="dtitle">Project demo - page {page + 1} of {TOTAL}: {PAGES[page][0]}</div>', unsafe_allow_html=True)
    PAGES[page][1](ctx)
    st.divider()
    nav_bar()
