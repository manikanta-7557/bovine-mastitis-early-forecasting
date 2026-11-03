"""
app.py -- Streamlit dashboard for the interpretable ANFIS (research prototype, NOT a diagnostic tool)
Run:  streamlit run app.py
Sections: Predict | Demo (15-page guided presentation) | How it works | Model facts.
All maths lives in app_core.py / explain_report.py (unit-tested). demo.py holds the Demo pages; ui_style.py the look.
"""
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import app_core
import demo
import explain_report as er
import facts as F
import plots
import ui_style as ui

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Mastitis detection with ANFIS", layout="wide")
ui.inject()


@st.cache_resource
def get_bundle(mode):
    return app_core.load(mode)


@st.cache_data
def get_df():
    p = ROOT / "data/processed/full_processed.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def get_facts():
    return F.load_facts(ROOT)


@st.cache_data
def get_examples(mode):
    d = get_df()
    return None if d is None else app_core.pick_examples(get_bundle(mode), d)


UNITS = {"temperature": "dataset scale (unit not documented)", "months_pp": "months after calving", "breed_enc": "0 = Jersey, 1 = Holstein",
         "hardness": "0 = normal, 1 = hard udder", "pain": "0 = no pain, 1 = pain"}

with st.sidebar:
    st.markdown("### Bovine health research")
    st.caption("Interpretable model workspace")
    mode = st.radio("Feature set", list(app_core.MODES), key="model_mode",
                    help="Sensor-only: the 8 udder sensors. Full-clinical: adds temperature, months postpartum, breed, hardness, pain.")
    st.info("Research prototype. Risk scores are not a veterinary diagnostic tool.")
    st.caption("Sensor-only uses eight raw udder sensors. Full-clinical adds recorded clinical features.")
bundle = get_bundle(mode); cols = bundle["cols"]; lo, hi = bundle["lo"], bundle["hi"]

ui.hero("Existing clinical mastitis detection", "An interpretable ANFIS research dashboard for sensor-only and full-clinical risk scores.")
section = st.radio("Section", ["Predict", "Demo", "How it works", "Model facts"], horizontal=True, key="section", label_visibility="collapsed")


# ---------------------------------------------------------------------------------------------- Predict
def show_analysis(an):
    out = an["out"]
    for w in out["warnings"]:
        st.warning(w)
    st.subheader("A. Prediction summary")
    a, b, c = st.columns(3)
    with a: ui.kpi("Presentation category", out["tier"], "not a clinical threshold")
    with b: ui.kpi("Raw model score", f"{out['score']:.3f}", "sum of rule contributions")
    with c: ui.kpi("Displayed score (0-1)", f"{out['score_clipped']:.2f}", "clipped for display" if out["clipped"] else "same as raw")
    st.progress(out["score_clipped"])
    ui.callout("The score is a <b>model output, not a calibrated clinical probability</b>. Low / Watch / High are presentation categories.")
    for s in an["narrative"]["summary"]:
        st.write(s)

    st.subheader("B. Input measurement analysis")
    st.caption(f"Comparison with the dominant rule (Rule {an['dominant']}) and with the closest rule per measurement. Membership 1 = at the rule centre. No clinical reference ranges are implied.")
    st.dataframe(pd.DataFrame([{"Measurement": m["feature"], "Entered": m["entered"], "Training range": f"{m['lo']:g} to {m['hi']:g}" + ("" if m["in_range"] else "  (OUTSIDE: clipped)"),
                                f"Rule {an['dominant']} centre": round(m["dom_centre"], 1), "Difference": round(m["dom_diff"], 1), "Membership": round(m["dom_mu"], 2),
                                "Closest rule": m["best_rule"], "Its centre": round(m["best_centre"], 1), "Its membership": round(m["best_mu"], 2)} for m in an["measurements"]]), hide_index=True)
    with st.expander("Membership of every measurement in every rule"):
        st.dataframe(pd.DataFrame(an["mu"].round(3), columns=an["cols"], index=[f"Rule {i + 1}" for i in range(len(an["mu"]))]))

    st.subheader("C. ANFIS rule analysis")
    st.dataframe(pd.DataFrame({"Rule": [r["rule"] for r in out["all_rules"]], "Raw firing strength": [f"{r['raw_firing']:.2e}" for r in out["all_rules"]],
                               "Firing share": [f"{r['firing_share']:.1%}" for r in out["all_rules"]], "Rule output (this cow)": [round(r["rule_output_for_this_cow"], 3) for r in out["all_rules"]],
                               "Contribution": [round(r["weighted_contribution"], 3) for r in out["all_rules"]]}), hide_index=True)
    st.latex(r"\text{Contribution}_i=\bar w_i\,f_i(x)\qquad \text{Score}=\sum_i \bar w_i\,f_i(x)")
    st.caption(f"Contributions add up to the raw score: {sum(r['weighted_contribution'] for r in out['all_rules']):.3f}. Rule outputs are linear (Sugeno) estimates, not probabilities, "
               "and can be below 0 or above 1. Raw firing strength = how well one rule matches; share = how it compares with the other rules, so it can read 100% even when the raw strength is tiny.")
    left, right = st.columns([3, 2])
    with right: st.pyplot(plots.contributions_bar(out["all_rules"], out["score_unrounded"]), clear_figure=True)
    with left:
        for r in out["all_rules"]:
            with st.expander(f"Rule {r['rule']}  -  share {r['firing_share']:.1%}, contribution {r['weighted_contribution']:+.3f}"):
                st.dataframe(pd.DataFrame(r["conditions"]).rename(columns={"centre": "rule centre", "width": "width (+-)", "your_value": "your value"}), hide_index=True)

    st.subheader("D. Why the model produced this result")
    n = an["narrative"]
    for s in n["contributions"] + n["closeness"] + n["activation"]:
        st.write("• " + s)
    st.caption("Fuzzy memberships of the score: " + ", ".join(f"{k} {v:.2f}" for k, v in out["tier_memberships"].items()))

    st.subheader("Veterinary report")
    if st.button("Generate Detailed Veterinary Explanation", type="primary", key="gen_report"):
        st.session_state["report_for"] = id(an)
    if st.session_state.get("report_for") == id(an):
        blocks = er.build_report(an)
        html = er.report_html(blocks)
        components.html(html, height=640, scrolling=True)
        d1, d2 = st.columns(2)
        d1.download_button("Download report (HTML, opens print dialog)", html, file_name="anfis_veterinary_explanation.html", mime="text/html")
        pdf = er.report_pdf(blocks)
        if pdf:
            d2.download_button("Download report (PDF)", pdf, file_name="anfis_veterinary_explanation.pdf", mime="application/pdf")
        else:
            d2.caption("PDF needs `pip install reportlab`; use the HTML report and Print > Save as PDF.")


if section.endswith("Predict"):
    ex = get_examples(mode)
    if ex:
        st.caption("Fill the form with a row from the held-out test cows (chosen by score, not clinically 'typical'):")
        cc = st.columns(len(ex))
        for k, (name, row) in enumerate(ex.items()):
            if cc[k].button(name, key=f"ex_{k}"):
                for c in cols:
                    st.session_state[f"in_{c}"] = float(row[c])
                st.session_state["example_note"] = f"{name}: this row's recorded label is {'mastitis' if row['label'] == 1 else 'healthy'}."
    if "example_note" in st.session_state:
        st.info(st.session_state["example_note"])
    st.subheader("Inputs")
    store = st.session_state.setdefault("input_store", {})        # survives switching sections (widget state alone would be dropped)
    vals = {}
    grid = st.columns(4)
    for i, c in enumerate(cols):
        j = cols.index(c); default = float(round((lo[j] + hi[j]) / 2, 1))
        st.session_state.setdefault(f"in_{c}", store.get(c, default))
        with grid[i % 4]:
            if c in ("breed_enc", "hardness", "pain"):
                vals[c] = float(st.selectbox(f"{c}  ({UNITS[c]})", [0.0, 1.0], key=f"in_{c}"))
            else:
                label = c + (f"  ({UNITS[c]})" if c in UNITS else "") + f"  [train {lo[j]:g} to {hi[j]:g}]"
                vals[c] = st.number_input(label, step=1.0, key=f"in_{c}")
        store[c] = vals[c]
    if st.button("Estimate risk", type="primary"):
        with st.spinner("Running the ANFIS rules..."):
            st.session_state["analysis"] = dict(mode=mode, vals=dict(vals), an=er.analyse(bundle, vals))
            st.session_state.pop("report_for", None)
    cur = st.session_state.get("analysis")
    if cur:
        if cur["mode"] != mode or cur["vals"] != vals:
            st.warning("Inputs or model changed since this result was calculated. Press 'Estimate risk' to update.")
        show_analysis(cur["an"])
    else:
        st.info("Enter the measurements (or load an example) and press 'Estimate risk'.")

# ---------------------------------------------------------------------------------------------- Demo
elif section.endswith("Demo"):
    d = get_df(); sb = get_bundle("Sensor-only")
    demo.render(dict(bundle=sb, facts=get_facts(), examples=None if d is None else app_core.pick_examples(sb, d)))

# ---------------------------------------------------------------------------------------------- How it works
elif section.endswith("How it works"):
    st.header("How it works")
    ui.flow([("Data input", "8 sensor values (+5 clinical fields in Full-clinical)", ""), ("Validation", "numbers; warn if outside training range", "alt"),
             ("Scaling", "training min-max, clip to 0-1", ""), ("Membership", "Gaussian curve per rule and measurement", "alt"),
             ("Rule firing", "product of memberships, then normalise", ""), ("Consequent", "linear output of each rule", "alt"),
             ("Final score", "sum of share x output", ""), ("Explanation", "text and report from the same numbers", "amb")], 4, 290)
    t = st.tabs(["Input and scaling", "Membership and rules", "Output and score", "Explanation"])
    with t[0]:
        st.markdown("Every measurement is put on a 0-1 scale with the **training** minimum and maximum, so large-valued sensors do not dominate. "
                    "A value outside the training range is clipped and a warning is shown, because the model was never trained there.")
        st.latex(r"x_j=\mathrm{clip}\!\left(\frac{v_j-\min_j}{\max_j-\min_j},0,1\right)")
    with t[1]:
        st.markdown("Each rule has, for every measurement, a bell curve with a centre and a width. The value's **membership** says how close it is to that centre. "
                    "A rule's **raw firing strength** is the product of its memberships: it is strong only if every condition matches.")
        st.latex(r"\mu_{ij}=e^{-\frac12\left(\frac{x_j-c_{ij}}{s_{ij}}\right)^2},\quad w_i=\prod_j \mu_{ij},\quad \bar w_i=\frac{w_i}{\sum_k w_k}")
        mf = st.selectbox("See the learned curves for", cols, key="hiw_mf")
        st.pyplot(plots.membership_curves(bundle, cols.index(mf)), clear_figure=True)
    with t[2]:
        st.markdown("Each rule has its own linear formula; the final score is the share-weighted sum. Rule outputs can lie outside 0-1, so the displayed score is clipped to 0-1.")
        st.latex(r"f_i(x)=\sum_j p_{ij}x_j+r_i,\qquad \text{score}=\sum_i \bar w_i f_i(x)")
    with t[3]:
        st.markdown("The explanation re-uses the same numbers: memberships, firing strengths, shares, outputs and contributions. Text is assembled from them with fixed sentence templates; "
                    "nothing is generated by a language model, and no clinical ranges are added.")

# ---------------------------------------------------------------------------------------------- Model facts
else:
    st.header("Model facts")
    st.write(f"Rules in this model: {bundle['model'].R}. Trained on the project's working clinical mastitis dataset associated with Mendeley record kbvcdw5b4m "
             "(Version 2 provenance is pending archive-level verification), 1,100 cows, cows in folds 1 to 4; tested once on 220 unseen cows (fold 0).")
    p = ROOT / "reports/final/fold0_results.csv"
    if p.exists():
        r = pd.read_csv(p); r = r[r.feature_set == bundle["key"]]
        if len(r):
            st.write(f"Fold-0 test: F1 {r.row_f1.iloc[0]:.3f}, AUC {r.row_auc.iloc[0]:.3f} (single fold, preliminary; see the Demo for cross-validation results).")

st.caption("Research prototype. Not a diagnostic tool. Results are preliminary.")
