"""
app.py -- Streamlit dashboard for the interpretable ANFIS (research prototype, NOT a diagnostic tool)
Run:  streamlit run app.py
Sections: Predict | Synthetic onset study | Demo (15-page guided presentation) | How it works | Model facts.
All maths lives in app_core.py / explain_report.py (unit-tested). demo.py holds the Demo pages; ui_style.py the look.
"""
from pathlib import Path
import secrets
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import app_core
import demo
import explain_report as er
import forecast_core as fc
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
def get_fc():
    p = ROOT / "models/scaler_FC7.json"; q = ROOT / fc.DATA_DIR / "daily_cow_data.csv"
    return None if not (p.exists() and q.exists()) else (fc.load_forecast(ROOT), fc.load_daily(ROOT), fc.heldout_cows(ROOT))


@st.cache_data
def get_example_pools(mode):
    d = get_df()
    return None if d is None else app_core.pick_example_pools(get_bundle(mode), d)


UNITS = {"temperature": "°C", "months_pp": "months after calving", "breed_enc": "0 = Jersey, 1 = Holstein",
         "hardness": "0 = normal, 1 = hard udder", "pain": "0 = no pain, 1 = pain"}

with st.sidebar:
    st.markdown("### ANFIS mastitis detection")
    mode = st.radio("Model", list(app_core.MODES), key="model_mode",
                    help="Sensor-only: the 8 udder sensors. Full-clinical: adds temperature, months postpartum, breed, hardness, pain.")
    st.caption("Sensor-only = raw 8 sensors (R8). Full-clinical = R13.")
bundle = get_bundle(mode); cols = bundle["cols"]; lo, hi = bundle["lo"], bundle["hi"]

ui.hero("Interpretable detection of clinical mastitis", "Neuro-fuzzy (ANFIS) decision support that explains every score with its fuzzy rules")
section = st.radio("Section", ["Predict", "Synthetic onset study", "Demo", "How it works", "Model facts"], horizontal=True, key="section", label_visibility="collapsed")


# ---------------------------------------------------------------------------------------------- Predict
def save_prediction(bundle, mode, vals, meta=None):
    analysis = er.analyse(bundle, vals)
    report = er.build_report(analysis, meta, seed=secrets.randbits(64))
    report_pdf = er.report_pdf(report)
    st.session_state["analysis"] = dict(mode=mode, vals=dict(vals), an=analysis, report=report, report_pdf=report_pdf)
    return analysis


def show_analysis(an, meta=None, report_blocks=None, report_pdf=None):
    out = an["out"]; fcst = an.get("task") == "forecast"
    for w in out["warnings"]:
        st.warning(w)
    st.subheader("A. Prediction summary")
    a, b, c = st.columns(3)
    with a: ui.kpi("Alert status" if fcst else "Presentation category", an["tier"], f"threshold {an['alert_threshold']:.2f} (validation-tuned)" if fcst else "not a clinical threshold")
    with b: ui.kpi("Raw model score", f"{out['score']:.3f}", "sum of rule contributions")
    with c: ui.kpi("Displayed score (0-1)", f"{out['score_clipped']:.2f}", "clipped for display" if out["clipped"] else "same as raw")
    st.progress(out["score_clipped"])
    ui.callout("The score is a <b>model output, not a calibrated clinical probability</b>. " + ("Alert / No alert uses a threshold tuned on validation data of a SYNTHETIC dataset." if fcst else "Low / Watch / High are presentation categories."))
    for s in an["narrative"]["summary"]:
        st.write(s)

    st.subheader("B. Input measurement analysis")
    st.caption(f"Comparison with the dominant rule (Rule {an['dominant']}) and with the closest rule per measurement. Membership 1 = at the rule centre. No clinical reference ranges are implied.")
    st.dataframe(pd.DataFrame([{"Measurement": m["feature"], "Entered": m["entered"], "Training range": f"{m['lo']:g} to {m['hi']:g}" + ("" if m["in_range"] else "  (OUTSIDE: clipped)"),
                                f"Rule {an['dominant']} centre": er._n(m["dom_centre"]), "Difference": er._n(m["dom_diff"]), "Membership": round(m["dom_mu"], 2),
                                "Closest rule": m["best_rule"], "Its centre": er._n(m["best_centre"]), "Its membership": round(m["best_mu"], 2)} for m in an["measurements"]]), hide_index=True)
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

    report_col, summary_col = st.columns([1.2, 1])
    with report_col:
        st.subheader("Printable veterinary report")
        blocks = report_blocks or er.build_report(an, meta)
        html = er.report_html(blocks, title="Veterinary explanation")
        components.html(html, height=480, scrolling=True)
        d1, d2 = st.columns(2)
        d1.download_button("Download report (HTML)", html, file_name="veterinary_explanation.html", mime="text/html")
        pdf = report_pdf if report_pdf is not None else er.report_pdf(blocks)
        if pdf:
            d2.download_button("Download report (PDF)", pdf, file_name="veterinary_explanation.pdf", mime="application/pdf")
        else:
            d2.caption("PDF needs reportlab; the HTML report can be printed or saved as PDF.")
    with summary_col:
        st.subheader("Veterinary report")
        with st.container(border=True):
            for heading, text in er.veterinary_summary(an):
                st.markdown(f"**{heading}**")
                st.write(text)


if section.endswith("Predict"):
    pools = get_example_pools(mode)
    if pools:
        positions = st.session_state.setdefault("example_positions", {})
        cc = st.columns(len(app_core.EXAMPLE_SCORE_RANGES))
        for k, name in enumerate(app_core.EXAMPLE_SCORE_RANGES):
            pool = pools[name]
            if cc[k].button(name, key=f"ex_{k}", disabled=not pool):
                position_key = f"{mode}:{name}"
                sample, positions[position_key] = app_core.next_example(pool, positions.get(position_key, 0))
                row = sample["row"]
                vals_sample = {c: float(row[c]) for c in cols}
                store = st.session_state.setdefault("input_store", {})
                for c in cols:
                    st.session_state[f"in_{c}"] = vals_sample[c]
                    store[c] = vals_sample[c]
                record = f"cow {row.get('cow_id', 'unknown')}, day {row.get('day', 'unknown')}"
                source = ("held-out test data" if sample["held_out"]
                          else f"training data (fold {int(row['fold'])}; illustrative, not a held-out test example)")
                save_prediction(bundle, mode, vals_sample, {"case": record, "record_note": source})
    st.subheader("Inputs")
    store = st.session_state.setdefault("input_store", {})        # survives switching sections (widget state alone would be dropped)
    vals = {}
    grid = st.columns(4)
    for i, c in enumerate(cols):
        j = cols.index(c); default = float(round((lo[j] + hi[j]) / 2, 1))
        st.session_state.setdefault(f"in_{c}", store.get(c, default))
        with grid[i % 4]:
            help_text = (f"Model training range: {lo[j]:g} to {hi[j]:g}. "
                         "Values outside this range are clipped before scoring. This is not a veterinary reference range.")
            if c in ("breed_enc", "hardness", "pain"):
                vals[c] = float(st.selectbox(f"{c}  ({UNITS[c]})", [0.0, 1.0], key=f"in_{c}", help=help_text))
            else:
                label = c + (f"  ({UNITS[c]})" if c in UNITS else "")
                vals[c] = st.number_input(label, step=1.0, key=f"in_{c}", help=help_text)
        store[c] = vals[c]
    if st.button("Estimate risk", type="primary"):
        with st.spinner("Running the ANFIS rules..."):
            save_prediction(bundle, mode, vals)
    cur = st.session_state.get("analysis")
    if cur:
        if cur["mode"] != mode or cur["vals"] != vals:
            st.warning("Inputs or model changed since this result was calculated. Press 'Estimate risk' to update.")
        show_analysis(cur["an"], report_blocks=cur.get("report"), report_pdf=cur.get("report_pdf"))
    else:
        st.info("Enter the measurements (or load an example) and press 'Estimate risk'.")

# ---------------------------------------------------------------------------------------------- Synthetic onset scenario
elif section.endswith("Synthetic onset study"):
    st.header("Synthetic 7-day onset scenario")
    ui.callout("<b>Computer-generated data only.</b> This section demonstrates model behavior on a synthetic 90-day farm dataset and a seven-day clinical-onset target. "
               "It is not real-animal evidence, does not establish farm performance, and is separate from the existing-clinical-mastitis detection results.", warn=True)
    pack = get_fc()
    if pack is None:
        st.info("Scenario files not found. Place the provided data under data/external/Synthetic_Data_for_Mastitis_Disease/ and run `python train_forecast_model.py --data data/external/Synthetic_Data_for_Mastitis_Disease`.")
    else:
        fb, daily, held = pack
        info = {c: daily[daily.cow_id == c].mastitis_status.max() for c in held}
        label = lambda c: f"{c}  ({'clinical onset in this study' if info[c] == 2 else 'no clinical onset'})"
        cow = st.selectbox("Held-out animal (never seen in training)", held, format_func=label, key="fc_cow")
        tr = fc.trajectory(fb, daily, cow); on = tr.onset_day.iloc[0]
        days = [int(x) for x in tr[tr.forecast_eligible == 1].day_index]
        default = int(max(min(days), min(days[-1], (on - 5) if on == on else days[len(days) // 2])))
        day = st.select_slider("Study day to explain (before clinical onset)", options=days, value=default if default in days else days[0], key="fc_day")
        st.pyplot(plots.forecast_trajectory(tr[tr.forecast_eligible == 1], fb["alert_threshold"], day), clear_figure=True)
        key = (cow, day); cached = st.session_state.get("fc_an")
        if not cached or cached[0] != key:
            vals_f = fc.vals_at(fb, tr, day); an_f = er.analyse(fb, vals_f); st.session_state["fc_an"] = (key, an_f); st.session_state.pop("report_for", None)
        an_f = st.session_state["fc_an"][1]
        recent = [float(x) for x in tr[(tr.day_index <= day) & (tr.day_index > day - 7)].score]
        show_analysis(an_f, dict(case=f"synthetic animal {cow}, study day {day}", recent=recent))
    p = ROOT / "reports/forecast/forecast_summary.csv"
    if p.exists():
        with st.expander("Cross-validated results on the synthetic study data"):
            sm = pd.read_csv(p); sm = sm[sm.target == "clinical_within_7d"]
            st.dataframe(sm[["feature_set", "model", "ROC_AUC", "PR_AUC", "precision", "recall", "F1", "onset_cows_alerted", "onset_cows", "median_lead_days", "false_alert_days_per_100"]].round(3), hide_index=True)
            st.caption("D8 = deviation from the cow's own previous 14 days; C8 = today's raw values only. Cow-grouped five-fold evaluation on synthetic data; these results are not clinical evidence.")

# ---------------------------------------------------------------------------------------------- Demo
elif section.endswith("Demo"):
    d = get_df(); sb = get_bundle("Sensor-only")
    demo.render(dict(root=ROOT, bundle=sb, facts=get_facts(), examples=None if d is None else app_core.pick_examples(sb, d)))

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

st.caption("Research prototype, not a diagnostic tool. The primary model detects existing clinical mastitis.")
