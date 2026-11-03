"""Research dashboard for clinical mastitis detection and a separate generated-data demonstration.

Run: ``streamlit run app.py``.

DEFENSE GUIDE
  Why separate the pages? Clinical detection and the generated onset study use different
    datasets and saved models; keeping them visibly distinct prevents cross-claiming evidence.
  Why retain the existing input and explanation path? It uses the same scaler and ANFIS
    artifacts as evaluation, so the interface does not introduce another scoring implementation.
  Alternatives rejected: presenting generated-data results as farm evidence or retraining
    artifacts from interactive controls, neither of which is supported by this repository.
"""
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

import app_core
import demo
import explain_report as er
import forecast_core as fc
import forecast_study as fs
import facts as F
import plots
import ui_style as ui

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Bovine mastitis research", layout="wide")
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
def get_synthetic_raw():
    p = ROOT / fc.DATA_DIR / "daily_cow_data.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def get_report(path):
    p = ROOT / path
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def get_examples(mode):
    d = get_df()
    return None if d is None else app_core.pick_examples(get_bundle(mode), d)


UNITS = {"temperature": "dataset scale (unit not documented)", "months_pp": "months after calving", "breed_enc": "0 = Jersey, 1 = Holstein",
         "hardness": "0 = normal, 1 = hard udder", "pain": "0 = no pain, 1 = pain"}

with st.sidebar:
    st.markdown("### Mastitis research")
    mode = st.radio("Model", list(app_core.MODES), key="model_mode",
                    help="Sensor-only: the 8 udder sensors. Full-clinical: adds temperature, months postpartum, breed, hardness, pain.")
    st.info("Research prototype, not a veterinary diagnosis. The main model detects existing clinical mastitis; the separate onset scenario uses computer-generated data only.")
    st.caption("Sensor-only = raw 8 sensors (R8). Full-clinical = R13.")
    section = st.radio(
        "Research workspace",
        ["Overview", "Synthetic onset study", "Explainable reasoning", "Temporal analysis",
         "Data explorer", "Model laboratory", "Evaluation", "About & methodology"],
        key="section",
    )
bundle = get_bundle(mode); cols = bundle["cols"]; lo, hi = bundle["lo"], bundle["hi"]

ui.hero("Bovine mastitis research dashboard", "Interpretable ANFIS for detection of existing clinical mastitis, with a separate computer-generated onset-study demonstration")


# ---------------------------------------------------------------------------------------------- Shared score explanation
def show_analysis(an, meta=None):
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

    st.subheader("Veterinary report")
    if st.button("Generate Detailed Veterinary Explanation", type="primary", key="gen_report"):
        st.session_state["report_for"] = id(an)
    if st.session_state.get("report_for") == id(an):
        blocks = er.build_report(an, meta)
        html = er.report_html(blocks)
        components.html(html, height=640, scrolling=True)
        d1, d2 = st.columns(2)
        d1.download_button("Download report (HTML, opens print dialog)", html, file_name="anfis_veterinary_explanation.html", mime="text/html")
        pdf = er.report_pdf(blocks)
        if pdf:
            d2.download_button("Download report (PDF)", pdf, file_name="anfis_veterinary_explanation.pdf", mime="application/pdf")
        else:
            d2.caption("PDF needs `pip install reportlab`; use the HTML report and Print > Save as PDF.")


if section == "Overview":
    st.header("Project overview")
    ui.callout(
        "The primary task is <b>detection of existing clinical mastitis</b> from udder-related measurements. "
        "The separate onset-study screens use computer-generated data and are demonstrations only; they are not evidence of performance on real animals.",
        warn=True,
    )
    d = get_df()
    pooled = get_report("reports/final/summary_pooled.csv")
    a, b, c = st.columns(3)
    if d is not None:
        with a:
            ui.kpi("Clinical detection records", f"{len(d):,}", f"{d.cow_id.nunique():,} animals in the working processed dataset")
    else:
        with a:
            ui.kpi("Clinical detection records", "Unavailable", "processed data file not found")
    with b:
        ui.kpi("Selected ANFIS", f"{bundle['model'].R} rules", f"{len(cols)} input measurements · {mode}")
    fc_pack = get_fc()
    with c:
        ui.kpi("Onset-study scope", "Generated data only", "separate 7-day demonstration" if fc_pack else "study artifacts unavailable")
    st.subheader("Evidence at a glance")
    if pooled is not None and "R8" in set(pooled.feature_set) and "ANFIS" in set(pooled.model):
        row = pooled[(pooled.feature_set == "R8") & (pooled.model == "ANFIS")]
        if not row.empty:
            m1, m2 = st.columns(2)
            with m1:
                ui.kpi("Sensor-only ANFIS F1", f"{row.F1.iloc[0]:.3f}", "pooled out-of-fold; 95% cow-bootstrap interval")
            with m2:
                ui.kpi("Sensor-only ANFIS AUC", f"{row.AUC.iloc[0]:.3f}", "same evaluation file and folds")
            st.caption(
                f"F1 95% CI: {row.F1_lo.iloc[0]:.3f}–{row.F1_hi.iloc[0]:.3f}; "
                f"AUC 95% CI: {row.AUC_lo.iloc[0]:.3f}–{row.AUC_hi.iloc[0]:.3f}. "
                "These are results on the project's working dataset, not external clinical validation."
            )
        else:
            st.info("Sensor-only ANFIS summary is not available in the saved pooled results.")
    else:
        st.info("Saved pooled evaluation results are unavailable.")
    st.markdown(
        "**Research tracks**  \n"
        "- **Clinical detection:** existing clinical mastitis labels, saved R8/R13 models, and rule-level explanations.  \n"
        "- **Synthetic onset study:** a distinct computer-generated dataset and FC7 artifact; its results must not be interpreted as real-animal evidence."
    )
    st.caption("Use Research workspace in the sidebar to inspect explanations, temporal patterns, datasets, model structure, and evaluation evidence.")

# ---------------------------------------------------------------------------------------------- Clinical detection explanation
elif section == "Explainable reasoning":
    ex = get_examples(mode)
    if ex:
        st.caption("Load a held-out example record selected by model score (not clinically 'typical'), or enter measurements below.")
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

# ---------------------------------------------------------------------------------------------- Synthetic onset scenario
elif section == "Synthetic onset study":
    st.header("Synthetic 7-day onset scenario")
    ui.callout("<b>Computer-generated data only.</b> This section demonstrates model behavior on a synthetic 90-day farm dataset and a seven-day clinical-onset target. "
               "It is not real-animal evidence, does not establish farm performance, and is separate from the existing-clinical-mastitis detection results.", warn=True)
    pack = get_fc()
    if pack is None:
        st.info("Scenario files not found. Place the provided data under data/external/Synthetic_Data_for_Mastitis_Disease/ and run `python train_forecast_model.py --data data/external/Synthetic_Data_for_Mastitis_Disease`.")
    else:
        fb, daily, held = pack
        info = {c: daily[daily.cow_id == c].mastitis_status.max() for c in held}
        label = lambda c: f"Held-out animal {held.index(c) + 1}  ({'clinical onset in this study' if info[c] == 2 else 'no clinical onset'})"
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
        show_analysis(an_f, dict(case=f"held-out synthetic study animal {held.index(cow) + 1}, study day {day}", recent=recent))
    p = ROOT / "reports/forecast/forecast_summary.csv"
    if p.exists():
        with st.expander("Cross-validated results on the synthetic study data"):
            sm = pd.read_csv(p); sm = sm[sm.target == "clinical_within_7d"]
            st.dataframe(sm[["feature_set", "model", "ROC_AUC", "PR_AUC", "precision", "recall", "F1", "onset_cows_alerted", "onset_cows", "median_lead_days", "false_alert_days_per_100"]].round(3), hide_index=True)
            st.caption("D8 = deviation from the cow's own previous 14 days; C8 = today's raw values only. Cow-grouped five-fold evaluation on synthetic data; these results are not clinical evidence.")

# ---------------------------------------------------------------------------------------------- Temporal analysis
elif section == "Temporal analysis":
    st.header("Temporal analysis")
    st.caption("Animal-level time series from the computer-generated study dataset only. These patterns are not real-animal evidence.")
    pack = get_fc()
    if pack is None:
        st.info("The synthetic study model or daily data are unavailable.")
    else:
        fb, daily, held = pack
        cow = st.selectbox(
            "Study animal",
            held,
            format_func=lambda value: f"Held-out animal {held.index(value) + 1}",
            key="temporal_cow",
        )
        tr = fc.trajectory(fb, daily, cow)
        if tr.empty:
            st.warning("No eligible observations are available for this study animal.")
        else:
            signal = st.selectbox("Measurement", fs.SIGNALS, key="temporal_signal")
            show_deviation = st.toggle("Show change from the prior 14-day mean", key="temporal_deviation")
            column = f"D_{signal}" if show_deviation else signal
            title = f"Change from prior baseline: {signal}" if show_deviation else signal.replace("_", " ")
            fig, ax = plt.subplots(figsize=(8, 3.4))
            ax.plot(tr.day_index, tr[column], color="#1f6f54", lw=1.8)
            onset = tr.onset_day.iloc[0]
            if pd.notna(onset):
                ax.axvline(onset, color="#c0504d", ls="--", lw=1.4, label="generated clinical onset")
                ax.legend(frameon=False, fontsize=8)
            ax.set(title=title, xlabel="Study day", ylabel="Deviation from prior mean" if show_deviation else "Recorded value")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.grid(axis="y", color="#e5ece8", lw=.8); fig.tight_layout()
            st.pyplot(fig, clear_figure=True); plt.close(fig)
            st.pyplot(plots.forecast_trajectory(tr, fb["alert_threshold"]), clear_figure=True)
            gaps = int((tr.day_index.diff().dropna() > 1).sum())
            m1, m2, m3 = st.columns(3)
            ui.kpi("Eligible study days", f"{len(tr):,}", f"{int(tr.day_index.min())}–{int(tr.day_index.max())}")
            ui.kpi("Missing day gaps", str(gaps), "within the eligible time-series")
            ui.kpi("Generated onset label", "Present" if pd.notna(onset) else "Not present", "synthetic target event")
            st.caption("The line chart shows recorded synthetic measurements; the dashed marker is the generated onset label, not a clinical assessment.")

# ---------------------------------------------------------------------------------------------- Data explorer
elif section == "Data explorer":
    st.header("Data explorer")
    clinical = get_df()
    synthetic = get_synthetic_raw()
    choices = []
    if clinical is not None:
        choices.append("Clinical detection dataset (processed)")
    if synthetic is not None:
        choices.append("Computer-generated daily study")
    if not choices:
        st.error("No supported dataset files were found.")
    else:
        dataset_name = st.selectbox("Dataset", choices, key="explorer_dataset")
        source = clinical if dataset_name.startswith("Clinical") else synthetic
        data = source.copy()
        identity = [name for name in ("cow_id", "farm_id", "animal_id") if name in data]
        subjects = data[identity[0]].nunique() if identity else None
        x1, x2, x3, x4 = st.columns(4)
        with x1:
            ui.kpi("Rows", f"{len(data):,}")
        with x2:
            ui.kpi("Columns", str(len(data.columns)))
        with x3:
            ui.kpi("Animals", f"{subjects:,}" if subjects is not None else "Not recorded")
        with x4:
            ui.kpi("Missing cells", f"{int(data.isna().sum().sum()):,}")
        if dataset_name.startswith("Computer"):
            ui.callout(
                "This dataset is computer-generated. Status, risk, and outcome columns are present for study auditing; "
                "they are not inputs to the saved ANFIS model.",
                warn=True,
            )
        else:
            st.caption("This view uses the project's processed clinical-detection data. Animal and farm identifiers are omitted from previews and downloads.")
        with st.expander("Missing values by column"):
            missing = data.isna().sum().sort_values(ascending=False)
            missing = missing[missing > 0]
            if missing.empty:
                st.success("No missing cells in this dataset.")
            else:
                st.dataframe(missing.rename("Missing cells").to_frame(), width="stretch")
        filter_options = [name for name in ("fold", "label", "breed", "mastitis_status", "lactation_stage", "risk_category") if name in data.columns]
        filter_col = st.selectbox("Optional category filter", ["No filter"] + filter_options, key="explorer_filter")
        filtered = data
        if filter_col != "No filter":
            values = sorted(data[filter_col].dropna().unique().tolist(), key=str)
            selected = st.multiselect("Keep values", values, default=values, key=f"explorer_values_{filter_col}")
            filtered = data[data[filter_col].isin(selected)]
        numeric = [name for name in data.select_dtypes(include="number").columns if name not in identity]
        if numeric:
            measure = st.selectbox("Distribution", numeric, key="explorer_measure")
            values = filtered[measure].dropna()
            if not values.empty:
                fig, ax = plt.subplots(figsize=(7, 2.8))
                ax.hist(values, bins=30, color="#1f6f54", alpha=.85)
                ax.set(title=f"{measure.replace('_', ' ')} distribution", xlabel=measure, ylabel="Records")
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                fig.tight_layout(); st.pyplot(fig, clear_figure=True); plt.close(fig)
        st.subheader(f"Records ({len(filtered):,})")
        page_size = st.selectbox("Rows per page", [25, 50, 100], index=1, key="explorer_page_size")
        page_count = max(1, (len(filtered) + page_size - 1) // page_size)
        page = st.selectbox("Page", list(range(1, page_count + 1)), key=f"explorer_page_{dataset_name}")
        start = (page - 1) * page_size
        public = filtered.drop(columns=identity, errors="ignore")
        st.dataframe(public.iloc[start:start + page_size], hide_index=True, width="stretch")
        st.download_button(
            "Download filtered records (identifiers removed)",
            public.to_csv(index=False),
            file_name="mastitis_research_records.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------------------------- Model laboratory
elif section == "Model laboratory":
    st.header("Model laboratory")
    st.caption("Inspect the saved model and reproducible training evidence. Browser controls do not retrain or overwrite model artifacts.")
    m1, m2, m3 = st.columns(3)
    with m1:
        ui.kpi("Model family", "First-order Sugeno ANFIS")
    with m2:
        ui.kpi("Rules", str(bundle["model"].R))
    with m3:
        ui.kpi("Input measurements", str(len(cols)))
    st.subheader("Inputs in the selected artifact")
    st.dataframe(pd.DataFrame({"Feature": cols, "Training minimum": lo, "Training maximum": hi}), hide_index=True, width="stretch")
    lab_tabs = st.tabs(["Architecture", "Membership functions", "Training evidence"])
    with lab_tabs[0]:
        ui.flow(
            [("Inputs", "Selected sensor and clinical measurements", ""),
             ("Premise", "Gaussian membership for each rule", "alt"),
             ("Rule firing", "Multiply memberships and normalize shares", ""),
             ("Consequent", "Learned linear Sugeno output", "alt"),
             ("Score", "Weighted sum with rule-level contributions", "amb")],
            3,
            220,
        )
        st.markdown("Subtractive clustering initializes a compact rule set. Hybrid learning fits the linear consequents and membership parameters. The displayed rules and curves come from the selected saved artifact.")
    with lab_tabs[1]:
        mf = st.selectbox("Input measurement", cols, key="laboratory_feature")
        st.pyplot(plots.membership_curves(bundle, cols.index(mf)), clear_figure=True)
        st.caption("Curves show the saved model's Gaussian memberships over the training range; they are not clinical reference ranges.")
    with lab_tabs[2]:
        run_info = ROOT / "reports/final/RUN_INFO.json"
        if run_info.exists():
            import json
            config = json.loads(run_info.read_text(encoding="utf-8"))
            st.write("Recorded settings for the final grouped evaluation:")
            st.json(config)
        rules = get_report("reports/final/anfis_rules_per_fold.csv")
        if rules is not None:
            chosen = rules[rules.feature_set == bundle["key"]]
            st.dataframe(chosen, hide_index=True, width="stretch")
        ga_summary = get_report("reports/final/ga_summary_R8.csv")
        ga_params = get_report("reports/final/ga_best_params_R8.csv")
        if ga_summary is not None and ga_params is not None:
            st.markdown("**Saved R8 grid-versus-genetic-search comparison**")
            st.dataframe(ga_summary.reset_index(drop=True), hide_index=True, width="stretch")
            st.dataframe(ga_params, hide_index=True, width="stretch")
        st.code("python train_final_models.py --root .\npython ga_tuning.py --root . --set R8 --out reports/final", language="powershell")
        st.caption("Training commands are documented for reproducibility; this dashboard does not execute them or change existing artifacts.")

# ---------------------------------------------------------------------------------------------- Evaluation
elif section == "Evaluation":
    st.header("Evaluation")
    eval_tabs = st.tabs(["Clinical detection", "Computer-generated onset study"])
    with eval_tabs[0]:
        pooled = get_report("reports/final/summary_pooled.csv")
        oof = get_report("reports/final/oof_predictions.csv")
        if pooled is None:
            st.info("The saved pooled clinical-detection results are unavailable.")
        else:
            feature_names = {
                "R8": "Sensor-only raw (R8)", "R13": "Full-clinical raw (R13)",
                "S5": "Sensor-only engineered (S5)", "F7": "Full-clinical engineered (F7)",
            }
            available = [name for name in feature_names if name in set(pooled.feature_set)]
            choice = st.selectbox("Feature set", available, format_func=lambda name: feature_names[name], key="evaluation_features")
            result = pooled[pooled.feature_set == choice].set_index("model")
            st.dataframe(result.reset_index()[["model", "F1", "F1_lo", "F1_hi", "AUC", "AUC_lo", "AUC_hi"]].round(3), hide_index=True, width="stretch")
            st.pyplot(plots.results_bars(result, "F1"), clear_figure=True)
            st.caption("Pooled out-of-fold results and cow-bootstrap intervals from reports/final/summary_pooled.csv. Scores are for the working dataset and are not external validation.")
            if oof is not None:
                curve_data = oof[oof.feature_set == choice]
                models = [name for name in ("ANFIS", "RandomForest", "GradBoost", "kNN", "DecisionTree", "LogReg", "SVM") if name in set(curve_data.model)]
                left, right = st.columns(2)
                fig_roc, ax_roc = plt.subplots(figsize=(5.2, 3.4))
                fig_pr, ax_pr = plt.subplots(figsize=(5.2, 3.4))
                for name in models:
                    rows = curve_data[curve_data.model == name]
                    fpr, tpr, _ = roc_curve(rows.label, rows.score)
                    precision, recall, _ = precision_recall_curve(rows.label, rows.score)
                    ax_roc.plot(fpr, tpr, lw=1.5, label=name)
                    ax_pr.plot(recall, precision, lw=1.5, label=name)
                ax_roc.plot([0, 1], [0, 1], color="#b8c4bd", ls="--", lw=1)
                ax_roc.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Pooled out-of-fold ROC")
                ax_pr.set(xlabel="Recall", ylabel="Precision", title="Pooled out-of-fold precision–recall")
                for ax in (ax_roc, ax_pr):
                    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                    ax.legend(fontsize=7, frameon=False); ax.grid(color="#e5ece8", lw=.7)
                fig_roc.tight_layout(); fig_pr.tight_layout()
                left.pyplot(fig_roc, clear_figure=True); right.pyplot(fig_pr, clear_figure=True)
                plt.close(fig_roc); plt.close(fig_pr)
                anfis_rows = curve_data[curve_data.model == "ANFIS"]
                cm = confusion_matrix(anfis_rows.label, anfis_rows.score >= 0.5, labels=[0, 1])
                st.caption("ANFIS confusion matrix at score threshold 0.5 (rows = observed label; columns = predicted label), calculated from the saved out-of-fold scores.")
                st.dataframe(pd.DataFrame(cm, index=["Observed healthy", "Observed mastitis"], columns=["Predicted healthy", "Predicted mastitis"]), width="stretch")
    with eval_tabs[1]:
        ui.callout("All results in this panel use computer-generated data. They are a method demonstration and do not establish clinical or farm performance.", warn=True)
        synthetic_results = get_report("reports/forecast/forecast_summary.csv")
        if synthetic_results is None:
            st.info("Saved computer-generated study results are unavailable.")
        else:
            horizon = st.selectbox(
                "Generated outcome window",
                ["clinical_within_7d", "clinical_within_14d"],
                format_func=lambda value: "Generated onset within 7 days" if value.endswith("7d") else "Generated onset within 14 days",
                key="study_evaluation_target",
            )
            result = synthetic_results[synthetic_results.target == horizon]
            st.dataframe(result[["feature_set", "model", "rows", "positives", "ROC_AUC", "PR_AUC", "precision", "recall", "F1", "onset_cows_alerted", "onset_cows", "median_lead_days", "false_alert_days_per_100"]].round(3), hide_index=True, width="stretch")
            st.pyplot(plots.forecast_results(synthetic_results, horizon), clear_figure=True)
            st.caption("The per-animal explorer uses the saved 7-day FC7 artifact. The 14-day results are an evaluation summary only; no individual 14-day model artifact is provided.")

# ---------------------------------------------------------------------------------------------- About and methodology
elif section == "About & methodology":
    st.header("About & methodology")
    about_tabs = st.tabs(["Project scope", "ANFIS method", "Guided demonstration", "Model facts"])
    with about_tabs[0]:
        st.markdown(
            "This research prototype studies **detection of existing clinical mastitis** using an interpretable neuro-fuzzy model. "
            "It is not a veterinary diagnostic tool, has not been validated on farms, and should not be used to guide treatment."
        )
        st.markdown(
            "The clinical-detection dataset and the computer-generated onset-study dataset are separate. "
            "The latter is an algorithm demonstration only; it is not evidence about real animals."
        )
        st.markdown("The project's working clinical dataset is associated with Mendeley record kbvcdw5b4m. Version 2 provenance is still pending archive-level verification.")
    with about_tabs[1]:
        ui.flow(
            [("Data input", "8 sensors (+5 clinical fields in Full-clinical)", ""),
             ("Scaling", "training-only min–max; clip and warn", "alt"),
             ("Membership", "Gaussian curve per rule and input", ""),
             ("Rule firing", "product of memberships, then normalize", "alt"),
             ("Consequent", "linear Sugeno output per rule", ""),
             ("Score", "sum of share × rule output", "alt"),
             ("Explanation", "same memberships, shares, and contributions", "amb")],
            4,
            270,
        )
        t = st.tabs(["Scaling", "Rules", "Score", "Explanation"])
        with t[0]:
            st.markdown("Inputs are scaled with the saved training minima and maxima. Values outside that range are clipped and shown with a warning.")
            st.latex(r"x_j=\mathrm{clip}\!\left(\frac{v_j-\min_j}{\max_j-\min_j},0,1\right)")
        with t[1]:
            st.markdown("Each rule uses Gaussian memberships; normalized firing shares weight the linear rule outputs.")
            st.latex(r"\mu_{ij}=e^{-\frac12\left(\frac{x_j-c_{ij}}{s_{ij}}\right)^2},\quad w_i=\prod_j \mu_{ij},\quad \bar w_i=\frac{w_i}{\sum_k w_k}")
        with t[2]:
            st.markdown("Rule outputs may lie outside 0–1. The dashboard clips only the displayed score; the raw score and rule contributions remain visible.")
            st.latex(r"f_i(x)=\sum_j p_{ij}x_j+r_i,\qquad \text{score}=\sum_i \bar w_i f_i(x)")
        with t[3]:
            st.markdown("Explanations are assembled from the ANFIS calculations and fixed text templates. No language model or clinical reference range is used.")
    with about_tabs[2]:
        d = get_df(); sb = get_bundle("Sensor-only")
        demo.render(dict(root=ROOT, bundle=sb, facts=get_facts(), examples=None if d is None else app_core.pick_examples(sb, d)))
    with about_tabs[3]:
        st.write(f"Selected model: {mode}. Rules: {bundle['model'].R}. Inputs: {len(cols)}.")
        st.write("The working clinical dataset contains cow-grouped records; folds 1–4 train the models and fold 0 is held out for the single-fold model report.")
        p = ROOT / "reports/final/fold0_results.csv"
        if p.exists():
            r = pd.read_csv(p); r = r[r.feature_set == bundle["key"]]
            if not r.empty:
                st.write(f"Saved fold-0 report for {mode}: F1 {r.row_f1.iloc[0]:.3f}, AUC {r.row_auc.iloc[0]:.3f}.")
        st.caption("See Evaluation for the pooled grouped-fold results and confidence intervals.")

st.caption("Research prototype, not a diagnostic tool. The primary model detects existing clinical mastitis.")
