"""
app.py -- Streamlit dashboard for the interpretable ANFIS (research prototype, NOT a diagnostic tool)
Run:  streamlit run app.py
All maths lives in app_core.py (unit-tested by test_app_core.py). This file only draws widgets.
"""
from pathlib import Path
import pandas as pd
import streamlit as st
import app_core

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Mastitis detection with ANFIS", page_icon="🐄", layout="wide")


@st.cache_resource
def get_bundle(mode):
    return app_core.load(mode)


@st.cache_data
def get_examples():
    p = ROOT / "data/processed/full_processed.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p); t = d[d.fold == 0]
    return {"Typical healthy cow": t[t.label == 0].iloc[5], "Typical mastitis cow": t[t.label == 1].iloc[5],
            "Another mastitis cow": t[t.label == 1].iloc[40]}


UNITS = {"temperature": "dataset scale (unit not documented)", "months_pp": "months after calving", "breed_enc": "0 = Jersey, 1 = Holstein",
         "hardness": "0 = normal, 1 = hard udder", "pain": "0 = no pain, 1 = pain"}

st.title("🐄 Interpretable detection of clinical mastitis")
st.write("A neuro-fuzzy (ANFIS) model that flags **existing** clinical mastitis from udder-sensor readings and explains "
         "its answer with fuzzy rules. Research prototype; not a veterinary diagnosis.")

with st.sidebar:
    mode = st.radio("Model", list(app_core.MODES), help="Sensor-only uses the 8 udder sensors. Full-clinical adds temperature, months postpartum, breed, hardness, pain.")
    with st.expander("How it works"):
        st.markdown("1. Each input gets a soft LOW / MEDIUM / HIGH score (Gaussian bell).\n2. A rule fires strongly only if all its conditions match.\n"
                    "3. Firing strengths are normalised to sum to 1.\n4. Each rule gives a linear estimate.\n5. The answer is the weighted sum.")
bundle = get_bundle(mode); cols = bundle["cols"]; lo, hi = bundle["lo"], bundle["hi"]

ex = get_examples()
if ex:
    st.caption("Fill the form with an example row from the held-out test cows:")
    cc = st.columns(len(ex))
    for k, (name, row) in enumerate(ex.items()):
        if cc[k].button(name):
            for c in cols:
                st.session_state[f"in_{c}"] = float(row[c])

st.subheader("Inputs")
vals = {}
grid = st.columns(4)
for i, c in enumerate(cols):
    j = cols.index(c); default = float((lo[j] + hi[j]) / 2)
    with grid[i % 4]:
        if c in ("breed_enc", "hardness", "pain"):
            vals[c] = float(st.selectbox(f"{c}  ({UNITS[c]})", [0.0, 1.0], index=0, key=f"in_{c}"))
        else:
            label = f"{c}" + (f"  ({UNITS[c]})" if c in UNITS else "") + f"  [train {lo[j]:g} to {hi[j]:g}]"
            if f"in_{c}" in st.session_state:       # value set by an example button: do not also pass value= (Streamlit warns)
                vals[c] = st.number_input(label, step=1.0, key=f"in_{c}")
            else:
                vals[c] = st.number_input(label, value=round(default, 1), step=1.0, key=f"in_{c}")

if st.button("Estimate risk", type="primary"):
    out = app_core.predict(bundle, vals)
    for w in out["warnings"]:
        st.warning(w)
    a, b = st.columns(2)
    a.metric("Risk category", out["tier"]); b.metric("Model score (0 to 1, clipped)", f"{out['score_clipped']:.2f}")
    st.progress(out["score_clipped"])
    st.caption(f"Raw model score {out['score']:.3f}. Category thresholds are a presentation choice, not clinically validated.")
    st.subheader("Why: the rules that fired for this cow")
    st.dataframe(pd.DataFrame({"Rule": [r["rule"] for r in out["top_rules"]],
                               "Firing share": [f"{r['firing_share']:.0%}" for r in out["top_rules"]],
                               "Rule output for this cow": [r["rule_output_for_this_cow"] for r in out["top_rules"]]}), hide_index=True)
    for r in out["top_rules"]:
        with st.expander(f"Rule {r['rule']} (firing share {r['firing_share']:.0%}): rule centre vs your values"):
            st.dataframe(pd.DataFrame(r["conditions"]).rename(columns={"centre": "rule centre", "width": "width (+-)", "your_value": "your value"}), hide_index=True)
    st.caption("Fuzzy memberships of the score: " + ", ".join(f"{k} {v:.2f}" for k, v in out["tier_memberships"].items()))

with st.expander("Model facts"):
    st.write(f"Rules in this model: {bundle['model'].R}. Trained on the working dataset (Mendeley kbvcdw5b4m, 1,100 cows), cows in folds 1 to 4; "
             "tested once on 220 unseen cows (fold 0).")
    p = ROOT / "reports/final/fold0_results.csv"
    if p.exists():
        r = pd.read_csv(p); r = r[r.feature_set == bundle["key"]]
        if len(r):
            st.write(f"Fold-0 test: F1 {r.row_f1.iloc[0]:.3f}, AUC {r.row_auc.iloc[0]:.3f} (single fold, preliminary).")
st.caption("Research prototype. Not a diagnostic tool. Detects existing clinical mastitis; it does not forecast.")
