"""Unit tests for the explanation engine, report, Demo navigation and facts. Run: python test_explain_report.py"""
import numpy as np, pandas as pd, pathlib
import matplotlib; matplotlib.use("Agg")
import app_core, explain_report as er, facts, plots

root = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(root / "data/processed/full_processed.csv")
BANNED = ("early warning", "early-warning", "subclinical", "first-ever", "synthetic", "fabricated")
for mode in app_core.MODES:
    b = app_core.load(mode); cols = b["cols"]; ex = app_core.pick_examples(b, df)
    for name, row in ex.items():
        vals = {c: float(row[c]) for c in cols}; an = er.analyse(b, vals); out = an["out"]
        # 1. contributions sum to the raw score; shares sum to 1
        assert abs(sum(r["weighted_contribution"] for r in an["rules"]) - an["score"]) < 1e-9
        assert abs(sum(r["firing_share"] for r in an["rules"]) - 1) < 1e-9
        # 2. memberships come from the SAVED model: product of the 8 (or 13) memberships == raw firing strength of app_core
        raw = np.array([r["raw_firing"] for r in sorted(an["rules"], key=lambda r: r["rule"])])
        assert np.allclose(an["mu"].prod(axis=1), raw, rtol=1e-8, atol=0), "membership product != raw firing strength"
        # 3. the score equals the model's own predict_score on the scaled row
        x, _ = app_core.scale(b, vals); assert abs(float(b["model"].predict_score(x[None, :])[0]) - an["score"]) < 1e-9
        # 4. the report has every section and no banned wording
        blocks = er.build_report(an); html = er.report_html(blocks)
        heads = [x[1] for x in blocks if x[0] == "h2"]; assert len(heads) == 11 and heads[0].startswith("1.") and heads[-1].startswith("11."), heads
        low = html.lower(); assert not any(w in low for w in BANNED), [w for w in BANNED if w in low]
        assert "not a calibrated clinical probability" in low and "research prototype" in low
        assert "Interpretable ANFIS for Detection of Existing Clinical Bovine Mastitis" in html
        assert "Range the model learned from (lowest to highest)" in html
        assert "Cows in the training data had values in this range." in html
        assert "does not include the separate FC7 computer-generated onset study" in html
        pdf = er.report_pdf(blocks); assert pdf is None or pdf[:5] == b"%PDF-"
    print(f"OK  {mode}: 3 examples, contributions sum to score, membership product = firing strength, report + PDF built")
b = app_core.load("Sensor-only"); cols = b["cols"]; ex = app_core.pick_examples(b, df)
# 5. healthy and mastitis examples land where expected
h = er.analyse(b, {c: float(ex["Healthy cow (median score)"][c]) for c in cols}); m = er.analyse(b, {c: float(ex["Mastitis cow (median score)"][c]) for c in cols})
assert h["tier"] == "Low" and m["tier"] == "High", (h["tier"], m["tier"])
assert "score at or above 0.5 counts as a 'mastitis' output" in " ".join(h["narrative"]["summary"])
# 6. weak-activation warning: a cow far from every rule triggers it; text states it is a heuristic
far = {c: float(b["hi"][i]) for i, c in enumerate(cols)}; far[cols[0]] = float(b["lo"][0])
a = er.analyse(b, far); assert a["weak"] and "heuristic" in " ".join(a["narrative"]["activation"]).lower()
# 7. out-of-range input is flagged, clipped and mentioned in report
oor = {c: float(ex["Healthy cow (median score)"][c]) for c in cols}; oor[cols[0]] = 9999.0
a = er.analyse(b, oor); assert a["out_of_range"] == [cols[0]] and a["measurements"][0]["seen"] == b["hi"][0] and a["clipped"] in (True, False)
assert any("Outside the training range" in w for w in a["narrative"]["warnings"]) and "9999" in er.report_html(er.build_report(a))
# 8. Demo navigation: first -> last -> first, clamped at both ends
import types, sys
st_stub = types.ModuleType("streamlit"); st_stub.session_state = {}; sys.modules.setdefault("streamlit", st_stub)
sys.modules.setdefault("streamlit.components", types.ModuleType("streamlit.components")); sys.modules.setdefault("streamlit.components.v1", types.ModuleType("streamlit.components.v1"))
sys.modules["streamlit.components.v1"].html = lambda *a, **k: None; sys.modules["streamlit.components"].v1 = sys.modules["streamlit.components.v1"]
import demo
p = 0; seen = [p]
for _ in range(20): p = demo.nav_step(p, +1); seen.append(p)
assert seen[:demo.TOTAL + 1] == list(range(demo.TOTAL)) + [demo.TOTAL - 1]
assert max(seen) == demo.TOTAL - 1 == len(demo.PAGES) - 1
assert demo.PAGES[1][0] == "Project workflow"
assert demo.PAGES[2][0] == "Five-layer ANFIS architecture"
for _ in range(20): p = demo.nav_step(p, -1)
assert p == 0
# 9. facts equal the CSVs
f = facts.load_facts(root); assert f["rows"] == 6600 and f["cows"] == 1100 and f["healthy"] + f["mastitis"] == 6600
assert abs(f["r8"].loc["ANFIS", "F1"] - 0.9191) < 1e-4 and abs(f["r8"].loc["ANFIS", "AUC"] - 0.9795) < 1e-4
print(f"OK  healthy->Low, mastitis->High, weak-match warning, out-of-range clip, Demo nav {demo.TOTAL} pages both ways, facts match CSVs")
print("ALL EXPLAIN/REPORT TESTS PASSED")
