"""
explain_report.py -- turns the ANFIS arithmetic for ONE cow into a structured explanation and a veterinary report
=================================================================================================================
No Streamlit code here (so it is unit-tested by test_explain_report.py).  Nothing in this file changes the model:
every number is read from the saved ANFIS (centres c, widths s, consequents theta) or recomputed from them with the
same formulas as anfis.py.

  analyse(bundle, vals)      -> everything the UI needs: per-measurement table, membership matrix, per-rule table,
                                plain-language narrative (all sentences are built from the numbers, none is canned
                                clinical text)
  build_report(an, meta)     -> list of report blocks (heading / paragraph / table / list) in 12 sections
  report_html(blocks)        -> standalone, print-ready HTML (A4, "Print / Save as PDF" button)
  report_pdf(blocks)         -> PDF bytes via reportlab (None if reportlab is not installed)

DEFENSE GUIDE
  Why membership = exp(-0.5 z^2)?  It is exactly layer 2 of anfis.py: mu_ij = exp(-0.5 ((x_j - c_ij) / s_ij)^2).  The raw
    firing strength of rule i is the PRODUCT of its eight memberships (layer 3); we test this identity.
  Why show raw firing AND share?  Share = firing / sum of firings, so it always adds to 100% even when every rule
    fits badly.  Showing both prevents false confidence.
  Why no clinical ranges?  The dataset page does not define the sensor units or healthy ranges; inventing them would be
    fabrication.  We only compare against what the MODEL learned (rule centres) and the training range.
  Alternatives rejected: SHAP/LIME on a black box (post hoc, approximates; ANFIS rules are the model itself);
    converting score to a probability (needs calibration we have not done).
"""
from __future__ import annotations

import html as _html
import io
from datetime import datetime

import numpy as np

import app_core
from app_core import LOW_MATCH

SENSOR_MEANING = {  # the dataset page does not name the columns; I = inhale limit, E = exhale limit is "very likely"
    "IUFL": "Udder size, inhale limit, front-left quarter", "EUFL": "Udder size, exhale limit, front-left quarter",
    "IUFR": "Udder size, inhale limit, front-right quarter", "EUFR": "Udder size, exhale limit, front-right quarter",
    "IURL": "Udder size, inhale limit, rear-left quarter", "EURL": "Udder size, exhale limit, rear-left quarter",
    "IURR": "Udder size, inhale limit, rear-right quarter", "EURR": "Udder size, exhale limit, rear-right quarter",
    "temperature": "Temperature (dataset scale; unit not documented)", "months_pp": "Months after calving",
    "breed_enc": "Breed (0 = Jersey, 1 = Holstein)", "hardness": "Udder hardness (0/1)", "pain": "Pain (0/1)",
}
ACTIVE_SHARE = 0.05      # a rule is called "active" in the report when its firing share is at least 5% (presentation choice)
NEGLIGIBLE = 5e-4        # contributions smaller than this print as 0.000 at 3 decimals
THRESHOLD = 0.5          # the score cut used when the model is EVALUATED (F1); not a clinical cut-off

DISCLAIMER = ("Research prototype from an academic soft-computing project. It is not a diagnostic device and not a "
              "substitute for veterinary examination. It detects patterns associated with existing clinical mastitis "
              "in one research dataset; it does not forecast disease and has not been validated on farms.")


# --------------------------------------------------------------------------------------------- analysis
def analyse(bundle: dict, vals: dict, out: dict | None = None) -> dict:
    """Full explanation data for one cow. `out` may be passed if app_core.predict was already run."""
    out = out or app_core.predict(bundle, vals)
    cols, lo, hi, m = bundle["cols"], bundle["lo"], bundle["hi"], bundle["model"]
    raw = np.array([float(vals[c]) for c in cols])
    x, _ = app_core.scale(bundle, vals)                                  # what the model actually sees (clipped to [0,1])
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    seen = lo + x * span                                                  # clipped value in raw units
    centre = lo + m.c * span                                              # (R, d) rule centres, raw units
    width = m.s * span                                                    # (R, d) rule widths, raw units
    z = (seen[None, :] - centre) / width                                  # signed distance in widths
    mu = np.exp(-0.5 * z ** 2)                                            # layer 2, (R, d)
    rules = out["all_rules"]                                              # sorted by firing share, descending
    dom = rules[0]["rule"] - 1                                            # dominant rule index (0-based)
    active = [r["rule"] for r in rules if r["firing_share"] >= ACTIVE_SHARE] or [rules[0]["rule"]]

    meas = []
    for j, c in enumerate(cols):
        best = int(np.argmax(mu[:, j]))
        meas.append(dict(feature=c, meaning=SENSOR_MEANING.get(c, c), entered=float(raw[j]), lo=float(lo[j]), hi=float(hi[j]),
                         in_range=bool(lo[j] <= raw[j] <= hi[j]), seen=float(seen[j]),
                         dom_centre=float(centre[dom, j]), dom_width=float(width[dom, j]), dom_diff=float(raw[j] - centre[dom, j]),
                         dom_z=float(z[dom, j]), dom_mu=float(mu[dom, j]),
                         best_rule=best + 1, best_centre=float(centre[best, j]), best_diff=float(raw[j] - centre[best, j]),
                         best_mu=float(mu[best, j])))
    an = dict(out=out, bundle_mode=bundle["mode"], key=bundle["key"], cols=cols, entered=raw, centre=centre, width=width, mu=mu, z=z,
              dominant=dom + 1, active=active, measurements=meas, rules=rules, score=out["score_unrounded"], clipped=out["clipped"],
              score_clipped=out["score_clipped"], tier=out["tier"], weak=out["max_raw_firing"] < LOW_MATCH,
              out_of_range=[mm["feature"] for mm in meas if not mm["in_range"]])
    an["narrative"] = narrate(an)
    return an


def _fmt(v, nd=3):
    return f"{v:+.{nd}f}"


def narrate(an: dict) -> dict:
    """Plain-language paragraphs built ONLY from the computed numbers."""
    out, rules = an["out"], an["rules"]
    score, tier = an["score"], an["tier"]
    side = "at or above" if score >= THRESHOLD else "below"
    summary = [f"The model produced a raw score of {score:.3f}"
               + (f"; because the displayed scale is 0 to 1 it is shown as {an['score_clipped']:.3f}" if an["clipped"] else "")
               + f". In this prototype's presentation categories that is '{tier}'. "
               f"When the model is evaluated in our experiments, a score {side} {THRESHOLD} counts as a 'mastitis' output; "
               "that is an evaluation convention, not a clinical cut-off, and the score is not a calibrated probability.",
               {"Low": "In plain terms: the eight measurements together look more like the healthy cows the model was trained on.",
                "Watch": "In plain terms: the measurements sit between the patterns the model learned for healthy and mastitis cows, so the output is uncertain.",
                "High": "In plain terms: the eight measurements together look more like the mastitis cows the model was trained on."}[tier]]
    top = sorted(rules, key=lambda r: -abs(r["weighted_contribution"]))
    contrib = []
    for r in top[:3]:
        if abs(r["weighted_contribution"]) < NEGLIGIBLE:
            continue
        direction = "adds to" if r["weighted_contribution"] > 0 else "pulls down"
        contrib.append(f"Rule {r['rule']} {direction} the score by {_fmt(r['weighted_contribution'])}: its firing share is {r['firing_share']:.1%} "
                       f"and its linear output for this cow is {r['rule_output_for_this_cow']:.3f} "
                       f"({r['firing_share']:.3f} x {r['rule_output_for_this_cow']:.3f}).")
    small = [r for r in rules if abs(r["weighted_contribution"]) < NEGLIGIBLE]
    if small:
        contrib.append("Rules with almost no effect (contribution below 0.0005): "
                       + ", ".join(f"Rule {r['rule']} (share {r['firing_share']:.2%})" for r in small) + ".")
    contrib.append("Adding all contributions gives "
                   f"{sum(r['weighted_contribution'] for r in rules):.3f}, which equals the raw score {score:.3f}.")
    d = an["dominant"] - 1
    order = np.argsort(-an["mu"][d])
    cols = an["cols"]
    near = [f"{cols[j]} (membership {an['mu'][d, j]:.2f}; entered {an['entered'][j]:g} vs centre {an['centre'][d, j]:.1f})" for j in order[:2]]
    far = [f"{cols[j]} (membership {an['mu'][d, j]:.2f}; entered {an['entered'][j]:g} vs centre {an['centre'][d, j]:.1f})" for j in order[::-1][:2]]
    close = [f"The dominant rule is Rule {an['dominant']} (firing share {rules[0]['firing_share']:.1%}). Its closest measurements to the rule centre are "
             + "; ".join(near) + ". The farthest are " + "; ".join(far) + ".",
             "A membership of 1 means the entered value sits exactly at the rule's centre; values near 0 are many rule-widths away. "
             "These are properties of the learned model, not clinical reference ranges."]
    mx = out["max_raw_firing"]
    if an["weak"]:
        act = [f"Weak match: the best rule's raw firing strength is {mx:.1e}, below the heuristic level of {LOW_MATCH:.0e}. "
               "No learned rule fits this cow well. The firing shares are still normalised to 100%, so they can look confident while the "
               "evidence is thin. Treat the output with extra caution. The 1e-5 level is a heuristic chosen by us (about 8% of held-out test rows fall below it), not a clinical rule."]
    else:
        act = [f"The best rule's raw firing strength is {mx:.1e}, above the heuristic weak-match level of {LOW_MATCH:.0e}. "
               "Raw strengths are small by construction because eight memberships (each below 1) are multiplied; the level is a heuristic, not a clinical rule."]
    act.append("Raw firing strength says how well one rule matches the cow in absolute terms; firing share only says how that rule compares with the others.")
    warns = []
    if an["out_of_range"]:
        warns.append("Outside the training range (values were clipped to the range before calculation): " + ", ".join(an["out_of_range"]) + ".")
    return dict(summary=summary, contributions=contrib, closeness=close, activation=act, warnings=warns)


# --------------------------------------------------------------------------------------------- report
def exam_points(an: dict) -> list:
    d = an["dominant"] - 1
    far = [an["cols"][j] for j in np.argsort(-np.abs(an["z"][d]))[:3]]
    pts = ["General clinical examination of all four quarters (visible swelling, heat, pain on palpation) and comparison between quarters.",
           "Examination of the milk itself and any laboratory or on-farm milk tests the veterinarian considers appropriate.",
           "Body temperature and general condition of the animal, and the cow's history (calving date, previous episodes).",
           "Confirmation that the sensor readings were taken correctly (sensor placement and calibration). The dataset does not document the exact meaning or unit of the sensor columns, so the readings should be checked against the instrument used.",
           f"Quarter measurements that differ most from the dominant rule's centre (in rule-widths): {', '.join(far)}. These are where the model's learned pattern and this cow differ most, so they may be worth re-measuring."]
    if an["out_of_range"]:
        pts.append("Entered values outside the training range: " + ", ".join(an["out_of_range"]) + ". Re-check these inputs.")
    if an["weak"]:
        pts.append("The model matched no learned rule well for this cow, so independent examination matters even more.")
    return pts


def build_report(an: dict, meta: dict | None = None) -> list:
    meta = meta or {}
    when = meta.get("when") or datetime.now().strftime("%d %b %Y, %H:%M")
    n = an["narrative"]; rules = an["rules"]; cols = an["cols"]
    B = [("h1", "Detailed model explanation for veterinary review"),
         ("note", DISCLAIMER),
         ("kv", [("Model", f"{an['bundle_mode']} ANFIS ({an['key']}), {an['out']['n_rules']} learned rules"), ("Generated", when)]
          + ([("Case", meta["case"])] if meta.get("case") else []))]
    B += [("h2", "1. Entered cow measurements"),
          ("table", ["Measurement", "Meaning", "Entered value", "Training range"],
           [[m["feature"], m["meaning"], f"{m['entered']:g}", f"{m['lo']:g} to {m['hi']:g}"] for m in an["measurements"]])]
    B += [("h2", "2. Model result and score"),
          ("table", ["Item", "Value"],
           [["Raw model score", f"{an['score']:.3f}"], ["Displayed score (clipped to 0-1)", f"{an['score_clipped']:.3f}"],
            ["Presentation category", f"{an['tier']} (not a clinically validated threshold)"],
            ["Weak-match check (heuristic)", "Weak match flagged" if an["weak"] else "Not flagged"]]),
          ("p", "The score is the model's output. It is not a calibrated clinical probability.")]
    B += [("h2", "3. Interpretation of the result")] + [("p", s) for s in n["summary"]]
    B += [("h2", "4. Measurement-wise comparison with learned rule centres"),
          ("p", f"Each row compares the entered value with the centre of the dominant rule (Rule {an['dominant']}) and with the rule whose centre is closest for that measurement. "
                "Membership is exp(-0.5 x ((value - centre) / width)^2): 1 at the centre, falling towards 0 with distance. No clinical reference ranges are implied."),
          ("table", ["Measurement", "Entered", f"Rule {an['dominant']} centre (+- width)", "Difference", "Membership", "Closest rule", "Its centre", "Membership"],
           [[m["feature"], f"{m['entered']:g}", f"{m['dom_centre']:.1f} (+-{m['dom_width']:.1f})", f"{m['dom_diff']:+.1f}", f"{m['dom_mu']:.2f}",
             f"Rule {m['best_rule']}", f"{m['best_centre']:.1f}", f"{m['best_mu']:.2f}"] for m in an["measurements"]])]
    B += [("h2", "5. Active ANFIS rules"),
          ("p", f"Rules with a firing share of at least {ACTIVE_SHARE:.0%} are called active here (a presentation choice). Active rules: "
                + ", ".join(f"Rule {a}" for a in an["active"]) + ".")]
    for a in an["active"]:
        i = a - 1; r = next(rr for rr in rules if rr["rule"] == a)
        B.append(("table", [f"Rule {a}: measurement", "Centre (+- width)", "Entered", "Membership"],
                  [[cols[j], f"{an['centre'][i, j]:.1f} (+-{an['width'][i, j]:.1f})", f"{an['entered'][j]:g}", f"{an['mu'][i, j]:.2f}"] for j in range(len(cols))]))
        B.append(("p", f"Rule {a}: raw firing strength {r['raw_firing']:.2e}, firing share {r['firing_share']:.1%}, linear output for this cow {r['rule_output_for_this_cow']:.3f}."))
    B += [("h2", "6. Rule firing-strength analysis")] + [("p", s) for s in n["activation"]]
    B.append(("table", ["Rule", "Raw firing strength", "Firing share"],
              [[f"Rule {r['rule']}", f"{r['raw_firing']:.2e}", f"{r['firing_share']:.2%}"] for r in rules]))
    B += [("h2", "7. Rule outputs and weighted contributions"),
          ("p", "Contribution of rule i = (firing share of rule i) x (output of rule i for this cow). Final raw score = sum of all contributions. "
                "Each rule output is a linear estimate and may be below 0 or above 1 in this type of model (first-order Sugeno)."),
          ("table", ["Rule", "Firing share", "Rule output (this cow)", "Contribution"],
           [[f"Rule {r['rule']}", f"{r['firing_share']:.3f}", f"{r['rule_output_for_this_cow']:.3f}", f"{r['weighted_contribution']:+.3f}"] for r in rules]
           + [["Sum (raw score)", "", "", f"{an['score']:+.3f}"]])]
    B += [("h2", "8. How the final score was obtained")] + [("p", s) for s in n["contributions"]] + [("p", s) for s in n["closeness"]]
    B += [("h2", "9. Important model warnings and limitations")]
    w = list(n["warnings"]) + (n["activation"][:1] if an["weak"] else [])
    B.append(("ul", w + [
        "Trained on the project's working clinical mastitis dataset associated with Mendeley record kbvcdw5b4m; Version 2 provenance is pending archive-level verification.",
        "No external farm validation. Performance numbers come from one research dataset (1,100 cows) evaluated with cow-grouped cross-validation.",
        "Low / Watch / High are presentation categories on the model score, not clinical thresholds.",
        "The model detects existing clinical mastitis patterns in this dataset. It does not forecast future disease.",
        "Learned rules describe patterns in the training data; they have not been reviewed by a veterinary expert."]))
    B += [("h2", "10. Suggested points for veterinary examination"),
          ("p", "General points for professional verification only. This report gives no diagnosis, treatment or medication advice."),
          ("ul", exam_points(an))]
    B += [("h2", "11. Closing disclaimer"), ("note", DISCLAIMER)]
    return B


# --------------------------------------------------------------------------------------------- renderers
_CSS = """
:root{--g:#1f6f54;--ink:#1b2b24;--mut:#5b6b63;--line:#d9e2dc;--bg:#f4f8f5}
*{box-sizing:border-box}body{font:14px/1.5 -apple-system,'Segoe UI',Roboto,Arial,sans-serif;color:var(--ink);margin:0;background:#fff}
.page{max-width:860px;margin:0 auto;padding:20px 22px}
h1{font-size:21px;color:var(--g);margin:0 0 8px}h2{font-size:15px;color:var(--g);border-bottom:1px solid var(--line);padding-bottom:3px;margin:20px 0 8px}
p{margin:6px 0}.note{background:var(--bg);border-left:3px solid var(--g);padding:8px 12px;border-radius:4px;color:#26382f;font-size:13px}
table{border-collapse:collapse;width:100%;margin:8px 0;font-size:12.5px}th,td{border:1px solid var(--line);padding:4px 7px;text-align:left;vertical-align:top}
th{background:var(--bg)}ul{margin:6px 0 6px 18px;padding:0}li{margin:3px 0}
.bar{display:flex;justify-content:flex-end;max-width:860px;margin:0 auto;padding:10px 22px 0}
.bar button{background:var(--g);color:#fff;border:0;border-radius:999px;padding:7px 18px;font-size:13px;cursor:pointer}
@media print{.bar{display:none}.page{padding:0}h2{break-after:avoid}table,.note{break-inside:avoid}@page{size:A4;margin:16mm}}
"""


def _e(s) -> str:
    return _html.escape(str(s))


def report_html(blocks: list, title: str = "ANFIS veterinary explanation") -> str:
    parts = []
    for b in blocks:
        k = b[0]
        if k == "h1": parts.append(f"<h1>{_e(b[1])}</h1>")
        elif k == "h2": parts.append(f"<h2>{_e(b[1])}</h2>")
        elif k == "p": parts.append(f"<p>{_e(b[1])}</p>")
        elif k == "note": parts.append(f'<div class="note">{_e(b[1])}</div>')
        elif k == "ul": parts.append("<ul>" + "".join(f"<li>{_e(x)}</li>" for x in b[1]) + "</ul>")
        elif k == "kv": parts.append("<table>" + "".join(f"<tr><th style='width:22%'>{_e(a)}</th><td>{_e(v)}</td></tr>" for a, v in b[1]) + "</table>")
        elif k == "table":
            parts.append("<table><tr>" + "".join(f"<th>{_e(h)}</th>" for h in b[1]) + "</tr>"
                         + "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in row) + "</tr>" for row in b[2]) + "</table>")
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title><style>{_CSS}</style></head><body><div class='bar'><button onclick='window.print()'>Print / Save as PDF</button></div>"
            f"<div class='page'>{''.join(parts)}</div></body></html>")


_PDF_SAFE = str.maketrans({"≥": ">=", "≤": "<=", "≈": "~", "−": "-", "–": "-", "—": "-", "→": "->",
                           "‘": "'", "’": "'", "“": '"', "”": '"', "±": "+-"})


def report_pdf(blocks: list) -> bytes | None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return None
    ss = getSampleStyleSheet(); green = colors.HexColor("#1f6f54"); pale = colors.HexColor("#f4f8f5"); line = colors.HexColor("#d9e2dc")
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, textColor=green, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=green, spaceBefore=10, spaceAfter=3, keepWithNext=1)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    cell = ParagraphStyle("c", parent=body, fontSize=7.6, leading=9.4)
    note = ParagraphStyle("n", parent=body, backColor=pale, borderPadding=5, leftIndent=4, rightIndent=4, spaceBefore=4, spaceAfter=6)
    P = lambda t, st: Paragraph(_html.escape(str(t).translate(_PDF_SAFE)), st)
    flow = []
    W = A4[0] - 32 * mm
    for b in blocks:
        k = b[0]
        if k == "h1": flow.append(P(b[1], h1))
        elif k == "h2": flow.append(P(b[1], h2))
        elif k == "p": flow.append(P(b[1], body))
        elif k == "note": flow.append(P(b[1], note))
        elif k == "ul": flow.append(ListFlowable([ListItem(P(x, body)) for x in b[1]], bulletType="bullet", leftIndent=12))
        else:
            rows = [[P(a, cell), P(v, cell)] for a, v in b[1]] if k == "kv" else [[P(h, cell) for h in b[1]]] + [[P(c, cell) for c in r] for r in b[2]]
            t = Table(rows, colWidths=[W * 0.22, W * 0.78] if k == "kv" else None, repeatRows=0 if k == "kv" else 1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, line), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   ("BACKGROUND", (0, 0), (-1, 0) if k != "kv" else (0, -1), pale),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
            flow += [t, Spacer(1, 4)]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                      title="ANFIS veterinary explanation").build(flow)
    return buf.getvalue()
