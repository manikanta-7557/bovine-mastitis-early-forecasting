"""
explain_report.py -- turns the ANFIS arithmetic for ONE cow into a structured explanation and a veterinary report
=================================================================================================================
No Streamlit code here (so it is unit-tested by test_explain_report.py).  Nothing in this file changes the model:
every number is read from the saved ANFIS (centres c, widths s, consequents theta) or recomputed from them with the
same formulas as anfis.py.

  analyse(bundle, vals)      -> everything the UI needs: per-measurement table, membership matrix, per-rule table,
                                plain-language narrative (all sentences are built from the numbers, none is canned
                                clinical text)
  build_report(an, meta)     -> short, plain-English veterinary report blocks
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
import random
import numpy as np

import app_core
from app_core import LOW_MATCH

SENSOR_MEANING = {  # the dataset page does not name the columns; I = inhale limit, E = exhale limit is "very likely"
    "IUFL": "Udder size, inhale limit, front-left quarter", "EUFL": "Udder size, exhale limit, front-left quarter",
    "IUFR": "Udder size, inhale limit, front-right quarter", "EUFR": "Udder size, exhale limit, front-right quarter",
    "IURL": "Udder size, inhale limit, rear-left quarter", "EURL": "Udder size, exhale limit, rear-left quarter",
    "IURR": "Udder size, inhale limit, rear-right quarter", "EURR": "Udder size, exhale limit, rear-right quarter",
    "temperature": "Temperature (°C)", "months_pp": "Months after calving",
    "breed_enc": "Breed (0 = Jersey, 1 = Holstein)", "hardness": "Udder hardness (0/1)", "pain": "Pain (0/1)",
}
_SIGNAL_NAMES = {"milk_conductivity_ms_cm": "Milk conductivity (mS/cm)", "milk_ph": "Milk pH", "milk_yield_l": "Milk yield (litres)", "milk_lactose_pct": "Milk lactose (%)",
                 "body_temp_c": "Body temperature (C)", "udder_surface_temp_c": "Udder surface temperature (C)", "rumination_min": "Rumination (min/day)", "activity_steps": "Activity (steps/day)"}
SENSOR_MEANING.update({f"D_{k}": f"{v}: today minus this cow's own mean of the previous 14 days" for k, v in _SIGNAL_NAMES.items()})
MANDATORY_DISCLAIMER = ("This is a research tool that looks for patterns in recorded measurements. "
                        "It is not a diagnosis or a substitute for veterinary examination.")
DISCLAIMER_FC = (MANDATORY_DISCLAIMER + " This isolated seven-day scenario uses computer-generated data only; "
                 "it is not a clinical result for a real cow. The score is not a probability or clinical finding.")
ACTIVE_SHARE = 0.05      # a rule is called "active" in the report when its firing share is at least 5% (presentation choice)
NEGLIGIBLE = 5e-4        # contributions smaller than this print as 0.000 at 3 decimals
THRESHOLD = 0.5          # the score cut used when the model is EVALUATED (F1); not a clinical cut-off

DISCLAIMER = MANDATORY_DISCLAIMER


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
    task = bundle.get("task", "detection")
    an = dict(task=task, alert_threshold=bundle.get("alert_threshold"), out=out, bundle_mode=bundle["mode"], key=bundle["key"], cols=cols, entered=raw, centre=centre, width=width, mu=mu, z=z,
              dominant=dom + 1, active=active, measurements=meas, rules=rules, score=out["score_unrounded"], clipped=out["clipped"],
              score_clipped=out["score_clipped"], tier=out["tier"], weak=out["max_raw_firing"] < LOW_MATCH,
              out_of_range=[mm["feature"] for mm in meas if not mm["in_range"]])
    if task == "forecast":
        an["tier"] = "Alert" if an["score"] >= an["alert_threshold"] else "No alert"
    an["narrative"] = narrate(an)
    return an


def _n(v, signed=False):
    """Number for display: 1 decimal for values of size >= 10 (sensor units), 3 significant digits for small values (deviations)."""
    v = float(v); t = "%.1f" % v if abs(v) >= 10 else "%.3g" % v
    return ("+" + t if v >= 0 else t) if signed else t


def _fmt(v, nd=3):
    return f"{v:+.{nd}f}"


def narrate(an: dict) -> dict:
    """Plain-language paragraphs built ONLY from the computed numbers."""
    out, rules = an["out"], an["rules"]
    score, tier = an["score"], an["tier"]
    side = "at or above" if score >= (an["alert_threshold"] if an["task"] == "forecast" else THRESHOLD) else "below"
    if an["task"] == "forecast":
        summary = [f"The model produced a raw scenario score of {score:.3f} for the synthetic-data question 'clinical mastitis starts within the next 7 days'. The alert threshold is {an['alert_threshold']:.3f} "
                   f"(chosen on validation data, not clinically). The score is {side} the threshold, so the tool shows '{tier}'. This is a model output, not a calibrated probability.",
                   {"Alert": "In plain terms: in the recent days this cow's signals moved away from her own recent baseline in a way that, in the synthetic training data, came before clinical onset.",
                    "No alert": "In plain terms: this cow's signals stayed close to her own recent baseline, or changed in ways that did not come before clinical onset in the synthetic training data."}[tier],
                   "The model was built and tested on a synthetic dataset only; this is a demonstration of the method, not a field prediction."]
    else:
        summary = [f"The model produced a raw score of {score:.3f}"
                   + (f"; because the displayed scale is 0 to 1 it is shown as {an['score_clipped']:.3f}" if an["clipped"] else "")
                   + f". In this prototype's presentation categories that is '{tier}'. "
                   f"When the model is evaluated in our experiments, a score at or above {THRESHOLD} counts as a 'mastitis' output; "
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
    near = [f"{cols[j]} (membership {an['mu'][d, j]:.2f}; entered {an['entered'][j]:g} vs centre {_n(an['centre'][d, j])})" for j in order[:2]]
    far = [f"{cols[j]} (membership {an['mu'][d, j]:.2f}; entered {an['entered'][j]:g} vs centre {_n(an['centre'][d, j])})" for j in order[::-1][:2]]
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
    if an.get("task") == "forecast":
        pts = ["Examine the udder and milk now (swelling, heat, pain, appearance of milk) regardless of the alert status; the sensor model does not replace this.",
               "Check the sensor records behind the largest changes: " + ", ".join(far) + " (largest departures from this cow's own baseline in the model's terms).",
               "Repeat the measurements at the next milking to see whether the change persists.",
               "Consider the cow's history and context (calving date, previous episodes, hygiene and housing conditions) as the veterinarian sees fit."]
        if an["weak"]: pts.append("No learned rule matched this cow well, so the alert status is less informative than usual.")
        return pts
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


def veterinary_summary(an: dict) -> list[tuple[str, str]]:
    """Return short sections suitable for a veterinary reader without model jargon."""
    fc = an.get("task") == "forecast"
    tier = an["tier"]
    if fc:
        status = ("In this computer-generated example, the model marked a change for follow-up."
                  if tier == "Alert" else
                  "In this computer-generated example, the model did not mark a change for follow-up.")
        concern = "This result comes from a computer-generated demonstration and does not describe a real cow."
    else:
        status = model_indication(an)
        cautions = []
        if an["weak"]:
            cautions.append("The model did not find a close match for these readings, so its result may be less reliable.")
        if an["out_of_range"]:
            cautions.append("Some readings were outside the range the model learned from. Please check that they were entered correctly.")
        if not cautions:
            if status == "Clinical mastitis pattern indicated by the model.":
                cautions.append("The pattern is a reason to examine the cow; it is not confirmation of mastitis.")
            elif status == "The model output is near the classification boundary.":
                cautions.append("The model result is uncertain, so it should not be used on its own to decide the cow's health.")
            else:
                cautions.append("A low model result does not rule out mastitis or another health problem.")
        concern = " ".join(cautions)

    return [
        ("Model-indicated status", status),
        ("What may need attention", concern),
        ("Overall health", "This tool does not assess the cow's overall health and cannot confirm or rule out disease."),
        ("Suggested veterinary follow-up",
         "Examine the udder and milk, assess the cow's general condition, and use any further checks you consider appropriate."),
        ("Important limitation",
         "This is a research tool that looks for patterns in recorded measurements. It is not a diagnosis or a substitute for veterinary examination."),
    ]


def model_indication(an: dict) -> str:
    """Translate a model result into the cautious wording used in veterinary reports."""
    if an.get("task") == "forecast":
        return ("The synthetic model marked a change for follow-up."
                if an["tier"] == "Alert" else
                "The synthetic model did not mark a change for follow-up.")
    score = float(an["score"])
    if 0.0 <= score <= 0.35:
        return "No strong clinical mastitis pattern indicated by the model."
    if 0.46 <= score <= 0.5:
        return "The model output is near the classification boundary."
    if 0.7 <= score <= 0.9999:
        return "Clinical mastitis pattern indicated by the model."
    return {
        "Low": "No strong clinical mastitis pattern indicated by the model.",
        "High": "Clinical mastitis pattern indicated by the model.",
        "Watch": "The model output is near the classification boundary.",
    }.get(an["tier"], "The model could not provide a clear result for these readings.")


def build_report(an: dict, meta: dict | None = None, seed: int | None = None) -> list:
    """Build a varied, plain-English report from this analysis and its actual model outputs."""
    meta = meta or {}
    rng = random.Random(seed)
    sections = veterinary_summary(an)
    if an.get("task") == "forecast":
        title = "Computer-generated example: veterinary explanation"
        intro = ("This is an isolated software demonstration using computer-generated data. "
                 "It is not a clinical result for a real cow.")
        disclaimer = DISCLAIMER_FC
    else:
        title = "Veterinary explanation: existing clinical mastitis detection"
        intro = rng.choice((
            "This report summarizes how the model interpreted the measurements entered for this cow.",
            "The following explanation is based on this cow's entered measurements and the saved model output.",
            "This veterinary-facing summary describes the model result for the measurements entered in this assessment.",
        ))
        disclaimer = DISCLAIMER
    blocks = [("h1", title), ("p", intro)]
    if meta.get("case"):
        blocks.append(("p", f"Record: {meta['case']}"))
    if meta.get("record_note"):
        blocks.append(("p", f"Record source: {meta['record_note']}."))
    for heading, text in sections:
        if heading == "Model-indicated status":
            text = (f"{text} The ANFIS score was {an['score']:.3f}"
                    + (f" (displayed as {an['score_clipped']:.3f} after clipping)." if an["clipped"] else ".")
                    + " This score is not a calibrated probability.")
        if heading == "What may need attention" and an.get("task") != "forecast":
            if an["weak"]:
                text += " The model's learned patterns did not closely match this set of readings."
            if an["out_of_range"]:
                text += " Out-of-range measurements were clipped before scoring."
        blocks.extend((("h2", heading), ("p", text)))

    measurements = sorted(an["measurements"], key=lambda item: item["dom_mu"], reverse=True)
    if measurements:
        closest = measurements[:2]
        farthest = min(measurements, key=lambda item: item["dom_mu"])
        detail = rng.choice((
            "Within the leading learned pattern, {a} ({av:g}) and {b} ({bv:g}) were the closest-matching entered measurements; {c} ({cv:g}) was less similar.",
            "The strongest pattern match was closest on {a} ({av:g}) and {b} ({bv:g}). The entered {c} value ({cv:g}) matched that pattern less closely.",
            "For the leading learned pattern, {a} ({av:g}) and {b} ({bv:g}) aligned most closely, while {c} ({cv:g}) was a weaker match.",
        )).format(a=closest[0]["feature"], av=closest[0]["entered"],
                 b=closest[-1]["feature"], bv=closest[-1]["entered"],
                 c=farthest["feature"], cv=farthest["entered"])
        blocks.extend((("h2", "Measurements considered"), ("p", detail),
                       ("p", "These are comparisons with learned model patterns, not veterinary reference ranges or findings about the animal.")))

    important_rules = sorted(an["rules"], key=lambda rule: abs(rule["weighted_contribution"]), reverse=True)[:2]
    if important_rules:
        rule_details = [
            f"Learned pattern {rule['rule']} matched {rule['firing_share']:.1%} of the model's combined pattern weight; "
            f"its output was {rule['rule_output_for_this_cow']:.3f} and its contribution was {rule['weighted_contribution']:+.3f}."
            for rule in important_rules
        ]
        explanation = rng.choice((
            "The model combined its learned patterns as follows: ",
            "The main learned-pattern contributions to this result were: ",
            "The result reflects the following strongest learned-pattern contributions: ",
        )) + " ".join(rule_details) + f" Together, all rule contributions produced the score {an['score']:.3f}."
        blocks.extend((("h2", "How the model combined patterns"), ("p", explanation),
                       ("p", "These are model calculations, not separate tests or clinical observations.")))
    blocks.append(("note", disclaimer))
    return blocks


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


def report_html(blocks: list, title: str = "Interpretable ANFIS for Detection of Existing Clinical Bovine Mastitis") -> str:
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
