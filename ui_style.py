"""ui_style.py -- shared visual system for the Streamlit research dashboard."""
import streamlit as st
import streamlit.components.v1 as components

CSS = """
<style>
:root{
    --green:#176b52;--green-dark:#104d3d;--teal:#2b7c78;--amber:#a96914;
    --ink:#172b25;--muted:#60716b;--line:#dce6e1;--soft:#eff5f1;
    --canvas:#f5f8f6;--surface:#fff;--shadow:0 12px 32px rgba(20,55,43,.055)
}
.stApp{background:radial-gradient(ellipse at 85% 0%,rgba(221,238,229,.48),transparent 36%),var(--canvas);color:var(--ink)}
.block-container{max-width:1440px;padding:1.7rem 2rem 3.25rem}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:rgba(255,255,255,.92);border-right:1px solid var(--line)}
[data-testid="stSidebar"] > div:first-child{padding-top:1.3rem}
h1,h2,h3{color:var(--ink);letter-spacing:-.025em}
h2{font-size:1.45rem!important;font-weight:700!important}
h3{font-size:1.13rem!important;font-weight:680!important}
.hero{position:relative;isolation:isolate;overflow:hidden;display:flex;align-items:center;justify-content:space-between;gap:24px;
    background:linear-gradient(112deg,#123f34 0%,#176b52 55%,#287a73 100%);color:#fff;border:1px solid rgba(255,255,255,.14);
    border-radius:22px;padding:30px 34px;margin:4px 0 24px;box-shadow:0 18px 42px rgba(19,77,59,.16)}
.hero:after{content:"";position:absolute;z-index:-1;width:340px;height:340px;right:5%;top:-235px;border-radius:50%;
    border:1px solid rgba(255,255,255,.14);box-shadow:0 0 0 38px rgba(255,255,255,.035),0 0 0 78px rgba(255,255,255,.025)}
.hero-copy{max-width:780px}
.hero-kicker{display:block;color:#c4e6d4;font-size:.69rem;font-weight:750;letter-spacing:.15em;margin-bottom:9px}
.hero h1{color:#fff;font-size:clamp(1.7rem,3vw,2.4rem);line-height:1.1;margin:0 0 9px;padding:0}
.hero p{max-width:700px;color:rgba(255,255,255,.82);font-size:1rem;line-height:1.55;margin:0}
.hero-chip{flex:0 0 auto;border:1px solid rgba(255,255,255,.24);border-radius:999px;padding:9px 14px;
    color:#f0fff7;background:rgba(255,255,255,.09);font-size:.76rem;font-weight:650;letter-spacing:.03em}
.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin-bottom:12px;box-shadow:var(--shadow)}
.card h4{margin:0 0 7px;color:var(--green);font-size:1rem}.card p,.card li{font-size:.93rem;color:var(--ink)}
.kpi{height:100%;background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:16px 18px;text-align:left;box-shadow:var(--shadow)}
.kpi .l{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.09em;font-weight:700}
.kpi .v{font-size:1.65rem;line-height:1.25;font-weight:750;color:var(--green);margin:5px 0}
.kpi .s{font-size:.78rem;color:var(--muted)}
.callout{border:1px solid #d8e9df;border-left:4px solid var(--green);background:var(--soft);padding:12px 15px;border-radius:10px;margin:12px 0;font-size:.91rem;line-height:1.55}
.callout.warn{border-color:#f0dfc0;border-left-color:var(--amber);background:#fff8ec}
.chip{display:inline-block;background:var(--soft);color:var(--green);border-radius:999px;padding:4px 11px;font-size:.76rem;margin-right:6px;font-weight:650}
.dprog{height:7px;background:#e4ece7;border-radius:99px;overflow:hidden;margin:9px 0 5px}
.dprog>div{height:100%;background:linear-gradient(90deg,var(--green),var(--teal))}
.dtitle{color:var(--muted);font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;font-weight:700}
.stButton>button,[data-testid="stDownloadButton"]>button{min-height:2.55rem;border-radius:10px;border:1px solid var(--line);
    font-weight:650;transition:transform .15s ease,box-shadow .15s ease,border-color .15s ease}
.stButton>button:hover,[data-testid="stDownloadButton"]>button:hover{transform:translateY(-1px);border-color:#a9c8b7;box-shadow:0 5px 14px rgba(23,107,82,.10)}
.stButton>button[kind="primary"]{background:linear-gradient(110deg,var(--green-dark),var(--green));border-color:transparent;color:#fff}
.stButton>button[kind="primary"]:hover{background:linear-gradient(110deg,#0b4032,#145f49);border-color:transparent}
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"]>div{border-radius:9px}
.stProgress>div>div{background:linear-gradient(90deg,var(--green),var(--teal));border-radius:99px}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:rgba(255,255,255,.78)}
[data-testid="stAlert"]{border-radius:12px}
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{height:46px;padding:0 15px}
.stTabs [aria-selected="true"]{color:var(--green);font-weight:700}
.st-key-section [data-testid="stRadio"]>div{gap:7px}
.st-key-section label[data-baseweb="radio"]{border:1px solid transparent;border-radius:10px;padding:8px 13px;background:rgba(255,255,255,.62)}
.st-key-section label[data-baseweb="radio"]:has(input:checked){background:#e8f2ec;border-color:#cfe2d7;color:var(--green)}
.st-key-section label[data-baseweb="radio"]>div:first-child{display:none}
.st-key-demo_nav [data-testid="stHorizontalBlock"]{gap:0;align-items:center;background:#fff;border:1px solid var(--line);border-radius:999px;padding:4px;max-width:520px;margin:0 auto}
.st-key-demo_nav button{border-radius:999px;border:0;background:transparent;font-weight:600;color:var(--green)}
.st-key-demo_nav button:hover:not(:disabled){background:var(--soft);color:var(--green)}
.st-key-demo_nav button:disabled{opacity:.35}
.st-key-demo_nav .cnt{text-align:center;font-variant-numeric:tabular-nums;font-weight:650;color:var(--ink)}
div[role="radiogroup"]{gap:.35rem}
@media (max-width:850px){.block-container{padding:1.2rem 1.2rem 2.5rem}.hero{padding:24px;border-radius:18px}.hero-chip{display:none}}
@media (max-width:600px){.block-container{padding:1rem .85rem 2rem}.hero{padding:21px 19px;margin-bottom:18px}.hero h1{font-size:1.55rem}.hero p{font-size:.9rem}.kpi .v{font-size:1.35rem}.st-key-section label[data-baseweb="radio"]{padding:7px 8px;font-size:.82rem}}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, sub):
    st.markdown(
        f'<div class="hero"><div class="hero-copy"><span class="hero-kicker">BOVINE HEALTH • RESEARCH DASHBOARD</span>'
        f'<h1>{title}</h1><p>{sub}</p></div><span class="hero-chip">INTERPRETABLE ANFIS</span></div>',
        unsafe_allow_html=True,
    )


def card(title, body_html):
    st.markdown(f'<div class="card"><h4>{title}</h4>{body_html}</div>', unsafe_allow_html=True)


def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def callout(text, warn=False):
    st.markdown(f'<div class="callout{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


FRAME_CSS = ("<style>body{margin:0;font:14px -apple-system,'Segoe UI',Roboto,Arial,sans-serif;color:#1b2b24}"
             ".row{display:flex;flex-wrap:wrap;gap:6px;align-items:stretch}.box{flex:1 1 130px;min-width:118px;background:#fff;border:1px solid #d9e2dc;"
             "border-top:4px solid #1f6f54;border-radius:10px;padding:8px 10px}.box b{display:block;color:#1f6f54;font-size:13px;margin-bottom:2px}"
             ".box span{font-size:12px;color:#5b6b63}.arrow{align-self:center;color:#0e7c86;font-size:20px}.box.alt{border-top-color:#0e7c86}"
             ".box.amb{border-top-color:#b7791f}code{background:#eef5f0;padding:0 4px;border-radius:4px;font-size:12px}</style>")


def frame(html, height=260):
    components.html(FRAME_CSS + html, height=height, scrolling=False)


def flow(steps, per_row=4, height=300):
    """steps: list of (title, caption, css_class). Draws boxes with arrows, wrapping every `per_row`."""
    rows = [steps[i:i + per_row] for i in range(0, len(steps), per_row)]
    out = ""
    for r in rows:
        out += '<div class="row" style="margin-bottom:10px">' + '<div class="arrow">&rarr;</div>'.join(
            f'<div class="box {c}"><b>{t}</b><span>{s}</span></div>' for t, s, c in r) + "</div>"
    frame(out, height)
