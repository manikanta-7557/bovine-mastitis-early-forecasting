import { useEffect, useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import {
  Activity, ArrowRight, BarChart3, BookOpen, Check, ChevronDown, CircleAlert,
  Download, FileText, FlaskConical, HeartPulse, Home, Info, Menu, Microscope,
  Network, RotateCcw, ShieldCheck, Sparkles, Stethoscope, Waves, X,
} from "lucide-react";

type Screen = "home" | "detection" | "reasoning" | "study" | "evaluation" | "reports" | "methodology";
type ModelMode = "Sensor-only" | "Full-clinical";
type JsonRecord = Record<string, unknown>;

const API = "/api";
const nav: { id: Screen; label: string; icon: typeof Home }[] = [
  { id: "home", label: "Overview", icon: Home },
  { id: "detection", label: "Clinical detection", icon: Stethoscope },
  { id: "reasoning", label: "Fuzzy reasoning", icon: Network },
  { id: "study", label: "Synthetic onset study", icon: Waves },
  { id: "evaluation", label: "Model evaluation", icon: BarChart3 },
  { id: "reports", label: "Reports", icon: FileText },
  { id: "methodology", label: "Methodology", icon: BookOpen },
];
const labels: Record<string, string> = {
  IUFL: "Front-left · inhale", EUFL: "Front-left · exhale", IUFR: "Front-right · inhale", EUFR: "Front-right · exhale",
  IURL: "Rear-left · inhale", EURL: "Rear-left · exhale", IURR: "Rear-right · inhale", EURR: "Rear-right · exhale",
  temperature: "Temperature (unit not documented)", months_pp: "Months after calving", breed_enc: "Breed (0 / 1)",
  hardness: "Udder hardness (0 / 1)", pain: "Pain (0 / 1)",
};
const RANGE_NOTE = "Cows in the training data had values in this range. A value outside it is adjusted to the nearest supported limit and flagged because the model has not seen such values during training.";

function InfoTip({ text }: { text: string }) {
  return (
    <span className="infotip" tabIndex={0} aria-label={text}>
      <Info size={13} strokeWidth={2} />
      <span className="infotip-bubble">{text}</span>
    </span>
  );
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

function formatName(value: string) {
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function number(value: unknown, digits = 3) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function App() {
  const [screen, setScreen] = useState<Screen>("home");
  const [menuOpen, setMenuOpen] = useState(false);
  const [health, setHealth] = useState<JsonRecord | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    request<JsonRecord>("/health").then(setHealth).catch((cause: Error) => setError(cause.message));
  }, []);

  function navigate(next: Screen) {
    setScreen(next);
    setMenuOpen(false);
    setError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${menuOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Activity size={21} strokeWidth={2.2} /></div>
          <div><strong>ANFIS <span>LAB</span></strong><small>Research workspace</small></div>
          <button className="icon-button mobile-close" onClick={() => setMenuOpen(false)} aria-label="Close menu"><X size={18} /></button>
        </div>
        <div className="side-label">WORKSPACE</div>
        <nav className="nav-list" aria-label="Main navigation">
          {nav.map(({ id, label, icon: Icon }) => (
            <button key={id} className={`nav-item ${screen === id ? "active" : ""}`} onClick={() => navigate(id)}>
              <Icon size={17} strokeWidth={1.8} /><span>{label}</span>{screen === id && <span className="nav-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="integrity-card"><ShieldCheck size={18} /><div><strong>Evidence stays separate</strong><small>Clinical data ≠ synthetic study</small></div></div>
          <div className="sidebar-foot">SIH26109 · Soft Computing<br />Research prototype · Not a diagnostic device</div>
        </div>
      </aside>

      {menuOpen && <button className="mobile-backdrop" onClick={() => setMenuOpen(false)} aria-label="Close navigation" />}
      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left"><button className="icon-button menu-button" onClick={() => setMenuOpen(true)} aria-label="Open menu"><Menu size={19} /></button><span>Research workspace</span><ChevronDown size={14} /></div>
          <div className="topbar-right"><span className="status-dot" /> Python inference API <span className="topbar-divider" /><span>SIH26109</span></div>
        </header>
        {error && <div className="global-error"><CircleAlert size={16} />{error}</div>}
        <div className="content">
          {screen === "home" && <HomeScreen health={health} onNavigate={navigate} />}
          {screen === "detection" && <DetectionScreen onError={setError} onNavigate={navigate} />}
          {screen === "reasoning" && <ReasoningScreen onNavigate={navigate} />}
          {screen === "study" && <StudyScreen onError={setError} />}
          {screen === "evaluation" && <EvaluationScreen onError={setError} />}
          {screen === "reports" && <ReportScreen onNavigate={navigate} />}
          {screen === "methodology" && <MethodologyScreen />}
        </div>
        <footer className="footer"><span>Interpretable ANFIS for detection of existing clinical bovine mastitis</span><span>Research prototype · Results are dataset-scoped</span></footer>
      </main>
    </div>
  );
}

function PageHeading({ eyebrow, title, description, children }: { eyebrow: string; title: string; description: string; children?: ReactNode }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{children}</div>;
}

function HomeScreen({ health, onNavigate }: { health: JsonRecord | null; onNavigate: (screen: Screen) => void }) {
  const models = (health?.models ?? {}) as Record<string, { available?: boolean; data_type?: string }>;
  return <>
    <section className="hero">
      <div className="hero-copy"><div className="hero-kicker"><span className="hero-kicker-dot" /> MASTITIS RESEARCH · SIH26109</div>
        <h1>Interpretable models.<br /><em>Evidence you can inspect.</em></h1>
        <p>A research workspace for detecting existing clinical bovine mastitis with ANFIS, alongside a separately labeled computer-generated onset study.</p>
        <div className="hero-actions"><button className="button button-light" onClick={() => onNavigate("detection")}>Explore clinical detection <ArrowRight size={16} /></button><button className="button button-ghost-light" onClick={() => onNavigate("methodology")}>Read methodology</button></div>
        <div className="hero-footnote"><ShieldCheck size={14} /> Scores are model outputs—not calibrated clinical probabilities or diagnoses.</div>
      </div>
      <div className="hero-art" aria-hidden="true"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit orbit-three" /><div className="orbit-center"><HeartPulse size={34} /></div><div className="orbit-node node-a"><Activity size={16} /></div><div className="orbit-node node-b"><Network size={16} /></div><div className="orbit-node node-c"><Microscope size={16} /></div><div className="orbit-caption">DATA → RULES → INSIGHT</div></div>
      <div className="hero-index">01 / 07</div>
    </section>
    <div className="section-intro"><div><div className="eyebrow">TWO EVIDENCE STREAMS</div><h2>One workspace, separate questions.</h2></div><p>Clinical detection and the generated-data study answer different questions and use different artifacts.</p></div>
    <div className="evidence-grid">
      <button className="evidence-card clinical-card" onClick={() => onNavigate("detection")}><div className="card-icon green-icon"><Stethoscope size={19} /></div><div className="card-meta">REAL CLINICAL DATA · R8 / R13</div><h3>Detection of existing clinical mastitis</h3><p>Score user-entered measurements with saved models and inspect the activated ANFIS rules.</p><div className="card-link">{models.R8?.available ? "Models available" : "Model artifacts unavailable"} <ArrowRight size={15} /></div></button>
      <button className="evidence-card generated-card" onClick={() => onNavigate("study")}><div className="card-icon amber-icon"><FlaskConical size={19} /></div><div className="card-meta">COMPUTER-GENERATED DATA · FC7</div><h3>Seven-day onset study</h3><p>An isolated software experiment. Not validated on real animals or farm records.</p><div className="card-link">{models.FC7?.available ? "Experimental study available" : "Study artifacts unavailable"} <ArrowRight size={15} /></div></button>
    </div>
    <section className="objective-strip"><div className="objective-title"><div className="eyebrow">RESEARCH OBJECTIVES</div><h2>Understand the model,<br />not just the score.</h2></div><div className="objective"><span>01</span><strong>Inspect the learned rules</strong><p>See rule firing, membership, and contribution values computed for the entered measurements.</p></div><div className="objective"><span>02</span><strong>Compare reported evidence</strong><p>Read evaluation results directly from versioned experiment output files.</p></div><div className="objective"><span>03</span><strong>State the limits clearly</strong><p>Keep dataset provenance and the absence of external farm validation visible.</p></div></section>
    <div className="integrity-banner"><Info size={18} /><p><strong>Research boundary.</strong> The real clinical dataset supports existing-status detection only. The onset experiment uses computer-generated data; its performance is not real-farm validation.</p><button onClick={() => onNavigate("methodology")}>Limitations <ArrowRight size={14} /></button></div>
  </>;
}

type DetectionResponse = {
  model: { configuration: string; key: string; rules: number };
  prediction: { score: number; displayed_score: number; presentation_category: string; is_calibrated_probability: boolean };
  warnings: string[];
  ranges: { feature: string; value: number; low: number; high: number; inside_learned_range: boolean; adjusted_value: number }[];
  rule_outputs: JsonRecord[];
  measurements: Record<string, number>;
};

function DetectionScreen({ onError, onNavigate }: { onError: (message: string) => void; onNavigate: (screen: Screen) => void }) {
  const [model, setModel] = useState<ModelMode>("Sensor-only");
  const [fields, setFields] = useState<string[]>([]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [identifier, setIdentifier] = useState("");
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [rangeMap, setRangeMap] = useState<Record<string, { low: number; high: number }>>({});

  useEffect(() => {
    setResult(null);
    setFields([]);
    setRangeMap({});
    request<{ features: string[]; ranges: Record<string, { low: number; high: number }> }>(`/model/schema?mode=${encodeURIComponent(model)}`)
      .then((response) => { setFields(response.features); setRangeMap(response.ranges); })
      .catch((cause: Error) => onError(cause.message));
  }, [model, onError]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    onError("");
    const missing = fields.filter((field) => values[field] === undefined || values[field].trim() === "");
    if (missing.length) {
      onError(`Enter all required measurements: ${missing.join(", ")}.`);
      return;
    }
    const measurements = Object.fromEntries(fields.map((field) => [field, Number(values[field])]));
    if (Object.values(measurements).some((value) => !Number.isFinite(value))) {
      onError("Every measurement must be a finite number.");
      return;
    }
    setBusy(true);
    const payload = { model, identifier: identifier || null, measurements };
    try {
      const response = await request<DetectionResponse>("/predict", {
        method: "POST", body: JSON.stringify(payload),
      });
      sessionStorage.setItem("anfis_last_input", JSON.stringify(payload));
      setResult(response);
      setRangeMap(Object.fromEntries(response.ranges.map((item) => [item.feature, { low: item.low, high: item.high }])));
    } catch (cause) {
      onError(cause instanceof Error ? cause.message : "Prediction failed.");
    } finally { setBusy(false); }
  }

  return <>
    <PageHeading eyebrow="SCREEN 02 · REAL CLINICAL DATA" title="Clinical detection" description="Enter measurements to score existing clinical mastitis patterns using the saved R8 or R13 model. No values are prefilled.">
      <span className="pill pill-green"><span className="status-dot" /> R8 / R13 models</span>
    </PageHeading>
    <div className="notice notice-green"><ShieldCheck size={17} /><div><strong>Existing-status detection only</strong><span>Research prototype—not a diagnosis. Output scores are not calibrated probabilities; category thresholds are not clinically validated.</span></div></div>
    <div className="workspace-grid">
      <form className="panel input-panel" onSubmit={submit}>
        <div className="panel-heading"><div><div className="eyebrow">INPUT MEASUREMENTS</div><h2>Configure a model</h2></div><span className="step-count">01 / 02</span></div>
        <label className="field-label">Model configuration</label>
        <div className="model-choice">
          {(["Sensor-only", "Full-clinical"] as ModelMode[]).map((choice) => <button type="button" className={`model-option ${model === choice ? "selected" : ""}`} key={choice} onClick={() => setModel(choice)}><span className="radio-dot" /><span><strong>{choice === "Sensor-only" ? "R8 · Sensor-only" : "R13 · Extended clinical"}</strong><small>{choice === "Sensor-only" ? "8 udder sensor measurements" : "8 sensors + 5 clinical/context inputs"}</small></span></button>)}
        </div>
        <label className="field-label" htmlFor="identifier">Optional case identifier</label><input id="identifier" className="text-input" value={identifier} maxLength={80} onChange={(event) => setIdentifier(event.target.value)} placeholder="Local reference only" />
        <div className="form-section-title"><span>Required measurements</span><span>{fields.length} fields · source order</span></div>
        <div className="measurement-grid">{fields.map((field, index) => {
          const range = rangeMap[field];
          return <div className="measurement-field" key={field}><label htmlFor={`measure-${field}`}><span className="field-index">{String(index + 1).padStart(2, "0")}</span>{labels[field] ?? formatName(field)}</label>
            <input id={`measure-${field}`} className="text-input" inputMode="decimal" type="number" step="any" value={values[field] ?? ""} onChange={(event) => setValues((current) => ({ ...current, [field]: event.target.value }))} placeholder="Enter measured value" required />
            {range && <small className="learned-range">Range the model learned from (lowest to highest): {number(range.low, 2)} – {number(range.high, 2)} <InfoTip text={RANGE_NOTE} /></small>}
          </div>;
        })}</div>
        <button className="button button-dark submit-button" type="submit" disabled={busy || fields.length === 0}>{busy ? <><span className="spinner" /> Scoring measurements…</> : <>Run model inference <ArrowRight size={16} /></>}</button>
      </form>
      <div className="result-column">
        {!result ? <div className="empty-panel"><div className="empty-icon"><Activity size={23} /></div><div className="eyebrow">READY FOR INPUT</div><h3>Prediction appears here</h3><p>Submit measured values to receive a score, learned-range checks, and actual rule outputs from the Python model.</p><div className="empty-points"><span><Check size={14} /> Actual saved model</span><span><Check size={14} /> Range adjustment warnings</span><span><Check size={14} /> Rule-level explanation</span></div></div> : <DetectionResult result={result} onNavigate={onNavigate} />}
      </div>
    </div>
  </>;
}

function DetectionResult({ result, onNavigate }: { result: DetectionResponse; onNavigate: (screen: Screen) => void }) {
  const percent = Math.max(0, Math.min(1, result.prediction.displayed_score)) * 100;
  const outside = result.ranges.filter((range) => !range.inside_learned_range);
  return <div className="panel output-panel"><div className="panel-heading"><div><div className="eyebrow">MODEL OUTPUT · {result.model.key}</div><h2>Detection result</h2></div><span className="result-tag">{result.prediction.presentation_category}</span></div>
    <div className="score-wrap"><div className="score-number">{number(result.prediction.displayed_score, 2)}</div><div><strong>Displayed model score</strong><span>Not a calibrated clinical probability</span></div></div>
    <div className="score-track"><span style={{ width: `${percent}%` }} /></div>
    <div className="score-caption"><span>0.00</span><span>1.00</span></div>
    <div className="result-divider" />
    <div className="result-detail"><span>Configuration</span><strong>{result.model.configuration} · {result.model.rules} learned rules</strong></div>
    <div className="result-detail"><span>Inputs outside learned range</span><strong>{outside.length === 0 ? <span className="inside-state"><Check size={14} /> All inside learned range</span> : <span className="outside-state"><CircleAlert size={14} /> {outside.length} adjusted</span>}</strong></div>
    {outside.length > 0 && <div className="warning-box"><strong>Outside range — adjusted</strong>{outside.map((range) => <span key={range.feature}>{range.feature}: {number(range.value, 3)} → {number(range.adjusted_value, 3)} (learned {number(range.low, 3)}–{number(range.high, 3)})</span>)}</div>}
    {result.warnings.map((warning) => <div className="warning-box" key={warning}><CircleAlert size={15} />{warning}</div>)}
    <div className="result-actions"><button className="button button-dark" onClick={() => onNavigate("reasoning")}>Inspect rule reasoning <ArrowRight size={15} /></button><button className="button button-outline" onClick={() => onNavigate("reports")}>Create report</button></div>
  </div>;
}

type RuleCondition = { feature: string; centre: number; width: number; your_value: number };
type ModelRule = { rule: number; raw_firing_strength?: number; firing_share?: number; rule_output_for_this_input?: number; weighted_contribution?: number; conditions?: RuleCondition[] };
type Explanation = { dominant_rule: number; active_rules: number[]; measurements: JsonRecord[]; narrative: Record<string, string[]>; rules: ModelRule[] };
type ExplainResponse = { model: string; prediction: { score: number; score_clipped: number; tier: string; warnings: string[] }; explanation: Explanation };

function ReasoningScreen({ onNavigate }: { onNavigate: (screen: Screen) => void }) {
  const [result, setResult] = useState<ExplainResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function loadLatest() {
    const raw = sessionStorage.getItem("anfis_last_input");
    if (!raw) { setError("Run a clinical detection first. No example rules or measurements are inserted here."); return; }
    setBusy(true); setError("");
    try { setResult(await request<ExplainResponse>("/explain", { method: "POST", body: raw })); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Explanation could not be loaded."); }
    finally { setBusy(false); }
  }

  useEffect(() => { void loadLatest(); }, []);
  return <>
    <PageHeading eyebrow="SCREEN 03 · ACTUAL SAVED MODEL" title="Fuzzy reasoning" description="Explore the rules activated by the measurements submitted to the clinical detection model. All values come from the Python explanation engine." />
    {!result ? <div className="empty-panel reasoning-empty"><div className="empty-icon"><Network size={23} /></div><div className="eyebrow">NO MODEL RESULT YET</div><h3>Reasoning needs an actual input</h3><p>{busy ? "Recomputing the saved model's explanation…" : error || "Submit measurements in Clinical detection to calculate rule activations and contributions."}</p>{busy ? <span className="spinner" /> : <button className="button button-dark" onClick={() => onNavigate("detection")}>Go to clinical detection <ArrowRight size={15} /></button>}</div> : <div className="reasoning-layout">
      <div className="panel reasoning-summary"><div className="panel-heading"><div><div className="eyebrow">{result.model} MODEL</div><h2>Input-specific rule activity</h2></div><button className="icon-button" onClick={() => void loadLatest()} title="Refresh actual explanation"><RotateCcw size={16} /></button></div>
        <div className="reasoning-score"><span>Score</span><strong>{number(result.prediction.score, 3)}</strong><small>{result.prediction.tier} · not calibrated</small></div>
        <div className="rule-list">{result.explanation.rules.map((rule) => {
          const share = Number(rule.firing_share ?? 0);
          return <div className="rule-row" key={String(rule.rule)}><div className="rule-row-head"><strong>Rule {String(rule.rule)}</strong><span>{(share * 100).toFixed(1)}% firing share</span></div><div className="rule-track"><span style={{ width: `${Math.min(share * 100, 100)}%` }} /></div><div className="rule-row-foot"><span>Raw firing {number(rule.raw_firing_strength, 4)}</span><span>Contribution {number(rule.weighted_contribution, 4)}</span></div></div>;
        })}</div>
      </div>
      <div className="reasoning-detail">
        <div className="panel"><div className="panel-heading"><div><div className="eyebrow">MEASUREMENT MEMBERSHIP</div><h2>Distance from learned rules</h2></div></div>
          <p className="panel-copy">Memberships and rule centres are read from the saved model for this input; they are not clinical reference ranges.</p>
          <div className="table-scroll"><table><thead><tr><th>Measurement</th><th>Entered</th><th>Dominant centre</th><th>Membership</th></tr></thead><tbody>{result.explanation.measurements.map((entry) => <tr key={String(entry.feature)}><td>{String(entry.feature)}</td><td>{number(entry.entered, 3)}</td><td>{number(entry.dom_centre, 3)}</td><td><span className="membership-pill">{number(entry.dom_mu, 3)}</span></td></tr>)}</tbody></table></div>
        </div>
        <div className="panel narrative-panel"><div className="eyebrow">MODEL-GENERATED EXPLANATION</div><h2>What contributed to the score</h2>{Object.values(result.explanation.narrative).flat().map((sentence, index) => <p key={`${index}-${sentence}`}>{sentence}</p>)}</div>
      </div>
    </div>}
  </>;
}

type CowOption = { cow_id: string; days: number[] };
type StudyResult = { data_notice: string; real_data_limitation: string; cow_id: string; observation_day: number; horizon_days: number; model_score: number; alert_threshold: number; alert: boolean; is_calibrated_probability: boolean; temporal_inputs: Record<string, number>; trajectory: { day_index: number; score: number; alert: boolean }[]; temporal_history: { day_index: number; inputs: Record<string, number> }[]; explanation: Explanation };

function StudyScreen({ onError }: { onError: (message: string) => void }) {
  const [cows, setCows] = useState<CowOption[]>([]);
  const [cowId, setCowId] = useState("");
  const [day, setDay] = useState("");
  const [result, setResult] = useState<StudyResult | null>(null);
  const [busy, setBusy] = useState(false);
  const selectedCow = cows.find((cow) => cow.cow_id === cowId);

  useEffect(() => {
    request<{ cows: CowOption[] }>("/forecast/cows").then((response) => {
      setCows(response.cows);
      if (response.cows.length) {
        const days = response.cows[0].days;
        setCowId(response.cows[0].cow_id);
        setDay(String(days[days.length - 1] ?? ""));
      }
    }).catch((cause: Error) => onError(cause.message));
  }, [onError]);

  async function submit(event: FormEvent) {
    event.preventDefault(); onError("");
    setBusy(true); setResult(null);
    try { setResult(await request<StudyResult>("/forecast", { method: "POST", body: JSON.stringify({ cow_id: cowId, day_index: Number(day) }) })); }
    catch (cause) { onError(cause instanceof Error ? cause.message : "Synthetic study request failed."); }
    finally { setBusy(false); }
  }
  return <>
    <PageHeading eyebrow="SCREEN 04 · FC7 EXPERIMENT" title="Synthetic onset study" description="A separate software experiment on computer-generated daily records. It does not establish future-onset performance on the real clinical dataset." />
    <div className="notice notice-amber"><FlaskConical size={18} /><div><strong>Experimental forecasting using computer-generated data — not validated on real farm records.</strong><span>The real clinical dataset does not establish future-onset forecasting. FC7 uses its saved model, past-only feature construction, and cow-held-out observations.</span></div></div>
    <form className="study-controls panel" onSubmit={submit}><div><label className="field-label" htmlFor="study-cow">Held-out synthetic cow</label><select id="study-cow" className="text-input" value={cowId} onChange={(event) => { setCowId(event.target.value); const next = cows.find((cow) => cow.cow_id === event.target.value); const days = next?.days ?? []; setDay(String(days[days.length - 1] ?? "")); setResult(null); }}><option value="" disabled>Select a cow</option>{cows.map((cow) => <option key={cow.cow_id} value={cow.cow_id}>{cow.cow_id}</option>)}</select></div>
      <div><label className="field-label" htmlFor="study-day">Eligible observation day</label><select id="study-day" className="text-input" value={day} onChange={(event) => { setDay(event.target.value); setResult(null); }}>{(selectedCow?.days ?? []).map((value) => <option key={value} value={value}>Day {value}</option>)}</select></div>
      <button className="button button-dark" type="submit" disabled={busy || !cowId}>{busy ? <><span className="spinner" /> Loading actual study row…</> : <>Score observation <ArrowRight size={15} /></>}</button>
    </form>
    {result ? <div className="study-results">
      <div className="study-summary panel"><div className="panel-heading"><div><div className="eyebrow">FC7 · DAY {result.observation_day}</div><h2>Seven-day model output</h2></div><span className={`result-tag ${result.alert ? "tag-alert" : ""}`}>{result.alert ? "Alert threshold met" : "Below alert threshold"}</span></div>
        <div className="score-wrap"><div className="score-number">{number(result.model_score, 3)}</div><div><strong>Model score for the next seven days</strong><span>Not a calibrated probability · threshold {number(result.alert_threshold, 3)}</span></div></div>
        <div className="notice notice-amber compact"><Info size={16} /><div><strong>Experimental synthetic-data result</strong><span>{result.real_data_limitation}</span></div></div>
        <h3 className="mini-heading">Actual temporal inputs (current day minus prior baseline)</h3><div className="input-chip-grid">{Object.entries(result.temporal_inputs).map(([key, value]) => <div className="input-chip" key={key}><span>{formatName(key)}</span><strong>{number(value, 3)}</strong></div>)}</div>
      </div>
      <div className="panel trajectory-panel"><div className="panel-heading"><div><div className="eyebrow">OBSERVED ELIGIBLE ROWS</div><h2>Study score trajectory</h2></div><span className="muted-small">No future observations generated</span></div><Trajectory data={result.trajectory} selectedDay={result.observation_day} threshold={result.alert_threshold} /></div>
      <div className="panel history-panel"><div className="panel-heading"><div><div className="eyebrow">OBSERVED INPUT HISTORY</div><h2>Recent temporal measurements</h2></div><span className="muted-small">Up to selected day · actual study rows</span></div><p className="panel-copy">These are saved FC7 features (current measurements minus the cow’s previous baseline), shown only for eligible observations up to Day {result.observation_day}.</p><TemporalHistory rows={result.temporal_history} /></div>
      <div className="panel"><div className="eyebrow">ACTUAL FC7 RULES</div><h2>Activated rule output</h2><RuleTable rules={result.explanation.rules} /></div>
    </div> : <div className="empty-panel study-empty"><div className="empty-icon amber-icon"><Waves size={23} /></div><h3>{cows.length ? "Select an eligible observation" : "Synthetic study data unavailable"}</h3><p>{cows.length ? "Scores and trajectories are computed only from existing held-out rows. The interface does not create historical measurements or future outcomes." : "Check the FC7 artifacts and generated study CSV paths in the repository."}</p></div>}
  </>;
}

function Trajectory({ data, selectedDay, threshold }: { data: StudyResult["trajectory"]; selectedDay: number; threshold: number }) {
  if (data.length < 2) return <p className="empty-inline">There are not enough eligible observed points for a trajectory chart.</p>;
  const width = 720, height = 190, pad = 22;
  const scores = data.map((item) => Math.max(0, Math.min(1, item.score)));
  const minDay = data[0].day_index, maxDay = data[data.length - 1].day_index;
  const x = (index: number) => pad + (index / Math.max(data.length - 1, 1)) * (width - 2 * pad);
  const y = (score: number) => height - pad - score * (height - 2 * pad);
  const line = scores.map((score, index) => `${index === 0 ? "M" : "L"} ${x(index)} ${y(score)}`).join(" ");
  const thresholdY = y(threshold);
  return <div className="chart-wrap"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Scores from actual eligible synthetic study observations">
    {[0, 0.5, 1].map((value) => <g key={value}><line x1={pad} x2={width - pad} y1={y(value)} y2={y(value)} className="chart-grid" /><text x="0" y={y(value) + 4} className="chart-label">{value.toFixed(1)}</text></g>)}
    <line x1={pad} x2={width - pad} y1={thresholdY} y2={thresholdY} className="threshold-line" /><path d={line} className="trajectory-line" />
    {scores.map((score, index) => <circle key={data[index].day_index} cx={x(index)} cy={y(score)} r={data[index].day_index === selectedDay ? 5 : 2.8} className={data[index].day_index === selectedDay ? "selected-point" : "trajectory-point"}><title>{`Day ${data[index].day_index}: model score ${score.toFixed(3)}`}</title></circle>)}
    <text x={pad} y={height - 2} className="chart-label">Day {minDay}</text><text x={width - 58} y={height - 2} className="chart-label">Day {maxDay}</text>
  </svg><div className="chart-legend"><span><i className="legend-line" /> Actual eligible model scores</span><span><i className="legend-dashed" /> Validation alert threshold</span><span><i className="legend-dot" /> Selected day {selectedDay}</span></div></div>;
}

function TemporalHistory({ rows }: { rows: StudyResult["temporal_history"] }) {
  const recent = rows.slice(-14);
  const features = Object.keys(recent[0]?.inputs ?? {});
  if (!recent.length) return <p className="empty-inline">No eligible historical measurements are available up to the selected observation.</p>;
  return <div className="table-scroll history-scroll"><table><thead><tr><th>Observation day</th>{features.map((feature) => <th key={feature}>{feature}</th>)}</tr></thead><tbody>{recent.map((row) => <tr key={row.day_index}><td>Day {row.day_index}</td>{features.map((feature) => <td key={feature}>{number(row.inputs[feature], 3)}</td>)}</tr>)}</tbody></table></div>;
}

function RuleTable({ rules }: { rules: ModelRule[] }) {
  return <div className="table-scroll"><table><thead><tr><th>Rule</th><th>Raw firing</th><th>Firing share</th><th>Rule output</th><th>Contribution</th><th>Condition</th></tr></thead><tbody>{rules.map((rule) => <tr key={String(rule.rule)}><td>Rule {String(rule.rule)}</td><td>{number(rule.raw_firing_strength, 5)}</td><td>{number(Number(rule.firing_share) * 100, 1)}%</td><td>{number(rule.rule_output_for_this_input, 4)}</td><td>{number(rule.weighted_contribution, 4)}</td><td><RuleConditions rule={rule} /></td></tr>)}</tbody></table></div>;
}

function RuleConditions({ rule }: { rule: ModelRule }) {
  const conditions = rule.conditions ?? [];
  if (!conditions.length) return <span>Not available</span>;
  return <details className="rule-conditions"><summary>Read IF–THEN rule</summary><p>IF {conditions.map((condition) => `${String(condition.feature)} is near its learned centre ${number(condition.centre, 2)} (width ${number(condition.width, 2)})`).join(" AND ")}, THEN this rule’s linear output for this input is {number(rule.rule_output_for_this_input, 4)}.</p></details>;
}

type ResultData = {
  detection: Record<string, { available: boolean; records: JsonRecord[] | null }>;
  synthetic_study: Record<string, { available: boolean; records: JsonRecord[] | null }>;
  confusion_matrices: { file: string; records: JsonRecord[] | null }[];
};
function EvaluationScreen({ onError }: { onError: (message: string) => void }) {
  const [data, setData] = useState<ResultData | null>(null);
  const [busy, setBusy] = useState(true);
  useEffect(() => { request<ResultData>("/results").then(setData).catch((cause: Error) => onError(cause.message)).finally(() => setBusy(false)); }, [onError]);
  const detection = data?.detection.pooled_grouped_five_fold;
  const synthetic = data?.synthetic_study.synthetic_onset_results;
  const records = detection?.records ?? [];
  const extraMetrics = data?.detection.additional_sensor_only_metrics?.records ?? [];
  const grouped = useMemo(() => records.reduce<Record<string, JsonRecord[]>>((acc, row) => { const key = String(row.feature_set); (acc[key] ??= []).push(row); return acc; }, {}), [records]);
  return <>
    <PageHeading eyebrow="SCREEN 05 · REPORTED ARTIFACTS" title="Model evaluation" description="Tables and comparisons are loaded from saved result files. Clinical detection and the generated-data onset study remain in separate panels." />
    <div className="notice notice-green"><BarChart3 size={18} /><div><strong>Values are not recreated in the browser</strong><span>Metrics below are read from reports/final/summary_pooled.csv and reports/forecast/forecast_summary.csv. No missing result is replaced with a placeholder metric.</span></div></div>
    {busy ? <div className="loading-panel"><span className="spinner" /> Reading experiment result files…</div> : <>
      <section className="evaluation-section"><div className="section-title-row"><div><div className="eyebrow">REAL CLINICAL DETECTION · GROUPED FIVE-FOLD</div><h2>Saved pooled results</h2></div><span className="pill pill-green">reports/final</span></div>
        {!detection?.available ? <EmptyArtifact name="reports/final/summary_pooled.csv" /> : Object.entries(grouped).map(([featureSet, rows]) => {
          const joinedRows = featureSet === "R8"
            ? rows.map((row) => ({ ...row, ...(extraMetrics.find((extra) => extra.model === row.model) ?? {}) }))
            : rows;
          const metrics = featureSet === "R8" ? ["F1", "AUC", "PR_AUC", "Brier"] : ["F1", "AUC"];
          return <div className="panel evaluation-panel" key={featureSet}><div className="panel-heading"><div><div className="eyebrow">{featureSet} FEATURE SET</div><h3>{featureSet === "R8" ? "Sensor-only" : featureSet === "R13" ? "Extended clinical" : featureSet}</h3></div><span className="muted-small">{rows.length} reported models</span></div><MetricsTable rows={joinedRows} metrics={metrics} /></div>;
        })}
        {data?.confusion_matrices.length === 0 ? <p className="empty-inline"><Info size={15} /> No confusion-matrix artifacts were found under reports/final/ or reports/forecast/.</p> : data?.confusion_matrices.map((matrix) => <div className="panel evaluation-panel" key={matrix.file}><div className="panel-heading"><h3>{matrix.file}</h3></div>{matrix.records?.length ? <GenericTable rows={matrix.records} /> : <p className="empty-inline">A confusion-matrix file exists but is not a CSV supported by this view.</p>}</div>)}
        <p className="integrity-note"><Info size={15} /> These results describe the saved grouped evaluation on this research dataset. They are not external clinical validation; model comparisons should be read directly, without assuming ANFIS is the best performer.</p>
      </section>
      <section className="evaluation-section synthetic-eval"><div className="section-title-row"><div><div className="eyebrow">COMPUTER-GENERATED DATA · FC7</div><h2>Separate onset-study results</h2></div><span className="pill pill-amber">reports/forecast</span></div>
        {!synthetic?.available ? <EmptyArtifact name="reports/forecast/forecast_summary.csv" /> : <>
          <div className="notice notice-amber compact"><FlaskConical size={16} /><div><strong>Experimental forecasting using computer-generated data — not validated on real farm records.</strong><span>Do not interpret these results as clinical performance.</span></div></div>
          {synthetic.records?.filter((row) => row.target === "clinical_within_7d").map((row, index) => <div className="panel evaluation-panel forecast-metric" key={`${String(row.feature_set)}-${String(row.model)}-${index}`}><div className="forecast-metric-head"><div><strong>{String(row.model)}</strong><span>{String(row.feature_set)} · {String(row.target)}</span></div><span className="metric-chip">F1 {number(row.F1, 3)}</span></div><MetricBars values={[["ROC-AUC", row.ROC_AUC], ["PR-AUC", row.PR_AUC], ["Precision", row.precision], ["Recall", row.recall]]} /></div>)}
        </>}
      </section>
    </>}
  </>;
}

function MetricsTable({ rows, metrics }: { rows: JsonRecord[]; metrics: string[] }) {
  return <div className="table-scroll"><table><thead><tr><th>Model</th><th>Feature set</th>{metrics.map((metric) => <th key={metric}>{metric}</th>)}<th>Evidence source</th></tr></thead><tbody>{rows.map((row, index) => <tr key={`${String(row.model)}-${index}`}><td><strong>{String(row.model)}</strong></td><td>{String(row.feature_set)}</td>{metrics.map((metric) => <td key={metric}><div className="metric-cell"><span>{number(row[metric], 3)}</span>{metric !== "Brier" && row[metric] !== null && row[metric] !== undefined && <span className="metric-bar"><i style={{ width: `${Math.max(0, Math.min(100, Number(row[metric]) * 100))}%` }} /></span>}</div></td>)}<td><span className="data-source-tag">Grouped evaluation</span></td></tr>)}</tbody></table></div>;
}

function GenericTable({ rows }: { rows: JsonRecord[] }) {
  const columns = Object.keys(rows[0] ?? {});
  return <div className="table-scroll"><table><thead><tr>{columns.map((column) => <th key={column}>{formatName(column)}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}>{columns.map((column) => <td key={column}>{String(row[column] ?? "—")}</td>)}</tr>)}</tbody></table></div>;
}

function MetricBars({ values }: { values: [string, unknown][] }) {
  return <div className="metric-bars">{values.map(([label, raw]) => { const value = typeof raw === "number" ? raw : Number(raw); return <div className="metric-bar-row" key={label}><span>{label}</span><div className="metric-bar"><i style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} /></div><strong>{number(value, 3)}</strong></div>; })}</div>;
}

function EmptyArtifact({ name }: { name: string }) { return <div className="empty-inline"><CircleAlert size={16} /> No report file is available at <code>{name}</code>; no metrics are shown.</div>; }

function ReportScreen({ onNavigate }: { onNavigate: (screen: Screen) => void }) {
  const [downloadBusy, setDownloadBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [hasInput, setHasInput] = useState(() => Boolean(sessionStorage.getItem("anfis_last_input")));
  async function download() {
    const raw = sessionStorage.getItem("anfis_last_input");
    if (!raw) { setStatus("Run a clinical detection with measured inputs before creating a report."); return; }
    setDownloadBusy(true); setStatus("");
    try {
      const response = await fetch(`${API}/report`, { method: "POST", headers: { "Content-Type": "application/json" }, body: raw });
      if (!response.ok) { const result = await response.json(); throw new Error(result.detail ?? `Report failed (${response.status})`); }
      const blob = await response.blob(); const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
      anchor.href = url; anchor.download = response.headers.get("content-disposition")?.match(/filename="([^"]+)"/)?.[1] ?? "clinical-detection.pdf"; anchor.click(); URL.revokeObjectURL(url);
      setHasInput(true);
      setStatus("PDF generated from the saved model's actual prediction and explanation.");
    } catch (cause) { setStatus(cause instanceof Error ? cause.message : "PDF generation failed."); }
    finally { setDownloadBusy(false); }
  }
  return <>
    <PageHeading eyebrow="SCREEN 06 · PDF OUTPUT" title="Reports" description="Generate a readable PDF from an actual clinical-model result. The report includes the measurements, learned-range warnings, explanation, and research limitations." />
    <div className="report-layout"><div className="panel report-card"><div className="report-icon"><FileText size={23} /></div><div className="eyebrow">CLINICAL DETECTION REPORT</div><h2>Detailed model explanation</h2><p>PDF content is produced by explain_report.py using the submitted values and saved R8/R13 model. It does not present the score as a calibrated probability or diagnosis.</p>
      <ul className="check-list"><li><Check size={15} /> Project title and research prototype disclaimer</li><li><Check size={15} /> Selected model configuration and input values</li><li><Check size={15} /> Actual ANFIS rules, firing strengths, and contributions</li><li><Check size={15} /> Learned-range and weak-match warnings</li><li><Check size={15} /> Research limitations and no external farm validation</li></ul>
      <button className="button button-dark" onClick={() => void download()} disabled={downloadBusy}>{downloadBusy ? <><span className="spinner" /> Generating PDF…</> : <><Download size={16} /> Download clinical report</>}</button>{status && <p className="form-status">{status}</p>}</div>
      <div className="report-preview"><div className="preview-toolbar"><span /><span /><span /><small>REPORT PREVIEW</small></div><div className="paper"><div className="paper-kicker">RESEARCH PROTOTYPE · SIH26109</div><h3>Detailed model explanation</h3><p>Interpretable ANFIS for detection of existing clinical bovine mastitis</p><div className="paper-line wide" /><div className="paper-line" /><div className="paper-block"><span>MODEL OUTPUT</span><div className="paper-line wide" /><div className="paper-line medium" /></div><div className="paper-block"><span>RULE CONTRIBUTIONS</span><div className="paper-line wide" /><div className="paper-line wide" /><div className="paper-line medium" /></div><div className="paper-disclaimer">Not a diagnosis · Not a calibrated clinical probability · No external farm validation</div></div></div></div>
    <div className="integrity-banner report-cta"><Info size={18} /><p>{hasInput ? "This report uses the last clinical measurements submitted in this browser session." : "No clinical measurements have been submitted in this browser session. Start with measured values; example or invented cow measurements are not preloaded."}</p><button onClick={() => onNavigate("detection")}>Open detection form <ArrowRight size={14} /></button></div>
  </>;
}

function MethodologyScreen() {
  return <>
    <PageHeading eyebrow="SCREEN 07 · METHODS & LIMITATIONS" title="Methodology" description="A concise account of the inference path, datasets, evaluation design, and limits documented in this repository." />
    <div className="method-hero"><div className="method-symbol"><Network size={27} /></div><div><div className="eyebrow">FIRST-ORDER SUGENO ANFIS</div><h2>Fuzzy rules with learned parameters</h2><p>Membership functions softly match inputs to rule centres. Normalized rule firing weights combine each rule’s linear output into one model score.</p></div></div>
    <div className="method-grid">
      <div className="panel method-card"><div className="method-step">01</div><h3>Fuzzification</h3><p>Gaussian membership functions convert each scaled measurement into a degree of match for each learned rule.</p><div className="formula">μ = exp(−½ ((x − c) / s)²)</div></div>
      <div className="panel method-card"><div className="method-step">02</div><h3>Rule firing</h3><p>Membership values are combined into raw rule firing strengths, then normalized to shares that sum to one.</p><div className="formula">w̄ᵢ = wᵢ / Σ wⱼ</div></div>
      <div className="panel method-card"><div className="method-step">03</div><h3>Weighted output</h3><p>Each first-order Sugeno rule has a linear output. Its share-weighted value contributes to the final score.</p><div className="formula">score = Σ w̄ᵢ · fᵢ(x)</div></div>
    </div>
    <div className="method-grid two-col">
      <div className="panel"><div className="eyebrow">REAL CLINICAL DATA</div><h2>Existing-status detection</h2><ul className="bullet-list"><li>R8 uses eight udder sensor measurements; R13 also includes the extended clinical/context fields supported by its saved scaler.</li><li>Cow-grouped splits keep a cow’s daily rows together. Scalers are fit on training data.</li><li><code>Milk_visibility</code> duplicates the target in the working file and is excluded from model features.</li><li>The current working file differs from Version 1; its Version 2 provenance remains unresolved.</li><li>There is no external farm validation; this is a research prototype, not a diagnostic tool.</li></ul></div>
      <div className="panel"><div className="eyebrow">SEPARATE COMPUTER-GENERATED STUDY</div><h2>Seven-day onset experiment</h2><ul className="bullet-list"><li>FC7 uses the saved model and generated daily dataset, with past-only temporal features and cow-level held-out data.</li><li>The UI plots existing eligible study rows only; it does not invent a future trajectory.</li><li>Synthetic scores do not establish clinical performance or real-farm validity.</li><li>The real clinical dataset has short histories and limited within-cow status transitions, so it cannot establish future-onset forecasting.</li></ul></div>
    </div>
    <div className="notice notice-amber"><CircleAlert size={18} /><div><strong>Interpretation limits</strong><span>Model scores are not calibrated probabilities. Presentation categories are not clinical cut-offs. Learned rules have not been reviewed by a veterinary expert, and dataset provenance remains unresolved.</span></div></div>
    <div className="comparison-callout"><Sparkles size={18} /><p><strong>Why ANFIS?</strong> Its value here is the ability to inspect a small set of data-learned fuzzy rules—not a claim that it outperforms Random Forest. The saved evaluation artifacts show model-specific comparisons directly.</p></div>
  </>;
}

export default App;
