# Project State and Roadmap
**Interpretable ANFIS for Bovine Mastitis Detection** (Soft Computing, M.Tech, VIT-AP; problem statement SIH26109). Review 1 materials said "early"; that word is no longer supported (claim C5 in `WORKFLOW.md`).
Last updated: 2026-10-25, v4 (see the Changelog at the end). Review II is on **Tuesday 2026-10-27**. Owners: A = Manikanta, B = teammate.

---

## 0. Read me first (for humans AND for any AI assistant helping with this project)

Rules that protect the project (the evaluating faculty is strict):
1. **No invented numbers, no invented citations.** Every number comes from `reports/*.csv` or `data/processed/data_prep_summary.json`. Every citation must be opened and verified by a human.
2. **Never claim "forecasting" or "predicting future onset."** The data only supports *detecting existing mastitis status* from measurements (section 3, point 7).
3. **Never claim ANFIS beats Random Forest or the Decision Tree.** Current results say it does not (section 2).
4. `Milk_visibility` is identical to the label (r = 1.0). It must never be a model input.
5. Tune only on validation folds. The test fold is scored once.
6. The student is still learning these concepts. Define every term before using it, give one command at a time when asked, and say what each command does and why.
7. When unsure what a number means, ask for the file. Do not guess.

Project instruction summary: every algorithm choice needs a first-principles explanation plus a "Defense Guide" (why this, what alternatives, why rejected). Pace: do not advance to the next phase until the student confirms understanding.

---

## 1. The project in plain English

| Term | Plain meaning | In our data |
|---|---|---|
| **Row** | One measurement of one cow on one day | 6,600 rows |
| **Cow_ID** | Which cow the row belongs to | 1,100 cows x 6 days (Day 1-6) |
| **Features** | The measurements a model may look at | 8 udder sensor readings (IUFL ... EURR = inhale and exhale limits of the 4 quarters), Temperature, Months after giving birth, Breed, Hardness, Pain |
| **Label (class1)** | The answer we want the model to give: 0 = healthy, 1 = mastitis | 3,961 healthy rows, 2,639 mastitis rows (about 40%) |
| **Train / test split** | Teach the model on some cows, then quiz it on cows it has never seen. Like studying from a textbook, then sitting an exam with new questions. | 880 cows to train, 220 cows to test |
| **Why split by cow** | One cow has 6 almost identical rows. If days of the same cow land in both train and test, the model just recognises the cow ("memorising answers"). Test scores would look better than they should. | `StratifiedGroupKFold` keeps each cow on one side |
| **Leakage** | A feature that secretly contains the answer | `Milk_visibility` is the label copied; dropped |
| **Baseline** | Ordinary, well-known models to compare against, so our result has meaning | Majority class, Logistic Regression, Decision Tree (depth 4), Random Forest |
| **Udder Swelling Index** | Average of the 8 sensor readings: "how swollen is the udder overall" | engineered in `data_prep.py` |
| **Udder Asymmetry Index** | How unequal the 4 quarters are (mastitis usually hits one quarter): (largest - smallest quarter) / average quarter | our own design choice, defend it |
| **Early-Warning set** (code name; in slides say "sensor-only") | Sensor-based features only (no human exam) | swelling, asymmetry, temperature, months, breed |
| **Full-Clinical set** (in slides say "sensor + clinical signs") | Early-Warning + Hardness + Pain (needs a person to feel the udder) | 7 features |
| **Fuzzy rule** | An IF-THEN rule with soft edges: "IF swelling is HIGH AND asymmetry is HIGH THEN risk is high". HIGH is a degree from 0 to 1, not yes/no. | each ANFIS rule |
| **ANFIS** | A set of fuzzy rules whose shapes and outputs are *learned from data*. Think of a panel of ~10 specialists; each votes with a strength equal to how well the cow matches their description; the final score is the weighted average. | `anfis.py` |
| **Subtractive clustering** | Finds ~10 "typical cows"; each becomes one rule. Avoids needing thousands of rules. | `subtractive_clustering.py` |
| **Hybrid learning** | Each round: (1) solve the rule outputs exactly with least squares, (2) nudge the rule shapes with gradient descent | `ANFIS.fit` |

**Metrics in plain words** (mastitis = "positive"):
- **Accuracy**: share of rows classified correctly. Misleading here because 60% are healthy.
- **Precision**: of cows flagged mastitis, how many truly were. Low precision = false alarms.
- **Recall (sensitivity)**: of cows that truly had mastitis, how many we caught. Low recall = missed sick cows. For a health screen, this matters most.
- **Specificity**: of healthy cows, how many we correctly cleared.
- **F1**: one number balancing precision and recall (1.0 perfect, 0 useless).
- **AUC**: probability that a random sick cow gets a higher risk score than a random healthy cow. 0.5 = coin flip, 1.0 = perfect ranking. Does not depend on the 0.5 threshold.
- **Row vs cow level**: row = each daily measurement judged separately; cow = the 6 daily scores averaged into one verdict per cow (the label is almost constant per cow, so cow level is the fairer view).

---

## 2. Current state: real results (reported by A on 2026-10-24, from `reports/comparison.csv`)

**Caution (2026-10-25): these results are on A's working CSV, whose source is unverified and which is NOT the Mendeley Version 1 CSV (section 2b).** Claude could not reproduce them because the working file has not been shared.

Test fold = 220 cows, 1,320 rows.

| Feature set | Model | Row F1 | Row AUC | Cow F1 |
|---|---|---:|---:|---:|
| Early-Warning | Majority class | 0.0000 | 0.5000 | 0.0000 |
| Early-Warning | Logistic Regression | 0.8590 | 0.9020 | 0.8590 |
| Early-Warning | Decision Tree (d=4) | 0.8987 | 0.9470 | 0.8944 |
| Early-Warning | Random Forest | 0.9301 | 0.9834 | 0.9467 |
| Early-Warning | **ANFIS (10 rules)** | 0.8694 | 0.9169 | 0.8675 |
| Full-Clinical | Majority class | 0.0000 | 0.5000 | 0.0000 |
| Full-Clinical | Logistic Regression | 0.8873 | 0.9177 | 0.8805 |
| Full-Clinical | Decision Tree (d=4) | 0.9211 | 0.9455 | 0.9212 |
| Full-Clinical | Random Forest | 0.9277 | 0.9761 | 0.9405 |
| Full-Clinical | **ANFIS (12 rules)** | 0.9076 | 0.9380 | 0.9091 |

Full metrics for the two headline pairs:

| Feature set | Model | Accuracy | Precision | Recall | Specificity | F1 | AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Early-Warning | Random Forest | 0.9455 | 0.9542 | 0.9072 | 0.9710 | 0.9301 | 0.9834 |
| Early-Warning | ANFIS | 0.9008 | 0.9179 | 0.8258 | 0.9508 | 0.8694 | 0.9169 |
| Full-Clinical | Random Forest | 0.9447 | 0.9730 | 0.8864 | 0.9836 | 0.9277 | 0.9761 |
| Full-Clinical | ANFIS | 0.9303 | 0.9658 | 0.8561 | 0.9798 | 0.9076 | 0.9380 |

Run facts: radius grid ra in {0.3, 0.4, 0.5, 0.6, 0.7}; validation AUC range 0.853-0.913 (Early) and 0.948-0.955 (Full); the script caps rules at 12 and the Full model used all 12; 7 test values were clipped by the train-only scaler; one cow's label changes across days.

**Not yet seen by Claude (paste these):** `reports/anfis_ra_grid.csv`, `reports/rules_Early-Warning.txt`, `reports/rules_Full-Clinical.txt`, `reports/example_explanations.txt`, `reports/anfis_results.csv` (chosen ra, epochs).

---

## 2b. Dataset facts checked against the source page (2026-10-25)

Source: Mendeley Data, "Data for: Clinical Mastitis in Cows based on Udder Parameter using Internet of Things (IoT)", CC BY 4.0; contributors Ankitha K, Manjaiah D H, Kartik M. URL: https://data.mendeley.com/datasets/kbvcdw5b4m

**Two versions exist** (the description text is identical in both):

| Version | DOI | Published | File listed on the page |
|---|---|---|---|
| 1 | 10.17632/kbvcdw5b4m.1 | 22 Oct 2020 | `clinical_mastitis_cows_version1.csv`, 413 KB |
| 2 | 10.17632/kbvcdw5b4m.2 | 20 Nov 2020 | `Clinical_Mastitis_cows_version2.zip`, 1.48 GB (contents not described on the page; the size suggests more than a CSV, possibly the milk images, but that is a guess) |

**Finding of 2026-10-25: our working `clinical_mastitis_cows.csv` is NOT the Version 1 CSV.** A uploaded the Version 1 CSV (422,719 bytes, SHA256 `11e1646b...b718b92`) to the Project. Claude compared it with the statistics A reported for the working file on 2026-10-07. Same shape (6,600 x 18, 1,100 cows x 6 days) and same breed, months-after-birth and previous-mastitis values, but different labels and different clinical columns:

| Statistic | Version 1 CSV (Project file) | Our working file (reported by A) |
|---|---|---|
| `class1` rows 0 / 1 | 5,490 / 1,110 (16.8% mastitis; 185 of 1,100 cows) | 3,961 / 2,639 (40%; inferred: 439 whole cows + 5 rows of cow10, since 2,639 = 439 x 6 + 5) |
| Cows whose label changes across days | 0 | 1 (cow10) |
| Correlation with label: Hardness / Pain / Temperature | 0.977 / 0.987 / 0.706 | 0.748 / 0.559 / 0.383 |
| Single-rule accuracy: Hardness / Pain | 0.994 / 0.996 | 0.872 / 0.773 |
| Test-fold row F1, Early-Warning: LogReg / Tree / RF / ANFIS | 0.972 / 0.972 / 0.972 / 0.972 | 0.859 / 0.899 / 0.930 / 0.869 |
| Test-fold row F1, Full-Clinical: LogReg / Tree / RF / ANFIS | 0.958 / 0.958 / 0.958 / 0.958 | 0.887 / 0.921 / 0.928 / 0.908 |

(Version 1 results were produced by Claude with the unchanged pipeline; the working-file column is as reported by A.) On Version 1 the problem is nearly trivial: Hardness, Pain and Milk_visibility almost equal the label and all four models tie.

**Provenance update (2026-10-25, later).** Per A's account, the working CSV (406.4 KB, SHA256 `e4166166...f532d8`, class1 3,961 / 2,639) was taken from the **Version 2 zip** (`Clinical_Mastitis_cows_version2.zip`, which A says also holds the milk images). **Still to confirm:** its SHA256 must equal the CSV extracted from that zip. What A's `verify_dataset_version.py` output shows (Version 1 vs working file): same 6,600 rows and the same Cow_ID+Day keys; `class1` and `Milk_visibility` each differ in exactly **1,529** rows, which equals 2,639 - 1,110, so **every change is healthy (0) -> mastitis (1)**; 1,529 = 254 x 6 + 5, consistent with 254 whole cows relabelled plus 5 days of cow10 (an inference until `compare_versions_deep.py` confirms it); the 8 sensor columns differ in about 1,490-1,507 cells each, Temperature in 351, Hardness in 725, Pain in 54. **Why Version 2 differs is not documented on the page** (re-measurement, relabelling from the milk images, correction or augmentation are all possible). Do **not** call it synthetic, fabricated or manipulated unless the authors or a paper say so (claim C14). An external AI-generated report (`ANFIS_Mastitis_Detection_Research.pdf`, reviewed in section 2c) quotes correlations between Version 1 and working-file sensor values in relabelled rows (IUFL -0.017, EUFL +0.046, Temperature 0.770); we have not reproduced those numbers (run `compare_versions_deep.py`, its section 6). **Decision D13:** main dataset = Version 2 (the authors' latest release, the file A actually downloaded), Version 1 = sensitivity check, differences disclosed; if the hash check fails, fall back to the source of truth we can document. **Rule: choose the dataset by provenance, never by which file makes ANFIS look better.**

| Topic | What the page says | What it means for us |
|---|---|---|
| Purpose | Data "collected from the udder of a cow to detect **clinical** mastitis" | Our label is *clinical* mastitis. Do **not** call it subclinical. Avoid "early" unless we can show detection earlier than visible signs. |
| Sensors | "four flex sensors and a temperature sensor", read via Raspberry Pi and an analog-to-digital converter | One flex sensor per quarter |
| `IUFL ... EURR` | Data attributes include "size of an udder (udder front left inhale and exhale limit, udder front right inhale and exhale limit, udder rear left ..., udder rear right ...)" = 4 quarters x 2 limits = 8 columns. Column names are not defined. | **I = inhale limit, E = exhale limit, very likely** (the 8 columns and the letters match). Write "very likely" until a source names the columns. Our earlier "inner/external" wording was wrong. The per-quadrant sum I+E is still a valid size measure. |
| `Hardness` | "the hardness of an udder from user input via a switch" | A human-entered flag (why it needs a person) |
| `Pain` | "pain due to swelling of an udder" | No further detail |
| `Milk_visibility` | Milk images processed; "0 indicates the normal milk and 1 indicates the abnormal milk" | Equals our label on every row, so in practice **the label is the milk-abnormality flag** |
| `class1` | "normal cows as 0 and mastitis cows as 1"; **how it was assigned is not explained** | Undocumented. State this limitation in the report. |
| Counts | Number of cows/days not stated | Our counts (1,100 cows x 6 days) come from our own file |

Notes:
- **cow10** (the only cow whose label changes): Days 1-5 Hardness 1, label 1; Day 6 Hardness 0, label 0 (temperature 43 and Pain 0 throughout; Jersey). Consistent with a label tied to milk appearance, but whether it is real recovery or a recording error cannot be told from the data. **Decision: keep, and log it in `JOURNAL.md`.** One cow of 1,100 (6 of 6,600 rows) cannot move the results.
- Mendeley Data deposits are not peer-reviewed by themselves. The title "Data for: ..." suggests a companion paper. **Not found yet.** One search found only a lead: a 2025 review ([Atlantis Press PDF](https://www.atlantis-press.com/article/126010069.pdf)) lists, as its reference [28], a 2021 paper by Ankitha and Manjaiah, "Comparison of KNN and SVM algorithms to detect clinical mastitis in cows using Internet of Animal Health Things", and says it found SVM (>90%) above KNN (~85%). That is second-hand: do **not** cite it or quote those numbers until someone opens the original. Avoid the phrase "peer-reviewed dataset" until a companion paper is verified.

## 2c. Review of the external research PDF (2026-10-25)

A attached `ANFIS_Mastitis_Detection_Research.pdf` (18 pages, written by another AI system; it never saw our code). Treat it as a list of leads, not as a source. 

| Part of the PDF | Verdict | Why |
|---|---|---|
| Drop `Milk_visibility` (target leakage); cow-grouped CV; train-only scaling; ridge LSE; Chiu accept/reject rule | **Already done** in our code | `data_prep.py`, `anfis.py`, `subtractive_clustering.py`. The PDF's "repository audit" lists bugs we do not have; it guessed ("likely lacks") |
| "The working file is manipulated / synthetic / fabricated; discard it" | **Do not repeat** | Overclaim. It is (per A) the authors' own Version 2. The correlation numbers are untraced; cause undocumented (C14) |
| "Use Version 1 as sole truth" | **A decision, not a fact** | On Version 1 all models tie (row F1 0.972 / 0.958), so it cannot separate methods. D13 keeps both: Version 2 main, Version 1 sensitivity |
| Wilcoxon signed-rank on 5 fold scores, p < 0.05 | **Wrong as stated** | With n = 5 pairs the smallest possible two-sided p is 2/32 = 0.0625. Use paired cow-bootstrap CIs on the pooled out-of-fold predictions and McNemar (D14) |
| EU AI Act as the reason for ANFIS | **Weak, drop or hedge** | The Act regulates AI systems affecting people; a cattle tool is not an obvious high-risk case; sources are Medium and a vendor glossary. Use "veterinarians need auditable reasoning" instead |
| Defence script "your proposal claimed 99% accuracy" | **Do not use** | Our proposal did not claim that; do not invent a story |
| Prune rules whose mean normalised firing strength is below 0.05 | **Adopt (cheap)** | Add to `explain.py` after Review II; report rules kept vs pruned |
| Feature ablation with / without Temperature; 5 seeds; save fold indices as JSON; SHA256 assertion in `data_prep.py` | **Adopt (Final Review)** | Cheap and good practice |
| Add a gradient-boosting baseline (Ebrahimi 2019 found GBT/deep learning best for subclinical mastitis) | **Adopt (Final Review)** | `HistGradientBoostingClassifier` needs no extra install |
| Citations (61 links) | **Verify each before use** | Several are off-topic (an EEG paper, a thin-film optics paper, a milling-roughness paper, Medium, Scribd, a tutorial site); "Pisani 2023" points to an author profile page, not a paper; "Yager 2010" looks like a misattributed edited book; the "99.46% KNN" figure and the 84.9% figure are unverified until the papers are opened |

---

## 3. What the numbers honestly say

1. **Ranking:** ANFIS beats Logistic Regression but is below the Decision Tree and Random Forest on F1 and AUC in both feature sets.
2. **Sensors alone are strong.** Random Forest reaches AUC 0.983 on the Early-Warning set. Hardness and Pain add nothing for RF (F1 0.9277 vs 0.9301). Potential contribution: "udder-sensor indices detect mastitis without human palpation." Caution: one test fold of 220 cows; a 0.003 gap is noise.
3. **Non-linear structure:** LogReg (AUC 0.90) vs RF (0.98) on the same features means interactions matter. Either the ANFIS needs more rules, or the two composite indices throw away information that RF extracts.
4. **The 12-rule cap was hit** on the Full set. The cap may be limiting accuracy (a fixable, cheap experiment).
5. **ANFIS recall (Early) is 0.826.** About 1 in 6 sick rows is missed at the 0.5 threshold. For screening, choose the threshold on validation to favour recall, and report it.
6. **Biggest viva risk:** a depth-4 Decision Tree is also interpretable and is *more* accurate than our ANFIS. We need evidence for what fuzzy rules add: a smooth continuous risk score, graded rule firing, robustness to sensor noise, a risk tier. These are experiments in Phase B (B6), not claims to make from memory.
7. **Detection, not forecasting.** The label is constant within 1,099 of 1,100 cows, so Days 1-6 are repeated measurements of one status. Say "detection from sensor measurements". The label is clinical mastitis (section 2b), so avoid the words "early" and "subclinical" unless you can show detection earlier than visible signs.
8. **Not clinically validated.** One dataset, one source; no claim about other farms.

---

## 4. Roadmap

### Phase A: Review II, now to Tue 2026-10-27 (30 marks)

| When | Owner | Task | Done when | Rubric |
|---|---|---|---|---|
| Sat night | A | Commit/push repo, add B | B can run all 4 scripts | - |
| Sat night | A+B | B runs the pipeline; compare `comparison.csv` | Numbers identical on both machines (if not, find why first) | Implementation |
| Sat night | A | Paste the 5 files listed in section 2 to Claude | Claude has rules + chosen radius | Prelim results |
| Sun | B | `python audit_odd_cow.py`: the one cow whose label changes | Decision (keep/drop) written in `JOURNAL.md` | Data prep |
| Sun | A (+B) | **T0.4 Dataset provenance:** (1) extract the CSV from the Version 2 zip (`unzip Clinical_Mastitis_cows_version2.zip "*.csv" -d v2/`); (2) `python verify_dataset_version.py data/raw/clinical_mastitis_cows.csv v2/<that csv>` must say IDENTICAL; (3) `python compare_versions_deep.py <Version 1 CSV> data/raw/clinical_mastitis_cows.csv` and paste the output to Claude (T0.5); (4) share the working CSV with Claude | Journal line: source + SHA256 + which official file it matches | Data prep |
| Sun | B | Try to find and open the 2021 KNN-vs-SVM paper (an unverified lead, section 2b); check whether it uses this dataset and defines `I*`/`E*` and the label rule | Verified citation, or a note "not found" | Lit survey, Data prep |
| Sun | A | Literature table, 6-8 rows: paper, method, black-box or interpretable, gap. Reuse the citations already verified in the Phase I proposal | Every row opened and checked by a human | Lit survey (3) |
| Sun | A | Research gap in 3 sentences | Mentions: black-box literature; no rule-level reasoning; detection from sensors | Lit survey |
| Sun | B | Read `anfis.py` and `subtractive_clustering.py` line by line; prepare a 15-minute explanation for A | B explains without the code open | SC justification (2) |
| Sun (optional) | A+B | **Diagnostics D1** (see B1 below): RF on raw 8 sensors vs on indices; RF feature importance; ANFIS with `MAX_RULES` 20 and radius grid 0.2-0.7 | Table of results, honest report whether it helps | Prelim results (4) |
| Sun | Claude | Architecture diagram, data-flow diagram, article draft, slides v1 with speaking notes | Generated from real numbers | System design (4), Article (4) |
| Mon | A | **T1.8** `leakage_replication.py` (10 min), paste the tables to Claude | `reports/leakage_replication.csv` | Results (4) |
| Mon | A+B | Cross-teach (15 min each way); trace every slide number to a file | No untraceable number | Viva (2) |
| Mon | A+B | Viva drill with section 5 questions, out loud | Each person answers 5 | Viva (2) |
| Mon 6 pm | A+B | **Freeze.** `git tag review2` | | - |
| Tue | A+B | Review II | | |

**Review II slide plan (11 slides):** 1 Title and team. 2 Problem (SIH26109) and why mastitis matters. 3 Research gap (black-box literature). 4 Dataset (1,100 cows, 6 days, 40% mastitis). 5 Data findings: leakage found, labels constant per cow so *detection*. 6 Architecture (4 layers). 7 Data prep and cow-grouped split. 8 Results table (baselines + ANFIS) with the honest reading. 9 What ANFIS adds: one real rule set, one explained cow, risk tier. 10 Limitations and Final Review plan (B1-B9). 11 Article plan (working title, abstract, 2 candidate venues to verify).

**Article for Review II (rubric item 6):** title, abstract (150 words), methodology outline, preliminary-results table, target venue. Suggested working title: *Interpretable Neuro-Fuzzy Detection of Bovine Mastitis from Udder Sensor Indices*. Venue candidates to check yourselves for scope, indexing and fees before naming one: MDPI *AgriEngineering* (where the MasPA benchmark appeared), *Computers and Electronics in Agriculture*.

### Phase B: Final Review (40 marks). Date: ____ (ask your faculty; fill in)

| # | Task | What exactly | Done when |
|---|---|---|---|
| **B1** | Fix the accuracy gap | (a) RF on 8 raw sensors vs on indices: if raw is much better, the indices lose information, so add features (max/min quadrant total, std across quadrants, inhale/exhale ratio). (b) `MAX_RULES` 12 -> 20-30, radius grid 0.15-0.7. (c) learning rate, epochs, ridge. (d) choose decision threshold on validation (target recall). (e) add per-quadrant expansion features (E minus I); the dataset page lists inhale and exhale limits per quarter (I*/E* mapping very likely). | Validation AUC/F1 table; keep only changes that help on validation |
| **B2** | GA tuning | Chromosome = (ra, rb factor, max_rules, learning rate). Fitness = validation AUC minus small penalty per rule. Population 20, 15 generations, tournament selection, blend crossover, Gaussian mutation, fixed seeds. Compare GA vs grid. | `ga_tune.py`, convergence plot, GA vs grid table |
| **B3** | Evaluation protocol | Grouped 5-fold CV using the existing `fold` column; radius chosen *inside* each training split (nested). Report mean +/- std and cow-bootstrap CIs. | One results table across 5 folds for every model |
| **B4** | Competing algorithms (need at least 2 beyond ANFIS) | Already: LogReg, Decision Tree, Random Forest. Add: SVM (RBF), MLP, Gradient Boosting; stage-1 plain Sugeno fuzzy (no learning) from our architecture; MasPA and Kumar et al. 2023 use **our dataset** (same 9 sensor features), so compare directly: reported numbers with their protocol, and our own re-run of their recipe (`leakage_replication.py`). Add kNN, SVM (RBF), gradient boosting and a MasPA-matched 9-feature set. | Same table, same folds |
| **B5** | Significance | Paired cow-bootstrap on F1/AUC differences; McNemar on cow-level predictions. Say "no significant difference" when that is true. | p-values or CIs in the table |
| **B6** | Interpretability evidence (answers the "Decision Tree is also interpretable" question) | Rule count and average conditions per rule (ANFIS vs tree); noise-robustness test (add 5/10/20% noise to sensors; compare performance drop of ANFIS vs tree vs RF); smoothness of the risk score; 3 worked case studies via `explain_row`. | Plots and a table, whichever way they come out |
| **B7** | Ablations | Indices vs raw sensors; alternative asymmetry formula (coefficient of variation); clustering with vs without the label; LSE-only vs hybrid; with vs without GA. | Ablation table |
| **B8** | Risk tier and operating point | Calibrate the Low/Watch/High tier thresholds on validation; report precision/recall at the chosen operating point. | Tier thresholds justified by data |
| **B9** | Demo | Streamlit `app.py`: enter or upload sensor readings -> score, tier, the fired rules as a bar chart. | Runs locally from README |
| **B10** | Outcome | Full paper draft (intro, related work, method, experiments, discussion, limitations); patent claim sketch (system and method claims around index-based fuzzy rule generation + explainable risk tier); one-page startup/deployment note. | Drafts in `docs/` |
| **B11** | Defense Sheet | Hardest questions + evidence files for each answer. | Printed sheet |

Work order: B1 -> B3 -> B4 -> B5 -> B2 -> B6/B7 -> B8 -> B9 -> B10 -> B11. B1 first, because the later work builds on the final model.

### Phase C: Project Report (20 marks)

| Report section | Source files / content |
|---|---|
| Abstract | Written last, from final tables |
| Introduction and problem definition | SIH26109, mastitis impact, detection vs forecasting stated clearly |
| Literature survey and gap | Verified table (Phase A) extended |
| Methodology / system design | 4-layer architecture, indices, subtractive clustering math, ANFIS layers, hybrid learning, GA |
| Implementation | Module map from `README.md`, git history, unit tests |
| Results and analysis | B3-B8 tables and figures (`reports/`) |
| Discussion, limitations, future scope | Section 3 caveats; other farms; real forecasting needs longitudinal onset data |
| Article / patent / startup documentation | B10 outputs |
| References and formatting | Every citation verified; consistent style |

---

## 5. Viva questions with answer skeletons (Review II)

1. **Why ANFIS if Random Forest is more accurate?** Accuracy is not the only goal: ANFIS gives rule-level reasons and a graded risk score. Be honest that RF scores higher (0.983 vs 0.917 AUC, Early set), and say Phase B tests what interpretability costs and what robustness it buys.
2. **A depth-4 Decision Tree is interpretable and beats you. Why ANFIS?** Admit it. Fuzzy rules give smooth boundaries and graded firing, which should handle sensor noise better; that is the B6 experiment. Do not claim it before the data.
3. **Why subtractive clustering?** Grid partitioning needs m^N rules (3 MFs and 7 inputs = 2,187); subtractive clustering needs no preset cluster count and produced 10-12 rules.
4. **Why hybrid learning?** Rule outputs are linear in their parameters, so least squares solves them exactly; gradient descent then only searches the smaller shape-parameter space.
5. **Why split by cow?** Rows of one cow are near-duplicates; random splitting leaks cows across train/test and inflates scores.
6. **Why drop `Milk_visibility`?** It equals the label on all 6,600 rows (r = 1.0); keeping it gives 100% accuracy that means nothing.
7. **Why this Asymmetry Index?** Mastitis usually affects one quarter, so the spread between quarters rises; dividing by the mean makes it scale-free. It is our design, and an ablation will test an alternative.
8. **Is this forecasting?** No. Labels are constant within 1,099 of 1,100 cows; this is detection from measurements. Forecasting needs time-stamped onset data.
9. **What does AUC 0.917 mean?** A random sick cow gets a higher risk score than a random healthy cow 91.7% of the time.
10. **Limitations?** One dataset and one test fold; indices may discard information; ANFIS currently below RF/DT; not clinically validated.

---

## 6. Context block for other AI assistants (copy and paste)

> I am working on an M.Tech Soft Computing project with a teammate: an interpretable ANFIS (first-order Sugeno, subtractive clustering, hybrid LSE + gradient descent learning, pure NumPy) for bovine mastitis *detection* from udder sensor readings (label = clinical mastitis, identical to the milk-abnormality flag) (Mendeley dataset DOI 10.17632/kbvcdw5b4m; our working CSV is not the Version 1 CSV: per A it comes from the Version 2 zip (hash check pending; .1 = 22 Oct 2020 CSV, .2 = 20 Nov 2020 zip); the two differ in 1,529 labels, all 0 -> 1, cause undocumented, so never call it synthetic; on the Version 1 CSV all models tie at row F1 0.972 / 0.958; I*/E* are very likely inhale/exhale limits; SIH26109). Data: 6,600 rows, 1,100 cows x 6 days, label class1 (about 40% mastitis). `Milk_visibility` equals the label and is dropped. The split is cow-grouped (880 train / 220 test cows). Two feature sets: Early-Warning (swelling index, asymmetry index, temperature, months postpartum, breed) and Full-Clinical (+ Hardness, Pain). Test-fold row AUC: Early-Warning LogReg 0.902, Decision Tree 0.947, Random Forest 0.983, ANFIS 0.917; Full-Clinical LogReg 0.918, Decision Tree 0.946, Random Forest 0.976, ANFIS 0.938. ANFIS is below DT and RF; do not claim otherwise. The label is constant per cow in 1,099/1,100 cows, so do not claim forecasting. Rules: never invent numbers or citations; tune on validation only; define terms in plain English; one step at a time. Repo files: data_prep.py, metrics.py, baselines.py, subtractive_clustering.py, anfis.py, explain.py, train_anfis.py, test_modules.py, audit_odd_cow.py. Review II is 2026-10-27; the Final Review needs GA tuning, grouped 5-fold CV, at least two competing algorithms, significance tests and a MasPA comparison (MasPA and Kumar et al. 2023 use the same dataset and report about 99% under a leaky recipe: oversample before splitting, random row split; `leakage_replication.py` measures the gap). ANFIS for mastitis has prior work (Mammadova & Keskin 2015; Kohli 2011), so never claim it is the first.

---

## 7. Commands cheat sheet

| Command | What it does | Output |
|---|---|---|
| `python test_modules.py` | Four correctness tests of clustering, gradient, least squares, hybrid learning | 4 x OK |
| `python data_prep.py` | Cleans data, drops leakage, splits by cow, builds indices, scales on train only | `data/processed/`, figs 1-7, summary JSON |
| `python baselines.py` | Trains LogReg, Tree, Forest on both feature sets | `reports/baseline_results.csv`, fig 8 |
| `python train_anfis.py` | Picks radius on validation, trains ANFIS, scores the test fold once | `anfis_results.csv`, `anfis_ra_grid.csv`, `comparison.csv`, rules, figs 9-10 |
| `python audit_odd_cow.py` | Lists cows whose label changes across days | printed table |

---

## 8. Open items

- [ ] Final Review date and Report deadline (ask faculty)
- [ ] Teammate's name and GitHub username
- [ ] The 5 report files in section 2
- [x] Odd cow found: cow10 (decision: keep, log in JOURNAL)
- [x] `I*`/`E*`: the dataset page lists inhale/exhale limits per quarter (write "very likely")
- [ ] Confirm by SHA256 that the working CSV equals the CSV inside the Version 2 zip (T0.4), and run `compare_versions_deep.py` (T0.5). Known: it is NOT Version 1; per A it is from Version 2
- [x] 12 papers collected and read 2026-10-26: see `docs/LITERATURE.md` (MasPA and Kumar et al. use our dataset; the external PDF mis-attributed the 99.46% result)
- [ ] Still unopened: Chiu 1994, Ebrahimi 2019, Ankitha & Manjaiah (KNN vs SVM), Mammadova & Keskin 2015 and Kohli 2011 (prior neuro-fuzzy mastitis work: read before writing the novelty claim)
- [ ] Run `leakage_replication.py` (T1.8) and bring the tables
- [ ] Companion paper: only an unverified lead (section 2b); open and verify it, or cite the dataset record itself
- [ ] Label rule (how class1 was assigned) stays undocumented: state it as a limitation
- [ ] Replace "early" / "subclinical" wording in slides, title and abstract
- [ ] Reproducibility check on teammate's machine

---

## Changelog
- **2026-10-26 (v5):** 12 papers read (`docs/LITERATURE.md`). Corrected: MasPA uses our dataset, not AMS data. Found: MasPA and Kumar both report 3,961 / 2,639 (matches the working file; supports D13) and both oversample before splitting and split by row. Added `leakage_replication.py` (T1.8). ANFIS for mastitis has prior work (Mammadova & Keskin 2015; Kohli 2011): novelty wording narrowed.
- **2026-10-25 (v4):** provenance resolved per A's account (working CSV from the Version 2 zip; hash check pending); derived the 1,529-row, all-0-to-1 difference; decision D13 (Version 2 main, Version 1 sensitivity); new section 2c (review of the external PDF); T0.4 reworded; T0.5 `compare_versions_deep.py` added; context block updated.
- **2026-10-25 (v3):** added dataset Versions 1 and 2 (DOIs, dates, files); I*/E* upgraded from "likely" to "very likely inhale/exhale limits"; added T0.4 dataset provenance task; companion paper recorded as an *unverified lead*; "early" removed from the title and context block; MasPA comparison made qualitative; open items updated.
- 2026-10-25 (v2): added section 2b (dataset facts from the source page) and the cow10 decision.
- 2026-10-24 (v1): first version (real results, honest reading, roadmap, viva questions).
- **2026-10-25 (v4):** Version 1 CSV compared with the working file: they differ (labels, Hardness, Pain, Temperature); results on Version 1 recorded (all models tie); provenance of the working file marked as the BLOCKER; section 2 results flagged as "provenance unverified"; context block updated.
