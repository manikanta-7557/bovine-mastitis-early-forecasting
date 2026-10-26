# WORKFLOW: from 2026-10-25 to the end of the project
**Interpretable ANFIS for bovine mastitis detection** | Owners: A = Manikanta, B = teammate | Companion file: `PROJECT_STATE_AND_ROADMAP.md` (rules, plain-English glossary, real results)

---

## 0. How to use this file (humans and AI models)

- Read order: `PROJECT_STATE_AND_ROADMAP.md` section 0 (rules) -> this file -> roadmap section 2 (real results).
- Every task has an ID (T3.2 ...). Use the ID in commit messages and `JOURNAL.md`.
- **Definition of Done (applies to every task):**
  1. Runs from a clean clone (`git clone`, `pip install -r requirements.txt`, one command).
  2. `python test_modules.py` still passes.
  3. Outputs land in `reports/` and a line is added to `reports/RUNLOG.md` (date, script, git commit, config).
  4. `JOURNAL.md` has an entry.
  5. The claims register (section 3) is updated.
  6. The other person can explain it in 5 minutes.
- **Non-negotiables:** no invented numbers or citations; tune on validation only; never use `Milk_visibility` as input; no "forecasting", "early", "subclinical" or "ANFIS beats RF/DT" claims unless the claims register says allowed.

---

## 1. Official rubric (from `0 Evaluation Criteria.docx`) and which tasks earn each mark

**Review II (30), Tuesday 2026-10-27**

| Criterion | Marks | Tasks |
|---|---:|---|
| Literature survey and refined gap | 3 | T1.1 |
| Methodology / system design | 4 | T1.2 |
| Implementation progress (aim 50-70% done, with preliminary results) | 8 | Modules 1-5 done; T0.2 |
| Dataset / data preparation | 3 | T0.3, T0.4, T1.4 |
| Preliminary results and analysis | 4 | T1.4 |
| Article / patent / startup development (article: title, abstract, methodology, preliminary results, target venue) | 4 | T1.3 |
| Presentation and viva | 2 | T1.4, T1.6 |
| Soft Computing justification | 2 | T1.5 |

**Final Review (40)**

| Criterion | Marks | Tasks |
|---|---:|---|
| Problem formulation and achievement of objectives | 3 | T8.1 |
| Methodology and technical correctness | 4 | T3.x, T4.1 |
| Complete implementation | 8 | T3.x, T5.x, T7.1, T8.2 |
| Experimental evaluation | 7 | T4.x, T5.x, T6.5 |
| Results, comparison and discussion | 5 | T4.2, T4.3, T8.3 |
| Application / novelty and effective use of Soft Computing | 5 | T6.1-T6.4 |
| Presentation and demonstration | 3 | T7.1, T8.4 |
| Viva-voce | 5 | T8.5 |

**Outcome achievement scale (separate table in the rubric):** idea only 1 | draft 2 | **submission-ready 3** | **submitted article/patent, or well-developed startup proposal 4**. Ask the faculty whether this is scored inside the 40 or separately. Aim for at least 3, and 4 if the article is actually submitted (T7.2).

**Report (20)**

| Section | Marks | Task |
|---|---:|---|
| Abstract, introduction, problem definition | 2 | T9.1 |
| Literature survey and gap | 3 | T9.1 |
| Methodology / system design | 3 | T9.1 |
| Implementation | 2 | T9.1 |
| Results and analysis | 4 | T9.1 |
| Discussion, limitations, future scope | 2 | T9.1 |
| Article / patent / startup documentation | 2 | T9.1 |
| References and citation quality | 1 | T9.2 |
| Organisation, formatting, presentation | 1 | T9.2 |

---

## 2. Stages and gates

Final Review and Report dates are **not known yet** (put them in section 10). Schedule backwards from them; if time is short use the cut order in section 9.

| Stage | When | Goal | Exit gate (all must be true) |
|---|---|---|---|
| 0 Sync | Sun 25 Oct | Both partners run the same pipeline | Identical `comparison.csv` and identical dataset SHA256 on both machines |
| 1 Review II package | Sun 25 - Mon 26 | Slides, diagrams, article skeleton, viva prep | Section 4 tasks T1.1-T1.7 done; repo tagged `review2` |
| 2 Review II | Tue 27 | Present honestly | - |
| 3 Freeze the model | Week 1 after Review II | Diagnose and fix the accuracy gap; lock the config | `config.json` committed; ANFIS config frozen BEFORE touching the test fold |
| 4 Rigorous evaluation | Weeks 1-2 | Nested grouped CV, more algorithms, significance tests | `oof_predictions.csv`, `cv_results.csv`, significance table |
| 5 GA and ablations | Week 2 | GA tuning vs grid vs random search; ablations | Convergence plot, comparison table |
| 6 Interpretability and robustness | Weeks 2-3 | Evidence for what fuzzy rules add | Rule metrics, noise-robustness plot, case studies, tier calibration |
| 7 Demo and outcome | Week 3 | Streamlit demo; article submission-ready | Demo runs from clean clone; article draft complete |
| 8 Final Review prep | Weeks 3-4 | Slides, defense sheet, rehearsal | Dry run with every number traced |
| 9 Report | Weeks 4+ | Phase IV report | Section checklist ticked |

---

## 3. Cross-cutting standards

**3.1 Single config.** Create `config.json` (task T3.0) holding: seeds, `TEST_FOLD`, `RA_GRID`, `MAX_RULES`, learning rate, epochs, feature-set lists, thresholds. Scripts read it; nothing is hard-coded twice.

**3.2 Provenance.** `reports/RUNLOG.md` gets one line per run: `date | script | git commit | config hash | output files`. Results files are regenerated by scripts, never edited by hand.

**3.3 Naming.** Say "sensor-only" and "sensor + clinical signs" in prose. The code names `Early-Warning` / `Full-Clinical` may stay, but slides and paper must not imply "early" or "subclinical" (the label is clinical mastitis).

**3.4 Git.** `main` always runs. Branches `feat/<task-id>-<name>`. Tags: `review2`, `final`, `report`. Merge by PR reviewed by the other partner.

**3.5 Claims register** (keep in `docs/CLAIMS.md`; a claim goes into any slide, paper or answer only if allowed):

| ID | Claim | Status | Evidence / what is needed |
|---|---|---|---|
| C1 | Sensor-only features detect clinical mastitis on unseen cows (ANFIS AUC 0.917, RF 0.983, single split) | Allowed, update after CV | `reports/comparison.csv` |
| C2 | ANFIS is more accurate than RF or Decision Tree | **Forbidden** | Currently false |
| C3 | Hardness and Pain add nothing for RF | Allowed with caveat (one test fold) | Re-test in CV (T4.1) |
| C4 | The model forecasts future onset | **Forbidden** | Labels constant per cow |
| C5 | "Early" or "subclinical" detection | **Forbidden** | Label is clinical mastitis |
| C6 | The dataset is peer-reviewed | **Forbidden** | Until the companion paper is found and cited |
| C7 | ANFIS gives rule-level explanations that RF cannot | Allowed | `rules_*.txt`, `explain_row` |
| C8 | ANFIS is more interpretable than a depth-4 tree | Unproven | T6.1 |
| C9 | ANFIS is more robust to sensor noise than the tree | Unproven | T6.2 (may turn out false) |
| C10 | Results generalise to other farms or breeds | **Forbidden** | One dataset |
| C11 | Our working CSV is the Version 2 file | **Per A, from the Version 2 zip; SHA256 check pending**; known NOT Version 1 | T0.4 |
| C16 | MasPA uses a different dataset (AMS data) | **Forbidden (was wrong)** | Its Methods use our 6,600-row dataset with 3,961 / 2,639 class counts |
| C17 | "First ANFIS / neuro-fuzzy model for mastitis" | **Forbidden** | Mammadova & Keskin 2015 and Kohli 2011 exist (titles seen in Kumar's references; to be opened). Use "to our knowledge, not on udder-sensor IoT data under cow-level validation" |
| C18 | The literature's 99% accuracy is wrong / the authors cheated | **Forbidden** | Say: their test sets overlap their training sets (duplicated rows, shared cows), so the number does not measure detection in an unseen cow. Quantify with T1.8 |
| C19 | We reproduced MasPA | **Only the recipe** | Same features and recipe, our own run (T1.8); their exact numbers depend on their random split |
| C13 | Conclusions drawn on the working file transfer to the Version 1 CSV | **Forbidden** | On Version 1 all models tie (row F1 0.972 Early-Warning, 0.958 Full-Clinical) and the label nearly equals Hardness, Pain and Milk_visibility |
| C12 | I* = inhale limit, E* = exhale limit | Allowed as "very likely" | Dataset page lists inhale/exhale limits per quarter; columns not named |
| C14 | Version 2 (or the working file) is "synthetic", "fabricated" or "manipulated" | **Forbidden** until the authors or a paper say so | Differences are measured (1,529 labels, all 0 -> 1, sensor values changed) but the cause is undocumented |
| C15 | EU AI Act requires interpretable models for our cattle tool | **Forbidden** unless a legal source on veterinary systems is opened | The Act targets AI systems affecting people; the PDF's sources were weak |

**3.6 Shared preprocessing.** To avoid training/serving mismatch, factor the index computation and train-only scaling from `data_prep.py` into `preprocess.py` (T3.1). `data_prep.py` was modified on disk after delivery: read the current file first and keep behaviour identical (re-run and diff `data_prep_summary.json`).

---

## 4. Task cards: Stages 0-2 (Review II)

### T0.1 Repo and collaboration  (A, 20 min)
- **How:** push the zip contents; add B; B clones and installs.
- **Accept:** B runs `python test_modules.py` successfully on his machine.

### T0.2 Reproducibility check  (A+B, 30 min)
- **How:** both run `data_prep.py`, `baselines.py`, `train_anfis.py`; compare `reports/comparison.csv` (`diff` or open side by side).
- **Accept:** identical, or differences < 1e-4 with a written cause (library versions: record `pip freeze` in the journal).

### T0.3 Evidence capture  (A, 10 min)
- **How:** send Claude `anfis_ra_grid.csv`, `anfis_results.csv`, `rules_*.txt`, `example_explanations.txt`.
- **Accept:** the numbers in slides come from these files.

### T0.4 Dataset provenance  (B, 30 min; marks: data 3)
- **Why:** the Mendeley page has Version 1 (`.1`, 22 Oct 2020, `clinical_mastitis_cows_version1.csv`, 413 KB) and Version 2 (`.2`, 20 Nov 2020, a 1.48 GB zip). The paper and report must cite the version actually used.
- **How:** download the small Version 1 CSV; run `python verify_dataset_version.py data/raw/clinical_mastitis_cows.csv <downloaded.csv>`; run the one-argument form on both partners' machines. **Result so far (2026-10-25): our working file is NOT identical to the Version 1 CSV** (different labels and clinical columns; see roadmap section 2b). Next: record where the working file came from; if the Version 2 zip (1.48 GB, contents not described on the page) is available, extract its CSV and compare with the same command.
- **Accept:** `JOURNAL.md` records the source of the working file, its SHA256 and which official Mendeley file (if any) it matches; both partners' SHA256 are identical; claim C11 is updated.
- **Update 2026-10-25 (later):** per A, the working CSV comes from the Version 2 zip. Remaining check: extract the CSV from the zip (`unzip Clinical_Mastitis_cows_version2.zip "*.csv" -d v2/`) and run `python verify_dataset_version.py data/raw/clinical_mastitis_cows.csv v2/<csv>`; expect IDENTICAL. If it is not identical, the source is again unknown: stop and tell Claude.

### T0.5 How exactly do Version 1 and Version 2 differ?  (A, 10 min; marks: data 3, viva 5)
- **Why:** `verify_dataset_version.py` says *that* the files differ; the faculty will ask *how*. Known from A's output: 1,529 labels differ, all 0 -> 1 (= 2,639 - 1,110); sensor columns differ in about 1,500 cells each.
- **How:** `python compare_versions_deep.py data/raw/<version1>.csv data/raw/clinical_mastitis_cows.csv` (file 1 = Version 1, file 2 = working file). Paste the full output to Claude. It prints label transitions, whole vs partly relabelled cows, per-column changes, mean values of relabelled rows, and the Version 1 vs Version 2 correlation inside relabelled rows.
- **Accept:** output saved in `reports/version_diff.txt` and a 3-line interpretation in `JOURNAL.md`. Describe only; do not guess the cause (C14).

### T1.1 Literature table and gap  (A, 3 h; marks: lit 3)
- **How:** one row per paper you have actually opened: reference, year, task, input data, method, black-box or interpretable, key result, limitation. Start from the citations verified in the Phase I proposal; add the MasPA paper (Ghafoor and Sitkowska 2021, *AgriEngineering* 3(3):575-583). **All 12 papers A collected are already tabulated in `docs/LITERATURE.md`: start there** and the dataset record cited with the DOI of the version we used (T0.4). A possible companion paper (2021, KNN vs SVM for clinical mastitis, by two of the dataset authors) is only an unverified lead seen in a 2025 review: open it before citing, or cite the dataset record alone.
- **Accept:** 6-8 rows, every DOI/URL opened by a human; a gap statement of at most 3 sentences that is true *for the reviewed set* (e.g. "none of the reviewed studies provide rule-level explanations from udder sensor readings"), not "nobody has ever".
- **Pitfall (corrected 2026-10-26):** MasPA's title says "AMS sensor data" but its Methods use **our dataset** (3,961 healthy / 2,639 mastitis). It is a direct comparator. Ghafoor and Sitkowska and Kumar et al. both oversample before splitting and split by row; see `docs/LITERATURE.md` and T1.8.

### T1.8 Leakage replication: why do papers report 99%?  (A runs, 10 min; marks: results 4, viva 5)
- **Why:** MasPA (2021) and Kumar (2023) report about 99% on this dataset. The faculty will ask why ours is lower. Their recipe oversamples *before* splitting and splits by *row*, so test rows are copies of, or neighbours of, training rows.
- **How:** `python leakage_replication.py data/raw/clinical_mastitis_cows.csv --seeds 5 --out reports/leakage_replication.csv`. It runs RF, DT, kNN and LogReg on the 9 features under four protocols (P1 paper-style, P2 random row split, P3 cow-grouped = ours, P4 cow-grouped + oversample train only) and prints accuracy, F1, AUC and two leakage diagnostics (share of test rows that are exact copies of training rows; share from cows seen in training).
- **Accept:** `reports/leakage_replication.csv` exists; paste the printed tables to Claude; the slide says "same data, same 9 features, same models: accuracy X under the published recipe, Y under cow-grouped" with both numbers read from the CSV. Never quote synthetic-test numbers.

### T1.2 Architecture and data-flow diagrams  (Claude drafts, B checks, 1 h; marks: design 4)
- **How:** Diagram 1 = the 4 layers (data and features; subtractive clustering; Sugeno ANFIS with hybrid learning; explainable output). Diagram 2 = data flow with file names: raw CSV -> `data_prep.py` -> `train/test.csv` -> `baselines.py` / `train_anfis.py` -> `reports/`.
- **Accept:** every box is a real file or function; names match the code; each arrow is a real data hand-off.

### T1.3 Article skeleton  (A with Claude, 2 h; marks: article 4)
- **How:** title, 150-200 word abstract with real numbers, methodology outline, preliminary-results table, 2 candidate venues. Working title: *Interpretable Neuro-Fuzzy Detection of Bovine Mastitis from Udder Sensor Indices*.
- **Accept:** every abstract number traces to a file; no "early", "forecast", "outperforms"; venues have a link to author guidelines, scope, indexing and fee checked by a human.

### T1.4 Review II slides + speaking notes  (Claude drafts, A+B finalise, 3 h; marks: results 4, data 3, presentation 2)
- **How:** the 11-slide plan in the roadmap (section 4).
- **Accept:** at most 12 slides; one honest results slide (ANFIS below DT and RF, stated plainly); each number has a source file in the speaking notes; a timed run-through fits the slot.

### T1.5 Soft Computing justification  (A, 1 h; marks: 2)
- **How:** one slide or page: why neuro-fuzzy here. Include the grid-partition explosion (3 MFs and 7 inputs = 3^7 = 2,187 rules) versus the 10-12 rules obtained; the interpretability argument; the honest counter-argument (a depth-4 tree is also readable) and how Final Review tests it.
- **Accept:** the math is shown, not just stated; the counter-argument appears.

### T1.6 Viva preparation  (A+B, 1.5 h; marks: viva 2)
- **How:** use the 10 questions in the roadmap section 5; each person answers 5 aloud, 60 seconds each, naming the evidence file.
- **Accept:** no answer contains a forbidden claim from the register.

### T1.7 Freeze and tag  (A, 10 min)
- **Accept:** `git tag review2`; a clean clone passes `python test_modules.py`.

---

## 5. Task cards: Stages 3-9 (Final Review and report)

### Stage 3: freeze the model

**T3.0 Config file** (A+B, 1 h). Create `config.json`; make `data_prep.py`, `train_anfis.py`, `baselines.py` read it. **Accept:** results identical to before the refactor.

**T3.1 Shared preprocessing** (B, 2 h). Create `preprocess.py` with `compute_indices(df_raw)` and `scale(df, scaler_params)`; `data_prep.py` and the demo call it. **Accept:** `data_prep_summary.json` unchanged; a unit test shows 5 random rows give identical features through both paths.

**T3.2 Diagnostics D1** (B writes `diagnostics.py`, A reviews, 4 h; marks: methodology 4, experimental 7).
- **Question:** why is ANFIS below RF/DT, and is it fixable?
- **Protocol:** use only the training cows. Within them make grouped inner folds (`StratifiedGroupKFold`, 4 folds). Never read the test fold.
- **Experiments:**
  - E1 information loss: RF on (a) the 5 current features, (b) raw 8 sensors + temperature + months + breed, (c) current + extra features.
  - E2 rule budget: ANFIS with `MAX_RULES` in {8, 12, 16, 20, 30} and `ra` in {0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}.
  - E3 extra features (keep at most 9 inputs): max, min, std of quadrant totals; per-quadrant expansion E minus I (the page lists inhale and exhale limits per quarter); mean expansion.
  - E4 optimiser: learning rate {0.003, 0.01, 0.03}, patience, ridge {1e-4, 1e-3, 1e-2}.
  - E5 threshold: choose on validation to reach a recall target.
- **Adoption rule:** adopt a change only if mean inner-fold AUC improves by more than one standard error AND rules stay at or below 20.
- **Accept:** `reports/diagnostics_*.csv`; every adopted change logged in `JOURNAL.md` with its evidence; if nothing helps, record that honestly (a valid result).

**T3.3 Freeze** (A+B, 30 min). Write the chosen config to `config.json`; commit; tag `model-frozen`. **Accept:** from here the test fold is scored only through scripts, once per config.

### Stage 4: rigorous evaluation (marks: experimental 7, results 5)

**T4.1 Nested grouped CV** (B, 6 h).
- **How:** `cv_eval.py`. Outer loop: for fold k in 0..4 (the existing `fold` column), test = fold k, train = the rest. Inner loop: choose `ra` (and any frozen-grid options) with `StratifiedGroupKFold(3)` on the training cows only. Fit on all training cows, score fold k once. Same folds for every model and both feature sets. Save pooled out-of-fold predictions to `reports/oof_predictions.csv` with columns `cow_id, day, fold, feature_set, model, y_true, score`.
- **Output:** `reports/cv_results.csv` (per fold) and a summary table (mean +/- std across folds; 95% cow-bootstrap CI on pooled predictions).
- **Accept:** an `assert` shows zero cow overlap in every fold; seeds fixed; rerun gives identical numbers; fold sizes about 220 cows each.

**T4.2 More competing algorithms** (B, 4 h). Required by the rubric: at least two beyond ANFIS (LogReg, Decision Tree and Random Forest already count). Add SVM (RBF), MLP, Gradient Boosting through the same CV code; add "plain Sugeno, no learning" (subtractive-clustering rules with LSE only, no gradient step) as the ablation of hybrid learning.
- **MasPA and Kumar et al.:** same dataset and the same 9 features (8 sensors + Temperature). Compare in two ways: (a) *reported* numbers, quoted with their protocol; (b) *our re-run* of their recipe vs ours with `leakage_replication.py` (T1.8). Add a MasPA-matched feature set (raw 9 features) to our table. Their protocol is not ours, so never put their 99% in the same column as our cow-grouped numbers without saying so.
- **Accept:** all models in one table on the same folds, hyperparameters chosen only on inner folds.

**T4.3 Significance tests** (B, 3 h). From `oof_predictions.csv`: (a) paired cow-bootstrap (2,000 resamples of cows; metric difference ANFIS minus model; 95% CI); (b) McNemar (exact binomial) on cow-level predictions at the chosen thresholds.
- **Accept:** a pairwise table; wording rule: if the CI contains 0, write "no significant difference", never "better".

### Stage 5: GA and ablations (marks: experimental 7, implementation 8)

**T5.1 Genetic Algorithm** (B writes `ga_tune.py`, A explains it, 6 h).
- **Chromosome:** `[ra in 0.15..0.8, rb_factor in 1.2..2.0, max_rules in 8..30 (integer), log10(lr) in -2.5..-1.3]`.
- **Fitness:** mean inner 3-fold AUC minus 0.01 x (rules / 30).
- **Settings:** population 20, 15 generations, elitism 2, tournament size 3, blend crossover (alpha 0.5, p 0.8), Gaussian mutation (p 0.2, sigma 10% of range), fixed seeds, cache repeated chromosomes.
- **Controls:** (i) grid search from `train_anfis.py`; (ii) random search with the same number of evaluations. This shows the GA beats chance, not just the grid.
- **Compute:** one fit took roughly 3 s in your run (42.7 s for about 14 fits); measure first. About 300 evaluations x ~9 s is ~45 min per feature set. Nested GA over 5 outer folds is hours: run it overnight, or use a smaller budget (population 12, 8 generations) for the nested run and the full budget on the fixed split.
- **Accept:** convergence plot (best and mean fitness per generation); table GA vs grid vs random search on validation and on the test fold; leakage check: the GA never sees the outer test fold.

**T5.2 Ablations** (B, 4 h). Change one thing at a time on the same folds: indices vs raw sensors; asymmetry index vs coefficient-of-variation version; clustering with vs without the label; LSE-only vs hybrid; with vs without GA; with vs without Hardness and Pain. **Accept:** one table, each row differs from the baseline in a single factor.

### Stage 6: interpretability and robustness (marks: novelty/effective use of SC 5)

**T6.1 Rule and tree complexity** (A, 2 h). Report number of rules, total conditions, rules covering at least 5% of test rows, versus tree leaves and node count at comparable accuracy (include a deeper tree as well). **Accept:** table plus one honest sentence on what it shows.

**T6.2 Noise robustness** (B writes `robustness.py`, 4 h). Add Gaussian noise (0, 2, 5, 10, 20% of training std) to the raw sensor columns of test rows, recompute indices with the train scaler (`preprocess.py`), repeat 20 times, record AUC and F1 for ANFIS, DT, RF, LR. **Hypothesis stated in advance:** ANFIS degrades more slowly than the tree. **Accept:** plot with error bars; the conclusion is written whichever way it comes out; update claim C9.

**T6.3 Case studies** (A, 2 h). Three cows with `explain_row`: a true positive, a true negative, a false negative. **Accept:** each has the fired rules in words and what a vet would check.

**T6.4 Operating point and risk tiers** (A, 2 h). Pick the threshold on validation (for example the highest precision with recall at least 0.90) and calibrate the Low/Watch/High tier edges. **Accept:** table of precision and recall at that threshold on the test fold; the observed mastitis rate rises across Low < Watch < High on held-out cows.

### Stage 7: demo and outcome

**T7.1 Streamlit demo** (B, 5 h; marks: demonstration 3, implementation 8). `app.py`: inputs for 8 sensor values, temperature, months, breed (and optional Hardness/Pain); preprocessing through `preprocess.py`; outputs score, tier, top-3 fired rules as a bar chart; banner "research prototype, not a diagnostic device". **Accept:** `streamlit run app.py` works from a clean clone; five test rows give exactly the same score as `predict_score`; out-of-range input shows a warning. Record a screen video as a fallback for the live demo.

**T7.2 Article** (A leads, 10 h; outcome 3-4). Sections: introduction, related work, data (with the label limitation), method, experiments, results, discussion, limitations, conclusion. `make_tables.py` generates every table from `reports/*.csv`. **Accept:** template of the chosen venue applied; page limit met; every table cell reproducible; similarity check run; authorship and affiliation confirmed with the faculty. To reach "submitted" (4 marks), submit and keep the confirmation email.

**T7.3 Secondary outcomes (only if time).**
- *Patent:* novel idea, inventive aspect, prior-art search, claims. Search Google Patents, Espacenet, WIPO PATENTSCOPE and the Indian Patent Office database; record at least 8 hits in a table; ANFIS and subtractive clustering are old, so any novelty must be in the specific combination (index-based aggregation + rule generation + explainable risk tier). Label as "not legal advice; check with the institution's IPR cell".
- *Startup:* problem, target users, product, USP, market. Every market number cited or marked as an assumption.

### Stage 8: Final Review preparation

**T8.1 Objectives table** (A, 1 h; marks: problem and objectives 3). Each objective: statement, evidence file, status (met / partly / not met). **Accept:** "not met" is allowed and explained.
**T8.2 Implementation completeness** (B, 2 h; marks: 8). README, tests, config, a clean-clone run, module map. **Accept:** a stranger reproduces everything in under 30 minutes.
**T8.3 Results discussion** (A+B, 3 h; marks: 5). Critical discussion against at least two other algorithms using the T4.3 table; say what ANFIS costs in accuracy and what it buys in explanations. **Accept:** no forbidden claim.
**T8.4 Final slides and demo script** (A+B, 4 h; marks: 3). 12-15 slides; a 5-minute live demo plus the recorded backup.
**T8.5 Defense sheet** (A+B, 4 h; marks: viva 5). 40 questions grouped by data, method, math, results, ethics, limitations; a 60-second answer and an evidence file for each. **Accept:** each partner answers 20 aloud without notes; the other grades them.

### Stage 9: project report (20 marks)

**T9.1 Write** (A+B, 15-20 h). Write each section from the finished tables, not before. Page budgets (adjust to the university template): abstract 1; introduction 3; literature 5; methodology 8; implementation 4; results 10; discussion and limitations 3; article/patent/startup documentation 3; references 2. **Accept:** every figure regenerates from `reports/figures`; every table traces to a CSV; limitations include clinical label, undocumented label rule, one dataset, no forecasting, and ANFIS versus RF/DT.

**T9.2 References and formatting** (A, 3 h). Each reference opened and verified, with DOI; consistent style; numbered figures and tables with captions; similarity check; proofread aloud. **Accept:** no unverifiable citation.

---

## 6. Decision log

| ID | Decision | Reason |
|---|---|---|
| D1 | Drop `Milk_visibility` | Identical to label |
| D2 | Cow-grouped stratified split | Rows of a cow are near-duplicates |
| D3 | Two composite indices | Tractability and interpretability; formula is our design |
| D4 | Keep cow10 | 6 of 6,600 rows; cause undocumented |
| D5 | Exclude `Previous_Mastits_status` | About 0.4% positive; near-zero variance |
| D6 | Frame as detection of clinical mastitis | Label constant per cow; dataset says clinical |
| D7 | Report ANFIS honestly | It trails DT and RF |
| D8 | Cluster on [inputs, label] of training data | Class-specific rules; label never used at prediction |
| D9 | Label rule undocumented | Stated as a limitation |
| D10 | Cite the dataset version actually used; verify with `verify_dataset_version.py` | Versions `.1` and `.2` exist with identical descriptions |
| D11 | Treat I*/E* as inhale/exhale limits, saying "very likely" | Page lists inhale/exhale limits for 4 quarters; letters match; columns not named |
| D12 | Choose the dataset by provenance, never by which file flatters ANFIS | Research integrity; the faculty will ask where the data came from |
| D15 | Corroboration for D13: MasPA (2021) and Kumar (2023) report 3,961 / 2,639 (about 40% mastitis), which matches the working file, not the Version 1 CSV (16.8%) | Independent published use of the working file's class balance. It does not explain why Version 1 differs |
| D13 | Main dataset = Version 2 (the working file); Version 1 = sensitivity table; disclose the 1,529 differing labels | Version 2 is the authors' latest release and the file A downloaded; on Version 1 all models tie, so it cannot separate methods. If the SHA256 check fails, revisit |
| D14 | Significance by paired cow-bootstrap CIs on pooled out-of-fold predictions plus McNemar, not Wilcoxon on 5 fold scores | With n = 5 pairs the smallest possible two-sided Wilcoxon p is 0.0625, so p < 0.05 is unreachable |

---

## 7. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| ANFIS stays below RF/DT | High | T3.2 fixes; T6.x interpretability evidence; honest framing (C2 forbidden) |
| Exams squeeze time | High | Section 9 cut order; B owns code tasks |
| Nested GA too slow | Medium | Smaller budget, overnight run, fixed-split comparison |
| Teammate unavailable | Medium | README, HANDOFF, JOURNAL; both can run everything |
| Label ambiguity | Medium | Limitation in report; find companion paper |
| Faculty: "papers report 99%, you report 0.87-0.93; why?" | High | `leakage_replication.py` (T1.8) turns the answer into a measured table; Bobbo 2023 as precedent for grouped splits |
| Overclaiming in viva | Medium | Claims register; drill T1.6 |

---

## 8. Operating protocol for AI assistants

1. Read the roadmap section 0 and this file before answering. Ask for the missing files; never guess numbers.
2. Define each term in plain English before using it. When asked for one step at a time, give one step and wait.
3. Write code with tests; run them; say what was and was not run (synthetic data only proves the code runs).
4. Cite only sources you opened; mark unverified items "to verify".
5. Update `JOURNAL.md` text and the claims register in your reply.
6. End each session with the handoff block below.

**Session handoff block (paste at the end of any AI session):**
```
DATE:
DONE (task IDs):
FILES CHANGED:
NUMBERS PRODUCED (file -> value):
CLAIMS REGISTER CHANGES:
OPEN QUESTIONS:
NEXT TASK ID:
```

**Prompt templates:**
- *Build a module:* "Task T<id> in WORKFLOW.md. Read the current files first. Write the code, tests, and a Defense Guide (why, alternatives, why rejected). Do not run on real data; tell me the command and the expected output file."
- *Check numbers:* "Here are the slides and `reports/*.csv`. List every number on the slides, the file it comes from, and any number with no source."
- *Viva drill:* "Ask me one question at a time from the defense sheet; grade my answer against the claims register."

---

## 9. Cut order if time collapses (cut from the top; never cut the bottom)

1. Patent and startup documents (T7.3)
2. SVM/MLP/Gradient Boosting (keep LogReg, DT, RF: already two or more)
3. Ablations beyond indices-vs-raw and LSE-only-vs-hybrid
4. GA nested run (keep GA vs grid vs random search on the fixed split)
5. Noise robustness (but then do not make claim C9)
- **Never cut:** nested grouped CV (T4.1), significance wording (T4.3), honest results discussion (T8.3), the Streamlit demo (3 marks), the defense sheet and drills (5 marks), the report's results section.

---

## 10. Open questions to fill in

- [ ] Final Review date: ____ ; Report deadline: ____
- [ ] Is "outcome achievement" scored inside the 40 or separately?
- [ ] Faculty's report template and page limit
- [ ] Who is a permitted co-author on the article (guide, affiliation)?
- [ ] Teammate's name and GitHub username
- [ ] SHA256 of the working CSV equals the CSV in the Version 2 zip? (T0.4); output of `compare_versions_deep.py` (T0.5)
- [ ] Companion paper: unverified lead only; opened and verified? Does it define the label rule?

---

## Changelog
- **2026-10-25 (v2):** added T0.4 (dataset provenance); claims C11 (version unverified) and C12 (I*/E* very likely inhale/exhale); decisions D10, D11; T1.1 now cites the dataset by version DOI; companion-paper lead marked unverified; Stage 0 gate now includes the dataset SHA256.
- 2026-10-25 (v1): first full workflow (stages 0-9, task cards, claims register, risk register, AI protocol).
- **2026-10-25 (v3):** working CSV found to differ from the official Version 1 CSV; T0.4 is now a BLOCKER; claims C11 updated and C13 added; decision D12 added.
- **2026-10-25 (v4):** provenance resolved per A's account (Version 2 zip, hash check pending); T0.5 added; claims C11 updated, C14 and C15 added; decisions D13 and D14 added; open questions updated.
- **2026-10-26 (v5):** 12 papers read (`docs/LITERATURE.md`); MasPA and Kumar found to use our dataset (claim C16 corrects an earlier error); claims C17-C19; decision D15; task T1.8 and `leakage_replication.py` added; T1.1 and T4.2 rewritten.
