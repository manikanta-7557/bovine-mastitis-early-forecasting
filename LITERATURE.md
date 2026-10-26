# Literature review: what the 12 papers actually say (checked 2026-10-26)

Owners: A = Manikanta, B = teammate. Written for humans and for other AI assistants. **Every fact below was read from the PDF A attached** (page text extracted and read). Anything not read is marked *unverified*. Do not add a number to a slide unless it is in this file or in `reports/`.

## 1. The three findings that change the project

1. **MasPA (2021) and Kumar et al. (2023) use our dataset, so they are direct comparators, not "a different data type".** MasPA states the raw data are 6,600 rows with **3,961 healthy (60.02%) and 2,639 mastitis (39.98%)**, exactly our working file's class counts (the Version 1 CSV has 5,490 / 1,110). Kumar et al. report 3,971 healthy of 6,600 (10 rows off, probably a typo). **We previously wrote that MasPA uses AMS data; that was wrong** (its title says AMS, its Methods use this IoT dataset). Corrected in claims C16.
2. **Their ~99% accuracy comes from a recipe that lets the test set see the training set.** Both papers (a) oversample the minority class *before* splitting, so copies of the same row land in train and test, and (b) split randomly by row although each cow has 6 daily rows. Neither groups by cow. `leakage_replication.py` measures the gap on our data (run it; do not quote a number before it is run).
3. **ANFIS has been tried on mastitis before.** Kumar's reference list contains Mammadova & Keskin (2015), "Application of neural network and adaptive neuro-fuzzy inference system to predict subclinical mastitis in dairy cattle" (*Indian J. Animal Research* 49(5):671-679) and Kohli (2011), a Lincoln University PhD thesis on neuro-fuzzy detection in automatic milking systems. We have seen only the titles, second-hand. **"First ANFIS for mastitis" is forbidden (claim C17).** Our novelty is narrower, see section 4.

The external AI report attached on 2026-10-25 credited the "99.46% KNN" result to "Ankitha et al. (2020)". **That is wrong**: Ankitha, Manjaiah and Kartik (2020) is the *dataset*; the 99.46% paper is Kumar, Sindhwani and Sachdeva (2023).

## 2. Literature table (12 papers read)

| # | Paper | Data and label | Method | Validation | Key result (as printed) | Use in our project | Cautions |
|---|---|---|---|---|---|---|---|
| 1 | **Ghafoor & Sitkowska (2021)**, MasPA, *AgriEngineering* 3(3):575-583, doi 10.3390/agriengineering3030037 | **Our dataset**; 8 sensor limits + Temperature (ID, breed, Hardness, Pain, milk photos dropped); mastitis 39.98% | 26 scikit-learn models via lazypredict; best = Random Forest (entropy, 100 trees); Streamlit app | RandomOverSampler on all data (3,961 + 3,961 = 7,922), **then** random 80/20 split (6,337 / 1,585); grid search; 10-fold CV | RF 99.12% test accuracy; 10-fold mean 98.10% (sd 0.043); sensitivity 99.36%, specificity 98.77%. Other test accuracies: XGB 98.68, DT 97.85, KNN 93.56, SVC 87.89, **LogReg 86.06**, NB 69.34 | Same-data comparator; our sensor-only feature set mirrors theirs; their LogReg 86% is close to our cow-grouped LogReg (row F1 0.859) | Oversample-before-split and row split (see section 1); says "three entries per cattle" (our file: six days x 1,100 cows); defines true positives as *healthy* cows; claims prediction "before it occurred" without a forecasting design |
| 2 | **Kumar, Sindhwani & Sachdeva (2023)**, "Detection of Mastitis Disease in Cow with Machine Learning Classifiers", doi 10.52865/DQFZ6016, pp. 112-129 (journal name not printed on the PDF: read it from the DOI page) | Our dataset, 8 sensors + Temperature; cites the Mendeley record as V1 (doi .1) yet reports 3,971 healthy / rest mastitis | RF, DT, SVM, kNN, NB | RandomOverSampler, 80/20 random split, grid search, 10-fold CV | Abstract: kNN 99.46% accuracy, RF 99.1%. Table 2: RF sensitivity 97.16, specificity 99.9; NB lowest | Second same-data comparator; adds NB and kNN | **Internally inconsistent:** the text gives 99.46% as RF's NPV and 99.4% as accuracy for RF and KNN; the conclusion says "from images"; the dataset reference points to an unrelated chemistry paper; same leakage recipe. Cite as "reported", never as truth |
| 3 | **Bobbo et al. (2023)**, *J. Dairy Sci.* 106:1942-1952, doi 10.3168/jds.2022-22292 | 3,891 monthly records of 1,038 buffaloes, 6 herds; label = high SCC (>200,000 cells/mL) at the *next* test day; prevalence 40.3% | GLM, SVM, RF, NN | Compares **splitting by record vs by animal ID** (80/20), recursive feature elimination inside CV | Test AUC by animal ID 0.840-0.841 vs by record 0.790-0.814; best test method NN (accuracy 76.2%, F1 0.726 by animal ID). Authors recommend splitting by ID for repeated measures | **Published precedent for our cow-grouped split** (cite it when asked "why group by cow?") | Buffaloes, subclinical, monthly data: do not compare numbers with ours. The differences between split types are modest there |
| 4 | **Guo, Dai & Hu (2025)**, *Front. Vet. Sci.* 12:1575525, doi 10.3389/fvets.2025.1575525 | 4,000 cows (2,821 after balancing); Apr+May milk records -> health status in June (a real forecast) | XGBoost best; GBDT, SVM, KNN, LogReg, LSTM | 80/20 split, 5-fold CV | XGBoost AUC 0.75, accuracy 71.36%; LSTM accuracy 65%, AUC 0.73 | Shows what honest *forecasting* looks like (AUC about 0.75), versus our *detection* task; supports C1 (no forecasting claim) | Subclinical, production-record data |
| 5 | **Han et al. (2025)**, TBESO-BP, *Front. Vet. Sci.* 12:1396799, doi 10.3389/fvets.2025.1396799 | Monthly DHI data, Jan-Jul 2022; regression target = somatic cell count | BP neural network with initial weights tuned by a boosted snake optimiser (TBESO); 6 alternative regressors | Train/test partition | Test R2 0.94, MAE 2.07, RMSE 5.33 | Precedent for **metaheuristic tuning of a neural predictor in mastitis work**, which supports our GA-tuned radii idea | Regression on SCC, not classification; no grouping by cow stated in what we read |
| 6 | **Zhou et al. (2026)**, *Animals* 16(2):204, doi 10.3390/ani16020204 | Farm sensor data (activity, rumination, conductivity, yield), 14 days before clinical mastitis onset | Quantile regression features + 11 ML models; SHAP | 5-fold CV (a 70/30 split is also mentioned) | Best: partial least squares, AUC 0.789, sensitivity 0.500, specificity 0.947, F1 0.625 | Contrast: **post-hoc** explanation (SHAP) vs our **intrinsic** rules (ANFIS) | Sensitivity only 0.50 at their threshold |
| 7 | **Tian et al. (2024)**, *Animals* 14(3):427, doi 10.3390/ani14030427 | 447 mastitic + 2,146 healthy cows, 5 farms, NE China; activity, rumination, yield, conductivity | 9 ML algorithms; z-standardisation helped | Four lactation-stage subgroups; split details not checked | Multilayer neural net best, random forest second | Shows standardisation matters (we scale on train only) | Different sensors; class-imbalanced |
| 8 | **Bonestroo et al. (2022)**, *Comput. Electron. Agric.* 198:107002 | 14 farms, AMS sensor data; forecast whether a subclinical case becomes chronic | Gradient-boosting trees | **Trained on 7 farms, tested on the other 7** (farm-level split) | AUC 0.964 (vs 0.941 frequent-SCC rule); MCC 0.712 vs 0.630 vs 0.552; accuracy 0.888 | Another grouped-validation precedent (grouping by farm); gradient boosting as baseline | Different data and task |
| 9 | **Khan, Thorup & Luo (2024)**, *IEEE Trans. Ind. Informatics* 20(7), starting page 9508; doi not read | 415 cows, 75 with mastitis, 4.75 years; IceQube leg sensor (behaviour) + yield, lactation, age | SVM, four kernels | **5-fold CV on randomly shuffled samples** (no cow grouping stated in what we read) | SVM accuracy 89.2%; RBF AUC 0.95 (sensitivity 88.5, specificity 90.4) | IoT decision-support framing, and a "published at IEEE with random-shuffle CV" example | Daily samples of the same cow may sit in train and test; do not over-criticise, say "does not state cow-level grouping" |
| 10 | **Deepika & Sharma (2024)**, *J. Inf. Optim. Sci.* 45(6):1519-1538, doi 10.47974/JIOS-1631 | Review of ensemble models (RF, boosting, bagging, stacking) for mastitis | - | - | States ensembles can be less interpretable than simpler models | Supports the interpretability argument (a review, cite as such) | A review; not primary evidence |
| 11 | **Rachah (2025)**, *Procedia Comput. Sci.* 270:5655-5664 (KES 2025) | Herd-level mastitis transmission (Corynebacterium) | SIS compartmental model vs RNN, Neural ODE, PINN | - | RNN RMSE 0.004 vs 0.050 for SIS | Low relevance (transmission, not detection). Optional one-line mention of hybrid mechanistic + learned models | Do not build on it |
| 12 | **Jang (1993)**, "ANFIS: Adaptive-Network-Based Fuzzy Inference System", *IEEE Trans. Syst., Man, Cybern.* 23(3):665-685 | - | ANFIS architecture, hybrid learning (least squares forward, gradient descent backward) | - | - | **The method paper.** Cite for the 5 layers, Sugeno rules, hybrid learning | We use first-order Sugeno consequents (Jang's Type 3) |

## 3. Not in the 12 (still unopened; do not cite until opened)

| Item | Why we want it | Status |
|---|---|---|
| Chiu (1994), "Fuzzy model identification based on cluster estimation", *J. Intelligent & Fuzzy Systems* 2(3):267-278 | The subtractive-clustering algorithm we implement | Get the PDF if possible |
| Ebrahimi et al. (2019), deep learning and gradient-boosted trees for subclinical mastitis (364,249 milkings; 84.9% accuracy reported second-hand) | State-of-the-art black-box reference | Unopened |
| Ankitha & Manjaiah, KNN vs SVM on this dataset (2020/2021; second-hand: KNN about 73%, SVM about 86%) | The dataset authors' own analysis; may define the label rule | Unopened; other sources disagree on the numbers |
| Mammadova & Keskin (2015); Kohli (2011) | Prior neuro-fuzzy mastitis work: **must read before writing the novelty claim** | Titles only, via Kumar's reference list |
| Takagi & Sugeno (1985) | Origin of Sugeno rules | Optional |

## 4. Research gap (draft, true for the reviewed set)

> Studies that use this IoT udder dataset (Ghafoor and Sitkowska 2021; Kumar et al. 2023) report about 99% accuracy with black-box tree and neighbour models, evaluated on test sets that share cows and duplicated rows with the training data, and neither offers rule-level explanations. Neuro-fuzzy methods have been applied to mastitis on milking-system data (Mammadova and Keskin 2015; Kohli 2011, titles only, to be read), but to our knowledge not to these udder-sensor data under cow-level validation. We therefore (i) quantify how much of the reported accuracy is leakage and (ii) test whether a subtractive-clustering ANFIS gives rule-based, auditable detection at accuracy close to the black-box baselines.

"To our knowledge" is required: our search was not exhaustive. Replace "titles only" after reading the two neuro-fuzzy papers.

## 5. Baselines suggested by the literature (for Final Review)

MasPA and Kumar between them used RF, DT, kNN, SVM, NB, XGBoost, LightGBM, Bagging, LogReg. We already have LogReg, DT, RF. Add **kNN, SVM (RBF), gradient boosting** (Bonestroo, Guo, MasPA all report boosted trees as strong) and a **MasPA-matched feature set** (8 sensors + Temperature only, no derived indices, no breed or months) so that our table is comparable with theirs.

## 6. Where each paper goes in the report

| Report section | Papers |
|---|---|
| Introduction and problem | Deepika & Sharma (review), MasPA (cost figures) |
| Literature survey | All of section 2, grouped: same-dataset (1, 2), validation design (3, 8, 9), forecasting vs detection (4), metaheuristic + NN (5), explainability (6, 10), sensors (7), method (12) |
| Methodology | Jang (1993), Chiu (1994, once opened), Bobbo (grouped split) |
| Results discussion | MasPA and Kumar numbers vs our cow-grouped numbers (after `leakage_replication.py`) |

---
Changelog: 2026-10-26: created from the 12 PDFs attached by A.
