# Interpretable ANFIS for detecting clinical bovine mastitis from udder-sensor data: report draft
Draft for the 9-section report structure of "0 Evaluation Criteria.docx". Every number is from `reports/final/` (run 1 Nov 2026) unless marked [verify]. Bracketed items are for the authors to fill.

## 1. Abstract
Clinical mastitis is among the costliest diseases of dairy cattle. Two earlier studies report about 99% accuracy on a public udder-sensor dataset (1,100 cows, six daily rows each) using random forest and nearest-neighbour models, with oversampling before the train-test split and random row splits. We evaluate eight models with cow-grouped five-fold validation on 1,100 cows, including a first-order Sugeno ANFIS whose rules come from subtractive clustering and whose parameters are learned by hybrid learning. On the eight raw sensors the ANFIS reaches F1 0.919 (95% CI 0.904 to 0.934) and AUC 0.980 with 4 to 5 rules, better than logistic regression, decision tree and SVM, statistically indistinguishable from gradient boosting at the cow level (McNemar p = 0.74), and below random forest (F1 0.983) and kNN (0.961). Replacing the eight sensors by two engineered indices lowers ANFIS to F1 0.865 and random forest to 0.914. Replicating the published recipe changes random-forest accuracy by only about one point (0.993 versus 0.984). Rules written in raw sensor units give a readable account of each decision. The task is detection of existing clinical mastitis, not forecasting.
Keywords: ANFIS, subtractive clustering, mastitis, interpretable machine learning, grouped cross-validation.

## 2. Introduction and problem definition
- Problem (Smart India Hackathon SIH26109, Ministry of Fisheries, Animal Husbandry and Dairying): clinical mastitis detection for dairy farms. Cost figures (about 6% of production value; up to USD 444 per case) are quoted in MasPA (Ghafoor and Sitkowska 2021) from earlier sources [verify originals].
- Task: binary classification of an existing clinical mastitis status (`class1`) from 8 udder-sensor readings and optional clinical inputs. Not forecasting; the data have six rows per cow with an almost constant label.
- Objectives: (O1) build a readable ANFIS; (O2) compare with seven baselines under cow-grouped validation; (O3) explain predictions through fired rules; (O4) measure how evaluation protocol and feature design change reported accuracy; (O5) provide a prototype dashboard.
- Contributions: a cow-grouped benchmark of eight models on this dataset with confidence intervals and paired tests; evidence that feature design matters more than split protocol here; a 4 to 5 rule ANFIS in raw sensor units.

## 3. Literature survey and research gap
Use the table in `docs/LITERATURE.md` (12 papers). Group as: same-dataset studies (Ghafoor and Sitkowska 2021; Kumar, Sindhwani and Sachdeva 2023), grouped validation precedents (Bobbo et al. 2023, by animal; Bonestroo et al. 2022, by farm), forecasting versus detection (Guo et al. 2025), post-hoc explanation (Zhou et al. 2026), metaheuristic tuning (Han et al. 2025), reviews (Deepika and Sharma 2024), method (Jang 1993).
Gap (within the literature reviewed, to our knowledge): studies on this dataset report about 99% with black-box models and no rule-level explanation; neuro-fuzzy mastitis work exists on milking-system data (Mammadova and Keskin 2015; Kohli 2011, [read before final submission]) but we found no application of a subtractive-clustering ANFIS to these udder-sensor data under cow-level validation. Do not claim "first".

## 4. Methodology and system design
**4.1 Data.** Mendeley Data kbvcdw5b4m (CC BY 4.0; Ankitha, Manjaiah, Kartik). Working file: 6,600 rows, 1,100 cows, 3,961 healthy and 2,639 mastitis rows, SHA256 e4166166...f532d8 (match with the official Version 2 zip [verify]). Version 1 (5,490 / 1,110) differs in 1,529 labels for undocumented reasons and was used only as a sensitivity check. `Milk_visibility` equals the label and is excluded.
**4.2 Validation.** StratifiedGroupKFold(5, shuffle, seed 42) grouped by cow; scalers fitted on training cows only; ANFIS radius chosen on an inner validation fold of training cows; pooled out-of-fold predictions over 1,100 cows; 95% CIs by 1,000 cow-level bootstrap resamples; paired comparison by exact McNemar test at cow level (a cow is positive if its mean daily score is at least 0.5).
**4.3 Feature sets.** S5 (swelling index, asymmetry index, temperature, months postpartum, breed), F7 (S5 + hardness + pain), R8 (8 raw sensors), R13 (8 sensors + temperature, months postpartum, breed, hardness, pain). Swelling index = mean of 8 sensors; asymmetry index = (max minus min of 4 quarter totals) divided by their mean (our own design).
**4.4 Models.** Majority, logistic regression, decision tree (depth 4), random forest (300 trees), kNN (k=5), SVM (RBF), gradient boosting, ANFIS.
**4.5 ANFIS.** Layer 1 Gaussian memberships exp(-(x-c)^2 / 2 sigma^2); layer 2 product firing strength; layer 3 normalisation; layer 4 first-order Sugeno consequent f_i = p_i.x + r_i; layer 5 weighted sum. Hybrid learning (Jang 1993): ridge least squares for consequents, Adam for centres and widths. Rules by subtractive clustering (Chiu 1994 [open the paper before submission]): alpha = 4/ra^2, rb = 1.5 ra, accept 0.5, reject 0.15, sigma = ra/sqrt(8), at most 12 rules; radius grid 0.3 to 0.7.
**4.6 System diagram.** Raw CSV -> cow-grouped split -> train-only scaling -> feature set -> (baselines | ANFIS) -> out-of-fold predictions -> metrics, CIs, McNemar -> rule explanations -> dashboard. (Redraw from Review II slide 6.)
**Defense guide:** grouped split because six rows per cow; bootstrap over cows because rows are not independent; McNemar at cow level because cows are the independent units; Wilcoxon over 5 folds rejected (minimum p = 0.0625).

## 5. Implementation
Python 3.13; numpy 2.5.3, pandas 3.0.5, scikit-learn 1.9.1, scipy, Streamlit. Modules: `data_prep.py`, `baselines.py`, `subtractive_clustering.py`, `anfis.py`, `explain.py`, `metrics.py`, `train_anfis.py`, `final_experiments.py`, `analyze_final.py`, `train_final_models.py`, `ga_tuning.py`, `app_core.py`, `app.py`, `leakage_replication.py`. Tests: `test_modules.py` (clustering, analytic vs numerical gradient 2.1e-9, least squares, hybrid learning) and `test_app_core.py` (dashboard path equals model path; AUC on 300 test rows 0.989 / 0.957). Full experiment runs in about 2 minutes plus 4 minutes of analysis on 2 CPU cores.

## 6. Results and analysis
**6.1 Pooled grouped 5-fold, 1,100 cows (F1 / AUC, 95% CI in `RESULTS.md`).**
| Model | S5 | F7 | R8 | R13 |
|---|---|---|---|---|
| ANFIS | 0.865 / 0.919 | 0.916 / 0.941 | 0.919 / 0.980 | 0.937 / 0.977 |
| Random forest | 0.914 / 0.978 | 0.928 / 0.978 | 0.983 / 0.999 | 0.975 / 0.997 |
| Gradient boosting | 0.910 / 0.975 | 0.932 / 0.979 | 0.922 / 0.984 | 0.939 / 0.988 |
| kNN | 0.872 / 0.924 | 0.906 / 0.941 | 0.961 / 0.990 | 0.933 / 0.974 |
| Decision tree | 0.884 / 0.940 | 0.925 / 0.946 | 0.890 / 0.909 | 0.929 / 0.929 |
| SVM | 0.853 / 0.888 | 0.904 / 0.911 | 0.898 / 0.959 | 0.910 / 0.943 |
| Logistic regression | 0.854 / 0.892 | 0.903 / 0.933 | 0.858 / 0.895 | 0.898 / 0.942 |
| ANFIS rules (mean, range) | 9.4 (7 to 12) | 9.6 (7 to 12) | 4.2 (4 to 5) | 7.0 (5 to 9) |
**6.2 Paired tests (ANFIS vs model, cow-level McNemar; uncorrected).** R8: vs logistic regression, decision tree, SVM p < 0.0001 (ANFIS better); vs gradient boosting p = 0.74 (no difference); vs kNN and random forest p < 0.0001 (they are better). R13: vs gradient boosting p = 0.58, kNN 0.15, decision tree 0.22 (no difference); vs logistic regression p < 0.0001 and SVM p = 0.0015 (ANFIS better); vs random forest p < 0.0001 (forest better). Engineered sets: ANFIS is below the tree-based models on S5 and close to them on F7.
**6.3 Single held-out fold (Review II protocol, fold 0).** R8 ANFIS: F1 0.936, AUC 0.983, 5 rules, cow-level F1 0.965. R13: F1 0.937, AUC 0.961. Review II engineered-feature results: sensor-only ANFIS F1 0.869, full-clinical 0.908 (reproduced).
**6.4 Evaluation protocol (`leakage_replication.py`, random forest, 8 sensors + temperature, 5 seeds).** Oversample then random row split: accuracy 0.993 (99.9% of test rows from cows seen in training; 44% exact copies of training rows). Cow-grouped: 0.984. About one point of difference. Under the cow-grouped split 28% of test rows are still exact copies of training rows from other cows (unexplained).
**6.5 GA tuning (R8).** GA over (ra, rb factor, learning rate), 58 fitness evaluations per fold versus 5 for the grid: F1 0.913 vs 0.919, AUC 0.977 vs 0.979. GA gave no gain.
**6.6 Rules.** The R8 model has 5 rules in raw units (`rules_R8.txt`). For example one rule is centred on a very large front-left quarter (IUFL about 318, EUFL about 365) with other quarters medium and outputs about 1.0; another is centred on all sensors low (150 to 195) with output about 0.1. [Add a figure and one worked example from `example_explanations_raw.txt`.]
**Interpretation.** Random forest and kNN on raw sensors are the best models; ANFIS trades about 6 F1 points against random forest for 4 to 5 readable rules. Feature design (two indices instead of eight sensors) costs more than the split protocol does.

## 7. Discussion, limitations, future scope
- ANFIS does not beat random forest or kNN. Its value is a compact rule base; whether vets find the rules useful is untested.
- Limits: one public dataset; six rows per cow; label nearly constant within a cow, so detection only; two undocumented dataset versions; 28% duplicated rows across cows unexplained; threshold fixed at 0.5; no external farm validation; first-order consequents mean a rule's output depends on the linear term, so "output at the centre" is not the output for a given cow; p-values uncorrected for multiple comparisons; Streamlit dashboard not clinically validated; temperature unit undocumented.
- Future: external data; calibration and threshold choice for a cost-sensitive setting; interval type-2 or zero-order ANFIS for simpler rules; veterinarian review of rules; explaining the duplicated rows with the dataset authors.

## 8. Article documentation
Working title: "Cow-grouped evaluation and interpretable neuro-fuzzy detection of clinical bovine mastitis from udder-sensor data". Outcome type: research article. Venue: [to be selected]. Status: draft. Outline: Introduction; Related work; Data and protocol; Models; Experiments; Results; Discussion; Conclusion. Claims allowed: sections 1 and 6 above. Claims forbidden: forecasting, early or subclinical detection, ANFIS superiority over RF, first ANFIS for mastitis, accusing earlier authors.

## 9. References (only items read or with known DOI; check formatting against the target venue)
1. Ghafoor, N.A.; Sitkowska, B. MasPA: A Machine Learning Application to Predict Risk of Mastitis in Cattle from AMS Sensor Data. AgriEngineering 2021, 3(3), 575-583. doi:10.3390/agriengineering3030037
2. Kumar, Sindhwani, Sachdeva. Detection of Mastitis Disease in Cow with Machine Learning Classifiers. 2023. doi:10.52865/DQFZ6016 [journal name from DOI page]
3. Bobbo et al. J. Dairy Sci. 2023, 106, 1942-1952. doi:10.3168/jds.2022-22292
4. Bonestroo et al. Comput. Electron. Agric. 2022, 198, 107002.
5. Guo, Dai, Hu. Front. Vet. Sci. 2025, 12, 1575525. doi:10.3389/fvets.2025.1575525
6. Han et al. Front. Vet. Sci. 2025, 12, 1396799. doi:10.3389/fvets.2025.1396799
7. Zhou et al. Animals 2026, 16(2), 204. doi:10.3390/ani16020204
8. Tian et al. Animals 2024, 14(3), 427. doi:10.3390/ani14030427
9. Jang, J.-S.R. ANFIS. IEEE Trans. Syst. Man Cybern. 1993, 23(3), 665-685.
10. Chiu, S.L. Fuzzy model identification based on cluster estimation. J. Intell. Fuzzy Syst. 1994, 2(3), 267-278 [open before citing].
11. Ankitha, Manjaiah, Kartik. Clinical Mastitis in Cows based on Udder Parameter using IoT. Mendeley Data, doi:10.17632/kbvcdw5b4m.2 (and .1 for Version 1).
12. Mammadova, Keskin 2015; Kohli 2011 [read before citing].
