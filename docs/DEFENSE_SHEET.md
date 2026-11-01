# Final Review defense sheet (hard questions and safe answers)
Numbers are from `reports/final/` (grouped 5-fold, 1,100 cows). Say "in our experiments", never "proves".

**1. Why ANFIS if random forest is better?** Random forest on raw sensors gets F1 0.983, ANFIS 0.919. We chose ANFIS for a 4 to 5 rule model written in sensor units that a vet can check; the price is about 6 F1 points against the forest. ANFIS is statistically tied with gradient boosting (cow-level McNemar p = 0.74) and better than logistic regression, decision tree and SVM on the raw sensors.
**2. Why not just use the best model?** If only accuracy mattered we would. The project question is whether an interpretable model can come close; the table answers it.
**3. Why are you below the 99% in the papers?** Mostly feature design: our two engineered indices lower random forest from 0.983 to 0.914. The split protocol matters little: 0.993 vs 0.984 for random forest. Our raw-sensor random forest reaches 0.983, so we can reproduce the order of magnitude under a stricter split.
**4. So was the leakage in the papers real?** The recipe does put the same cows (99.9% of test rows) and copied rows (44%) in train and test. Its measured effect on random-forest accuracy is about one point. We do not say the authors erred; we replicated the recipe, not their code.
**5. Why group by cow?** Six correlated rows per cow. Bobbo et al. (2023) split by animal and found different AUC than by record.
**6. What is the unit of evaluation?** Rows for F1/AUC; cows for McNemar (a cow is positive if its mean score over six days is at least 0.5) and for bootstrap resampling.
**7. Why not Wilcoxon over folds?** Five folds give a minimum p of 0.0625.
**8. Multiple comparisons?** Yes, many tests; p-values are uncorrected and descriptive. The CIs and effect sizes are the main evidence.
**9. Explain subtractive clustering.** Each point gets a density score from its neighbours; the densest point becomes a centre, its neighbourhood is suppressed, repeat until the remaining density falls under 15% of the first (Chiu 1994; we have not yet read the paper itself, so describe our implementation).
**10. Explain hybrid learning.** With memberships fixed, the output is linear in the consequent parameters, so ridge least squares solves them exactly; centres and widths are updated by Adam. Analytic gradients match numerical ones to 2.1e-9 (test).
**11. Why first-order Sugeno?** More expressive than zero-order with fewer rules; drawback: a rule's output depends on the linear term, so the printed "output at centre" is not the output for a given cow. The dashboard shows firing share and the cow's values next to the rule centre.
**12. Did GA help?** No: GA 0.913 vs grid 0.919 F1 with 58 vs 5 evaluations per fold. We report it as a negative result.
**13. Why did the engineered features do worse?** Averaging eight sensors and one max-min ratio discard which quarter is swollen. Raw sensors keep it; rules in raw units show one swollen quarter per rule.
**14. Why two dataset versions, and which is right?** The authors do not document the difference (1,529 labels). We use 3,961 / 2,639, the split both published studies report, and treat Version 1 as a sensitivity check. [Say "verified" for provenance only after the hash check.]
**15. Is it a clinical tool?** No. It detects existing clinical mastitis in this dataset; no external farm validation; thresholds not clinically validated; the dashboard is a prototype.
**16. Can it forecast?** No. Labels are nearly constant within a cow; any forecast claim is unsupported.
**17. Duplicated rows?** 28% of test rows are exact copies of training rows from other cows even under cow-grouping; unexplained, listed as a limitation.
**18. What would you do next?** External farm data, calibrated thresholds, veterinarian review of the rules, contact with the dataset authors about the duplicates and versions.
**19. What did your teammate contribute?** Streamlit prototype and model upload (Madhur Sathya); his stored models were replaced by models trained on our pipeline after we found they did not match the reported results.
**20. How do I know your numbers are reproducible?** Fixed seed, scripts in the repo, `reports/final` outputs, tests, and the run log; rerunning gives the same pooled table.
