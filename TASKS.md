# Tasks - Review II is Tuesday 2026-10-27

A = Manikanta, B = teammate. Tick boxes in your PR. Anything unfinished by Monday night moves to the Final Review list at the bottom.

## Saturday night (Oct 24)
- [ ] A: create the private GitHub repo, push this folder, add B as collaborator
- [ ] B: accept invite, clone, `pip install -r requirements.txt`, put the CSV in `data/raw/`
- [ ] A+B: run `test_modules.py`, `data_prep.py`, `baselines.py`, `train_anfis.py` on BOTH machines
- [ ] A: send Claude `reports/comparison.csv` and `reports/anfis_ra_grid.csv`
- [ ] B: confirm your numbers equal A's (same `comparison.csv`). Different = investigate first

## Sunday evening (Oct 25), time-boxed
| When (IST) | Owner | Task | Done when |
|---|---|---|---|
| 18:00-18:30 | A | Send Claude: the working CSV (406 KB) + `reports/` files: `comparison.csv`, `anfis_ra_grid.csv`, `anfis_results.csv`, `rules_*.txt`, `example_explanations.txt`. **Claude cannot build slides or numbers without these** | Files attached in chat |
| 18:30-19:00 | A | T0.4: extract the CSV from the Version 2 zip (`unzip Clinical_Mastitis_cows_version2.zip "*.csv" -d v2/`), run `python verify_dataset_version.py data/raw/clinical_mastitis_cows.csv v2/<csv>`; T0.5: `python compare_versions_deep.py <version1.csv> data/raw/clinical_mastitis_cows.csv > reports/version_diff.txt`; paste both outputs to Claude | IDENTICAL (or tell Claude it is not); diff output pasted |
| 18:00-20:00 | B | `python audit_odd_cow.py` (decide keep/drop cow10, log in JOURNAL); read `anfis.py` + `subtractive_clustering.py` line by line; prepare the 15-min explanation | B can explain without the code open |
| 18:00-20:00 | A | Get the PDFs from the college library: Ankitha & Manjaiah (KNN vs SVM, clinical mastitis, IoAHT); Ankitha et al. "Detection of Mastitis Disease in Cow with Machine Learning Classifiers"; Ebrahimi et al. 2019 (GBT/deep learning, subclinical mastitis); Jang 1993 (ANFIS); Chiu 1994 (subtractive clustering); MasPA (Ghafoor & Sitkowska 2021). Open each; do not cite unopened papers | PDFs in `docs/papers/` (not committed if licensed) |
| 20:00-21:30 | A | Literature table (6-8 rows) + research gap in 3 sentences (T1.1) from papers actually opened | Every row opened and checked |
| Tonight | Claude | After A's files: reproduce numbers, Version 1 vs Version 2 table, architecture + data-flow diagrams, slides v1 with speaking notes, article skeleton, viva sheet | Delivered before A sleeps or first thing Monday |

## Monday (Oct 26): freeze day (revised 11:00 IST after the 12 papers)
| When (IST) | Owner | Task | Done when |
|---|---|---|---|
| 11:00-11:30 | A | Run `python leakage_replication.py data/raw/clinical_mastitis_cows.csv --seeds 5` (T1.8); paste the printed tables to Claude. **Also attach the working CSV and the `reports/` files (comparison.csv, anfis_ra_grid.csv, anfis_results.csv, rules_*.txt, example_explanations.txt)** | Tables pasted; files attached |
| 11:00-11:30 | A | Hash check + `compare_versions_deep.py` (T0.4, T0.5) if not done yet; paste outputs | IDENTICAL or tell Claude |
| 11:00-13:00 | A | Read `docs/LITERATURE.md` sections 1-2; open MasPA p.2 (3,961 / 2,639; oversampling sentence), Kumar p.5, Bobbo results, Jang section III; tick each fact you verified | You can explain the 3 findings without the file |
| 11:00-13:00 | B | Find and open Mammadova & Keskin 2015 (ANFIS, subclinical mastitis) and Kohli 2011 (neuro-fuzzy, AMS); also Chiu 1994. Note: data, label, validation, result, rules or black box | 5 lines per paper in JOURNAL |
| 11:00-13:00 | B | `audit_odd_cow.py`; read `anfis.py` + `subtractive_clustering.py`; prepare the 15-min explanation | Explains without the code open |
| 13:00-14:30 | Claude | Slides v1 with speaking notes, architecture + data-flow diagrams, article skeleton, viva sheet, built from A's real files | Delivered to chat |
| 14:30-15:30 | A+B | Review slides; every number traced to a file in `reports/`; banned words: early, subclinical, synthetic, forecasting | Each number has a source |
| 15:30-16:30 | A+B | Cross-teach (15 min each way), then swap 5 questions each | Neither stalls on "why" |
| 16:30-17:30 | A+B | Viva drill: 10 questions out loud, timed; one full rehearsal | Two clean run-throughs |
| 17:30-18:00 | A+B | Commit; `git tag review2`; push | Tag exists |
| 18:00 | A+B | **Freeze.** No new features, no re-running experiments | - |

## Tuesday (Oct 27): Review II

## After Review II (Final Review list)
- [ ] GA tuning of cluster radii (ra, rb)
- [ ] Grouped 5-fold CV, 2+ competing algorithms (add gradient boosting), significance by paired cow-bootstrap + McNemar (not Wilcoxon on 5 folds), MasPA/Kumar comparison on the same dataset (reported numbers + our re-run of their recipe), add kNN and SVM and a MasPA-matched 9-feature set
- [ ] Ablation of the Asymmetry Index formula (e.g. coefficient of variation)
- [ ] Streamlit demo (`app.py`)
- [ ] Submission-ready paper, patent and startup notes, full report

---
Changelog: 2026-10-25: replaced the I*/E* task (the dataset page already answers it: inhale/exhale limits, very likely) with a dataset-version check and a companion-paper search.
Changelog (later 2026-10-25): working CSV found to differ from the Version 1 CSV; provenance is now the blocker.
Changelog (evening 2026-10-25): provenance resolved per A (Version 2 zip, hash check pending); replaced the Sunday/Monday sections with a time-boxed schedule.
Changelog (2026-10-26): Monday schedule revised after the 12 papers; T1.8 leakage replication added; MasPA is a same-dataset comparator.
