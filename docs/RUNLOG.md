# Experiment run log

No run log existed when this file was added. Records marked **RETROSPECTIVE** were
assembled from observed command output and current artifacts; exact historical timestamps,
environment/package versions, and the revision at execution were not captured. They are
not independent reproductions. New records should be appended with a timestamp, environment,
revision, data/config fingerprints, outputs and exit status.

| Timestamp (local, +05:30) | Record | Environment | Command / outcome | Revision / dataset / config | Outputs and measured summary | Independent reproduction? |
|---|---|---|---|---|---|---|
| 2026-10-29, exact time not captured | RETROSPECTIVE | Historical environment not captured; current audit runtime is Python 3.12.10, NumPy 2.2.4, pandas 2.2.3, scikit-learn 1.8.0 (not asserted as historical) | `python test_modules.py` — exit 0, all four tests passed | Execution revision unknown; current audited HEAD `deda8694d6632b03d0a0b8c40e65ce2f53dddd43`; working CSV SHA-256 `e4166166bf83d91a63a62d1ab5e8bd66f9f7ce6687233a67abc4d99918f532d8`; config did not exist | Test output only; no report/model outputs | No |
| 2026-10-29, exact time not captured | RETROSPECTIVE | Historical environment not captured | `python data_prep.py` — exit 0 | Execution revision unknown; same working CSV hash as above, measured during 2026-10-30 audit; settings from source: seed 42, StratifiedGroupKFold(5), test fold 0; config did not exist | `data/processed/train.csv`, `test.csv`, `full_processed.csv`, `data_prep_summary.json`, figures 1–7; summary reports 5280/1320 rows and 880/220 cows with zero overlap | No |
| 2026-10-29, exact time not captured | RETROSPECTIVE | Historical environment not captured | `python baselines.py` — exit 0 | Execution revision unknown; current working CSV hash above; source random state 42; fixed baseline parameters in `baselines.py`; config did not exist | `reports/baseline_results.csv`, `reports/figures/fig8_baseline_roc.png`; RF row F1/AUC: Early-Warning 0.9301/0.9834, Full-Clinical 0.9277/0.9761 | No |
| 2026-10-29, exact time not captured | RETROSPECTIVE | Historical environment not captured | `python train_anfis.py` — exit 0 | Execution revision unknown; current working CSV hash above; source radius grid 0.3–0.7, max 12 rules, validation fold 1, max 100 epochs, learning rate 0.01; config did not exist | `reports/anfis_results.csv`, `anfis_ra_grid.csv`, `comparison.csv`, rules/explanations, `models/anfis_*.npz`, figures 9–10; ANFIS row F1/AUC 0.8694/0.9169 (sensor-only) and 0.9076/0.9380 (Full-Clinical) | No |
| 2026-10-30, exact time not captured | Audit; read-only | Python 3.12.10, NumPy 2.2.4, pandas 2.2.3, scikit-learn 1.8.0 | `python verify_dataset_version.py data/raw/clinical_mastitis_cows.csv data/raw/clinical_mastitis_cows_version1.csv` — exit 0 | HEAD `deda8694d6632b03d0a0b8c40e65ce2f53dddd43`; no config used | Printed both hashes and found differing columns/cells; working file is not identical to Version 1. No output file written. | No |
| 2026-10-30, exact time not captured | Audit; read-only | Python 3.12.10, NumPy 2.2.4, pandas 2.2.3, scikit-learn 1.8.0 | `python compare_versions_deep.py data/raw/clinical_mastitis_cows_version1.csv data/raw/clinical_mastitis_cows.csv` — exit 0 | HEAD `deda8694d6632b03d0a0b8c40e65ce2f53dddd43`; no config used | Printed aligned row/label/feature differences; no output file written | No |
| 2026-10-30, exact time not captured | Audit; read-only | Python 3.12.10, pandas 2.2.3 | `python audit_odd_cow.py data/raw/clinical_mastitis_cows.csv` — exit 0 | HEAD `deda8694d6632b03d0a0b8c40e65ce2f53dddd43`; working CSV SHA-256 above | Printed cow10 daily labels and selected clinical values; no output file written | No |

These are local command records only. “Independent reproduction” requires a separate
environment or independent operator and has not been demonstrated by these entries.
