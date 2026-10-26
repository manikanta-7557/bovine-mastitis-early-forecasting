"""
compare_versions_deep.py - HOW exactly do two versions of the Mendeley CSV differ?

Usage:
    python compare_versions_deep.py data/raw/clinical_mastitis_cows_version1.csv data/raw/clinical_mastitis_cows.csv
    (file 1 = Version 1, file 2 = the Version 2 / working file)

Why: verify_dataset_version.py says THAT the files differ (16,152 cells). This script says HOW:
which rows changed label, whether whole cows were relabelled, whether the changes sit only in those
rows, and whether the changed rows look like edits of the originals or brand-new numbers.
It only describes; it does not decide WHY (the Mendeley page does not say).
"""
import sys

import numpy as np
import pandas as pd

KEYS = ["Cow_ID", "Day"]
SENS = ["IUFL", "EUFL", "IUFR", "EUFR", "IURL", "EURL", "IURR", "EURR"]
OTHER = ["Temperature", "Hardness", "Pain", "Previous_Mastits_status", "Months after giving birth"]


def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def main(p1, p2):
    a, b = load(p1), load(p2)
    m = a.merge(b, on=KEYS, suffixes=("_v1", "_v2"), validate="one_to_one")
    assert len(m) == len(a) == len(b), "files do not align one-to-one on Cow_ID + Day"
    print(f"Aligned rows: {len(m)}")

    print("\n1. LABEL TRANSITIONS (class1: file 1 -> file 2)")
    print(pd.crosstab(m["class1_v1"], m["class1_v2"], rownames=["file1"], colnames=["file2"]).to_string())
    m["flip"] = m["class1_v1"] != m["class1_v2"]

    per_cow = m.groupby("Cow_ID").agg(n=("flip", "size"), flipped=("flip", "sum"))
    whole = per_cow[per_cow.flipped == per_cow.n]
    part = per_cow[(per_cow.flipped > 0) & (per_cow.flipped < per_cow.n)]
    print(f"\n2. COWS: {len(whole)} cows have ALL rows relabelled; "
          f"{len(part)} cows partly relabelled {part.flipped.to_dict()}")

    cols = [c for c in SENS + OTHER + ["Milk_visibility"] if c in a.columns and c in b.columns]
    changed_any = pd.Series(False, index=m.index)
    for c in cols:
        changed_any |= m[f"{c}_v1"] != m[f"{c}_v2"]
    print(f"\n3. ROWS WITH ANY CHANGE: {int(changed_any.sum())} "
          f"(label flipped: {int((changed_any & m.flip).sum())}, label NOT flipped: {int((changed_any & ~m.flip).sum())})")

    print("\n4. PER COLUMN: cells changed in flipped rows vs other rows; change = file2 - file1")
    rows = []
    for c in cols:
        d = m[f"{c}_v2"] - m[f"{c}_v1"]
        ch = d != 0
        f = ch & m.flip
        rows.append([c, int(f.sum()), int((ch & ~m.flip).sum()),
                     round(float(d[ch].mean()), 2) if ch.any() else 0.0,
                     round(float(d[f].mean()), 2) if f.any() else 0.0,
                     round(float(d[f].std()), 2) if f.sum() > 1 else 0.0])
    print(pd.DataFrame(rows, columns=["column", "changed_in_flipped", "changed_in_other",
                                      "mean_change_all", "mean_change_flipped", "std_change_flipped"]).to_string(index=False))

    print("\n5. MEAN VALUES: do the relabelled rows now look like the original mastitis rows?")
    healthy = m[(m.class1_v1 == 0) & ~m.flip]
    mast = m[m.class1_v1 == 1]
    flipped = m[m.flip]
    out = []
    for c in SENS + ["Temperature", "Hardness", "Pain"]:
        out.append([c, round(healthy[f"{c}_v1"].mean(), 2), round(mast[f"{c}_v1"].mean(), 2),
                    round(flipped[f"{c}_v1"].mean(), 2), round(flipped[f"{c}_v2"].mean(), 2)])
    print(pd.DataFrame(out, columns=["column", "orig_healthy(f1)", "orig_mastitis(f1)",
                                     "relabelled_rows(f1 values)", "relabelled_rows(f2 values)"]).to_string(index=False))

    print("\n6. IN RELABELLED ROWS: correlation of file-1 and file-2 values "
          "(near 1 = edited from the same base, near 0 = replaced by new numbers)")
    for c in SENS + ["Temperature"]:
        if len(flipped) > 2:
            print(f"   {c}: {flipped[f'{c}_v1'].corr(flipped[f'{c}_v2']):.3f}")

    if "cow10" in set(m["Cow_ID"]):
        print("\n7. cow10 in both files")
        print(m.loc[m.Cow_ID == "cow10", ["Day", "Hardness_v1", "Hardness_v2", "Milk_visibility_v1",
                                           "Milk_visibility_v2", "class1_v1", "class1_v2"]].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
