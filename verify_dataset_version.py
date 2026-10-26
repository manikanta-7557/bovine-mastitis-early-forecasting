"""
verify_dataset_version.py - which Mendeley version is our CSV, and do two copies match?

Why: the Mendeley page has Version 1 (clinical_mastitis_cows_version1.csv, 413 KB, published
22 Oct 2020) and Version 2 (a 1.48 GB zip, published 20 Nov 2020). The paper and report must
cite the DOI of the version we actually used (.1 or .2). This script gives a fingerprint of
our file and, if you pass a second CSV, compares them.

Usage:
    python verify_dataset_version.py data/raw/clinical_mastitis_cows.csv
    python verify_dataset_version.py data/raw/clinical_mastitis_cows.csv path/to/clinical_mastitis_cows_version1.csv

What to do with the output:
  * Run it on BOTH partners' machines. The SHA256 line must be identical.
  * Download the Version 1 CSV (small) from https://data.mendeley.com/datasets/kbvcdw5b4m and compare.
  * Write the result in JOURNAL.md (which version, SHA256).
"""
import hashlib
import os
import sys

import pandas as pd


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def describe(path: str) -> pd.DataFrame:
    df = load(path)
    print(f"File      : {path}")
    print(f"Size      : {os.path.getsize(path) / 1024:.1f} KB")
    print(f"SHA256    : {sha256(path)}")
    print(f"Shape     : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Columns   : {list(df.columns)}")
    if "Cow_ID" in df.columns:
        print(f"Cows      : {df['Cow_ID'].nunique()}  (rows per cow: {df.groupby('Cow_ID').size().value_counts().to_dict()})")
    if "class1" in df.columns:
        print(f"class1    : {df['class1'].value_counts().sort_index().to_dict()}")
    return df


def compare(a: pd.DataFrame, b: pd.DataFrame) -> None:
    print("\n=== COMPARISON ===")
    if a.shape == b.shape and list(a.columns) == list(b.columns) and a.equals(b):
        print("IDENTICAL content (same columns, same values, same row order).")
        return
    only_a = [c for c in a.columns if c not in b.columns]
    only_b = [c for c in b.columns if c not in a.columns]
    print(f"Columns only in file 1: {only_a} | only in file 2: {only_b}")
    print(f"Rows: file 1 = {len(a)}, file 2 = {len(b)}")
    if not {"Cow_ID", "Day"} <= set(a.columns) & set(b.columns):
        print("Cannot align rows (need Cow_ID and Day in both files).")
        return
    common = [c for c in a.columns if c in b.columns and c not in ("Cow_ID", "Day")]
    outer = a[["Cow_ID", "Day"]].merge(b[["Cow_ID", "Day"]], on=["Cow_ID", "Day"], how="outer", indicator=True)
    print(f"Rows only in file 1: {(outer['_merge'] == 'left_only').sum()} | only in file 2: {(outer['_merge'] == 'right_only').sum()}")
    both = a.merge(b, on=["Cow_ID", "Day"], how="inner", suffixes=("_1", "_2"))  # inner merge keeps dtypes
    total = 0
    for c in common:
        x, y = both[f"{c}_1"], both[f"{c}_2"]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            diff = int((~((x == y) | (x.isna() & y.isna()))).sum())
        else:
            diff = int((x.astype(str) != y.astype(str)).sum())
        if diff:
            print(f"  column {c}: {diff} differing cells")
            total += diff
    print(f"Total differing cells in the {len(both)} aligned rows: {total}")
    if total == 0 and (outer["_merge"] == "both").all():
        print("Same data; only row order or formatting differs.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    first = describe(sys.argv[1])
    if len(sys.argv) > 2:
        print()
        second = describe(sys.argv[2])
        compare(first, second)
