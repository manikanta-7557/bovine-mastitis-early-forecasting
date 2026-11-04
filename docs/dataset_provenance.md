# Dataset provenance audit

**Audit date:** 2026-10-30 (local time, +05:30)
**Conclusion:** **Provenance unresolved**

No training, data-preparation, or experiment script was run. Neither existing CSV was
modified. No archive or extracted file was downloaded because the live official Version 2
record exposed no downloadable files at the time of this audit.

## Official records checked

Both record pages and Mendeley public metadata were checked on 2026-10-30:

| Version | Official record URL | DOI | Published | Public metadata |
|---|---|---|---|---|
| 1 | <https://data.mendeley.com/datasets/kbvcdw5b4m/1> | `10.17632/kbvcdw5b4m.1` | 2020-10-22 | Available; CC BY 4.0 |
| 2 | <https://data.mendeley.com/datasets/kbvcdw5b4m/2> | `10.17632/kbvcdw5b4m.2` | 2020-11-20 | Available; CC BY 4.0 |

The public metadata endpoint checked was
<https://data.mendeley.com/public-api/datasets/kbvcdw5b4m?version=2> (and the same URL
with `version=1`). It identifies the dataset, DOI, version dates, contributors and license.
It reports `size: 0` for each requested version. On the Version 2 page, the Files section
contained no listed file, the server-rendered file list was empty, and “Download All” was
disabled. No archive filename or file download URL was exposed. Thus, although the record
metadata marks Version 2 available, the archive content was not accessible from the record
during this audit.

Older project notes mention a `Clinical_Mastitis_cows_version2.zip` reportedly sized
1.48 GB. That filename and size could not be confirmed on the live official record checked
here and are not treated as verified archive metadata. No third-party copy was used.

## Version 2 archive and extracted CSV

| Requested item | Finding |
|---|---|
| Download source | Official Version 2 Mendeley record above; its current file listing offered no file |
| Archive filename | Not exposed / unverified |
| Download date | Not downloaded; official record inspected 2026-10-30 |
| Archive size and SHA-256 | Not available |
| Extracted CSV filename, size and SHA-256 | Not available |
| Byte-for-byte match to working CSV | Cannot be tested |

The repository tree was searched for a Version 2 archive, extracted Version 2 CSV, and
matching archive names; none was found. The official archive therefore could not be
obtained and its authenticity/content cannot be compared. A file with similar dimensions
or label counts would not establish provenance.

## Read-only comparison of local CSVs

Since no Version 2 file was available, the two local CSVs were compared without writing to
either:

| File | Bytes | SHA-256 | Shape | `class1` counts |
|---|---:|---|---|---|
| Working `data/raw/clinical_mastitis_cows.csv` | 416,118 | `e4166166bf83d91a63a62d1ab5e8bd66f9f7ce6687233a67abc4d99918f532d8` | 6,600 × 18 | 0: 3,961; 1: 2,639 |
| Local Version 1 `data/raw/clinical_mastitis_cows_version1.csv` | 422,719 | `11e1646bda161d234b2be4017ea66b0e1fab54f4dd56ae454eb8ffdb8b718b92` | 6,600 × 18 | 0: 5,490; 1: 1,110 |

Both files have the same 18 column names in the same order. They align one-to-one on
`Cow_ID` + `Day`: 6,600 matched rows, no rows unique to either file. This alignment is used
only to describe differences; matching schema and row keys do not authenticate a source.

| Column | Differing aligned cells |
|---|---:|
| `IUFL` | 1,505 |
| `EUFL` | 1,502 |
| `IUFR` | 1,486 |
| `EUFR` | 1,487 |
| `IURL` | 1,507 |
| `EURL` | 1,499 |
| `IURR` | 1,489 |
| `EURR` | 1,489 |
| `Temperature` | 351 |
| `Hardness` | 725 |
| `Pain` | 54 |
| `Milk_visibility` | 1,529 |
| `class1` | 1,529 |
| **Total changed cells** | **16,152** |

The label transition comparison from Version 1 to the working file is:

| Version 1 `class1` | Working `class1` | Rows |
|---:|---:|---:|
| 0 | 0 | 3,961 |
| 0 | 1 | 1,529 |
| 1 | 1 | 1,110 |
| 1 | 0 | 0 |

No cause for the differences is inferred. The comparison only establishes that the local
working CSV is not byte-identical and not content-identical to the local Version 1 CSV.
It does not establish that the working file is Version 2.

The working and Version 1 CSV sizes and hashes were recomputed before comparison and again
afterwards; both remained unchanged. No copy or extraction was created.

## Conclusion and artifact needed

**Provenance unresolved.** The official Version 1 and Version 2 records can be identified,
and their metadata was inspected. However, the current official Version 2 page exposed no
downloadable archive/file and returned a dataset size of zero. The working CSV is confirmed
to differ from the available local Version 1 CSV, but it is **not authenticated as Version
2**.

Needed to resolve this: an accessible archive/file asset from the official Version 2 record
(`<https://data.mendeley.com/datasets/kbvcdw5b4m/2>`) or confirmation from the dataset
owner/Mendeley that identifies the official Version 2 file. Record the archive source,
download date, byte size and SHA-256; extract to a separate provenance directory; then
compare the extracted CSV hash and full content with the preserved working file. Until that
evidence exists, do not label results as Version 2 results.
