# Corpus documentation

This document records information available in the scripts and the author's description. Items marked **Pending** have not been established; no source URL, license or citation is invented.

| Directory | Role | Provenance and reuse conditions |
| --- | --- | --- |
| `rhythmic_corpus_100plays/` | Rhythmic Corpus of 100 Spanish Golden Age Plays, used for encoder cross-validation | Álvaro Cuéllar, v3, original-text component; CC BY 4.0; cite the dataset and Cuéllar (2024) |
| `crossval_12folds_mroberta_tokenized/` | Shared cross-validation fragments created using the mRoBERTa tokenizer, as confirmed by the author | Derived from the preceding corpus, licensed CC BY 4.0; cite the dataset and chapter |
| `salv_original_corpus/` | Original corpus for the SALV use case | García-Reidy (2019), cited below. Pending: exact dataset version/download date and corpus reuse conditions |
| `salv_expanded_corpus/` | Expanded corpus used for the SALV experiment | Original SALV corpus supplemented with plays from ETSO (Cuéllar and Vega García-Luengos, 2017–2026), selected using textual integrity criteria. [12 added plays](salv_added_plays.md); pending: editions, dataset retrieval dates and reuse conditions |
| `salv_splits_seed44/` | Derived SALV training, validation, test and disputed-play fragments | Derived from the expanded corpus; reuse conditions pending |

## Provenance and references

The source corpus is *Rhythmic Corpus of 100 Spanish Golden Age Plays*, created by Álvaro Cuéllar. We use the original-text component of version v3, available at [https://doi.org/10.5281/zenodo.17537343](https://doi.org/10.5281/zenodo.17537343).

The corpus is described in Cuéllar (2024), *Stylometry and Spanish Golden Age Theatre: An Evaluation of Authorship Attribution in a Control Group of One Hundred Undisputed Plays*: [https://doi.org/10.17885/heiup.1157.c19368](https://doi.org/10.17885/heiup.1157.c19368).

The source dataset is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Please cite both the dataset version and the chapter when using this corpus.

The original SALV corpus comes from García-Reidy (2019), and the additional SALV plays from the ETSO project. Added plays were selected using textual integrity criteria. The original corpus contains 73 text files and the expanded corpus 85. The [inventory of the 12 additions](salv_added_plays.md) is obtained by subtracting the original filenames from the expanded filenames: six Belmonte plays and six Claramonte plays. All original filenames are retained. Per-play editions and the precise operational textual integrity criteria remain to be documented.

- Cuéllar, Álvaro (2024). “Stylometry and Spanish Golden Age Theatre: An Evaluation of Authorship Attribution in a Control Group of One Hundred Undisputed Plays.” In *Digital Stylistics in Romance Studies and Beyond*, pp. 101–117. Heidelberg: Heidelberg University Publishing. [Chapter DOI](https://doi.org/10.17885/heiup.1157.c19368).
- García-Reidy, Alejandro (2019). “Deconstructing the Authorship of Siempre ayuda la verdad: A Play by Lope de Vega?” *Neophilologus*, 103(4), 493–510. [DOI: 10.1007/s11061-019-09607-8](https://doi.org/10.1007/s11061-019-09607-8).
- Cuéllar, Álvaro, and Germán Vega García-Luengos (2017–2026). *ETSO: Estilometría aplicada al Teatro del Siglo de Oro*. [Project website](https://etso.es/). Accessed 2026-10-12, as supplied by the author; this is not an established corpus download date.

Machine-readable references are provided in [references.bib](../references.bib). The 100-play source dataset is licensed CC BY 4.0. Reuse conditions for the SALV source corpora remain pending.

## Shared cross-validation fragments

The saved `dataset_stats.txt` records 100 plays, 12 folds, 5315 fragments, chunk size 510 and stride 0. The author identifies `BSC-LT/mRoBERTa` as the tokenizer used to create these shared fragments. Fragments are constructed with a maximum of 510 content tokens, without special tokens, to reserve two positions for sequence boundary special tokens within the 512-token input limit. Each model reconstructs the input sequence using its own tokenizer and adds its special tokens.

All 100 `.txt` files in `rhythmic_corpus_100plays/` are used, with no exclusions. The [play inventory](corpus_100plays_inventory.md) lists each identifier, filename-derived title, author, original file and preserved fold assignment.

The current `src/corpus_preprocessing/generate_crossval_folds.py` reads UTF-8 text, applies NFKC Unicode normalization, converts CRLF and CR to LF, removes spaces and tabs immediately before and after line breaks, collapses repeated horizontal whitespace, and trims surrounding whitespace. It preserves internal line breaks and does not apply lowercasing or explicit punctuation removal. It uses `BSC-LT/mRoBERTa` without special tokens and divides the token offset sequence into non-overlapping windows of at most 510 tokens, retaining the corresponding slices of normalized text and the final shorter window. Each fragment stores text, author, play identifier, zero-based position and its retokenized token count.

Whole plays are assigned before fragmentation: filenames are grouped by author, each author's list is shuffled with seed 44, and plays are distributed round-robin over 12 folds. The generator uses unsorted `os.listdir()` enumeration, so the seed alone does not guarantee the preserved assignment. Use the inventory and saved fold CSVs for the experimental partition. Regenerated data are written to `crossval_12folds_mroberta_tokenized_regenerated/`; the preserved experimental folds remain in `crossval_12folds_mroberta_tokenized/`.

For each iteration `i`, `src/run_crossval.py` uses fold `i` for test, `(i + 1) % 12` for validation and the other ten folds for training. All fragments of a play stay together. Each play is tested once, used for validation once and included in training ten times.

| Iteration (zero-based) | Test fold | Validation fold | Training folds |
| --- | --- | --- | --- |
| 0 | 00 | 01 | 02, 03, 04, 05, 06, 07, 08, 09, 10, 11 |
| 1 | 01 | 02 | 00, 03, 04, 05, 06, 07, 08, 09, 10, 11 |
| 2 | 02 | 03 | 00, 01, 04, 05, 06, 07, 08, 09, 10, 11 |
| 3 | 03 | 04 | 00, 01, 02, 05, 06, 07, 08, 09, 10, 11 |
| 4 | 04 | 05 | 00, 01, 02, 03, 06, 07, 08, 09, 10, 11 |
| 5 | 05 | 06 | 00, 01, 02, 03, 04, 07, 08, 09, 10, 11 |
| 6 | 06 | 07 | 00, 01, 02, 03, 04, 05, 08, 09, 10, 11 |
| 7 | 07 | 08 | 00, 01, 02, 03, 04, 05, 06, 09, 10, 11 |
| 8 | 08 | 09 | 00, 01, 02, 03, 04, 05, 06, 07, 10, 11 |
| 9 | 09 | 10 | 00, 01, 02, 03, 04, 05, 06, 07, 08, 11 |
| 10 | 10 | 11 | 00, 01, 02, 03, 04, 05, 06, 07, 08, 09 |
| 11 | 11 | 00 | 01, 02, 03, 04, 05, 06, 07, 08, 09, 10 |

## SALV splits

The current `src/corpus_preprocessing/generate_salv_splits.py` uses `BSC-LT/mRoBERTa`, chunk size 510, no overlap and split seed 44. The maximum of 510 content tokens likewise reserves two positions for special tokens when reconstructing a sequence with a maximum length of 512 tokens. It sorts input filenames, normalizes Unicode with NFKC, and unifies line endings. It selects test plays first, then up to two validation plays per author, and assigns the remaining plays to training. Test counts are 8 for Alarcón, 3 for Belmonte, 3 for Claramonte, 8 for Lope and 8 for Tirso, capped by the number available. Plays whose filename author prefix is `DUDA` form a separate disputed-play set. Partitioning precedes fragmentation.

These statements describe the current generator, not independently verified historical execution. Training runs use seeds 1–5 over the fixed data partitions.

## Information needed for publication

For the SALV corpora, confirm dataset locations, editions/versions and download dates; document the operational textual integrity criteria and the edition of every play added to SALV; confirm the historical preprocessing configuration; and state the applicable licenses, redistribution permissions and any additional required dataset citations. Repository code is licensed GPL-3.0-only; see [LICENSE](../LICENSE). Dataset licensing is separate.
