# Data

| Directory | Role | Provenance and reuse conditions |
| --- | --- | --- |
| `rhythmic_corpus_100plays/` | Rhythmic Corpus of 100 Spanish Golden Age Plays, used for encoder cross-validation | Álvaro Cuéllar, v3, original-text component; CC BY 4.0; cite the dataset and Cuéllar (2024) |
| `crossval_12folds_mroberta_tokenized/` | Shared cross-validation fragments created using the mRoBERTa tokenizer | Derived from the preceding corpus, licensed CC BY 4.0; cite the dataset and chapter |
| `salv_original_corpus/` | Original corpus for the *Siempre ayuda la verdad* use case | García-Reidy (2019), OSF; downloaded February 2026; no license specified |
| `salv_expanded_corpus/` | Expanded corpus used for the *Siempre ayuda la verdad* experiment | Original *Siempre ayuda la verdad* corpus supplemented with plays from ETSO (Cuéllar and Vega García-Luengos, 2017–2026), selected using textual integrity criteria. [12 added plays](salv_added_plays.md) |
| `salv_splits_seed44/` | Derived *Siempre ayuda la verdad* training, validation, test and disputed-play fragments | Derived from the expanded corpus |

## Sources

- **100-play corpus:** original texts from [Cuéllar's dataset, v3](https://doi.org/10.5281/zenodo.17537343), licensed CC BY 4.0. Cite the dataset and [Cuéllar (2024)](https://doi.org/10.17885/heiup.1157.c19368).
- **Original *Siempre ayuda la verdad* corpus:** 73 texts from the [OSF repository](https://osf.io/axgjc/?view_only=08d7916575e643d2a30c37f60dec422e) linked in [García-Reidy (2019)](https://doi.org/10.1007/s11061-019-09607-8), downloaded in February 2026. No license is specified in OSF.
- **Expanded *Siempre ayuda la verdad* corpus:** all 73 original texts plus [12 plays](salv_added_plays.md) from [ETSO](https://etso.es/), selected using textual integrity criteria.

Bibliographic entries are in [references.bib](../references.bib).

## Cross-validation preprocessing

All 100 plays are included, producing 5315 fragments across 12 folds. See the [play inventory](corpus_100plays_inventory.md) for saved assignments.

The generator applies NFKC normalization, unifies line endings, removes spaces around line breaks and collapses repeated horizontal whitespace. It preserves internal line breaks, case and punctuation.

Fragments use the `BSC-LT/mRoBERTa` tokenizer with up to 510 content tokens and no overlap. Text slices follow token offsets, retaining the final shorter fragment. Each encoder retokenizes the fragments and adds its own special tokens within the 512-token limit.

Whole plays are grouped by author, shuffled with seed 44 and assigned round-robin to folds before fragmentation. The generator uses unsorted filenames, so regenerating folds may change assignments. Use the saved CSVs for the experimental partitions; regenerated files go to `crossval_12folds_mroberta_tokenized_regenerated/`.

Iteration `i` uses fold `i` for test, `(i + 1) % 12` for validation and the remaining ten for training. All fragments of a play stay in the same fold.

## *Siempre ayuda la verdad* preprocessing

The generator sorts filenames, applies NFKC normalization and unifies line endings. It selects test plays first: 8 each for Alarcón, Lope and Tirso, and 3 each for Belmonte and Claramonte, capped by availability. Up to two remaining plays per author are selected for validation; the rest form the training set. Plays labelled `DUDA` form a separate disputed-play set.

Partitioning uses seed 44 and precedes fragmentation. Fragments use `BSC-LT/mRoBERTa`, at most 510 content tokens and no overlap. Use the saved CSVs for the reported experiments; this description reflects the current preprocessing script.
