# Review archive

Build the publication candidate from the repository root using only Python's
standard library:

```sh
python src/scripts/package_publication.py
```

The default destination is `dist/AA2-review.tar.gz`. Existing archives are never
overwritten; use `--output` for a new candidate. This is a local package, not an
external publication. Its `MANIFEST.json` records the path, size, SHA-256 of the
packaged bytes and SHA-256 of the source bytes for every included file.

## Included

Code, tests, experiment configurations, one shared environment specification,
license and references; the preserved cross-validation source texts and folds;
the expanded SALV corpus and preserved splits; baseline and truncation reports;
encoder summaries, per-fold/per-seed metrics, training histories, predictions,
CSV confusion matrices, fold provenance and probability arrays. These retain
the evidence needed to inspect reported results without bundling trained models.

The canonical baseline report is
`outputs/baseline_experiments/mfw_baselines/results.json`. It was moved unchanged
from `src/baseline2/resultados.json`; SHA-256:
`e5de6974f19d6d72a53e4109fc6d769bcd9c3542a2e0dd2d19caf88997b4ff57`.
The script uses this same default destination and requires `--overwrite` to
replace it. Prefer `--output` when reproducing results. Its recorded code hash
identifies the historical script, before the overwrite guard was added. Keep
the historical package versions in the report: Python 3.10.21 and SciPy 1.15.2
differ from the current shared environment snapshot. The environment name alone
does not imply identical installed versions.

The BETO diagnostic is stored in `outputs/diagnostics/beto_truncation_results.json`.
Run `python src/scripts/check_beto_truncation.py` to regenerate it; the tokenizer
uses the standard Hugging Face cache, with optional `--cache-dir` and
`--local-files-only`. Regeneration replaces this diagnostic report.

## Excluded

The local tree contains 63 model weight files totaling 42,451,988,628 bytes
(39.54 GiB), plus tokenizer files. Best models and intermediate checkpoints are
excluded from the archive and ignored by Git. Preserve them locally; if trained
weights are needed for reuse, deposit them separately and add their persistent
identifier and checksums when available.

Python caches, R session history, logs, plots and representation arrays are
excluded. CSV confusion matrices are retained instead of their duplicate NPY
versions; probability arrays are retained because they support further analysis.
Plots can be reconstructed from data where sufficient inputs are available;
embedding visualizations require the locally retained representations or a new
model run. No claim is made that every excluded visualization can be reproduced
from this archive alone.

All 73 original SALV text files are byte-identical to files in the expanded
85-play corpus. The archive includes the expanded corpus once, and omits
`data/salv_original_corpus/`. The original subset consists of the expanded
filenames minus the 12 entries in `data/salv_added_plays.md`.
The original directory remains in the local tree. The standalone baseline
requirements file duplicated the root environment and was removed, along with
the empty `.Rhistory` and Python caches.

Historical output paths are normalized **in the archive only**: original fold
CSV paths become `data/crossval_12folds_mroberta_tokenized/fold_XX.csv`, and model,
history and plot references become repository-relative experiment paths. Model
and plot references may point to excluded local artifacts. Metrics and corpus
bytes are unchanged. The manifest identifies every transformed file through its
different source and packaged hashes. Unknown personal paths cause packaging to
fail rather than being silently removed. Original local reports remain intact.

## Before submission

The archive is a review candidate. SALV redistribution conditions and editions
remain unresolved, as documented in `data/README.md`. Historical model revisions,
environment details and paper/release metadata remain unresolved in `README.md`.
This cleanup does not establish complete reproducibility or resolve those
publication requirements. Do not interpret the existence of an archive as
confirmation of dataset redistribution permissions.
