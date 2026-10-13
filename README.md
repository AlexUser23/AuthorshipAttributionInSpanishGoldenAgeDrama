# Authorship attribution in Spanish Golden Age theatre

Research repository for a study of fine-tuned contextual encoders on the Rhythmic Corpus of 100 Spanish Golden Age Plays, with a use case on the disputed play *Siempre ayuda la verdad* (SALV). The study compares BETO, mRoBERTa, MrBERT and MrBERT-es with Classic Delta, Cosine Delta, a linear SVM and a majority-author baseline.

## Repository structure

- `data/rhythmic_corpus_100plays/`: source texts for the cross-validation experiment.
- `data/crossval_12folds_mroberta_tokenized/`: shared fragments and play-level folds used across encoders.
- `data/salv_original_corpus/`: original corpus for the SALV use case.
- `data/salv_expanded_corpus/`: expanded corpus used in the SALV experiment.
- `data/salv_splits_seed44/`: training, validation, test and disputed-play CSVs.
- `src/baseline2/`: MFW-based stylometric baselines using the shared data and fold rotation.
- `outputs/baseline_experiments/mfw_baselines/results.json`: canonical preserved baseline report and default script destination.
- `src/run_crossval.py`: encoder training and evaluation with rotating test and validation folds.
- `src/run_salv_5runs.py`: MrBERT-es SALV experiment with training seeds 1–5.
- `src/corpus_preprocessing/`: corpus fragmentation and partitioning scripts.
- `outputs/crossval_experiments/` and `outputs/salv_experiments/`: preserved experimental outputs and checkpoints.
- `configs/`: documentary experiment configurations extracted from the current scripts.

See [data documentation](data/README.md) for corpus provenance and preprocessing details, and [references.bib](references.bib) for the references supplied by the author.

## Source corpus

The source corpus is *Rhythmic Corpus of 100 Spanish Golden Age Plays*, created by Álvaro Cuéllar. We use the original-text component of version v3, available at [https://doi.org/10.5281/zenodo.17537343](https://doi.org/10.5281/zenodo.17537343).

The corpus is described in Cuéllar (2024), *Stylometry and Spanish Golden Age Theatre: An Evaluation of Authorship Attribution in a Control Group of One Hundred Undisputed Plays*: [https://doi.org/10.17885/heiup.1157.c19368](https://doi.org/10.17885/heiup.1157.c19368).

The source dataset is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Please cite both the dataset version and the chapter when using this corpus.

All 100 original-text files are included, with no exclusions. See the [play inventory and fold assignments](data/corpus_100plays_inventory.md) and [preprocessing and fold rotation](data/README.md#shared-cross-validation-fragments).

## Experiment configurations

The JSON files document the current code. They are **not loaded by the scripts** and do not independently establish the configuration of every historical run. Shared encoder hyperparameters follow the author's confirmation that only learning rates differ across cross-validation models. SALV uses the same shared training values, but has different labels, partitions and training seeds.

| Configuration | Model identifier | Learning rate |
| --- | --- | --- |
| [BETO](configs/beto_crossval.json) | `dccuchile/bert-base-spanish-wwm-cased` | `4e-6` |
| [mRoBERTa](configs/mroberta_crossval.json) | `BSC-LT/mRoBERTa` | `1e-5` |
| [MrBERT](configs/mrbert_crossval.json) | `BSC-LT/MrBERT` | `1e-4` |
| [MrBERT-es](configs/mrbert_es_crossval.json) | `BSC-LT/MrBERT-es` | `1e-5` |
| [MrBERT-es SALV](configs/mrbert_es_salv.json) | `BSC-LT/MrBERT-es` | `1e-5` |

Missing values use JSON `null`, rather than an empty numeric string. Each configuration lists them in `missing_fields`. The learning rates confirmed by the author are BETO `4e-6`, mRoBERTa `1e-5`, MrBERT `1e-4` and MrBERT-es `1e-5`, kept fixed across all twelve iterations. The exact model/tokenizer identifiers confirmed by the author are `BSC-LT/MrBERT` and `BSC-LT/MrBERT-es`. Model and tokenizer revisions are unknown for all experiments. Commented learning rates and old directory names were not treated as confirmed assignments.

Shared settings: at most 45 epochs; batch size 4; gradient accumulation 4 (nominal effective batch size 16 on one device); maximum sequence length 512; gradient clipping 1.0; linear scheduler with 10% warmup; AdamW with weight decay 0.01, except parameters whose names contain `bias` or `LayerNorm.weight` (0.0). Evaluation takes place after each epoch. The best checkpoint maximizes validation accuracy, requires a strict improvement, and is reloaded for final evaluation. Early stopping occurs after 10 epochs without improvement. Other optimizer arguments use installed PyTorch defaults.

Shared text fragments use a maximum of 510 content tokens to reserve two positions for special tokens within the 512-token sequence limit. Each encoder retokenizes the fragment and reconstructs the sequence with its own tokenizer.

Cross-validation uses training seed 44, resets it each fold, reserves fold `i` for test and `(i + 1) % n_folds` for validation, and trains on the remaining folds. Whole-play predictions support majority voting and summed probabilities. SALV uses fixed data splits with seed 44 and training seeds 1, 2, 3, 4 and 5. Fragment vote shares for the disputed play are not probabilities of authorship.

## Environment and execution status

The Transformer experiments and stylometric baselines were executed in the shared `qlora3` environment, as confirmed by the author. The preserved baseline results are available in [outputs/baseline_experiments/mfw_baselines/results.json](outputs/baseline_experiments/mfw_baselines/results.json). Dependencies for both are included in [environment.yml](environment.yml). Activate the environment before using the script entry points:

```bash
conda activate qlora3
```

The following versions were read from the currently installed `qlora3` environment on 2026-10-12, using package metadata without importing the training libraries:

| Component | Installed version |
| --- | --- |
| Python | 3.10.19 |
| torch | 2.10.0 |
| transformers | 5.0.0 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| scikit-learn | 1.7.2 |
| scipy | 1.15.3 |
| matplotlib | 3.10.8 |
| tqdm | 4.67.2 |
| umap-learn | 0.5.11 |
| tokenizers | 0.22.2 |

This is a snapshot of the current environment, not confirmation that these exact versions were installed during every historical experiment. Historical package versions, GPU/CUDA details, hardware, runtime and model revisions remain to be confirmed. A version-pinned Linux x86_64 snapshot is provided in [environment.yml](environment.yml), exported from `qlora3` on 2026-10-13 without local installation paths. It includes Conda and pip dependencies, including the installed CUDA runtime packages. Creating a fresh environment from this file has not yet been validated; historical package versions remain unconfirmed.

To create a separate environment, run from the repository root:

```bash
conda env create --name aa2-repro --file environment.yml
conda activate aa2-repro
python -m pip check
```

GPU execution also requires a compatible NVIDIA driver on the host.

### Execution resources

The experiments were submitted through Slurm at CESGA using the `qlora3` environment. The submission setup requested one NVIDIA A100 GPU, 32 CPUs and 64 GB RAM, with a wall-time limit of 5 hours 30 minutes per job. This limit is not the measured runtime. The same submission file was reused for the different experiments, as confirmed by the author.

Existing paths in the four scripts are constructed from `Path(__file__).resolve()` relative to the repository root. The scripts still keep their parameters in source code. The fragment generator writes to a separate regenerated-data directory; the training scripts continue to use the preserved experimental folds.

The script entry points are:

```bash
python src/baseline2/baselines.py
python src/run_crossval.py
python src/run_salv_5runs.py
python src/corpus_preprocessing/generate_crossval_folds.py
python src/corpus_preprocessing/generate_salv_splits.py
```

These commands describe entry points, not a verified reproduction procedure. Review the following limitations before running them:

- Cross-validation is currently configured for BETO. Supplying a JSON configuration is not supported by the unchanged script.
- The cross-validation fragment generator currently uses `BSC-LT/mRoBERTa` and chunk size 510, and writes regenerated data to `data/crossval_12folds_mroberta_tokenized_regenerated/`, keeping the preserved experimental folds separate. Its unsorted file enumeration can produce fold assignments different from the preserved corpus. The shared corpus statistics record 510 tokens per fragment, stride 0, 12 folds, 100 plays and 5315 fragments.
- Training scripts point to existing output directories. Cross-validation can write into those directories; the SALV parent process rejects a nonempty output directory. Preprocessing scripts can rewrite their destination CSVs. Preserve the archived outputs and data when preparing a future reproduction workflow.
- Both training scripts evaluate test data each epoch, while checkpoint selection in the code uses validation accuracy. This behavior should be described when reporting the experimental protocol.
- The cross-validation generator uses unsorted directory enumeration, whereas the current SALV generator sorts filenames. A fixed seed alone does not establish that regenerated historical partitions are identical; use the preserved CSVs for the recorded experiments.
- Stylometric baselines use 500 training-selected MFW, training-only population z-scores and the same 12-fold rotation. Full-play evaluation classifies complete original texts independently rather than aggregating fragment predictions. See [baseline documentation](src/baseline2/README.md) for distances, SVM selection and pooled test metrics.

No training, model download or corpus regeneration was performed to prepare these files.

## Publication, licensing and citation

See [publication package policy](PUBLICATION.md) for the minimal review archive,
checkpoint exclusions, duplicate handling and historical path normalization.
Build it with `python src/scripts/package_publication.py`. Local historical
outputs are preserved; the package includes metrics and predictions without
the approximately 40 GiB of model weights. Baseline reruns require a new
`--output` path or explicit `--overwrite` to replace the canonical report.

Paper title, author list, publication identifier and repository release/DOI are pending. Historical environment versions and experimental configurations also require confirmation before claiming full reproducibility.

The 100-play source corpus is licensed CC BY 4.0; cite both its v3 dataset release and Cuéllar (2024). The code in this repository, including the stylometric baselines and Transformer experiments, is licensed under the GNU General Public License v3.0 only (GPL-3.0-only). See [LICENSE](LICENSE) for the full license text. This code license does not replace the source datasets’ licenses. Reuse permissions for the SALV corpora remain to be established separately. See [data/README.md](data/README.md) for corpus references and conditions.
