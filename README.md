# Authorship attribution in Spanish Golden Age theatre

Research repository for a study of fine-tuned contextual encoders on the Rhythmic Corpus of 100 Spanish Golden Age Plays, with a use case on the disputed play *Siempre ayuda la verdad* (SALV). The study compares BETO, mRoBERTa, MrBERT and MrBERT-es with Classic Delta, Cosine Delta, a linear SVM and a majority-author baseline.

## Repository structure

- `data/rhythmic_corpus_100plays/`: source texts for the cross-validation experiment.
- `data/crossval_12folds_mroberta_tokenized/`: shared fragments and play-level folds used across encoders.
- `data/salv_original_corpus/`: original corpus for the SALV use case.
- `data/salv_expanded_corpus/`: expanded corpus used in the SALV experiment.
- `data/salv_splits_seed44/`: training, validation, test and disputed-play CSVs.
- `src/baseline2/`: MFW-based stylometric baselines using the shared data and fold rotation.
- `outputs/baseline_experiments/mfw_baselines/results.json`: baseline results.
- `src/run_crossval.py`: encoder training and evaluation with rotating test and validation folds.
- `src/run_salv_5runs.py`: MrBERT-es SALV experiment with training seeds 1–5.
- `src/corpus_preprocessing/`: corpus fragmentation and partitioning scripts.
- `outputs/crossval_experiments/` and `outputs/salv_experiments/`: experimental results; checkpoints are excluded from Git.
- `configs/`: experiment parameters.

See [data documentation](data/README.md) for corpus provenance and preprocessing details, and [references.bib](references.bib) for references.

## Experiments

| Configuration | Model identifier | Learning rate |
| --- | --- | --- |
| [BETO](configs/beto_crossval.json) | `dccuchile/bert-base-spanish-wwm-cased` | `4e-6` |
| [mRoBERTa](configs/mroberta_crossval.json) | `BSC-LT/mRoBERTa` | `1e-5` |
| [MrBERT](configs/mrbert_crossval.json) | `BSC-LT/MrBERT` | `1e-4` |
| [MrBERT-es](configs/mrbert_es_crossval.json) | `BSC-LT/MrBERT-es` | `1e-5` |
| [MrBERT-es SALV](configs/mrbert_es_salv.json) | `BSC-LT/MrBERT-es` | `1e-5` |

Models and tokenizers are loaded by identifier. The JSON files document the parameters; scripts do not load them. Set the model, learning rate and output directory in the training script before running another configuration. Cross-validation is configured for BETO by default.

Shared training settings: up to 45 epochs, batch size 4, gradient accumulation 4, maximum sequence length 512, AdamW, gradient clipping 1.0 and a linear scheduler with 10% warmup. The best checkpoint is selected by validation accuracy, with early stopping after 10 epochs without improvement. See `configs/` for all parameters.

Cross-validation uses 12 folds and training seed 44. SALV uses fixed partitions and training seeds 1–5. Play-level Transformer predictions aggregate fragment votes or class probabilities. Vote shares for the disputed play are not probabilities of authorship.

See [data preprocessing](data/README.md) and the [baseline protocol](src/baseline2/README.md).

## Installation

[environment.yml](environment.yml) contains the Linux x86_64 environment exported on 2026-10-13. It is a snapshot of the installed environment, not a record of every historical run; installation in a clean environment has not been validated.

```bash
conda env create --name aa2-repro --file environment.yml
conda activate aa2-repro
python -m pip check
```

GPU training requires a compatible NVIDIA driver. Experiments ran through Slurm at CESGA, requesting one NVIDIA A100, 32 CPUs and 64 GB RAM per job.

## Execution

Run from the repository root:

```bash
python src/baseline2/baselines.py
python src/run_crossval.py
python src/run_salv_5runs.py
```

Preprocessing:

```bash
python src/corpus_preprocessing/generate_crossval_folds.py
python src/corpus_preprocessing/generate_salv_splits.py
```

- Use the saved partitions for the reported experiments. Regenerating cross-validation folds can change assignments because filenames are not sorted.
- Set new output directories before rerunning training. Cross-validation can overwrite existing outputs; SALV rejects a nonempty output directory. SALV preprocessing can overwrite saved CSVs. Baselines require a new `--output` path or explicit `--overwrite` when the report already exists.
- Transformer scripts evaluate test data each epoch; checkpoint selection uses validation accuracy.

## Data and license

The 100-play corpus is the original-text component of [Cuéllar's dataset, v3](https://doi.org/10.5281/zenodo.17537343), licensed CC BY 4.0. Cite the dataset and [Cuéllar (2024)](https://doi.org/10.17885/heiup.1157.c19368). SALV sources and download information are documented in [data/README.md](data/README.md); the original OSF repository does not specify a license.

Code is licensed GPL-3.0-only; see [LICENSE](LICENSE). Dataset licenses are separate. References are in [references.bib](references.bib).

## Local archive

See [archive contents](PUBLICATION.md) for packaging code, data and results without model weights.
