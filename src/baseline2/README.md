# MFW baselines

Classic Delta, Cosine Delta, linear SVM and a majority-author baseline use the same saved partitions as the Transformer experiments.

## Execution

With the environment installed as described in the [main README](../../README.md), run from the repository root:

```bash
python src/baseline2/baselines.py
```

Inputs are the twelve fold CSVs and the original texts in `data/rhythmic_corpus_100plays/`. Results are saved to `outputs/baseline_experiments/mfw_baselines/results.json`. To rerun without replacing the report:

```bash
python src/baseline2/baselines.py --output /path/new_results.json
```

Use `--corpus` for another data directory. Replacing an existing report requires `--overwrite`.

## Method

- Fragments and complete plays are evaluated separately. Complete plays inherit their fragments' folds and are classified from their full texts.
- Twelve rotations use fold `i` for test, `(i + 1) % 12` for validation and the other ten for training. Validation is reserved for every method.
- Tokenization lowercases Unicode words, retains accents and single-letter words, and excludes numbers and punctuation.
- Features are the 500 most frequent words in training, with alphabetical tie-breaking. Frequencies are divided by all words in each sample. Vocabulary selection and z-score normalization use training data only (`ddof=0`); constant features are removed.
- Classic Delta selects the nearest training sample by mean absolute difference; Cosine Delta uses cosine distance. Distance ties select the first training sample. Cosine Delta rejects zero vectors.
- Linear SVM uses one-vs-one classification. `C` is selected from `{0.1, 1, 10}` by validation accuracy; ties favor the smaller value. The selected model is not refitted on validation data.
- The majority baseline predicts the most frequent training author; ties are resolved alphabetically.

Accuracy and macro-F1 are calculated over pooled test predictions for all nine authors. The report does not contain per-fold metrics or predictions. MFW include both function and content words.

## Results and tests

The report contains global metrics, the SVM `C` values selected per fold, package versions and code/corpus hashes. Its code hash identifies the version used to produce the results, before the overwrite guard was added.

```bash
python -m unittest discover -s src/baseline2 -v
```

References: [Evert et al. (2017)](https://doi.org/10.1093/llc/fqx023) and [SVC documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.svm.SVC.html).
