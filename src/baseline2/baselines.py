"""Three training-only MFW baselines and a majority-author baseline."""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import warnings

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


REPO_ROOT = Path(__file__).resolve().parents[2]

N_FOLDS = 12
N_MFW = 500
C_VALUES = (0.1, 1.0, 10.0)
MODELS = ("classic_delta", "cosine_delta", "linear_svm", "majority")


def word_counts(text):
    """Lowercase Unicode words; retain accents and single-letter words."""
    counts = Counter(re.findall(r"\b[^\W\d_]+\b", text.lower()))
    if not counts:
        raise ValueError("A sample contains no word tokens.")
    return counts


@dataclass
class Sample:
    counts: Counter
    author: str
    play: str
    fold: int


def load_corpus(root):
    """Use the existing CSV partitions and original complete-play texts."""
    fragments, plays = [], []
    metadata, positions, seen_texts = {}, {}, {}
    digest = hashlib.sha256()

    def read(path):
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(data).digest())
        return data.decode("utf-8-sig")

    folder = root / "crossval_12folds_mroberta_tokenized"
    files = sorted(folder.glob("*.csv"))
    if [p.name for p in files] != [f"fold_{i:02d}.csv" for i in range(N_FOLDS)]:
        raise ValueError("Expected exactly fold_00.csv through fold_11.csv.")
    for fold, path in enumerate(files):
        # splitlines(keepends=True) preserves newlines inside quoted CSV fields.
        rows = list(csv.DictReader(read(path).splitlines(keepends=True)))
        if not rows or not {"texto", "autor", "comedia", "posicion"} <= rows[0].keys():
            raise ValueError(f"Empty fold or missing columns: {path}")
        for row in rows:
            text, author, play = row["texto"], row["autor"], row["comedia"]
            if not all(value and value.strip() for value in (text, author, play)):
                raise ValueError(f"Empty text or label: {path}")
            if metadata.setdefault(play, (author, fold)) != (author, fold):
                raise ValueError(f"Play spans authors or folds: {play}")
            if seen_texts.setdefault(text, fold) != fold:
                raise ValueError("Identical fragment text occurs in different folds.")
            positions.setdefault(play, []).append(int(row["posicion"]))
            fragments.append(Sample(word_counts(text), author, play, fold))

    paths = sorted((root / "rhythmic_corpus_100plays").glob("*.txt"))
    if {p.stem for p in paths} != set(metadata):
        raise ValueError("The complete plays and CSV play identifiers differ.")
    seen_plays = set()
    for path in paths:
        play = path.stem
        if sorted(positions[play]) != list(range(len(positions[play]))):
            raise ValueError(f"Duplicate or missing fragment positions: {play}")
        text = read(path)
        if text in seen_plays:
            raise ValueError("Duplicate complete-play text.")
        seen_plays.add(text)
        author, fold = metadata[play]
        plays.append(Sample(word_counts(text), author, play, fold))
    return {"fragment": fragments, "full_play": plays}, digest.hexdigest()


def relative_frequencies(samples, vocabulary):
    """Divide by all word tokens, including words outside the vocabulary."""
    rows = []
    for sample in samples:
        total = sample.counts.total()
        rows.append([sample.counts[word] / total for word in vocabulary])
    return np.array(rows, dtype=np.float64)


def fit_features(train, n_mfw=N_MFW):
    """Select MFW and fit population z-scores exclusively on training data."""
    total = Counter()
    for sample in train:
        total.update(sample.counts)
    # Alphabetical tie-breaking makes the vocabulary independent of row order.
    vocabulary = sorted(total, key=lambda word: (-total[word], word))[:n_mfw]
    if len(vocabulary) < n_mfw:
        raise ValueError(f"Training vocabulary has fewer than {n_mfw} words.")
    x = relative_frequencies(train, vocabulary)
    # Exact constant columns can acquire tiny nonzero std through round-off.
    keep = np.ptp(x, axis=0) > 0
    if not keep.any():
        raise ValueError("No variable MFW features remain.")
    vocabulary = [word for word, retain in zip(vocabulary, keep) if retain]
    x = x[:, keep]
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    return (x - mean) / std, (vocabulary, mean, std)


def transform(samples, fitted):
    vocabulary, mean, std = fitted
    return (relative_frequencies(samples, vocabulary) - mean) / std


def nearest_author(train, test, authors, metric):
    """Classic Delta = cityblock / feature count; Cosine Delta = 1 - cosine."""
    if metric == "cosine" and (np.any(np.linalg.norm(train, axis=1) == 0)
                               or np.any(np.linalg.norm(test, axis=1) == 0)):
        raise ValueError("Cosine Delta is undefined for a zero vector.")
    distances = cdist(test, train, metric=metric)
    if metric == "cityblock":
        distances /= train.shape[1]
    # np.argmin picks the first training row in an exact distance tie.
    return authors[distances.argmin(axis=1)]


def select_svm(train, authors, validation, validation_authors):
    """Choose C by validation accuracy; smaller C wins ties. No refitting."""
    best, best_score = None, -1.0
    for c in C_VALUES:
        model = SVC(C=c, kernel="linear", tol=1e-4, cache_size=200,
                    class_weight=None, probability=False, shrinking=True,
                    decision_function_shape="ovo", max_iter=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train, authors)
        score = accuracy_score(validation_authors, model.predict(validation))
        if score > best_score:
            best, best_score = model, score
    return best


def split_samples(samples, fold):
    """Reserve validation for every method, including Delta and majority."""
    val_fold = (fold + 1) % N_FOLDS
    return ([s for s in samples if s.fold not in (fold, val_fold)],
            [s for s in samples if s.fold == val_fold],
            [s for s in samples if s.fold == fold])


def majority_author(authors):
    """Most frequent training label; alphabetical tie-breaking."""
    counts = Counter(authors)
    return min(counts, key=lambda author: (-counts[author], author))


def evaluate(samples):
    """12 rotations: test i, validation (i+1)%12, training the other ten."""
    labels = sorted({s.author for s in samples})
    truth, predictions = [], {name: [] for name in MODELS}
    chosen_c = []
    for fold in range(N_FOLDS):
        train, val, test = split_samples(samples, fold)
        authors = np.array([s.author for s in train])
        if set(authors) != set(labels):
            raise ValueError(f"Training fold {fold} lacks an author.")
        x_train, fitted = fit_features(train)
        x_val = transform(val, fitted)
        svm = select_svm(x_train, authors, x_val, [s.author for s in val])
        chosen_c.append(svm.C)
        # Test samples enter only after the model has been selected.
        x_test = transform(test, fitted)
        predictions["classic_delta"].extend(nearest_author(x_train, x_test, authors, "cityblock"))
        predictions["cosine_delta"].extend(nearest_author(x_train, x_test, authors, "cosine"))
        predictions["linear_svm"].extend(svm.predict(x_test))
        predictions["majority"].extend([majority_author(authors)] * len(test))
        truth.extend(s.author for s in test)
        print(f"  fold {fold:02d}: finished (SVM C={svm.C:g})", flush=True)
    scores = {name: {"accuracy": accuracy_score(truth, pred),
                     "macro_f1": f1_score(truth, pred, labels=labels, average="macro", zero_division=0)}
              for name, pred in predictions.items()}
    return {"n_samples": len(truth), "n_authors": len(labels),
            "svm_c_by_test_fold": chosen_c, "scores": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "outputs/baseline_experiments/mfw_baselines/results.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing result report.")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("Output already exists; choose --output or explicitly use --overwrite.")
    datasets, corpus_hash = load_corpus(args.corpus)
    results = {}
    with threadpool_limits(limits=1):
        for level, samples in datasets.items():
            print(level, flush=True)
            results[level] = evaluate(samples)
    report = {
        "protocol": {"folds": N_FOLDS, "train_folds": 10,
                     "validation_fold": "(test_fold + 1) % 12", "mfw": N_MFW,
                     "culling": 0, "standard_deviation_ddof": 0,
                     "svm": "SVC(kernel='linear'), one-vs-one",
                     "svm_c_candidates": C_VALUES,
                     "aggregation": "pooled held-out predictions"},
        "reproducibility": {
            "python": platform.python_version(),
            "packages": {name: version(name) for name in
                         ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
            "corpus_sha256": corpus_hash,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "results": results,
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
