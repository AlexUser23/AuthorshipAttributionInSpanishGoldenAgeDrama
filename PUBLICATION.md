# Local archive

From the repository root:

```sh
python src/scripts/package_publication.py
```

The default output is `dist/AA2-review.tar.gz`. To use another destination,
pass `--output`. Existing archives are not overwritten.

The archive includes code, configurations, environment dependencies, references,
corpora, saved partitions, metrics and predictions. It excludes model weights,
checkpoints, caches, logs, plots and representation arrays. The original SALV
corpus is omitted because its 73 texts are also in the expanded corpus.

Absolute paths in reports are converted to repository-relative paths in the
archive. The source files are preserved. `MANIFEST.json` records file sizes
and SHA-256 hashes of source and packaged files.
