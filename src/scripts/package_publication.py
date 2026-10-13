"""Package code, data and results without model weights."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


ROOT = Path(__file__).resolve().parents[2]
PERSONAL_PATH = re.compile(r"/(?:home|mnt|Users|scratch|lustre)/[^\s\"',]+")
TEXT_SUFFIXES = {".py", ".md", ".json", ".csv", ".txt", ".yml", ".bib"}


def included(path):
    rel = path.relative_to(ROOT)
    if any(part.startswith(".") for part in rel.parts):
        return rel.as_posix() == ".gitignore"
    if rel.parts[0] not in {"src", "data", "configs", "outputs"}:
        return rel.as_posix() in {"README.md", "PUBLICATION.md", "LICENSE", "environment.yml", "references.bib"}
    if any(part in {"__pycache__", "best_model", "representations", "salv_original_corpus"}
           or part.startswith("checkpoint-") or part.endswith("_regenerated")
           for part in rel.parts):
        return False
    if path.suffix == ".npy":
        # Keep probability arrays; omit duplicate confusion matrices.
        return rel.parts[0] == "outputs" and "probs" in path.stem
    return path.suffix in TEXT_SUFFIXES


def publication_bytes(path):
    data = path.read_bytes()
    if path.relative_to(ROOT).parts[0] != "outputs" or path.suffix not in TEXT_SUFFIXES:
        return data
    experiment = Path(*path.relative_to(ROOT).parts[:3])

    def replace(match):
        old = match.group()
        fold = re.search(r"(?:^|/)(fold_\d\d\.csv)$", old)
        if fold:
            return "data/crossval_12folds_mroberta_tokenized/" + fold[1]
        run = re.search(r"/(fold_\d\d|seed_\d+)/(.+)$", old)
        if run:
            return (experiment / run[1] / run[2]).as_posix()
        if old.endswith("/per_class_metrics_global.csv"):
            return (experiment / "per_class_metrics_global.csv").as_posix()
        raise ValueError(f"Unmapped historical path in {path.relative_to(ROOT)}")

    return PERSONAL_PATH.sub(replace, data.decode("utf-8")).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/AA2-review.tar.gz")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Archive already exists; choose another --output.")
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and included(p))
    entries, payloads = [], []
    for path in files:
        source = path.read_bytes()
        payload = publication_bytes(path)
        if path.suffix in TEXT_SUFFIXES and PERSONAL_PATH.search(payload.decode("utf-8")):
            raise ValueError(f"Personal path remains in {path.relative_to(ROOT)}")
        if path.suffix == ".json":
            json.loads(payload)
        name = path.relative_to(ROOT).as_posix()
        entries.append({"path": name, "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "source_sha256": hashlib.sha256(source).hexdigest()})
        payloads.append((name, payload))
    manifest = json.dumps({"files": entries}, indent=2).encode() + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as archive:
        for name, payload in payloads + [("MANIFEST.json", manifest)]:
            info = tarfile.TarInfo("AA2/" + name)
            info.size, info.mode, info.mtime = len(payload), 0o644, 0
            archive.addfile(info, io.BytesIO(payload))
    print(f"{len(files)} files; {sum(len(p) for _, p in payloads):,} uncompressed bytes")
    print(f"Archive: {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
