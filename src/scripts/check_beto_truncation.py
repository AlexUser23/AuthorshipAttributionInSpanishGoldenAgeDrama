"""Check preserved cross-validation fragments against BETO's 512-token limit."""

import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH = 512


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, cache_dir=args.cache_dir, local_files_only=args.local_files_only,
    )
    folder = REPO_ROOT / "data/crossval_12folds_mroberta_tokenized"
    paths = [folder / f"fold_{i:02d}.csv" for i in range(12)]
    total, longest = 0, 0
    affected = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                ids = tokenizer.encode(
                    row["texto"], add_special_tokens=True, truncation=False,
                )
                length = len(ids)
                total += 1
                longest = max(longest, length)
                if length > MAX_LENGTH:
                    affected.append({
                        "file": path.name, "row_index": row_index,
                        "play": row["comedia"], "position": int(row["posicion"]),
                        "tokens_including_special": length,
                        "content_tokens_removed_at_512": length - MAX_LENGTH,
                    })
    report = {
        "tokenizer": MODEL,
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "corpus": str(folder.relative_to(REPO_ROOT)),
        "max_length": MAX_LENGTH,
        "special_tokens_per_sequence": tokenizer.num_special_tokens_to_add(pair=False),
        "total_fragments": total,
        "max_tokens_including_special": longest,
        "fragments_exceeding_limit": len(affected),
        "percent_exceeding_limit": 100 * len(affected) / total if total else 0,
        "all_fragments_fit": bool(total) and not affected,
        "affected_fragments": affected,
    }
    output = REPO_ROOT / "outputs/diagnostics/beto_truncation_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Fragments: {total}; exceeding 512: {len(affected)}; maximum: {longest}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
