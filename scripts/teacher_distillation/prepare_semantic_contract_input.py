#!/usr/bin/env python3
"""Create and seal candidate-hidden P9-009 semantic-contract generation input."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tiny_qwen_coder.data.loading import load_normalized_training_records_jsonl
from tiny_qwen_coder.data.python_corpus_io import _write_records_jsonl
from tiny_qwen_coder.distillation.semantic_contracts import prepare_semantic_contract_records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_normalized_training_records_jsonl(args.input, expected_language="python")
    prepared = prepare_semantic_contract_records(records)
    _write_records_jsonl(prepared, args.output)
    digest = _sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n",
        encoding="ascii",
    )
    print(f"records={len(prepared)}")
    print(f"output={args.output}")
    print(f"output_sha256={digest}")


if __name__ == "__main__":
    main()
