"""Materialize the canonical pre-split Python P0 training corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_qwen_coder.data.python_corpus_io import (
    materialize_canonical_python_p0,
    python_p0_summary_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/python/p0.yaml"),
        help="Python P0 composition config",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base/qwen35-4b.yaml"),
        help="Pinned base/tokenizer config",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require tokenizer files to already exist locally",
    )
    args = parser.parse_args()

    result, output_path, summary_path = materialize_canonical_python_p0(
        config_path=args.config,
        base_config=args.base_config,
        local_files_only=args.local_files_only,
    )
    print(python_p0_summary_json(result))
    print(f"records={output_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
