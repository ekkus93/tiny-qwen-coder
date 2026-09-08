"""Verify and import the frozen repaired V4-2000 corpus for local student training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NoReturn

from tiny_qwen_coder.distilled_salvage import (
    distilled_salvage_import_receipt_json,
    import_repaired_v4_2000_corpus,
)


def main(argv: list[str] | None = None) -> NoReturn:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Downloaded salvage ZIP")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/python/qwen38-27b-v4-2000-salvage-v1"),
        help="Immutable local dataset destination",
    )
    args = parser.parse_args(argv)
    receipt = import_repaired_v4_2000_corpus(args.archive, output_dir=args.output_dir)
    print(distilled_salvage_import_receipt_json(receipt), end="")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
