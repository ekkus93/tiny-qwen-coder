"""Build the canonical Python P0 input and seal a durable copy for teacher generation."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from tiny_qwen_coder.data.python_corpus_io import (
    build_canonical_python_p0,
    write_python_p0_jsonl,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="ascii")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/python/p0.yaml"),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base/qwen35-4b.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Durable accepted.jsonl path; on Colab point this at mounted Google Drive",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    result = build_canonical_python_p0(
        config_path=args.config,
        base_config=args.base_config,
        local_files_only=args.local_files_only,
    )
    output = write_python_p0_jsonl(result, args.output)
    digest = _sha256(output)
    checksum = output.with_suffix(output.suffix + ".sha256")
    _atomic_write(checksum, f"{digest}  {output.name}\n")
    if _sha256(output) != digest:
        raise RuntimeError("teacher input changed immediately after it was written")
    print(f"records={len(result.accepted_records)}")
    print(f"input={output}")
    print(f"input_sha256={digest}")
    print(f"checksum={checksum}")


if __name__ == "__main__":
    main()
