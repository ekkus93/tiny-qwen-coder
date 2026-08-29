"""Build and freeze the canonical Python P0 dataset manifest and split artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiny_qwen_coder.data.python_corpus_io import freeze_canonical_python_p0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/python/p0.yaml"),
        help="Python P0 composition/split config",
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

    artifacts = freeze_canonical_python_p0(
        config_path=args.config,
        base_config=args.base_config,
        local_files_only=args.local_files_only,
    )
    print(f"manifest_sha256={artifacts.manifest_sha256}")
    print(f"accepted={artifacts.accepted_path}")
    print(f"train={artifacts.train_path}")
    print(f"validation={artifacts.validation_path}")
    print(f"composition={artifacts.composition_path}")
    print(f"manifest={artifacts.manifest_path}")
    print(f"manifest_checksum={artifacts.manifest_checksum_path}")
    print(f"contamination={artifacts.manifest.contamination.status.value}")


if __name__ == "__main__":
    main()
