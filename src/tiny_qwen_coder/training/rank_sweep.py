"""P9-001 controlled Python LoRA rank-sweep protocol validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NoReturn

import yaml

_PROTOCOL_PATH = Path("configs/train/python/p9_rank_sweep_v1.yaml")
_EXPECTED_RANKS = (8, 16, 32, 64)
_BASELINE_CONFIG = "configs/train/python/p0.yaml"
EXPECTED_PROTOCOL_SHA256 = "7460583554e81b972b4cf50de964c529f9147872218ad49d987422ffc9c3b056"
EXPECTED_FIXED_PAYLOAD_SHA256 = "b83145ca896102cac467e5eeb26c2f173d5984059da5c91d736f277c9ede59c2"
_ALLOWED_TOP_LEVEL_DIFFERENCES = frozenset({"adapter_id", "output_dir"})


class RankSweepError(ValueError):
    """Raised when the P9-001 sweep stops being a rank-only experiment."""


@dataclass(frozen=True, slots=True)
class RankSweepCandidate:
    """One exact training configuration participating in the rank sweep."""

    rank: int
    config_path: str
    smoke_config_path: str | None
    baseline: bool
    config_sha256: str
    adapter_id: str
    output_dir: str


@dataclass(frozen=True, slots=True)
class RankSweepValidation:
    """Machine-readable proof that only LoRA rank varies across candidates."""

    schema_version: int
    sweep_id: str
    protocol_path: str
    protocol_sha256: str
    baseline_rank: int
    baseline_config_path: str
    candidates: tuple[RankSweepCandidate, ...]
    fixed_payload_sha256: str


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RankSweepError(f"could not read {path}") from exc


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RankSweepError(f"{context} must be a mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise RankSweepError(f"{context} keys must be strings")
        result[key] = item
    return result


def _load_yaml(path: Path, *, context: str) -> dict[str, object]:
    try:
        payload: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RankSweepError(f"could not read {context} {path}") from exc
    return _mapping(payload, context=context)


def _require_keys(
    mapping: Mapping[str, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    context: str,
) -> None:
    allowed = required | optional
    unknown = sorted(set(mapping) - allowed)
    missing = sorted(required - set(mapping))
    if unknown:
        raise RankSweepError(f"{context} contains unknown field(s): {', '.join(unknown)}")
    if missing:
        raise RankSweepError(f"{context} is missing field(s): {', '.join(missing)}")


def _string(mapping: Mapping[str, object], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RankSweepError(f"{context}.{key} must be a non-empty string")
    return value


def _positive_int(mapping: Mapping[str, object], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RankSweepError(f"{context}.{key} must be a positive integer")
    return value


def _boolean(mapping: Mapping[str, object], key: str, *, context: str) -> bool:
    value = mapping.get(key, False)
    if not isinstance(value, bool):
        raise RankSweepError(f"{context}.{key} must be a boolean")
    return value


def _candidate_rows(protocol: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    value = protocol.get("candidates")
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RankSweepError("rank sweep candidates must be a sequence")
    rows = tuple(_mapping(item, context=f"candidates[{index}]") for index, item in enumerate(value))
    if not rows:
        raise RankSweepError("rank sweep must contain candidates")
    return rows


def _normalized_fixed_payload(mapping: Mapping[str, object], *, context: str) -> dict[str, object]:
    payload = dict(mapping)
    for key in _ALLOWED_TOP_LEVEL_DIFFERENCES:
        if key not in payload:
            raise RankSweepError(f"{context} is missing required training field {key}")
        payload.pop(key)
    lora = _mapping(payload.get("lora"), context=f"{context}.lora")
    if "rank" not in lora:
        raise RankSweepError(f"{context}.lora.rank is missing")
    lora.pop("rank")
    payload["lora"] = lora
    return payload


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _validate_smoke_config(
    path: Path,
    *,
    training_config: str,
    rank: int,
) -> None:
    payload = _load_yaml(path, context=f"rank {rank} smoke config")
    _require_keys(
        payload,
        required=frozenset(
            {
                "schema_version",
                "training_config",
                "output_dir",
                "max_steps",
                "train_samples",
                "validation_samples",
            }
        ),
        context=f"rank {rank} smoke config",
    )
    if payload["schema_version"] != 1:
        raise RankSweepError(f"rank {rank} smoke config schema_version must be 1")
    if _string(payload, "training_config", context=f"rank {rank} smoke config") != training_config:
        raise RankSweepError(f"rank {rank} smoke config points at the wrong training config")
    expected_output = f"artifacts/train/python/p9-rank-r{rank}-smoke"
    if _string(payload, "output_dir", context=f"rank {rank} smoke config") != expected_output:
        raise RankSweepError(f"rank {rank} smoke output_dir is not canonical")
    expected_ints = {"max_steps": 1, "train_samples": 8, "validation_samples": 4}
    for key, expected in expected_ints.items():
        if _positive_int(payload, key, context=f"rank {rank} smoke config") != expected:
            raise RankSweepError(f"rank {rank} smoke config {key} must equal {expected}")


def validate_rank_sweep(
    protocol_path: Path = _PROTOCOL_PATH,
    *,
    repo_root: Path = Path("."),
) -> RankSweepValidation:
    """Validate the frozen P9-001 protocol and prove rank is the only training variable."""

    resolved_protocol = repo_root / protocol_path
    protocol_sha = _sha256(resolved_protocol)
    if protocol_sha != EXPECTED_PROTOCOL_SHA256:
        raise RankSweepError("rank sweep protocol SHA-256 does not match frozen P9-001 v1")
    protocol = _load_yaml(resolved_protocol, context="rank sweep protocol")
    _require_keys(
        protocol,
        required=frozenset({"schema_version", "sweep_id", "baseline_rank", "candidates"}),
        context="rank sweep protocol",
    )
    if protocol["schema_version"] != 1:
        raise RankSweepError("rank sweep protocol schema_version must be 1")
    sweep_id = _string(protocol, "sweep_id", context="rank sweep protocol")
    baseline_rank = _positive_int(protocol, "baseline_rank", context="rank sweep protocol")
    if baseline_rank != 16:
        raise RankSweepError("P9-001 baseline rank must remain 16")

    rows = _candidate_rows(protocol)
    ranks: list[int] = []
    candidate_specs: list[tuple[int, str, str | None, bool]] = []
    for index, row in enumerate(rows):
        context = f"candidates[{index}]"
        _require_keys(
            row,
            required=frozenset({"rank", "config"}),
            optional=frozenset({"smoke_config", "baseline"}),
            context=context,
        )
        rank = _positive_int(row, "rank", context=context)
        config_path = _string(row, "config", context=context)
        baseline = _boolean(row, "baseline", context=context)
        smoke_value = row.get("smoke_config")
        smoke_config: str | None
        if smoke_value is None:
            smoke_config = None
        elif isinstance(smoke_value, str) and smoke_value.strip():
            smoke_config = smoke_value
        else:
            raise RankSweepError(f"{context}.smoke_config must be a non-empty string")
        ranks.append(rank)
        candidate_specs.append((rank, config_path, smoke_config, baseline))

    if tuple(ranks) != _EXPECTED_RANKS:
        raise RankSweepError(f"P9-001 ranks must be exactly {_EXPECTED_RANKS!r} in ascending order")
    baselines = [spec for spec in candidate_specs if spec[3]]
    if len(baselines) != 1 or baselines[0][:2] != (16, _BASELINE_CONFIG):
        raise RankSweepError(
            "P9-001 must use configs/train/python/p0.yaml as the sole r=16 baseline"
        )

    baseline_mapping = _load_yaml(repo_root / _BASELINE_CONFIG, context="rank 16 baseline config")
    baseline_fixed = _normalized_fixed_payload(baseline_mapping, context="rank 16 baseline config")
    fixed_json = _canonical_json(baseline_fixed)
    fixed_sha = hashlib.sha256(fixed_json.encode("utf-8")).hexdigest()
    if fixed_sha != EXPECTED_FIXED_PAYLOAD_SHA256:
        raise RankSweepError("rank sweep fixed training payload drifted from frozen P0")

    candidates: list[RankSweepCandidate] = []
    for rank, config_path, smoke_config, baseline in candidate_specs:
        config_file = repo_root / config_path
        mapping = _load_yaml(config_file, context=f"rank {rank} training config")
        fixed = _normalized_fixed_payload(mapping, context=f"rank {rank} training config")
        if fixed != baseline_fixed:
            raise RankSweepError(f"rank {rank} changes a training variable other than LoRA rank")
        lora = _mapping(mapping["lora"], context=f"rank {rank} training config.lora")
        if _positive_int(lora, "rank", context=f"rank {rank} training config.lora") != rank:
            raise RankSweepError(f"rank {rank} config does not declare lora.rank={rank}")

        if baseline:
            expected_adapter = "language/python/p0"
            expected_output = "artifacts/train/python/p0"
            if smoke_config is not None:
                raise RankSweepError(
                    "r=16 reuses canonical P0 and must not define a new smoke config"
                )
        else:
            expected_adapter = f"language/python/p9-rank-r{rank}"
            expected_output = f"artifacts/train/python/p9-rank-r{rank}"
            if smoke_config is None:
                raise RankSweepError(f"rank {rank} candidate requires a smoke config")
            _validate_smoke_config(repo_root / smoke_config, training_config=config_path, rank=rank)

        adapter_id = _string(mapping, "adapter_id", context=f"rank {rank} training config")
        output_dir = _string(mapping, "output_dir", context=f"rank {rank} training config")
        if adapter_id != expected_adapter:
            raise RankSweepError(f"rank {rank} adapter_id is not canonical")
        if output_dir != expected_output:
            raise RankSweepError(f"rank {rank} output_dir is not canonical")
        candidates.append(
            RankSweepCandidate(
                rank=rank,
                config_path=config_path,
                smoke_config_path=smoke_config,
                baseline=baseline,
                config_sha256=_sha256(config_file),
                adapter_id=adapter_id,
                output_dir=output_dir,
            )
        )

    return RankSweepValidation(
        schema_version=1,
        sweep_id=sweep_id,
        protocol_path=str(protocol_path),
        protocol_sha256=protocol_sha,
        baseline_rank=baseline_rank,
        baseline_config_path=_BASELINE_CONFIG,
        candidates=tuple(candidates),
        fixed_payload_sha256=fixed_sha,
    )


def rank_sweep_validation_json(report: RankSweepValidation) -> str:
    """Serialize rank-sweep validation deterministically."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def rank_sweep_main(argv: Sequence[str] | None = None) -> int:
    """Validate and print the frozen P9-001 rank-sweep protocol."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=_PROTOCOL_PATH)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    report = validate_rank_sweep(args.protocol, repo_root=args.repo_root)
    print(rank_sweep_validation_json(report), end="")
    return 0


def main() -> NoReturn:
    raise SystemExit(rank_sweep_main())


if __name__ == "__main__":
    main()
