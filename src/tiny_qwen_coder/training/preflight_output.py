"""Filesystem-safety checks for adapter-training preflight."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tiny_qwen_coder.training.plan import AdapterTrainingError, AdapterTrainingPlan


@dataclass(frozen=True, slots=True)
class OutputPreflightEvidence:
    """Validated destination for a new training run."""

    repo_root: str
    allowed_root: str
    output_dir: str


def _reject_symlink_components(path: Path, *, stop: Path) -> None:
    current = path
    while True:
        if current.exists() and current.is_symlink():
            raise AdapterTrainingError(f"training output path contains symlink component: {current}")
        if current == stop:
            return
        parent = current.parent
        if parent == current:
            raise AdapterTrainingError("training output path is not beneath repository root")
        current = parent


def verify_training_output_path(
    plan: AdapterTrainingPlan,
    *,
    repo_root: Path,
) -> OutputPreflightEvidence:
    """Require a fresh, non-symlinked destination beneath ``artifacts/train``."""

    root = repo_root.resolve()
    configured = Path(plan.config.output_dir)
    lexical = configured if configured.is_absolute() else root / configured
    _reject_symlink_components(lexical, stop=root)

    allowed = (root / "artifacts" / "train").resolve(strict=False)
    output = lexical.resolve(strict=False)
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise AdapterTrainingError(
            f"training output must be beneath {allowed}; got {output}"
        ) from exc
    if output == allowed:
        raise AdapterTrainingError("training output must be a run-specific directory")
    if output.exists():
        raise AdapterTrainingError(f"training output already exists: {output}")
    return OutputPreflightEvidence(
        repo_root=str(root),
        allowed_root=str(allowed),
        output_dir=str(output),
    )
