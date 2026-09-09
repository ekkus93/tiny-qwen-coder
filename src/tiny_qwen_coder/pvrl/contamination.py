"""Fail-closed protected-data and partition contamination controls for PVRL.

PVRL-105 extends the existing protected-benchmark registry into the executable
environment pipeline.  A clean report is content-addressed against the exact
environment material, protected examples, environment-author few-shot context,
partition inventory, and textual-overlap policy.  Reports contain only
identities, hashes, and overlap scores; protected source text is never serialized
into an admitted environment manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum

from tiny_qwen_coder.evaluation.contamination import (
    HighOverlapConfig,
    ProtectedBenchmarkExample,
    contamination_text_overlap_score,
    normalized_contamination_sha256,
)
from tiny_qwen_coder.evaluation.protected_benchmarks import ProtectedBenchmarkRegistry
from tiny_qwen_coder.pvrl.environment_manifest import (
    ContaminationEvidence,
    ContaminationStatus,
    EnvironmentManifest,
)

_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")
_LANGUAGE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_CHECKER_IDS = (
    "author-few-shot",
    "exact-content",
    "exact-task-id",
    "normalized-content",
    "partition-overlap",
    "text-overlap",
)


class PVRLContaminationError(ValueError):
    """Raised when contamination evidence cannot be produced or trusted."""


class PVRLContaminationRejectedError(PVRLContaminationError):
    """Raised when an environment with contamination findings is offered for admission."""


class TaskPartitionRole(StrEnum):
    """Scientifically distinct PVRL task-set roles from specification section 21."""

    TRAINING_CURRICULUM = "training_curriculum"
    CALIBRATION_DIAGNOSTIC = "calibration_diagnostic"
    DEVELOPMENT_EVALUATION = "development_evaluation"
    QUALIFICATION_EVALUATION = "qualification_evaluation"


@dataclass(frozen=True, slots=True)
class ContaminationTextComponent:
    """One exact text component included in contamination comparison."""

    component_id: str
    text: str

    def __post_init__(self) -> None:
        _require_exact_non_empty(self.component_id, field_name="component_id")
        if not self.text or not self.text.strip():
            raise PVRLContaminationError("contamination text must not be empty")

    @property
    def exact_sha256(self) -> str:
        """Return the exact UTF-8 byte identity of this text."""

        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def normalized_sha256(self) -> str:
        """Return the canonical normalized text identity."""

        return normalized_contamination_sha256(self.text)


@dataclass(frozen=True, slots=True)
class EnvironmentContaminationInput:
    """Exact environment text material checked before curriculum admission."""

    environment_id: str
    environment_material_sha256: str
    task_id: str
    role: TaskPartitionRole
    components: tuple[ContaminationTextComponent, ...]
    input_sha256: str

    def __post_init__(self) -> None:
        _require_exact_non_empty(self.environment_id, field_name="environment_id")
        _require_sha256(
            self.environment_material_sha256,
            field_name="environment_material_sha256",
        )
        _require_exact_non_empty(self.task_id, field_name="task_id")
        _validate_components(self.components, context="environment contamination input")
        if not any(item.component_id == "specification" for item in self.components):
            raise PVRLContaminationError(
                "environment contamination input must contain the specification component"
            )
        _require_sha256(self.input_sha256, field_name="input_sha256")
        if self.input_sha256 != _environment_input_sha256(self):
            raise PVRLContaminationError(
                "environment contamination input changed while retaining stale input_sha256"
            )


@dataclass(frozen=True, slots=True)
class AuthorFewShotExample:
    """One environment-author demonstration that must itself be protected-data clean."""

    example_id: str
    components: tuple[ContaminationTextComponent, ...]
    task_id: str | None = None

    def __post_init__(self) -> None:
        _require_exact_non_empty(self.example_id, field_name="author few-shot example_id")
        if self.task_id is not None:
            _require_exact_non_empty(self.task_id, field_name="author few-shot task_id")
        _validate_components(self.components, context=f"author few-shot {self.example_id!r}")


@dataclass(frozen=True, slots=True)
class PartitionTask:
    """One already-reserved task in a scientific PVRL partition."""

    task_id: str
    role: TaskPartitionRole
    components: tuple[ContaminationTextComponent, ...]
    environment_id: str | None = None

    def __post_init__(self) -> None:
        _require_exact_non_empty(self.task_id, field_name="partition task_id")
        if self.environment_id is not None:
            _require_exact_non_empty(self.environment_id, field_name="partition environment_id")
        _validate_components(self.components, context=f"partition task {self.task_id!r}")

    @classmethod
    def from_environment(cls, value: EnvironmentContaminationInput) -> PartitionTask:
        """Freeze a checked environment as one partition-inventory entry."""

        return cls(
            task_id=value.task_id,
            role=value.role,
            components=value.components,
            environment_id=value.environment_id,
        )


@dataclass(frozen=True, slots=True)
class PVRLContaminationFinding:
    """One protected-data or cross-partition collision without embedding source text."""

    checker_id: str
    finding_type: str
    subject_id: str
    conflicting_id: str
    subject_sha256: str | None = None
    conflicting_sha256: str | None = None
    score: float | None = None

    def __post_init__(self) -> None:
        if self.checker_id not in _CHECKER_IDS:
            raise PVRLContaminationError("finding checker_id is not a declared PVRL checker")
        if not _COMPONENT_ID_PATTERN.fullmatch(self.finding_type):
            raise PVRLContaminationError(
                "finding_type must be a stable lowercase component identifier"
            )
        _require_exact_non_empty(self.subject_id, field_name="finding subject_id")
        _require_exact_non_empty(self.conflicting_id, field_name="finding conflicting_id")
        for field_name, value in (
            ("subject_sha256", self.subject_sha256),
            ("conflicting_sha256", self.conflicting_sha256),
        ):
            if value is not None:
                _require_sha256(value, field_name=field_name)
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise PVRLContaminationError("contamination finding score must be between zero and one")


@dataclass(frozen=True, slots=True)
class PVRLContaminationReport:
    """Content-addressed PVRL-105 contamination evidence for one environment."""

    schema_version: int
    environment_id: str
    environment_material_sha256: str
    language: str
    candidate_task_id: str
    candidate_role: TaskPartitionRole
    candidate_input_sha256: str
    protected_registry_sha256: str
    protected_inventory_sha256: str
    author_few_shot_inventory_sha256: str
    partition_inventory_sha256: str
    overlap_threshold: float
    overlap_shingle_size: int
    overlap_min_tokens: int
    checker_ids: tuple[str, ...]
    status: ContaminationStatus
    findings: tuple[PVRLContaminationFinding, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise PVRLContaminationError(
                f"unsupported contamination schema_version {self.schema_version}; "
                f"expected {_SCHEMA_VERSION}"
            )
        _require_exact_non_empty(self.environment_id, field_name="environment_id")
        if not _LANGUAGE_ID_PATTERN.fullmatch(self.language):
            raise PVRLContaminationError("contamination language must be a stable language ID")
        _require_exact_non_empty(self.candidate_task_id, field_name="candidate_task_id")
        for field_name, value in (
            ("environment_material_sha256", self.environment_material_sha256),
            ("candidate_input_sha256", self.candidate_input_sha256),
            ("protected_registry_sha256", self.protected_registry_sha256),
            ("protected_inventory_sha256", self.protected_inventory_sha256),
            ("author_few_shot_inventory_sha256", self.author_few_shot_inventory_sha256),
            ("partition_inventory_sha256", self.partition_inventory_sha256),
            ("evidence_sha256", self.evidence_sha256),
        ):
            _require_sha256(value, field_name=field_name)
        HighOverlapConfig(
            threshold=self.overlap_threshold,
            shingle_size=self.overlap_shingle_size,
            min_tokens=self.overlap_min_tokens,
        )
        if self.checker_ids != _CHECKER_IDS:
            raise PVRLContaminationError(
                "contamination report must declare the complete frozen PVRL-105 checker set"
            )
        ordered = tuple(sorted(self.findings, key=_finding_sort_key))
        if ordered != self.findings:
            raise PVRLContaminationError("contamination findings must be deterministically sorted")
        if self.status is ContaminationStatus.NOT_RUN:
            raise PVRLContaminationError("a completed PVRL-105 report cannot have not_run status")
        if self.status is ContaminationStatus.CLEAN and self.findings:
            raise PVRLContaminationError("clean contamination report cannot contain findings")
        if self.status is ContaminationStatus.FINDINGS and not self.findings:
            raise PVRLContaminationError("findings contamination report requires findings")
        if self.evidence_sha256 != _report_sha256(self):
            raise PVRLContaminationError(
                "contamination report changed while retaining stale evidence_sha256"
            )


def _require_exact_non_empty(value: str, *, field_name: str) -> None:
    if not value or not value.strip():
        raise PVRLContaminationError(f"{field_name} must not be empty")
    if value != value.strip():
        raise PVRLContaminationError(
            f"{field_name} must not contain leading or trailing whitespace"
        )


def _require_sha256(value: str, *, field_name: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise PVRLContaminationError(f"{field_name} must be a lowercase SHA-256 digest")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
        "ascii"
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_components(
    components: tuple[ContaminationTextComponent, ...],
    *,
    context: str,
) -> None:
    if not components:
        raise PVRLContaminationError(f"{context} must contain text components")
    ids = tuple(item.component_id for item in components)
    if tuple(sorted(ids)) != ids:
        raise PVRLContaminationError(f"{context} components must be sorted by component_id")
    if len(ids) != len(set(ids)):
        raise PVRLContaminationError(f"{context} component IDs must be unique")


def _component_payload(component: ContaminationTextComponent) -> dict[str, str]:
    return {
        "component_id": component.component_id,
        "exact_sha256": component.exact_sha256,
        "normalized_sha256": component.normalized_sha256,
    }


def _environment_input_payload_fields(
    *,
    environment_id: str,
    environment_material_sha256: str,
    task_id: str,
    role: TaskPartitionRole,
    components: Sequence[ContaminationTextComponent],
) -> dict[str, object]:
    return {
        "environment_id": environment_id,
        "environment_material_sha256": environment_material_sha256,
        "task_id": task_id,
        "role": role,
        "components": [_component_payload(item) for item in components],
    }


def _environment_input_payload(value: EnvironmentContaminationInput) -> dict[str, object]:
    return _environment_input_payload_fields(
        environment_id=value.environment_id,
        environment_material_sha256=value.environment_material_sha256,
        task_id=value.task_id,
        role=value.role,
        components=value.components,
    )


def _environment_input_sha256(value: EnvironmentContaminationInput) -> str:
    return _sha256(_environment_input_payload(value))


def create_environment_contamination_input(
    manifest: EnvironmentManifest,
    *,
    task_id: str,
    role: TaskPartitionRole,
    specification: str,
    artifact_contents: Mapping[str, bytes],
) -> EnvironmentContaminationInput:
    """Bind exact UTF-8 environment material to the PVRL contamination checker.

    Every manifest artifact is required.  This intentionally scans public
    workspace, hidden grader, hidden reference, and support text so protected
    prompt/test/solution material cannot be hidden in a non-specification file.
    Self-contained PVRL v0 is text-only; non-UTF-8 artifacts fail closed rather
    than receiving an unjustified clean contamination status.
    """

    _require_exact_non_empty(task_id, field_name="task_id")
    if not specification or not specification.strip():
        raise PVRLContaminationError("environment specification must not be empty")
    specification_sha256 = hashlib.sha256(specification.encode("utf-8")).hexdigest()
    if specification_sha256 != manifest.specification_sha256:
        raise PVRLContaminationError(
            "environment specification bytes do not match manifest specification_sha256"
        )

    expected_paths = tuple(item.path for item in manifest.artifacts)
    supplied_paths = tuple(sorted(artifact_contents))
    if supplied_paths != expected_paths:
        raise PVRLContaminationError(
            "contamination input must provide exact bytes for every manifest artifact"
        )
    manifest_by_path = {item.path: item for item in manifest.artifacts}
    components: list[ContaminationTextComponent] = [
        ContaminationTextComponent(component_id="specification", text=specification)
    ]
    for path in supplied_paths:
        content = artifact_contents[path]
        if not isinstance(content, bytes):
            raise PVRLContaminationError(f"artifact {path!r} contamination content must be bytes")
        if hashlib.sha256(content).hexdigest() != manifest_by_path[path].sha256:
            raise PVRLContaminationError(f"artifact {path!r} bytes do not match manifest SHA-256")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PVRLContaminationError(
                f"artifact {path!r} is not UTF-8 text and cannot receive v0 contamination clearance"
            ) from exc
        components.append(ContaminationTextComponent(component_id=f"artifact:{path}", text=text))

    ordered_components = tuple(sorted(components, key=lambda item: item.component_id))
    input_sha256 = _sha256(
        _environment_input_payload_fields(
            environment_id=manifest.environment_id,
            environment_material_sha256=manifest.material_sha256,
            task_id=task_id,
            role=role,
            components=ordered_components,
        )
    )
    return EnvironmentContaminationInput(
        environment_id=manifest.environment_id,
        environment_material_sha256=manifest.material_sha256,
        task_id=task_id,
        role=role,
        components=ordered_components,
        input_sha256=input_sha256,
    )


def _author_inventory_payload(examples: Sequence[AuthorFewShotExample]) -> list[dict[str, object]]:
    return [
        {
            "example_id": example.example_id,
            "task_id": example.task_id,
            "components": [_component_payload(item) for item in example.components],
        }
        for example in examples
    ]


def _partition_inventory_payload(entries: Sequence[PartitionTask]) -> list[dict[str, object]]:
    return [
        {
            "task_id": entry.task_id,
            "role": entry.role,
            "environment_id": entry.environment_id,
            "components": [_component_payload(item) for item in entry.components],
        }
        for entry in entries
    ]


def _protected_prompt_text(example: ProtectedBenchmarkExample) -> str:
    return "\n".join(f"<{message.role}>\n{message.content}" for message in example.prompt_messages)


def _protected_components(
    example: ProtectedBenchmarkExample,
) -> tuple[ContaminationTextComponent, ...]:
    components = [
        ContaminationTextComponent(component_id="prompt", text=_protected_prompt_text(example)),
        *(
            ContaminationTextComponent(
                component_id=f"prompt-message:{index:04d}:{message.role}",
                text=message.content,
            )
            for index, message in enumerate(example.prompt_messages)
        ),
    ]
    if example.solution is not None:
        components.append(
            ContaminationTextComponent(component_id="solution", text=example.solution)
        )
    components.extend(
        ContaminationTextComponent(component_id=f"test:{index:04d}", text=text)
        for index, text in enumerate(example.test_texts)
    )
    return tuple(sorted(components, key=lambda item: item.component_id))


def _prepare_protected_inventory(
    registry: ProtectedBenchmarkRegistry,
    examples: Sequence[ProtectedBenchmarkExample],
    *,
    language: str,
) -> tuple[tuple[ProtectedBenchmarkExample, tuple[ContaminationTextComponent, ...]], ...]:
    benchmarks = registry.list_benchmarks(language=language)
    if not benchmarks:
        raise PVRLContaminationError(
            f"no protected benchmarks are registered for contamination language {language!r}"
        )
    expected_ids = {item.id for item in benchmarks}
    identities: set[tuple[str, str]] = set()
    provided_ids: set[str] = set()
    prepared: list[tuple[ProtectedBenchmarkExample, tuple[ContaminationTextComponent, ...]]] = []
    for example in examples:
        if example.language != language:
            raise PVRLContaminationError(
                f"protected example {example.record_id!r} uses language {example.language!r}; "
                f"expected {language!r}"
            )
        try:
            benchmark = registry.resolve(language, example.benchmark_id)
        except ValueError as exc:
            raise PVRLContaminationError(
                f"protected example {example.record_id!r} references an unregistered benchmark"
            ) from exc
        if (
            example.dataset_id != benchmark.dataset_id
            or example.dataset_revision != benchmark.dataset_revision
        ):
            raise PVRLContaminationError(
                f"protected example {example.record_id!r} does not match registered dataset identity"
            )
        identity = (example.benchmark_id, example.record_id)
        if identity in identities:
            raise PVRLContaminationError(
                f"duplicate protected example identity {example.benchmark_id}/{example.record_id}"
            )
        identities.add(identity)
        provided_ids.add(example.benchmark_id)
        prepared.append((example, _protected_components(example)))
    missing = tuple(sorted(expected_ids - provided_ids))
    unexpected = tuple(sorted(provided_ids - expected_ids))
    if missing or unexpected:
        raise PVRLContaminationError(
            "protected example coverage does not match registered benchmarks; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return tuple(sorted(prepared, key=lambda item: (item[0].benchmark_id, item[0].record_id)))


def _protected_registry_sha256(
    registry: ProtectedBenchmarkRegistry,
    *,
    language: str,
) -> str:
    return _sha256(
        [
            {
                "language": item.language,
                "id": item.id,
                "dataset_id": item.dataset_id,
                "dataset_revision": item.dataset_revision,
                "source_configs": item.source_configs,
            }
            for item in registry.list_benchmarks(language=language)
        ]
    )


def _protected_inventory_sha256(
    prepared: Sequence[tuple[ProtectedBenchmarkExample, tuple[ContaminationTextComponent, ...]]],
) -> str:
    return _sha256(
        [
            {
                "benchmark_id": example.benchmark_id,
                "dataset_id": example.dataset_id,
                "dataset_revision": example.dataset_revision,
                "record_id": example.record_id,
                "components": [_component_payload(item) for item in components],
            }
            for example, components in prepared
        ]
    )


def _finding_sort_key(
    item: PVRLContaminationFinding,
) -> tuple[str, str, str, str, str, str, float]:
    return (
        item.checker_id,
        item.finding_type,
        item.subject_id,
        item.conflicting_id,
        item.subject_sha256 or "",
        item.conflicting_sha256 or "",
        item.score if item.score is not None else -1.0,
    )


def _append_text_collision_findings(
    findings: list[PVRLContaminationFinding],
    *,
    checker_id: str,
    subject_prefix: str,
    subject_components: Sequence[ContaminationTextComponent],
    conflicting_prefix: str,
    conflicting_components: Sequence[ContaminationTextComponent],
    overlap: HighOverlapConfig,
) -> None:
    for subject in subject_components:
        for conflicting in conflicting_components:
            subject_id = f"{subject_prefix}:{subject.component_id}"
            conflicting_id = f"{conflicting_prefix}:{conflicting.component_id}"
            if subject.exact_sha256 == conflicting.exact_sha256:
                findings.append(
                    PVRLContaminationFinding(
                        checker_id=checker_id
                        if checker_id in {"author-few-shot", "partition-overlap"}
                        else "exact-content",
                        finding_type="exact-content",
                        subject_id=subject_id,
                        conflicting_id=conflicting_id,
                        subject_sha256=subject.exact_sha256,
                        conflicting_sha256=conflicting.exact_sha256,
                        score=1.0,
                    )
                )
                continue
            if subject.normalized_sha256 == conflicting.normalized_sha256:
                findings.append(
                    PVRLContaminationFinding(
                        checker_id=checker_id
                        if checker_id in {"author-few-shot", "partition-overlap"}
                        else "normalized-content",
                        finding_type="normalized-content",
                        subject_id=subject_id,
                        conflicting_id=conflicting_id,
                        subject_sha256=subject.normalized_sha256,
                        conflicting_sha256=conflicting.normalized_sha256,
                        score=1.0,
                    )
                )
                continue
            score = contamination_text_overlap_score(
                subject.text, conflicting.text, overlap=overlap
            )
            if score >= overlap.threshold:
                findings.append(
                    PVRLContaminationFinding(
                        checker_id=checker_id
                        if checker_id in {"author-few-shot", "partition-overlap"}
                        else "text-overlap",
                        finding_type="text-overlap",
                        subject_id=subject_id,
                        conflicting_id=conflicting_id,
                        subject_sha256=subject.normalized_sha256,
                        conflicting_sha256=conflicting.normalized_sha256,
                        score=round(score, 6),
                    )
                )


def _append_protected_findings(
    findings: list[PVRLContaminationFinding],
    *,
    subject_prefix: str,
    task_id: str | None,
    components: Sequence[ContaminationTextComponent],
    protected: Sequence[tuple[ProtectedBenchmarkExample, tuple[ContaminationTextComponent, ...]]],
    overlap: HighOverlapConfig,
    category_checker: str | None = None,
) -> None:
    for example, protected_components in protected:
        protected_prefix = f"protected:{example.benchmark_id}:{example.record_id}"
        if task_id is not None and task_id == example.record_id:
            findings.append(
                PVRLContaminationFinding(
                    checker_id=category_checker or "exact-task-id",
                    finding_type="exact-task-id",
                    subject_id=f"{subject_prefix}:task-id",
                    conflicting_id=f"{protected_prefix}:task-id",
                )
            )
        _append_text_collision_findings(
            findings,
            checker_id=category_checker or "protected",
            subject_prefix=subject_prefix,
            subject_components=components,
            conflicting_prefix=protected_prefix,
            conflicting_components=protected_components,
            overlap=overlap,
        )


def _validate_author_examples(
    values: Iterable[AuthorFewShotExample],
) -> tuple[AuthorFewShotExample, ...]:
    ordered = tuple(sorted(values, key=lambda item: item.example_id))
    ids = tuple(item.example_id for item in ordered)
    if len(ids) != len(set(ids)):
        raise PVRLContaminationError("author few-shot example IDs must be unique")
    return ordered


def _validate_partition_entries(values: Iterable[PartitionTask]) -> tuple[PartitionTask, ...]:
    ordered = tuple(
        sorted(
            values,
            key=lambda item: (item.role.value, item.task_id, item.environment_id or ""),
        )
    )
    identities = tuple((item.role, item.task_id, item.environment_id) for item in ordered)
    if len(identities) != len(set(identities)):
        raise PVRLContaminationError("partition inventory contains duplicate task identities")
    return ordered


def _append_partition_pair_findings(
    findings: list[PVRLContaminationFinding],
    left: PartitionTask,
    right: PartitionTask,
    *,
    overlap: HighOverlapConfig,
) -> None:
    left_prefix = f"partition:{left.role.value}:{left.task_id}"
    right_prefix = f"partition:{right.role.value}:{right.task_id}"
    if (
        left.environment_id is not None
        and right.environment_id is not None
        and left.environment_id == right.environment_id
    ):
        findings.append(
            PVRLContaminationFinding(
                checker_id="partition-overlap",
                finding_type=(
                    "environment-role-conflict"
                    if left.role is not right.role
                    else "environment-task-conflict"
                ),
                subject_id=f"{left_prefix}:environment-id",
                conflicting_id=f"{right_prefix}:environment-id",
            )
        )
    if left.role is right.role:
        return
    if left.task_id == right.task_id:
        findings.append(
            PVRLContaminationFinding(
                checker_id="partition-overlap",
                finding_type="exact-task-id",
                subject_id=f"{left_prefix}:task-id",
                conflicting_id=f"{right_prefix}:task-id",
            )
        )
    _append_text_collision_findings(
        findings,
        checker_id="partition-overlap",
        subject_prefix=left_prefix,
        subject_components=left.components,
        conflicting_prefix=right_prefix,
        conflicting_components=right.components,
        overlap=overlap,
    )


def _append_author_partition_findings(
    findings: list[PVRLContaminationFinding],
    *,
    task_id: str,
    role: TaskPartitionRole,
    components: Sequence[ContaminationTextComponent],
    author: Sequence[AuthorFewShotExample],
    overlap: HighOverlapConfig,
) -> None:
    if role not in {
        TaskPartitionRole.DEVELOPMENT_EVALUATION,
        TaskPartitionRole.QUALIFICATION_EVALUATION,
    }:
        return
    subject_prefix = f"partition:{role.value}:{task_id}"
    for example in author:
        conflicting_prefix = f"author-few-shot:{example.example_id}"
        if example.task_id is not None and task_id == example.task_id:
            findings.append(
                PVRLContaminationFinding(
                    checker_id="partition-overlap",
                    finding_type="exact-task-id",
                    subject_id=f"{subject_prefix}:task-id",
                    conflicting_id=f"{conflicting_prefix}:task-id",
                )
            )
        _append_text_collision_findings(
            findings,
            checker_id="partition-overlap",
            subject_prefix=subject_prefix,
            subject_components=components,
            conflicting_prefix=conflicting_prefix,
            conflicting_components=example.components,
            overlap=overlap,
        )


def check_environment_contamination(
    candidate: EnvironmentContaminationInput,
    protected_examples: Sequence[ProtectedBenchmarkExample],
    *,
    manifest: EnvironmentManifest,
    language: str,
    registry: ProtectedBenchmarkRegistry,
    author_few_shots: Iterable[AuthorFewShotExample],
    partition_entries: Iterable[PartitionTask],
    overlap: HighOverlapConfig | None = None,
) -> PVRLContaminationReport:
    """Run the complete PVRL-105 protected-data and freshness audit.

    The audit checks the candidate, every author few-shot, and every supplied
    scientific partition against the protected registry.  Both inventories are
    explicit required inputs, including when intentionally empty, so callers
    cannot obtain a clean result by silently accepting omitted defaults. It then checks all
    cross-role partition pairs and the candidate against each reserved task.
    Development/qualification candidates are additionally required to remain
    disjoint from author demonstrations.  Any finding blocks clean admission.
    """

    if candidate.environment_id != manifest.environment_id:
        raise PVRLContaminationError("candidate contamination input targets another environment")
    if candidate.environment_material_sha256 != manifest.material_sha256:
        raise PVRLContaminationError("candidate contamination input has stale environment material")
    if manifest.contamination.status is not ContaminationStatus.NOT_RUN:
        raise PVRLContaminationError(
            "contamination checks cannot be rerun over a manifest with completed evidence"
        )
    selected_overlap = overlap or HighOverlapConfig()
    author = _validate_author_examples(author_few_shots)
    partitions = _validate_partition_entries(partition_entries)
    protected = _prepare_protected_inventory(registry, protected_examples, language=language)

    findings: list[PVRLContaminationFinding] = []
    _append_protected_findings(
        findings,
        subject_prefix=f"candidate:{candidate.task_id}",
        task_id=candidate.task_id,
        components=candidate.components,
        protected=protected,
        overlap=selected_overlap,
    )

    for example in author:
        _append_protected_findings(
            findings,
            subject_prefix=f"author-few-shot:{example.example_id}",
            task_id=example.task_id,
            components=example.components,
            protected=protected,
            overlap=selected_overlap,
            category_checker="author-few-shot",
        )

    for entry in partitions:
        _append_protected_findings(
            findings,
            subject_prefix=f"partition:{entry.role.value}:{entry.task_id}",
            task_id=entry.task_id,
            components=entry.components,
            protected=protected,
            overlap=selected_overlap,
            category_checker="partition-overlap",
        )

    for left_index, left in enumerate(partitions):
        for right in partitions[left_index + 1 :]:
            _append_partition_pair_findings(findings, left, right, overlap=selected_overlap)

    candidate_partition = PartitionTask.from_environment(candidate)
    for entry in partitions:
        if (
            entry.environment_id == candidate.environment_id
            and entry.role is candidate.role
            and entry.task_id == candidate.task_id
        ):
            continue
        _append_partition_pair_findings(
            findings,
            candidate_partition,
            entry,
            overlap=selected_overlap,
        )

    for entry in partitions:
        _append_author_partition_findings(
            findings,
            task_id=entry.task_id,
            role=entry.role,
            components=entry.components,
            author=author,
            overlap=selected_overlap,
        )
    _append_author_partition_findings(
        findings,
        task_id=candidate.task_id,
        role=candidate.role,
        components=candidate.components,
        author=author,
        overlap=selected_overlap,
    )

    ordered_findings = tuple(sorted(set(findings), key=_finding_sort_key))
    status = ContaminationStatus.FINDINGS if ordered_findings else ContaminationStatus.CLEAN
    protected_registry_sha256 = _protected_registry_sha256(registry, language=language)
    protected_inventory_sha256 = _protected_inventory_sha256(protected)
    author_few_shot_inventory_sha256 = _sha256(_author_inventory_payload(author))
    partition_inventory_sha256 = _sha256(_partition_inventory_payload(partitions))
    fields: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "environment_id": manifest.environment_id,
        "environment_material_sha256": manifest.material_sha256,
        "language": language,
        "candidate_task_id": candidate.task_id,
        "candidate_role": candidate.role,
        "candidate_input_sha256": candidate.input_sha256,
        "protected_registry_sha256": protected_registry_sha256,
        "protected_inventory_sha256": protected_inventory_sha256,
        "author_few_shot_inventory_sha256": author_few_shot_inventory_sha256,
        "partition_inventory_sha256": partition_inventory_sha256,
        "overlap_threshold": selected_overlap.threshold,
        "overlap_shingle_size": selected_overlap.shingle_size,
        "overlap_min_tokens": selected_overlap.min_tokens,
        "checker_ids": _CHECKER_IDS,
        "status": status,
        "findings": [asdict(item) for item in ordered_findings],
    }
    evidence_sha256 = _sha256(fields)
    return PVRLContaminationReport(
        schema_version=_SCHEMA_VERSION,
        environment_id=manifest.environment_id,
        environment_material_sha256=manifest.material_sha256,
        language=language,
        candidate_task_id=candidate.task_id,
        candidate_role=candidate.role,
        candidate_input_sha256=candidate.input_sha256,
        protected_registry_sha256=protected_registry_sha256,
        protected_inventory_sha256=protected_inventory_sha256,
        author_few_shot_inventory_sha256=author_few_shot_inventory_sha256,
        partition_inventory_sha256=partition_inventory_sha256,
        overlap_threshold=selected_overlap.threshold,
        overlap_shingle_size=selected_overlap.shingle_size,
        overlap_min_tokens=selected_overlap.min_tokens,
        checker_ids=_CHECKER_IDS,
        status=status,
        findings=ordered_findings,
        evidence_sha256=evidence_sha256,
    )


def _report_payload(report: PVRLContaminationReport) -> dict[str, object]:
    payload = asdict(report)
    payload.pop("evidence_sha256")
    return payload


def _report_sha256(report: PVRLContaminationReport) -> str:
    return _sha256(_report_payload(report))


def contamination_report_json(report: PVRLContaminationReport) -> str:
    """Serialize contamination evidence deterministically without protected text."""

    return json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"


def admit_environment_contamination(
    manifest: EnvironmentManifest,
    report: PVRLContaminationReport,
) -> EnvironmentManifest:
    """Attach only clean PVRL-105 evidence to an otherwise immutable environment."""

    if manifest.contamination.status is not ContaminationStatus.NOT_RUN:
        raise PVRLContaminationError("completed contamination evidence cannot be overwritten")
    if report.environment_id != manifest.environment_id:
        raise PVRLContaminationError("contamination report targets another environment_id")
    if report.environment_material_sha256 != manifest.material_sha256:
        raise PVRLContaminationError("contamination report targets stale environment material")
    if report.status is not ContaminationStatus.CLEAN:
        raise PVRLContaminationRejectedError(
            "environment cannot be admitted while contamination findings are present"
        )
    return replace(
        manifest,
        contamination=ContaminationEvidence(
            status=ContaminationStatus.CLEAN,
            checker_ids=report.checker_ids,
            evidence_sha256=report.evidence_sha256,
        ),
    )


__all__ = [
    "AuthorFewShotExample",
    "ContaminationTextComponent",
    "EnvironmentContaminationInput",
    "PVRLContaminationError",
    "PVRLContaminationFinding",
    "PVRLContaminationRejectedError",
    "PVRLContaminationReport",
    "PartitionTask",
    "TaskPartitionRole",
    "admit_environment_contamination",
    "check_environment_contamination",
    "contamination_report_json",
    "create_environment_contamination_input",
]
