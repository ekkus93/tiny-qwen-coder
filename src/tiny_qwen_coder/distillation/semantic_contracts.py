"""Independent executable semantic verification for repaired teacher-distilled records.

P9-009 deliberately keeps candidate answers out of the verifier prompt.  A second,
separately-seeded teacher generation sees only the original task and emits an
executable behavioral contract.  The contract must first validate its own reference
solution before it may grade the frozen candidate answer.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from tiny_qwen_coder.config import ExecutionConfig
from tiny_qwen_coder.data.deduplication import normalized_record_fingerprint
from tiny_qwen_coder.data.loading import load_normalized_training_records_jsonl
from tiny_qwen_coder.data.records import NormalizedTrainingRecord, SourceProvenance, TrainingMessage
from tiny_qwen_coder.distillation.finalize import _atomic_write_text, _write_records
from tiny_qwen_coder.evaluation.execution import (
    DirectExecutionHarness,
    ExecutionFile,
    ExecutionLimits,
    ExecutionRequest,
)

SEMANTIC_CONTRACT_POLICY_ID = "python-semantic-contract-v1"
SEMANTIC_CONTRACT_SEED = 2718
MIN_ASSERTIONS = 8
EXPECTED_SOURCE_MANIFEST_SHA256 = "7e299de17cbb62bff3cf5f50c61ad6ad30ca571c9297b9935ff1307cd52e0eb7"
EXPECTED_SOURCE_OUTPUT_SHA256 = "7f07f7253e98bf8ed295b12d72ebdf03c44b2e08a8d9f74aaa02dce5b760d966"
EXPECTED_ACCEPTED_RECORDS = 1557
EXPECTED_TRAIN_RECORDS = 1479
EXPECTED_VALIDATION_RECORDS = 78
DERIVED_MANIFEST_ID = "dataset/python/qwen38-27b-v4-2000-semantic-v1"

_SYSTEM_PROMPT = """You are an independent executable semantic verifier for Python coding tasks.
You do not see the candidate solution and must not guess or reconstruct it.

Given only the original user task, produce exactly one JSON object with these fields and no others:
{"schema_version":1,"verifiable":true|false,"entry_point":string|null,
 "reference_solution":string|null,"tests":string|null,"reason":string}

For verifiable=true:
- entry_point is the exact public function or class name required by the task;
- reference_solution is deterministic Python 3 using only the standard library;
- tests contains at least eight explicit Python assert statements covering normal and edge cases;
- tests must exercise the entry point and must not redefine it;
- do not use files, network, subprocesses, input(), randomness, wall-clock time, or manual judgment.

If the task cannot be checked safely and deterministically from a standalone Python function/class
contract, set verifiable=false, set entry_point/reference_solution/tests to null, and explain why.
Do not emit Markdown fences or prose outside the JSON object.
"""

_PYTHON_FENCE_RE = re.compile(
    r"```(?P<label>python|python3|py)[ \t]*\n(?P<code>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "asyncio",
        "ctypes",
        "http",
        "importlib",
        "multiprocessing",
        "os",
        "pathlib",
        "random",
        "requests",
        "secrets",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "time",
        "urllib",
    }
)
_FORBIDDEN_CALL_NAMES = frozenset(
    {"__import__", "breakpoint", "compile", "eval", "exec", "input", "open"}
)


class SemanticContractError(RuntimeError):
    """Raised when P9-009 semantic-contract evidence cannot be trusted."""


@dataclass(frozen=True, slots=True)
class SemanticContract:
    """Strict verifier output parsed from one separately generated response."""

    schema_version: int
    verifiable: bool
    entry_point: str | None
    reference_solution: str | None
    tests: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class SemanticVerificationResult:
    """Durable, content-minimized result for one candidate/contract pair."""

    schema_version: int
    index: int
    source_record_sha256: str
    candidate_response_sha256: str
    contract_response_sha256: str
    accepted: bool
    reason: str
    entry_point: str | None
    assertion_count: int
    reference_passed: bool
    candidate_passed: bool


@dataclass(frozen=True, slots=True)
class SemanticFilteredCorpusSummary:
    """Census emitted before P9-009 training can be considered."""

    schema_version: int
    policy_id: str
    policy_sha256: str
    source_manifest_sha256: str
    source_output_sha256: str
    input_records: int
    semantic_verified: int
    semantic_rejected: int
    train_records: int
    validation_records: int
    result_sha256: str
    contract_input_sha256: str
    contract_config_sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    )


def semantic_contract_policy_sha256() -> str:
    """Return the stable hash of the exact verifier system policy."""

    return _sha256_text(_SYSTEM_PROMPT)


def _assistant_response(record: NormalizedTrainingRecord) -> str:
    if not record.messages or record.messages[-1].role != "assistant":
        raise SemanticContractError("source record must end with an assistant response")
    return record.messages[-1].content


def _last_user_task(record: NormalizedTrainingRecord) -> str:
    for message in reversed(record.messages[:-1]):
        if message.role == "user" and message.content.strip():
            return message.content
    raise SemanticContractError("source record has no non-empty user task")


def prepare_semantic_contract_record(record: NormalizedTrainingRecord) -> NormalizedTrainingRecord:
    """Create one verifier input without exposing the candidate answer to the verifier."""

    fingerprint = normalized_record_fingerprint(record)
    candidate_sha256 = _sha256_text(_assistant_response(record))
    task = _last_user_task(record)
    prompt_sha256 = _canonical_sha256(
        {
            "schema_version": 1,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": task},
            ],
        }
    )
    metadata = dict(record.provenance.source_metadata)
    metadata.update(
        {
            "semantic_contract.candidate_response_sha256": candidate_sha256,
            "semantic_contract.policy_id": SEMANTIC_CONTRACT_POLICY_ID,
            "semantic_contract.policy_sha256": semantic_contract_policy_sha256(),
            "semantic_contract.prompt_sha256": prompt_sha256,
            "semantic_contract.source_record_sha256": fingerprint.record_sha256,
        }
    )
    provenance = SourceProvenance(
        source_id=record.provenance.source_id,
        revision=record.provenance.revision,
        license=record.provenance.license,
        split=record.provenance.split,
        record_id=record.provenance.record_id,
        url=record.provenance.url,
        source_metadata=tuple(sorted(metadata.items())),
    )
    return NormalizedTrainingRecord(
        schema_version=1,
        messages=(
            TrainingMessage(role="system", content=_SYSTEM_PROMPT),
            TrainingMessage(role="user", content=task),
            TrainingMessage(role="assistant", content="semantic-contract-placeholder"),
        ),
        language="python",
        provenance=provenance,
        validation=None,
    )


def prepare_semantic_contract_records(
    records: tuple[NormalizedTrainingRecord, ...],
) -> tuple[NormalizedTrainingRecord, ...]:
    """Prepare the exact candidate-hidden verifier inputs in source order."""

    if len(records) != EXPECTED_ACCEPTED_RECORDS:
        raise SemanticContractError(
            f"expected {EXPECTED_ACCEPTED_RECORDS} repaired records; got {len(records)}"
        )
    return tuple(prepare_semantic_contract_record(record) for record in records)


def _strip_transport_reasoning(text: str) -> str:
    """Handle both normal and historical closing-only Qwen reasoning transport."""

    stripped = text.strip()
    if "<think>" in stripped and "</think>" not in stripped:
        raise SemanticContractError("contract response opened <think> without closing </think>")
    if "</think>" in stripped:
        stripped = stripped.rsplit("</think>", maxsplit=1)[1].strip()
    return stripped


def parse_semantic_contract(text: str) -> SemanticContract:
    """Parse one exact JSON semantic contract, failing closed on schema drift."""

    payload_text = _strip_transport_reasoning(text)
    try:
        raw: object = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise SemanticContractError(f"semantic contract is invalid JSON: {exc}") from exc
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise SemanticContractError("semantic contract must be a JSON object")
    expected_fields = {
        "schema_version",
        "verifiable",
        "entry_point",
        "reference_solution",
        "tests",
        "reason",
    }
    if set(raw) != expected_fields:
        raise SemanticContractError("semantic contract field set drifted")
    if raw.get("schema_version") != 1:
        raise SemanticContractError("semantic contract schema_version must be 1")
    verifiable = raw.get("verifiable")
    if not isinstance(verifiable, bool):
        raise SemanticContractError("semantic contract verifiable must be boolean")
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise SemanticContractError("semantic contract reason must be non-empty")

    entry_point = raw.get("entry_point")
    reference_solution = raw.get("reference_solution")
    tests = raw.get("tests")
    if not verifiable:
        if any(value is not None for value in (entry_point, reference_solution, tests)):
            raise SemanticContractError("unverifiable contract must null executable fields")
        return SemanticContract(1, False, None, None, None, reason.strip())

    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        raise SemanticContractError("verifiable contract entry_point must be a Python identifier")
    if not isinstance(reference_solution, str) or not reference_solution.strip():
        raise SemanticContractError("verifiable contract reference_solution must be non-empty")
    if not isinstance(tests, str) or not tests.strip():
        raise SemanticContractError("verifiable contract tests must be non-empty")
    return SemanticContract(
        1,
        True,
        entry_point,
        reference_solution,
        tests,
        reason.strip(),
    )


def _forbidden_ast_reason(tree: ast.AST) -> str | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", maxsplit=1)[0] in _FORBIDDEN_IMPORT_ROOTS:
                    return f"forbidden_import:{alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", maxsplit=1)[0]
            if root in _FORBIDDEN_IMPORT_ROOTS:
                return f"forbidden_import:{node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALL_NAMES:
                return f"forbidden_call:{node.func.id}"
    return None


def validate_semantic_contract_static(contract: SemanticContract) -> tuple[bool, str, int]:
    """Validate deterministic executable structure before any untrusted execution."""

    if not contract.verifiable:
        return False, "unverifiable", 0
    assert contract.entry_point is not None
    assert contract.reference_solution is not None
    assert contract.tests is not None
    try:
        reference_tree = ast.parse(contract.reference_solution)
        tests_tree = ast.parse(contract.tests)
    except SyntaxError as exc:
        return False, f"syntax_error:{exc.lineno or 0}", 0

    reference_names = {
        node.name
        for node in reference_tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    if contract.entry_point not in reference_names:
        return False, "reference_missing_entry_point", 0
    test_definitions = {
        node.name
        for node in tests_tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    if contract.entry_point in test_definitions:
        return False, "tests_redefine_entry_point", 0
    if any(
        isinstance(node, ast.Name)
        and node.id == contract.entry_point
        and isinstance(node.ctx, ast.Store | ast.Del)
        for node in ast.walk(tests_tree)
    ):
        return False, "tests_rebind_entry_point", 0

    assertions = tuple(node for node in ast.walk(tests_tree) if isinstance(node, ast.Assert))
    assertion_count = sum(
        any(
            isinstance(child, ast.Name) and child.id == contract.entry_point
            for child in ast.walk(assertion)
        )
        for assertion in assertions
    )
    if assertion_count < MIN_ASSERTIONS:
        return False, f"too_few_entry_point_assertions:{assertion_count}", assertion_count
    for tree in (reference_tree, tests_tree):
        forbidden = _forbidden_ast_reason(tree)
        if forbidden is not None:
            return False, forbidden, assertion_count
    return True, "static_valid", assertion_count


def _entry_point_names(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def extract_candidate_code(candidate: str, *, entry_point: str) -> str:
    """Select exactly one Python block defining the independently chosen entry point."""

    fenced = tuple(match.group("code").strip() for match in _PYTHON_FENCE_RE.finditer(candidate))
    blocks = fenced if fenced else (candidate.strip(),)
    matching = tuple(block for block in blocks if entry_point in _entry_point_names(block))
    if len(matching) != 1:
        raise SemanticContractError(
            f"candidate must contain exactly one Python block defining {entry_point!r}; "
            f"found {len(matching)}"
        )
    return matching[0]


def _execution_request(code: str) -> ExecutionRequest:
    return ExecutionRequest(
        image="python:3.11-alpine3.24",
        command=(str(Path(sys.executable).resolve()), "-I", "semantic_check.py"),
        files=(ExecutionFile.from_text("semantic_check.py", code),),
    )


def _execute(code: str, *, harness: DirectExecutionHarness, execution: ExecutionConfig) -> bool:
    result = harness.run(
        _execution_request(code),
        execution,
        limits=ExecutionLimits(
            cpus=1.0,
            memory_mebibytes=512,
            pids=32,
            workspace_mebibytes=32,
            temp_mebibytes=32,
            max_output_bytes=65_536,
            max_input_bytes=524_288,
            open_files=64,
        ),
    )
    return result.succeeded


def verify_candidate_against_contract(
    candidate_record: NormalizedTrainingRecord,
    contract_response: str,
    *,
    index: int,
    harness: DirectExecutionHarness,
    execution: ExecutionConfig,
) -> SemanticVerificationResult:
    """Self-test one independent contract, then grade exactly one frozen candidate."""

    fingerprint = normalized_record_fingerprint(candidate_record)
    candidate = _assistant_response(candidate_record)
    candidate_sha256 = _sha256_text(candidate)
    contract_sha256 = _sha256_text(contract_response)
    try:
        contract = parse_semantic_contract(contract_response)
    except SemanticContractError as exc:
        return SemanticVerificationResult(
            1,
            index,
            fingerprint.record_sha256,
            candidate_sha256,
            contract_sha256,
            False,
            f"contract_parse:{exc}",
            None,
            0,
            False,
            False,
        )
    static_ok, reason, assertion_count = validate_semantic_contract_static(contract)
    if not static_ok:
        return SemanticVerificationResult(
            1,
            index,
            fingerprint.record_sha256,
            candidate_sha256,
            contract_sha256,
            False,
            reason,
            contract.entry_point,
            assertion_count,
            False,
            False,
        )
    assert contract.reference_solution is not None
    assert contract.tests is not None
    assert contract.entry_point is not None

    reference_code = contract.reference_solution.rstrip() + "\n\n" + contract.tests.lstrip()
    if not _execute(reference_code, harness=harness, execution=execution):
        return SemanticVerificationResult(
            1,
            index,
            fingerprint.record_sha256,
            candidate_sha256,
            contract_sha256,
            False,
            "reference_self_test_failed",
            contract.entry_point,
            assertion_count,
            False,
            False,
        )
    try:
        candidate_code = extract_candidate_code(candidate, entry_point=contract.entry_point)
    except SemanticContractError as exc:
        return SemanticVerificationResult(
            1,
            index,
            fingerprint.record_sha256,
            candidate_sha256,
            contract_sha256,
            False,
            f"candidate_extract:{exc}",
            contract.entry_point,
            assertion_count,
            True,
            False,
        )
    candidate_test = candidate_code.rstrip() + "\n\n" + contract.tests.lstrip()
    candidate_passed = _execute(candidate_test, harness=harness, execution=execution)
    return SemanticVerificationResult(
        1,
        index,
        fingerprint.record_sha256,
        candidate_sha256,
        contract_sha256,
        candidate_passed,
        "semantic_pass" if candidate_passed else "candidate_test_failed",
        contract.entry_point,
        assertion_count,
        True,
        candidate_passed,
    )


def _json_object(path: Path, *, context: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticContractError(f"could not read {context} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SemanticContractError(f"{context} must be an object")
    return {str(key): item for key, item in value.items()}


def _source_partition_hashes(source_dir: Path) -> tuple[set[str], set[str]]:
    train = load_normalized_training_records_jsonl(
        source_dir / "train.jsonl", expected_language="python"
    )
    validation = load_normalized_training_records_jsonl(
        source_dir / "validation.jsonl", expected_language="python"
    )
    if len(train) != EXPECTED_TRAIN_RECORDS or len(validation) != EXPECTED_VALIDATION_RECORDS:
        raise SemanticContractError("source train/validation record counts drifted")
    return (
        {normalized_record_fingerprint(record).record_sha256 for record in train},
        {normalized_record_fingerprint(record).record_sha256 for record in validation},
    )


def _membership(
    records: tuple[NormalizedTrainingRecord, ...], *, train_hashes: set[str]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for index, record in enumerate(records):
        fp = normalized_record_fingerprint(record)
        output.append(
            {
                "unique_index": index,
                "prompt_sha256": fp.prompt_sha256,
                "record_sha256": fp.record_sha256,
                "source_id": record.provenance.source_id,
                "source_record_id": record.provenance.record_id,
                "partition": "train" if fp.record_sha256 in train_hashes else "validation",
            }
        )
    return output


def _records_sha(records: tuple[NormalizedTrainingRecord, ...]) -> str:
    hashes = tuple(normalized_record_fingerprint(record).record_sha256 for record in records)
    return _canonical_sha256(hashes)


def build_semantically_filtered_corpus(
    *,
    source_dir: Path,
    contract_records: tuple[NormalizedTrainingRecord, ...],
    output_dir: Path,
    contract_input_sha256: str,
    contract_config_sha256: str,
    harness: DirectExecutionHarness | None = None,
) -> SemanticFilteredCorpusSummary:
    """Execute all P9-009 contracts and freeze a split-preserving survivor corpus."""

    if output_dir.exists():
        raise SemanticContractError(f"output directory already exists: {output_dir}")
    source_manifest_path = source_dir / "dataset-manifest.json"
    if _file_sha256(source_manifest_path) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise SemanticContractError("source repaired dataset manifest drifted")
    source_manifest = _json_object(source_manifest_path, context="source dataset manifest")
    source_identity = _json_object(
        source_dir / "source-run-identity.json", context="source identity"
    )
    if source_identity.get("output_sha256") != EXPECTED_SOURCE_OUTPUT_SHA256:
        raise SemanticContractError("source repaired output identity drifted")
    contamination = source_manifest.get("contamination")
    if not isinstance(contamination, dict) or contamination.get("status") != "clean":
        raise SemanticContractError("source contamination evidence is not clean")

    accepted = load_normalized_training_records_jsonl(
        source_dir / "accepted.jsonl", expected_language="python"
    )
    if len(accepted) != EXPECTED_ACCEPTED_RECORDS:
        raise SemanticContractError("source accepted record count drifted")
    if len(contract_records) != len(accepted):
        raise SemanticContractError("contract record count does not match source accepted corpus")

    train_hashes, validation_hashes = _source_partition_hashes(source_dir)
    if train_hashes & validation_hashes:
        raise SemanticContractError("source train/validation partitions overlap")
    accepted_hashes = {normalized_record_fingerprint(record).record_sha256 for record in accepted}
    if accepted_hashes != train_hashes | validation_hashes:
        raise SemanticContractError("source accepted corpus is not train plus validation")

    resolved_harness = harness or DirectExecutionHarness(allow_reduced_isolation=True)
    execution = ExecutionConfig(timeout_seconds=5.0, network_enabled=False)
    results: list[SemanticVerificationResult] = []
    survivors: list[NormalizedTrainingRecord] = []
    paired_records = zip(accepted, contract_records, strict=True)
    for index, (candidate, contract_record) in enumerate(paired_records):
        expected_input = prepare_semantic_contract_record(candidate)
        if contract_record.messages[:-1] != expected_input.messages[:-1]:
            raise SemanticContractError(f"contract verifier prompt drift at index {index}")
        metadata = dict(contract_record.provenance.source_metadata)
        expected_metadata = dict(expected_input.provenance.source_metadata)
        for key in (
            "semantic_contract.source_record_sha256",
            "semantic_contract.candidate_response_sha256",
            "semantic_contract.policy_id",
            "semantic_contract.policy_sha256",
            "semantic_contract.prompt_sha256",
        ):
            if metadata.get(key) != expected_metadata.get(key):
                raise SemanticContractError(
                    f"contract metadata binding drift at index {index}: {key}"
                )
        result = verify_candidate_against_contract(
            candidate,
            _assistant_response(contract_record),
            index=index,
            harness=resolved_harness,
            execution=execution,
        )
        results.append(result)
        if result.accepted:
            survivors.append(candidate)

    survivor_tuple = tuple(survivors)
    survivor_hashes = {
        normalized_record_fingerprint(record).record_sha256 for record in survivor_tuple
    }
    train = tuple(
        record
        for record in survivor_tuple
        if normalized_record_fingerprint(record).record_sha256 in train_hashes
    )
    validation = tuple(
        record
        for record in survivor_tuple
        if normalized_record_fingerprint(record).record_sha256 in validation_hashes
    )
    if not train or not validation:
        raise SemanticContractError(
            "semantic filtering produced an empty train or validation partition"
        )
    if survivor_hashes != {
        normalized_record_fingerprint(record).record_sha256 for record in (*train, *validation)
    }:
        raise SemanticContractError("filtered train/validation union does not equal survivors")

    output_dir.mkdir(parents=True)
    accepted_path = output_dir / "accepted.jsonl"
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    results_path = output_dir / "semantic-verification-results.jsonl"
    summary_path = output_dir / "semantic-verification-summary.json"
    manifest_path = output_dir / "dataset-manifest.json"
    manifest_sidecar = output_dir / "dataset-manifest.sha256"
    _write_records(accepted_path, survivor_tuple)
    _write_records(train_path, train)
    _write_records(validation_path, validation)
    result_text = "".join(
        json.dumps(asdict(result), sort_keys=True, separators=(",", ":")) + "\n"
        for result in results
    )
    _atomic_write_text(results_path, result_text)
    result_sha256 = _file_sha256(results_path)

    summary = SemanticFilteredCorpusSummary(
        schema_version=1,
        policy_id=SEMANTIC_CONTRACT_POLICY_ID,
        policy_sha256=semantic_contract_policy_sha256(),
        source_manifest_sha256=EXPECTED_SOURCE_MANIFEST_SHA256,
        source_output_sha256=EXPECTED_SOURCE_OUTPUT_SHA256,
        input_records=len(accepted),
        semantic_verified=len(survivor_tuple),
        semantic_rejected=len(accepted) - len(survivor_tuple),
        train_records=len(train),
        validation_records=len(validation),
        result_sha256=result_sha256,
        contract_input_sha256=contract_input_sha256,
        contract_config_sha256=contract_config_sha256,
    )
    _atomic_write_text(summary_path, json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n")

    memberships = _membership(survivor_tuple, train_hashes=train_hashes)
    tokenizer = source_manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        raise SemanticContractError("source dataset manifest tokenizer is invalid")
    manifest = {
        "schema_version": 1,
        "manifest_id": DERIVED_MANIFEST_ID,
        "language": "python",
        "seed": 1729,
        "tokenizer": tokenizer,
        "counts": {
            "input_records": len(accepted),
            "semantic_verified": len(survivor_tuple),
            "semantic_rejected": len(accepted) - len(survivor_tuple),
            "train_records": len(train),
            "validation_records": len(validation),
        },
        "checksums": {
            "input_records_sha256": _records_sha(accepted),
            "unique_corpus_sha256": _records_sha(survivor_tuple),
            "train_records_sha256": _records_sha(train),
            "validation_records_sha256": _records_sha(validation),
            "split_membership_sha256": _canonical_sha256(memberships),
        },
        "memberships": memberships,
        "contamination": contamination,
        "semantic_verification": {
            "policy_id": SEMANTIC_CONTRACT_POLICY_ID,
            "policy_sha256": semantic_contract_policy_sha256(),
            "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "source_output_sha256": EXPECTED_SOURCE_OUTPUT_SHA256,
            "contract_input_sha256": contract_input_sha256,
            "contract_config_sha256": contract_config_sha256,
            "minimum_assertions": MIN_ASSERTIONS,
            "result_sha256": result_sha256,
        },
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _atomic_write_text(manifest_path, manifest_text)
    manifest_sha256 = _file_sha256(manifest_path)
    _atomic_write_text(
        manifest_sidecar,
        f"{manifest_sha256}  dataset-manifest.json\n",
        encoding="ascii",
    )
    return summary


__all__ = [
    "DERIVED_MANIFEST_ID",
    "EXPECTED_ACCEPTED_RECORDS",
    "MIN_ASSERTIONS",
    "SEMANTIC_CONTRACT_POLICY_ID",
    "SEMANTIC_CONTRACT_SEED",
    "SemanticContract",
    "SemanticContractError",
    "SemanticFilteredCorpusSummary",
    "SemanticVerificationResult",
    "build_semantically_filtered_corpus",
    "extract_candidate_code",
    "parse_semantic_contract",
    "prepare_semantic_contract_record",
    "prepare_semantic_contract_records",
    "semantic_contract_policy_sha256",
    "validate_semantic_contract_static",
    "verify_candidate_against_contract",
]
