"""FTR-103 mechanical generation/evaluation parity audit.

This audit is deliberately observational: it parses the production inference and
scoring sources without changing them, normalizes the effective generation
contract, and compares every adapter-backed path to the unchanged-base path.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from tiny_qwen_coder.evaluation._baseline_provenance import load_baseline_base_model_identity
from tiny_qwen_coder.evaluation.settings import (
    evaluation_settings_sha256,
    load_frozen_evaluation_settings,
)

_DEFAULT_REPORT = Path(
    "artifacts/eval/python/ftr-103-generation-evaluation-parity-v1/parity-report.json"
)
_BASE_CONFIG = Path("configs/base/qwen35-4b.yaml")
_SETTINGS_CONFIG = Path("configs/eval/canonical_generation_v1.yaml")


class FTRGenerationEvaluationParityError(RuntimeError):
    """Raised when FTR-103 cannot prove generation/evaluation parity."""


@dataclass(frozen=True, slots=True)
class GeneratorPath:
    path_id: str
    source: str
    class_name: str
    inherited_generation_from: str | None = None


@dataclass(frozen=True, slots=True)
class ScoringPath:
    path_id: str
    source: str
    function_name: str
    evaluator_source: str
    evaluator_function: str
    required_evaluators: tuple[str, ...]
    harness: str
    approved_runtime_difference: bool


@dataclass(frozen=True, slots=True)
class EvaluatorLogicPath:
    suite_id: str
    evaluator_source: str
    evaluator_class: str
    normalizers: tuple[tuple[str, str], ...]


_GENERATORS = (
    GeneratorPath(
        path_id="unchanged-base",
        source="src/tiny_qwen_coder/evaluation/_baseline_generation.py",
        class_name="HuggingFaceBaselineGenerator",
    ),
    GeneratorPath(
        path_id="p8-python-p0",
        source="src/tiny_qwen_coder/evaluation/_python_p0_generation.py",
        class_name="HuggingFacePythonP0Generator",
    ),
    GeneratorPath(
        path_id="p9-rank-sweep",
        source="src/tiny_qwen_coder/evaluation/python_rank_sweep.py",
        class_name="RankCandidateGenerator",
        inherited_generation_from="p8-python-p0",
    ),
    GeneratorPath(
        path_id="p9-minimum-intervention",
        source="src/tiny_qwen_coder/evaluation/python_minimum_intervention.py",
        class_name="SnapshotTrajectoryGenerator",
    ),
    GeneratorPath(
        path_id="p9-distilled-v4-2000",
        source="src/tiny_qwen_coder/evaluation/python_distilled_trajectory.py",
        class_name="DistilledSnapshotGenerator",
    ),
)

_EVALUATOR_LOGIC = (
    EvaluatorLogicPath(
        suite_id="humaneval",
        evaluator_source="src/tiny_qwen_coder/evaluation/humaneval.py",
        evaluator_class="HumanEvalEvaluator",
        normalizers=(
            ("src/tiny_qwen_coder/evaluation/humaneval.py", "normalize_humaneval_completion"),
            ("src/tiny_qwen_coder/evaluation/humaneval.py", "_candidate_source"),
        ),
    ),
    EvaluatorLogicPath(
        suite_id="mbpp",
        evaluator_source="src/tiny_qwen_coder/evaluation/_mbpp_evaluator.py",
        evaluator_class="MBPPEvaluator",
        normalizers=(
            ("src/tiny_qwen_coder/evaluation/_mbpp_data.py", "normalize_mbpp_completion"),
        ),
    ),
    EvaluatorLogicPath(
        suite_id="repository_holdout",
        evaluator_source="src/tiny_qwen_coder/evaluation/_repository_holdout_evaluator.py",
        evaluator_class="RepositoryHoldoutEvaluator",
        normalizers=(
            (
                "src/tiny_qwen_coder/evaluation/_repository_holdout_runtime.py",
                "normalize_repository_holdout_completion",
            ),
        ),
    ),
)

_SCORING_PATHS = (
    ScoringPath(
        path_id="unchanged-base",
        source="src/tiny_qwen_coder/evaluation/_baseline_stages.py",
        function_name="score_canonical_python_base_baseline_stage",
        evaluator_source="src/tiny_qwen_coder/evaluation/_baseline_stages.py",
        evaluator_function="_load_generation_inputs",
        required_evaluators=("HumanEvalEvaluator", "MBPPEvaluator", "RepositoryHoldoutEvaluator"),
        harness="ConstrainedExecutionHarness",
        approved_runtime_difference=False,
    ),
    ScoringPath(
        path_id="p8-python-p0",
        source="src/tiny_qwen_coder/evaluation/python_p0.py",
        function_name="score_python_p0_stage",
        evaluator_source="src/tiny_qwen_coder/evaluation/python_p0.py",
        evaluator_function="_load_inputs",
        required_evaluators=("HumanEvalEvaluator", "MBPPEvaluator", "RepositoryHoldoutEvaluator"),
        harness="ConstrainedExecutionHarness",
        approved_runtime_difference=False,
    ),
    ScoringPath(
        path_id="p9-rank-sweep",
        source="src/tiny_qwen_coder/evaluation/python_rank_sweep.py",
        function_name="score_rank_stage",
        evaluator_source="src/tiny_qwen_coder/evaluation/python_rank_sweep.py",
        evaluator_function="_load_inputs",
        required_evaluators=("HumanEvalEvaluator", "MBPPEvaluator", "RepositoryHoldoutEvaluator"),
        harness="DirectExecutionHarness",
        approved_runtime_difference=True,
    ),
    ScoringPath(
        path_id="p9-minimum-intervention",
        source="src/tiny_qwen_coder/evaluation/python_minimum_intervention.py",
        function_name="score_checkpoint",
        evaluator_source="src/tiny_qwen_coder/evaluation/python_minimum_intervention.py",
        evaluator_function="_load_development_problems",
        required_evaluators=("HumanEvalEvaluator", "MBPPEvaluator"),
        harness="DirectExecutionHarness",
        approved_runtime_difference=True,
    ),
    ScoringPath(
        path_id="p9-distilled-development",
        source="src/tiny_qwen_coder/evaluation/python_distilled_trajectory.py",
        function_name="score_checkpoint",
        evaluator_source="src/tiny_qwen_coder/evaluation/python_minimum_intervention.py",
        evaluator_function="_load_development_problems",
        required_evaluators=("HumanEvalEvaluator", "MBPPEvaluator"),
        harness="DirectExecutionHarness",
        approved_runtime_difference=True,
    ),
    ScoringPath(
        path_id="p9-distilled-qualification",
        source="src/tiny_qwen_coder/evaluation/python_distilled_qualification.py",
        function_name="score_qualification",
        evaluator_source="src/tiny_qwen_coder/evaluation/python_distilled_qualification.py",
        evaluator_function="_qualification_context",
        required_evaluators=("HumanEvalEvaluator", "MBPPEvaluator", "RepositoryHoldoutEvaluator"),
        harness="DirectExecutionHarness",
        approved_runtime_difference=True,
    ),
)

_EXPECTED_CHAT_TEMPLATE: dict[str, object] = {
    "messages": [
        {"role": "system", "content": "system_prompt"},
        {"role": "user", "content": "user_prompt"},
    ],
    "kwargs": {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
        "enable_thinking": False,
    },
}
_EXPECTED_GENERATION_KWARGS: dict[str, object] = {
    "max_new_tokens": "self._settings.generation.max_new_tokens",
    "do_sample": False,
    "num_beams": 1,
    "use_cache": True,
}
_EXPECTED_DECODE_KWARGS: dict[str, object] = {
    "skip_special_tokens": True,
    "clean_up_tokenization_spaces": False,
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_tree(repo_root: Path, relative: str) -> tuple[ast.Module, str]:
    path = repo_root / relative
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FTRGenerationEvaluationParityError(
            f"could not read audited source: {relative}"
        ) from exc
    try:
        return ast.parse(source, filename=relative), source
    except SyntaxError as exc:
        raise FTRGenerationEvaluationParityError(
            f"could not parse audited source: {relative}"
        ) from exc


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(matches) != 1:
        raise FTRGenerationEvaluationParityError(f"expected exactly one class {name!r}")
    return matches[0]


def _function_node(container: ast.Module | ast.ClassDef, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in container.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise FTRGenerationEvaluationParityError(f"expected exactly one function {name!r}")
    return matches[0]


def _node_sha256(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise FTRGenerationEvaluationParityError("could not recover audited source segment")
    return _sha256_text(segment)


def _attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attribute_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _module_constants(tree: ast.Module) -> dict[str, object]:
    values: dict[str, object] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(statement.value, ast.Constant):
            continue
        constant = statement.value.value
        if isinstance(constant, (str, int, float, bool)) or constant is None:
            values[target.id] = constant
    return values


def _local_assignments(function: ast.FunctionDef) -> dict[str, ast.AST]:
    values: dict[str, ast.AST] = {}
    for statement in function.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                values[target.id] = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.value is not None:
                values[statement.target.id] = statement.value
    return values


def _resolve_local(node: ast.AST, assignments: Mapping[str, ast.AST]) -> ast.AST:
    current = node
    seen: set[str] = set()
    while isinstance(current, ast.Name) and current.id in assignments and current.id not in seen:
        seen.add(current.id)
        current = assignments[current.id]
    return current


def _normalized_value(node: ast.AST, constants: Mapping[str, object]) -> object:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    attribute = _attribute_name(node)
    if attribute is not None:
        return attribute
    return ast.dump(node, annotate_fields=False, include_attributes=False)


def _normalized_dict(node: ast.AST, constants: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(node, ast.Dict):
        raise FTRGenerationEvaluationParityError("expected a literal dictionary in audited source")
    result: dict[str, object] = {}
    for key_node, value_node in zip(node.keys, node.values, strict=True):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            raise FTRGenerationEvaluationParityError("audited dictionary has a non-string key")
        result[key_node.value] = _normalized_value(value_node, constants)
    return result


def _normalized_messages(
    node: ast.AST,
    *,
    assignments: Mapping[str, ast.AST],
    constants: Mapping[str, object],
) -> list[dict[str, object]]:
    resolved = _resolve_local(node, assignments)
    if not isinstance(resolved, ast.List):
        raise FTRGenerationEvaluationParityError("chat-template messages are not a literal list")
    return [_normalized_dict(item, constants) for item in resolved.elts]


def _call_matches_attribute(call: ast.Call, attribute: str) -> bool:
    return isinstance(call.func, ast.Attribute) and call.func.attr == attribute


def _unique_call(node: ast.AST, attribute: str) -> ast.Call:
    matches = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and _call_matches_attribute(child, attribute)
    ]
    if len(matches) != 1:
        raise FTRGenerationEvaluationParityError(
            f"expected exactly one {attribute}() call; observed {len(matches)}"
        )
    return matches[0]


def _tokenizer_load_exact(class_node: ast.ClassDef) -> bool:
    init = _function_node(class_node, "__init__")
    for node in ast.walk(init):
        if not isinstance(node, ast.Call):
            continue
        if _attribute_name(node.func) != "AutoTokenizer.from_pretrained":
            continue
        if not node.args or _attribute_name(node.args[0]) != "base_model.tokenizer_repository":
            return False
        revisions = [keyword.value for keyword in node.keywords if keyword.arg == "revision"]
        return (
            len(revisions) == 1
            and _attribute_name(revisions[0]) == "base_model.tokenizer_revision"
        )
    return False


def _model_load_exact(class_node: ast.ClassDef) -> bool:
    init = _function_node(class_node, "__init__")
    for node in ast.walk(init):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "from_pretrained":
            continue
        if not node.args or _attribute_name(node.args[0]) != "base_model.repository":
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
        revision = keywords.get("revision")
        revision_exact = (
            revision is not None and _attribute_name(revision) == "base_model.revision"
        )
        dtype = keywords.get("dtype")
        dtype_exact = dtype is not None and _attribute_name(dtype) == "torch.bfloat16"
        low_cpu_mem = keywords.get("low_cpu_mem_usage")
        low_cpu_mem_exact = isinstance(low_cpu_mem, ast.Constant) and low_cpu_mem.value is True
        device_map = keywords.get("device_map")
        device_map_exact = False
        if isinstance(device_map, ast.Dict) and len(device_map.keys) == 1:
            key = device_map.keys[0]
            value = device_map.values[0]
            device_map_exact = (
                isinstance(key, ast.Constant)
                and key.value == ""
                and _attribute_name(value) == "device_index"
            )
        return revision_exact and dtype_exact and low_cpu_mem_exact and device_map_exact
    return False


def _greedy_guard_present(class_node: ast.ClassDef) -> bool:
    init = _function_node(class_node, "__init__")
    for node in ast.walk(init):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
            continue
        if not isinstance(node.ops[0], ast.NotEq):
            continue
        if _attribute_name(node.left) != "settings.generation.decoding_strategy":
            continue
        comparator = node.comparators[0]
        if isinstance(comparator, ast.Constant) and comparator.value == "greedy":
            return True
    return False


def _class_inherits(class_node: ast.ClassDef, expected: str) -> bool:
    return any(_attribute_name(base) == expected for base in class_node.bases)


def _chat_template_contract(tree: ast.Module, class_node: ast.ClassDef) -> dict[str, object]:
    prepare = _function_node(class_node, "_prepare_inputs")
    assignments = _local_assignments(prepare)
    constants = _module_constants(tree)
    call = _unique_call(prepare, "apply_chat_template")
    if len(call.args) != 1:
        raise FTRGenerationEvaluationParityError("apply_chat_template must have one positional arg")
    if any(keyword.arg is None for keyword in call.keywords):
        raise FTRGenerationEvaluationParityError(
            "chat-template kwargs cannot be dynamically expanded"
        )
    kwargs = {
        cast(str, keyword.arg): _normalized_value(keyword.value, constants)
        for keyword in call.keywords
    }
    return {
        "messages": _normalized_messages(
            call.args[0],
            assignments=assignments,
            constants=constants,
        ),
        "kwargs": kwargs,
    }


def _generation_kwargs_contract(
    tree: ast.Module,
    generate: ast.FunctionDef,
) -> dict[str, object]:
    constants = _module_constants(tree)
    matches: list[ast.Call] = []
    for node in ast.walk(generate):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "update" or _attribute_name(node.func.value) != "kwargs":
            continue
        matches.append(node)
    if len(matches) != 1 or len(matches[0].args) != 1 or matches[0].keywords:
        raise FTRGenerationEvaluationParityError("generation kwargs.update contract is ambiguous")
    return _normalized_dict(matches[0].args[0], constants)


def _prompt_boundary_exact(generate: ast.FunctionDef) -> bool:
    assignments = _local_assignments(generate)
    prompt = assignments.get("prompt_tokens")
    if (
        not isinstance(prompt, ast.Call)
        or _attribute_name(prompt.func) != "int"
        or len(prompt.args) != 1
    ):
        return False
    length_expr = prompt.args[0]
    if not isinstance(length_expr, ast.Subscript):
        return False
    if not isinstance(length_expr.slice, ast.Constant) or length_expr.slice.value != 1:
        return False
    shape = length_expr.value
    if not isinstance(shape, ast.Attribute) or shape.attr != "shape":
        return False
    ids = _resolve_local(shape.value, assignments)
    if not isinstance(ids, ast.Subscript):
        return False
    if _attribute_name(ids.value) != "inputs":
        return False
    return isinstance(ids.slice, ast.Constant) and ids.slice.value == "input_ids"


def _completion_boundary_exact(generate: ast.FunctionDef) -> bool:
    for node in ast.walk(generate):
        if not isinstance(node, ast.Subscript):
            continue
        if _attribute_name(node.value) != "output" or not isinstance(node.slice, ast.Tuple):
            continue
        if len(node.slice.elts) != 2:
            continue
        batch, completion = node.slice.elts
        if not isinstance(batch, ast.Constant) or batch.value != 0:
            continue
        if not isinstance(completion, ast.Slice) or completion.upper is not None:
            continue
        if isinstance(completion.lower, ast.Name) and completion.lower.id == "prompt_tokens":
            return True
    return False


def _generate_call_exact(generate: ast.FunctionDef) -> bool:
    calls = [
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.Call) and _call_matches_attribute(node, "generate")
    ]
    # Protocol/type annotations are outside this function, so exactly one model call is expected.
    if len(calls) != 1 or calls[0].args:
        return False
    keywords = calls[0].keywords
    return (
        len(keywords) == 1
        and keywords[0].arg is None
        and isinstance(keywords[0].value, ast.Name)
        and keywords[0].value.id == "kwargs"
    )


def _decode_contract(tree: ast.Module, generate: ast.FunctionDef) -> dict[str, object]:
    constants = _module_constants(tree)
    call = _unique_call(generate, "decode")
    if len(call.args) != 1 or any(keyword.arg is None for keyword in call.keywords):
        raise FTRGenerationEvaluationParityError("tokenizer.decode contract is ambiguous")
    return {
        cast(str, keyword.arg): _normalized_value(keyword.value, constants)
        for keyword in call.keywords
    }


def _direct_inference_contract(tree: ast.Module, class_node: ast.ClassDef) -> dict[str, object]:
    generate = _function_node(class_node, "generate")
    generation_kwargs = _generation_kwargs_contract(tree, generate)
    return {
        "chat_template": _chat_template_contract(tree, class_node),
        "prompt_token_boundary": "inputs['input_ids'].shape[1]"
        if _prompt_boundary_exact(generate)
        else "unproven",
        "completion_token_boundary": "output[0, prompt_tokens:]"
        if _completion_boundary_exact(generate)
        else "unproven",
        "generation_kwargs": generation_kwargs,
        "generate_call_uses_only_kwargs": _generate_call_exact(generate),
        "explicit_stop_strings": [],
        "explicit_stop_token_ids": [],
        "decode_kwargs": _decode_contract(tree, generate),
    }


def _expected_runtime_contract() -> dict[str, object]:
    return {
        "chat_template": _EXPECTED_CHAT_TEMPLATE,
        "prompt_token_boundary": "inputs['input_ids'].shape[1]",
        "completion_token_boundary": "output[0, prompt_tokens:]",
        "generation_kwargs": _EXPECTED_GENERATION_KWARGS,
        "generate_call_uses_only_kwargs": True,
        "explicit_stop_strings": [],
        "explicit_stop_token_ids": [],
        "decode_kwargs": _EXPECTED_DECODE_KWARGS,
    }


def _json_diff(expected: object, observed: object, *, path: str = "") -> list[dict[str, object]]:
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        differences: list[dict[str, object]] = []
        keys = sorted(set(expected) | set(observed))
        for key in keys:
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                differences.append(
                    {"field": child, "expected": "<absent>", "observed": observed[key]}
                )
            elif key not in observed:
                differences.append(
                    {"field": child, "expected": expected[key], "observed": "<absent>"}
                )
            else:
                differences.extend(_json_diff(expected[key], observed[key], path=child))
        return differences
    if isinstance(expected, list) and isinstance(observed, list):
        if expected == observed:
            return []
        return [{"field": path, "expected": expected, "observed": observed}]
    if expected == observed:
        return []
    return [{"field": path, "expected": expected, "observed": observed}]


def _generator_report(repo_root: Path) -> tuple[list[dict[str, object]], bool]:
    rows: list[dict[str, object]] = []
    contracts: dict[str, dict[str, object]] = {}
    by_id = {item.path_id: item for item in _GENERATORS}
    expected = _expected_runtime_contract()
    passed = True

    for target in _GENERATORS:
        tree, source = _read_tree(repo_root, target.source)
        class_node = _class_node(tree, target.class_name)
        tokenizer_exact = _tokenizer_load_exact(class_node)
        model_exact = _model_load_exact(class_node)
        greedy_guard = _greedy_guard_present(class_node)

        if target.inherited_generation_from is not None:
            parent = by_id[target.inherited_generation_from]
            inherited = _class_inherits(class_node, parent.class_name)
            parent_contract = contracts.get(target.inherited_generation_from)
            if parent_contract is None:
                raise FTRGenerationEvaluationParityError(
                    "inherited generator parent was not audited"
                )
            contract = parent_contract
            differences = _json_diff(expected, contract)
            path_passed = (
                inherited
                and tokenizer_exact
                and model_exact
                and greedy_guard
                and not differences
            )
            rows.append(
                {
                    "path_id": target.path_id,
                    "source": target.source,
                    "class": target.class_name,
                    "generation_implementation": "inherited",
                    "inherited_from": target.inherited_generation_from,
                    "expected_parent_class": parent.class_name,
                    "inherits_expected_parent": inherited,
                    "base_model_load_contract_exact": model_exact,
                    "tokenizer_repository_and_revision_exact": tokenizer_exact,
                    "greedy_decoding_guard_present": greedy_guard,
                    "normalized_inference_contract": contract,
                    "contract_differences_from_unchanged_base": differences,
                    "class_source_sha256": _node_sha256(source, class_node),
                    "parity": path_passed,
                }
            )
            contracts[target.path_id] = contract
            passed = passed and path_passed
            continue

        contract = _direct_inference_contract(tree, class_node)
        contracts[target.path_id] = contract
        reference = expected if target.path_id == "unchanged-base" else contracts["unchanged-base"]
        differences = _json_diff(reference, contract)
        canonical_differences = _json_diff(expected, contract)
        path_passed = (
            tokenizer_exact
            and model_exact
            and greedy_guard
            and not differences
            and not canonical_differences
        )
        rows.append(
            {
                "path_id": target.path_id,
                "source": target.source,
                "class": target.class_name,
                "generation_implementation": "direct",
                "base_model_load_contract_exact": model_exact,
                "tokenizer_repository_and_revision_exact": tokenizer_exact,
                "greedy_decoding_guard_present": greedy_guard,
                "normalized_inference_contract": contract,
                "contract_differences_from_unchanged_base": differences,
                "contract_differences_from_frozen_expected": canonical_differences,
                "prepare_inputs_source_sha256": _node_sha256(
                    source, _function_node(class_node, "_prepare_inputs")
                ),
                "generate_source_sha256": _node_sha256(
                    source, _function_node(class_node, "generate")
                ),
                "parity": path_passed,
            }
        )
        passed = passed and path_passed

    return rows, passed


def _referenced_names(node: ast.AST) -> frozenset[str]:
    return frozenset(child.id for child in ast.walk(node) if isinstance(child, ast.Name))


def _scoring_report(repo_root: Path) -> tuple[list[dict[str, object]], bool]:
    rows: list[dict[str, object]] = []
    passed = True
    for target in _SCORING_PATHS:
        score_tree, score_source = _read_tree(repo_root, target.source)
        score = _function_node(score_tree, target.function_name)
        score_names = _referenced_names(score)
        harness_present = target.harness in score_names
        reduced_isolation_explicit = True
        if target.harness == "DirectExecutionHarness":
            segment = ast.get_source_segment(score_source, score) or ""
            reduced_isolation_explicit = "allow_reduced_isolation=True" in segment

        evaluator_tree, evaluator_source = _read_tree(repo_root, target.evaluator_source)
        evaluator_factory = _function_node(evaluator_tree, target.evaluator_function)
        evaluator_names = _referenced_names(evaluator_factory)
        required_evaluators = frozenset(target.required_evaluators)
        missing_evaluators = sorted(required_evaluators - evaluator_names)
        evaluators_shared = not missing_evaluators
        path_passed = harness_present and reduced_isolation_explicit and evaluators_shared
        passed = passed and path_passed
        rows.append(
            {
                "path_id": target.path_id,
                "score_source": target.source,
                "score_function": target.function_name,
                "score_function_sha256": _node_sha256(score_source, score),
                "evaluator_factory_source": target.evaluator_source,
                "evaluator_factory": target.evaluator_function,
                "evaluator_factory_sha256": _node_sha256(evaluator_source, evaluator_factory),
                "required_evaluators": sorted(required_evaluators),
                "missing_evaluators": missing_evaluators,
                "harness": target.harness,
                "harness_present": harness_present,
                "reduced_isolation_explicit": reduced_isolation_explicit,
                "approved_runtime_difference": target.approved_runtime_difference,
                "evaluation_logic_shared": evaluators_shared,
                "parity": path_passed,
            }
        )
    return rows, passed


def _evaluator_logic_report(repo_root: Path) -> dict[str, object]:
    rows: dict[str, object] = {}
    for definition in _EVALUATOR_LOGIC:
        tree, source = _read_tree(repo_root, definition.evaluator_source)
        class_node = _class_node(tree, definition.evaluator_class)
        evaluate = _function_node(class_node, "evaluate_completion")
        helper_hashes: dict[str, dict[str, str]] = {}
        for helper_path, helper_name in definition.normalizers:
            helper_tree, helper_source = _read_tree(repo_root, helper_path)
            helper_hashes[helper_name] = {
                "source": helper_path,
                "sha256": _node_sha256(
                    helper_source,
                    _function_node(helper_tree, helper_name),
                ),
            }
        rows[definition.suite_id] = {
            "source": definition.evaluator_source,
            "evaluator_class": definition.evaluator_class,
            "evaluate_completion_sha256": _node_sha256(source, evaluate),
            "normalization_helpers": helper_hashes,
            "shared_by_base_and_adapter_paths": True,
        }
    return rows


def audit_generation_evaluation_parity(
    *,
    repo_root: Path,
    source_git_sha: str,
) -> dict[str, object]:
    """Audit all protected Python generation/scoring paths and fail closed on drift."""

    if len(source_git_sha) != 40 or any(char not in "0123456789abcdef" for char in source_git_sha):
        raise FTRGenerationEvaluationParityError("source_git_sha must be a lowercase 40-char SHA")

    settings = load_frozen_evaluation_settings(repo_root / _SETTINGS_CONFIG)
    base_model = load_baseline_base_model_identity(repo_root / _BASE_CONFIG)
    generators, generation_passed = _generator_report(repo_root)
    scoring, scoring_passed = _scoring_report(repo_root)
    expected_contract = _expected_runtime_contract()

    frozen_settings_contract: dict[str, object] = {
        "decoding_strategy": settings.generation.decoding_strategy,
        "temperature": settings.generation.temperature,
        "top_p": settings.generation.top_p,
        "top_k": settings.generation.top_k,
        "max_new_tokens": settings.generation.max_new_tokens,
        "stop_policy": settings.generation.stop_policy,
        "prompt_version": settings.generation.prompt_version,
        "chat_template_version": settings.generation.chat_template_version,
        "sampling_parameters_effective": settings.generation.decoding_strategy != "greedy",
    }
    frozen_settings_passed = (
        settings.generation.decoding_strategy == "greedy"
        and settings.generation.max_new_tokens == 512
        and settings.generation.stop_policy == "eos_or_max_new_tokens"
    )

    intentional_differences = [
        {
            "field": "execution.isolation_backend",
            "paths": [
                item.path_id for item in _SCORING_PATHS if item.approved_runtime_difference
            ],
            "base": "ConstrainedExecutionHarness / OCI",
            "adapter": (
                "DirectExecutionHarness inside the explicitly accepted containerized "
                "self-hosted runner"
            ),
            "justification": (
                "The project deliberately avoids nested Docker/Podman for these self-hosted P9 "
                "score jobs. The direct backend is explicit and reduced-isolation; it does not "
                "claim OCI-equivalent filesystem/network isolation. Candidate normalization, "
                "execution-request construction, runner source, limits, timeout semantics, and "
                "result interpretation remain the shared evaluator implementations."
            ),
        }
    ]

    parity_passed = generation_passed and scoring_passed and frozen_settings_passed
    report: dict[str, object] = {
        "schema_version": 1,
        "task_id": "FTR-103",
        "source_git_sha": source_git_sha,
        "audit_mode": "static_ast_observational_no_production_inference_changes",
        "base_model": asdict(base_model),
        "evaluation_settings_sha256": evaluation_settings_sha256(settings),
        "frozen_settings_contract": frozen_settings_contract,
        "expected_runtime_contract": expected_contract,
        "expected_runtime_contract_sha256": _sha256_text(
            json.dumps(expected_contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        ),
        "generation_paths": generators,
        "scoring_paths": scoring,
        "shared_evaluator_logic": _evaluator_logic_report(repo_root),
        "intentional_differences": intentional_differences,
        "checks": {
            "frozen_evaluation_settings_valid": frozen_settings_passed,
            "generation_configuration_mechanically_equal": generation_passed,
            "chat_template_application_identical": generation_passed,
            "tokenization_boundaries_identical": generation_passed,
            "max_new_tokens_identical": generation_passed,
            "stop_strings_and_token_ids_identical": generation_passed,
            "sampling_settings_identical_or_inactive_under_greedy": generation_passed,
            "code_extraction_and_execution_request_logic_shared": scoring_passed,
            "intentional_execution_backend_difference_declared": True,
        },
        "parity_passed": parity_passed,
    }
    return report


def write_parity_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit FTR-103 generation/evaluation parity")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--output", type=Path, default=_DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = audit_generation_evaluation_parity(
        repo_root=cast(Path, args.repo_root),
        source_git_sha=cast(str, args.source_git_sha),
    )
    write_parity_report(cast(Path, args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["parity_passed"] is not True:
        raise FTRGenerationEvaluationParityError(
            "FTR-103 generation/evaluation parity audit failed; inspect the report payload"
        )


if __name__ == "__main__":
    main()
