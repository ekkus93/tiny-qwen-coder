"""Temporary one-shot patch helper for PR #28; removed by the patch commit."""

from pathlib import Path


def _replace_once(path: Path, old: str, new: str, *, context: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"expected {context} anchor was not found")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


execution = Path("src/tiny_qwen_coder/evaluation/execution.py")
_replace_once(
    execution,
    """_COPY_AND_EXEC_SCRIPT = (
    'set -eu; cp -R /input/. /workspace/; chmod -R u+rwX /workspace; cd /workspace; exec \"$@\"'
)
""",
    """_COPY_AND_EXEC_SCRIPT = (
    'set -eu; cp -R /input/. /workspace/; '
    'for path in /workspace/* /workspace/.[!.]* /workspace/..?*; do '
    '[ -e \"$path\" ] || continue; chmod -R u+rwX \"$path\"; done; '
    'cd /workspace; exec \"$@\"'
)
""",
    context="constrained-execution bootstrap",
)

artifacts = Path("src/tiny_qwen_coder/evaluation/_baseline_artifacts.py")
text = artifacts.read_text(encoding="utf-8")
required_start = text.find("_REQUIRED_ARTIFACTS: tuple[tuple[str, str], ...] = (")
if required_start < 0:
    raise SystemExit("expected required artifact inventory was not found")
required_end = text.find("\n)\n\n\ndef _atomic_write", required_start)
if required_end < 0:
    raise SystemExit("expected required artifact inventory terminator was not found")
required_end += 2
constants = """
_CODING_AGGREGATES: tuple[tuple[str, str], ...] = (
    ("HumanEval", "humaneval/humaneval-aggregate.json"),
    ("MBPP", "mbpp/mbpp-aggregate.json"),
    (
        "repository holdout",
        "repository-holdout/repository-holdout-aggregate.json",
    ),
)
"""
text = text[:required_end] + constants + text[required_end:]

helper_anchor = "def freeze_python_baseline(\n"
helper = '''def _require_valid_coding_aggregates(output_dir: Path) -> None:
    """Reject incomplete coding metrics even when artifact files themselves are present."""

    for benchmark_name, relative_path in _CODING_AGGREGATES:
        context = f"{benchmark_name} aggregate"
        mapping = _read_json_mapping(output_dir / relative_path, context=context)
        harness_errors = _expect_int(mapping, "harness_errors", context=context)
        if harness_errors != 0:
            raise PythonBaselineError(
                f"{context} records {harness_errors} harness errors; baseline is invalid"
            )
        pass_at_1 = mapping.get("pass_at_1")
        if isinstance(pass_at_1, bool) or not isinstance(pass_at_1, (int, float)):
            raise PythonBaselineError(f"{context}.pass_at_1 must be a numeric value")
        if not 0.0 <= float(pass_at_1) <= 1.0:
            raise PythonBaselineError(f"{context}.pass_at_1 must be between zero and one")


'''
if helper_anchor not in text:
    raise SystemExit("expected freeze helper anchor was not found")
text = text.replace(helper_anchor, helper + helper_anchor, 1)

ordered_anchor = "    ordered = tuple(artifacts)\n    manifest = PythonBaselineManifest(\n"
ordered_replacement = (
    "    ordered = tuple(artifacts)\n"
    "    _require_valid_coding_aggregates(output_dir)\n"
    "    manifest = PythonBaselineManifest(\n"
)
if ordered_anchor not in text:
    raise SystemExit("expected freeze aggregate-validation anchor was not found")
text = text.replace(ordered_anchor, ordered_replacement, 1)

validate_anchor = '''    if actual_ids != required_ids:
        raise PythonBaselineError(
            "frozen Python baseline artifact inventory does not match the required P6-005 set"
        )
    return resolved
'''
validate_replacement = '''    if actual_ids != required_ids:
        raise PythonBaselineError(
            "frozen Python baseline artifact inventory does not match the required P6-005 set"
        )
    _require_valid_coding_aggregates(output_dir)
    return resolved
'''
if validate_anchor not in text:
    raise SystemExit("expected verify aggregate-validation anchor was not found")
artifacts.write_text(text.replace(validate_anchor, validate_replacement, 1), encoding="utf-8")
