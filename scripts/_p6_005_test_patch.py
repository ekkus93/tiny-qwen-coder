"""Temporary one-shot test-fixture patch helper for PR #28."""

from pathlib import Path

path = Path("tests/unit/test_python_base_baseline.py")
text = path.read_text(encoding="utf-8")
old = '''    for relative_path in _REQUIRED_ARTIFACTS:
        path = output / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artifact:{relative_path}\\n", encoding="utf-8")
    (output / "provenance.json").write_text(baseline_provenance_json(provenance), encoding="utf-8")
'''
new = '''    for relative_path in _REQUIRED_ARTIFACTS:
        path = output / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"artifact:{relative_path}\\n", encoding="utf-8")
    valid_coding_aggregate = '{"harness_errors":0,"pass_at_1":0.0}\\n'
    for relative_path in (
        "humaneval/humaneval-aggregate.json",
        "mbpp/mbpp-aggregate.json",
        "repository-holdout/repository-holdout-aggregate.json",
    ):
        (output / relative_path).write_text(valid_coding_aggregate, encoding="utf-8")
    (output / "provenance.json").write_text(baseline_provenance_json(provenance), encoding="utf-8")
'''
if old not in text:
    raise SystemExit("expected baseline freeze fixture block was not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
