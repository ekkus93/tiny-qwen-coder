from pathlib import Path

import yaml


WORKFLOW = Path(".github/workflows/python-p0-general-tool-regression.yml")


def test_p8_002_workflow_is_manual_only_and_compact() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "on:\n  workflow_dispatch:\n" in text
    assert "push:" not in text
    assert 'retention-days: 7' in text
    assert 'retention-days: 3' in text
    assert "33422910444" in text
    assert "33301242379" in text
    assert "c94606250e112f72362eb883a55f7b2c8af854d445f6bb6194352c2806a8f276" in text


def test_p8_002_workflow_has_gpu_generation_then_hosted_scoring() -> None:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs = data["jobs"]
    generate = jobs["generate"]
    score = jobs["score"]
    assert generate["runs-on"] == ["self-hosted", "linux", "x64"]
    assert score["runs-on"] == "ubuntu-24.04"
    assert score["needs"] == "generate"
    assert generate["timeout-minutes"] == 60
    assert score["timeout-minutes"] == 30
