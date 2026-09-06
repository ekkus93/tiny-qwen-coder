from __future__ import annotations

import json
from pathlib import Path


def replace_required(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"missing expected text in {path}: {old!r}")
    path.write_text(text.replace(old, new), encoding="utf-8")


for path in (
    Path("src/tiny_qwen_coder/distillation/finalize.py"),
    Path("src/tiny_qwen_coder/distillation/diagnostics.py"),
):
    replace_required(
        path,
        "from tiny_qwen_coder.distillation.v2_input import strip_v2_teacher_input_policy",
        "from tiny_qwen_coder.distillation.input_policy import strip_teacher_input_policy",
    )
    replace_required(
        path,
        "strip_v2_teacher_input_policy(record)",
        "strip_teacher_input_policy(record)",
    )

readme = Path("scripts/teacher_distillation/README.md")
text = readme.read_text(encoding="utf-8")
old_notebooks = """There are now two executable notebooks:

- [`qwen38_teacher_distillation_colab.ipynb`](qwen38_teacher_distillation_colab.ipynb) — preserved v1 workflow and historical 16/500/2,000 progression.
- [`qwen38_teacher_distillation_v2_colab.ipynb`](qwen38_teacher_distillation_v2_colab.ipynb) — current bounded v2 successor study after the v1 2,000-candidate diagnosis.

**Open the current v2 workflow in Colab:** [qwen38_teacher_distillation_v2_colab.ipynb](https://colab.research.google.com/github/ekkus93/tiny-qwen-coder/blob/master/scripts/teacher_distillation/qwen38_teacher_distillation_v2_colab.ipynb)
"""
new_notebooks = """There are now three executable notebooks:

- [`qwen38_teacher_distillation_colab.ipynb`](qwen38_teacher_distillation_colab.ipynb) — preserved v1 workflow and historical 16/500/2,000 progression.
- [`qwen38_teacher_distillation_v2_colab.ipynb`](qwen38_teacher_distillation_v2_colab.ipynb) — preserved bounded v2 high-reasoning study.
- [`qwen38_teacher_distillation_v3_colab.ipynb`](qwen38_teacher_distillation_v3_colab.ipynb) — current bounded v3 per-record answer-budget study.

**Open the current v3 workflow in Colab:** [qwen38_teacher_distillation_v3_colab.ipynb](https://colab.research.google.com/github/ekkus93/tiny-qwen-coder/blob/master/scripts/teacher_distillation/qwen38_teacher_distillation_v3_colab.ipynb)
"""
if old_notebooks not in text:
    raise RuntimeError("README notebook block did not match expected v2 text")
text = text.replace(old_notebooks, new_notebooks)
old_row = "| `prepare_teacher_v2_input.py` | Add and SHA-256 bind the teacher-only concise-answer v2 policy. |\n"
new_row = old_row + "| `prepare_teacher_v3_input.py` | Compute, inject, and SHA-256 bind the canonical Qwen3.5-4B per-record final-answer budget. |\n"
if old_row not in text:
    raise RuntimeError("README v2 preparation row missing")
text = text.replace(old_row, new_row)
if "## Bounded v3 per-record answer-budget contract" not in text:
    text += """

## Bounded v3 per-record answer-budget contract

The 200-record v2 high-reasoning study fixed the v1 runaway-generation problem: all 200 candidates stopped normally and protected-benchmark contamination was clean. Its remaining failure was narrow and specifically student-envelope related: 153/200 records (76.5%) fit the 2,048-token student boundary, below the predeclared 80% floor. Prompt lengths were not the driver; rejected final answers were substantially longer than accepted answers.

v3 therefore keeps the successful v2 teacher/runtime/generation contract fixed and changes only the teacher-only input policy:

- Qwen3.8-27B at the same pinned revision, native BF16, vLLM 0.28.0;
- reasoning effort remains `high`;
- sampling parameters, seed, 16-record shard size, 16,384-token context, and 8,192-token teacher generation cap remain unchanged;
- the same deterministic 200 source records are used for the bounded study;
- the Qwen3.5-4B canonical tokenizer measures each source prompt before teacher-only policy injection;
- each record receives `min(1792, 2048 - student_prompt_tokens - 128)` as its approximate final-answer budget;
- the 128-token reserve protects chat-template boundaries and imperfect token-budget adherence;
- the 8,192 teacher generation cap is deliberately not reduced because it includes hidden reasoning, while the new budget applies only to the final answer;
- the dynamic instruction is stripped before Python quality checks, student tokenization, contamination checks, or corpus writing.

The existing formal scale floor remains >=90% normal stops, >=80% student-length acceptance among normal stops, and contamination `clean`. For v3, the notebook uses a stricter **85% student-length scale gate** to provide margin before spending on a fresh 2,000-record run. Do not lower either gate after observing the result.
"""
readme.write_text(text, encoding="utf-8")

source_notebook = Path("scripts/teacher_distillation/qwen38_teacher_distillation_v2_colab.ipynb")
target_notebook = Path("scripts/teacher_distillation/qwen38_teacher_distillation_v3_colab.ipynb")
notebook = json.loads(source_notebook.read_text(encoding="utf-8"))
notebook_text = json.dumps(notebook)
notebook_text = notebook_text.replace("V2_", "V3_").replace("v2", "v3").replace("V2", "V3")
notebook = json.loads(notebook_text)
notebook["metadata"]["colab"]["name"] = target_notebook.name
notebook["cells"][0]["source"] = [
    "# Qwen3.8-27B bounded v3 teacher study on Google Colab\n",
    "\n",
    "This is the successor to the bounded v2 high-reasoning study. v2 achieved\n",
    "100% normal stops and clean contamination, but only 76.5% of normally stopped\n",
    "records fit the Qwen3.5-4B 2,048-token training envelope. The measured failure\n",
    "was final-answer verbosity rather than prompt length.\n",
    "\n",
    "v3 keeps the pinned Qwen3.8-27B teacher, native BF16 weights, vLLM 0.28.0,\n",
    "high reasoning effort, 16,384-token context, 8,192-token generation cap,\n",
    "sampling parameters, seed, and 16-record shard size. The only scientific\n",
    "intervention is a SHA-bound per-record final-answer budget computed with the\n",
    "canonical Qwen3.5-4B tokenizer: min(1792, 2048 - prompt_tokens - 128).\n",
    "\n",
    "The first v3 generation is the same deterministic 200-record source subset.\n",
    "Do not scale to 2,000 unless normal stops are >=90%, contamination is clean,\n",
    "and student-length acceptance is >=85% for this v3 scale decision.\n",
]
for cell in notebook["cells"]:
    source = "".join(cell.get("source", []))
    source = source.replace(">=80%", ">=85%").replace("≥80%", "≥85%")
    if "scripts/teacher_distillation/qualify_teacher_study.py" in source:
        marker = '    "--output",\n'
        if "--minimum-student-length-accept-rate-given-stop" not in source:
            if marker not in source:
                raise RuntimeError("v3 qualification insertion marker missing")
            source = source.replace(
                marker,
                '    "--minimum-student-length-accept-rate-given-stop",\n'
                '    "0.85",\n'
                + marker,
            )
    cell["source"] = source.splitlines(keepends=True)
target_notebook.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
