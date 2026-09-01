from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"{label} did not match expected state")
    return text.replace(old, new, 1)


todo = Path("docs/TODO.md")
text = todo.read_text(encoding="utf-8")
old = """## P8-002 — Run general/tool regression suite

- [ ] instruction following.
- [ ] JSON.
- [ ] reasoning.
- [ ] shell/Git.
- [ ] tool-call formatting/selection.

Acceptance criteria:

- Regressions quantified, not hand-waved.
"""
new = """## P8-002 — Run general/tool regression suite

- [x] instruction following.
- [x] JSON.
- [x] reasoning.
- [x] shell/Git.
- [x] tool-call formatting/selection.

Acceptance criteria:

- Regressions quantified, not hand-waved.

Implementation note: canonical run `33554096916` evaluated the exact frozen 12-case `general_tool_regression` v1 suite against the accepted P6-005 baseline and independently verified the evidence. The unchanged base scored `2/12` and Python P0 also scored `2/12`: zero pass/fail regressions, zero improvements, and no category-level score delta. Both models preserved the two JSON passes. Two `preserved_fail` cases changed output despite equal scores: arithmetic changed from base `14` to adapter `17 - 5 + 8 = 20`, and Git staging changed from `git add -A` to `git add .`. The frozen suite is also sensitive to exact trailing-newline/tool-block formatting; this is recorded as a future-suite concern and is not retroactively changed after observing P0.
"""
todo.write_text(replace_once(text, old, new, "P8-002 TODO block"), encoding="utf-8")

doc = Path("docs/P8_002_GENERAL_TOOL_REGRESSION.md")
text = doc.read_text(encoding="utf-8")
marker = "## Comparison\n"
accepted = """## Accepted canonical result

Canonical workflow run `33554096916` on source `0485c375bcc384d0a0dfdfb423db740a4b78b109` passed GPU generation, P6 baseline validation, scoring, independent verification, and evidence upload. The final evidence artifact is GitHub Actions artifact `9818707785` (`p8-002-python-p0-general-tool-regression-0485c375bcc384d0a0dfdfb423db740a4b78b109`), with uploaded ZIP SHA-256 `d023e9d1563518fcba4ff1528d0290996004df943f834798d057ff7b22acc74b`.

| Category | Base | Python P0 | Delta |
| --- | ---: | ---: | ---: |
| instruction following | `0/2` | `0/2` | `0` |
| JSON structured output | `2/2` | `2/2` | `0` |
| simple reasoning | `0/2` | `0/2` | `0` |
| shell reasoning | `0/2` | `0/2` | `0` |
| Git reasoning | `0/2` | `0/2` | `0` |
| tool-call formatting/selection | `0/2` | `0/2` | `0` |
| **Overall** | **`2/12`** | **`2/12`** | **`0`** |

There were zero base-pass→adapter-fail regressions and zero base-fail→adapter-pass improvements. The two JSON cases were preserved passes. All ten remaining cases were preserved fails.

Pass/fail equality does not mean every output was identical. In particular, the arithmetic case changed from base `14` to adapter `17 - 5 + 8 = 20`, and the Git staging case changed from base `git add -A` to adapter `git add .`. Those behavioral mutations are preserved in the evidence for P8-004 analysis.

The frozen v1 suite also reveals exact-format sensitivity: several otherwise plausible responses include a trailing newline and therefore fail exact-text/tool-block matching. Because the suite was frozen before P0 training, P8-002 does not alter those expectations after seeing results. Any normalization change belongs in a future suite version and must not rewrite this comparison.

"""
if "## Accepted canonical result\n" not in text:
    text = replace_once(text, marker, accepted + marker, "P8-002 comparison marker")
doc.write_text(text, encoding="utf-8")
