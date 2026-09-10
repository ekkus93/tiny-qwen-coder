# PVRL-G1 — Environment Integrity Gate Closure

Status: **PASSED**

Date: 2026-09-09

Authoritative machine-readable evidence:

`docs/evidence/PVRL_G1_ENVIRONMENT_INTEGRITY.json`

## Scope

PVRL-G1 closes the PVRL-100 executable-environment-integrity stage. It requires the independently implemented PVRL-101 through PVRL-105 controls to compose as one fail-closed admission boundary rather than merely pass in isolation.

The audit started from master SHA:

`fe73c38da027c4fa4a384f5c3239ee3c2db6e045`

That SHA already contained merged PVRL-101 through PVRL-105.

## Component evidence

The following component implementations were independently CI-green before G1 closure:

- **PVRL-101 — immutable environment schema:** PR #65, implementation head `b80d81f72feeaf832bbd0bb022e372d387e7b9a4`, CI run `34387785484`, merged as `ed8b841ef50c5b97609a8cc166fe2a4dba248ae6`.
- **PVRL-102 — oracle/reference provenance:** PR #66, implementation head `e19dfcb2c1f1e98954714495e166770dc04b9af7`, CI run `34391045923`, merged as `194877f33c1117eb17b06d934e4d189599aa6f6d`.
- **PVRL-103 — reference-first validation:** PR #67, implementation head `f4ed93c2ba414987306db1f9b6e30a797e63231a`, CI run `34393664208`, merged as `96dccab1e718aa8425e062fac5bbd789c4ae00c7`.
- **PVRL-104 — hardened execution sandbox:** PR #68, implementation head `72bf480527680a584650e5c8bda17ff6d7a79971`, CI run `34398358972`, merged as `ee15ead30089f8798060b887113ac4fce08c13ca`.
- **PVRL-105 — contamination controls:** PR #69, implementation head `7f04fe0fbf04a6d14d9407cd5d94a97829cc3a25`, CI run `34411799714`, merged as `fe73c38da027c4fa4a384f5c3239ee3c2db6e045`.

## Cross-cutting audit finding

The component audit found one substantive closure gap. PVRL manifests correctly allow intermediate authored-candidate states with reference validation and contamination checks not yet run. That is necessary during environment construction, but before this gate there was no single API that prevented such an intermediate candidate from being treated as fully environment-integrity-admitted.

PVRL-G1 therefore adds `admit_environment_integrity`. The admission function requires the actual evidence objects, not merely status flags or caller-provided hashes. It verifies that all of the following bind to the same final immutable environment:

- the final evidence-bearing `EnvironmentManifest`;
- the exact `HardenedEnvironmentMaterial` artifact bytes;
- the successful `ReferenceValidationReport` attached to the manifest;
- the clean `PVRLContaminationReport` attached to the manifest;
- the `OracleProvenanceAssessment` created against the final manifest.

Successful admission emits content-addressed `EnvironmentIntegrityEvidence`. It records identities and hashes only; task text, hidden grader text, reference code, and protected benchmark material are not serialized into the gate record.

## Integration proof

The G1 integration test constructs a real immutable environment and exercises the complete composition path:

1. create PVRL-101 immutable environment material;
2. construct the PVRL-104 hardened material boundary;
3. execute PVRL-103 reference-first validation twice in fresh hardened sandboxes;
4. attach successful reference-validation evidence;
5. run PVRL-105 protected-data and partition contamination checks;
6. attach only clean contamination evidence;
7. create the PVRL-102 oracle-independence assessment against the final evidence-bearing manifest;
8. admit the environment through the new PVRL-G1 boundary.

The same tests prove fail-closed rejection for missing validation, missing contamination evidence, stale hardened material, a stale pre-final oracle assessment, reuse of a report from another environment, and post-admission evidence tampering.

The first PR #70 CI run, `34414146277`, stopped at pinned Ruff formatting before semantic gates ran. The formatting-only correction changed no logic. Exact-head replacement run **`34414512064`** on commit `7c72c44084158be2dc4373a5dc9dbe1e9ddba121` passed the fast static preflight, frozen environment sync, strict mypy, and the complete pytest suite.

## Gate decision

All PVRL-G1 conditions are satisfied:

- immutable environment schema: **PROVEN**;
- reference-first validation: **PROVEN**;
- sandbox/grader isolation: **PROVEN**;
- contamination controls: **PROVEN**;
- cross-cutting final admission: **PROVEN**.

PVRL-G1 is therefore **PASSED**.

## FTR dependency remains blocking for model-training interpretation

PVRL-G1 does not satisfy or bypass the separate FTR integrity prerequisite. At closure time, FTR-102 PR #63 remains open and FTR-G1 is not satisfied.

Therefore this gate authorizes continued PVRL infrastructure work, including PVRL-200, but it does **not** authorize interpreting PVRL model-training experiments as scientifically valid before FTR-G1 is satisfied.

## Next work

The next dependency-ordered PVRL stage is **PVRL-200 — Agent harness and exact trajectory identity**, beginning with **PVRL-201 — Define Leaf-lite tool contract**.
