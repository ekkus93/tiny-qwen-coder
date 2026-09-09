# FTR-000 completion marker

FTR-000 is complete as of repository commit `1fa3af57b4b1cdfec56ae654c827b2c925bf029b`.

The authoritative machine-readable boundary is `docs/evidence/FTR_000_REMEDIATION_BOUNDARY.json`.

It freezes the remediation-track starting master SHA, the exact Qwen3.5-4B and Qwen3.8-27B revisions, the frozen unchanged-base benchmark artifacts and scores, the failed P9-007 qualification evidence, the P9-008 forensic evidence, the repaired v4-2000 corpus identities, and the qualification-freshness boundary.

P9-007 is explicitly a failed model qualification despite a successful operational workflow. Its consumed qualification set is historical/diagnostic evidence only for successor experiments whose design was influenced by those outcomes; it is not a fresh qualification set.
