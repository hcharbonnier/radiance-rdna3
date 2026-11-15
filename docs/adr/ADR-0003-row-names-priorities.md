# ADR-3 — Names and priorities of the IQR rows

- Status: accepted (draft 2026-11-15)
- Deciders: kernels lead, integration lead

## Context

The selector resolves an op by matching rows of the op's schema and ranking them
by priority. Recorded fixtures and `data/kbench_rows` identify rows by name, so
the names must match libr4d's where the same case is replayed. The IQR rows
implement `gemm_nt_q` (schema already declared by libref) and later
`moe_gemm_q`.

## Decision

- `gemm_iqr_nt_m64` — op `gemm_nt_q`, family `gemm`, priority
  `R3D_PRIO_GEMM_M16` (60). Band: `M <= 16` in W0/W3, extended to 64 in W4.
- `gemm_iqr_tiled` — op `gemm_nt_q`, family `gemm`, priority
  `R3D_PRIO_GEMM_TILED` (40). Band: `M >= 17`, added in W4.
- `moe_gemm_iqr` — op `moe_gemm_q`, family `moe`, priority `R3D_PRIO_ONLY`
  in W5 (no competing libr3d row for the IQR dtype).
- Constraints mirror the int8 twin's: `dtype` in `i8a8h`, `group` 128,
  `K % 128 == 0`, `N % 16 == 0`, and the trellis run dividing `N` where the
  form has one. The shim also refuses forms the decoder does not implement.

Rows are append-only sections per family in `r3d_rows.cpp`; the integrator
merges them, and a parallel branch does not renumber or edit existing rows.

## Consequences

- The fixture coverage names line up with libr4d, so `--op gemm_iqr` and the
  coverage files are directly comparable.
- A future band split (e.g. GEMV vs 16-token) keeps one row name and an
  internal dispatch, because the fixture records the row name.
- Priorities are spacing-compatible with later insertions (R3D values step by
  ten inside a family).
