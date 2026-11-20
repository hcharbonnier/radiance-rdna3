# ADR-4 — Numerical parity policy for the IQR MoE

- Status: proposed (draft 2026-11-15; decided before W5)
- Deciders: kernels lead, programme lead

## Context

`moe_gemm_iqr` has no recorded fixture. libr4d's kernel orders its reductions
by expert run, 16-row tile and K-block; libr3d's port may choose a different
tile or run map for gfx11. Dense IQR rows are held to bit equality with the
recorded fixtures; the MoE cannot be, unless the order is reproduced exactly.

## Decision (proposed)

1. Build a host reference: decode each expert's IQR planes to int8 with the same
   host code as the layout hook, run the existing int8 GEMM reference, and
   compare the kernel output against it.
2. Accept a bounded difference only when it comes from f32 summation order:
   report max absolute error, max relative error and the 99.9th percentile, on
   every shape in the W5 matrix.
3. Where the difference exceeds 1 ulp of f32 accumulation for any output, the
   lot does not close until either libr4d's order is reproduced or the
   programme lead accepts the bound in writing.
4. The end-to-end model check is deterministic generation: same prompt, same
   seed, same tokens across two runs; a cross-card comparison is out of scope
   without RDNA4 access.

## Consequences

- The MoE reaches M2 only with a documented error table.
- The dense rows keep the stricter bit-equality rule; this ADR does not relax
  W0/W3/W4.
