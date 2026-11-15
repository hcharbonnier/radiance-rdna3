# ADR-6 — MTP is out of scope for v1

- Status: accepted (2026-11-15)
- Deciders: programme lead

## Context

The `qwen4exp_fp8` plugin declares `mtp_enter`, `mtp_couple_propose` and
`logit_rerank` only when `num-speculative-tokens > 0`. The one-card Flash-Next
configuration uses 3 speculative tokens, but the programme's first service can
run at 0.

## Decision

- v1 serves with `num-speculative-tokens: 0`; MTP rows are not implemented.
- The service configuration and the demonstration pin that value.
- W13 (MTP) stays optional and out of every milestone.
- The 27B spike also runs with 0 to keep the first end-to-end path small; the
  DFlash drafter is benchmarked separately only if it resolves.

## Consequences

- Decode throughput lacks the drafter's gain; the comparison with RDNA4 in the
  report must state the same setting, or explicitly note the difference.
- Re-enabling speculation later is additive: rows plus the two existing ops.
