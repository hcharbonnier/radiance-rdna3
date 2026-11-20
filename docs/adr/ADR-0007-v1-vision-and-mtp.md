# ADR-7 — v1 includes the vision tower and Flash-Next MTP

- Status: accepted (2026-11-16)
- Deciders: user
- Supersedes: ADR-6 (MTP out of scope for v1); completes ADR-5 (the tower is
  ported and must now be served)

## Context

**Vision.** W10 ported the tower (`attn_dense` hd72, axial rope hd72, the patch
ops) and W15 added the two remaining rows (`gemm_nt_bias_bf16`,
`grid_embed_bf16`); all are green in kbench. But the serving build has no
`RAD_WITH_FFMPEG`, so `capabilities.image=false` and every media part is
refused. ADR-5 allowed a media-free artifact for M5; that fallback is a test
artifact, not a target.

**MTP.** Flash-Next (`qwen3.8-next-flash-iqr-v4.rad`) carries a 48th MTP layer
(full attention, theta 1e7). Its routed experts are `w4nl` `u4`
(`moe_gemm_w4nl`, served since W15) and its shared expert IQR; the trunk's
dense, attention and HC rows already cover the rest of the layer. The
`qwen4exp_fp8` plugin declares `mtp_enter`, `mtp_couple_propose` and
`logit_rerank` only when `num-speculative-tokens > 0`; `libr3d` has no device
rows for `mtp_enter` and `logit_rerank` (`libr4d`, `libavx` and `libref` do), so
v1 pins the value at 0 (ADR-6). The 78 kbench cases left "Not covered" are the
MTP 48 tables and `ngram_hint`.

## Decision

- The **vision tower is in v1's target**: the shipped configuration is built
  with `RAD_WITH_FFMPEG=ON`, the container's media branch is served, and at
  least one image turn is demonstrated end to end (preprocess, tower, text
  answer).
- **Flash-Next MTP is in v1's target**: the service runs with
  `num-speculative-tokens > 0` once `libr3d` serves `mtp_enter` and
  `logit_rerank` on the device, with the same contracts as `libr4d`. The MTP 48
  cases kbench skips become required coverage, not "Not covered".
- ADR-6's `num-speculative-tokens: 0` no longer defines the served
  configuration; it remains the fallback while the two rows are written.
- The 27B recipe is **unchanged**: `num-speculative-tokens: 0` and KV i8 (its
  h128 windowed drafter requires bf16). This decision covers Flash-Next's MTP
  head, not the 27B's drafter.
- ADR-5's media-free artifact stays permitted for tests but no longer satisfies
  the target, and `docs/GUIDE.md`'s "build without ffmpeg" behaviour is a
  degraded mode, not the shipped one.

## Consequences

- New work: `mtp_enter` and `logit_rerank` device rows in `libr3d` plus device
  tests; an FFMPEG-enabled build on the host; a small image corpus and one
  vision turn in the demonstration; kbench MTP coverage (the 48-row cases and
  `moe_gemm_q` `w4nl64a8i8h` at MTP 48).
- Measurements: decode throughput is reported at the speculative setting, and
  any comparison with RDNA4 states both sides' setting.
- `ngram_hint` stays without an oracle: it is a sampler hint, not an op the
  served path needs, so its "Not covered" cases are still not regressions.
- Phase closure (checklist §8) gains two acceptance lines: an image turn
  served, and a speculative Flash-Next run with its kbench coverage.

## Outcome (2026-11-16)

- Vision: **delivered** (W26). The host has FFmpeg 8.0.1 and the `libav*-dev`
  packages; the serving build is configured with `-DRAD_WITH_FFMPEG=ON`
  (libavformat 62.3), the tower (856.68 MiB) is in VRAM, `capabilities.image`
  is true, and 512²/1024² image turns are answered correctly. Video and audio
  are advertised by the processor but untested.
- MTP: **delivered** (W25). `libr3d` serves `mtp_enter_e4m3` and
  `logit_rerank_iqr`/`_i8`; two pre-existing `gemm_nt_q_w2a8_m16` faults were
  fixed on the way (a joined weight's header was read from VRAM, and the MT
  chunk did not divide K at M=1/K=2560). The service runs at
  `num-speculative-tokens: 3` with 1.82× the decode throughput and identical
  greedy text (40.3% acceptance); the MTP 48 table coverage remains with R23.