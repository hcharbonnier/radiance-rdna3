# ADR-1 — IQR form definitions shared with libr4d

- Status: accepted (draft 2026-11-15)
- Deciders: kernels lead, integration lead

## Context

libr4d owns the IQR weight definitions and host layout hooks
(`libr4d/r4d_iqr.h`, `r4d_iqr_layout.cpp`). libr3d must serve the same stored
bytes on gfx11 but may not depend on libr4d, neither at build time nor at run
time (plan §2.4, §3.3). The definitions are pure host C++ plus device-side
constexpr helpers; the layout order must be byte-identical or the recorded
fixtures cannot be replayed.

## Decision

Duplicate the definitions in `libr3d/r3d_iqr.h` with the namespace `r3d_iqr` and
libr3d's own private dtype tags. The host hooks live in
`libr3d/r3d_iqr_layout.cpp`, and the device decoder in
`libr3d/r3d_iqr_dec.h`.

Add a host-only parity test, `tests/relayout_iqr_parity_test.cpp`, that compiles
both `r4d_rad_relayout_iqr` and `r3d_rad_relayout_iqr` into one executable and
requires byte equality on every form, plus equality of the inverse. The test is
the consistency contract; neither library includes the other.

Rejected alternatives:

- A neutral header in `abi/`: the form word is a kernel-private storage layout,
  not an ABI concept. Moving it to the ABI would freeze plugin internals.
- A build dependency on libr4d: forbidden by the plan and would drag gfx12
  units into an RDNA3 build.
- Reading the stored form through an opaque core API: no such API exists, and
  the hooks are where the core expects storage decisions.

## Consequences

- Two copies to keep aligned. The parity test fails loudly when either drifts.
- A form added to libr4d is not automatically served by libr3d; `built()` lists
  what libr3d's kernels read, and a new form requires both the decoder and the
  list.
- No collision with libr4d's dtype tags (see ADR-2).
