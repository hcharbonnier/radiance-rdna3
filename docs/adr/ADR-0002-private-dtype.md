# ADR-2 — Private dtypes and layout tags for IQR in libr3d

- Status: accepted (draft 2026-11-15)
- Deciders: kernels lead, integration lead

## Context

A weight's stored form rides in its dtype: the layout hook fills the tensor's
dtype with a tag the core treats as opaque, and only the owning library decodes
it. `abi/rad_types.h` reserves values at or above `RAD_DT_PLUGIN_BASE` (256) for
plugins. libr4d, libhalo and libr3dh already use `RAD_DT_PLUGIN_BASE + 0..9`
for other weight forms, and libr4d's IQR uses `0x40000000`/`0x50000000`.
Tools may load several libraries and compare tags without a matching card.

## Decision

Use two tagged bases above the small enum range:

- `R3D_DT_IQR_W = 0x60000000u` — weight buffer;
- `R3D_DT_IQR_S = 0x70000000u` — scale buffer;
- the low 28 bits carry the form word; layout tags are `"r3d.iqr.w"` and
  `"r3d.iqr.s"` (and `"r3d.iqr.embed"`/`"r3d.iqr.lm"` for those tables).

`kernel_word()` clears the run bits, as libr4d does, because the kernel takes the
run as an argument.

## Rationale

- The values are far outside the `RAD_DT_PLUGIN_BASE + n` ladder, so they cannot
  collide with a library that later claims another small plugin dtype.
- They cannot collide with libr4d's IQR tags, so a host tool can hold planes
  from both libraries in one process (the parity test does).
- The core only sizes these tensors through the layout hook; it never calls
  `rad_dtype_bytes` on them.

## Consequences

- A fixture or tool that assumes `0x40000000` must select the libr3d row first;
  the kbench replay asks the selected row's hook, so no change is needed.
- The tag is stable for the whole programme; changing it invalidates caches.
