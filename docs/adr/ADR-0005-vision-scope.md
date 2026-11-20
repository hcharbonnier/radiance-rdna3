# ADR-5 — Vision tower in the first milestone

- Status: accepted (2026-11-15); completed by ADR-7 (2026-11-16) — the tower is
  now required in the served target, and the media-free artifact stays a test
  artifact only.
- Deciders: user, programme lead

## Context

`qwen3.8-next-flash-iqr-v4.rad` declares `m.media`; the architecture plugin
declares the vision tower and `declare` will not complete if its ops cannot be
resolved. Porting the tower (ViT, ViT attention, grid embedding, row gather) is
3–4 weeks and is the last textual dependency but blocks M5.

## Decision

If the tower threatens the schedule, the first milestone may be demonstrated
with a container built without the media branch, used only as a test artifact.
The shipped 113 GiB container is never modified. The vision port (W10) remains in
the programme; M5 is met either by the ported tower or by the media-free
artifact, clearly labelled.

The user approved this fallback on 2026-11-15.

## Consequences

- ADR-5 removes the tower from the critical path to a text-serving milestone.
- The test artifact has a different hash and is not a deliverable container; the
  reproduction procedure names both artifacts.
- No change to the original container, `arch/common` or the plugin.
