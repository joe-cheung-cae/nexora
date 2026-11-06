# ADR-001: Modular targets and dependency direction

- Status: initialization decision superseded by [ADR-003](ADR-003-kokkos-only-foundation.md); historical implementation remains until approved migration
- Originally recorded at `2026-11-06T11:01:56+08:00`
- Supersession documented at `2026-11-06T14:49:39+08:00`

## Original decision and current implementation

The bootstrap uses separate core, execution API, native CPU execution, data, participant API, and simulation targets. CLI and tests are leaves. Optional native CUDA and MPI paths are diagnostic probes only. Its dependency checker and exported package still encode this graph.

## Superseding direction

Retain modular boundaries and a directed dependency graph, but replace the execution/storage foundation with required Kokkos. Do not develop the old native CPU/CUDA routes or add native HIP, providers, or a DSL runtime. [The overview](../architecture/overview.md) owns the target graph; [the Kokkos design](../architecture/kokkos-design.md) owns public foundation/data/kernel contracts. Raw vendor-runtime and MPI types stay inside optional interoperability implementations.

The target graph, checker, exported dependencies, and package-consumer tests must migrate together after [phase-1 approval](../architecture/phase-1-design-review.md). Existing checker success is bootstrap evidence only. No target implementation is claimed by this ADR update.
