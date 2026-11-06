# ADR-003: Kokkos as the sole foundation

- Status: Kokkos-only direction selected; detailed design and implementation plan pending phase-1 approval
- Recorded at `2026-11-06T14:49:39+08:00`
- Supersedes the foundation choices in [ADR-001](ADR-001-module-boundaries.md) and [ADR-002](ADR-002-language-and-backend-baseline.md)

## Context

The initialization bootstrap has a native CPU execution API and host storage, with optional CUDA/MPI probes. Continuing this graph alongside Kokkos would create competing ownership, synchronization, dispatch, and portability contracts. The selected direction removes that parallel route.

## Decision

Use Kokkos as the only execution and numerical-storage foundation. It is required for the default CPU build. Kokkos Serial is the default execution space, OpenMP is an explicit CPU configuration, and CUDA/HIP are optional Kokkos configurations. MPI remains an optional communication integration. The numerical algorithms share Kokkos-based kernels rather than separate native implementations.

Nexora owns domain semantics, identifiers, mesh/field relationships, participant lifecycle, numerical contracts, communication and checkpoint policy. Kokkos owns parallel dispatch, execution/memory-space semantics, numerical allocations and copies. Foundation adapters may enforce Nexora policy; they must not become a provider system, custom DSL runtime, duplicated execution layer, or generic replacement for Kokkos Views and policies. See the authoritative [Kokkos design](../architecture/kokkos-design.md).

An explicitly approved complete external engine may retain its internal runtime; Warp may only occur as that engine's internal dependency. Application, service, worker, or process wrappers cannot restore a generic Nexora operator provider or a second computing foundation.

The proposed lock is Kokkos 5.2.2, commit `8e15454876298832f013ebcaf08d53825b5bda40`, C++20, and CMake 3.25.2+. [Build and validation](../architecture/build-and-validation.md) gives official-source verification and offline provisioning. Missing Kokkos is an error; automatic downloads and native fallback are prohibited.

## Consequences and implementation boundary

The default build remains offline and reproducible from provisioned inputs, but is no longer dependency-free. Public foundation and numerical data contracts can expose Kokkos types; raw CUDA/HIP and MPI types stay private to interoperability implementations. Installed Nexora packages must find and propagate their matching Kokkos dependency.

Retire native CPU/CUDA/HIP parallel implementation routes, provider selection, and DSL runtime proposals. The [migration map](../architecture/migration-roadmap.md) tracks old artifacts and destinations. Existing probes and metadata cannot establish Kokkos runtime support. Configured capabilities, compilation, device execution, numerical validation, and performance validation must be recorded separately.

This ADR changes documentation only. The [phase-1 review gate](../architecture/phase-1-design-review.md) must approve the detailed contracts, dependency policy, migration sequence, and representative acceptance before source, CMake, presets, CI, or tests change. Numerical formulas, defaults, tolerances, and conventions remain unchanged absent explicit authorization and evidence. Nexora's license remains unresolved under [ADR-014](ADR-014-license-and-provenance.md).
