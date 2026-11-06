# ADR-002: Language and backend baseline

- Status: C++17 initialization baseline superseded by [ADR-003](ADR-003-kokkos-only-foundation.md); replacement baseline proposed for phase-1 approval
- Originally recorded at `2026-11-06T11:01:56+08:00`
- Updated at `2026-11-06T14:49:39+08:00`

## Original baseline

The unchanged bootstrap uses C++17, CMake 3.24+, a native CPU path, and optional MPI/CUDA diagnostic probes. This describes current files, not a continuing architectural option.

## Proposed replacement

Use Kokkos 5.2.2 at commit `8e15454876298832f013ebcaf08d53825b5bda40`, C++20 throughout Kokkos/Nexora/consumers, CMake 3.25.2+, and Ninja. Kokkos is required even for the default CPU build. Default to Kokkos Serial; select OpenMP or one Kokkos GPU backend explicitly. MPI remains opt-in through its C API at the communication boundary. Do not preserve a dependency-free native CPU alternative.

Kokkos 5.2.2 enforces C++20 or newer and has a general CMake minimum of 3.22; its CUDA-as-CMake-language mode requires 3.25.2. The proposed Nexora floor intentionally uses the stricter value for every configuration. Compiler floors, exact upstream evidence, configuration restrictions, and unvalidated platforms are recorded once in [build and validation](../architecture/build-and-validation.md).

## Consequences

A locally supplied, pinned dependency replaces the former zero-dependency policy while preserving offline default operation. Upgrade source/build/preset/CI/package contracts only after [phase-1 review](../architecture/phase-1-design-review.md). Successful configuration or compilation is not device execution, numerical validation, or performance evidence.
