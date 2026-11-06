# ADR-002: C++17 and opt-in platform probes

- Status: accepted for the initialization baseline
- Recorded at `2026-11-06T11:01:56+08:00`

## Decision

Use C++17 and CMake 3.24 or newer. The required build is CPU-only. MPI is optional through its C API; CUDA is optional and requires an explicit `CMAKE_CUDA_ARCHITECTURES`. Optional configuration never downloads dependencies. The initial CUDA and MPI functionality is a CLI smoke probe, not a complete backend or solver.

## Consequences

The default build is suitable for offline development and CI. GPU support claims require actual device execution. Unsupported combinations must be reported as such instead of inferred from successful compilation.
