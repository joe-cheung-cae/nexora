# Repository guidance

## Names

- Product/project: **Nexora**.
- CMake package/project: **Nexora**. Exported targets use `nexora::`.
- CLI executable: **`nexora`**.
- Language: C++17.
- C++ namespaces: **`nexora::`**.
- CMake options and preprocessor configuration macros: **`NEXORA_*`**.

Do not introduce the former CFDX name into new public interfaces.

## Engineering rules

- Preserve the dependency direction in `docs/architecture/overview.md`; the dependency checker is authoritative for target links and public header includes.
- Make the smallest focused change that solves the task. Do not change numerical formulas, defaults, tolerances, or physical conventions unless explicitly requested and backed by validation evidence.
- Do not add automatic downloads or mandatory third-party dependencies. Keep optional dependencies behind explicit options and describe them.
- Public headers must not expose CUDA or MPI implementation types. Keep platform-specific details in optional implementation targets.
- Adding or changing state requires considering ownership, communication, and checkpoint implications; note when those facilities are not implemented yet.
- Add tests for new modules and behavior. Numerical changes require correctness evidence. Performance changes require correctness evidence and a reproducible benchmark.
- Unsupported features must fail clearly; do not silently substitute a different backend, algorithm, or capability.
- Do not claim a configuration is validated merely because it compiles. Record actual test hardware and execution where relevant.
- Preserve third-party and legacy provenance. Do not copy code of unclear origin into the project.

## Build and validation

The required baseline is CPU C++17 / CMake 3.24+ / Ninja. Keep the CPU-only configure path free of CUDA, MPI, and network requirements. Run the relevant configure, build, and CTest preset after changes when verification is requested or needed to substantiate a claim.
