# Contributing

## Development setup

Use CMake 3.24+, Ninja, and a C++17 compiler. The CPU-only presets are the default development path and require no third-party package download.

```sh
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

For MPI changes, use the `mpi-debug` preset and run the two-rank smoke test. For CUDA changes, select `NEXORA_GPU_BACKEND=CUDA`, set `CMAKE_CUDA_ARCHITECTURES` explicitly, and record the compiler/toolkit, GPU model, and device-test results. Do not describe compilation without device execution as GPU validation.

## Change expectations

- Keep changes focused and preserve established public contracts.
- Add tests for new modules and behavior. Numerical or performance changes need correctness evidence; optimizations also need a repeatable benchmark against the same workload and tolerance.
- Keep optional integrations opt-in. A default configure/build must not fetch dependencies.
- Update architecture rules or an ADR when changing module boundaries, public API, dependency policy, or supported configurations.
- Document unsupported or unvalidated configurations explicitly. Never silently fall back to a different numerical method or execution backend.
- Follow [AGENTS.md](AGENTS.md) for repository-specific collaboration rules.

## License and provenance

The project license is pending maintainer decision. Confirm provenance and reuse rights before vendoring or redistributing third-party code. Any dependency addition must document its source, version, license, and whether it is required at build or runtime.
