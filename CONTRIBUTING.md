# Contributing

## Read the design boundary first

The [architecture overview](docs/architecture/overview.md) is the authoritative entry point. Kokkos is the selected sole execution/storage foundation and a required CPU dependency. This phase changes documentation only; the [phase-1 review](docs/architecture/phase-1-design-review.md) gates source, build, preset, CI, and test migration.

The proposed target uses C++20, CMake 3.25.2+, Ninja, and exact Kokkos 5.2.2. The [build and validation specification](docs/architecture/build-and-validation.md) supplies the lock, compiler requirements, offline acquisition/install/consumption plan, configuration matrix, and acceptance evidence. All new target command examples there are pending approval and implementation.

The [initial CUDA contract](docs/architecture/build-and-validation.md#cuda-build-contract) selects `nvcc_wrapper` with CXX, CUDA+Serial and one explicit architecture. The same verified package/toolchain contract applies to installed consumers. At cutover `NEXORA_GPU_BACKEND` is removed and any supplied value is rejected; the verified local Kokkos package determines the allowed execution-space set. These are proposed changes, not current configure behavior.

## Existing bootstrap checks

The current implementation still uses C++17, CMake 3.24+, and Ninja; it has no Kokkos dependency. The existing check sequence below is **bootstrap-only**, retained for inspection/regression evidence. It is not the target design's build procedure.

```sh
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

`cpu-release` is the optimized bootstrap preset. `mpi-debug` exercises an optional MPI C API probe, not distributed physics. The old `NEXORA_GPU_BACKEND=CUDA` option only enables a native CUDA diagnostic probe; it must not be extended into a second execution implementation. Current package tests cover old exports, including `nexora::exec_cpu`. The current dependency checker only knows the bootstrap graph.

## Change expectations after design approval

- Keep changes focused and preserve numerical contracts. Implement the accepted Kokkos boundaries; remove obsolete native/provider routes according to the migration map.
- Add meaningful tests for new behavior. Numerical changes require explicit approval and correctness evidence; optimization needs repeatable measurements for the same workload and tolerance.
- Require Kokkos even on CPU; keep GPU and MPI opt-in. Default configure, build, test, install, and downstream consumption must not fetch dependencies. Missing, wrong-version, or incompatible Kokkos installations must produce a clear error.
- Export required Kokkos dependencies transitively and test installed-package consumption in a clean, offline downstream build. Check compiler/ABI, execution-space, and architecture compatibility, not just a package's version string.
- Update architecture rules, dependency checks, package tests, and ADRs when changing supported contracts or configurations.
- Report unsupported and unvalidated configurations explicitly. Never silently substitute another backend or numerical method. Device tests that skip do not pass device acceptance; compile-only evidence is not runtime validation.
- Record exact source/toolchain/configuration, hardware, command, exit status, test counts, numerical tolerance, and ISO 8601 timestamp with seconds and timezone. Use the evidence states in [build and validation](docs/architecture/build-and-validation.md).
- Follow [AGENTS.md](AGENTS.md) for repository-specific collaboration rules.

## License and provenance

The Nexora license is pending maintainer decision. Confirm provenance and reuse rights before importing legacy or third-party material. The dependency policy in [third_party/README.md](third_party/README.md) records Kokkos's upstream license separately from Nexora, required attribution, local provisioning, and the lock upgrade process.
