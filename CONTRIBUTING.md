# Contributing

Search [existing issues](https://github.com/joe-cheung-cae/nexora/issues) before reporting a bug or proposing a change. In your pull request, describe the change, link any relevant issue, and note which checks you ran.

## Read the design boundary first

The [architecture overview](docs/architecture/overview.md) is the authoritative entry point. The Serial CPU foundation, minimal reference state lifecycle, and managed Serial field binding and leases are implemented. See the [Phase 1 review](docs/architecture/phase-1-design-review.md) for the remaining tasks. C++20, CMake 3.25.2+, Ninja and a locally provisioned exact Kokkos 5.2.2 Serial-only package are required. See [build and validation](docs/architecture/build-and-validation.md) for acquisition and package requirements, and the [CPU foundation](docs/architecture/kokkos-cpu-slice-validation.md) and [reference lifecycle](docs/architecture/reference-lifecycle-validation.md) records for actual results.

## Current CPU checks

Test builds require a locally provisioned **GoogleTest 1.17.0**. The Serial validator supplies it alongside Kokkos from checksum-verified archives. For a native single-config build, the same checkout search convention described below applies to `.cache/dependencies/googletest-1.17.0/install-<lowercase-build-type>`. To use your own installation, pass its `lib/cmake/GTest` directory as `GTest_DIR`. Use `-DBUILD_TESTING=OFF` to build the libraries and CLI with Kokkos alone. [Offline provisioning commands](docs/architecture/serial-ci-validation.md#重跑) · [GoogleTest development plan](docs/architecture/googletest-development-plan.md)

For a native single-config build, CMake also searches a provisioned `.cache/dependencies/kokkos-5.2.2/install-<lowercase-build-type>` prefix in the checkout. Non-MSVC builds also search `install-release`. The normal CMake package paths take precedence. To select another verified installation, set `Kokkos_DIR` to its `lib/cmake/Kokkos` directory. CMake caches the resolved package. When switching between Debug and Release on MSVC, use separate build directories or `--fresh` so CMake searches for the matching installation. Use `--fresh` when reusing a build directory created by the old bootstrap, so the removed GPU option is not retained in its cache.

```sh
cmake --fresh --preset cpu-debug -DKokkos_DIR="<kokkos-debug-prefix>/lib/cmake/Kokkos" \
  -DGTest_DIR="<gtest-debug-prefix>/lib/cmake/GTest"
cmake --build --preset cpu-debug
ctest --preset cpu-debug
cmake --fresh --preset cpu-release -DKokkos_DIR="<kokkos-release-prefix>/lib/cmake/Kokkos" \
  -DGTest_DIR="<gtest-release-prefix>/lib/cmake/GTest" -DNEXORA_ENABLE_DOUBLE_PRECISION=ON
cmake --build --preset cpu-release
ctest --preset cpu-release
```

CTest discovers each GoogleTest case under `<target>.<suite>.<case>`. Run `ctest --preset cpu-debug -N -L gtest` to list the C++ cases or `ctest --preset cpu-debug -R '^nexora_reference_tests\.Reference\.EulerConvergence$'` to execute one. Running the full preset also exercises the CLI and relocated installation.

The default `nexora::real` type is `float`; configure with `-DNEXORA_ENABLE_DOUBLE_PRECISION=ON` for `double`. Use a separate clean build and install prefix for each precision and rebuild downstream consumers. The installed generated configuration fixes the type and rejects conflicting consumer requests or conflicting compiler-macro overrides. Existing reference numerical fields, parameters, operators and transport use `real`; time contracts remain `double`. Field metadata defaults to `real_scalar_type` and must match actual storage. Both precisions require numerical validation, including the independent Decimal oracle described in [real numerical validation](docs/architecture/real-numerics-validation.md).

The same CTest suite includes `nexora_reference_tests`: real Kokkos accepted/trial arrays, commit/restore/failure/retry/replay, retained const snapshots, requested-duration rounding and the existing Euler convergence criteria. The `nexora_reference_transport_tests` run local receive in the actual reference window and cover rollback, retry, and retained accepted read leases. The Reference method and participant are internal build targets. The installed CLI can check, plan, and execute [Reference cases](docs/architecture/reference-case.md). The `nexora_field_binding_tests` exercise native View binding, alias arbitration, and lease lifetime. The `nexora_transport_tests` exercise external ownership, explicit local packing, and per-consumer completion. The `nexora_transport_receive_tests` exercise stable-ID unpack into trial and ghost regions and verify that validation failures leave the target unchanged before retry. Installed-package tests cover schema-only consumption, typed data headers, transitive Kokkos computation, and both time include paths. Keep every participant and retained snapshot inside the application's Kokkos runtime lifetime.

The package is required offline; configuration does not download it. `nexora::data` propagates Kokkos; the new installed consumer uses only that exported Nexora target. Legacy exec exports and their bootstrap consumer have been removed. Use a clean install prefix when verifying the package: installing over an existing prefix does not remove obsolete headers or libraries. `NEXORA_GPU_BACKEND` is rejected for every value; the old native CUDA source is dormant. OpenMP and GPU profiles are future work. The `mpi-debug` preset retains an optional MPI C API smoke probe.

The [Serial validation runner](docs/architecture/serial-ci-validation.md) provisions the checksum-verified archives in a separate step, then uses the local Kokkos and GoogleTest packages for the dependency check, CTest, Decimal oracle and relocated package consumer. For routine validation use float Debug and double Release. For precision, packaging or merge acceptance run both precisions with each build type. Use matching Debug and Release Kokkos packages for MSVC. The CI matrix requests all four combinations on Linux GCC, Linux Clang and Windows MSVC, with the separate MPI smoke on Linux. See the runner record for the profiles actually executed.

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

Submit original contributions that you are authorized to license under Nexora's [BSD 3-Clause License](LICENSE). Confirm provenance and reuse rights before importing legacy or third-party material. The [dependency policy](third_party/README.md) records Kokkos's license, required notices, local provisioning, and the lock upgrade process.
