# Nexora

*Different Methods. One Physical World.*

Nexora is a modular C++ foundation for multiphysics and HPC simulation. The initialization stage establishes module boundaries, an offline build, and a testable participant lifecycle.

## Build

The default build uses a C/C++ toolchain with C++17, CMake 3.24 or newer, and Ninja for the presets below. The CPU build and its CTest suite run offline with the standard library.

```sh
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

Use `cpu-release` for an optimized build. Presets put build trees under `build/`; user-specific overrides belong in the ignored `CMakeUserPresets.json`.

MPI is optional and uses the MPI C API. Configure with `NEXORA_ENABLE_MPI=ON` (the `mpi-debug` preset selects this option) and run its two-rank smoke test with `ctest --preset mpi-debug`.

CUDA is opt-in with `NEXORA_GPU_BACKEND=CUDA`. Set `CMAKE_CUDA_ARCHITECTURES` to the target device architecture. On a CUDA runner, replace `80` with the matching architecture and run:

```sh
cmake -S . -B build/cuda -G Ninja -DNEXORA_GPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/cuda
ctest --test-dir build/cuda -L cuda --output-on-failure
```

The CUDA smoke test needs a visible device. The MPI and CUDA options build CLI diagnostics and keep the public CPU libraries free of either runtime.

```sh
cmake -S . -B build/cuda-smoke -G Ninja -DNEXORA_GPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/cuda-smoke
ctest --test-dir build/cuda-smoke -L cuda --output-on-failure
```

Replace `80` with the intended device architecture. A CTest result marked skipped means no usable device was available; it is not a passing device validation.

## Install and consume

```sh
cmake --install build/cpu-debug --prefix <prefix>
```

Installed CMake package consumers use:

```cmake
find_package(Nexora CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE nexora::simulation nexora::exec_cpu)
```

The installed target/API surface is limited to what the current build exports. `nexora --version` reports the build version; `nexora inspect-build` reports configured capabilities and build metadata. Optional capabilities must be reported as unavailable when they were not built.

## Architecture and current status

The enforced dependency direction and current baseline are described in [the architecture overview](docs/architecture/overview.md), with decisions in [docs/adr](docs/adr/). For the first solver slice, select representative cases and capture numerical and performance baselines. Review provenance and reuse rights before importing external code.

The [archived v1.2 development plan](docs/archive/CFDX_Multimethod_HPC_Architecture_Plan_v1.2_NVIDIA_Integration.md) records the original architecture proposal.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) before changing the project.

## License status

No project license has been selected. See [LICENSE](LICENSE) and ADR-014. Do not assume this notice grants rights to use, modify, or redistribute project or legacy material.
