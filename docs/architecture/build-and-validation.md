# Build and validation specification

- Revision: `2026-11-19T13:44:51+08:00`
- Status: limited Serial CPU slice implemented; full [Phase 1](phase-1-design-review.md) and other profiles remain unapproved
- Official-source verification: `2026-11-06T15:45:05+08:00` (CUDA compilation/package contract; no build or dependency acquisition)
- Entry point: [architecture overview](overview.md); foundation contracts: [Kokkos design](kokkos-design.md)

The current CPU slice and local provisioning are described below. Later CUDA and complete Phase 1 sections remain proposed targets. Legacy exec exports and their bootstrap consumer have been removed; core, schema, data and control exports remain. The CI workflow provisions Kokkos and the test-only GoogleTest package before configuring Nexora. See [Serial CI validation](serial-ci-validation.md) for the workflow, local replay and actual runner results, and [CPU slice validation](kokkos-cpu-slice-validation.md) for the original slice.

## Real precision configuration

`nexora::real`, declared in `nexora/core/Real.hpp` and included by `Types.hpp`, defaults to `float`. Set `-DNEXORA_ENABLE_DOUBLE_PRECISION=ON` to select `double`. Generated and installed `nexora/core/Config.hpp` fixes the configuration as 0/1 for both the library and its consumers; conflicting compiler-macro overrides and explicitly conflicting downstream precision requests are rejected. Precision is a package configuration, so use separate clean build and install prefixes for the two choices and rebuild consumers; do not mix headers and libraries from different configurations. An overlaid install does not remove stale files.

Existing reference numerical state, parameters, operators and local transport use `real`; time values retain `double`. `FieldDescriptor` defaults to `real_scalar_type` (float32/float64 for the configured alias); explicitly typed external/integer storage must declare its actual dtype. The double reference keeps its existing thresholds; float is checked against a documented roundoff bound, and both use an independent 80-digit Decimal oracle. Current migration results are recorded in [real numerical validation](real-numerics-validation.md); [alias configuration validation](real-precision-validation.md) is the earlier evidence before numerical migration.

## Dependency and language lock

| Item | Requirement | Evidence or rationale |
| --- | --- | --- |
| Kokkos | **5.2.2**, commit **`8e15454876298832f013ebcaf08d53825b5bda40`** | [Official release](https://github.com/kokkos/kokkos/releases/tag/5.2.2), [release commit](https://github.com/kokkos/kokkos/commit/8e15454876298832f013ebcaf08d53825b5bda40); release date shown as 2026-09-10; exact seconds/timezone were not obtained from the release summary |
| GoogleTest | **1.17.0**, required when `BUILD_TESTING=ON` | Exact local config package with compiler-matched libraries. [Source, digest and license](../../third_party/README.md#googletest-provenance-record) · [Test development plan](googletest-development-plan.md) |
| C++ | **20**, consistent for Kokkos, Nexora and downstream consumers | Pinned [standard selection](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/cmake/kokkos_pick_cxx_std.cmake) and [standard validation](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/cmake/kokkos_test_cxx_std.cmake) |
| CMake | **3.25.2+** for every Nexora target configuration | Kokkos's [top-level minimum is 3.22](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/CMakeLists.txt), but [CUDA as a CMake language requires 3.25.2](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/cmake/kokkos_enable_options.cmake); use one stricter project floor |
| Generator / architecture | Ninja; 64-bit target; record exact Ninja version | Reproducible runner profile; Kokkos's pinned top-level source treats 32-bit as experimental |
| Default CPU execution | Kokkos Serial, with GPU and MPI disabled | Current limited slice; not a general upstream support claim |
| Kokkos license | `Apache-2.0 WITH LLVM-exception` | [Pinned upstream LICENSE](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/LICENSE) |

The official release archive SHA-256 is `d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f`, verified against its official manifest; the annotated tag resolves to the locked commit. See [provenance](../../third_party/README.md). `latest`, `develop`, a moving branch, and a version range are not acceptable dependency locks.

The live [requirements page](https://kokkos.org/kokkos-core-wiki/get-started/requirements.html) and [configuration guide](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html) are useful references, but the exact commit's source takes precedence where a general documentation minimum differs. In particular, a generic CMake 3.16 claim must not override the pinned source requirements above.

## Compiler baseline versus Nexora validation

The following are upstream floors from the pinned [compiler checks](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/cmake/kokkos_compiler_id.cmake). They do not prove every compiler/library/driver combination works for Nexora.

| Configuration | Upstream compiler floor | Nexora target treatment |
| --- | --- | --- |
| Serial / OpenMP CPU | GCC 10.4.0 or Clang 14.0.0 | First acceptance profiles; verify the compiler's C++20 standard library and OpenMP runtime when enabled |
| CUDA | NVCC 12.2.0; upstream also checks Clang CUDA 15.0.0 | Initial CUDA acceptance uses `nvcc_wrapper` with NVCC and a toolkit-compatible host compiler; other CUDA compilation routes are outside the initial profile |
| HIP | HIPCC 6.2.0 | Future optional Kokkos profile; ROCm/driver/GPU combination requires its own execution evidence |
| MSVC CPU | 19.30 | Upstream floor only; no Nexora Windows validation claimed |
| Other compilers/platforms | Consult pinned checks and upstream support notes | IntelLLVM, NVHPC, SYCL and AppleClang are not implicitly accepted Nexora profiles |

Choose exact compiler, standard library, toolkit, driver and OS patch versions when provisioning each runner; save them with the artifact manifest. Meeting a minimum is only a configuration prerequisite. GPU toolkit/host-compiler compatibility and explicit device architecture selection remain necessary. Do not derive AppleClang support from the LLVM Clang version number alone.

## Offline dependency lifecycle

**Provisioning process:**

1. A separate, explicitly networked acquisition job obtains the official 5.2.2 source distribution or a Git checkout at the locked commit. The Nexora configure/build never performs this step.
2. For an archive, obtain the release's [SHA-256 manifest](https://github.com/kokkos/kokkos/releases/download/5.2.2/kokkos-5.2.2-SHA-256.txt), signature and provenance listed on the release page; verify the artifact and record the resulting digest and verification method. For Git, verify the full commit and release-tag association and preserve the checkout identity. Any packaged Git snapshot also receives an artifact SHA-256.
3. Store the source, license/notices, provenance and manifest in an approved local artifact cache or mirror. Record local patches explicitly; the acquired archive has none. No binaries or large archives belong in this repository.
4. On the offline machine, verify the cached artifact digest before unpacking; build and install Kokkos with the chosen C++ standard, compiler, architecture and execution-space options. Give each incompatible configuration a separate prefix. Pre-provision any compiler/OpenMP/GPU/MPI runtime dependencies as well.
5. Configure Nexora against that exact installed Kokkos using `Kokkos_DIR` or a controlled `CMAKE_PREFIX_PATH`. Native single-config builds also search the [checkout's local install prefixes](../../CONTRIBUTING.md#current-cpu-checks) after the standard package paths. Do not introduce download-based FetchContent or a package-manager bootstrap. Missing or incompatible dependencies fail with an actionable error.
6. Build, test and install Nexora, then build and run a separate downstream consumer with the network disabled. Record source IDs, digests, configuration, command outputs and execution evidence. Keep local paths in a manifest, not embedded as mandatory absolute paths in exported targets.

“Offline” means all inputs are already provisioned. It does not mean dependency-free. Reproducibility means the source/toolchain/options/hardware profile is recorded and repeatable; it does not promise bitwise-identical numerical results across execution spaces or compilers. Never change numerical tolerances merely to make a new backend pass.

## Current Serial provisioning and proposed package examples

Use `tools/validate_serial.py` for the CI-equivalent Serial checks. Its provisioning phase checks the exact source SHA-256, installs a compiler- and configuration-specific static package, and records file digests and provenance. For test builds it also verifies and provisions GoogleTest into a separate prefix. Its validation phase uses local `Kokkos_DIR` and `GTest_DIR` to configure, build, check the dependency graph, run CTest and the Decimal oracle, then install, relocate and exercise the package in a separate consumer. Linux CI runs this phase in a fresh network namespace. See the [Serial CI validation record](serial-ci-validation.md) for reproducible commands and platform status.

The following is the separately networked acquisition and offline local provisioning used for the current Serial slice. Choose paths appropriate to the machine. Confirm the manifest line matches before extraction and retain license notices as described in [third-party provenance](../../third_party/README.md).

```sh
curl -fLO https://github.com/kokkos/kokkos/releases/download/5.2.2/kokkos-5.2.2.tar.gz
curl -fLO https://github.com/kokkos/kokkos/releases/download/5.2.2/kokkos-5.2.2-SHA-256.txt
shasum -a 256 kokkos-5.2.2.tar.gz
git ls-remote https://github.com/kokkos/kokkos.git 'refs/tags/5.2.2*'
tar -xzf kokkos-5.2.2.tar.gz
cmake -S kokkos-5.2.2 -B <kokkos-build> -G Ninja \
  -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=<kokkos-prefix> -DCMAKE_INSTALL_LIBDIR=lib \
  -DBUILD_SHARED_LIBS=OFF -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_HIP=OFF -DKokkos_ENABLE_SYCL=OFF \
  -DKokkos_ENABLE_TESTS=OFF -DKokkos_ENABLE_EXAMPLES=OFF \
  -DKokkos_ENABLE_BENCHMARKS=OFF
cmake --build <kokkos-build>
cmake --install <kokkos-build>
```

The slice checks 64-bit target, exact version, Serial-only devices, C++20/CXX and matching compiler ID and exact version. Installed Nexora also compares Kokkos architecture, options and TPL metadata. Source revision, digest and custom standard-library or ABI settings require manifest verification; these checks do not prove arbitrary ABI compatibility. See [CONTRIBUTING](../../CONTRIBUTING.md) for current Nexora commands, including `--fresh` for old caches. The installed CPU consumer links only `nexora::data` and obtains Kokkos transitively. The legacy exec export and consumer have been removed.

**The following design examples are proposed targets beyond the current limited slice.** Placeholder paths must be replaced with verified local inputs. The example uses a Serial dependency with no Kokkos tests, examples or benchmarks; this avoids silently adding their dependencies. Nexora's own acceptance tests remain required.

```sh
# 批准后目标 / PROPOSED — pending approval and implementation; local verified source only.
cmake -S <local-kokkos-5.2.2-source> -B <kokkos-serial-build> -G Ninja \
  -DCMAKE_CXX_COMPILER=<absolute-cxx-path> -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=<kokkos-serial-prefix> \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_HIP=OFF \
  -DKokkos_ENABLE_TESTS=OFF -DKokkos_ENABLE_EXAMPLES=OFF \
  -DKokkos_ENABLE_BENCHMARKS=OFF
cmake --build <kokkos-serial-build>
cmake --install <kokkos-serial-build>
cmake -S . -B <nexora-serial-build> -G Ninja \
  -DCMAKE_CXX_COMPILER=<absolute-cxx-path> -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_PREFIX_PATH=<kokkos-serial-prefix> -DBUILD_TESTING=ON
cmake --build <nexora-serial-build>
ctest --test-dir <nexora-serial-build> --output-on-failure
cmake --install <nexora-serial-build> --prefix <nexora-prefix>
```

Use a clean install prefix when validating removal of old headers and libraries. Installing over an existing prefix does not remove obsolete files; clean or replace that prefix before relying on its contents. The current Nexora package propagates Kokkos through `nexora::data`, which publicly exposes the designated Kokkos numerical storage contract. The exported `nexora::data_schema` depends on core, and the participant and simulation targets receive schema without transitive Kokkos compile or link requirements. Package discovery still validates the exact Kokkos dependency. The old `nexora::exec_cpu` export has been removed.

```cmake
# 批准后目标 / PROPOSED — pending approval and implementation, NexoraConfig.cmake dependency.
include(CMakeFindDependencyMacro)
set(Kokkos_LAUNCH_COMPILER OFF) # Require the explicitly selected compiler.
find_dependency(Kokkos 5.2.2 EXACT CONFIG)
# Validate the recorded Kokkos profile/toolchain contract described below.
# Import NexoraTargets.cmake only after its required dependency is available.
```

```cmake
# 批准后目标 / PROPOSED — pending approval and implementation, downstream consumer.
cmake_minimum_required(VERSION 3.25.2)
project(nexora_consumer LANGUAGES CXX)
find_package(Nexora CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_compile_features(my_app PRIVATE cxx_std_20)
target_link_libraries(my_app PRIVATE nexora::simulation nexora::data)
```

The exact revision/digest must also be checked through the provisioned manifest: `EXACT` package version matching alone cannot detect patched source or the wrong compiler/backend ABI. The installed consumer must receive `Kokkos::kokkos` transitively; it must not hard-code raw CUDA/HIP libraries. GPU consumers must use a compatible device-capable compilation workflow and explicit architecture, not an ordinary host-only compiler simply because `find_package` succeeded. Required MPI package propagation, if an exported communication target references MPI, must likewise be tested offline.

<a id="cuda-build-contract"></a>
## First CUDA compilation and installed-consumer contract

**批准后目标 — the first CUDA path is `nvcc_wrapper` with CMake's `CXX` language.** It uses the same Kokkos-based `.cpp` numerical source and explicit template instantiations as CPU. Nexora may retain `LANGUAGES C CXX` for the MPI C API probe; its C compiler is an ordinary compatible host C compiler. CUDA kernels, private stream-interop test code and downstream numerical consumers are compiled as `CXX` with the wrapper. This does not add a native CUDA backend.

The first CUDA profile has exactly Kokkos **CUDA + Serial** enabled, one explicitly selected NVIDIA GPU architecture, C++20 without extensions, and CUDA relocatable device code disabled. Kokkos and Nexora are built in fresh, separate build directories. OpenMP + CUDA, HIP, CMake `CUDA`-language compilation, mixed-language Kokkos, Clang CUDA, compiler-launcher redirection, and RDC/device functions linked across translation units are outside this initial profile; each needs a separate reviewed build/consumer profile. The independent CPU OpenMP profile is proposed, not implemented. CMake 3.25.2 stays the common Nexora floor even though the selected wrapper route does not itself require Kokkos's stricter CUDA-language floor.

### Fixed-source basis

The following evidence is the official **5.2.2 tag associated with locked commit `8e15454876298832f013ebcaf08d53825b5bda40`**, not the `develop` examples returned by Context7:

| Source | Consequence for this profile |
| --- | --- |
| [Top-level language selection](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/CMakeLists.txt#L69) | Default compile language is CXX; enabling `Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE` with CUDA changes it. Keep that option and `Kokkos_ENABLE_MULTIPLE_CMAKE_LANGUAGES` OFF. |
| [nvcc_wrapper](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/bin/nvcc_wrapper) | Compiles `.cpp` through NVCC, takes the host compiler from `NVCC_WRAPPER_DEFAULT_COMPILER`, and selects NVCC through `CUDA_ROOT` or PATH. Set both explicitly; the wrapper's fallback architecture is not a Nexora target selection. |
| [Architecture handling](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/cmake/kokkos_arch.cmake#L936) | `Kokkos_ARCH_<GPU_TOKEN>=ON` selects one GPU architecture. Wrapper mode supplies compile/link architecture flags; exported `Kokkos_CUDA_ARCHITECTURES` records the numeric architecture. Multiple GPU architecture selections fail upstream. |
| [Installed package metadata](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/cmake/KokkosConfigCommon.cmake.in) | Exports `Kokkos_DEVICES`, `Kokkos_ARCH`, compiler identity/version, `Kokkos_CXX_STANDARD`, enabled options and `Kokkos_CUDA_ARCHITECTURES`; `kokkos_check` checks requested devices/architectures. |
| [Installed compiler selection](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/cmake/KokkosConfig.cmake.in#L28) | A non-wrapper consumer can otherwise trigger automatic launcher redirection. Nexora explicitly sets `Kokkos_LAUNCH_COMPILER=OFF` before discovery and requires the wrapper instead. `Kokkos_COMPILE_LANGUAGE` is exported. |

Select the host compiler and exact NVCC/toolkit/driver/OS versions when an actual runner is allocated; no exact runner versions or GPU model are known here. The minimum compiler table above is not an exact toolchain lock. Record those actual versions, host standard library/ABI, wrapper digest, target GPU token, toolkit path, and package digest in the provisioning manifest. Preserve the same `NVCC_WRAPPER_DEFAULT_COMPILER` and `CUDA_ROOT` during configure, build, tests and consumer builds; changing an environment variable must not silently change a previously configured toolchain.

### Configuration interface and profile discovery

**批准后目标:** remove `NEXORA_GPU_BACKEND` entirely. Passing it with *any* value, including `NONE`, must produce a migration error pointing to local Kokkos package selection; old presets must be replaced at cutover. Do not add a replacement provider enum. Default CPU provisioning selects a Serial-only package. CUDA is opt-in by explicitly selecting its verified local `Kokkos_DIR` or controlled prefix; finding an incidental system CUDA-enabled package without that explicit selection must not switch the default CPU profile. Discovery never downloads or upgrades a package.

Normalize and validate the **exported** `Kokkos_DEVICES` set after `find_package(Kokkos 5.2.2 EXACT CONFIG REQUIRED)`; do not infer capability from stale `Kokkos_ENABLE_*` cache values, the presence of a CUDA toolkit, or wrapper availability. The initial allowed sets are `{SERIAL}`, `{OPENMP,SERIAL}`, and `{CUDA,SERIAL}`. A CUDA package cannot be silently treated as the default CPU profile. Any other set is rejected until its profile is implemented and accepted. Selecting a package determines compiled instantiations; runtime method/space selection and validation status remain separate as specified in the Kokkos design.

For the CUDA set, require exported `Kokkos_COMPILE_LANGUAGE=CXX`, `Kokkos_CXX_STANDARD=20`, `Kokkos_CXX_COMPILER_ID=NVIDIA`, the selected single architecture, and the intended options (no mixed-language mode or RDC). Cross-check these against the provisioned manifest and imported target requirements. `kokkos_check` is a subset check, so it cannot by itself reject unwanted additional backends or options. A package with the correct version but wrong profile must fail.

The CUDA wrapper is the `CMAKE_CXX_COMPILER`, set before the first `project()` call via the configure command/toolchain file. For Nexora and its consumer use `<kokkos-cuda-prefix>/bin/nvcc_wrapper` from that verified installation. Its actual wrapper content and resolved host/NVCC toolchain must match the dependency manifest. Do not demand string equality with exported `Kokkos_CXX_COMPILER`, which may retain the source-tree wrapper path used to build Kokkos; do not fall back to that old source-tree path either. The installed package must work with the source and build trees unavailable.

`Kokkos_ARCH_<GPU_TOKEN>` is an input **only when building Kokkos**. Nexora and its downstream consumer inherit the corresponding compile/link flags through `nexora::data → Kokkos::kokkos` and validate exported architecture metadata. They do not reconfigure an installed package by setting Kokkos build options. `CMAKE_CUDA_ARCHITECTURES` is not the architecture selector for this CXX-wrapper profile; supplying it as a Nexora override is rejected with instructions to provision/select the matching Kokkos package. Also reject conflicting manually supplied `-arch`/`-gencode` options. For example, `AMPERE80` maps to `80`/`sm_80` in the pinned source; this illustrates the naming scheme and does not select the unknown runner's GPU.

A proposed discovery skeleton follows; the prose above additionally requires manifest, compiler and imported-target checks before acceptance. This is a documentation sketch, not a complete or executed implementation.

```cmake
# 批准后目标 / PROPOSED — pending approval and implementation.
project(Nexora LANGUAGES C CXX) # No CUDA language in this profile.
if(DEFINED NEXORA_GPU_BACKEND)
  message(FATAL_ERROR "NEXORA_GPU_BACKEND was removed; select a verified local Kokkos package")
endif()
if(DEFINED CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "This CXX-wrapper profile inherits architecture from installed Kokkos")
endif()
set(Kokkos_LAUNCH_COMPILER OFF)
find_package(Kokkos 5.2.2 EXACT CONFIG REQUIRED)
set(nexora_kokkos_devices "${Kokkos_DEVICES}")
list(TRANSFORM nexora_kokkos_devices TOUPPER)
list(REMOVE_DUPLICATES nexora_kokkos_devices)
list(SORT nexora_kokkos_devices)
if(NOT nexora_kokkos_devices STREQUAL "SERIAL"
   AND NOT nexora_kokkos_devices STREQUAL "OPENMP;SERIAL"
   AND NOT nexora_kokkos_devices STREQUAL "CUDA;SERIAL")
  message(FATAL_ERROR "Kokkos execution-space combination has no accepted Nexora build profile")
endif()
if("CUDA" IN_LIST nexora_kokkos_devices)
  kokkos_check(DEVICES CUDA SERIAL)
  if(NOT Kokkos_COMPILE_LANGUAGE STREQUAL "CXX"
     OR NOT Kokkos_CXX_STANDARD STREQUAL "20"
     OR NOT Kokkos_CXX_COMPILER_ID STREQUAL "NVIDIA")
    message(FATAL_ERROR "CUDA requires the matched C++20 nvcc_wrapper Kokkos installation")
  endif()
  # Then validate manifest, wrapper/toolchain, architecture, options and a
  # CXX compile/link probe using Kokkos::kokkos; no runtime-support claim yet.
endif()
```

### CUDA dependency, Nexora and consumer sequence

**批准后目标 — not executed.** All paths below are placeholders for verified local inputs. `<GPU_TOKEN>` is one token from the pinned Kokkos architecture list, selected against actual hardware and toolkit support. The example fixes `lib` and `bin` install directories so its package paths are unambiguous. It does not acquire dependencies or select a real runner version.

```sh
# 批准后目标 / PROPOSED — pending approval and implementation.
export NVCC_WRAPPER_DEFAULT_COMPILER="<absolute-host-cxx-path>"
export CUDA_ROOT="<verified-cuda-toolkit-prefix>"
cmake -S "<local-kokkos-5.2.2-source>" -B "<kokkos-cuda-build>" -G Ninja \
  -DCMAKE_CXX_COMPILER="<local-kokkos-5.2.2-source>/bin/nvcc_wrapper" \
  -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="<kokkos-cuda-prefix>" \
  -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_BINDIR=bin \
  -DBUILD_SHARED_LIBS=OFF \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_OPENMP=OFF -DKokkos_ENABLE_HIP=OFF \
  -DKokkos_ENABLE_SYCL=OFF \
  "-DKokkos_ARCH_<GPU_TOKEN>=ON" \
  -DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=OFF \
  -DKokkos_ENABLE_MULTIPLE_CMAKE_LANGUAGES=OFF \
  -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=OFF \
  -DKokkos_ENABLE_TESTS=OFF -DKokkos_ENABLE_EXAMPLES=OFF \
  -DKokkos_ENABLE_BENCHMARKS=OFF
cmake --build "<kokkos-cuda-build>"
cmake --install "<kokkos-cuda-build>"
cmake -S . -B "<nexora-cuda-build>" -G Ninja \
  -DCMAKE_C_COMPILER="<absolute-host-c-path>" \
  -DCMAKE_CXX_COMPILER="<kokkos-cuda-prefix>/bin/nvcc_wrapper" \
  -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_DIR="<kokkos-cuda-prefix>/lib/cmake/Kokkos" \
  -DKokkos_LAUNCH_COMPILER=OFF -DNEXORA_ENABLE_MPI=OFF \
  -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=OFF
cmake --build "<nexora-cuda-build>"
ctest --test-dir "<nexora-cuda-build>" --output-on-failure
cmake --install "<nexora-cuda-build>" --prefix "<nexora-cuda-prefix>"
cmake -S "<standalone-consumer-source>" -B "<consumer-cuda-build>" -G Ninja \
  -DCMAKE_CXX_COMPILER="<kokkos-cuda-prefix>/bin/nvcc_wrapper" \
  -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_BUILD_TYPE=Release \
  -DNexora_DIR="<nexora-cuda-prefix>/lib/cmake/Nexora" \
  -DKokkos_DIR="<kokkos-cuda-prefix>/lib/cmake/Kokkos" \
  -DKokkos_LAUNCH_COMPILER=OFF
cmake --build "<consumer-cuda-build>"
"<consumer-cuda-build>/my_app"
```

Use the downstream `LANGUAGES CXX` CMake example above; its `main.cpp` acceptance fixture must initialize Kokkos, construct the exported Nexora typed data, launch a small Kokkos CUDA operation, copy/observe its result correctly, and finalize in the documented lifetime order. This tests the installed numerical contract rather than linking an empty executable. The consumer uses only exported Nexora targets; transitive Kokkos flags and libraries must suffice. All of its C++ translation units use the wrapper for this initial profile. There is no silent switch to an ordinary host compiler or automatic `launch_compiler` component. A separate host-only mixed consumer is a later profile, not implied support.

`NexoraConfig.cmake` must find the exact Kokkos version before importing Nexora targets and compare the located package with Nexora's recorded execution-space set, compile language, C++ standard, architecture/options and toolchain ABI contract. A same-version Serial package cannot satisfy an installed CUDA Nexora package. The contract must survive a clean offline downstream build with source/build trees absent; any absolute source-wrapper path, unstated environment dependency, or need to link raw CUDA runtime libraries manually is a failure. Toolkit/host compiler environment is a declared provisioned input, not part of relocatable project sources. Private interop tests may discover/link the CUDA runtime explicitly; that remains outside the installed public physics contract.

### CUDA negative acceptance

**批准后目标 — none of these checks has run.** P1-02 package checks and P1-05/P1-10 device acceptance must include:

| Negative input / condition | Required outcome |
| --- | --- |
| `NEXORA_GPU_BACKEND=NONE`, `CUDA`, or any other old value | Configure rejects the removed interface and explains package selection; no ignored cache option |
| CUDA Nexora package with missing, wrong-version, Serial-only, HIP, mixed-language or mismatched-architecture Kokkos | Configure rejects before numerical targets build; never downloads or substitutes a backend |
| Ordinary host C++ compiler, bare NVCC in place of the selected wrapper, missing/different wrapper, or changed host/NVCC environment | Fail the declared compiler/toolchain contract; no implicit Kokkos launcher redirection |
| Missing manifest architecture, two GPU tokens, only `CMAKE_CUDA_ARCHITECTURES`, or conflicting `-arch`/`-gencode` | Reject provisioning/configuration; actual numerical compile/link commands must carry the single manifest architecture, not a wrapper default |
| C++17, inconsistent standard/library ABI, RDC ON, or unsupported device set/options | Reject the profile; version equality does not waive configuration compatibility |
| Consumer can build only while the original Kokkos/Nexora source or build tree exists | Installed-package acceptance fails; do not preserve accidental source-tree paths as requirements |
| Driver/device absent or incompatible, device test skipped, or CXX compile probe alone passes | Record compile-only or unavailable runtime; CUDA numerical/runtime acceptance remains unfulfilled |

## Configuration policy

| Target configuration | Required provisioned inputs | Intended acceptance level | Current Kokkos implementation evidence |
| --- | --- | --- | --- |
| Serial CPU | Locked Kokkos Serial + C++20 compiler | Required default | Limited slice implemented; see [validation](kokkos-cpu-slice-validation.md) for results |
| OpenMP CPU | Locked Kokkos OpenMP + compiler OpenMP runtime | Explicit CPU performance profile | Not implemented or run |
| CUDA | Locked Kokkos CUDA + toolkit/host compiler/driver + explicit GPU architecture | Optional initial GPU profile | Not implemented or run |
| HIP | Locked Kokkos HIP + ROCm/driver + explicit GPU architecture | Optional later profile | Not implemented or run |
| CPU + MPI | Above CPU inputs + MPI C API/runtime | Optional distributed profile | Not implemented or run |
| CUDA or HIP + MPI | Matching GPU inputs + MPI runtime | Optional; host-staging baseline first | Not implemented or run |

Except for the limited Serial CPU slice, these are future target combinations, not supported-runtime claims. Use one selected GPU backend per build; CUDA+HIP in one Nexora build is rejected by policy. Unsupported execution-space requests must fail explicitly. GPU-aware MPI/direct device buffers require separate capability and lifetime validation; ordinary MPI discovery does not establish that capability. The future Serial and OpenMP profiles use the same kernels. The legacy native CPU execution layer and its bootstrap consumer have been removed.

Execution-space selection belongs to the verified installed Kokkos configuration and approved Nexora instantiation policy. The [first CUDA contract](#cuda-build-contract) defines the replacement configuration interface: remove/reject `NEXORA_GPU_BACKEND`, derive the permitted profile from exported `Kokkos_DEVICES`, and validate compiler/architecture compatibility. HIP and GPU+MPI rows are later target configurations, not additional combinations accepted by the initial CUDA profile. Record enabled spaces, actual selected execution space and build/validation status separately.

## Evidence and acceptance

The detailed representative cases and task dependencies are owned by [phase-1 design review](phase-1-design-review.md), especially P1-01 (lock), P1-02 (build), P1-05 (CPU/CUDA instantiation), P1-10 (acceptance/performance), and P1-11 (cutover). This document adds build-specific acceptance:

- In an environment with network access disabled, configure/build/test/install the default Serial profile using only the verified local dependency prefix; run a clean installed-package consumer.
- Verify missing Kokkos, wrong version, incompatible backend/configuration, unsupported combinations and missing GPU architecture fail explicitly without native fallback or network access.
- Check build and install target graphs, public includes and transitive package dependencies against the new architecture; replace the bootstrap checker policy together with the new targets.
- For each accepted profile, preserve manifest, exact commands, exit statuses, passed/failed/skipped counts, build metadata, compiler/runtime versions, hardware/device details, and ISO 8601 timestamps with seconds and timezone.
- Distinguish **configured**, **compiled**, **executed**, **numerically validated**, and **performance measured**. A configuration flag proves only requested/available build capability; a skipped or missing-device test establishes no device execution. Record performance only with workload, synchronization, repetitions, tolerance and comparison baseline fixed.

See [CPU slice validation](kokkos-cpu-slice-validation.md) for actual results and limits. GPU and MPI numerical acceptance and full Phase 1 remain outstanding.
