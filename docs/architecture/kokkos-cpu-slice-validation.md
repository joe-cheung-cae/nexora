# 首个 Kokkos CPU 基础切片：实现与实测记录

记录时间：`2026-11-06T16:34:20+08:00`。最终离线复验区间：`2026-11-06T16:31:21+08:00` 至 `2026-11-06T16:31:38+08:00`。

维护者明确批准了这个有限切片的源码、CMake、presets、测试和文档修改。**此记录不批准或关闭完整 Phase 1。** 当前架构入口仍为 [overview](overview.md)。没有改变物理公式、默认值或容差，也未实现 Field/State、参与者事务、SPH/FVM 或 GPU/MPI 数值计算。

## 实现边界

- 必需 `find_package(Kokkos 5.2.2 EXACT CONFIG REQUIRED)`，没有下载代码或 native fallback。默认 CPU 只接受导出 `Kokkos_DEVICES=SERIAL` 的本地包；GPU/MPI 默认关闭。
- 使用 C++20、CMake 3.25.2+、Ninja。检查 64-bit、Kokkos C++20/CXX、compiler ID 和精确版本；安装配置还核对构建 Nexora 时的 compiler、Kokkos ARCH/OPTIONS/TPLS。来源、commit、digest 和自定义 stdlib/ABI flags 仍需核对制品记录，配置检查不证明任意 ABI 组合兼容。
- `nexora::data` 公开传递 `Kokkos::kokkos`；core 不增加 Kokkos 直接依赖。CLI 通过 Kokkos 报告编译能力。此时旧 data→exec_api、participant→data 等边仍属过渡状态，未完成 schema 拆分。
- 整数测试由应用局部 `Kokkos::ScopeGuard` 管理运行时；其后声明 execution instance、typed `Kokkos::View<int*, ExecutionSpace>` 和 host mirror，逆序析构保证它们先于 finalize 释放。`RangePolicy`/`parallel_for` 计算 32 个 `3*i-7`，实例 fence 后使用 **create_mirror** 分配独立 host 存储并 **deep_copy**，遍历校验每个元素；最后一项为 86，无浮点容差。
- 安装消费者编译同一份测试源码，计算 target 只链接 `nexora::data`，没有直接链接 Kokkos、添加其 include 路径或使用旧 CpuContext。旧 bootstrap consumer 保留为独立回归项。
- 新测试仅有局部 managed View；不存在借用外部 owner、通信或 checkpoint。它不覆盖这些后续契约。未增加 allocator、queue、event 或 backend 分派框架。
- 任何 `NEXORA_GPU_BACKEND` 值均报迁移错误；`CudaSmoke.cu` 保留为不参与构建的旧文件。OpenMP、CUDA/HIP 包明确拒绝，相关设计保留为后续目标。

## 依赖与机器配置

| 项目 | 实际值 |
|---|---|
| Kokkos release / commit | 5.2.2 / `8e15454876298832f013ebcaf08d53825b5bda40` |
| 官方源制品 | [kokkos-5.2.2.tar.gz](https://github.com/kokkos/kokkos/releases/download/5.2.2/kokkos-5.2.2.tar.gz) |
| 源制品 SHA-256 | `d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f` |
| 核验 | 与[官方 SHA-256 清单](https://github.com/kokkos/kokkos/releases/download/5.2.2/kokkos-5.2.2-SHA-256.txt)一致；`git ls-remote` 的 annotated tag `ec149afc3998d5a1a0f3db6e1639114234bfafa4` peeled 至锁定 commit |
| 清单 SHA-256 | `52e49a388d1b78fec33c913bd2ad820cbba3a0bae5fd4e517382ec807dfeb5b4` |
| 许可证 / patches | `Apache-2.0 WITH LLVM-exception`；无本地 patches；bundled desul、mdspan、gtest notices 见 [third_party](../../third_party/README.md) |
| 本地源 / 包 | `/private/tmp/nexora-kokkos-5.2.2/kokkos-5.2.2` / `/private/tmp/nexora-kokkos-5.2.2/install-release` |
| Kokkos 配置 | Release 静态库；Serial ON；OpenMP/CUDA/HIP/SYCL OFF；tests/examples/benchmarks OFF；C++20、extensions OFF |
| Kokkos_OPTIONS | `DEPRECATED_CODE_5;DEPRECATION_WARNINGS;IMPL_REF_COUNT_BRANCH_UNLIKELY;IMPL_MDSPAN;COMPLEX_ALIGN;LIBDL`；ARCH/TPLS 为空 |
| 主机 | Apple M2 Pro、arm64、12 physical/logical CPUs、16 GiB RAM；本测试使用 Serial |
| OS / SDK | macOS 27.2 build 26B123 / SDK 27.2 |
| 编译器 / stdlib | `/usr/bin/c++`，AppleClang `21.0.0.21000401`；libc++ `_LIBCPP_VERSION=220106`，`_LIBCPP_ABI_VERSION=1` |
| 工具 | CMake 4.4.3 / Ninja 1.13.2 |

同一 Release Kokkos 安装供 Nexora Debug 和 Release 使用，本机两者都执行通过；这不表示启用了 Kokkos Debug instrumentation。未独立验证上游签名，已完成的验证是官方摘要比对和 tag/commit 关联。依赖缓存位于 `/private/tmp`，是本地临时制品，需保管后才能作为长期离线输入；源码/大制品/二进制未加入 Git。

实际静态库摘要（SHA-256）：

```text
libkokkoscore.a       8ec81e18e997069104c9a299c4c5022a2cf17a757355391c58cb28f12643302c
libkokkoscontainers.a f8c3c8a4d115cf805869320ac6babf120234c4b8ca07334e785dbb7a0d0b9071
libkokkossimd.a       ae396c6ff1bbff0807480bbec511af97b8a56708d9f6f3a513932e1cf9f8de06
```

接口查证使用 Context7 `/kokkos/kokkos` 和 `/kitware/cmake`；前者返回 develop 示例，不能作为锁定版本证据。随后核对已校验 5.2.2 源码中的 [ScopeGuard](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/core/src/impl/Kokkos_ScopeGuard.hpp)、[View copy/mirror](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/core/src/Kokkos_CopyViews.hpp)、Serial/RangePolicy 和安装包元数据，并用本切片实际编译执行验证接口。

## 实际命令与结果

依赖在独立联网准备步骤下载和校验；该过程的完整命令已保留在本地 `results/kokkos-cpu-slice/dependency-evidence.txt`，可复用命令见 [build-and-validation](build-and-validation.md#current-serial-provisioning-and-proposed-package-examples)。配置/构建/安装只使用本地解包源，未使用 FetchContent 或 package-manager bootstrap。

Nexora 首次从 fresh Debug/Release 配置编译；以下每条 configure/build/ctest 命令均实际使用前缀 `/usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network*)'`，因此测试内的安装及消费者进程同样被禁止联网：

```sh
cd /Users/chao/workspace/repo/nexora
cmake --fresh --preset cpu-debug -DKokkos_DIR=/private/tmp/nexora-kokkos-5.2.2/install-release/lib/cmake/Kokkos
cmake --build --preset cpu-debug
ctest --preset cpu-debug --output-on-failure
cmake --fresh --preset cpu-release -DKokkos_DIR=/private/tmp/nexora-kokkos-5.2.2/install-release/lib/cmake/Kokkos
cmake --build --preset cpu-release
ctest --preset cpu-release --output-on-failure
python3 tools/check_dependencies.py --graph build/cpu-debug/nexora-target-graph.json
python3 tools/check_dependencies.py --graph build/cpu-release/nexora-target-graph.json
```

最终复验因安装配置和负例增加而重新执行 build/CTest/checker；CMake 重新生成，计算源码无需再次编译。每项命令的最终秒级起止时间、参数、工作目录及 exit code 见本地 `results/kokkos-cpu-slice/commands.json`。

| 验证 | 实际结果 |
|---|---|
| CPU Debug | 配置、编译、执行通过；11 passed / 0 failed / 0 skipped；最终 CTest `2026-11-06T16:31:21+08:00`–`2026-11-06T16:31:28+08:00` |
| CPU Release | 配置、编译、执行通过；11 passed / 0 failed / 0 skipped；最终 CTest `2026-11-06T16:31:28+08:00`–`2026-11-06T16:31:35+08:00` |
| Kokkos 整数路径 | 每个 profile 的源码测试和安装消费者都验证全部 32 项；输出 `Verified 32 Kokkos integer results on Serial. Final value 86` |
| 安装 package CTest | 每个 profile 安装并搬迁 Nexora 前缀；独立 consumer 项目 2/2，通过旧消费者及新的 Kokkos 消费者；它们是顶层 11 项中 package 项的子测试 |
| 错误配置 | 每个 package 测试运行 13 个拒绝检查，全部命中预期诊断；见下表 |
| 依赖检查 | Debug/Release target graph、include 边界、允许/禁止 Kokkos 边的 self-checks 全部通过 |
| 双前缀隔离消费 | 单独安装 Release，搬迁 Nexora 和 Kokkos 前缀，在网络及原始路径不可读条件下 configure/build/run：2/2 passed，0 failed/skipped |
| CLI | `cxx_language=202002`，`cpu_execution=kokkos_serial`，`kokkos_version=5.2.2`，`kokkos_execution_space=Serial`，`cuda_smoke=off`，`mpi_smoke=off`；这是编译元数据，运行证据来自上述测试 |

所有表中成功操作 exit code 为 0；拒绝用例的 configure 必须非零，并同时检查具体诊断，不能用任意失败冒充成功。

| 输入 | 覆盖及预期拒绝 |
|---|---|
| 缺 Kokkos | 隔离 CMake package 搜索，仅提供空 fixture root；源码/安装消费者各 1 项，报找不到 Kokkos 配置 |
| 错误版本 | 模拟 5.2.1 ConfigVersion，源码/安装消费者各 1 项，版本不符合 EXACT 5.2.2 |
| C++17 | 使用真实 Kokkos 包，源码/安装消费者各 1 项，报 requires C++20 |
| OpenMP+Serial | 模拟 5.2.2 的错误设备集合，源码/安装消费者各 1 项，报 Serial-only |
| Kokkos C++23 | 模拟同版本错误 standard 元数据，源码/安装消费者各 1 项，报 C++20/CXX |
| Kokkos compiler version 0.0.1 | 模拟同版本不匹配 compiler 元数据，源码/安装消费者各 1 项，报 compiler identity/version |
| 旧 `NEXORA_GPU_BACKEND=NONE` | 源码 1 项，报 was removed |

模拟包只用于验证拒绝行为，未构建其他 Kokkos 版本、OpenMP 或不匹配 compiler 的真实库；不宣称这些组合曾执行。

独立隔离消费使用本地 `/private/tmp/nexora-cpu-slice-validation`（以下简写为 `V`）。先执行 `cmake --install build/cpu-release --prefix "$V/nexora-installed"`（网络禁止，exit 0），然后重命名为 `nexora-relocated`；复制 Kokkos 安装为 `kokkos-relocated`，将消费者 CMake/main 和同一整数测试复制到 `consumer-source`。之后以下命令均通过 `sandbox-exec -f "$V/detached.sb"` 执行：

```sh
cmake -S "$V/consumer-source" -B "$V/consumer-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DNexora_DIR="$V/nexora-relocated/lib/cmake/Nexora" \
  -DKokkos_DIR="$V/kokkos-relocated/lib/cmake/Kokkos"
cmake --build "$V/consumer-build"
ctest --test-dir "$V/consumer-build" --verbose
```

`detached.sb` 内容：

```text
(version 1)
(allow default)
(deny network*)
(deny file-read* (subpath "/Users/chao/workspace/repo/nexora"))
(deny file-read* (subpath "/private/tmp/nexora-kokkos-5.2.2"))
```

同一 sandbox 中先用 Python 读取原 `CMakeLists.txt` 和原 Kokkos header、连接本机 socket，三者均得到 PermissionError，证明禁止策略生效。没有删除或隐藏工作区。隔离消费完成于 `2026-11-06T16:31:38+08:00`。

## 修改文件

| 文件 | 修改用途 |
|---|---|
| `CMakeLists.txt`、`CMakePresets.json` | C++20/CMake floor、必需 exact 包、Serial profile、旧选项拒绝、新测试接入 |
| `cmake/NexoraKokkos.cmake`（新增）、`cmake/NexoraConfig.cmake.in` | 共用 profile 校验、安装 find_dependency 与构建 compiler/options 对照 |
| `src/data/CMakeLists.txt` | Kokkos 公开传递依赖 |
| `apps/nexora/CMakeLists.txt`、`apps/nexora/main.cpp` | CLI 链接与 Kokkos 编译元数据，停用旧 CUDA source |
| `tests/kokkos/CMakeLists.txt`、`tests/kokkos/kokkos_tests.cpp`（新增） | typed View 整数计算及全部元素验证 |
| `tests/package/CMakeLists.txt`、`tests/package/VerifyInstall.cmake`、`tests/package/consumer/CMakeLists.txt` | 搬迁安装消费、复用计算测试、13 个负例 |
| `tools/check_dependencies.py` | Kokkos 允许/必需 target 边、头边界和自检 |
| `AGENTS.md`、`README.md`、`CONTRIBUTING.md`、`third_party/README.md` | 当前有限批准、实际构建方法、来源/摘要/许可证 |
| `docs/architecture/overview.md`、`build-and-validation.md`、`migration-roadmap.md`、`phase-1-design-review.md`、本文件（新增） | 当前实现、后续目标、旧层和实测证据分开记录 |

## 保留项与未验证项

- 旧 `Buffer`、`ExecContext/CpuContext`、`CompletionToken`、`MemorySpace`、`FieldView`、descriptor/registry、单参与者 Simulation 及相关源码/导出/旧测试暂留，未扩展。Kokkos 计算不调用它们；它们通过不计作新路径验证。
- 未完成 typed 字段语义、owner/lease、accepted/trial、通信、checkpoint、SPH/FVM 代表性切片、旧 exec 删除或完整 Phase 1。无物理数值/浮点容差验证，无性能测量。
- 未验证 OpenMP、CUDA/HIP、MPI、GCC/Linux、MSVC/Windows 或其他 AppleClang/macOS 组合。现有 CI 未增加依赖供给，未运行且不能作为此切片证据；全新 CI runner 当前需要先准备 Kokkos，旧 Windows GPU 参数需后续迁移。
- 本机 Kokkos 的 upstream tests 未启用。本记录的 numerical validation 仅指精确整数基础设施测试；没有把编译或元数据当作运行证明。
- Nexora 自身许可证和 legacy 复用权利仍未解决。本次只记录 Kokkos 自身许可证，不据此推断项目授权。

本地完整原始日志保存在忽略目录 `results/kokkos-cpu-slice/`；Git 中此记录保留必要命令、配置、摘要与结果。工作止于此 CPU 基础切片。
