# 字段迁移提交前复验

完整工作树验证完成时间：`2026-11-11T10:50:26+08:00`。本轮审阅字段授权、外部 owner、本地 pack/unpack、reference accepted/trial 生命周期、real 精度及安装导出，未发现需要修复的本次变更缺陷。没有修改数值公式或验收阈值；本记录只覆盖下述 Serial CPU 配置。

## 本地依赖与工具链

原 `/private/tmp/nexora-kokkos-5.2.2` 缓存已不存在。经维护者授权，重新下载官方 Kokkos 5.2.2 release 归档和摘要清单，保存在已被 Git 忽略的 `.cache/dependencies/kokkos-5.2.2/`。配置、编译和测试使用本地安装，不隐式获取依赖。没有另加禁网沙箱。

- 来源：[官方归档](https://github.com/kokkos/kokkos/releases/download/5.2.2/kokkos-5.2.2.tar.gz)，SHA-256 `d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f`，与本轮下载的官方清单及仓库锁一致。清单 SHA-256 为 `52e49a388d1b78fec33c913bd2ad820cbba3a0bae5fd4e517382ec807dfeb5b4`。
- commit 关联沿用[来源记录](../../third_party/README.md)：`8e15454876298832f013ebcaf08d53825b5bda40`。许可 `Apache-2.0 WITH LLVM-exception`，保留上游及 bundled dependency notices，无本地补丁。
- Kokkos 为 Release 静态安装，Serial ON，OpenMP/CUDA/HIP/SYCL 和上游 tests/examples/benchmarks OFF；C++20，extensions OFF。配置、构建、安装在 `2026-11-11T10:49:25+08:00` 至 `2026-11-11T10:49:31+08:00` 完成，均退出 0。
- 本轮安装的 `libkokkoscore.a` SHA-256 为 `b02d99a10f3397d9e2ee8e119ba7f219717e422fe44cac1db7370c3506765a98`。完整库摘要、确切 configure 参数和环境见本机 `results/current-change-review/manifest.json` 与 `dependency/commands.json`；新构建制品不沿用旧临时安装的二进制摘要。
- 硬件：Apple M2 Pro、arm64、12 physical/logical CPUs、16 GiB RAM。OS 为 macOS 27.2.1 build 26B131；AppleClang 21.0.0 (`clang-2100.4.1.1`)，CMake 4.4.3、Ninja 1.13.2、Python 3.14.7。硬件读数另存 `hardware.json`。

## 完整工作树验证

四个新 build 目录为 `build/current-review-<precision>-<mode>`；`Kokkos_DIR` 指向仓库内 `.cache/dependencies/kokkos-5.2.2/install-release/lib/cmake/Kokkos`。float 使用默认选项，double 显式设置 `NEXORA_ENABLE_DOUBLE_PRECISION=ON`，MPI 均关闭。

```sh
cmake -S . -B <build> -G Ninja -DKokkos_DIR=<local-prefix>/lib/cmake/Kokkos \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ -DCMAKE_BUILD_TYPE=<Debug-or-Release> \
  -DCMAKE_CXX_EXTENSIONS=OFF -DBUILD_TESTING=ON -DNEXORA_ENABLE_MPI=OFF
cmake --build <build> --parallel 2
python3 tools/check_dependencies.py --graph <build>/nexora-target-graph.json
ctest --test-dir <build> --output-on-failure
python3 tools/check_reference_precision.py --log <build>/Testing/Temporary/LastTest.log \
  --precision <float-or-double> --output <report.json>
```

| 配置 | 配置开始 / 验证完成 | 配置、构建、依赖检查、Decimal | CTest |
|---|---|---|---|
| float Debug | `2026-11-11T10:50:02+08:00` / `2026-11-11T10:50:23+08:00` | 均退出 0 | 15 passed / 0 failed / 0 skipped，13.11 s |
| double Debug | `2026-11-11T10:50:02+08:00` / `2026-11-11T10:50:23+08:00` | 均退出 0 | 15 passed / 0 failed / 0 skipped，13.11 s |
| float Release | `2026-11-11T10:50:02+08:00` / `2026-11-11T10:50:26+08:00` | 均退出 0 | 15 passed / 0 failed / 0 skipped，13.49 s |
| double Release | `2026-11-11T10:50:02+08:00` / `2026-11-11T10:50:26+08:00` | 均退出 0 | 15 passed / 0 failed / 0 skipped，13.49 s |

每项 CTest 包含干净安装、搬迁和三个独立消费测试，以及精度宏/CMake 冲突拒绝、旧接口删除、schema 依赖边界、字段 lease、外部 owner、收发、失败恢复和重试。Decimal 使用 80 位精度，独立检查 16/32/64 步及非二进制步长；最大离散分量误差为 float `1.0456617304142321e-7`、double `1.9918532962658262e-16`，守恒及观测阶继续通过既定判据。

另执行 `git diff --check`、Python 解析、依赖策略自检和当前 include 边界检查，均通过。私有 allocation tracker 适配器已对照本轮校验的 Kokkos 5.2.2 头文件核对：managed subview 共享原分配记录，record 的 base/size 和 tracker 保活行为符合当前实现的使用方式。

## 提交拆分

精度配置先独立提交，字段与 reference 的接口迁移和配套回归测试保持同一提交；原生 View 数值测试及 Decimal 检查器作为测试提交，架构与验证记录作为文档提交。字段描述符与 reference 调用方相互依赖，不制造临时兼容接口。

第一个精度配置提交 `b66d08f` 的完整源码快照已与实际提交逐文件比对一致。其 float Debug 和 double Release 从 `2026-11-11T10:51:46+08:00` 至 `2026-11-11T10:51:59+08:00` 分别通过配置、构建、依赖检查和 12/12 CTest；此时 reference 尚保留原 double 数值状态，后续提交完成 real 迁移。

字段与 reference 迁移提交 `953ab99` 的精确暂存快照另在 `2026-11-11T10:53:47+08:00` 至 `2026-11-11T10:54:01+08:00` 完成 float Debug 配置、构建、依赖检查和 15/15 CTest（10.45 s），包含搬迁安装消费。中间状态日志位于 `results/current-change-review/data-intermediate/`。测试提交 `012ca27` 加入原生 View 和独立 Decimal 检查器后，代码与上表四配置验证的完整工作树一致。

本机原始日志、逐步命令及秒级时间保存在被忽略的 `results/current-change-review/`。本轮已配置、编译、执行并数值验证上述 Serial CPU 组合；未验证 OpenMP、CUDA/HIP、MPI 数值通信、持久 checkpoint、完整 P1 或产品 solver。
