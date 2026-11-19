# GoogleTest 接入与测试开发计划

本轮覆盖 Nexora 现有的 14 个 C++ 测试程序。开发者可以用 CTest 单独运行一次字段授权失败、一次 Reference 回滚或一个 SPH 数值基准，直接看到失败的断言。CLI 的命令行行为、MPI smoke 和可搬迁安装包继续由 CTest 启动真实进程检查。

## 实施与后续安排

| 顺序 | 工作与交付 | 完成条件 | 状态 |
| --- | --- | --- | --- |
| 1. 固定依赖 | 供应 GoogleTest 1.17.0，记录源码 commit、归档 SHA-256、许可证、编译器及安装文件摘要。默认 CPU 测试从本地 CMake config 包加载。 | 缺包或版本不符时，测试配置失败并指出原因。关闭测试时可独立构建 CLI 和库。 | 本机通过 |
| 2. 迁移现有场景 | 将 core、data、simulation、Kokkos、reference、SPH 的行为检查改成具名 GoogleTest 用例。数值程序共用一个管理 Kokkos 生命周期的入口。初始化前的 3 个拒绝用例分别进入 data 和 reference 的独立程序。 | Release 仍执行断言。每个用例可单独筛选。精度专属场景只在对应构建中注册。 | 本机通过 |
| 3. 验收本机 Serial | 在 Debug/Release 下分别运行 float/double。对照原有 Decimal oracle 和 SPH 固定基准，验证数据、依赖图和安装消费者。 | 四组配置均通过；安装包搬迁后，3 个外部消费者在禁用 GTest 查找时编译并运行。 | 本机通过，结果见下表 |
| 4. 验收托管 CI | 在包含本变更的 PR 上运行既有 Linux GCC、Linux Clang、Windows MSVC 矩阵和独立 Linux MPI smoke。 | 核对每个 job 的 JUnit、精度报告和依赖清单。Linux Serial 核对禁网探针，MSVC 核对 Debug/Release CRT，MPI smoke 核对两个 rank 的执行结果。 | 7/7 通过，见下方托管 CI 记录 |
| 5. 随功能增量开发 | 模块作者将后续修复的最小复现场景加入所属 suite。对状态、拓扑、归约的改动，复用对应模块现有夹具和独立期望值。 | 修复类 PR 能单跑用例重现旧行为，并在修复后通过。涉及精度或安装契约的变更复跑四配置矩阵。 | 按后续 PR 执行 |

步骤 4 的产物用于判断相应平台是否通过。涉及新设备后端或物理方法的验收，继续使用对应项目任务中的实机和数值标准。[Phase 1 验收要求](phase-1-design-review.md)

## 新增或定位用例

1. 在现有模块测试文件中添加 `TEST(Suite, Behavior)`。先用 `ASSERT_*` 检查后续读取所依赖的前提，再检查结果。保留数值基准规定的误差界。对运算或比较的诊断输出，可在断言后使用 `<<` 附上 stable ID、索引或输入值。
2. 新建测试程序时使用 `nexora_add_test`。需要已初始化 Kokkos 的程序传 `KOKKOS`，由公共入口建立 `Kokkos::ScopeGuard`。维护新增目标的直接依赖检查。安装消费者只链接要验证的导出目标。
3. 构建后用 `-N -L gtest` 列出 GoogleTest 用例，用 `-R` 按名选择。合并前运行完整 CTest preset，以覆盖 CLI 和安装包。[GoogleTest 断言](https://google.github.io/googletest/primer.html) · [CTest 用例发现](https://cmake.org/cmake/help/v3.25/module/GoogleTest.html)

以下命令在已供应依赖的 checkout 中先复跑单个场景，再运行整套测试。

```sh
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug -N -L gtest
ctest --preset cpu-debug -R '^nexora_reference_tests\.Reference\.EulerConvergence$'
ctest --preset cpu-debug
```

## 本机验收记录

基于 `40970c1` 的工作树，于 `2026-11-19T13:40:01+08:00` 至 `2026-11-19T13:41:09+08:00` 运行。使用 macOS 27.2.1 arm64、Apple M2 Pro、AppleClang 21.0.0.21000401、CMake 4.4.3、Ninja 1.13.2、C++20 和 Kokkos 5.2.2 Serial。验证脚本使用 `--require-network-blocked --measure-irregular`，在验证前后均确认网络受限。本地原始日志位于 `results/gtest-serial/{Debug,Release}/`。[离线供应与四配置复跑命令](serial-ci-validation.md#重跑)

| 配置 | GoogleTest 用例 | CTest 合计 | Decimal oracle | 搬迁后独立消费者 |
| --- | ---: | ---: | --- | ---: |
| Debug / float | 69 | 76/76 | 通过 | 3/3 |
| Release / float | 69 | 76/76 | 通过 | 3/3 |
| Debug / double | 70 | 77/77 | 通过 | 3/3 |
| Release / double | 70 | 77/77 | 通过 | 3/3 |

float 独有一个 Reference 步长超出可表示范围的场景，double 独有两个接近 `DBL_MAX` 的 SPH 归约场景。每组另含 7 项原生 CTest 检查，3 个消费者计在其中的安装包检查内。四组依赖图检查与独立非规则结构计量均通过。单独配置 `BUILD_TESTING=OFF` 且禁用 GTest 查找的 Release 构建也通过了编译与依赖图检查。

## 托管 CI 验收记录

提交 `d0f1e75` 的 PR 验证于 `2026-11-19T13:48:07+08:00` 启动，最后一个任务于 `2026-11-19T13:52:54+08:00` 通过。JUnit、Decimal 报告和两份依赖清单均已逐任务核对。每个精度配置都通过了 3 个搬迁安装消费者。[GitHub Actions 运行与验收产物](https://github.com/joe-cheung-cae/nexora/actions/runs/44467644385)

| 托管环境 | 编译器 | 构建类型 | 每种构建的 CTest 结果 |
| --- | --- | --- | --- |
| Ubuntu 24.04 x86_64 | GCC 13.3.0、Clang 18.1.3 | Debug、Release | float 76/76，double 77/77 |
| Windows Server 2022 x64 | MSVC 19.44.35230 | Debug、Release | float 76/76，double 77/77 |
| Ubuntu 24.04 x86_64，Open MPI | GCC 13.3.0 | Debug | float 76/76，含双进程 smoke |

四个 Linux Serial 任务的网络探针在验证前后均通过。Windows Debug 和 Release 使用各自配置的 Kokkos 与 GoogleTest，GoogleTest 采用匹配的动态 CRT。MPI 任务实际执行 `mpiexec -n 2`，两个 rank 均满足 `MPI_THREAD_FUNNELED`，归约和为 3，程序成功完成 `MPI_Finalize`。[双进程 MPI 任务](https://github.com/joe-cheung-cae/nexora/actions/runs/44467644385/job/135369170602)
