# Serial CI 验证

对应 [开发计划任务 1](foundation-development-plan.md#任务-1-固定基线并恢复-ci)，基线为 `51dd068dc339ebb33cadb44968fc464e1ca4481a`。本次验证包含任务 1 和任务 2 的工作树改动。

## 依赖供应和 CI

[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) 分别配置 Linux GCC 13、Linux Clang 18 和 Windows Visual Studio 2022 MSVC x64 的 Debug 和 Release。三个编译器均使用 Ninja。每个构建类型运行 float 和 double。Windows 通过 Developer PowerShell 选定 amd64 编译器和目标。独立 Linux/GCC Debug job 运行可选的两进程 MPI C API smoke。 [Visual Studio 工具链设置](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell?view=vs-2022)

[验证脚本](../../tools/validate_serial.py) 先把官方 Kokkos 5.2.2 源归档与锁定的 SHA-256 比对。使用 `--download` 时也获取官方摘要清单，确认归档和摘要清单后再使用。源码 commit `8e15454876298832f013ebcaf08d53825b5bda40` 与归档的关联沿用已核验的[依赖记录](../../third_party/README.md#kokkos-provenance-record)。脚本保留上游许可和版权文件。源码无补丁，以 C++20、Serial-only、静态库安装，保留编译器、构建选项和完整安装文件的 SHA-256。cache key 纳入操作系统、架构、编译器二进制与版本、工具链输入、构建选项、锁定源码摘要及脚本版本。每个 build type 独立供应 Kokkos，MSVC 的 Debug 和 Release 库与各自的 CRT 配置匹配。

当前验证脚本也供应锁定的 GoogleTest 1.17.0，按相同编译器和构建类型分别安装。验证阶段从本地 `Kokkos_DIR` 和 `GTest_DIR` 加载依赖。GoogleTest 用例通过 CTest 逐项发现和执行。[迁移与开发计划](googletest-development-plan.md)

验证脚本依次执行配置、编译、依赖图检查、测试和独立 Decimal oracle。最后由包测试执行干净安装、搬迁、独立消费者构建与运行，并检查拒绝路径。GitHub Actions 的 Linux Serial job 用 `unshare --net` 创建隔离网络命名空间，降回 runner 用户执行验证。Windows Serial 和 MPI smoke job 使用本地依赖。

## 受控 runner 实测

Apple M2 Pro arm64，macOS 27.2.1 build 26B131，AppleClang `21.0.0.21000401`，CMake `4.4.3`，Ninja `1.13.2`，Python `3.14.7`。MPI 关闭。

两个本地安装使用同一个官方归档，SHA-256 为 `d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f`，摘要清单 SHA-256 为 `52e49a388d1b78fec33c913bd2ad820cbba3a0bae5fd4e517382ec807dfeb5b4`。源码摘要在 `2026-11-11T11:30:48+08:00` 完成核验，Debug 和 Release 安装分别在 `2026-11-11T11:30:55+08:00` 与 `2026-11-11T11:30:56+08:00` 完成。两个安装各记录 599 项文件摘要。

| Kokkos 构建类型 | `lib/libkokkoscore.a` SHA-256 |
| --- | --- |
| Debug | `4e84741a6c89114f47b42347dd226f7f9b0b92cdb84e8642d2d1dbd68b7585e1` |
| Release | `aa3316d9745fc105ade0a63ac594ac2d21b4e0dfc7a65941621f013ccfc9c318` |

验证运行于 Codex 受控执行沙箱。每次 `verify` 的同一进程在开始前与结束后检测 IPv4、IPv6 的本地和远端 TCP 连接、UDP 发送以及回环绑定。每轮 10 个操作均返回 `EPERM`，配置、编译、测试、安装及消费者进程继承该网络限制。另测嵌套 `sandbox-exec`，系统拒绝应用第二层沙箱。

| Profile | 配置开始至搬迁包测试完成 | Nexora CTest | 独立消费者 | Decimal oracle |
| --- | --- | --- | --- | --- |
| float Debug | `2026-11-11T11:30:55+08:00` 至 `2026-11-11T11:31:15+08:00` | 16/16 | 3/3 | 通过 |
| double Debug | `2026-11-11T11:31:15+08:00` 至 `2026-11-11T11:31:37+08:00` | 16/16 | 3/3 | 通过 |
| float Release | `2026-11-11T11:30:56+08:00` 至 `2026-11-11T11:31:19+08:00` | 16/16 | 3/3 | 通过 |
| double Release | `2026-11-11T11:31:19+08:00` 至 `2026-11-11T11:31:42+08:00` | 16/16 | 3/3 | 通过 |

各组配置、31 个 Ninja 步骤、依赖图检查及验证命令退出码均为 0，CTest 失败和跳过均为 0。80 位 Decimal oracle 覆盖 16/32/64 步及非二进制步长，最大离散分量误差为 float `1.0456617304142321e-7`、double `1.9918532962658262e-16`，采用[现有判据](real-numerics-validation.md)。Kokkos 配置报告未使用 `CMAKE_C_COMPILER`，该依赖只声明 CXX。

每个 profile 的搬迁包检查验证三项独立消费者，并在干净目录执行配置拒绝用例。

| 输入 | 验证结果 |
| --- | --- |
| 缺少 Kokkos | Nexora 与已安装消费者均报找不到 exact 5.2.2 配置 |
| Kokkos 5.2.1 | Nexora 与已安装消费者均报告候选版本 5.2.1 不匹配 |
| `NEXORA_GPU_BACKEND=NONE` | Nexora 配置明确报告旧选项已移除 |
| 消费者请求与安装包相反的精度 | 消费者配置报 `Nexora precision mismatch` |
| C++17 或同版本的非 Serial、错误标准、错误编译器配置 | Nexora 与已安装消费者均报告对应的不兼容项 |

归档与安装完整性也分别验证。篡改的归档于 `2026-11-11T11:31:16+08:00` 因 SHA-256 不符在安装前被拒绝。复制安装后改动 `Kokkos_Core.hpp`，于 `2026-11-11T11:32:02+08:00` 因安装文件摘要不符在配置前被拒绝。

本地日志保存在 Git 忽略的 `results/serial-ci-20261111/`，包含命令、时间戳、退出码、依赖 manifest、JUnit、Decimal 输出和负例日志。工作流运行时以每个 job 的 artifact 保留相同输出。Linux GCC、Linux Clang、Windows MSVC 及 Linux MPI 为已配置的工作流 profile，本地无 Git remote，未触发 GitHub Actions。OpenMP、CUDA 和 HIP 继续按各自验收。

review 发现 job 级 `env` 无法引用 `runner.temp`。工作流改为在首个步骤用 `$RUNNER_TEMP` 生成路径并写入 `$GITHUB_ENV`。于 `2026-11-11T11:34:24+08:00` 验证旧写法被本地作用域检查拒绝，新写法通过作用域和 Bash 语法检查。带空格的 POSIX 路径和带盘符的 Windows 路径模拟均覆盖供应、验证、cache 和 artifact 的路径传递。 [GitHub Actions 上下文表](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability) · [Runner 的环境变量合并](https://github.com/actions/runner/blob/main/src/Runner.Worker/StepsRunner.cs)

修复后于 `2026-11-11T11:35:03+08:00` 至 `2026-11-11T11:35:25+08:00` 用工作流的实际路径设置脚本、带空格的临时路径和缓存的 Release Kokkos 安装复跑 double Release。依赖图、CTest 16/16、独立消费者 3/3、Decimal 和禁网探针均通过，日志位于 `workflow-path-retest/`。

本地另用 Open MPI 5.0.10 对可选 smoke 作诊断，使用同一个 Debug Kokkos 安装及 `verify --precisions float --mpi`。在 `2026-11-11T11:32:31+08:00` 至 `2026-11-11T11:32:44+08:00` 完成配置、编译和依赖图检查。15 个运行测试通过 14 个，`nexora_mpi_smoke` 的启动器在启动 ranks 前绑定 socket 返回 `EPERM`，CTest 退出码为 8。此轮停在运行测试，日志位于 `logs-mpi-diagnostic/debug-float/ctest.log`。

## 重跑

准备官方归档。需要网络获取时在 `provision` 中显式加 `--download`。在网络隔离环境内执行 `verify`，保留 `--require-network-blocked` 以使探针失败时停止验证。以下按当前的两个归档和本机工具链演示，按 profile 使用新目录。只执行一组精度时传 `--precisions float` 或 `--precisions double`。

```sh
nexora_run=$(mktemp -d /tmp/nexora-serial-ci.XXXXXX)
nexora_archive="$PWD/.cache/dependencies/kokkos-5.2.2/kokkos-5.2.2.tar.gz"
gtest_archive="$PWD/.cache/dependencies/googletest-1.17.0/googletest-1.17.0.tar.gz"

for config in Debug Release; do
  python3 tools/validate_serial.py --cc /usr/bin/cc --cxx /usr/bin/c++ --config "$config" \
    provision --archive "$nexora_archive" --gtest-archive "$gtest_archive" \
    --prefix "$nexora_run/kokkos-$config" --gtest-prefix "$nexora_run/gtest-$config" \
    --logs "$nexora_run/logs" --jobs 2 || exit
  python3 tools/validate_serial.py --cc /usr/bin/cc --cxx /usr/bin/c++ --config "$config" \
    verify --prefix "$nexora_run/kokkos-$config" --gtest-prefix "$nexora_run/gtest-$config" \
    --work-dir "$nexora_run/build" \
    --logs "$nexora_run/logs-$config" --jobs 2 --require-network-blocked || exit
done
```
