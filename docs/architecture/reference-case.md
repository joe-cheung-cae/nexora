# Serial Reference case v1

`nexora validate <case>`、`nexora plan <case>` 和 `nexora run <case>` 接受同一份 Reference case。三条命令共用读取、解析和预检。该入口对应 [P1-09 的 Serial 子集](phase-1-design-review.md#2-实施任务与依赖)。

## 文件格式

每行一项 `key=value`。键区分大小写，键和值两端允许空白。支持空行和以 `#` 开头的整行注释。逗号分隔的列表按文件顺序绑定稳定 ID 和初值。输入错误包含文件名和从 1 开始的行号，三条命令使用相同的诊断。

| 键 | v1 约束与默认值 |
| --- | --- |
| `version` | 必填，值为 `1`。 |
| `fixture` | 必填，值为 `reference`。 |
| `end` | 必填，有限 `double`，大于 `start`。 |
| `step` | 必填，有限正 `double`，须满足 Reference 的时间步约束和当前 `real` 精度。 |
| `start` | 有限 `double`，默认采用 `Simulation` 的 `0`。 |
| `lambda` | 有限正 `real`，默认采用 Reference `Config` 的 `1`。 |
| `ids` | 非空、唯一的无符号 64 位整数列表。默认 `0`。 |
| `x` | 有限 `real` 初值列表，默认 `1`。 |
| `q` | 有限 `real` 初值列表，默认 `0`。 |
| `execution` | 默认 `serial`，该版本支持本地 Kokkos Serial。 |
| `precision` | `float` 或 `double`，默认采用当前安装包的 `nexora::real`。显式请求必须与构建匹配。 |

`ids`、`x`、`q` 的默认值来自 Reference `Config`，三个列表必须等长。`x`、`q` 和 `lambda` 的十进制值直接解析为构建的 `real`。未知键、重复键、未知版本和不受支持的执行或精度请求会被拒绝。Reference v1 拒绝 `quantum` 配置。

```text
version=1
fixture=reference
end=0.35
step=0.125
lambda=2
ids=10,30
x=1,2
q=0,1
```

## 计划和运行

预检使用 Reference 与 `Simulation` 的现有时间步约束。每一步先按请求步长计算舍入后的 `double` 端点。若该端点恰好为 `end`，保留原请求步长。若端点超过 `end`，以 `end - current` 形成末步并重新校验实际端点及步长。不可推进、不能抵达目标或违反 `real` 数值步长约束的计划在预检拒绝。`run` 按预检得到的请求步长推进，积分仍使用 `TimeInterval::duration()`。v1 预检最多保留 1,000,000 个请求步。超限返回 `resource_exhausted`。

`validate` 在通过时输出 `valid=true`。`plan` 分别报告 `built`、`available`、`validated`，并列出参与者、实体数、字段精度与布局、授权、复制和同步边界。`validation_evidence=task6-local-serial` 表示构建配置与下方有实测记录的 profile 匹配。匹配条件为原生 Darwin 27.2.0 arm64、AppleClang `21.0.0.21000401`、默认 Ninja Debug 或 Release、默认架构和 SDK 选择、MPI 关闭及 exact Kokkos 5.2.2 Serial，覆盖 float 和 double。自定义工具链、编译或链接参数、编译器 launcher 与 IPO 属于单独的构建组合。实测源码与所用依赖制品的摘要记录在验收证据中。

其他构建报告 `validated=false`，普通 `run` 会在预检后拒绝运行。启用 `BUILD_TESTING` 的构建提供 `nexora run --fixture <case>`，用于显式执行受控夹具并收集证据。普通运行在已匹配的 profile 上直接使用 `nexora run <case>`。

`run` 在应用的 Kokkos `ScopeGuard` 内创建 Reference 和 `Simulation`，将预检生成的全部时长一次交给 `Simulation::run`。每个成功步发布字段和时钟。运行期间的首个失败结束该算例并执行清理。成功输出包含 `accepted_time`、`state_version`、`accepted_steps` 及按 case 顺序排列的稳定 ID 与 `x/q`。浮点输出使用各自类型的 `max_digits10`。预检失败退出码为 2。运行期间发现的数值或资源失败退出码为 1，报告失败步并省略成功摘要。

## 本地验收

Apple M2 Pro arm64，macOS 27.2.1 build 26B131，AppleClang `21.0.0.21000401`，CMake `4.4.3`，Ninja `1.13.2`，Python `3.14.7`。沿用并重新校验 [exact Kokkos 5.2.2 Serial 安装](serial-ci-validation.md) 的 Debug/Release 依赖摘要。每次验证前后 10 个 IPv4/IPv6 TCP、UDP 和 bind 探针均返回 `EPERM`。配置、构建、测试、安装和独立消费继承该网络限制。

| 配置 | 配置开始至包测试完成 | CTest | 独立消费者 | Decimal |
| --- | --- | --- | --- | --- |
| float Debug | `2026-11-11T13:57:53+08:00` 至 `2026-11-11T13:58:23+08:00` | 20/20 | 3/3 | 通过 |
| double Debug | `2026-11-11T13:58:23+08:00` 至 `2026-11-11T13:59:01+08:00` | 20/20 | 3/3 | 通过 |
| float Release | `2026-11-11T13:57:53+08:00` 至 `2026-11-11T13:58:25+08:00` | 20/20 | 3/3 | 通过 |
| double Release | `2026-11-11T13:58:25+08:00` 至 `2026-11-11T13:59:04+08:00` | 20/20 | 3/3 | 通过 |

四组依赖图、安装搬迁和原有数值判据通过，CTest 无失败或跳过。搬迁后的普通 `run` 输出与私有测试驱动直接调用 `Simulation`/Reference 的输出逐字节相同。驱动使用独立固定的非二进制步长序列和 `x/q` 期望。CLI 测试还检查大绝对时间下保留请求步长、末步缩短、`real` 最大有限值，以及恰好 1,000,000 步的计划通过和超限拒绝。源树与搬迁后测试均包含带空格的 case 路径，核对三入口的文件行号、退出码及非法输入诊断。

`2026-11-11T13:57:52+08:00` 至 `2026-11-11T13:58:25+08:00`，在带空格的构建路径以 `CMAKE_INSTALL_BINDIR=tools` 和自定义 `CMAKE_CXX_FLAGS=-g` 复验。测试构建报告 built/available 为 true、validated 为 false，普通 `run` 退出 2，显式 fixture 输出与直接库调用一致。关闭测试后，`--fixture` 退出 2。另于 `2026-11-11T13:59:10+08:00` 至 `2026-11-11T13:59:17+08:00` 验证匹配证据的默认 Release 构建在 `BUILD_TESTING=OFF` 时仍可普通运行，且拒绝 `--fixture`。

Apple 链接器继续报告重排测试的重复 `libnexora_data.a` 链接，链接及执行通过。Linux GCC、Linux Clang、Windows MSVC 托管矩阵及 MPI、OpenMP、CUDA/HIP 本轮未执行。

## 独立复审

**PASSED**。`2026-11-11T13:58:39+08:00` 至 `2026-11-11T13:59:18+08:00`，独立复审在干净构建目录运行禁网 `verify`，Debug/Release × float/double 四组均为 20/20，无失败或跳过。四组的普通 `run` 均完成 3 步。复审确认三入口诊断一致、CLI 与直接库输出逐字节一致、搬迁后 CLI 和独立消费者通过，Decimal 与依赖检查通过。预检仅经私有 Reference Participant，依赖检查拒绝 CLI 直接包含方法状态头。

`2026-11-11T13:59:31+08:00` 至 `2026-11-11T13:59:39+08:00`，独立复审在 `BUILD_TESTING=OFF`、`CMAKE_INSTALL_BINDIR=tools` 且搬迁前缀和 case 路径带空格的配置中验证普通 `validate`、`plan`、`run` 通过，`--fixture` 被拒。运行时数值溢出返回退出码 1，无成功摘要。复审另通过 39 个独立探针和无验证证据构建的运行门禁。

本地验收与独立复审的 59 个源码和测试路径摘要完全一致。源码摘要清单 SHA-256 为 `b921bf696885160d3473e8f73d3e947d0f06a3e53eda1a88649e6ff9736aa572`。

## 复跑命令

在受控禁网环境中运行以下命令。前缀按 [依赖供给命令](serial-ci-validation.md#重跑) 生成。本轮并行运行两个构建类型，每个 `verify` 依次运行 float 与 double。

```sh
nexora_kokkos_run=/tmp/nexora-serial-ci.ODRp5I
nexora_run=$(mktemp -d /tmp/nexora-task6-review.XXXXXX)
for config in Debug Release; do
  lower=$(printf '%s' "$config" | tr '[:upper:]' '[:lower:]')
  python3 tools/validate_serial.py --cc /usr/bin/cc --cxx /usr/bin/c++ --config "$config" \
    verify --prefix "$nexora_kokkos_run/kokkos-$lower" --work-dir "$nexora_run/build" \
    --logs "$nexora_run/logs-$lower" --jobs 2 --require-network-blocked || exit
done

case_file="$PWD/cases/reference/cli-nonbinary-tail.case"
build="$nexora_run/build/release-double"
relocated_cli="$build/tests/package/Release/relocated/bin/nexora"
"$relocated_cli" validate "$case_file"
"$relocated_cli" plan "$case_file"
"$relocated_cli" run "$case_file" > "$nexora_run/relocated.out"
"$build/tests/reference/nexora_reference_cli_oracle" > "$nexora_run/direct.out"
cmp "$nexora_run/relocated.out" "$nexora_run/direct.out"

custom_build="$nexora_run/custom build with spaces"
cmake -S . -B "$custom_build" -G Ninja -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON \
  -DCMAKE_CXX_FLAGS=-g -DCMAKE_INSTALL_BINDIR=tools \
  -DKokkos_DIR="$nexora_kokkos_run/kokkos-debug/lib/cmake/Kokkos"
cmake --build "$custom_build" --parallel 2
ctest --test-dir "$custom_build" --output-on-failure \
  -R '^(nexora_cli_case_suite|nexora_install_consumer)$'
```

`results/task6-20261111/` 保存构建与测试命令、JUnit、能力门禁日志、依赖清单、禁网探针、Decimal 输出和独立复审记录。`results/task6-20261111/review-manifest.json` 固定 Git 基线 `51dd068dc339ebb33cadb44968fc464e1ca4481a`、任务 5 的 PASSED 清单、任务 6 改动和累计工作树摘要。该基线保留原始开发计划。
