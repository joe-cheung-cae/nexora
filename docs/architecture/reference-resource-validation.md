# Reference 进程内恢复与资源预算

本记录对应 [底层开发计划任务 5](foundation-development-plan.md#任务-5-完成状态历史-恢复与快照预算)。实现位于私有 Reference State 和 Participant，由现有同步 `Simulation` 推进单参与者窗口。

## 状态与缓存

| 内容 | Accepted、trial 与恢复契约 |
| --- | --- |
| 稳定 ID | 生命周期内共享一个不可变 HostSpace View。每次试算的 ID issuer 绑定本次 trial stamp。 |
| `x/q` | 独立 trial Views 由原 relaxation 公式完整覆写。既有 `real`、`double` 时间与数值判据继续适用。 |
| Replay history | 单个 `uint64_t` Kokkos 元素。begin 在选定 Serial 实例上从 accepted 显式复制，advance 在候选内更新。 |
| Control | Host 端 `uint64_t sequence` 从 accepted 复制，advance 在候选内递增。超过容量的请求在 begin 拒绝。 |
| Diagnostic | 隔离夹具以 `history = 33 × history + sequence` 和 `diagnostic = history + sequence` 验证依赖，使用模 `2^64` 的无符号运算。`sequence` 先递增，初值可单独配置。该诊断只供恢复验证。 |
| Trial diagnostic cache | 键为 history binding 的完整 `FieldStamp`，含 topology、geometry、layout、accepted 版本、trial epoch、时间、storage incarnation。advance 完成并 fence 后重建，prepare_commit 查验同一绑定。commit、restore 和 finalize 丢弃该缓存。接收只改 `x/q`，与该缓存的输入无关。 |

begin 检查时间、版本、预算与 accepted history 读取授权，复制候选历史后准备本次窗口的 authority。advance 完成数值与诊断后才交回调用方。prepare_commit 核对版本和诊断，关闭 trial 签发并准备只读 accepted authority。commit 发布候选并转移资源票据，`Simulation` 随后推进 accepted 时钟和步数。恢复关闭 trial 签发，完成相关 Serial 访问，释放候选与其票据。若恢复失败，Participant 与 Simulation 均进入终止试算的 failed 状态，保留 accepted 快照供检查及显式关闭。retry 使用新 attempt，accepted 逻辑时间和步数由成功提交推进。

Reference 固定本地实体顺序。测试通过不匹配的 layout 键和实际重试产生的旧 attempt 验证诊断拒绝。实际 CSR 重排由 [StructureState 验证](structure-execution-validation.md)覆盖。

## 保活与额度

默认限制为 **64 个快照记录**及 **256 MiB 受控载荷**。这两个限额可为私有 Reference fixture 单独配置。

| 计量项 | 字节数与释放条件 |
| --- | --- |
| 共享 ID allocation | `n × sizeof(uint64_t)`。参与者和快照共享同一张票据。 |
| 当前、候选和仍被快照引用的旧 bank | 每个 bank 为 `2 × n × sizeof(real) + sizeof(uint64_t)`。发布转移候选票据，旧 bank 由最后一个快照保活。 |
| 已签发 accepted 快照 | 一份记录。同一快照的副本共享记录、字段租约和 bank 票据。 |
| 独立 packed transport | 每份单字段 pack 为 `n × (sizeof(uint64_t) + sizeof(real))`，占一个记录。消费者完成或销毁时释放该记录和载荷票据。 |
| 外部 owner | 通过 `known_external_owner_bytes` 单列。Reference 的 bank 都是 managed Views，接收只在同步调用期间借用输入，因此自身长期持有的外部 owner 字节为 0。 |

字节按逻辑数组载荷计算，Kokkos 对齐与分配元数据、host 控制标量、临时校验结构和 C++ 记录开销由运行时另行管理。初始化、begin、签发快照和 pack 在分配相应载荷或签发记录前预留额度。超限返回 `resource_exhausted`，释放旧记录后可按同一输入重试。

显式 finalize 先关闭签发，再完成该 Serial producer，最后释放参与者的 Views、authority、票据和执行实例。已交付的只读快照与独立 packed 记录可由消费者延迟完成，各自持有预算账户及数据。应用必须在 Kokkos `ScopeGuard` 结束前释放所有这些记录。提取的原生 View 只在对应快照或 transport 记录保持有效期间用于访问，单独复制 View 不延长授权或预算生命周期。

## 未来 checkpoint 清单

在一致的 accepted 边界保存稳定 ID 与实体命名空间、字段语义、`x/q`、replay history、`Control::sequence`、`lambda`、accepted 时间与版本、accepted 步数和已发布的 attempt 元数据。若恢复后继续使用同一诊断计数域，记录 attempt 高水位以避免重复。保留构建精度和时间契约及现行 R0 所需的配置与 provenance。

诊断由 history 与 control 重算。按稳定 ID 重建索引与字段绑定。trial 缓存、lease、预算票据、allocation incarnation 与执行实例由新会话建立。Reference 当前恢复在进程内完成。 [持久恢复契约](checkpoint-restart.md#checkpoint-content)

## 验收

`nexora_reference_resource_tests` 在提交后的非零 accepted 状态分别注入 begin、advance、receive 和 prepare 故障。测试用预先独立复制的值核对当前及旧快照的 ID、`x/q`、历史、控制和诊断，同时核对 allocation identity、stamp、Simulation 时钟与预算。接收使用同 epoch、不同值的独立发送者，故障位于解包及有限值检查之后。恢复失败用例触发 `Simulation::failed`，拒绝再次试算和提交，显式 finalize 回收候选，旧快照仍可读。部分初始化故障覆盖已分配 Views 的回收，选定实例在清理完成前保活。

两-bank 上限用例在提交后恰好用尽字节额度。旧快照的任一副本仍在时，下一次 begin 返回 `resource_exhausted`，Kokkos Tools 记录到 0 次分配，accepted 保持原值。最后一个副本释放后，同一 begin 成功。显式关闭后，快照和慢 packed 消费者分别释放各自记录与载荷。带种子的 history/control 重放与连续推进对照通过，诊断拒绝错误 layout、旧 attempt 及错误 storage incarnation。

Apple M2 Pro arm64，macOS 27.2.1 build 26B131，AppleClang `21.0.0.21000401`，CMake `4.4.3`，Ninja `1.13.2`，Python `3.14.7`。本次沿用任务 1 的 exact Kokkos 5.2.2 Serial Debug 和 Release 安装，并再次核对安装文件摘要。配置、构建、CTest、安装搬迁及独立消费继承同一受控禁网环境。每次验证前后 10 个 IPv4/IPv6 TCP、UDP 和 bind 探针均返回 `EPERM`。 [依赖供给记录](serial-ci-validation.md)

| 配置 | 配置开始至包验证完成 | CTest | 独立消费者 | Decimal |
| --- | --- | --- | --- | --- |
| float Debug | `2026-11-11T13:27:16+08:00` 至 `2026-11-11T13:27:40+08:00` | 19/19 | 3/3 | 通过 |
| double Debug | `2026-11-11T13:27:40+08:00` 至 `2026-11-11T13:28:04+08:00` | 19/19 | 3/3 | 通过 |
| float Release | `2026-11-11T13:27:16+08:00` 至 `2026-11-11T13:27:44+08:00` | 19/19 | 3/3 | 通过 |
| double Release | `2026-11-11T13:27:44+08:00` 至 `2026-11-11T13:28:09+08:00` | 19/19 | 3/3 | 通过 |

四组的依赖图、安装正反例和独立 Decimal 检查通过，CTest 失败和跳过均为 0。Apple 链接器继续报告重排测试的重复 `libnexora_data.a` 链接，链接与执行均通过。本地证据覆盖单参与者 Serial。Linux GCC、Linux Clang、Windows MSVC 托管矩阵及 MPI、OpenMP、CUDA/HIP 本轮未执行。

独立复审于 `2026-11-11T13:28:58+08:00` 至 `2026-11-11T13:29:24+08:00` 在禁网目录复跑四组，每组 19/19，依赖图、搬迁消费与 Decimal 通过，失败和跳过为 0。复审用原地污染探针验证故障测试能识别 accepted 改写，并独立核对预算边界、恢复失败和最后一个消费者释放。复审结果为 **PASSED**。源码哈希与日志保存在 `results/task5-20261111/independent-review/`。

在受控禁网环境中运行以下命令。前缀按 [依赖供给命令](serial-ci-validation.md#重跑) 生成。本轮并行运行两个构建类型，每个 `verify` 依次运行 float 与 double。

```sh
nexora_kokkos_run=/tmp/nexora-serial-ci.ODRp5I
nexora_run=$(mktemp -d /tmp/nexora-task5-final.XXXXXX)
for config in Debug Release; do
  lower=$(printf '%s' "$config" | tr '[:upper:]' '[:lower:]')
  python3 tools/validate_serial.py --cc /usr/bin/cc --cxx /usr/bin/c++ --config "$config" \
    verify --prefix "$nexora_kokkos_run/kokkos-$lower" --work-dir "$nexora_run/build" \
    --logs "$nexora_run/logs-$lower" --jobs 2 --require-network-blocked || exit
done
```

`results/task5-20261111/` 保存配置与构建命令、退出码、JUnit、依赖清单、禁网探针、Decimal 输出及复审清单。复审基线为 `51dd068dc339ebb33cadb44968fc464e1ca4481a` 和任务 1–4 的 PASSED 工作树。
