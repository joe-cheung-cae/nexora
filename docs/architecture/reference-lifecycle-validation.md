# 步骤 2：参考状态生命周期实现与验证

记录时间：`2026-11-06T17:35:17+08:00`。禁网复验区间：`2026-11-06T17:34:56+08:00` 至 `2026-11-06T17:35:17+08:00`。

维护者明确授权本次最小 reference 闭环。**这是 Serial CPU 基础之上的有限步骤 2，不代表完整 Phase 1 获批或完成。** 保留[原 reference 公式、参数约定和容差](phase-1-design-review.md#31-保留的-reference-夹具)；没有选择新模型。步骤 1 的历史证据保留在 [CPU 基础记录](kokkos-cpu-slice-validation.md)。

## 实现和存储所有权

| 落点 | 本次实现 |
|---|---|
| `src/methods/reference/State.hpp/.cpp` | 方法状态：稳定 `uint64_t` ID、FP64 `x/q`、accepted 时间、`StateVersion`、单调尝试编号和必要阶段；直接使用 typed Kokkos Views |
| `src/participants/reference/Participant.hpp/.cpp` | 适配现有 ParticipantIF；拥有方法状态，提供 accepted 只读快照；报告 rollback 能力及 `1/lambda` 步长上限 |
| `include/nexora/core/Time.hpp` | 原样移入 TimeInterval；`requested_step/duration()` 保持原语义，旧 TimeConstraint 头继续包含该定义 |
| 根 CMake 与两个内部 CMake | 增加内部 `nexora_methods_reference`、`nexora_participants_reference`；不安装这些目标或头，不增加公共 reference API |
| `tests/reference/` | 同一 CPU 回归套件中的真实数组生命周期、失败重试和数值测试 |
| `tools/check_dependencies.py`、`tests/package/` | 新内部 target/header 向下依赖规则及反例；TimeInterval 新旧包含路径的安装消费；确认 reference 目标不导出 |

现有 `Simulation` 和 `ParticipantIF` 的生命周期实现无需修改。方法的初始配置是 host 输入向量；成功初始化后释放这些向量，运行状态仅由 Kokkos Views 持有。稳定 ID 非空、唯一且与 x/q 长度一致；本切片不改变顺序或拓扑。x/q 的 descriptor 复用现有 FieldDescriptor 的语义子集：`Association::node`、`entity_set=reference_samples`、单位 `1`、FP64 单分量、point value、read、accepted 时间及状态版本。未拆分通用 schema 或重新设计 registry。

生命周期采用以下顺序：

1. 应用创建 `Kokkos::ScopeGuard`，再初始化参与者和 Simulation。没有初始化 Kokkos 时明确返回 `invalid_state`，不分配数组。
2. `begin_window` 校验有限时间、正 duration、连续 accepted 起点和 `0 < dt*lambda <= 1`，为 x/q 新分配独立 trial 存储。ID 只读共享。
3. `advance_trial` 使用显式 `Kokkos::Serial` 实例及 `RangePolicy`，逐元素计算 `d=dt*lambda*x`、`trial_x=x-d`、`trial_q=q+d`。实例 fence 后，通过独立 `create_mirror` 和显式 `deep_copy` 检查全部结果有限性。
4. 只有成功计算并完成 `prepare_commit` 后，`commit_window` 才发布 trial，增加 accepted 版本并推进时间。方法释放 mutable trial handles，后续试算不写这组已发布数组。
5. 主动恢复或数值失败时丢弃 trial，accepted 数组、时间、版本和 accepted 尝试编号不变。现有 Simulation 在 advance 失败后自动恢复，允许合法重试。尝试计数不回退，因此重放数值相同不要求尝试编号相同。

快照返回 `Kokkos::View<const double*>` 和 `Kokkos::View<const uint64_t*>`；元素不可写，元数据为副本。方法从不根据 View 引用计数授予写权限或复用 published 存储。旧快照可以跨提交、恢复、参与者 finalize 和销毁持续读取，靠 managed View 保活；新 snapshot 请求在 finalize 后失败。**应用必须在 Kokkos finalize 前销毁所有参与者及保留快照。** 本切片同步使用、单一调用方；没有并发调用保证。没有快照预算或背压，调用者持续保留历史快照会持续保留其数组。

`ports()` 返回空集合：本次只读输出通过 reference 的 `snapshot()`，未实现耦合端口。可报告的 host 异常转换为 Status；致命 Kokkos/runtime abort 不承诺可恢复，未注入或验证这种故障。

## 固定依赖与接口证据

继续使用步骤 1 已准备的本地 Kokkos **5.2.2**，commit `8e15454876298832f013ebcaf08d53825b5bda40`，源制品 SHA-256 `d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f`。许可证为 `Apache-2.0 WITH LLVM-exception`，无本地 patch；来源、notices、库摘要和完整工具链见 [CPU 基础记录](kokkos-cpu-slice-validation.md#依赖与机器配置)。本次没有获取新依赖。

- 安装包：`/private/tmp/nexora-kokkos-5.2.2/install-release/lib/cmake/Kokkos`；Serial-only、C++20、extensions OFF，GPU/MPI 关闭。
- 主机：Apple M2 Pro / arm64 / 12 CPU / 16 GiB；macOS 27.2 build 26B123，SDK 27.2；本测试仅 Serial。
- 编译器：AppleClang `21.0.0.21000401`，libc++ `_LIBCPP_VERSION=220106` / ABI 1；CMake 4.4.3，Ninja 1.13.2。
- Nexora Debug/Release 均链接同一 Release Kokkos 库；不声称运行了 Kokkos Debug instrumentation。

按仓库要求使用 Context7 resolve/query 检索 Kokkos View、所有权和执行；细分 View 查询未命中，通用查询返回滚动分支资料。因此具体接口以已校验的锁定源码为准，并实际编译执行：managed/const View 采用当前包的 [Kokkos_View.hpp](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/core/src/Kokkos_View.hpp)，复制与 mirror 核对 [Kokkos_CopyViews.hpp](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/core/src/Kokkos_CopyViews.hpp)，运行时边界核对 [ScopeGuard](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/core/src/impl/Kokkos_ScopeGuard.hpp)。本地包没有启用 legacy View 实现。未把 develop 或未来版本描述当作 5.2.2 API 证据。

执行时文件 SHA-256（fixture 输入、reference 和断言在测试源码中）：

```text
tests/reference/reference_tests.cpp  ac9609ef4928dba33301a760ff0da24d02ff7873343244864a50b07be702c4e9
src/methods/reference/State.cpp      ae050f93bec40c423a46b284a1178aa2138de6f1f6ecbca0079643e92d6951c6
```

## 实际命令与结果

工作目录 `/Users/chao/workspace/repo/nexora`。以下每条命令都使用 `sandbox-exec -p '(version 1)(allow default)(deny network*)'` 前缀执行，子进程同样禁网。只有本地预供给的依赖；configure/build/test/install 不下载。

```sh
cmake --fresh --preset cpu-debug -DKokkos_DIR=/private/tmp/nexora-kokkos-5.2.2/install-release/lib/cmake/Kokkos
cmake --build --preset cpu-debug
ctest --preset cpu-debug --verbose
python3 tools/check_dependencies.py --graph build/cpu-debug/nexora-target-graph.json
cmake --fresh --preset cpu-release -DKokkos_DIR=/private/tmp/nexora-kokkos-5.2.2/install-release/lib/cmake/Kokkos
cmake --build --preset cpu-release
ctest --preset cpu-release --verbose
python3 tools/check_dependencies.py --graph build/cpu-release/nexora-target-graph.json
```

全部命令 exit code 0。本地完整日志及每条命令的参数、秒级起止时间保存在忽略目录 `results/reference-lifecycle/`，清单为 `commands.json`。`--fresh` 清理配置缓存，构建仍复用未变化源码的现有产物；新增/变化源码已实际编译。

| 检查 | 实际结果 |
|---|---|
| CPU Debug | configured / compiled / executed；CTest `2026-11-06T17:34:58+08:00`–`2026-11-06T17:35:06+08:00`，12 passed / 0 failed / 0 skipped |
| CPU Release | configured / compiled / executed；CTest `2026-11-06T17:35:09+08:00`–`2026-11-06T17:35:17+08:00`，12 passed / 0 failed / 0 skipped |
| 安装消费 | 各 profile 的 package 测试安装并搬迁 Nexora，复制独立 consumer 源码后配置、构建和运行；各 2/2 passed。新 core/Time 头独立 TU 和旧包含路径通过，reference target 未导出；原 transitive Kokkos 整数计算全部 32 项通过 |
| 配置拒绝 | 各 profile 的既有 13 项检查全部命中预期诊断，含源码/consumer 缺包、模拟 5.2.1、C++17、错误 devices/standard/compiler 元数据和旧 GPU 选项；模拟包不代表构建或执行过这些组合 |
| 依赖检查 | 两个配置的实际 target graph、头边界、允许与禁止依赖 self-check 全部通过 |

新 reference 测试实际覆盖：

- 非单位 `lambda=2`，ID `{17,3,42}`，三个正负/非零 x/q 元素全部精确验证；trial、主动 restore 和 precommit 期间 accepted 值、指针、时间、版本不变，成功提交后才替换数组。
- 保留三个连续 accepted 快照，跨后续计算、restore、finalize 和参与者销毁逐元素复验；编译期断言 x/q/ID 元素不能赋值。
- 有限初值 `x=double_max, q=double_max/2` 在 `dt=1` 产生 trial 溢出，报告 `numerical_failure`；Simulation 自动恢复，accepted 全部不变。`dt=.25` 重试成功，与全新参与者直接走 `.25` 的全部值、时间和版本一致，尝试编号分别为 2/1。
- 主动恢复后的合法两步运行与全新重放一致；正在试算时 finalize 自动恢复。拒绝重复 initialize/finalize、非法阶段调用、错误 interval、重复 ID、形状错误、空数据、NaN/Inf 和非法 lambda。
- Kokkos 未初始化时拒绝且 allocation callback 观察为零次分配。绝对时间 `2^52`、requested step `.75` 的端点差为 `1`，计算仍使用 `.75`，得到 `x=.25, q=.75`。

默认 reference 为 `lambda=1, x0=1, q0=0, T=1`。Debug/Release 数值结果相同；使用原容差，没有放宽：

| 步数 | 最终 x | 最终 q | 对 exp(-1) 的 x 误差 | 最大 abs(x+q-1) |
|---|---|---|---|---|
| 16 | 0.35607413045179281 | 0.64392586954820719 | 0.01180531071964952 | 0 |
| 32 | 0.36205528925631664 | 0.63794471074368364 | 0.0058241519151256971 | 4.4408920985006262e-16 |
| 64 | 0.36498652424390737 | 0.63501347575609268 | 0.0028929169275349609 | 4.4408920985006262e-16 |

观测阶为 **1.0193161223839076 / 1.0095230910767712**，均在 `0.9–1.1`；每一步守恒误差低于 `1e-12`；最终值与独立显式 Euler 闭式离散值比较，满足 `1e-12 + 1e-10*abs(ref)`。这是 reference 的 numerically validated 结果，不适用于其他方法或设备。

## 保留项与未实现边界

仍保留旧 `Buffer`、`ExecContext`、`FieldView`、旧 exec 导出/测试、dummy 参与者测试与休眠 CUDA smoke 源码。新 reference 不调用旧 exec，没有扩展第二条计算路线；依赖图中旧 data/participant 边尚待后续整体迁移。

未实现通用 Field/ReadField/WriteField、lease/外部 owner、TransportView、历史快照预算与背压、异步通信、持久 checkpoint、CLI reference case、安装 reference API、SPH/FVM、GPU/OpenMP、MPI 数值或多参与者耦合。未做性能认证、分配失败注入、计数器耗尽运行或致命 runtime 错误恢复测试；这些不能宣称通过。现有 CI 仍缺 Kokkos 供给，本地结果不等于 CI 已通过。完整 Phase 1 保持开放。
