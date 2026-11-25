# Reference 窗口的 lease 与本地收发

本页记录早期 Serial 基线。现行单求解器生命周期和状态发布契约见 [Kokkos 数据与执行设计](kokkos-design.md#7-单求解器的同步推进与状态发布)。

当前内部 reference 参与者复用 `ParticipantIF` 与 `Simulation` 的同步事务。reference 算子保留既有显式 Euler 公式、时间步和容差。真实耦合端口、CLI、MPI、设备异步和持久 checkpoint 属于后续范围。

## 窗口契约

- 初始化后，参与者为 accepted `x/q` 持有只读 authority。每次打开窗口分配独立 trial `x/q`，为它们签发写授权。稳定 ID 存储保持不变，窗口为这块只读存储建立同 epoch 的 ID 授权。reference 算子从 accepted 读 lease 和 trial 写 lease 获得原生 Kokkos Views，完成并 fence 后才结束这些局部访问。
- `trial_input_stamp()` 在算子完成后提供本次试算的逻辑 stamp，包含基准 accepted 版本、窗口、单调 attempt 和窗口终点。预提交前，`receive_trial()` 可用本地 `TransportView` 替换指定 trial `x` 或 `q`。输入使用 coupling purpose 和 reference 的字段语义，按稳定 ID 映射到目标。解包及有限性检查同步完成，写 lease 不会逃逸。
- 一次无效接收会使整个试算无法提交。调用方可主动恢复，也可由 `Simulation::commit_window()` 在预提交失败后恢复。恢复丢弃全部 trial 和待发布授权，保留 accepted 数组、版本、时间及其快照。重开同一时间窗会使用新 attempt，旧 payload 无法应用到重试。
- 预提交完成可能失败的只读授权登记，将 trial 的写授权关掉。此时公开快照仍读取上一版 accepted。提交成功才发布新的 Views、时钟、版本和只读授权。快照副本共享原版的 `x/q` 读 lease，稳定 ID 由 const managed View 保活。旧版数据可跨提交、恢复、finalize 和参与者销毁保留，快照须先于应用的 Kokkos finalize 释放。

本轮为单调用方 Serial fixture。同一窗口内的字段替换顺序执行，接收失败时以窗口为单位丢弃。Context7 返回的是 develop 分支通用执行空间说明，具体镜像、复制和同步行为以本地锁定的 5.2.2 包及编译运行结果核验。

## CPU 验证

使用已预供给的 Kokkos 5.2.2 Serial-only Release 包，commit `8e15454876298832f013ebcaf08d53825b5bda40`。环境为 arm64、macOS 27.2 build 26B123、SDK 27.2、AppleClang 21.0.0.21000401、CMake 4.4.3、Ninja 1.13.2。依赖来源和摘要见 [CPU 基础记录](kokkos-cpu-slice-validation.md#依赖与机器配置)。本轮未启用额外的禁网沙箱。

| 配置 | 最终复验 | 结果 |
|---|---|---|
| CPU Debug | `2026-11-06T18:52:54+08:00`–`2026-11-06T18:53:02+08:00` | CTest 16/16，搬迁安装消费 3/3，依赖检查通过 |
| CPU Release | `2026-11-06T18:52:54+08:00`–`2026-11-06T18:53:02+08:00` | CTest 16/16，搬迁安装消费 3/3，依赖检查通过 |

专用测试覆盖乱序 ID 替换、解包成功后主动恢复、同一试算中先成功替换一个字段再因另一字段失败而整窗恢复、非有限值拒绝、过期 attempt 拒绝、同端点重试、预提交期间仅能看到旧 accepted，以及旧快照和其副本跨提交及参与者销毁保活。保留原 reference 公式的数值与收敛回归。安装回归确认 reference 头和 target 仍在内部边界。

专用集成测试及 Nexora 相关编译单元另以 `-Wall -Wextra -Werror -fsanitize=address,undefined` 编译，链接同一预构建的 Kokkos 库，执行无 sanitizer 诊断，完成于 `2026-11-06T18:53:02+08:00`。配置、构建、测试、依赖检查和 sanitizer 运行退出码均为 0。日志位于本地忽略目录 `results/reference-window/`。


## 本地发送与两参与者复验

`send_trial()` 仅在算子完成后、预提交前签发 trial 字段和稳定 ID 的读 lease。发送端复用现有打包和 Serial fence，输出独立的只读 `TransportView`，随后归还两个源 lease。消费者可保留旧载荷，发送方仍能在预提交时切换 authority，也能开启下一窗口。外部内存等调用方仍可沿用保留源 lease 至消费者完成的默认策略。

CPU 集成测试用两个拥有独立 `Simulation` 的 reference 参与者验证相同窗口内的乱序 ID 收发及分别提交。接收失败由接收方事务回滚，调用方显式恢复发送方试算，使下一次重试的窗口和 attempt 对齐。旧载荷可读取并保持原值，接收端会拒绝其过期 attempt。发送快照还覆盖移动、跨提交、后续窗口回滚、finalize 和发送方销毁后的读取。

使用上述锁定的 Kokkos 5.2.2 Serial-only 包重新构建并执行 CTest。CPU Debug 在 `2026-11-06T18:57:28+08:00` 至 `2026-11-06T18:57:37+08:00` 通过 16/16，CPU Release 在 `2026-11-06T18:57:30+08:00` 至 `2026-11-06T18:57:39+08:00` 通过 16/16。两组搬迁安装消费均通过 3/3，依赖检查通过。专用测试以严格警告、ASan 和 UBSan 编译运行，结束于 `2026-11-06T18:57:37+08:00`，无诊断。日志位于 `results/reference-sender/`。
