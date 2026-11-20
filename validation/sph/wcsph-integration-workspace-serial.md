# WCSPH 三维积分工作区验证

## 工作区生命周期

`Participant3D` 从 accepted 快照读取位置、速度、密度和压力加速度，每次窗口试算执行一次 Velocity Verlet 推进。工作区使用 Kokkos Serial 的 managed `HostSpace` Views，受参与者的资源预算约束。[积分与窗口实现](../../src/participants/sph/Participant3D.cpp)

- 漂移阶段写入候选坐标和半步速度。格表整理完成后，候选几何持有独立坐标。坐标暂存可复用为连续性阶段的速度暂存，半步速度留至最后一次半踢。
- 连续性阶段通过格表的双向排列保持稳定粒子 ID 与旧密度率、预测密度率和校正密度的对应。校正完成后，速度暂存、旧行映射与两组速率即可释放。压力计算读取校正密度，末次半踢完成后生成新的速度和声学 CFL 上限。
- 工作区按参与者跨步复用。如果输出字段或压力工作区的预算预留失败，则释放已用完的连续性缓冲，只重试本次字节预留。半步速度与校正密度保留至本步计算完成。
- 几何、密度、压力、压力加速度与速度一并发布为只读状态。预算或数值检查失败时，`Simulation` 恢复该窗口，保留原 accepted 快照、时钟和 CFL 上限。调用方可用合法步长重新开窗。[窗口契约](../../include/nexora/simulation/Simulation.hpp)

## Serial 回归

回归用例覆盖稳定 ID 的跨格推进、预算拒绝与重试、accepted 和 trial 快照的保活与发布，以及全空和仅固定壁两种零 owned 路径。使用 [Serial 验证器](../../tools/validate_serial.py) 检查 Debug 和 Release 的 float、double 配置，以及安装后的独立消费。[积分窗口测试](../../tests/methods/sph/verlet_window_tests.cpp) · [连续性测试](../../tests/methods/sph/continuity_tests.cpp)

在 Apple M2 Pro、macOS arm64、Apple Clang 21.0.0 和 Kokkos 5.2.2 Serial 上，Release 复验完成于 `2026-11-20T17:24:10+08:00`，Debug 复验完成于 `2026-11-20T17:24:45+08:00`。两种构建的常规 CTest 均为 float 132/132、double 131/131，各含 66 项 SPH 测试。四组的依赖检查、精度校验和安装后消费测试均通过。
