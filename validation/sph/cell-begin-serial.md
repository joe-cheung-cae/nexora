# SPH-002 三维 cellBegin 的 Serial 验证

本切片在私有 `nexora_methods_sph` 中实现单进程、非周期、固定半径的三维格搜索，由 `nexora_participants_sph` 的 `Participant3D` 在 Simulation 窗口中运行。代表性密度计算采用选型实验的三维 M₄ 核，固定 `h = 1`、`R = 2`。真实物理算例及其参数由 BASE-001/SPH-001 另行验证。[任务范围](../../docs/architecture/migration-roadmap.md#sph-cell-begin)

<a id="simulation-window"></a>
## Simulation 窗口

`Simulation simulation(participant)` 使用已有的 `ParticipantIF`。初始化时，`Participant3D` 从初始字段建立首份独立格索引，并计算所有 owned 粒子的初始密度。从这一步起，accepted 快照可按稳定 ID 读取邻居和密度。[参与者接口](../../src/participants/sph/Participant3D.hpp)

| 操作 | 可观察的状态 |
|---|---|
| `simulation.begin_window(dt)` 后调用 `participant.set_trial_fields(end_fields)` | 将终点坐标、质量、稳定 ID 和 ghost 标记复制到本次 attempt。提供者可在返回后释放或修改自己的缓冲。省略输入时沿用 accepted 几何。 |
| `simulation.advance_trial()` | 有新字段时从 accepted 格索引更新，按本步坐标重新判距。密度通过写 lease 写入独立候选分配，完成后可用 `trial_stamp()` 和 `trial_snapshot(stamp)` 读取。此时 `snapshot()` 仍返回原 accepted 状态。 |
| `simulation.commit_window()` | 在预提交阶段准备只读 accepted 密度绑定。有读者保留 trial 时另复制密度。几何、密度和版本一起发布，Simulation 随后推进时钟。 |
| `simulation.restore_window()`，或 `advance_trial()/commit_window()` 失败 | 丢弃候选和本次输入，accepted stamp、时钟与数值保持原值。重试可重新提供终点字段，使用新的 attempt。先前签发的快照仍可读，旧 stamp 无法再取得当前 trial。 |

`topology` 跟踪稳定 ID 集合和 ownership，`geometry` 跟踪按 ID 对齐的坐标或成员变化，`layout` 跟踪槽位顺序、owned 行和输入映射。质量单独变化仍会重算密度。快照仅提供按行读密度和按稳定 ID 枚举邻居。快照可保留到 participant 关闭之后，应用须在 Kokkos 结束前释放它们。[窗口实现](../../src/participants/sph/Participant3D.cpp)

默认预算为 64 份外部快照记录和 256 MiB 受控载荷。预算覆盖 staged 输入、快照保留的格索引和粒子字段，以及密度和预提交副本。共享格表和映射按同一分配计一次。建格排序、身份匹配的临时工作区和 Kokkos 分配开销另计，`max_cells` 继续限定稠密格数。[几何状态与预算](../../src/participants/sph/Geometry3D.cpp)

### 窗口验证结果

在 Apple M2 Pro、macOS 27.2.1 arm64（26B131）上，使用 AppleClang 21.0.0.21000401、C++20、CMake 4.4.3、Ninja 1.13.2、Kokkos 5.2.2 Serial 和 GoogleTest 1.17.0 实跑。四组离线验证从 `2026-11-20T08:20:45+08:00` 持续至 `2026-11-20T08:22:01+08:00`，均从空构建目录开始，依赖检查和测试退出 0，零跳过。

| 配置 | 三维窗口测试 | 三维方法测试 | 常规 CTest（含前两列） | 安装包 CTest / 包内消费者 |
|---|---:|---:|---:|---:|
| Debug / Release float，各组 | 7/7 | 17/17 | 99/99 | 1/1、3/3 |
| Debug / Release double，各组 | 7/7 | 17/17 | 100/100 | 1/1、3/3 |

- 27 粒子回放算例包含 20 个 owned 和 7 个 ghost。先提交一次格内移动，再让一个 owned 与一个 ghost 跨三个轴交换位置并打乱输入顺序。每次读取初始、试算、提交、恢复或重试快照时，都从对应的原始粒子字段全扫描建立显式 CSR，对每个 owned ID 精确比对邻居集合，并用独立 M₄ 公式核对密度。显式恢复保留上一已提交的 stamp 与时钟。相同步长和字段重试后，逐行密度与被撤销的 trial 逐位一致，旧 attempt 的 stamp 被拒绝。[窗口回归](../../tests/methods/sph/participant_3d_tests.cpp) · [独立 CSR](../../tests/methods/sph/cell_grid_oracle.hpp)
- `2026-11-20T08:22:10+08:00` 在 Release double 的上述回放中用 Kokkos Tools 观测到初建、格内复用、越格重建、重试再次重建。回滚后的重试从上次成功提交的格索引重新出发。
- 另验几何非法和密度溢出时自动恢复、缩短步长重试、预提交失败时有无 trial 读者两条路径、跨实例 stamp 拒绝、资源额度回收、空输入和纯 ghost。保留的快照在回滚、后续提交和 participant 关闭后仍可按对应阶段复核。

源码基于 `1cb423fa17c4a1ec357e7319c466d58be28fa264` 加本次工作树。本机 `results/sph-002-window-20261120T082038/` 保存 JUnit、依赖证明、安装包日志、`window-replay-trace.log` 及 `source-sha256.txt`。源码和验证输入摘要清单 SHA-256 为 `3d6c5eef3006cd3c6a8e873a07064864454b8c5c4ae17e67fe0f931c4dbf4313`。

## 方法与对照

- 调用者提供指定的 `Kokkos::Serial` 实例、HostSpace 粒子字段和稠密格数预算。网格按 `floor(x/R)` 等三个轴向格号确定包围盒，检查容量，再按格号及完整 64 位稳定 ID 排序。重复 ID、非有限坐标、非正或非有限质量、非法 ghost 标记及不够的格数预算均返回错误。[建格](../../src/methods/sph/neighbors/CellGrid.cpp)
- 每个快照独立持有重排后的位置 SoA 和质量，并持有只读的格表、ID、ghost 标记、双向槽位置换及 owned 行映射。格表以源粒子数收尾，空格两端相等。每个 owned 目标最多扫描 9 个连续 x 区间，覆盖裁剪后的 27 格。按该步坐标与闭支撑域筛选非自身粒子，ghost 只读。[遍历](../../src/methods/sph/neighbors/CellGrid.hpp)
- 支撑判定要求各轴的实际分量差绝对值均不超过 2，并以 binary64 按 x、y、z 顺序累加平方距离，与 4 作闭区间比较。实现在平方距离等于 4 时用减法余项检查轴向边界。例如 `-denorm_min` 与 `2` 的轴向差可能舍入为 2，两者实际在支撑域外。正好相距 2 的边进入邻居列表，密度权重为零。
- 输出行按 `owned_slots()` 映射回稳定 ID。密度为自身项 `m_i / π` 加各邻居的 `m_j W(r)`。核值与累加使用 double，写入选定的 `real`。非有限或超过输出精度上限的结果使本次求值失败。[密度核](../../src/methods/sph/Density3D.cpp)
- 测试在原始输入上逐 owned 行扫描全部源，独立判断分量边界，以稳定 ID 建立显式 CSR。逐行精确比较完整邻居集合并检查重复边，再沿 CSR 逐行用等价的截断幂形式计算参考密度。密度相对误差限为 `ε(real) + 32 × (邻居数 + 1) × ε(double)`，涵盖输出舍入及不同累加顺序。该主机的 `long double` 有 53 位有效精度。[逐粒子测试](../../tests/methods/sph/cell_grid_tests.cpp)

## 跨步使用

首步调用 `CellGrid::build(exec, fields, max_cells)`。后续把本步全部粒子字段传给 `previous.update(exec, fields, max_cells)`，成功后在返回的新快照上调用 `evaluate_density_3d(exec, next)`。相邻两步可同时保留，各自用本步的位置、质量重算密度。

| 本步变化 | 格表与槽位的处理 |
|---|---|
| 所有粒子均在原格，输入顺序相同 | 复用格表、ID、ghost 标记、owned 行及双向映射。按现有映射打包本步位置与质量。 |
| 只有输入顺序改变，或重排同时有格内移动 | 按新坐标定位格，在该格内按完整稳定 ID 查找旧槽位，验证一一对应并建立新的双向映射。格表与 owned 行仍共享。 |
| 任一 owned 或 ghost 粒子越格，成员或 ownership 改变 | 重新排序并构建格表、owned 行及双向映射。即使两个粒子交换了格子、各格计数相同，也会重建。 |

每次更新都校验字段及本次稠密格数预算。返回错误时旧快照仍可使用或重试。所有成功路径均在指定 Serial 实例上完成本步字段打包，再发布快照。邻居和密度按当前几何重新计算。[更新实现](../../src/methods/sph/neighbors/CellGrid.cpp)

## 跨步离线执行结果

`2026-11-19T17:27:05+08:00` 至 `2026-11-19T17:28:55+08:00`，在下节记录的同一 Apple M2 Pro 和工具链上完成四组全新离线构建。每组均通过依赖检查、全部 17 项三维测试和安装包检查，零跳过。

| 配置 | 常规 CTest（含三维用例） | 安装包 CTest | 包内消费者 |
|---|---:|---:|---:|
| Debug / Release float，各组 | 92/92 | 1/1 | 3/3 |
| Debug / Release double，各组 | 93/93 | 1/1 | 3/3 |

- 四粒子定向用例使一条边进入支撑域、另一条边离开，期间所有粒子均在原格。下一步两个 ghost 交换格子，再下一步两个 owned 粒子交换格子。每步按稳定 ID 对照 CSR，最后再次核对保留的旧快照。另有用例覆盖连续输入重排、ownership 和成员改变、空集合、预算拒绝及失败重试。
- 三维轨迹把 54 个粒子分布在 27 格中。初始步与之后 7 次更新均打乱输入顺序，并为各粒子生成新坐标和质量。第 2、4、6 次更新分别沿 x、y、z 交换两个粒子的格归属，各格计数保持相同。每一步都精确核对全部 36 个 owned 粒子的邻居 ID，并按上文容差核对密度。
- `2026-11-19T17:34:58+08:00` 在 Release double 上用 Kokkos Tools 复跑这两条轨迹。三维轨迹共观测到 4 次格表分配（初建及 3 次越格），其余 4 步只刷新字段并重映射输入顺序。8 个快照各新分配一组坐标和质量。四粒子轨迹依次为初建、复用、重建、重建、复用。

源码基于 `1cb423fa17c4a1ec357e7319c466d58be28fa264` 加本次工作树。本机 `results/sph-002-reuse-20261119T172658/` 保存 JUnit、依赖与安装包日志、Kokkos Tools 分配记录及 `source-sha256.txt`。源码与验证输入摘要清单的 SHA-256 为 `7e04af7285447559adb673098bd4cc2d576958352ea5702bb9f176886150ba9a`。

## 首版单步离线执行结果

运行时间为 `2026-11-19T16:29:35+08:00` 至 `2026-11-19T16:30:36+08:00`。主机为 Apple M2 Pro、macOS 27.2.1 arm64（26B131）。使用 AppleClang 21.0.0.21000401、C++20、CMake 4.4.3、Ninja 1.13.2、Kokkos 5.2.2 Serial 及 GoogleTest 1.17.0。四组配置均从独立空目录构建，禁网检查、依赖检查、常规 CTest 和迁移安装包检查均退出 0，零跳过。两个精度的常规总数含各自的一维精度专用用例。

| 配置 | 三维用例（计入常规 CTest） | 常规 CTest | 安装包 CTest | 包内消费者 |
|---|---:|---:|---:|---:|
| Debug float | 12/12 | 87/87 | 1/1 | 3/3 |
| Release float | 12/12 | 87/87 | 1/1 | 3/3 |
| Debug double | 12/12 | 88/88 | 1/1 | 3/3 |
| Release double | 12/12 | 88/88 | 1/1 | 3/3 |

随机三维算例含 137 个乱序粒子，其中 117 个 owned 目标同时核对邻居 ID 和密度。高邻居数算例验证单个目标能接收全部 4096 个 ghost 邻居。其他用例覆盖空输入、纯 ghost、连续空格的精确 offset、相同位置不同 ID、截断内外及双向舍入边界、跨格重建与原快照重算、容量拒绝和密度溢出。

源码基于 `1cb423fa17c4a1ec357e7319c466d58be28fa264` 加本次工作树。构建、JUnit、包验证和依赖清单保存在本机 `results/sph-002-serial-20261119T162930/`，相关源码和验证输入摘要清单 `source-sha256.txt` 的 SHA-256 为 `7e62d824ab88576a8adc5f167a92027b3a89fa493daff144155bfc9a63241cff`。

## 复跑

在仓库根目录执行以下命令，使用与构建类型匹配的本地依赖。每次全新验证选用新的构建和日志目录。[离线依赖供应](../../docs/architecture/serial-ci-validation.md)

```sh
for config in Debug Release; do
  python3 tools/validate_serial.py --cc /usr/bin/cc --cxx /usr/bin/c++ \
    --config "$config" verify \
    --prefix ".cache/dependencies/gtest-validation/kokkos-$config" \
    --gtest-prefix ".cache/dependencies/gtest-validation/gtest-$config" \
    --work-dir build/sph-002-rerun --logs "results/sph-002-rerun/$config" \
    --jobs 4 --precisions float double --require-network-blocked || exit 1
done
```

已有构建可单独运行三维用例。

```sh
ctest --test-dir build/sph-002-rerun/release-double -R 'Sph(CellGrid|Participant3D)\.' --output-on-failure
```

## 接入边界

三维密度既可通过方法接口调用，也可由 `Simulation` 管理单进程 Serial 试算、提交和恢复。WCSPH 在同一试算格表上计算压力加速度。终点位置可由调用方提供，或在全 owned 模式由 Velocity Verlet 从初始状态推进。MPI halo、持久 checkpoint 和其他执行空间待各自接入与验证。[压力窗口](wcsph-pressure-serial.md) · [多步积分](wcsph-verlet-serial.md) · [P1-07 一维固定 CSR](serial.md)
