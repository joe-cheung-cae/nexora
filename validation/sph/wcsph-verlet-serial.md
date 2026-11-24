# 三维 WCSPH 的 Velocity Verlet 与 CFL 验证

Kokkos Serial 的三维窗口接收全 owned 流体的位置、质量、稳定 ID、归属、初始速度和显式初始密度。速度和密度按流体输入行排列。初始化后，调用方可释放或改写这些缓冲。每个算例指定统一的三维 M₄ 平滑长度 `h`（默认 1 m），支撑半径为 `2h`，边界非周期。Tait 表压与对称压力加速度使用本阶段流体密度。算例可另供恒定重力与 Adami 固定壁。[参与者接口](../../src/participants/sph/Participant3D.hpp) · [压力公式](wcsph-pressure-serial.md)

## 调用

下面的片段位于返回 `nexora::Status` 的函数内，应用已初始化 Kokkos。调用方给出参考密度、声速、Tait 指数和 CFL 安全系数，初始化、准备各一次，之后每个 `advance` 执行并提交一步。

```cpp
using namespace nexora;
sph::WcsphVerletParameters parameters{{rho0, c0, gamma}, cfl};
sph::Participant3D fluid(initial_fields, max_cells, parameters, initial_velocity, initial_density);
Simulation simulation(fluid);
if (auto s = simulation.initialize(); !s.ok())
    return s;
if (auto s = simulation.prepare(); !s.ok())
    return s;
for (std::size_t step = 0; step < step_count; ++step) {
    const auto limit = fluid.time_constraint(simulation.time().accepted_time).maximum_step;
    if (auto s = simulation.advance(std::min(requested_dt, limit)); !s.ok())
        return s;
}
return simulation.finalize();
```

需要检查中间状态时可用 `begin_window(dt)`、`advance_trial()`、`trial_snapshot(stamp)`，再选择 `commit_window()` 或 `restore_window()`。快照以 `row_id(row)` 标识粒子，同时提供 `position`、`velocity`、`density`、`pressure` 和 `pressure_acceleration`。省略初始密度时，只能读取初态核求和诊断，开窗返回 `unsupported`。普通 ghost 的同阶段重构可由独立压力算子在给定格表上评估。

## 一步内的计算与发布

| 次序 | 计算与使用的状态 |
|---|---|
| 开窗 | 从 accepted 快照读取声学步长上限。如果请求的 `dt` 更大，开窗失败，时钟和 accepted 状态保持原值。 |
| 半步速度与位置 | 从 accepted 的 $x_n,v_n,a_n$ 求 $v_{1/2}=v_n+(\Delta t/2)a_n$，再求 $x_{n+1}=x_n+\Delta t\,v_{1/2}$。两次半步所用的 $a$ 都是压力加速度与重力之和。使用请求的 `dt`，即使绝对时间相加发生舍入。 |
| 更新邻域与压力 | 按新坐标更新三维格表。同格移动复用 `cellBegin` 后重新判距，跨格时重建。在旧状态和新位置的预测速度上分别求连续性密度率，取梯形平均更新 $\rho_{n+1}$，再求 $p_{n+1}$ 和 $a_{n+1}$。 |
| 后半步速度 | 用新格表的 `sorted_to_input` 找回同一粒子的半步速度，求 $v_{n+1}=v_{1/2}+(\Delta t/2)a_{n+1}$。半步速度、位置和最终速度分别存为配置的 `real`；乘加使用 double。 |
| 提交或恢复 | 一次发布位置、速度、密度、压力、加速度和下一步的 CFL 上限。试算期间对外查询的 CFL 上限始终来自 accepted 快照。恢复或失败会丢弃候选，旧快照在 Kokkos 生命周期内仍可读。 |

速度为笛卡尔系的 m/s，压力加速度为 m/s²。每个 owned 粒子的字段组包含八个 `real`（密度、表压、三分量加速度和速度），位置随几何快照保活。格表收集完新位置后，同一试算缓冲改存连续性计算所需的速度。成功步可复用半步速度、阶段缓冲、校正密度、密度率与行映射。预算吃紧时释放已用完的工作区并重新申请所需额度。持有 trial 的读者时，预提交为全部可重绑定字段创建副本，防止后续提交影响读者。[积分算子](../../src/methods/sph/Verlet3D.cpp) · [窗口与预算](../../src/participants/sph/Participant3D.cpp) · [Velocity Verlet 公式](https://manual.gromacs.org/documentation/2024-current/reference-manual/algorithms/molecular-dynamics.html)

## 声学 CFL 上限

$$
\Delta t_{\max}=\frac{C\,h}{c_0+\max_i\lVert\mathbf v_i\rVert_2},
\qquad 0<C\le1.
$$

采用配置的参考声速 $c_0$，用已提交的全步速度求最大欧氏模长。初始时计算一次，每次积分在试算速度完成后计算候选上限。空粒子集合和全静止情形取 $Ch/c_0$。等于返回上限的正步长可用，大于上限一个 double 间隔的步长也会被拒绝，由调用方决定是否缩步。该判据用于本轮压力驱动算例。[声学 CFL 的定义，式 23](https://www.sphinxsys.org/html/theory.html)

## 逐粒子对照

参考轨迹用独立的全粒子对 CSR 核求和生成测试所供初始密度。每步按旧速度和新位置的预测速度全扫描求两阶段密度率，用校正密度和独立 CSR 计算压力力，完成两次半步速度更新和一次位置更新。生产轨迹只在格表上判距。两个轨迹按稳定 ID 精确核对邻居集合，并逐粒子核对位置、速度、密度、表压和三个加速度分量。在实际位置上另做一次独立 CSR 检查，区分累计轨迹舍入与单步力误差。[CSR 与压力对照](../../tests/methods/sph/wcsph_oracle.hpp) · [积分回归](../../tests/methods/sph/verlet_window_tests.cpp)

- 两个无相互作用的粒子连续匀速推进 6 步。质量不等的受压粒子对推进 12 步（每步 0.02），各步检查三个方向的总动量以及随速度调整的 CFL 上限。
- 8 个粒子的三维算例采用乱序输入、不同质量和初速，在 8 步中改变邻居集合。另一算例从已提交状态撤销 0.04 的试算，改用 0.01 重试，检查原 accepted 与保留的 trial。
- 一次合法 CFL 步使新格表超过容量，检查自动回滚和较短步长重试。另测空输入、非法初速或 CFL、普通 ghost、缺显式密度时拒绝开窗，以及输入缓冲在初始化后被修改时的独立状态保留。

第 $n$ 步的跨轨迹对照取 $e_n=256(n+1)\epsilon_{real}$。位置、速度的分量误差按 $\max(1,|\mathrm{reference}|)$ 缩放，密度按参考密度缩放，表压按 Tait 压力尺度缩放，加速度按各邻居绝对贡献的压力尺度和乘 $4e_n$ 限定。相同几何上的力对照使用压力算子更紧的单步容差。两粒子总动量残差按各轴动量绝对值之和乘 $64(n+1)\epsilon_{real}$ 检查。

## 核求和密度版本的历史实测

四组验证于 `2026-11-20T10:01:24+08:00` 至 `2026-11-20T10:02:50+08:00` 在 Apple M2 Pro、macOS 27.2.1 arm64（26B131）上完成。使用 AppleClang 21.0.0.21000401、C++20、CMake 4.4.3、Ninja 1.13.2、Kokkos 5.2.2 Serial 及 GoogleTest 1.17.0。每组从空目录配置与构建，通过禁网、依赖、数值和迁移安装包检查，退出码为 0，零跳过。

| 构建配置 | 积分回归（计入常规测试） | 常规 CTest | 安装包测试 | 包内消费者 |
|---|---:|---:|---:|---:|
| Debug、Release 的 float（各组） | 7/7 | 116/116 | 1/1 | 3/3 |
| Debug、Release 的 double（各组） | 7/7 | 117/117 | 1/1 | 3/3 |

下表为 12 步质量不等粒子对轨迹相对显式 CSR 的最大绝对差，取遍所有步与粒子。位置、速度同时取遍三个分量。Debug 与 Release 在同一精度下的记录相同。FP32 的密度与加速度对照值保留了末次存为 `real` 前的参考精度，位置、速度和压力则与按配置精度存储的参考值比较。

| 精度 | 位置 (m) | 速度 (m/s) | 密度 (kg/m³) | 表压 (Pa) | 加速度 x / y / z (m/s²) |
|---|---:|---:|---:|---:|---|
| float | 0 | 0 | 2.65e−8 | 0 | 1.18e−7 / 5.86e−8 / 2.82e−8 |
| double | 1.39e−17 | 3.33e−16 | 1.11e−16 | 1.33e−15 | 8.88e−16 / 6.66e−16 / 3.33e−16 |

在 `2026-11-20T10:02:32+08:00`，Release double 的 Kokkos Tools 跟踪记录了这 12 步中的 10 次格表复用。第 10、12 步分别重建格表。每步均执行前半步、漂移与后半步。计入初始化，密度、表压、压力加速度和 CFL 上限各计算 13 次。

本机 `results/wcsph-verlet-20261120T095800/` 保存四组 JUnit、`trajectory-errors.xml`、安装包日志、`window-trace.log` 与源码清单。源码基于 `1cb423fa17c4a1ec357e7319c466d58be28fa264` 加工作树，`source-sha256.txt` 的 SHA-256 为 `093350396be5c81a4e9f47fa6328ef9257119b38d3218c1b53440ff0c7f9c6e6`。本轮证据覆盖单进程、全 owned、固定光滑长度的纯压力推进；ghost 积分、MPI、设备执行空间和真实流体基准仍需独立验证。

## 复跑当前连续性积分

从仓库根目录运行，使用已供应的本地依赖，每次选择新的构建目录和结果目录。[离线验证工具链](../../docs/architecture/build-and-validation.md)

```sh
run=wcsph-verlet-rerun
for config in Debug Release; do
  python3 tools/validate_serial.py --cc /usr/bin/cc --cxx /usr/bin/c++ \
    --config "$config" verify \
    --prefix ".cache/dependencies/gtest-validation/kokkos-$config" \
    --gtest-prefix ".cache/dependencies/gtest-validation/gtest-$config" \
    --work-dir "build/$run" --logs "results/$run/$config" \
    --jobs 4 --precisions float double --require-network-blocked || exit 1
done
ctest --test-dir "build/$run/release-double" --no-tests=error -R 'Sph(VerletWindow|Continuity)\.' --output-on-failure
"build/$run/release-double/tests/methods/sph/nexora_sph_tests" \
  --gtest_filter=SphVerletWindow.UnequalMassTrajectoryMatchesIndependentCsrAndMomentum \
  --gtest_output="xml:results/$run/trajectory-errors.xml"
```
