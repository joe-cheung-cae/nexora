# WCSPH 三维压力算子的 Serial 验证

单进程 Kokkos Serial 上，静态构型测试按格网的 owned 行数分配输出，将只读邻域和密度输出传给 `evaluate_density_3d`。纯流体压力测试调用生产步进使用的 `evaluate_step_pressure_3d`，普通 ghost 构型调用支持本地支撑域重构的 `evaluate_pressure_3d`。两者均用有限正的参考密度、声速和指数建立 Tait 状态方程。实际 WCSPH 推进使用显式初态密度和随后由连续性方程校正的密度。[压力接口](../../src/methods/sph/Pressure3D.hpp) · [格索引](cell-begin-serial.md) · [Velocity Verlet](wcsph-verlet-serial.md)

## 由密度到压力加速度

1. 对每个 owned 粒子，用给定构型的坐标和三维 M₄ 核计算密度，自身质量项计一次。沿用固定光滑长度 `h = 1`、支撑半径 `R = 2` 和非周期边界。
2. 从本步密度求 Tait 表压。参考密度为 $\rho_0$，参考声速为 $c_0$，指数为 $\gamma$。$\rho_i=\rho_0$ 时表压为零，低于参考密度时保留负表压。[Becker 与 Teschner，式 7](https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf)

   $$
   p_i = \frac{\rho_0 c_0^2}{\gamma}\left[\left(\frac{\rho_i}{\rho_0}\right)^\gamma-1\right].
   $$

3. 复用同一份 `cellBegin`，重新判距并逐 owned 行累加压力加速度。$\nabla_i W_{ij}$ 是对目标位置的梯度。重合粒子和截断边界的核梯度为零。[Price，固定光滑长度的式 31](https://arxiv.org/html/1012.1885) · [压力算子](../../src/methods/sph/Pressure3D.cpp)

   $$
   \mathbf{a}^{p}_i = -\sum_{j\ne i} m_j\left(\frac{p_i}{\rho_i^2}+\frac{p_j}{\rho_j^2}\right)\nabla_i W_{ij}.
   $$

密度、表压和加速度以配置的 `real` 存储。压力系数直接计算 `p / (ρ * ρ)`，中间平方也按 `real` 舍入。压力力和连续性方程仅累加归一化距离 `0 < sqrt(d²)/h < 2` 的粒子对。接近参考密度时用 `log1p` 和 `expm1` 保留微小压差。密度非正、压力或最终加速度超出输出范围都会使本次试算失败。

**Ghost 输入必须覆盖参与受力的 ghost 自身的完整支撑域。** 每个对 owned 目标有非零梯度贡献的 ghost，会用当前格表中的全部本地源重算自身密度和表压。例如，距目标 `2.5` 的粒子可影响距目标 `1.25` 的 ghost，进而改变目标的加速度。按距离从 owned 集合选源时，覆盖至 `2R` 是充分条件。本阶段只发布 owned 行，MPI halo 交换待另行验证。

## 算子回归

压力算子按格表的 owned 行顺序写入调用方提供的标量压力和三分量加速度数组。普通 ghost 只作源。该算例使用完整的本地支撑域，按同阶段密度重构普通 ghost 压力；固定壁积分时，壁压按 Adami 外推。[算子回归](../../tests/methods/sph/pressure_3d_tests.cpp) · [固定壁](adami-hydrostatic-serial.md)

独立对照先全粒子两两扫描并构造显式 CSR，按稳定 ID 建行，再从这些行计算参考密度、Tait 压力及三个力分量。生产算子直接遍历格表。邻居 ID 集合要求完全相等。压力绝对误差限为 `64 ε × B[(ρ/ρ₀)^γ + 1]`，其中 `B = ρ₀ c₀² / γ`、`ε = epsilon(real)`。每个加速度分量用逐邻居绝对贡献的对应压力尺度之和乘 `256 ε`，可检查力相互抵消的情形。密度使用既有的逐行舍入界。[独立 CSR](../../tests/methods/sph/cell_grid_oracle.hpp) · [压力参考](../../tests/methods/sph/wcsph_oracle.hpp)

| 检查 | 判据 |
|---|---|
| 格内位移、质量变化与三轴跨格重排 | 在各构型上按稳定 ID 对照独立 CSR 的邻居、核密度、压力和三个加速度分量。 |
| 修改 ownership、成员和输入顺序 | 新旧构型的 owned 行分别按各自的稳定 ID 与 CSR 核对。 |
| 两个质量不等的粒子 | 正压相斥、负压相吸，力方向符合解析解，质量加权加速度之和在舍入误差内为零。 |
| 远端源、重合、支撑边界、孤立及空目标 | 改变间接源质量只经 ghost 改变目标压力加速度；重合、边界和孤立情形的加速度为零；空输入和纯 ghost 均产生零个 owned 行。 |
| 参考密度 | 检查零表压及参考密度两侧相邻 double 值的压差。 |

积分路径另核对成功发布、数值及预算拒绝后已提交字段和时间的保留。[积分窗口回归](../../tests/methods/sph/verlet_window_tests.cpp)

## 历史外供窗口实测

以下数据记录原外供窗口通过压力算子时的执行情况。

`2026-11-20T08:49:53+08:00` 至 `2026-11-20T08:51:00+08:00`，在 Apple M2 Pro、macOS 27.2.1 arm64（26B131）运行。工具链为 AppleClang 21.0.0.21000401、C++20、CMake 4.4.3、Ninja 1.13.2、Kokkos 5.2.2 Serial、GoogleTest 1.17.0。四个独立目录均完成禁网检查、配置、构建、依赖检查、常规测试和迁移安装包验证，退出码均为 0，零跳过。

| 配置 | WCSPH 回归（计入常规 CTest） | 常规 CTest | 安装包测试 | 包内消费者 |
|---|---:|---:|---:|---:|
| Debug float、Release float（各组） | 10/10 | 109/109 | 1/1 | 3/3 |
| Debug double、Release double（各组） | 10/10 | 110/110 | 1/1 | 3/3 |

在 `2026-11-20T08:50:44+08:00` 用 Kokkos Tools 复跑 Release double 的三轴跨格用例。初始建表一次，同格移动刷新一次且复用格表，跨格试算和回滚后的跨格重试各建表一次。四个阶段各执行一次 owned 密度、owned 状态方程、ghost 标记、ghost 密度与状态方程、压力加速度。持有试算读者时观测到密度、压力和加速度各复制一次。

源码基于 `1cb423fa17c4a1ec357e7319c466d58be28fa264` 加本次工作树。JUnit、安装包及依赖日志、`window-trace.log`、`source-sha256.txt` 保存在本机 `results/wcsph-pressure-20261120T084700/final/`。源码与验证输入摘要清单的 SHA-256 为 `2d70a2e7a2d3c12246eaa3f08a1f8b8e5ba5fabbd7236b9f29e803908c35a868`。本轮覆盖单进程 Serial 压力窗口。自由液面等真实算例、时间积分、MPI 与其他执行空间另行验收。

## 复跑

从仓库根目录使用本地供应的 Kokkos 和 GoogleTest 运行完整矩阵。将 `run` 设为尚未使用的目录名。[离线依赖与工具链](../../docs/architecture/build-and-validation.md)

```sh
run=wcsph-pressure-rerun
for config in Debug Release; do
  python3 tools/validate_serial.py --cc /usr/bin/cc --cxx /usr/bin/c++ \
    --config "$config" verify \
    --prefix ".cache/dependencies/gtest-validation/kokkos-$config" \
    --gtest-prefix ".cache/dependencies/gtest-validation/gtest-$config" \
    --work-dir "build/$run" --logs "results/$run/$config" \
    --jobs 4 --precisions float double --require-network-blocked || exit 1
done
ctest --test-dir "build/$run/release-double" --no-tests=error -R 'SphWcsphOperators[.]' --output-on-failure
```
