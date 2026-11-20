# P1-07 SPH Serial CPU 验证

`nexora_methods_sph` 使用指定的 Kokkos Serial 实例计算 [P1-01b](../baselines/sph-density-1d-v1.md) 的固定 CSR 一维 M₄ 密度。每个 owned 行写入一个独立 trial 值，ghost 只供读取。支持边界和核权重直接由粒子距离求得。同一次 Kokkos 归约把逐行 `real` 密度写入 trial，并用普通 `double` 累加 owned 和。归约值有限且未超过 `real` 上限才转换和发布。 [密度算子](../../src/methods/sph/Density.cpp) · [SPH participant](../../src/participants/sph/Participant.cpp)

## 状态与验证范围

- 所有数值 Views 位于 HostSpace。participant 验证输入的 ID、owned 行、CSR 边和字段，持有位置与质量的只读 lease。固定邻域的完整性由输入给定，基准用独立扫描核对。试算密度由写 lease 独占。提交时仍有 trial 读授权，则为 accepted 另取独立存储。其余情况下复用试算密度的原分配并签发新的 accepted 读授权与 stamp。原 trial stamp 随窗口关闭而失效。
- 重排使用已有双向排列同时迁移行列、字段和 CSR，发布时推进 layout 并保留 geometry、topology 和 accepted 时间。首次密度发布前的快照使用当前位置分配的 `storage_incarnation`，密度快照使用其密度分配的身份。重排拒绝过期和其他实例的 stamp。已签发的读快照在新状态发布、回滚和关闭后继续有效。
- 跨窗口读取持有完整 `Snapshot`。按索引读取返回数值，副本共享同一快照记录和读授权。预算分别绑定整套不可变布局与密度分配，重排另预留临时排列和 CSR 的字节数。最后一个持有者释放后归还额度。默认上限为 64 份快照记录与 256 MiB 受控载荷，受控载荷按 Kokkos 数组逻辑字节计；进程 RSS 包含主机输入、校验容器和运行时开销。 [快照接口](../../src/participants/sph/Participant.hpp)
- 支持范围为单进程同步 Serial、固定粒子集合和本地 ghost 输入。快照与恢复限进程内。CUDA、跨 rank 通信、动态邻域、持久 restart 和生产 SPH 分别保留在相应任务。 [验收矩阵](../../docs/architecture/phase-1-design-review.md#acceptance)

## 常规公式复验

以 `e39ca40` 为基线，在工作树中直接求值一维和三维密度、Tait 压力、Adami 壁面、连续性方程、声学 CFL 与 Velocity Verlet。测试用实际的物理、几何和预算失败验证回滚与重试。保留冻结 CSR 对照、长邻域、守恒、静水初态和快照保活检查。

在 Apple M2 Pro、macOS 27.2.1 arm64（26B131）、AppleClang 21.0.0.21000401、CMake 4.4.3、Ninja 1.13.2、Kokkos 5.2.2 Serial 和 GoogleTest 1.17.0 上，重新构建了已有的四套配置。各配置用 `ctest -R '^nexora_sph_tests\.'` 执行注册的 SPH 测试。

| 配置 | 通过 | 完成时间 |
| --- | ---: | --- |
| Release float | 63/63 | `2026-11-20T15:04:37+08:00` |
| Release double | 63/63 | `2026-11-20T15:04:38+08:00` |
| Debug float 和 Debug double（各组） | 63/63 | `2026-11-20T15:04:54+08:00` |

Release double 的依赖检查及全仓 CTest（含安装包消费者）也通过，共 129/129 项，于 `2026-11-20T15:05:40+08:00` 完成。另以 Release double 执行 `nexora_sph_benchmark 256` 和 `nexora_sph_tank_trajectory 6 <输出目录>`。冻结一维基准与重排对照通过。三维水箱诊断执行了 601 步，达到 2.34765625 s，216 个流体粒子全程无穿壁记录。两项均于 `2026-11-20T15:05:28+08:00` 完成。

### 数值检查与 CFL 复验

以 `d60beb7` 为基线，压力、Verlet 和连续性更新采用有序范围判断。CFL 对初始化或最终半步已验证的有限速度取最大速度平方，并验证所得时间步。负例按校验行为合并，仍覆盖无效坐标与初速、容量拒绝和密度试算回滚重试。

沿用上述工具链和四套构建目录。float 与 double 的 Release SPH 测试各通过 63/63 项，于 `2026-11-20T15:32:57+08:00` 完成。float 与 double 的 Debug SPH 测试也各通过 63/63 项，于 `2026-11-20T15:33:15+08:00` 完成。Release double 的依赖检查及全仓 CTest（含安装包消费者）通过，共 129/129 项，于 `2026-11-20T15:33:52+08:00` 完成。

## 历史离线执行

Apple M2 Pro arm64（8P+4E，16 GB），macOS 27.2.1 build 26B131，AppleClang 21.0.0.21000401，CMake 4.4.3，Ninja 1.13.2，Python 3.14.7。Kokkos 5.2.2（`8e15454876298832f013ebcaf08d53825b5bda40`）用校验过的官方归档分别供应 Debug/Release，两组安装均核对逐文件摘要。MPI 关闭。两轮验证前后的 10 项 IPv4/IPv6 网络探针均返回 EPERM。 [依赖来源](../../third_party/README.md#kokkos-provenance-record)

四组 Serial 配置分别从干净构建目录完成编译和依赖检查，20 项常规 CTest、1 项安装包 CTest 均通过，零跳过。安装包搬迁后又分别构建并执行 3 个独立消费者，各配置的 Reference Decimal oracle 均通过。

| 配置 | 配置开始 | 搬迁消费完成 | CTest | 独立消费者 |
| --- | --- | --- | --- | --- |
| Debug float | `2026-11-19T11:39:07+08:00` | `2026-11-19T11:39:34+08:00` | 21/21 | 3/3 |
| Debug double | `2026-11-19T11:39:34+08:00` | `2026-11-19T11:40:00+08:00` | 21/21 | 3/3 |
| Release float | `2026-11-19T11:39:07+08:00` | `2026-11-19T11:39:33+08:00` | 21/21 | 3/3 |
| Release double | `2026-11-19T11:39:33+08:00` | `2026-11-19T11:39:58+08:00` | 21/21 | 3/3 |

SPH 测试按每个固定输入生成一次独立 scalar oracle，在首次提交、行列重排和重算后核对结果。覆盖空行、4096 邻居行和闭支持域两侧。保留完整 accepted 或 trial 快照，检查跨提交、恢复和关闭读取、最后一份副本释放后的预算恢复，以及失败重排原子性。另执行冻结算例的分数复核，两个 CSR 布局、逐 ID 密度及 owned 和均精确匹配。 [CTest 用例](../../tests/methods/sph/sph_tests.cpp) · [冻结基准](../baselines/sph-density-1d-v1.md)

源码基线 `1e4e10c7a4643a3ab97129e210e2efd369598e65` 加工作树修改。最终源码、夹具和构建输入清单 SHA-256 为 `960797be742b7d5ebf3f2349eb75a7f3eedaf6c0e7bb933980b01879c02fe064`。本地 JUnit、依赖清单、精确复核、计时样本和源码清单存于 `results/p1-07-serial-simplified-20261119T113427/final/`。精简前的原始计量与验证保存在 `results/p1-07-view-lifetime-20261119/`，此前的依赖检查反例见 `results/p1-07-review-fixes-20261119/`。

## 端到端样本

Release float/double 各执行 1 次预热、3 次计时和 1 次独立 Kokkos Tools 计量。小例为冻结六粒子算例。大例生成 262144 个以 8 m 间隔平移的六粒子组，将质量按 `1 + (组号 mod 4)/8` 缩放，另加入 4096 邻居的 owned 行。大例有 1310721 行、1576961 列、3936256 条 CSR 边。独立计量轮按 stable ID 校验首次 accepted、重排后的已发布值和重排后再次求值，另外检查重排逐值精确保留。大例归约对照独立逆序求和及随项数增长的舍入界。冻结算例继续使用原有 `8ε`／`16ε`。

端到端使用连续 wall-clock，从输入构造前到第二次窗口的 accepted 快照可读，覆盖 participant 装配、字段转换与初始化、第一次窗口、行列与字段重排、释放旧快照和第二次窗口。旧快照释放归入重排阶段。Kokkos runtime 在计时前初始化，清理另行计时。Kokkos Views 峰值按回调记录累计，拷贝统计为 `deep_copy` 报告的本地主机字节。进程峰值 RSS 按整个默认运行记录，包含小例和大例、主机临时容器、运行时及计量轮。当前窗口的数据驻留 HostSpace，ghost 由输入供给。

Release float 计量从 `2026-11-19T11:40:26+08:00` 至 `2026-11-19T11:40:29+08:00`，double 从 `2026-11-19T11:40:29+08:00` 至 `2026-11-19T11:40:31+08:00`。两种精度依次执行。热样本取 3 次中位数；对照列来自上一版相同硬件、输入与计时范围。

| 用例 | 本轮首轮 ms | 上一版热端到端 ms | 本轮热端到端 ms | 本轮首次窗口 ms | 本轮重排 ms | 首次窗口百万边/s |
| --- | --- | --- | --- | --- | --- | --- |
| 六粒子 float | 0.2169 | 0.04075 | 0.06563 | 0.005208 | 0.03267 | 2.88 |
| 六粒子 double | 0.2796 | 0.04054 | 0.04913 | 0.00475 | 0.02442 | 3.16 |
| 大例 float | 491.7 | 386.4 | 407 | 8.835 | 203.1 | 446 |
| 大例 double | 463.6 | 376.5 | 384.7 | 5.48 | 192.5 | 718 |

大例端到端的三次热样本范围为 float 394.80–416.12 ms、double 377.90–396.84 ms。两种精度的中位数均高于上一版。大例重排重算的精确参考和为 `73654285/12 kg/m`，float 与 double 的相对误差分别为 `1.36e-8` 和 `2.39e-12`。 [基准与测量阶段](../../benchmarks/sph/sph_benchmark.cpp)

| 用例 | 峰值 Kokkos 字节 | 分配 / 释放次数 | 累计分配字节 | 本地复制次数 / 字节 | fence 次数 / 回调 ms |
| --- | --- | --- | --- | --- | --- |
| 六粒子 float | 832 | 27 / 27 | 856 | 4 / 60 | 33 / 0.002251 |
| 六粒子 double | 960 | 27 / 27 | 1000 | 4 / 108 | 33 / 0.002042 |
| 大例 float | 200507600 | 27 / 27 | 205750488 | 4 / 15769610 | 33 / 0.001916 |
| 大例 double | 236224720 | 27 / 27 | 246710488 | 4 / 28385298 | 33 / 0.001624 |

四个用例的 Kokkos Views 峰值、分配次数与字节数、复制次数与字节数、fence 次数均与上一版相同。退出计量时在用 View 字节归零。默认进程峰值 RSS 取 Darwin `wait4` 的 `ru_maxrss`，float 为 362168320 字节（上一版 365379584），double 为 381124608 字节（上一版 327106560）。

基准的受控载荷上限为 512 MiB。另于 `2026-11-19T11:40:53+08:00` 至 `2026-11-19T11:40:57+08:00` 运行 Release double 的 524288 组上限输入，正常退出，峰值 Views 472154320 字节，进程峰值 RSS 738639872 字节。

精简前后 Release 大例的分阶段配对复测、原因和修正复验见[性能归因记录](serial-release-crossover.md)。

## 复跑

本次在禁网环境中分别对 `serial_config=Debug` 和 `serial_config=Release` 执行以下命令。`kokkos_prefix` 指向对应配置已供应的本地安装，`serial_work` 与 `serial_logs` 指向新的构建和日志目录。

```sh
python3 tools/validate_serial.py --cc /usr/bin/cc --cxx /usr/bin/c++ --config "$serial_config" verify \
  --prefix "$kokkos_prefix" --work-dir "$serial_work" --logs "$serial_logs" \
  --jobs 3 --precisions float double --require-network-blocked
```

将 `nexora_release_build` 分别设为 `$serial_work/release-float` 与 `$serial_work/release-double`，依次计量。可执行文件默认测试上表的两个规模，也接受显式组数。

```sh
cmake --build "$nexora_release_build" --target nexora_sph_benchmark
"$nexora_release_build/benchmarks/sph/nexora_sph_benchmark"
```

P1-07 的 CUDA 数值、设备交接和端到端计时待 Linux NVIDIA runner 和 P1-02/04/05 的 CUDA 配置、交接与执行条件。 [CUDA 构建契约](../../docs/architecture/build-and-validation.md#cuda-build-contract)
