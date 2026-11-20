# Nexora Phase 1：Kokkos 基础与代表性切片评审

修订时间 `2026-11-20T19:20:01+08:00`。**当前已实现 Serial CPU 基础与 Reference 同步窗口，包括字段权限、本地传输、最小历史恢复和预算。P1-07 的 Serial CPU 代表性切片通过审查修复后的回归验收。完整 Phase 1 仍开放。**

唯一架构入口：[overview](overview.md)。关键类型和生命周期：[kokkos-design](kokkos-design.md)。当前源码、旧类型/target 处置及归档逐项映射：[migration-roadmap](migration-roadmap.md)。依赖版本和离线过程：[build-and-validation](build-and-validation.md)。

## 1. 当前基础与 Phase 1 出口

当前实现包含 exact Kokkos Serial 包、typed View 执行测试、经 `nexora::data` 的安装消费，以及使用真实 Kokkos 数组的 reference 状态/参与者。后者复用现有 Simulation，覆盖独立试算、提交、主动恢复、数值失败、重试和重放。旧 host Buffer、同步 ExecContext、旧字段 View 与其导出已删除；字段元数据及单参与者控制保留。没有真实方法求解器、分布式 halo 或持久 checkpoint。实际证据见 [CPU 切片验证](kokkos-cpu-slice-validation.md)和[参考生命周期验证](reference-lifecycle-validation.md)；dummy 测试和历史 CPU/设备描述不替代新路径的执行证据。

P1 直接交付 Kokkos 默认 CPU 和可选 CUDA 的**同一算子源码**、typed 字段、权限/外部 owner、accepted/trial 与恢复，以及 SPH 非规则邻域、FVM 面通量/单元累积两个代表性结构切片。简单标量 reference 保留用于基础生命周期测试。P1 不关闭完整 SPH/DEM 或 FVM solver 的物理验证，不提前宣称 FEM/LBM/MBD 已可用。

**完整 P1 关闭条件：** 下述 V1–V10 的 P1 项目全部满足，含真实 CPU/CUDA 的代表性数值和端到端性能；旧通用执行层删除。可以先验收并发布明确标注的 CPU 切片；若 CUDA 实机证据缺失，完整 P1 保持开放，不能把 CPU 结果代替它。MPI 诊断保持可选，P2 MPI 数值与持久重启另有门槛。HIP 只在后续真实硬件通过后列为运行支持。

## 2. 实施任务与依赖

下表定义**完整任务的实施落点与验收**。当前落实 P1-01a/02 的 Serial 构建子集及 P1-03/04 的字段绑定、整组读写 lease、外部 HostSpace owner、稳定 ID、双向排列、CSR 与本地交换。P1-05/06 的 Reference Serial 子集使用显式执行实例，已验收独立 accepted/trial、最小 replay history 与控制恢复、失败重试和快照预算。`TimeInterval` 已下移 core。完整 P1 还需设备异步交接、CPU/CUDA 同源实例化与实机验证，以及 SPH/FVM 代表性方法的状态与恢复验收。内部 target 使用 `nexora_*`，导出 target 使用 `nexora::*`。 [实体与拓扑验证](entity-topology-validation.md) · [状态重排与整组授权](structure-execution-validation.md) · [Reference 进程内恢复与预算](reference-resource-validation.md)

| 任务 | 文件/目录及 target | 接口与前置依赖 | 通过条件 |
|---|---|---|---|
| **P1-01 基线、来源与锁定** | `docs/architecture/`、`third_party/` 依赖清单、拟 `validation/baselines/`；无新功能 target | 前置：维护者批准具体实施包。分为 01a 工具链/依赖/现状锁定（Kokkos commit、C++20、来源、消费面，放行 02），01b reference 及 SPH/FVM 数学基准（输入/公式/容差，各自放行 06/07/08）。真实 SPH/DEM BASE-001 独立追踪。 | 01a 有可追溯版本/许可证/制品摘要；01b 记录已有基线和缺口。SPH/FVM 基准未批准只阻塞对应数值任务和完整 P1，不阻塞通用字段/owner 工作。 |
| **P1-02 Kokkos 构建与包** | 根 `CMakeLists.txt`、`CMakePresets.json`、`cmake/NexoraConfig.cmake.in`、`cmake/NexoraTargetGraph.cmake`、`tests/package/`；`build_options/core/data_schema/data` | 前置 P1-01a。C++20、必需 exact Kokkos installed package、Serial 默认；首个 CUDA 按 [nvcc_wrapper/CXX 契约](build-and-validation.md#cuda-build-contract)，仅 CUDA+Serial、单一 GPU 架构、匹配安装消费者；MPI 独立可选。删除旧 `NEXORA_GPU_BACKEND`，从已验证的 `Kokkos_DEVICES` 确定构建组合，配置拒绝不兼容包/编译器。 | V1 的依赖/安装冒烟子项：断网构建与消费可直接创建 Kokkos View/parallel_for，不依赖尚未完成的新字段接口；缺依赖失败，不联网/fallback。完整已迁移 consumer、搬迁和旧包清理由 P1-11 最终验收。 |
| **P1-03 语义与 typed 字段** | `include/nexora/data/{FieldDescriptor,FieldRegistry,FieldBinding}.hpp`、`src/data/`、`tests/data/`、拟 `include/nexora/core/Time.hpp`；`nexora::data_schema/data/core` | 前置 P1-02。拆 descriptor/FieldStamp；ReadField/WriteField 直接绑定 typed View；ID、permutation、topology/layout/incarnation；将共享 TimeInterval 下移 core，保持 duration 语义；取消 FieldKernelView 构想。 | V2；形状、版本、访问和稳定 ID 正反例；Kokkos 类型仅在指定头子集；core/schema 无设备 API。 |
| **P1-04 lease、外部 owner 与交换快照** | `include/nexora/data/{FieldAccess,TransportView}.hpp`、`src/data/`、`tests/data/`、`tests/interop/`；`nexora::data`，必要私有 `nexora_interop_cuda` | 前置 P1-03；CUDA 部分集成 P1-05。整 allocation 权限仲裁、managed/unmanaged、可观察外部 owner、进程内 consumer-complete、显式 pack/staging。只实现 P1 本地 TransportView fixture，MPI request 在 P2。 | V3/V4 低层子项；提前释放 caller、多读者、写冲突、关闭、部分失败均可观察；正确复制/转换与保活，不新建 allocator/queue/event。窗口恢复 V7 在 P1-06 之后验收。 |
| **P1-05 typed 算子与 execution 实例** | `src/methods/reference/{Relaxation,Instantiations}.hpp/.cpp`、`tests/methods/`、必要 `src/interop/cuda/` 私有实现；`nexora_methods_reference` | 前置 P1-02/03；与 P1-04 联调。同一 template source 实例化 Serial/OpenMP（选中时）/Cuda；每次 dispatch 显式 exec 实例；仅在必要边界 fence。 | V3/V5；CPU/CUDA 同输入数值；非默认实例、跨实例交接和外部 stream fixture；拒绝未构建组合；原生类型仅私有互操作。 |
| **P1-06 参考参与者与窗口事务** | `src/participants/reference/`、`include/nexora/participant/`、`src/simulation/`、`tests/simulation/`；`nexora_participants_reference`、`participant_api/simulation` | 前置 P1-03/04/05 及 P1-01b reference 基准。独立 accepted/trial、历史与控制、缓存键、恢复/重放、快照背压；保持高层同步接口，不对 Simulation 加设备模板。 | V5/V7 的 reference 子项；真实 Kokkos 数组上的失败/恢复/重试，与已接受状态/时钟一致；旧 dummy 的失败语义继续覆盖。SPH/FVM 的 V7 在对应 07/08 集成后汇总。 |
| **P1-07 SPH 代表性切片** | `validation/sph/fixed_csr/`、`tests/methods/sph/`、`benchmarks/sph/`，共用 `nexora_sph_fixed_csr_fixture`。 | 前置 P1-01b 对应数学基准、P1-03、P1-04、P1-05、P1-06。用固定 CSR 验证非规则邻域、稳定 ID、粒子场和获准阶段公式及归约。 | V6-SPH/V8；零/长邻域、重排、版本失效、CPU/CUDA、历史回放与含转换的端到端耗时；不能只提交规则 SAXPY 结果。 |
| **P1-08 FVM 代表性切片** | `src/methods/fvm/` 的 FaceFlux/CellAccumulation/Topology；`tests/methods/fvm/`、`validation/fvm/`、`benchmarks/fvm/`；`nexora_methods_fvm` | 前置 P1-01b 对应数学基准、P1-03、P1-04、P1-05、P1-06。cell/face 分域、owner/neighbour/patch/CSR；独立 face 输出→cell 累积，保留原守恒方向约定。 | V6-FVM/V8；内部面唯一逻辑通量、边界面、非均匀连接；CPU/CUDA 和全局通量预算通过；完整压力校正在 P3。 |
| **P1-09 case 与装配能力** | `apps/nexora/`、拟 `schemas/reference/`、`cases/reference/`、CLI tests；`nexora` | 前置 P1-02/06；schema 可在契约冻结后准备。versioned key=value reference fixture、validate/plan/run 共用校验；built/available/validated 分开；取消 provider 选择键。 | V9；非法/重复键、时间/量子/NaN/Inf、未支持组合明确失败；plan 列复制、布局转换与同步边界；run 不隐式选另一设备。 |
| **P1-10 验收与端到端性能** | `tests/`、`validation/`、`benchmarks/`、`.github/workflows/ci.yml` 与受控 GPU runner 文档；测试叶 target | 前置 P1-02、P1-03、P1-04、P1-05、P1-06、P1-07、P1-08、P1-09；各 profile 的预检可随对应任务开展，汇总关闭须全部满足。固定独立 oracle、重复策略、manifest 和时序统计。受控致命错误只在隔离测试进程，不在正常进程强行恢复设备。 | V1–V9 的候选行为与 V10 预检，全部结果按实际 profile 记录；V1 最终包消费和 V10 最终删除检查由 P1-11 完成。硬件/驱动/编译器/架构/输入/hash/时间/skip 俱全，无设备不算通过。 |
| **P1-11 旧层切换与最终验收** | 已提前移除 `include/nexora/exec/`、`src/exec/`、`FieldView.hpp`、旧 core 数值容器及旧 exec targets/tests；更新 checker、CMake exports 和安装消费者。 | 本次明确授权源码及导出清理，独立于其余 P1 验收的顺序；不存在旧兼容别名或替代 facade。最终交付仍须完成全部相关 consumer、目标图与已启用设备路径验收。 | V1/V10 最终干净构建、安装消费、依赖正反例及完整配置证据仍待汇总；本次删除不关闭 P1-11 全部验收或完整 P1。 |

验收依赖主线为 `01a → 02 → 03 → (04,05) → 06 → (07,08,09) → 10 → 11`，01b 的各数学基准分别约束对应数值任务。04/05 的低层 CPU 子项可分别完成后进入 06，CUDA 联调证据在 10 汇总；07/08 可并行。本次用户授权已提前实施旧 core 容器与 exec 删除，不等待全部验收前置；不发布双基础设施、长期兼容 facade 或 native fallback。P1-02 只要求早期包冒烟，P1-11 完成最终包消费/V10 删除检查，避免循环依赖。

工程量在数学基准、runner 和外部消费面确认后重新估计；原 10–17 人日估计作废，不能用于新范围排期。

## 3. 现有数值与代表性工作负载

### 3.1 保留的 reference 夹具

保留原稿的无量纲 `x,q`、`Association::node`、`entity_set=reference_samples` 与单位 `1`，公式为 `d=dt×λ×x`、`trial_x=x−d`、`trial_q=q+d`。有限输入、正步长及 `0<dt×λ≤1`；积分使用现有 `TimeInterval::duration()`，不得用舍入绝对时间端点相减替代。该模型已作为内部 Serial 基础 fixture 实现并执行，不是产品 solver，也不提供 GPU 证据。

double 配置的原稿数值门槛不变：`λ=1 s⁻¹, x₀=1, q₀=0, T=1 s`，16/32/64 步；对照 `exp(−t)` 与独立显式 Euler 离散值，两次观测阶 `0.9–1.1`，`abs(x+q−1)≤1e−12`；每分量 CPU/CUDA `abs(diff)≤1e−12+1e−10×abs(ref)`。这些容差只对 reference 的 double 配置生效，不能套到 SPH、FVM 或真实耦合。现有 reference 字段和算子已迁 `real`，时间保留 double。float 对此单位幅值、λ=1 的夹具采用 `8Nε + N|real(dt)−dt|` 绝对舍入界，保持两次观测阶 `0.9–1.1`；double 不放宽旧门槛。两种配置另以 Python Decimal 80 位闭式 Euler/解析解核验，适用条件、界的依据与实际误差见[数值迁移验证](real-numerics-validation.md)。

### 3.2 SPH 非规则邻域切片

必须包含不规则空间分布、不同邻居数、空邻域、长邻域、boundary/ghost 标志及至少一次粒子重排。数据规模包含可人工检查的小例和超出缓存/体现设备带宽的案例；冻结邻域列表、self inclusion、支持半径和对称性约定。测试执行邻域 gather、获准的密度或相互作用阶段以及全局归约；独立 scalar oracle 在测试中计算同一公式，不能成为产品 native CPU 路线。

`P1-01b` 的 [SPH 一维密度小算例](../../validation/baselines/sph-density-1d-v1.md) 已冻结核函数、输入、精确参考答案、误差判据及 SHA-256。P1-07 按此基准精确核对稳定 ID、CSR 关系和重排，并按精度核对密度与归约。真实 `BASE-001/SPH-001` 继续独立验收。

P1-07 的 Serial CPU 验证包含密度与归约边界、重排和版本失效、快照预算、失败恢复和重放，以及小／大规模端到端与内存计量。固定 CSR 实现由一维测试与显式 benchmark 共用。三维静态几何、密度与压力通过格索引和算子接口验证，三维 WCSPH 时间推进通过内部 `sph::Participant3D` 的 Simulation 窗口验证。CUDA 实机数值和性能仍待验收。[Serial CPU 验证记录](../../validation/sph/serial.md) · [三维邻域设计与实施](migration-roadmap.md#sph-cell-begin)

#### 3.2.1 Serial 固定 CSR 验证夹具

固定 CSR 在初始化时用计算精度下的距离校验每条边是否位于闭支持域内。重排保留邻接关系。M₄ 算子按原 CSR 顺序计算每个 owned 行，单计自身项，ghost 只读。核权重由现行分段多项式直接求值，再乘邻居质量。[密度算子](../../validation/sph/fixed_csr/Density.cpp) · [边界回归](../../tests/methods/sph/sph_tests.cpp)

一次 Kokkos `parallel_reduce` 写入每行的 `real` 密度，并将该值累加到普通 `double` 诊断和。归约结果有限且落在 `real` 的范围内才转换并交给 participant。失败时恢复已提交状态。[Kokkos 归约语义](https://kokkos.org/kokkos-core-wiki/API/core/parallel-dispatch/parallel_reduce.html#semantics)

跨窗口读取持有完整 `Snapshot`，通过数量及按索引取值的接口读取布局和密度。副本共享快照记录与读授权，布局和密度各自持有载荷预算。重排单独预留临时空间；提交时如有读者保留 trial，参与者先为 accepted 密度分配并复制。Serial 四配置及两种精度的基准结果见验证记录。[快照接口](../../validation/sph/fixed_csr/Participant.hpp) · [Serial 验证](../../validation/sph/serial.md)

### 3.3 FVM 面通量与单元累积切片

保留控制体、积分面通量和符号方向契约：
`d(V_c U_c)/dt + Σ_f s_cf F_f = V_c S_c`，每个内部面只有一个逻辑 `F_f`，两侧符号相反。这沿用历史物理约定，不改离散公式。

夹具包含非均匀体积、不同每单元面数、内部面和 boundary patch、owner/neighbour，与 cell-face CSR 的双向一致性；两阶段为获准的面通量计算，然后用 Kokkos cell gather/累积验证预算与结果。可用预先规定的 face-flux 数据独立验证累积，但**只拷贝预给通量不足以关闭 face-flux 计算任务**：P1-01 仍须冻结有独立 oracle 的具体标量通量公式/边界方案和误差标准。完整压力–速度耦合、GCL、收敛阶在 P3 另验，不以结构切片冒充完整 FVM 认证。

新增基准若涉及数值公式、默认值或容差选择，须记录独立数学决策。SPH 小算例记录见 §3.2，FVM 基准待定。性能样本必须和数值样本使用相同算法约定。

<a id="acceptance"></a>
## 4. 构建、数值、生命周期与性能验收矩阵

**以下是完整 Phase 1 的通过条件；有限 CPU 切片的单独证据见[基础切片记录](kokkos-cpu-slice-validation.md)和[参考生命周期记录](reference-lifecycle-validation.md)，不代表这些完整门槛通过。** 每个报告记录 ISO 8601 秒级时间及 timezone；CPU 型号/线程/NUMA、GPU 型号/内存/架构、OS、compiler/stdlib、Kokkos commit/digest/options、toolkit/driver、MPI 版本（适用时）、命令、输入/参考 hash 与 passed/failed/skipped。时序/生命周期测试要证明实际发生预期先后关系，不能仅凭偶然结果正确。

| ID / 任务 | 工作负载与正反例 | 通过条件与证据 |
|---|---|---|
| **V1 离线构建与安装** / 02,11 | 默认 Serial Debug/Release；选定 GCC/Clang profile，MSVC/macOS 单独执行；本地 exact Kokkos；新 build 目录；移动 Nexora/Kokkos 安装前缀后的独立 consumer | 网络禁用仍 configure/build/CTest/install/consume/run；consumer 实际创建 View、执行算子并读回；缺包、错版本/ABI/profile、缺架构明确失败；新旧语言标准不混用。旧 exec 包消费示例替换。CUDA 追加 [构建与消费负向矩阵](build-and-validation.md#cuda-build-contract)：wrapper/host/NVCC/架构及包组合匹配，source/build 树不可用时仍能独立消费；旧 `NEXORA_GPU_BACKEND` 任意值拒绝。 |
| **V2 字段、ID 和布局契约** / 03 | rank-1 粒子标量、rank-2 交换、SPH LayoutLeft 向量、FEM 类 rank-3 fixture、LayoutStride subview；空字段、溢出、错单位/association/dtype、旧版本、重叠写 | schema 与 actual View 校验一致；只读无法获写授权；重复/丢失 stable ID、非双射 permutation、容量截断拒绝；重排不改变稳定身份，旧绑定新提交失败。 |
| **V3 执行实例与拷贝** / 04,05 | CPU 与 CUDA 同 source；H2H/H2D/D2H/D2D；两个独立 execution instances；同实例数据链、跨实例交接、显式非默认 CUDA stream fixture | 检查实际结果与先后关系；同实例无逐 stage host wait；跨实例 fence 或获准 bridge 建依赖；producer 未完成不得交 MPI/SDK；host 不直接解引用 device pointer。内存复制只对已验证空间组合开放。 |
| **V4 owner / 多消费者 / 关闭** / 04 | managed 末 handle 提前释放、unmanaged 真 owner 计数；两个读消费者，一快一慢；读写冲突、外部 resize、caller 退出、初始化中断、flight 预留失败、close | 生命周期记录证明最后访问后恰好一次释放；写授权不会由引用计数推断；有任何 reader 时不可复用 bank；无在途工作才能销毁 owner/stream/ScopeGuard；部分失败无漏释放或 UAF。通过受控同步/延迟安排强制覆盖在途状态。 |
| **V5 reference CPU/CUDA 数值** / 05,06 | §3.1 固定公式/输入/精度；小样本与较大样本；同一 template CPU 和 CUDA 实机执行 | §3.1 原门槛全部通过；记录独立 oracle 与收敛表。OpenMP 若启用同验。不能以 CUDA smoke 或编译通过替代。 |
| **V6-SPH 非规则邻域** / 07 | §3.2；邻域遍历+阶段算子+全局归约；CPU/CUDA，重排前后按 stable ID 比较；空邻域、长邻域和边界样本 | CSR/ID 精确检查；按冻结公式与容差数值对照；错误位置/拓扑版本会重建或拒绝；重复 reduction 与并发写路径受控；不得用规则标量循环代替。 |
| **V6-FVM 通量与累积** / 08 | §3.3；cell→face flux→cell accumulation；边界面与非规则 CSR；CPU/CUDA，对照 independent scalar oracle | 每内部面唯一值/相反贡献；全局内部通量抵消、边界预算与单元结果满足既有容差；不更改 pressure/measure convention。单元独占 gather 与经批准累积策略分别验证。 |
| **V7 数值失败、恢复与重放** / 06,07,08 | trial 或 prepare_commit 失败、显式 restore、合法 retry、旧 accepted 输出仍有 reader、历史/控制/拓扑改变、重放与连续运行。SPH 通过实际的物理、几何和资源条件触发失败。其他模块可用故障注入。 | accepted 数组、历史、时间、步骤计数不受失败 trial 污染；尝试数可增加；缓存按 epoch/版本失效；新 trial 独立 storage；restore 失败进入 failed，禁止继续提交。设备致命失败只验证隔离进程退出/不发布，不要求设备恢复成功。 |
| **V8 端到端性能** / 07,08,10 | reference 只作辅助；SPH/FVM 小/大规模、代表性度分布；包含 layout pack/unpack、host/device copy、allocation、同步、归约读回和窗口准备/提交 | 完整完成时间、吞吐、峰值 host/device memory、分配次数/bytes、复制量、转换成本、fence 次数/等待、预热/稳态分别报告；同公式/输入/容差下对比 CPU/CUDA；只有 kernel launch 时间无效。无既有数字时先建立基线，不声称 speedup 或编造百分比门槛。 |
| **V9 capability / CLI** / 09 | 同一 fixture 的 validate/plan/run；未知/重复键、schema、NaN/Inf、非法 step/quantum、缺 device/未编译 space；built 但未 validated | 校验与装配一致、错误退出明确；plan 显式列布局转换、copy 和主机决策点；发布的支持矩阵只含实测组合；未知组合不 fallback。保留 inspect-build/version 的诊断意义，变更旧键需迁移说明。 |
| **V10 架构与清理** / 10,11 | 默认 CPU、CUDA、MPI 选择图；include/target/source/export 正反例；全文与安装面扫描；公开 Kokkos 与私有 native types | checker 区分 schema/data；缺可选节点允许、启用后缺必需节点失败；越层、公共原生 CUDA/HIP/MPI 类型、旧 exec 链接失败；新包无旧别名，Kokkos 依赖真实传递。旧 native CUDA smoke 的通用计算用途删除。 |
| **V11 P2 通信与持久恢复** / P2 | 2/4 rank、空分区、halo/reverse sum、contact history 迁移；host staging；send complete 与 remote applied 区分；中断 checkpoint | 稳定 ID/owner 唯一，物理预算符合相应 baseline；request 完成前缓冲不释放；文件只有 accepted 逻辑数据，schema/checksum 和一致 manifest；IO-001 仅按 [R0 与 CP-01–08](checkpoint-restart.md#restart-acceptance) 验收相同 binary/rank/分区/设备配置，R1–R3 明确延期且拒绝越界恢复。P1 本地 TransportView fixture 不替代此项。 |
| **V12 后续 HIP/外部引擎** / P6/M/I | 同一算法 source 的 AMD 实机；engine 完整历史/约束/控制状态恢复；按实际 workload 独立选择 | 复跑相关 V1–V11，记录真实硬件与依赖；编译成功只记 compiled；不以 CPU 或 CUDA 结果授予 HIP 支持；外部精度/formulation 不满足则拒绝。 |

性能报告必须划清计时边界：冷启动包含初始化/分配/编译或加载（若第三方引擎适用）；稳态从窗口输入准备至 accepted 输出实际可用，所有异步操作完成。MPI 后续还需计入通信与 pack。既报告总耗时也报告可加总/可解释的阶段重叠，不能把重叠的分项简单相加当总时间。采样记录 warmup、重复次数、分布/波动与系统负载，回归阈值须由取得的基线另行批准。

## 5. 后续阶段与集成范围

[迁移路线](migration-roadmap.md#legacy-p0-p7) 对归档 §27 的 P0–P7、[§30 全部任务](migration-roadmap.md#legacy-task-ids)及 [MBD-001–020 / INT-001–030](follow-on-task-plan.md) 作逐项重映射。P2 的文件、target、接口、精确前置和通过条件以 [P2 任务表](migration-roadmap.md#p2-task-ids) 为准；M/I 候选需另行设计和批准，不能把归档当实施规范。本稿的阶段出口为：

| 阶段 | 修订后出口与前置 |
|---|---|
| P0 / P1 | P1 从 Kokkos 起步；BASE-001 的真实 SPH/DEM 原始基线仍开放；scalar reference 不能关闭它。 |
| P2 | 在 P1 typed fields/生命周期后完成真实 SPH/DEM、映射/载荷账本、多参与者窗口、可选 MPI、完整内存 checkpoint 和 R0 同配置持久重启；按数学/通信前置拆开验收。 |
| P3 | FVM 全 solver；复用 P1 face-flux 结构证据，完整离散、线性求解器、压力/连续性和 FVM–DEM 各自验收。 |
| P4 | FEM/CHT/FSI；保留 FE space、quadrature history、mixed formulation 与强耦合恢复，依赖对应方法和 P2 协议。 |
| P5 | LBM populations/时间层/stencil/halo/单位独立建模，保持固定量子与恢复 parity；复用基础设施，不强制 FVM 或粒子布局。 |
| P6 | 新 Kokkos execution space 的实机认证和性能调优（HIP 等）；删除“第二套 backend”和 Kokkos/DSL/provider 实验。 |
| P7 | 大规模、AMR/APR、复杂模型、产品发布和受控外部引擎/服务；每项独立证据，不整包自动放行。 |
| M0–M7 | 保留机械 owner、frames、wrench/impulse、引擎恢复与耦合物理门槛。Nexora 自有机械数组/算子使用 Kokkos；第三方完整 engine 内部运行时不变成通用 provider。 |
| I0–I7 | 保留 accepted 快照、OpenUSD/viewer、可选 engine 与通信边界；取消 I2 Warp 通用算子 provider/JIT/AOT 路线及相关前置，Newton 等按 engine 独立审核。 |

单方法 P3/P4/P5 可在 P1 契约稳定后并行准备；跨方法运行必须等待 P2 耦合及恢复证据。首次两参与者测试不等待所有方法完成。MPI、HDF5、OpenUSD、外部 engine 都是按需引入的依赖，不能替代默认必需 Kokkos 的地位。

<a id="approval"></a>
## 6. 实现所需审批与未决事项

Kokkos 单一路线已定，本表不要求再次确认选型。本次旧 core 容器及 exec 清理已有明确授权；其余尚未授权事项按实际范围形成具体实施批准：

| 项目 | 本稿提出的具体决定 | 尚需维护者批准/提供 |
|---|---|---|
| A1 源码与构建实施包 | 按 P1-01–11 实施；Kokkos 5.2.2 exact commit、C++20、CMake 3.25.2+、Serial 默认；一次切换删除 exec API/导出 | 本次旧容器与 exec 的不兼容删除已获授权；其余范围及仓库外 consumer 的迁移按实际需求确认。无需保留永久旧 facade。 |
| A2 数值基准 | 保留 §3.1 原 reference 公式与容差。SPH 一维小算例已按 §3.2 固定出处、原创输入、独立答案与容差。 | 按 §3.3 冻结 FVM 阶段公式、输入、独立答案和容差。真实 SPH/DEM 参考来源和复用权利由 `BASE-001` 跟踪。 |
| A3 实机资源 | CPU Serial 为默认；另验 OpenMP profile；CUDA 需要 Linux/NVIDIA runner，显式架构与匹配 NVCC/host compiler；HIP 留 P6 | 提供可持续使用的 runner、OS、GPU/架构、driver、NVCC/toolkit、host compiler/stdlib 精确配置，按 [首个 CUDA 契约](build-and-validation.md#cuda-build-contract) 锁定并批准；目前均未知，不能用上游最低版本代替实际配置。CPU 可以先交付，完整 P1 CUDA 验收保持开放至取得证据。 |
| A4 依赖制品与分发 | 独立获取并校验 Kokkos tag/commit/source digest，存入本地 artifact cache；configure 不联网；保留 Apache-2.0 WITH LLVM-exception notices | 确定离线制品保管位置/来源记录；项目自身许可证仍按 ADR-014 单独决策，不阻止本轮设计。 |
| A5 外部 engine 与高级集成 | 本轮只确定边界，不选择 Chrono/Newton 的生产引擎，不建设 Warp provider | 未来立项时批准具体 engine/formulation/恢复级别及复用权利；不作为 P1 通用基础设施实施的前置。 |
| A6 P2 持久重启实施包 | IO-001 仅 R0；按现行兼容、完整性和 CP-01–08 关闭，不扩展恢复等级 | P2 立项时批准实际无损 codec/容器、原子发布与受支持存储系统、同配置 profile 和完整状态清单；R1–R3 需另行设计和批准。不阻塞 P1。 |

以上未决项不影响已确定的架构方向。受限 CPU 切片及本次清理的授权和证据不批准或关闭本表其余工作；不得把局部计算冒烟等同于完整 P1。
