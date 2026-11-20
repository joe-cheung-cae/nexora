# Kokkos 迁移路线与历史计划对照

> **Serial CPU 基础、参考状态生命周期、CPU 字段访问与本地传输已实现；本次明确授权提前删除旧 core 数值容器与 exec 层，完整迁移和 P1 验收仍开放。** 以下盘点以 bootstrap 为历史基线，当前 `data` 公开传递 Kokkos，`data_schema` 与 `participant_api` 无 Kokkos 传递编译依赖。当前接口见 [core 数据结构](core-data-structures.md)，历史运行证据见 [CPU 切片验证](kokkos-cpu-slice-validation.md)和[参考生命周期验证](reference-lifecycle-validation.md)。OpenMP/GPU 未实现；旧 CUDA 源休眠，旧 GPU 选项任意值均拒绝；可选 MPI smoke 未验证。

当前架构入口为 [架构总览](overview.md)，执行、字段和存储契约见 [Kokkos 设计](kokkos-design.md)，首阶段工作和验收见 [Phase 1 设计](phase-1-design-review.md)。下文用 `P1-01`–`P1-11` 表示首期计划任务。远期机械与外部集成的 50 个历史 ID 见[后续任务对照](follow-on-task-plan.md)。共享值类型 `TimeInterval` 已在步骤 2 移至 `core/Time.hpp`，保留原类型与 duration 语义和旧头兼容包含；`TimeConstraint` 及 `TrialReport` 保留 participant 职责，MBD engine 的独立报告仍为后续目标。

<a id="source-inventory"></a>
## 源码现状与类型去向

| 旧类型或契约 | 当前仓库 | 迁移处置与条件 |
|---|---|---|
| `Buffer` | 历史 host 字节数组和借用接口。 | 已移除；数值存储使用 managed Kokkos View，外部借用走可信 owner 边界。 |
| `ExecContext` / `CpuContext` | 历史同步执行接口。 | 已移除；方法直接使用 Kokkos execution space，控制错误用 `Status`/`Result`。 |
| `CompletionToken` / `Event` | 历史同步完成占位类型。 | 已移除；完成和所有权按 Kokkos 同步点及明确 owner/lease 管理。 |
| `AllocationRef` / `BorrowedBuffer` / 旧 allocation lease | 仅见于旧架构或 Phase 1 草案，源码中没有这些分配或租约类型。 | **取消通用分配引用和字节租约**。普通存储使用 managed View 的持有关系。确有外部借用时，在边界保留外部 owner，满足写入完成、消费完成和释放顺序，再构造受限的 unmanaged View。由 `P1-04` 验证受控借用；真实外部库按集成单独验收。 |
| `ExecutionQueue` / `ExecutionEvent`，旧 `BackendKind` / `DeviceId` | 仅为旧设计提案。旧 native CUDA 源中的 stream/event 现已休眠。 [探针](../../apps/nexora/CudaSmoke.cu) | **取消抽象**。方法使用具体 Kokkos execution space；配置保留构建与设备能力语义。同一实例顺序、跨实例 fence、主机可见性和外部通信交接分别按 Kokkos 设计执行，由 `P1-04/05` 验证。 |
| `MemorySpace` | 历史 host/device/unified 枚举。 | 已移除；space、可访问性和布局由 View 与绑定表达。 |
| `FieldView<T>` | 历史 host 借用字段接口。 | 已移除；字段使用 typed View 与 `ReadField`/`WriteField`。 |
| `KernelView<T>` / `FieldKernelView<T>` | 前者为历史裸指针描述，后者仅为草案。 | 前者已删除，后者取消；内核直接使用 typed View/subview。 |
| `FieldBinding` / `TransportView` | `FieldBinding.hpp` 已提供 managed Serial typed 绑定与同步读写 lease。外部 HostSpace 数组和本地 `TransportView` 已复用这些 lease。 | **收敛到领域绑定与边界传输**。`P1-03` 保留必要字段语义及版本验证，`P1-04` 验证拷贝、借用和数据交接。具体跨 rank pack/halo 留给 P2。 |
| `FieldDescriptor` / `FieldRegistry` | 已拆分。描述符验证语义，Registry 按 `FieldId` 保存描述符并拒绝重复 ID。`FieldStamp` 保存动态版本和有效时间，布局及内存位置由 View 表达。 [描述符](../../include/nexora/data/FieldDescriptor.hpp) [注册表](../../include/nexora/data/FieldRegistry.hpp) | **保留语义元数据和注册表职责**。拆开物理语义、版本与实际 View 布局；不从字符串或旧枚举推断真实存储。字段及状态 owner 仍是方法或参与者。`P1-03` 移除 exec 依赖；跨端口物理兼容见 `CPL-001`。 |
| `ArrayView<T>`、`Vec`、`Tensor` | 旧 core 容器已删除。 | 数值使用 Kokkos 原生 View/Array；ID/版本、`Status`/`Result`、checked cast 和时间值保留在 core。详见 [core 数据结构](core-data-structures.md)。 |
| `ParticipantIF`、`PortDescriptor`、`TimeConstraint`、`Simulation` | 已实现同步、单参与者生命周期。调用方持有参与者；Simulation 持有接受时钟，校验端口与时间、调用试算和提交，按已有契约尝试恢复。旧 dummy 测试保留，步骤 2 新增真实 Kokkos reference 的提交/恢复/重试/重放测试。 [接口](../../include/nexora/participant/) [实现](../../src/simulation/Simulation.cpp) [参考测试](../../tests/reference/reference_tests.cpp) | **保留并扩展已验证参考子集**。当前仅同步 Serial accepted/trial 与只读输出；完整 `P1-06` 的历史、缓存和快照预算仍开放。多参与者协调、域分解与持久化重启各自留给后续任务。 |
| typed operator set、`reference*`、provider 选择 | 内部 Reference State/Participant 以 Kokkos Serial 运行 x/q 公式。CLI 装配版本化 Reference case，并提供帮助、版本、构建信息和可选 MPI 诊断。 [方法](../../src/methods/reference/State.cpp) · [case 格式与验证](reference-case.md) | **保留类型化算子契约**。后续实施同源可选设备实例及代表性 SPH/FVM 阶段。 |

<a id="targets"></a>
## 模块、构建目标和检查器

箭头从目标指向直接链接依赖；表格以切片前 bootstrap 为基线，以下列出的切片差异覆盖表中旧图。内部 target 为 `nexora_*`，安装导出的名字为 `nexora::*`。实际直连与 checker 允许的边分别核对。 [顶层 CMake](../../CMakeLists.txt) [检查器](../../tools/check_dependencies.py)

| 当前模块或 target | 当前实际直连及用途 | 迁移去向 |
|---|---|---|
| `build_options`、`core` | `build_options` 无直连，`core → build_options`。两者均为 INTERFACE，负责语言特性、公共与生成头文件。 [core CMake](../../src/core/CMakeLists.txt) | 保留。Kokkos 依赖落在实际消费 View/执行 API 的边界，不加入 core 公共依赖。语言与工具链按选定 Kokkos 版本确定。 |
| `exec_api`、`exec_cpu` | 旧 target、源码、头和安装导出已移除。 | 本次提前完成 P1-11 的旧层清理；最终构建与消费验收仍须按实际执行记录。 |
| `data_schema`、`data` | INTERFACE `data_schema → core`；STATIC `data → data_schema, Kokkos::kokkos`。 [data CMake](../../src/data/CMakeLists.txt) | 已移除 `data → exec_api`。CPU 外部借用、本地 pack 及按稳定 ID 接收解包已实现，设备异步交接仍待实施。 |
| `participant_api`、`simulation` | INTERFACE `participant_api → data_schema`；STATIC `simulation → participant_api`。两者均已导出。 [simulation CMake](../../src/simulation/CMakeLists.txt) | 保留生命周期边界。Kokkos 状态及算子实例放进实际方法/参与者实现。 |
| `methods_reference`、`participants_reference` | 步骤 2 新增内部 STATIC 目标。`methods_reference → data, Kokkos::kokkos`；`participants_reference → participant_api, methods_reference, data`。不导出、不安装。 [方法 CMake](../../src/methods/reference/CMakeLists.txt) [参与者 CMake](../../src/participants/reference/CMakeLists.txt) | 只实现本次 Serial 状态闭环；其他设备、公共安装 API 和完整数据契约分别验收。 |
| `nexora` CLI | `nexora` 直接链接 core、simulation、内部 `participants_reference` 和 Kokkos，以版本化 case 装配 Serial Reference。可选私有 `MPI::MPI_C` 用于独立探针。 [CLI CMake](../../apps/nexora/CMakeLists.txt) · [case 格式与验证](reference-case.md) | 后续设备组合按各自的实机验收与能力清单装配。 |
| 原草案 `exec_cuda`、`reference_contract`、`reference_cpu`、`reference_cuda` | 均无 CMake 声明或实现。 | 取消 backend 专属执行目标。参考参与者与算子签名继续按职责组织，同一算子模板实例化所选 Kokkos execution space；内部目标只随实际使用者增加。 |
| 测试和安装消费 | 旧 exec 测试与消费已删除；保留字段、生命周期、schema、simulation 与 Kokkos 安装消费测试。 | 干净安装与独立消费及现行正反例按实际执行结果记录；不以旧测试证明新路径。 |
| CMake 目标图、依赖检查与 CI | 配置后生成直接边 JSON，checker 检查目标与包含边界。CI 在配置前供应锁定的 Kokkos。 [目标图](../../cmake/NexoraTargetGraph.cmake) [检查器](../../tools/check_dependencies.py) [执行记录](serial-ci-validation.md) | 持续检查旧路径拒绝、Kokkos 消费边、可选目标及安装导出。 |

<a id="cutover"></a>
## Phase 1 顺序与旧层删除条件

1. `P1-01a` 锁定工具链与现状，登记当前 CPU 命令与输出、包消费、依赖图和可选探针覆盖。`P1-01b` 的数学基准只约束对应数值任务。真实 SPH/DEM 的可复现基线由 `BASE-001` 继续追踪。版本、来源、许可与选项由构建策略记录。
2. `P1-02 → P1-03` 打通所选 Kokkos 的 CPU 构建和初步包冒烟，然后迁移 typed 字段、ID 与版本契约。`P1-03 → P1-04` 验证外部 owner、借用、复制和完成交接，`P1-03 → P1-05` 接同一 typed 算子及 CPU/可选 CUDA 实例。
3. `P1-04/05 → P1-06` 接参考参与者、accepted/trial 与恢复。`P1-07/08` 分别用批准的数值基准验证 SPH 非规则邻域与 FVM 面通量切片；两者依赖对应字段、执行和生命周期契约。完整方法和 DEM 仍在后续阶段。`P1-09` 在构建能力与参考参与者确定后完成 CLI 校验和装配。
4. `P1-10` 汇总候选组合的数值、生命周期与端到端性能证据，并预检依赖和包消费。最终清理与安装消费由 `P1-11` 验收。CPU 必须实际运行。可选 GPU、MPI 各以实际组合执行结果标注状态，无设备或未执行的组合保持待验收。可选 MPI 诊断通过只说明其探针路径。
5. 本次已明确授权提前移除旧 core 容器、exec 及其旧导出。`P1-11` 最终验收仍需重新生成目标图并执行干净构建、测试和安装消费检查，且不代替完整 P1 的数值与设备验收。

   - 新 CPU 路径需通过正常 CLI、参考试算、提交、回滚与重放；覆盖零长度、形状、版本和 owner 生命周期。代表性切片使用批准的基准与容差。
   - 源码、测试与安装消费者使用新入口；记录 `Buffer`、旧 exec headers/targets 及旧 core 容器的源与包兼容变更。仓库外消费者须迁移；包配置须解析 Kokkos。
   - 干净安装并检索头、链接、包配置、target graph 与 CI；旧 prefix 不会因覆盖安装自动清理，须替换或清理。复跑禁用能力负例；未获实机证据的 CUDA 不声明运行验收。

CPU 和可选 GPU 均以 Kokkos 作为唯一计算及数值存储基础。具体任务分工、数值检查和设备验收以 [Phase 1 设计](phase-1-design-review.md) 为准。

<a id="legacy-p0-p7"></a>
## 归档 §27 的阶段重映射

| 历史阶段 | 当前范围、依赖与出口 |
|---|---|
| P0 | 由 `P1-01` 接架构、依赖和现状基线，`P1-02` 接实际构建。登记参考算例和可复现报告规范。`BASE-001` 的真实 SPH/DEM 基线独立完成，约束相应方法和外部对照。 |
| P1 | `P1-01`–`P1-11`。CPU Kokkos、typed 字段与算子、参考参与者窗口、SPH/FVM 代表性切片、CLI、测试和旧层切换。CUDA 作为可选 Kokkos GPU 验收，MPI 为可选诊断。 |
| P2 | 首个真实 SPH–DEM 窗口。基于 Kokkos 实现邻域、密度阶段、接触历史和力算子，逐步验收显式耦合、守恒 ledger、回滚与重启；按真实用例增设可选 MPI halo、反向规约和历史迁移。机械增量 M0–M3 按[机械任务前置](follow-on-task-plan.md#mbd-task-ids)另行设计和批准。 |
| P3 | FVM 扩散、输运、不可压流及 FVM–DEM。复用 P1 face-flux 结构切片与 P2 耦合契约，按收敛阶、连续性和体平均源项/冲量验收。代数库如被选中，按独立外部边界认证。 |
| P4 | FEM 热、弹性及混合流，CHT/FSI 强耦合，连接 M4/M6。依赖字段契约、P2 窗口/恢复接口及对应方法；验收 patch/MMS、热流或功预算、重算与唯一惯性。 |
| P5 | LBM 单相 lattice、单位、端口、streaming/forcing 和运动边界。复用 typed Kokkos 数据/算子，按所选通信组合验证 halo、质量、checkpoint 和单位。 |
| P6 | 改为同一源码在选定新增 Kokkos execution space 上的可移植性认证。HIP 等候选必须有依赖/工具链和实机证据，报告数值、回滚、端到端性能。旧第二套自研 backend 与 DSL/provider 实验取消。 |
| P7 | 大规模、AMR/APR、复杂模型及产品化。以真实 workload 分项验收；外部引擎/求解器、I/O、场景与服务按隔离边界独立评估。 |

P3、P4、P5 的单方法可以在 P1 字段契约确定后分别准备设计，实施需另行批准；跨方法耦合依赖 P2 协议及证据。各阶段继续分别提交构建/依赖、数值、耦合/恢复和性能证据。参考 Phase 1 小算例与 P2 真实算例的通过标准分别记录。 [归档 §27](../archive/CFDX_Multimethod_HPC_Architecture_Plan_v1.2_NVIDIA_Integration.md#sec-27)

<a id="legacy-task-ids"></a>
## 归档 §30 的任务逐项重映射

### P0/P1 主干

| 历史任务 | 处置及理由 | 现行任务、前置和验收 |
|---|---|---|
| `ARCH-001` | **迁移**。依赖约束仍必需，白名单改用目标 Kokkos 图。 | `P1-01a` 记录基线与边界，`P1-02` 产出候选图，`P1-11` 验收 checker 的 include/target/export 正反例。 |
| `BASE-001` | **保留**。首期小样本只验证平台结构，真实 SPH/DEM 需要独立数值和性能证据。 | `P1-01b` 和 `P1-10` 只负责首期结构夹具。真实基线落点为拟 `validation/{sph,dem}/`、`benchmarks/{sph,dem}/`，只有测试/基准叶 target，不新增产品实现。前置 `P1-01a` 及参考来源/复用权利批准。交付冻结输入、公式/物理约定/原容差、独立参考、计时边界、来源/输入哈希、可重跑命令、硬件/工具链/config manifest、原始观测与报告；接口是版本化 baseline manifest。通过条件：在记录的配置上可复跑，数值满足既有判据、计时覆盖规定边界，缺基准或缺数值审批保持开放；性能阈值没有既有依据时单独审批。供 `SPH-001`、`DEM-001`、`LEGACY-001`、`CPL-004` 使用。 |
| `BUILD-001` | **迁移**。最小 CPU 构建需显式解析必需 Kokkos。 | `P1-02` 依赖 `P1-01a`，验收离线依赖解析、候选导出与包冒烟，`P1-11` 验收最终独立消费。 |
| `CORE-001` | **保留并迁移后续校验**。已有 core 值类型和 host 契约适用。 | 复用既有 core 测试；`P1-03` 在 `P1-02` 之后验证 ID/版本、checked cast 与共享 `TimeInterval` 的边界，保持 duration 语义。 |
| `EXEC-001` | **取消原生实现**。Kokkos Serial 承担 CPU 存储与执行。 | `P1-03`、`P1-04`、`P1-05` 分别验证 CPU 存储、保活和完成点；旧目标、头与调用者已提前删除，`P1-11` 最终验收仍开放。 |
| `EXEC-002` | **取消原生 executor，迁移验收**。CUDA 使用同一 typed 算子的 Kokkos 实例。 | `P1-04`、`P1-05` 定位私有互操作与设备复制；`P1-10` 记录实机实例、跨实例顺序和释放证据，`P1-11` 迁移原生探针。 |
| `DATA-001` | **迁移**。元数据与真实 typed View 的职责分开。 | `P1-03` 依赖 `P1-02`，校验单位存在、association、形状与版本，保留元数据 registry。 |
| `DATA-002` | **迁移**。采用真实 owner 与完成条件，零拷贝取决于边界兼容性。 | `P1-04` 依赖 `P1-03`，验收受控 owner、镜像/复制、借用和完成交接；设备联调由 `P1-05`、`P1-10` 验收。真实外部旧数组由 `LEGACY-001` 另审。 |
| `PART-001` | **迁移**。已有同步 API 支撑 Kokkos 参考参与者。 | `P1-06` 依赖 `P1-01b` 的 reference 基准及 `P1-03`、`P1-04`、`P1-05`，验证接受、重试与恢复。 |
| `LEGACY-001` | **延期，需另行设计和批准**。实际外部求解器及复用许可未确定。 | `BASE-001`、`P1-04`、`P1-06` 后评估候选 `src/adapters/external_sph/`（私有 `nexora_external_sph_adapter`）。只接完整外部参与者，验收物理约定、恢复与复制成本。 |
| `CASE-001` | **迁移**。装配校验使用所选 Kokkos 组合与同一份 case。 | `P1-09` 依赖 `P1-02`、`P1-06`；版本、未知键、非法时间及缺能力明确拒绝。 |
| `TEST-001` | **合并**。平台证据及最终安装清理分开验收。 | `P1-10` 验证实际 CPU 和可选设备上的候选行为，`P1-11` 完成独立包消费和最终依赖检查；编译、跳过、执行分开记录。 |

<a id="p2-task-ids"></a>
### P2 真实数值与耦合

下表是已采纳的 P2 任务范围和拟改动落点。P2 的具体实现包仍须独立评审批准。全表共同验收前置是 `P1-11` 的 CPU 切换，表中另列任务依赖。申报 CUDA 路径还须取得相应 P1 CUDA 证据；`MPI-001`–`MPI-003` 只约束启用 MPI 的交付。P1 的任务与验收 ID 以[首期计划](phase-1-design-review.md#acceptance)为准。

| 任务与处置 | 拟修改/新增文件与 target | 契约与精确前置 | 通过条件 |
|---|---|---|---|
| `SPH-001` **保留并迁移**。真实 density 阶段超出 P1 结构夹具。 | `src/methods/sph/{State,Density,Instantiations}.*`、`src/participants/sph/`、`tests/methods/sph/`、`validation/sph/`、`benchmarks/sph/`；`nexora_methods_sph`、`nexora_participants_sph`。 | 三维动态密度默认直接遍历 `SPH-002` 的格区间，读取 typed 粒子字段，按批准的公式、精度和边界写 owned trial density。前置 `BASE-001`、`P1-05`、`P1-06`、`P1-07`、`SPH-002`。 | 独立基准匹配密度和原物理约定；零/极端邻域与重排后的稳定 ID 对照通过。已启用 CPU/CUDA 分别实跑并记录含准备/归约的耗时。 |
| `SPH-002` **保留**。三维动态 SPH 默认直接遍历 [`cellBegin` 格索引](#sph-cell-begin)。 | `src/methods/sph/neighbors/`、`tests/methods/sph/`；`nexora_methods_sph`。 | 方法持有按格重排的粒子字段、稳定 ID 与槽位双射，以及按拓扑、几何和布局版本校验的格索引。前置 `P1-03`、`P1-07`。 | 同域字段和历史随粒子重排，陈旧绑定拒绝或更新。按稳定 ID 对照独立全扫描与显式 CSR，覆盖空格、极端邻域、支持边界、ghost 和跨格移动。 |
| `DEM-001` **保留并迁移**。稳定 contact key 和 history 是恢复输入。 | `src/methods/dem/contacts/`、`src/participants/dem/`、`tests/methods/dem/`、`validation/dem/`；`nexora_methods_dem`、`nexora_participants_dem`。 | 以稳定实体对及已批准的接触标识绑定唯一 owner，typed View 保存 accepted/trial history；交付首例所需的受限接触/积分阶段。前置 `BASE-001`、`P1-03`、`P1-05`、`P1-06`。 | 接触/脱离、重排、失败回滚和重放保留正确历史，拒绝重复接触。选定力学小例对照已批准 DEM 基线；跨 rank 迁移另验 `MPI-003`。 |
| `CPL-001` **保留**。端口物理兼容须在读写前确定。 | `src/coupling/ports/`、`tests/coupling/ports/`；`nexora_coupling`。 | 以 `PortDescriptor`、`FieldDescriptor` 和快照 stamp 检查方向、单位、frame、association、measure 与有效时间。前置 `P1-03`、`P1-04`、`P1-06`。 | 兼容端口可签发所需交换；错误语义在提交或修改目标状态之前明确拒绝。 |
| `CPL-002` **保留并迁移**。映射关系与计算基础无关，实现改用 typed Kokkos 数据。 | `src/coupling/mapping/`、`tests/coupling/mapping/`；`nexora_coupling`。 | 按下述[现行映射定义](#p2-mapping-contract)绑定 `H` 与共轭 `Hᵀ` 的源/目标稳定 ID、权重和版本键；空间映射与载荷方向保持已批准物理约定。前置 `CPL-001`、`P1-05`。 | 常量、线性、合力/矩及离散功检查分别与独立 oracle 对照；陈旧映射或失配拒绝。 |
| `CPL-003` **保留**。载荷来源、方向和时间积分需要独立审计。 | `src/coupling/{loads,temporal}/`、`tests/coupling/loads/`；`nexora_coupling`。 | `LoadLedger`/`ImpulseAccumulator` 以来源、端口、window/attempt 与区间记账，保持力/力矩/冲量语义。前置 `CPL-002`。 | 非等长子步冲量与反力预算吻合，重复载荷、错误区间和旧 attempt 在提交前被拒绝。 |
| `TIME-001` **迁移**。复用同步单参与者契约，新增多参与者一致窗口。 | `src/simulation/`、`src/coupling/`、`tests/simulation/`；`nexora::simulation`、`nexora_coupling`。 | 由窗口协调者统一 `begin → trial → prepare → publish/restore`，所有参与者准备成功后才发布时钟及 accepted 状态。前置 `P1-06`、`CPL-003`。 | 任一参与者试算/准备失败均恢复一致时刻、历史与账本，重试使用新 attempt；部分发布故障进入不可继续状态。分布式失败决策另验 `MPI-001`。 |
| `CKPT-001` **保留并迁移**。重放依赖参与者拥有的完整状态。 | `src/participants/{sph,dem}/`、`tests/simulation/`、`tests/methods/{sph,dem}/`；`nexora_participants_sph`、`nexora_participants_dem`、`nexora::simulation`。 | 内存快照捕获 accepted 主状态、历史、控制、随机/ID 生成状态及版本；缓存按明确键重建。前置 `TIME-001`、`SPH-001`、`DEM-001`。 | 同输入连续运行与恢复重放满足已批准等价准则；恢复前终止相关访问，拒绝不完整快照或无法证明恢复的参与者，并准备 [CP-01 完整捕获清单](checkpoint-restart.md#restart-acceptance)。 |
| `MPI-001` **迁移**。真实通信需要领域传输完成契约，旧诊断只证明 MPI C API 接通。 | `src/parallel/`、`src/parallel/mpi/`、`cmake/NexoraMPI.cmake`、`tests/parallel/`；`nexora_parallel_api`、可选 `nexora_parallel_mpi`。 | typed 快照打包、host staging、请求保活与明确送达/应用完成；MPI 类型私有，单进程路径直接运行。前置 `P1-02`、`P1-04`、`P1-05`。 | MPI off/on 均构建，实际两 rank 验证零负载、请求状态、发送缓冲寿命及一致失败决策。探针和设备直传各自单列。 |
| `MPI-002` **保留**。halo/reverse sum 是可选分区路径的物理边界。 | `src/parallel/halo/`、`src/methods/sph/neighbors/`、`tests/parallel/halo/`；`nexora_parallel_api`、`nexora_parallel_mpi`、`nexora_methods_sph`。 | 按稳定 ID 定义 ghost owner、halo 路线、版本和反向累积，按实例完成 pack/unpack。前置 `MPI-001`、`SPH-002`。 | 实际 2/4 rank 覆盖空分区、ghost、重复/丢失 ID，反向累积与单进程参考一致，失败交换不污染 trial。GPU-aware 设备直传需另验。 |
| `MPI-003` **保留**。跨 rank 迁移需要保持历史和接触所有权。 | `src/parallel/migration/`、`src/participants/dem/`、`tests/parallel/migration/`；`nexora_parallel_api`、`nexora_parallel_mpi`、`nexora_participants_dem`。 | 粒子与接触历史按稳定 ID 迁移，成功完成后才换 owner。前置 `DEM-001`、`MPI-002`。 | 分别在固定 2 或 4 rank 的运行中迁移所有权，于获准安全点发布，覆盖跨界接触、空 rank 和失败路径；无重复接触、丢历史或提前发布，复算满足参考准则。R0 loader 不执行重分区：若快照在在线迁移后生成，只恢复快照当时的分区/owner；改变 rank 或分区的重启属于延期 R1。 |
| `IO-001` **保留并迁移**。持久重启与内存快照、输出场景是独立边界。 | `src/io/checkpoint/`、`tests/io/checkpoint/`、`validation/restart/`；`nexora_io_checkpoint`。HDF5 为独立可选依赖，格式选型随实施包审批。 | 只序列化 accepted 逻辑 schema、稳定 ID、状态/历史与 manifest。前置 `CKPT-001`；启用 MPI 重启时还须 `MPI-001`、`MPI-002`、`MPI-003`。 | 按 [R0 定义](checkpoint-restart.md#restart-levels)和 [CP-02–08](checkpoint-restart.md#restart-acceptance)验收；比较恢复与不中断运行，验证完整性及中断写入拒绝，记录运行组合和恢复级别。 |
| `CPL-004` **保留并迁移**。真实 SPH–DEM 窗口验证整条平台路径。 | `apps/nexora/`、`src/participants/{sph,dem}/`、`cases/verification/coupling/sph_dem/`、`tests/coupling/sph_dem/`、`validation/coupling/`、`benchmarks/coupling/`；`nexora`、`nexora_coupling`、`nexora_participants_sph`、`nexora_participants_dem`。 | 真实参与者声明稳定 ID 的运动/载荷端口，按批准基准接首例所需的力和时间推进阶段，通过载荷账本交换并发布 accepted 快照。串行前置 `P1-09`、`BASE-001`、`SPH-001`、`SPH-002`、`DEM-001`、`CPL-001`、`CPL-002`、`CPL-003`、`TIME-001`、`CKPT-001`、`IO-001`；申报 MPI 组合追加 `MPI-001`、`MPI-002`、`MPI-003`。 | 正常 case 校验→typed 计算→守恒交换→accepted 提交→持久重启/重放，全链记录数值、失败恢复与端到端性能。MPI 组合另报实测跨 rank 一致性与通信开销；串行交付不冒充 MPI 支持。 |

<a id="sph-cell-begin"></a>
#### SPH-002 三维动态邻域、密度与 WCSPH 压力

**已实现的 Serial 窗口。** `sph::Participant3D` 持有 accepted 和 trial 的格索引、字段、密度及版本。窗口试算按稳定 ID 检查本次输入，格归属、成员与 ownership 相同就复用 `cellBegin`，任一变化即重建。三维 M₄ 核按试算坐标判距与累加。提交发布整份候选快照，恢复保留原 accepted 状态。提交、恢复及重试均按 owned ID 对照独立显式 CSR。[源码与验证记录](../../validation/sph/cell-begin-serial.md#simulation-window)

传入 `WcsphParameters{reference_density, sound_speed, gamma}` 可启用 Tait 表压及对称压力加速度。外供终点模式中，每个参与受力的 ghost 从完整本地支撑域重算同阶段密度。全 owned、单进程 Serial 的时间积分模式另接收初始速度、CFL 安全系数和固定的算例平滑长度 `h`（默认 1 m）。格宽和核支持半径均为 `2h`，核值和核梯度分别按 `h⁻³`、`h⁻⁴` 缩放，声学 CFL 随 `h` 缩放。每个窗口从 accepted 加速度出发，半步更新速度并移动粒子，在更新后的格表用预测速度再求一次密度变化率。两端变化率校正流体密度后重算压力力，再完成速度更新和下一步 CFL 估计。[压力公式与 ghost 输入约定](../../validation/sph/wcsph-pressure-serial.md) · [Velocity Verlet 与 CFL 验证](../../validation/sph/wcsph-verlet-serial.md)

积分模式还可接收恒定重力与单独的固定壁粒子集。壁源随流体进入同一邻域索引，每次力评估从当前流体外推壁压，再反解 Tait 壁密度。积分和字段发布只遍历流体行。静水箱壁层统一取 ceil(2h/壁粒距)，完整包覆实体壁与底角，顶部对大气开放。现行推进需要显式物理密度，几何初态的核求和另作诊断。三档算例均保持物理质量，用推进后的压力、速度和穿墙检验静水平衡。[修复与实测](../../validation/sph/hydrostatic-continuity-repair.md) 原来的两种质量分布与轨迹对照属于历史实验。[现行初态验收](../../validation/sph/hydrostatic-initial-resolution.md) · [历史 Adami 固定壁实验](../../validation/sph/adami-hydrostatic-serial.md)

1. **按格组织数据。** 用 Kokkos 按 `(cell, 稳定粒子 ID)` 排序，将位置各分量和质量等热字段按同一置换写成 SoA，并保留 owned 目标到粒子槽位的映射。同域字段、归属标记及积分历史须按稳定 ID 对齐。`cellBegin` 含格数加一项，末尾哨兵等于源粒子数。格 `c` 的连续槽位为 `[cellBegin[c], cellBegin[c+1])`，空格两端相等，下一格的起点兼作本格的 `cellEnd`。
2. **在密度核内直接遍历。** 固定支持半径 `R`、格宽为 `R`、x 为最快变化的格编号时，三维内部目标周围最多 27 格。沿每个合法的 `(y,z)` 合并相邻 x 格，非周期边界先裁剪，每行最多扫描 9 个连续区间。对当前阶段坐标现场检查支持域并累加，只写 owned 目标的试算密度。ghost 可作只读源。以粒子身份识别自身，自身项只加一次，同位置的不同 ID 仍分别参与筛选。
3. **跨阶段检查有效性。** 压力与相互作用阶段可复用仍有效的格索引，并读取该阶段的坐标和物理字段。积分更新位置后检查格归属，粒子仍在原格时可复用区间并重新判距，越格时更新置换、相关字段和格边界。成员、ghost 集合、布局或网格参数变化时须重新校验绑定。试算重排与索引更新受窗口发布和回滚约束，恢复后按 accepted 版本复用或重建索引。

显式 CSR 保留作回归对照，也是固定几何下高复用压力计算的候选优化。[P1-07 一维冻结算例](../../validation/baselines/sph-density-1d-v1.md)继续按原行列顺序、闭支持域（含零权重边）、自身单计和原容差核对固定 CSR、逐 ID 密度及 owned 和。

**CPU 选型依据。** 独立的 Kokkos 5.2.2 实验于 `2026-11-19T15:42:18+08:00` 至 `2026-11-19T15:47:42+08:00` 在 Apple M2 Pro 上运行。下表取 OpenMP 8 线程、FP64、固定几何的三维密度核，含 262,144 个粒子、229,376 个 owned 目标及 3,895,709 条非自身有向边。三种方法的六种启动顺序各重复三次。累计耗时从格索引与字段重排开始，包含可选建表，至最后一遍密度核完成。各列为 18 轮中位数，峰值 RSS 按单方法独立进程测量。

| 方法 | 1 遍（ms） | 2 遍（ms） | 3 遍（ms） | 8 遍（ms） | 峰值 RSS（MiB） |
|---|---:|---:|---:|---:|---:|
| `cellBegin` 直接遍历 | 25.88 | 37.72 | 49.48 | 108.80 | 39.13 |
| 显式 CSR | 37.89 | 38.90 | 39.92 | 45.32 | 56.72 |
| 建 CSR 时临时缓存每格区间 | 29.87 | 30.94 | 31.97 | 37.26 | 56.95 |

三种方法的全部 owned 密度逐位一致。单遍直接遍历的累计耗时和峰值内存均最低。普通 CSR 从第 3 遍、临时缓存区间的 CSR 从第 2 遍起在累计耗时中位数上优于直接遍历。独立的固定两遍配对复测中，缓存版在 17/17 组更快，普通 CSR 在 2/17 组更快。Serial 和 OpenMP 4 线程也观察到相同的中位数交叉点。这些遍数仅作该密度负载的实验依据，不设自动切换阈值。本地原始记录位于 `results/sph-density-cell-csr-20261119T073659Z/`。

**后续验收。** OpenMP 按 owned 行比较连续 SoA 读取的线程扩展、带宽或缓存计数，并计入排序、字段搬移、越格更新和归约。CUDA、HIP 分别在 NVIDIA、AMD 实机验证 Kokkos 执行空间，比较按行与组内协作遍历时的合并访问、实际内存流量、同步和峰值空间。各配置均按稳定 ID 核对邻居与密度，并记录端到端耗时。实际压力核须在对应设备上按几何有效期和实测复用次数复验 CSR 的收益；上述独立 CPU 实验只作为规划证据。

<a id="p2-mapping-contract"></a>
#### CPL-002 已采纳的映射定义

以下仅迁入原有物理约定，没有选择新映射算法、权重、默认值或容差。`H` 将同一接口上离散 A 的速度映射到 B：`u_B = H u_A`；`F_A`、`F_B` 为分别作用于两侧的、已积分的功共轭载荷，反力传递为 `F_A = −Hᵀ F_B`。相同评价时刻、参考系和速度约束下，`u_Aᵀ F_A + u_Bᵀ F_B = 0`，只说明接口功率配对，不自动证明不同时间积分器下的全步能量守恒，也不消除分区迭代误差。

traction/体积力密度必须先按已批准的面积/体积/积分测度转成集成载荷，不能对密度量直接套普通和。一般 FEM 投影不能假定为对角测度除法。常量/线性再现、合力、关于同一参考点的合矩及功率分别验收；权重和为 1 不代表力矩或能量自动守恒。映射权重、受力位置、覆盖范围、符号与时序须随首例物理基准冻结，仍有缺项时 `CPL-002` 不关闭，不能从归档补入未经采纳的算法。

§30.3 的真实链路由 `CPL-004` 承接。§30.4 的机械链路按[现行 MBD 任务对照](follow-on-task-plan.md#mbd-task-ids)分项审查。外部引擎独立评估，不阻塞 Nexora 自有 Kokkos 机械参考与耦合。 [归档 §30](../archive/CFDX_Multimethod_HPC_Architecture_Plan_v1.2_NVIDIA_Integration.md#sec-30)

<a id="legacy-mbd"></a>
## 归档 M0–M7 的范围变化

各阶段为候选范围，需另行设计和批准。[MBD-001–020 逐项对照](follow-on-task-plan.md#mbd-task-ids)记录现行前置与退出条件。

| 阶段 | 当前范围 |
|---|---|
| M0 | 保留 frame、ID、运动/惯性/接触 owner 与负例校验。P1 审查设计边界，真实机械配置按 MBD/CPL 依赖交付。 |
| M1 | 保留受限机械参考和参与者契约。Nexora 自有计算和数值状态使用 Kokkos；Chrono/Newton 等候选仅以外部引擎边界评估自己的模型、状态和恢复。 |
| M2 | SPH–MBD 的运动批量变换和分段力/矩规约由同一 typed Kokkos 算子承接。依次评审 `MBD-008`、`MBD-009`、`MBD-010`，取消 CPU/CUDA 两套实现与 provider 选择。 |
| M3 | `MBD-011` 保留 DEM–MBD 接触 owner、历史、反力与力矩检查。可选机械 MPI 规约由 `MBD-015` 单独验收。 |
| M4 | 保留 SPH–DEM–MBD、轨迹迭代、机械子步和完整回滚。P2 多参与者事务与 M2/M3 耦合各自通过后进入强耦合验证。 |
| M5 | 保留独立的 FVM–MBD 和 LBM–MBD，分别依赖 P3/P5，沿用 GCL/压力与量子/单位/恢复门槛。 |
| M6 | 保留 FEM–MBD 全阶及独立 ROM。依赖 P4 及机械 ownership/time 契约，分别验证虚功、惯性归属、响应和收敛。 |
| M7 | 按 workload 推进 Kokkos GPU 批量算子、大规模约束和可选联合接触算法。以同 formulation 的参考对照、硬件执行和端到端收益放行，取消机械 DSL/provider 路线。 |

[归档 §36.4–36.5](../archive/CFDX_Multimethod_HPC_Architecture_Plan_v1.2_NVIDIA_Integration.md#sec-36)

<a id="legacy-integrations"></a>
## 归档 I0–I7 的范围变化

各阶段为候选范围，需另行设计和批准。[INT-001–030 逐项对照](follow-on-task-plan.md#integration-task-ids)记录现行前置与退出条件。

| 阶段 | 当前范围 |
|---|---|
| I0 | 保留外部边界的 ID、单位、控制权与 API 隔离。按使用场景评审 `INT-001`、`INT-002`、`INT-003`；机械端口追加 `MBD-001`。 |
| I1 | 可选离线 OpenUSD/只读 viewer 仍可基于已接受快照交付，依赖 I0 和一个可运行案例。 |
| I2 | **取消内部 Warp 单算子、Python/JIT/AOT provider 和 runtime bridge 任务**（`INT-005`–`INT-007`）。Warp 仅可作为经批准的完整外部动力学引擎的内部依赖；应用、服务或进程包装均不得借此提供通用算子 provider 或第二套计算/存储基础设施。 |
| I3 | 可选 Newton 独立机构按 I0、M1 的参与者与引擎边界验收。能力描述限具体引擎和已实测组合，接受/恢复包含引擎自己的隐状态。 |
| I4 | SPH–Newton（`INT-011`）和 DEM–Newton（`INT-012`）分别依赖 `INT-010` 及本方法的机械闭环。取消旧 I2 前置；强耦合还需按 `MBD-013` 验证，三方追加 `MBD-014`。 |
| I5 | 可选在线控制/状态/快照继续依赖 I0/I1 及 accepted publisher。按幂等、权限、背压和单一权威时钟验收。 |
| I6 | SHM、CUDA IPC、Fabric 或多 rank gateway 分别由 `INT-020`、`INT-021`、`INT-022`、`INT-023` 评估，采用所选链路的测量与前置。跨边界复制或借用须有明确 owner、完成条件和失败路径。 |
| I7 | 可按实际组合做离线发行、多部署形态、外部渲染或远端参与者认证。移除 Warp provider/AOT 交付和通用 provider profile。 |

历史 `INT-003/004` 的借用和 ABI 边界若有实际使用者，由 `P1-04` 提供底层契约后单独实现；`INT-029/030` 只覆盖实际启用和认证的集成。 [归档 §46.3–46.7](../archive/CFDX_Multimethod_HPC_Architecture_Plan_v1.2_NVIDIA_Integration.md#sec-46)
