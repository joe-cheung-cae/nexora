# Kokkos 迁移路线与历史计划对照

> **目标路线已定，代码迁移待实施。** 源码盘点于 `2026-11-06T14:52:47+08:00`，任务对照复核于 `2026-11-06T15:46:36+08:00`。当前实现仍为原有 CPU 骨架及可选 CUDA/MPI CLI 探针。本次文档修订未改动源码、构建脚本或测试，也未取得 Kokkos 执行证据。

当前架构入口为 [架构总览](overview.md)，执行、字段和存储契约见 [Kokkos 设计](kokkos-design.md)，首阶段工作和验收见 [Phase 1 设计](phase-1-design-review.md)。下文用 `P1-01`–`P1-11` 表示首期计划任务。远期机械与外部集成的 50 个历史 ID 见[后续任务对照](follow-on-task-plan.md)。共享值类型 `TimeInterval` 计划在 P1-03 移至 `core/Time.hpp`，保留原类型与 duration 语义；`TimeConstraint` 及 `TrialReport` 保留 participant 职责，MBD engine 使用独立报告，由 participant 转换。

<a id="source-inventory"></a>
## 源码现状与类型去向

| 旧类型或契约 | 当前仓库 | 迁移处置与条件 |
|---|---|---|
| `Buffer` | 已实现。独占、可移动的 host 字节数组（`std::unique_ptr<std::byte[]>`），只从左值借出 `ArrayView`。exec 单测及安装消费示例使用。 [源码](../../include/nexora/exec/Buffer.hpp) | **移除**。参与者和算子的数值存储直接使用类型化 managed `Kokkos::View`，不新增字节分配器。`P1-03/04/06` 覆盖所有权和恢复，测试与安装消费者迁移后由 `P1-11` 删除。 |
| `ExecContext` / `CpuContext` | 已实现。`CpuContext` 是别名。调用方线程先逐个检查依赖状态，再同步调用函数。唯一编译单元 `CpuExecutor.cpp` 实现依赖检查。 [头文件](../../include/nexora/exec/ExecContext.hpp) [实现](../../src/exec/cpu/CpuExecutor.cpp) | **移除**。方法以具体 Kokkos execution-space 实例提交类型化算子；前置失败由调用侧 `Status`/`Result` 短路，完成顺序按设计规定的实例与 fence 建立。`P1-05/06` 验证，`P1-11` 切换调用者。 |
| `CompletionToken` / `Event` | 已实现。`Event` 是别名，`wait()` 只返回已经保存的 `Status`。类型不记录设备事件、异步任务或保活引用。 [源码](../../include/nexora/exec/CompletionToken.hpp) | **移除**。同步公开边界继续返回 `Status`/`Result`；异步的完成与所有权分别由 Kokkos 同步点和明确的 owner/lease 管理。`P1-04/05/06` 验证，`P1-11` 删除。 |
| `AllocationRef` / `BorrowedBuffer` / 旧 allocation lease | 仅见于旧架构或 Phase 1 草案，源码中没有这些分配或租约类型。 | **取消通用分配引用和字节租约**。普通存储使用 managed View 的持有关系。确有外部借用时，在边界保留外部 owner，满足写入完成、消费完成和释放顺序，再构造受限的 unmanaged View。由 `P1-04` 验证受控借用；真实外部库按集成单独验收。 |
| `ExecutionQueue` / `ExecutionEvent`，旧 `BackendKind` / `DeviceId` | 仅为旧设计提案。可选 CLI 探针中的私有 CUDA stream/event 属于独立诊断。 [探针](../../apps/nexora/CudaSmoke.cu) | **取消抽象**。方法使用具体 Kokkos execution space；配置保留构建与设备能力语义。同一实例顺序、跨实例 fence、主机可见性和外部通信交接分别按 Kokkos 设计执行，由 `P1-04/05` 验证。 |
| `MemorySpace` | 已实现的 `host/device/unified` 枚举。`FieldDescriptor` 默认记录 `host`。枚举不控制分配或可访问性。 [源码](../../include/nexora/exec/MemorySpace.hpp) | **迁移并移除旧枚举**。实际 memory space、访问限制、layout、extent 和 stride 由 View/具体绑定决定。端口保留独立的领域元数据。完成描述符和消费面迁移后移除 exec include，见 `P1-03/11`。 |
| `FieldView<T>` | 已实现的连续借用 host view，持有 `ArrayView<T>`、分量数和 `LayoutVersion`，校验形状；当前只有 data 单测使用。 [源码](../../include/nexora/data/FieldView.hpp) [测试](../../tests/data/data_tests.cpp) | **迁移用途**。求解字段使用带领域语义的 typed View 绑定；显式 host 检查或导出使用可访问的 host View/镜像。用形状和版本回归覆盖旧测试意图后，在 `P1-11` 清理旧头。 |
| `KernelView<T>` / `FieldKernelView<T>` | `KernelView<T>` 已在 `FieldView.hpp` 定义，仅含指针、大小和步幅。未见调用点。`FieldKernelView<T>` 仅为旧草案提案。 [源码](../../include/nexora/data/FieldView.hpp) | **删除前者，取消后者**。内核直接捕获类型化的 View 或 subview，不从通用指针描述重造 Kokkos 视图。由 `P1-03/05/11` 完成。 |
| `FieldBinding` / `TransportView` | 旧草案提案，现有头文件没有对应定义。 | **收敛到领域绑定与边界传输**。`P1-03` 保留必要字段语义及版本验证，`P1-04` 验证拷贝、借用和数据交接。具体跨 rank pack/halo 留给 P2。 |
| `FieldDescriptor` / `FieldRegistry` | 已实现。描述符验证名称、单位非空、分量数、有效时间区间及 association；另有 layout、stride、设备和各类版本字段。Registry 验证后按 `FieldId` 保存描述符，拒绝重复 ID。 [描述符](../../include/nexora/data/FieldDescriptor.hpp) [注册表](../../include/nexora/data/FieldRegistry.hpp) | **保留语义元数据和注册表职责**。拆开物理语义、版本与实际 View 布局；不从字符串或旧枚举推断真实存储。字段及状态 owner 仍是方法或参与者。`P1-03` 移除 exec 依赖；跨端口物理兼容见 `CPL-001`。 |
| `ArrayView<T>`，ID/版本、`Status`/`Result`、checked cast、`Vec`/`Tensor` | 已实现的 core 基础。`ArrayView` 是不持有 owner 的 host 范围。 [源码目录](../../include/nexora/core/) | **保留 host/API 职责**。数值数组与内核参数迁入 Kokkos。单独审查设备可调用类型，保持 core 对后端无依赖。 |
| `ParticipantIF`、`PortDescriptor`、`TimeConstraint`、`Simulation` | 已实现同步、单参与者生命周期。调用方持有参与者；参与者实现 accepted checkpoint 和恢复；Simulation 持有接受时钟，校验端口与时间、调用试算和提交，按已有契约尝试恢复。现有测试采用 dummy 参与者。 [接口](../../include/nexora/participant/) [实现](../../src/simulation/Simulation.cpp) [测试](../../tests/simulation/simulation_tests.cpp) | **保留并接真实参考状态**。`P1-06` 验证以 Kokkos 存储区分 accepted/trial、恢复与已接受输出。多参与者协调、域分解与持久化重启各自留给后续任务。 |
| typed operator set、`reference*`、provider 选择 | 当前无 reference 方法、求解方法、provider 注册表或计算 DSL。CLI 只处理帮助、版本、构建信息和 CUDA/MPI 诊断。 [入口](../../apps/nexora/main.cpp) | **保留类型化算子契约，取消 provider 设施**。`P1-05/06/07/08/09` 接同一算子源码、参考参与者、代表性 SPH/FVM 阶段与 CLI 装配。CPU 与可选 GPU 是该算子的 Kokkos 实例。 |

<a id="targets"></a>
## 模块、构建目标和检查器

箭头从目标指向当前实际的直接链接依赖。内部 target 为 `nexora_*`，安装导出的名字为 `nexora::*`。实际直连与 checker 允许的边分别核对。 [顶层 CMake](../../CMakeLists.txt) [检查器](../../tools/check_dependencies.py)

| 当前模块或 target | 当前实际直连及用途 | 迁移去向 |
|---|---|---|
| `build_options`、`core` | `build_options` 无直连，`core → build_options`。两者均为 INTERFACE，负责语言特性、公共与生成头文件。 [core CMake](../../src/core/CMakeLists.txt) | 保留。Kokkos 依赖落在实际消费 View/执行 API 的边界，不加入 core 公共依赖。语言与工具链按选定 Kokkos 版本确定。 |
| `exec_api`、`exec_cpu` | INTERFACE `exec_api → core`；STATIC `exec_cpu → exec_api`。两者均已导出，`exec_cpu` 只编译 `CpuExecutor.cpp`。 [exec CMake](../../src/exec/CMakeLists.txt) | `P1-11` 一并删除源码、头、目标与旧包导出。目标图中不保留旧 exec 别名或第二条 CPU 执行路径。 |
| `data` | INTERFACE `data → core, exec_api`；现有字段和注册表均为头文件实现。 [data CMake](../../src/data/CMakeLists.txt) | 拆去 `exec_api` 边。新增 `data_schema → core` 承载纯元数据；`data → data_schema, Kokkos::kokkos` 承载 typed 字段绑定，并声明公开传递依赖。 |
| `participant_api`、`simulation` | INTERFACE `participant_api → data`；STATIC `simulation → participant_api`。两者均已导出。 [simulation CMake](../../src/simulation/CMakeLists.txt) | 保留生命周期边界，`participant_api` 依赖改为 `data_schema` 与实际使用的 core，`simulation` 保留指向 `participant_api` 的直接边。Kokkos 状态及算子实例放进实际方法/参与者实现。 |
| `nexora` CLI | `nexora → core`，可选私有 `CUDA::cudart`、`MPI::MPI_C` 只供原探针。当前开关为 `NEXORA_GPU_BACKEND=NONE/CUDA` 与 `NEXORA_ENABLE_MPI`。MPI C API 从 C++ 链接的能力在配置期检查。 [CLI CMake](../../apps/nexora/CMakeLists.txt) [MPI 检查](../../cmake/NexoraMPI.cmake) | 保留 CLI 装配入口及诊断的可验证语义。CUDA 探针迁成 Kokkos 设备诊断，移除旧 `CudaSmoke.cu` 的 native kernel、分配、stream/event 和直接 runtime 链接。可选 MPI 探针仍是独立诊断，真实通信由后续专项验收。 |
| 原草案 `exec_cuda`、`reference_contract`、`reference_cpu`、`reference_cuda` | 均无 CMake 声明或实现。 | 取消 backend 专属执行目标。参考参与者与算子签名继续按职责组织，同一算子模板实例化所选 Kokkos execution space；内部目标只随实际使用者增加。 |
| 测试和安装消费 | `nexora_exec_tests → exec_cpu`，`nexora_data_tests → data`。`nexora_install_consumer` 另起项目，直连已安装的 `nexora::simulation` 和 `nexora::exec_cpu`，实际调用 `Buffer`/`CpuContext`。 [exec 测试](../../tests/exec/CMakeLists.txt) [安装消费者](../../tests/package/consumer/) | 原测试验证意图迁到 Kokkos 执行、字段与生命周期测试；独立消费项目走新导出目标，实际执行 Kokkos 路径。更新包依赖解析并做干净安装与独立消费验证。 |
| CMake 目标图、依赖检查与 CI | 配置后生成直接边 JSON。checker 维护目标/包含白名单，要求旧 exec 目标存在，检查越层 include；当前按整个 CLI 目录放行 CUDA/MPI 头。已有 CPU CI 及两 rank MPI smoke CI；CUDA CTest 可在无设备时跳过。 [目标图](../../cmake/NexoraTargetGraph.cmake) [检查器](../../tools/check_dependencies.py) [CI](../../.github/workflows/ci.yml) | `P1-02/11` 同步更新目标图、安装导出、配置矩阵及 checker。检查 Kokkos 实际消费边、可选目标和平台源文件归属。保留 MPI/CUDA 平台头隔离；编译或跳过分别记录，实机执行单独验收。 |

<a id="cutover"></a>
## Phase 1 顺序与旧层删除条件

1. `P1-01a` 锁定工具链与现状，登记当前 CPU 命令与输出、包消费、依赖图和可选探针覆盖。`P1-01b` 的数学基准只约束对应数值任务。真实 SPH/DEM 的可复现基线由 `BASE-001` 继续追踪。版本、来源、许可与选项由构建策略记录。
2. `P1-02 → P1-03` 打通所选 Kokkos 的 CPU 构建和初步包冒烟，然后迁移 typed 字段、ID 与版本契约。`P1-03 → P1-04` 验证外部 owner、借用、复制和完成交接，`P1-03 → P1-05` 接同一 typed 算子及 CPU/可选 CUDA 实例。
3. `P1-04/05 → P1-06` 接参考参与者、accepted/trial 与恢复。`P1-07/08` 分别用批准的数值基准验证 SPH 非规则邻域与 FVM 面通量切片；两者依赖对应字段、执行和生命周期契约。完整方法和 DEM 仍在后续阶段。`P1-09` 在构建能力与参考参与者确定后完成 CLI 校验和装配。
4. `P1-10` 汇总候选组合的数值、生命周期与端到端性能证据，并预检依赖和包消费。最终清理与安装消费由 `P1-11` 验收。CPU 必须实际运行。可选 GPU、MPI 各以实际组合执行结果标注状态，无设备或未执行的组合保持待验收。可选 MPI 诊断通过只说明其探针路径。
5. `P1-11` 原子切换发布路径。切换前确认以下条件，切换后重新生成目标图并执行干净构建、测试和安装消费检查。

   - 新 CPU 路径通过正常 CLI、参考试算、提交、回滚与重放；测试覆盖零长度、形状、版本和 owner 生命周期。保留既有物理约定和数值公式，代表性切片使用批准的基准与容差。
   - 源码、测试、独立安装消费者及支持的下游用法都迁到新的已记录入口。`Buffer`、旧 exec headers、`nexora::exec_api`、`nexora::exec_cpu` 的源与包兼容变更列入迁移说明。包配置可解析实际导出的 Kokkos 依赖。
   - 检索全部已装和在树 headers、target links、包配置、生成的 target graph 及 CI 后，清理旧 exec 源和链接。更新检查器与对外能力报告，复跑禁用能力负例。CUDA 路径启用时走同一 Kokkos 实现，未获得实机证据时按构建/未验收状态记录。

`P1-02`–`P1-06` 的开发分支可以短暂联调旧调用点；发布切换完成后，CPU 和可选 GPU 都走唯一的 Kokkos 计算及数值存储路径。具体任务分工、数值检查和设备验收以 [Phase 1 设计](phase-1-design-review.md) 为准。

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
| `EXEC-001` | **取消原生实现**。Kokkos Serial 承担 CPU 存储与执行。 | `P1-03`、`P1-04`、`P1-05` 分别验证 CPU 存储、保活和完成点，`P1-11` 删除旧目标、头与调用者。 |
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
| `SPH-001` **保留并迁移**。真实 density 阶段超出 P1 结构夹具。 | `src/methods/sph/{State,Density,Instantiations}.*`、`src/participants/sph/`、`tests/methods/sph/`、`validation/sph/`、`benchmarks/sph/`；`nexora_methods_sph`、`nexora_participants_sph`。 | typed 粒子/邻域输入和 trial density 输出，同一公式与已批准的精度/边界。前置 `BASE-001`、`P1-05`、`P1-06`、`P1-07`、`SPH-002`。 | 独立基准匹配密度和原物理约定；零/极端邻域与重排后的稳定 ID 对照通过。已启用 CPU/CUDA 分别实跑并记录含准备/归约的耗时。 |
| `SPH-002` **保留**。稳定 ID 与局部顺序仍须隔离。 | `src/methods/sph/neighbors/`、`tests/methods/sph/`；`nexora_methods_sph`。 | 定义 CSR、permutation 与按拓扑/几何/布局版本失效的邻域绑定；方法持有重排数据。前置 `P1-03`、`P1-07`。 | 双射、所有同域字段与历史一同重排、陈旧绑定拒绝或重建；用稳定 ID 对比重排前后关系，覆盖空/极端邻域与 ghost。 |
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
