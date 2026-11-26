# Nexora 当前架构入口

修订日期 `2026-11-26`。**这是唯一有效的架构入口。Kokkos Serial CPU 已实现参考状态生命周期、P1-04 字段访问与本地传输、P1-07 SPH 代表性切片，以及三维 WCSPH 的密度、压力、中心差分时间积分、恒定重力与 Adami 壁面。完整 Phase 1 仍开放。**

## 1. 决策与文档效力

Nexora 自有数值计算和存储统一使用 Kokkos，包括默认 CPU、可选 CUDA 及后续经实机验收的 HIP。Kokkos 是必需依赖，不存在绕过它的 native CPU/CUDA/HIP 通用实现、隐式 fallback、可选 Kokkos provider、通用 kernel DSL、编译器或自研 portability runtime。允许同一方法选择不同 Kokkos policy、布局和调优参数，数学公式、物理约定及容差仍须独立审批。

有效设计按本入口分工，发生矛盾时修正文档，不能用历史提案覆盖新决策：

| 文档 | 唯一负责的内容 |
|---|---|
| [ADR-003](../adr/ADR-003-kokkos-only-foundation.md) | 已确定的单一路线及替代关系；实现授权边界 |
| [Core 与数值数据结构](core-data-structures.md) | core 的值契约、Kokkos 原生数值接口及迁移边界 |
| [Kokkos 数据与执行设计](kokkos-design.md) | 关键 C++20 声明、字段权限、状态、同步、通信和关闭契约 |
| [Phase 1 评审与验收](phase-1-design-review.md) | 可执行任务、前置依赖、CPU/CUDA 与多方法代表性验收、实现审批事项 |
| [源码迁移与路线图](migration-roadmap.md) | 源码事实、新旧类型和 target 处置、归档 §27/§30 与 M/I 路线重映射 |
| [后续机械与集成任务](follow-on-task-plan.md) | 历史 MBD/INT 每项处置及候选前置；未立项项需另行设计和批准 |
| [Checkpoint 与重启](checkpoint-restart.md) | R0 的同配置范围、兼容检查/拒绝/验收；R1–R3 延期边界 |
| [构建与依赖政策](build-and-validation.md) | 版本锁定、工具链、离线供给、安装消费与配置能力 |
| [Serial CPU 基础设施验收](serial-foundation-acceptance.md) | 任务 1–7 的本地复跑、结构夹具计量、交付基线与能力边界 |

先前设计文档的文件清单与一致性检查见[文档审查记录](documentation-audit.md)，该记录不新增架构规范；实际实施与验证见 [CPU 基础切片](kokkos-cpu-slice-validation.md)、[参考状态生命周期](reference-lifecycle-validation.md)、[CPU 字段绑定](field-binding-validation.md)和[外部内存与本地传输](external-transport-validation.md)。

本次提交前的审阅、本地依赖重建及 float/double Debug/Release 复验见[字段迁移复查](field-migration-review.md)。

[ADR-001](../adr/ADR-001-module-boundaries.md)、[ADR-002](../adr/ADR-002-language-and-backend-baseline.md) 已被替代，只解释 bootstrap 源码。[归档 v1.2](../archive/CFDX_Multimethod_HPC_Architecture_Plan_v1.2_NVIDIA_Integration.md) 全篇为历史材料，其中的目录树、配置、代码、阶段和验收不是当前要求。历史中的物理问题可作为调研线索，须经现行计划重新采纳。Nexora 自有代码与文档采用 [BSD 3-Clause License](../../LICENSE)。复用第三方或 legacy 材料前，按 [ADR-014](../adr/ADR-014-license-and-provenance.md) 审查来源及授权。

## 2. 当前实现（源码事实）

首个切片要求本地 exact Kokkos 5.2.2 Serial-only、C++20 与 CMake 3.25.2+。`data → Kokkos::kokkos` 公开传递依赖；typed View 测试和只链接 `nexora::data` 的独立安装消费者执行 Kokkos 路径。CLI 报告 Kokkos 构建元数据。早期检查见 [CPU 切片验证](kokkos-cpu-slice-validation.md)。旧 `ArrayView`、`Vec`、`Tensor`、`FieldView`、`KernelView` 和 exec 层及其导出已移除；core 保留领域值契约，数值使用 Kokkos 原生类型。`nexora::real` 默认 `float`，可由构建选项 `NEXORA_ENABLE_DOUBLE_PRECISION=ON` 切换为 `double`，并由安装配置固定；现有 reference 数值字段、参数、算子和本地传输均随 real 精度选择，字段描述符默认 dtype 与之匹配；仿真时钟、字段时间戳、调度步长、运动壁的周期与采样时间使用 `double`。WCSPH 的材料与运动参数、邻域几何、密度、压力、积分和工作缓冲均使用 `real`；积分前检查原始步长转换为 `real` 后仍为有限正值。接口与迁移说明见 [core 数据结构](core-data-structures.md)，此前旧层切换的 Serial Debug/Release 测试和安装消费结果见[切换验证](core-data-structures-validation.md)；当前字段/算子的 float/double 数值证据见[数值迁移验证](real-numerics-validation.md)。

同步单参与者的运行入口为 `Simulation::initialize()`、`run(steps)`（或 `run(dt, count)`）和 `finalize()`。`run` 依次消费已计划的请求时长。每步由 `ParticipantIF::step` 对当前已接受约束作一次严格准入，完成全部数值和发布准备后发布状态，随后由 `Simulation` 推进时钟和计数一次。任一步失败即返回原 `Status`，保留前一成功状态和时钟，随后由调用方关闭。步长使用 `core/Time.hpp` 的 `TimeInterval::duration()`，绝对端点使用 double 加法生成。见 [Kokkos 单步契约](kokkos-design.md#7-单求解器的同步推进与状态发布)。

内部 `nexora_methods_reference` 和 `nexora_participants_reference` 以 typed `x/q` Views 保存数值。每步由已有的 accepted 读权读取，向独立临时数组求值并验证。在作用域内完成权限交接后一次发布新状态。旧快照持续持有其原读权和预算记录。

`nexora::data_schema` 导出纯字段语义、registry 和 `FieldStamp`，`participant_api` 经由它依赖 core。`nexora::data` 提供 managed Kokkos View 的 typed 绑定与同步 Serial lease。私有签发器以整分配仲裁，检查字段形状与版本，允许多个读者或一个写者。可信适配器可导入持有真实 owner 的一维 HostSpace 外部数组；本地 `TransportView` 显式打包只读载荷，源 lease 可在打包完成后释放，也可保留至消费者完成。本地接收在写入前验证版本、字段形状和稳定 ID 集合，再按 ID 解包到持有写 lease 的 trial 或 ghost 区域。设备异步交接和通用实体重排仍是后续任务。见 [外部内存与本地传输](external-transport-validation.md)及 [CPU 本地接收验证](local-receive-validation.md)。

`data` 内部以只读 Kokkos Views 保存稳定实体顺序、双向排列与独立行列顺序的 CSR 邻接。本地接收复用实体 ID 双射检查。见 [实体与拓扑验证](entity-topology-validation.md)。

测试树 `tests/validation/structure` 的专用 `StructureState` 验证夹具以候选 bank 一次发布实体顺序、主字段、被动历史和 CSR。Reference 步进、快照和结构夹具的本地收发使用整组 allocation 授权，数值工作由选定的 Serial 实例执行。见 [状态重排与整组授权](structure-execution-validation.md)。

Reference 的数值步同时发布主字段、历史、控制序列、诊断与版本。只读快照和独立传输缓冲各自持票，最后一个持有者释放后回收额度。

P1-07 的 Serial 验证夹具用 Simulation 运行固定 CSR 的一维 M₄ 密度与 owned 归约。逐行输出与归约均使用配置的 `real`，共享一次遍历。夹具 participant 负责稳定 ID、字段授权、行列重排和结果发布，跨步读者持有完整快照。私有目标 `nexora_sph_fixed_csr_fixture` 仅供一维测试及显式构建的 `nexora_sph_benchmark` 使用。[夹具源码](../../validation/sph/fixed_csr/) · [SPH Serial 验证](../../validation/sph/serial.md)

给定三维粒子状态可直接建立或更新 `CellGrid`，以只读 `NeighborView3D` 调用密度、压力或连续性算子。格网和版本比较按实际分配前预留受控额度。四组静水初态直接运行密度和固定壁压力算子。静水箱轨迹测试用解析剖面构造初态和核对误差，推进时按粒子的实际密度求压力。[格索引](../../validation/sph/cell-begin-serial.md) · [重构验收](../../validation/sph/sph-3d-refactor-serial.md) · [水箱验收](../../validation/sph/hydrostatic-continuity-repair.md)

私有 `sph::WcsphParticipant3D` 通过 `Simulation::run` 默认执行三维中心差分积分。初始化在初始几何上计算密度变化率和压力加速度。步进用已接受的导数推进粒子位置、速度和密度，更新邻域，在新几何上求步末导数并随历史一起提交。压力加速度字段只含压力项，积分时加恒定重力。首步用单步欧拉建立速度、密度历史，等步长采用中心差分；变步长以及每 39 个连续中心差分步后的重新同步采用两步 Adams–Bashforth。统一固定平滑长度 h 由 `WcsphVerletParameters::smoothing_length` 给定，格宽及支持半径为 2h。所有运动壁统一通过 `PrescribedWallTrajectory3D::motion` 提供绝对位姿与物理导数。调用方拥有壁运动；速度驱动算例由调用方以步初线速度和固定轴角速度前向欧拉形成候选位姿，SPH 只采样并变换参考粒子。步末导数与采样壁几何一同提交。未来 MBD 可经同一入口单向供数，双向载荷交换与联合恢复仍属后续任务。[运动壁时序与误差](../../validation/sph/velocity-driven-wall-serial.md)

应用层私有 `app::PrescribedMotion` 提供表达式运动装配，所有模式均经同一 `motion` 回调供数。`pose` 模式直接计算平移和固定轴转角及一、二阶导数；`velocity` 模式由调用方保存初始位姿，以步初速率欧拉推进并提供端点速率与加速度。表达式通过锁定的 Lepton 主机标量解释器解析和符号求导，变量 `t` 为未取模的物理时间。该 C++ 装配尚无 CLI 或场景文件入口，不安装导出。[运动装配与验证](../../validation/sph/velocity-driven-wall-serial.md)

声学 CFL 与各壁组采样限制中最小的值决定下一步上限。有限轨迹的右端点可以提交，随后可用步长为零。格网可从已接受几何重建。[WCSPH 生命周期](sph-3d-refactor-plan.md) · [积分公式与 Serial 验收](../../validation/sph/central-difference-serial.md)

CLI 的 `validate`、`plan` 和 `run` 共用版本化 Reference case 预检。`run` 装配现有 `Simulation` 与内部 Reference participant，并发布可核对的 accepted 数值摘要。 [case 格式与验证](reference-case.md)

任何 `NEXORA_GPU_BACKEND` 值均被拒绝，旧 native CUDA 源文件休眠。OpenMP、CUDA/HIP 尚未实现。完整 SPH 求解器仍待验收，FVM/FEM/LBM/DEM/MBD 求解器、MPI halo 和持久 checkpoint 尚未实现。CI 先供应锁定的 Kokkos，再用本地安装验证 Serial。可选 MPI smoke 有独立配置。已运行的平台、配置和禁网记录见 [Serial CI 验证](serial-ci-validation.md)。

当前生产目标的直接依赖如下。

```mermaid
flowchart TD
  CLI["nexora CLI"] --> C["core"]
  CLI --> K["Kokkos::kokkos"]
  CLI --> S
  CLI --> R
  CLI -. optional smoke .-> MPI["MPI::MPI_C"]
  S["simulation"] --> P["participant_api"]
  P --> DS["data_schema"]
  DS --> C
  F["data"] --> DS
  F --> K
  R["participants_reference"] --> P
  R --> M["methods_reference"]
  R --> F
  M --> F
  M --> K
  SP["participants_sph"] --> P
  SP --> F
  SP --> K
  SP --> SM["methods_sph"]
  SM --> C
  SM --> K
```

实际源文件与 target 的完整证据见[源码盘点](migration-roadmap.md#source-inventory)。`tools/check_dependencies.py` 检查当前边界，其通过不证明下述目标图全部实现。

## 3. 批准后模块图

箭头从使用者指向依赖。除上一节列明的实现外，新边和新模块仍是设计目标；后续模块按实际使用者引入，避免预建空框架。

```mermaid
flowchart TD
  App["应用装配 / 配置能力校验"] --> Sim["simulation · 时间与提交协调"]
  App --> Part["具体 participant · 状态授权与生命周期"]
  Sim --> API["participant API · 高层控制契约"]
  Sim --> CPL["coupling · 映射 / 载荷账本 / 窗口"]
  CPL --> API
  Part --> API
  Part --> Methods["FVM / FEM / SPH / LBM / DEM · 各自 state / topology / operator"]
  Part --> MBD["MBD engine contract / 可选外部 engine adapter"]
  Methods --> Data["data · typed 字段绑定 / 访问租约"]
  CPL --> Data
  Part --> Transport["通信 / checkpoint · 快照与完成条件"]
  Transport --> Data
  MBD --> Schema["data schema / mechanics · 物理语义"]
  API --> Schema
  Data --> Schema
  Data --> K["Kokkos · View / spaces / layout / policies / copy / fence"]
  Methods --> K
  Schema --> Core["core · ID / Status / 时间值"]
```

`exec` 取消，不重命名成另一套 execution abstraction。Kokkos 初始化由应用拥有的 `Kokkos::ScopeGuard` 管理；具体参与者持有类型化 execution space 实例。外部 stream 的特殊桥接只存在于私有 interop 文件。`data` 的薄层只增加字段语义、授权和保活，不复制 View 索引、布局、allocation、parallel_for、copy 或 fence API。控制层既不模板化为某种 GPU，也不持有 CUDA/HIP/MPI 原生类型。

## 4. 批准后 CMake target 依赖图

图是直接依赖允许集合，具体 target 只链接实际用到的项。P1 导出 `nexora::core`、`nexora::data_schema`、`nexora::data`、`nexora::participant_api`、`nexora::simulation`；`build_options` 是导出链所需 compile policy，不是功能 API。方法、具体参与者、interop 和后续集成先为内部 `nexora_*` target，公开安装须另验消费与版本契约。图上无前缀的内部节点对应同名 `nexora_` target。

```mermaid
flowchart TD
  cli["nexora"] --> sim["nexora::simulation"]
  cli --> pr["participants_reference / participants_x"]
  cli --> K["Kokkos::kokkos"]
  sim --> api["nexora::participant_api"]
  pr --> api
  pr --> mx["methods_reference / methods_x"]
  pr --> data["nexora::data"]
  mx --> data
  mx --> K
  data --> ds["nexora::data_schema"]
  data --> K
  api --> ds
  api --> core["nexora::core"]
  ds --> core
  core --> opt["nexora::build_options"]
  sim -. P2 .-> cpl["coupling"]
  cpl --> api
  cpl --> data
  cpl --> par["parallel_api"]
  par --> data
  mpi["parallel_mpi · opt-in"] --> par
  mpi --> MP["MPI::MPI_C · private headers"]
  io["io_checkpoint · P2"] --> data
  pm["participant_mbd"] --> api
  pm --> me["mbd_engine_api"]
  me --> ds
  me --> core
  eng["mbd_engine_adapter · opt-in"] --> me
  eng --> data
  eng --> SDK["external engine SDK · private headers"]
  bridge["interop_cuda · private opt-in"] --> data
  bridge --> RT["CUDA::cudart · private headers"]
  test["interop tests · P1"] --> bridge
  cli -. optional smoke .-> MP
  cli -. P2 injection .-> mpi
  pr -. P2 .-> par
  pr -. P2 .-> io
  eng -. certified interop .-> bridge
```

- `data_schema` 位于 `include/nexora/data/` 的元数据子集，`FieldDescriptor`/`FieldRegistry` 不再依赖执行或 memory-space 枚举。`data` 的 typed storage 头公开依赖 `Kokkos::kokkos`，这是有意的 API/ABI 变化。
- 模板化方法直接使用 Kokkos；相同 source 显式实例化已构建的 CPU/CUDA 类型，不设 `methods_x_cpu` / `methods_x_cuda` 两套实现库。内部实例化 object 单元可按编译器需要组织，不能成为第二实现路线。
- geometry/topology/numerics 按首次真实使用再提取共享 target，只依赖 `core/data` 和实际数值依赖；不得反向依赖 methods/participant/simulation。physics 保存物理模型与参数，不持有设备 runtime。
- P2 的 concrete participant 按用途链接 `parallel_api`、`io_checkpoint`；应用注入 `parallel_mpi`。外部 SDK 和 MPI 原生头保持私有；静态库的传递链接需求仍必须由安装配置解析，不能用 `PRIVATE` 掩盖真实依赖。
- 图中 CLI 的 optional smoke 边是私有诊断边 `nexora → MPI::MPI_C`；该探针保留，但不是 parallel_mpi 数值支持证据。旧 CUDA native 探针改用 Kokkos，CLI 不保留直接 CUDA runtime 链接；原生 stream 测试放在私有 interop target。
- Kokkos CUDA 构建自身的公开编译/链接要求可经 `Kokkos::kokkos` 传递。禁止的是 Nexora 公共契约暴露原生 `cudaStream_t`/`hipStream_t`/`MPI_Comm`，不是假装设备工具链没有依赖。
- 完整迁移时同时核对 target 边、header 所属子集、可选节点、源文件归属和安装导出；反例必须失败。当前已更新 schema 与 typed data 的 CMake 直接边、头边界检查和安装包测试。[Serial CI 验证](serial-ci-validation.md)涵盖本地 Kokkos 供应、依赖图及搬迁消费。

## 5. 交付与支持口径

单一基础设施不要求五种方法使用同一算法、rank 或布局。共同契约是物理语义、稳定 ID、版本、访问权、事务和可观测通信完成；具体离散结构见[方法数据边界](kokkos-design.md#method-state)。经批准的完整外部引擎可保有内部运行时，Warp 只可作为这种引擎的内部依赖；不得通过外部应用、服务或进程包装重建 Nexora 通用算子 provider 或第二套计算基础设施。

后续任务必须使用现行文件中的定义与精确任务 ID；归档链接只用于核对出处，不提供实施授权或补充未迁入的规范。P2 持久恢复仅采纳 [R0 同配置重启](checkpoint-restart.md#restart-levels)；尚未立项的 M/I 及高级能力需另行设计和批准。

Phase 1 直接建立 Kokkos 基础及 SPH/FVM 代表性结构切片；完整方法、分布式数值及持久恢复按后续阶段关闭。CPU/CUDA 实机数值、生命周期和端到端性能都通过后才可宣称完整 P1。HIP 必须另有 AMD 实机证据。构建成功、诊断通过和运行验收是三个不同状态。

首个 CUDA 构建按 [nvcc_wrapper/CXX 与安装消费契约](build-and-validation.md#cuda-build-contract)：锁定 Kokkos 的 CUDA+Serial、C++20、单一 GPU 架构，匹配的 wrapper/工具链用于 Nexora 和下游数值消费者；删除旧 `NEXORA_GPU_BACKEND`，从已验证的安装包推导构建组合。尚无 runner 配置或构建执行证据。

本次授权涵盖旧 core 数值容器及 exec 层的提前清理；其余未授权范围按 [Phase 1 的具体变更包](phase-1-design-review.md#approval)执行。Kokkos 单一路线本身不再列为待决项。
