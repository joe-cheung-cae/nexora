# MBD 与外部集成后续任务对照

> **需另行设计和批准。** 本表接续[迁移路线](migration-roadmap.md)，逐项解释旧任务的当前去向。保留或迁移仅指候选范围，合并指承接既有契约，延期等待明确用例和证据，取消项无交付物且不作为前置。候选目录和内部 target 仅供定位，未创建，也不改变 [target 许可边](overview.md) 或扩大 Phase 1 的实施范围。逐项批准时冻结所选接口、依赖、发布组合、验收和复用权利。

表中的 `P1-*`、`SPH-*`、`DEM-*`、`CPL-*`、`TIME-*`、`CKPT-*`、`MPI-*` 分别在[首期计划](phase-1-design-review.md)和 [P2 任务表](migration-roadmap.md#p2-task-ids)定义。M/I 前置按本页各行执行，原归档只用于追溯；依赖某个 P3/P4/P5 出口时，必须交付行内列明的可运行接口和证据。这些后续阶段尚无获准的实施任务表；立项时必须把所依赖出口拆成具体任务 ID 并批准，不能用阶段名或本表候选门槛直接开工。首期评审只确定边界。

Nexora 自有机械和流体状态、算子及数值存储统一使用 Kokkos。外部 Chrono/Newton 等只能按完整动力学引擎和机械端口接入，明确 owner、复制、完成与恢复。**Warp 仅可作为经批准的完整外部动力学引擎的内部依赖。** 外部应用、服务、进程或代理均不得包装 Warp 以提供通用算子 provider、JIT/AOT 执行路线或第二套计算/存储基础设施。查看[机械执行与恢复边界](kokkos-design.md)和[阶段范围](migration-roadmap.md#legacy-mbd)。

<a id="mbd-task-ids"></a>
## MBD-001–020

对应[归档 §36.5](../archive/CFDX_Multimethod_HPC_Architecture_Plan_v1.2_NVIDIA_Integration.md#sec-36)。以下各项除取消项外均需另行设计和批准。

| 任务 | 处置及理由 | 现行前置与候选落点 | 候选退出门槛 |
|---|---|---|---|
| `MBD-001` | **保留** frame/pose/load 值类型。机械功和取矩点需要公共语义。 | `P1-03`；`include/nexora/mechanics/`（纯语义子集，候选 `nexora::data_schema`）。 | 换点、换系、姿态与功率恒等式通过，拒绝缺失单位、frame 或参考点。 |
| `MBD-002` | **保留** motion/inertia/contact owner 校验。耦合不得重复计算同一物理贡献。 | `MBD-001`、`CPL-001`；`src/coupling/mechanics/`、`src/coupling/contact/`（`nexora_coupling`）。 | 同一物体的运动/惯性重复配置和同一接触对的重复 owner 均被拒绝，显式列出载荷去向。 |
| `MBD-003` | **保留** `MbdModel` 与 `CoordinateLayout`。广义坐标和速度不能按同一长度解码。 | `MBD-001`；`include/nexora/mechanics/`、`schemas/mbd/`（候选 `nexora_mbd_engine_api`）。 | 验证质量惯量、q/v 长度、关节 frame 及约束声明，越界模型失败。 |
| `MBD-004` | **迁移**受限参考引擎到 Kokkos。参考值由独立解析例给出。 | `MBD-003`、`P1-05`；`src/mbd/reference/`（候选 `nexora_mbd_reference`）。 | 已批准 formulation 的自由落体、旋转和小机构通过；不支持的关节/约束明确拒绝。 |
| `MBD-005` | **迁移** engine/participant 契约。引擎报告独立于高层生命周期。 | `MBD-003`、`P1-06`；`include/nexora/mechanics/`、`src/participants/mbd/`（候选 `nexora_mbd_engine_api`、`nexora_participant_mbd`）。 | `MbdEngineIF` 使用 `MbdAdvanceReport`，由 participant 转换并绑定语义端口。dummy engine/fluid 验证 trial、commit、restore 及失败传播。 |
| `MBD-006` | **延期** Chrono 私有适配。须先确定可用版本、许可和实际模型。 | `MBD-005`、`P1-02`；`src/adapters/chrono/`（候选 `nexora_mbd_engine_adapter` 的实现）。 | 依赖 on/off、ID/frame/load 转换、单机构独立数值对照通过；强耦合支持取决于完整恢复认证。 |
| `MBD-007` | **迁移**完整快照与重放。engine hidden state 由具体 owner 提供。 | `MBD-004`、`MBD-005`、`CKPT-001`；`src/mbd/checkpoint/`、`src/participants/mbd/`（候选 `nexora_mbd_reference`、`nexora_participant_mbd`）。 | 参考引擎捕获接触/控制/积分器/约束历史并重放；外部引擎另行证明，缺完整能力时拒绝需要恢复的生命周期。 |
| `MBD-008` | **迁移**刚体边界变换到单一 typed Kokkos 算子。取消两套 CPU/CUDA 实现。 | `MBD-001`、`P1-03`、`P1-05`；`src/mbd/boundary/`（候选 `nexora_mbd_boundary`）。 | 多 body、COM 偏置、重排、frame 与所选 execution instance 顺序通过；CUDA 申报需实机。 |
| `MBD-009` | **迁移**分段合力/矩规约到 Kokkos。已有精度、取矩点及来源契约继续适用。 | `MBD-008`、`CPL-003`；`src/mbd/boundary/`、`src/coupling/loads/`（候选 `nexora_mbd_boundary`、`nexora_coupling`）。 | 以独立参考验证不同原点的合力/矩、分段来源与已批准 FP64 容差。 |
| `MBD-010` | **保留并迁移** SPH–MBD 案例。将流体边界/载荷与 engine 生命周期分离。 | `MBD-005`、`MBD-007`、`MBD-008`、`MBD-009`、`SPH-001`；`src/coupling/sph_mbd/`、`cases/verification/coupling/sph_mbd/`（候选 `nexora_coupling`）。 | 规定运动、静水及自由体分步验证，闭环力/矩、功和恢复符合各自已批准基线。 |
| `MBD-011` | **保留** DEM 边界代理与接触过滤。history 和成对反力有明确 owner。 | `DEM-001`、`MBD-002`、`MBD-005`、`MBD-007`；`src/coupling/dem_mbd/`（候选 `nexora_coupling`）。 | 偏心冲击的反力/矩、成对冲量、history 恢复和唯一 owner 通过。 |
| `MBD-012` | **保留**轨迹及载荷时间积分。区分瞬时值、波形与积分量。 | `MBD-007`、`CPL-003`、`TIME-001`；`src/coupling/temporal/`（`nexora_coupling`）。 | 运动 COM、时变 Jacobian、事件冲量及不同子步组合的时间预算通过。 |
| `MBD-013` | **保留**强耦合 feasible-motion 迭代。先验证可恢复的双向闭环。 | `MBD-010`、`MBD-012`；`src/coupling/iterations/`（候选 `nexora_coupling`）。 | 旋转残差、关节约束、added-mass 案例和失败时无 accepted 发布通过。 |
| `MBD-014` | **保留**三方机械子步组。复用已验收的两条机械耦合。 | `MBD-011`、`MBD-012`、`MBD-013`；`src/coupling/sph_dem_mbd/`、`cases/verification/coupling/sph_dem_mbd/`（候选 `nexora_coupling`）。 | 三方同一权威时钟和积分者，避免重复载荷；冲量/功预算与失败恢复通过。 |
| `MBD-015` | **迁移** MPI owner/reduce/distribute 到可选通信边界。底层不暴露 MPI 类型。 | `MPI-001`、`MPI-002`、`MBD-002`、`MBD-009`；`src/parallel/mpi/`、`src/coupling/mechanics/`（候选 `nexora_parallel_mpi`、`nexora_coupling`）。 | 1/2/4 rank 对照、空贡献 rank、排除 ghost 和通信失败时不提交。 |
| `MBD-016` | **保留** FVM–MBD 独立适配。运动网格/GCL 和牵引力与其他方法分别验证。 | `MBD-012` 及 P3 已验收的网格几何/面通量/牵引接口；`src/coupling/fvm_mbd/`（候选 `nexora_coupling`）。 | 选定路线的 GCL、压力约定、合力/矩与强耦合重放。 |
| `MBD-017` | **保留** LBM–MBD 独立适配。moving links、物理单位和时间量子属于具体方法。 | `MBD-012` 及 P5 已验收的 lattice/运动边界/单位接口；`src/coupling/lbm_mbd/`（候选 `nexora_coupling`）。 | 移动边界动量交换、固定时间量子及 populations/parity 恢复通过。 |
| `MBD-018` | **保留** FEM–MBD trace/constraint 适配。先验收全阶刚柔接口。 | `MBD-002`、`MBD-012` 及 P4 已验收的 FEM trace/约束/历史恢复；`src/coupling/fem_mbd/`（候选 `nexora_coupling`）。 | 虚功、唯一质量归属、连接假设、约束与重放独立验证。 |
| `MBD-019` | **延期** ROM 导入与模式恢复。依赖独立 ADR、模型来源和适用性批准。 | `MBD-018`；`src/mbd/rom/`（候选 `nexora_mbd_rom`）。 | 模式归一化、静力、频率、接口重构、恢复和模态数收敛通过。 |
| `MBD-020` | **合并**到所选机械交付的诊断和发布验收。避免无证据的综合能力表。 | `MBD-005`、`MBD-007`。逐配置追加拟发布行的任务 ID，例如 SPH 双向 `MBD-010`、三方 `MBD-014`、机械 MPI `MBD-015`；`validation/mbd/`、`benchmarks/mbd/`（测试/基准叶目标）。 | 只列实际通过的模型、引擎版本、设备、恢复级别；记录独立正确性、全链耗时和峰值内存。 |

<a id="integration-task-ids"></a>
## INT-001–030

对应[归档 §46.5–46.7](../archive/CFDX_Multimethod_HPC_Architecture_Plan_v1.2_NVIDIA_Integration.md#sec-46)和[现行 I0–I7 范围](migration-roadmap.md#legacy-integrations)。以下各项除取消项外均需另行设计和批准。

| 任务 | 处置及理由 | 现行前置与候选落点 | 候选退出门槛 |
|---|---|---|---|
| `INT-001` | **迁移**外部契约与隔离。按实际使用方划分数值快照、场景、控制边界，移除 provider profile。 | `P1-03`、`P1-11`；机械端口追加 `MBD-001`。候选 `docs/architecture/`、`include/nexora/interop/`（`nexora_interop_api`）。 | core 无 SDK/脚本 runtime 依赖，边界检查和能力拒绝有正反例。 |
| `INT-002` | **迁移**中性描述符校验。字段逻辑语义与外部 payload 布局分离。 | `INT-001`、`P1-03`；`include/nexora/interop/`、`src/scene/`（候选 `nexora_interop_api`、`nexora_scene`）。 | dtype、shape/stride、frame、时间、ID、容量与版本正反例通过；仅经验证的载荷可导入。 |
| `INT-003` | **合并**底层 lease 与完成契约到 `P1-04`，实际外部导出延期到具体消费者。 | `INT-002`、`P1-04`；`src/interop/`（候选私有 `nexora_interop_export`）。 | 两消费者、迟到完成、失败取消与恰好一次释放通过；GPU 组合有实机证据。 |
| `INT-004` | **延期**可选 C ABI/Python 数据桥。只在外部消费者需要版本化载荷时立项。 | `INT-002`、`INT-003`；`src/capi/`、`bindings/python/`（候选 `nexora_capi`、Python 扩展叶目标）。 | ABI 尺寸/版本、错误、数据所有权与完成条件测试通过；Nexora CPU core 的构建与运行独立于 Python。 |
| `INT-005` | **取消** Warp 内部批量算子 provider。Nexora 自有算子按 `P1-05` 使用 Kokkos。 | 无后继依赖、目录或 target。 | 不进入交付或能力清单。 |
| `INT-006` | **取消** Warp provider 的 stream/runtime bridge、预热及 profile。具体 engine 内部依赖由该 engine 的审批与锁定覆盖。 | 无后继依赖、目录或 target。 | 不进入交付或能力清单。 |
| `INT-007` | **取消** Warp AOT provider 和原生加载实验。编译产物同样会形成第二套算子路线。 | 无后继依赖、目录或 target。 | 不进入交付或能力清单。 |
| `INT-008` | **延期** Newton 模型与 solver 能力审计。只考虑一个获准的完整 engine。 | `MBD-003`、`MBD-005`、`INT-002`；`src/adapters/newton/`（候选 `nexora_mbd_engine_adapter` 的实现）。 | 有精确版本、许可、受支持关节/约束、精度、状态及恢复能力清单；不可用项明确拒绝。 |
| `INT-009` | **迁移** Newton 位姿/力矩/q-v 转换。避免用数组布局代替机械约定。 | `INT-008`、`MBD-001`；`src/adapters/newton/conventions/`（候选 `nexora_mbd_engine_adapter`）。 | 偏 COM、frame、功共轭、四元数与 q/v 顺序通过独立对照。 |
| `INT-010` | **迁移** Newton 完整 trial/restore/commit。独立认证 engine 隐状态。 | `INT-009`、`MBD-007`；`src/adapters/newton/checkpoint/`（候选 `nexora_mbd_engine_adapter`）。 | 独立机构重放、试算/恢复失败测试和隐状态清单齐全；缺完整恢复时拒绝事务能力。 |
| `INT-011` | **保留** SPH–Newton 双向候选。完整 engine 替换需要真实守恒证据。 | `INT-010`、`MBD-010`；`src/coupling/sph_mbd/`、`cases/verification/coupling/sph_newton/`（候选 `nexora_coupling`）。 | 浮体/铰接体、反力矩、冲量与回滚对照参考引擎；缺实测不开放强耦合。 |
| `INT-012` | **保留** DEM–Newton 接触唯一 owner。避免引擎和 DEM 同时积分相同惯性/接触。 | `INT-010`、`MBD-011`；`src/coupling/dem_mbd/`、`cases/verification/coupling/dem_newton/`（候选 `nexora_coupling`）。 | 偏心冲击、历史恢复、唯一接触 owner 和成对反力矩通过。 |
| `INT-013` | **保留**可选 OpenUSD 受限导入。缺失语义须显式报告。 | `INT-002`；`src/adapters/openusd/`（候选 `nexora_openusd_adapter`）。 | 验证单位、frame、关节和不支持的非均匀 scale，未知字段不产生无依据的物理输入。 |
| `INT-014` | **保留** USD/sidecar 及实体绑定。只消费已接受的场景快照。 | `INT-013`、`P1-06`；`src/scene/`、`src/adapters/openusd/`（候选 `nexora_scene`、`nexora_openusd_adapter`）。 | 一个实际案例的稳定 ID、粒子/表面/刚体表达、单位与 frame 往返一致。 |
| `INT-015` | **保留**可选 Kit 只读客户端。展示入口隔离 solver 运行和主状态。 | `INT-014`；`integrations/omniverse/kit/`（独立客户端，不链接 core target）。 | 文件加载、单写 stage、卸载清理和读取旧快照行为通过。 |
| `INT-016` | **保留**版本化协议和兼容检查。wire schema 独立于 View 或进程 ABI。 | `INT-001`、`INT-002`；`protocols/`、`schemas/`（候选 `nexora_integration_protocol`）。 | 新旧客户端版本协商、有界解码和不支持版本拒绝通过。 |
| `INT-017` | **保留**控制 facade 与幂等账本。命令只能由权威窗口协调者应用。 | `INT-016`、`TIME-001`；`src/service/facade/`（候选 `nexora_service_facade`）。 | 重复/超时请求不重复推进；记录命令 ID、应用版本和窗口结果。 |
| `INT-018` | **保留**可选 gRPC 会话和订阅。背压与访问寿命是服务端专项。 | `INT-017`、`INT-003`；`integrations/grpc/`（候选 `nexora_grpc_service`）。 | 有界内存、分块、权限、deadline/断线与 snapshot lease 通过。 |
| `INT-019` | **保留** Kit 在线快照/控制。保留唯一权威时钟。 | `INT-015`、`INT-018`；`integrations/omniverse/kit/`（独立客户端控制插件）。 | 授权暂停/恢复/更新、只读快照、有界背压和断线恢复通过。 |
| `INT-020` | **延期** SHM 本机交换。性能证据须证明普通复制路径不足。 | `INT-018`、`INT-003`；`integrations/shared_memory/`（候选 `nexora_shared_memory_transport`）。 | 环满、重连、lease/generation、旧 writer 退出及快照完整性通过。 |
| `INT-021` | **取消专用原生分配，延期 IPC 优化**。现有 Kokkos/外部合规 owner 与显式复制边界承担寿命。 | `INT-020`、`INT-003` 和实际 CUDA/OS 性能证据；`integrations/cuda_ipc/`（候选私有 `nexora_cuda_ipc_transport`）。 | 仅导出可证明与 Kokkos 或获准外部 owner 兼容的存储；否则显式 staging 或拒绝。实机检验跨进程释放和 stale writer，禁止新增求解数值分配器。 |
| `INT-022` | **延期** Fabric 展示优化。只在实测展示瓶颈成立时评估。 | `INT-019`；`integrations/omniverse/fabric/`（独立客户端组件）。 | 重载时旧 handle 失效，等价快照可见，测得端到端显示收益。 |
| `INT-023` | **迁移** rank-aware gateway 到可选 MPI 边界。服务线程不能隐式发起 collective。 | `INT-018`、`MPI-001`、`MPI-002`；`src/service/mpi_gateway/`（候选 `nexora_mpi_gateway`）。 | 由指定通信线程协调空 rank、断线和失败决定；2/4 rank 实测数值一致与缓冲完成。 |
| `INT-024` | **延期** Newton worker 隔离。worker 仅承载完整已批准的动力学引擎。 | `INT-010`、`INT-018`；`integrations/newton/worker/`（独立 worker 与候选私有 adapter）。 | 报告 IPC/复制、权限、重连和控制权，验证 fail-stop 和重放。进程隔离不允许提供通用算子。 |
| `INT-025` | **延期**远端参与者窗口。受控远端完整参与者与其恢复协议须分别认证。 | `INT-016`、`INT-018`、`TIME-001`、`CKPT-001`；MPI gateway 模式追加 `INT-023`，Newton worker 模式追加 `INT-024`。候选 `integrations/grpc/client/`、`src/coupling/remote/`（`nexora_remote_participant`）。 | retry/restore/commit 幂等，断线和未知完成状态时不提交，不重复载荷或推进。 |
| `INT-026` | **延期** Kit/Isaac 引擎桥。只评估明确产品需求和完整引擎。 | `INT-019`、`MBD-002`、`MBD-005`；`integrations/omniverse/isaac/`（独立插件）。 | 精确版本、明确机械 owner、同一物体只有一个 engine 推进；恢复能力按选定组合另验。 |
| `INT-027` | **延期**独立渲染、流媒体与资产库。三类需求分别立项。 | 离线场景路径 `INT-014`，在线路径 `INT-018`；`integrations/{rendering,streaming,assets}/`（独立叶组件）。 | 以真实场景验证对应能力、许可、资源上界与性能，保持求解器可独立运行。 |
| `INT-028` | **保留**安全及资产 resolver 硬化。外部数据/协议进入时验证。 | USD 路径 `INT-013`，网络路径 `INT-018`；`src/scene/`、`src/service/`（所选 adapter/service 私有实现）。 | 权限、URI/路径穿越、配额、超大/畸形输入负例通过。 |
| `INT-029` | **合并**到各已批准集成的离线发行验收。取消通用 provider/AOT 发布物。 | `P1-02`、`P1-11`，发布清单逐项列出组件前置 ID，例如 USD `INT-014`、Newton `INT-010`、在线控制 `INT-018`；`packaging/`、CI 文档（包装/消费叶目标）。 | 每个获准组合记录真实版本、摘要、许可证、ABI、安装消费与禁网运行结果。 |
| `INT-030` | **合并**到具体组合的全链验收。整合数值、恢复、通信、内存和性能归因。 | `P1-10`、`INT-029`，组合清单加实际链路 ID，例如 SPH–Newton `INT-011`、DEM–Newton `INT-012`、远端窗口 `INT-025`；`validation/integration/`、`benchmarks/integration/`（验证/基准叶目标）。 | 按选定组合记录五线结果和可重跑清单，区分配置、编译、执行、数值与性能，不以孤立 smoke 宣称端到端支持。 |
