# 本轮文档修订与一致性审查记录

审查时间：`2026-11-06T15:58:40+08:00`。实现基线：`3495e7e80d9439a6af0e00be1d444f52d6fc8552`。**本轮仅文档修订，无正式源码、CMake、preset、CI 或测试代码改动。** 本记录是变更清单与审查证据，不是另一份架构规范；唯一入口仍是 [overview](overview.md)。

## 修改文档清单

| 文件 | 本轮处置 |
|---|---|
| [README](../../README.md) | 将 Kokkos 单一路线和当前尚未迁移状态放到入口；旧命令转到贡献说明 |
| [AGENTS](../../AGENTS.md) | 禁止重新引入 native/provider/DSL runtime；固定文档审批边界与离线必需依赖政策 |
| [CONTRIBUTING](../../CONTRIBUTING.md) | 当前 bootstrap 命令与目标开发/验证政策分开 |
| [third_party/README](../../third_party/README.md) | 必需 Kokkos、精确 revision、来源/许可证/制品摘要及升级政策 |
| [overview](overview.md) | 唯一架构入口；当前/目标模块图、target 图与有效文档分工 |
| [kokkos-design](kokkos-design.md) | 新增完整数据、执行、状态、通信、外部引擎和关闭设计 |
| [phase-1-design-review](phase-1-design-review.md) | 重写用户提供的未跟踪基线；P1-01–11 任务、V1–V12 验收与实现审批 |
| [migration-roadmap](migration-roadmap.md) | 新增源码精确盘点、旧类型/target 去向、删除条件和阶段/任务重映射 |
| [build-and-validation](build-and-validation.md) | 版本/工具链锁定、离线构建；追加首个 CUDA wrapper/CXX、架构、配置匹配、旧选项删除与独立消费契约 |
| [checkpoint-restart](checkpoint-restart.md) | 新增 R0 同配置范围、兼容/拒绝、逻辑数据、发布与 CP-01–08；R1–R3 延期 |
| [follow-on-task-plan](follow-on-task-plan.md) | 新增全部 20 个 MBD、30 个 INT 历史任务的处置/理由与候选前置/落点/门槛；均需另行设计和批准 |
| [ADR-001](../adr/ADR-001-module-boundaries.md) | 标记 bootstrap 决策被 ADR-003 替代 |
| [ADR-002](../adr/ADR-002-language-and-backend-baseline.md) | 标记旧 C++17/平台基线被替代，保留当前源码解释 |
| [ADR-003](../adr/ADR-003-kokkos-only-foundation.md) | 新增单一路线决策，区分已定选型与待实施细节 |
| [ADR-014](../adr/ADR-014-license-and-provenance.md) | 保留项目许可证待定；区分 Kokkos 自身许可与 Nexora/legacy 复用权利 |
| [历史 v1.2 路线图](../archive/CFDX_Multimethod_HPC_Architecture_Plan_v1.2_NVIDIA_Integration.md) | 全篇历史标识和受影响章节提示；§27/§30 入口改指新重映射，原正文保留 |
| 本记录 | 新增审查结果和交付索引 |

## 冲突检索与处置

使用 `rg` 盘点全部仓库 Markdown，并检查实际 `include/src/apps/tests`、CMake/presets、依赖检查器、安装消费和 CI。全文扫描关键词包含 `Kokkos/provider/DSL/native/backend/Buffer/AllocationRef/ExecutionQueue/ExecutionEvent/KernelView/exec_cpu/C++17/无第三方依赖` 等。所有保留命中归为以下明确身份：

| 保留命中 | 身份与处理 |
|---|---|
| 当前 C++17、Buffer/ExecContext/exec_cpu、原生 CUDA smoke | 源码事实，放在“当前实现”或迁移表；禁止作为目标扩展 |
| 新文档中的旧类型/provider/DSL | 删除、取消或禁止项，逐项说明替代职责；不是仍有效的接口 |
| 归档里的旧目录、示例、配置、原生/provider 路线 | 全篇 Superseded，受影响章节另加提示；不是现行要求 |
| 原生 CUDA stream 的限定引用 | 仅为私有互操作的具体需求、所有权与测试边界，不构成第二条执行路线 |
| 旧无第三方 CPU 要求 | 已替代；目标 CPU 必需 Kokkos，离线表示预置且可复验的依赖 |
| 官方滚动页面与固定版本的差异 | 以 5.2.2 固定 commit 源码确认要求；不照搬 develop 或未来 API |

归档 §30 的 25 个主干/P2、§36.5 的 20 个 MBD、§46.5–46.7 的 30 个 INT，共 75 个历史任务 ID 均逐项映射处置与理由。13 个 P2 任务补齐文件/target/接口/明确前置/通过条件；M/I 未立项项仅为候选，需另行设计和批准。P0–P7、M0–M7、I0–I7 均有范围与前置说明。历史正文只插入归档提示，不悄悄改写旧论证。旧绝对 Markdown 行号引用已从 Phase 1 清理，采用当前文件/明确锚点。

## 复核问题修复与现行定义完整性

本次接续复核先确认已有 stage 链权限、字段构造和 stream/instance 关闭顺序已修复，保留它们；新增修复如下：

1. 历史任务映射和 P2 精确前置补全；`CPL-004` 不再使用“上述相关任务”。`BASE-001` 补目录、测试叶 target、前置、baseline manifest、可重跑命令、硬件/工具链及原始数值/性能证据。未来阶段出口需在立项时拆具体 ID 并批准，不从归档自动继承。
2. R0 的实际条件已迁入现行 checkpoint 规范；同 binary/依赖、rank/分区、execution/device 与数值配置，缺失/不匹配拒绝。IO-001 和 V11 直接引用 CP 验收；R1–R3 均延期。明确 `MPI-003` 只在固定 rank 数运行中迁移，R0 loader 不重分区；快照按记录的分区恢复。
3. Context7 查询编译/安装接口后，以锁定 5.2.2 官方源码核实。首个 CUDA 使用 `nvcc_wrapper`/CXX、C++20、CUDA+Serial 和单一显式架构；安装消费者复用匹配工具链及传递目标，禁止隐式 launcher/native fallback。删除并拒绝旧 `NEXORA_GPU_BACKEND`，以已验证的 Kokkos 包确定构建组合；追加目标命令、消费契约与负向矩阵。实际 runner 未知，精确版本和运行结果均待提供。
4. Warp 在设计、路线图、ADR、AGENTS 和归档替代提示中统一为获准完整外部引擎内部依赖；不得借外部应用、服务、worker 或进程重新建立通用 provider/第二套基础设施。
5. 所有现行归档链接逐一检查为出处/历史追溯。发现 `CPL-002` 的 `H/Hᵀ` 原来只有名称，已将原有映射方向、负号、已积分载荷、参考系/时序和功率适用条件迁入现行定义；没有改公式、选算法或新容差。尚未立项的机械/高级任务保留设计缺口，必须在具体计划中补齐，归档不能补作有效规范。

本次接续修改 10 份已有文档（AGENTS、CONTRIBUTING、ADR-003、overview、kokkos-design、phase-1-design-review、migration-roadmap、build-and-validation、历史路线图、本记录），新增 checkpoint-restart 和 follow-on-task-plan 两份。README、ADR-001/002/014、third_party/README 的已有修复仅验证；它们在本轮工作区中仍包含前一轮改动。

## 独立评审与验证

两轮独立审查已发现并修正：stage 链授权与异步排序的冲突、外部 stream 先于 execution instance 析构的风险、fence 与 host 可访问性的混淆、字段绑定不可构造的声明、MBD engine 对 participant 类型的隐性依赖、验收前运行入口与支持认证的区别，以及 P1 任务/最终删除验收的循环前置。

已执行的文档检查：

- 全部 17 份仓库 Markdown、264 个本地链接/锚点检查通过，代码围栏无未关闭项；75 个历史 ID 各有唯一处置行。13 个 P2 和 50 个 MBD/INT 行的前置 ID 可解析、无循环，保留任务不依赖已取消的 INT-005/006/007。归档去除新增 Superseded 提示后与 HEAD 原正文逐字节一致。
- 全文一致性检索通过；旧 runtime/provider/native、旧类型/target、C++17/无依赖、R0–R3、Warp 与 CUDA 选项等命中逐类检查。模糊“上述相关任务”只剩已废止归档原文，现行任务已用 ID 替换。`git diff --check` 通过；改动和未跟踪新文件均为 Markdown。
- 与接续开始时 69 个仓库文件的 SHA-256 快照比较，本次没有非 Markdown 文件变化。对源码、根 CMake/presets、`include/src/apps/tests/tools/cmake/.github` 与 LICENSE 的 diff 为空；没有启动迁移。
- 上一轮曾对已有 CPU Debug、CPU Release、MPI Debug target graph 运行当前依赖检查器并通过；本次未重复运行。该历史结果只说明旧 bootstrap 图符合旧规则，不验证目标架构。追加只读复核确认 R0、CUDA 包契约及上述定义补齐无剩余必须修复项。

**尚未执行：** Kokkos provisioning/build、声明编译、目标安装消费、CPU/CUDA/HIP 数值运行、MPI 数值、真实生命周期及性能矩阵。本轮未改变数值公式、默认值或容差；SPH/FVM 基准缺口保留为实施前数学审批，不虚构已存在的 solver。审批清单见 [Phase 1](phase-1-design-review.md#approval)。
