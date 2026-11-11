# Checkpoint 与重启契约

修订时间：`2026-11-06T15:58:40+08:00`。**批准后目标，尚未实现或验收。** 本文补齐现行计划已经采纳的 `CKPT-001/IO-001` 契约，不授权实现，不增加跨 rank、跨设备或跨版本恢复范围。唯一入口是 [overview](overview.md)，任务落点见[当前 P2 任务表](migration-roadmap.md#p2-task-ids)。

## 1. 内存回滚与持久重启分开

- **P1 单参与者窗口回滚**：同一进程内保存 accepted 状态，试算失败后安全结束访问并丢弃 trial。Serial Reference 已覆盖 Kokkos 主状态、最小 replay history 与控制序列。 [实现和后续捕获清单](reference-resource-validation.md#未来-checkpoint-清单)
- **CKPT-001**：P2 多参与者完整内存恢复点；包括影响 replay 的主状态、历史、积分器、控制、耦合数据。它不意味着存在可跨进程读取的文件。
- **IO-001 / R0**：把一个已一致接受的恢复点持久化，并在新进程中按下述相同配置加载、继续计算。仅保留 pose/velocity 或可视化字段不足以实现 R0。

Kokkos fence、MPI request complete、writer 完成、全局 checkpoint 发布是不同条件；具体保活/数据复制遵循 [Kokkos 设计](kokkos-design.md)。Kokkos 或设备致命失败后不承诺同进程修复；只有已发布的完整持久恢复点及通过兼容检查的新进程可用于 R0。

<a id="restart-levels"></a>
## 2. 重启等级及采纳状态

| 名称 | 本文中的精确定义 | 采纳与限制 |
|---|---|---|
| **R0：同配置重启** | 相同可执行文件及加载的数值模块/依赖制品、相同 rank 数和分区/owner 映射、相同方法/模型版本、execution space、精度/数学模式、获验硬件及运行配置，从完整 accepted checkpoint 继续 | **保留为 IO-001 的唯一首期持久重启目标**。不是已实现能力。串行 rank 数定义为 1；MPI 组合必须另有同 rank 验收。下节条件全部成立才能读取。 |
| **R1：重新分区重启** | 在相同方法版本下改变 rank 数或分区/owner 分布，重新分配拓扑及历史 | **延期，需另行设计和批准**。目前请求改变 rank 数或重分区必须拒绝；稳定 ID 存在不自动获得此能力。 |
| **R2：跨执行/设备配置重启** | 在执行空间、设备放置或硬件配置之间迁移状态，例如 CPU↔CUDA、CUDA↔HIP、改变 rank-to-device 绑定；可能涉及布局转换 | **延期，需另行设计和批准**。R0 不承诺换设备、换 space 或更换精度；跨布局字段可复制也不证明重启兼容。 |
| **R3：schema/算法版本迁移** | 通过显式迁移器读取不同 checkpoint schema、方法/模型或算法版本 | **延期，需另行设计和批准**。未知或不同版本直接拒绝；无自动迁移或“尽力恢复”。 |

等级不是继承关系或自动回退阶梯：符合 R0 不说明 R1/R2/R3 可用。即使 binary 内容相同，只要请求改变 rank/设备/模型等受约束条件，仍超出 R0。后续等级未立项，其名称只用于说明被拒绝的范围，不构成实施计划。

<a id="compatibility"></a>
## 3. R0 兼容条件与拒绝策略

首次加载先校验 manifest 和配置，再创建恢复候选；必须在任何 accepted 状态发布前完成全部校验。相同源码 commit 或文件名不足以证明“相同 binary”。

| 检查项 | R0 要求 | 不满足时 |
|---|---|---|
| 程序及依赖身份 | 可执行文件、实际加载的 Nexora 数值模块及依赖的制品摘要一致；记录 source commit/dirty 状态、Kokkos lock、编译/链接选项作溯源；动态库同名但 bytes 不同不等价 | 拒绝，不能仅用项目版本字符串放行 |
| 编译与数值配置 | 相同 compiler/stdlib、C++ 标准、Kokkos configuration、method/solver/integrator 版本、标量类型、舍入/fast-math 等数学模式、归约/确定性策略、线程数和执行调优配置 | 拒绝；不得转换精度、改容差或偷偷选其他实现 |
| MPI 与分区 | MPI 开关/实现/版本、communicator 逻辑成员与 rank 数、稳定 ID 对应的 partition/owner 映射一致；无需保存原始 MPI_Comm 值，新进程重建句柄 | rank 数、分区或 ownership 不同拒绝；不在 loader 内调用再分区 |
| CPU/GPU 放置 | 相同获验硬件与软件 profile（设备型号/架构、driver/toolkit、OS/runtime）、线程/NUMA 放置，以及 rank-to-device 绑定；GPU 记录可核验设备身份与映射，不能只靠 ordinal 相同判定 | 缺少可验证 profile 或发生设备替换/重映射拒绝；R0 不授权换设备测试 |
| case 和物理约定 | resolved case、模型参数、单位/坐标/压力约定、边界条件、积分器与容差、随机数算法及状态匹配；启动时使用 checkpoint 记录的 accepted 时间/步骤 | 不允许利用 restart 修改数值配置；唯一可不同的是明确定义为非数值的日志/输出路径，且不能改变载荷、控制或推进逻辑 |
| schema 与编码 | 当前程序所要求的 exact schema、字段 dtype/shape/association/units、整数宽度和浮点编码/字节序一致；必需字段和模型历史完整 | 缺字段、未知 schema、尺寸溢出、编码不支持、未声明的多余必需语义均拒绝；不默认填零或自动迁移 |
| 快照完整性 | 同一个 checkpoint ID、同一 accepted window/time/state version；完整文件列表、sizes/checksums、rank 清单与完成标记一致 | 临时目录、混合 generation、缺文件、截断、checksum 不符或有未完成发布均拒绝 |
| 实体与状态一致性 | 稳定 ID 唯一、连接范围合法、每实体 owner 唯一、所有参与者/coupling history 对齐；恢复所需实体集合、历史和时间层齐全 | 重复/丢失 ID、错误拓扑、混合 trial/accepted、非法时间层拒绝 |
| 外部引擎（如未来启用） | 另获批准的完整 engine、固定 solver/版本/配置，能恢复全部隐状态且通过同配置 replay 认证 | 仅有 pose/velocity、引擎无完整恢复或未认证时拒绝 R0；不能因端口能交换数组就承诺重启 |

R0 首个验收 profile 的具体硬件/driver/OS 版本仍待提供，上表不是假定任何机器已经获验。设备身份、绝对路径、PID 和内存地址用途不同：路径可重定位，PID/指针必须重建；不得因为路径变化拒绝相同制品，也不得因为地址相同接受不兼容状态。不放宽现有数值公式、默认值或容差。

<a id="checkpoint-content"></a>
## 4. 完整逻辑状态与持久格式边界

IO-001 必须明确序列化 schema 与 codec，保存以下逻辑内容。下面是要求清单，不是已实现的文件格式，也不预先选定 HDF5 或给它增加必需依赖：

| 类别 | 必需数据 |
|---|---|
| manifest / provenance | checkpoint/schema/run 标识、accepted 时间与窗口/步骤、rank/分区清单、程序与依赖摘要、数值配置/profile、各文件 sizes/checksums、发布完成状态 |
| 拓扑与物理场 | domain/association/entity-set、稳定 ID、connectivity、边界/patch、geometry、主状态、物理语义和必要版本；rank 内局部布局的可重建映射 |
| 方法历史 | SPH 积分/自适应历史、DEM contact key 与切向/滚动等模型历史；FVM 若所选方案要求的 face flux；FEM 材料历史；LBM populations/parity/掩码。只保存本次实际参与且已获验的方法，不因此新增方法实现。 |
| 控制与积分器 | 时间层/子步、历史导数、控制状态、随机数完整状态、适应性判据历史、影响轨迹的 warm-start；未来 MBD 还需 q/v、约束/接触/控制/engine hidden state |
| 耦合与分布式 | accepted 载荷历史/ledger、端口版本、ownership、恢复所需映射参数；未完成消息或外部请求不能成为恢复的隐式输入 |

可重建缓存必须列名、依赖版本和重建算法；若重建会改变已承诺的 replay 行为，则把相应数据升级为持久历史，不得仅称“缓存”省略。恢复重新分配 typed Kokkos Views、显式 copy、重建索引与权限；绝不序列化 View 对象、指针、引用计数、lease、stream、MPI request 或 C++ 对象镜像。所有外部会话/授权必须重建，新 run 的绑定身份隔离，旧句柄永远无效；物理逻辑版本按 checkpoint 恢复。

持久数值 payload 必须无损 round-trip（同 dtype/编码，不能通过低精度显示格式），整数/ID/时间层/状态计数精确恢复。R0 延续运行的误差比较继续使用对应获准 baseline 的原有判据；并行归约的既有非确定性不能靠新容差掩盖。只有基准已经要求逐位一致时才要求后续轨迹逐位一致，不能凭“相同 binary”自行新增该承诺。

<a id="checkpoint-publication"></a>
## 5. 发布、加载与失败行为

**写入：** 到达一致 accepted 边界 → 签发不可变快照并保活 → 完成所需 Kokkos copy 和相关通信 → 受内存预算约束的 writer 写临时数据 → 每份数据关闭且确认完成 → MPI 模式由全部参与 rank 作成功判定 → 最后发布完整 manifest。只发布全部成功的 generation；失败保留上一个有效恢复点，不能覆盖它。writer 没有消费完成前不能归还快照存储。

首个持久介质/codec 在 IO-001 实施设计中确认；本契约仅要求在选定且实测的存储语义下形成一致恢复点，不承诺对任意文件系统、断电或设备故障容错。若选同一文件系统 rename 发布，必须验证该存储的可见性与持久化条件；跨文件系统/对象存储发布不在首期范围。HDF5 是可选候选，版本、串行/并行模式和线程要求须另行审核，不能继承归档中的默认选择。

**读取：** 只接受已发布 manifest → 在不修改 active accepted 状态的情况下完成 header/制品/配置/文件完整性检查 → 在独立恢复候选中解码并重建 typed Views/索引 → 校验跨参与者/rank 一致性并完成必要拷贝 → 全部成功后一次发布 restored accepted 状态。任一失败不发布部分恢复、不悄悄回到初始条件继续运行；返回明确错误或终止本次重启。可以显式选择另一个已验证 checkpoint 再尝试，不能自动把失败称作恢复成功。

进程内 trial rollback 继续遵守 CKPT-001/participant 契约；IO-001 的 loader 不复用进程内 lease/event，不把设备异常转换成可恢复的数值失败。

<a id="restart-acceptance"></a>
## 6. IO-001 / R0 验收

| ID | 场景 | 通过条件 |
|---|---|---|
| CP-01 完整捕获 | 对实际参与的方法和耦合状态逐项建立保存/重建清单 | 所有影响后续演化的主状态/历史/控制均归类；不完整引擎明确拒绝能力 |
| CP-02 同配置继续 | 同一个已批准 workload 连续运行，与 accepted 中途写文件、退出进程、相同 binary/profile 新进程加载后继续相比 | 读取点无损；ID、owner、时间层/accepted 计数精确一致；最终数值与守恒量满足原有判据；不得调整容差 |
| CP-03 相同 rank | 串行 rank=1；仅在 MPI-001/002/003 的对应路径已通过后，分别测相同 2→2、4→4 的 profile | 每个组合单独报告；不把 1→2、2→4 当 R0；空 rank、contact history 和全局接受时间正确 |
| CP-04 不兼容拒绝 | binary/动态库摘要、Kokkos config、dtype、schema、case、rank 数、partition/owner、device binding/profile 任一不匹配 | 在发布恢复状态前拒绝；不隐式转换、不重新分区、不 fallback |
| CP-05 不完整恢复点 | writer 中断、缺 rank 文件、truncate/checksum 错误、混合 generation、丢历史/重复 ID、临时 manifest | 全部拒绝，上一完整 checkpoint 不受污染 |
| CP-06 加载及保活失败 | 解码/分配/复制/验证失败、输出积压、提前销毁 caller、加载后旧导出句柄再用 | 不发布部分状态；快照到最后消费者后才释放；背压受控；旧会话/lease 拒绝 |
| CP-07 等级越界 | 请求重新分区、换设备/space/精度、跨方法/schema 版本恢复，或外部 engine 未认证 | 明确报未支持/需另行批准；不由 R0 通过推导 R1/R2/R3 |
| CP-08 证据 | 固定输入/参考 hash、所有 binary/dependency/profile/checkpoint 摘要、命令、停止/重启点、日志及带时区秒级时间戳 | 配置、编译、执行与数值结论分开；未实施、无硬件或 skipped 不算通过 |

CP-01 主要由 CKPT-001 准备，CP-02–08 是 IO-001 的关闭条件。当前所有 CP 项均未实施/未执行；P1 单参与者回滚验证不关闭 IO-001，CPU 验收也不自动关闭 CUDA/MPI R0。
