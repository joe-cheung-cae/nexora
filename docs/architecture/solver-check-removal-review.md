# 求解器防御性检查移除审批清单

2026-11-26 已实施 A01–A11 和 B01–B16。求解器已解除对 `checked_integer_cast` 的依赖，公共 API 保留。下表记录基于 `da628e9` 工作树的审批依据，行号对应修改前的文件。

覆盖三维 WCSPH 和 Reference 的初始化、数值算子、步进、发布和读取，向上追到 Simulation，向下追到整数容量、拓扑、字段授权和资源预算。可独立调用的算子、Reference 结构重排和一维固定 CSR 验证夹具均标明了适用范围。CLI 仅记录 Reference 的输入和运行边界。以下编号可独立审批；同一谓词在不同入口有不同结论时，按入口注明。

## A　建议先移除的重复检查

| 编号 | 代码位置 | 拟移除内容与依据 |
|---|---|---|
| A01 | [EntityCapacity.hpp 第 10–14 行](../../src/data/EntityCapacity.hpp#L10) | `checked_entity_count` 已把无符号 64 位数量限制到 `LocalIndex`。在支持的 ABI 上，`size_t` 能表示全部 32 位局部索引。保留第一次拒绝，返回结果时直接转换。 |
| A02 | [EntityCapacity.hpp 第 16–22 行](../../src/data/EntityCapacity.hpp#L16) | `checked_local_index` 在原实现中先校验数量及 `index < count`，最后一次受检转换不可能失败。A04 迁移调用方后，后续清理删除了该私有函数。 |
| A03 | [EntityCapacity.hpp 第 24–30 行](../../src/data/EntityCapacity.hpp#L24) | `checked_array_bytes` 已要求元素字节数非零且 `count <= SIZE_MAX / element_size`。这也证明 `count` 能转成 `size_t`。移除后面的受检转换与错误转发，仍然在乘法前拒绝越界。此项与平台位宽无关。 |
| A04 | [Permutation.hpp 第 31、43–45、52–57 行](../../src/data/Permutation.hpp#L31) | `match_entity_ids` 两个循环都遵守 `i < 已校验的 count`，各删一次 `checked_local_index(i, count).value()`，直接转换循环索引。源 ID 唯一、目标 ID 存在且未重复仍需检查。 |
| A05 | [EntityOrder.cpp 第 97–103、149–151 行](../../src/data/EntityOrder.cpp#L97) | 外部排列的两个元素和 CSR 列号本来就是 `LocalIndex`，关联的 `EntityOrder` 数量在构造时已受限。三个调用分别换成与对应行数或列数的直接比较，保留外层原报错。 |
| A06 | [EntityOrder.cpp 第 144–148 行](../../src/data/EntityOrder.cpp#L144) | CSR 已固定最后一个 offset 等于条目数，并遍历所有相邻项验证非递减。中间项若越过末值，后面必有下降。可删逐项的 `offsets[i] > indices.size()`。两种失败在现代码中返回同一状态和消息，检查完成前也没有按 offset 访问条目。 |
| A07 | [CellGrid.cpp 第 262–266 行](../../src/methods/sph/neighbors/CellGrid.cpp#L262) · [State.cpp 第 78、93–99 行](../../src/methods/reference/State.cpp#L93) | 两处都先挡住超过 `UINT32_MAX` 的实体数。支持的 64 位 ABI 上，排序项、三分量坐标以及 Reference 的两个实数字段加一个历史标量都远小于 `SIZE_MAX`。可删第二道字节上限；保留实体上限、实际分配和额度检查。 |
| A08 | [Continuity3D.cpp 第 64、92–95、104–110 行](../../src/methods/sph/Continuity3D.cpp#L64) · [Continuity3D.hpp 第 33–46 行](../../src/methods/sph/Continuity3D.hpp#L33) | 延迟生成预测速度时，`write<false>` 写完三个分量后恒返回 true。`VelocityFailure::prediction`、产生它的条件分支和处理它的报错均可删。当前步进在需要壁速度时检查其可表示性，校正密度在发布前检查。 |
| A09 | [CellGrid.cpp 第 209–216、229 行](../../src/methods/sph/neighbors/CellGrid.cpp#L209) | 在 `try_reuse<true>` 中，质量字段沿用旧票据，构造的 `Result<Ticket>` 恒为成功。直接拷贝票据，仅对重新申请质量存储的分支检查失败。票据仍随新格网保活。 |
| A10 | [State.cpp 第 112–134 行](../../src/methods/reference/State.cpp#L112) · [Participant.cpp 第 172–178 行](../../src/participants/reference/Participant.cpp#L172) | 私有的 `State::initialize` 只会正常完成或抛异常，唯一显式返回值为成功。可改为 `void`，去掉调用者检查恒成功状态的分支。外围异常转换、失败清理和随后的字段签发仍需保留。 |
| A11 | [State.cpp 第 158–166、220–224 行](../../src/methods/reference/State.cpp#L158) · [Participant.cpp 第 194–198、269–279 行](../../src/participants/reference/Participant.cpp#L269) | Reference 的私有步前检查和快照函数各重查一次已初始化状态。两者都只能从持有 accepted 访问权的 participant 进入；初始化失败会撤销状态，成功关闭会释放访问权。可删这两条内部 phase 检查，使内部快照直接返回值。保留入口拒绝、时刻匹配、版本与控制计数器检查、快照预算和授权失败处理。 |

构建拒绝非 64 位目标；`LocalIndex` 为 `uint32_t`，`Offset` 为 `uint64_t`。涉及固定步幅的结论针对仓库实际支持的 ABI 和 float/double 两种配置。若扩大平台范围，可以把所需的位宽关系写成一个编译期约束。[构建入口](../../cmake/NexoraKokkos.cmake#L17) · [索引类型](../../include/nexora/core/Types.hpp#L45) · [支持配置](build-and-validation.md)

## B　可进一步移除，需明确适用范围或诊断变化

| 编号 | 位置 | 拟处理范围与代价 |
|---|---|---|
| B01 | [CheckedCast.hpp 第 7–25 行](../../include/nexora/core/CheckedCast.hpp#L7) | 这是已安装的 host API。生产代码仅在 `EntityCapacity.hpp` 第 11、14、22、27 行直接调用；均为无符号源。A01–A03 后只剩第 11 行真正可能失败。可在该处直接比较 `count` 和 `LocalIndex::max()`，使求解器脱离这个模板，同时保留它对外提供的受检转换。若要求连公共函数一起删除，安装消费者需迁移，负数转无符号和有符号窄化的拒绝契约也须另决。两个 `static_assert` 与 `if constexpr` 属于编译期限制或选择。[核心用例](../../tests/core/core_tests.cpp#L23) · [安装消费者](../../tests/package/consumer/main.cpp#L64) |
| B02 | [EntityCapacity.hpp 第 34–49 行](../../src/data/EntityCapacity.hpp#L34) | 先检查行数的 32 位上限已经足以保证随后 `rows + 1` 可用 64 位表示，offset 数组字节数也有界。去掉 `rows == UINT64_MAX` 特判会将该输入的消息从 CSR 哨兵溢出改为整数超界。条目数是独立的 64 位输入，其乘法及与 offsets 字节数相加的溢出检查仍有测试反例。 |
| B03 | [EntityOrder.cpp 第 101–109、141 行](../../src/data/EntityOrder.cpp#L101) · [Permutation.hpp 第 23–35 行](../../src/data/Permutation.hpp#L23) | 两张等长且值域合法的有限排列只需检查一个方向的复合为恒等。删另一方向会改变同时有多种错误时最先报告的诊断。两个命名空间各自校验再比较相等，也可在相等后只校验一次，但会改变非法且不相等时的消息。独立外部映射仍需边界及稳定 ID 对应校验。对已有 EntityOrder 再算排列字节数、行列数也可按 A07 的已限界数据处理；通用 CSR 接口仍接受未限界的条目数。 |
| B04 | [WcsphStep3D.cpp 第 126–149、316–325 行](../../src/participants/sph/WcsphStep3D.cpp#L126) · [CellGrid.cpp 第 262–267 行](../../src/methods/sph/neighbors/CellGrid.cpp#L262) | 流体数、各壁组之和先分别受 `size_t` 加法和最终 `LocalIndex` 上限保护，私有初始格网再检查总数。可在读取流体数组和分配壁记录前，以 `LocalIndex` 的剩余额度逐组检查，覆盖零壁和空壁组，再删除私有入口的重复总数检查。独立建网入口仍需限界。提前拒绝可能改变同时具有坏 ownership 与超量输入时的首个报错。 |
| B05 | [WcsphStep3D.cpp 第 69、138、288、464 行](../../src/participants/sph/WcsphStep3D.cpp#L69) · [WcsphParticipant3D.cpp 第 152 行](../../src/participants/sph/WcsphParticipant3D.cpp#L152) · [Geometry3D.cpp 第 12 行](../../src/participants/sph/Geometry3D.cpp#L12) | 已受限的粒子数、owned 行数乘固定步幅，包括初始几何副本、壁粒子记录、压力 scratch、积分缓冲、发布字段与行表，均可改成直接计算字节数。须逐个保留真正的预算申请、失败转发、票据寿命；第 69 行无壁副本沿另一条路径，要依赖此前首次建网成功的数量上限。 |
| B06 | [WcsphStep3D.cpp 第 408–415 行](../../src/participants/sph/WcsphStep3D.cpp#L408) · [Verlet3D.cpp 第 278–293 行](../../src/methods/sph/Verlet3D.cpp#L278) | `try_step` 的唯一现有调用者经 `ParticipantIF::step` 准入。原始 double 步长已为有限正值，float 的已发布 CFL 上限封顶于 `real::max()`，壁上限只会进一步收紧。可在明确这个私有步进前提后删第 411 行整条前置谓词。**保留第 414 行**，因为合法的小正 double 仍会转成 float 零。[准入入口](../../include/nexora/participant/ParticipantIF.hpp#L45) · [小步长用例](../../tests/methods/sph/verlet_window_tests.cpp#L610) |
| B07 | [CellGrid.cpp 第 257、287、349–351 行](../../src/methods/sph/neighbors/CellGrid.cpp#L349) | 仅更新位置时，粒子数已对过，owned 数直接取自旧格网；复用条件中这两项可省。外部更新可以改变所有权或数量，保留其快速拒绝。相同形状推导 `cell_begin` 长度还依赖旧格网完整且空格网保有一个哨兵；要删第 351 行，需先界定已移走对象的可调用状态。 |
| B08 | [Verlet 漂移](../../src/methods/sph/Verlet3D.cpp) · [数值步](../../src/participants/sph/WcsphStep3D.cpp) | 已实施。半步速度与漂移位置写入候选缓冲，格网在位置转为整数之前拒绝无效值。 |
| B09 | [Pressure3D.cpp 第 21–37 行](../../src/methods/sph/Pressure3D.cpp#L21) · [PrescribedMovingWall3D.cpp 第 165–169、214–219 行](../../src/methods/sph/PrescribedMovingWall3D.cpp#L165) | ghost 压力先将相同类型的密度写入临时值；可直接对原值检查正且有限，再求 EOS。独立壁面采样及其坐标包装已删除。生产漂移位置通过格网准入检查。 |
| B10 | [WcsphParticipant3D.cpp 第 159–215、325–340 行](../../src/participants/sph/WcsphParticipant3D.cpp#L159) · [WcsphStep3D.cpp 第 271–275 行](../../src/participants/sph/WcsphStep3D.cpp#L271) | 字段签发闭包目前只收到三种内部枚举常量。已提交的 ghost 也必来自当前持有的壁组。可通过穷举派发或让一次查找同时返回粒子与所属组，消除未知内部组的失败状态。直接删除兜底会让函数失去有效返回值；外部传入未知壁 ID 仍需拒绝。 |
| B11 | [WcsphStep3D.cpp 第 88–95、167–177 行](../../src/participants/sph/WcsphStep3D.cpp#L88) · [WcsphParticipant3D.cpp 第 152–157 行](../../src/participants/sph/WcsphParticipant3D.cpp#L152) | 这几处检查 `!result.ok()` 后又检查 `resource_exhausted`。当前具体的 `ResourceBudget::reserve` 只有这一种失败码。可仅在绑定该具体函数的路径合并谓词；泛型回调如继续允许别的失败，保留按错误码决定是否重试的语义。 |
| B12 | [CellGrid.cpp 第 247 行](../../src/methods/sph/neighbors/CellGrid.cpp#L247) · [State.cpp 第 106 行](../../src/methods/reference/State.cpp#L106) · [WcsphParticipant3D.cpp 第 233 行](../../src/participants/sph/WcsphParticipant3D.cpp#L233) | 锁定的 Kokkos 5.2.2 初始化后为 initialized，正常 finalize 清除 initialized 并设置 finalized，且拒绝重复初始化。在该生命周期中，`!is_initialized()\|\|is_finalized()` 可只留前半。相同表达式共 **11 处**，另有 [EntityOrder.cpp 第 15、90、139 行](../../src/data/EntityOrder.cpp#L15)、[FieldAccess.cpp 第 187 行](../../src/data/FieldAccess.cpp#L187)、[Permutation.hpp 第 23 行](../../src/data/Permutation.hpp#L23)、[TransportBuilder.hpp 第 41 行](../../src/data/TransportBuilder.hpp#L41)、[TransportReceiver.hpp 第 19 行](../../src/data/TransportReceiver.hpp#L19)、[固定 CSR 夹具第 213 行](../../validation/sph/fixed_csr/Participant.cpp#L213)。版本依据为本地锁定源码 `core/src/impl/Kokkos_Core.cpp` 第 749、785–786、1024–1045、1124–1128 行。 |
| B13 | [TransportBuilder.hpp 第 72–101、111–117 行](../../src/data/TransportBuilder.hpp#L72) · [TransportReceiver.hpp 第 83–117 行](../../src/data/TransportReceiver.hpp#L83) | 当前本机载荷由私有构造器产生，打包时令 ID 数与值行数相等，值列数取描述符的分量数。接收方已经核对描述符，可将源端这两项形状复查缩为一次源行数与目标实体数比较。有效的一维 ID lease 的实体轴还蕴含 `components == 1`。保留与目标 ID 长度、单位、命名空间、目标轴、所有权、版本和稳定 ID 双射有关的独立比较。 |
| B14 | [固定 CSR 的 Density.cpp 第 23–28 行](../../validation/sph/fixed_csr/Density.cpp#L23) · [Participant.cpp 第 53–56、223、271–285、323–350 行](../../validation/sph/fixed_csr/Participant.cpp#L271) | 仅限一维验证夹具。现有唯一算子调用从已验证 CSR、同一行数分配的输出和已验证等长的源字段组装输入，可去掉算子内三项形状复查。已限界的行列乘固定步幅也可直接计算。保留条目容量和组合额度。重排时若删入口基数检查而等到 `Permutation::between` 才报错，额度或分配失败可能先发生。 |
| B15 | [StructureState.cpp 第 166–169、208–213 行](../../tests/validation/structure/StructureState.cpp#L166) · [固定 CSR 夹具第 399 行](../../validation/sph/fixed_csr/Participant.cpp#L399) | 这些重排从有效 CSR 和双射产生新图，再经过通用构造器复验。同一结构夹具替换 q 时也复制并复验既有实体顺序。若跳过，须先解决只读顺序的分配授权，以及新旧快照共存问题；通用 CSR 构造器继续接受外部表，不能直接去掉其入口检查。 |
| B16 | [FieldStamp.hpp 第 25–31 行](../../include/nexora/data/FieldStamp.hpp#L25) | 有限时刻不可能落入倒置区间。若集中为一次成员关系检查，可省 `valid_until < valid_from` 的单独判定，但倒置区间将收到另一条具体错误消息。仍需拒绝非有限时刻与端点。 |

## B05–B07、B09–B14 执行记录

2026-11-26 已按批准范围实施九项清理。SPH 对已限界的粒子数和 owned 行数直接计算固定步幅字节数，实际预算申请和失败回滚仍由原路径处理。私有步进采用入口已验证的原始步长，缩窄后仍检查正值。位置更新复用格网时沿用粒子数和 owned 数，外部更新继续校验两者。

ghost 压力直接校验原密度。已提交壁粒子的时间只需查找运动壁组，外部未知 ID 仍返回错误。具体预算申请失败时根据可驱逐资源决定是否重试，泛型回调继续区分失败码。本地传输使用私有打包器已保证的源端形状，固定 CSR 密度核使用已验证的行列尺寸；条目容量、组合额度和重排基数仍在入口校验。Kokkos 活动状态检查统一使用 `is_initialized()`。

连续性预测的声明和生产调用统一为 `void write(...)`。测试从已提交状态核对邻居、密度变化率及失败后的回滚。24³ 水箱在 CTest 和 Serial 验证入口均关闭，待引入高精度 SPH 算法后恢复。

两套 Kokkos 5.2.2 Serial Release 暂存构建于 `2026-11-26T14:45:23+08:00` 完成首轮 CTest，float 185/185、double 183/183，均无跳过项。安装消费者、Reference 独立精度对照和依赖检查均通过。另行运行的结构重排计量暴露了清理前即可复现的旧预期。暂存夹具的分配按创建 6、打包 2、重排 10、接收 4、提交 4 计，共 26 次；保留故障注入位置的显式 fence 后共 13 次。已按这条路径同步测试断言。

## 执行前的范围核对

2026-11-26 对照当时的 Git 暂存区、工作树和新增的结构测试夹具复核。B08 已由工作树覆盖，下表记录随后获批的九个编号及其边界。

| 编号 | 基于当前源码的执行范围 |
|---|---|
| B05 | 几何版本比较夹具已删除。只处理初始几何、壁粒子记录、压力和积分缓冲、发布字段与 owned 行表中已受粒子数上限约束的固定步幅乘法。保留实际预算申请、驱逐重试，以及独立于粒子数的壁组和回调字节数检查。[分配点](../../src/participants/sph/WcsphStep3D.cpp#L69) |
| B06、B07 | 私有步进只移除调用方已保证的原始步长检查，保留转成 `real` 后的正值检查。位置更新只省去复用时的粒子数和 owned 数比较，保留外部更新检查及空格网哨兵。[步长](../../src/participants/sph/WcsphStep3D.cpp#L398) · [格网复用](../../src/methods/sph/neighbors/CellGrid.cpp#L349) |
| B09 | 独立壁面采样及 `checked_position` 已删除。剩余工作是让 ghost 压力直接校验原密度再求 EOS，保留拒绝非正、非有限密度的行为。[ghost 压力](../../src/methods/sph/Pressure3D.cpp#L30) |
| B10、B11 | 简化内部字段分派和已提交壁粒子的所属组查找，外部未知壁 ID 继续报错。仅对直接调用 `ResourceBudget::reserve` 的路径合并失败判断，泛型回调仍按错误码决定重试。[字段签发](../../src/participants/sph/WcsphParticipant3D.cpp#L152) · [预算重试](../../src/participants/sph/WcsphStep3D.cpp#L167) |
| B12 | 原列 11 处，删除 `Permutation::create` 后剩 10 处。按锁定的 Kokkos 生命周期简化这 10 处双重判断，保留各入口自己的状态和授权检查。[实体与 CSR 入口](../../src/data/EntityOrder.cpp#L75) |
| B13、B14 | 本地传输仅合并私有载荷构造已经保证的源端形状检查。固定 CSR 夹具仅清理内部密度算子的三项形状复查及已限界的固定步幅乘法。目标长度、稳定 ID、独立条目容量、组合预算和重排入口检查仍有各自的失败用例。[传输接收](../../src/data/TransportReceiver.hpp#L101) · [夹具密度算子](../../validation/sph/fixed_csr/Density.cpp#L23) |

B08 的 `kick_drift` 已返回 `void`，生产调用直接写入候选缓冲，独立 `kick_drift_3d` 继续检查半步速度及位置溢出。另列的 `continuity_rate_3d` 和预测速度的测试观测分支也已从工作树删除。接下来验证参与者的弹道轨迹、半踢溢出回滚和两阶段非有限变化率的密度校正。[漂移调用](../../src/participants/sph/WcsphStep3D.cpp#L468) · [回归用例](../../tests/methods/sph/continuity_tests.cpp#L32)

复核时暂存的连续性头文件已声明非模板 `write`，暂存调用仍使用 `write<false>` 和 `write<true>`，部分暂存测试仍调用旧的独立连续性接口。后续纳入这批改动时需一并核对声明、实现、调用方和测试，从同一暂存快照重建 float 与 double 并运行回归。24³ 水箱验收继续暂停。

当时延期的 B02–B04、B15、B16 已获批准，见下文执行记录。

## 其余防御性操作及移除影响

这些位置也检查坏输入或失败。表中列出各组实际建立的条件，供决定是否另行改变接口或数值行为。一般的循环终止、物理分支和单纯转发下层失败没有单独编号。

| 层或路径 | 检查位置 | 直接去掉的后果 |
|---|---|---|
| WCSPH 初态、壁输入 | [WcsphStep3D.cpp 第 47–49、126–163、310–331 行](../../src/participants/sph/WcsphStep3D.cpp#L310)；[SphBoundary.cpp 第 14–27、44–47、58–86 行](../../src/methods/sph/SphBoundary.cpp#L44) | 非空空指针、异长字段、带 ghost 的流体、重复壁组或粒子 ID、非法质量或运动壁参考点可进入排序、用户运动回调或数值核。参与者在壁组装配和建格时验证外部输入。 |
| 格网、分箱与邻居 | [CellGrid.cpp 第 27、89–93、112–200、247–278、292–345、349–357、390–400 行](../../src/methods/sph/neighbors/CellGrid.cpp#L247)；[NeighborView3D.hpp 第 53–75 行](../../src/methods/sph/neighbors/NeighborView3D.hpp#L53) | 外部输入、位置更新与已提交格网重建的前提不同。保留质量和 ghost 值、更新形状、ID 唯一、浮点坐标转整数前的范围、实际平方支持半径、稠密格数乘积及总预算检查。有限正 h 的平方仍可能上溢或下溢。格边界修正、自身排除和截断距离决定邻居 ID。 |
| 密度、压力、固定壁 | [Density3D.cpp 第 18–54 行](../../src/methods/sph/Density3D.cpp#L18)、[Density3D.hpp 第 12–35 行](../../src/methods/sph/Density3D.hpp#L12)、[Pressure3D.cpp 第 14–36、52、71–83、128–145、146–216、243–246 行](../../src/methods/sph/Pressure3D.cpp#L71)、[AdamiWallPressure3D.cpp 第 13–39 行](../../src/methods/sph/AdamiWallPressure3D.cpp#L13)、[TaitEos.cpp 第 8–13、24–36 行](../../src/methods/sph/TaitEos.cpp#L8) | 外部初始密度、重构鬼粒子密度、EOS 压力、Adami 逆算密度和最终加速度是不同的新数值。去掉各自最后的检查可暴露零密度或非有限结果。近参考密度的 `log1p/expm1` 和零距离、截断点分支会影响物理结果。 |
| 连续性和 Verlet | [连续性校正](../../src/methods/sph/Continuity3D.cpp) · [Verlet](../../src/methods/sph/Verlet3D.cpp) | 保留步长准入、壁速度、校正密度、终速和 CFL 检查。即使每个分量有限，速度平方和仍可能溢出。 |
| 运动壁时间与回调 | [PrescribedMovingWall3D.hpp 第 64–66 行](../../src/methods/sph/PrescribedMovingWall3D.hpp#L64)；[PrescribedMovingWall3D.cpp 第 22–29、38–89、98–127、165–188、206–219 行](../../src/methods/sph/PrescribedMovingWall3D.cpp#L38)；[WcsphStep3D.cpp 第 197–208、217–230、252–268、418–432 行](../../src/participants/sph/WcsphStep3D.cpp#L418) | 保留外部回调存在性和异常转换、有效域、实际可分辨的时长、真实使用的运动学分量及有限端点。否则可能越域采样、重复调用回调、提前拒绝合法端点或将无效壁速度传给流体。仅有平移时的速度缓存和旋转轴归一化也有可观测用途。 |
| WCSPH 权限、预算和原子发布 | [WcsphParticipant3D.cpp 第 145–215、220–261、265–298、301–346 行](../../src/participants/sph/WcsphParticipant3D.cpp#L265)；[WcsphStep3D.cpp 第 139–151、167–198、283–293、434–487、512–515、556–568、604–615 行](../../src/participants/sph/WcsphStep3D.cpp#L604)；[ResourceBudget.hpp 第 35–53、70–77 行](../../src/participants/ResourceBudget.hpp#L35) | 壁组数可以因空组而超过壁粒子数，回调持有字节数也独立于粒子数；相关乘法与累加不能套用 B05。有限预算、驱逐重试、字段授予、相位/版本、读回调重入、失败后的临时缓冲清理均可触发。删除会改变最后一次成功状态、可读快照或剩余额度。 |
| Reference 数值和读取 | [State.cpp 第 21–37、65–108、158–185、220–223 行](../../src/methods/reference/State.cpp#L65)；[Participant.cpp 第 81–101、107–143、153–186、194–247、255–299 行](../../src/participants/reference/Participant.cpp#L107) | 外部输入须非空、等长、ID 唯一、数值有限。步长要同时满足 double 调度和 real 积分的稳定性与可表示性。已接受时间匹配、版本和控制计数器未耗尽、候选 q 有限、预算与字段签发成功都是独立条件。直接 initialize 与 case 预检共享有缓存的校验，正常路径只执行一次。 |
| Reference 结构、固定 CSR 夹具 | [StructureState.cpp 第 53–86、117–175、189–213 行](../../tests/validation/structure/StructureState.cpp#L53)；[固定 CSR Participant.cpp 第 44–56、109–151、186–188、207–238、259–285、306–350、399–429 行](../../validation/sph/fixed_csr/Participant.cpp#L109)；[Density.cpp 第 46–49 行](../../validation/sph/fixed_csr/Density.cpp#L46) | 保留外部数据长度、有限性、元数据、稳定 ID、重排及版本验证；一维图还需验证 owned 行覆盖、自边、重复边、支持域和边的互反性。默认空快照有既定读取语义。固定图密度归约可溢出。结构夹具已迁入测试树并移除故障注入，回滚验证使用实际输入拒绝及版本耗尽路径。 |
| 时间、端口、Simulation | [TimeConstraint.hpp 第 24–58 行](../../include/nexora/participant/TimeConstraint.hpp#L24)、[ParticipantIF.hpp 第 45–53 行](../../include/nexora/participant/ParticipantIF.hpp#L45)、[Port.hpp 第 18–23 行](../../include/nexora/participant/Port.hpp#L18)、[Simulation.cpp 第 36–56、70–86、94–102 行](../../src/simulation/Simulation.cpp#L36) | 已耗尽时间域、无效约束、非法固定量子、不能前进的舍入端点有不同诊断。端口来自可独立实现的 participant；运行阶段、重入和已接受步数的保护确保先成功发布，再推进时钟。 |
| 实体拓扑、版本与错误值 | [EntityCapacity.hpp 第 10–60 行](../../src/data/EntityCapacity.hpp#L10)、[EntityOrder.hpp 第 31–54 行](../../src/data/EntityOrder.hpp#L31)、[EntityOrder.cpp 第 6–43、53–76、90–109、139–151、169–172 行](../../src/data/EntityOrder.cpp#L90)、[Permutation.hpp 第 18–57 行](../../src/data/Permutation.hpp#L18)、[Result.hpp 第 10–24 行](../../include/nexora/core/Result.hpp#L10) | 首次窄化、命名空间、数量、双射、稳定 ID、CSR 首尾与单调性、分配失败和版本耗尽均有外部触发点。`Result` 拒绝用成功状态构造失败值，空值访问另有既定行为。A/B 项只省去已由这些前提推出的部分。 |
| 字段、外部内存和本地传输 | [FieldDescriptor.hpp 第 48–65 行](../../include/nexora/data/FieldDescriptor.hpp#L48)、[FieldRegistry.hpp 第 8–14 行](../../include/nexora/data/FieldRegistry.hpp#L8)、[FieldIssuer.hpp 第 48–59、94–138、188–194、228–249 行](../../src/data/FieldIssuer.hpp#L124)、[FieldAccess.cpp 第 82–94、115–165、194–228、236–253、263–304 行](../../src/data/FieldAccess.cpp#L236)、[ExternalField.hpp 第 19–47 行](../../src/data/ExternalField.hpp#L19)、[FieldBinding.hpp 第 39、91 行](../../include/nexora/data/FieldBinding.hpp#L39)、[TransportView.hpp 第 96–99 行](../../include/nexora/data/TransportView.hpp#L96)、[TransportBuilder.hpp 第 25–54、72–108 行](../../src/data/TransportBuilder.hpp#L25)、[TransportReceiver.hpp 第 19–36、74–120 行](../../src/data/TransportReceiver.hpp#L74) | 保留实际 owner、非活动或已移走的 lease/request、整分配互斥、存储 incarnation、字段类型与版本、视图 stride/对齐/容量、打包总字节数、跨字段形状与 ID 检查。空数组也可能携带错误元数据；移动后的请求仍可能显示 `Result::ok()` 但已失去 authority。 |
| CLI 预检与运行边界 | [ReferenceCase.cpp 第 46–76、93–109、122–179、182–214、223–235 行](../../apps/nexora/ReferenceCase.cpp#L140)；[ReferenceCommand.cpp 第 73–88、113–133 行](../../apps/nexora/ReferenceCommand.cpp#L73) | 解析有符号文本、拒绝非法数值与 case 版本、限制计划长度、检测尾步和舍入停滞，服务于可独立执行的 validate/plan。移除容量预估可能延后到逐步枚举才失败，改变诊断优先次序或资源成本。运行后还核对实际端点并处理 finalize 失败。 |

## B02–B04、B08、B15、B16 执行记录

CSR 原始行列数在入口限界，行数等于 `UINT64_MAX` 时返回 `invalid_argument` 和 `integer conversion out of range`。已构造的实体顺序直接用于计算 offsets 字节数；独立条目乘法与合计字节数仍检查溢出。排列按稳定 ID 生成，命名空间相等时验证一次，错误仍按源、目标、匹配的顺序返回。

WCSPH 在流体字段形状验证后、读取 ghost 标记前，按局部索引上限接纳流体和各壁组。超量返回 `resource_exhausted` 和 `WCSPH source count exceeds capacity`。独立建网入口继续检查粒子容量。生产漂移直接写候选缓冲，格网拒绝非有限位置并回滚；独立 `kick_drift_3d` 继续在写出结果时报告半步速度或位置溢出。

结构夹具与固定 CSR 夹具从已验证的行列双射生成只读邻接，生成的 offsets 和 indices 直接归入新快照。只替换字段时独立复制已验证的 ID 顺序，签发新读权，并沿用原有只读 CSR。重排或替换失败时，原快照继续可读。有限时刻和端点统一按闭区间判断，倒置区间返回 `field time is outside its validity interval`。

## 审批后的验证

1. 整数与拓扑改动先覆盖负转无符号、两侧窄化、最大有效索引、空顺序、坏排列、倒退的中间 offset、条目乘法与总字节数溢出。公共转换若有改动，同时验证安装消费者。[核心测试](../../tests/core/core_tests.cpp#L23) · [实体与拓扑测试](../../tests/data/entity_topology_tests.cpp#L230) · [安装消费者](../../tests/package/consumer/main.cpp#L64)
2. WCSPH 分别在 float 和 double Serial 下核对格数、逐行邻居 ID、压力和密度、步长下溢、候选失败后的已提交行、版本、时钟及预算。24³ 水箱物理验收暂缓，待引入 SPH 高精度算法后在 float Release 恢复。[格网测试](../../tests/methods/sph/cell_grid_tests.cpp#L507) · [连续性测试](../../tests/methods/sph/continuity_tests.cpp#L130) · [压力测试](../../tests/methods/sph/pressure_3d_tests.cpp#L196) · [步进与回滚](../../tests/methods/sph/verlet_window_tests.cpp#L565) · [仓库验证入口](../../CONTRIBUTING.md)
3. Reference、公共字段和夹具按批准范围复核关闭、失败再试、快照跨步保活、用尽额度和版本、重排后 ID 与字段对齐、移动后的句柄和 CLI 错误消息。[Reference 资源测试](../../tests/reference/reference_resource_tests.cpp#L78) · [字段请求测试](../../tests/data/field_group_tests.cpp#L116) · [固定 CSR 测试](../../tests/methods/sph/sph_tests.cpp#L182) · [CLI 套件](../../tests/package/RunCliSuite.cmake)

审批前独立枚举了 2,000 组小型 CSR offsets 和 66,282 对小型等长映射，分别核对 A06 与 B03 的接受或拒绝判定。实施 A06 后，在既有 CSR 用例中补入 `{0, 2, 3, 6, 5}`，验证中间 offset 超过末值、最后一项才下降时仍返回 `invalid_argument`。[回归用例](../../tests/data/entity_topology_tests.cpp#L174)

### 执行结果

macOS 27.2.1 arm64，Apple Clang 21.0.0，C++20，CMake 4.4.3，Ninja 1.13.2，Kokkos 5.2.2 Serial。以下配置均重新编译并执行完整 CTest，无跳过项。各配置的依赖检查、Reference 的 80 位 Decimal 独立对照和安装后消费者均通过。另核对了 Ninja 记录的头文件依赖，float Debug 和 double Release 的求解器编译单元均已脱离 `CheckedCast.hpp`。

| 配置 | CTest 通过数 | 完成时间 |
|---|---:|---|
| float Debug | 183/183 | `2026-11-26T14:08:00+08:00` |
| double Release | 181/181 | `2026-11-26T14:07:54+08:00` |
| float Release | 183/184 | `2026-11-26T14:09:30+08:00` |

float Release 唯一失败项是 24³ 静水箱。终点箱内压力 RMS 误差为 **6,714.27 Pa**，门限为 2,943 Pa；全程 RMS 速度峰值为 **0.312283 m/s**，门限为 0.0767203 m/s。用本次修改前的暂存源码、相同 Release 编译选项和同一 Kokkos 安装重建对照程序，两版均走完 2,404 步，并触发相同的两项超限。`steps.csv`、`frames.csv`、`profiles.csv` 和 `crossings.csv` 逐字节一致。两份 `steps.csv` 的 SHA-256 均为 `9e84e48320ae5ec28ea45c860f26bb5347d757a9fddb5708e77d780a67a55842`。[水箱验收入口](../../tests/methods/sph/HydrostaticTankAcceptance.cmake) · [既有水箱验证记录](../../validation/sph/hydrostatic-continuity-repair.md)

### 工作树复核

逐项复核 A01–A11 和 B01 时补齐了 A08。当时将延迟预测改为返回 `void`。此后删除步进观测及其单独的预测分支，生产路径按需写入预测值，壁速度回调仍可报告失败。两种精度的 Ninja 编译依赖均确认求解器已脱离 `CheckedCast.hpp`，安装后的公共头与源码逐字节一致。[预测速度实现](../../src/methods/sph/Continuity3D.hpp#L33) · [壁速度回调](../../src/methods/sph/Continuity3D.cpp#L101)

在上述 macOS arm64、Apple Clang 与 Kokkos Serial 环境中重新构建后，CTest 于 `2026-11-26T14:16:12+08:00` 完成。float Release 通过 183/183，double Release 通过 181/181，均包含安装消费者测试，Reference 独立精度对照也均通过。两套配置均未注册 24³ 水箱物理验收，待引入 SPH 高精度算法后恢复。[水箱测试开关](../../tests/methods/sph/CMakeLists.txt#L22)
