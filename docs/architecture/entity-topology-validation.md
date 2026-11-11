# Serial 实体顺序与 CSR 验证

本记录对应 [开发计划任务 2](foundation-development-plan.md#任务-2-实体顺序-双射和-csr-基础) 和 `P1-03` 的私有 `data` 实现。语义及版本变更条件见 [Kokkos 设计](kokkos-design.md#3-稳定-id局部顺序与版本)。

## 所有权与失效

| 对象 | 持有内容和校验 |
| --- | --- |
| `EntityOrder` | `DomainId`、`Association`、`entity_set`、`LayoutVersion` 和独立的 rank-1 const ID View。`GlobalEntityId` 在主机边界解包为 `uint64_t`。相同命名空间内的 ID 唯一。空顺序仍校验命名空间。 |
| `Permutation` | `new_to_old` 和 `old_to_new` 为 const `LocalIndex` Views，并保活源和目标顺序。局部索引随布局失效。按稳定 ID 验证两张表的值、方向及互逆关系。 |
| `CsrAdjacency` | 保活独立的行和列顺序，以 const `Offset` 和 `LocalIndex` Views 保存关系。消费前可检查 topology、两侧 layout 和顺序。重复边、自环、行内顺序保留原样。 |
| 本地 transport | 发送快照和目标 ID read lease 在整个预检期间保活。接收仍独立检查字段语义、shape、时效、epoch 和授权，完成共享 ID 双射校验后才向 trial 或 ghost 的写 lease 解包。 |

这些对象由调用方持有。旧快照在保活期间可以继续读取。排列和邻接的 `validate_for` 会拒绝变化后的布局。已删除 ID 的退役记录及发布新布局由状态所有者维护。Reference 的 accepted/trial 版本、恢复和 storage incarnation 沿用现有生命周期，状态级原子重排属于任务 3。

## 夹具

- `nexora_entity_topology_tests` 通过写死的顺序、双向映射、字段期望和稳定 ID 边表比较原序及重排夹具。覆盖分域 CSR、空图、空行、长行、只读快照、ID 不匹配、错链、越界、旧版本和计数溢出。容量负例使用计数参数。
- `nexora_transport_receive_tests` 和 `nexora_reference_transport_tests` 覆盖接收写入前拒绝、租约、跨布局 ID 对齐、trial/ghost 解包及实际窗口的失败恢复和重试。
- 复用 `nexora_field_binding_tests` 的 rank-1、两种 rank-2 轴序、`LayoutStride` subview、rank-3 和空字段用例。安装搬迁消费检查新增头仍为 `data` 私有。

## 集成结果

`2026-11-11T11:30:55+08:00` 至 `2026-11-11T11:31:42+08:00`，在 Apple M2 Pro arm64 上用 AppleClang 和 exact Kokkos 5.2.2 Serial 跑完 float/double × Debug/Release。每组 CTest 16/16、独立消费者 3/3、依赖图和 Decimal oracle 均通过。验证在禁网环境中完成。工具链、命令、负例及日志见 [Serial CI 验证](serial-ci-validation.md)。
