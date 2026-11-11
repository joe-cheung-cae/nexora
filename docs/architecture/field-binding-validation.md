# 通用字段绑定与 CPU 读写 lease

复验区间为 `2026-11-06T18:13:28+08:00` 至 `2026-11-06T18:13:42+08:00`。本次交付 P1-03/04 的 managed Kokkos Serial 子集。

## 数据与访问契约

- `nexora::data_schema` 只依赖 core，包含字段描述符、注册表和 `FieldStamp`。字段名、单位、association 等语义留在描述符，版本和有效时间移到 stamp，内存位置、extent 和 stride 由原生 View 提供。原来从描述符读取有效时间或版本的调用方改读 stamp。参考快照保留原有数值生命周期，现通过 `stamp()` 提供相同的时间和版本。
- `nexora::data` 公开传递 Kokkos。`ReadField` 暴露元素为 const 的 View，`WriteField` 暴露原生可写 View。私有签发器校验 association、单位、dtype、rank、各轴 extent、分量数、stamp 和存储归属。分量轴由调用方声明，支持 rank-1、rank-2、rank-3、空字段和常见有填充或非连续的 subview。检查 stride 按递增顺序能否证明无重叠，并检查对齐、可触及字节范围和 LocalIndex 容量。无法证明不重叠的特殊布局会被拒绝。
- 签发器使用锁定版本 Kokkos 的 managed allocation record 区分分配。跨字段及互不重叠的 subview 仍共享整分配的权限；多读者可共存，overwrite 和 accumulate 都必须独占。accepted 或由 const View 注册的分配只读。权限计数独立于 Kokkos 引用计数；allocation tracker 只用于识别别名和保活，Kokkos 的内部类型限制在私有实现内。
- 每次注册生成单调递增的 incarnation。关闭后停止签发，已有 grant 继续保活。签发器及最后一个 grant 销毁后才能重新注册同一分配，重新注册会产生新的 incarnation。调用方持有 grant 直到 Serial 访问完成，复制出的原生 View 不延长访问授权。签发器、grant 和 retained Views 均应先于 Kokkos finalize 销毁。

本次验证限于 managed View。后续 CPU 外部 owner、unmanaged 借用和本地 `TransportView` 的实现与验收见 [外部内存与本地传输](external-transport-validation.md)。异步完成、整链多分配授权、实体重排、持久 checkpoint 超出本次记录的范围。后续参考窗口接入见 [reference 窗口集成](reference-window-validation.md)。

Context7 的 View allocation identity 查询未命中，执行空间查询返回 develop 资料。具体身份与保活以本地锁定源码和实测为准。 [Kokkos View](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/core/src/Kokkos_View.hpp) · [Kokkos allocation tracker](https://github.com/kokkos/kokkos/blob/8e15454876298832f013ebcaf08d53825b5bda40/core/src/impl/Kokkos_SharedAlloc.hpp)

## CPU 验证

使用本地 Kokkos 5.2.2 Serial-only Release 包，commit `8e15454876298832f013ebcaf08d53825b5bda40`。本次重新比对 `libkokkoscore.a` SHA-256，与 [CPU 基础切片的制品记录](kokkos-cpu-slice-validation.md#依赖与机器配置) 一致。实际环境为 arm64、macOS 27.2 build 26B123、SDK 27.2、AppleClang 21.0.0.21000401、CMake 4.4.3、Ninja 1.13.2。Debug 和 Release 使用同一个 Release Kokkos 库。

两个 profile 分别使用新构建目录，依次配置、编译、执行 CTest 和依赖检查，全部退出码为 0。本轮使用已预供给的本地依赖，未启用额外的禁网沙箱。逐命令日志保存在本地忽略目录 `results/field-binding/`。

| 配置 | CTest | 依赖及安装包检查 |
|---|---|---|
| CPU Debug | `2026-11-06T18:13:31+08:00`–`2026-11-06T18:13:41+08:00`，13 通过，0 失败或跳过 | 目标图和头边界通过，搬迁后的安装消费 3/3 通过 |
| CPU Release | `2026-11-06T18:13:32+08:00`–`2026-11-06T18:13:42+08:00`，13 通过，0 失败或跳过 | 目标图和头边界通过，搬迁后的安装消费 3/3 通过 |

新增测试实际执行 Serial overwrite、accumulate 和只读归约，逐项验证租约冲突和精确结果。测试还覆盖不同布局、managed 空分配、不同字段及 disjoint subview 的同分配仲裁、独立分配、旧 stamp、只读分配、无运行时、unmanaged 拒绝、移动赋值和关闭后的保活。安装消费测试验证 schema 和 typed data 的导出边、私有签发器安装边界、schema 及 Kokkos 消费者的构建运行，以及既有 13 个错误配置。参考数值测试继续使用原公式和容差。
