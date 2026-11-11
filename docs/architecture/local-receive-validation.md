# CPU 本地接收与解包

私有 `TransportReceiver` 接收已经准备完成的 `TransportView`、目标 `WriteField` 和目标稳定 ID 的 `ReadField`，复用三方已有的生命周期与访问权限。调用方提供计划中的完整源 stamp，并在本地同步调用期间持有这三个对象。成功和失败都由调用方决定何时结束快照和归还 lease。

- halo 只写声明为 ghost 的区域。coupling 可写 ghost 或带有效试算标记的未接受区域。接收要求 overwrite 权限，拒绝 visualization、checkpoint、accumulate 和 accepted 目标。
- 源 stamp 必须与计划完全匹配，包括源 layout 和 storage incarnation。源和目标必须同拓扑、几何、accepted 版本、试算 epoch 及逻辑时间，但各自可采用不同的本地 layout 和存储。目标 ID 绑定与目标值的 stamp 除各自 storage incarnation 外必须一致。
- 要求同一逻辑字段的语义、dtype、单位、分量数和实体命名空间。目标可绑定 rank-1 标量或 rank-2 实体/分量字段，轴顺序沿用签发 lease 时确认的角色。源和目标区域必须拥有同一组稳定 ID。调用方可通过绑定 subview 指定更小的目标区域，区域之外的值不参与接收。
- 解包先验证形状和双方 ID 的唯一性及完整性，建立源行到目标行的双射。成功后用同一 Serial 实例覆盖目标并 fence。校验失败或映射准备失败发生在第一笔目标写入之前，调用方可用原写 lease 重试。该契约涵盖写入前的失败，执行期间的致命运行时错误仍由运行环境处理。

本轮覆盖进程内、同一逻辑版本的 copy，适用于 CPU trial 或 ghost 子区间。MPI、跨域映射、设备异步交接、跨版本迁移和持久恢复留在后续阶段。Context7 返回 develop 的通用执行空间说明，具体 View、RangePolicy 和 fence 接口沿用已验证的锁定 5.2.2 Serial 路径。

## CPU 验证

使用已预供给的 Kokkos 5.2.2 Serial-only Release 包，commit `8e15454876298832f013ebcaf08d53825b5bda40`。环境为 arm64、macOS 27.2 build 26B123、SDK 27.2、AppleClang 21.0.0.21000401、CMake 4.4.3、Ninja 1.13.2。使用新构建目录，本轮未启用额外的禁网沙箱。依赖来源和摘要见 [CPU 基础记录](kokkos-cpu-slice-validation.md#依赖与机器配置)。

专用测试覆盖乱序稳定 ID 的标量试算写入和 component-first 非连续 ghost 子视图，并逐项检查区域之外的哨兵值。对错误源 incarnation、不同试算 attempt、旧目标 ID layout、错误单位、非法 purpose、错误分量/ID 数量、缺失和重复 ID，逐项校验目标仍保持原值。修正输入后复用原写 lease，核对重试结果。

| 配置 | 最终复验 | 结果 |
|---|---|---|
| CPU Debug | `2026-11-06T18:34:53+08:00`–`2026-11-06T18:35:01+08:00` | CTest 15/15，搬迁安装消费 3/3，目标图和头边界检查通过 |
| CPU Release | `2026-11-06T18:34:54+08:00`–`2026-11-06T18:35:02+08:00` | CTest 15/15，搬迁安装消费 3/3，目标图和头边界检查通过 |

两组配置、构建、CTest 和依赖检查均退出 0。安装回归检查私有接收器头未进入安装树。专用测试和 `FieldAccess.cpp` 还以 `-Wall -Wextra -Werror -fsanitize=address,undefined` 编译，连接同一预构建的 Kokkos 库，执行成功且无 sanitizer 诊断，完成于 `2026-11-06T18:34:59+08:00`。日志位于本地忽略目录 `results/local-receive/`。
