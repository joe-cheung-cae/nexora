# P1-04 CPU 外部内存与本地传输验证

复验区间为 `2026-11-06T18:21:34+08:00` 至 `2026-11-06T18:21:49+08:00`。本轮覆盖 Kokkos Serial 下的外部 HostSpace 内存和进程内传输。

## 所有权与访问

- 私有 `ExternalField<T>` 接收一维数组的真实分配基址、字节容量、元素数和步幅，以及指向同一基址的共享 owner。包装前检查 owner 的共享所有权、空指针、对齐、地址与跨度溢出及容量，再创建 unmanaged `LayoutStride` View。空字段仍需有效的 owner 和基址。可信适配器须保证实际分配容量和元素生命周期，导入前完成 producer 访问，并在签发器及 lease 存续期间禁止外部 resize、free 或绕过授权的写入。
- 外部内存复用 managed 字段的 `FieldIssuer`、`ReadLease` 和 `WriteLease`。同一外部 owner 的控制块或重叠的字节范围只能登记一次，范围冲突检查覆盖 managed 与 unmanaged。写者独占，accepted 和 const 导入只读。关闭停止新签发。已有 lease 仍有效，最后一次释放发生在签发器和所有 lease 都销毁之后。
- 私有 `TransportBuilder` 消费数值字段与稳定 ID 的两个读取 lease。它检查实体命名空间、ID 类型、唯一性、数量、轴角色及版本一致性。两个不同的源分配允许有不同的 storage incarnation。`TransportView` 保留数值字段的源描述符、stamp 与 purpose。
- 显式 Serial pack 将 rank-1 标量或 rank-2 实体/分量字段转换成只读 `LayoutRight[entity, component]`，同时复制稳定 ID。实例 fence 完成后才交接给消费者。每个消费者获得独立快照并保留两份源 lease，即使已复制也会阻止源分配复用，直至该消费者调用 `complete()` 或销毁记录。调用方须在最后一次使用后结束记录，原生 View 的副本不会延长源授权。准备失败由 RAII 归还两份 lease。

本轮实现本地 pack 与消费者完成。后续本地接收与解包见 [CPU 本地接收验证](local-receive-validation.md)。MPI、设备异步交接、持久 checkpoint 和全链多分配事务仍待验证或实现。reference 窗口接入另见 [reference 窗口集成](reference-window-validation.md)。运行时必须晚于签发器、传输记录和保留 Views 关闭。

Context7 未返回针对这两个查询的接口文档，接口以 Kokkos 官方互操作说明、锁定版本本地源码和实际编译运行核对。 [Kokkos 互操作说明](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Interoperability.html) · [执行空间契约](https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html)

## CPU 结果

使用已预供给的 Kokkos 5.2.2 Serial-only Release 库，commit `8e15454876298832f013ebcaf08d53825b5bda40`。测试环境为 arm64、macOS 27.2 build 26B123、SDK 27.2、AppleClang 21.0.0.21000401、CMake 4.4.3、Ninja 1.13.2。依赖来源和摘要见 [CPU 基础记录](kokkos-cpu-slice-validation.md#依赖与机器配置)。本轮复验未启用额外的禁网沙箱。

| 配置 | CTest | 安装与依赖检查 |
|---|---|---|
| CPU Debug | `2026-11-06T18:21:38+08:00`–`2026-11-06T18:21:48+08:00`，14 通过，0 失败或跳过 | 搬迁后安装消费 3/3 通过，目标图和头边界通过 |
| CPU Release | `2026-11-06T18:21:39+08:00`–`2026-11-06T18:21:48+08:00`，14 通过，0 失败或跳过 | 搬迁后安装消费 3/3 通过，目标图和头边界通过 |

两组都在新构建目录配置、编译、运行 CTest 和依赖检查，退出码均为 0。命令及输出保存在本地忽略目录 `results/external-transport/`。安装消费者验证只读载荷与私有导入、签发和打包头的安装边界。

新增测试实际执行 strided unmanaged 读取、Serial 写入和 component-first 打包，逐项对照精确结果及稳定 ID。还验证无运行时、错误 owner、对齐/容量/步幅、重复或重叠导入、旧 stamp、错误 purpose/ID/布局、失败后重新获写、空载荷和关闭后的保活。两个消费者先后完成时，最后一个完成才释放已无其它 owner 的外部值数组与 ID 数组，释放计数均为一次。

`2026-11-06T18:22:31+08:00` 至 `2026-11-06T18:22:37+08:00`，另用 AppleClang 对 `FieldAccess.cpp` 和专用 transport 测试启用 `-Wall -Wextra -Werror -fsanitize=address,undefined`，编译并执行成功，退出码 0，无 sanitizer 诊断。链接的 Kokkos Release 库沿用本地制品。

最终测试源码复验于 `2026-11-06T18:23:40+08:00` 至 `2026-11-06T18:23:48+08:00` 完成。Debug、Release 均为 CTest 14/14，含搬迁安装消费 3/3，依赖检查通过。完整日志为 `results/external-transport/final-cpu-debug.log` 和 `results/external-transport/final-cpu-release.log`。
