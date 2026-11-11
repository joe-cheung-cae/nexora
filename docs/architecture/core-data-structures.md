# Core 与数值数据结构

设计修订时间：`2026-11-09T20:24:13+08:00`。本文描述本次明确授权的接口切换；旧数值容器与 exec 层切换的运行结果见[切换验证](core-data-structures-validation.md)，`real` 别名配置见[精度验证](real-precision-validation.md)，现有数值字段与算子的两种精度验证见[数值迁移验证](real-numerics-validation.md)。当前实现以 [架构入口](overview.md)为准。完整 P1、设备异步交接和持久 checkpoint 仍未完成。

## 职责和接口

| 用途 | 当前接口 | 边界 |
|---|---|---|
| 身份、索引、版本、时间戳和试算 epoch | [`Types.hpp`](../../include/nexora/core/Types.hpp) | core 中的领域值；稳定 ID 与局部索引分开 |
| 可配置实数标量 | [`Real.hpp`](../../include/nexora/core/Real.hpp)（`Types.hpp` 也包含） | `nexora::real` 默认 `float`，启用双精度时为 `double`；由包配置固定 |
| 时间区间 | [`Time.hpp`](../../include/nexora/core/Time.hpp) | `duration()` 返回 `requested_step`；participant 旧头继续兼容包含 |
| 错误与整数转换 | [`Status.hpp`](../../include/nexora/core/Status.hpp)、[`Result.hpp`](../../include/nexora/core/Result.hpp)、[`CheckedCast.hpp`](../../include/nexora/core/CheckedCast.hpp) | host 控制契约；不替代设备故障处理 |
| 数值数组、分量与多维存储 | Kokkos 原生 `View`、`Array` | data/method 按实际 space、布局与访问模式选择；core 不引入 Kokkos 或容器别名 |
| 字段语义和访问 | [`FieldDescriptor`](../../include/nexora/data/FieldDescriptor.hpp)、[`FieldBinding`](../../include/nexora/data/FieldBinding.hpp) 中 `ReadField`/`WriteField` | 在原生 View 上增加语义、版本、访问授权与保活；不复制 View API |

`nexora::core` 是仅依赖 `nexora::build_options` 的 interface target；`nexora::data` 公开传递 `Kokkos::kokkos`。方法直接使用 Kokkos，无新容器 facade 或通用执行层。当前仅 Serial 子集已实现；示例说明原生接口，不表示所有 rank、布局或设备均已纳入当前字段绑定的支持范围。

## 精度配置

`nexora::real` 的类型由构建选项 `NEXORA_ENABLE_DOUBLE_PRECISION` 决定：默认 `OFF` 为 `float`，配置 `-DNEXORA_ENABLE_DOUBLE_PRECISION=ON` 为 `double`。生成并安装的 `nexora/core/Config.hpp` 将选择固定为 0/1；不可用编译命令行宏覆盖包配置，下游明确请求与已安装包冲突的精度会被拒绝。切换精度时使用独立的干净 build 和 install prefix，并重建消费者；不得混用不同配置的头与库。

现有 reference 的 x/q、lambda、算子中间量和本地传输载荷已统一使用 `real`。`TimeInterval`、`TimeStamp`、有效期、accepted 时钟与时间协调仍为 `double`。`FieldDescriptor` 默认使用 `real_scalar_type`，随配置对应 `float32/float64`；整数或其他显式类型的外部字段仍须声明实际 dtype。reference 公式与 double 原门槛不变，float 按机器 epsilon、步数和时间步转换误差验证，详见[数值迁移验证](real-numerics-validation.md)。

时间步在方法边界从 double 转为 real：检查 double 请求的稳定条件、real 可表达范围及转换后非零正值，再检查 real 有效因子。通过校验的 real 步长用于该窗口算子，时间仍按原 double 请求推进。不可表达、转换归零或有效因子下溢的请求明确拒绝，不提交状态。reference 公布考虑可表达性的保守正下界，并在上限转换会越过稳定边界时向下调整；通用时间 bounds 校验不能代替方法严格检查。安装配置宏校验后固定为生成的字面量，避免重复展开调用方表达式改变精度。端点、动态宏与数值回归见[迁移复查记录](real-migration-review.md)。

## 原生 Kokkos 示例

以下示例在应用已初始化 Kokkos 且 Views 在 runtime 关闭前销毁的作用域中使用。实际 execution/memory space 由选定方法及构建决定；示例使用当前 Serial 的 HostSpace。`View` 的固定维放在动态维之后，布局由显式类型决定。参见 Kokkos [View 接口](https://kokkos.org/kokkos-core-wiki/API/core/view/view.html)。

```cpp
#include <Kokkos_Core.hpp>
#include <Kokkos_Array.hpp>
#include <nexora/core/Real.hpp>
#include <cstddef>

void example(std::size_t n) {
    using VectorField = Kokkos::View<nexora::real*[3], Kokkos::LayoutRight, Kokkos::HostSpace>;
    using TensorField = Kokkos::View<nexora::real*[3][3], Kokkos::LayoutRight, Kokkos::HostSpace>;
    VectorField velocity("velocity", n);       // n 个三分量值
    TensorField stress("stress", n);          // n 个 3×3 值
    if (n > 0) {
        velocity(0, 1) = 2.0;
        stress(0, 1, 2) = 3.0;
    }
    VectorField::const_type read_velocity = velocity; // 元素只读的 View
    Kokkos::Array<nexora::real, 3> local{1.0f, 2.0f, 3.0f}; // 内嵌定长值
    (void)read_velocity;
    (void)local;
}
```

示例允许空数组，但不会索引它。`const VectorField` 只约束句柄自身，不使元素只读，读取授权须使用元素为 const 的 View 或 `const_type`。`Kokkos::Array` 是带内嵌存储的定长 aggregate，不是线性代数或张量运算库；需要具体数学操作时按实际方法实现或另行选择有依据的依赖，本次不新增 Kokkos Kernels。参见 [Kokkos Array](https://kokkos.org/kokkos-core-wiki/API/core/stl-compat/Array.html)。实际 API 以仓库锁定的 Kokkos 5.2.2 头文件及构建结果为准。

## 所有权、同步与迁移

- managed View 的句柄保活其分配，但引用计数并不授予写权限。字段访问仍由 lease 仲裁，当前同步 Serial 路径以整分配约束多读或单写；参见 [字段绑定测试](../../tests/data/field_binding_tests.cpp) 和 [reference 窗口](reference-window-validation.md)。
- 只有内存空间对主机可访问时才直接从主机读写；其他情况使用合适的镜像、复制和明确完成条件。跨 execution instance、设备与外部系统的交接须建立正确排序并验证完成，当前设备异步交接未实现。
- 外部数组通过可信 adapter 导入，校验空间、形状、容量、对齐和权限，保留真实 owner 直到全部使用完成，且禁止外部提前释放、重分配或违反写约束。裸指针和 unmanaged View 本身不能证明所有权或设备身份。当前支持范围为一维 HostSpace 外部 owner，详见 [外部内存与本地传输](external-transport-validation.md)及[传输测试](../../tests/data/transport_tests.cpp)。
- accepted/trial、旧快照与 lease 保持生命周期边界；本地发送打包与接收解包不等于跨进程通信或持久 checkpoint。持久化应记录逻辑 accepted 状态、版本和恢复所需元数据，不序列化 View 对象；R0 和延期范围见 [checkpoint 与重启](checkpoint-restart.md)。
- 旧 `ArrayView`、`Vec`、`Tensor`、`FieldView`、`KernelView` 与 exec 层及目标已移除，属于源代码和包接口不兼容变更。`ArrayView::at()` 的异常及禁止从临时 `vector` 借用的校验不是原生 View 的源兼容契约；不得仅靠裸指针重建旧借用语义。根据实际用途选择 managed View 或上述可信外部导入。仓库外消费者需要迁移。
- 验证安装时使用干净 prefix；覆盖安装不会删除旧头或旧库，应清理或替换已有 prefix 后再判断安装面。本次 Serial Debug/Release 的执行范围见[验证记录](core-data-structures-validation.md)，不代表完整 P1 已验收。
