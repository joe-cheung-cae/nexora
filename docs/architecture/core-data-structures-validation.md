# Core 数据结构切换验证

验证完成时间：`2026-11-09T19:35:52+08:00`。接口设计见 [core 数据结构](core-data-structures.md)。本记录仅覆盖本次旧容器与执行层删除后的 Serial CPU 构建、执行和安装消费，不关闭完整 P1。

## 变更范围

- 删除 core 的 `ArrayView`、`Vec`、`Tensor`，以及依赖它们的旧 `Buffer`、`FieldView/KernelView`；删除旧 exec 实现、目标、导出和专属测试。
- 保留 core 的 ID、索引、版本、时间、`Status/Result` 和 checked cast。core 仍只依赖 `build_options`，不公开依赖 Kokkos。
- 数值直接使用原生 Kokkos `View/Array`，字段语义与权限复用 `ReadField/WriteField`。现有 reference 公式、默认值和容差不变。
- 更新目标/include 检查器、安装消费者和现行文档。安装时使用干净 prefix，不依赖覆盖安装清理历史文件。

## 依赖与环境

| 项目 | 实际配置 |
|---|---|
| 主机 | arm64，macOS 27.2 build 26B123；具体 CPU 型号受沙箱读取限制，未取得 |
| 编译器 | AppleClang 21.0.0，`clang-2100.4.1.1`，C++20，无语言扩展 |
| 构建工具 | CMake 4.4.3，Ninja 1.13.2，Python 3.14.7 |
| Kokkos | 本地已预供给的 5.2.2 Serial-only Release 包；Nexora Debug/Release 均使用该包 |
| 版本与许可 | commit `8e15454876298832f013ebcaf08d53825b5bda40`，`Apache-2.0 WITH LLVM-exception`；本轮未引入补丁或新依赖 |
| 源制品 SHA-256 | `d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f` |
| `libkokkoscore.a` SHA-256 | `8ec81e18e997069104c9a299c4c5022a2cf17a757355391c58cb28f12643302c` |

本轮重新计算源归档和静态库摘要，与[既有来源记录](../../third_party/README.md)及 [CPU 切片制品记录](kokkos-cpu-slice-validation.md)一致。没有获取依赖或 configure/build 下载；没有另加禁网沙箱验证。完整工具链、制品和本地路径见本机 [environment.json](../../results/core-data-structures/environment.json)。

## 命令与结果

两个构建目录均为本轮新建。对 `Debug` 和 `Release` 各执行下列流程，`<build>` 分别为 `/tmp/nexora-core-kokkos-01a1c002/debug` 和 `/tmp/nexora-core-kokkos-01a1c002/release`：

```sh
cmake -S . -B <build> -G Ninja \
  -DKokkos_DIR=<verified-prefix>/lib/cmake/Kokkos \
  -DCMAKE_BUILD_TYPE=<Debug-or-Release> -DCMAKE_CXX_EXTENSIONS=OFF \
  -DBUILD_TESTING=ON -DNEXORA_ENABLE_MPI=OFF
cmake --build <build> --parallel 4
python3 tools/check_dependencies.py --graph <build>/nexora-target-graph.json
ctest --test-dir <build> --output-on-failure
```

| 配置 | 开始 / 完成时间 | 配置、编译、依赖检查 | CTest 执行 |
|---|---|---|---|
| Serial Debug | `2026-11-09T19:34:32+08:00` / `2026-11-09T19:34:49+08:00` | 均退出 0 | 15/15 通过，11.20 s |
| Serial Release | `2026-11-09T19:34:32+08:00` / `2026-11-09T19:34:51+08:00` | 均退出 0 | 15/15 通过，11.44 s |

测试数量从前次 16 项降至 15 项，原因是删除旧 `nexora_exec_tests`。保留的 simulation/reference 测试继续覆盖失败、恢复、重试、提交与旧快照；旧 FieldView 的形状、空范围及版本验证由 typed 字段测试覆盖。

- 原生接口测试验证 32 个整数标量、固定三分量 rank-2 View、固定 2×2 rank-3 View 和 kernel 内的 `Kokkos::Array`，逐元素精确对照 host 结果。没有引入新浮点容差或物理公式。
- 元素 const 类型在编译期不可写；managed 只读 View 在原可写句柄离开作用域后继续保活。显式 execution-instance 复制和 fence 后检查 host mirror，另验证 stride=3 的列 subview。
- 每个配置的安装测试在干净目录安装并搬迁包，再独立配置、编译、执行控制、schema 和 Kokkos 消费者。控制与 schema 不传递 Kokkos 编译/链接依赖，数值消费者只链接 `nexora::data`。
- 安装断言拒绝旧 exec targets 和已删除公共头；依赖 checker 的正反例拒绝旧 target/include、core/schema 的 Kokkos 越界以及向上依赖。原有缺包、错版本、错编译器、非 Serial 配置和 C++17 等配置负例仍在安装测试中执行。

原始命令、秒级时间和退出码见本机 [Debug commands.json](../../results/core-data-structures/debug/commands.json) 与 [Release commands.json](../../results/core-data-structures/release/commands.json)；各目录保存 configure/build/dependencies/ctest 日志。`results/` 是本机忽略目录，不随源码发布。

## 文档示例与支持边界

提取设计文档中的完整 C++ 示例，以本地 Kokkos 5.2.2、C++20 和 `-Wall -Wextra -Wpedantic` 编译；在同一 `ScopeGuard` 内以 `n=0`、`n=1` 运行。编译和执行均退出 0，时间为 `2026-11-09T19:35:51+08:00` 至 `2026-11-09T19:35:52+08:00`。确切命令和探针摘要见本机 [documented-interface.json](../../results/core-data-structures/documented-interface.json)。

本轮已配置、编译并执行 Serial CPU 路径，验证上述整数数据结构行为及既有 reference 数值回归。未配置或运行 OpenMP、CUDA/HIP、MPI 数值通信、设备异步交接、SPH/FVM 产品算子或持久 checkpoint；不据此声明这些能力或完整 P1 验收。
