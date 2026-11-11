# real 精度配置验证

验证完成时间：`2026-11-09T19:45:11+08:00`。

## 接口与环境

`nexora::real` 定义于 `core/Real.hpp`，也可通过 `core/Types.hpp` 使用。默认 `NEXORA_ENABLE_DOUBLE_PRECISION=OFF` 对应 `float`，配置 `-DNEXORA_ENABLE_DOUBLE_PRECISION=ON` 对应 `double`。生成的 `core/Config.hpp` 随安装包固定选择；相同编译宏被接受，相反宏值编译失败。安装消费者显式请求不同精度时，`find_package` 报 `Nexora precision mismatch`。

别名只影响显式使用 `real` 的代码。时间、既有 FP64 reference 的数据/公式/容差及 `FieldDescriptor` 默认 `float64` 均未改变；real 字段须按实际元素类型填写 dtype。

本机为 arm64、macOS 27.2 build 26B123，AppleClang 21.0.0 (`clang-2100.4.1.1`)、C++20、CMake 4.4.3、Ninja 1.13.2。依赖为既有 Kokkos 5.2.2 Serial-only Release 包，commit `8e15454876298832f013ebcaf08d53825b5bda40`，许可 `Apache-2.0 WITH LLVM-exception`，无新增补丁。

本轮重新核验源归档 SHA-256 `d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f` 和 `libkokkoscore.a` SHA-256 `8ec81e18e997069104c9a299c4c5022a2cf17a757355391c58cb28f12643302c`，与[既有来源](../../third_party/README.md)和[制品记录](kokkos-cpu-slice-validation.md)一致。完整环境及核验时间见本机 [environment.json](../../results/real-precision/environment.json)。没有获取新依赖或 configure/build 下载；未另加禁网沙箱。

## 命令与结果

在本轮新建的 `/tmp/nexora-real-01a1c002/{float,double}-{debug,release}` 中分别配置、编译和执行。float 两项不传精度开关，以验证默认值；double 两项增加 `-DNEXORA_ENABLE_DOUBLE_PRECISION=ON`：

```sh
cmake -S . -B <build> -G Ninja \
  -DKokkos_DIR=<verified-prefix>/lib/cmake/Kokkos \
  -DCMAKE_BUILD_TYPE=<Debug-or-Release> -DCMAKE_CXX_EXTENSIONS=OFF \
  -DBUILD_TESTING=ON -DNEXORA_ENABLE_MPI=OFF
cmake --build <build> --parallel 4
python3 tools/check_dependencies.py --graph <build>/nexora-target-graph.json
ctest --test-dir <build> --output-on-failure
```

| 配置 | 开始 / 完成时间 | 配置、编译、依赖检查 | CTest |
|---|---|---|---|
| float Debug | `2026-11-09T19:44:03+08:00` / `2026-11-09T19:44:23+08:00` | 均退出 0 | 15/15，11.66 s |
| float Release | `2026-11-09T19:44:03+08:00` / `2026-11-09T19:44:26+08:00` | 均退出 0 | 15/15，12.45 s |
| double Debug | `2026-11-09T19:44:03+08:00` / `2026-11-09T19:44:24+08:00` | 均退出 0 | 15/15，11.60 s |
| double Release | `2026-11-09T19:44:03+08:00` / `2026-11-09T19:44:26+08:00` | 均退出 0 | 15/15，12.39 s |

- core 静态断言验证实际标量类型，并确认时间成员仍为 `double`。
- Kokkos probe 使用 `View<real*>` 和 kernel 内的 `Array<real, 2>` 计算 32 个精确可表示的小数。显式复制、fence 后逐元素对照 `(2*i+1)/4`，两种精度均精确相等，最后一项为 `15.75`；没有新增浮点容差。
- 每项配置完成干净安装、包搬迁和三个独立消费者的编译与执行。消费者用上游独立期望值检查实际类型；typed View、ReadField/WriteField 和 TransportView 元素类型随 `real` 匹配。
- 每项安装验收验证相同宏编译成功、相反宏因指定配置诊断编译失败、相反 CMake 精度请求配置失败。原有旧层缺席、导出依赖和包兼容负例保留。
- simulation/reference 提交、恢复、重试、保活及既有 FP64 数值回归全部通过。

原始命令、逐步骤时间和日志在本机 [results/real-precision](../../results/real-precision)，每项配置有 `commands.json` 及 configure/build/dependencies/ctest 日志。该目录被 Git 忽略，不随源码发布。

设计文档完整示例另外使用 float/double 两套生成头编译，并以 `n=0`、`n=1` 执行，均退出 0；完成时间为 `2026-11-09T19:45:11+08:00`，命令见本机 [documented-interface.json](../../results/real-precision/documented-interface.json)。

以上是 Serial CPU 两种别名配置及现有回归证据，不声明现有 solver 已整体切换精度，不提供 GPU/OpenMP/MPI、跨精度 checkpoint 或不同配置头库混用的支持证据。切换精度应使用独立干净构建/安装前缀，并重建消费者。
