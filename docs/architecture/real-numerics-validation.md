# 现有字段与算子的 real 迁移验证

执行完成时间：`2026-11-09T20:23:35+08:00`。本次已授权迁移现有数值字段与算子，时间类型保留 double；这是 Kokkos Serial CPU 的 float/double 数值证据，不关闭完整 P1。

后续复查在 `2026-11-10T08:19:34+08:00` 修复了时间约束端点和动态精度宏问题，重新执行四种配置及独立数值验证；见[复查记录](real-migration-review.md)。以下保留本轮原始迁移证据。

## 实现范围

- reference 的 x/q accepted/trial Views、配置数组、lambda、算子 dt 与中间量、发送及接收载荷统一为 `nexora::real`；保持 `d=dt*lambda*x`、`trial_x=x-d`、`trial_q=q+d` 的顺序和物理约定。
- `FieldDescriptor` 默认采用 `real_scalar_type`，随构建为 float32/float64；实际字段描述符和 payload dtype 一致。整数 ID 保持 uint64；外部显式 dtype 的校验仍保留，错精度负例按配置选择另一种类型。
- double 保留于 `TimeInterval`、时间戳/有效期、accepted 时钟和时间协调。请求时间步先通过 double 稳定性检查，再校验 real 转换范围、有限正值和 real 有效因子，成功后保存本窗口已验证的 real dt。转换溢出、归零或有效因子下溢均在分配 trial、增加 attempt 前拒绝，不提交状态。
- data 的字段访问、归约、外部 owner、pack/unpack 及相应测试改用 real。整数探针和独立测试 oracle 不属于数值状态。原有 lease、独立 trial、旧快照、失败恢复和重放契约保持。

实现见 [State](../../src/methods/reference/State.cpp)、[Participant](../../src/participants/reference/Participant.cpp) 和 [FieldDescriptor](../../include/nexora/data/FieldDescriptor.hpp)。精度开关仍为 `NEXORA_ENABLE_DOUBLE_PRECISION`，默认 OFF/float，ON/double；安装包配置与冲突拒绝不变。

## 验证基准与误差界

使用原 reference 输入 `lambda=1, x0=1, q0=0, T=1`，16/32/64 步，分别比较独立闭式 Euler 解 `(1-dt)^N` 和解析解 `exp(-T)`；两次观测阶必须为 0.9–1.1。另覆盖 `dt=0.1`、10 步，检查 double 时间步向 real 转换的影响。

- double 保留原门槛：守恒误差不超过 `1e-12`，离散分量误差不超过 `1e-12 + 1e-10*abs(oracle)`。
- float 的本夹具绝对界为 `8*N*epsilon + N*abs(real(dt)-dt)`。单位幅值、lambda=1 的稳定衰减中，x/q 非负且总量约为 1；每步至多四次数值运算，精确更新矩阵在 1-范数不放大误差。`N*epsilon` 很小时，8 倍 epsilon 为每步舍入及其累积留出保守余量；第二项覆盖步长转换。该界只用于本次 N≤64 的 reference，不转用到其他物理模型或归约。
- 本机 `long double` 和 `double` 均为 53 位尾数，不能把 C++ long-double 对照宣称为高于 FP64 的 oracle。因此额外使用 [独立检查器](../../tools/check_reference_precision.py)，以 Python 标准库 Decimal 的 80 位精度重算闭式/解析解、最终守恒及收敛阶。17 位日志经 float 解析、`Decimal.from_float` 恢复实际二进制值；float 状态还检查能否精确表示为 float32。阈值按传入 precision 独立重算，不信任日志打印的阈值。

## 环境、命令与执行结果

实际环境：arm64、macOS 27.2 build 26B123，AppleClang 21.0.0 (`clang-2100.4.1.1`)、C++20、CMake 4.4.3、Ninja 1.13.2、Python 3.14.7。Kokkos 为本地 5.2.2 Serial-only Release 包，commit `8e15454876298832f013ebcaf08d53825b5bda40`，许可 `Apache-2.0 WITH LLVM-exception`，无本地补丁。四个 Nexora 配置均使用同一依赖。

本次重新核验源归档 SHA-256 `d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f` 和 `libkokkoscore.a` SHA-256 `8ec81e18e997069104c9a299c4c5022a2cf17a757355391c58cb28f12643302c`，与[依赖来源](../../third_party/README.md)和[既有制品记录](kokkos-cpu-slice-validation.md)一致。未获取新依赖或触发构建下载；未另加禁网沙箱。

在本轮新建的 `/tmp/nexora-real-numerics-01a1c002/<precision>-<mode>` 执行配置、编译、依赖检查、CTest 和 Decimal 检查。float 不传精度开关，double 显式增加 `-DNEXORA_ENABLE_DOUBLE_PRECISION=ON`：

```sh
cmake -S . -B <build> -G Ninja -DKokkos_DIR=<verified-prefix>/lib/cmake/Kokkos \
  -DCMAKE_BUILD_TYPE=<Debug-or-Release> -DCMAKE_CXX_EXTENSIONS=OFF \
  -DBUILD_TESTING=ON -DNEXORA_ENABLE_MPI=OFF
cmake --build <build> --parallel 4
python3 tools/check_dependencies.py --graph <build>/nexora-target-graph.json
ctest --test-dir <build> --output-on-failure
python3 tools/check_reference_precision.py --log <build>/Testing/Temporary/LastTest.log \
  --precision <float-or-double> --output <report.json>
```

| 配置 | 开始 / 完成时间 | 配置、编译、依赖、Decimal | CTest |
|---|---|---|---|
| float Debug | `2026-11-09T20:23:14+08:00` / `2026-11-09T20:23:34+08:00` | 均退出 0 | 15/15，11.54 s |
| double Debug | `2026-11-09T20:23:14+08:00` / `2026-11-09T20:23:34+08:00` | 均退出 0 | 15/15，11.62 s |
| float Release | `2026-11-09T20:23:14+08:00` / `2026-11-09T20:23:35+08:00` | 均退出 0 | 15/15，11.62 s |
| double Release | `2026-11-09T20:23:14+08:00` / `2026-11-09T20:23:35+08:00` | 均退出 0 | 15/15，11.60 s |

每项包含干净安装、包搬迁、独立消费者、匹配/冲突精度宏与 CMake 请求、dtype 拒绝、外部 owner、alias/lease、收发快照、恢复/重试/重放，以及步长转换异常。大绝对时间下的 requested duration 回归继续通过。

## 数值观测

Debug/Release 在本机得到相同 reference 最终值。下列离散误差为 Decimal 80 位 oracle 的 x/q 两分量最大值；守恒列为 C++ 运行时对各子步的最大观测，最终状态另由 Decimal 精确核验。

| 精度 | 步数 | 最大离散分量误差 | 解析 x 误差 | 最大观测守恒误差 |
|---|---:|---:|---:|---:|
| float | 16 | 3.56794539e-8 | 1.18052868e-2 | 5.96046448e-8 |
| float | 32 | 4.71947016e-8 | 5.82413951e-3 | 1.19209290e-7 |
| float | 64 | 1.04566173e-7 | 2.89302149e-3 | 1.49011612e-7 |
| double | 16 | 1.38235777e-17 | 1.18053107e-2 | 0 |
| double | 32 | 1.99185330e-16 | 5.82415192e-3 | 4.44089210e-16 |
| double | 64 | 9.87903885e-17 | 2.89291693e-3 | 4.44089210e-16 |

float 观测阶为 `1.0193162726064826 / 1.0094678709055236`，double 为 `1.0193161223839090 / 1.0095230910767743`。float 的 16/32/64 步舍入界分别为 `1.52587890625e-5 / 3.0517578125e-5 / 6.103515625e-5`，实际离散误差与守恒误差均低于界；double 维持旧门槛。

`dt=0.1` 的 x/q 最大离散误差：float 为 `3.004674680466572e-8`，double 为 `4.733728239392221e-17`，均通过含步长转换项的相应门槛。

完整 80 位结果、每步命令/秒级时间和原始日志见本机 [results/real-numerics](../../results/real-numerics)，含四个配置的 `commands.json`、`numerical-report.json` 及 configure/build/dependencies/ctest/decimal 日志。`results/` 被 Git 忽略，不随源码发布。

本轮已配置、编译、执行并数值验证上述 Serial CPU 组合。没有验证 OpenMP、CUDA/HIP、MPI 数值通信、持久 checkpoint 或尚不存在的产品 solver，不能把本结果扩展到这些能力或跨精度恢复。
