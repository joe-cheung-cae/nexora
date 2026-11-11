# real 迁移复查与修复

完成时间：`2026-11-10T08:19:34+08:00`。本次按数值路径、时间边界、安装配置三路并行复查，主代理集成并重新验证。审查当前工作树，保留原有未提交变更；没有写入 Mem。

## 发现并修复的问题

### 1. 公布的时间步上限可能无法执行

float 下 `lambda=real{0x1.34p126}` 时，旧 `time_constraint()` 返回的 double 上限约为 `9.77034265618524746e-39`。转为 subnormal float 后向上舍入，real 乘积成为 `1.00000011920928955`，导致按公布上限调用 begin 仍被严格稳定性检查拒绝。旧下界 0 也没有反映 dt 转换归零或有效因子下溢的限制。

[Participant](../../src/participants/reference/Participant.cpp) 现公布保守的正下界 `max(real_denorm_min, real_denorm_min/lambda)`；上界先沿用 double 候选，仅当 double 请求乘积或 real 转换乘积越过 1 时，向下调整一个 real 可表示值。上述复现的修复后上限为 `9.77034194643667398e-39`，下限为 `1.40129846432481707e-45`。

回归使用五种 lambda（1、real 最小次正规数、real 最大值和两个确定的极大 float 值），分别按公布的精确上下端点执行 begin、advance、prepare_commit、commit，检查 accepted 时间及 stamp 仍等于 double 端点。原转换归零、超范围和因子下溢拒绝用例保留。

公共时间约束的 8ε 宽容和 State 的严格数值检查均未改变。通用 bounds 校验是初筛，不能代替方法检查；本次保证上述公布端点可执行，不声称所有容忍越界值都能计算。数值公式、精度选择和原有误差门槛没有改变。

### 2. 动态预定义宏可绕过配置固定性

旧生成头在校验后保留调用方宏文本。float 消费者预定义 `NEXORA_ENABLE_DOUBLE_PRECISION=__COUNTER__` 时，首次展开为 0 而通过校验，Real.hpp 再次展开为 1，导致同包两个编译单元分别得到 4 字节和 8 字节的 real。

[Config.hpp.in](../../cmake/Config.hpp.in) 现保留冲突值诊断，之后无条件将宏重新定义为生成的 0/1 字面量。[安装消费测试](../../tests/package/consumer/CMakeLists.txt) 新增动态宏展开回归，同时检查宏值和实际类型；普通匹配宏、相反宏和相反 CMake 请求的原检查继续执行。

另用新安装包运行独立多编译单元消费者：float 输出 `normal=4 overridden=4`，double 输出 `normal=8 overridden=8`，均退出 0。源代码、命令、输出和时间见本机 [multi-tu.json](../../results/real-review/multi-tu.json) 所记录的临时项目。

## 未发现的问题与边界

- 未发现遗漏的生产 double 数值字段、Kokkos 数值 kernel 或传输转换。保留的显式 double 运算是请求时间步和稳定性/范围边界校验。
- 三个生产编译单元 State、Participant、FieldAccess 在各配置生成头下，通过 `-Wdouble-promotion -Wfloat-conversion -Wimplicit-float-conversion -Werror -fsyntax-only`；Kokkos 头作为 system include。该检查证明这些编译单元没有相应诊断，不等于验证未构建的设备路径。
- `FieldIssuer` 的 float/double 分支属于实际 dtype 校验，应保留。泛型 transport 按源元素类型打包并匹配接收类型，没有自动精度转换。
- 未发现时间误改。新增编译期断言覆盖 TimeConstraint 的上下界/量子、TrialReport reached_time、FieldStamp 有效期和 TimeState accepted_time；原 TimeInterval、TimeStamp 和 Snapshot 时间断言继续通过。

## 重新验证

环境为 arm64、macOS 27.2 build 26B123、AppleClang 21.0.0 (`clang-2100.4.1.1`)、C++20、CMake 4.4.3、Ninja 1.13.2、Python 3.14.7。依赖仍为锁定 commit `8e15454876298832f013ebcaf08d53825b5bda40` 的 Kokkos 5.2.2 Serial-only Release 包，许可 `Apache-2.0 WITH LLVM-exception`，无补丁或新依赖；源归档和静态库摘要再次与[既有记录](real-numerics-validation.md)比对一致。未执行依赖下载或另加禁网沙箱。

四个目录均为本轮新建：`/tmp/nexora-real-review-01a1c002/<precision>-<mode>`。沿用[数值迁移验证](real-numerics-validation.md)的 configure/build/dependency/CTest/80 位 Decimal 命令，额外执行上述严格转换诊断。float 不传精度选项，double 显式设置 `NEXORA_ENABLE_DOUBLE_PRECISION=ON`。

| 配置 | 开始 / 完成时间 | 构建、转换诊断、依赖、Decimal | CTest |
|---|---|---|---|
| float Debug | `2026-11-10T08:17:53+08:00` / `2026-11-10T08:18:17+08:00` | 均退出 0 | 15/15，11.98 s |
| double Debug | `2026-11-10T08:17:53+08:00` / `2026-11-10T08:18:17+08:00` | 均退出 0 | 15/15，11.97 s |
| float Release | `2026-11-10T08:17:53+08:00` / `2026-11-10T08:18:19+08:00` | 均退出 0 | 15/15，12.12 s |
| double Release | `2026-11-10T08:17:53+08:00` / `2026-11-10T08:18:19+08:00` | 均退出 0 | 15/15，11.94 s |

每项包含新增端点和动态宏回归，以及干净安装/搬迁、独立消费、旧接口缺失、精度冲突、外部 owner/lease、收发、恢复/重试/重放。80 位 Decimal 独立验证继续通过，三层最大离散分量误差仍为 float `1.0456617304142321e-7`、double `1.9918532962658262e-16`，收敛阶和守恒判据不变。

原始日志、逐步骤确切命令、秒级时间、环境/制品清单和数值 JSON 见本机 [results/real-review](../../results/real-review)。该目录被 Git 忽略，不随源码发布。本次未验证 OpenMP、CUDA/HIP、MPI 数值通信、持久 checkpoint 或完整 P1。
