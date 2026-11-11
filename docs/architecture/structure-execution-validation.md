# Serial 状态重排与整组授权

本记录对应 [底层开发计划的任务 3 和 4](foundation-development-plan.md#任务-3-在实际状态中实现原子重排)。两项实现均使用锁定的 Kokkos 5.2.2 Serial。状态和授权的版本语义见 [Kokkos 设计](kokkos-design.md)。

## 结构状态与安全点

`StructureState` 位于 `src/participants/reference`，由现有 `nexora_participants_reference` 静态目标构建。测试提供输入并调用同一实现。状态由单一调用者同步操作，accepted bank 只签发读取权限。单个本地域的实体全部为 owned，ghost 目标仅用于收发夹具。

| 数据 | 重排与发布规则 |
| --- | --- |
| 稳定 ID、几何 `x`、标量 `q`、被动历史 | 同一个 `new_to_old` 将所有字段 gather 到独立候选分配。新字段的 storage incarnation 由各自的新分配产生。 |
| CSR 邻接 | `new_to_old` 选择旧行，`old_to_new` 重映射列索引。保留每行稳定 ID 邻接顺序，重建行列局部索引及其 layout 关联。 |
| 逻辑 stamp | 纯重排只推进 layout。topology、geometry、accepted state version 和时钟继承原值。 |
| 快照与授权 | 候选分配、Kokkos 工作、诊断及只读授权准备完毕后关闭旧 issuer 的新签发，并以一次 bank 交换发布。已签发的旧读取授权随旧快照保活。 |

`nexora_structure_reorder_tests` 用独立字面量核对原序、重排、逆排的 ID、字段、历史和 CSR 边表。它在部分 gather 完成后及候选授权准备完毕后分别注入失败，检查 accepted bank、字段与版本。夹具随后发布一组调用方提供的新值，验证旧快照穿过下一次提交及状态析构，旧 stamp、顺序和 CSR 在新布局上的校验失败。真实 `TransportBuilder` 和 `TransportReceiver` 把重排前的快照解包到重排后的 ghost 目标，并检查 stale stamp 拒绝时目标保持原值。

## Reference 执行与授权

reference participant 在初始化时选定一个 `Kokkos::Serial` 实例。初始化用该实例分配三个 HostSpace Views，从仍由配置持有的 host 缓冲复制，完成 fence 后才释放输入。试算分配由同一实例完成，随后由私有 `relaxation` 按传入的实例和 native const/mutable Views 执行。诊断按 host 可访问性选择读取方式，并在所用实例完成后作数值判断。公式、步长转换和容差沿用 reference 的既有判据。

`FieldIssuer` 先验证每个请求并保留 typed View 和 stamp。请求的权限、authority 和 binding 仅由 `FieldIssuer` 持有和消费。整组预留 grant 和锁记录，对每个 allocation 合并同模式别名，拒绝不同模式混用，再按 storage incarnation 顺序锁定。所有权限检查通过后一起生效。组内同一 allocation 的绑定共享一次授权计数，最后一个绑定释放时归还。

| 同步操作 | 一次取得的权限及完成边界 |
| --- | --- |
| 试算 | accepted `x/q` read 加 trial `x/q` overwrite。kernel 和有限值诊断完成前保留整组。 |
| 快照 | accepted `x/q` read 随快照保活。 |
| 发送 | trial 字段与 ID read。先在实例完成后检查 ID，再打包。Reference 在 pack 完成且缓冲独立后释放源授权。需延长源保活的外部输入由消费者各自持有授权及 owner。 |
| 接收 | trial 目标 overwrite 加 ID read。语义、版本、形状和 ID 校验通过后才写入，解包和诊断完成后释放。 |

`nexora_field_group_tests` 覆盖 pending 字段不可访问、移后及关闭请求拒绝、subview 与重复请求、多个 reader、独占 writer、整组冲突与反序申请。`nexora_transport_tests` 使用整组授权给两个独立消费者打包，检查 caller 提前释放后仍可读取，并观察最后一个消费者完成时值和 ID 的 owner 各释放一次。原有 reference 数值和收发测试继续覆盖窗口恢复、不同轴序与独立 Decimal 判据。

## 本地复验

Apple M2 Pro arm64，macOS 27.2.1 build 26B131，AppleClang `21.0.0.21000401`，CMake `4.4.3`，Ninja `1.13.2`，Python `3.14.7`。验证复用任务 1/2 记录中按构建类型供应的 Debug 和 Release Kokkos 5.2.2。每次运行重新核对安装文件摘要，开始和结束各 10 个 IPv4/IPv6 TCP、UDP 和 bind 探针均返回 `EPERM`。 [Kokkos 供给记录](serial-ci-validation.md)

| 配置 | 配置开始至包验证完成 | CTest | 独立消费者 | Decimal |
| --- | --- | --- | --- | --- |
| float Debug | `2026-11-11T11:54:23+08:00` 至 `2026-11-11T11:54:48+08:00` | 18/18 | 3/3 | 通过 |
| double Debug | `2026-11-11T11:54:48+08:00` 至 `2026-11-11T11:55:12+08:00` | 18/18 | 3/3 | 通过 |
| float Release | `2026-11-11T11:54:23+08:00` 至 `2026-11-11T11:54:51+08:00` | 18/18 | 3/3 | 通过 |
| double Release | `2026-11-11T11:54:51+08:00` 至 `2026-11-11T11:55:18+08:00` | 18/18 | 3/3 | 通过 |

四组依赖检查和安装搬迁正反例均通过，CTest 失败和跳过均为 0。Apple 链接器对重排测试的直接与传递 data 链接报重复静态库警告，链接及执行通过。用 `BUILD_TESTING=OFF` 于 `2026-11-11T11:56:14+08:00` 至 `2026-11-11T11:56:19+08:00` 完成干净 Release 构建和依赖检查，CTest 注册为 0，`nm -C` 在 `libnexora_participants_reference.a` 确认 `StructureState::create`、`reorder` 和 `commit_values` 符号。

`2026-11-11T11:56:14+08:00` 至 `2026-11-11T11:56:15+08:00`，Kokkos Tools 回调记录 `Participant::initialize` 期间三次 HostSpace 分配、三次显式复制和一次 `reference initial state ready` fence。`finalize` 的 HostSpace 释放 fence 出现在初始化返回之后。探针源文件和编译命令随日志保留。

独立复审在 `2026-11-11T11:54:10+08:00` 至 `2026-11-11T11:55:07+08:00` 再次跑完四组禁网验证，并验证关闭测试的构建。复审确认篡改 pending 字段的代码不能编译。四处 host 分配分别注入 `bad_alloc` 时返回 `resource_exhausted`，且随后可重新取得完整授权。独立复审通过，原始日志位于 `/private/tmp/nexora-r34-review.tYcddB`。

本次执行范围为本地 Serial。Linux GCC、Linux Clang 和 Windows MSVC 的托管矩阵尚未触发。MPI、OpenMP、CUDA/HIP 和任务 5 的完整历史、控制及快照预算属于各自的验收范围。

## 重跑命令与证据

设置 `nexora_kokkos_run` 为按 [任务 1/2 命令](serial-ci-validation.md#重跑) 供应了 `kokkos-debug` 和 `kokkos-release` 的目录。下列循环在受限网络环境中执行，各构建类型同时验证 float 和 double。

```sh
nexora_kokkos_run=/tmp/nexora-serial-ci.ODRp5I
nexora_run=$(mktemp -d /tmp/nexora-task34-final.XXXXXX)
for config in Debug Release; do
  lower=$(printf '%s' "$config" | tr '[:upper:]' '[:lower:]')
  python3 tools/validate_serial.py --cc /usr/bin/cc --cxx /usr/bin/c++ --config "$config" \
    verify --prefix "$nexora_kokkos_run/kokkos-$lower" --work-dir "$nexora_run/build" \
    --logs "$nexora_run/logs-$lower" --jobs 2 --require-network-blocked || exit
done

cmake -S . -B "$nexora_run/build/no-tests" -G Ninja \
  -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNEXORA_ENABLE_MPI=OFF \
  -DKokkos_DIR="$nexora_kokkos_run/kokkos-release/lib/cmake/Kokkos"
cmake --build "$nexora_run/build/no-tests" --parallel 2
python3 tools/check_dependencies.py --graph "$nexora_run/build/no-tests/nexora-target-graph.json"
ctest --test-dir "$nexora_run/build/no-tests" -N
nm -C "$nexora_run/build/no-tests/src/participants/reference/libnexora_participants_reference.a"
```

开发复验的命令、退出码、JUnit、Decimal、探针和工作区文件摘要存于 `results/task34-20261111/`。复审清单使用 `51dd068dc339ebb33cadb44968fc464e1ca4481a` 和任务 1/2 的通过清单标识共享基线。
