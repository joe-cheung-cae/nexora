# Serial CPU 基础设施验收

本页记录早期 Serial 基线。现行单求解器生命周期和状态发布契约见 [Kokkos 数据与执行设计](kokkos-design.md#7-单求解器的同步推进与状态发布)。

本记录对应[底层开发计划任务 7](foundation-development-plan.md#任务-7-汇总验收并固定交付边界)，涵盖 `P1-10/11` 的 Serial 子项。验收对象为 C++20、Ninja、exact Kokkos 5.2.2 Serial-only 的 Nexora 源码及搬迁安装。同步 Reference 窗口验证数值、时间和事务，独立的四实体 `StructureState` 验证数据布局、CSR 和原子重排。两套夹具均使用实际的 Kokkos Views、字段授权和本地传输。 [架构总览](overview.md#2-当前实现源码事实) · [底层开发计划](foundation-development-plan.md#本阶段交付边界)

## 任务 1–6 验收对应

下表将各阶段的历史验收对应到累计工作树中的可执行检查。各阶段的验证日志随原记录保留。任务 7 的执行证据按本轮 profile 登记，并以累计源码摘要和可重放补丁标识。

| 前置任务 | 累计复跑检查 | 可观察结果 |
| --- | --- | --- |
| 1 固定基线与 CI | `tools/validate_serial.py verify` 核验本地 Kokkos 安装的 profile 与文件摘要，分构建类型和精度执行配置、构建、依赖检查、CTest、Decimal 及搬迁包测试。`nexora_install_consumer` 从空前缀安装后搬迁，另建三个消费者。 | 检查 `nexora::simulation` 的控制 API 和 `nexora::data_schema` 的纯语义消费。`nexora::data` 消费者经导出 target 获得 Kokkos 并执行算子。配置负例覆盖缺包、错版本、旧 GPU 参数、C++17、Kokkos 设备/标准/编译器不符及消费端精度冲突。编译负例检查精度宏冲突。 [Serial CI 验证](serial-ci-validation.md) · [搬迁与配置负例](../../tests/package/VerifyInstall.cmake) · [消费者与导出断言](../../tests/package/consumer/CMakeLists.txt) |
| 2 实体顺序、双射与 CSR | `nexora_entity_topology_tests`、`nexora_field_binding_tests`、`nexora_transport_tests`、`nexora_transport_receive_tests`。 | 独立字面量核对稳定 ID、双向排列和稳定 ID 边表。覆盖分域 CSR、空图、空行、长行、重复或缺失 ID、跨域、非互逆映射、非法 offsets/列索引、旧布局与容量溢出。绑定覆盖 rank-1、两种 rank-2 轴序、`LayoutStride` subview、rank-3 和空字段。传输覆盖空 pack、两种 rank-2 轴序、跨布局 ID 对齐与 strided ghost 目标。接收负例校验目标缓冲保留原值，成功时也检查 subview 外的数据未改动。 [实体与拓扑验证](entity-topology-validation.md) · [字段绑定](../../tests/data/field_binding_tests.cpp) · [本地接收](../../tests/data/transport_receive_tests.cpp) |
| 3 原子重排 | `nexora_structure_reorder_tests` 直接编译 `tests/validation/structure` 中的 `StructureState`。 | 原序、重排、逆排分别按独立期望核对 ID、`x/q`、被动历史和 CSR，保留行内稳定 ID 顺序。纯重排推进 layout 并更换受影响字段的 storage incarnation，原 topology、geometry、accepted 版本和时钟继承。实际无效输入、版本耗尽及非有限提交值检查原 bank 及签发能力。验证旧绑定拒绝新签发，旧顺序和旧 CSR 保留原布局，旧快照穿过重排、下一次值提交及状态析构仍可读。跨布局接收把重排前的 packed 快照按 ID 解包到重排后的 ghost。 [结构与执行验证](structure-execution-validation.md) · [重排测试](../../tests/data/structure_reorder_tests.cpp) |
| 4 显式实例和字段交接 | `nexora_field_group_tests`、`nexora_transport_tests`、`nexora_transport_receive_tests`、`nexora_reference_transport_tests`、`nexora_reference_tests` 及 Decimal 检查。 | 实际 reference 算子沿传入的 Serial 实例使用 native Views，主机诊断在相关 fence 后读取。整组授权检查 managed subview 别名、重复请求、多读者、独占写、冲突时权限原样保留及相反顺序的并发申请。传输检查 caller 提前释放后两位独立消费者仍能读取，最后一次释放使值与 ID 的 owner 各销毁一次。校验不通过时保留接收目标和 accepted 状态。 [结构与执行验证](structure-execution-validation.md#reference-执行与授权) · [整组授权测试](../../tests/data/field_group_tests.cpp) · [传输保活测试](../../tests/data/transport_tests.cpp) |
| 5 事务、历史与资源 | `nexora_reference_resource_tests`、`nexora_reference_tests`、`nexora_reference_transport_tests`、`nexora_simulation_tests`。 | begin、advance、receive、prepare 故障逐一检查 accepted 数值、ID、历史、控制、诊断、stamp、allocation identity、时钟与预算。恢复失败使 Simulation 拒绝后续试算和提交。核对连续推进与重放、旧 layout/attempt/storage incarnation 缓存拒绝、达到上限时分配次数为零、最后一个旧快照释放后同一请求重试、部分初始化回收和显式关闭后的延迟消费。 [Reference 资源验证](reference-resource-validation.md) · [故障与预算测试](../../tests/reference/reference_resource_tests.cpp) |
| 6 应用装配 | `nexora_cli_case_suite`、`nexora_install_consumer` 内的搬迁 CLI suite，以及 `--version`、`inspect-build` 和无效命令 smoke。 | 合法 case 的 `validate`、`plan`、`run` 共用预检。检查计划的 built/available/validated、精度、布局及访问边界。`cli-nonbinary-tail.case` 的 accepted 时间、版本、步数和 `x/q` 摘要与直接调用 `Simulation`/Reference 的驱动逐字节一致。非法版本、键、数值、时间、步长、精度及执行请求返回非零，三入口的同一负例诊断一致。 [case 格式与验证](reference-case.md) · [CLI suite](../../tests/package/RunCliSuite.cmake) · [直接库调用驱动](../../tests/reference/cli_oracle.cpp) |

## 任务 7 汇总判据

- **构建、包及边界。** 对四组本地 profile 分别核对命令退出状态、运行 JUnit 和搬迁包 JUnit，预期每组 19 个运行用例及 1 个包用例。包内三个独立消费者另行计数。运行 JUnit 与预期测试名称逐项比对。`verify` 的网络记录包含 8 项 TCP/UDP 操作和另列的 IPv4/IPv6 回环 bind，禁网记录核对全部实际结果。 [CI 验证脚本](../../tools/validate_serial.py)
- **架构。** `check_dependencies.py --graph` 同时执行 CMake 图、源码与公开头 include 边界检查及规则自检。安装消费核对导出的直接依赖，排除旧 exec targets、旧数值容器头以及实体拓扑、签发器和 Reference 等私有头。CMake 图允许 CLI 装配 Reference participant，源码检查将方法状态与预算实现限制在私有层。 `data_schema/core` 保持纯语义，`participant_api` 与 `simulation` 经语义层控制，Kokkos 由数值层和应用生命周期管理。 [依赖检查器](../../tools/check_dependencies.py) · [安装边界](../../tests/package/VerifyInstall.cmake) · [公开 target 规则](overview.md#4-批准后-cmake-target-依赖图)
- **CLI 正常入口。** 测试 suite 同时覆盖匹配验证证据的普通运行，以及未验证构建的拒绝与测试专用 `--fixture` 入口。任务 7 的搬迁 CLI 正常运行证据包含 `plan` 的 `validated=true` 和普通 `run` 的退出码及摘要比对。 [运行门禁](reference-case.md#计划和运行) · [CLI suite](../../tests/package/RunCliSuite.cmake)
- **数值。** `nexora_reference_tests` 运行 16、32、64 步及 `dt=0.1` 的非二进制步长。`check_reference_precision.py` 用 80 位 Decimal 独立核验离散解、守恒与收敛，应用原精度门槛，观测阶仍须在 0.9–1.1。CLI 的尾步 case 另核对实际装配路径和输出。 [数值验证判据](real-numerics-validation.md#验证基准与误差界) · [Decimal 检查器](../../tools/check_reference_precision.py)
- **结构计量。** 不规则四实体夹具的一次完整执行记录 allocation、copy、fence 及峰值在途载荷。 [结构夹具](../../tests/data/structure_fixture.hpp)

## 结构计量口径

测量复用任务 3 的四实体夹具。输入稳定 ID 顺序为 `10,30,20,40`，目标为 `20,10,40,30`，CSR 为 4 行 5 条边，包含空行。输入还包含 `x/q/history`、版本 stamp、字段描述和提交值。编译后的输入来自夹具和测量源码的固定字面量。运行包括构造输入和 accepted bank、打包、重排、拒绝旧 stamp 后合法接收、提交值、销毁状态对象、释放仍可读的旧快照及最后一个传输载荷、清理输入并 finalize Kokkos。每个 bank 按固定独立期望核对稳定 ID、CSR 边序、字段和版本，接收核对 ghost 值。 [夹具](../../tests/data/structure_fixture.hpp) · [测量程序](../../tests/data/serial_irregular_measurement.cpp)

- **分配与峰值。** `Kokkos::Tools::Experimental` 的分配和释放回调按同一指针配对，并校验标签、内存空间和字节数。报告用单调分配 ID 留存配对，避免记录进程地址。字节为回调报告的 View 分配 span，包含 Kokkos 的 span 取整。例如 `csr.indices` 和 `reorder.indices` 各有 5 个 4 B 的 `LocalIndex`，元素共 20 B，回调各报 24 B。峰值是任一事件后的未释放字节之和，同时单列 packed 传输载荷峰值。Serial 的算子和传输同步执行，这里的在途载荷指仍由候选、快照、映射或传输消费者持有的存储。最后一个消费者完成后，回调账目归零。
- **复制与同步。** `deep_copy` 按 begin/end 配对，字节为回调给出的目标字节，含 ghost 初值的标量填充。另按标签记录 `parallel_for`，包括重排 gather、CSR 构造、pack/unpack 和 Kokkos 标量填充 kernel。`EntityOrder::create` 和 CSR 构造的主机写入、`match_entity_ids` 的主机映射写入归入主机循环，未纳入 deep copy 字节。旧 stamp 的接收调用在预检返回，未到达分配、拷贝、kernel 或 fence。显式 fence 按源码标签识别，Kokkos 内部 fence 单列。两类 fence 都核对 begin/end 配对，工具设置不请求额外全局 fence。
- **排除项。** 分配统计排除 allocator 头与额外对齐开销、STL/元数据、输入和观测器自己的主机存储及进程 RSS。copy 统计排除主机循环、kernel 的读写流量和物理传输量。`StructureState` 的这条路径不使用 `ResourceBudget`。任务 5 的额度与最后消费者验证由 `nexora_reference_resource_tests` 单独提供。 [资源预算验收](reference-resource-validation.md)

## 本地复跑结果

Apple M2 Pro arm64，12 核，16 GB，macOS 27.2.1 build 26B131，Darwin 27.2.0，AppleClang `21.0.0.21000401`，CMake `4.4.3`，Ninja `1.13.2`，Python `3.14.7`。以原有 exact Kokkos 5.2.2 Serial-only 静态安装重新验证每个安装文件摘要，MPI 关闭。Kokkos 源 commit 为 `8e15454876298832f013ebcaf08d53825b5bda40`，归档 SHA-256 为 `d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f`。两个 `verify` 进程的前后各 10 项网络探针均返回 `EPERM`，包括 IPv4/IPv6 回环 bind。 [依赖来源与许可](../../third_party/README.md#kokkos-provenance-record)

| Profile | 配置至普通搬迁 CLI 对照完成 | CTest 通过/失败/跳过 | 独立消费者 通过/失败/跳过 | 计量执行 通过/失败/跳过 |
| --- | --- | --- | --- | --- |
| Debug float | `2026-11-11T14:14:52+08:00` 至 `2026-11-11T14:15:59+08:00` | 20/0/0 | 3/0/0 | 2/0/0 |
| Debug double | `2026-11-11T14:15:22+08:00` 至 `2026-11-11T14:15:59+08:00` | 20/0/0 | 3/0/0 | 2/0/0 |
| Release float | `2026-11-11T14:14:52+08:00` 至 `2026-11-11T14:16:06+08:00` | 20/0/0 | 3/0/0 | 2/0/0 |
| Release double | `2026-11-11T14:15:28+08:00` 至 `2026-11-11T14:16:06+08:00` | 20/0/0 | 3/0/0 | 2/0/0 |

四组运行 JUnit 的 19 个名称与任务 6 一致，另有 1 项搬迁包测试。四组依赖图、允许与拒绝边界自检、安装导出及负例、80 位 Decimal 判据均通过。Decimal 文件保存 16/32/64 步误差、守恒、非二进制步长及观测阶，float 和 double 的两段观测阶均在 1.009–1.020。另在带空格的 case 路径上执行搬迁后的普通 `validate`、`plan`、`run`，每项退出 0。四组 `plan` 均报告 `validated=true` 与 `validation_evidence=task6-local-serial`，普通 `run` 均完成 3 步，stdout 与直接库驱动逐字节一致。命令、退出码、输入和输出摘要见 `matrix-summary.json` 与分组日志。

| 历史一次夹具执行（2026-11-11，迁移前） | float | double |
| --- | ---: | ---: |
| 分配 / 释放事件 | 28 / 28 | 28 / 28 |
| 累计分配 / 释放字节 | 656 / 656 B | 832 / 832 B |
| `deep_copy` 次数 / 字节 | 10 / 208 B | 10 / 320 B |
| `parallel_for` 启动 | 7 | 7 |
| 显式 fence / Kokkos 内部释放 fence（历史测量） | 15 / 28 | 15 / 28 |
| 重排阶段峰值在途 View 字节 | 496 B | 608 B |
| packed View 峰值 | 48 B | 64 B |
| finalize 后仍持有的 View 字节 | 0 B | 0 B |

各组两次计量的原始 JSON 逐字节一致，同精度的 Debug 与 Release 也一致。`ledger-replay.json` 逐笔重放 allocation ID 配对，并记录峰值时仍持有的 View。重排峰值包含旧 bank、packed 快照、新 bank 和重排临时量。关闭状态对象时旧快照仍可读，最后的 packed 消费者释放后归零。Apple 链接器对结构测试和计量目标报告重复 `libnexora_data.a`，四组链接和执行通过。

`2026-11-11T14:15:38+08:00` 至 `2026-11-11T14:16:08+08:00`，在干净 Git archive 上应用候选补丁并执行默认 Debug float 禁网 `verify`。20 项 CTest、3 个独立消费者、Decimal 和依赖检查均通过。默认入口记录计量未请求，计量可执行文件未生成。日志位于 `replay-default/`。

## 独立复审

**PASSED**。`2026-11-11T14:15:19+08:00` 至 `2026-11-11T14:15:58+08:00`，独立复审先将已校验的 Kokkos Debug/Release 安装搬迁到带空格的前缀，再从干净目录禁网复跑四组。各组 20/20 CTest、3/3 独立消费者，无失败或跳过。普通搬迁 CLI 三入口均退出 0，`plan` 为 `validated=true`，3 步输出与直接库驱动逐字节一致。schema 消费编译命令无 Kokkos include，typed data 消费者执行 Kokkos 算子，Decimal、依赖图及导出负例通过。

独立复审按 allocation ID 重放各组事件并再次执行计量，计数、字节与本地原始报告一致。独立源码清单 SHA-256 为 `ab056f1f75301bd4c241ce5cce647200bd6c1b10716516a7c3568657afbb9197`，复跑前后相同。清单所含 61 个路径与本地冻结的 105 个构建输入逐项吻合。`2026-11-11T14:15:46+08:00`，复审从干净 Git archive 重放候选补丁并比对 76 个工作树路径。独立证据归档于 `independent-review/`，候选补丁和清单归档于 `candidate/`。

## 交付边界

| 能力 | 范围与状态 |
| --- | --- |
| 本地 Serial | Debug/Release × `float`/`double`，使用各构建类型对应的 exact Kokkos Serial 包，MPI 关闭。Reference 的 `real` 在构建及安装时固定，时间继续使用 `double`。配置、编译、执行、数值和安装消费分别有各自的检查。 |
| 托管工作流 | Linux GCC、Linux Clang、Windows MSVC 与 Linux MPI smoke 已配置。托管 job 执行证据待各 runner 实际运行后登记。历史本地 MPI smoke 在启动 ranks 前的 socket bind 被沙箱拒绝。 [CI 状态](serial-ci-validation.md#受控-runner-实测) |
| 其他执行和通信 | OpenMP、CUDA/HIP、设备异步交接、MPI 数值通信和 MPI halo 属于独立配置、实现和实机验收。当前 ghost/halo 名称用于进程内传输夹具。 [构建和实机规则](build-and-validation.md) · [传输契约](kokkos-design.md#6-transportview-与通信完成) |
| 结构与事务 | `StructureState` 验证单一 owned 域的纯排列及后续值发布。Reference 窗口使用固定本地顺序，串行推进单参与者事务。实体删除及 ID 退役记录由未来可变状态的所有者负责。多参与者一致提交、分布式拓扑和生产缓存各有独立验收。 [结构状态](structure-execution-validation.md#结构状态与安全点) · [Reference 状态](reference-resource-validation.md#状态与缓存) |
| 资源与恢复 | 资源额度按受控的 bank、共享 ID 和 packed buffer 载荷及快照记录计数。运行时元数据、host 临时校验结构和进程总内存需单独计量。故障恢复在进程内保持 accepted 状态。持久 checkpoint 与新进程 R0 重启保留在 `IO-001`，R1–R3 延期。 [资源口径](reference-resource-validation.md#保活与额度) · [持久恢复规范](checkpoint-restart.md#restart-levels) |
| 完整 Phase 1 与产品求解 | 现行 Phase 1 仍要求 SPH/FVM 代表性切片、适用的设备实机数值与互操作、生命周期和端到端性能。真实求解器、物理耦合和外部引擎各按后续任务验收。 [Phase 1 出口](phase-1-design-review.md#1-当前基础与-phase-1-出口) · [验收矩阵](phase-1-design-review.md#acceptance) |

## 复跑命令

在具备网络隔离的环境中执行。`NEXORA_KOKKOS_LOCAL` 指向包含已供应 `kokkos-debug` 与 `kokkos-release` 安装的目录。依赖前缀须和编译器、构建类型及当前锁匹配。 [依赖供给命令](serial-ci-validation.md#重跑)

```sh
nexora_run=$(mktemp -d /tmp/nexora-task7.XXXXXX)
for config in Debug Release; do
  lower=$(printf '%s' "$config" | tr '[:upper:]' '[:lower:]')
  python3 tools/validate_serial.py --cc /usr/bin/cc --cxx /usr/bin/c++ --config "$config" \
    verify --prefix "$NEXORA_KOKKOS_LOCAL/kokkos-$lower" --work-dir "$nexora_run/build" \
    --logs "$nexora_run/logs-$lower" --jobs 2 --require-network-blocked --measure-irregular || exit 1
  for precision in float double; do
    build="$nexora_run/build/$lower-$precision"
    cli="$build/tests/package/$config/relocated/bin/nexora"
    case_file="$PWD/cases/reference/cli-nonbinary-tail.case"
    "$cli" validate "$case_file" || exit 1
    "$cli" plan "$case_file" > "$build/plan.out" || exit 1
    rg -x 'validated=true' "$build/plan.out" || exit 1
    "$cli" run "$case_file" > "$build/relocated.out" || exit 1
    "$build/tests/reference/nexora_reference_cli_oracle" > "$build/direct.out" || exit 1
    cmp "$build/relocated.out" "$build/direct.out" || exit 1
  done
done
```

`--measure-irregular` 仅在显式指定时构建并运行测试目录的 `nexora_serial_irregular_measurement`。可单独用 `cmake --build "$build" --target nexora_serial_irregular_measurement` 和 `"$build/tests/data/nexora_serial_irregular_measurement"` 重测已配置的 profile。观测 JSON 记录有序事件和摘要，`irregular-measurement.json` 记录输入源码、可执行文件和观测结果 SHA-256、执行命令及状态。

## 交付基线与复原

本轮结果目录为 `results/task7-20261111/`。目录由 Git 忽略，交付时可整体导出。`evidence-index.json` 记录 profile 命令日志、JUnit、Decimal 输出、Kokkos 供应清单与网络探针、搬迁 CLI 与直接库调用的 stdout、计量 JSON 和补丁的摘要。`prior-baselines/` 保存已核对的前置清单链。复原源码只需基线与补丁。

Git 基线为 `51dd068dc339ebb33cadb44968fc464e1ca4481a`。任务 6 的 PASSED 清单 SHA-256 为 `5eaa3aa4216e299ce2bc36278e414812ed166b21f03314139ce841264d1b9663`。`review-manifest.json` 记录相对任务 6 的改动、累计工作树及输入摘要。`worktree.patch` 相对 Git 基线生成，包含新增文件。可用以下命令在空目录复原，应用前先与清单中的 `worktree_patch_sha256` 比对。

```sh
base=51dd068dc339ebb33cadb44968fc464e1ca4481a
patch_file="$PWD/results/task7-20261111/worktree.patch"
replay=$(mktemp -d /tmp/nexora-replay.XXXXXX)
shasum -a 256 "$patch_file"
git archive "$base" | tar -xf - -C "$replay"
git -C "$replay" apply --check "$patch_file"
git -C "$replay" apply "$patch_file"
```

复原后可逐项与 `review-manifest.json` 的 `sha256` 及 `tested-source.json` 比对。`patch-replay.log` 保留对干净 Git archive 的实际校验和应用结果。

## 测试树迁移后的结构计量

`2026-11-26T14:30:30+08:00`，上述 Apple M2 Pro / AppleClang 工具链的 `build/sph-must-fix-20261126/debug-float` 与 `release-double` 显式构建并成功执行 `nexora_serial_irregular_measurement`。本次仅运行该计量目标，不替代完整 CTest 验收。结构夹具迁入测试树，删除故障注入专用 fence；CSR 直接保留传入的只读 EntityOrder，不再额外复制一份顺序。

| 当前一次执行 | Debug float | Release double |
| --- | ---: | ---: |
| 分配 / 释放事件 | 26 / 26 | 26 / 26 |
| 累计分配 / 释放字节 | 592 / 592 B | 768 / 768 B |
| `deep_copy` 次数 / 字节 | 9 / 176 B | 9 / 288 B |
| `parallel_for` 启动 / 结束 | 7 / 7 | 7 / 7 |
| 显式 fence / 内部释放 fence | 12 / 26 | 12 / 26 |
| 峰值在途 View 字节 / 阶段 | 432 B / reorder | 576 B / commit |
| packed View 峰值 / 最终存活字节 | 48 B / 0 B | 64 B / 0 B |

独立源码账本以 `r=sizeof(real)` 计：create 6、pack 2、reorder 10、receive 4、commit 4 次分配，共 `416+44r` 字节。七个四实数复制/填充加两个四 uint64 ID 复制给出 9 次、`64+28r` 字节。Kernel 为 pack、两次字段 gather、两次 CSR、ghost 标量填充及 unpack。十二个显式 fence 分布为 create 1、pack 2、reorder 2、receive 4、commit 3；释放 26 个分配产生 26 个 Kokkos 内部 fence，全部 begin/end 配对。

重排峰值保留原 bank、新 bank、payload、双向排列及临时 CSR，为 `320+28r`。提交峰值保留三个字段 bank 和 payload，其中新旧提交 bank 共享重排后的 CSR，为 `256+40r`。实际峰值取两者最大值，因此 double 的峰值阶段是 commit。上方旧表保留历史实测记录。
