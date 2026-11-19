# P1-01b SPH 一维密度小算例

本轮定案记录适用于 `sph-density-1d` v1。采用一维归一化三次样条 M₄，固定 `h = 1 m`，计算一次粒子线密度（kg/m）。公式依据 Monaghan 的式 (2.6) 和 (2.9)。算例数据、标准答案及复核程序为 Nexora 原创，采用 BSD 3-Clause License。[论文 DOI](https://doi.org/10.1088/0034-4885/68/8/R01) · [公式原页](https://planets.utsc.utoronto.ca/~pawel/PHYD57/monaghan-sph2005.pdf#page=8) · [Nexora LICENSE](../../LICENSE)

## 输入与标准答案

[冻结数据](sph-density-1d-v1.json)定义位置、质量、两套 CSR 排列、精确结果及误差系数。文件原始字节 SHA-256 为 `2d35b201f8d13e16a2c00868bf98a8c677b9fe48fe378032964fbebd43e3f4a6`。

- 令 `q = |x_i - x_j| / h`。核权重 `W = (2 / (3h)) f(q)`。当 `0 ≤ q < 1`，`f(q) = 1 - 3q²/2 + 3q³/4`。当 `1 ≤ q < 2`，`f(q) = (2 - q)³/4`。当 `q ≥ 2`，权重为零。
- CSR 列举距离 `≤ 2h` 的非自身粒子，保留支持边界的零权重条目。每行的粒子 ID 唯一，两个 owned 粒子之间的边双向存在。每个 owned 粒子另加一次自身项，`ρ_i = m_i W(0) + Σ_j m_j W(q_ij)`，求和只遍历该行。ID 50 的空行只贡献自身项。
- ID 40 是带 boundary 标记的 owned 粒子，按同一公式求值。ID 60 是只读 ghost，可贡献质量和位置。密度输出与全局求和只包含 owned 粒子，求和为计算结果核对值。
- CSR 索引为当前列顺序下的零基局部索引。重排置换行列，保持每行的邻居 ID 顺序与 topology version，推进 layout version。两种布局都必须逐项匹配各自的 ID、offsets 和 indices，并按稳定 ID 比较密度。

手算时，`q = 0, 1/2, 1, 3/2, 2` 的权重依次为 `32, 23, 8, 1, 0` 除以 `48 m`。例如 ID 20 的密度为 `(3×32 + 1×1 + 2×8 + 4×23 + 6×8)/48 = 253/48 kg/m`。五个 owned 粒子的密度和为 `946/48 kg/m`。

## 数值判据

两种布局及各执行配置分别对照同一精确分数。结果须为有限值，并满足 `abs(实际值 - 精确值) ≤ k × ε × abs(精确值)`。`ε` 取 `nexora::real` 在 1 处的机器 epsilon（float 为 `2^-23`，double 为 `2^-52`）。逐粒子密度 `k = 8`，owned 密度和 `k = 16`。

此例的输入与未归一化权重均可用二进制精确表示。按上述多项式直接计算，每个非负项至多经过归一化常数舍入和两次乘法舍入。每行最多再做四次加法，全局和最多再做四次，各路径无消减。按单位舍入误差 `u = ε/2` 逐步累计，`8ε` 和 `16ε` 为这个小算例保留了舍入余量。

## 独立复核

从仓库根目录执行。用全粒子扫描和精确分数重建邻接与答案，计算密度时使用代数等价的截断幂形式。

```sh
python3 -I - <<'PY'
import hashlib
import json
from fractions import Fraction
from pathlib import Path

data = Path("validation/baselines/sph-density-1d-v1.json").read_bytes()
assert hashlib.sha256(data).hexdigest() == "2d35b201f8d13e16a2c00868bf98a8c677b9fe48fe378032964fbebd43e3f4a6"
case = json.loads(data, parse_float=Fraction)
particles = {p["id"]: p for p in case["particles"]}
owned = [i for i, p in particles.items() if p["role"] == "owned"]
source_order = case["initial"]["column_ids"]
h = Fraction(case["h_m"])

def distance_ratio(i, j):
    return abs(particles[i]["x_m"] - particles[j]["x_m"]) / h

neighbors = {i: [j for j in source_order if j != i and distance_ratio(i, j) <= 2]
             for i in owned}
for name in ("initial", "reordered"):
    layout = case[name]
    assert sorted(layout["row_ids"]) == sorted(owned)
    assert sorted(layout["column_ids"]) == sorted(particles)
    offsets, indices = [0], []
    for i in layout["row_ids"]:
        indices.extend(layout["column_ids"].index(j) for j in neighbors[i])
        offsets.append(len(indices))
    assert (offsets, indices) == (layout["offsets"], layout["indices"]), name

def weight(q):
    return (max(2 - q, 0)**3 - 4 * max(1 - q, 0)**3) / (6 * h)

density = {str(i): sum(p["mass_kg"] * weight(distance_ratio(i, j))
                       for j, p in particles.items())
           for i in owned}
expected = case["expected"]
assert density == {i: Fraction(rho) for i, rho in expected["density_kg_per_m_by_id"].items()}
assert sum(density.values()) == Fraction(expected["owned_density_sum_kg_per_m"])
print("SPH density, owned sum and both CSR layouts verified")
PY
```

`2026-11-19T09:59:14+08:00` 用 Python 3.14.7 复核通过。SHA-256、原序和重排 CSR、逐粒子密度及 owned 密度和均与记录精确一致。

P1-07/P1-10 负责 Kokkos 执行、长邻域、大规模、状态重放和性能验收。真实 SPH/DEM 基准仍由 `BASE-001` 单独跟踪。
