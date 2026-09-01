# tcgen05.mma.ws 吞吐量测试结论

GB200（`sm_100`，152 SM，2062 MHz）上 1SM dense `tcgen05.mma.ws` 的流水线吞吐。数字是实测拟合，**不是** NVIDIA 官方 II。默认引用 **TS、最大 pipeline depth**。

**II（Initiation Interval，发起间隔）**：流水线稳态下，相邻两条 MMA 间隔多少个 GPU cycle（cycle/MMA）。对 depth 拟合 `CyclesPerMMA ≈ II + b/DEPTH` 时，常数项就是稳态 II；`b/DEPTH` 是 batch 头尾 fill / lastuse / commit / wait 的摊销。II ≠ latency（单条从发到结果可用，通常远大于 II）。

---

## TLDR

1. **只用单个 `b0` collector 的 no-reuse / discard 基线下，WS 相对 AS 的收益只在小 M。** 每条 MMA 都从 SMEM 重新 fill B。BF16 TS、N=256、depth=256：64×256 上 `tcgen05.mma.ws` 为 6857 FLOPs/cyc，AS `tcgen05.mma` 为 4080（半 datapath，II 仍是 N/2），**WS 约 1.68×**；128×256 上 WS 7573 vs AS 8161，**WS 约 0.93×，略慢**。M=32 没有同机 AS 对照；单 `b0::discard` 时它和 M=64 同一条 II，FLOPs 是 64×256 的一半（3429）。这不是 no-reuse 场景的上限；显式指定并交替使用 `b0/b1::discard` 后，M=64 也能打满（见第 3 条）。
2. **单 collector 时，B reuse 可以打满 1SM peak，且相对 discard 的加速在小 M 更大。** `fill / use / lastuse` 之后：64×256 / 128×256 BF16 分别到 8128 / 8160（99%+）。相对单 b0 discard，M=32 约 **2.3–3.1×**（32×256 为 2.34×），M=64 约 1.2–1.7×，M=128 约 1.1–1.3×。省略 qualifier = `b0::discard`。
3. **M=64 即使每条 B 都不 reuse，也可以用两个 collector ping-pong 打满。** 让同一个 D accumulator 交替发 `b0::discard` / `b1::discard`，BF16 TS 的 N=64/128/256 分别为 **16.50 / 32.50 / 64.50 cyc/MMA**，对齐理想 `N/4`。N=256 即使用两个不同的 B SMEM descriptor 仍为 64.50，排除了同地址特例。只把 D 拆成 D0/D1、仍全走 b0 没有效果。

---

## 1. 测试 setting

| 项 | 范围 |
|---|---|
| 指令 | `tcgen05.mma.ws`，`cta_group::1` only |
| Format（WS 编号，和 AS bench 0–5 不同） | 0=BF16 K=16，1=TF32 K=8，2=E4M3 K=32，3=S8 K=32，4=F4 K=64 |
| M | 32, 64, 128 |
| N | 64, 128, 256 |
| AB | SS（A 在 SMEM）、TS（A 在 TMEM） |
| Collector | discard（省略 qualifier = `b0::discard`）、双 discard ping-pong（`b0/b1::discard`）、reuse（`b0::fill / use / lastuse`） |
| 不测 | MX、2SM、`b2/b3`、idesc B-shift、正确性校验 |

每条 MMA 的工作量是 `2MNK` FLOPs。`CyclesPerMMA` = `cycles / (DEPTH × 1000)`。Reuse 每个 batch 的编译期序列（`DEPTH ≥ 2`）：

```
mma_fill(0)
mma_use(1)  × (DEPTH - 2)   // #pragma unroll
mma_lastuse(1)
commit + mbarrier wait
```

每个 batch 重新 `fill` 是 **本 kernel 的写法**，不是测出来的硬件事实：batch 末尾用了 PTX 的 `lastuse`（读完即丢），而且没有做过「commit 之后不 fill、直接 `use`」的对照。PTX 只要求 `use` 时「先前的 fill 仍然 valid」，没有写 commit/mbarrier 会清 collector。

数据文件：

- `ws_tput.csv`：discard，450 点（无 `Collector` 列，早期 sweep）
- `ws_reuse.csv`：reuse，450 点
- `as_gb200_m64_128.csv`：同机 AS `tcgen05.mma` 1SM 对照（M=64/128）

复现：

```bash
python3 benchmark_ws.py 0 1 2 3 4 --mode all --collector discard -o ws_tput.csv
python3 benchmark_ws.py 0 1 2 3 4 --mode all --collector reuse -o ws_reuse.csv --overwrite

make umma_ws_tput.out MMA_FORMAT=0 MMA_M=64 MMA_N=256 MMA_K=16 \
    MMA_DEPTH=256 CTA_GROUP=1 AB_LAYOUT=1 WS_COLLECTOR=1
./umma_ws_tput.out
```

`umma_ws_tput.out` 不把宏算进依赖；换配置必须 `make clean`。`benchmark_ws.py` 每个点都会 clean。内核 `umma_ws_tput.cu`，`WS_COLLECTOR=0/1` 是编译期分支。双 discard collector 的复现命令：

```bash
make -B umma_ws_tput.out MMA_FORMAT=0 MMA_M=64 MMA_N=256 MMA_K=16 \
    MMA_DEPTH=256 CTA_GROUP=1 AB_LAYOUT=1 WS_COLLECTOR=0 \
    WS_DISCARD_COLLECTORS=2 WS_B_DESCRIPTORS=2
./umma_ws_tput.out
```

`WS_B_DESCRIPTORS=2` 让 b0/b1 从两块不同的 SMEM B tile 读取；设为 1 则两者使用同一个 descriptor。D0/D1 轮转只用于下文的一次性消融，当前 bench 固定使用单 D accumulator。

---

## 2. 峰值模型和 N/8

官方 dense peak（每 SM）：BF16 8192 / TF32 4096 / E4M3·S8 16384 / F4 32768 FLOPs/cycle，即 **`512 × K`**。

算满这条峰值，发起间隔必须是

```
II = 2MNK / (512K) = MN / 256
```

| M | 模型 II | 含义 |
|---|---|---|
| 32 | **N/8** | 相对 M=64 每条只做一半输出，算满峰值只需一半周期 |
| 64 | **N/4** | |
| 128 | **N/2** | 和 AS M=128 的经验 II 相同 |

一种和 idesc（N 按 8 列编码）对得上的读法——**不是 spec**：N 按 8 列一组推进，M=32/64/128 分别是每组 1/2/4 cycle，所以 II = `1×(N/8)`、`2×(N/8)`、`4×(N/8)`。

---

## 3. Discard：每条都重新 fill B

省略 collector qualifier 时，M=32 和 M=64 的 II **完全相同**（B 是 K×N，和 M 无关）。32×256 与 64×256 的总 cycle 可以对到同一条数。

拟合（TS）：

| M | 稳态 II | N=64 / 128 / 256 |
|---|---|---|
| 32、64 | **N/4 + 12** | 28 / 44 / 76 |
| 128 | **N/2 + 10** | 42 / 74 / 138 |

**`+12` / `+10` 是 discard 相对计算 II 多付的、几乎不随 N 变的常数**，单位 cycle/MMA。reuse 之后正好被削掉：M=64 三条 N 的 discard−reuse 都是 **11.96**，M=128 都是 **9.96**。解读：每条 MMA 都 `b0::discard`、从 SMEM 再 fill 一遍 B，和计算不能完全重叠，于是叠在计算 II 上。不是 latency，也不是文档里的字段。

为何 12 和 10 差 2：没有 spec。B 是 K×N、与 M 无关，更长的 M=128 计算窗（N/2）可能多重叠掉 ~2 cycle。AS 把同类 B 流量藏进 `N/2`，WS discard 把它露在外面。

M=32 的 `N/4+12` **不是**「计算 II（N/8）+ 12」。32 和 64 的 discard II 完全一样，说明此时被 **B 路径**卡住（B 与 M 无关），地板就是 M=64 那条 `N/4+12`。reuse 后 M=32 才落到 N/8，所以它比 discard 快的不止 12 cycle（32×256 快约 44）。

周期曲线与 format 无关，FLOPs 只随 K 变。因此 M=32 discard 的 FLOPs 正好是同 N 下 M=64 的一半：同样的 II，一半的 `2MNK`。

相对 AS（同机 BF16 TS，N=256，depth=256）：

| Shape | AS II / FLOPs | WS discard II / FLOPs | vs AS |
|---|---|---|---|
| 64×256 | 128.5 / 4080 | 76.5 / 6857 | **1.68×** |
| 128×256 | 128.5 / 8161 | 138.5 / 7573 | **0.93×** |

M=64 AS 走半条 datapath、II 仍是 `N/2`，所以只有 ~4080。WS discard 已经比它快，但还没到 8192。

SS 相对 TS（discard）：M=32 几乎无差；M=64 大约 **+4**；M=128 仅 N=64 大约 **+6**。

### 3.1 双 discard collector ping-pong

单 collector 的 `N/4+12` 不是不可避免的非-reuse 成本。交替指定两个 collector：

```text
tcgen05.mma.ws...collector::b0::discard [D], [A], B0, idesc, p
tcgen05.mma.ws...collector::b1::discard [D], [A], B1, idesc, p
tcgen05.mma.ws...collector::b0::discard [D], [A], B2, idesc, p
tcgen05.mma.ws...collector::b1::discard [D], [A], B3, idesc, p
...
```

每条仍是 `discard`，没有 `fill -> use`，因此不是 B reuse。推断是 b0 参与当前 MMA 时，b1 可以接收下一条 B，反之亦然；两条 fill 路径的开销与 M=64 的计算窗重叠，aggregate II 回到 `N/4`。这是实测解释，不是 PTX 对 microarchitecture overlap 的保证。

BF16 TS、depth=256、单 D accumulator：

| N | b0 only | b0/b1 discard | 理想 II | 双 collector FLOPs/cyc |
|---:|---:|---:|---:|---:|
| 64 | 28.4490 | **16.4979** | 16 | 7945 |
| 128 | 44.4533 | **32.4951** | 32 | 8067 |
| 256 | 76.4493 | **64.4962** | 64 | 8129 |

N=256 消融：

| D accumulator | collector | B descriptor | II |
|---|---|---|---:|
| D0 | b0 | same | 76.4493 |
| D0/D1 | b0 | same | 76.6994 |
| D0 | b0/b1 | same | **64.4962** |
| D0/D1 | b0/b1 | same | **64.4974** |
| D0 | b0/b1 | distinct B0/B1 | **64.4974** |

所以需要拆的是 **B collector**，不是 accumulator。单 D 保留正常的 K 维累加依赖；第一条 `enable-input-d=false`，后续为 true。由于 MMA 是异步操作，真实 kernel 仍须按其完成机制保护 SMEM B stage 的生命周期，不能因使用 `discard` 就立刻覆盖源数据。

---

## 4. Reuse：露出计算 II

`fill / use / lastuse` 之后，B-fill 税被摊进 batch 头尾，稳态 II 落到第 2 节的模型：

| M | reuse II | N=64 / 128 / 256（F4 depth=1024，最贴模型） |
|---|---|---|
| 32 | **N/8** | 9.17 / 16.17 / 32.18 |
| 64 | **N/4** | 16.15 / 32.15 / 64.15 |
| 128 | **N/2** | 32.15 / 64.15 / 128.15 |

五个 format 的 cycle 彼此对齐（BF16/TF32 最大 depth 只有 256，小 N 还剩 ~0.5 cycle；E4M3/S8/F4 更深，更贴 `N/2^k`）。

32×64 没有完全落到 8：BF16 depth=256 为 9.52（peak 的 84%），F4 depth=1024 为 9.17（87%）。大 N 已经 98–99.9%。

大 shape 贴满 1SM peak（TS，最大 depth）：

| Format | peak | 64×256 reuse | 128×256 reuse |
|---|---|---|---|
| BF16 | 8192 | 8128（99.2%） | 8160（99.6%） |
| TF32 | 4096 | 4064（99.2%） | 4080（99.6%） |
| E4M3 / S8 | 16384 | 16320（99.6%） | 16352（99.8%） |
| F4 | 32768 | 32691（99.8%） | 32730（99.9%） |

---

## 5. Reuse 相对 discard 的加速

加速比 = discard II / reuse II。小 M、小 N 更大，因为 discard 多付的 B-fill 占 II 的比例更高。**不是** M=32 的 datapath 突然变宽。

BF16 TS depth=256（其余 format 同 cycle、同加速比，F4 因 depth 更深略高）：

| M×N | discard II | reuse II | 加速 | FLOPs discard → reuse |
|---|---|---|---|---|
| 32×64 | 28.45 | 9.52 | **2.99×** | 2303 → 6886 |
| 32×128 | 44.46 | 16.56 | **2.68×** | 2948 → 7915 |
| 32×256 | 76.46 | 32.63 | **2.34×** | 3429 → 8035 |
| 64×64 | 28.46 | 16.50 | 1.72× | 4606 → 7945 |
| 64×128 | 44.46 | 32.50 | 1.37× | 5897 → 8065 |
| 64×256 | 76.46 | 64.50 | 1.19× | 6857 → 8128 |
| 128×64 | 42.46 | 32.50 | 1.31× | 6174 → 8067 |
| 128×128 | 74.45 | 64.49 | 1.15× | 7042 → 8130 |
| 128×256 | 138.46 | 128.50 | 1.08× | 7573 → 8160 |

全 format、TS、最大 depth：

| M | 加速范围 | 中位 |
|---|---|---|
| 32 | 2.34–3.07× | 2.72× |
| 64 | 1.19–1.74× | 1.37× |
| 128 | 1.08–1.31× | 1.16× |

32×256 五个 format 都是 **2.34–2.37×**，对应 `(N/4+12) / (N/8)`。

---

## 6. 和 AS 比

| Shape | AS | WS discard | WS reuse |
|---|---|---|---|
| 64×256 BF16 | 4080（半 datapath，II=N/2） | 6857 | **8128**（满 peak，II=N/4） |
| 128×256 BF16 | 8161（II=N/2） | 7573 | **8160**（II=N/2，对齐 AS） |

AS 的 M=64 不缩短 II，所以只有一半 FLOPs。WS reuse 的 II 随 M 缩放，三个 M 都能打到同一条 `512K` peak。

---

## 7. SS vs TS（reuse）

SS 付的是 A-descriptor 税，不是 collector：

| N | SS − TS（reuse II） |
|---|---|
| 64 | 大约 **+6～7**（M=32 的 32×64 为 +6，64/128 为 +7） |
| 128 | M=32 大约 +2，M≥64 大约 0 |
| 256 | ~0 |

N 大时 SS/TS 重合；看峰值用 TS 即可。

---

## 8. 还没做 / 不支持

本 bench 只测「同一份 A、同一份 B、连发 DEPTH 条」的发指令速率。下面几项是当时为了隔离 collector 收益主动砍掉的，不是硬件做不到。

| 项 | 是什么 | 为什么没做 |
|---|---|---|
| `b2/b3` 与更深 collector schedule | b0/b1 的 `discard` ping-pong 已验证；还可测试四 collector 或把 `fill/use/lastuse` 与下一块 B 的预取组合。 | 两个 collector 已足以让 M=64 回到理想 II，四 buffer 是否对其他形状有收益尚未测。 |
| idesc `[30:31]` B-shift | 这两位是 `.ws` 专用：「尝试 B reuse 时允许的最大 shift」，0/1/2/3 = 不 shift / 最多 8 / 16 / 32。给卷积滑动窗口用：B 不必整块重填，可以在 collector 里错位再用。本 bench 这两位恒为 0。 | 测的是整 tile 复用，不是 conv 的窗口错位。 |
| 换 A | 真 WS GEMM 是 B 不动、A 按 M/K 流过。本 bench 的 A（SMEM desc 或 TMEM）在 batch 内固定，DEPTH 条都是同一对 A×B。 | 固定 A 才能把 II 差单独归因到 B collector，不和换 A 的 TMEM/SMEM 流量搅在一起。 |
| 正确性 / vs cuBLAS | 只记 `clock64`，不读回 D，不对 `D=A*B`。 | 吞吐 microbench；reuse 的 `use` 是否真算满，是靠 II 落到 `MN/256` 间接看的，没有数值对照。 |
| 稀疏 `mma.ws.sp` | 合法指令，仍是 `cta_group::1`。A 稀疏，K 加倍，额外要 TMEM 里的 sparsity metadata。N 只支持 {64,128}。 | 和 dense 不是同一条指令/同一套 idesc；当时范围是 dense 吞吐。 |
| Collector 跨 commit 保活 | 见第 1 节：本 kernel 用了 `lastuse`，观察不到。 | 没做对照实验，不是结论。 |

**PTX 直接标 Invalid、不是「还没做」：**

- **2SM（`cta_group::2`）**：Table 42 里 `.ws` + CTA group 2 对 f16/tf32/f8/i8 都是 Invalid。指令语法也写死 `.cta_group::1`。
- **MX（`kind::mxf8f6f4` / `mxf4` / `mxf4nvf4`）**：`.ws` 全部 Invalid。WS 没有 block-scale 这条路。
