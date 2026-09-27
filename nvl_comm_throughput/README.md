# NVLink collective throughput

## 开发 / 测试矩阵（先记录计划，再实现并更新实测）

目标：同一消息量 sweep 下比较 PyTorch/NCCL Host API、TMA、copy engine (CE)。
单机、每 GPU 一个进程；固定 4 GPUs。所有通信包括 self shard 的复制。

当前实验：TMA 每条指令固定搬运 **16 KiB**，warp 数固定为资源限制下的最大值，
CTA 数扫描 **4 / 8 / 16 / 32 / 64**。在每个 CTA 配置下使用相同的 per-rank 输出数据量 sweep。
NCCL 与 CE baseline 每个输出数据量只测一次。
这里 16 KiB 是满 tile 大小；当小消息 shard 不足 16 KiB 或存在尾块时，按实际字节数复制。

| Pattern | Backend | Memory / operation | 实现 | 正确性 / sweep |
|---|---|---|---|---|
| allgather | torch NCCL | `dist.all_gather_into_tensor` | 已实现 | 4 卡通过 |
| allgather | TMA | symmetric peer mappings，逐 peer push | 已实现 | 4 卡通过 |
| allgather | TMA | multicast VA，`multimem.cp.async.bulk` | 已实现 | 4 卡通过 |
| allgather | CE | symmetric peer mappings，D2D async copy | 已实现 | 4 卡通过 |
| allgather | CE | multicast VA，D2D async copy | 已实现 | 4 卡通过 |
| alltoall | torch NCCL | `dist.all_to_all_single` | 已实现 | 4 卡通过 |
| alltoall | TMA | symmetric peer mappings，逐 peer push | 已实现 | 4 卡通过 |
| alltoall | CE | symmetric peer mappings，逐 peer push | 已实现 | 4 卡通过 |

Symmetric memory 用 PyTorch CUDA symmetric allocator / rendezvous 管理相同布局和
peer 映射，不要求各进程虚拟地址数值相同。Multicast VA 映射同一组 backing buffers；
不是 CTA-cluster 的 shared-memory multicast。不引入 NVSHMEM runtime 依赖。
不支持的组合必须显式报错 / 标记，不能悄悄用其他实现替代。

## 消息量与指标

定义 sweep 横轴 `B = output bytes per rank`。`--sizes`、`--min-exp / --max-exp`
和绘图 `--sizes` 均采用输出量口径。默认 B 为 4 KiB 到 1 GiB 的 2 次幂。
两种 pattern 都有 `S = B/P` 的 shard，要求 B 是 `16*P` 的倍数。

| Pattern | 每 rank 输入 | 每目标 shard S | 每 rank 输出 | 每 rank 远端接收 R |
|---|---:|---:|---:|---:|
| Allgather | B/P | B/P | B | B*(P-1)/P |
| Alltoall | B | B/P | B | B*(P-1)/P |

4 卡、**1 GiB output/rank** 时，allgather 输入 256 MiB，alltoall 输入 1 GiB；
每个目标 shard 都为 256 MiB，两者每 rank 都接收 **768 MiB** 远端数据。
因此该口径对齐了远端接收量。TMA 的 16 KiB tile 是另一个独立概念。

- `payload_GBps = B/t/1e9`，包括 self shard 的逻辑输出。
- `remote_GBps = R/t/1e9`，每 GPU 有效远端接收带宽，是主比较指标。
- `aggregate_remote_GBps = P*R/t/1e9`，全体 GPU 接收量。
- `rx_util_pct = 100*remote_GBps/peak_oneway_GBps`。
- multicast allgather 的发送端有效注入量为 S，unicast 为 `(P-1)*S`；
  不把 multicast 接收总量解释为单个发送端实际链路字节。实际物理流量需硬件计数器。

新生成的 CSV 保存 `input_bytes`、`output_bytes`、`shard_bytes`、`remote_bytes_per_rank`，
并以 `size_basis=output_bytes_per_rank` 明确标记口径。

分母用显式 `--peak-gbps` 指定每 GPU **单方向** NVLink 峰值；不把双向规格直接当分母。
未提供时利用率留空，保留实测 GB/s。GB 使用十进制，消息 KiB/MiB 使用二进制。

## 测试约束

1. 记录 GPU / NVLink topology、PyTorch、CUDA、NCCL、环境变量和完整参数。
2. 先做 4 rank 的小尺寸、tile 边界、非 2 次幂尺寸正确性，再跑 sweep。
3. 数据编码包含源 rank、目标 shard 与元素位置；每个配置都做完整输出校验。
4. 所有方法使用相同的 stream barrier 包装，确保输入 ready / 远端写完成；
   barrier 包含在计时内，因此小消息结果是 collective latency 而非纯链路吞吐。
5. 默认 CUDA Graph，warmup 后多个 trial；每 trial 取所有 rank 的最大 event 时间，
   再报告 trial 中位数 / 最小值 / 最大值。分配、rendezvous、校验和编译不计时。
6. TMA 使用 global -> shared -> remote global；必须等待完整 bulk group 完成后再复用
   staging / 宣告 collective 完成。可 sweep CTA 数，不能把 CTA 数直接称为占用 SM 数。
7. CE 明确调用 CUDA memcpy API，不使用 `Tensor.copy_` 代替；用 Nsight Systems
   验证实际 GPU memcpy 活动。TMA 检查编译指令，必要时 sanitizer。
8. 不在未校验通过时报告性能；记录失败，保留原始 CSV / metadata。

## 参考

- [用户指定的 ThunderKittens 评论](https://github.com/HazyResearch/ThunderKittens/issues/169#issuecomment-3844927779)：点对点 CE / TMA 实验参考，不直接作为 collective 性能结论。
- [PTX asynchronous copy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-cp-async-bulk)：multimem bulk copy / completion 语义。
- [PyTorch symmetric memory](https://docs.pytorch.org/docs/stable/symmetric_memory.html)：peer buffers 与 multicast allocation。

本机初始检查：4 × NVIDIA GB200，任意 GPU pair 为 NV18；CUDA toolkit 13.1，
PyTorch 2.9.1+CUDA 13.1。GPU 操作需要沙箱外执行。

## 运行

依赖：CUDA toolkit >= 13.1（提供 PTX 9.1 multimem bulk copy）、PyTorch 2.9 的
`torch.distributed._symmetric_memory` CUDA allocator、NCCL、支持 multicast 的 NVLink
GPU；默认编译 SM100。这里使用 PyTorch 私有 API，升级 PyTorch 后应先跑 smoke。
绘图另需 matplotlib。

```bash
cd nvl_comm_throughput
make
# 正确性：最小对齐、tile 前后、非 2 次幂；每项校验三种数据 epoch。
NCCL_DEBUG=WARN torchrun --standalone --nproc-per-node=4 benchmark.py \
  --sizes 128 4096 65536 65600 131008 131072 131136 \
  --iters 3 --trials 2 --warmup 2 --output results/smoke.csv

# 4 卡全矩阵 sweep（使用新时间戳目录）。
bash sweep.sh --peak-gbps 900

# 每 rank 输出 1 GiB；两种 pattern 的远端接收量相同。
NCCL_DEBUG=WARN torchrun --standalone --nproc-per-node=4 benchmark.py \
  --sizes 1073741824 --peak-gbps 900 --output results/output1g.csv

# CTA 作为横轴，只展示 1 GiB output/rank。
python plot.py measurements/gb200_4gpu/sweep4.csv --x-axis ctas \
  --sizes 1073741824 --output assets/ctas.png

```

GB200 的 [1.8 TB/s 双向规格](https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/)
在本实验取 900 GB/s 单向作为参考分母；这是规格利用率，不是测得的物理链路计数器利用率。
默认保留 NCCL 环境和自动选算法，metadata 记录实际设置；不强制 ring 或禁用 NVLS。
`--eager` 可测 host 提交开销，不能与 graph 结果混为一条曲线。

每次输出 CSV、同名 `.metadata.json`（拓扑、版本、环境、参数、源码和二进制 hash）。
拒绝覆盖已有文件；启动新实验需选择新路径。`sweep.sh` 另外保存日志，并将 PNG / SVG 更新到 `assets/`。
同一时刻只跑一个性能实验，避免 GPU 相互干扰。缓存不 flush；小尺寸可能驻留 L2，
大尺寸用于观察稳态吞吐。当前 TMA 为每 warp 一个固定 16 KiB staging tile；
CE 在一个 stream 上按 rank 轮转顺序提交 peer copies。两者都有进一步调优空间。

## TMA warp / SMEM 配置（修订）

每个 warp 的 lane 0 独立发起 load、等待、store；拥有独立的 `tile_bytes` staging
和 16 B 对齐的 barrier 区域。任务按 `CTA * warps_per_CTA + warp_id` 分片。
因此增加 warp 会增加独立在途搬运，而不是仅增加不工作的线程。

warp 数固定按最大可用驻留数量选择，不提供手动指定或 sweep 参数：枚举 1–32 warps/CTA，调用
`cudaOccupancyMaxActiveBlocksPerMultiprocessor`，最大化
`warps_per_CTA * resident_blocks_per_SM`，受真实寄存器、SMEM、线程和 CTA 上限约束；
并列时选更大的 CTA。默认 grid sweep 为 4 / 8 / 16 / 32 / 64 CTAs；
`--ctas 0` 可额外选择 `SM_count * resident_blocks_per_SM` 的满驻留参考配置。
CTA 数不能直接解释为实际占用 SM 数。

GB200 本机查询：152 SM，64 warps/SM，228 KiB SMEM/SM，227 KiB opt-in SMEM/CTA。
各 backend 单独查 occupancy；CSV 记录 warps、SMEM、registers、blocks/SM 和设备上限。
16 KiB 对应 14 warps/CTA；小消息任务不足时不会实际用满这些 warp。

## 已执行测试（2026-09-27）

当前 4 卡完整 sweep：19 个 per-rank 输出数据量，3 条 TMA 路径各 5 个 CTA 配置，
5 条 NCCL/CE baseline 各测一次，共 `19 * (3*5 + 5) = 380` 项全部通过
三种数据 epoch 的完整正确性校验。TMA 固定 16 KiB tile、14 warps/CTA。

已归档数据：`measurements/gb200_4gpu/sweep4.csv`，环境与源码 hash：同名 `.metadata.json`。
per-rank 输出量图和 CTA 图分别是 `assets/bandwidth.png`、`assets/ctas.png`（附 SVG）。
输出量图仅展示 TMA 的 16 / 32 CTA 配置及全部 NCCL/CE baseline；CTA 图保留完整扫描。
恢复 output/rank 口径后的 22 项回归全部通过，记录在 `measurements/gb200_4gpu/output_restore_smoke4.csv`。
实测汇总见 [RESULTS.md](RESULTS.md)。
