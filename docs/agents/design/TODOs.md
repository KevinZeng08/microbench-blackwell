# MicroBench TODOs

## 定义

MicroBench 测的是**原子化组件的能力**：一条指令、一条硬件路径、一个 API 原语，在锁死设定下的延迟 / 吞吐 / 合法空间。

不是 e2e 系统分数。测完应能回答：这个原子能做什么、上限在哪、默认路径踩了哪条税。

---

## 单卡计算、访存

（现有仓库已有不少）

- [ ] L2 cache 相关带宽、延迟等指标

---

## 节点内通信

NVLink性能、行为等

- [ ] TMA
- [ ] register ld/st
- [ ] NVLS multimem指令 (性能、red指令deterministic)
- [ ] copy engine
- [ ] NCCL LSA Device API

---

## 存储

这块不太了解，可以从Mooncake等项目学习

GPU direct storage

---

## 节点间通信

GPU direct RDMA

- [ ] NCCL GIN

---

## 参考资料

- [Mooncake](https://github.com/kvcache-ai/Mooncake) / [Transfer Engine](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html) / [Mooncake Store](https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html)
- [GPUDirect Storage / cuFile](https://docs.nvidia.com/gpudirect-storage/)
- NCCL 2.30（Host API、Device API：LSA / Multimem / GIN）
- NVSHMEM 3.7
