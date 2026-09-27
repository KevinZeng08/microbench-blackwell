#!/usr/bin/env python3
"""Single-node, torchrun-launched NVLink collective benchmark. See README.md."""
import argparse
import csv
import ctypes as C
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

ROOT = Path(__file__).resolve().parent
BACKENDS = ("nccl", "tma_sym", "tma_mc", "ce_sym", "ce_mc")


def args_parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patterns", nargs="+", choices=("allgather", "alltoall"),
                   default=["allgather", "alltoall"])
    p.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    p.add_argument("--min-exp", type=int, default=12)
    p.add_argument("--max-exp", type=int, default=30)
    p.add_argument("--sizes", type=int, nargs="+", help="explicit OUTPUT bytes per rank")
    p.add_argument("--ctas", type=int, nargs="+", default=[4, 8, 16, 32, 64],
                   help="0 = SM count * occupancy blocks/SM")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--eager", action="store_true", help="include host submission gaps")
    p.add_argument("--peak-gbps", type=float, help="per-GPU ONE-WAY NVLink peak, decimal GB/s")
    p.add_argument("--output", type=Path, default=ROOT / "results" / "sweep.csv")
    a = p.parse_args()
    a.tile_bytes = 16384  # Fixed per-instruction TMA message; only grid size is swept.
    if not 0 <= a.min_exp <= a.max_exp <= 40:
        p.error("require 0 <= min-exp <= max-exp <= 40")
    a.sizes = sorted(set(a.sizes or [2**i for i in range(a.min_exp, a.max_exp + 1)]))
    if min(a.sizes) <= 0 or min(a.ctas) < 0 or min(a.iters, a.trials, a.warmup) <= 0:
        p.error("sizes, iters, trials, warmup must be positive; CTAs must be >=0")
    if a.peak_gbps is not None and a.peak_gbps <= 0:
        p.error("peak-gbps must be positive")
    return a


class Native:
    def __init__(self):
        self.lib = C.CDLL(str(ROOT / "libnvl_comm.so"))
        ptr, size, integer = C.c_void_p, C.c_size_t, C.c_int
        self.lib.launch_tma.argtypes = [ptr, ptr, ptr, size, integer, integer,
                                        integer, integer, integer, integer, ptr]
        self.lib.configure_tma.argtypes = [integer, integer, integer, C.POINTER(integer)]
        self.lib.launch_ce.argtypes = [C.c_uint64, C.POINTER(C.c_uint64), C.c_uint64,
                                       size, integer, integer, integer, ptr]
        self.lib.fill_data.argtypes = [ptr, size, size, integer, integer, integer, ptr]
        self.lib.verify_data.argtypes = [ptr, size, size, integer, integer, integer, ptr, ptr]
        for name in ("launch_tma", "configure_tma", "launch_ce", "fill_data", "verify_data"):
            getattr(self.lib, name).restype = integer

    def call(self, name, *args):
        error = getattr(self.lib, name)(*args)
        if error:
            raise RuntimeError(f"{name} failed: CUDA {'driver' if name == 'launch_ce' else 'runtime'} error {error}")


def command(*cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=20)
    except (OSError, subprocess.SubprocessError) as e:
        return str(e)


def metadata(a, world):
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "world_size": world, "torch": torch.__version__, "cuda": torch.version.cuda,
        "size_basis": "output_bytes_per_rank",
        "nccl": torch.cuda.nccl.version(),
        "gpu": [str(torch.cuda.get_device_properties(i)) for i in range(world)],
        "topology": command("nvidia-smi", "topo", "-m"),
        "nvidia_smi": command("nvidia-smi"), "nvcc": command("nvcc", "--version"),
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("NCCL_", "CUDA_", "TORCH_"))},
        "sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                   for f in ("benchmark.py", "kernels.cu", "libnvl_comm.so")},
        "timing": "median of per-trial max-rank CUDA-event average, two symmetric barriers per op",
    }


def main():
    a = args_parse()
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world != 4 or int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        raise ValueError("requires single-node torchrun with 4 GPUs")
    alignment = 16 * world
    if any(b % alignment for b in a.sizes):
        raise ValueError(f"output bytes must be divisible by {alignment}")
    torch.cuda.set_device(local)
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError("requires SM90+; default binary targets SM100")
    for peer in range(world):
        if peer != local and not torch.cuda.can_device_access_peer(local, peer):
            raise RuntimeError(f"GPU {local} cannot access GPU {peer}")
    dist.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=torch.device("cuda", local))
    cpu_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    native = Native()
    stream = torch.cuda.Stream()
    control = symm.empty(4096, dtype=torch.uint8, device="cuda")
    control_handle = symm.rendezvous(control, dist.group.WORLD)
    if rank == 0:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        # Avoid silently overwriting a completed experiment.
        with a.output.with_suffix(".metadata.json").open("x") as f:
            json.dump(metadata(a, world), f, indent=2)
    dist.barrier(group=cpu_group)
    fields = ["pattern", "backend", "world", "size_basis", "output_bytes", "input_bytes", "shard_bytes",
              "remote_bytes_per_rank",
              "ctas", "tile_bytes", "mode", "iters", "trials", "latency_us", "min_us", "max_us",
              "warps_per_cta", "dynamic_smem_bytes", "occupancy_blocks_per_sm",
              "occupancy_warps_per_sm", "sm_count", "max_warps_per_sm", "smem_per_sm",
              "max_smem_per_cta", "registers_per_thread", "static_smem_bytes",
              "payload_GBps", "remote_GBps", "aggregate_remote_GBps", "logical_inject_GBps",
              "peak_oneway_GBps", "rx_util_pct", "correct", "trial_us"]
    output_file = a.output.open("x", newline="") if rank == 0 else None
    writer = csv.DictWriter(output_file, fieldnames=fields) if rank == 0 else None
    if writer:
        writer.writeheader()
    try:
        for pattern_name in a.patterns:
            alltoall = int(pattern_name == "alltoall")
            for output_bytes in a.sizes:
                shard = output_bytes // world
                input_bytes = output_bytes if alltoall else shard
                remote_bytes = shard * (world - 1)
                # NCCL baseline uses ordinary output allocation. Custom methods share
                # symmetric storage; this is their required transport setup, outside timing.
                src = torch.empty(input_bytes, dtype=torch.uint8, device="cuda")
                dst = symm.empty(output_bytes, dtype=torch.uint8, device="cuda")
                handle = symm.rendezvous(dst, dist.group.WORLD)
                nccl_dst = torch.empty_like(dst)
                errors = torch.zeros(1, dtype=torch.int64, device="cuda")
                peers = (C.c_uint64 * world)(*handle.buffer_ptrs)
                dev_peers = handle.buffer_ptrs_dev
                stream.wait_stream(torch.cuda.current_stream())
                for backend in a.backends:
                    if alltoall and backend.endswith("_mc"):
                        continue  # Not a valid combination, explicitly absent from matrix.
                    mc = handle.multicast_ptr if backend.endswith("_mc") else 0
                    if backend.endswith("_mc") and not mc:
                        raise RuntimeError(f"{backend}: multicast allocation unsupported (no fallback)")
                    out = nccl_dst if backend == "nccl" else dst
                    for requested_ctas in a.ctas if backend.startswith("tma") else [0]:
                        config = (C.c_int * 9)()
                        if backend.startswith("tma"):
                            native.call("configure_tma", alltoall, bool(mc), a.tile_bytes, config)
                        warps, smem, blocks_per_sm, sm_count, max_warps, smem_per_sm, max_smem, regs, static_smem = config
                        ctas = requested_ctas or blocks_per_sm * sm_count
                        def operation():
                            control_handle.barrier(channel=0, timeout_ms=60000)
                            if backend == "nccl":
                                if alltoall:
                                    dist.all_to_all_single(out, src)
                                else:
                                    dist.all_gather_into_tensor(out, src)
                            elif backend.startswith("tma"):
                                native.call("launch_tma", src.data_ptr(), dev_peers, mc,
                                            shard, rank, world, alltoall, ctas, a.tile_bytes,
                                            warps, stream.cuda_stream)
                            else:
                                native.call("launch_ce", src.data_ptr(), peers, mc,
                                            shard, rank, world, alltoall, stream.cuda_stream)
                            control_handle.barrier(channel=1, timeout_ms=60000)

                        def validate(epoch, run):
                            with torch.cuda.stream(stream):
                                native.call("fill_data", src.data_ptr(), input_bytes, shard,
                                            rank, alltoall, epoch, stream.cuda_stream)
                                out.fill_(0xA5)
                                errors.zero_()
                                run()
                                native.call("verify_data", out.data_ptr(), output_bytes, shard, rank,
                                            alltoall, epoch, errors.data_ptr(), stream.cuda_stream)
                            stream.synchronize()
                            count = int(errors.item())
                            counts = [None] * world
                            dist.all_gather_object(counts, count, group=cpu_group)
                            if any(counts):
                                raise RuntimeError(f"{pattern_name}/{backend}/input={input_bytes}/{ctas}/{warps}: mismatches {counts}")

                        validate(1, operation)
                        with torch.cuda.stream(stream):
                            for _ in range(a.warmup):
                                operation()
                        stream.synchronize()
                        dist.barrier(group=cpu_group)
                        graph = None
                        if not a.eager:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=stream):
                                for _ in range(a.iters):
                                    operation()
                            # New contents + poisoned output detect stale graph / missing writes.
                            validate(2, graph.replay)
                        times = []
                        for _ in range(a.trials):
                            dist.barrier(group=cpu_group)
                            start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                            with torch.cuda.stream(stream):
                                start.record()
                                if graph is not None:
                                    graph.replay()
                                else:
                                    for _ in range(a.iters):
                                        operation()
                                stop.record()
                            stop.synchronize()
                            elapsed = start.elapsed_time(stop) * 1000 / a.iters
                            per_rank = [None] * world
                            dist.all_gather_object(per_rank, elapsed, group=cpu_group)
                            times.append(max(per_rank))
                        validate(3, graph.replay if graph is not None else operation)
                        us = statistics.median(times)
                        remote = remote_bytes / us / 1000
                        row = dict(pattern=pattern_name, backend=backend, world=world,
                                   size_basis="output_bytes_per_rank", output_bytes=output_bytes,
                                   input_bytes=input_bytes, shard_bytes=shard,
                                   remote_bytes_per_rank=remote_bytes,
                                   ctas=ctas, tile_bytes=a.tile_bytes if ctas else 0,
                                   warps_per_cta=warps, dynamic_smem_bytes=smem,
                                   occupancy_blocks_per_sm=blocks_per_sm,
                                   occupancy_warps_per_sm=blocks_per_sm * warps,
                                   sm_count=sm_count, max_warps_per_sm=max_warps,
                                   smem_per_sm=smem_per_sm, max_smem_per_cta=max_smem,
                                   registers_per_thread=regs, static_smem_bytes=static_smem,
                                   mode="eager" if a.eager else "graph", iters=a.iters,
                                   trials=a.trials, latency_us=us, min_us=min(times), max_us=max(times),
                                   payload_GBps=output_bytes / us / 1000, remote_GBps=remote,
                                   aggregate_remote_GBps=remote * world,
                                   logical_inject_GBps=shard / us / 1000 if mc else remote,
                                   peak_oneway_GBps=a.peak_gbps or "",
                                   rx_util_pct=remote / a.peak_gbps * 100 if a.peak_gbps else "",
                                   correct=True, trial_us=json.dumps(times))
                        if writer:
                            writer.writerow(row)
                            output_file.flush()
                            print(f"{pattern_name:9s} {backend:7s} input={input_bytes:11d} "
                                  f"output={output_bytes:11d} CTA={ctas:3d} W={warps:2d} "
                                  f"{us:9.3f} us {remote:8.2f} GB/s remote/rank", flush=True)
                        del graph
                # All ranks finish reading before releasing mappings.
                stream.synchronize()
                dist.barrier(group=cpu_group)
                del dst, handle, nccl_dst, out, src, errors
    finally:
        if output_file:
            output_file.close()
        # Release VMM / multicast mappings while the CUDA primary context and
        # process group are still alive (PyTorch 2.9 teardown is order-sensitive).
        stream.synchronize()
        del control_handle, control
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
