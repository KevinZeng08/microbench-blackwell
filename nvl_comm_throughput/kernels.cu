#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// The Python owner retains every allocation and peer mapping through completion.
// One independent issuing lane per warp: local global -> shared -> remote global.
// Each warp owns its staging tile and mbarrier; no CTA-wide synchronization.
// The unicast AG path reuses a local tile for all destinations.
template<bool Multicast, bool Alltoall>
__global__ void tma_push(const char* src, char** peers, char* multicast,
                         size_t shard, int rank, int world, int tile) {
    extern __shared__ __align__(16) char storage[];
    const unsigned warp = threadIdx.x / 32;
    const unsigned warps = blockDim.x / 32;
    if (threadIdx.x % 32 != 0) return;
    char* staging = storage + warp * (tile + 16);
    const unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(staging));
    const unsigned bar = static_cast<unsigned>(__cvta_generic_to_shared(staging + tile));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    unsigned phase = 0;
    const size_t tiles = (shard + tile - 1) / tile;
    const size_t tasks = tiles * (Alltoall ? world : 1);
    for (size_t task = size_t(blockIdx.x) * warps + warp; task < tasks;
         task += size_t(gridDim.x) * warps) {
        const int dst_rank = Alltoall ? (int(task / tiles) + rank) % world : 0;
        const size_t offset = (task % tiles) * tile;
        const unsigned bytes = static_cast<unsigned>(min(size_t(tile), shard - offset));
        const char* input = src + (Alltoall ? dst_rank * shard : 0) + offset;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                     :: "r"(bar), "r"(bytes) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                     "[%0], [%1], %2, [%3];"
                     :: "r"(smem), "l"(input), "r"(bytes), "r"(bar) : "memory");
        asm volatile("{ .reg .pred ready; WAIT: "
                     "mbarrier.try_wait.parity.shared::cta.b64 ready, [%0], %1; "
                     "@!ready bra WAIT; }" :: "r"(bar), "r"(phase) : "memory");
        phase ^= 1;
        const size_t out_offset = rank * shard + offset;
        if constexpr (Multicast) {
            asm volatile("multimem.cp.async.bulk.global.shared::cta.bulk_group "
                         "[%0], [%1], %2;"
                         :: "l"(multicast + out_offset), "r"(smem), "r"(bytes) : "memory");
        } else {
            const int count = Alltoall ? 1 : world;
            for (int i = 0; i < count; ++i) {
                const int p = Alltoall ? dst_rank : (rank + i) % world;
                asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                             :: "l"(peers[p] + out_offset), "r"(smem), "r"(bytes) : "memory");
            }
        }
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        // Full completion (NOT .read): destination writes must finish before barrier.
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    }
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(bar) : "memory");
}

// Choose the largest resident warp count permitted by the actual compiled
// kernel (registers, block limits, threads and dynamic shared memory included).
// Called before timing/capture; launch_tma itself never changes function attrs.
extern "C" int configure_tma(int alltoall, int multicast, int tile, int* config) {
    if (tile < 16 || tile > 32768 || tile % 16 ||
        (alltoall && multicast)) return int(cudaErrorInvalidValue);
    const void* kernel = multicast ? (const void*)tma_push<true, false> :
        alltoall ? (const void*)tma_push<false, true> : (const void*)tma_push<false, false>;
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) return int(err);
    cudaDeviceProp prop;
    if ((err = cudaGetDeviceProperties(&prop, device)) != cudaSuccess) return int(err);
    cudaFuncAttributes attr;
    if ((err = cudaFuncGetAttributes(&attr, kernel)) != cudaSuccess) return int(err);
    int max_dynamic = int(prop.sharedMemPerBlockOptin - attr.sharedSizeBytes);
    if ((err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   max_dynamic)) != cudaSuccess) return int(err);
    int best_warps = 0, best_blocks = 0;
    for (int w = 1; w <= 32; ++w) {
        if (w * 32 > attr.maxThreadsPerBlock || w * (tile + 16) > max_dynamic) continue;
        int blocks = 0;
        if ((err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                 &blocks, kernel, w * 32, w * (tile + 16))) != cudaSuccess) return int(err);
        if (blocks > 0 && blocks * w >= best_blocks * best_warps) {
            best_warps = w;
            best_blocks = blocks;
        }
    }
    if (!best_warps) return int(cudaErrorInvalidConfiguration);
    config[0] = best_warps;
    config[1] = best_warps * (tile + 16);
    config[2] = best_blocks;
    config[3] = prop.multiProcessorCount;
    config[4] = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    config[5] = int(prop.sharedMemPerMultiprocessor);
    config[6] = int(prop.sharedMemPerBlockOptin);
    config[7] = attr.numRegs;
    config[8] = int(attr.sharedSizeBytes);
    return 0;
}

__device__ uint32_t pattern(size_t i, int source, int destination, int epoch) {
    return uint32_t(i) * 2654435761u ^ uint32_t(i >> 32) * 2246822519u ^
           uint32_t(source + 1) * 3266489917u ^ uint32_t(destination + 1) * 668265263u ^
           uint32_t(epoch) * 374761393u;
}

__global__ void fill_kernel(uint32_t* data, size_t words, size_t shard_words,
                            int rank, bool alltoall, int epoch) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
         i += size_t(blockDim.x) * gridDim.x)
        data[i] = pattern(i % shard_words, rank, alltoall ? i / shard_words : 0, epoch);
}

__global__ void verify_kernel(const uint32_t* data, size_t words, size_t shard_words,
                              int rank, bool alltoall, int epoch, unsigned long long* errors) {
    unsigned long long count = 0;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
         i += size_t(blockDim.x) * gridDim.x)
        count += data[i] != pattern(i % shard_words, i / shard_words,
                                    alltoall ? rank : 0, epoch);
    if (count) atomicAdd(errors, count);
}

extern "C" int launch_tma(const void* src, void* peers, void* multicast,
                          size_t shard, int rank, int world, int alltoall,
                          int ctas, int tile, int warps, cudaStream_t stream) {
    if (!shard || shard % 16 || tile < 16 || tile > 32768 || tile % 16 ||
        ctas < 1 || warps < 1 || warps > 32 || rank < 0 || rank >= world || (alltoall && multicast))
        return int(cudaErrorInvalidValue);
    const int smem_bytes = warps * (tile + 16);
    if (multicast)
        tma_push<true, false><<<ctas, warps * 32, smem_bytes, stream>>>(
            (const char*)src, (char**)peers, (char*)multicast, shard, rank, world, tile);
    else if (alltoall)
        tma_push<false, true><<<ctas, warps * 32, smem_bytes, stream>>>(
            (const char*)src, (char**)peers, nullptr, shard, rank, world, tile);
    else
        tma_push<false, false><<<ctas, warps * 32, smem_bytes, stream>>>(
            (const char*)src, (char**)peers, nullptr, shard, rank, world, tile);
    return int(cudaGetLastError());
}

extern "C" int launch_ce(uint64_t src, const uint64_t* peers, uint64_t multicast,
                         size_t shard, int rank, int world, int alltoall, CUstream stream) {
    if (alltoall && multicast) return int(CUDA_ERROR_INVALID_VALUE);
    for (int i = 0; i < (multicast ? 1 : world); ++i) {
        const int p = (rank + i) % world;
        CUdeviceptr dst = (multicast ? multicast : peers[p]) + rank * shard;
        CUdeviceptr input = src + (alltoall ? p * shard : 0);
        CUresult result = cuMemcpyDtoDAsync(dst, input, shard, stream);
        if (result != CUDA_SUCCESS) return int(result);
    }
    return 0;
}

extern "C" int fill_data(void* data, size_t bytes, size_t shard, int rank,
                         int alltoall, int epoch, cudaStream_t stream) {
    fill_kernel<<<256, 256, 0, stream>>>((uint32_t*)data, bytes / 4, shard / 4,
                                        rank, alltoall, epoch);
    return int(cudaGetLastError());
}

extern "C" int verify_data(const void* data, size_t bytes, size_t shard, int rank,
                           int alltoall, int epoch, void* errors, cudaStream_t stream) {
    verify_kernel<<<256, 256, 0, stream>>>((const uint32_t*)data, bytes / 4, shard / 4,
                                          rank, alltoall, epoch, (unsigned long long*)errors);
    return int(cudaGetLastError());
}
