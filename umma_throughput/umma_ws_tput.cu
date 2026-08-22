// tcgen05.mma.ws throughput microbenchmark (1SM, dense)
// Collector: WS_COLLECTOR=0 omit qualifier (b0 discard); =1 b0 fill / use / lastuse
// Compile-time macros:
//   MMA_FORMAT: 0=BF16, 1=TF32, 2=E4M3, 3=S8, 4=F4
//   MMA_M: 32, 64, 128
//   MMA_N: 64, 128, 256
//   AB_LAYOUT: 0=SS (A from SMEM), 1=TS (A from TMEM)
//   WS_COLLECTOR: 0=b0 discard (default), 1=b0 fill/use/lastuse
//   CTA_GROUP must be 1

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef MMA_FORMAT
#error "MMA_FORMAT must be defined (0=BF16, 1=TF32, 2=E4M3, 3=S8, 4=F4)"
#endif
#ifndef MMA_M
#error "MMA_M must be defined (32, 64, or 128)"
#endif
#ifndef MMA_N
#error "MMA_N must be defined (64, 128, or 256)"
#endif
#ifndef MMA_K
#error "MMA_K must be defined"
#endif
#ifndef MMA_DEPTH
#error "MMA_DEPTH must be defined"
#endif
#ifndef CTA_GROUP
#error "CTA_GROUP must be defined (must be 1 for mma.ws)"
#endif
#ifndef AB_LAYOUT
#error "AB_LAYOUT must be defined (0=SS_MODE, 1=TS_MODE)"
#endif
#ifndef WS_COLLECTOR
#define WS_COLLECTOR 0
#endif

#define SS_MODE 0
#define TS_MODE 1
#define WS_COLLECTOR_DISCARD 0
#define WS_COLLECTOR_REUSE   1

#if MMA_FORMAT == 0
    #define MMA_KIND "f16"
#elif MMA_FORMAT == 1
    #define MMA_KIND "tf32"
#elif MMA_FORMAT == 2 || MMA_FORMAT == 4
    #define MMA_KIND "f8f6f4"
#elif MMA_FORMAT == 3
    #define MMA_KIND "i8"
#else
    #error "MMA_FORMAT must be 0=BF16, 1=TF32, 2=E4M3, 3=S8, 4=F4"
#endif

#if CTA_GROUP != 1
#error "tcgen05.mma.ws only supports cta_group::1"
#endif
#if MMA_M != 32 && MMA_M != 64 && MMA_M != 128
#error "tcgen05.mma.ws dense M must be 32, 64, or 128"
#endif
#if MMA_N != 64 && MMA_N != 128 && MMA_N != 256
#error "tcgen05.mma.ws dense N must be 64, 128, or 256"
#endif
#if AB_LAYOUT != SS_MODE && AB_LAYOUT != TS_MODE
#error "AB_LAYOUT must be 0 (SS) or 1 (TS)"
#endif
#if MMA_FORMAT == 0 && MMA_K != 16
#error "BF16 requires MMA_K=16"
#endif
#if MMA_FORMAT == 1 && MMA_K != 8
#error "TF32 requires MMA_K=8"
#endif
#if (MMA_FORMAT == 2 || MMA_FORMAT == 3) && MMA_K != 32
#error "E4M3/S8 require MMA_K=32"
#endif
#if MMA_FORMAT == 4 && MMA_K != 64
#error "F4 requires MMA_K=64"
#endif
#if WS_COLLECTOR != WS_COLLECTOR_DISCARD && WS_COLLECTOR != WS_COLLECTOR_REUSE
#error "WS_COLLECTOR must be 0 (discard) or 1 (b0 fill/use/lastuse)"
#endif
#if WS_COLLECTOR == WS_COLLECTOR_REUSE && MMA_DEPTH < 2
#error "B-collector reuse requires MMA_DEPTH >= 2"
#endif

enum class MMAFormat : uint8_t {
    BF16 = 0,
    TF32 = 1,
    E4M3 = 2,
    S8   = 3,
    F4   = 4,
};

template <MMAFormat Fmt> struct MMATraits;

template <> struct MMATraits<MMAFormat::BF16> {
    struct A { using Elem = nv_bfloat16; static constexpr int Bits = 16; };
    struct B { using Elem = nv_bfloat16; static constexpr int Bits = 16; };
    struct D { using Elem = float; };
};

template <> struct MMATraits<MMAFormat::TF32> {
    struct A { using Elem = float; static constexpr int Bits = 32; };
    struct B { using Elem = float; static constexpr int Bits = 32; };
    struct D { using Elem = float; };
};

template <> struct MMATraits<MMAFormat::E4M3> {
    struct A { using Elem = __nv_fp8_e4m3; static constexpr int Bits = 8; };
    struct B { using Elem = __nv_fp8_e4m3; static constexpr int Bits = 8; };
    struct D { using Elem = float; };
};

template <> struct MMATraits<MMAFormat::S8> {
    struct A { using Elem = int8_t; static constexpr int Bits = 8; };
    struct B { using Elem = int8_t; static constexpr int Bits = 8; };
    struct D { using Elem = int32_t; };
};

template <> struct MMATraits<MMAFormat::F4> {
    struct A { using Elem = uint8_t; static constexpr int Bits = 4; };
    struct B { using Elem = uint8_t; static constexpr int Bits = 4; };
    struct D { using Elem = float; };
};

using MT = MMATraits<static_cast<MMAFormat>(MMA_FORMAT)>;

__host__ __device__ constexpr int next_power_of_2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

__device__ __forceinline__
uint64_t make_smem_desc(const void* ptr, int height) {
    int addr = static_cast<int>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
    desc |= (addr >> 4) & 0x3FFF;
    desc |= ((height * 16) >> 4) << 16;
    desc |= (8ULL << 32);
    desc |= (1ULL << 46);
    return desc;
}

__device__ inline uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}"
        : "+r"(pred)
        : "r"(0xFFFFFFFF)
    );
    return pred;
}

__device__ __forceinline__ void barrier_sync() {
    __syncthreads();
}

template <MMAFormat Fmt>
__device__ constexpr uint32_t make_i_desc();

template <> __device__ constexpr uint32_t make_i_desc<MMAFormat::BF16>() {
    uint32_t desc = 0;
    desc |= (1U << 4);                      // dtype = FP32
    desc |= (1U << 7);                      // atype = BF16
    desc |= (1U << 10);                     // btype = BF16
    desc |= ((MMA_N >> 3) << 17);           // N / 8
    desc |= ((MMA_M >> 4) << 24);           // M / 16
    return desc;
}

template <> __device__ constexpr uint32_t make_i_desc<MMAFormat::TF32>() {
    uint32_t desc = 0;
    desc |= (1U << 4);                      // dtype = FP32
    desc |= (2U << 7);                      // atype = TF32
    desc |= (2U << 10);                     // btype = TF32
    desc |= ((MMA_N >> 3) << 17);           // N / 8
    desc |= ((MMA_M >> 4) << 24);           // M / 16
    return desc;
}

template <> __device__ constexpr uint32_t make_i_desc<MMAFormat::E4M3>() {
    uint32_t desc = 0;
    desc |= (1U << 4);                      // dtype = FP32
    desc |= (2U << 7);                      // atype = E4M3
    desc |= (2U << 10);                     // btype = E4M3
    desc |= ((MMA_N >> 3) << 17);           // N / 8
    desc |= ((MMA_M >> 4) << 24);           // M / 16
    return desc;
}

template <> __device__ constexpr uint32_t make_i_desc<MMAFormat::S8>() {
    uint32_t desc = 0;
    desc |= (2U << 4);                      // dtype = S32
    desc |= (1U << 7);                      // atype = INT8
    desc |= (1U << 10);                     // btype = INT8
    desc |= ((MMA_N >> 3) << 17);           // N / 8
    desc |= ((MMA_M >> 4) << 24);           // M / 16
    return desc;
}

template <> __device__ constexpr uint32_t make_i_desc<MMAFormat::F4>() {
    uint32_t desc = 0;
    desc |= (1U << 4);                      // dtype = FP32
    desc |= (5U << 7);                      // atype = E2M1
    desc |= (5U << 10);                     // btype = E2M1
    desc |= ((MMA_N >> 3) << 17);           // N / 8
    desc |= ((MMA_M >> 4) << 24);           // M / 16
    return desc;
}

template <typename T>
__device__ __forceinline__ T fill_value(int i) {
    T val;
    uint8_t* p = reinterpret_cast<uint8_t*>(&val);
    #pragma unroll
    for (int b = 0; b < sizeof(T); b++)
        p[b] = static_cast<uint8_t>(((i + b) % 127) + 1);
    return val;
}

constexpr int A_SIZE = (MMA_M * MMA_K) * MT::A::Bits / 8;
constexpr int B_SIZE = (MMA_N * MMA_K) * MT::B::Bits / 8;

__global__ __launch_bounds__(128)
void umma_ws_tput_kernel() {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    extern __shared__ __align__(128) char smem[];
    auto* A = reinterpret_cast<MT::A::Elem*>(smem);
    auto* B = reinterpret_cast<MT::B::Elem*>(smem + A_SIZE);

    constexpr int A_NUMEL = A_SIZE / sizeof(MT::A::Elem);
    constexpr int B_NUMEL = B_SIZE / sizeof(MT::B::Elem);
    for (int i = tid; i < A_NUMEL; i += blockDim.x)
        A[i] = fill_value<MT::A::Elem>(i);
    for (int i = tid; i < B_NUMEL; i += blockDim.x)
        B[i] = fill_value<MT::B::Elem>(i);

    barrier_sync();

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ uint64_t mbar;
    __shared__ int tmem_addr;

    const int mbar_addr = static_cast<int>(__cvta_generic_to_shared(&mbar));
    if (warp_id == 0 && elect_sync()) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                    :: "r"(mbar_addr), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    int tmem_cols = MMA_N;
#if AB_LAYOUT == TS_MODE
    const int tmem_a_offset = tmem_cols;
    tmem_cols += 8;
#endif
    const int tmem_alloc_cols = next_power_of_2(tmem_cols < 32 ? 32 : tmem_cols);

    if (warp_id == 0) {
        const int tmem_addr_smem = static_cast<int>(__cvta_generic_to_shared(&tmem_addr));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                    :: "r"(tmem_addr_smem), "r"(tmem_alloc_cols));
    }
    barrier_sync();

    const uint32_t tmem_d = tmem_addr;

#if AB_LAYOUT == TS_MODE
    const uint32_t tmem_a = tmem_addr + tmem_a_offset;

    constexpr int A_ROW_BYTES = MMA_K * MT::A::Bits / 8;
    static_assert(A_ROW_BYTES == 32, "Each A row should be 256 bits = 32 bytes");

    uint32_t a_regs[8];
    if (tid >= MMA_M) {
        for (int i = 0; i < 8; i++) a_regs[i] = 0;
    } else {
        const uint32_t* row = reinterpret_cast<const uint32_t*>(
            reinterpret_cast<const char*>(A) + tid * A_ROW_BYTES);
        for (int i = 0; i < 8; i++) a_regs[i] = row[i];
    }
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(tmem_a),
                   "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
                   "r"(a_regs[4]), "r"(a_regs[5]), "r"(a_regs[6]), "r"(a_regs[7]));
#endif

    barrier_sync();

    constexpr uint32_t i_desc = make_i_desc<static_cast<MMAFormat>(MMA_FORMAT)>();
    uint64_t b_desc = make_smem_desc(B, MMA_N);

#if AB_LAYOUT == SS_MODE
    uint64_t a_desc = make_smem_desc(A, MMA_M);
#define ISSUE_MMA_WS(COLLECTOR, pred_var) \
    asm volatile( \
        "{\n\t" \
        ".reg .pred p;\n\t" \
        "setp.ne.b32 p, %4, 0;\n\t" \
        "tcgen05.mma.ws.cta_group::1.kind::" MMA_KIND COLLECTOR " [%0], %1, %2, %3, p;\n\t" \
        "}" \
        :: "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(pred_var) \
    )
#else
#define ISSUE_MMA_WS(COLLECTOR, pred_var) \
    asm volatile( \
        "{\n\t" \
        ".reg .pred p;\n\t" \
        "setp.ne.b32 p, %4, 0;\n\t" \
        "tcgen05.mma.ws.cta_group::1.kind::" MMA_KIND COLLECTOR " [%0], [%1], %2, %3, p;\n\t" \
        "}" \
        :: "r"(tmem_d), "r"(tmem_a), "l"(b_desc), "r"(i_desc), "r"(pred_var) \
    )
#endif

#if WS_COLLECTOR == WS_COLLECTOR_DISCARD
    auto mma = [&](int pred) { ISSUE_MMA_WS("", pred); };
#else
    auto mma_fill = [&](int pred) { ISSUE_MMA_WS(".collector::b0::fill", pred); };
    auto mma_use = [&](int pred) { ISSUE_MMA_WS(".collector::b0::use", pred); };
    auto mma_lastuse = [&](int pred) { ISSUE_MMA_WS(".collector::b0::lastuse", pred); };
#endif

    constexpr int NUM_ITERS = 1000;
    uint64_t start_clock, end_clock;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock));

    for (int iter = 0, phase = 0; iter < NUM_ITERS; iter++) {
        if (warp_id == 0 && elect_sync()) {
#if WS_COLLECTOR == WS_COLLECTOR_DISCARD
            mma(0);
            #pragma unroll
            for (int m = 1; m < MMA_DEPTH; m++) {
                mma(1);
            }
#else
            mma_fill(0);
            #pragma unroll
            for (int m = 1; m < MMA_DEPTH - 1; m++) {
                mma_use(1);
            }
            mma_lastuse(1);
#endif
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                        :: "r"(mbar_addr) : "memory");
        }

        asm volatile(
            "{\n\t"
            ".reg .pred P1;\n\t"
            "LAB_WAIT:\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
            "@P1 bra.uni DONE;\n\t"
            "bra.uni LAB_WAIT;\n\t"
            "DONE:\n\t"
            "}"
            :: "r"(mbar_addr), "r"(phase)
        );
        phase ^= 1;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock));

    asm volatile("tcgen05.fence::after_thread_sync;");

    barrier_sync();
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                    :: "r"(tmem_addr), "r"(tmem_alloc_cols));
    }

    if (warp_id == 0 && elect_sync()) {
        uint64_t cycles = end_clock - start_clock;
        uint64_t total_mmas = (uint64_t)MMA_DEPTH * NUM_ITERS;
        printf("RESULT,%d,%d,%d,%d,%llu,%llu,%.4f\n",
               MMA_M, MMA_N, MMA_K, MMA_DEPTH,
               (unsigned long long)cycles, (unsigned long long)total_mmas,
               (double)cycles / total_mmas);
    }
}

int main() {
    umma_ws_tput_kernel<<<1, 128, A_SIZE + B_SIZE>>>();

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Done!\n");
    return 0;
}
