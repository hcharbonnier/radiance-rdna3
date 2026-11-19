/* r3d_gemm_iqr_splitk.hip -- split-K for the IQR GEMV (M = 1, narrow N), on gfx11.
 *
 * The GEMV in r3d_gemm_iqr_m16.hip gives one workgroup (one wave) to a 16-row weight tile. On the
 * 27B's down shape (N = 5120, K = 17408) that is 320 workgroups for 96 CUs: three waves a CU at
 * best, and the W0 table measures 51.5 % of the weight-stream ceiling. The wide q|gate shape
 * (N = 12288, 768 workgroups) reaches 86 %, so the limit is parallelism, not the decoder.
 *
 * This unit adds a split over K for that band only. `S` contiguous slices of whole 128-K groups run
 * as blockIdx.x (the tile is blockIdx.y, so the resident blocks cover every slice of a few tiles);
 * each slice writes one f32 partial per output column into the row's scratch arena
 * ([S][N] floats, split-major), and a second kernel adds the partials in ascending split order and
 * applies the row scale once, as the unsplit epilogue does.
 *
 * ---- the arithmetic ------------------------------------------------------------------------------
 *
 * Within a slice the kernel is the GEMV: per group the exact i32 sub-sum, then one group term
 * `(float)sub * sa * sb`, groups ascending, so the int32 sums are the reference's. The reducer then
 * forms ((p0 + p1) + p2) + ... in split order and multiplies by row_scale once. That is not the
 * unsplit chain's single ascending sum over all groups: the slice sums bracket the group terms
 * differently, a reassociation of the same f32 additions. The difference is a few f32 ulps, far
 * below the device test's two-ulp bf16 bound (tests/iqr_device_test.cpp). A split is a whole number
 * of 128-K groups, so no scale group straddles two slices.
 *
 * ---- the split rule ------------------------------------------------------------------------------
 *
 * Only M = 1 (the GEMV band) and N <= kIqrSplitMaxN qualify. The split is the largest of 8, 4, 2
 * that divides K / 128, so the slice count is deterministic for a geometry and the scratch hook,
 * the shim and this launcher all agree. Anything else returns 1 and keeps the unsplit path:
 * r3d_gemm_iqr_m16.hip's kernel is untouched.
 */
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_common.h"
#include "r3d_iqr.h"
#include "r3d_iqr_dec.h"
#include "r3d_launch.h"

namespace {

using namespace r3d_iqr;
using namespace r3d_iqr_dev;

constexpr int kIqrWave = 32;
constexpr int kIqrSplitMaxN = 8192;   /* the narrow band: above it N already fills the card */
constexpr int kIqrSplitMax = 8;

/* One slice: the GEMV's group fold over kb in [kbeg, kend), one f32 partial an output column. */
template <class F>
__global__ void __launch_bounds__(kIqrWave) R3D_WAVES(4)
gemm_iqr_m16_split_kernel(const int8_t* __restrict__ a, const float* __restrict__ as,
                          const uint8_t* __restrict__ b, const uint8_t* __restrict__ bs,
                          float* __restrict__ part, int N, int K, uint32_t run_tiles) {
    const uint32_t lane = threadIdx.x & (kIqrWave - 1);
    const uint32_t rt = blockIdx.y;
    const uint32_t nkb = (uint32_t)(K / 128);
    const uint32_t kg = nkb / gridDim.x;          /* whole groups a slice: K % (S * 128) == 0 */
    const uint32_t kbeg = blockIdx.x * kg;
    const uint32_t kend = kbeg + kg;
    Ctx c;
    c.W = b;
    c.S = bs;
    c.N = (uint32_t)N;
    c.nkb = nkb;
    c.run_tiles = run_tiles;
    c.tab = nullptr;

    /* The GEMV's row partials, with the 33-word pitch that avoids the 16-way bank conflict. */
    __shared__ int sP[16 * 33];
    const int row0 = (int)rt * 16;

    float acc = 0.0f;
    for (uint32_t kb = kbeg; kb < kend; ++kb) {
        const typename Dec<F>::Raw r = Dec<F>::fetch(c, rt, kb, lane, true);
        uint32_t w[16];
        Dec<F>::rows(r, c, 1.0f, w);
        Dec<F>::unzero(c, rt, kb, w);
        const uint32_t a4 = *(const uint32_t*)(a + kb * 128 + 4u * lane);
#pragma unroll
        for (int q = 0; q < 16; ++q) sP[q * 33 + lane] = dot4_i8(a4, w[q], 0);
        __syncthreads();
        if (lane < 16) {
            int sub = 0;
#pragma unroll
            for (int l = 0; l < 32; ++l) sub += sP[lane * 33 + l];
            const float sa = as[kb];   /* M = 1: every weight row shares the token's scale */
            const float sb = fold_f32<F>(c, row0 + lane, kb);
            acc += (float)sub * sa * sb;
        }
        __syncthreads();
    }
    if (lane < 16) part[(size_t)blockIdx.x * N + row0 + lane] = acc;
}

/* The partials, ascending split, then the row scale once: one output column a thread. */
template <class F>
__global__ void __launch_bounds__(256)
gemm_iqr_m16_split_reduce_kernel(const float* __restrict__ part, const uint8_t* __restrict__ bs,
                                 uint16_t* __restrict__ y, int N, int splits) {
    const int n = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (n >= N) return;
    float v = part[n];
    for (int s = 1; s < splits; ++s) v += part[(size_t)s * N + n];
    Ctx c{};
    c.S = bs;
    c.N = (uint32_t)N;
    y[n] = f32_to_bf16(v * row_scale<F>(c, (uint32_t)n));
}

template <class F>
static int launch_split(const uint8_t* b, const uint8_t* bs, long a, long as, long y, int N, int K,
                        uint32_t run_tiles, int splits, long scratch, long stream) {
    /* Slices vary fastest, then the tile: the resident blocks cover every slice of a few tiles, so a
     * tile's K run is walked by concurrent waves, not one slice after another. */
    const dim3 grid((unsigned)splits, (unsigned)(N / 16)), block(kIqrWave);
    R3D_KLAUNCH(R3D_KNAME((gemm_iqr_m16_split_kernel<F>)), grid, block, (hipStream_t)stream,
                (const int8_t*)a, (const float*)as, b, bs, (float*)scratch, N, K, run_tiles);
    if (hipGetLastError() != hipSuccess) return -4;
    constexpr int kThreads = 256;
    const dim3 rgrid((unsigned)((N + kThreads - 1) / kThreads)), rblock(kThreads);
    R3D_KLAUNCH(gemm_iqr_m16_split_reduce_kernel<F>, rgrid, rblock, (hipStream_t)stream,
                (const float*)scratch, bs, (uint16_t*)y, N, splits);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

}  // namespace

using namespace r3d_iqr;
using namespace r3d_iqr_dev;

/* The split rule, shared by the row's scratch hook (r3d_rows.cpp), the shim
 * (r3d_gemm_iqr_m16.hip) and the launcher below. 1 means the unsplit path. */
extern "C" int r3d_iqr_split_slices(long long N, long long K) {
    /* The row's shape contract: N a multiple of 16, K of 128. Anything else is not this path. */
    if (N < 16 || N > kIqrSplitMaxN || (N & 15) || K < 128 || (K & 127)) return 1;
    const long long nkb = K / 128;
    if (nkb < 2) return 1;
    if (nkb % kIqrSplitMax == 0) return kIqrSplitMax;
    if (nkb % 4 == 0) return 4;
    if (nkb % 2 == 0) return 2;
    return 1;
}

/* The launcher. The shim has validated the geometry, the form and the scratch size. */
extern "C" int r3d_gemm_iqr_splitk(long a, long as, long b, long bs, long y, int N, int K,
                                   uint32_t word, int splits, long scratch, long stream) {
    if (!a || !as || !b || !bs || !y || !scratch || splits < 2) return -1;
    /* Every K byte must be read exactly once: GEMV's shape contract, and whole slices. */
    if (N < 16 || (N & 15) || K < 128 || (K & 127)) return -2;
    if ((K / 128) % splits) return -2;
    const Form f = form(word);
    const uint32_t lrun = (uint32_t)((word >> 22) & 15u);
    const uint32_t run_tiles = (1u << lrun) / 16u;
    const uint8_t* B_ = (const uint8_t*)b;
    const uint8_t* S_ = (const uint8_t*)bs;

#define R3D_IQR_SPLIT_CASE(B, SC)                                                                 \
    do {                                                                                          \
        using F = FormT<kTrellisMad, B, 0, 16, 0, SC>;                                             \
        return launch_split<F>(B_, S_, a, as, y, N, K, run_tiles, splits, scratch, stream);        \
    } while (0)
#define R3D_IQR_SPLIT_SC(B)                                                                       \
    do {                                                                                          \
        if (f.scale == kRowF32) { R3D_IQR_SPLIT_CASE(B, kRowF32); }                                \
        R3D_IQR_SPLIT_CASE(B, kBf16);                                                              \
    } while (0)

    switch (f.bits) {
        case 2: R3D_IQR_SPLIT_SC(2);
        case 3: R3D_IQR_SPLIT_SC(3);
        case 4: R3D_IQR_SPLIT_SC(4);
        case 5: R3D_IQR_SPLIT_SC(5);
        case 6: R3D_IQR_SPLIT_SC(6);
        case 7: R3D_IQR_SPLIT_SC(7);
        default: return -6;
    }
#undef R3D_IQR_SPLIT_SC
#undef R3D_IQR_SPLIT_CASE
}
