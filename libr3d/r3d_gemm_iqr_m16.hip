/* r3d_gemm_iqr_m16.hip -- gemm_nt_q over an IQR weight (r3d_iqr.h) at M <= 16, on gfx11.
 *
 * W0/W3 slice: the computed trellis (`tcq_code=mad`) with one f32 scale a row and a factor a 128-K
 * block, or a bf16 scale a (row, 128). The table and grid forms follow with the MoE lot; the shim
 * refuses them explicitly.
 *
 * ---- the arithmetic ------------------------------------------------------------------------------
 *
 * The reference grid is the int8 one (docs/OPS.md): per 128-K group,
 *   sub = sum over the group of a[m][k] * w[n][k], an exact integer
 *   acc += (sub * a_scale[m][g]) * w_scale[n][g]
 * and, under one scale a row (kRowF32), the row's own factor is taken in the epilogue. The kernel
 * reproduces that order: groups ascend, each group's sub is an i32 dot, the two f32 factors
 * multiply left to right, and the epilogue multiplies the row scale once.
 *
 * ---- the tile --------------------------------------------------------------------------------------
 *
 * A workgroup is one wave and serves one 16-row tile of the weight. Each lane fetches its lane
 * segment of a 16x128 unit (four columns of all 16 rows), decodes it to 16 packed dwords, and:
 *   - at M = 1 (the GEMV band) dots its four columns with the activation's four bytes, writes the
 *     16 row partials to LDS, and lanes 0..15 fold the group and take the scales;
 *   - above 1 the tile is staged in LDS as 16 rows of 128 int8, then each lane owns whole
 *     (token, row) pairs, each a full 128-K dot of 32 v_dot4, so no cross-lane fold is needed.
 *
 * The tiled row, above the decode band, serves many tokens from the same decoded unit: its
 * workgroup is four waves, each owning sixteen tokens, and wave 0 decodes the unit once per K
 * block into LDS while all four waves consume it (gemm_iqr_resident_kernel, below). The
 * arithmetic per (token, row) is this kernel's, so the two are bit-identical.
 *
 * K and N must be multiples of 128 and 16 and the trellis run must divide N; the shim refuses
 * everything else.
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
constexpr int kMTMax = 16;
/* i32 sub of four signed int8 pairs, exact: gfx11's dot4 is signed x unsigned, so both operands
 * are negated (libr4d's form), which is the signed product (r4d_gemm_iqr.hip:284). */
__device__ __forceinline__ int dot4_i8(uint32_t a4, uint32_t w4, int acc) {
    return __builtin_amdgcn_sudot4(true, (int)w4, true, (int)a4, acc, false);
}

template <class F, int MT>
__global__ void __launch_bounds__(kIqrWave) R3D_WAVES(4)
gemm_iqr_m16_kernel(const int8_t* __restrict__ a, const float* __restrict__ as,
                    const uint8_t* __restrict__ b, const uint8_t* __restrict__ bs,
                    uint16_t* __restrict__ y, int M, int N, int K, int a_ld, int y_ld,
                    uint32_t run_tiles) {
    const uint32_t lane = threadIdx.x & (kIqrWave - 1);
    const uint32_t rt = blockIdx.x;
    const uint32_t nkb = (uint32_t)(K / 128);
    Ctx c;
    c.W = b;
    c.S = bs;
    c.N = (uint32_t)N;
    c.nkb = nkb;
    c.run_tiles = run_tiles;
    c.tab = nullptr;   /* the W0 forms compute their values; the table forms stage LDS with W5 */

    /* One 16x128 int8 tile, and the GEMV's row partials (a stride of 33 words avoids a 16-way bank
     * conflict when the 16 row owners read one lane column). */
    __shared__ uint8_t sW[16 * 128];
    __shared__ int sP[16 * 33];

    const int row0 = (int)rt * 16;

    if constexpr (MT == 1) {
        float acc = 0.0f;
        for (uint32_t kb = 0; kb < nkb; ++kb) {
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
        if (lane < 16)
            y[(long long)0 * y_ld + row0 + lane] =
                f32_to_bf16(acc * row_scale<F>(c, row0 + lane));
        return;
    }

    constexpr int NP = (MT * 16 + kIqrWave - 1) / kIqrWave;   /* (token, row) pairs a lane */
    float acc[NP];
#pragma unroll
    for (int i = 0; i < NP; ++i) acc[i] = 0.0f;

    for (uint32_t kb = 0; kb < nkb; ++kb) {
        const typename Dec<F>::Raw r = Dec<F>::fetch(c, rt, kb, lane, true);
        uint32_t w[16];
        Dec<F>::rows(r, c, 1.0f, w);
        Dec<F>::unzero(c, rt, kb, w);
#pragma unroll
        for (int q = 0; q < 16; ++q) *(uint32_t*)(sW + q * 128 + 4u * lane) = w[q];
        __syncthreads();

#pragma unroll
        for (int i = 0; i < NP; ++i) {
            const int p = (int)lane + kIqrWave * i;
            if constexpr (MT * 16 < kIqrWave) {
                if (p >= MT * 16) continue;
            }
            const int m = p / 16, q = p % 16;
            if (m >= M) continue;
            int sub = 0;
            const int8_t* ap = a + (long long)m * a_ld + (long long)kb * 128;
            const uint32_t* wp = (const uint32_t*)(sW + q * 128);
#pragma unroll
            for (int c4 = 0; c4 < 32; ++c4)
                sub = dot4_i8(*(const uint32_t*)(ap + 4 * c4), wp[c4], sub);
            const float sa = as[m * nkb + kb];
            const float sb = fold_f32<F>(c, row0 + q, kb);
            acc[i] += (float)sub * sa * sb;
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < NP; ++i) {
        const int p = (int)lane + kIqrWave * i;
        if constexpr (MT * 16 < kIqrWave) {
            if (p >= MT * 16) continue;
        }
        const int m = p / 16, q = p % 16;
        if (m >= M) continue;
        y[(long long)m * y_ld + row0 + q] = f32_to_bf16(acc[i] * row_scale<F>(c, row0 + q));
    }
}

/* ---- the resident prefill tile --------------------------------------------------------------------
 *
 * Above the decode band the tiled row serves many tokens from one weight tile. The decode-band kernel
 * above already keeps the decoded 16x128 tile for its 16 tokens; the resident kernel keeps the same
 * unit across a workgroup's W 16-token slices, so the weight's codes are fetched and decoded once
 * per (16-row tile, 128-K block) and reused W times instead of once:
 *
 *   - a workgroup is W waves, wave w owning tokens [m0 + 16w, m0 + 16w + 16);
 *   - wave 0 decodes the unit once (all 32 lane segments) into LDS, padded to 33 dwords a row so the
 *     16 rows' owners read 16 different banks;
 *   - a barrier, then every wave runs the decode band's (token, row) pairs over their own tokens;
 *   - a second barrier, then the next K block.
 *
 * Per (token, row) the arithmetic is the decode band's, in the same order: groups ascend, each
 * group's sub is the exact i32 sudot4 sum, its two f32 factors multiply left to right, and the
 * epilogue takes the row scale once. The output is therefore bit-identical to r3d_gemm_iqr_m16's
 * (and to the chunked tiled launch it replaces), only the weight traffic changes.
 */
constexpr int kIqrResidentWaves = 4;   /* 16-token slices a workgroup: 64 tokens a weight tile */
constexpr int kIqrSWPitch = 33;        /* dwords a staged weight row: 33 mod 32 = 1, bank-free */

template <class F, int W>
__global__ void __launch_bounds__(W * kIqrWave) R3D_WAVES(4)
gemm_iqr_resident_kernel(const int8_t* __restrict__ a, const float* __restrict__ as,
                         const uint8_t* __restrict__ b, const uint8_t* __restrict__ bs,
                         uint16_t* __restrict__ y, int M, int N, int K, int a_ld, int y_ld,
                         uint32_t run_tiles) {
    const uint32_t lane = threadIdx.x & (kIqrWave - 1);
    const uint32_t wave = threadIdx.x / kIqrWave;
    const uint32_t nkb = (uint32_t)(K / 128);
    const uint32_t nm = (uint32_t)((M + 16 * W - 1) / (16 * W));
    /* Consecutive blocks share a weight tile, so the L2 stream a group of blocks reads is one. */
    const uint32_t rt = (uint32_t)blockIdx.x / nm;
    const int m0 = (int)((uint32_t)blockIdx.x % nm) * (16 * W) + (int)wave * 16;
    Ctx c;
    c.W = b;
    c.S = bs;
    c.N = (uint32_t)N;
    c.nkb = nkb;
    c.run_tiles = run_tiles;
    c.tab = nullptr;

    /* The decoded 16x128 int8 tile, shared by the workgroup's W slices. */
    __shared__ uint8_t sW[16 * kIqrSWPitch * 4];
    const int row0 = (int)rt * 16;

    constexpr int NP = 8;   /* (token, row) pairs a lane: 16 tokens x 16 rows over one wave */
    float acc[NP];
#pragma unroll
    for (int i = 0; i < NP; ++i) acc[i] = 0.0f;

    for (uint32_t kb = 0; kb < nkb; ++kb) {
        if (wave == 0) {
            const typename Dec<F>::Raw r = Dec<F>::fetch(c, rt, kb, lane, true);
            uint32_t w[16];
            Dec<F>::rows(r, c, 1.0f, w);
            Dec<F>::unzero(c, rt, kb, w);
#pragma unroll
            for (int q = 0; q < 16; ++q)
                *(uint32_t*)(sW + q * (kIqrSWPitch * 4) + 4u * lane) = w[q];
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < NP; ++i) {
            const int p = (int)lane + kIqrWave * i;
            const int m = m0 + p / 16, q = p % 16;
            if (m >= M) continue;
            int sub = 0;
            const int8_t* ap = a + (long long)m * a_ld + (long long)kb * 128;
            const uint32_t* wp = (const uint32_t*)(sW + q * (kIqrSWPitch * 4));
#pragma unroll
            for (int c4 = 0; c4 < 32; ++c4)
                sub = dot4_i8(*(const uint32_t*)(ap + 4 * c4), wp[c4], sub);
            const float sa = as[m * nkb + kb];
            const float sb = fold_f32<F>(c, row0 + q, kb);
            acc[i] += (float)sub * sa * sb;
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < NP; ++i) {
        const int p = (int)lane + kIqrWave * i;
        const int m = m0 + p / 16, q = p % 16;
        if (m >= M) continue;
        y[(long long)m * y_ld + row0 + q] = f32_to_bf16(acc[i] * row_scale<F>(c, row0 + q));
    }
}

int mt_for(int M) {
    int mt = 1;
    while (mt < M && mt < kMTMax) mt *= 2;
    return mt;
}

/* ------------------------------------------------------------------ the shim */

int form_supported(const Form& f) {
    /* W0: the computed trellis, L = 16, one scale a row or a bf16 scale a (row, 128). */
    return f.kind == kTrellisMad && f.L == 16 && f.lrb == 0 &&
           (f.scale == kRowF32 || f.scale == kBf16) && f.bits >= 2 && f.bits <= 7;
}

}  // namespace

using namespace r3d_iqr;

/* The launcher, on addresses, so the templates stay in this unit. */
template <class F, int MT>
static int launch_mt(const uint8_t* b, const uint8_t* bs, long a, long as, long y, int M, int N,
                     int K, int a_ld, int y_ld, uint32_t run_tiles, long stream) {
    const dim3 grid((unsigned)(N / 16)), block(kIqrWave);
    R3D_KLAUNCH(R3D_KNAME((gemm_iqr_m16_kernel<F, MT>)), grid, block, (hipStream_t)stream,
                (const int8_t*)a, (const float*)as, b, bs, (uint16_t*)y, M, N, K, a_ld, y_ld,
                run_tiles);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

#define R3D_IQR_MT_CASE(B, SC)                                                                    \
    do {                                                                                          \
        using F = FormT<kTrellisMad, B, 0, 16, 0, SC>;                                             \
        switch (mt) {                                                                             \
            case 1:  return launch_mt<F, 1>(B_, S_, a, as, y, M, N, K, a_ld, y_ld, run, stream);   \
            case 2:  return launch_mt<F, 2>(B_, S_, a, as, y, M, N, K, a_ld, y_ld, run, stream);   \
            case 4:  return launch_mt<F, 4>(B_, S_, a, as, y, M, N, K, a_ld, y_ld, run, stream);   \
            case 8:  return launch_mt<F, 8>(B_, S_, a, as, y, M, N, K, a_ld, y_ld, run, stream);   \
            default: return launch_mt<F, 16>(B_, S_, a, as, y, M, N, K, a_ld, y_ld, run, stream);  \
        }                                                                                         \
    } while (0)

/* The resident tile's launch: one workgroup a (weight tile, 64-token block). */
template <class F>
static int launch_resident(const uint8_t* b, const uint8_t* bs, long a, long as, long y, int M,
                           int N, int K, int a_ld, int y_ld, uint32_t run_tiles, long stream) {
    constexpr int W = kIqrResidentWaves;
    const int nt = N / 16;
    const int nm = (M + 16 * W - 1) / (16 * W);
    const dim3 grid((unsigned)(nt * nm)), block(W * kIqrWave);
    R3D_KLAUNCH(R3D_KNAME((gemm_iqr_resident_kernel<F, W>)), grid, block, (hipStream_t)stream,
                (const int8_t*)a, (const float*)as, b, bs, (uint16_t*)y, M, N, K, a_ld, y_ld,
                run_tiles);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_gemm_iqr_m16(long a, long as, long b, long bs, long y, int M, int N, int K,
                                int a_ld, int y_ld, uint32_t word, long stream) {
    const Form f = form(word);
    if (!form_supported(f)) return -6;
    if (!a || !as || !b || !bs || !y) return -1;
    if (M < 1 || M > kMTMax || N < 1 || K < 128) return -2;
    if (N % 16 || K % 128) return -2;
    const unsigned lrun = (unsigned)((word >> 22) & 15u);
    if (lrun < 4 || (N % (int)(1u << lrun))) return -2;
    const uint32_t run = (1u << lrun) / 16u;
    const int mt = mt_for(M);
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = N;
    if (a_ld < K || a_ld % 4) return -3;
    if (y_ld < N) return -3;
    if (((uintptr_t)a & 3u) || ((uintptr_t)as & 3u) || ((uintptr_t)b & 15u) ||
        ((uintptr_t)bs & 15u) || ((uintptr_t)y & 1u))
        return -5;
    const uint8_t* B_ = (const uint8_t*)b;
    const uint8_t* S_ = (const uint8_t*)bs;

#define R3D_IQR_SC_CASE(B)                                                                        \
    do {                                                                                          \
        if (f.scale == kRowF32) { R3D_IQR_MT_CASE(B, kRowF32); }                                  \
        R3D_IQR_MT_CASE(B, kBf16);                                                                \
    } while (0)

    switch (f.bits) {
        case 2: R3D_IQR_SC_CASE(2);
        case 3: R3D_IQR_SC_CASE(3);
        case 4: R3D_IQR_SC_CASE(4);
        case 5: R3D_IQR_SC_CASE(5);
        case 6: R3D_IQR_SC_CASE(6);
        case 7: R3D_IQR_SC_CASE(7);
        default: return -6;
    }
#undef R3D_IQR_SC_CASE
#undef R3D_IQR_MT_CASE
}

/* The tiled row's launch: the resident tile (gemm_iqr_resident_kernel). A workgroup decodes a
 * 16x128 weight unit once per K block and reuses it across its four 16-token slices, so the weight
 * is read once per 64 tokens instead of once per 16. The arithmetic per (token, row) is the decode
 * band's, so the result is bit-identical to the chunked launch this replaces. */
extern "C" int r3d_gemm_iqr_tiled(long a, long as, long b, long bs, long y, int M, int N, int K,
                                  int a_ld, int y_ld, uint32_t word, long stream) {
    const Form f = form(word);
    if (!form_supported(f)) return -6;
    if (!a || !as || !b || !bs || !y) return -1;
    if (M < 1 || N < 1 || K < 128) return -2;
    if (N % 16 || K % 128) return -2;
    const unsigned lrun = (unsigned)((word >> 22) & 15u);
    if (lrun < 4 || (N % (int)(1u << lrun))) return -2;
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = N;
    if (a_ld < K || a_ld % 4 || y_ld < N) return -3;
    if (((uintptr_t)a & 3u) || ((uintptr_t)as & 3u) || ((uintptr_t)b & 15u) ||
        ((uintptr_t)bs & 15u) || ((uintptr_t)y & 1u))
        return -5;
    const uint8_t* B_ = (const uint8_t*)b;
    const uint8_t* S_ = (const uint8_t*)bs;

#define R3D_IQR_RES_CASE(B, SC)                                                                   \
    do {                                                                                          \
        using F = FormT<kTrellisMad, B, 0, 16, 0, SC>;                                             \
        return launch_resident<F>(B_, S_, a, as, y, M, N, K, a_ld, y_ld, run, stream);             \
    } while (0)
#define R3D_IQR_RES_SC(B)                                                                         \
    do {                                                                                          \
        if (f.scale == kRowF32) { R3D_IQR_RES_CASE(B, kRowF32); }                                  \
        R3D_IQR_RES_CASE(B, kBf16);                                                                \
    } while (0)

    const uint32_t run = (1u << lrun) / 16u;
    switch (f.bits) {
        case 2: R3D_IQR_RES_SC(2);
        case 3: R3D_IQR_RES_SC(3);
        case 4: R3D_IQR_RES_SC(4);
        case 5: R3D_IQR_RES_SC(5);
        case 6: R3D_IQR_RES_SC(6);
        case 7: R3D_IQR_RES_SC(7);
        default: return -6;
    }
#undef R3D_IQR_RES_SC
#undef R3D_IQR_RES_CASE
}

extern "C" int r3d_rad_gemm_nt_q_iqr_m16(const RadArgs* arg, RadStream s) {
    if (!arg) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(arg, "dtype");
    if (dt && std::strcmp(dt, "i8a8h") != 0) return R3D_NO(RAD_E_DTYPE);

    const RadTensor* A = r3d_opt(arg, 0);
    const RadTensor* AS = r3d_opt(arg, 1);
    const RadTensor* B = r3d_opt(arg, 2);
    const RadTensor* BS = r3d_opt(arg, 3);
    const RadTensor* Y = r3d_opt(arg, 4);
    const RadTensor* ASUM = r3d_opt(arg, 5);
    const RadTensor* BREF = r3d_opt(arg, 6);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(BS) || !r3d_p(Y)) return R3D_NO(RAD_E_INVAL);
    if (r3d_has(ASUM) || r3d_has(BREF)) return R3D_NO(RAD_E_DTYPE);

    if (!r3d_is_i8(A)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(AS)) return R3D_NO(RAD_E_DTYPE);
    if (!is_weight_dtype(B->dtype) || !is_scale_dtype(BS->dtype)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_bf16(Y)) return R3D_NO(RAD_E_DTYPE);

    const Form fw = form(B->dtype), fs = form(BS->dtype);
    if (kernel_word(B->dtype) != kernel_word(BS->dtype)) return R3D_NO(RAD_E_DTYPE);
    if (!form_supported(fw)) return R3D_NO(RAD_E_UNSUPPORTED);
    (void)fs;

    const long long N = r3d_geti_or(arg, "N", 0), K = r3d_geti_or(arg, "K", 0);
    if (N < 1 || K < 1) return R3D_NO(RAD_E_SHAPE);
    if (K % 128 || N % 16) return R3D_NO(RAD_E_SHAPE);
    if (r3d_geti_or(arg, "group", 128) != 128) return R3D_NO(RAD_E_SHAPE);
    const unsigned lrun = (unsigned)((B->dtype >> 22) & 15u);
    if (lrun < 4 || (N % (long long)(1u << lrun))) return R3D_NO(RAD_E_SHAPE);

    const long long ynel = r3d_numel(Y);
    if (ynel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = ynel / N;
    if (M < 1 || M > kMTMax) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(A) < M * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(AS) < M * (K / 128)) return R3D_NO(RAD_E_SHAPE);
    /* The private planes are described by their layout (r3d_iqr.h): [N, K] and the form's scale
     * plane with the stored dtype; the core allocated layout.bytes for each. A `block=1x*` row
     * scale lays its plane out over N*K logical entries (packed), not [N, K/128], so the scale
     * operand's extent is the layout's, not the group count. */
    if (r3d_numel(B) != N * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(BS) < 1) return R3D_NO(RAD_E_SHAPE);

    const long long ap = r3d_row_pitch(A);
    if (ap < K || ap % 4) return R3D_NO(RAD_E_STRIDE);
    long long yrows = 0;
    const long long yp = r3d_rows_pitch(Y, N, &yrows);
    if (yp < N || yrows < M) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(r3d_p(A), 4) || !r3d_aligned(r3d_p(AS), 4) || !r3d_aligned(r3d_p(B), 16) ||
        !r3d_aligned(r3d_p(BS), 16) || !r3d_aligned(r3d_p(Y), 2))
        return R3D_NO(RAD_E_ALIGN);

    const int rc = r3d_gemm_iqr_m16(r3d_addr(A), r3d_addr(AS), r3d_addr(B), r3d_addr(BS),
                                    r3d_addr(Y), (int)M, (int)N, (int)K, (int)ap, (int)yp,
                                    B->dtype, (long)s);
    if (rc == -1) return R3D_NO(RAD_E_INVAL);
    if (rc == -2) return R3D_NO(RAD_E_SHAPE);
    if (rc == -3) return R3D_NO(RAD_E_STRIDE);
    if (rc == -5) return R3D_NO(RAD_E_ALIGN);
    if (rc == -6) return R3D_NO(RAD_E_UNSUPPORTED);
    if (rc == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}

extern "C" int r3d_rad_gemm_nt_q_iqr_tiled(const RadArgs* arg, RadStream s) {
    if (!arg) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(arg, "dtype");
    if (dt && std::strcmp(dt, "i8a8h") != 0) return R3D_NO(RAD_E_DTYPE);

    const RadTensor* A = r3d_opt(arg, 0);
    const RadTensor* AS = r3d_opt(arg, 1);
    const RadTensor* B = r3d_opt(arg, 2);
    const RadTensor* BS = r3d_opt(arg, 3);
    const RadTensor* Y = r3d_opt(arg, 4);
    const RadTensor* ASUM = r3d_opt(arg, 5);
    const RadTensor* BREF = r3d_opt(arg, 6);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(BS) || !r3d_p(Y)) return R3D_NO(RAD_E_INVAL);
    if (r3d_has(ASUM) || r3d_has(BREF)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_i8(A) || !r3d_is_f32(AS) || !r3d_is_bf16(Y)) return R3D_NO(RAD_E_DTYPE);
    if (!is_weight_dtype(B->dtype) || !is_scale_dtype(BS->dtype)) return R3D_NO(RAD_E_DTYPE);

    const Form fw = form(B->dtype);
    if (kernel_word(B->dtype) != kernel_word(BS->dtype)) return R3D_NO(RAD_E_DTYPE);
    if (!form_supported(fw)) return R3D_NO(RAD_E_UNSUPPORTED);

    const long long N = r3d_geti_or(arg, "N", 0), K = r3d_geti_or(arg, "K", 0);
    if (N < 1 || K < 1 || K % 128 || N % 16) return R3D_NO(RAD_E_SHAPE);
    if (r3d_geti_or(arg, "group", 128) != 128) return R3D_NO(RAD_E_SHAPE);
    const unsigned lrun = (unsigned)((B->dtype >> 22) & 15u);
    if (lrun < 4 || (N % (long long)(1u << lrun))) return R3D_NO(RAD_E_SHAPE);

    const long long ynel = r3d_numel(Y);
    if (ynel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = ynel / N;
    if (M < 1) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(A) < M * K || r3d_numel(AS) < M * (K / 128)) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(B) != N * K || r3d_numel(BS) < 1) return R3D_NO(RAD_E_SHAPE);

    const long long ap = r3d_row_pitch(A);
    if (ap < K || ap % 4) return R3D_NO(RAD_E_STRIDE);
    long long yrows = 0;
    const long long yp = r3d_rows_pitch(Y, N, &yrows);
    if (yp < N || yrows < M) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(r3d_p(A), 4) || !r3d_aligned(r3d_p(AS), 4) || !r3d_aligned(r3d_p(B), 16) ||
        !r3d_aligned(r3d_p(BS), 16) || !r3d_aligned(r3d_p(Y), 2))
        return R3D_NO(RAD_E_ALIGN);

    const int rc = r3d_gemm_iqr_tiled(r3d_addr(A), r3d_addr(AS), r3d_addr(B), r3d_addr(BS),
                                      r3d_addr(Y), (int)M, (int)N, (int)K, (int)ap, (int)yp,
                                      B->dtype, (long)s);
    if (rc == -1) return R3D_NO(RAD_E_INVAL);
    if (rc == -2) return R3D_NO(RAD_E_SHAPE);
    if (rc == -3) return R3D_NO(RAD_E_STRIDE);
    if (rc == -5) return R3D_NO(RAD_E_ALIGN);
    if (rc == -6) return R3D_NO(RAD_E_UNSUPPORTED);
    if (rc == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}
