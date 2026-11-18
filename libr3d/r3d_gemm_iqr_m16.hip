/* r3d_gemm_iqr_m16.hip -- gemm_nt_q over an IQR weight (r3d_iqr.h) at M <= 16, on gfx11.
 *
 * W0/W3 slice: the computed trellis (`tcq_code=mad`) with one f32 scale a row and a factor a 128-K
 * block, or a bf16 scale a (row, 128); W14 adds IQR-V1's scalar codebook (`kTable`, u2..u5) and
 * vector codebook (`kGrid`, u8/u9/u10/u12, V = 4), both under a bf16 scale a (row, 128); W15 adds
 * the rotated int8 grid (`kSigned`, 8 bits) under the same bf16 scale, which one row serves beside
 * the computed trellis of its geometry. The other forms the container can declare (E4M3 scales,
 * tcq_rb grids, the read trellis, zero codes) stay refused: no kernel and no measured container
 * weight exercises them yet.
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
 * The computed trellis at prefill heights (M >= kIqrPairMinM) takes the register-tiled kernel
 * instead (gemm_iqr_resident_pair_kernel, below): each lane owns two tokens by four rows, and the
 * unit is decoded with all four waves in parallel, one column a lane. The arithmetic per
 * (token, row) is still this kernel's, in the same order, so its output is bit-identical too.
 *
 * Above kIqrWmmaMinM rows the tiled row runs the same residency on the int8 wave matrix
 * (gemm_iqr_wmma_kernel, below): the decoded 16x128 unit is the B fragment, a 16-token
 * activation tile the A fragment, and the eight 16x16x16 steps of a group chain one exact i32
 * sub-sum. The group fold and the epilogue are the resident tile's, in the same order, so the
 * output is bit-identical too; only the issue width changes. Small M keeps the v_dot4 tile.
 * W13 measured the wave matrix 25-30 % slower than the v_dot4 tile at every prefill shape, so
 * kIqrWmmaMinM sits above any M: the path stays built and covered, switched off, for W14.
 *
 * K and N must be multiples of 128 and 16 and the trellis run must divide N; the shim refuses
 * everything else.
 *
 * ---- split-K ------------------------------------------------------------------------------------
 *
 * At M = 1 with a narrow N (the `down` shape) one wave a 16-row tile leaves the card underfilled.
 * The shim hands that band to r3d_gemm_iqr_splitk.hip when the row's scratch holds the partials the
 * scratch hook sizes; the kernel below is untouched and serves every other geometry, so a caller
 * without scratch still gets the exact bytes it got before. The split's arithmetic and its one
 * documented rounding difference are described in that unit.
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

template <class F, int MT, int GS>
__global__ void __launch_bounds__(kIqrWave) R3D_WAVES(4)
gemm_iqr_m16_kernel(const int8_t* __restrict__ a, const float* __restrict__ as,
                    const uint8_t* __restrict__ b, const uint8_t* __restrict__ bs,
                    uint16_t* __restrict__ y, int M, int N, int K, int a_ld, int y_ld,
                    uint32_t run_tiles) {
    static_assert(GS == 1 || GS == 2, "a scale group is one or two 128-K blocks");
    const uint32_t lane = threadIdx.x & (kIqrWave - 1);
    const uint32_t rt = blockIdx.x;
    const uint32_t nkb = (uint32_t)(K / 128);
    const uint32_t ngrp = nkb / (uint32_t)GS;
    Ctx c;
    c.W = b;
    c.S = bs;
    c.N = (uint32_t)N;
    c.nkb = nkb;
    c.run_tiles = run_tiles;
    c.tab = nullptr;
    c.ngrp = ngrp;
    c.gs = (uint32_t)GS;

    /* One 16x128 int8 tile, the GEMV's row partials (a stride of 33 words avoids a 16-way bank
     * conflict when the 16 row owners read one lane column), and the form's table when it has one. */
    __shared__ uint8_t sW[16 * 128];
    __shared__ int sP[16 * 33];
    __shared__ __align__(16) int8_t sTab[Dec<F>::kTab];
    if constexpr (Dec<F>::kTab > 4) {   /* a table form; the computed trellis reads none */
        Dec<F>::stage(sTab, c, threadIdx.x, blockDim.x);
        __syncthreads();
        c.tab = sTab;
    }

    const int row0 = (int)rt * 16;

    if constexpr (MT == 1) {
        float acc = 0.0f;
        for (uint32_t g = 0; g < ngrp; ++g) {
            int sub = 0;
            for (int j = 0; j < GS; ++j) {
                const uint32_t kb = g * GS + (uint32_t)j;
                const typename Dec<F>::Raw r = Dec<F>::fetch(c, rt, kb, lane, true);
                uint32_t w[16];
                Dec<F>::rows(r, c, 1.0f, w);
                Dec<F>::unzero(c, rt, kb, w);
                const uint32_t a4 = *(const uint32_t*)(a + kb * 128 + 4u * lane);
#pragma unroll
                for (int q = 0; q < 16; ++q) sP[q * 33 + lane] = dot4_i8(a4, w[q], 0);
                __syncthreads();
                if (lane < 16) {
#pragma unroll
                    for (int l = 0; l < 32; ++l) sub += sP[lane * 33 + l];
                }
                __syncthreads();
            }
            if (lane < 16) {
                const float sa = as[g];   /* M = 1: every weight row shares the token's scale */
                const float sb = fold_f32<F>(c, row0 + lane, g * GS);
                acc += (float)sub * sa * sb;
            }
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

    for (uint32_t g = 0; g < ngrp; ++g) {
        int sub[NP];
#pragma unroll
        for (int i = 0; i < NP; ++i) sub[i] = 0;
        for (int j = 0; j < GS; ++j) {
            const uint32_t kb = g * GS + (uint32_t)j;
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
                const int8_t* ap = a + (long long)m * a_ld + (long long)kb * 128;
                const uint32_t* wp = (const uint32_t*)(sW + q * 128);
#pragma unroll
                for (int c4 = 0; c4 < 32; ++c4)
                    sub[i] = dot4_i8(*(const uint32_t*)(ap + 4 * c4), wp[c4], sub[i]);
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
            const float sa = as[m * (int)ngrp + (int)g];
            const float sb = fold_f32<F>(c, row0 + q, g * GS);
            acc[i] += (float)sub[i] * sa * sb;
        }
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

template <class F, int W, int GS>
__global__ void __launch_bounds__(W * kIqrWave) R3D_WAVES(4)
gemm_iqr_resident_kernel(const int8_t* __restrict__ a, const float* __restrict__ as,
                         const uint8_t* __restrict__ b, const uint8_t* __restrict__ bs,
                         uint16_t* __restrict__ y, int M, int N, int K, int a_ld, int y_ld,
                         uint32_t run_tiles) {
    static_assert(GS == 1 || GS == 2, "a scale group is one or two 128-K blocks");
    const uint32_t lane = threadIdx.x & (kIqrWave - 1);
    const uint32_t wave = threadIdx.x / kIqrWave;
    const uint32_t nkb = (uint32_t)(K / 128);
    const uint32_t ngrp = nkb / (uint32_t)GS;
    const uint32_t nm = ((uint32_t)M + 16u * W - 1u) / (16u * W);
    /* Consecutive blocks share a weight tile, so the L2 stream a group of blocks reads is one. */
    const uint32_t rt = (uint32_t)blockIdx.x / nm;
    const uint32_t m0 = ((uint32_t)blockIdx.x % nm) * (16u * W) + wave * 16u;
    Ctx c;
    c.W = b;
    c.S = bs;
    c.N = (uint32_t)N;
    c.nkb = nkb;
    c.run_tiles = run_tiles;
    c.tab = nullptr;
    c.ngrp = ngrp;
    c.gs = (uint32_t)GS;

    /* The decoded 16x128 int8 tile, shared by the workgroup's W slices, and the form's table. */
    __shared__ uint32_t sW[16 * kIqrSWPitch];
    __shared__ __align__(16) int8_t sTab[Dec<F>::kTab];
    if constexpr (Dec<F>::kTab > 4) {   /* a table form; the computed trellis reads none */
        Dec<F>::stage(sTab, c, threadIdx.x, blockDim.x);
        __syncthreads();
        c.tab = sTab;
    }
    const int row0 = (int)rt * 16;

    constexpr int NP = 8;   /* (token, row) pairs a lane: 16 tokens x 16 rows over one wave */
    float acc[NP];
#pragma unroll
    for (int i = 0; i < NP; ++i) acc[i] = 0.0f;

    for (uint32_t g = 0; g < ngrp; ++g) {
        int sub[NP];
#pragma unroll
        for (int i = 0; i < NP; ++i) sub[i] = 0;
        for (int j = 0; j < GS; ++j) {
            const uint32_t kb = g * GS + (uint32_t)j;
            if (wave == 0) {
                const typename Dec<F>::Raw r = Dec<F>::fetch(c, rt, kb, lane, true);
                uint32_t w[16];
                Dec<F>::rows(r, c, 1.0f, w);
                Dec<F>::unzero(c, rt, kb, w);
#pragma unroll
                for (int q = 0; q < 16; ++q) sW[q * kIqrSWPitch + lane] = w[q];
            }
            __syncthreads();

#pragma unroll
            for (int i = 0; i < NP; ++i) {
                const int p = (int)lane + kIqrWave * i;
                const uint32_t mu = m0 + (uint32_t)(p / 16);
                if (mu >= (uint32_t)M) continue;
                const int m = (int)mu, q = p % 16;
                const int8_t* ap = a + (long long)m * a_ld + (long long)kb * 128;
                const uint32_t* wp = sW + q * kIqrSWPitch;
#pragma unroll
                for (int c4 = 0; c4 < 32; ++c4)
                    sub[i] = dot4_i8(*(const uint32_t*)(ap + 4 * c4), wp[c4], sub[i]);
            }
            __syncthreads();
        }
#pragma unroll
        for (int i = 0; i < NP; ++i) {
            const int p = (int)lane + kIqrWave * i;
            const uint32_t mu = m0 + (uint32_t)(p / 16);
            if (mu >= (uint32_t)M) continue;
            const int m = (int)mu, q = p % 16;
            const float sa = as[m * (int)ngrp + (int)g];
            const float sb = fold_f32<F>(c, row0 + q, g * GS);
            acc[i] += (float)sub[i] * sa * sb;
        }
    }

#pragma unroll
    for (int i = 0; i < NP; ++i) {
        const int p = (int)lane + kIqrWave * i;
        const uint32_t mu = m0 + (uint32_t)(p / 16);
        if (mu >= (uint32_t)M) continue;
        const int m = (int)mu, q = p % 16;
        y[(long long)m * y_ld + row0 + q] = f32_to_bf16(acc[i] * row_scale<F>(c, row0 + q));
    }
}

/* ---- the wave-matrix prefill tile ----------------------------------------------------------------
 *
 * Above kIqrWmmaMinM rows the tiled row keeps the resident tile's residency and changes only the
 * issue: each group's 128-K sub-sum is eight V_WMMA_I32_16X16X16_IU8 steps over the same exact
 * integer grid. Per 128 products |a * w| <= 127 * 127 and their sum is 2,064,512, far inside i32,
 * so the eight steps chain an exact sub with no rounding; the group then folds in f32 exactly as
 * the resident tile does, in ascending group order, and the row scale is the epilogue's. The
 * output is therefore bit-identical to the v_dot4 tile, not merely within tolerance.
 *
 *   - a workgroup is W waves and serves one 16-row weight tile; wave w owns tokens
 *     [m0 + 16w, m0 + 16w + 16);
 *   - wave 0 decodes the 16x128 unit into LDS once per K block, pitched to 33 dwords, and every
 *     wave reads it as the B fragment (lane l holds weight row l & 15, the step's 16 k);
 *   - each wave stages its own 16 x 128 activation tile into LDS with the A fragment's lane map
 *     (lane l holds token row l & 15), then issues the eight steps;
 *   - the D fragment's lane l element e is row 2e + (l >> 4) of the token tile and column l & 15,
 *     so the scales are: a_scale of the D row, the weight row's fold factor, the row scale last.
 *
 * The activation stage reads dwords, so the shim's 4-byte alignment contract is enough; the
 * fragment read then uses the decoded tile's own pitch. Tail rows re-read the last real row and
 * drop the store.
 */
constexpr int kIqrWmmaWaves = 4;    /* 16-token slices a workgroup: 64 tokens a weight tile */
/* The W13 wave-matrix tile is kept but switched off: measured on the W7900 it is 25-30% slower
 * than the v_dot4 resident tile at every tiled shape (12288x2560 M=512 2796 us against 2120,
 * M=4096 23013 against 17499, 5120x17408 M=512 7375 against 5732, M=4096 66038 against 49617),
 * which also cost the 27B 25% of its prefill. Raise the threshold past a launch's M to force the
 * v_dot4 tile; the 64-token crossover was never demonstrated. Marked for W14: the issue and the
 * staging, not the residency, are what the measurement rejects. */
constexpr int kIqrWmmaMinM = 1 << 30;    /* at or above this the wave matrix runs; raise past M to
                                          * force the v_dot4 tile (the switch, there is no other) */

template <class F, int W, int GS>
__global__ void __launch_bounds__(W * kIqrWave) R3D_WAVES(4)
gemm_iqr_wmma_kernel(const int8_t* __restrict__ a, const float* __restrict__ as,
                     const uint8_t* __restrict__ b, const uint8_t* __restrict__ bs,
                     uint16_t* __restrict__ y, int M, int N, int K, int a_ld, int y_ld,
                     uint32_t run_tiles) {
    static_assert(GS == 1 || GS == 2, "a scale group is one or two 128-K blocks");
    const uint32_t lane = threadIdx.x & (kIqrWave - 1);
    const uint32_t wave = threadIdx.x / kIqrWave;
    const uint32_t nkb = (uint32_t)(K / 128);
    const uint32_t ngrp = nkb / (uint32_t)GS;
    const uint32_t nm = ((uint32_t)M + 16u * W - 1u) / (16u * W);
    const uint32_t rt = (uint32_t)blockIdx.x / nm;
    const uint32_t m0 = ((uint32_t)blockIdx.x % nm) * (16u * W) + wave * 16u;
    Ctx c;
    c.W = b;
    c.S = bs;
    c.N = (uint32_t)N;
    c.nkb = nkb;
    c.run_tiles = run_tiles;
    c.tab = nullptr;
    c.ngrp = ngrp;
    c.gs = (uint32_t)GS;

    /* The decoded 16x128 weight unit (the B fragment), each wave's 16x128 activation tile, and the
     * form's table. */
    __shared__ uint32_t sW[16 * kIqrSWPitch];
    __shared__ uint32_t sA[W][16 * kIqrSWPitch];
    __shared__ __align__(16) int8_t sTab[Dec<F>::kTab];
    if constexpr (Dec<F>::kTab > 4) {   /* a table form; the computed trellis reads none */
        Dec<F>::stage(sTab, c, threadIdx.x, blockDim.x);
        __syncthreads();
        c.tab = sTab;
    }
    const int row0 = (int)rt * 16;

    /* The A fragment's row is lane & 15; the D fragment's rows are 2e + (lane >> 4). */
    const uint32_t q = lane & 15u;
    const uint32_t half = lane >> 4;
    uint32_t dm[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const uint32_t mu = m0 + (uint32_t)(2 * e) + half;
        dm[e] = mu < (uint32_t)M ? mu : (uint32_t)(M - 1);
    }

    v8float acc = v8float{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    for (uint32_t g = 0; g < ngrp; ++g) {
        v8int sub = v8int{ 0, 0, 0, 0, 0, 0, 0, 0 };
        for (int j = 0; j < GS; ++j) {
            const uint32_t kb = g * GS + (uint32_t)j;
            if (wave == 0) {
                const typename Dec<F>::Raw r = Dec<F>::fetch(c, rt, kb, lane, true);
                uint32_t w[16];
                Dec<F>::rows(r, c, 1.0f, w);
                Dec<F>::unzero(c, rt, kb, w);
#pragma unroll
                for (int qq = 0; qq < 16; ++qq) sW[qq * kIqrSWPitch + lane] = w[qq];
            }
            /* Dwords, so the shim's 4-byte alignment is enough; lane l reads dword l of each row. */
            for (int i = (int)lane; i < 16 * (128 / 4); i += kIqrWave) {
                const int row = i / (128 / 4), d = i % (128 / 4);
                const uint32_t rr = m0 + (uint32_t)row < (uint32_t)M ? m0 + (uint32_t)row
                                                                    : (uint32_t)(M - 1);
                sA[wave][row * kIqrSWPitch + d] =
                    *(const uint32_t*)(a + (size_t)rr * a_ld + (size_t)kb * 128 + 4u * (uint32_t)d);
            }
            __syncthreads();

            const uint32_t* ap = sA[wave] + (size_t)q * kIqrSWPitch;
            const uint32_t* wp = sW + (size_t)q * kIqrSWPitch;
#pragma unroll
            for (int k = 0; k < 128; k += 16) {
                const r3d_u32x4 aq = *(const r3d_u32x4*)(ap + k / 4);
                const r3d_u32x4 bq = *(const r3d_u32x4*)(wp + k / 4);
                sub = wmma_i8(v4int{ (int)aq.x, (int)aq.y, (int)aq.z, (int)aq.w },
                              v4int{ (int)bq.x, (int)bq.y, (int)bq.z, (int)bq.w }, sub);
            }
            __syncthreads();
        }
        const float sb = fold_f32<F>(c, row0 + q, g * GS);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float t = (float)sub[e] * as[(size_t)dm[e] * ngrp + g];
            acc[e] += t * sb;
        }
    }

    const float rowsc = row_scale<F>(c, row0 + q);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const uint32_t mu = m0 + (uint32_t)(2 * e) + half;
        if (mu >= (uint32_t)M) continue;
        y[(size_t)mu * y_ld + row0 + q] = f32_to_bf16(acc[e] * rowsc);
    }
}

int mt_for(int M) {
    int mt = 1;
    while (mt < M && mt < kMTMax) mt *= 2;
    return mt;
}

/* ------------------------------------------------------------------ the shim */

int form_supported(const Form& f) {
    /* W0: the computed trellis, L = 16, one scale a row or a bf16 scale a (row, 128). W14 adds the
     * two IQR-V1 forms the decoded-lane kernels already read: a scalar codebook (kTable, u2..u5) and
     * a vector codebook (kGrid, V = 4, u8/u9/u10/u12), both under a bf16 scale a (row, 128). W15
     * adds the rotated int8 grid (kSigned, 8 bits) the same decoded-lane kernels already sign-extend,
     * so a row serves both it and the computed trellis of one geometry. The built() list is the
     * authority for every form, so a width or scale the layout cannot produce is refused here with
     * the same code. */
    if (is_trellis(f.kind))
        return f.kind == kTrellisMad && f.L == 16 && f.lrb == 0 &&
               (f.scale == kRowF32 || f.scale == kBf16) && f.bits >= 2 && f.bits <= 7 &&
               built(kernel_word(word(f)));
    if (f.kind == kSigned)
        return f.bits == 8 && f.scale == kBf16 && f.lv == 0 && f.L == 0 && f.lrb == 0 &&
               built(kernel_word(word(f)));
    if (f.kind != kTable && f.kind != kGrid) return 0;
    if (!built(kernel_word(word(f)))) return 0;
    if (f.kind == kTable)
        return f.scale == kBf16 && f.bits >= 2 && f.bits <= 5 && f.lv == 0 && f.L == 0 &&
               f.lrb == 0;
    return f.scale == kBf16 && f.lv == 2 && f.L == 0 && f.lrb == 0 && f.bits != 11 &&
           f.bits >= 8 && f.bits <= 12;
}

/* Stage a computed trellis's 16x128 unit into sW with all four waves decoding in parallel: each
 * wave takes eight of the unit's 32 lane segments, four lanes a segment and one column a lane.
 * The state chain is per column, so this is the same arithmetic as Dec::rows() (state extraction,
 * the masked hash and its byte sum), only split across the wave's 32 lanes; the byte the value
 * would have packed into column e is stored at that byte, and rows() places state q at w[15-q].
 * The caller barriers before the tile is consumed. */
template <class F>
__device__ __forceinline__ void stage_mad_parallel(uint32_t* sW, const Ctx& c, uint32_t rt,
                                                   uint32_t kb, uint32_t wave, uint32_t lane) {
    static_assert(F::KIND == kTrellisMad && F::LRB == 0,
                  "the parallel trellis stage serves the computed trellis, one scale a row");
    const uint32_t seg = wave * 8u + (lane >> 2u);
    const uint32_t e = lane & 3u;
    const typename Dec<F>::Raw r = Dec<F>::fetch(c, rt, kb, seg, true);
    constexpr uint32_t NW = Dec<F>::NW;
    uint32_t cw[NW], mw[NW];
#pragma unroll
    for (uint32_t i = 0; i < NW; ++i) cw[i] = Dec<F>::colw(r, e, i);
#pragma unroll
    for (uint32_t i = 0; i < NW; ++i)
        mw[i] = i + 1u < NW ? __builtin_amdgcn_alignbit(cw[i + 1u], cw[i], 16u) : 0u;
    constexpr Form f{ F::KIND, F::B, F::LV, F::L, F::LRB, F::SC, 0 };
    const uint32_t* ab = (const uint32_t*)(c.W + aux_off(f, c.N, (int64_t)c.nkb * 128));
    const uint32_t A = ab[0], Bc = ab[1];
    uint8_t* sb = (uint8_t*)sW;
#pragma unroll
    for (uint32_t q = 0; q < 16u; ++q) {
        const uint32_t st = Dec<F>::state1(cw, mw, q);
        const uint32_t h = (__umul24(st, A) + Bc) & 0x3F3F3F3Fu;
        const int v = __builtin_amdgcn_sudot4(false, (int)h, false, 0x01010101, -126, false);
        sb[(15u - q) * (kIqrSWPitch * 4u) + 4u * seg + e] = (uint8_t)v;
    }
}

/* ---- GS=1 path, restored from the W26 kernel: the W27 group loop costs 17% on the
 * 27B (GS=1); GS=2 keeps the grouped kernels below. ---- */

template <class F, int W>
__global__ void __launch_bounds__(W * kIqrWave) R3D_WAVES(4)
gemm_iqr_resident_gs1_kernel(const int8_t* __restrict__ a, const float* __restrict__ as,
                         const uint8_t* __restrict__ b, const uint8_t* __restrict__ bs,
                         uint16_t* __restrict__ y, int M, int N, int K, int a_ld, int y_ld,
                         uint32_t run_tiles) {
    const uint32_t lane = threadIdx.x & (kIqrWave - 1);
    const uint32_t wave = threadIdx.x / kIqrWave;
    const uint32_t nkb = (uint32_t)(K / 128);
    const uint32_t nm = ((uint32_t)M + 16u * W - 1u) / (16u * W);
    /* Consecutive blocks share a weight tile, so the L2 stream a group of blocks reads is one. */
    const uint32_t rt = (uint32_t)blockIdx.x / nm;
    const uint32_t m0 = ((uint32_t)blockIdx.x % nm) * (16u * W) + wave * 16u;
    Ctx c;
    c.W = b;
    c.S = bs;
    c.N = (uint32_t)N;
    c.nkb = nkb;
    c.run_tiles = run_tiles;
    c.tab = nullptr;

    /* The decoded 16x128 int8 tile, shared by the workgroup's W slices, and the form's table. */
    __shared__ uint32_t sW[16 * kIqrSWPitch];
    __shared__ __align__(16) int8_t sTab[Dec<F>::kTab];
    if constexpr (Dec<F>::kTab > 4) {   /* a table form; the computed trellis reads none */
        Dec<F>::stage(sTab, c, threadIdx.x, blockDim.x);
        __syncthreads();
        c.tab = sTab;
    }
    const int row0 = (int)rt * 16;

    constexpr int NP = 8;   /* (token, row) pairs a lane: 16 tokens x 16 rows over one wave */
    float acc[NP];
#pragma unroll
    for (int i = 0; i < NP; ++i) acc[i] = 0.0f;

    for (uint32_t kb = 0; kb < nkb; ++kb) {
        if constexpr (Dec<F>::MAD && F::LRB == 0) {
            stage_mad_parallel<F>(sW, c, rt, kb, wave, lane);
        } else if (wave == 0) {
            const typename Dec<F>::Raw r = Dec<F>::fetch(c, rt, kb, lane, true);
            uint32_t w[16];
            Dec<F>::rows(r, c, 1.0f, w);
            Dec<F>::unzero(c, rt, kb, w);
#pragma unroll
            for (int q = 0; q < 16; ++q) sW[q * kIqrSWPitch + lane] = w[q];
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < NP; ++i) {
            const int p = (int)lane + kIqrWave * i;
            const uint32_t mu = m0 + (uint32_t)(p / 16);
            if (mu >= (uint32_t)M) continue;
            const int m = (int)mu, q = p % 16;
            int sub = 0;
            const int8_t* ap = a + (long long)m * a_ld + (long long)kb * 128;
            const uint32_t* wp = sW + q * kIqrSWPitch;
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
        const uint32_t mu = m0 + (uint32_t)(p / 16);
        if (mu >= (uint32_t)M) continue;
        const int m = (int)mu, q = p % 16;
        y[(long long)m * y_ld + row0 + q] = f32_to_bf16(acc[i] * row_scale<F>(c, row0 + q));
    }
}

/* ---- the register-tiled computed-trellis tile ---------------------------------------------------
 *
 * The resident tile above gives each lane one weight row and eight tokens, so each c4 step is one
 * activation load, one weight load and one dot4: every activation dword feeds one row, and the
 * tile's other fifteen rows load it again. At the 27B's ffn_gate_up (N = 34816, K = 5120) that is
 * 256 activation dwords and 256 weight dwords for 256 dot4 a lane per K block, and the measured
 * 15.6-17.3 TFLOP/s does not move when the trellis decode is split across the four waves (W33:
 * 3-4 %), so the decode is not the limiter; the activation stream and the dot4 chain's latency are.
 *
 * Here each lane owns two tokens by four rows: an 8-output register tile. One c4 step is two
 * activation dwords and four weight dwords feeding eight dot4, so an activation dword is consumed
 * by four rows in registers instead of one, and the load count per dot4 falls from 2 to 0.75. The
 * four row quads of a wave read the same activation pair (a 4-way broadcast). A 4x4 tile was
 * measured on the rotated int8 grid first and is *not* used: its sixteen group sums plus sixteen
 * accumulators spilled 88 VGPRs and ran 24-39 % slower; 2x4 is the widest tile that stays resident.
 *
 * The decode is the parallel stage above, so the kernel's registers hold the 2x4 compute instead of
 * the 64-state decode. The arithmetic per (token, row) is the resident tile's, in the same order:
 * each c4 accumulates in ascending k into the group's exact i32 sub, groups ascend, the group's
 * two f32 factors multiply left to right, and the row scale is the epilogue's. The output is
 * therefore bit-identical. A tail token past M loads the last real row (in bounds) and drops its
 * store. GS = 1 only: the 27B's trunk is one 128-K scale a block.
 */
constexpr int kIqrPairWaves = 4;    /* 16-token slices a workgroup: 64 tokens a weight tile */
constexpr int kIqrPairMinM = 128;   /* served prefill heights; below, the old tile skips idle
                                     * tail outputs instead of clamping them */

template <class F, int W>
__global__ void __launch_bounds__(W * kIqrWave) R3D_WAVES(4)
gemm_iqr_resident_pair_kernel(const int8_t* __restrict__ a, const float* __restrict__ as,
                              const uint8_t* __restrict__ b, const uint8_t* __restrict__ bs,
                              uint16_t* __restrict__ y, int M, int N, int K, int a_ld, int y_ld,
                              uint32_t run_tiles) {
    const uint32_t lane = threadIdx.x & (kIqrWave - 1);
    const uint32_t wave = threadIdx.x / kIqrWave;
    static_assert(F::KIND == kTrellisMad && F::LRB == 0,
                  "the pair tile serves the computed trellis, one scale a row");
    static_assert(W == 4, "the parallel stage gives each of four waves eight segments");
    const uint32_t nkb = (uint32_t)(K / 128);
    constexpr uint32_t kBlockTokens = 16u * (uint32_t)W;
    const uint32_t nm = ((uint32_t)M + kBlockTokens - 1u) / kBlockTokens;
    /* Consecutive blocks share a weight tile, so the L2 stream a group of blocks reads is one. */
    const uint32_t rt = (uint32_t)blockIdx.x / nm;
    const uint32_t m0 = ((uint32_t)blockIdx.x % nm) * kBlockTokens;
    Ctx c;
    c.W = b;
    c.S = bs;
    c.N = (uint32_t)N;
    c.nkb = nkb;
    c.run_tiles = run_tiles;
    c.tab = nullptr;
    c.ngrp = nkb;
    c.gs = 1u;

    __shared__ uint32_t sW[16 * kIqrSWPitch];
    const int row0 = (int)rt * 16;

    /* This lane's token pair and its quad of weight rows. */
    const uint32_t tq = lane >> 2, rq = lane & 3u;
    const uint32_t tm = m0 + wave * 16u + 2u * tq;
    int mrow[2];
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        const uint32_t mt = tm + (uint32_t)t;
        mrow[t] = mt < (uint32_t)M ? (int)mt : M - 1;   /* tail: read in bounds, drop the store */
    }
    const int rbase = row0 + (int)(4u * rq);

    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;

    for (uint32_t kb = 0; kb < nkb; ++kb) {
        stage_mad_parallel<F>(sW, c, rt, kb, wave, lane);
        __syncthreads();

        const int8_t* ap[2];
#pragma unroll
        for (int t = 0; t < 2; ++t)
            ap[t] = a + (long long)mrow[t] * a_ld + (long long)kb * 128;
        const uint32_t* wp = sW + (size_t)(4u * rq) * kIqrSWPitch;

        int sub[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) sub[i] = 0;
#pragma unroll
        for (int c4 = 0; c4 < 32; ++c4) {
            uint32_t av[2], wv[4];
#pragma unroll
            for (int t = 0; t < 2; ++t) av[t] = *(const uint32_t*)(ap[t] + 4 * c4);
#pragma unroll
            for (int r = 0; r < 4; ++r) wv[r] = wp[(size_t)r * kIqrSWPitch + c4];
#pragma unroll
            for (int t = 0; t < 2; ++t)
#pragma unroll
                for (int r = 0; r < 4; ++r)
                    sub[t * 4 + r] = dot4_i8(av[t], wv[r], sub[t * 4 + r]);
        }
        __syncthreads();

        float sa[2], sb[4];
#pragma unroll
        for (int t = 0; t < 2; ++t) sa[t] = as[(size_t)mrow[t] * nkb + kb];
#pragma unroll
        for (int r = 0; r < 4; ++r) sb[r] = fold_f32<F>(c, (uint32_t)(rbase + r), kb);
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            if (tm + (uint32_t)t >= (uint32_t)M) continue;
#pragma unroll
            for (int r = 0; r < 4; ++r)
                acc[t * 4 + r] += (float)sub[t * 4 + r] * sa[t] * sb[r];
        }
    }

    float rs[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) rs[r] = row_scale<F>(c, (uint32_t)(rbase + r));
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        if (tm + (uint32_t)t >= (uint32_t)M) continue;
#pragma unroll
        for (int r = 0; r < 4; ++r)
            y[(long long)mrow[t] * y_ld + rbase + r] = f32_to_bf16(acc[t * 4 + r] * rs[r]);
    }
}

}  // namespace

using namespace r3d_iqr;

/* The launcher, on addresses, so the templates stay in this unit. */
template <class F, int MT, int GS = 1>
static int launch_mt(const uint8_t* b, const uint8_t* bs, long a, long as, long y, int M, int N,
                     int K, int a_ld, int y_ld, uint32_t run_tiles, long stream) {
    const dim3 grid((unsigned)(N / 16)), block(kIqrWave);
    R3D_KLAUNCH(R3D_KNAME((gemm_iqr_m16_kernel<F, MT, GS>)), grid, block, (hipStream_t)stream,
                (const int8_t*)a, (const float*)as, b, bs, (uint16_t*)y, M, N, K, a_ld, y_ld,
                run_tiles);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The decode band's dispatch: one launch a form, MT chosen from M. */
template <uint32_t KIND, uint32_t B, uint32_t LV, uint32_t L, uint32_t LRB, uint32_t SC,
          uint32_t GS = 1>
static int launch_mt_form(const uint8_t* b, const uint8_t* bs, long a, long as, long y, int M,
                          int N, int K, int a_ld, int y_ld, uint32_t run_tiles, long stream,
                          int mt) {
    using F = FormT<KIND, B, LV, L, LRB, SC>;
    switch (mt) {
        case 1:  return launch_mt<F, 1, GS>(b, bs, a, as, y, M, N, K, a_ld, y_ld, run_tiles, stream);
        case 2:  return launch_mt<F, 2, GS>(b, bs, a, as, y, M, N, K, a_ld, y_ld, run_tiles, stream);
        case 4:  return launch_mt<F, 4, GS>(b, bs, a, as, y, M, N, K, a_ld, y_ld, run_tiles, stream);
        case 8:  return launch_mt<F, 8, GS>(b, bs, a, as, y, M, N, K, a_ld, y_ld, run_tiles, stream);
        default: return launch_mt<F, 16, GS>(b, bs, a, as, y, M, N, K, a_ld, y_ld, run_tiles, stream);
    }
}

/* The resident tile's launch: one workgroup a (weight tile, 64-token block). */
template <class F, int GS = 1>
static int launch_resident(const uint8_t* b, const uint8_t* bs, long a, long as, long y, int M,
                           int N, int K, int a_ld, int y_ld, uint32_t run_tiles, long stream) {
    constexpr int W = kIqrResidentWaves;
    const long long nt = N / 16;
    const long long nm = ((long long)M + 16 * W - 1) / (16 * W);
    const long long blocks = nt * nm;
    if (blocks > 0x7FFFFFFFll) return -2;   /* no launch geometry this large */
    const dim3 grid((unsigned)blocks), block(W * kIqrWave);
    R3D_KLAUNCH(R3D_KNAME((gemm_iqr_resident_kernel<F, W, GS>)), grid, block, (hipStream_t)stream,
                (const int8_t*)a, (const float*)as, b, bs, (uint16_t*)y, M, N, K, a_ld, y_ld,
                run_tiles);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The wave-matrix tile's launch: the same (weight tile, 64-token block) geometry, the group on
 * the int8 WMMA. */
template <class F, int GS = 1>
static int launch_wmma(const uint8_t* b, const uint8_t* bs, long a, long as, long y, int M,
                       int N, int K, int a_ld, int y_ld, uint32_t run_tiles, long stream) {
    constexpr int W = kIqrWmmaWaves;
    const long long nt = N / 16;
    const long long nm = ((long long)M + 16 * W - 1) / (16 * W);
    const long long blocks = nt * nm;
    if (blocks > 0x7FFFFFFFll) return -2;   /* no launch geometry this large */
    const dim3 grid((unsigned)blocks), block(W * kIqrWave);
    R3D_KLAUNCH(R3D_KNAME((gemm_iqr_wmma_kernel<F, W, GS>)), grid, block, (hipStream_t)stream,
                (const int8_t*)a, (const float*)as, b, bs, (uint16_t*)y, M, N, K, a_ld, y_ld,
                run_tiles);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

template <class F>
static int launch_resident_gs1(const uint8_t* b, const uint8_t* bs, long a, long as, long y, int M,
                               int N, int K, int a_ld, int y_ld, uint32_t run_tiles, long stream) {
    constexpr int W = kIqrResidentWaves;
    const long long nt = N / 16;
    const long long nm = ((long long)M + 16 * W - 1) / (16 * W);
    const long long blocks = nt * nm;
    if (blocks > 0x7FFFFFFFll) return -2;
    const dim3 grid((unsigned)blocks), block(W * kIqrWave);
    R3D_KLAUNCH(R3D_KNAME((gemm_iqr_resident_gs1_kernel<F, W>)), grid, block, (hipStream_t)stream,
                (const int8_t*)a, (const float*)as, b, bs, (uint16_t*)y, M, N, K, a_ld, y_ld,
                run_tiles);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The register-tiled computed-trellis launch: one workgroup a (weight tile, 64-token block). */
template <class F>
static int launch_resident_pair(const uint8_t* b, const uint8_t* bs, long a, long as, long y, int M,
                                int N, int K, int a_ld, int y_ld, uint32_t run_tiles, long stream) {
    constexpr int W = kIqrPairWaves;
    const long long nt = N / 16;
    const long long nm = ((long long)M + 16 * W - 1) / (16 * W);
    const long long blocks = nt * nm;
    if (blocks > 0x7FFFFFFFll) return -2;
    const dim3 grid((unsigned)blocks), block(W * kIqrWave);
    R3D_KLAUNCH(R3D_KNAME((gemm_iqr_resident_pair_kernel<F, W>)), grid, block, (hipStream_t)stream,
                (const int8_t*)a, (const float*)as, b, bs, (uint16_t*)y, M, N, K, a_ld, y_ld,
                run_tiles);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* The tiled row's dispatch: the same residency and wave-matrix switch for every form. */
template <uint32_t KIND, uint32_t B, uint32_t LV, uint32_t L, uint32_t LRB, uint32_t SC,
          uint32_t GS = 1>
static int launch_tiled_form(const uint8_t* b, const uint8_t* bs, long a, long as, long y, int M,
                             int N, int K, int a_ld, int y_ld, uint32_t run_tiles, long stream) {
    using F = FormT<KIND, B, LV, L, LRB, SC>;
    if constexpr (GS == 1) {
        /* The computed trellis's served prefill heights take the 2x4 register tile; below
         * kIqrPairMinM its padded rows outweigh the load saving, so the old tile stays. */
        if constexpr (F::KIND == kTrellisMad && F::LRB == 0) {
            if (M >= kIqrPairMinM)
                return launch_resident_pair<F>(b, bs, a, as, y, M, N, K, a_ld, y_ld, run_tiles,
                                               stream);
        }
        return launch_resident_gs1<F>(b, bs, a, as, y, M, N, K, a_ld, y_ld, run_tiles, stream);
    }
    if (M >= kIqrWmmaMinM) return launch_wmma<F, GS>(b, bs, a, as, y, M, N, K, a_ld, y_ld,
                                                    run_tiles, stream);
    return launch_resident<F, GS>(b, bs, a, as, y, M, N, K, a_ld, y_ld, run_tiles, stream);
}

/* The split-K slice rule and launcher (r3d_gemm_iqr_splitk.hip). */
extern "C" int r3d_iqr_split_slices(long long N, long long K);
extern "C" int r3d_gemm_iqr_splitk(long a, long as, long b, long bs, long y, int N, int K,
                                   uint32_t word, int splits, long scratch, long stream);

/* The row's scratch hook: [splits][N] f32 partials for the split band, nothing otherwise. The hook
 * runs at declare, where the m16 row's M is the band's upper bound (16), so it keys on N and K
 * only: the split is chosen by the geometry, not by the launch's M. N and K are the logical
 * extents. Attaching scratch makes the row non-concurrent (rad_kernel_concurrent is
 * `scratch == nullptr`), the price of the partials; the narrow band pays it at every height because
 * the hook cannot see a launch's M. */
extern "C" int64_t r3d_rad_gemm_nt_q_iqr_m16_scratch(const RadArgs* a) {
    if (!a) return 0;
    long long N = 0, K = 0;
    if (!rad_args_geti(a, "N", &N) || !rad_args_geti(a, "K", &K)) return 0;
    /* IQ8R's 256-wide group has no split kernel: its two blocks fold once, so the split would cut
     * a group. */
    if (r3d_geti_or(a, "group", 128) != 128) return 0;
    const int splits = r3d_iqr_split_slices(N, K);
    if (splits <= 1) return 0;
    const long long bytes = (long long)splits * N * 4;
    return (bytes + 255) & ~255ll;   /* the arena is page-aligned; keep the request 256-multiple */
}

extern "C" int r3d_gemm_iqr_m16(long a, long as, long b, long bs, long y, int M, int N, int K,
                                int a_ld, int y_ld, uint32_t word, int group, long scratch,
                                long scratch_bytes, long stream) {
    const Form f = form(word);
    if (!form_supported(f)) return -6;
    if (!a || !as || !b || !bs || !y) return -1;
    if (M < 1 || M > kMTMax || N < 1 || K < 128) return -2;
    if (group != 128 && group != 256) return -2;
    if (N % 16 || K % group) return -2;
    /* The 256-wide group is IQ8R's rotated int8 grid; no other form has a layout at that width. */
    if (group == 256 && (f.kind != kSigned || f.bits != 8 || f.scale != kBf16)) return -6;
    const unsigned lrun = (unsigned)((word >> 22) & 15u);
    if (is_trellis(f.kind) && (lrun < 4 || (N % (int)(1u << lrun)))) return -2;
    const uint32_t run = is_trellis(f.kind) ? (1u << lrun) / 16u : 0u;
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

    /* The 256-wide group: two 128-K blocks per fold. */
    if (group == 256)
        return launch_mt_form<kSigned, 8, 0, 0, 0, kBf16, 2>(B_, S_, a, as, y, M, N, K, a_ld, y_ld,
                                                             run, stream, mt);

    /* The split band: the GEMV (M = 1) at a narrow N, when the caller's scratch holds the
     * partials the row's hook asked for. Without it the unsplit path below runs, unchanged, so a
     * direct caller that allocates no scratch still gets the old kernel and bytes. Only the
     * computed trellis has a split kernel; the other forms keep the unsplit GEMV. */
    if (mt == 1 && f.kind == kTrellisMad) {
        const int splits = r3d_iqr_split_slices(N, K);
        if (splits > 1 && scratch && !((uintptr_t)scratch & 3u) &&
            scratch_bytes >= (long long)splits * N * (long long)sizeof(float))
            return r3d_gemm_iqr_splitk(a, as, b, bs, y, N, K, word, splits, scratch, stream);
    }

#define R3D_IQR_M16_FORM(KIND, B, LV, L, LRB, SC)                                                 \
    return launch_mt_form<KIND, B, LV, L, LRB, SC>(B_, S_, a, as, y, M, N, K, a_ld, y_ld, run,    \
                                                    stream, mt)
#define R3D_IQR_M16_SC(KIND, B, LV, L, LRB)                                                       \
    do {                                                                                          \
        if (f.scale == kRowF32) { R3D_IQR_M16_FORM(KIND, B, LV, L, LRB, kRowF32); }               \
        R3D_IQR_M16_FORM(KIND, B, LV, L, LRB, kBf16);                                             \
    } while (0)
#define R3D_IQR_M16_BF16(KIND, B, LV, L, LRB)                                                     \
    R3D_IQR_M16_FORM(KIND, B, LV, L, LRB, kBf16)

    switch (f.kind) {
        case kSigned:
            switch (f.bits) {
                case 8: R3D_IQR_M16_BF16(kSigned, 8, 0, 0, 0);
                default: return -6;
            }
        case kTrellisMad:
            switch (f.bits) {
                case 2: R3D_IQR_M16_SC(kTrellisMad, 2, 0, 16, 0);
                case 3: R3D_IQR_M16_SC(kTrellisMad, 3, 0, 16, 0);
                case 4: R3D_IQR_M16_SC(kTrellisMad, 4, 0, 16, 0);
                case 5: R3D_IQR_M16_SC(kTrellisMad, 5, 0, 16, 0);
                case 6: R3D_IQR_M16_SC(kTrellisMad, 6, 0, 16, 0);
                case 7: R3D_IQR_M16_SC(kTrellisMad, 7, 0, 16, 0);
                default: return -6;
            }
        case kTable:
            switch (f.bits) {
                case 2: R3D_IQR_M16_BF16(kTable, 2, 0, 0, 0);
                case 3: R3D_IQR_M16_BF16(kTable, 3, 0, 0, 0);
                case 4: R3D_IQR_M16_BF16(kTable, 4, 0, 0, 0);
                case 5: R3D_IQR_M16_BF16(kTable, 5, 0, 0, 0);
                default: return -6;
            }
        case kGrid:
            switch (f.bits) {
                case 8: R3D_IQR_M16_BF16(kGrid, 8, 2, 0, 0);
                case 9: R3D_IQR_M16_BF16(kGrid, 9, 2, 0, 0);
                case 10: R3D_IQR_M16_BF16(kGrid, 10, 2, 0, 0);
                case 12: R3D_IQR_M16_BF16(kGrid, 12, 2, 0, 0);
                default: return -6;
            }
        default: return -6;
    }
#undef R3D_IQR_M16_BF16
#undef R3D_IQR_M16_SC
#undef R3D_IQR_M16_FORM
}

/* The tiled row's launch: above kIqrWmmaMinM rows the int8 wave matrix
 * (gemm_iqr_wmma_kernel), below it the v_dot4 resident tile (gemm_iqr_resident_kernel). Both decode
 * a 16x128 weight unit once per K block and reuse it across their four 16-token slices; their
 * arithmetic per (token, row) is the decode band's, so either is bit-identical to the chunked
 * launch this replaces. */
extern "C" int r3d_gemm_iqr_tiled(long a, long as, long b, long bs, long y, int M, int N, int K,
                                  int a_ld, int y_ld, uint32_t word, int group, long stream) {
    const Form f = form(word);
    if (!form_supported(f)) return -6;
    if (!a || !as || !b || !bs || !y) return -1;
    if (M < 1 || N < 1 || K < 128) return -2;
    if (group != 128 && group != 256) return -2;
    if (N % 16 || K % group) return -2;
    if (group == 256 && (f.kind != kSigned || f.bits != 8 || f.scale != kBf16)) return -6;
    const unsigned lrun = (unsigned)((word >> 22) & 15u);
    if (is_trellis(f.kind) && (lrun < 4 || (N % (int)(1u << lrun)))) return -2;
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = N;
    if (a_ld < K || a_ld % 4 || y_ld < N) return -3;
    if (((uintptr_t)a & 3u) || ((uintptr_t)as & 3u) || ((uintptr_t)b & 15u) ||
        ((uintptr_t)bs & 15u) || ((uintptr_t)y & 1u))
        return -5;
    const uint8_t* B_ = (const uint8_t*)b;
    const uint8_t* S_ = (const uint8_t*)bs;
    const uint32_t run = is_trellis(f.kind) ? (1u << lrun) / 16u : 0u;

    if (group == 256)
        return launch_tiled_form<kSigned, 8, 0, 0, 0, kBf16, 2>(B_, S_, a, as, y, M, N, K, a_ld,
                                                                 y_ld, run, stream);

#define R3D_IQR_TILED_FORM(KIND, B, LV, L, LRB, SC)                                               \
    return launch_tiled_form<KIND, B, LV, L, LRB, SC>(B_, S_, a, as, y, M, N, K, a_ld, y_ld, run, \
                                                       stream)
#define R3D_IQR_TILED_SC(KIND, B, LV, L, LRB)                                                     \
    do {                                                                                          \
        if (f.scale == kRowF32) { R3D_IQR_TILED_FORM(KIND, B, LV, L, LRB, kRowF32); }             \
        R3D_IQR_TILED_FORM(KIND, B, LV, L, LRB, kBf16);                                           \
    } while (0)
#define R3D_IQR_TILED_BF16(KIND, B, LV, L, LRB)                                                   \
    R3D_IQR_TILED_FORM(KIND, B, LV, L, LRB, kBf16)

    switch (f.kind) {
        case kSigned:
            switch (f.bits) {
                case 8: R3D_IQR_TILED_BF16(kSigned, 8, 0, 0, 0);
                default: return -6;
            }
        case kTrellisMad:
            switch (f.bits) {
                case 2: R3D_IQR_TILED_SC(kTrellisMad, 2, 0, 16, 0);
                case 3: R3D_IQR_TILED_SC(kTrellisMad, 3, 0, 16, 0);
                case 4: R3D_IQR_TILED_SC(kTrellisMad, 4, 0, 16, 0);
                case 5: R3D_IQR_TILED_SC(kTrellisMad, 5, 0, 16, 0);
                case 6: R3D_IQR_TILED_SC(kTrellisMad, 6, 0, 16, 0);
                case 7: R3D_IQR_TILED_SC(kTrellisMad, 7, 0, 16, 0);
                default: return -6;
            }
        case kTable:
            switch (f.bits) {
                case 2: R3D_IQR_TILED_BF16(kTable, 2, 0, 0, 0);
                case 3: R3D_IQR_TILED_BF16(kTable, 3, 0, 0, 0);
                case 4: R3D_IQR_TILED_BF16(kTable, 4, 0, 0, 0);
                case 5: R3D_IQR_TILED_BF16(kTable, 5, 0, 0, 0);
                default: return -6;
            }
        case kGrid:
            switch (f.bits) {
                case 8: R3D_IQR_TILED_BF16(kGrid, 8, 2, 0, 0);
                case 9: R3D_IQR_TILED_BF16(kGrid, 9, 2, 0, 0);
                case 10: R3D_IQR_TILED_BF16(kGrid, 10, 2, 0, 0);
                case 12: R3D_IQR_TILED_BF16(kGrid, 12, 2, 0, 0);
                default: return -6;
            }
        default: return -6;
    }
#undef R3D_IQR_TILED_BF16
#undef R3D_IQR_TILED_SC
#undef R3D_IQR_TILED_FORM
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
    const long long group = r3d_geti_or(arg, "group", 128);
    if (N < 1 || K < 1) return R3D_NO(RAD_E_SHAPE);
    if (group != 128 && group != 256) return R3D_NO(RAD_E_SHAPE);
    if (K % group || N % 16) return R3D_NO(RAD_E_SHAPE);
    const unsigned lrun = (unsigned)((B->dtype >> 22) & 15u);
    if (is_trellis(fw.kind) && (lrun < 4 || (N % (long long)(1u << lrun))))
        return R3D_NO(RAD_E_SHAPE);

    const long long ynel = r3d_numel(Y);
    if (ynel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = ynel / N;
    if (M < 1 || M > kMTMax) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(A) < M * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(AS) < M * (K / group)) return R3D_NO(RAD_E_SHAPE);
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
                                    B->dtype, (int)group, (long)arg->scratch, arg->scratch_bytes,
                                    (long)s);
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
    const long long group = r3d_geti_or(arg, "group", 128);
    if (N < 1 || K < 1) return R3D_NO(RAD_E_SHAPE);
    if (group != 128 && group != 256) return R3D_NO(RAD_E_SHAPE);
    if (K % group || N % 16) return R3D_NO(RAD_E_SHAPE);
    const unsigned lrun = (unsigned)((B->dtype >> 22) & 15u);
    if (is_trellis(fw.kind) && (lrun < 4 || (N % (long long)(1u << lrun))))
        return R3D_NO(RAD_E_SHAPE);

    const long long ynel = r3d_numel(Y);
    if (ynel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = ynel / N;
    if (M < 1) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(A) < M * K || r3d_numel(AS) < M * (K / group)) return R3D_NO(RAD_E_SHAPE);
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
                                      B->dtype, (int)group, (long)s);
    if (rc == -1) return R3D_NO(RAD_E_INVAL);
    if (rc == -2) return R3D_NO(RAD_E_SHAPE);
    if (rc == -3) return R3D_NO(RAD_E_STRIDE);
    if (rc == -5) return R3D_NO(RAD_E_ALIGN);
    if (rc == -6) return R3D_NO(RAD_E_UNSUPPORTED);
    if (rc == -4) return R3D_NO(RAD_E_DEVICE);
    return RAD_OK;
}
