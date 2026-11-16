/* r3d_iqr_logits.hip -- logits_gemm_vocab over an IQR head, on gfx11.
 *
 * The head is up to four width classes of the vocabulary: each class is an IQR weight of rows<i>
 * rows and its ids, the global vocabulary id of each row. The activation is rotated by the same
 * 128-wide Hadamard the head's codes were trained against and held at sixteen bits: one f32 scale
 * a (token, 128) over integers q = 128 hi + lo, |q| <= 127 * 128, hi and lo int8. Each 128-K
 * block's dot is then two exact int8 sums over the same decoded weight tile, the first scaled by
 * 128 in the integer accumulator: an exact integer per (row, token, block), folded once into f32,
 * and the row's own factor taken in the epilogue. The class's ids scatter the f32 logits; ids
 * outside this rank's share are dropped. Nothing on the path is bf16.
 *
 * W12: this replaces the per-class int8 GEMM and bf16 logits plane. The int8 grid cost the head
 * 2e-5 KL on its own (libr4d/r4d_logits_vocab.hip); sixteen bits are below bf16's rounding, and the
 * accumulation stays exact until the fold. At the decode height (M = 1) the wave's two halves take
 * the hi and lo planes of the same tile and join their exact sums with v_permlanex16, so no lane
 * sits idle and the dot chain halves; the bits are unchanged.
 *
 * W13: one launch covers every class. The grid's z axis selects a class, x its 16-row tiles and y
 * the token tiles; a block whose tile is past its class's rows returns at once. The classes are
 * independent and write disjoint ids, and a launch's class is uniform per block, so the decoder's
 * form is a uniform switch rather than four host launches. The f32 fold is one inline for both
 * tilings (the M = 1 split and MT > 1), so the two cannot drift in arithmetic.
 */
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_common.h"
#include "r3d_fwht.h"
#include "r3d_iqr.h"
#include "r3d_iqr_dec.h"
#include "r3d_launch.h"

#include <cmath>
#include <string>

namespace {

using namespace r3d_iqr;
using namespace r3d_iqr_dev;

constexpr uint32_t kThreads = 32;
/* |q| <= 127 * 128: hi = floor((q + 64) / 128) stays in int8, lo = q - 128 hi in [-64, 63]. */
constexpr float kQMax = 127.0f * 128.0f;

constexpr uint32_t kMaxClass = 4;

/* One width class, as the one launch sees it: its codes and scales, its ids (the class's rows,
 * global ids), its row count and its trellis run. blockIdx.z picks one. */
struct Cls {
    const uint8_t* b;
    const uint8_t* bs;
    const int32_t* ids;
    int N;
    uint32_t run_tiles;
    uint32_t bits;
    uint32_t scale;
};
struct Head {
    Cls c[kMaxClass];
    uint32_t n;
};

__device__ __forceinline__ float ld_bf(const uint16_t* p, uint32_t i) {
    return bf16_to_f32((uint16_t)ldg_u16(p + i));
}
__device__ __forceinline__ float wave_max_f(float v) {
    v = fmaxf(v, r3d_xor_lane<16>(v));
    v = fmaxf(v, r3d_xor_lane<8>(v));
    v = fmaxf(v, r3d_xor_lane<4>(v));
    v = fmaxf(v, r3d_xor_lane<2>(v));
    v = fmaxf(v, r3d_xor_lane<1>(v));
    return v;
}

/* The f32 fold, one inline for both tilings: a block's exact integer at the activation's group
 * scale and the block's factor, then the row's own factor once in the epilogue. The order is the
 * head's: (float)total * sa * sb, added block by block, and acc * row_scale at the end. */
template <class F>
struct Fold {
    __device__ __forceinline__ static float block(const Ctx& c, const float* __restrict__ xs,
                                                  int tok, uint32_t row, uint32_t kb, int total) {
        const float sa = xs[(size_t)tok * c.nkb + kb];
        const float sb = fold_f32<F>(c, row, kb);
        return (float)total * sa * sb;
    }
    __device__ __forceinline__ static float row(const Ctx& c, uint32_t r, float acc) {
        return acc * row_scale<F>(c, r);
    }
};

/* The head's activation: one wave a (token, 128-group), the FWHT of the rotated weights' input,
 * then q = 128 hi + lo at sixteen bits under the group's scale. An all-zero group is scale 1. */
__global__ __launch_bounds__(kThreads) void logits_quant_kernel(
    const uint16_t* __restrict__ x, int8_t* __restrict__ qh, int8_t* __restrict__ ql,
    float* __restrict__ s, uint32_t n, uint32_t x_ld) {
    constexpr int HAD = 128, VPL = HAD / 32;
    static_assert(VPL == 4, "one wave's 32 lanes hold the group's 128 columns as four each");
    const uint32_t row = blockIdx.x, kg = blockIdx.y;
    const uint32_t nkb = n / 128u;
    const uint32_t lane = threadIdx.x & 31;
    const uint16_t* a = x + (size_t)row * x_ld + (size_t)kg * HAD;
    int8_t* qhr = qh + (size_t)row * n + (size_t)kg * HAD;
    int8_t* qlr = ql + (size_t)row * n + (size_t)kg * HAD;
    const int base = lane * VPL;
    float v[VPL];
#pragma unroll
    for (int j = 0; j < VPL; ++j) v[j] = ld_bf(a, base + j);
    r3d_fwht_wave<HAD>(v, lane);
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < VPL; ++j) m = fmaxf(m, fabsf(v[j]));
    m = wave_max_f(m);
    const float sc = m > 0.0f ? m / kQMax : 1.0f;
    const float inv = 1.0f / sc;
    uint32_t hp = 0, lp = 0;
#pragma unroll
    for (int j = 0; j < VPL; ++j) {
        /* NaN -> 0, as the int8 form's f32_to_i8_rtne does; fmaxf/fminf would otherwise turn it
         * into the negative rail. */
        const float t = v[j] * inv;
        const int q = (int)rintf(fminf(fmaxf(t == t ? t : 0.0f, -kQMax), kQMax));
        const int hi = (q + 64) >> 7, lo = q - hi * 128;
        hp |= ((uint32_t)hi & 0xFFu) << (8u * (uint32_t)j);
        lp |= ((uint32_t)lo & 0xFFu) << (8u * (uint32_t)j);
    }
    if (lane == 0) s[(size_t)row * nkb + kg] = sc;
    *(uint32_t*)(qhr + (uint32_t)base) = hp;
    *(uint32_t*)(qlr + (uint32_t)base) = lp;
}

/* i32 sub of four signed int8 pairs, exact (r3d_gemm_iqr_m16.hip): gfx11's dot4 is signed x
 * unsigned, so both operands are negated, which is the signed product. */
__device__ __forceinline__ int dot4_i8(uint32_t a4, uint32_t w4, int acc) {
    return __builtin_amdgcn_sudot4(true, (int)w4, true, (int)a4, acc, false);
}

/* One wave a 16-row tile of one class, MT tokens a block. Each /128-K block decodes the lane's
 * four columns of all 16 rows once, stages the 16 x 128 int8 tile in LDS, then every (token, row)
 * pair is a full 128-K dot of 32 v_dot4 per plane: hi scaled by 128 in the integer, lo added, the
 * block folded once into f32 and the row's factor taken after the K loop. */
template <class F, int MT>
__device__ __forceinline__ void logits16_tile(
    const Cls& cl, uint32_t* __restrict__ sW, const int8_t* __restrict__ xh,
    const int8_t* __restrict__ xl, const float* __restrict__ xs, float* __restrict__ logits,
    int M, int K, int ld, int voff, int n_vocab, uint32_t rt, int tok0) {
    const uint32_t lane = threadIdx.x & (kThreads - 1);
    const uint32_t nkb = (uint32_t)(K / 128);
    Ctx c;
    c.W = cl.b;
    c.S = cl.bs;
    c.N = (uint32_t)cl.N;
    c.nkb = nkb;
    c.run_tiles = cl.run_tiles;
    c.tab = nullptr;   /* the head's classes are the computed trellis */

    const int row0 = (int)rt * 16;
    constexpr int NP = (MT * 16 + (int)kThreads - 1) / (int)kThreads;
    float acc[NP];
#pragma unroll
    for (int i = 0; i < NP; ++i) acc[i] = 0.0f;

    for (uint32_t kb = 0; kb < nkb; ++kb) {
        const typename Dec<F>::Raw r = Dec<F>::fetch(c, rt, kb, lane, true);
        uint32_t w[16];
        Dec<F>::rows(r, c, 1.0f, w);
        Dec<F>::unzero(c, rt, kb, w);
#pragma unroll
        for (int q = 0; q < 16; ++q) sW[q * 33 + (int)lane] = w[q];
        __syncthreads();

        if constexpr (MT == 1) {
            /* The decode height: 16 rows and two planes over 32 lanes. Lanes 0-15 take the hi
             * plane, 16-31 the lo, each its row's full 128-K sum, so both halves of the wave do
             * 32 v_dot4 instead of 16 lanes doing 64. The low lane takes its partner's lo sum with
             * v_permlanex16 (no LDS round trip) and forms the same exact integer the unsplit path
             * does; the f32 fold, the block factor and the row's factor are untouched. */
            const bool hi = lane < 16u;
            const int q = (int)(lane & 15u);
            const int tok = tok0;
            if (tok < M) {
                const int8_t* ap = (hi ? xh : xl) + (long long)tok * K + (long long)kb * 128;
                const uint32_t* wp = sW + q * 33;
                int s = 0;
#pragma unroll
                for (int c4 = 0; c4 < 32; ++c4)
                    s = dot4_i8(*(const uint32_t*)(ap + 4 * c4), wp[c4], s);
                const int partner = (int)r3d_xor_lane_u32<16>((uint32_t)s);
                if (hi) {
                    const int total = s * 128 + partner;
                    acc[0] += Fold<F>::block(c, xs, tok, (uint32_t)(row0 + q), kb, total);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < NP; ++i) {
                const int p = (int)lane + (int)kThreads * i;
                const int m = p / 16, q = p % 16;
                const int tok = tok0 + m;
                if (tok >= M) continue;
                const int8_t* h = xh + (long long)tok * K + (long long)kb * 128;
                const int8_t* l = xl + (long long)tok * K + (long long)kb * 128;
                const uint32_t* wp = sW + q * 33;
                int sh = 0, sl = 0;
#pragma unroll
                for (int c4 = 0; c4 < 32; ++c4) {
                    const uint32_t ww = wp[c4];
                    sh = dot4_i8(*(const uint32_t*)(h + 4 * c4), ww, sh);
                    sl = dot4_i8(*(const uint32_t*)(l + 4 * c4), ww, sl);
                }
                /* q = 128 hi + lo: the exact integer this block adds before any f32 rounding */
                const int total = sh * 128 + sl;
                acc[i] += Fold<F>::block(c, xs, tok, (uint32_t)(row0 + q), kb, total);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < NP; ++i) {
        const int p = (int)lane + (int)kThreads * i;
        if constexpr (MT * 16 < (int)kThreads) {
            if (p >= MT * 16) continue;
        }
        const int m = p / 16, q = p % 16;
        const int tok = tok0 + m;
        if (tok >= M) continue;
        const int32_t id = cl.ids[row0 + q];
        const long long rel = (long long)id - voff;
        if (rel < 0 || rel >= n_vocab) continue;
        logits[(size_t)tok * ld + rel] = Fold<F>::row(c, (uint32_t)(row0 + q), acc[i]);
    }
}

/* A launch's class is uniform per block, so its width and scale select the tile's form in one
 * uniform switch: no lane diverges and no class needs a launch of its own. */
template <int MT>
__device__ __forceinline__ void dispatch_class(
    const Cls& cl, uint32_t* __restrict__ sW, const int8_t* __restrict__ xh,
    const int8_t* __restrict__ xl, const float* __restrict__ xs, float* __restrict__ logits,
    int M, int K, int ld, int voff, int n_vocab, uint32_t rt, int tok0) {
#define R3D_LOGITS_TILE(KIND, B, L, SC)                                                           \
    logits16_tile<FormT<KIND, B, 0, L, 0, SC>, MT>(cl, sW, xh, xl, xs, logits, M, K, ld,           \
                                                   voff, n_vocab, rt, tok0)
    if (cl.bits == 8) R3D_LOGITS_TILE(kSigned, 8, 0, kBf16);
    switch (cl.bits) {
        case 2: if (cl.scale == kRowF32) R3D_LOGITS_TILE(kTrellisMad, 2, 16, kRowF32); else R3D_LOGITS_TILE(kTrellisMad, 2, 16, kBf16); break;
        case 3: if (cl.scale == kRowF32) R3D_LOGITS_TILE(kTrellisMad, 3, 16, kRowF32); else R3D_LOGITS_TILE(kTrellisMad, 3, 16, kBf16); break;
        case 4: if (cl.scale == kRowF32) R3D_LOGITS_TILE(kTrellisMad, 4, 16, kRowF32); else R3D_LOGITS_TILE(kTrellisMad, 4, 16, kBf16); break;
        case 5: if (cl.scale == kRowF32) R3D_LOGITS_TILE(kTrellisMad, 5, 16, kRowF32); else R3D_LOGITS_TILE(kTrellisMad, 5, 16, kBf16); break;
        case 6: if (cl.scale == kRowF32) R3D_LOGITS_TILE(kTrellisMad, 6, 16, kRowF32); else R3D_LOGITS_TILE(kTrellisMad, 6, 16, kBf16); break;
        case 7: if (cl.scale == kRowF32) R3D_LOGITS_TILE(kTrellisMad, 7, 16, kRowF32); else R3D_LOGITS_TILE(kTrellisMad, 7, 16, kBf16); break;
        default: break;
    }
#undef R3D_LOGITS_TILE
}

/* The head, one launch: z picks the class, x its 16-row tiles, y the token tiles. A block whose
 * tile is past its own class's rows returns at once; the classes are independent and write
 * disjoint ids. */
template <int MT>
__global__ __launch_bounds__(kThreads) R3D_WAVES(4) void logits16_kernel(
    Head h, const int8_t* __restrict__ xh, const int8_t* __restrict__ xl,
    const float* __restrict__ xs, float* __restrict__ logits, int M, int K, int ld, int voff,
    int n_vocab) {
    const uint32_t rt = blockIdx.x;
    /* the argument block is indexed at constant indices, never at run time */
    Cls cl = h.c[0];
#pragma unroll
    for (uint32_t i = 1; i < kMaxClass; ++i)
        if ((uint32_t)blockIdx.z == i) cl = h.c[i];
    if (rt * 16u >= (uint32_t)cl.N) return;
    const int tok0 = (int)blockIdx.y * MT;
    if (tok0 >= M) return;

    /* 16 x 128 int8 as 32 dwords a row, plus one pad dword: 33 mod 32 is 1, so the 16 rows'
     * same-column reads land in 16 different LDS banks instead of all in the one a 32-dword
     * stride would give. The pad is never read; the row's dwords keep lane order. */
    __shared__ uint32_t sW[16 * 33];
    dispatch_class<MT>(cl, sW, xh, xl, xs, logits, M, K, ld, voff, n_vocab, rt, tok0);
}

int mt_for(int M) {
    int mt = 1;
    while (mt < M && mt < 16) mt *= 2;
    return mt;
}

int form_supported(const Form& f) {
    /* The forms the head's layout hook stores: the computed trellis, L = 16, one f32 a row under
     * a factor a 128-K block, or a bf16 scale a (row, 128). W15 adds the rotated int8 grid
     * (kSigned, 8 bits) under a bf16 scale a (row, 128), decoded by the same lane as the trellis. */
    if (f.kind == kSigned)
        return f.bits == 8 && f.scale == kBf16 && f.lv == 0 && f.L == 0 && f.lrb == 0 &&
               built(kernel_word(word(f)));
    return f.kind == kTrellisMad && f.L == 16 && f.lrb == 0 &&
           (f.scale == kRowF32 || f.scale == kBf16) && f.bits >= 2 && f.bits <= 7;
}

/* The launcher, on addresses, so the templates stay in this unit. One launch covers every class:
 * z selects it, x its row tiles (the widest class's count; the others return past their rows) and
 * y the token tiles. */
template <int MT>
static int launch_head_mt(const int8_t* xh, const int8_t* xl, const float* xs, const Head& h,
                          float* y, int M, int K, int ld, int voff, int n_vocab, int max_tiles,
                          long stream) {
    const dim3 grid((unsigned)max_tiles, (unsigned)((M + MT - 1) / MT), h.n);
    R3D_KLAUNCH(R3D_KNAME((logits16_kernel<MT>)), grid, dim3(kThreads), (hipStream_t)stream,
                h, xh, xl, xs, y, M, K, ld, voff, n_vocab);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

static int launch_head(const int8_t* xh, const int8_t* xl, const float* xs, const Head& h,
                       float* y, int M, int K, int ld, int voff, int n_vocab, int max_tiles,
                       long stream) {
    switch (mt_for(M)) {
        case 1:  return launch_head_mt<1>(xh, xl, xs, h, y, M, K, ld, voff, n_vocab, max_tiles, stream);
        case 2:  return launch_head_mt<2>(xh, xl, xs, h, y, M, K, ld, voff, n_vocab, max_tiles, stream);
        case 4:  return launch_head_mt<4>(xh, xl, xs, h, y, M, K, ld, voff, n_vocab, max_tiles, stream);
        case 8:  return launch_head_mt<8>(xh, xl, xs, h, y, M, K, ld, voff, n_vocab, max_tiles, stream);
        default: return launch_head_mt<16>(xh, xl, xs, h, y, M, K, ld, voff, n_vocab, max_tiles, stream);
    }
}

int64_t align256(int64_t v) { return (v + 255) & ~255ll; }

}  // namespace

extern "C" {

/* The two int8 activation planes, then the group scales. */
int64_t r3d_logits_gemm_vocab_iqr_scratch(const RadArgs* a) {
    if (!a) return 0;
    const RadTensor* x = r3d_opt(a, 0);
    const long long n_embd = r3d_geti_or(a, "n_embd", 0);
    long long M = r3d_geti_or(a, "M", 0);
    if (x && x->rank >= 2) M = x->shape[0] > M ? x->shape[0] : M;
    long long rows = 0;
    for (const char* k : { "rows0", "rows1", "rows2", "rows3" }) {
        const long long r = r3d_geti_or(a, k, 0);
        rows = r > rows ? r : rows;
    }
    if (M <= 0 || n_embd <= 0 || rows <= 0) return 0;
    if (n_embd % 128) return 0;
    return align256(M * n_embd) + align256(M * n_embd) +
           align256(M * (n_embd / 128) * 4) + 256;
}

int r3d_rad_logits_gemm_vocab_iqr(const RadArgs* a, RadStream stream) {
    if (!a || a->n_t < 15 || !a->scratch) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0);
    const RadTensor* lg = r3d_opt(a, 14);
    if (!r3d_p(x) || !r3d_wp(lg)) return R3D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || lg->dtype != RAD_F32) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(x) || !r3d_contig(lg)) return R3D_NO(RAD_E_STRIDE);

    const long long n_embd = r3d_geti_or(a, "n_embd", 0);
    const long long n_vocab = r3d_geti_or(a, "n_vocab", 0);
    const long long voff = r3d_geti_or(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_embd % 128 || n_vocab <= 0 || voff < 0) return R3D_NO(RAD_E_SHAPE);

    long long M = 0, x_ld = 0;
    /* `M` is the band's capacity, not the step's row count: the engine issues a one-row decode
     * step inside a band declared at max_logit_rows (core/build rad_builder bands, docs/OPS.md)
     * and passes the band's upper bound, with the step's extent on the tensors themselves. The
     * reference row does the same (libr4d/r4d_logits_vocab.hip: g->M = x->shape[0]). So the extent
     * is read from `x` and only a capacity below it is a caller bug. */
    const long long band_M = r3d_geti_or(a, "M", 0);
    x_ld = r3d_rows_pitch(x, n_embd, &M);
    if (x_ld < n_embd || M <= 0 || band_M < M) return R3D_NO(RAD_E_SHAPE);
    long long lrows = 0;
    const long long ld = r3d_rows_pitch(lg, n_vocab, &lrows);
    if (ld < n_vocab || lrows < M) return R3D_NO(RAD_E_SHAPE);
    if (((uintptr_t)r3d_wp(lg) & 3u)) return R3D_NO(RAD_E_ALIGN);

    long long rows[4] = { 0, 0, 0, 0 };
    long long maxrows = 0;
    for (int i = 0; i < 4; ++i) {
        rows[i] = r3d_geti_or(a, ("rows" + std::to_string(i)).c_str(), 0);
        maxrows = rows[i] > maxrows ? rows[i] : maxrows;
    }
    if (maxrows <= 0) return R3D_NO(RAD_E_SHAPE);

    /* The classes' weights: each present one is the IQR pair and its ids. */
    const RadTensor* w[4] = { r3d_opt(a, 1), r3d_opt(a, 4), r3d_opt(a, 7), r3d_opt(a, 10) };
    const RadTensor* sc[4] = { r3d_opt(a, 2), r3d_opt(a, 5), r3d_opt(a, 8), r3d_opt(a, 11) };
    const RadTensor* id[4] = { r3d_opt(a, 3), r3d_opt(a, 6), r3d_opt(a, 9), r3d_opt(a, 12) };
    Head head{};
    int max_tiles = 0;
    for (int i = 0; i < 4; ++i) {
        if (rows[i] <= 0) { w[i] = sc[i] = id[i] = nullptr; continue; }
        if (!r3d_p(w[i]) || !r3d_p(sc[i]) || !r3d_p(id[i])) return R3D_NO(RAD_E_INVAL);
        if (!is_weight_dtype(w[i]->dtype) || !is_scale_dtype(sc[i]->dtype)) return R3D_NO(RAD_E_DTYPE);
        if (id[i]->dtype != RAD_I32 || r3d_numel(id[i]) < rows[i]) return R3D_NO(RAD_E_SHAPE);
        if (kernel_word(w[i]->dtype) != kernel_word(sc[i]->dtype)) return R3D_NO(RAD_E_DTYPE);
        const Form f = form(w[i]->dtype);
        if (!form_supported(f)) return R3D_NO(RAD_E_UNSUPPORTED);
        if (r3d_numel(w[i]) != rows[i] * n_embd || r3d_numel(sc[i]) < 1)
            return R3D_NO(RAD_E_SHAPE);
        if (((uintptr_t)r3d_p(w[i]) & 15u) || ((uintptr_t)r3d_p(sc[i]) & 15u))
            return R3D_NO(RAD_E_ALIGN);
        const unsigned lrun = (unsigned)((w[i]->dtype >> 22) & 15u);
        if (is_trellis(f.kind) && (lrun < 4 || rows[i] % (long long)(1u << lrun)))
            return R3D_NO(RAD_E_SHAPE);
        Cls& c = head.c[head.n++];
        c.b = (const uint8_t*)r3d_p(w[i]);
        c.bs = (const uint8_t*)r3d_p(sc[i]);
        c.ids = (const int32_t*)r3d_p(id[i]);
        c.N = (int)rows[i];
        c.run_tiles = is_trellis(f.kind) ? (1u << lrun) / 16u : 0u;
        c.bits = f.bits;
        c.scale = f.scale;
        const int tiles = (int)(rows[i] / 16);
        max_tiles = tiles > max_tiles ? tiles : max_tiles;
    }

    const int64_t need = r3d_logits_gemm_vocab_iqr_scratch(a);
    if (need <= 0 || a->scratch_bytes < need) return R3D_NO(RAD_E_SCRATCH);
    int8_t* qh = (int8_t*)a->scratch;
    int8_t* ql = qh + align256(M * n_embd);
    float* s = (float*)((uint8_t*)ql + align256(M * n_embd));

    const dim3 qgrid((uint32_t)M, (uint32_t)(n_embd / 128));
    R3D_KLAUNCH(R3D_KNAME(logits_quant_kernel), qgrid, dim3(kThreads), (hipStream_t)stream,
                (const uint16_t*)r3d_p(x), qh, ql, s, (uint32_t)n_embd, (uint32_t)x_ld);

    if (launch_head(qh, ql, s, head, (float*)r3d_wp(lg), (int)M, (int)n_embd, (int)ld, (int)voff,
                    (int)n_vocab, max_tiles, (long)stream))
        return R3D_NO(RAD_E_DEVICE);
    return hipGetLastError() == hipSuccess ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

}  // extern "C"
