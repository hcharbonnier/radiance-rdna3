/* r3d_gemm_iqr_gated.hip -- gemm_nt_q_gated over a rotated int8 (i8a8h) IQR weight, the 27B's
 * feed-forward fold in one device kernel: the IQR decode of the gate's and the up's 16-row tiles in
 * one workgroup, the gate, the bf16 rounding the unfused pair stores, the 128-wide Hadamard and the
 * per-group int8 quantiser.
 *
 * The workgroup is one 128-column group of the gated output, sixteen waves: waves 0..7 decode the
 * group's eight gate tiles of the [2n, K] weight, waves 8..15 its eight up tiles (unit rt = kg*8 + t,
 * and n/16 + kg*8 + t), and each wave runs the plain m16 kernel's K loop over its 16 rows -- the
 * same i32 sub-sums, groups ascending, `(sub * sa) * sb` left to right, the row scale in the
 * epilogue -- so the f32 sums and their bf16 store are gemm_iqr_m16's bytes. The gate and up values
 * meet in LDS, where the gated product is rounded to bf16 and a wave a token runs
 * gated_had_quant_i8g's pass: the silu/sigmoid gate, the 128-wide FWHT, amax/127 per (token, 128),
 * and row-major int8 codes.
 *
 * No [M, 2n] bf16 intermediate and no scratch: the op's scratch hook answers 0, so the dispatch hands
 * the launch none. `out_bf16`, when bound, receives the gated product the codes were taken over, as
 * the pair writes it.
 *
 * The rotator's width and the quantiser's group are both 128, as the ffn fold declares them.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_fwht.h"
#include "r3d_iqr.h"
#include "r3d_iqr_dec.h"
#include "r3d_launch.h"

#include <cmath>
#include <cstring>

namespace {

using namespace r3d_iqr;
using namespace r3d_iqr_dev;

constexpr int kWave = 32;
constexpr int kUnits = 16;        /* eight gate tiles and eight up tiles: one 128-column group */
constexpr int kHalfUnits = 8;
constexpr int kMTMax = 16;
constexpr int kGroup = 128;
constexpr float kTopI8 = 127.0f;
/* One unit's 16x128 int8 tile (a stride of 33 words avoids a 16-way bank conflict when the 16 row
 * owners read one lane column), which also holds the GEMV's i32 partials. */
constexpr int kPartBytes = 16 * 33 * 4;

/* i32 sub of four signed int8 pairs, exact: gfx11's dot4 is signed x unsigned, so both operands
 * are negated (libr4d's form), which is the signed product (r4d_gemm_iqr.hip:284). */
__device__ __forceinline__ int dot4_i8(uint32_t a4, uint32_t w4, int acc) {
    return __builtin_amdgcn_sudot4(true, (int)w4, true, (int)a4, acc, false);
}

__device__ __forceinline__ float clamp_top(float c, float top) {
    if (c > top) c = top;
    if (c < -top) c = -top;
    return c;
}
__device__ __forceinline__ uint32_t i8_codes4(const float* v, float inv) {
    const uint32_t b0 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[0] * inv, kTopI8));
    const uint32_t b1 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[1] * inv, kTopI8));
    const uint32_t b2 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[2] * inv, kTopI8));
    const uint32_t b3 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[3] * inv, kTopI8));
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}
__device__ __forceinline__ float wave_max_f(float v) {
    v = fmaxf(v, r3d_xor_lane<16>(v));
    v = fmaxf(v, r3d_xor_lane<8>(v));
    v = fmaxf(v, r3d_xor_lane<4>(v));
    v = fmaxf(v, r3d_xor_lane<2>(v));
    v = fmaxf(v, r3d_xor_lane<1>(v));
    return v;
}

/* MT is the token tile mt_for(M) selects, as r3d_gemm_iqr_m16 uses it: M = 1 is the GEMV's
 * lane-partial fold, above it each lane owns whole (token, row) pairs of the unit. `act` is 0 for
 * silu and 1 for sigmoid, the two the pair's gated quantiser serves. */
template <class F, int MT>
__global__ __launch_bounds__(kUnits * kWave) R3D_WAVES(4)
void gemm_iqr_gated_kernel(const int8_t* __restrict__ a, const float* __restrict__ as,
                           const uint8_t* __restrict__ b, const uint8_t* __restrict__ bs,
                           uint8_t* __restrict__ q, float* __restrict__ s,
                           uint16_t* __restrict__ yo, int M, int N, int K, int a_ld,
                           uint32_t y_ld, int act, uint32_t run_tiles) {
    const uint32_t lane = threadIdx.x & (kWave - 1);
    const uint32_t u = threadIdx.x / kWave;
    const uint32_t kg = blockIdx.x;
    const uint32_t n = (uint32_t)N / 2u;
    const uint32_t nkb = (uint32_t)K / 128u;
    const uint32_t ng = n / 128u;
    /* Waves 0..7 are the group's gate tiles, 8..15 its up tiles. */
    const uint32_t half = u / kHalfUnits, tile = u % kHalfUnits;
    const uint32_t rt = kg * kHalfUnits + tile + half * (n / 16u);
    const uint32_t row0 = rt * 16u;

    Ctx c;
    c.W = b;
    c.S = bs;
    c.N = (uint32_t)N;
    c.nkb = nkb;
    c.run_tiles = run_tiles;
    c.tab = nullptr;   /* the W0 forms compute their values; the table forms stage LDS with W5 */

    /* One unit's staging, and the token tile's bf16 gate and up values (gate columns 0..127, up
     * 128..255). MT == 1 uses the staging as the GEMV's partials. */
    __shared__ __align__(16) uint8_t s_stage[(size_t)kUnits * kPartBytes];
    __shared__ uint16_t s_val[MT][2 * kGroup];

    if constexpr (MT == 1) {
        int* const sP = (int*)(s_stage + (size_t)u * kPartBytes);
        float acc = 0.0f;
        for (uint32_t kb = 0; kb < nkb; ++kb) {
            const typename Dec<F>::Raw r = Dec<F>::fetch(c, rt, kb, lane, true);
            uint32_t w[16];
            Dec<F>::rows(r, c, 1.0f, w);
            Dec<F>::unzero(c, rt, kb, w);
            const uint32_t a4 = *(const uint32_t*)(a + kb * 128u + 4u * lane);
#pragma unroll
            for (int qi = 0; qi < 16; ++qi) sP[qi * 33 + lane] = dot4_i8(a4, w[qi], 0);
            __syncwarp();
            if (lane < 16) {
                int sub = 0;
#pragma unroll
                for (int l = 0; l < 32; ++l) sub += sP[lane * 33 + l];
                const float sa = as[kb];   /* M = 1: every weight row shares the token's scale */
                const float sb = fold_f32<F>(c, row0 + lane, kb);
                acc += (float)sub * sa * sb;
            }
            __syncwarp();
        }
        if (lane < 16) {
            const float v = acc * row_scale<F>(c, row0 + lane);
            s_val[0][half * kGroup + tile * 16u + lane] = f32_to_bf16(v);
        }
    } else {
        uint32_t* const sW = (uint32_t*)(s_stage + (size_t)u * kPartBytes);
        constexpr int NP = (MT * 16 + kWave - 1) / kWave;   /* (token, row) pairs a lane */
        float acc[NP];
#pragma unroll
        for (int i = 0; i < NP; ++i) acc[i] = 0.0f;

        for (uint32_t kb = 0; kb < nkb; ++kb) {
            const typename Dec<F>::Raw r = Dec<F>::fetch(c, rt, kb, lane, true);
            uint32_t w[16];
            Dec<F>::rows(r, c, 1.0f, w);
            Dec<F>::unzero(c, rt, kb, w);
#pragma unroll
            for (int qi = 0; qi < 16; ++qi) sW[qi * 33 + lane] = w[qi];
            __syncwarp();

#pragma unroll
            for (int i = 0; i < NP; ++i) {
                const int p = (int)lane + kWave * i;
                const int m = p / 16, qi = p % 16;
                if (m >= M) continue;
                int sub = 0;
                const int8_t* ap = a + (long long)m * a_ld + (long long)kb * 128;
                const uint32_t* wp = sW + qi * 33;
#pragma unroll
                for (int c4 = 0; c4 < 32; ++c4)
                    sub = dot4_i8(*(const uint32_t*)(ap + 4 * c4), wp[c4], sub);
                const float sa = as[m * (int)nkb + (int)kb];
                const float sb = fold_f32<F>(c, row0 + qi, kb);
                acc[i] += (float)sub * sa * sb;
            }
            __syncwarp();
        }

#pragma unroll
        for (int i = 0; i < NP; ++i) {
            const int p = (int)lane + kWave * i;
            const int m = p / 16, qi = p % 16;
            if (m >= M) continue;
            s_val[m][half * kGroup + tile * 16u + qi] =
                f32_to_bf16(acc[i] * row_scale<F>(c, row0 + qi));
        }
    }
    __syncthreads();

    /* gated_had_quant_i8g's pass, a wave a token: the pair reads the same bf16 gate and up values
     * back from the [M, 2n] tensor this kernel never materialises. */
    if (u < (uint32_t)M) {
        const uint32_t tok = u;
        const uint32_t base = lane * 4u;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float xg = bf16_to_f32(s_val[tok][base + (uint32_t)j]);
            const float sg = 1.0f / (1.0f + expf(-xg));
            const uint16_t h = f32_to_bf16(
                (act ? sg : xg * sg) * bf16_to_f32(s_val[tok][kGroup + base + (uint32_t)j]));
            if (yo) yo[(size_t)tok * y_ld + (size_t)kg * kGroup + base + (uint32_t)j] = h;
            v[j] = bf16_to_f32(h);
        }
        r3d_fwht_wave<kGroup>(v, (int)lane);
        float m = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) m = fmaxf(m, fabsf(v[j]));
        m = wave_max_f(m);
        const float sc = m > 0.0f ? m / kTopI8 : 1.0f;
        const float inv = 1.0f / sc;
        if (lane == 0) s[(size_t)tok * ng + kg] = sc;
        *(uint32_t*)(q + (size_t)tok * n + (size_t)kg * kGroup + base) = i8_codes4(v, inv);
    }
}

int mt_for(int M) {
    int mt = 1;
    while (mt < M && mt < kMTMax) mt *= 2;
    return mt;
}

/* ------------------------------------------------------------------ the shim */

bool form_supported(const Form& f) {
    /* W0: the computed trellis, L = 16, one scale a row or a bf16 scale a (row, 128). */
    return f.kind == kTrellisMad && f.L == 16 && f.lrb == 0 &&
           (f.scale == kRowF32 || f.scale == kBf16) && f.bits >= 2 && f.bits <= 7;
}

/* The launcher, on addresses, so the templates stay in this unit. */
template <class F, int MT>
static int launch_fused(const int8_t* a, const float* as, const uint8_t* b, const uint8_t* bs,
                        uint8_t* q, float* s, uint16_t* yo, int M, int N, int K, int a_ld,
                        uint32_t y_ld, int act, uint32_t run_tiles, long stream) {
    const dim3 grid((unsigned)((N / 2) / kGroup)), block(kUnits * kWave);
    R3D_KLAUNCH(R3D_KNAME((gemm_iqr_gated_kernel<F, MT>)), grid, block, (hipStream_t)stream, a, as,
                b, bs, q, s, yo, M, N, K, a_ld, y_ld, act, run_tiles);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

}  // namespace

using namespace r3d_iqr;

#define R3D_IQR_GATED_MT_CASE(B, SC)                                                              \
    do {                                                                                          \
        using F = FormT<kTrellisMad, B, 0, 16, 0, SC>;                                             \
        switch (mt) {                                                                             \
            case 1:  return launch_fused<F, 1>(A_, AS_, B_, S_, Q_, SC_, OB_, M, N, K, ap, y_ld,  \
                                               act, run_tiles, stream);                            \
            case 2:  return launch_fused<F, 2>(A_, AS_, B_, S_, Q_, SC_, OB_, M, N, K, ap, y_ld,  \
                                               act, run_tiles, stream);                            \
            case 4:  return launch_fused<F, 4>(A_, AS_, B_, S_, Q_, SC_, OB_, M, N, K, ap, y_ld,  \
                                               act, run_tiles, stream);                            \
            case 8:  return launch_fused<F, 8>(A_, AS_, B_, S_, Q_, SC_, OB_, M, N, K, ap, y_ld,  \
                                               act, run_tiles, stream);                            \
            default: return launch_fused<F, 16>(A_, AS_, B_, S_, Q_, SC_, OB_, M, N, K, ap, y_ld, \
                                                act, run_tiles, stream);                           \
        }                                                                                         \
    } while (0)

extern "C" {

/* No intermediate exists, so the op asks the dispatch for no arena: the hook answers 0 and the
 * launch reads no scratch. */
int64_t r3d_rad_gemm_nt_q_iqr_gated_scratch(const RadArgs* a) {
    (void)a;
    return 0;
}

int r3d_rad_gemm_nt_q_iqr_gated(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 6) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(a, "dtype");
    if (!dt || std::strcmp(dt, "i8a8h") != 0) return R3D_NO(RAD_E_DTYPE);
    /* The pair's gated quantiser takes every name but "sigmoid" as silu, but the op's schema
     * requires one of the two and the row refused the rest: 0 is silu, 1 sigmoid. */
    const char* act_s = r3d_gets(a, "act");
    if (!act_s || (std::strcmp(act_s, "silu") != 0 && std::strcmp(act_s, "sigmoid") != 0))
        return R3D_NO(RAD_E_UNSUPPORTED);
    const int act = std::strcmp(act_s, "sigmoid") == 0 ? 1 : 0;

    const RadTensor* A = r3d_opt(a, 0), *AS = r3d_opt(a, 1), *B = r3d_opt(a, 2),
                    *BS = r3d_opt(a, 3), *Q = r3d_opt(a, 4), *SC = r3d_opt(a, 5),
                    *OB = r3d_opt(a, 6);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(BS) || !r3d_wp(Q) || !r3d_wp(SC))
        return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_i8(A) || !r3d_is_f32(AS) || !r3d_is_i8(Q) || !r3d_is_f32(SC))
        return R3D_NO(RAD_E_DTYPE);
    if (!is_weight_dtype(B->dtype) || !is_scale_dtype(BS->dtype)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(OB) && OB->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);

    const long long N = r3d_geti_or(a, "N", 0), K = r3d_geti_or(a, "K", 0),
                    group = r3d_geti_or(a, "group", 0);
    if (N < 2 || (N & 1) || K < 128) return R3D_NO(RAD_E_SHAPE);
    if (group != kGroup) return R3D_NO(RAD_E_UNSUPPORTED);
    if (K % (long long)kGroup) return R3D_NO(RAD_E_SHAPE);
    const long long n = N / 2;
    /* The pair's quantiser and this kernel both cut `n` into 128-column groups; the gated shim
     * refused an uneven `n`, and its gate_up extent refused a q wider than n. */
    if (n % kGroup) return R3D_NO(RAD_E_SHAPE);
    long long qm = 0, qn = 0;
    if (r3d_rows_of(Q, &qm, &qn) != 0 || qn != n) return R3D_NO(RAD_E_SHAPE);
    /* The band's `M` is the largest step (16); the tensors hold this step's rows, which is what the
     * launch's arithmetic uses. */
    const long long M = qm;
    if (M < 1 || M > kMTMax) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(Q)) return R3D_NO(RAD_E_STRIDE);

    /* The same forms, private planes, extents and alignment the m16 row serves. */
    const Form fw = form(B->dtype);
    if (kernel_word(B->dtype) != kernel_word(BS->dtype)) return R3D_NO(RAD_E_DTYPE);
    if (!form_supported(fw)) return R3D_NO(RAD_E_UNSUPPORTED);
    const unsigned lrun = (unsigned)((B->dtype >> 22) & 15u);
    if (lrun < 4 || (N % (long long)(1u << lrun))) return R3D_NO(RAD_E_SHAPE);

    if (r3d_numel(A) < M * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(AS) < M * (K / kGroup)) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(B) != N * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(BS) < 1) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(SC) < M * (n / kGroup)) return R3D_NO(RAD_E_SHAPE);
    if (r3d_has(OB) && r3d_numel(OB) < M * n) return R3D_NO(RAD_E_SHAPE);

    const long long ap = r3d_row_pitch(A);
    if (ap < K || ap % 4) return R3D_NO(RAD_E_STRIDE);
    /* `out_bf16`'s row pitch, as gated_had_quant_i8g reads it: n unless its second extent is wider. */
    uint32_t y_ld = (uint32_t)n;
    if (r3d_has(OB) && OB->rank >= 2 && OB->shape[1] > n) y_ld = (uint32_t)OB->shape[1];
    if (!r3d_aligned(r3d_p(A), 4) || !r3d_aligned(r3d_p(AS), 4) || !r3d_aligned(r3d_p(B), 16) ||
        !r3d_aligned(r3d_p(BS), 16) || !r3d_aligned(r3d_wp(Q), 4) ||
        (r3d_has(OB) && !r3d_aligned(r3d_wp(OB), 2)))
        return R3D_NO(RAD_E_ALIGN);

    const int mt = mt_for((int)M);
    const uint32_t run_tiles = (1u << lrun) / 16u;
    const long stream = (long)(uintptr_t)s;
    const uint8_t* B_ = (const uint8_t*)r3d_p(B);
    const uint8_t* S_ = (const uint8_t*)r3d_p(BS);
    const int8_t* A_ = (const int8_t*)r3d_p(A);
    const float* AS_ = (const float*)r3d_p(AS);
    uint8_t* Q_ = (uint8_t*)r3d_wp(Q);
    float* SC_ = (float*)r3d_wp(SC);
    uint16_t* OB_ = r3d_has(OB) ? (uint16_t*)r3d_wp(OB) : nullptr;

#define R3D_IQR_GATED_SC_CASE(B) do {                                                             \
        if (fw.scale == kRowF32) { R3D_IQR_GATED_MT_CASE(B, kRowF32); }                           \
        R3D_IQR_GATED_MT_CASE(B, kBf16);                                                          \
    } while (0)

    switch (fw.bits) {
        case 2: R3D_IQR_GATED_SC_CASE(2);
        case 3: R3D_IQR_GATED_SC_CASE(3);
        case 4: R3D_IQR_GATED_SC_CASE(4);
        case 5: R3D_IQR_GATED_SC_CASE(5);
        case 6: R3D_IQR_GATED_SC_CASE(6);
        case 7: R3D_IQR_GATED_SC_CASE(7);
        default: return R3D_NO(RAD_E_UNSUPPORTED);
    }
#undef R3D_IQR_GATED_SC_CASE
}

}  // extern "C"

#undef R3D_IQR_GATED_MT_CASE
