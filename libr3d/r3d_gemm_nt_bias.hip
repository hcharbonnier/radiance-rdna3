/* r3d_gemm_nt_bias.hip -- gemm_nt_bias at the vision tower's heights.
 *
 *   y[M,N] = a[M,K] @ b[N,K]^T + bias[N], then optionally act, then optionally + res,
 *   each step rounded to the output's dtype before the next reads it.
 *
 * The vision tower (arch/common/rad_block_vit.h) makes every linear this op: the patch
 * embedding, qkv, the attention projection and both MLP layers, each with its bias, and the
 * projection and down layers with the residual stream in the epilogue. The MERGER's two layers
 * and the deepstack mergers use the same shapes with a fused activation.
 *
 * ---- the arithmetic -------------------------------------------------------------------------
 *
 * One thread owns a pair of output columns for one row, and its two sums walk k ascending:
 *
 *     acc = 0; for k ascending: acc = acc + a[m][k] * b[n][k]
 *
 * which is libref's `quant_tile` with no scale streams: one multiply and one add a k, no FMA
 * (`-ffp-contract=off`, as the reference is built). The bias is added to the f32 sum, then the
 * result narrowed once -- the reference stores the biased product to bf16 before its epilogue.
 * `act` (`gelu` the exact erf form, `gelu_tanh` PyTorch's approximation) is evaluated on the
 * narrowed value and narrowed again, and `res` is added to that and narrowed. Each step therefore
 * sees exactly the bytes a separate tensor would hold, so the fused row equals the unfused chain
 * bit for bit.
 *
 * ---- the tile -------------------------------------------------------------------------------
 *
 * A block carries a 32-row x 64-column tile with four waves: a thread owns four rows by four
 * columns, so one activation value feeds four outputs and one weight value four rows. Both
 * operands stage to LDS as the checkpoint's bf16 bytes -- not f32 -- in a k-major layout (one
 * staged row is the block's whole width for one k), so a thread's four rows, or its four columns,
 * are two aligned 32-bit reads that widen to f32 in registers. The staged row pitch is padded to
 * 36 and 68 halfwords (18 and 34 dwords), which spreads a k row across the banks: the compute
 * loop's reads are conflict-free and the transposed store in is four-way. Rows and columns past
 * the band clamp to the last real one, so the tile walk stays branch-free; the store drops them.
 *
 * k is walked in 64-wide tiles, ascending, so every output's chain is still libref's ascending k
 * with one separate multiply and one separate add a step. The sixteen accumulators a thread are
 * sixteen independent chains, so the add latency is hidden without reassociating anything.
 *
 * ---- pitches --------------------------------------------------------------------------------
 *
 * `a_ld`, `b_ld`, `y_ld` and `res_ld` are row pitches in elements. The activation is a slice of
 * a wider buffer in the tower (a norm's output), the weight is the checkpoint's dense plane, and
 * the output is the residual stream's own row; all three may be wider than their logical width.
 * Every read is a halfword, so no pointer alignment is required.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <cmath>
#include <cstring>

namespace {

constexpr int kThr = 128;       /* four waves: eight 4-row groups by sixteen 4-column groups */
constexpr int kMT  = 32;        /* rows one block carries */
constexpr int kNT  = 64;        /* columns one block carries */
constexpr int kKT  = 64;        /* k one staged tile covers */
constexpr int kRM  = 4;         /* rows one thread accumulates */
constexpr int kRN  = 4;         /* columns one thread accumulates */
constexpr int kSA  = kMT + 4;   /* halfwords a staged activation k row: 18 dwords */
constexpr int kSB  = kNT + 4;   /* halfwords a staged weight k row: 34 dwords */

/* The tile map, pinned: four warps stage two k a lane over a 64-wide tile, and sixteen 4-column
 * groups span the block's rows. A change that breaks these breaks the staging or the epilogue. */
static_assert(kThr / kWave == 4, "the stage walks four warps");
static_assert(kKT == 2 * kWave, "a warp's lanes stage two k each");
static_assert(kThr == 16 * (kMT / kRM), "sixteen 4-row groups");
static_assert(kNT == kRN * 16, "sixteen 4-column groups");
static_assert(kSA % 2 == 0 && kSB % 2 == 0, "a staged k row must keep 32-bit reads aligned");

/* The epilogue's activation, on the value the bias left (the reference's gemm_epilogue reads the
 * stored tensor back, so the argument is what bf16 holds). The double libm calls are narrowed
 * once, as ref_erf and ref_tanh do; the f32 expressions keep the reference's grouping. */
__device__ __forceinline__ float bias_act(int act, float x) {
    if (act == 1) {
        const float t = x * 0.70710678118654752f;
        const float e = (float)std::erf((double)t);
        return 0.5f * x * (1.0f + e);
    }
    if (act == 2) {
        const float k0 = 0.7978845608028654f, k1 = 0.044715f;
        float p = k1 * x;
        p = p * x;
        p = p * x;
        const float z = k0 * (x + p);
        const float th = (float)std::tanh((double)z);
        return 0.5f * x * (1.0f + th);
    }
    return x;
}

/* One output element: the biased product, the optional activation, the optional residual. `res`
 * is null when the band omits it. The store narrows once, as each separate tensor would. */
__device__ __forceinline__ void store_one(uint16_t* y, int y_ld, int m, int n, float acc,
                                          const uint16_t* bias, const uint16_t* res, int res_ld,
                                          int act) {
    float v = acc + bf16_to_f32((uint16_t)ldg_u16(bias + n));
    uint16_t b = f32_to_bf16(v);
    if (act) {
        const float a = bias_act(act, bf16_to_f32(b));
        b = f32_to_bf16(a);
    }
    if (res) {
        const float r = bf16_to_f32((uint16_t)ldg_u16(res + (size_t)m * res_ld + n));
        b = f32_to_bf16(r + bf16_to_f32(b));
    }
    y[(size_t)m * y_ld + n] = b;
}

/* A packed bf16 pair (little endian: the low half is the first element) widened to two f32. Both
 * are exact: bf16 is f32's top half, so the widening is a shift or a mask, not a rounding. */
__device__ __forceinline__ void bf16x2_to_f32(uint32_t u, float& lo, float& hi) {
    lo = float_of(u << 16);
    hi = float_of(u & 0xffff0000u);
}

/* y[M,N] = a[M,K] @ b[N,K]^T + bias, with the optional epilogue. Every output's chain is the
 * reference's ascending k; a 32 x 64 tile stages both operands as bf16 in a k-major LDS layout
 * and each thread accumulates four rows by four columns of it. */
__global__ __launch_bounds__(kThr) R3D_WAVES(8) void gemm_nt_bias_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
    const uint16_t* __restrict__ bias, const uint16_t* __restrict__ res,
    uint16_t* __restrict__ Y, int M, int N, int K, int a_ld, int b_ld, int res_ld, int y_ld,
    int act) {
    /* sA[k][r] and sB[k][n]: one staged row holds the block's whole width for one k, so the four
     * rows (or columns) a thread owns are four adjacent halfwords. */
    __shared__ __align__(16) uint16_t sA[kKT * kSA];
    __shared__ __align__(16) uint16_t sB[kKT * kSB];

    const int tid = (int)threadIdx.x;
    const int lane = tid & (kWave - 1);
    const int wv = tid / kWave;            /* the warp: stages rows wv, wv + 4, ... */
    const int rg = tid >> 4;               /* rows rg*4 .. rg*4 + 3 */
    const int cg = tid & 15;               /* columns cg*4 .. cg*4 + 3 */
    const int m0 = (int)blockIdx.y * kMT;
    const int n0 = (int)blockIdx.x * kNT;

    float acc[kRM][kRN];
#pragma unroll
    for (int i = 0; i < kRM; ++i)
#pragma unroll
        for (int j = 0; j < kRN; ++j) acc[i][j] = 0.0f;

    for (int kt = 0; kt < K; kt += kKT) {
        /* Stage the whole tile, transposed: a warp owns one row at a time and its 32 lanes cover
         * the tile's 64 k two each, so a row's 128 bytes are one coalesced request; the two
         * halfwords land in two k rows of LDS. k past K stages as +0 and is never consumed (the
         * compute loop stops at kl). */
        for (int r = wv; r < kMT; r += 4) {
            int rr = m0 + r;
            if (rr >= M) rr = M - 1;
            const uint16_t* src = A + (size_t)rr * a_ld + kt + 2 * lane;
            uint16_t v0 = 0, v1 = 0;
            if (kt + 2 * lane < K) v0 = (uint16_t)ldg_u16(src);
            if (kt + 2 * lane + 1 < K) v1 = (uint16_t)ldg_u16(src + 1);
            sA[(2 * lane) * kSA + r] = v0;
            sA[(2 * lane + 1) * kSA + r] = v1;
        }
        for (int r = wv; r < kNT; r += 4) {
            int nn = n0 + r;
            if (nn >= N) nn = N - 1;
            const uint16_t* src = B + (size_t)nn * b_ld + kt + 2 * lane;
            uint16_t v0 = 0, v1 = 0;
            if (kt + 2 * lane < K) v0 = (uint16_t)ldg_u16(src);
            if (kt + 2 * lane + 1 < K) v1 = (uint16_t)ldg_u16(src + 1);
            sB[(2 * lane) * kSB + r] = v0;
            sB[(2 * lane + 1) * kSB + r] = v1;
        }
        group_barrier();

        const int kl = K - kt < kKT ? K - kt : kKT;
        const uint16_t* pa = sA + rg * 4;
        const uint16_t* pb = sB + cg * 4;
#pragma unroll 4
        for (int kk = 0; kk < kl; ++kk) {
            float a0, a1, a2, a3, b0, b1, b2, b3;
            bf16x2_to_f32(*(const uint32_t*)(pa), a0, a1);
            bf16x2_to_f32(*(const uint32_t*)(pa + 2), a2, a3);
            bf16x2_to_f32(*(const uint32_t*)(pb), b0, b1);
            bf16x2_to_f32(*(const uint32_t*)(pb + 2), b2, b3);
            acc[0][0] = acc[0][0] + a0 * b0;
            acc[0][1] = acc[0][1] + a0 * b1;
            acc[0][2] = acc[0][2] + a0 * b2;
            acc[0][3] = acc[0][3] + a0 * b3;
            acc[1][0] = acc[1][0] + a1 * b0;
            acc[1][1] = acc[1][1] + a1 * b1;
            acc[1][2] = acc[1][2] + a1 * b2;
            acc[1][3] = acc[1][3] + a1 * b3;
            acc[2][0] = acc[2][0] + a2 * b0;
            acc[2][1] = acc[2][1] + a2 * b1;
            acc[2][2] = acc[2][2] + a2 * b2;
            acc[2][3] = acc[2][3] + a2 * b3;
            acc[3][0] = acc[3][0] + a3 * b0;
            acc[3][1] = acc[3][1] + a3 * b1;
            acc[3][2] = acc[3][2] + a3 * b2;
            acc[3][3] = acc[3][3] + a3 * b3;
            pa += kSA;
            pb += kSB;
        }
        group_barrier();
    }

    const int m = m0 + rg * 4;
    const int n = n0 + cg * 4;
#pragma unroll
    for (int i = 0; i < kRM; ++i)
        if (m + i < M)
#pragma unroll
            for (int j = 0; j < kRN; ++j)
                if (n + j < N) store_one(Y, y_ld, m + i, n + j, acc[i][j], bias, res, res_ld, act);
}

}  // namespace

/* The launcher. `res` is 0 when the band has no residual (and `res_ld` is then ignored), `act` is
 * 0 none, 1 exact GELU, 2 tanh GELU. Returns 0, -1 a null operand, -2 a shape the grid cannot
 * carry, -3 a pitch short of the operand, -4 the HIP launch failed. */
extern "C" int r3d_gemm_nt_bias_bf16(long a, long b, long bias, long res, long y, int M, int N,
                                     int K, int a_ld, int b_ld, int res_ld, int y_ld, int act,
                                     long stream) {
    if (!a || !b || !bias || !y) return -1;
    if (M < 1 || N < 1 || K < 1 || act < 0 || act > 2) return -2;
    if (a_ld < K || b_ld < K || y_ld < N || (res && res_ld < N)) return -3;
    const long long tiles_m = ((long long)M + kMT - 1) / kMT;
    const long long tiles_n = ((long long)N + kNT - 1) / kNT;
    /* gfx11 caps grid y and z at 65535; x is 2^31. */
    if (tiles_n > 0x7fffffffLL || tiles_m > 65535LL) return -2;
    R3D_KLAUNCH(gemm_nt_bias_kernel, dim3((uint32_t)tiles_n, (uint32_t)tiles_m),
                dim3(kThr), (hipStream_t)(uintptr_t)stream, (const uint16_t*)(uintptr_t)a,
                (const uint16_t*)(uintptr_t)b, (const uint16_t*)(uintptr_t)bias,
                (const uint16_t*)(uintptr_t)res, (uint16_t*)(uintptr_t)y, M, N, K, a_ld, b_ld,
                res_ld, y_ld, act);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ============================================================== the plugin ABI entry point
 *
 * `gemm_nt_bias`'s operands are a, b, bias, res? (libref's pGemmBias/oGemmBias), its parameters
 * M, N, K, dtype and an optional act. This row's band is bf16, the vision tower's dtype. */
enum { G_A = 0, G_B, G_BIAS, G_RES, G_Y };

#define R3D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R3D_NO(_s); } while (0)

static int dt_bf16(const RadTensor* t) { return r3d_is_bf16(t) ? RAD_OK : RAD_E_DTYPE; }

/* A rows view reads the last axis one element apart. `r3d_rows_pitch` checks it for rank >= 2;
 * the rank-1 planes (a bias, a whole activation) need the same promise. */
static int last_stride_1(const RadTensor* t) {
    return t && t->rank >= 1 && r3d_stride(t, (int)t->rank - 1) == 1 ? RAD_OK : RAD_E_STRIDE;
}

extern "C" int r3d_rad_gemm_nt_bias(const RadArgs* a, RadStream s) {
    const RadTensor* A = r3d_opt(a, G_A);
    const RadTensor* B = r3d_opt(a, G_B);
    const RadTensor* bias = r3d_opt(a, G_BIAS);
    const RadTensor* res = r3d_opt(a, G_RES);
    const RadTensor* Y = r3d_opt(a, G_Y);
    if (!A || !B || !bias || !Y) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(A));
    R3D_NEED(dt_bf16(B));
    R3D_NEED(dt_bf16(bias));
    R3D_NEED(dt_bf16(Y));
    if (res) R3D_NEED(dt_bf16(res));
    R3D_NEED(last_stride_1(A));
    R3D_NEED(last_stride_1(B));
    R3D_NEED(last_stride_1(bias));
    R3D_NEED(last_stride_1(Y));
    if (res) R3D_NEED(last_stride_1(res));

    if (const char* nm = r3d_gets(a, "dtype"))
        if (std::strcmp(nm, "bf16")) return R3D_NO(RAD_E_UNSUPPORTED);

    int act = 0;
    if (const char* nm = r3d_gets(a, "act")) {
        if (!std::strcmp(nm, "none")) act = 0;
        else if (!std::strcmp(nm, "gelu")) act = 1;
        else if (!std::strcmp(nm, "gelu_tanh")) act = 2;
        else return R3D_NO(RAD_E_UNSUPPORTED);
    }

    const long long K = r3d_geti_or(a, "K", r3d_dim(B, (int)B->rank - 1));
    const long long N = r3d_geti_or(a, "N", r3d_dim(B, (int)B->rank - 2));
    if (K < 1 || N < 1) return R3D_NO(RAD_E_SHAPE);
    /* The row count comes from the output, never from `M`, which is the band's upper bound. */
    const long long nel = r3d_numel(Y);
    if (nel < N || nel % N) return R3D_NO(RAD_E_SHAPE);
    const long long M = nel / N;
    if (M < 1) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(A) < M * K || r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(bias) < N) return R3D_NO(RAD_E_SHAPE);
    if (res && r3d_numel(res) < M * N) return R3D_NO(RAD_E_SHAPE);

    long long arows = 0, brows = 0, yrows = 0;
    const long long a_ld = r3d_rows_pitch(A, K, &arows);
    const long long b_ld = r3d_rows_pitch(B, K, &brows);
    const long long y_ld = r3d_rows_pitch(Y, N, &yrows);
    if (a_ld < 0 || b_ld < 0 || y_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (arows < M || brows < N || yrows < M) return R3D_NO(RAD_E_SHAPE);
    long long res_ld = N, resrows = 0;
    if (res) {
        res_ld = r3d_rows_pitch(res, N, &resrows);
        if (res_ld < 0) return R3D_NO(RAD_E_STRIDE);
        if (resrows < M) return R3D_NO(RAD_E_SHAPE);
    }
    /* Every read is a halfword; the pitches must still fit the launcher's ints. */
    if (a_ld > 0x7fffffffLL || b_ld > 0x7fffffffLL || y_ld > 0x7fffffffLL ||
        res_ld > 0x7fffffffLL || (long long)M * N > 0x7fffffffLL)
        return R3D_NO(RAD_E_SHAPE);

    const int e = r3d_gemm_nt_bias_bf16(r3d_addr(A), r3d_addr(B), r3d_addr(bias),
                                        res ? r3d_addr(res) : 0L, r3d_addr(Y), (int)M, (int)N,
                                        (int)K, (int)a_ld, (int)b_ld, (int)res_ld, (int)y_ld, act,
                                        (long)(uintptr_t)s);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_INVAL : e == -2 ? RAD_E_SHAPE
                        : e == -3 ? RAD_E_STRIDE : RAD_E_DEVICE);
}
