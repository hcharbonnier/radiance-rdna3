/* r3d_gemm_nt_n1.hip -- the N = 1 GEMV of gemm_nt, the shared expert's gate.
 *
 *   y[m] = a[m, K] . b[0, K]        f32 accumulate, one narrowing at the store
 *
 * The Flash-Next shared gate is a `gemm_nt` with a one-row weight, so N has no parallelism to
 * give and the skinny m16 row's wave-per-run-of-columns map has a single column to cover. This
 * row serves that band with one lane per output row: a wave owns 32 rows, and every lane walks
 * its own dot in ascending k.
 *
 * ---- the order ------------------------------------------------------------------------------
 *
 * The chain is libref's, not merely the same formula. ref_gemm.h's quant_tile fills a tile with
 * `sub += a[k] * b[k]` over ascending k (REF_KC chunks add into the same `sub`), and with no
 * scales it stores `sub` after one f32 -> bf16 rounding. This kernel is
 *
 *     acc = 0; for k ascending: acc = acc + a[k] * b[k]
 *
 * with the multiply and the add both f32 and neither contracted (`-ffp-contract=off` is set on
 * every libr3d HIP unit), so the stored bf16 is bit-identical to libref's `gemm_nt` at N = 1.
 * There is no cross-lane reduction and no split-K: both would bracket the products differently.
 *
 * bf16 widens exactly to f32, so the products are the reference's products; the one rounding left
 * is the store's, and it is the same software RTNE codec libref uses.
 *
 * ---- the lane map ---------------------------------------------------------------------------
 *
 * One lane owns one output row, loads 16 bytes (eight bf16) of a and of b at a time, and adds the
 * eight products in ascending k. A block is one wave (32 rows), which spreads the row count over
 * every CU instead of packing eight waves into one block's CU. K is a multiple of 128, the row's
 * constraint, so the 16-element step has no tail and every 16-byte load is aligned (a row pitch is
 * a multiple of eight elements and both bases are 16-byte aligned; the shim checks that).
 *
 * The weight is read once per row, but every lane of a wave reads the same address, which the
 * hardware broadcasts, and the line stays in L1 across the rows of a CU. b is never staged: the
 * dot is a dependent f32 add chain, so the useful latency hiding comes from unrolling the loads,
 * not from another LDS round trip.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_common.h"
#include "r3d_launch.h"

namespace {

constexpr int kBlockThreads = 32;            // one wave, 32 output rows
constexpr int kUnit         = 8;             // bf16 elements in one 16-byte load
constexpr int kStep         = 16;            // two loads per operand, one loop iteration

__device__ __forceinline__ r3d_u32x4 ldg4(const uint16_t* p)
{
    return *(const __attribute__((address_space(1))) r3d_u32x4*)p;
}

/* y[m] = a[m, K] . b[0, K], one lane a row, ascending k. */
__global__ __launch_bounds__(kBlockThreads) R3D_WAVES(8) void gemv_n1_kernel(
    const uint16_t* __restrict__ A, const uint16_t* __restrict__ B,
    uint16_t* __restrict__ Y, int M, int K, int a_ld, int y_ld)
{
    const int m = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (m >= M) return;

    const uint16_t* x = A + (size_t)m * a_ld;
    float acc = 0.0f;

    /* Unrolled so several global loads are in flight while the add chain advances; the adds stay
     * in ascending k, one f32 rounding each. */
    #pragma unroll 4
    for (int k = 0; k < K; k += kStep) {
        const r3d_u32x4 a0 = ldg4(x + k);
        const r3d_u32x4 b0 = ldg4(B + k);
        const r3d_u32x4 a1 = ldg4(x + k + kUnit);
        const r3d_u32x4 b1 = ldg4(B + k + kUnit);

        acc = acc + bf16w_lo(a0.x) * bf16w_lo(b0.x);
        acc = acc + bf16w_hi(a0.x) * bf16w_hi(b0.x);
        acc = acc + bf16w_lo(a0.y) * bf16w_lo(b0.y);
        acc = acc + bf16w_hi(a0.y) * bf16w_hi(b0.y);
        acc = acc + bf16w_lo(a0.z) * bf16w_lo(b0.z);
        acc = acc + bf16w_hi(a0.z) * bf16w_hi(b0.z);
        acc = acc + bf16w_lo(a0.w) * bf16w_lo(b0.w);
        acc = acc + bf16w_hi(a0.w) * bf16w_hi(b0.w);

        acc = acc + bf16w_lo(a1.x) * bf16w_lo(b1.x);
        acc = acc + bf16w_hi(a1.x) * bf16w_hi(b1.x);
        acc = acc + bf16w_lo(a1.y) * bf16w_lo(b1.y);
        acc = acc + bf16w_hi(a1.y) * bf16w_hi(b1.y);
        acc = acc + bf16w_lo(a1.z) * bf16w_lo(b1.z);
        acc = acc + bf16w_hi(a1.z) * bf16w_hi(b1.z);
        acc = acc + bf16w_lo(a1.w) * bf16w_lo(b1.w);
        acc = acc + bf16w_hi(a1.w) * bf16w_hi(b1.w);
    }

    Y[(size_t)m * y_ld] = f32_to_bf16(acc);
}

}  // namespace

/* y[M,1] = a[M,K] @ b[1,K]^T, bf16 in and out, f32 accumulate, bit-identical to libref's N = 1
 * gemm_nt chain.
 *
 *   a  [M][K] bf16, `a_ld` elements per row (0 = K)
 *   b  [1][K] bf16, dense
 *   y  [M][1] bf16, `y_ld` elements per row (0 = 1)
 *
 * Returns 0, -1 a null pointer, -2 an unsupported shape, -3 an unreadable pitch, -4 the HIP launch
 * failed. */
extern "C" int r3d_gemm_nt_bf16_n1(long a, long b, long y, int M, int N, int K, int a_ld, int y_ld,
                                   long stream)
{
    if (!a || !b || !y) return -1;
    if (M < 1 || N != 1 || K < 1) return -2;
    if (K % 128) return -2;
    if (!a_ld) a_ld = K;
    if (!y_ld) y_ld = 1;
    if (a_ld < K || y_ld < 1) return -3;
    if (a_ld % kUnit) return -3;

    const dim3 grid((unsigned)(((long long)M + kBlockThreads - 1) / kBlockThreads));
    R3D_KLAUNCH(gemv_n1_kernel, grid, dim3(kBlockThreads), (hipStream_t)stream,
                (const uint16_t*)a, (const uint16_t*)b, (uint16_t*)y, M, K, a_ld, y_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ============================================================== the plugin ABI entry point
 *
 * gemm_nt's operand list is [a, b, y] and its parameters M, N, K, dtype (libref's pGemm/oGemm).
 * The row exists for N = 1 only; any other N is refused here rather than served by the m16 or
 * tiled rows under a shared name. */
enum { G_A = 0, G_B, G_Y };

#define R3D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R3D_NO(_s); } while (0)
static int dt_bf16(const RadTensor* t) { return r3d_is_bf16(t) ? RAD_OK : RAD_E_DTYPE; }

extern "C" int r3d_rad_gemm_nt_bf16_n1(const RadArgs* a, RadStream s)
{
    const RadTensor* A = r3d_opt(a, G_A);
    const RadTensor* B = r3d_opt(a, G_B);
    const RadTensor* Y = r3d_opt(a, G_Y);
    if (!A || !B || !Y) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(A));
    R3D_NEED(dt_bf16(B));
    R3D_NEED(dt_bf16(Y));

    const long long K = r3d_geti_or(a, "K", r3d_dim(B, (int)B->rank - 1));
    const long long N = r3d_geti_or(a, "N", r3d_dim(B, (int)B->rank - 2));
    if (K < 1 || N != 1) return R3D_NO(RAD_E_SHAPE);
    if (K % 128) return R3D_NO(RAD_E_SHAPE);
    if (K > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);

    /* The row count comes from the output, never from `M`, which is the band's upper bound. At
     * N = 1 that is the output's whole numel. */
    const long long M = r3d_numel(Y);
    if (M < 1 || M > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(A) < M * K || r3d_numel(B) < N * K) return R3D_NO(RAD_E_SHAPE);

    long long arows = 0, yrows = 0;
    const long long a_ld = r3d_rows_pitch(A, K, &arows);
    const long long y_ld = r3d_rows_pitch(Y, N, &yrows);
    if (a_ld < 0 || y_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (arows < M || yrows < M) return R3D_NO(RAD_E_SHAPE);
    if (a_ld < K || y_ld < N) return R3D_NO(RAD_E_STRIDE);
    /* The weight is dense: the dot reads b[0][k] at k. */
    if (!r3d_contig(B)) return R3D_NO(RAD_E_STRIDE);
    /* 16-byte loads at a row base, and a tile base a multiple of eight elements. */
    if (a_ld % kUnit) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(A->data, 16) || !r3d_aligned(B->data, 16)) return R3D_NO(RAD_E_ALIGN);

    const int e = r3d_gemm_nt_bf16_n1(r3d_addr(A), r3d_addr(B), r3d_addr(Y), (int)M, (int)N,
                                      (int)K, (int)a_ld, (int)y_ld, (long)(uintptr_t)s);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_INVAL : e == -2 ? RAD_E_SHAPE
                    : e == -3 ? RAD_E_STRIDE : RAD_E_DEVICE);
}

#undef R3D_NEED
