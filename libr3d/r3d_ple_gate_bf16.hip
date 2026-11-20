/* r3d_ple_gate_bf16.hip -- Per-Layer Embedding's signed-square-root dot gate.
 *
 *   K[c] = grouped_rmsnorm(k)[c] * (w_key   + wadd)
 *   Q[c] = grouped_rmsnorm(q)[c] * (w_query + wadd)
 *   s[c] = sum_d K[c][d] * Q[c][d] / sqrt(n)
 *   s[c] = sign(s[c]) * sqrt(max(|s[c]|, gate_eps))
 *   gv[c] = sigmoid(s[c]) * v
 *   gvn   = grouped_rmsnorm(gv) * (w_conv + wadd)
 *
 * libref/ref_ple.cpp is the oracle, docs/OPS.md the contract. One workgroup a (row, stream), as
 * in libr4d: the two norms, the dot and the gate's own rms are four reductions over the same n.
 * Values are kept in registers across the phases, and gv is narrowed to bf16 before its rms
 * because the reference norms the bf16 tensor it wrote.
 *
 * The sums are double, the reference's: it accumulates each group in double and returns a float
 * scale, so a float tree here would differ in the last bits before the first narrowing. The
 * sigmoid uses the double exp libref uses, for the same reason.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_hc_ple_common.h"
#include "r3d_launch.h"

namespace {

constexpr int kPgThreads = 256;

__device__ __forceinline__ double pg_reduce(double v, double* lds) {
    const uint32_t t = threadIdx.x;
    lds[t] = v;
    __syncthreads();
    for (uint32_t w = kPgThreads / 2; w > 0; w >>= 1) {
        if (t < w) lds[t] += lds[t + w];
        __syncthreads();
    }
    const double r = lds[0];
    __syncthreads();
    return r;
}

__global__ __launch_bounds__(kPgThreads) void ple_gate_kernel(
    const uint16_t* __restrict__ k, const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ v, const float* __restrict__ wk, const float* __restrict__ wq,
    const float* __restrict__ wc, uint16_t* __restrict__ gv, uint16_t* __restrict__ gvn,
    uint32_t n, uint32_t hc, uint32_t k_ld, uint32_t q_ld, uint32_t v_ld, uint32_t g_ld,
    uint32_t gn_ld, float eps, float wadd, float geps, uint32_t per_thread) {
    __shared__ double lds[kPgThreads];
    float mine_k[16], mine_q[16], mine_gv[16];

    const uint32_t row = blockIdx.x / hc;
    const uint32_t c = blockIdx.x % hc;
    const uint32_t t = threadIdx.x;
    const uint32_t base = c * n;   /* this stream's first column of the wide row */

    /* ---- the two norms, from values each thread then keeps ------------------------------------ */
    double sk = 0.0, sq = 0.0;
    for (uint32_t i = 0; i < per_thread; ++i) {
        const uint32_t d = i * kPgThreads + t;
        if (d >= n) break;
        const float kv = bf16_to_f32(k[(size_t)row * k_ld + base + d]);
        const float qv = bf16_to_f32(q[(size_t)row * q_ld + base + d]);
        mine_k[i] = kv;
        mine_q[i] = qv;
        sk += (double)kv * (double)kv;
        sq += (double)qv * (double)qv;
    }
    const float rk = (float)(1.0 / sqrt(pg_reduce(sk, lds) / (double)n + (double)eps));
    const float rq = (float)(1.0 / sqrt(pg_reduce(sq, lds) / (double)n + (double)eps));

    /* ---- the dot between them, over the NARROWED normed values -------------------------------- */
    double dot = 0.0;
    for (uint32_t i = 0; i < per_thread; ++i) {
        const uint32_t d = i * kPgThreads + t;
        if (d >= n) break;
        const float kk = r3d_hcple::narrow(R3D_DT_BF16, mine_k[i] * rk * (wk[base + d] + wadd));
        const float qq = r3d_hcple::narrow(R3D_DT_BF16, mine_q[i] * rq * (wq[base + d] + wadd));
        dot += (double)kk * (double)qq;
    }
    float s = (float)(pg_reduce(dot, lds) / sqrt((double)n));

    const float mag = sqrtf(fabsf(s) < geps ? geps : fabsf(s));
    s = s > 0.0f ? mag : (s < 0.0f ? -mag : 0.0f);
    const float gate = r3d_hcple::sigmoid(s);

    /* ---- the gated value, narrowed, and its own rms from the narrowed bytes ------------------- */
    double sg = 0.0;
    for (uint32_t i = 0; i < per_thread; ++i) {
        const uint32_t d = i * kPgThreads + t;
        if (d >= n) break;
        const float x =
            r3d_hcple::narrow(R3D_DT_BF16, gate * bf16_to_f32(v[(size_t)row * v_ld + d]));
        mine_gv[i] = x;
        gv[(size_t)row * g_ld + base + d] = f32_to_bf16(x);
        sg += (double)x * (double)x;
    }
    const float rg = (float)(1.0 / sqrt(pg_reduce(sg, lds) / (double)n + (double)eps));

    for (uint32_t i = 0; i < per_thread; ++i) {
        const uint32_t d = i * kPgThreads + t;
        if (d >= n) break;
        gvn[(size_t)row * gn_ld + base + d] =
            f32_to_bf16(mine_gv[i] * rg * (wc[base + d] + wadd));
    }
}

}  // namespace

extern "C" {

int r3d_ple_gate(long k, long q, long v, long wk, long wq, long wc, long gv, long gvn, int M,
                 int n, int hc, int k_ld, int q_ld, int v_ld, int g_ld, int gn_ld, float eps,
                 float wadd, float geps, long stream) {
    if (!k || !q || !v || !wk || !wq || !wc || !gv || !gvn) return -1;
    if (M <= 0 || n <= 0 || hc <= 0) return -2;
    const int per_thread = (n + kPgThreads - 1) / kPgThreads;
    if (per_thread > 16) return -2;
    if (k_ld < hc * n || q_ld < hc * n || v_ld < n || g_ld < hc * n || gn_ld < hc * n) return -3;
    R3D_KLAUNCH(ple_gate_kernel, dim3((uint32_t)(M * hc)), dim3(kPgThreads), (hipStream_t)stream,
                (const uint16_t*)k, (const uint16_t*)q, (const uint16_t*)v, (const float*)wk,
                (const float*)wq, (const float*)wc, (uint16_t*)gv, (uint16_t*)gvn, (uint32_t)n,
                (uint32_t)hc, (uint32_t)k_ld, (uint32_t)q_ld, (uint32_t)v_ld, (uint32_t)g_ld,
                (uint32_t)gn_ld, eps, wadd, geps, (uint32_t)per_thread);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

int r3d_rad_ple_gate(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 8) return R3D_NO(RAD_E_INVAL);
    const RadTensor* k = r3d_opt(a, 0), *q = r3d_opt(a, 1), *v = r3d_opt(a, 2),
                    *wk = r3d_opt(a, 3), *wq = r3d_opt(a, 4), *wc = r3d_opt(a, 5),
                    *gv = r3d_opt(a, 6), *gvn = r3d_opt(a, 7);
    if (!r3d_p(k) || !r3d_p(q) || !r3d_p(v) || !r3d_p(wk) || !r3d_p(wq) || !r3d_p(wc) ||
        !r3d_wp(gv) || !r3d_wp(gvn))
        return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(k) || !r3d_is_bf16(q) || !r3d_is_bf16(v) || !r3d_is_bf16(gv) ||
        !r3d_is_bf16(gvn))
        return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(wk) || !r3d_is_f32(wq) || !r3d_is_f32(wc)) return R3D_NO(RAD_E_DTYPE);

    const long long n = r3d_geti_or(a, "n", 0);
    const long long hc = r3d_geti_or(a, "hc", 0);
    if (n <= 0 || hc <= 0) return R3D_NO(RAD_E_SHAPE);
    const long long per_thread = (n + kPgThreads - 1) / kPgThreads;
    if (per_thread > 16) return R3D_NO(RAD_E_UNSUPPORTED);

    const float eps = (float)r3d_getf(a, "eps", 1e-6);
    const float wadd = (float)r3d_getf(a, "wadd", 0.0);
    const float geps = (float)r3d_getf(a, "gate_eps", 1e-6);

    long long T = 0, tq = 0, tv = 0, tg = 0, tn = 0;
    const long long k_ld = r3d_rows_pitch(k, hc * n, &T);
    const long long q_ld = r3d_rows_pitch(q, hc * n, &tq);
    const long long v_ld = r3d_rows_pitch(v, n, &tv);
    const long long g_ld = r3d_rows_pitch(gv, hc * n, &tg);
    const long long gn_ld = r3d_rows_pitch(gvn, hc * n, &tn);
    if (k_ld < 0 || q_ld < 0 || v_ld < 0 || g_ld < 0 || gn_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (T <= 0 || tq != T || tv != T || tg != T || tn != T) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(wk) < hc * n || r3d_numel(wq) < hc * n || r3d_numel(wc) < hc * n)
        return R3D_NO(RAD_E_SHAPE);

    const int rc = r3d_ple_gate((long)(uintptr_t)r3d_p(k), (long)(uintptr_t)r3d_p(q),
                                (long)(uintptr_t)r3d_p(v), (long)(uintptr_t)r3d_p(wk),
                                (long)(uintptr_t)r3d_p(wq), (long)(uintptr_t)r3d_p(wc),
                                (long)(uintptr_t)r3d_wp(gv), (long)(uintptr_t)r3d_wp(gvn), (int)T,
                                (int)n, (int)hc, (int)k_ld, (int)q_ld, (int)v_ld, (int)g_ld,
                                (int)gn_ld, eps, wadd, geps, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

}  /* extern "C" */
