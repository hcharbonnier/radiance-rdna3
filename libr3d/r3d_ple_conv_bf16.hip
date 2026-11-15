/* r3d_ple_conv_bf16.hip -- Per-Layer Embedding's dilated depthwise convolution and residual.
 *
 *   y[t][c] = resid[t][c] + silu( sum_j w[c][j] * x[t - (width-1-j)*dilation][c] )
 *
 * The taps are `dilation` apart, so a width-4, dilation-3 layer carries 9 timesteps of history,
 * not 3. The indexing mirrors libref/ref_ple.cpp line for line: a state slot holds one relative
 * position, a read at `src < 0` comes from the window at `src + hist`, and `has_init` 0 makes that
 * window zeros rather than the slot's stale contents.
 *
 * One thread per channel: a depthwise convolution never mixes channels, so only time must stay in
 * one thread. A thread walks a 32-token tile with consecutive threads on consecutive channels, so
 * a wave reads one contiguous slice of a token row. The window write-back is in the same launch
 * and lives in tile 0, which is sound because hist <= 16 < 32 (the second static_assert below).
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_hc_ple_common.h"
#include "r3d_launch.h"

namespace {

enum {
    kPcThreads = 256,
    kPcBT = 32,
    kPcMaxW = 8,
    kPcMaxHist = 16,
};
static_assert(kPcMaxHist < kPcBT, "the conv history must fit inside tile 0 for the fused write-back");

__global__ __launch_bounds__(kPcThreads) void ple_conv_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, uint16_t* __restrict__ cs,
    const int32_t* __restrict__ cu, const uint16_t* __restrict__ resid,
    const int32_t* __restrict__ sidx, const int32_t* __restrict__ hini,
    const int32_t* __restrict__ nacc, uint16_t* __restrict__ y, uint32_t n, uint32_t wid,
    uint32_t dil, uint32_t hist, uint32_t xld, uint32_t rld, uint32_t yld, uint32_t wld,
    long long cs_slot, long long cs_ch, long long cs_tok, long long state_len) {
    const uint32_t c = blockIdx.x * kPcThreads + threadIdx.x;
    if (c >= n) return;
    const uint32_t s = blockIdx.z;
    const int32_t bos = cu[s];
    const int32_t slen = cu[s + 1] - bos;
    if (slen <= 0) return;
    const int32_t t0 = (int32_t)(blockIdx.y * kPcBT);
    if (t0 >= slen) return;
    const int32_t t1 = min(t0 + (int32_t)kPcBT, slen);

    const int32_t slot = sidx ? sidx[s] : (int32_t)s;
    if (slot < 0) return;
    const bool warm = cs && (!hini || hini[s] != 0);
    const long long off = nacc ? (long long)nacc[s] - 1 : 0;
    if (off < 0 || (hist && off + (long long)hist > state_len)) return;

    float wt[kPcMaxW];
#pragma unroll
    for (uint32_t j = 0; j < kPcMaxW; ++j)
        wt[j] = j < wid ? bf16_to_f32(w[(size_t)c * wld + j]) : 0.0f;

    const uint16_t* __restrict__ xc = x + (size_t)bos * xld + c;
    const uint16_t* __restrict__ rc = resid ? resid + (size_t)bos * rld + c : nullptr;
    uint16_t* __restrict__ dst = cs ? cs + (size_t)slot * cs_slot + (size_t)c * cs_ch : nullptr;
    const uint16_t* __restrict__ hc = warm ? dst + off * cs_tok : nullptr;
    uint16_t* __restrict__ yc = y + (size_t)bos * yld + c;

    for (int32_t t = t0; t < t1; ++t) {
        float acc = 0.0f;
#pragma unroll
        for (uint32_t j = 0; j < kPcMaxW; ++j) {
            if (j < wid) {
                const int32_t src = t - (int32_t)hist + (int32_t)(j * dil);
                const float v =
                    src >= 0 ? bf16_to_f32(xc[(size_t)src * xld])
                             : (hc ? bf16_to_f32(hc[(size_t)(src + (int32_t)hist) * cs_tok]) : 0.0f);
                acc += v * wt[j];
            }
        }
        const float r = rc ? bf16_to_f32(rc[(size_t)t * rld]) : 0.0f;
        yc[(size_t)t * yld] = f32_to_bf16(r + r3d_hcple::silu(acc));
    }

    if (!dst || blockIdx.y != 0) return;
    if (hist == 0) return;
    uint16_t keep[kPcMaxHist];
    if (nacc) {
#pragma unroll
        for (uint32_t i = 0; i + 1 < kPcMaxHist; ++i)
            if (i + 1 < hist) keep[i] = hc ? hc[(size_t)(i + 1) * cs_tok] : (uint16_t)0;
#pragma unroll
        for (uint32_t i = 0; i + 1 < kPcMaxHist; ++i)
            if (i + 1 < hist) dst[(size_t)i * cs_tok] = keep[i];
        for (int32_t t = 0; t < slen && (long long)hist - 1 + t < state_len; ++t)
            dst[(size_t)(hist - 1 + (uint32_t)t) * cs_tok] = xc[(size_t)t * xld];
        return;
    }
#pragma unroll
    for (uint32_t i = 0; i < kPcMaxHist; ++i) {
        if (i < hist) {
            const int32_t src = slen - (int32_t)hist + (int32_t)i;
            keep[i] = src >= 0 ? xc[(size_t)src * xld]
                               : (hc ? hc[(size_t)(src + (int32_t)hist) * cs_tok] : (uint16_t)0);
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < kPcMaxHist; ++i)
        if (i < hist) dst[(size_t)i * cs_tok] = keep[i];
}

}  // namespace

extern "C" {

int r3d_ple_conv(long x, long w, long cs, long cu, long resid, long cidx, long hini, long nacc,
                 long y, int n, int width, int dilation, int x_ld, int r_ld, int y_ld, int w_ld,
                 long long cs_slot, long long cs_ch, long long cs_tok, long long state_len,
                 int n_seq, int T, long stream) {
    if (!x || !w || !cu || !y) return -1;
    if (n <= 0 || width <= 0 || width > kPcMaxW || dilation <= 0) return -2;
    const int hist = (width - 1) * dilation;
    if (hist > kPcMaxHist) return -2;
    if (x_ld < n || y_ld < n || (resid && r_ld < n) || w_ld < width) return -3;
    if (n_seq <= 0 || T <= 0) return -2;
    const uint32_t cb = (uint32_t)((n + kPcThreads - 1) / kPcThreads);
    const uint32_t tb = (uint32_t)((T + kPcBT - 1) / kPcBT);
    R3D_KLAUNCH(ple_conv_kernel, dim3(cb, tb, (uint32_t)n_seq), dim3(kPcThreads),
                (hipStream_t)stream, (const uint16_t*)x, (const uint16_t*)w, (uint16_t*)cs,
                (const int32_t*)cu, (const uint16_t*)resid, (const int32_t*)cidx,
                (const int32_t*)hini, (const int32_t*)nacc, (uint16_t*)y, (uint32_t)n,
                (uint32_t)width, (uint32_t)dilation, (uint32_t)hist, (uint32_t)x_ld,
                (uint32_t)r_ld, (uint32_t)y_ld, (uint32_t)w_ld, cs_slot, cs_ch, cs_tok,
                state_len);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

int r3d_rad_ple_conv(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 9) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0), *w = r3d_opt(a, 1), *cs = r3d_opt(a, 2),
                    *cu = r3d_opt(a, 3), *resid = r3d_opt(a, 4), *cidx = r3d_opt(a, 5),
                    *hini = r3d_opt(a, 6), *nacc = r3d_opt(a, 7), *y = r3d_opt(a, 8);
    if (!r3d_p(x) || !r3d_p(w) || !r3d_p(cu) || !r3d_wp(y)) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(x) || !r3d_is_bf16(w) || !r3d_is_bf16(y)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(resid) && !r3d_is_bf16(resid)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(cs) && !r3d_is_bf16(cs)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_i32(cu)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(cidx) && !r3d_is_i32(cidx)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(hini) && !r3d_is_i32(hini)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(nacc) && !r3d_is_i32(nacc)) return R3D_NO(RAD_E_DTYPE);

    const long long n = r3d_geti_or(a, "n", 0);
    const long long wid = r3d_geti_or(a, "width", 0);
    const long long dil = r3d_geti_or(a, "dilation", 1);
    if (n <= 0 || wid <= 0 || dil <= 0) return R3D_NO(RAD_E_SHAPE);
    if (wid > kPcMaxW) return R3D_NO(RAD_E_UNSUPPORTED);
    const long long hist = (wid - 1) * dil;
    if (hist > kPcMaxHist) return R3D_NO(RAD_E_UNSUPPORTED);

    long long T = 0, ty = 0, tr = 0, wrows = 0;
    const long long xld = r3d_rows_pitch(x, n, &T);
    const long long yld = r3d_rows_pitch(y, n, &ty);
    const long long rld = r3d_has(resid) ? r3d_rows_pitch(resid, n, &tr) : 0;
    const long long wld = r3d_rows_pitch(w, wid, &wrows);
    if (xld < 0 || yld < 0 || wld < 0 || (r3d_has(resid) && rld < 0)) return R3D_NO(RAD_E_STRIDE);
    if (T <= 0 || ty != T || (r3d_has(resid) && tr != T)) return R3D_NO(RAD_E_SHAPE);
    if (wrows < n) return R3D_NO(RAD_E_SHAPE);

    const long long N = r3d_numel(cu) - 1;
    if (N <= 0) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(cu)) return R3D_NO(RAD_E_STRIDE);

    long long cs_slot = 0, cs_ch = 0, cs_tok = 0, state_len = 0;
    const bool have_cs = r3d_has(cs) && hist > 0;
    if (have_cs) {
        if (cs->rank != 3) return R3D_NO(RAD_E_SHAPE);
        if (r3d_dim(cs, 1) < n || r3d_dim(cs, 2) < hist) return R3D_NO(RAD_E_SHAPE);
        cs_slot = r3d_stride(cs, 0);
        cs_ch = r3d_stride(cs, 1);
        cs_tok = r3d_stride(cs, 2);
        state_len = r3d_dim(cs, 2);
    }

    const int rc = r3d_ple_conv((long)(uintptr_t)r3d_p(x), (long)(uintptr_t)r3d_p(w),
                                have_cs ? (long)(uintptr_t)r3d_wp(cs) : 0L,
                                (long)(uintptr_t)r3d_p(cu),
                                r3d_has(resid) ? (long)(uintptr_t)r3d_p(resid) : 0L,
                                r3d_has(cidx) ? (long)(uintptr_t)r3d_p(cidx) : 0L,
                                r3d_has(hini) ? (long)(uintptr_t)r3d_p(hini) : 0L,
                                r3d_has(nacc) ? (long)(uintptr_t)r3d_p(nacc) : 0L,
                                (long)(uintptr_t)r3d_wp(y), (int)n, (int)wid, (int)dil, (int)xld,
                                (int)rld, (int)yld, (int)wld, cs_slot, cs_ch, cs_tok, state_len,
                                (int)N, (int)T, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

}  /* extern "C" */
