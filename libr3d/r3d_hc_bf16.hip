/* r3d_hc_bf16.hip -- HyperConnections' read, write, enter and projection.
 *
 *   h is [M, hc*n] bf16, one wide residual stream cut into hc connections of n.
 *   read:  N[c*n+i] = bf16(h * rsqrt(mean(h_c^2)+eps) * (w+wadd))
 *          d[j]      = bf16(silu(sum_i N[i] * down[j][i] / hc))
 *          gv[j]     = sum_k d[k] * up[j][k]                 j is a connection's column
 *          x[i]      = sum_c bf16(sigmoid(gv[c*n+i])) * N[c*n+i] / hc
 *          inj[c]    = bf16(2 * sigmoid(sum_i N[i] * injw[c][i] / hc))
 *          q / rq    = the quantised and rotated copies of x, exactly quant_act's and
 *                      had_quant_act's group arithmetic
 *   write:   h += bf16(inj[c]) * bf16(y[i])
 *   enter:   h[c*n+i] = bf16(x[i])
 *   project: for a connection with on[r] set, h -= g * v where
 *            g = (float)(sum_i h[i] * v[i] * scale)        (double sum, as libref writes it)
 *
 * libref/ref_hc.cpp is the oracle and docs/OPS.md the contract. The read is split over four
 * launches because each phase has a different natural grid:
 *
 *   norm    one block a row, hc threads: the hc double sums and their scales;
 *   down    one block a (row, 16-output tile): the block rebuilds N in LDS (the normalisation is
 *           cheap beside the weight stream) and each of 16 threads owns one output row, summing
 *           i ascending in f32 exactly as ref does; the first tile also writes `inj`;
 *   up      one block a (row, 256 columns): each thread owns one column and sums c ascending;
 *   quant   one block a (row, group): q is a sequential group scan, rq a wave transform in LDS.
 *
 * `d` and the per-stream scales travel in the row's scratch: M*(lowrank*2 + hc*4) bytes. The
 * split-K weight stream is read once. The accumulations are ascending and f32 (double for the
 * norm), with no FMA, so the read is bit-identical to the reference, not merely close.
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_hc_ple_common.h"
#include "r3d_launch.h"

namespace {

constexpr int kHcThreads = 256;
constexpr int kHcDownTile = 16;
constexpr int kHcQuantScratch = 256;   /* the deepest rotated group the quantiser transforms */

/* Scratch layout: d [M, lowrank] bf16, then the stream scales [M, hc] f32, 16-byte aligned. */
static inline long long hc_sc_off(long long M, long long lr) {
    return ((M * lr * 2 + 15) / 16) * 16;
}
static inline long long hc_scratch_bytes(long long M, long long lr, long long hc) {
    return hc_sc_off(M, lr) + M * hc * 4;
}

/* A mixing weight: bf16 row-major, or the E4M3 row with its scales in the row's tail (the
 * layout hook below). `K` is the dot length and `G` the scale group. */
struct MixW {
    const void* base;
    int f8;
    int ld;     /* bf16 row pitch in elements */
    int stride; /* E4M3 row stride in bytes */
    int K;
    int G;
};

__device__ __forceinline__ float mix_get(const MixW& w, long long row, int i) {
    if (!w.f8) return bf16_to_f32(((const uint16_t*)w.base)[row * w.ld + i]);
    const uint8_t* r = (const uint8_t*)w.base + row * w.stride;
    return e4m3_to_f32(r[i]) * ((const float*)(r + w.K))[i / w.G];
}

/* ---- the per-stream norm scale ------------------------------------------------------------- */
__global__ __launch_bounds__(kHcThreads) void hc_norm_kernel(
    const uint16_t* __restrict__ h, const float* __restrict__ w, float* __restrict__ sc, int M,
    int n, int hc, int h_ld, float eps) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const int c = (int)threadIdx.x;
    if (c >= hc) return;
    const uint16_t* hp = h + (long long)row * h_ld + (long long)c * n;
    double ss = 0.0;
    for (int i = 0; i < n; ++i) {
        const float v = bf16_to_f32(hp[i]);
        ss += (double)v * (double)v;
    }
    const float mean = (float)(ss / (double)n);
    sc[(long long)row * hc + c] = 1.0f / sqrtf(mean + eps);
}

/* ---- down: d and inj ----------------------------------------------------------------------- */
__global__ __launch_bounds__(kHcThreads) void hc_down_kernel(
    const uint16_t* __restrict__ h, const float* __restrict__ w, const void* __restrict__ down,
    const uint16_t* __restrict__ injw, uint16_t* __restrict__ d, const float* __restrict__ sc,
    uint16_t* __restrict__ inj, int M, int n, int hc, int lr, int h_ld, int injw_ld, int inj_ld,
    int f8, int down_ld, int down_stride, int down_G, float wadd) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long HN = (long long)hc * n;
    extern __shared__ float N[];
    const float* scr = sc + (long long)row * hc;
    const uint16_t* hr = h + (long long)row * h_ld;
    for (long long j = threadIdx.x; j < HN; j += kHcThreads) {
        const int c = (int)(j / n);
        N[j] = r3d_hcple::narrow(R3D_DT_BF16, bf16_to_f32(hr[j]) * scr[c] * (w[j] + wadd));
    }
    __syncthreads();

    MixW dw{down, f8, down_ld, down_stride, (int)HN, down_G};
    const int j0 = (int)blockIdx.y * kHcDownTile;
    const int j1 = min(j0 + kHcDownTile, lr);
    const int t = (int)threadIdx.x;
    if (t < j1 - j0) {
        const int j = j0 + t;
        float acc = 0.0f;
        for (long long i = 0; i < HN; ++i) acc += N[i] * mix_get(dw, j, (int)i);
        d[(long long)row * lr + j] = f32_to_bf16(r3d_hcple::silu(acc / (float)hc));
    }
    /* The injection gains are one dot a stream; the first tile owns them so no other block repeats
     * the write. */
    if (blockIdx.y == 0 && injw && inj && t < hc) {
        float acc = 0.0f;
        const uint16_t* base = injw + (long long)t * injw_ld;
        for (long long i = 0; i < HN; ++i) acc += N[i] * bf16_to_f32(base[i]);
        inj[(long long)row * inj_ld + t] =
            f32_to_bf16(2.0f * r3d_hcple::sigmoid(acc / (float)hc));
    }
}

/* ---- up and x ------------------------------------------------------------------------------ */
__global__ __launch_bounds__(kHcThreads) void hc_up_kernel(
    const uint16_t* __restrict__ h, const float* __restrict__ w, const void* __restrict__ up,
    const uint16_t* __restrict__ d, const float* __restrict__ sc, void* __restrict__ x, int M,
    int n, int hc, int lr, int h_ld, int x_ld, int x_dt, int f8, int up_ld, int up_stride,
    int up_G, float wadd) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const int i = (int)blockIdx.y * kHcThreads + (int)threadIdx.x;
    if (i >= n) return;
    extern __shared__ float dsh[];
    for (int k = (int)threadIdx.x; k < lr; k += kHcThreads)
        dsh[k] = bf16_to_f32(d[(long long)row * lr + k]);
    __syncthreads();

    const float* scr = sc + (long long)row * hc;
    const uint16_t* hr = h + (long long)row * h_ld;
    MixW uw{up, f8, up_ld, up_stride, lr, up_G};
    float acc = 0.0f;
    for (int c = 0; c < hc; ++c) {
        const long long j = (long long)c * n + i;
        const float Nj =
            r3d_hcple::narrow(R3D_DT_BF16, bf16_to_f32(hr[j]) * scr[c] * (w[j] + wadd));
        float gv = 0.0f;
        for (int k = 0; k < lr; ++k) gv += dsh[k] * mix_get(uw, j, k);
        acc += r3d_hcple::narrow(R3D_DT_BF16, r3d_hcple::sigmoid(gv)) * Nj;
    }
    r3d_hcple::st_elem(x_dt, x, (long long)row * x_ld + i, acc / (float)hc);
}

/* ---- q and rq ------------------------------------------------------------------------------ */
__global__ __launch_bounds__(kHcThreads) void hc_quant_kernel(
    const void* __restrict__ x, int x_dt, int M, int n, int x_ld, int qg, int ngrp, void* __restrict__ q,
    float* __restrict__ qs, int q_dt, int q_ld, int qs_ld, int rg, int rgrp, void* __restrict__ rq,
    float* __restrict__ rs, int rq_dt, int rq_ld, int rs_ld) {
    const int row = (int)blockIdx.x;
    const int gi = (int)blockIdx.y;
    if (row >= M) return;
    __shared__ float s[kHcQuantScratch];
    const int t = (int)threadIdx.x;

    if (gi < ngrp) {
        if (t == 0) {
            const long long k0 = (long long)gi * qg;
            const int kn = (int)min((long long)qg, (long long)n - k0);
            float amax = 0.0f;
            for (int i = 0; i < kn; ++i) {
                const float v = r3d_hcple::ld_elem(x_dt, x, (long long)row * x_ld + k0 + i);
                amax = fmaxf(amax, fabsf(v));
            }
            const float top = q_dt == R3D_DT_I8 ? 127.0f : 448.0f;
            const float scale = amax > 0.0f ? amax * (1.0f / top) : 1.0f;
            const float inv = 1.0f / scale;
            for (int i = 0; i < kn; ++i) {
                const float v = r3d_hcple::ld_elem(x_dt, x, (long long)row * x_ld + k0 + i) * inv;
                const float c = fminf(fmaxf(v, -top), top);
                r3d_hcple::st_elem(q_dt, q, (long long)row * q_ld + k0 + i, c);
            }
            qs[(long long)row * qs_ld + gi] = scale;
        }
    }

    if (gi < rgrp) {
        if (t < 32) {
            const uint32_t lane = (uint32_t)t;
            const long long k0 = (long long)gi * rg;
            for (int i = t; i < rg; i += 32)
                s[i] = r3d_hcple::ld_elem(x_dt, x, (long long)row * x_ld + k0 + i);
            __syncwarp();
            if (lane == 0) {
                for (uint32_t h = 1; h < (uint32_t)rg; h <<= 1) {
                    for (uint32_t b = 0; b < (uint32_t)rg; b += h << 1) {
                        for (uint32_t j = 0; j < h; ++j) {
                            const float a = s[b + j], c = s[b + j + h];
                            s[b + j] = a + c;
                            s[b + j + h] = a - c;
                        }
                    }
                }
            }
            __syncwarp();
            float amax = 0.0f;
            for (int i = t; i < rg; i += 32) amax = fmaxf(amax, fabsf(s[i]));
            amax = r3d_xor_max<32>(amax);
            const float top = rq_dt == R3D_DT_I8 ? 127.0f : 448.0f;
            const float scale = amax > 0.0f ? amax * (1.0f / top) : 1.0f;
            const float inv = 1.0f / scale;
            for (int i = t; i < rg; i += 32) {
                const float v = fminf(fmaxf(s[i] * inv, -top), top);
                r3d_hcple::st_elem(rq_dt, rq, (long long)row * rq_ld + k0 + i, v);
            }
            if (lane == 0) rs[(long long)row * rs_ld + gi] = scale;
        }
    }
}

/* ---- the write half ------------------------------------------------------------------------ */
__global__ __launch_bounds__(kHcThreads) void hc_write_kernel(
    const uint16_t* __restrict__ y, const uint16_t* __restrict__ inj, uint16_t* __restrict__ h,
    int M, int n, int hc, int y_ld, int inj_ld, int h_ld) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long HN = (long long)hc * n;
    const uint16_t* yr = y + (long long)row * y_ld;
    const uint16_t* ir = inj + (long long)row * inj_ld;
    uint16_t* hr = h + (long long)row * h_ld;
    for (long long j = threadIdx.x; j < HN; j += kHcThreads) {
        const int c = (int)(j / n);
        const int i = (int)(j - (long long)c * n);
        const float hv = bf16_to_f32(hr[j]);
        const float gain = bf16_to_f32(ir[c]);
        const float yv = bf16_to_f32(yr[i]);
        hr[j] = f32_to_bf16(hv + gain * yv);
    }
}

/* ---- enter --------------------------------------------------------------------------------- */
__global__ __launch_bounds__(kHcThreads) void hc_enter_kernel(
    const void* __restrict__ x, int x_dt, int x_ld, void* __restrict__ h, int h_dt, int h_ld,
    int M, int n, int hc) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long HN = (long long)hc * n;
    for (long long j = threadIdx.x; j < HN; j += kHcThreads) {
        const int i = (int)(j % n);
        r3d_hcple::st_elem(h_dt, h, (long long)row * h_ld + j,
                           r3d_hcple::ld_elem(x_dt, x, (long long)row * x_ld + i));
    }
}

/* ---- project ------------------------------------------------------------------------------- */
__global__ __launch_bounds__(kHcThreads) void hc_project_kernel(
    uint16_t* __restrict__ h, const float* __restrict__ v, const int32_t* __restrict__ on,
    int M, int n, int hc, int h_ld, int on_ld, float scale) {
    const int row = (int)blockIdx.x;
    const int c = (int)blockIdx.y;
    if (row >= M) return;
    if (on && on[(long long)row * on_ld] == 0) return;
    __shared__ float g;
    const long long base = (long long)c * n;
    uint16_t* hr = h + (long long)row * h_ld;
    if (threadIdx.x == 0) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i)
            acc += (double)bf16_to_f32(hr[base + i]) * (double)v[i];
        g = (float)(acc * (double)scale);
    }
    __syncthreads();
    for (int i = (int)threadIdx.x; i < n; i += kHcThreads)
        hr[base + i] = f32_to_bf16(bf16_to_f32(hr[base + i]) - g * v[i]);
}

}  // namespace

extern "C" {

int64_t r3d_hc_read_scratch_bytes(int M, int lowrank, int hc) {
    if (M <= 0 || lowrank <= 0 || hc <= 0) return -1;
    return hc_scratch_bytes(M, lowrank, hc);
}

int r3d_hc_read(long h, long w, long down, long up, long injw, long x, long inj, long q, long qs,
                long rq, long rs, long scratch, long long scratch_bytes, int M, int n, int hc,
                int lr, float eps, float wadd, int group, int h_ld, int x_ld, int inj_ld,
                int injw_ld, int q_ld, int qs_ld, int rq_ld, int rs_ld, int mix_e4m3, int q_dt, int rq_dt,
                int x_dt, int down_ld, int down_stride, int down_G, int up_ld, int up_stride,
                int up_G, long stream) {
    if (!h || !w || !down || !up || !x || !scratch) return -1;
    if (M <= 0 || n <= 0 || hc <= 0 || lr <= 0 || h_ld < hc * n || x_ld < n) return -2;
    if (scratch_bytes < hc_scratch_bytes(M, lr, hc)) return -2;

    const long long sc_off = hc_sc_off(M, lr);
    uint16_t* d = (uint16_t*)scratch;
    float* sc = (float*)((char*)scratch + sc_off);

    const int norm_thr = kHcThreads;
    R3D_KLAUNCH(hc_norm_kernel, dim3((uint32_t)M), dim3(norm_thr), (hipStream_t)stream,
                (const uint16_t*)h, (const float*)w, sc, M, n, hc, h_ld, eps);

    const int down_blocks = (lr + kHcDownTile - 1) / kHcDownTile;
    R3D_KLAUNCH_S(hc_down_kernel, dim3((uint32_t)M, (uint32_t)down_blocks), dim3(kHcThreads),
                  (size_t)hc * n * sizeof(float), (hipStream_t)stream, (const uint16_t*)h,
                  (const float*)w, (const void*)down, injw ? (const uint16_t*)injw : nullptr, d, sc,
                  inj ? (uint16_t*)inj : nullptr, M, n, hc, lr, h_ld,
                  injw ? injw_ld : 0, inj ? inj_ld : 0, mix_e4m3, down_ld, down_stride, down_G,
                  wadd);

    R3D_KLAUNCH_S(hc_up_kernel, dim3((uint32_t)M, (uint32_t)((n + kHcThreads - 1) / kHcThreads)),
                  dim3(kHcThreads), (size_t)lr * sizeof(float), (hipStream_t)stream,
                  (const uint16_t*)h, (const float*)w, (const void*)up, d, sc, (void*)x, M, n, hc,
                  lr, h_ld, x_ld, x_dt, mix_e4m3, up_ld, up_stride, up_G, wadd);

    if (q || rq) {
        const int qg = (group > 0 && group <= n) ? group : n;
        const int ngrp = q ? (n + qg - 1) / qg : 0;
        const int rgrp = rq ? n / (group > 0 ? group : n) : 0;
        const int gx = ngrp > rgrp ? ngrp : rgrp;
        if (gx > 0)
            R3D_KLAUNCH(hc_quant_kernel, dim3((uint32_t)M, (uint32_t)gx), dim3(kHcThreads),
                        (hipStream_t)stream, (const void*)x, x_dt, M, n, x_ld, qg, ngrp,
                        (void*)q, (float*)qs, q_dt, q_ld, qs_ld, rq ? (group > 0 ? group : n) : 0,
                        rgrp, (void*)rq, (float*)rs, rq_dt, rq_ld, rs_ld);
    }
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

int r3d_hc_write(long y, long inj, long h, int M, int n, int hc, int y_ld, int inj_ld, int h_ld,
                 long stream) {
    if (!y || !inj || !h) return -1;
    if (M <= 0 || n <= 0 || hc <= 0 || y_ld < n || inj_ld < hc || h_ld < hc * n) return -2;
    R3D_KLAUNCH(hc_write_kernel, dim3((uint32_t)M), dim3(kHcThreads), (hipStream_t)stream,
                (const uint16_t*)y, (const uint16_t*)inj, (uint16_t*)h, M, n, hc, y_ld, inj_ld,
                h_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

int r3d_hc_enter(long x, long h, int M, int n, int hc, int x_ld, int h_ld, int x_dt, int h_dt,
                 long stream) {
    if (!x || !h) return -1;
    if (M <= 0 || n <= 0 || hc <= 0 || x_ld < n || h_ld < hc * n) return -2;
    R3D_KLAUNCH(hc_enter_kernel, dim3((uint32_t)M), dim3(kHcThreads), (hipStream_t)stream,
                (const void*)x, x_dt, x_ld, (void*)h, h_dt, h_ld, M, n, hc);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

int r3d_hc_project(long h, long v, long on, int M, int n, int hc, float scale, int h_ld, int on_ld,
                   long stream) {
    if (!h || !v) return -1;
    if (M <= 0 || n <= 0 || hc <= 0 || h_ld < hc * n) return -2;
    R3D_KLAUNCH(hc_project_kernel, dim3((uint32_t)M, (uint32_t)hc), dim3(kHcThreads),
                (hipStream_t)stream, (uint16_t*)h, (const float*)v, (const int32_t*)on, M, n, hc,
                h_ld, on_ld, scale);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ---- the shims ----------------------------------------------------------------------------- */

extern "C" int64_t r3d_rad_hc_read_scratch(const RadArgs* a) {
    const long long M = r3d_geti_or(a, "M", 0);
    const long long n = r3d_geti_or(a, "n", 0);
    const long long hc = r3d_geti_or(a, "hc", 0);
    const long long lr = r3d_geti_or(a, "lowrank", 0);
    if (M <= 0 || n <= 0 || hc <= 0 || lr <= 0) return R3D_NO(RAD_E_SHAPE);
    return hc_scratch_bytes(M, lr, hc);
}

}  /* extern "C" */

/* The joined mixing weight: E4M3 codes and a f32 scale per 128 of K (down) or K/4 (up), both
 * row-major, packed into one row with the scales in its tail. The bytes and the tag are libr4d's
 * (`r4d.hc.e4m3.*`): a container baked on an RDNA4 build then loads on gfx11 without a re-pack,
 * and the two libraries' layouts are held byte-identical by construction. */
namespace {

struct HcGeom { long long rows, K, G; };

/* The role of selected plane i, and whether the selection names exactly these two roles. */
const char* hc_role(const RadEncoding* e, const int* sel, int i) {
    return (e && sel && sel[i] >= 0 && sel[i] < e->n_planes) ? e->plane[sel[i]].role : "";
}

bool hc_selects(const RadEncoding* e, const int* sel, int n) {
    return n == 2 && !std::strncmp(hc_role(e, sel, 0), "codes", RAD_ENC_STR) &&
           !std::strncmp(hc_role(e, sel, 1), "scale", RAD_ENC_STR);
}

bool hc_dims(const RadTensor* t, long long rows, long long cols) {
    return t && t->rank == 2 && t->shape[0] == rows && t->shape[1] == cols;
}

bool hc_mix_geom(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                 const RadTensor* pl, int n, HcGeom* g) {
    if (!enc || n != 2 || !rad_enc_is(enc, "affine")) return false;
    if (enc->n_planes != 2 || enc->plane[0].dtype != RAD_F8E4M3 || enc->plane[0].block[1] != 1)
        return false;
    const int s = rad_enc_find(enc, "scale");
    if (s < 0 || enc->plane[s].dtype != RAD_F32 || enc->plane[s].kind != RAD_PLANE_TILED)
        return false;
    const long long n_ = rad_param_geti(p, n_p, "n", 0);
    const long long hc = rad_param_geti(p, n_p, "hc", 0);
    const long long lr = rad_param_geti(p, n_p, "lowrank", 0);
    const long long HN = hc * n_;
    if (HN <= 0 || lr <= 0) return false;
    if (operand == 2) {
        g->rows = lr;
        g->K = HN;
        g->G = 128;
    } else if (operand == 3) {
        g->rows = HN;
        g->K = lr;
        g->G = lr / 4;
    } else {
        return false;
    }
    if (g->G <= 0 || g->G % 16 || g->K % g->G) return false;
    /* The down stream's scale group is 128; the up stream has one group per four lowrank
     * columns, so its length is the one that must be a multiple of 64. */
    if (operand == 2 && g->K % 128) return false;
    if (operand == 3 && g->K % 64) return false;
    if (enc->plane[s].block[0] != 1 || enc->plane[s].block[1] != g->G) return false;
    if (!hc_selects(enc, sel, n)) return false;
    if (!hc_dims(&pl[0], g->rows, g->K) || !hc_dims(&pl[1], g->rows, g->K / g->G))
        return false;
    return true;
}

}  // namespace

extern "C" int r3d_rad_layout_hc_mix(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                     int n, RadLayout* out) {
    if (operand != 2 && operand != 3) return RAD_E_UNSUPPORTED;
    HcGeom g;
    if (!hc_mix_geom(p, n_p, operand, enc, sel, pl, n, &g)) return RAD_E_DTYPE;
    const long long stride = ((g.K + 4 * (g.K / g.G)) + 15) / 16 * 16;
    std::memset(out, 0, sizeof(*out));
    out->tag = g.G == 128 ? "r4d.hc.e4m3.g128" : "r4d.hc.e4m3.q4";
    out->dtype = RAD_F8E4M3;
    out->rank = 2;
    out->shape[0] = g.rows;
    out->shape[1] = stride;
    out->align = 16;
    out->bytes = g.rows * stride;
    return RAD_OK;
}

extern "C" int r3d_rad_relayout_hc_mix(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, void* dst,
                                       int64_t dst_bytes) {
    if (operand != 2 && operand != 3) return RAD_E_UNSUPPORTED;
    HcGeom g;
    if (!hc_mix_geom(p, n_p, operand, enc, sel, pl, n, &g)) return RAD_E_DTYPE;
    const long long stride = ((g.K + 4 * (g.K / g.G)) + 15) / 16 * 16;
    const long long ng = g.K / g.G;
    if (dst_bytes != g.rows * stride || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* codes = (const uint8_t*)pl[0].data;
    const float* sc = (const float*)pl[1].data;
    uint8_t* d = (uint8_t*)dst;
    const long long used = g.K + 4 * ng;
    for (long long r = 0; r < g.rows; ++r) {
        uint8_t* row = d + r * stride;
        std::memcpy(row, codes + r * g.K, (size_t)g.K);
        std::memcpy(row + g.K, sc + r * ng, (size_t)(4 * ng));
        if (stride > used) std::memset(row + used, 0, (size_t)(stride - used));
    }
    return RAD_OK;
}

extern "C" int r3d_rad_unrelayout_hc_mix(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel, const void* src,
                                         int64_t src_bytes, const RadTensor* pl, int n) {
    if (operand != 2 && operand != 3) return RAD_E_UNSUPPORTED;
    HcGeom g;
    if (!hc_mix_geom(p, n_p, operand, enc, sel, pl, n, &g)) return RAD_E_DTYPE;
    const long long stride = ((g.K + 4 * (g.K / g.G)) + 15) / 16 * 16;
    const long long ng = g.K / g.G;
    if (src_bytes != g.rows * stride || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* codes = (uint8_t*)pl[0].data;
    float* sc = (float*)pl[1].data;
    for (long long r = 0; r < g.rows; ++r) {
        const uint8_t* row = s + r * stride;
        std::memcpy(codes + r * g.K, row, (size_t)g.K);
        std::memcpy(sc + r * ng, row + g.K, (size_t)(4 * ng));
    }
    return RAD_OK;
}

extern "C" int r3d_rad_hc_read(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 11) return R3D_NO(RAD_E_INVAL);
    const RadTensor* h = r3d_opt(a, 0), *w = r3d_opt(a, 1), *down = r3d_opt(a, 2),
                    *up = r3d_opt(a, 3), *injw = r3d_opt(a, 4), *x = r3d_opt(a, 5),
                    *inj = r3d_opt(a, 6), *q = r3d_opt(a, 7), *qs = r3d_opt(a, 8),
                    *rq = r3d_opt(a, 9), *rs = r3d_opt(a, 10);
    if (!r3d_p(h) || !r3d_p(w) || !r3d_p(down) || !r3d_p(up) || !r3d_wp(x))
        return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(h) || !r3d_is_f32(w)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(w) || !r3d_contig(h)) return R3D_NO(RAD_E_STRIDE);
    const int x_dt = r3d_dt_code(x->dtype);
    if (x_dt != R3D_DT_BF16 && x_dt != R3D_DT_F16 && x_dt != R3D_DT_F32)
        return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(inj) && !r3d_is_bf16(inj)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(injw) && !r3d_is_bf16(injw)) return R3D_NO(RAD_E_DTYPE);

    const long long n = r3d_geti_or(a, "n", 0);
    const long long hc = r3d_geti_or(a, "hc", 0);
    const long long lr = r3d_geti_or(a, "lowrank", 0);
    if (n <= 0 || hc <= 1 || lr <= 0) return R3D_NO(RAD_E_SHAPE);
    if (hc > 8) return R3D_NO(RAD_E_UNSUPPORTED);
    const long long HN = hc * n;
    if (HN > 12288 || lr > 12288) return R3D_NO(RAD_E_UNSUPPORTED);   /* one 48 KiB LDS stage */

    const char* mix = r3d_gets(a, "mix");
    bool mix_e4m3 = false;
    if (mix) {
        if (!std::strcmp(mix, "e4m3")) mix_e4m3 = true;
        else if (std::strcmp(mix, "bf16")) return R3D_NO(RAD_E_INVAL);
    }

    long long M = 0;
    const long long h_ld = r3d_rows_pitch(h, HN, &M);
    if (h_ld < 0 || M <= 0) return R3D_NO(RAD_E_STRIDE);
    long long Mx = 0;
    const long long x_ld = r3d_rows_pitch(x, n, &Mx);
    if (x_ld < 0 || Mx != M) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(x)) return R3D_NO(RAD_E_STRIDE);
    if (r3d_numel(w) < HN) return R3D_NO(RAD_E_SHAPE);

    long long inj_ld = 0;
    if (r3d_has(inj)) {
        long long Mi = 0;
        inj_ld = r3d_rows_pitch(inj, hc, &Mi);
        if (inj_ld < 0 || Mi != M) return R3D_NO(RAD_E_SHAPE);
        if (!r3d_contig(inj)) return R3D_NO(RAD_E_STRIDE);
    }
    long long down_ld = HN, down_stride = 0, down_G = 0;
    if (mix_e4m3) {
        if (!r3d_is_e4m3(down) || !r3d_is_e4m3(up)) return R3D_NO(RAD_E_DTYPE);
        if (HN % 128 || lr % 64) return R3D_NO(RAD_E_SHAPE);
        down_G = 128;
        down_stride = (int)(((HN + 4 * (HN / 128)) + 15) / 16 * 16);
        if (r3d_numel(down) < lr * down_stride) return R3D_NO(RAD_E_SHAPE);
    } else {
        if (!r3d_is_bf16(down) || !r3d_is_bf16(up)) return R3D_NO(RAD_E_DTYPE);
        if (!r3d_contig(down) || !r3d_contig(up)) return R3D_NO(RAD_E_STRIDE);
        long long dr = 0;
        down_ld = (int)r3d_rows_pitch(down, HN, &dr);
        if (down_ld < 0 || dr != lr) return R3D_NO(RAD_E_SHAPE);
    }
    long long up_ld = lr, up_stride = 0, up_G = 0;
    if (mix_e4m3) {
        up_G = (int)(lr / 4);
        up_stride = (int)(((lr + 4 * (lr / up_G)) + 15) / 16 * 16);
        if (r3d_numel(up) < HN * up_stride) return R3D_NO(RAD_E_SHAPE);
    } else {
        long long ur = 0;
        up_ld = (int)r3d_rows_pitch(up, lr, &ur);
        if (up_ld < 0 || ur != HN) return R3D_NO(RAD_E_SHAPE);
    }

    long long injw_ld = 0;
    if (r3d_has(injw)) {
        long long ir = 0;
        injw_ld = (int)r3d_rows_pitch(injw, HN, &ir);
        if (injw_ld < 0 || ir != hc) return R3D_NO(RAD_E_SHAPE);
        if (!r3d_contig(injw)) return R3D_NO(RAD_E_STRIDE);
        /* A weight without an output would compute scalars nobody reads; the reference refuses
         * the pair. */
        if (!r3d_has(inj)) return R3D_NO(RAD_E_SHAPE);
    }

    const long long group = r3d_geti_or(a, "group", 0);
    long long q_ld = 0, qs_ld = 0, rq_ld = 0, rs_ld = 0, qg = n, ngrp = 0, rgrp = 0;
    int q_dt = R3D_DT_U8, rq_dt = R3D_DT_U8;
    if (r3d_has(q) || r3d_has(qs)) {
        if (!r3d_has(q) || !r3d_has(qs)) return R3D_NO(RAD_E_INVAL);
        q_dt = r3d_dt_code(q->dtype);
        if (q_dt != R3D_DT_U8 && q_dt != R3D_DT_I8) return R3D_NO(RAD_E_DTYPE);
        if (!r3d_is_f32(qs)) return R3D_NO(RAD_E_DTYPE);
        qg = (group > 0 && group <= n) ? group : n;
        ngrp = (n + qg - 1) / qg;
        long long Mq = 0, Ms = 0;
        q_ld = r3d_rows_pitch(q, n, &Mq);
        qs_ld = r3d_rows_pitch(qs, ngrp, &Ms);
        if (q_ld < 0 || qs_ld < 0 || Mq != M || Ms != M) return R3D_NO(RAD_E_SHAPE);
    }
    if (r3d_has(rq) || r3d_has(rs)) {
        if (!r3d_has(rq) || !r3d_has(rs)) return R3D_NO(RAD_E_INVAL);
        rq_dt = r3d_dt_code(rq->dtype);
        if (rq_dt != R3D_DT_U8 && rq_dt != R3D_DT_I8) return R3D_NO(RAD_E_DTYPE);
        if (!r3d_is_f32(rs)) return R3D_NO(RAD_E_DTYPE);
        if (group <= 0 || group > kHcQuantScratch || (group & (group - 1)) || n % group)
            return R3D_NO(RAD_E_UNSUPPORTED);
        rgrp = n / group;
        long long Mr = 0, Msr = 0;
        rq_ld = r3d_rows_pitch(rq, n, &Mr);
        rs_ld = r3d_rows_pitch(rs, rgrp, &Msr);
        if (rq_ld < 0 || rs_ld < 0 || Mr != M || Msr != M) return R3D_NO(RAD_E_SHAPE);
    }

    const long long need = hc_scratch_bytes(M, lr, hc);
    if (!a->scratch || a->scratch_bytes < need) return R3D_NO(RAD_E_SHAPE);

    const float eps = (float)r3d_getf(a, "eps", 1e-6);
    const float wadd = (float)r3d_getf(a, "wadd", 0.0);
    const int rc = r3d_hc_read((long)(uintptr_t)r3d_p(h), (long)(uintptr_t)r3d_p(w),
                               (long)(uintptr_t)r3d_p(down), (long)(uintptr_t)r3d_p(up),
                               r3d_has(injw) ? (long)(uintptr_t)r3d_p(injw) : 0L,
                               (long)(uintptr_t)r3d_wp(x),
                               r3d_has(inj) ? (long)(uintptr_t)r3d_wp(inj) : 0L,
                               r3d_has(q) ? (long)(uintptr_t)r3d_wp(q) : 0L,
                               r3d_has(qs) ? (long)(uintptr_t)r3d_wp(qs) : 0L,
                               r3d_has(rq) ? (long)(uintptr_t)r3d_wp(rq) : 0L,
                               r3d_has(rs) ? (long)(uintptr_t)r3d_wp(rs) : 0L,
                               (long)(uintptr_t)a->scratch, a->scratch_bytes, (int)M, (int)n,
                               (int)hc, (int)lr, eps, wadd, (int)group, (int)h_ld, (int)x_ld,
                               (int)inj_ld, (int)injw_ld, (int)q_ld, (int)qs_ld, (int)rq_ld,
                               (int)rs_ld, mix_e4m3 ? 1 : 0, q_dt, rq_dt, x_dt, (int)down_ld,
                               (int)down_stride, (int)down_G, (int)up_ld, (int)up_stride,
                               (int)up_G, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

extern "C" int r3d_rad_hc_write(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* y = r3d_opt(a, 0), *inj = r3d_opt(a, 1), *h = r3d_opt(a, 2);
    if (!r3d_p(y) || !r3d_p(inj) || !r3d_wp(h)) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(y) || !r3d_is_bf16(inj) || !r3d_is_bf16(h)) return R3D_NO(RAD_E_DTYPE);
    const long long n = r3d_geti_or(a, "n", 0), hc = r3d_geti_or(a, "hc", 0);
    if (n <= 0 || hc <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0, Mi = 0;
    const long long h_ld = r3d_rows_pitch(h, hc * n, &M);
    const long long y_ld = r3d_rows_pitch(y, n, &My);
    const long long inj_ld = r3d_rows_pitch(inj, hc, &Mi);
    if (h_ld < 0 || y_ld < 0 || inj_ld < 0 || M <= 0) return R3D_NO(RAD_E_STRIDE);
    if (My != M || Mi != M) return R3D_NO(RAD_E_SHAPE);
    const int rc = r3d_hc_write((long)(uintptr_t)r3d_p(y), (long)(uintptr_t)r3d_p(inj),
                                (long)(uintptr_t)r3d_wp(h), (int)M, (int)n, (int)hc, (int)y_ld,
                                (int)inj_ld, (int)h_ld, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

extern "C" int r3d_rad_hc_enter(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 2) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0), *h = r3d_opt(a, 1);
    if (!r3d_p(x) || !r3d_wp(h)) return R3D_NO(RAD_E_INVAL);
    const int x_dt = r3d_dt_code(x->dtype), h_dt = r3d_dt_code(h->dtype);
    if (x_dt < 0 || h_dt != R3D_DT_BF16) return R3D_NO(RAD_E_DTYPE);
    const long long n = r3d_geti_or(a, "n", 0), hc = r3d_geti_or(a, "hc", 0);
    if (n <= 0 || hc <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long h_ld = r3d_rows_pitch(h, hc * n, &M);
    const long long x_ld = r3d_rows_pitch(x, n, &Mx);
    if (h_ld < 0 || x_ld < 0 || M <= 0 || Mx != M) return R3D_NO(RAD_E_SHAPE);
    const int rc = r3d_hc_enter((long)(uintptr_t)r3d_p(x), (long)(uintptr_t)r3d_wp(h), (int)M,
                                (int)n, (int)hc, (int)x_ld, (int)h_ld, x_dt, h_dt, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

extern "C" int r3d_rad_hc_project(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* h = r3d_opt(a, 0), *v = r3d_opt(a, 1), *on = r3d_opt(a, 2);
    if (!r3d_wp(h) || !r3d_p(v)) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(h) || !r3d_is_f32(v)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(on) && !r3d_is_i32(on)) return R3D_NO(RAD_E_DTYPE);
    const long long n = r3d_geti_or(a, "n", 0), hc = r3d_geti_or(a, "hc", 0);
    if (n <= 0 || hc <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long h_ld = r3d_rows_pitch(h, hc * n, &M);
    if (h_ld < 0 || M <= 0) return R3D_NO(RAD_E_STRIDE);
    if (r3d_numel(v) < n) return R3D_NO(RAD_E_SHAPE);
    long long on_ld = 1;
    if (r3d_has(on)) {
        long long Mo = 0;
        on_ld = r3d_rows_pitch(on, 1, &Mo);
        if (on_ld < 0 || Mo != M) return R3D_NO(RAD_E_SHAPE);
    }
    const float scale = (float)r3d_getf(a, "scale", 1.0);
    const int rc = r3d_hc_project((long)(uintptr_t)r3d_wp(h), (long)(uintptr_t)r3d_p(v),
                                  r3d_has(on) ? (long)(uintptr_t)r3d_p(on) : 0L, (int)M, (int)n,
                                  (int)hc, scale, (int)h_ld, (int)on_ld, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}
