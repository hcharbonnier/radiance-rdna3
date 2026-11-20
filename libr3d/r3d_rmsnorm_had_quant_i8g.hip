/* r3d_rmsnorm_had_quant_i8g.hip -- rmsnorm_had_quant_i8g: the residual add, the RMS norm, a
 * 128-wide block Hadamard and the per-group int8 quantiser in one pass.
 *
 * The fused producer for an int8 trunk's rotated linears (dtype i8a8h): it is `rmsnorm_quant_fp8`'s
 * row with `had_quant_act_i8g`'s quantiser on the norm's pass, so the normalised bf16 row is never
 * written to VRAM and read back. Byte-identical to `rmsnorm` + `had_quant_act_i8g`:
 *
 *   - the residual sum is stored narrowed to bf16 before the variance reads it, and the variance is
 *     taken over the stored value, as `rmsnorm_quant_fp8` takes it;
 *   - the norm output is the bf16-rounded value, and the codes are taken over that value;
 *   - the rotation is the same 128-wide Hadamard as had_quant_act_i8g over four contiguous columns
 *     a lane, so the codes are its bytes. It runs through the wave's shared-memory slice rather
 *     than the DPP exchange of r3d_fwht_wave, because this kernel calls it inside the group loop
 *     and a DPP exchange in a loop body miscompiles on gfx1102 (see r3d_had_quant_act_i8.hip);
 *   - the scale is the same expression: amax / 127 per scale group, the rotation's 1/128 being the
 *     weight's, as had_quant_act_i8g folds nothing. A group of 256 (IQ8R) rotates each 128-wide
 *     half on its own and takes one amax and scale over the pair. `out_bf16` receives the
 *     unrotated bf16 norm output.
 *
 * One workgroup a row, a wave a scale group after the norm reduction, so a 5120-column row runs
 * its 40 groups of 128 (or 20 of 256) across the workgroup's waves.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include <cmath>
#include <cstdint>

namespace {

constexpr uint32_t kThreads = 256;   /* the wide row */
constexpr uint32_t kNarrow = 32;     /* one wave: a row of at most 256 columns */
constexpr float kTopI8 = 127.0f;

typedef const uint16_t __attribute__((address_space(1))) *r3d_gu16d;
typedef const float __attribute__((address_space(1))) *r3d_gf32d;

__device__ __forceinline__ uint16_t ld_bf16(const uint16_t* p, uint32_t i) {
    return (uint16_t)(*(r3d_gu16d)(p + i));
}
__device__ __forceinline__ float gain_of(const void* w, uint32_t i, int wdt) {
    return wdt == R3D_DT_F32 ? *(r3d_gf32d)((const float*)w + i)
                             : bf16_to_f32(ld_bf16((const uint16_t*)w, i));
}

/* The 128-wide Hadamard over a wave's shared slice. Same butterfly network and operand order as
 * r3d_fwht_wave (low keeps v + partner, high keeps partner - v), so the results are bit-identical;
 * only the exchange medium changes (LDS instead of DPP), which is what makes it safe to call from
 * the group loop on every RDNA3 target. */
__device__ __forceinline__ void fwht128_lds(float* v, int lane, float* sh) {
    constexpr int VPL = 4, HAD = 128;
    const int base = lane * VPL;
#pragma unroll
    for (int j = 0; j < VPL; ++j) sh[base + j] = v[j];
    __syncwarp();
#pragma unroll
    for (int h = 1; h < HAD; h <<= 1) {
#pragma unroll
        for (int j = 0; j < VPL; ++j) {
            const int i = base + j, p = i ^ h;
            const float a = sh[i], b = sh[p];
            v[j] = (i & h) ? b - a : a + b;
        }
        __syncwarp();
#pragma unroll
        for (int j = 0; j < VPL; ++j) sh[base + j] = v[j];
        __syncwarp();
    }
}

/* THR is the row's workgroup; WDT the gain's dtype; RES the fused residual add. The rotation is
 * always 128 wide; GRP (128 or 256) is the scale group, so a 256 group is two independent 128-wide
 * rotations under one shared amax and scale, as IQ8R's weight scale expects. It divides n. */
template <int THR, int WDT, bool RES, int GRP>
__global__ __launch_bounds__(THR) void rmsnorm_had_quant_i8g_kernel(
    const uint16_t* __restrict__ x, uint16_t* __restrict__ res, const void* __restrict__ w,
    uint8_t* __restrict__ q, float* __restrict__ s, uint16_t* __restrict__ ob, uint32_t n,
    float eps, float wadd, uint32_t x_ld, uint32_t res_ld, uint32_t ob_ld, uint32_t q_ld,
    uint32_t s_ld) {
    __shared__ float lds[THR / kWave];
    __shared__ float fw[THR / kWave][128];
    const uint32_t row = blockIdx.x;
    const uint16_t* __restrict__ xr = x + (size_t)row * x_ld;
    uint16_t* __restrict__ rr = RES ? res + (size_t)row * res_ld : nullptr;
    uint8_t* __restrict__ qr = q + (size_t)row * q_ld;
    float* __restrict__ sr = s + (size_t)row * s_ld;
    uint16_t* __restrict__ obr = ob ? ob + (size_t)row * ob_ld : nullptr;

    /* The row the norm is taken over: the residual plane, once the sum has been written to it. */
    if (RES) {
        for (uint32_t i = threadIdx.x; i < n; i += THR)
            rr[i] = f32_to_bf16(bf16_to_f32(ld_bf16(xr, i)) + bf16_to_f32(ld_bf16(rr, i)));
    }
    /* Not `__restrict__`: with RES this aliases `rr`, which the same thread has just written. */
    const uint16_t* nr = RES ? rr : xr;

    /* block_sum's barrier publishes pass one's stores to every thread's pass two. */
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += THR) {
        const float v = bf16_to_f32(ld_bf16(nr, i));
        ss += v * v;
    }
    ss = block_sum<THR>(ss, lds);
    const float rs = 1.0f / sqrtf(ss / (float)n + eps);

    const uint32_t wave = threadIdx.x / kWave, lane = threadIdx.x & (kWave - 1);
    constexpr uint32_t kHadW = 128, kPerLane = 4;
    static_assert(GRP == 128 || GRP == 256, "a 256 group is two 128-wide rotations");
    constexpr uint32_t kSub = (uint32_t)(GRP / 128);
    const uint32_t ngrp = n / (uint32_t)GRP;
    for (uint32_t g = wave; g < ngrp; g += (uint32_t)(THR / kWave)) {
        uint32_t base[kSub];
        float v[kSub][kPerLane];
        uint16_t hb[kSub][kPerLane];
#pragma unroll
        for (uint32_t h = 0; h < kSub; ++h) {
            base[h] = g * (uint32_t)GRP + h * kHadW + kPerLane * lane;
#pragma unroll
            for (uint32_t j = 0; j < kPerLane; ++j) {
                const uint32_t i = base[h] + j;
                /* The bf16 round trip is the pair's: rmsnorm stores these bytes and
                 * had_quant_act_i8g reads them back. */
                hb[h][j] = f32_to_bf16(bf16_to_f32(ld_bf16(nr, i)) * rs * (gain_of(w, i, WDT) + wadd));
                v[h][j] = bf16_to_f32(hb[h][j]);
            }
        }
        if (obr) {
#pragma unroll
            for (uint32_t h = 0; h < kSub; ++h) {
                *(uint32_t*)(obr + base[h]) =
                    (uint32_t)hb[h][0] | ((uint32_t)hb[h][1] << 16);
                *(uint32_t*)(obr + base[h] + 2) =
                    (uint32_t)hb[h][2] | ((uint32_t)hb[h][3] << 16);
            }
        }
#pragma unroll
        for (uint32_t h = 0; h < kSub; ++h) fwht128_lds(v[h], (int)lane, fw[wave]);
        float m = 0.0f;
#pragma unroll
        for (uint32_t h = 0; h < kSub; ++h)
#pragma unroll
            for (uint32_t j = 0; j < kPerLane; ++j) m = fmaxf(m, fabsf(v[h][j]));
        m = wave_max(m);
        const float sc = m > 0.0f ? m / kTopI8 : 1.0f;
        if (lane == 0) sr[g] = sc;
        const float inv = 1.0f / sc;
#pragma unroll
        for (uint32_t h = 0; h < kSub; ++h)
            *(uint32_t*)(qr + base[h]) = i8_codes4(v[h], inv);   /* row-major (the i8a8 GEMM's layout) */
    }
}

}  // namespace

extern "C" {

int r3d_rad_rmsnorm_had_quant_i8g(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 5) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0), *res = r3d_opt(a, 1), *w = r3d_opt(a, 2),
                    *q = r3d_opt(a, 3), *sc = r3d_opt(a, 4), *ob = r3d_opt(a, 5);
    if (!r3d_p(x) || !r3d_p(w) || !r3d_wp(q) || !r3d_wp(sc)) return R3D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || q->dtype != RAD_I8 || sc->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);
    int wdt = R3D_DT_N;
    if (r3d_is_bf16(w)) wdt = R3D_DT_BF16;
    else if (r3d_is_f32(w)) wdt = R3D_DT_F32;
    else return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(ob) && ob->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);

    const long long n = r3d_geti_or(a, "n", w->rank ? r3d_dim(w, (int)w->rank - 1) : 0);
    const long long group = r3d_geti_or(a, "group", 0);
    if (n <= 0) return R3D_NO(RAD_E_SHAPE);
    if (group != 128 && group != 256) return R3D_NO(RAD_E_UNSUPPORTED);
    if (n % group) return R3D_NO(RAD_E_UNSUPPORTED);
    const float eps = (float)r3d_getf(a, "eps", 1e-6);
    const float wadd = (float)r3d_getf(a, "wadd", 0.0);
    /* `residual` is optional on its own, as in rmsnorm_quant_fp8: a graph whose stream is carried
     * elsewhere (hyper-connections) binds none and `wadd` still reaches the gain. */
    const bool use_res = r3d_has(res);
    if (use_res && res->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);

    const long long ngrp = n / group;
    long long M = 0, Mres = 0, Mob = 0, Mq = 0, Ms = 0;
    const long long x_ld = r3d_rows_pitch(x, n, &M);
    const long long q_ld = r3d_rows_pitch(q, n, &Mq);
    const long long s_ld = r3d_rows_pitch(sc, ngrp, &Ms);
    if (x_ld < 0 || q_ld < 0 || s_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (!M || Mq < M || Ms < M) return R3D_NO(RAD_E_SHAPE);
    long long res_ld = 0, ob_ld = 0;
    if (use_res) {
        res_ld = r3d_rows_pitch(res, n, &Mres);
        if (res_ld < 0) return R3D_NO(RAD_E_STRIDE);
        if (Mres < M) return R3D_NO(RAD_E_SHAPE);
    }
    if (r3d_has(ob)) {
        ob_ld = r3d_rows_pitch(ob, n, &Mob);
        if (ob_ld < 0) return R3D_NO(RAD_E_STRIDE);
        if (Mob < M) return R3D_NO(RAD_E_SHAPE);
    }
    if (r3d_numel(w) < n) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(q) < M * n || r3d_numel(sc) < M * ngrp) return R3D_NO(RAD_E_SHAPE);
    if (use_res && r3d_numel(res) < M * n) return R3D_NO(RAD_E_SHAPE);
    /* A lane stores four codes as one dword, and four bf16 values as two dwords. */
    if ((q_ld % 4) || ((uintptr_t)r3d_wp(q) & 3u)) return R3D_NO(RAD_E_STRIDE);
    if (r3d_has(ob) && ((ob_ld % 4) || ((uintptr_t)r3d_wp(ob) & 3u))) return R3D_NO(RAD_E_STRIDE);

    const int thr = n <= 256 ? (int)kNarrow : (int)kThreads;
    const uint16_t* xp = (const uint16_t*)r3d_p(x);
    uint16_t* rp = use_res ? (uint16_t*)r3d_wp(res) : nullptr;
    const void* wp = r3d_p(w);
    uint8_t* qp = (uint8_t*)r3d_wp(q);
    float* sp = (float*)r3d_wp(sc);
    uint16_t* op = r3d_has(ob) ? (uint16_t*)r3d_wp(ob) : nullptr;

#define R3D_RHQ(THR, WDT, RES, GRP)                                                              \
    R3D_KLAUNCH(R3D_KNAME(rmsnorm_had_quant_i8g_kernel<THR, WDT, RES, GRP>), dim3((uint32_t)M),  \
                dim3(THR), (hipStream_t)s, xp, rp, wp, qp, sp, op, (uint32_t)n, eps, wadd,       \
                (uint32_t)x_ld, (uint32_t)res_ld, (uint32_t)ob_ld, (uint32_t)q_ld,               \
                (uint32_t)s_ld)
#define R3D_RHQ_GRP(THR, WDT, RES)                                                               \
    do {                                                                                         \
        if (group == 256) R3D_RHQ(THR, WDT, RES, 256);                                           \
        else              R3D_RHQ(THR, WDT, RES, 128);                                           \
    } while (0)
#define R3D_RHQ_W(THR)                                                                           \
    do {                                                                                         \
        if (wdt == R3D_DT_F32) {                                                                 \
            if (use_res) R3D_RHQ_GRP(THR, R3D_DT_F32, true);                                     \
            else         R3D_RHQ_GRP(THR, R3D_DT_F32, false);                                    \
        } else {                                                                                 \
            if (use_res) R3D_RHQ_GRP(THR, R3D_DT_BF16, true);                                    \
            else         R3D_RHQ_GRP(THR, R3D_DT_BF16, false);                                   \
        }                                                                                        \
    } while (0)
    if (thr == (int)kNarrow) R3D_RHQ_W(kNarrow); else R3D_RHQ_W(kThreads);
#undef R3D_RHQ_W
#undef R3D_RHQ_GRP
#undef R3D_RHQ
    return (int)hipGetLastError();
}

}  // extern "C"
