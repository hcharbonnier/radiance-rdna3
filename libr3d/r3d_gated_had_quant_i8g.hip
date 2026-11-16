/* r3d_gated_had_quant_i8g.hip -- gated_had_quant_i8g: a gated product, a 128-wide block Hadamard,
 * then per-group int8 (row-major), in one pass.
 *
 * mode 0 (silu) is down_proj's input, mode 1 (sigmoid) a gated attention o_proj's. The gate and up
 * arrive as two halves of a fused gate_up of row pitch 2*n. The gated product is rounded to bf16 as
 * the unfused pair stores it, the 128-wide Hadamard shrinks the group maximum, and the codes are
 * row-major int8 (the i8a8 dot2 GEMM reads stored order), with one scale per (row, group). The
 * rotation's 1/128 is the weight's, so the scale is amax / 127 and folds nothing.
 *
 * One wave a (token, 128-column group): the wave runs the group's whole FWHT with no group loop,
 * which is the form gfx1102 needs (a DPP exchange in a loop body miscompiles there) and lets a
 * 17 408-wide feed-forward run 136 groups across the card.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_fwht.h"
#include <cmath>

namespace {

constexpr uint32_t kThreads = 32;
constexpr float kTopI8 = 127.0f;

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

template <int MODE>
__global__ __launch_bounds__(kThreads) void gated_had_quant_i8g_kernel(
    const uint16_t* __restrict__ gu, uint16_t* __restrict__ yo, uint8_t* __restrict__ q,
    float* __restrict__ s, uint32_t n, uint32_t gu_ld, uint32_t y_ld) {
    constexpr int HAD = 128, VPL = HAD / 32;
    const uint32_t row = blockIdx.x, kg = blockIdx.y;
    const uint32_t nkb = n / 128u;
    const uint32_t lane = threadIdx.x & 31;
    const uint16_t* g = gu + (size_t)row * gu_ld + (size_t)kg * HAD;
    const uint16_t* u = g + n;
    uint8_t* qr = q + (size_t)row * n + (size_t)kg * HAD;
    uint16_t* yop = yo ? yo + (size_t)row * y_ld + (size_t)kg * HAD : nullptr;

    const int base = lane * VPL;
    float v[VPL];
#pragma unroll
    for (int j = 0; j < VPL; ++j) {
        const float xg = ld_bf(g, base + j);
        const float sg = 1.0f / (1.0f + expf(-xg));   /* full exp: the reference's sigmoid */
        /* The unfused pair stores the gated product as bf16 and the quantiser reads it back. */
        const uint16_t h = f32_to_bf16((MODE == 0 ? xg * sg : sg) * ld_bf(u, base + j));
        if (yop) yop[base + j] = h;
        v[j] = bf16_to_f32(h);
    }
    r3d_fwht_wave<HAD>(v, lane);
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < VPL; ++j) m = fmaxf(m, fabsf(v[j]));
    m = wave_max_f(m);
    const float sc = m > 0.0f ? m / kTopI8 : 1.0f;
    const float inv = 1.0f / sc;
    if (lane == 0) s[(size_t)row * nkb + kg] = sc;
    /* The codes in row-major order: each lane's VPL values are columns base .. base+VPL-1, so lane
     * l writes its own four (VPL = 4) bytes at l*4, and 32 lanes cover the group. */
    static_assert(VPL == 4, "one wave's 32 lanes hold the group's 128 columns as four each");
    *(uint32_t*)(qr + (uint32_t)base) = i8_codes4(v, inv);
}

}  // namespace

extern "C" {

int r3d_rad_gated_had_quant_i8g(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    /* The schema's order: gate_up, q, scale, out_bf16. */
    const RadTensor* gu = r3d_opt(a, 0), *q = r3d_opt(a, 1), *sc = r3d_opt(a, 2),
                    *y = r3d_opt(a, 3);
    if (!r3d_p(gu) || !r3d_wp(q) || !r3d_wp(sc)) return R3D_NO(RAD_E_INVAL);
    if (gu->dtype != RAD_BF16 || q->dtype != RAD_I8 || sc->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);

    long long M = 0, n = 0;
    if (r3d_rows_of(q, &M, &n) != 0) return R3D_NO(RAD_E_INVAL);
    if (n <= 0 || n % 128) return R3D_NO(RAD_E_SHAPE);
    const long long grp = r3d_geti_or(a, "group", 128);
    if (grp != 128) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(gu) < M * 2 * n) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(sc) < M * (n / 128)) return R3D_NO(RAD_E_SHAPE);
    uint32_t y_ld = (uint32_t)n;
    if (y) {
        if (y->dtype != RAD_BF16 || r3d_numel(y) < M * n) return R3D_NO(RAD_E_SHAPE);
        if (y->rank >= 2 && y->shape[1] > n) y_ld = (uint32_t)y->shape[1];
    }

    const char* act = r3d_gets(a, "act");
    const long long mode = act && 0 == std::strcmp(act, "sigmoid") ? 1 : 0;
    const uint32_t gu_ld = (uint32_t)(gu->rank >= 2 ? gu->shape[1] : gu->stride[0]);
    dim3 grid((unsigned)M, (unsigned)(n / 128)), blk(kThreads);
    if (mode == 0)
        R3D_KLAUNCH(R3D_KNAME(gated_had_quant_i8g_kernel<0>), grid, blk, (hipStream_t)s,
                    (const uint16_t*)r3d_p(gu), (uint16_t*)(y ? r3d_wp(y) : nullptr),
                    (uint8_t*)r3d_wp(q), (float*)r3d_wp(sc), (uint32_t)n, gu_ld, y_ld);
    else
        R3D_KLAUNCH(R3D_KNAME(gated_had_quant_i8g_kernel<1>), grid, blk, (hipStream_t)s,
                    (const uint16_t*)r3d_p(gu), (uint16_t*)(y ? r3d_wp(y) : nullptr),
                    (uint8_t*)r3d_wp(q), (float*)r3d_wp(sc), (uint32_t)n, gu_ld, y_ld);
    return (int)hipGetLastError();
}

}  // extern "C"
