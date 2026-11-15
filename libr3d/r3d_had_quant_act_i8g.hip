/* r3d_had_quant_act_i8g.hip -- had_quant_act_i8g: a 128-wide block Hadamard, then per-group int8.
 *
 * The rotated int8 form for the int8 trunk GEMMs (i8a8): the 128-wide Hadamard shrinks the group
 * maximum, and the codes are row-major (the i8a8 dot2 GEMM reads stored order), with one scale per
 * (row, group). This is quant_act_i8g with the rotation; the unrotated form is a separate op.
 *
 * A separate op, not a parameter: it writes different numbers and a different code layout than
 * quant_act_i8 (step-ordered) or had_quant_act_i8 (step-ordered), so a swapped graph would run as
 * a different model. The rotation's 1/128 is the weight's (a `w8a8_fwht128`/IQR plane stores it),
 * so the scale is amax / 127 and folds nothing, as libref and libr4d's i8g quantiser do.
 *
 * One wave a (token, 128-column group): the wave runs the group's whole FWHT with no group loop,
 * which is the form gfx1102 needs (a DPP exchange in a loop body miscompiles there) and lets a
 * 5120- or 17408-wide activation run its 40 or 136 groups across the card.
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

__global__ __launch_bounds__(kThreads) void had_quant_act_i8g_kernel(
    const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n) {
    constexpr int HAD = 128, VPL = HAD / 32;
    static_assert(VPL == 4, "one wave's 32 lanes hold the group's 128 columns as four each");
    const uint32_t row = blockIdx.x, kg = blockIdx.y;
    const uint32_t nkb = n / 128u;
    const uint32_t lane = threadIdx.x & 31;
    const uint16_t* a = x + (size_t)row * n + (size_t)kg * HAD;
    uint8_t* qr = q + (size_t)row * n + (size_t)kg * HAD;

    const int base = lane * VPL;
    float v[VPL];
#pragma unroll
    for (int j = 0; j < VPL; ++j) v[j] = ld_bf(a, base + j);
    r3d_fwht_wave<HAD>(v, lane);
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < VPL; ++j) m = fmaxf(m, fabsf(v[j]));
    m = wave_max_f(m);
    const float sc = m > 0.0f ? m / kTopI8 : 1.0f;
    const float inv = 1.0f / sc;
    if (lane == 0) s[(size_t)row * nkb + kg] = sc;
    *(uint32_t*)(qr + (uint32_t)base) = i8_codes4(v, inv);   // row-major (the i8a8 GEMM's layout)
}

}  // namespace

extern "C" {

int r3d_rad_had_quant_act_i8g(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0), *q = r3d_opt(a, 1), *sc = r3d_opt(a, 2);
    if (!r3d_p(x) || !r3d_wp(q) || !r3d_wp(sc)) return R3D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || q->dtype != RAD_I8 || sc->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(x)) return R3D_NO(RAD_E_STRIDE);

    long long M = 0, K = 0;
    if (r3d_rows_of(x, &M, &K) != 0) return R3D_NO(RAD_E_INVAL);
    if (K <= 0 || K % 128) return R3D_NO(RAD_E_SHAPE);
    const long long grp = r3d_geti_or(a, "group", 128);
    if (grp != 128) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(q) < M * K || r3d_numel(sc) < M * (K / 128)) return R3D_NO(RAD_E_SHAPE);

    dim3 grid((unsigned)M, (unsigned)(K / 128)), blk(kThreads);
    R3D_KLAUNCH(had_quant_act_i8g_kernel, grid, blk, (hipStream_t)s, (const uint16_t*)r3d_p(x),
                (uint8_t*)r3d_wp(q), (float*)r3d_wp(sc), (uint32_t)K);
    return (int)hipGetLastError();
}

}  // extern "C"
