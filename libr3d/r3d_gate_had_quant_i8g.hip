/* r3d_gate_had_quant_i8g.hip -- gate_had_quant_i8g: act(gate) * x, a 128-wide block Hadamard,
 * then per-group int8 (row-major), in one pass.
 *
 * The gate arrives in its own operand rather than as the first half of a fused gate_up plane, as
 * gated_had_quant_i8g reads it: a gated attention's [q|gate] projection leaves the gate a column
 * slice and the attention output a separate buffer with its own pitch. mode 0 is silu (a gated
 * feed-forward), mode 1 sigmoid (a gated attention o_proj). The product is rounded to bf16 as the
 * unfused pair stores it, the 128-wide Hadamard shrinks the group maximum, and the codes are
 * row-major int8 with one scale per (row, group). The rotation's 1/128 is the weight's, so the
 * scale is amax / 127 and folds nothing. A group of 256 (IQ8R) is two independent 128-wide
 * rotations under one amax and one scale.
 *
 * One wave a (token, scale group): the wave runs the group's whole FWHT with no group loop, which
 * is the form gfx1102 needs (a DPP exchange in a loop body miscompiles there).
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_fwht.h"
#include "r3d_launch.h"
#include <cmath>
#include <cstdio>

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

/* SUB is the number of 128-wide rotation blocks one scale covers: 1 (group 128), or 2 for IQ8R's
 * 256-wide group, where each half is rotated on its own and the amax is taken over the pair.
 *
 * The op is issued on the whole projection row, not on one head: with a gated attention the
 * operands are [M, heads*head_dim] while the `n` parameter names the head width, and the gate is a
 * rank-3 view [M, heads, head_dim] whose head stride is wider than head_dim (a [q|gate] projection
 * leaves it as a column slice of each head's block). Every rotation block lies inside one head
 * because head_dim is a multiple of 128, so a block's gate base is head*gate_hs + in_head. */
template <int MODE, int SUB>
__global__ __launch_bounds__(kThreads) void gate_had_quant_i8g_kernel(
    const uint16_t* __restrict__ gate, const uint16_t* __restrict__ x, uint16_t* __restrict__ yo,
    uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n, uint32_t q_ld, uint32_t gate_ld,
    uint32_t gate_hs, uint32_t head_dim, uint32_t x_ld, uint32_t y_ld) {
    constexpr int HAD = 128, VPL = HAD / 32;
    static_assert(SUB == 1 || SUB == 2, "a 256 group is two 128-wide rotations");
    const uint32_t row = blockIdx.x, kg = blockIdx.y;
    const uint32_t nkb = n / (uint32_t)(SUB * HAD);
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t c0 = kg * (uint32_t)(SUB * HAD);
    const uint32_t head = c0 / head_dim;
    const uint16_t* g = gate + (size_t)row * gate_ld + (size_t)head * gate_hs + (c0 - head * head_dim);
    const uint16_t* xr = x + (size_t)row * x_ld + c0;
    uint8_t* qr = q + (size_t)row * q_ld + c0;
    uint16_t* yop = yo ? yo + (size_t)row * y_ld + c0 : nullptr;

    const int base = lane * VPL;
    float v[SUB][VPL];
    float m = 0.0f;
#pragma unroll
    for (int h = 0; h < SUB; ++h) {
        const uint16_t* gh = g + h * HAD;
        const uint16_t* xh = xr + h * HAD;
        uint16_t* yh = yop ? yop + h * HAD : nullptr;
#pragma unroll
        for (int j = 0; j < VPL; ++j) {
            const float xg = ld_bf(gh, base + j);
            const float sg = 1.0f / (1.0f + expf(-xg));   /* full exp: the reference's sigmoid */
            /* The unfused pair stores the gated product as bf16 and the quantiser reads it back. */
            const uint16_t hb =
                f32_to_bf16((MODE == 0 ? xg * sg : sg) * ld_bf(xh, base + j));
            if (yh) yh[base + j] = hb;
            v[h][j] = bf16_to_f32(hb);
        }
        r3d_fwht_wave<HAD>(v[h], lane);
#pragma unroll
        for (int j = 0; j < VPL; ++j) m = fmaxf(m, fabsf(v[h][j]));
    }
    m = wave_max_f(m);
    const float sc = m > 0.0f ? m / kTopI8 : 1.0f;
    const float inv = 1.0f / sc;
    if (lane == 0) s[(size_t)row * nkb + kg] = sc;
    static_assert(VPL == 4, "one wave's 32 lanes hold the group's 128 columns as four each");
#pragma unroll
    for (int h = 0; h < SUB; ++h)
        *(uint32_t*)(qr + h * HAD + (uint32_t)base) = i8_codes4(v[h], inv);
}

}  // namespace

extern "C" {

int r3d_rad_gate_had_quant_i8g(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 4) { std::fprintf(stderr, "gate_had: n_t %d\n", a ? (int)a->n_t : -1); return R3D_NO(RAD_E_INVAL); }
    /* The schema's order: gate, x, q, scale, out_bf16. */
    const RadTensor* gate = r3d_opt(a, 0), *x = r3d_opt(a, 1), *q = r3d_opt(a, 2),
                    *sc = r3d_opt(a, 3), *y = r3d_opt(a, 4);
    if (!r3d_p(gate) || !r3d_p(x) || !r3d_wp(q) || !r3d_wp(sc)) return R3D_NO(RAD_E_INVAL);
    if (gate->dtype != RAD_BF16 || x->dtype != RAD_BF16 || q->dtype != RAD_I8 ||
        sc->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);

    /* The row is issued on the whole projection: q and x are [M, n] with `n` the full width, while
     * the `n` parameter names the head width (heads * head_dim = n). */
    long long M = 0, n = 0;
    if (r3d_rows_of(q, &M, &n) != 0) return R3D_NO(RAD_E_INVAL);
    if (n <= 0 || n % 128) return R3D_NO(RAD_E_SHAPE);
    const long long grp = r3d_geti_or(a, "group", 128);
    if (grp != 128 && grp != 256) return R3D_NO(RAD_E_SHAPE);
    if (n % grp) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(gate) < M * n || r3d_numel(x) < M * n) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(sc) < M * (n / grp)) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(q)) return R3D_NO(RAD_E_STRIDE);

    /* The gate is [M, n] contiguous, or [M, heads, head_dim] with its own head and row pitches: a
     * [q|gate] projection leaves it a column slice of each head's block. head_dim tiles 128, so a
     * rotation block never crosses a head. */
    uint32_t head_dim, gate_hs;
    if (gate->rank >= 3) {
        if (gate->rank > 3 || gate->shape[2] <= 0 || gate->shape[2] % 128 ||
            gate->stride[2] != 1 || gate->stride[1] < gate->shape[2] ||
            gate->shape[1] * gate->shape[2] != n)
            return R3D_NO(RAD_E_SHAPE);
        head_dim = (uint32_t)gate->shape[2];
        gate_hs = (uint32_t)gate->stride[1];
    } else {
        if (gate->rank != 2 || gate->stride[1] != 1) return R3D_NO(RAD_E_STRIDE);
        head_dim = (uint32_t)n;
        gate_hs = (uint32_t)n;
    }
    if (grp > head_dim) return R3D_NO(RAD_E_SHAPE);
    if (x->rank < 2 || x->stride[1] != 1 || x->stride[0] < n) return R3D_NO(RAD_E_STRIDE);
    const uint32_t gate_ld = (uint32_t)gate->stride[0], x_ld = (uint32_t)x->stride[0];
    const uint32_t q_ld = (uint32_t)q->stride[0];
    uint32_t y_ld = (uint32_t)n;
    if (y) {
        if (y->dtype != RAD_BF16 || r3d_numel(y) < M * n || y->rank < 2 || y->stride[1] != 1)
            return R3D_NO(RAD_E_SHAPE);
        y_ld = (uint32_t)y->stride[0];
    }

    const char* act = r3d_gets(a, "act");
    long long mode = 0;
    if (act && std::strcmp(act, "sigmoid") == 0) mode = 1;
    else if (act && act[0] && std::strcmp(act, "silu") != 0) return R3D_NO(RAD_E_UNSUPPORTED);
    /* A lane stores four codes as one dword. */
    if (((uintptr_t)r3d_wp(q) & 3u) || ((uintptr_t)r3d_wp(sc) & 3u)) return R3D_NO(RAD_E_ALIGN);

    dim3 grid((unsigned)M, (unsigned)(n / grp)), blk(kThreads);
    const int sub = grp == 256 ? 2 : 1;
    const uint16_t* gp = (const uint16_t*)r3d_p(gate);
    const uint16_t* xp = (const uint16_t*)r3d_p(x);
    uint16_t* yp = (uint16_t*)(y ? r3d_wp(y) : nullptr);
    uint8_t* qp = (uint8_t*)r3d_wp(q);
    float* sp = (float*)r3d_wp(sc);
    if (mode == 0) {
        if (sub == 2)
            R3D_KLAUNCH(R3D_KNAME(gate_had_quant_i8g_kernel<0, 2>), grid, blk, (hipStream_t)s,
                        gp, xp, yp, qp, sp, (uint32_t)n, q_ld, gate_ld, gate_hs, head_dim, x_ld, y_ld);
        else
            R3D_KLAUNCH(R3D_KNAME(gate_had_quant_i8g_kernel<0, 1>), grid, blk, (hipStream_t)s,
                        gp, xp, yp, qp, sp, (uint32_t)n, q_ld, gate_ld, gate_hs, head_dim, x_ld, y_ld);
    } else {
        if (sub == 2)
            R3D_KLAUNCH(R3D_KNAME(gate_had_quant_i8g_kernel<1, 2>), grid, blk, (hipStream_t)s,
                        gp, xp, yp, qp, sp, (uint32_t)n, q_ld, gate_ld, gate_hs, head_dim, x_ld, y_ld);
        else
            R3D_KLAUNCH(R3D_KNAME(gate_had_quant_i8g_kernel<1, 1>), grid, blk, (hipStream_t)s,
                        gp, xp, yp, qp, sp, (uint32_t)n, q_ld, gate_ld, gate_hs, head_dim, x_ld, y_ld);
    }
    return (int)hipGetLastError();
}

}  // extern "C"
