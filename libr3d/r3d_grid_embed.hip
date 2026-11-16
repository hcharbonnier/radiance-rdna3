/* r3d_grid_embed.hip -- grid_embed, the ViT's learned position table.
 *
 *   x[m] += pe(m),  pe(m) = table resampled bilinearly at row m's grid coordinate,
 *   with the grid's corners on the table's (align_corners) and edge taps clamped.
 *
 * The vision tower (arch/common/rad_block_vit.h) adds the 48 x 48 learned table to the patch
 * embedding before the first block, resampled to each image's patch grid. `coord` is [4, M] i32 --
 * (row, column, grid height, grid width) -- and the table is `side x side` rows of n.
 *
 * ---- the arithmetic -------------------------------------------------------------------------
 *
 * This is libref's ref_grid_embed transcribed. Per axis:
 *
 *     src = (float)idx * (float)(side - 1) / (float)(size - 1 > 1 ? size - 1 : 1)
 *     fl  = floor(src);  tap k = clamp((int)fl + k, 0, side - 1)
 *     w[k] = max(1 - |src - fl - k|, 0)
 *
 * The four taps are then summed in f32 in the reference's order -- (h0,w0), (h0,w1), (h1,w0),
 * (h1,w1) -- the sum rounded once to x's dtype, and that value added to x and rounded again. A
 * block owns one row, so the coordinate and its four weights are computed once and shared; each
 * thread sweeps the row's columns.
 *
 * x and the table are read at their row pitches and their last axis must be contiguous, which is
 * what the core's buffers are; every access is a halfword, so no alignment is required. The
 * coordinate's own strides are honoured, as libref's are.
 */

#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <cmath>
#include <cstring>

namespace {

constexpr int kThr = 256;

/* One axis of the resample, exactly ref_grid_embed's lambda. */
__device__ __forceinline__ void ge_axis(int idx, int size, int side, int t[2], float w[2]) {
    const int den = size - 1 > 1 ? size - 1 : 1;
    const float src = (float)idx * (float)(side - 1) / (float)den;
    const float fl = std::floor(src);
    const int tap0 = (int)fl;
    for (int k = 0; k < 2; ++k) {
        int tap = tap0 + k;
        tap = tap < 0 ? 0 : (tap > side - 1 ? side - 1 : tap);
        t[k] = tap;
        const float d = std::fabs(src - fl - (float)k);
        const float one = 1.0f - d;
        w[k] = one > 0.0f ? one : 0.0f;
    }
}

/* One block a row: thread 0 reads the coordinate and forms the taps, every thread then walks the
 * row's n columns. */
__global__ __launch_bounds__(kThr) void grid_embed_kernel(
    uint16_t* __restrict__ x, const uint16_t* __restrict__ table,
    const int32_t* __restrict__ coord, int M, int n, int side, int x_ld, int t_ld, int ccs,
    int cms) {
    __shared__ int th[2], tw[2];
    __shared__ float wh[2], ww[2];

    const int r = (int)blockIdx.x;
    if (r >= M) return;
    if (threadIdx.x == 0) {
        const int gr = coord[0 * ccs + (long)r * cms];
        const int gc = coord[1 * ccs + (long)r * cms];
        const int gh = coord[2 * ccs + (long)r * cms];
        const int gw = coord[3 * ccs + (long)r * cms];
        ge_axis(gr, gh, side, th, wh);
        ge_axis(gc, gw, side, tw, ww);
    }
    group_barrier();

    const float w00 = wh[0] * ww[0], w01 = wh[0] * ww[1];
    const float w10 = wh[1] * ww[0], w11 = wh[1] * ww[1];
    const uint16_t* r00 = table + (size_t)(th[0] * side + tw[0]) * t_ld;
    const uint16_t* r01 = table + (size_t)(th[0] * side + tw[1]) * t_ld;
    const uint16_t* r10 = table + (size_t)(th[1] * side + tw[0]) * t_ld;
    const uint16_t* r11 = table + (size_t)(th[1] * side + tw[1]) * t_ld;
    uint16_t* xr = x + (size_t)r * x_ld;

    for (int i = (int)threadIdx.x; i < n; i += kThr) {
        float acc = 0.0f;
        acc = acc + bf16_to_f32((uint16_t)ldg_u16(r00 + i)) * w00;
        acc = acc + bf16_to_f32((uint16_t)ldg_u16(r01 + i)) * w01;
        acc = acc + bf16_to_f32((uint16_t)ldg_u16(r10 + i)) * w10;
        acc = acc + bf16_to_f32((uint16_t)ldg_u16(r11 + i)) * w11;
        const float pe = bf16_to_f32(f32_to_bf16(acc));
        xr[i] = f32_to_bf16(bf16_to_f32((uint16_t)ldg_u16(xr + i)) + pe);
    }
}

}  // namespace

/* The launcher. Returns 0, -1 a null operand, -2 a shape/geometry the kernel cannot carry, -3 a
 * pitch, -4 the HIP launch failed. */
extern "C" int r3d_grid_embed_bf16(long x, long table, long coord, int M, int n, int side,
                                   int x_ld, int t_ld, int ccs, int cms, long stream) {
    if (!x || !table || !coord) return -1;
    if (M < 1 || n < 1 || side < 1) return -2;
    if (x_ld < n || t_ld < n || ccs < 0 || cms < 1) return -3;
    R3D_KLAUNCH(grid_embed_kernel, dim3((uint32_t)M), dim3(kThr),
                (hipStream_t)(uintptr_t)stream, (uint16_t*)(uintptr_t)x,
                (const uint16_t*)(uintptr_t)table, (const int32_t*)(uintptr_t)coord, M, n, side,
                x_ld, t_ld, ccs, cms);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ============================================================== the plugin ABI entry point
 *
 * `grid_embed`'s operands are x (inout), table (weight), coord (input); parameters M, n, side,
 * dtype. This row's band is the vision tower's bf16. */
enum { G_X = 0, G_TAB, G_CRD };

#define R3D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R3D_NO(_s); } while (0)

extern "C" int r3d_rad_grid_embed(const RadArgs* a, RadStream s) {
    const RadTensor* X = r3d_opt(a, G_X);
    const RadTensor* T = r3d_opt(a, G_TAB);
    const RadTensor* C = r3d_opt(a, G_CRD);
    if (!X || !T || !C) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(r3d_is_bf16(X) ? RAD_OK : RAD_E_DTYPE);
    R3D_NEED(r3d_is_bf16(T) ? RAD_OK : RAD_E_DTYPE);
    R3D_NEED(r3d_is_i32(C) ? RAD_OK : RAD_E_DTYPE);

    if (const char* nm = r3d_gets(a, "dtype"))
        if (std::strcmp(nm, "bf16")) return R3D_NO(RAD_E_UNSUPPORTED);

    const long long n = r3d_geti_or(a, "n", r3d_dim(T, (int)T->rank - 1));
    const long long side = r3d_geti_or(a, "side", 0);
    if (n < 1 || side < 1) return R3D_NO(RAD_E_SHAPE);
    /* The last axis is the row's width; a parameter that disagrees with it would make the pitch
     * describe a different plane. */
    if (X->rank >= 2 && r3d_dim(X, (int)X->rank - 1) != n) return R3D_NO(RAD_E_SHAPE);
    if (T->rank >= 2 && r3d_dim(T, (int)T->rank - 1) != n) return R3D_NO(RAD_E_SHAPE);

    long long xrows = 0, trows = 0;
    const long long x_ld = r3d_rows_pitch(X, n, &xrows);
    const long long t_ld = r3d_rows_pitch(T, n, &trows);
    if (x_ld < 0 || t_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (trows < side * side) return R3D_NO(RAD_E_SHAPE);
    /* [4, M] component-major, read at its own strides; a view of a wider plane is read in place. */
    if (C->rank < 2 || r3d_dim(C, 0) < 4 || r3d_dim(C, 1) < xrows) return R3D_NO(RAD_E_SHAPE);
    const long long ccs = r3d_stride(C, 0), cms = r3d_stride(C, 1);
    if (ccs < 0 || cms < 1 || ccs > 0x7fffffffLL || cms > 0x7fffffffLL)
        return R3D_NO(RAD_E_STRIDE);
    if (x_ld > 0x7fffffffLL || t_ld > 0x7fffffffLL || xrows > 0x7fffffffLL ||
        n > 0x7fffffffLL || side > 0x7fffffffLL)
        return R3D_NO(RAD_E_SHAPE);

    const int e = r3d_grid_embed_bf16(r3d_addr(X), r3d_addr(T), r3d_addr(C), (int)xrows, (int)n,
                                      (int)side, (int)x_ld, (int)t_ld, (int)ccs, (int)cms,
                                      (long)(uintptr_t)s);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_INVAL : e == -2 ? RAD_E_SHAPE
                        : e == -3 ? RAD_E_STRIDE : RAD_E_DEVICE);
}
