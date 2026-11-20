/* r3d_dflash_conv.hip -- DFlash2's grouped dynamic depthwise convolution, `dflash_conv`.
 *
 * A block-diffusion drafter convolves its hidden stream with a short dynamic kernel whose
 * coefficients are projected per token: at tap j the gain is base[j, h] + delta[t, j, h / group].
 * The blocks are `block` rows starting at row 0, so the shifted tap never reaches into the previous
 * block. This file transcribes libref's ref_dflash_conv:
 *
 *   out[t][h] = sum over j with j == 0 or t % block >= j of
 *               (base[j][h] + delta[t][j][h / group]) * x[t - j][h]
 *
 * in ascending j, each term (base + delta) * x rounded as written in f32, the sum rounded once to
 * out's dtype. The library compiles -ffp-contract=off, so no FMA changes a term's rounding.
 *
 * Parallelism. One block covers 128 threads of eight channels each, and the grid's y index is the
 * row: a thread reads one 16-byte chunk of x (and of x's predecessor row), the two base chunks and
 * two delta scalars, and writes one 16-byte chunk of out. The tap mask is a scalar branch, since t
 * is uniform across the block.
 *
 * Cost. At the 27B's draft height (T = 4, H = 5120, block 4) the grid is 5 x 4 blocks: launch
 * latency, not arithmetic, and the row's reason to exist is that without it the drafter's whole
 * weight group is pinned to the container's mapping (the plan cannot copy a widened weight into a
 * pool it never chose). At a context pass (T up to a prefill chunk) the kernel is bandwidth-bound
 * on x: 3 reads and 1 write of 16 bytes per 8 channels.
 *
 * Agreement with the reference: the delta plane is a slice of the [T, 2, taps, NG] coefficient
 * projection and is read at its own row pitch (2*taps*NG); a packed pitch would silently read the
 * other half's coefficients, so the shim refuses any other pitch.
 *
 * The shim refuses, none of it reachable from a row whose constraints hold: a dtype other than bf16
 * on any operand, a padded x/out/base row, `out` aliasing `x` (tap j reads row t - j), a delta pitch
 * that is not 2*taps*NG, and a geometry the body's lane map cannot serve.
 */
#include <hip/hip_runtime.h>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3dh_dflash_conv_body.h"

#include <rad_types.h>

#define DFC_THREADS 128
#define DFC_VEC     8      /* bf16 lanes per thread: one dwordx4 in and one out */

/* The band's `T` is a ceiling on the rows of a step, never the row count: the extents come from
 * the tensors (R24). */
__global__ __launch_bounds__(DFC_THREADS) void dflash_conv_kernel(
    const r3dhc_u16* __restrict__ x, const r3dhc_u16* __restrict__ delta,
    const r3dhc_u16* __restrict__ base, r3dhc_u16* __restrict__ out,
    int H, int dpitch, int NG, int vpr, int block) {
    const int hv = blockIdx.x * DFC_THREADS + threadIdx.x;
    r3dhc_dflash_conv_body<2, 16>(x, delta, base, out, H, dpitch, NG, vpr, block, blockIdx.y, hv);
}

/* x     [T, H]        bf16, row pitch H
 * delta [T, taps, NG] bf16, row pitch dpitch = 2*taps*NG (a slice of the [T, 2, taps, NG]
 *                     projection: the packed pitch would read the other half's coefficients)
 * base  [taps, H]     bf16, contiguous
 * out   [T, H]        bf16. Must not alias x: tap 1 of row t reads input row t-1. */
static int dflash_conv_launch(const RadTensor* x, const RadTensor* d, const RadTensor* bs,
                             const RadTensor* o, int NG, int bsz, int dpitch, void* stream) {
    const long long T = x->shape[0], H = x->shape[1];
    if (T <= 0) return 0;

    const int vpr = (int)(H / DFC_VEC);
    dim3 grid((vpr + DFC_THREADS - 1) / DFC_THREADS, (unsigned)T), blk(DFC_THREADS);
    R3D_KLAUNCH(R3D_KNAME(dflash_conv_kernel), grid, blk, (hipStream_t)(uintptr_t)stream,
                (const r3dhc_u16*)x->data, (const r3dhc_u16*)d->data, (const r3dhc_u16*)bs->data,
                (r3dhc_u16*)o->data, (int)H, dpitch, NG, vpr, bsz);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ================================================================== shim
 *
 * libref's operand list is {x, delta, base, out} and its parameter list is {T, hidden_size, taps,
 * group_size, NG, block_size, dtype}. `x` and `out` are [T, H] and `base` the [taps, H] weight;
 * `delta` may be passed as the [T, taps, NG] slice (rank 3) or the flat [T, taps*NG] form (rank 2),
 * and in both cases at the projection's row pitch.
 */
extern "C" int r3d_rad_dflash_conv(const RadArgs* a, RadStream s) {
    enum { DF_X = 0, DF_DELTA, DF_BASE, DF_OUT };
    const RadTensor* x  = r3d_opt(a, DF_X);
    const RadTensor* d  = r3d_opt(a, DF_DELTA);
    const RadTensor* bs = r3d_opt(a, DF_BASE);
    const RadTensor* o  = r3d_opt(a, DF_OUT);
    if (!x || !d || !bs || !o) return R3D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || d->dtype != RAD_BF16 || bs->dtype != RAD_BF16 ||
        o->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (x->rank != 2 || o->rank != 2 || d->rank < 2) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(x) || !r3d_contig(o) || !r3d_contig(bs)) return R3D_NO(RAD_E_STRIDE);
    if (x->data == o->data) return R3D_NO(RAD_E_INVAL);

    const long long T = x->shape[0];
    const long long H = x->shape[1];
    if (T <= 0 || o->shape[0] != T || o->shape[1] != H) return R3D_NO(RAD_E_SHAPE);

    const int taps  = (int)r3d_geti_or(a, "taps", 2);
    const int group = (int)r3d_geti_or(a, "group_size", 16);
    const int NG    = (int)r3d_geti_or(a, "NG", 0);
    const int bsz   = (int)r3d_geti_or(a, "block_size", 0);
    if (taps != 2 || group != 16) return R3D_NO(RAD_E_UNSUPPORTED);
    if (NG <= 0 || bsz < 1) return R3D_NO(RAD_E_INVAL);
    if (H % (group * DFC_VEC) || NG != H / group) return R3D_NO(RAD_E_SHAPE);
    if (rad_tensor_numel(bs) < (long long)taps * H) return R3D_NO(RAD_E_SHAPE);

    /* delta is a slice of the [T, 2, taps, NG] projection: its row pitch is 2*taps*NG, not the
     * slice's own width. Nothing bounds the gap from the view's own extent -- the view lives inside
     * the full plane -- so the pitch is the whole check. */
    const long long dpitch = r3d_row_pitch(d);
    if (dpitch != 2LL * taps * NG) return R3D_NO(RAD_E_STRIDE);

    /* One dwordx4 per thread: every row start is a multiple of 8 channels because H is. */
    if (!r3d_aligned(x->data, 16) || !r3d_aligned(o->data, 16) || !r3d_aligned(bs->data, 16))
        return R3D_NO(RAD_E_STRIDE);

    return dflash_conv_launch(x, d, bs, o, NG, bsz, (int)dpitch, s);
}
