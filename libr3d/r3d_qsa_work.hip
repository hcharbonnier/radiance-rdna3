/* r3d_qsa_work.hip -- qsa_work: the gather that turns a batch into the flat block list
 * qsa_block_key reads (docs/OPS.md, "qsa_work").
 *
 * The blocks that complete in this step are found from `cu` and `pos`; each one's `ratio` raw
 * index keys are copied into consecutive rows of `stage`, and its destination page and first-token
 * position are written beside it. A block completing at decode has its keys partly in this step's
 * `k` and partly in the per-sequence tail (`state`, the linear KV state qsa_tail_store fills), so
 * the block-key kernel never learns that there were two sources.
 *
 * `state` is [slots, 1, ring, n] and `sidx[s][0]` names the sequence's slot; a sequence with no
 * slot gets page -1 and no gather. The tail row a key at position q reads is `q % ring`, the same
 * law qsa_tail_store writes with. The ring is ratio + n_spec, not ratio: see r3d_qsa_tail_store.hip.
 *
 * The launch is bounded on the host: one workgroup per potential block, i.e. one per `ratio`
 * tokens plus at most one straddling block per sequence (`work`). The real count is device data;
 * slots past it get page -1, which qsa_block_key skips, so the bound costs empty workgroups and
 * never a wrong answer.
 *
 * `nc` -- each sequence's complete-block count, (last position + 1) / ratio -- is computed here on
 * the way, by the first `seqs` workgroups, which saves the scorer and the selection a launch.
 *
 *   k     [T][n]                  bf16, this step's raw index keys (a column slice of the
 *                                 projection, read at its row pitch)
 *   pos   [T]                     i32
 *   cu    [seqs + 1]              i32
 *   state [slots, 1, ring, n]     bf16, the tail group's cache for this layer
 *   sidx  [seqs][sidx_ld]         i32, column 0 is the sequence's slot (-1: none)
 *   bt    [seqs][bt_ld]           i32, the block-key group's block table (page ids)
 *   rope  [3][T]                  i32, optional: this step's rotary components (M-RoPE)
 *   stage [work * ratio][n]       bf16
 *   page  [work]                  i32, -1 past the real count or with no tail slot
 *   bpos  [work][bstep]           i32, first-token position (or its three components under rope)
 *   nc    [seqs]                  i32, complete blocks per sequence
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_launch.h"

namespace {

constexpr int kThreads = 128;
/* The sequence scan's bound, checked by the launcher against the real count from cu_seqlens. It is
 * the same ceiling libr4d's launcher enforces, so both libraries serve the same models. */
constexpr int kMaxSeq = 256;

__global__ __launch_bounds__(kThreads) void qsa_work_kernel(
    const uint16_t* __restrict__ k, const int32_t* __restrict__ pos,
    const int32_t* __restrict__ cu, const uint16_t* __restrict__ state,
    const int32_t* __restrict__ sidx, const int32_t* __restrict__ bt,
    uint16_t* __restrict__ stage, int32_t* __restrict__ page, int32_t* __restrict__ bpos,
    int32_t* __restrict__ nc, int T, int n, int ratio, int ring, int seqs, int max_work,
    int slots, int k_ld, int s_slot, int s_row, int sidx_ld, int bt_ld, int bt_cols, int g_ld,
    const int32_t* __restrict__ rope, int rplane, int bstep)
{
    const int w = (int)blockIdx.x;

    /* Complete blocks per sequence, written by the first `seqs` workgroups. `cu` is validated by
     * the launcher; the range test here only keeps a malformed batch from reading outside pos. */
    if (w < seqs && threadIdx.x == 0) {
        const int t0 = cu[w], t1 = cu[w + 1];
        nc[w] = (t1 > t0 && t0 >= 0 && t1 <= T) ? (pos[t1 - 1] + 1) / ratio : 0;
    }
    if (w >= max_work) return;

    /* The sequence owning work item w, by a running count recomputed in every workgroup (seqs is
     * small; a prefix would need a second launch). A block completes when its last token,
     * b * ratio + ratio - 1, is in [p0, p1]. Both bounds use non-negative numerators: C division
     * truncates toward zero, so (p1 - (ratio - 1)) / ratio would be 0, not -1, for
     * p1 < ratio - 1. */
    int s = -1, b = 0, run = 0;
    for (int i = 0; i < seqs && i < kMaxSeq; ++i) {
        const int t0 = cu[i], t1 = cu[i + 1];
        if (t1 <= t0 || t0 < 0 || t1 > T) continue;
        const int p0 = pos[t0], p1 = pos[t1 - 1];
        if (p0 < 0) continue;
        const int lo = p0 / ratio;
        const int hi = (p1 + 1) / ratio - 1;
        const int cnt = hi >= lo ? hi - lo + 1 : 0;
        if (w < run + cnt) { s = i; b = w - run + lo; break; }
        run += cnt;
    }
    if (s < 0) {
        /* Past the real count: page -1 makes the host's bound safe. */
        if (threadIdx.x == 0) {
            page[w] = -1;
            for (int c = 0; c < (rope ? 3 : 1); ++c) bpos[(size_t)w * bstep + c] = 0;
        }
        return;
    }

    const int t0 = cu[s];
    const int p0 = pos[t0];
    /* Column 0 is the committed slot; -1 marks a sequence with no state, whose blocks cannot be
     * built (page -1, no gather). */
    const int si = sidx[(size_t)s * sidx_ld];

    if (threadIdx.x == 0) {
        page[w] = (si < 0 || si >= slots || b >= bt_cols) ? -1 : bt[(size_t)s * bt_ld + b];
        if (!rope) {
            bpos[(size_t)w * bstep] = b * ratio;
        } else {
            /* Rotate a block key at its first token's position, three components side by side. A
             * first token in this step reads its own components; one from an earlier step is text
             * (only a decode row or a prompt's last chunk leaves a block open, and a prompt ends
             * in text), so it rotates at its index plus the shift this step's first row of the
             * sequence carries. */
            const int f = b * ratio;
            for (int c = 0; c < 3; ++c)
                bpos[(size_t)w * bstep + c] =
                    f >= p0 ? rope[(size_t)c * rplane + t0 + (f - p0)]
                            : f + (rope[(size_t)t0] - p0);
        }
    }
    if (si < 0 || si >= slots) return;

    /* The block's raw keys, each from this step or from the tail. */
    for (int i = 0; i < ratio; ++i) {
        const int q = b * ratio + i;
        const uint16_t* src;
        if (q >= p0) {
            const int row = t0 + (q - p0);
            if (row < 0 || row >= T) continue;
            src = k + (size_t)row * k_ld;
        } else {
            src = state + (size_t)si * s_slot + (size_t)(q % ring) * s_row;
        }
        uint16_t* dst = stage + (size_t)(w * ratio + i) * g_ld;
        for (int d = (int)threadIdx.x; d < n; d += kThreads) dst[d] = src[d];
    }
}

}  // namespace

extern "C" {

int r3d_rad_qsa_work(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 11) return R3D_NO(RAD_E_INVAL);
    const RadTensor* k  = r3d_opt(a, 0), *po = r3d_opt(a, 1), *cu = r3d_opt(a, 2),
                   *tl = r3d_opt(a, 3), *sx = r3d_opt(a, 4), *bt = r3d_opt(a, 5),
                   *rp = r3d_opt(a, 6), *stg = r3d_opt(a, 7), *pg = r3d_opt(a, 8),
                   *bp = r3d_opt(a, 9), *nc = r3d_opt(a, 10);
    if (!r3d_p(k) || !r3d_p(po) || !r3d_p(cu) || !r3d_p(tl) || !r3d_p(sx) ||
        !r3d_p(bt) || !r3d_p(stg) || !r3d_p(pg) || !r3d_p(bp) || !r3d_p(nc))
        return R3D_NO(RAD_E_INVAL);

    const long long n = r3d_geti_or(a, "n", 0);
    const long long ratio = r3d_geti_or(a, "ratio", 0);
    const long long ring = r3d_geti_or(a, "ring", 0);
    if (n <= 0 || ratio <= 0 || ring < ratio) return R3D_NO(RAD_E_INVAL);

    if (k->rank != 2 || bt->rank != 2 || stg->rank != 2) return R3D_NO(RAD_E_SHAPE);
    /* The tail state arrives as the group declared it: [slots, heads, ring, n], heads 1. */
    if (tl->rank != 4 || tl->shape[1] != 1 || tl->shape[2] < ring || tl->shape[3] != n)
        return R3D_NO(RAD_E_SHAPE);

    const long long T = k->shape[0];
    const long long n_seq = r3d_numel(cu) - 1;
    const long long slots = tl->shape[0];
    if (T <= 0 || n_seq <= 0 || r3d_numel(po) < T || k->shape[1] != n ||
        stg->shape[1] != n)
        return R3D_NO(RAD_E_SHAPE);
    if (bt->shape[0] < n_seq || r3d_dim(sx, 0) < n_seq) return R3D_NO(RAD_E_SHAPE);

    /* `work` is a capacity; the real count is `page`'s extent, as the reference takes it. */
    const long long mw = r3d_numel(pg);
    if (mw < n_seq || stg->shape[0] < mw * ratio || r3d_numel(nc) < n_seq)
        return R3D_NO(RAD_E_SHAPE);
    if (mw > 0x7fffffffLL || T > 0x7fffffffLL || n > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);

    long long bstep = 1;
    if (bp->rank == 2) {
        if (bp->shape[0] < mw || bp->stride[1] != 1 || bp->stride[0] < bp->shape[1] ||
            (rp && bp->shape[1] < 3))
            return R3D_NO(RAD_E_SHAPE);
        bstep = bp->stride[0];
    } else {
        /* The one-component form is a flat plane; the reference reads it at its own stride. */
        if (rp || bp->rank > 1 || r3d_numel(bp) < mw) return R3D_NO(RAD_E_SHAPE);
        bstep = bp->rank == 1 ? bp->stride[0] : 1;
    }
    if (rp && (rp->rank != 2 || rp->shape[0] < 3 || rp->shape[1] < T ||
               rp->stride[1] != 1 || rp->stride[0] < rp->shape[1] ||
               rp->dtype != RAD_I32))
        return R3D_NO(RAD_E_SHAPE);

    if (n_seq > kMaxSeq) return R3D_NO(RAD_E_SHAPE);
    /* The kernel reads these as flat i32 / bf16 planes with a single row pitch. */
    if (!r3d_contig(po) || !r3d_contig(cu) || !r3d_contig(pg) || !r3d_contig(nc))
        return R3D_NO(RAD_E_STRIDE);
    if (k->stride[1] != 1 || stg->stride[1] != 1 || tl->stride[3] != 1 ||
        bt->stride[1] != 1 || (sx->rank >= 2 && sx->stride[1] != 1))
        return R3D_NO(RAD_E_STRIDE);
    const long long k_ld = k->stride[0];
    const long long g_ld = stg->stride[0];
    const long long s_row = tl->stride[2];
    /* The kernel scales these by an unsigned row index; a negative pitch would wrap it. */
    if (k_ld < n || g_ld < n || s_row < n || tl->stride[0] < 0 || sx->stride[0] < 0 ||
        bt->stride[0] < 0 || bstep < 0)
        return R3D_NO(RAD_E_STRIDE);
    if (k->dtype != RAD_BF16 || tl->dtype != RAD_BF16 || stg->dtype != RAD_BF16)
        return R3D_NO(RAD_E_DTYPE);

    R3D_KLAUNCH(qsa_work_kernel, dim3((unsigned)mw), dim3(kThreads), (hipStream_t)s,
                (const uint16_t*)r3d_p(k), (const int32_t*)r3d_p(po), (const int32_t*)r3d_p(cu),
                (const uint16_t*)r3d_p(tl), (const int32_t*)r3d_p(sx),
                (const int32_t*)r3d_p(bt), (uint16_t*)r3d_wp(stg), (int32_t*)r3d_wp(pg),
                (int32_t*)r3d_wp(bp), (int32_t*)r3d_wp(nc), (int)T, (int)n, (int)ratio,
                (int)ring, (int)n_seq, (int)mw, (int)slots, (int)k_ld, (int)tl->stride[0],
                (int)s_row, (int)sx->stride[0], (int)bt->stride[0], (int)bt->shape[1],
                (int)g_ld, (const int32_t*)r3d_p(rp), (int)(rp ? rp->stride[0] : 0),
                (int)bstep);
    return (int)hipGetLastError();
}

}  // extern "C"
