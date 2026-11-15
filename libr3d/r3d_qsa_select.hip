/* r3d_qsa_select.hip -- qsa_select: the topk highest-scoring complete blocks a query has passed,
 * as page ids, plus the partial page holding the query (docs/OPS.md, "qsa_select").
 *
 * A workgroup per query. The per-query bound is n = min(nc[seq], nb, (pos + 1) / ratio): block b is
 * entirely in this query's past when b * ratio + ratio - 1 <= pos. `cu` joins a query row to its
 * sequence, as in qsa_score. A row no sequence holds reads sequence 0, as libref's does.
 *
 * Live blocks are those whose page id is >= 0 and whose score is above -1e30 (unsigned, the
 * default) or finite (signed). The unsigned ranking clamps a live score to max(score, 0); the
 * signed ranking is the float's total order over all 32 bits, with -0 and +0 one key. The two
 * order keys below are the r4d kernel's, so both libraries select the same set and order.
 *
 * The selection is a four-digit radix threshold and a prefix-count emit: entries strictly above
 * the topk-th largest come first in block order, then the entries equal to it in block order until
 * topk are taken. Every slot comes from a workgroup prefix count, never from an atomic's
 * completion order, so equal scores reproduce bit for bit.
 *
 * The partial page bt[seq][n] is appended last when (pos + 1) % ratio != 0 and n < nb; it is
 * never a candidate. -1 pads the row to topk + 1. seqused is (nsel - 1) * ratio + pos % ratio + 1,
 * or 0 for an empty table.
 *
 *   score [M][nb] f32, bt [nseq][nb] i32, nc [nseq] i32, pos [M] i32, cu [nseq+1] i32
 *   sel [M][topk+1] i32, nsel [M] i32, seqused [M] i32
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_launch.h"

namespace {

constexpr int kThreads = 256;
constexpr int kBins    = 256;

/* The order key, by the op's `signed` parameter. Key 0 means "no block" in both; only a NaN's
 * bit pattern could otherwise map to it in the signed order, and a NaN is not live. */
template <bool Signed>
__device__ __forceinline__ uint32_t key_of(float v, int page)
{
    if (page < 0) return 0u;
    if (Signed) {
        if (!__builtin_isfinite(v)) return 0u;
        const uint32_t bits = v == 0.0f ? 0u : __float_as_uint(v);
        return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    }
    if (!(v > -1e30f)) return 0u;
    /* A select, not fmaxf: fmaxf(-0.0f, 0.0f) may return -0.0f, whose sign bit would rank it
     * above every positive score. The +1 keeps a live zero above key 0, which means no block. */
    return __float_as_uint(v > 0.0f ? v : 0.0f) + 1u;
}

/* The exclusive workgroup prefix of a per-thread count, and the total. In-place in `scan`, which
 * the caller does not reuse before the trailing barrier. */
__device__ __forceinline__ uint32_t wg_prefix(uint32_t count, uint32_t* scan, uint32_t* total,
                                              int t)
{
    scan[t] = count;
    __syncthreads();
    if (t == 0) {
        uint32_t acc = 0;
        for (int i = 0; i < kThreads; ++i) {
            const uint32_t c = scan[i];
            scan[i] = acc;
            acc += c;
        }
        *total = acc;
    }
    __syncthreads();
    return scan[t];
}

template <bool Signed>
__global__ __launch_bounds__(kThreads) void qsa_select_kernel(
    const float* __restrict__ score, const int32_t* __restrict__ bt,
    const int32_t* __restrict__ nc, const int32_t* __restrict__ pos,
    const int32_t* __restrict__ cu,
    int32_t* __restrict__ sel, int32_t* __restrict__ nsel, int32_t* __restrict__ seqused,
    int nb, int topk, int ratio, int nseq, int s_ld, int t_ld, int o_ld,
    int nc_step, int pos_step, int cu_step, int ns_step, int su_step)
{
    __shared__ uint32_t hist[kBins];
    __shared__ uint32_t scan[kThreads];
    __shared__ uint32_t sh_T, sh_need, sh_total;
    __shared__ int      sh_seq, sh_done;

    const int m = (int)blockIdx.x;
    const int t = (int)threadIdx.x;

    if (t == 0) {
        int s = 0;
        for (int i = 0; i < nseq; ++i) {
            const long long lo = (long long)i * cu_step;
            const long long hi = (long long)(i + 1) * cu_step;
            if (m >= cu[lo] && m < cu[hi]) { s = i; break; }
        }
        sh_seq = s;
    }
    __syncthreads();
    const int seq = sh_seq;
    const long long p = pos[(size_t)m * pos_step];
    const long long nc_s = nc[(size_t)seq * nc_step];
    long long n = nc_s < nb ? nc_s : nb;
    const long long own = (p + 1) / ratio;
    if (own < n) n = own;
    if (n < 0) n = 0;

    const float*   sc = score + (size_t)m * s_ld;
    const int32_t* tb = bt + (size_t)seq * t_ld;
    int32_t*       ou = sel + (size_t)m * o_ld;

    /* ---- the exact topk-th largest key and how many entries equal to it are taken. */
    uint32_t T = 0u, need = 0u;
    if (n > topk) {
        uint32_t fixed = 0u, want = (uint32_t)topk;
        bool done = false;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = t; i < kBins; i += kThreads) hist[i] = 0u;
            __syncthreads();
            const uint32_t mask = shift == 24 ? 0u : (0xFFFFFFFFu << (shift + 8));
            for (long long b = t; b < n; b += kThreads) {
                const uint32_t k = key_of<Signed>(sc[b], tb[b]);
                if (k != 0u && (k & mask) == (fixed & mask))
                    atomicAdd(&hist[(k >> shift) & 255u], 1u);
            }
            __syncthreads();
            if (t == 0) {
                uint32_t run = 0u, after = 0u;
                int bin = -1;
                for (int i = kBins - 1; i >= 0; --i) {
                    const uint32_t c = hist[i];
                    if (run < want && run + c >= want) { bin = i; after = run; }
                    run += c;
                }
                /* The first digit counts every live key. topk or fewer live keys are all taken,
                 * in block order, which is what T = 0 tells the emit. */
                if (shift == 24 && run <= want) {
                    sh_T = 0u; sh_need = 0u; sh_done = 1;
                } else if (bin >= 0) {
                    sh_T = fixed | ((uint32_t)bin << shift);
                    sh_need = want - after;
                    sh_done = 0;
                } else {
                    sh_T = 0u; sh_need = 0u; sh_done = 1;
                }
            }
            __syncthreads();
            if (sh_done) { done = true; break; }
            fixed = sh_T;
            want  = sh_need;
        }
        if (!done) { T = fixed; need = want; }
    }

    /* ---- entries above the threshold, in block order. */
    uint32_t out_count = 0u;
    for (long long b0 = 0; b0 < n; b0 += kThreads) {
        const long long b = b0 + t;
        const uint32_t k = b < n ? key_of<Signed>(sc[b], tb[b]) : 0u;
        const bool hi = k != 0u && (n <= topk || k > T);
        const uint32_t pre = wg_prefix(hi ? 1u : 0u, scan, &sh_total, t);
        if (hi) {
            const uint32_t slot = out_count + pre;
            if (slot < (uint32_t)topk) ou[slot] = tb[b];
        }
        out_count += sh_total;
        __syncthreads();
    }

    /* ---- the ties at the threshold, in block order, after the entries above it. */
    uint32_t taken_eq = 0u;
    for (long long b0 = 0; b0 < n; b0 += kThreads) {
        const long long b = b0 + t;
        const uint32_t k = b < n ? key_of<Signed>(sc[b], tb[b]) : 0u;
        const bool eq = k != 0u && n > topk && k == T;
        const uint32_t pre = wg_prefix(eq ? 1u : 0u, scan, &sh_total, t);
        if (eq) {
            const uint32_t rank = taken_eq + pre;
            if (rank < need)
                ou[(uint32_t)topk - need + rank] = tb[b];
        }
        taken_eq += sh_total;
        __syncthreads();
    }

    if (t == 0) {
        uint32_t out = (n <= topk || T == 0u) ? out_count : (uint32_t)topk;
        if (n < nb && (p + 1) % ratio != 0) {
            const int32_t part = tb[n];
            if (part >= 0) ou[out++] = part;
        }
        for (uint32_t i = out; i < (uint32_t)(topk + 1); ++i) ou[i] = -1;
        nsel[(size_t)m * ns_step] = (int32_t)out;
        const int32_t su = out == 0u ? 0
                          : (int32_t)(((long long)out - 1) * ratio + (p % ratio) + 1);
        seqused[(size_t)m * su_step] = su;
    }
}

/* The per-query scalar operands may be rank 1 or [N, 1]; this returns the element step, or -1. */
long long scalar_step(const RadTensor* t)
{
    if (!t || t->rank < 1) return -1;
    if (rad_tensor_is_contiguous(t)) return 1;
    if (t->rank == 1) return t->stride[0];
    if (t->rank == 2 && t->shape[1] == 1) return t->stride[0];
    return -1;
}

}  // namespace

extern "C" {

int r3d_rad_qsa_select(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 8) return R3D_NO(RAD_E_INVAL);
    const RadTensor* sc = r3d_opt(a, 0), *bt = r3d_opt(a, 1), *nc = r3d_opt(a, 2),
                   *po = r3d_opt(a, 3), *cu = r3d_opt(a, 4), *se = r3d_opt(a, 5),
                   *ns = r3d_opt(a, 6), *su = r3d_opt(a, 7);
    if (!r3d_p(sc) || !r3d_p(bt) || !r3d_p(nc) || !r3d_p(po) || !r3d_p(cu) ||
        !r3d_wp(se) || !r3d_wp(ns) || !r3d_wp(su))
        return R3D_NO(RAD_E_INVAL);
    if (sc->rank != 2 || bt->rank != 2 || se->rank != 2) return R3D_NO(RAD_E_SHAPE);

    const long long topk  = r3d_geti_or(a, "topk", 0);
    const long long ratio = r3d_geti_or(a, "ratio", 0);
    const long long sgn   = r3d_geti_or(a, "signed", 0);
    if (topk <= 0 || ratio <= 0) return R3D_NO(RAD_E_SHAPE);
    if (topk > 0x7fffffffLL || ratio > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    if (sgn != 0 && sgn != 1) return R3D_NO(RAD_E_INVAL);

    const long long M    = sc->shape[0];
    const long long nb   = bt->shape[1] < sc->shape[1] ? bt->shape[1] : sc->shape[1];
    const long long nseq = r3d_numel(cu) - 1;
    if (M <= 0 || nb <= 0 || nseq <= 0) return R3D_NO(RAD_E_SHAPE);
    if (M > 0x7fffffffLL || nb > 0x7fffffffLL || nseq > 0x7fffffffLL)
        return R3D_NO(RAD_E_SHAPE);
    if (bt->shape[0] < nseq || r3d_numel(nc) < nseq) return R3D_NO(RAD_E_SHAPE);
    if (se->shape[0] < M || se->shape[1] < topk + 1) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(ns) < M || r3d_numel(po) < M || r3d_numel(su) < M)
        return R3D_NO(RAD_E_SHAPE);

    if (!r3d_is_f32(sc) || !r3d_is_i32(bt) || !r3d_is_i32(nc) || !r3d_is_i32(po) ||
        !r3d_is_i32(cu) || !r3d_is_i32(se) || !r3d_is_i32(ns) || !r3d_is_i32(su))
        return R3D_NO(RAD_E_DTYPE);

    const long long s_ld = r3d_row_pitch(sc);
    const long long t_ld = r3d_row_pitch(bt);
    const long long o_ld = r3d_row_pitch(se);
    if (s_ld < nb) R3D_STR("score row pitch", "pitch", s_ld, "nb", nb);
    if (t_ld < nb) R3D_STR("bt row pitch", "pitch", t_ld, "nb", nb);
    if (o_ld < topk + 1) R3D_STR("sel row pitch", "pitch", o_ld, "topk + 1", topk + 1);

    const long long nc_step = scalar_step(nc), pos_step = scalar_step(po);
    const long long cu_step = scalar_step(cu), ns_step = scalar_step(ns);
    const long long su_step = scalar_step(su);
    if (nc_step < 1 || pos_step < 1 || cu_step < 1 || ns_step < 1 || su_step < 1)
        return R3D_NO(RAD_E_STRIDE);
    if (s_ld > 0x7fffffffLL || t_ld > 0x7fffffffLL || o_ld > 0x7fffffffLL ||
        nc_step > 0x7fffffffLL || pos_step > 0x7fffffffLL || cu_step > 0x7fffffffLL ||
        ns_step > 0x7fffffffLL || su_step > 0x7fffffffLL)
        return R3D_NO(RAD_E_SHAPE);

    const int kM = (int)M, knb = (int)nb, ktopk = (int)topk, kratio = (int)ratio;
    const int knseq = (int)nseq;
    if (sgn)
        R3D_KLAUNCH(qsa_select_kernel<true>, dim3((unsigned)kM), dim3(kThreads), (hipStream_t)s,
                    (const float*)r3d_p(sc), (const int32_t*)r3d_p(bt), (const int32_t*)r3d_p(nc),
                    (const int32_t*)r3d_p(po), (const int32_t*)r3d_p(cu), (int32_t*)r3d_wp(se),
                    (int32_t*)r3d_wp(ns), (int32_t*)r3d_wp(su), knb, ktopk, kratio, knseq,
                    (int)s_ld, (int)t_ld, (int)o_ld, (int)nc_step, (int)pos_step, (int)cu_step,
                    (int)ns_step, (int)su_step);
    else
        R3D_KLAUNCH(qsa_select_kernel<false>, dim3((unsigned)kM), dim3(kThreads), (hipStream_t)s,
                    (const float*)r3d_p(sc), (const int32_t*)r3d_p(bt), (const int32_t*)r3d_p(nc),
                    (const int32_t*)r3d_p(po), (const int32_t*)r3d_p(cu), (int32_t*)r3d_wp(se),
                    (int32_t*)r3d_wp(ns), (int32_t*)r3d_wp(su), knb, ktopk, kratio, knseq,
                    (int)s_ld, (int)t_ld, (int)o_ld, (int)nc_step, (int)pos_step, (int)cu_step,
                    (int)ns_step, (int)su_step);
    return (int)hipGetLastError();
}

}  // extern "C"
