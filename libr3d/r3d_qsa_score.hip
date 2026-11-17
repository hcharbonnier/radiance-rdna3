/* r3d_qsa_score.hip -- qsa_score: the QSA indexer's score of each compressed block of key
 * history against a query.
 *
 *   score[m][b] = ( sum over heads h of relu( q[m][h] . bk[bt[seq(m)][b]] ) ) / sqrt(n)
 *
 * for b below min(nc[seq(m)], the table's width, the score plane's width). The relu is per
 * (block, head) and the sum is over the heads, in that order: independent votes, each of which
 * can only argue for a block (docs/OPS.md, "qsa_score").
 *
 * `bt` holds page ids, not block indices: the attention KV group is paged at the compress ratio,
 * so a block and a page are the same thing, and the sequence's block table is the list scored.
 * A negative entry scores -infinity rather than zero, because zero is a real score (every head
 * vetoed the block) and padding must not outrank it. `nc` is the sequence's live complete-block
 * count and bounds the loop; an entry at or past the key store's page count is left unwritten
 * (the reference refuses such a launch on the host, which a device kernel cannot do).
 *
 * A thread owns one (query, block) pair and walks that block's heads and lanes in the reference's
 * order: the dot is an ascending f32 chain and the relus are summed over the heads in order, so
 * the result is bit-identical to libref's. A scalar row does that directly; the tiled row below
 * stages a tile of query rows and block pages in LDS with coalesced loads and keeps the same
 * per-(row, block) arithmetic, so it is exact too.
 *
 *   q [rows, heads*n] (dtype), bk [pages, ... , n] (dtype, one page is n contiguous values),
 *   bt [seqs, nb] i32, cu [seqs+1] i32, nc [seqs] i32 -> score [rows, nb] f32
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"

namespace {

constexpr int kThreads = 256;
/* gridDim.y is 16 bits on HIP, so rows past it are split onto z. `rows` is bounded by INT_MAX in
 * the shim, well inside 65535 * 65535. */
constexpr unsigned kGridY = 65535;

/* The tiled row: one workgroup covers a rectangle of query rows and block-table columns. The
 * original row let each thread read a different page at the same offset, so a wave touched 32
 * pages and each 32-byte sector carried two useful bytes. Here the pages of the tile are read
 * once into LDS, coalesced, and the queries of the tile likewise; each thread then keeps several
 * (row, block) chains, which gives it independent f32 chains to interleave without changing any
 * chain's order. */
constexpr int kTileThreads = 256;      /* eight waves, two workgroups per CU */
constexpr int kTileRows = 32;          /* query rows per tile */
constexpr int kRowGroups = 8;          /* a thread owns kRowsPerThread rows */
constexpr int kRowsPerThread = 4;
constexpr int kBlockPairs = 32;        /* a thread owns two blocks kBlockPairs apart */
constexpr int kTileBlocksMax = 2 * kBlockPairs;
static_assert(kRowGroups * kBlockPairs == kTileThreads, "one thread per (row group, block pair)");
static_assert(kTileRows == kRowGroups * kRowsPerThread, "the row groups cover the tile");
constexpr int kLdsBudget = 32 * 1024;  /* two workgroups stay resident per CU */

/* One activation element, widened to f32 exactly. DT is an R3D_DT_* code; q and bk may differ (the
 * indexer's query is the activation dtype, its block-key group is bf16 on this model). */
template <int DT>
__device__ __forceinline__ float qsa_load(const void* p, long long i) {
    if constexpr (DT == R3D_DT_BF16)
        return bf16_to_f32(((const uint16_t*)p)[i]);
    else if constexpr (DT == R3D_DT_F16)
        return f16_to_f32(((const uint16_t*)p)[i]);
    else
        return ((const float*)p)[i];
}

/* The stored element of DT: two bytes for bf16/f16, four for f32. */
template <int DT>
struct QsaElem {
    using type = std::conditional_t<DT == R3D_DT_F32, float, uint16_t>;
};

template <int DT>
__device__ __forceinline__ float qsa_widen(typename QsaElem<DT>::type v) {
    if constexpr (DT == R3D_DT_BF16)
        return bf16_to_f32(v);
    else if constexpr (DT == R3D_DT_F16)
        return f16_to_f32(v);
    else
        return v;
}

/* Two adjacent stored elements of a two-byte dtype, widened exactly. */
template <int DT>
__device__ __forceinline__ void qsa_unpack(uint32_t v, float* lo, float* hi) {
    if constexpr (DT == R3D_DT_BF16) {
        *lo = bf16_to_f32((uint16_t)v);
        *hi = bf16_to_f32((uint16_t)(v >> 16));
    } else {
        *lo = f16_to_f32((uint16_t)v);
        *hi = f16_to_f32((uint16_t)(v >> 16));
    }
}

/* An LDS row stride that keeps two-byte rows on a 32-bit boundary and spreads equal offsets of
 * successive rows/pages over different banks. */
__host__ __device__ __forceinline__ int qsa_lds_stride(int n, int elem_bytes) {
    return elem_bytes == 4 ? n + 1 : n + 2 + (n & 1);
}

/* ---- the scalar row: the original shape, kept for tiles that cannot fit their stage ---------- */

template <int QDT, int KDT>
__global__ __launch_bounds__(kThreads) void qsa_score_kernel(
    const void* __restrict__ q, const void* __restrict__ bk, const int32_t* __restrict__ bt,
    const int32_t* __restrict__ cu, const int32_t* __restrict__ nc, float* __restrict__ out,
    int n, int heads, int nb, int nseq, int pages, int q_rs, int q_cs, int bk_pg, int bt_rs,
    int bt_cs, int o_rs, int o_cs, float rn) {
    /* Every lane of a workgroup has the same query row, so its sequence and live block count are
     * computed once. A row no sequence holds is skipped, as in the reference. */
    __shared__ int sh_seq, sh_nbs;
    const int m = (int)((unsigned)blockIdx.y + (unsigned)blockIdx.z * kGridY);
    const int b = (int)blockIdx.x * kThreads + (int)threadIdx.x;
    if (threadIdx.x == 0) {
        int s = -1;
        for (int i = 0; i < nseq; ++i)
            if (m >= cu[i] && m < cu[i + 1]) { s = i; break; }
        int live = 0;
        if (s >= 0) {
            live = nc[s];
            if (live > nb) live = nb;
            if (live < 0) live = 0;
        }
        sh_seq = s;
        sh_nbs = live;
    }
    __syncthreads();
    const int s = sh_seq;
    if (s < 0 || b >= nb || b >= sh_nbs) return;

    const int pg = bt[(size_t)s * bt_rs + (size_t)b * bt_cs];
    float v;
    if (pg < 0) {
        v = -INFINITY;
    } else {
        /* The reference rejects such a page on the host before writing anything; a launch cannot
         * read the table, so the kernel leaves the element alone rather than fault. */
        if (pg >= pages) return;
        const long long qrow = (long long)m * q_rs;
        const long long krow = (long long)pg * bk_pg;
        float sum = 0.0f;
        for (int h = 0; h < heads; ++h) {
            const long long qh = qrow + (long long)h * n * q_cs;
            float d = 0.0f;
            for (int i = 0; i < n; ++i)
                d += qsa_load<QDT>(q, qh + (long long)i * q_cs) *
                     qsa_load<KDT>(bk, krow + i);
            sum += fmaxf(d, 0.0f);
        }
        v = sum / rn;
    }
    out[(size_t)m * o_rs + (size_t)b * o_cs] = v;
}

/* ---- the tiled row ------------------------------------------------------------------------- */

/* `btile` block columns, kTileRows query rows. The query rows of a tile need not share a sequence;
 * the kernel splits the tile into runs of equal sequence and stages the block pages of one run at
 * a time, so a boundary inside the tile costs a second pass, not a fallback. */
template <int QDT, int KDT>
__global__ __launch_bounds__(kTileThreads) void qsa_score_tiled_kernel(
    const void* __restrict__ q, const void* __restrict__ bk, const int32_t* __restrict__ bt,
    const int32_t* __restrict__ cu, const int32_t* __restrict__ nc, float* __restrict__ out,
    int n, int heads, int nb, int nseq, int pages, int rows, int q_rs, int q_cs, int bk_pg,
    int bt_rs, int bt_cs, int o_rs, int o_cs, float rn, int btile) {
    using QE = typename QsaElem<QDT>::type;
    using KE = typename QsaElem<KDT>::type;
    const int qstride = qsa_lds_stride(n, (int)sizeof(QE));
    const int kstride = qsa_lds_stride(n, (int)sizeof(KE));
    const size_t q_elems = (size_t)kTileRows * qstride;

    extern __shared__ __align__(16) unsigned char smem[];
    QE* qs = (QE*)smem;
    KE* ks = (KE*)(smem + q_elems * sizeof(QE));
    int* pgs = (int*)(smem + q_elems * sizeof(QE) + (size_t)btile * kstride * sizeof(KE));

    __shared__ int sh_seq[kTileRows];
    __shared__ int sh_live[kTileRows];
    __shared__ int sh_gstart[kTileRows], sh_gend[kTileRows], sh_gseq[kTileRows];
    __shared__ int sh_ngrp, sh_maxlive;

    const int m0 = (int)((unsigned)blockIdx.y + (unsigned)blockIdx.z * kGridY) * kTileRows;
    const int b0 = (int)blockIdx.x * btile;
    const int tid = (int)threadIdx.x;

    /* Which sequence holds each row of the tile, and its live complete-block count. */
    if (tid < kTileRows) {
        const int m = m0 + tid;
        int s = -1;
        if (m < rows) {
            for (int i = 0; i < nseq; ++i)
                if (m >= cu[i] && m < cu[i + 1]) { s = i; break; }
        }
        int live = 0;
        if (s >= 0) {
            live = nc[s];
            if (live > nb) live = nb;
            if (live < 0) live = 0;
        }
        sh_seq[tid] = s;
        sh_live[tid] = live;
    }
    __syncthreads();

    /* Contiguous runs of one sequence; every row of a run shares its live count. */
    if (tid == 0) {
        int ng = 0, maxlive = 0;
        for (int r = 0; r < kTileRows;) {
            const int s = sh_seq[r];
            int e = r + 1;
            while (e < kTileRows && sh_seq[e] == s) ++e;
            sh_gstart[ng] = r;
            sh_gend[ng] = e;
            sh_gseq[ng] = s;
            if (s >= 0 && sh_live[r] > maxlive) maxlive = sh_live[r];
            ++ng;
            r = e;
        }
        sh_ngrp = ng;
        sh_maxlive = maxlive;
    }
    __syncthreads();
    /* No sequence of this tile holds a block this far out. Uniform, so the early exit is safe. */
    if (b0 >= sh_maxlive) return;

    /* Stage the tile's query rows a head at a time: a wave reads one row's head, so its lanes
     * touch consecutive columns, and the LDS stage stays small enough for two workgroups. */
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = kTileThreads / 32;
    const int rp = tid / kBlockPairs;    /* 0..kRowGroups-1: this thread's four rows */
    const int bp = tid & (kBlockPairs - 1);
    const int rbase = rp * kRowsPerThread;
    /* Blocks bp and bp + kBlockPairs, not neighbours: the page stride is an odd number of
     * 32-bit words, so consecutive lanes then hit consecutive LDS banks. */
    const int lb0 = bp;
    const int lb1_out = bp + kBlockPairs;
    const int lb1 = lb1_out < btile ? lb1_out : bp;

    for (int g = 0; g < sh_ngrp; ++g) {
        const int s = sh_gseq[g];
        const int ga = sh_gstart[g], gb = sh_gend[g];
        const int live = s < 0 ? 0 : sh_live[ga];
        if (live <= b0) continue;  /* uniform: the whole run has no live column in this tile */
        int nlive = live - b0;
        if (nlive > btile) nlive = btile;

        /* Pages of this run's block columns, coalesced: a wave walks one page's elements. */
        for (int lb = warp; lb < btile; lb += nwarps) {
            const int b = b0 + lb;
            int pg = -1;
            if (s >= 0 && lb < nlive && b < nb)
                pg = bt[(size_t)s * bt_rs + (size_t)b * bt_cs];
            if (lane == 0) pgs[lb] = pg;
            KE* dst = ks + (size_t)lb * kstride;
            if (pg >= 0 && pg < pages) {
                const KE* src = (const KE*)bk + (size_t)pg * bk_pg;
                for (int i = lane; i < n; i += 32) dst[i] = src[i];
            } else {
                for (int i = lane; i < n; i += 32) dst[i] = (KE)0;
            }
        }
        __syncthreads();

        /* Per (row, block) the reference's order: heads outer, then the ascending f32 dot. The
         * query stage is one head, refilled for each head; the sums stay in registers. */
        const bool active = rbase < gb && rbase + kRowsPerThread > ga && lb0 < nlive;
        float sum[kRowsPerThread][2];
        for (int r = 0; r < kRowsPerThread; ++r)
            sum[r][0] = sum[r][1] = 0.0f;
        for (int h = 0; h < heads; ++h) {
            for (int r = warp; r < kTileRows; r += nwarps) {
                const int m = m0 + r;
                QE* dst = qs + (size_t)r * qstride;
                if (m < rows) {
                    const QE* src =
                        (const QE*)q + (long long)m * q_rs + (long long)h * n * q_cs;
                    for (int i = lane; i < n; i += 32) dst[i] = src[(long long)i * q_cs];
                } else {
                    for (int i = lane; i < n; i += 32) dst[i] = (QE)0;
                }
            }
            __syncthreads();
            if (active) {
                float d[kRowsPerThread][2];
                for (int r = 0; r < kRowsPerThread; ++r)
                    d[r][0] = d[r][1] = 0.0f;
                const QE* qh = qs + (size_t)rbase * qstride;
                const KE* kb0 = ks + (size_t)lb0 * kstride;
                const KE* kb1 = ks + (size_t)lb1 * kstride;
                if constexpr (sizeof(QE) == 2 && sizeof(KE) == 2) {
                    /* Two columns a load: the LDS rows are even and 4-byte aligned, and each
                     * chain still adds column i before i + 1, so the dot is unchanged. */
                    int i = 0;
#pragma unroll 4
                    for (; i + 1 < n; i += 2) {
                        uint32_t qw[kRowsPerThread];
                        for (int r = 0; r < kRowsPerThread; ++r)
                            qw[r] = *(const uint32_t*)(qh + (size_t)r * qstride + i);
                        float k00, k01, k10, k11;
                        qsa_unpack<KDT>(*(const uint32_t*)(kb0 + i), &k00, &k01);
                        qsa_unpack<KDT>(*(const uint32_t*)(kb1 + i), &k10, &k11);
                        for (int r = 0; r < kRowsPerThread; ++r) {
                            float q0, q1;
                            qsa_unpack<QDT>(qw[r], &q0, &q1);
                            d[r][0] += q0 * k00;
                            d[r][0] += q1 * k01;
                            d[r][1] += q0 * k10;
                            d[r][1] += q1 * k11;
                        }
                    }
                    if (i < n) {
                        for (int r = 0; r < kRowsPerThread; ++r) {
                            const float qv = qsa_widen<QDT>(qh[(size_t)r * qstride + i]);
                            d[r][0] += qv * qsa_widen<KDT>(kb0[i]);
                            d[r][1] += qv * qsa_widen<KDT>(kb1[i]);
                        }
                    }
                } else {
                    for (int i = 0; i < n; ++i) {
                        for (int r = 0; r < kRowsPerThread; ++r) {
                            const float qv = qsa_widen<QDT>(qh[(size_t)r * qstride + i]);
                            d[r][0] += qv * qsa_widen<KDT>(kb0[i]);
                            d[r][1] += qv * qsa_widen<KDT>(kb1[i]);
                        }
                    }
                }
                for (int r = 0; r < kRowsPerThread; ++r) {
                    sum[r][0] += fmaxf(d[r][0], 0.0f);
                    sum[r][1] += fmaxf(d[r][1], 0.0f);
                }
            }
            __syncthreads();
        }
        if (active) {
            for (int r = 0; r < kRowsPerThread; ++r) {
                const int rr = rbase + r;
                if (rr < ga || rr >= gb) continue;
                const int m = m0 + rr;
                if (m >= rows) continue;
                for (int t = 0; t < 2; ++t) {
                    const int lb = t ? lb1_out : lb0;
                    const int b = b0 + lb;
                    if (lb >= btile || b >= nb || b >= live) continue;
                    const int pg = pgs[lb];
                    if (pg >= pages) continue;  /* the reference leaves this element unwritten */
                    const float v = pg < 0 ? -INFINITY : sum[r][t] / rn;
                    out[(size_t)m * o_rs + (size_t)b * o_cs] = v;
                }
            }
        }
        __syncthreads();  /* the next run stages its own pages over these */
    }
}

/* libref's row_stride/laststride: fold the trailing extents until they span `cols`, and read the
 * next axis out as the row pitch. A column slice therefore reads its own pitch, not the buffer's. */
long long qsa_row_stride(const RadTensor* t, long long cols) {
    if (!t || t->rank < 2) return cols;
    long long span = 1;
    for (int i = (int)t->rank - 1; i >= 0; --i) {
        span *= t->shape[i];
        if (span == cols) return (i > 0) ? t->stride[i - 1] : cols;
        if (span > cols) break;
    }
    return t->stride[t->rank - 2];
}

long long qsa_last_stride(const RadTensor* t) { return t->rank ? t->stride[t->rank - 1] : 1; }

/* The LDS bytes a tiled launch needs, and the block width that fits; 0 when no tile fits and the
 * scalar row must be used. The low bound is nb itself so a narrow table stages nothing unused. */
int qsa_tile_blocks(long long nb, int n, int qdt, int kdt, int* lds_bytes) {
    const int qel = qdt == R3D_DT_F32 ? 4 : 2;
    const int kel = kdt == R3D_DT_F32 ? 4 : 2;
    const int qstride = qsa_lds_stride(n, qel);
    const int kstride = qsa_lds_stride(n, kel);
    const long long qbytes = (long long)kTileRows * qstride * qel;
    const long long per_block = (long long)kstride * kel + 4;
    if (qbytes + 2 * per_block > kLdsBudget) return 0;
    long long nb_even = nb + (nb & 1);
    if (nb_even < 2) nb_even = 2;
    long long btile = (kLdsBudget - qbytes) / per_block;
    if (btile > kTileBlocksMax) btile = kTileBlocksMax;
    if (btile > nb_even) btile = nb_even;
    btile &= ~1LL;
    if (btile < 2) return 0;
    *lds_bytes = (int)(qbytes + btile * per_block);
    return (int)btile;
}

template <int QDT, int KDT>
int qsa_emit(const void* q, const void* bk, const int32_t* bt, const int32_t* cu,
             const int32_t* nc, float* out, int n, int heads, int nb, int nseq, int pages,
             int rows, int q_rs, int q_cs, int bk_pg, int bt_rs, int bt_cs, int o_rs, int o_cs,
             float rn, hipStream_t stream) {
    int lds_bytes = 0;
    const int btile = qsa_tile_blocks(nb, n, QDT, KDT, &lds_bytes);
    if (btile > 0) {
        const long long tiles = (long long)(rows + kTileRows - 1) / kTileRows;
        const unsigned gy = (unsigned)(tiles < (long long)kGridY ? tiles : (long long)kGridY);
        const unsigned gz = (unsigned)((tiles + (long long)kGridY - 1) / kGridY);
        const dim3 grid((unsigned)((nb + btile - 1) / btile), gy, gz);
        R3D_KLAUNCH_S(R3D_KNAME(qsa_score_tiled_kernel<QDT, KDT>), grid, dim3(kTileThreads),
                      lds_bytes, stream, q, bk, bt, cu, nc, out, n, heads, nb, nseq, pages, rows,
                      q_rs, q_cs, bk_pg, bt_rs, bt_cs, o_rs, o_cs, rn, btile);
        return (int)hipGetLastError();
    }
    const unsigned gy = (unsigned)(rows < (long long)kGridY ? rows : (long long)kGridY);
    const unsigned gz = (unsigned)((rows + (long long)kGridY - 1) / kGridY);
    const dim3 grid((unsigned)((nb + kThreads - 1) / kThreads), gy, gz);
    R3D_KLAUNCH(R3D_KNAME(qsa_score_kernel<QDT, KDT>), grid, dim3(kThreads), stream, q, bk, bt, cu,
                nc, out, n, heads, nb, nseq, pages, q_rs, q_cs, bk_pg, bt_rs, bt_cs, o_rs, o_cs,
                rn);
    return (int)hipGetLastError();
}

template <int QDT>
int qsa_emit_key(int kdt, const void* q, const void* bk, const int32_t* bt, const int32_t* cu,
                 const int32_t* nc, float* out, int n, int heads, int nb, int nseq, int pages,
                 int rows, int q_rs, int q_cs, int bk_pg, int bt_rs, int bt_cs, int o_rs,
                 int o_cs, float rn, hipStream_t stream) {
    switch (kdt) {
        case R3D_DT_BF16:
            return qsa_emit<QDT, R3D_DT_BF16>(q, bk, bt, cu, nc, out, n, heads, nb, nseq, pages,
                                              rows, q_rs, q_cs, bk_pg, bt_rs, bt_cs, o_rs, o_cs,
                                              rn, stream);
        case R3D_DT_F16:
            return qsa_emit<QDT, R3D_DT_F16>(q, bk, bt, cu, nc, out, n, heads, nb, nseq, pages,
                                             rows, q_rs, q_cs, bk_pg, bt_rs, bt_cs, o_rs, o_cs,
                                             rn, stream);
        case R3D_DT_F32:
            return qsa_emit<QDT, R3D_DT_F32>(q, bk, bt, cu, nc, out, n, heads, nb, nseq, pages,
                                             rows, q_rs, q_cs, bk_pg, bt_rs, bt_cs, o_rs, o_cs,
                                             rn, stream);
        default:
            return R3D_NO(RAD_E_DTYPE);
    }
}

}  // namespace

extern "C" {

int r3d_rad_qsa_score(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 6) return R3D_NO(RAD_E_INVAL);
    const RadTensor* q = r3d_opt(a, 0), *bk = r3d_opt(a, 1), *bt = r3d_opt(a, 2),
                    *cu = r3d_opt(a, 3), *nc = r3d_opt(a, 4), *o = r3d_opt(a, 5);
    if (!r3d_p(q) || !r3d_p(bk) || !r3d_p(bt) || !r3d_p(cu) || !r3d_p(nc) || !r3d_wp(o))
        return R3D_NO(RAD_E_INVAL);
    const long long n = r3d_geti_or(a, "n", 0), heads = r3d_geti_or(a, "heads", 0);
    if (n <= 0 || heads <= 0) return R3D_NO(RAD_E_SHAPE);
    if (bt->rank != 2 || o->rank != 2 || bk->rank < 2) return R3D_NO(RAD_E_SHAPE);

    const long long rows = o->shape[0];
    const long long nb = bt->shape[1] < o->shape[1] ? bt->shape[1] : o->shape[1];
    const long long nseq = r3d_numel(cu) - 1;
    const long long pages = bk->shape[0];
    if (rows <= 0 || nb <= 0 || nseq <= 0 || pages <= 0) return R3D_NO(RAD_E_SHAPE);
    if (bt->shape[0] < nseq || r3d_numel(nc) < nseq) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(q) / (heads * n) < rows) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(bk) / pages != n) return R3D_NO(RAD_E_SHAPE);
    const long long bk_pg = r3d_row_pitch(bk);
    if (bk_pg < 0) return R3D_NO(RAD_E_STRIDE);

    if (bt->dtype != RAD_I32 || cu->dtype != RAD_I32 || nc->dtype != RAD_I32 ||
        o->dtype != RAD_F32)
        return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(cu) || !r3d_contig(nc)) return R3D_NO(RAD_E_STRIDE);
    const int qdt = r3d_dt_code(q->dtype), kdt = r3d_dt_code(bk->dtype);
    if ((qdt != R3D_DT_BF16 && qdt != R3D_DT_F16 && qdt != R3D_DT_F32) ||
        (kdt != R3D_DT_BF16 && kdt != R3D_DT_F16 && kdt != R3D_DT_F32))
        return R3D_NO(RAD_E_DTYPE);

    const long long q_rs = qsa_row_stride(q, heads * n), q_cs = qsa_last_stride(q);
    const long long bt_rs = bt->stride[0], bt_cs = bt->stride[1];
    const long long o_rs = o->stride[0], o_cs = o->stride[1];
    if (q_rs < 0 || q_cs <= 0 || bt_rs < 0 || bt_cs <= 0 || o_rs < 0 || o_cs <= 0)
        return R3D_NO(RAD_E_STRIDE);
    /* The kernel takes 32-bit pitches and extents, as the rest of the library does. */
    const long long lim = 0x7fffffffLL;
    if (rows > lim || nb > lim || nseq > lim || pages > lim || n > lim || heads > lim ||
        q_rs > lim || q_cs > lim || bk_pg > lim || bt_rs > lim || bt_cs > lim || o_rs > lim ||
        o_cs > lim)
        return R3D_NO(RAD_E_SHAPE);

    /* libref's `rn`, and the division below, in f32. */
    const float rn = std::sqrt((float)n);
    const void* qp = r3d_p(q);
    const void* kp = r3d_p(bk);
    const int32_t* btp = (const int32_t*)r3d_p(bt);
    const int32_t* cup = (const int32_t*)r3d_p(cu);
    const int32_t* ncp = (const int32_t*)r3d_p(nc);
    float* op = (float*)r3d_wp(o);
    const hipStream_t stream = (hipStream_t)s;

    switch (qdt) {
        case R3D_DT_BF16:
            return qsa_emit_key<R3D_DT_BF16>(
                kdt, qp, kp, btp, cup, ncp, op, (int)n, (int)heads, (int)nb, (int)nseq,
                (int)pages, (int)rows, (int)q_rs, (int)q_cs, (int)bk_pg, (int)bt_rs,
                (int)bt_cs, (int)o_rs, (int)o_cs, rn, stream);
        case R3D_DT_F16:
            return qsa_emit_key<R3D_DT_F16>(
                kdt, qp, kp, btp, cup, ncp, op, (int)n, (int)heads, (int)nb, (int)nseq,
                (int)pages, (int)rows, (int)q_rs, (int)q_cs, (int)bk_pg, (int)bt_rs,
                (int)bt_cs, (int)o_rs, (int)o_cs, rn, stream);
        case R3D_DT_F32:
            return qsa_emit_key<R3D_DT_F32>(
                kdt, qp, kp, btp, cup, ncp, op, (int)n, (int)heads, (int)nb, (int)nseq,
                (int)pages, (int)rows, (int)q_rs, (int)q_cs, (int)bk_pg, (int)bt_rs,
                (int)bt_cs, (int)o_rs, (int)o_cs, rn, stream);
        default:
            return R3D_NO(RAD_E_DTYPE);
    }
}

}  // extern "C"
