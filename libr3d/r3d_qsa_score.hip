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
 * the result is bit-identical to libref's. `q`'s row pitch and odd-column stride are the
 * reference's row_stride/laststride, so a column slice of a wider projection reads correctly.
 *
 *   q [rows, heads*n] (dtype), bk [pages, ... , n] (dtype, one page is n contiguous values),
 *   bt [seqs, nb] i32, cu [seqs+1] i32, nc [seqs] i32 -> score [rows, nb] f32
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"

namespace {

constexpr int kThreads = 256;
/* gridDim.y is 16 bits on HIP, so rows past it are split onto z. `rows` is bounded by INT_MAX in
 * the shim, well inside 65535 * 65535. */
constexpr unsigned kGridY = 65535;

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

template <int QDT, int KDT>
int qsa_emit(const void* q, const void* bk, const int32_t* bt, const int32_t* cu,
             const int32_t* nc, float* out, int n, int heads, int nb, int nseq, int pages,
             int rows, int q_rs, int q_cs, int bk_pg, int bt_rs, int bt_cs, int o_rs, int o_cs,
             float rn, hipStream_t stream) {
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
