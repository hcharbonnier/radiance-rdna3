/* r3d_ngram_hint.hip -- ngram_hint: the row ids of a draft pass's newest token, one pass early.
 *
 *   ids[s][h] = ids_hash(the n-gram ending at sequence s's newest token)
 *
 * The hash and both tail layouts are ngram_ids' (docs/OPS.md). The op only changes which id is the
 * newest: a history pass's last row (with `hist`) or this round's single row. The rows are a
 * prefetch hint; nothing reads them as model ids. The schema is libr4d's, copied slot for slot.
 *
 * One thread a sequence, as the reference writes it: at decode the hint is a handful of 64-bit
 * modulos, and at prefill it is still negligible beside the gather it feeds.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_launch.h"

namespace {

__device__ __forceinline__ long long ng_wrap_mul(long long a, long long b) {
    return (long long)((unsigned long long)a * (unsigned long long)b);
}
__device__ __forceinline__ long long ng_floor_mod(long long a, long long b) {
    const long long r = a % b;
    return r < 0 ? r + b : r;
}

constexpr uint32_t kNgThreads = 256;
constexpr uint32_t kNgMaxShift = 8;

__global__ __launch_bounds__(kNgThreads) void ngram_hint_kernel(
    const int32_t* __restrict__ tok, const int32_t* __restrict__ pos,
    const int32_t* __restrict__ cu, const long long* __restrict__ mult,
    const long long* __restrict__ vsz, const long long* __restrict__ off,
    const int32_t* __restrict__ sidx, int32_t* __restrict__ tail, int32_t* __restrict__ ids,
    uint32_t n_seq, uint32_t heads, uint32_t ngram, uint32_t per, long long eos, uint32_t ids_ld,
    uint32_t hist, uint32_t slots) {
    const uint32_t s = blockIdx.x * kNgThreads + threadIdx.x;
    if (s >= n_seq) return;
    const uint32_t bos = (uint32_t)cu[s], eot = (uint32_t)cu[s + 1];
    const int32_t slot = sidx ? sidx[s] : (int32_t)s;
    int32_t* const row = ids + (long long)s * ids_ld;
    /* A sequence with nothing to hint still gets ids a gather can read: its heads' first rows. */
    if (eot <= bos || slot < 0) {
        for (uint32_t h = 0; h < heads; ++h) row[h] = (int32_t)off[h];
        return;
    }
    const uint32_t ctx = ngram - 1u;
    int32_t* const commit = tail + (long long)((uint32_t)slot % slots) * 2 * ctx;
    int32_t* const chain = commit + ctx;

    const uint32_t last = eot - 1u;
    const long long x = (long long)tok[last];
    long long back[kNgMaxShift];
    if (hist) {
        uint32_t first = last;
        const int32_t at = pos[last];
        while (first > bos && pos[first - 1u] == at) --first;
        const uint32_t have = first - bos;
        for (uint32_t k = 0; k < ctx; ++k)
            back[k] = k < have ? (long long)tok[first - 1u - k]
                               : (long long)commit[ctx - 1u - (k - have)];
    } else {
        for (uint32_t k = 0; k < ctx; ++k) back[k] = (long long)chain[ctx - 1u - k];
    }
    for (uint32_t k = 0; k < ctx; ++k) {
        const int32_t v = k + 1u < ctx ? (int32_t)back[ctx - 2u - k] : (int32_t)x;
        if (hist) commit[k] = v;
        chain[k] = v;
    }

    long long sh[kNgMaxShift];
    sh[0] = x;
    bool cut = false;
    for (uint32_t k = 1; k < ngram; ++k) {
        if (cut || sh[k - 1] == eos) { cut = true; sh[k] = eos; continue; }
        sh[k] = back[k - 1u];
    }
    long long mixed = 0;
    for (uint32_t k = 0; k < ngram; ++k) {
        const long long term = ng_wrap_mul(sh[k], mult[k]);
        mixed = k == 0 ? term : (mixed ^ term);
        if (k == 0) continue;
        for (uint32_t j = 0; j < per; ++j) {
            const uint32_t h = (k - 1u) * per + j;
            row[h] = (int32_t)(ng_floor_mod(mixed, vsz[h]) + off[h]);
        }
    }
}

}  // namespace

extern "C" {

int r3d_ngram_hint(long tok, long pos, long cu, long mult, long vsz, long off, long cidx,
                   long tail, long ids, int n_seq, int heads, int ngram, int eos, int hist,
                   int ids_ld, int slots, long stream) {
    if (!tok || !pos || !cu || !mult || !vsz || !off || !tail || !ids) return -1;
    if (n_seq <= 0 || heads <= 0 || ngram < 2 || ngram > (int)kNgMaxShift || eos < 0 ||
        ids_ld < heads || slots < 1)
        return -2;
    if (heads % (ngram - 1)) return -2;
    const uint32_t blocks = (uint32_t)((n_seq + kNgThreads - 1) / kNgThreads);
    R3D_KLAUNCH(ngram_hint_kernel, dim3(blocks), dim3(kNgThreads), (hipStream_t)stream,
                (const int32_t*)tok, (const int32_t*)pos, (const int32_t*)cu,
                (const long long*)mult, (const long long*)vsz, (const long long*)off,
                (const int32_t*)cidx, (int32_t*)tail, (int32_t*)ids, (uint32_t)n_seq,
                (uint32_t)heads, (uint32_t)ngram, (uint32_t)(heads / (ngram - 1)), (long long)eos,
                (uint32_t)ids_ld, (uint32_t)(hist ? 1 : 0), (uint32_t)slots);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

int r3d_rad_ngram_hint(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 9) return R3D_NO(RAD_E_INVAL);
    const RadTensor* tok = r3d_opt(a, 0), *pos = r3d_opt(a, 1), *cu = r3d_opt(a, 2),
                    *mult = r3d_opt(a, 3), *vsz = r3d_opt(a, 4), *off = r3d_opt(a, 5),
                    *cidx = r3d_opt(a, 6), *tail = r3d_opt(a, 7), *ids = r3d_opt(a, 8);
    if (!r3d_p(tok) || !r3d_p(pos) || !r3d_p(cu) || !r3d_p(mult) || !r3d_p(vsz) || !r3d_p(off) ||
        !r3d_p(tail) || !r3d_wp(ids))
        return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_i32(tok) || !r3d_is_i32(pos) || !r3d_is_i32(cu) || !r3d_is_i32(tail) ||
        !r3d_is_i32(ids) || (r3d_has(cidx) && !r3d_is_i32(cidx)))
        return R3D_NO(RAD_E_DTYPE);
    if (!(mult->dtype == RAD_I64 && vsz->dtype == RAD_I64 && off->dtype == RAD_I64))
        return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(tok) || !r3d_contig(pos) || !r3d_contig(cu) || !r3d_contig(tail))
        return R3D_NO(RAD_E_STRIDE);

    const long long heads = r3d_geti_or(a, "heads", 0);
    const long long ngram = r3d_geti_or(a, "ngram", 0);
    const long long eos = r3d_geti_or(a, "eos", -1);
    const long long hist = r3d_geti_or(a, "hist", 0);
    if (heads <= 0 || ngram < 2 || eos < 0) return R3D_NO(RAD_E_SHAPE);
    if (ngram > (long long)kNgMaxShift) return R3D_NO(RAD_E_UNSUPPORTED);
    if (heads % (ngram - 1)) return R3D_NO(RAD_E_SHAPE);

    const long long N = r3d_numel(cu) - 1;
    long long rows = 0;
    const long long ids_ld = r3d_rows_pitch(ids, heads, &rows);
    if (ids_ld < 0 || N <= 0 || rows < N) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(mult) < ngram || r3d_numel(vsz) < heads || r3d_numel(off) < heads)
        return R3D_NO(RAD_E_SHAPE);
    if (r3d_has(cidx) && r3d_numel(cidx) < N) return R3D_NO(RAD_E_SHAPE);
    const long long slots = r3d_numel(tail) / (2 * (ngram - 1));
    if (slots < 1) return R3D_NO(RAD_E_SHAPE);

    const int rc = r3d_ngram_hint((long)(uintptr_t)r3d_p(tok), (long)(uintptr_t)r3d_p(pos),
                                  (long)(uintptr_t)r3d_p(cu), (long)(uintptr_t)r3d_p(mult),
                                  (long)(uintptr_t)r3d_p(vsz), (long)(uintptr_t)r3d_p(off),
                                  (long)(uintptr_t)r3d_p(cidx), (long)(uintptr_t)r3d_p(tail),
                                  (long)(uintptr_t)r3d_wp(ids), (int)N, (int)heads, (int)ngram,
                                  (int)eos, (int)(hist ? 1 : 0), (int)ids_ld, (int)slots, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

}  /* extern "C" */
