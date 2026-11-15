/* r3d_moe_iqr.hip -- moe_gemm_q over routed experts stored as IQR weights (r3d_iqr.h).
 *
 *   y[i, n] = a[token(i), :] . w[e(i), n, :], one output row per routed slot i,
 *
 * where the expert of slot i is the run `expert_offset` places it in and the token is
 * `sorted_tok[i] / top_k` (or row i itself under a_order "sorted"). The weights are the same
 * computed trellis the dense row reads (r3d_gemm_iqr_m16.hip): a 16-row x 128-K unit per lane
 * segment, decoded where it is read, one f32 scale a weight row and one f32 factor a 128-K block.
 *
 * The arithmetic per (token, output row, 128-K block) is the dense row's, so the two agree bit for
 * bit: an exact i32 v_dot4 sub-sum, then (sub * a_scale) * fold in f32 (no contraction), blocks
 * ascending, and the weight row's own scale once at the store. This is libr4d's r4d_moe_iqr
 * grouping (r4d_moe_iqr.hip) without its biased-accumulator trick: the same factors in the same
 * order, one extra rounded multiply instead of one fma.
 *
 * ---- two grids ---------------------------------------------------------------------------------
 *
 *   decode   a block owns one 16-row tile of the sorted rows and four 16-column weight tiles (a
 *            wave each). It walks the expert runs the tile reaches, so a tile straddling experts is
 *            correct and an expert with no row costs nothing. Runs are at most 16 rows here.
 *
 *   prefill  above 1.5 routed rows an expert (the decode band), a run map in the row's scratch
 *            gives a block one run of up to 64 rows. A wave decodes its 16x128 weight unit once
 *            per K block and reuses it over the run's four 16-row token tiles, the same residency
 *            as the dense row's `gemm_iqr_resident_kernel`.
 *
 * In both, a wave owns 16 output columns of one weight class part; a run's rows are token rows
 * 0 .. nrows-1, padding rows read the run's last row and write nothing. Rows past `off[n_expert]`
 * (dropped slots) are zeroed: the decode grid zeroes its tile first, the run map carries a tail
 * entry for them.
 *
 * ---- forms -------------------------------------------------------------------------------------
 *
 * W5 builds the computed trellis (`tcq_code=mad`, L = 16) under one f32 scale a row (kRowF32),
 * at 2 to 7 code bits, no tcq_rb. K is a whole number of 128-K blocks, N a whole number of 64
 * columns and of the trellis run; equal classes (N_odd = N, K_odd = K); `parts` divides N in whole
 * 16-row tiles. The other forms, unequal classes and bf16 protected experts are refused by name.
 */
#include <hip/hip_runtime.h>

#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_common.h"
#include "r3d_iqr.h"
#include "r3d_iqr_dec.h"
#include "r3d_launch.h"
#include "r3d_moe_tab.h"

namespace {

using namespace r3d_iqr;
using namespace r3d_iqr_dev;

constexpr uint32_t kNThreads = 128;          /* four waves, one 16-column weight tile each */
constexpr uint32_t kRTB = kNThreads / kWave; /* weight tiles a block */
constexpr uint32_t kCols = 16 * kRTB;        /* output columns a block */
constexpr uint32_t kPitch = 33;              /* dwords a staged 128-K row: 33 mod 32 = 1 */
constexpr uint32_t kDecRows = 16;            /* sorted rows a decode block's tile holds */
constexpr uint32_t kMapRows = 64;            /* sorted rows a run-map entry holds */
constexpr uint32_t kMaxExpert = 512;         /* moe_scatter's ceiling */
constexpr uint32_t kMapTail = 0xffffffffu;   /* a run-map entry owning no expert */

/* The same exact four-pair signed dot the dense row uses: gfx11's dot4 is signed x unsigned, so
 * both operands are negated, which is the signed product. */
__device__ __forceinline__ int dot4_i8(uint32_t a4, uint32_t w4, int acc) {
    return __builtin_amdgcn_sudot4(true, (int)w4, true, (int)a4, acc, false);
}

/* A routed slot's token: f / top_k by the launch's reciprocal (0: top_k is 1). */
__device__ __forceinline__ uint32_t slot_token(uint32_t f, uint32_t inv_k) {
    return inv_k ? __umulhi(f, inv_k) : f;
}

/* ---- the run map ------------------------------------------------------------------------------
 *
 * One count slot, then two u32 per entry (expert, first sorted row). A run of R rows takes
 * ceil(R / kMapRows) entries, and the dropped-slot tail one entry a kMapRows rows; the count in
 * map[0] is clamped to the grid the scratch was sized for, and blocks past it return.
 *
 * Built by one workgroup, as libr4d's map is: the prefix is a DPP scan a wave and one exchange of
 * wave totals, integer so order-free. */
__global__ __launch_bounds__(256) void moe_iqr_blockmap_kernel(
    const int32_t* __restrict__ off, uint32_t n_expert, uint32_t T, uint32_t FM, uint32_t cap,
    uint32_t* __restrict__ map) {
    constexpr uint32_t NT = 256, NW = NT / kWave, kTiles = (kMaxExpert + NT) / NT;
    __shared__ uint32_t s_cnt[kMaxExpert + 1];
    __shared__ int32_t s_wt[kTiles][NW];
    const uint32_t tid = threadIdx.x, lane = tid % kWave, wave = tid / kWave;

    /* Both ends of every expert this thread owns, issued together. */
    int32_t lo[kTiles], hi[kTiles];
#pragma unroll
    for (uint32_t i = 0; i < kTiles; ++i) {
        const uint32_t e = tid + i * NT;
        lo[i] = off[e < n_expert ? e : n_expert];
        hi[i] = off[e + 1 < n_expert ? e + 1 : n_expert];
    }
    const uint32_t live = (uint32_t)off[n_expert];
#pragma unroll
    for (uint32_t i = 0; i < kTiles; ++i) {
        const uint32_t e = tid + i * NT;
        if (e < n_expert) s_cnt[e] = ((uint32_t)(hi[i] - lo[i]) + FM - 1) / FM;
    }
    if (tid == 0) s_cnt[n_expert] = T > live ? (T - live + FM - 1) / FM : 0u;
    __syncthreads();

    uint32_t carry = 0;
    for (uint32_t b = 0, tl = 0; b <= n_expert; b += NT, ++tl) {
        const uint32_t e = b + tid;
        const int32_t v = e <= n_expert ? (int32_t)s_cnt[e] : 0;
        const int32_t x = (int32_t)wave_prefix_u32((uint32_t)v, (int)lane);
        if (lane == kWave - 1) s_wt[tl][wave] = x;
        __syncthreads();
        int32_t below = 0, tot = 0;
#pragma unroll
        for (uint32_t k = 0; k < NW; ++k) {
            const int32_t c = s_wt[tl][k];
            below += (k < wave) ? c : 0;
            tot += c;
        }
        if (e <= n_expert) s_cnt[e] = carry + (uint32_t)(below + x - v);
        carry += (uint32_t)tot;
    }
    if (tid == 0) map[0] = carry < cap ? carry : cap;
    __syncthreads();

#pragma unroll
    for (uint32_t i = 0; i < kTiles; ++i) {
        const uint32_t e = tid + i * NT;
        if (e >= n_expert) continue;
        const uint32_t run = (uint32_t)(hi[i] - lo[i]);
        uint32_t o = s_cnt[e];
        for (uint32_t r = 0; r < run; r += FM, ++o) {
            if (o >= cap) break;
            map[1 + 2 * o + 0] = e;
            map[1 + 2 * o + 1] = (uint32_t)lo[i] + r;
        }
    }
    if (tid == 0) {
        uint32_t o = s_cnt[n_expert];
        for (uint32_t r = live; r < T; r += FM, ++o) {
            if (o >= cap) break;
            map[1 + 2 * o + 0] = kMapTail;
            map[1 + 2 * o + 1] = r;
        }
    }
}

/* ---- the GEMM ---------------------------------------------------------------------------------
 *
 * `NT` token tiles of 16 rows a run (1 in the decode band, 4 under the run map). `tw` carries both
 * classes' weight and scale tables; expert e is entry e/2 of class e & 1. */
template <class F, int NT, bool MAP>
__global__ __launch_bounds__(kNThreads) R3D_WAVES(4) void moe_iqr_kernel(
    const int8_t* __restrict__ a, const float* __restrict__ as, R3dMoeTabs tw,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    const uint32_t* __restrict__ map, uint16_t* __restrict__ y, int T, int N, int K, int a_ld,
    int as_ld, int y_ld, int n_expert, uint32_t inv_k, int a_sorted, int ash, int parts,
    int run_tiles) {
    constexpr uint32_t RT = MAP ? kMapRows : kDecRows;
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const uint32_t nkb = (uint32_t)(K / 128);

    /* This wave's 16-column weight tile. With equal classes `fc = fy = N / parts`, so the tile's
     * weight row is its output column; `part` keeps the mapping for a split N. */
    const uint32_t ty = (uint32_t)blockIdx.y * kRTB + wave;
    const uint32_t ft = (uint32_t)(N / parts) / 16u;
    const uint32_t part = ty / ft;
    const uint32_t tin = ty - part * ft;
    const uint32_t fy = (uint32_t)(N / parts);
    const uint32_t row0 = part * fy + tin * 16u;
    const uint32_t out_col = row0;
    const uint32_t q = lane & 15u;          /* the weight row (and output column) this lane owns */
    const uint32_t half = lane >> 4;        /* the token row parity of its first pair */

    __shared__ uint32_t s_w[kRTB][16 * kPitch];
    __shared__ uint32_t s_a[RT * kPitch];
    __shared__ int32_t s_rs[kDecRows], s_re[kDecRows], s_e[kDecRows];
    __shared__ int32_t s_nruns;
    __shared__ int32_t s_rtok[RT];

    /* One run's GEMM. All of the block's threads call it, so its barriers are uniform. */
    auto process = [&](uint32_t e, uint32_t rs, uint32_t re) {
        const uint32_t nrows = re - rs;
        const uint32_t cls = e & 1u, e2 = e >> 1;
        Ctx c;
        c.W = r3d_tab_at(tw.w[cls], (int)e2);
        c.S = r3d_tab_at(tw.s[cls], (int)e2);
        c.tab = nullptr;
        c.N = (uint32_t)N;
        c.nkb = nkb;
        c.run_tiles = (uint32_t)run_tiles;

        /* The run's token rows; padding rows read the run's last row. */
        if (tid < RT) {
            const uint32_t m = tid;
            const uint32_t sr = rs + (m < nrows ? m : nrows - 1u);
            const int32_t f = sorted[sr];
            s_rtok[m] = f < 0 ? -1 : (a_sorted ? (int32_t)sr : (int32_t)slot_token((uint32_t)f,
                                                                                    inv_k));
        }
        __syncthreads();

        float acc[NT][8];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[j][i] = 0.0f;

        const float rscw = row_scale<F>(c, row0 + q);
        for (uint32_t kb = 0; kb < nkb; ++kb) {
            /* The run's activation rows, shared by the block's four weight tiles. */
            for (uint32_t i = tid; i < RT * 32u; i += kNThreads) {
                const uint32_t m = i >> 5, d = i & 31u;
                int32_t tk = s_rtok[m];
                if (tk < 0) tk = 0;
                s_a[m * kPitch + d] =
                    *(const uint32_t*)(a + (size_t)tk * a_ld + (size_t)kb * 128u + 4u * d);
            }
            /* This wave's decoded 16x128 weight unit. */
            const typename Dec<F>::Raw r = Dec<F>::fetch(c, row0 / 16u, kb, lane, true);
            uint32_t w[16];
            Dec<F>::rows(r, c, 1.0f, w);
#pragma unroll
            for (uint32_t qq = 0; qq < 16; ++qq) s_w[wave][qq * kPitch + lane] = w[qq];
            __syncthreads();

            const float sb = fold_f32<F>(c, row0 + q, kb);
            const uint32_t* wp = s_w[wave] + q * kPitch;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t m = (uint32_t)j * 16u + half + 2u * (uint32_t)i;
                    if (m >= nrows) continue;
                    int sub = 0;
                    const uint32_t* ap = s_a + m * kPitch;
#pragma unroll
                    for (int c4 = 0; c4 < 32; ++c4) sub = dot4_i8(ap[c4], wp[c4], sub);
                    const int32_t tk = s_rtok[m];
                    const float sa = as[(size_t)(tk < 0 ? 0 : tk) * as_ld +
                                        (size_t)(kb >> (uint32_t)ash)];
                    acc[j][i] += (float)sub * sa * sb;
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t m = (uint32_t)j * 16u + half + 2u * (uint32_t)i;
                if (m >= nrows || s_rtok[m] < 0) continue;
                y[(size_t)(rs + m) * y_ld + out_col + q] = f32_to_bf16(acc[j][i] * rscw);
            }
        }
    };

    if constexpr (MAP) {
        if (blockIdx.x >= map[0]) return;
        const uint32_t e = map[1 + 2 * blockIdx.x];
        const uint32_t rs = map[2 + 2 * blockIdx.x];
        if (e == kMapTail) {
            /* Dropped slots: this wave's columns of the entry's rows. */
            const uint32_t rT = rs + kMapRows < (uint32_t)T ? rs + kMapRows : (uint32_t)T;
            for (uint32_t i = lane; i < (rT - rs) * 16u; i += kWave) {
                const uint32_t rr = i >> 4, cc = i & 15u;
                y[(size_t)(rs + rr) * y_ld + out_col + cc] = 0;
            }
            return;
        }
        uint32_t oe = (uint32_t)off[e + 1];
        if (oe > (uint32_t)T) oe = (uint32_t)T;
        if (oe > rs + kMapRows) oe = rs + kMapRows;
        if (oe > rs) process(e, rs, oe);
    } else {
        const uint32_t r0 = (uint32_t)blockIdx.x * kDecRows;
        if (r0 >= (uint32_t)T) return;
        const uint32_t r1 = r0 + kDecRows < (uint32_t)T ? r0 + kDecRows : (uint32_t)T;

        /* Every row this block's columns own starts at zero; the runs below overwrite the live
         * ones, and dropped slots stay zero. */
        for (uint32_t i = lane; i < kDecRows * 16u; i += kWave) {
            const uint32_t rr = i >> 4, cc = i & 15u;
            if (r0 + rr < (uint32_t)T) y[(size_t)(r0 + rr) * y_ld + out_col + cc] = 0;
        }

        const uint32_t live = (uint32_t)off[n_expert];
        if (tid == 0) {
            /* The first expert whose run ends past r0, then the runs in the tile. */
            uint32_t lo = 0, hi = (uint32_t)n_expert;
            while (lo < hi) {
                const uint32_t mid = (lo + hi) >> 1;
                if ((uint32_t)off[mid + 1] <= r0) lo = mid + 1;
                else hi = mid;
            }
            uint32_t i = r0, e = lo, nr = 0;
            while (i < r1 && i < live) {
                while (e < (uint32_t)n_expert && (uint32_t)off[e + 1] <= i) ++e;
                if (e >= (uint32_t)n_expert) break;
                const uint32_t oa = (uint32_t)off[e], ob = (uint32_t)off[e + 1];
                const uint32_t rsa = oa > i ? oa : i;
                const uint32_t reb = ob < r1 ? ob : r1;
                if (reb > rsa) {
                    s_rs[nr] = (int32_t)rsa;
                    s_re[nr] = (int32_t)reb;
                    s_e[nr] = (int32_t)e;
                    ++nr;
                }
                i = reb;
                ++e;
            }
            s_nruns = (int32_t)nr;
        }
        __syncthreads();
        const int nruns = s_nruns;
        for (int run = 0; run < nruns; ++run) process((uint32_t)s_e[run], (uint32_t)s_rs[run],
                                                      (uint32_t)s_re[run]);
    }
}

/* ---- the launcher ----------------------------------------------------------------------------- */

template <class F, int NT, bool MAP>
int launch_form(const int8_t* a, const float* as, R3dMoeTabs tw, const int32_t* sorted,
                const int32_t* off, uint32_t* map, uint16_t* y, int T, int N, int K, int a_ld,
                int as_ld, int y_ld, int n_expert, uint32_t inv_k, int a_sorted, int ash,
                int parts, int run_tiles, uint32_t gx, long stream) {
    const dim3 grid(gx, (unsigned)(N / (int)kCols)), block(kNThreads);
    R3D_KLAUNCH(R3D_KNAME((moe_iqr_kernel<F, NT, MAP>)), grid, block, (hipStream_t)stream, a, as,
                tw, sorted, off, map, y, T, N, K, a_ld, as_ld, y_ld, n_expert, inv_k, a_sorted,
                ash, parts, run_tiles);
    return hipGetLastError() == hipSuccess ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

int form_supported(const Form& f) {
    /* W5: the computed trellis, L = 16, one f32 scale a row, no tcq_rb. */
    return f.kind == kTrellisMad && f.L == 16 && f.lrb == 0 && f.scale == kRowF32 &&
           f.bits >= 2 && f.bits <= 7;
}

/* The run-map scratch: [count][2 * cap] u32, cap = n_expert + ceil(T / 64) + 1. */
int64_t map_bytes(long long T, long long n_expert) {
    const long long cap = n_expert + (T + kMapRows - 1) / kMapRows + 1;
    const long long bytes = (1 + 2 * cap) * (long long)sizeof(uint32_t);
    return (bytes + 255) & ~255ll;
}

int64_t map_cap(long long T, long long n_expert) {
    return n_expert + (T + kMapRows - 1) / kMapRows + 1;
}

}  // namespace

using namespace r3d_iqr;
using namespace r3d_iqr_dev;

extern "C" int r3d_rad_moe_gemm_iqr(const RadArgs* a, RadStream s) {
    if (!a) return R3D_NO(RAD_E_INVAL);
    R3dMoeCall c;
    int rc = r3d_moe_call(a, true, &c);
    if (rc != RAD_OK) return rc;

    if (c.a->dtype != RAD_I8 || c.a_scale->dtype != RAD_F32 || c.y->dtype != RAD_BF16)
        return R3D_NO(RAD_E_DTYPE);
    if (c.sorted->dtype != RAD_I32 || c.off->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);
    if (!is_weight_dtype(c.w->dtype) || !is_scale_dtype(c.w_scale->dtype))
        return R3D_NO(RAD_E_DTYPE);

    const Form f = form(c.w->dtype);
    if (!form_supported(f) || kernel_word(c.w->dtype) != kernel_word(c.w_scale->dtype))
        return R3D_NO(RAD_E_DTYPE);
    if (c.n_expert > (long long)kMaxExpert) return R3D_NO(RAD_E_UNSUPPORTED);
    if (c.n_expert >= 2 &&
        (c.w_odd->dtype != c.w->dtype || c.w_odd_scale->dtype != c.w_scale->dtype))
        return R3D_NO(RAD_E_DTYPE);

    const long long N = c.N, K = c.K, T = c.T, E = c.n_expert;
    if (T == 0) return RAD_OK;   /* an empty pass launches nothing */
    const long long group = r3d_geti_or(a, "group", 128);
    const long long a_group = r3d_geti_or(a, "a_group", 128);
    if (group != 128) return R3D_NO(RAD_E_SHAPE);
    if (a_group != 128 && a_group != 256) return R3D_NO(RAD_E_UNSUPPORTED);
    if (N <= 0 || K <= 0) return R3D_NO(RAD_E_SHAPE);
    if (K % 128 || K % a_group) return R3D_NO(RAD_E_SHAPE);
    if (N % (long long)kCols) return R3D_NO(RAD_E_SHAPE);
    const long long parts = r3d_geti_or(a, "parts", 1);
    if (parts < 1 || parts > N / 16 || N % (16 * parts)) return R3D_NO(RAD_E_SHAPE);
    const long long run = 1ll << f.lrun;
    if (f.lrun < 4 || f.lrun > 15 || N % run) return R3D_NO(RAD_E_SHAPE);
    if (T > 0x7FFFFFFFll) return R3D_NO(RAD_E_SHAPE);
    if (N > 0x7FFFFFFFll || K > 0x7FFFFFFFll) return R3D_NO(RAD_E_SHAPE);

    /* The activation and its scales must carry every token the slots name. The contract's own
     * check uses T / top_k, which is one row short when top_k does not divide T. */
    long long ar = 0, ak = 0, asr = 0, asc = 0;
    if (r3d_rows_of(c.a, &ar, &ak) != RAD_OK || r3d_rows_of(c.a_scale, &asr, &asc) != RAD_OK)
        return R3D_NO(RAD_E_SHAPE);
    const long long trows = c.a_sorted ? T : (T + c.top_k - 1) / c.top_k;
    if (ar < trows || asr < trows || asc < K / a_group) return R3D_NO(RAD_E_SHAPE);

    /* IQR weights are pointer tables (libr4d's r4d_moe_iqr takes the same contract): an entry is a
     * whole layout buffer with its own aux and scales, so a stacked plane's stride[0] would not be
     * the entry's byte size and is refused rather than read at the wrong offset. */
    if (c.w->stride[0] != 0 || c.w_scale->stride[0] != 0 ||
        (E >= 2 && (c.w_odd->stride[0] != 0 || c.w_odd_scale->stride[0] != 0)))
        return R3D_NO(RAD_E_UNSUPPORTED);

    R3dMoeTabs tw;
    rc = r3d_moe_q_tabs(c, c.w->dtype, (int)f.bits, c.w_scale->dtype, 32, 1, 1, &tw);
    if (rc != RAD_OK) return rc;

    const long long a_ld64 = c.a->stride[0], as_ld64 = c.a_scale->stride[0];
    const long long y_ld64 = c.y->stride[0];
    if (a_ld64 > 0x7FFFFFFFll || as_ld64 > 0x7FFFFFFFll || y_ld64 > 0x7FFFFFFFll)
        return R3D_NO(RAD_E_STRIDE);
    const int a_ld = (int)a_ld64, as_ld = (int)as_ld64;
    const int y_ld = (int)y_ld64;
    if (a_ld < K || a_ld % 4 || as_ld < K / a_group || y_ld < N)
        return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(r3d_p(c.a), 4) || !r3d_aligned(r3d_p(c.a_scale), 4) ||
        !r3d_aligned(r3d_p(c.y), 2))
        return R3D_NO(RAD_E_ALIGN);

    /* The band: a run of a row or two is the decode grid; longer runs take the map. */
    const bool band = 2 * T <= 3 * E;
    uint32_t* map = nullptr;
    uint32_t gx = (uint32_t)((T + kDecRows - 1) / kDecRows);
    if (!band) {
        const long long cap = map_cap(T, E);
        const long long need = map_bytes(T, E);
        if (!a->scratch || a->scratch_bytes < need) return R3D_NO(RAD_E_SCRATCH);
        if (!r3d_aligned(a->scratch, 4)) return R3D_NO(RAD_E_ALIGN);
        map = (uint32_t*)a->scratch;
        R3D_KLAUNCH(moe_iqr_blockmap_kernel, dim3(1), dim3(256), (hipStream_t)s,
                    (const int32_t*)r3d_p(c.off), (uint32_t)E, (uint32_t)T, (uint32_t)kMapRows,
                    (uint32_t)cap, map);
        if (hipGetLastError() != hipSuccess) return R3D_NO(RAD_E_DEVICE);
        gx = (uint32_t)cap;
    }

    const uint32_t inv_k = c.top_k == 1 ? 0u : (uint32_t)(((uint64_t)1 << 32) /
                                                         (uint64_t)c.top_k + 1u);
    const int run_tiles = (int)(run / 16);
    const int ash = a_group == 256 ? 1 : 0;
    const int8_t* pa = (const int8_t*)r3d_p(c.a);
    const float* psa = (const float*)r3d_p(c.a_scale);
    const int32_t* ps = (const int32_t*)r3d_p(c.sorted);
    const int32_t* po = (const int32_t*)r3d_p(c.off);
    uint16_t* py = (uint16_t*)r3d_wp(c.y);

    if (band) {
#define R3D_MOE_IQR_DEC(B)                                                                        \
    do {                                                                                          \
        using F = FormT<kTrellisMad, B, 0, 16, 0, kRowF32>;                                       \
        return launch_form<F, 1, false>(pa, psa, tw, ps, po, nullptr, py, (int)T, (int)N, (int)K,  \
                                        a_ld, as_ld, y_ld, (int)E, inv_k, c.a_sorted, ash,        \
                                        (int)parts, run_tiles, gx, (long)s);                      \
    } while (0)
        switch (f.bits) {
            case 2: R3D_MOE_IQR_DEC(2);
            case 3: R3D_MOE_IQR_DEC(3);
            case 4: R3D_MOE_IQR_DEC(4);
            case 5: R3D_MOE_IQR_DEC(5);
            case 6: R3D_MOE_IQR_DEC(6);
            case 7: R3D_MOE_IQR_DEC(7);
            default: return R3D_NO(RAD_E_UNSUPPORTED);
        }
#undef R3D_MOE_IQR_DEC
    } else {
#define R3D_MOE_IQR_MAP(B)                                                                        \
    do {                                                                                          \
        using F = FormT<kTrellisMad, B, 0, 16, 0, kRowF32>;                                       \
        return launch_form<F, 4, true>(pa, psa, tw, ps, po, map, py, (int)T, (int)N, (int)K,      \
                                       a_ld, as_ld, y_ld, (int)E, inv_k, c.a_sorted, ash,         \
                                       (int)parts, run_tiles, gx, (long)s);                       \
    } while (0)
        switch (f.bits) {
            case 2: R3D_MOE_IQR_MAP(2);
            case 3: R3D_MOE_IQR_MAP(3);
            case 4: R3D_MOE_IQR_MAP(4);
            case 5: R3D_MOE_IQR_MAP(5);
            case 6: R3D_MOE_IQR_MAP(6);
            case 7: R3D_MOE_IQR_MAP(7);
            default: return R3D_NO(RAD_E_UNSUPPORTED);
        }
#undef R3D_MOE_IQR_MAP
    }
}

/* The run map's bytes, or 0 in the decode band. Sized from M * top_k, as the other grouped rows
 * size theirs: the hook runs at declare, where the sorted row count is not yet an operand. */
extern "C" int64_t r3d_rad_moe_gemm_iqr_scratch(const RadArgs* a) {
    if (!a) return 0;
    long long n_expert = 0, top_k = 0, M = 0;
    if (!rad_args_geti(a, "n_expert", &n_expert) || !rad_args_geti(a, "top_k", &top_k) ||
        !rad_args_geti(a, "M", &M))
        return 0;
    if (n_expert <= 0 || n_expert > (long long)kMaxExpert || top_k <= 0 || M <= 0) return 0;
    if (M > 0x7FFFFFFFll / top_k) return 0;   /* the launch refuses a T past int32 */
    const long long T = M * top_k;
    if (2 * T <= 3 * n_expert) return 0;
    return map_bytes(T, n_expert);
}
