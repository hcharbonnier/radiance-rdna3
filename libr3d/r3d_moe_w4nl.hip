/* r3d_moe_w4nl.hip -- moe_gemm_q over routed experts stored in the `w4nl64a8i8h` codebook form.
 *
 *   y[i, n] = a[token(i), :] . w[e(i), n, :], one output row per routed slot i,
 *
 * where the expert of slot i is the run `expert_offset` places it in and the token is
 * `sorted_tok[i] / top_k` (or row i itself under a_order "sorted"). Each expert is a container's
 * codebook weight (r3d_moe_w4nl.h): u4 codes indexing a sixteen-entry f32 table, an E4M3 scale per
 * (row, 64 of K), and one fixed f32 level (2^-13) for the whole weight. A weight element is
 * `table[code] * scale * level`.
 *
 * The row reads the same stored operands libr4d's moe_gemm_w4a8i8 does: the codes and the table are
 * one joined buffer, the two scale planes another (abi/rad_planes.h). The layout hook
 * (r3d_iqr_layout.cpp) joins them and refuses a table or second level that is not the w4nl one.
 *
 * ---- the arithmetic ----------------------------------------------------------------------------
 *
 * Every w4nl level is an exact multiple of 1/8, so a code is decoded to `table8 = table * 8` and
 * the 1/8 is folded into the group's scale. Each 64-K group's sub-sum is then an exact i32, and the
 * group is folded as
 *
 *     acc += (sub * a_scale[token][group]) * (scale_e4m3 * level * 1/8)
 *
 * with the two multiplies and the add rounded separately (no fma), groups ascending, exactly
 * libref's `quant_dot` chain. The reference's f32 sub-sum of `a * table` is exact for the same
 * reason, and moving a factor of eight across the two multiplies is exact, so the device result is
 * bit-exact against libref on every shape. libr4d sums a 128-K block's two sub-sums before the
 * activation scale (`(sub0*sb0 + sub1*sb1) * sa`); that differs in the last bits and is the bounded
 * difference ADR-4 permits. This row takes libref's stronger order, which also makes the reference
 * test and the oracle agree bit for bit.
 *
 * ---- two grids ---------------------------------------------------------------------------------
 *
 *   decode   a block owns one 16-row tile of the sorted rows and four 16-column weight tiles (a
 *            wave each). It walks the expert runs the tile reaches, so a tile straddling experts is
 *            correct and an expert with no row costs nothing. A run is at most 16 rows here.
 *
 *   prefill  above 1.5 routed rows an expert (the decode band), a run map in the row's scratch
 *            gives a block one run of up to 64 rows. A wave decodes its 16x128 weight unit once per
 *            K block and reuses it over the run's four 16-row token tiles.
 *
 * In both, a wave owns 16 output columns of one weight class part; a run's rows are token rows
 * 0 .. nrows-1, padding rows read the run's last row and write nothing. Rows past `off[n_expert]`
 * (dropped slots) are zeroed: the decode grid zeroes its tile first, the run map carries a tail
 * entry for them.
 *
 * ---- geometry ----------------------------------------------------------------------------------
 *
 * N a multiple of 64, K a multiple of 128, `group` 128 (the op's base scale group; the weight's
 * scales are one per 64), `a_group` 128 or 256, equal classes (N_odd = N, K_odd = K), `parts`
 * dividing N in whole 64-column blocks. The two model geometries are N=2560 K=640 (the down
 * projections, parts 1) and N=1280 K=2560 (the MTP head, parts 2). bf16 protected experts are not
 * served here; the bf16 row serves them.
 */
#include <hip/hip_runtime.h>

#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include "r3d_moe_tab.h"
#include "r3d_moe_w4nl.h"

namespace {

using namespace r3d_w4nl;

constexpr uint32_t kNThreads = 128;          /* four waves, one 16-column weight tile each */
constexpr uint32_t kRTB = kNThreads / kWave; /* weight tiles a block */
constexpr uint32_t kCols = 16 * kRTB;        /* output columns a block */
constexpr uint32_t kDecRows = 16;            /* sorted rows a decode block's tile holds */
constexpr uint32_t kMapRows = 64;            /* sorted rows a run-map entry holds */
constexpr uint32_t kMaxExpert = 512;         /* moe_scatter's ceiling */
constexpr uint32_t kMapTail = 0xffffffffu;   /* a run-map entry owning no expert */
constexpr uint32_t kPitchA = 132;            /* bytes a staged 128-K int8 activation row */
constexpr uint32_t kPitchW = 136;            /* int16 a staged 128-K weight row */

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
 * wave totals, integer so order-free.
 */
__global__ __launch_bounds__(256) void moe_w4nl_blockmap_kernel(
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
 * classes' code/table and scale/level tables; expert e is entry e/2 of class e & 1.
 */
template <int NT, bool MAP>
__global__ __launch_bounds__(kNThreads) R3D_WAVES(4) void moe_w4nl_kernel(
    const int8_t* __restrict__ a, const float* __restrict__ as, R3dMoeTabs tw,
    const int32_t* __restrict__ sorted, const int32_t* __restrict__ off,
    const uint32_t* __restrict__ map, uint16_t* __restrict__ y, int T, int N, int K, int a_ld,
    int as_ld, int y_ld, int n_expert, uint32_t inv_k, int a_sorted, int ash, int parts) {
    constexpr uint32_t RT = MAP ? kMapRows : kDecRows;
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const uint32_t nkb = (uint32_t)(K / 128);
    const uint32_t srow = (uint32_t)(K / (int)kGroup);
    const uint32_t wrow_bytes = (uint32_t)(K / 2);
    const int64_t codes_at = codes_off();
    const int64_t table_at = table_off((int64_t)N, (int64_t)K);
    const int64_t scale_at = scale_off();
    const int64_t level_at = level_off((int64_t)N, (int64_t)K);

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

    __shared__ int8_t s_a[RT * kPitchA];
    __shared__ int16_t s_w[kRTB][16 * kPitchW];
    __shared__ int16_t s_tab[16];
    __shared__ int32_t s_rtok[RT];
    __shared__ int32_t s_rs[kDecRows], s_re[kDecRows], s_e[kDecRows];
    __shared__ int32_t s_nruns;

    /* One run's GEMM. All of the block's threads call it, so its barriers are uniform. */
    auto process = [&](uint32_t e, uint32_t rs, uint32_t re) {
        const uint32_t nrows = re - rs;
        const uint32_t cls = e & 1u, e2 = e >> 1;
        const uint8_t* wb = r3d_tab_at(tw.w[cls], (int)e2);
        const uint8_t* sb = r3d_tab_at(tw.s[cls], (int)e2);
        const uint8_t* codes = wb + codes_at;
        const float* tab = (const float*)(wb + table_at);
        const uint8_t* sbytes = sb + scale_at;
        const float sublevel = (*(const float*)(sb + level_at)) * kEighth;

        /* The run's token rows; padding rows read the run's last row. */
        if (tid < RT) {
            const uint32_t m = tid;
            const uint32_t sr = rs + (m < nrows ? m : nrows - 1u);
            const int32_t f = sorted[sr];
            s_rtok[m] = f < 0 ? -1 : (a_sorted ? (int32_t)sr
                                               : (int32_t)slot_token((uint32_t)f, inv_k));
        }
        /* The table's levels in eighths, read from the stored table once a run. The w4nl entries
         * are exact multiples of 1/8, so this conversion is exact. */
        if (tid < 16) s_tab[tid] = (int16_t)__float2int_rn(tab[tid] * 8.0f);
        __syncthreads();

        float acc[NT][8];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[j][i] = 0.0f;

        for (uint32_t kb = 0; kb < nkb; ++kb) {
            /* The run's activation rows, shared by the block's four weight tiles. */
            for (uint32_t i = tid; i < RT * 32u; i += kNThreads) {
                const uint32_t m = i >> 5, d = i & 31u;
                int32_t tk = s_rtok[m];
                if (tk < 0) tk = 0;
                *(uint32_t*)(s_a + m * kPitchA + 4u * d) =
                    *(const uint32_t*)(a + (size_t)tk * a_ld + (size_t)kb * 128u + 4u * d);
            }
            /* This wave's 16x128 weight unit, decoded through the table. */
            {
                const uint32_t qq = lane & 15u, hf = lane >> 4;
                const uint8_t* cseg =
                    codes + (size_t)(row0 + qq) * wrow_bytes + (size_t)kb * 64u + hf * 32u;
                int16_t* wp = s_w[wave] + qq * kPitchW + hf * 64u;
#pragma unroll
                for (uint32_t j = 0; j < 32u; ++j) {
                    const uint8_t byte = cseg[j];
                    wp[2u * j] = s_tab[byte & 15u];
                    wp[2u * j + 1u] = s_tab[byte >> 4];
                }
            }
            __syncthreads();

            for (uint32_t sub = 0; sub < 2u; ++sub) {
                const uint32_t g = kb * 2u + sub;
                const float sw =
                    e4m3_to_f32(sbytes[(size_t)(row0 + q) * srow + g]) * sublevel;
                const uint32_t sai = g >> ash;
#pragma unroll
                for (int j = 0; j < NT; ++j) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const uint32_t m = (uint32_t)j * 16u + half + 2u * (uint32_t)i;
                        if (m >= nrows) continue;
                        const int8_t* ap = s_a + m * kPitchA + sub * 64u;
                        const int16_t* wp = s_w[wave] + q * kPitchW + sub * 64u;
                        int sum = 0;
#pragma unroll
                        for (uint32_t k = 0; k < 64u; ++k) sum += (int)ap[k] * (int)wp[k];
                        const int32_t tk = s_rtok[m];
                        const float sa =
                            as[(size_t)(tk < 0 ? 0 : tk) * as_ld + sai];
                        acc[j][i] += (float)sum * sa * sw;
                    }
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
                y[(size_t)(rs + m) * y_ld + out_col + q] = f32_to_bf16(acc[j][i]);
            }
        }
        __syncthreads();
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
        for (int run = 0; run < nruns; ++run)
            process((uint32_t)s_e[run], (uint32_t)s_rs[run], (uint32_t)s_re[run]);
    }
}

/* ---- the launcher ----------------------------------------------------------------------------- */

/* The class tables of a moe_gemm_q call: pointer arrays of joined buffers, as the loader stores a
 * codebook weight. A stacked plane cannot be read here: an entry is a whole joined buffer whose
 * internal offsets the kernel derives, so a rank-3 stack's stride[0] is not the entry's byte size
 * and is refused rather than read at the wrong offset. */
int w4nl_tabs(const RadTensor* w, const RadTensor* ws, long long entries, long long N,
              long long K, long long nkb, R3dTab* tw, R3dTab* ts) {
    if (!w || !w->data || !ws || !ws->data || w->rank != 3 || ws->rank != 3)
        return R3D_NO(RAD_E_SHAPE);
    if (w->stride[0] != 0 || ws->stride[0] != 0) return R3D_NO(RAD_E_UNSUPPORTED);
    if (w->shape[0] < entries || w->shape[1] < N || w->shape[2] < K)
        return R3D_NO(RAD_E_SHAPE);
    if (ws->shape[0] < entries || ws->shape[1] < N || ws->shape[2] < nkb)
        return R3D_NO(RAD_E_SHAPE);
    tw->tab = (const void* const*)w->data;
    tw->s0 = 0;
    tw->ld = (int)K;
    ts->tab = (const void* const*)ws->data;
    ts->s0 = 0;
    ts->ld = (int)nkb;
    return RAD_OK;
}

int launch_form(int NT_map, const int8_t* a, const float* as, R3dMoeTabs tw,
                const int32_t* sorted, const int32_t* off, uint32_t* map, uint16_t* y, int T,
                int N, int K, int a_ld, int as_ld, int y_ld, int n_expert, uint32_t inv_k,
                int a_sorted, int ash, int parts, uint32_t gx, long stream) {
    const dim3 grid(gx, (unsigned)(N / (int)kCols)), block(kNThreads);
    if (!NT_map) {
        R3D_KLAUNCH(R3D_KNAME((moe_w4nl_kernel<1, false>)), grid, block, (hipStream_t)stream, a,
                    as, tw, sorted, off, map, y, T, N, K, a_ld, as_ld, y_ld, n_expert, inv_k,
                    a_sorted, ash, parts);
    } else {
        R3D_KLAUNCH(R3D_KNAME((moe_w4nl_kernel<4, true>)), grid, block, (hipStream_t)stream, a,
                    as, tw, sorted, off, map, y, T, N, K, a_ld, as_ld, y_ld, n_expert, inv_k,
                    a_sorted, ash, parts);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : R3D_NO(RAD_E_DEVICE);
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

extern "C" int r3d_rad_moe_gemm_w4nl(const RadArgs* a, RadStream s) {
    if (!a) return R3D_NO(RAD_E_INVAL);
    R3dMoeCall c;
    int rc = r3d_moe_call(a, true, &c);
    if (rc != RAD_OK) return rc;

    if (c.a->dtype != RAD_I8 || c.a_scale->dtype != RAD_F32 || c.y->dtype != RAD_BF16)
        return R3D_NO(RAD_E_DTYPE);
    if (c.sorted->dtype != RAD_I32 || c.off->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);
    if (c.w->dtype != RAD_U4 || c.w_scale->dtype != RAD_F8E4M3) return R3D_NO(RAD_E_DTYPE);
    if (c.n_expert >= 2 &&
        (c.w_odd->dtype != c.w->dtype || c.w_odd_scale->dtype != c.w_scale->dtype))
        return R3D_NO(RAD_E_DTYPE);
    if (c.n_expert > (long long)kMaxExpert) return R3D_NO(RAD_E_UNSUPPORTED);

    const long long N = c.N, K = c.K, T = c.T, E = c.n_expert;
    if (T == 0) return RAD_OK;   /* an empty pass launches nothing */
    const long long group = r3d_geti_or(a, "group", 128);
    const long long a_group = r3d_geti_or(a, "a_group", 128);
    if (group != r3d_w4nl::kBaseGroup) return R3D_NO(RAD_E_SHAPE);
    if (a_group != 128 && a_group != 256) return R3D_NO(RAD_E_UNSUPPORTED);
    if (N <= 0 || K <= 0) return R3D_NO(RAD_E_SHAPE);
    if (K % 128 || K % a_group) return R3D_NO(RAD_E_SHAPE);
    if (N % (long long)kCols) return R3D_NO(RAD_E_SHAPE);
    const long long parts = r3d_geti_or(a, "parts", 1);
    if (parts < 1 || (N / parts) % (long long)kCols) return R3D_NO(RAD_E_SHAPE);
    if (T > 0x7FFFFFFFll || N > 0x7FFFFFFFll || K > 0x7FFFFFFFll) return R3D_NO(RAD_E_SHAPE);

    /* The activation and its scales must carry every token the slots name. The contract's own
     * check uses T / top_k, which is one row short when top_k does not divide T. */
    long long ar = 0, ak = 0, asr = 0, asc = 0;
    if (r3d_rows_of(c.a, &ar, &ak) != RAD_OK || r3d_rows_of(c.a_scale, &asr, &asc) != RAD_OK)
        return R3D_NO(RAD_E_SHAPE);
    const long long trows = c.a_sorted ? T : (T + c.top_k - 1) / c.top_k;
    if (ar < trows || asr < trows || asc < K / a_group) return R3D_NO(RAD_E_SHAPE);

    const long long n_even = (E + 1) / 2, n_odd = E / 2;
    R3dMoeTabs tw;
    rc = w4nl_tabs(c.w, c.w_scale, n_even, N, K, K / (long long)r3d_w4nl::kGroup, &tw.w[0], &tw.s[0]);
    if (rc != RAD_OK) return rc;
    if (n_odd == 0) {
        tw.w[1] = tw.w[0];
        tw.s[1] = tw.s[0];
    } else {
        rc = w4nl_tabs(c.w_odd, c.w_odd_scale, n_odd, N, K, K / (long long)r3d_w4nl::kGroup, &tw.w[1],
                       &tw.s[1]);
        if (rc != RAD_OK) return rc;
    }

    const long long a_ld64 = c.a->stride[0], as_ld64 = c.a_scale->stride[0];
    const long long y_ld64 = c.y->stride[0];
    if (a_ld64 > 0x7FFFFFFFll || as_ld64 > 0x7FFFFFFFll || y_ld64 > 0x7FFFFFFFll)
        return R3D_NO(RAD_E_STRIDE);
    const int a_ld = (int)a_ld64, as_ld = (int)as_ld64, y_ld = (int)y_ld64;
    if (a_ld < K || a_ld % 4 || as_ld < K / a_group || y_ld < N) return R3D_NO(RAD_E_STRIDE);
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
        R3D_KLAUNCH(moe_w4nl_blockmap_kernel, dim3(1), dim3(256), (hipStream_t)s,
                    (const int32_t*)r3d_p(c.off), (uint32_t)E, (uint32_t)T, (uint32_t)kMapRows,
                    (uint32_t)cap, map);
        if (hipGetLastError() != hipSuccess) return R3D_NO(RAD_E_DEVICE);
        gx = (uint32_t)cap;
    }

    const uint32_t inv_k = c.top_k == 1 ? 0u : (uint32_t)(((uint64_t)1 << 32) /
                                                         (uint64_t)c.top_k + 1u);
    const int ash = a_group == 256 ? 2 : 1;
    const uint8_t* pa = (const uint8_t*)r3d_p(c.a);
    const float* psa = (const float*)r3d_p(c.a_scale);
    const int32_t* ps = (const int32_t*)r3d_p(c.sorted);
    const int32_t* po = (const int32_t*)r3d_p(c.off);
    uint16_t* py = (uint16_t*)r3d_wp(c.y);

    return launch_form(band ? 0 : 1, (const int8_t*)pa, psa, tw, ps, po, map, py, (int)T, (int)N,
                       (int)K, a_ld, as_ld, y_ld, (int)E, inv_k, c.a_sorted, ash, (int)parts, gx,
                       (long)s);
}

/* The run map's bytes, or 0 in the decode band. Sized from M * top_k, as the other grouped rows
 * size theirs: the hook runs at declare, where the sorted row count is not yet an operand. */
extern "C" int64_t r3d_rad_moe_gemm_w4nl_scratch(const RadArgs* a) {
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
