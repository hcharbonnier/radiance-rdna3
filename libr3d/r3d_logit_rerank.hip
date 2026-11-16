/* r3d_logit_rerank.hip -- logit_rerank over the trunk's exact head: the IQR lm form (r3d_iqr.h)
 * and the int8 form Flash-Next's `output.weight` is stored in.
 *
 * A coarse draft head proposes R candidate ids a row; this rescores only those against the exact
 * head and re-sorts on the exact values, which recovers the top-1 agreement a 2-bit head loses.
 * R weight rows are read instead of the whole head: at Flash-Next's shape an 8-way rescore of one
 * row reads 40 KB where the full i8 head is 635 MB.
 *
 * Two stored forms, one arithmetic:
 *   - IQR (`logit_rerank_iqr`): the lm rows `logits_gemm_iqr` stores (r3d_iqr.h). B bits a code,
 *     a bf16 scale per (row, 128 of K) in the row's tail, and the int8 table after the last row;
 *     the code's value is `table[code] * scale`.
 *   - int8 (`logit_rerank_i8`): the container's `i8*bf16[1x128]` head, joined into `K` codes and
 *     `K/128` bf16 scales in each row's tail (`r3d.i8lm.r128`, libr4d's fp8 lm-head row geometry).
 * The logical head both decode to is the f32 the reference reads.
 *
 * One workgroup a row, one wave a candidate. A wave walks K with a lane stride and folds with the
 * same xor butterfly as the rest of this library. Lane 0 of wave 0 then runs the reference's sort:
 * descending value, ties to the LOWER id, and a candidate outside this rank's rows is a pad that
 * sorts last with id -1 (`vocab_off` is the shard's first column and ids are global).
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_common.h"
#include "r3d_iqr.h"
#include "r3d_launch.h"
#include "rad_planes.h"

namespace {

constexpr int kRerankThreads = 256;
constexpr int kRerankWaves = kRerankThreads / 32;
constexpr int kRerankRMax = 32;

/* `IQR` picks the row's decode: B-bit table codes, or one int8 a column. */
template <bool IQR>
__global__ __launch_bounds__(kRerankThreads) void logit_rerank_kernel(
    const uint16_t* __restrict__ X, const int32_t* __restrict__ IN,
    const uint8_t* __restrict__ W, const uint16_t* __restrict__ S, const int8_t* __restrict__ tab,
    long w_ld, long scale_row, uint32_t bits, int32_t* __restrict__ OI,
    float* __restrict__ OV, float* __restrict__ OP, int R, long K, long voff, long nv, long x_ld,
    long in_ld, long oi_ld, long ov_ld) {
    __shared__ float s_sc[kRerankRMax];
    __shared__ int32_t s_id[kRerankRMax];

    const int m = (int)blockIdx.x;
    const int lane = (int)(threadIdx.x & 31u);
    const int wv = (int)(threadIdx.x >> 5);
    const uint16_t* xr = X + (long)m * x_ld;
    const uint32_t cmask = (bits >= 32u) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    const long cb = IQR ? (long)K * (long)bits / 8 : (long)K;   /* codes, before the scales */

    for (int r = wv; r < R; r += kRerankWaves) {
        const int32_t gid = IN[(long)m * in_ld + r];
        const long loc = (long)gid - voff;
        const bool own = gid >= 0 && loc >= 0 && loc < nv;
        float acc = 0.0f;
        if (own) {
            /* IQR rows are packed (codes, then the row's scales); the int8 head is the container's
             * joined codes + scale planes, read in place. */
            const uint8_t* cr = IQR ? W + (size_t)loc * (size_t)w_ld : W + (long)loc * K;
            const uint16_t* sr = IQR ? (const uint16_t*)(cr + cb) : S + (long)loc * scale_row;
            for (long k = lane; k < K; k += 32) {
                float c;
                if constexpr (IQR) {
                    const long bit = k * (long)bits;
                    const uint32_t wv2 =
                        (uint32_t)cr[bit >> 3] | ((uint32_t)cr[(bit >> 3) + 1] << 8);
                    c = (float)(int32_t)(int8_t)tab[(wv2 >> (uint32_t)(bit & 7)) & cmask];
                } else {
                    c = (float)(int32_t)(int8_t)cr[k];
                }
                acc = fmaf(c * bf16_to_f32(sr[k >> 7]), bf16_to_f32(xr[k]), acc);
            }
            acc = wave_sum(acc);
        }
        if (lane == 0) {
            s_sc[r] = own ? acc : -INFINITY;
            s_id[r] = own ? gid : -1;
        }
    }
    __syncthreads();
    if (wv != 0 || lane != 0) return;

    /* The oracle's next_best: among the unused candidates with an id >= 0, the largest value,
     * ties to the lower id; when none is left the slot is a pad (-1, -inf). */
    bool used[kRerankRMax];
    for (int r = 0; r < R; ++r) used[r] = false;
    for (int o = 0; o < R; ++o) {
        int best = -1;
        for (int c = 0; c < R; ++c) {
            if (used[c] || s_id[c] < 0) continue;
            if (best < 0 || s_sc[c] > s_sc[best] ||
                (s_sc[c] == s_sc[best] && s_id[c] < s_id[best]))
                best = c;
        }
        const int32_t oid = best >= 0 ? s_id[best] : -1;
        const float ov = best >= 0 ? s_sc[best] : -INFINITY;
        if (best >= 0) used[best] = true;
        OI[(long)m * oi_ld + o] = oid;
        OV[(long)m * ov_ld + o] = ov;
        if (OP) {
            float* p = OP + ((long)m * R + o) * 2;
            *(int32_t*)p = oid;
            p[1] = ov;
        }
    }
}

/* ---- the container's int8 head -------------------------------------------------------------
 *
 * `output.weight` is `i8*bf16[1x128]`, the form logits_gemm_bf16 already reads. The row joins the
 * two planes with `rad_planes_layout` (the same `rad.planes` form, no byte moved) and addresses
 * the codes and scale planes in place: the header, the codes plane, then the scale plane. A weight
 * is stored once, so this row must not ask for a second form.
 */

struct I8LogitsGeom {
    long long rows, dim;
};

bool i8l_geom(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
              const RadTensor* pl, int n, I8LogitsGeom* g) {
    if (operand != 1) return false;
    g->rows = rad_param_geti(p, n_p, "n_vocab", 0);
    g->dim = rad_param_geti(p, n_p, "n_embd", 0);
    if (g->rows <= 0 || g->dim <= 0 || (g->dim % 128) != 0) return false;
    if (!enc || n != 2 || !rad_enc_is(enc, "affine")) return false;
    if (enc->n_planes != 2 || enc->plane[0].dtype != RAD_I8 || enc->plane[0].block[0] != 1 ||
        enc->plane[0].block[1] != 1)
        return false;
    const int s = rad_enc_find(enc, "scale");
    if (s < 0 || enc->plane[s].dtype != RAD_BF16 || enc->plane[s].kind != RAD_PLANE_TILED ||
        enc->plane[s].block[0] != 1 || enc->plane[s].block[1] != 128)
        return false;
    const auto role = [&](int i) {
        return (sel && sel[i] >= 0 && sel[i] < enc->n_planes) ? enc->plane[sel[i]].role : "";
    };
    if (std::strncmp(role(0), "codes", RAD_ENC_STR) != 0 ||
        std::strncmp(role(1), "scale", RAD_ENC_STR) != 0)
        return false;
    if (!pl || pl[0].rank != 2 || pl[0].shape[0] != g->rows || pl[0].shape[1] != g->dim)
        return false;
    if (pl[1].rank != 2 || pl[1].shape[0] != g->rows || pl[1].shape[1] != g->dim / 128)
        return false;
    return true;
}

/* ---- the shared launch --------------------------------------------------------------------- */

int logit_rerank_run(const RadArgs* a, RadStream s, bool iqr) {
    if (!a || a->n_t < 5) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0), *w = r3d_opt(a, 1), *in = r3d_opt(a, 2),
                    *idx = r3d_opt(a, 3), *val = r3d_opt(a, 4), *pr = r3d_opt(a, 5);
    if (!r3d_p(x) || !r3d_p(w) || !r3d_p(in) || !r3d_wp(idx) || !r3d_wp(val))
        return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(x)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_i32(in) || !r3d_is_i32(idx) || !r3d_is_f32(val)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(pr) && !r3d_is_f32(pr)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(x) || !r3d_contig(w) || !r3d_contig(in) || !r3d_contig(idx) ||
        !r3d_contig(val) || (r3d_has(pr) && !r3d_contig(pr)))
        return R3D_NO(RAD_E_STRIDE);

    const long long K = r3d_geti_or(a, "n_embd", 0);
    const long long NV = r3d_geti_or(a, "n_vocab", 0);
    const long long R = r3d_geti_or(a, "R", 0);
    const long long voff = r3d_geti_or(a, "vocab_off", 0);
    if (K <= 0 || NV <= 0 || R <= 0) return R3D_NO(RAD_E_SHAPE);
    if (R > kRerankRMax) return R3D_NO(RAD_E_SHAPE);
    if (voff < 0) return R3D_NO(RAD_E_SHAPE);
    if ((K % 128) != 0) return R3D_NO(RAD_E_SHAPE);   /* the stored scale block */

    uint32_t bits = 0;
    long long w_ld = 0, scale_row = 0;
    long long codes_off = 0, scale_off = 0;
    if (iqr) {
        if (!r3d_iqr::is_weight_dtype(w->dtype)) return R3D_NO(RAD_E_DTYPE);
        const r3d_iqr::Form f = r3d_iqr::form(w->dtype & r3d_iqr::kWordMask);
        if (!r3d_iqr::lm_built(r3d_iqr::word(f))) return R3D_NO(RAD_E_DTYPE);
        bits = f.bits;
        w_ld = (long long)r3d_iqr::lm_row(f, K);
        /* The joined descriptor covers the rows; the table sits past them, inside the allocation
         * the layout hook sized (lm_bytes), and is addressed from the base. */
        if (r3d_numel(w) < NV * w_ld) return R3D_NO(RAD_E_SHAPE);
    } else {
        /* The container's `i8*bf16[1x128]`, joined by rad_planes_layout: the header, the codes
         * plane, then the scale plane. Both operands are the same stored bytes as
         * logits_gemm_bf16's, so a weight is not asked for in two forms. */
        if (!r3d_is_i8(w)) return R3D_NO(RAD_E_DTYPE);
        codes_off = rad_planes_round((long long)sizeof(RadPlanesHeader) +
                                     2 * (long long)sizeof(RadPlanesEntry));
        scale_off = rad_planes_round(codes_off + NV * K);
        const long long planes_bytes =
            rad_planes_round(scale_off + NV * (K / 128) * 2);
        const bool joined = w->rank == 1 && r3d_numel(w) >= planes_bytes;
        const bool logical = w->rank == 2 && r3d_dim(w, 0) == NV && r3d_dim(w, 1) >= K;
        if (!joined && !logical) return R3D_NO(RAD_E_SHAPE);
        scale_row = K / 128;   /* bf16 elements a scale row */
    }

    long long M = 0, Mi = 0, Mo = 0, Mv = 0;
    const long long x_ld = r3d_rows_pitch(x, K, &M);
    const long long in_ld = r3d_rows_pitch(in, R, &Mi);
    const long long oi_ld = r3d_rows_pitch(idx, R, &Mo);
    const long long ov_ld = r3d_rows_pitch(val, R, &Mv);
    if (x_ld < 0 || in_ld < 0 || oi_ld < 0 || ov_ld < 0 || M <= 0) return R3D_NO(RAD_E_STRIDE);
    if (Mi < M || Mo < M || Mv < M) return R3D_NO(RAD_E_SHAPE);
    if (r3d_has(pr) && r3d_numel(pr) < M * R * 2) return R3D_NO(RAD_E_SHAPE);

    const uint8_t* wb = (const uint8_t*)r3d_p(w);
    const uint8_t* wbase = iqr ? wb : wb + codes_off;
    const uint16_t* sbase = iqr ? nullptr : (const uint16_t*)(wb + scale_off);
    const int8_t* tab = iqr ? (const int8_t*)(wb + r3d_iqr::lm_tab(
                                          r3d_iqr::form(w->dtype & r3d_iqr::kWordMask), NV, K))
                            : nullptr;
#define R3D_RERANK_LAUNCH(IQR_)                                                                  \
    R3D_KLAUNCH(logit_rerank_kernel<IQR_>, dim3((unsigned)M), dim3(kRerankThreads),               \
                (hipStream_t)s, (const uint16_t*)r3d_p(x), (const int32_t*)r3d_p(in), wbase,     \
                sbase, tab, (long)w_ld, (long)scale_row, bits, (int32_t*)r3d_wp(idx),            \
                (float*)r3d_wp(val), r3d_has(pr) ? (float*)r3d_wp(pr) : nullptr, (int)R,         \
                (long)K, (long)voff, (long)NV, x_ld, in_ld, oi_ld, ov_ld)
    if (iqr) R3D_RERANK_LAUNCH(true);
    else R3D_RERANK_LAUNCH(false);
#undef R3D_RERANK_LAUNCH
    return hipGetLastError() == hipSuccess ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

}  // namespace

extern "C" int r3d_rad_logit_rerank_iqr(const RadArgs* a, RadStream s) {
    return logit_rerank_run(a, s, true);
}

extern "C" int r3d_rad_logit_rerank_i8(const RadArgs* a, RadStream s) {
    return logit_rerank_run(a, s, false);
}

extern "C" int r3d_rad_layout_i8_logits(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* pl, int n, RadLayout* out) {
    I8LogitsGeom g;
    if (!out) return RAD_E_INVAL;
    if (operand != 1) return RAD_E_UNSUPPORTED;
    if (!i8l_geom(p, n_p, operand, enc, sel, pl, n, &g)) return RAD_E_DTYPE;
    /* The same stored form logits_gemm_bf16 reads: the container's planes, joined, no byte moved. */
    return rad_planes_layout(enc, sel, pl, n, out);
}