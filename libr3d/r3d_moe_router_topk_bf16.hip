/* r3d_moe_router_topk_bf16.hip -- router_topk: softmax over all n_expert logits, the top_k by
 * value, and -- if `norm` -- the kept weights renormalised to sum to one.
 *
 * One wave per row (one block per row); the selection is top_k passes of a wave minimum, not a
 * sort. That matches the reference exactly (largest remaining value, ties to the lower id); a sort
 * with another tie rule would be a different router on tied rows, which are not rare. Selection
 * runs on the logits: softmax is monotone, so the order is the same and the comparison exact.
 *
 * gfx1102: a wave reduction (the max, the sum, the per-round min) in a loop body miscompiles for
 * a non-first wave -- wave 0 is correct, wave 1+ drop out. Two things keep this safe:
 *   * one wave per block (one block per row): the __syncthreads is a single-wave barrier, so there
 *     is no cross-wave sync whose ordering the compiler can scramble;
 *   * a separate LDS slot per top_k round: the loop never reuses a slot, so a read-after-write
 *     across iterations cannot be folded (the max and the sum use their own slots, rounds 0 and 1).
 * Every wave reduction is an LDS pass, not a DPP butterfly (a DPP exchange miscompiles here).
 *
 * Verified: single-row (equal, distinct, random) and multi-row (M >= 2) all pick the right ids and
 * weights.
 */
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_launch.h"
#include "r3d_codecs.h"

#define R3D_BARRIER() do { __syncthreads(); asm volatile("" ::: "memory"); } while (0)

namespace {

constexpr int kW = 32;

/* (order-preserving value, ~id); 0 = absent or NaN. The key's high half is the logit's bits with
 * the sign folded (so larger logit -> larger key) and the low half is the complemented id, so a
 * tie goes to the lower id. */
__device__ __forceinline__ uint32_t topk_key(uint16_t raw, uint32_t e, bool ok) {
    const bool nan = ((raw & 0x7F80u) == 0x7F80u) && (raw & 0x7Fu);
    const uint16_t u = (raw == 0x8000u) ? (uint16_t)0u : raw;   // -0.0 folds to +0.0
    const uint16_t k = (u & 0x8000u) ? (uint16_t)~u : (uint16_t)(u | 0x8000u);
    return nan ? 0u : (((uint32_t)k << 16) | (~e & 0xFFFFu));
}
__device__ __forceinline__ uint16_t topk_key_raw(uint32_t key) {
    const uint32_t k = key >> 16;
    return (uint16_t)((k & 0x8000u) ? (k & 0x7FFFu) : ~k);
}

/* One row, one wave (the block is a single wave). `wmin` holds 32 slots per round: round 0 the
 * max, round 1 the sum, rounds 2..(1+top_k) the per-round minimums. A separate slot per round so
 * the loop never reuses one (gfx1102). */
template <int MAXOWN>
__device__ __forceinline__ void r3d_router_topk_row(
    const unsigned short* __restrict__ lg, int32_t* __restrict__ ids,
    unsigned short* __restrict__ w, uint32_t m, uint32_t n_expert, uint32_t top_k,
    uint32_t lg_ld, int norm, uint32_t lane, __shared__ uint32_t* wmin)
{
    const unsigned short* row = lg + (size_t)m * lg_ld;

    uint16_t raw[MAXOWN];
#pragma unroll
    for (int i = 0; i < MAXOWN; ++i) {
        const uint32_t e = lane + i * kW;
        raw[i] = row[e < n_expert ? e : 0u];
    }
    float v[MAXOWN];
    uint32_t key[MAXOWN];
#pragma unroll
    for (int i = 0; i < MAXOWN; ++i) {
        const uint32_t e = lane + i * kW;
        const bool ok = e < n_expert;
        v[i] = ok ? bf16_to_f32(raw[i]) : -INFINITY;
        key[i] = topk_key(raw[i], e, ok) & (0u - (uint32_t)ok);
    }

    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < MAXOWN; ++i) mx = fmaxf(mx, v[i]);

    // Wave max through LDS (round 0).
    wmin[lane] = __builtin_bit_cast(uint32_t, mx);
    R3D_BARRIER();
#pragma unroll
    for (int i = 0; i < kW; ++i)
        mx = fmaxf(mx, __builtin_bit_cast(float, wmin[i]));

    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < MAXOWN; ++i) sum += __expf(v[i] - mx);
    // Wave sum through LDS (round 1).
    wmin[kW + lane] = __builtin_bit_cast(uint32_t, sum);
    R3D_BARRIER();
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < kW; ++i) s += __builtin_bit_cast(float, wmin[kW + i]);
    sum = s;
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    uint32_t P = 0xFFFFFFFFu, mykey = 0u, nsel = 0;
    for (uint32_t r = 0; r < top_k; ++r) {
        uint32_t d = 0xFFFFFFFFu;
#pragma unroll
        for (int i = 0; i < MAXOWN; ++i) {
            const uint32_t t = P - key[i];
            d = d < t ? d : t;
        }
        // Wave minimum through LDS, round (2 + r): a fresh slot each round.
        const uint32_t base = (2 + r) * kW;
        wmin[base + lane] = d;
        R3D_BARRIER();
        uint32_t mn = 0xFFFFFFFFu;
#pragma unroll
        for (int i = 0; i < kW; ++i) mn = mn < wmin[base + i] ? mn : wmin[base + i];
        const uint32_t best = P - mn;
        if ((best & 0xFFFFu) == 0u) break;
        if (lane == r) mykey = best;
        P = best - 1u;
        ++nsel;
    }
    const float p = __expf(bf16_to_f32(topk_key_raw(mykey)) - mx) * inv;
    const uint16_t pb = f32_to_bf16(p);
    float wsum = 0.0f;
    for (uint32_t r = 0; r < nsel; ++r)
        wsum += __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, p),
                                                                    (int)r));
    if (lane < nsel) {
        ids[(size_t)m * top_k + lane] = (int32_t)(~mykey & 0xFFFFu);
        w[(size_t)m * top_k + lane] =
            (norm && wsum > 0.0f) ? f32_to_bf16(bf16_to_f32(pb) * (1.0f / wsum)) : pb;
    }
}

template <int MAXOWN>
__global__ __launch_bounds__(kW) void r3d_router_topk_kernel(
    const unsigned short* __restrict__ lg, int32_t* __restrict__ ids,
    unsigned short* __restrict__ w, uint32_t M, uint32_t n_expert, uint32_t top_k,
    uint32_t lg_ld, int norm)
{
    __shared__ uint32_t wmin[kW * (2 + kW)];   // a slot per round, top_k <= kW
    const uint32_t lane = threadIdx.x;
    const uint32_t m = blockIdx.x;             // one block per row
    r3d_router_topk_row<MAXOWN>(lg, ids, w, m, n_expert, top_k, lg_ld, norm, lane, wmin);
}

}  // namespace



/* ------------------------------------------------------------------ router_sigmoid_topk
 *
 * The GLM-style router: scores[e] = sigmoid(logit[e]), the selection runs on
 * select[e] = scores[e] + bias[e] (a per-expert correction bias), and the kept weights are the
 * scores themselves, scaled by route_scale (and renormalised to sum to one if `norm`). Ties go to
 * the lower id, as in router_topk.
 *
 * The selection key is 64-bit (the 32-bit select's monotone bits, the low 32 the complemented id),
 * so a 64-bit LDS slot per round. The scores are staged in LDS so the weight is read back without a
 * global reload. */
template <int MAXOWN>
__device__ __forceinline__ void r3d_router_sigmoid_topk_row(
    const unsigned short* __restrict__ lg, const float* __restrict__ bias,
    int32_t* __restrict__ ids, unsigned short* __restrict__ w,
    uint32_t m, uint32_t n_expert, uint32_t top_k, uint32_t lg_ld,
    float scale, int norm, uint32_t lane,
    __shared__ uint64_t* wmin, __shared__ float* s_score)
{
    const unsigned short* row = lg + (size_t)m * lg_ld;

    uint64_t key[MAXOWN];
#pragma unroll
    for (int i = 0; i < MAXOWN; ++i) {
        const uint32_t e = lane + i * kW;
        const bool ok = e < n_expert;
        const float lgt = ok ? bf16_to_f32(row[e]) : -INFINITY;
        const float sc = 1.0f / (1.0f + __expf(-lgt));
        s_score[lane * MAXOWN + i] = sc;
        const float sel = ok ? (sc + bias[e]) : -INFINITY;
        const uint32_t bits = __builtin_bit_cast(uint32_t, sel);
        const uint32_t u = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        key[i] = (((uint64_t)u) << 32) | (~e & 0xFFFFFFFFull);
    }

    uint64_t P = ~0ull, mykey = 0ull, nsel = 0;
    for (uint32_t r = 0; r < top_k; ++r) {
        uint64_t d = ~0ull;  // min over (P - key[i]); P - key is smaller for the larger key
#pragma unroll
        for (int i = 0; i < MAXOWN; ++i) {
            const uint64_t t = P - key[i];
            d = d < t ? d : t;
        }
        const uint64_t base = (uint64_t)(2 + r) * kW;
        wmin[base + lane] = d;
        R3D_BARRIER();
        uint64_t mn = ~0ull;
#pragma unroll
        for (int i = 0; i < kW; ++i) mn = mn < wmin[base + i] ? mn : wmin[base + i];
        const uint64_t best = P - mn;
        if (best == 0ull) break;
        if (lane == r) mykey = best;
        P = best - 1ull;
        ++nsel;
    }

    const uint32_t id = (uint32_t)(~mykey & 0xFFFFFFFFull);
    const float sc = s_score[(id % kW) * MAXOWN + (id / kW)];
    // Sum the kept scores over lanes 0..nsel-1 (a wave reduce).
    wmin[lane] = (uint64_t)__builtin_bit_cast(uint32_t, sc);
    R3D_BARRIER();
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kW; ++i) {
        const uint32_t b = (uint32_t)wmin[i];
        if (i < nsel) sum += __builtin_bit_cast(float, b);
    }
    if (lane < nsel) {
        const float weight = sc * (norm ? (scale / (sum + 1e-20f)) : scale);
        ids[(size_t)m * top_k + lane] = (int32_t)id;
        w[(size_t)m * top_k + lane] = f32_to_bf16(weight);
    }
}

template <int MAXOWN>
__global__ __launch_bounds__(kW) void r3d_router_sigmoid_topk_kernel(
    const unsigned short* __restrict__ lg, const float* __restrict__ bias,
    int32_t* __restrict__ ids, unsigned short* __restrict__ w,
    uint32_t M, uint32_t n_expert, uint32_t top_k, uint32_t lg_ld, float scale, int norm)
{
    __shared__ uint64_t wmin[kW * (2 + kW)];
    __shared__ float s_score[kW * MAXOWN];
    const uint32_t lane = threadIdx.x;
    const uint32_t m = blockIdx.x;
    r3d_router_sigmoid_topk_row<MAXOWN>(lg, bias, ids, w, m, n_expert, top_k, lg_ld, scale,
                                        norm, lane, wmin, s_score);
}


extern "C" {

/* moe_scatter's public dispatch, reused by the fused router_topk_scatter. */
int r3d_rad_moe_scatter(const RadArgs*, RadStream);

int r3d_rad_moe_router_topk_bf16(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const long long ne = r3d_geti_or(a, "n_expert", 0), tk = r3d_geti_or(a, "top_k", 0);
    const long long norm_ll = r3d_geti_or(a, "norm", 0);
    const int norm = (int)norm_ll;
    if (ne <= 0 || tk <= 0 || tk > kW) return R3D_NO(RAD_E_SHAPE);
    if (ne > (long long)16 * kW) return R3D_NO(RAD_E_SHAPE);   // MAXOWN=16 for this row
    const RadTensor* L = r3d_opt(a, 0), *I = r3d_opt(a, 1), *W = r3d_opt(a, 2);
    if (!r3d_p(L) || !r3d_wp(I) || !r3d_wp(W)) return R3D_NO(RAD_E_INVAL);
    if (L->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    const uint32_t lg_ld = (uint32_t)r3d_row_pitch(L);
    if (lg_ld < ne) return R3D_NO(RAD_E_SHAPE);
    /* The rows are the step's, not the band's: a declare binds `M` to the band's upper bound while
     * the activations carry the step's shape (the same rule the scatter phase at the bottom of this
     * file follows). Reading the capacity here would run, read and write rows the step does not
     * have. */
    const long long M = (long long)r3d_dim(L, 0);
    if (M < 1 || r3d_geti_or(a, "M", M) < M) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(I) < M * tk || r3d_numel(W) < M * tk) return R3D_NO(RAD_E_SHAPE);

    dim3 grid((unsigned)M), blk(kW);           // one block (one wave) per row
    const unsigned short* lg = (const unsigned short*)r3d_p(L);
    int32_t* ip = (int32_t*)r3d_wp(I);
    unsigned short* wp = (unsigned short*)r3d_wp(W);
    if (ne <= (long long)2 * kW)
        R3D_KLAUNCH(r3d_router_topk_kernel<2>, grid, blk, (hipStream_t)s, lg, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, norm);
    else if (ne <= (long long)8 * kW)
        R3D_KLAUNCH(r3d_router_topk_kernel<8>, grid, blk, (hipStream_t)s, lg, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, norm);
    else
        R3D_KLAUNCH(r3d_router_topk_kernel<16>, grid, blk, (hipStream_t)s, lg, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, norm);
    return (int)hipGetLastError();
}


/* router_topk_scatter: the pair as one op. Phase 1 is the top-k selection over the whole n_expert
 * (the router_topk kernel); phase 2 is the counting sort over this rank's slice (n_local), which
 * reuses moe_scatter's dispatch. Byte-identical to the two reference launches, in order. */
int r3d_rad_router_topk_scatter(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 6) return R3D_NO(RAD_E_INVAL);
    const long long ne = r3d_geti_or(a, "n_expert", 0), tk = r3d_geti_or(a, "top_k", 0),
                    nl = r3d_geti_or(a, "n_local", ne);
    const int norm = (int)r3d_geti_or(a, "norm", 0);
    if (ne <= 0 || tk <= 0 || tk > (long long)kW || tk > ne) return R3D_NO(RAD_E_SHAPE);
    if (nl <= 0 || nl > ne) return R3D_NO(RAD_E_SHAPE);
    if (ne > (long long)16 * kW) return R3D_NO(RAD_E_SHAPE);

    const RadTensor* L = r3d_opt(a, 0), *I = r3d_opt(a, 1), *W = r3d_opt(a, 2),
                    *ST = r3d_opt(a, 3), *OFF = r3d_opt(a, 4), *CNT = r3d_opt(a, 5);
    if (!r3d_p(L) || !r3d_wp(I) || !r3d_wp(W) || !r3d_wp(ST) || !r3d_wp(OFF) || !r3d_wp(CNT))
        return R3D_NO(RAD_E_INVAL);
    if (L->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    const uint32_t lg_ld = (uint32_t)r3d_row_pitch(L);
    if (lg_ld < ne) return R3D_NO(RAD_E_SHAPE);
    /* The step's rows, not the band's capacity: a decode step of one row inside a band declared at
     * max_seqs would otherwise run, read and write rows it does not have. */
    const long long M = (long long)r3d_dim(L, 0);
    if (M < 1 || r3d_geti_or(a, "M", M) < M) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(I) < M * tk || r3d_numel(W) < M * tk || r3d_numel(ST) < M * tk)
        return R3D_NO(RAD_E_SHAPE);

    // Phase 1: the top-k selection (the router_topk kernel, over the whole n_expert).
    dim3 grid((unsigned)M), blk(kW);
    const unsigned short* lg = (const unsigned short*)r3d_p(L);
    int32_t* ip = (int32_t*)r3d_wp(I);
    unsigned short* wp = (unsigned short*)r3d_wp(W);
    if (ne <= (long long)2 * kW)
        R3D_KLAUNCH(r3d_router_topk_kernel<2>, grid, blk, (hipStream_t)s, lg, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, norm);
    else if (ne <= (long long)8 * kW)
        R3D_KLAUNCH(r3d_router_topk_kernel<8>, grid, blk, (hipStream_t)s, lg, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, norm);
    else
        R3D_KLAUNCH(r3d_router_topk_kernel<16>, grid, blk, (hipStream_t)s, lg, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, norm);
    if (hipGetLastError() != hipSuccess) return R3D_NO(RAD_E_INVAL);

    // Phase 2: the counting sort over this rank's slice (n_local), via moe_scatter's dispatch.
    RadTensor t2[4] = { *I, *ST, *OFF, *CNT };
    RadParam sp[4] = {
        { "n_expert", RAD_P_INT, (long long)nl, 0, nullptr, 0.0 },
        { "top_k", RAD_P_INT, (long long)tk, 0, nullptr, 0.0 },
        { "expert_base", RAD_P_INT, r3d_geti_or(a, "expert_base", 0), 0, nullptr, 0.0 },
        { "protect", RAD_P_STR, 0, 0, r3d_gets(a, "protect"), 0.0 },
    };
    RadArgs sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.t = t2; sa.n_t = 4; sa.p = sp; sa.n_p = 4;
    sa.rank = a->rank; sa.world_size = a->world_size;
    return r3d_rad_moe_scatter(&sa, s);
}


int r3d_rad_router_sigmoid_topk(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 4) return R3D_NO(RAD_E_INVAL);
    const long long ne = r3d_geti_or(a, "n_expert", 0), tk = r3d_geti_or(a, "top_k", 0);
    const int norm = (int)r3d_geti_or(a, "norm", 0);
    const double scale_d = r3d_getf(a, "route_scale", 0.0);
    const float scale = (float)scale_d;
    if (ne <= 0 || tk <= 0 || tk > (long long)kW || tk > ne) return R3D_NO(RAD_E_SHAPE);
    if (ne > (long long)16 * kW) return R3D_NO(RAD_E_SHAPE);
    if (scale <= 0.0f) return R3D_NO(RAD_E_SHAPE);

    const RadTensor* L = r3d_opt(a, 0), *B = r3d_opt(a, 1), *I = r3d_opt(a, 2), *W = r3d_opt(a, 3);
    if (!r3d_p(L) || !r3d_p(B) || !r3d_wp(I) || !r3d_wp(W)) return R3D_NO(RAD_E_INVAL);
    if (L->dtype != RAD_BF16 || B->dtype != RAD_F32) return R3D_NO(RAD_E_DTYPE);
    const uint32_t lg_ld = (uint32_t)r3d_row_pitch(L);
    if (lg_ld < ne) return R3D_NO(RAD_E_SHAPE);
    /* The step's rows, not the band's capacity (see the note in the scatter phase below). */
    const long long M = (long long)r3d_dim(L, 0);
    if (M < 1 || r3d_geti_or(a, "M", M) < M) return R3D_NO(RAD_E_SHAPE);
    /* `B` is the per-expert correction bias `[ne]`, indexed `bias[e]` for `e < n_expert`: its
     * length is the expert count, not the step's rows. */
    if (r3d_numel(I) < M * tk || r3d_numel(W) < M * tk || r3d_numel(B) < ne)
        return R3D_NO(RAD_E_SHAPE);

    dim3 grid((unsigned)M), blk(kW);
    const unsigned short* lg = (const unsigned short*)r3d_p(L);
    const float* bp = (const float*)r3d_p(B);
    int32_t* ip = (int32_t*)r3d_wp(I);
    unsigned short* wp = (unsigned short*)r3d_wp(W);
    if (ne <= (long long)2 * kW)
        R3D_KLAUNCH(r3d_router_sigmoid_topk_kernel<2>, grid, blk, (hipStream_t)s, lg, bp, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, scale, norm);
    else if (ne <= (long long)8 * kW)
        R3D_KLAUNCH(r3d_router_sigmoid_topk_kernel<8>, grid, blk, (hipStream_t)s, lg, bp, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, scale, norm);
    else
        R3D_KLAUNCH(r3d_router_sigmoid_topk_kernel<16>, grid, blk, (hipStream_t)s, lg, bp, ip, wp,
                    (uint32_t)M, (uint32_t)ne, (uint32_t)tk, lg_ld, scale, norm);
    return (int)hipGetLastError();
}

}  // extern "C"


/* ------------------------------------------------------- router_gemm_topk_scatter
 *
 * The router's GEMM, its top-k and the counting sort as the one op docs/OPS.md declares, plus the
 * shared expert's one-column gate when the caller brings `gate_b` and `gate`. libr3d serves it by
 * composing the *device* launches it stands in for, in order:
 *
 *   1. `gemm_nt` of a [M,K] against the router b [n_expert,K] into logits -- the same skinny row
 *      the unfused router resolves to: gemm_nt_bf16_m16 at M <= 16 (split-K as its law picks) and
 *      gemm_nt_bf16_tiled above, so the bytes are the pair's;
 *   2. the shared gate at N = 1 over the same a, in 16-row chunks through the m16 row (its own
 *      band is M <= 16, and a row of the product depends on no other, so chunking is exact);
 *   3. `router_topk_scatter` over those logits, the device row itself.
 *
 * It is therefore not a single-launch fusion: the arrival-counter form is libr4d's, and the
 * selection still waits on a stream barrier. It exists so the op resolves in device instead of
 * falling back to the host `avx_router_gemm_topk_scatter`, and every output is what the separate
 * launches would write. Scratch is the m16 router's f32 partial arena; the tiled band asks for
 * none. A tuned point for the unfused router's gemm_nt would reach it where the pair's tuning
 * differs; no shipped configuration tunes these GEMMs (docs/OPS.md). */

extern "C" {
int r3d_rad_gemm_nt_bf16_m16(const RadArgs*, RadStream);
int r3d_rad_gemm_nt_bf16_tiled(const RadArgs*, RadStream);
}

namespace {

/* r3d_gemm_bf16_nt_m16.hip's split law: the largest ladder step whose slab is whole 8-k steps. */
int rg_pick_sk(long long K, int maxsk) {
    static const int ladder[] = { 16, 8, 4, 2, 1 };
    for (int s : ladder)
        if (s <= maxsk && K % ((long long)s * 8) == 0) return s;
    return 1;
}

/* One gemm_nt launch over a triple of the fused op's operands, with the same split/impl the
 * unfused row reads (the fused op carries no tune keys, so the row defaults). */
int rg_gemm(const RadArgs* a, RadStream s, const RadTensor& x, const RadTensor& w,
            const RadTensor& y, long long N, long long K, bool m16) {
    RadTensor t[3] = { x, w, y };
    RadParam p[3] = { RAD_INT("N", N), RAD_INT("K", K), RAD_STR("dtype", "bf16") };
    RadArgs ga;
    std::memset(&ga, 0, sizeof ga);
    ga.t = t; ga.n_t = 3;
    ga.p = p; ga.n_p = 3;
    ga.scratch = a->scratch;
    ga.scratch_bytes = a->scratch_bytes;
    ga.rank = a->rank;
    ga.world_size = a->world_size;
    return m16 ? r3d_rad_gemm_nt_bf16_m16(&ga, s) : r3d_rad_gemm_nt_bf16_tiled(&ga, s);
}

/* The shared expert's gate: a [M,K] x gate_b [1,K]^T -> gate [M,1] through the m16 row, whose
 * band is 16 rows, one chunk at a time. `a_ld` is the activation's row pitch the m16 shim would
 * read (K on a dense buffer), so a sliced activation keeps its stride. A row of the product
 * reads only its own activation row, so the chunking changes no bit. `nt` is the m16 row's
 * default (non-temporal weight streaming). */
int rg_gate(const RadArgs* a, RadStream s, const RadTensor& x, const RadTensor& gb,
            const RadTensor& y, long long M, long long K, long long a_ld) {
    const int sk = rg_pick_sk(K, 16);
    if (sk > 1 && (!a->scratch || a->scratch_bytes < (long long)sk * 16 * 4))
        return R3D_NO(RAD_E_SCRATCH);
    const unsigned short* xp = (const unsigned short*)r3d_p(&x);
    unsigned short* yp = (unsigned short*)r3d_wp(&y);
    for (long long i = 0; i < M; i += 16) {
        const int mc = (int)((M - i) < 16 ? (M - i) : 16);
        const int e = r3d_gemm_nt_bf16_m16(
            (long)(uintptr_t)(xp + (size_t)i * (size_t)a_ld), (long)r3d_addr(&gb),
            (long)(uintptr_t)(yp + i), mc, 1, (int)K, (int)a_ld, 1, sk, 8, 1,
            (long)(uintptr_t)(sk > 1 ? a->scratch : nullptr), (long)(uintptr_t)s);
        if (e != 0)
            return R3D_NO(e == -1 ? RAD_E_INVAL : e == -2 ? RAD_E_SHAPE
                             : e == -3 ? RAD_E_STRIDE : RAD_E_DEVICE);
    }
    return RAD_OK;
}

}  // namespace

extern "C" int r3d_rad_router_gemm_topk_scatter(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 9) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0), *wr = r3d_opt(a, 1), *wg = r3d_opt(a, 2),
                    *lg = r3d_opt(a, 3), *ids = r3d_opt(a, 4), *w = r3d_opt(a, 5),
                    *st = r3d_opt(a, 6), *off = r3d_opt(a, 7), *cnt = r3d_opt(a, 8);
    const RadTensor* gate = r3d_opt(a, 9);   /* absent slot has null data */
    if (!r3d_p(x) || !r3d_p(wr) || !r3d_wp(lg) || !r3d_wp(ids) || !r3d_wp(w) || !r3d_wp(st) ||
        !r3d_wp(off) || !r3d_wp(cnt))
        return R3D_NO(RAD_E_INVAL);
    /* The gate weight and its output come together or not at all, as the reference states. */
    if (r3d_has(wg) != r3d_has(gate)) return R3D_NO(RAD_E_INVAL);

    const long long ne = r3d_geti_or(a, "n_expert", 0);
    const long long tk = r3d_geti_or(a, "top_k", 0);
    const long long nl = r3d_geti_or(a, "n_local", ne);
    if (ne <= 0 || ne > 512 || tk <= 0 || tk > kW || tk > ne) return R3D_NO(RAD_E_SHAPE);
    if (nl <= 0 || nl > ne) return R3D_NO(RAD_E_SHAPE);

    if (x->dtype != RAD_BF16 || wr->dtype != RAD_BF16 || lg->dtype != RAD_BF16 ||
        w->dtype != RAD_BF16)
        return R3D_NO(RAD_E_DTYPE);
    if (wg && (wg->dtype != RAD_BF16 || gate->dtype != RAD_BF16)) return R3D_NO(RAD_E_DTYPE);
    if (ids->dtype != RAD_I32 || st->dtype != RAD_I32 || off->dtype != RAD_I32 ||
        cnt->dtype != RAD_I32)
        return R3D_NO(RAD_E_DTYPE);

    long long M = 0, K = 0;
    if (r3d_rows_of(x, &M, &K) != RAD_OK) return R3D_NO(RAD_E_SHAPE);
    if (M < 1 || M > 64) return R3D_NO(RAD_E_SHAPE);
    /* `M` is the band's geometry, not necessarily this step's rows: a declare binds the band's
     * upper bound while the activation carries the step's shape. The selection reads `M` and
     * writes that many rows, so hand it the rows the GEMM will actually write; a band that does
     * not even cover the step is a shape this row cannot serve. */
    const long long bandM = r3d_geti_or(a, "M", M);
    if (bandM < M) return R3D_NO(RAD_E_SHAPE);
    if (r3d_geti_or(a, "K", K) != K) return R3D_NO(RAD_E_SHAPE);
    if (r3d_dim(wr, 0) != ne || r3d_dim(wr, 1) != K) return R3D_NO(RAD_E_SHAPE);
    /* The activation may be a row slice of a wider buffer, so take its pitch as the m16 and tiled
     * rows do; the weight and the logits are the dedicated dense planes. */
    long long xrows = 0;
    const long long a_ld = r3d_rows_pitch(x, K, &xrows);
    if (a_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (xrows < M) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(wr) || !r3d_contig(lg)) return R3D_NO(RAD_E_STRIDE);
    if (a_ld % 8) return R3D_NO(RAD_E_STRIDE);          /* the m16 arm's 16-byte row bases */
    if (K % 32) return R3D_NO(RAD_E_SHAPE);             /* the tiled arm's 32-code chunk */
    if (r3d_numel(lg) < M * ne) return R3D_NO(RAD_E_SHAPE);

    const long long T = M * tk;
    if (T > 1024) return R3D_NO(RAD_E_SHAPE);            /* the sort's one-workgroup form */
    if (!r3d_contig(ids) || !r3d_contig(w) || !r3d_contig(st) || !r3d_contig(off) ||
        !r3d_contig(cnt))
        return R3D_NO(RAD_E_STRIDE);
    if (r3d_numel(ids) < T || r3d_numel(w) < T || r3d_numel(st) < T) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(off) < nl + 1 || r3d_numel(cnt) < nl) return R3D_NO(RAD_E_SHAPE);
    if (wg) {
        if (r3d_dim(wg, 0) != 1 || r3d_dim(wg, 1) != K) return R3D_NO(RAD_E_SHAPE);
        if (!r3d_contig(wg) || !r3d_contig(gate)) return R3D_NO(RAD_E_STRIDE);
        if (r3d_numel(gate) < M) return R3D_NO(RAD_E_SHAPE);
    }
    if (!r3d_aligned(r3d_p(x), 16) || !r3d_aligned(r3d_p(wr), 16) ||
        (wg && !r3d_aligned(r3d_p(wg), 16)))
        return R3D_NO(RAD_E_ALIGN);

    /* 1) The router GEMM: the same row the unfused router resolves to, m16 at decode heights and
     * the tiled row above. */
    int rc = rg_gemm(a, s, *x, *wr, *lg, ne, K, M <= 16);
    if (rc != RAD_OK) return rc;

    /* 2) The shared expert's gate, at N = 1 over the same activation. */
    if (wg) {
        rc = rg_gate(a, s, *x, *wg, *gate, M, K, a_ld);
        if (rc != RAD_OK) return rc;
    }

    /* 3) The selection and the sort, the device row itself; every parameter it reads is a
     * parameter of this op under the same name, except `M` which becomes the step's rows. */
    RadTensor t6[6] = { *lg, *ids, *w, *st, *off, *cnt };
    constexpr int kMaxParams = 32;   /* the schema's nine plus the tuning axes; no heap in a launch */
    RadParam pc[kMaxParams];
    int n_pc = 0;
    if (a->p && a->n_p > 0) {
        if (a->n_p > kMaxParams) return R3D_NO(RAD_E_UNSUPPORTED);
        std::memcpy(pc, a->p, (size_t)a->n_p * sizeof(RadParam));
        n_pc = a->n_p;
    }
    bool m_set = false;
    for (int i = 0; i < n_pc; ++i)
        if (pc[i].key && !std::strcmp(pc[i].key, "M") && pc[i].kind == RAD_P_INT) {
            pc[i].ival = M; m_set = true;
        }
    if (!m_set) pc[n_pc++] = RadParam{ "M", RAD_P_INT, (long long)M, 0, nullptr, 0.0 };
    RadArgs fa;
    std::memset(&fa, 0, sizeof fa);
    fa.t = t6; fa.n_t = 6;
    fa.p = pc; fa.n_p = n_pc;
    fa.rank = a->rank; fa.world_size = a->world_size;
    return r3d_rad_router_topk_scatter(&fa, s);
}
