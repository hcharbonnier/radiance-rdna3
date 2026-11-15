/* r3d_vision_bf16.hip -- W10-lite: the two ops the 27B's vision tower needs and libr3d lacked.
 *
 *   attn_dense at head_dim 72, bf16, gqa 1, non-causal: the ViT's full attention over one image.
 *   rope at mode axial, head_dim 72, sections 18 18: the ViT's 2-D rotary over (row, column).
 *
 * Both are transcriptions of libref, which is the oracle the selftest compares them to:
 *
 *   attn_dense (ref_attn.cpp:attn_dense)
 *     * scores: the dot over d ascending, then `dot * scale`, f32, no FMA;
 *     * the two-pass softmax: the maximum pass, then the score recomputed so the sum accumulates
 *       ascending like the reference's; every key of the segment is visible (non-causal, no
 *       window);
 *     * `sum += w` ascending and `acc[d] += w * v[d]` ascending, f32;
 *     * `out = acc[d] * (1 / sum)`, narrowed once to bf16.
 *     The one deliberate divergence is `expf` in place of `(float)exp((double)x)`: the device
 *     double transcendental is far too slow for M = 16384, and the f32 ULP it can differ by is far
 *     below the bf16 store's 1/256. The selftest states the bound (one bf16 code) and adds a
 *     uniform-score case where exp(0) is exactly 1 and the answer is bit-identical.
 *
 *   rope axial (ref_attn.c:ref_rope, rope_freqs ROPE_AXIAL)
 *     * NeoX pairs (i, i + rotary_dim / 2), q heads then k heads, v untouched;
 *     * section c owns a contiguous run of the rotary_dim / 2 frequencies and restarts the ladder
 *       at theta^(-2j / (2 * sections[c])) for its j-th;
 *     * the angle is `pos[c] * inv` in f32, and cos/sin are the double libm values narrowed once,
 *       as ref_cos/ref_sin do; the two products are f32 and the stores narrow once.
 *
 * A launcher takes pointers as longs and geometry as ints, checks what a __global__ cannot and
 * issues one kernel; the shim the row table names reads RadArgs and maps refusals to RAD codes.
 * Neither allocates nor synchronises.
 */

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kHd    = 72;       /* the vision tower's head width */
constexpr int kHalf  = kHd / 2;  /* NeoX pairs: i and i + 36 */
constexpr int kThr   = 128;      /* four wave32s: one thread a (row, head) or a rotary pair */

/* ================================================================== attn_dense, hd 72 */
/* One thread owns one (query row, query head). Its q head is widened into registers once; the two
 * passes walk the segment's keys in ascending order, so every f32 sum has the reference's
 * bracketing. With gqa 1 a head group is one head; the general ratio is kept because it costs
 * nothing here and keeps the kernel's contract honest. */
__global__ __launch_bounds__(kThr) void attn_dense_h72_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v, uint16_t* __restrict__ o,
    const int32_t* __restrict__ cu, int total_q, int n_seq, int q_heads, int gqa,
    long q_ld, long k_ld, long v_ld, long o_ld, float scale) {
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)total_q * q_heads;
    if (tid >= total) return;
    const int i = (int)(tid / q_heads);
    const int h = (int)(tid - (long)i * q_heads);

    /* The sequence that owns row i, exactly ref's seq_of scan. */
    int s = 0;
    while (s + 1 < n_seq && cu[s + 1] <= i) ++s;
    const int b0 = cu[s], b1 = cu[s + 1];
    const int kh = h / gqa;

    const uint16_t* qrow = q + (size_t)i * q_ld + (size_t)h * kHd;
    float qv[kHd];
#pragma unroll
    for (int d = 0; d < kHd; ++d) qv[d] = bf16_to_f32((uint16_t)ldg_u16(qrow + d));

    float mx = -INFINITY;
    for (int j = b0; j < b1; ++j) {
        const uint16_t* krow = k + (size_t)j * k_ld + (size_t)kh * kHd;
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < kHd; ++d) dot += qv[d] * bf16_to_f32((uint16_t)ldg_u16(krow + d));
        dot *= scale;
        if (dot > mx) mx = dot;
    }

    float acc[kHd];
#pragma unroll
    for (int d = 0; d < kHd; ++d) acc[d] = 0.0f;

    float sum = 0.0f;
    for (int j = b0; j < b1; ++j) {
        const uint16_t* krow = k + (size_t)j * k_ld + (size_t)kh * kHd;
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < kHd; ++d) dot += qv[d] * bf16_to_f32((uint16_t)ldg_u16(krow + d));
        const float w = expf(dot * scale - mx);
        sum += w;
        const uint16_t* vrow = v + (size_t)j * v_ld + (size_t)kh * kHd;
#pragma unroll
        for (int d = 0; d < kHd; ++d) acc[d] += w * bf16_to_f32((uint16_t)ldg_u16(vrow + d));
    }

    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    uint16_t* orow = o + (size_t)i * o_ld + (size_t)h * kHd;
#pragma unroll
    for (int d = 0; d < kHd; ++d) orow[d] = f32_to_bf16(acc[d] * inv);
}

}  // namespace

extern "C" int r3d_attn_dense_h72(long q, long k, long v, long o, long cu, int total_q,
                                  int n_seq, int q_heads, int gqa, long q_ld, long k_ld,
                                  long v_ld, long o_ld, float scale, long stream) {
    if (!q || !k || !v || !o || !cu) return -1;
    if (total_q <= 0 || n_seq <= 0 || q_heads <= 0 || gqa <= 0 || q_heads % gqa) return -2;
    if (q_ld < (long)q_heads * kHd || k_ld < (long)(q_heads / gqa) * kHd ||
        v_ld < (long)(q_heads / gqa) * kHd || o_ld < (long)q_heads * kHd)
        return -3;
    const long total = (long)total_q * q_heads;
    const long grid = (total + kThr - 1) / kThr;
    if (grid > 0x7fffffffL) return -2;
    R3D_KLAUNCH(attn_dense_h72_kernel, dim3((uint32_t)grid), dim3(kThr),
                (hipStream_t)(uintptr_t)stream, (const uint16_t*)(uintptr_t)q,
                (const uint16_t*)(uintptr_t)k, (const uint16_t*)(uintptr_t)v,
                (uint16_t*)(uintptr_t)o, (const int32_t*)(uintptr_t)cu, total_q, n_seq,
                q_heads, gqa, q_ld, k_ld, v_ld, o_ld, scale);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

static inline int dt_bf16(const RadTensor* t) { return r3d_is_bf16(t) ? RAD_OK : RAD_E_DTYPE; }
static inline int dt_i32(const RadTensor* t) { return r3d_is_i32(t) ? RAD_OK : RAD_E_DTYPE; }

#define R3D_NEED(x) do { const int _s = (x); if (_s != RAD_OK) return R3D_NO(_s); } while (0)

static inline int launch_ok(int rc) { return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE); }

/* The dense attention of the vision tower. `M` is the band; the row count, the heads and the
 * pitches come from the operands, as every shim here does. The row's constraints select only the
 * geometry this kernel implements: bf16, head_dim 72, gqa 1, non-causal, no window, no blocks.
 * `cu_div` is the one optional parameter libref would divide the boundaries by; 1 is the vision
 * tower's form. */
extern "C" int r3d_rad_attn_dense(const RadArgs* a, RadStream s) {
    const RadTensor* q = r3d_opt(a, 0);
    const RadTensor* k = r3d_opt(a, 1);
    const RadTensor* v = r3d_opt(a, 2);
    const RadTensor* cu = r3d_opt(a, 3);
    const RadTensor* o = r3d_opt(a, 4);
    if (!q || !k || !v || !cu || !o) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(q));
    R3D_NEED(dt_bf16(k));
    R3D_NEED(dt_bf16(v));
    R3D_NEED(dt_bf16(o));
    R3D_NEED(dt_i32(cu));

    if (r3d_geti_or(a, "head_dim", 0) != kHd) return R3D_NO(RAD_E_SHAPE);
    const long long gqa = r3d_geti_or(a, "gqa", 1);
    if (gqa <= 0) return R3D_NO(RAD_E_SHAPE);
    if (r3d_geti_or(a, "causal", 0) != 0) return R3D_NO(RAD_E_UNSUPPORTED);
    if (const char* nm = r3d_gets(a, "dtype"))
        if (std::strcmp(nm, "bf16")) return R3D_NO(RAD_E_UNSUPPORTED);
    /* The kernel has no window and no block walk. A caller that names one is a different op's
     * band, and the row's constraints do not carry optional keys (a band that omits one would not
     * resolve to this row), so the shim states the refusal here. */
    if (r3d_geti_or(a, "window", 0) != 0 || r3d_geti_or(a, "window_right", 0) != 0 ||
        r3d_geti_or(a, "block", 0) != 0 || r3d_geti_or(a, "cu_div", 1) != 1)
        return R3D_NO(RAD_E_UNSUPPORTED);

    long long q_tok = 0, q_nh = 0, q_ld = 0;
    long long k_tok = 0, k_nh = 0, k_ld = 0;
    long long v_tok = 0, v_nh = 0, v_ld = 0;
    long long o_tok = 0, o_nh = 0, o_ld = 0;
    if (!r3d_head_rows(q, kHd, &q_tok, &q_nh, &q_ld)) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_head_rows(k, kHd, &k_tok, &k_nh, &k_ld)) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_head_rows(v, kHd, &v_tok, &v_nh, &v_ld)) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_head_rows(o, kHd, &o_tok, &o_nh, &o_ld)) return R3D_NO(RAD_E_STRIDE);
    if (q_nh % gqa) return R3D_NO(RAD_E_SHAPE);
    const long long kv_nh = q_nh / gqa;
    if (k_nh != kv_nh || v_nh != kv_nh) return R3D_NO(RAD_E_SHAPE);
    if (k_tok != q_tok || v_tok != q_tok || o_tok != q_tok || o_nh != q_nh)
        return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(cu)) return R3D_NO(RAD_E_STRIDE);
    const long long n_seq = r3d_numel(cu) - 1;
    if (n_seq <= 0) return R3D_NO(RAD_E_SHAPE);

    /* libref's default is `1.0f / sqrt((float)head_dim)`, not the double quotient narrowed. */
    const float dflt_scale = 1.0f / std::sqrt((float)kHd);
    const float scale = (float)r3d_getf(a, "scale", (double)dflt_scale);
    return launch_ok(r3d_attn_dense_h72(r3d_addr(q), r3d_addr(k), r3d_addr(v), r3d_addr(o),
                                        r3d_addr(cu), (int)q_tok, (int)n_seq, (int)q_nh, (int)gqa,
                                        (long)q_ld, (long)k_ld, (long)v_ld, (long)o_ld, scale,
                                        (long)(uintptr_t)s));
}

/* ================================================================== rope, mode axial, hd 72 */
/* One block a token; a thread owns one (rope head, frequency). Section c owns a contiguous run of
 * the 36 frequencies and restarts its ladder at theta^(-j / sections[c]), the form rope_freqs
 * spells theta^(-2j / (2 * sections[c])). Cos and sin are the double libm values narrowed once, as
 * ref_cos/ref_sin evaluate them; the angle itself is f32, as in ref_rope. */
__global__ __launch_bounds__(kThr) void rope_axial_h72_kernel(
    uint16_t* __restrict__ qkv, const int32_t* __restrict__ pos, uint32_t n_rope_heads,
    uint32_t row_pitch, uint32_t pos_cs, uint32_t pos_rs, uint32_t sec0, uint32_t sec1,
    float theta, float scale) {
    const uint32_t m = blockIdx.x;
    const uint32_t total = n_rope_heads * (uint32_t)kHalf;
    uint16_t* __restrict__ row = qkv + (size_t)m * row_pitch;
    const float div = scale != 0.0f ? scale : 1.0f;

    for (uint32_t j = threadIdx.x; j < total; j += kThr) {
        const uint32_t h = j / (uint32_t)kHalf, i = j - h * (uint32_t)kHalf;
        const uint32_t c = i < sec0 ? 0u : 1u;
        const uint32_t s = c ? sec1 : sec0;
        const uint32_t jj = i - (c ? sec0 : 0u);
        /* ref: 1.0f / ref_pow(theta, (float)(2*j) / (float)(2*sec[c])), the double pow narrowed
         * once; the reciprocal is f32. */
        const float inv = 1.0f / (float)std::pow((double)theta,
                                                 (double)((float)(2u * jj) / (float)(2u * s)));
        const float p = (float)pos[(size_t)c * pos_cs + (size_t)m * pos_rs] / div;
        const float ang = p * inv;
        const float co = (float)std::cos((double)ang);
        const float si = (float)std::sin((double)ang);
        const uint32_t i0 = h * (uint32_t)kHd + i;
        const uint32_t i1 = h * (uint32_t)kHd + i + (uint32_t)kHalf;
        const float x0 = bf16_to_f32((uint16_t)ldg_u16(row + i0));
        const float x1 = bf16_to_f32((uint16_t)ldg_u16(row + i1));
        ((uint16_t*)row)[i0] = f32_to_bf16(x0 * co - x1 * si);
        ((uint16_t*)row)[i1] = f32_to_bf16(x0 * si + x1 * co);
    }
}

extern "C" int r3d_rope_axial_h72(long qkv, long pos, int M, int n_head, int n_head_kv,
                                  int qkv_ld, int pos_cs, int pos_rs, int sec0, int sec1,
                                  float theta, float scale, long stream) {
    if (!qkv || !pos) return -1;
    if (M <= 0 || n_head <= 0 || n_head_kv < 0) return -2;
    if (qkv_ld < (n_head + 2 * n_head_kv) * kHd) return -3;
    if (pos_cs < 0 || pos_rs < 0 || sec0 <= 0 || sec1 <= 0 || sec0 + sec1 != kHalf) return -2;
    if (!(theta > 0.0f) || !std::isfinite(theta)) return -2;
    R3D_KLAUNCH(rope_axial_h72_kernel, dim3((uint32_t)M), dim3(kThr),
                (hipStream_t)(uintptr_t)stream, (uint16_t*)(uintptr_t)qkv,
                (const int32_t*)(uintptr_t)pos, (uint32_t)(n_head + n_head_kv),
                (uint32_t)qkv_ld, (uint32_t)pos_cs, (uint32_t)pos_rs, (uint32_t)sec0,
                (uint32_t)sec1, theta, scale);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* "18 18" -> two sections. Axial needs the sum to be rotary_dim / 2 and at least the component
 * count; this kernel serves the two-section form the vision tower declares. */
static int rope_sections2(const char* v, int* s0, int* s1) {
    int n = 0, sec[2] = { 0, 0 };
    while (v && *v && n < 2) {
        char* end = nullptr;
        const long x = std::strtol(v, &end, 10);
        if (end == v || x <= 0) return -1;
        sec[n++] = (int)x;
        v = end;
    }
    while (v && (*v == ' ' || *v == ',')) ++v;
    if (!v || *v || n != 2) return -1;
    *s0 = sec[0];
    *s1 = sec[1];
    return 0;
}

/* The vision tower's axial rotary. The row's constraints keep the band on this kernel; the shim
 * still checks every form the kernel does not implement (positions dtype, sections count, the
 * yarn/llama3 rescalings that would change the ladder). */
extern "C" int r3d_rad_rope_axial(const RadArgs* a, RadStream s) {
    const RadTensor* qkv = r3d_opt(a, 0);
    const RadTensor* pos = r3d_opt(a, 1);
    if (!qkv || !pos) return R3D_NO(RAD_E_INVAL);
    R3D_NEED(dt_bf16(qkv));
    R3D_NEED(dt_i32(pos));

    const long long head_dim = r3d_geti_or(a, "head_dim", 0);
    const long long n_head = r3d_geti_or(a, "n_head", 0);
    const long long n_head_kv = r3d_geti_or(a, "n_head_kv", 0);
    if (head_dim != kHd || n_head <= 0 || n_head_kv < 0) return R3D_NO(RAD_E_SHAPE);
    long long rot = r3d_geti_or(a, "rotary_dim", head_dim);
    if (rot != kHd) return R3D_NO(RAD_E_SHAPE);

    const char* mode = r3d_gets(a, "mode");
    if (!mode || std::strcmp(mode, "axial")) return R3D_NO(RAD_E_UNSUPPORTED);
    if (r3d_getf(a, "yarn_factor", 0.0) != 0.0) return R3D_NO(RAD_E_UNSUPPORTED);
    if (r3d_getf(a, "llama3_factor", 0.0) != 0.0) return R3D_NO(RAD_E_UNSUPPORTED);
    if (const char* lad = r3d_gets(a, "ladder")) {
        /* Axial overrides every frequency with its section ladder, so the base denominator does
         * not reach the result; an unknown spelling is still a typo and is refused. */
        if (std::strcmp(lad, "rotary") && std::strcmp(lad, "head"))
            return R3D_NO(RAD_E_UNSUPPORTED);
    }

    int sec0 = 0, sec1 = 0;
    if (rope_sections2(r3d_gets(a, "sections"), &sec0, &sec1) != 0)
        return R3D_NO(RAD_E_SHAPE);
    if (sec0 + sec1 != kHalf) return R3D_NO(RAD_E_SHAPE);

    const long long width = (n_head + 2 * n_head_kv) * kHd;
    long long Mq = 0;
    const long long qkv_ld = r3d_rows_pitch(qkv, width, &Mq);
    if (qkv_ld < 0) return R3D_NO(RAD_E_STRIDE);

    /* [C, M] component-major, read at its own strides (a view of a wider plane is read in place). */
    if (pos->rank < 2 || r3d_dim(pos, 0) < 2 || r3d_dim(pos, 1) <= 0) return R3D_NO(RAD_E_SHAPE);
    const long long M = r3d_dim(pos, 1);
    const long long pcs = r3d_stride(pos, 0), prs = r3d_stride(pos, 1);
    if (pcs < 0 || prs < 0 || pcs > 0x7fffffffLL || prs > 0x7fffffffLL) return R3D_NO(RAD_E_STRIDE);
    if (Mq < M) return R3D_NO(RAD_E_SHAPE);

    const float theta = (float)r3d_getf(a, "theta", 10000.0);
    const float scale = (float)r3d_getf(a, "scale", 1.0);
    if (!(theta > 0.0f) || !std::isfinite(theta)) return R3D_NO(RAD_E_INVAL);

    return launch_ok(r3d_rope_axial_h72(r3d_addr(qkv), r3d_addr(pos), (int)M, (int)n_head,
                                        (int)n_head_kv, (int)qkv_ld, (int)pcs, (int)prs, sec0,
                                        sec1, theta, scale, (long)(uintptr_t)s));
}
