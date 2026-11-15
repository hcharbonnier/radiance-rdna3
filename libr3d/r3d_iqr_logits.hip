/* r3d_iqr_logits.hip -- logits_gemm_vocab over an IQR head, on gfx11.
 *
 * The head is up to four width classes of the vocabulary (r3d_iqr.h's lm form): each class is an
 * IQR weight of rows<i> rows and its ids, the global vocabulary id of each row. The activation is
 * rotated by the same 128-wide Hadamard the head's codes were trained against, quantised to int8
 * with one f32 a (token, 128), and each class's GEMM writes bf16, which the scatter widens to the
 * op's f32 logits at the ids in range.
 *
 * W4 fallback: one quantise, class GEMMs and scatters, the class logits narrowed to bf16 between
 * the GEMM and the f32 logits. W11 replaces the class path with the head's own 16-bit wave matrix
 * so the accumulation stays f32; until then this is the row that lets a container serve.
 */
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_common.h"
#include "r3d_fwht.h"
#include "r3d_iqr.h"
#include "r3d_launch.h"

#include <cmath>
#include <string>

namespace {

using namespace r3d_iqr;

/* r3d_gemm_iqr_m16.hip: the chunked m16 launch the class GEMMs use until W11's head kernel. */
extern "C" int r3d_gemm_iqr_tiled(long a, long as, long b, long bs, long y, int M, int N, int K,
                                  int a_ld, int y_ld, uint32_t word, long stream);

constexpr uint32_t kThreads = 32;
constexpr float kTopI8 = 127.0f;

__device__ __forceinline__ float ld_bf(const uint16_t* p, uint32_t i) {
    return bf16_to_f32((uint16_t)ldg_u16(p + i));
}
__device__ __forceinline__ float clamp_top(float c, float top) {
    if (c > top) c = top;
    if (c < -top) c = -top;
    return c;
}
__device__ __forceinline__ uint32_t i8_codes4(const float* v, float inv) {
    const uint32_t b0 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[0] * inv, kTopI8));
    const uint32_t b1 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[1] * inv, kTopI8));
    const uint32_t b2 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[2] * inv, kTopI8));
    const uint32_t b3 = (uint32_t)(uint8_t)f32_to_i8_rtne(clamp_top(v[3] * inv, kTopI8));
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}
__device__ __forceinline__ float wave_max_f(float v) {
    v = fmaxf(v, r3d_xor_lane<16>(v));
    v = fmaxf(v, r3d_xor_lane<8>(v));
    v = fmaxf(v, r3d_xor_lane<4>(v));
    v = fmaxf(v, r3d_xor_lane<2>(v));
    v = fmaxf(v, r3d_xor_lane<1>(v));
    return v;
}

/* The head's activation: one wave a (token, 128-group), the FWHT of the rotated weights' input,
 * then int8 with the group's scale. */
__global__ __launch_bounds__(kThreads) void logits_quant_kernel(
    const uint16_t* __restrict__ x, uint8_t* __restrict__ q, float* __restrict__ s, uint32_t n,
    uint32_t x_ld) {
    constexpr int HAD = 128, VPL = HAD / 32;
    static_assert(VPL == 4, "one wave's 32 lanes hold the group's 128 columns as four each");
    const uint32_t row = blockIdx.x, kg = blockIdx.y;
    const uint32_t nkb = n / 128u;
    const uint32_t lane = threadIdx.x & 31;
    const uint16_t* a = x + (size_t)row * x_ld + (size_t)kg * HAD;
    uint8_t* qr = q + (size_t)row * n + (size_t)kg * HAD;
    const int base = lane * VPL;
    float v[VPL];
#pragma unroll
    for (int j = 0; j < VPL; ++j) v[j] = ld_bf(a, base + j);
    r3d_fwht_wave<HAD>(v, lane);
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < VPL; ++j) m = fmaxf(m, fabsf(v[j]));
    m = wave_max_f(m);
    m = m > 0.0f ? m : 1e-8f;
    const float inv = kTopI8 / m;
    if (lane == 0) s[(size_t)row * nkb + kg] = m * (1.0f / sqrtf((float)HAD)) / kTopI8;
    *(uint32_t*)(qr + (uint32_t)base) = i8_codes4(v, inv);
}

/* A class's ids scatter its bf16 rows to the f32 logits; ids outside this rank's vocabulary are
 * dropped (the head is sharded by `vocab_offset`). */
__global__ __launch_bounds__(128) void logits_scatter_kernel(
    const uint16_t* __restrict__ y, const int32_t* __restrict__ ids, float* __restrict__ logits,
    uint32_t n_vocab, uint32_t rows, int32_t voff, uint32_t y_ld, uint32_t ld) {
    const uint32_t m = blockIdx.x, r = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const int64_t id = (int64_t)ids[r] - voff;
    if (id < 0 || id >= (int64_t)n_vocab) return;
    logits[(size_t)m * ld + id] = bf16_to_f32((uint16_t)ldg_u16(y + (size_t)m * y_ld + r));
}

int64_t align256(int64_t v) { return (v + 255) & ~255ll; }

}  // namespace

extern "C" {

/* A's int8 codes, its scales, then the class logits' bf16 plane. */
int64_t r3d_logits_gemm_vocab_iqr_scratch(const RadArgs* a) {
    if (!a || a->n_t < 15) return 0;
    const RadTensor* x = r3d_opt(a, 0);
    const long long n_embd = r3d_geti_or(a, "n_embd", 0);
    long long M = r3d_geti_or(a, "M", 0);
    if (x && x->rank >= 2) M = x->shape[0] > M ? x->shape[0] : M;
    long long rows = 0;
    for (const char* k : { "rows0", "rows1", "rows2", "rows3" }) {
        const long long r = r3d_geti_or(a, k, 0);
        rows = r > rows ? r : rows;
    }
    if (M <= 0 || n_embd <= 0 || rows <= 0) return 0;
    if (n_embd % 128) return 0;
    return align256(M * n_embd) + align256(M * (n_embd / 128) * 4) + align256(M * rows * 2) + 256;
}

int r3d_rad_logits_gemm_vocab_iqr(const RadArgs* a, RadStream stream) {
    /* W11: the class GEMM through the decode row disagrees with the host reference on the 27B's
     * head form (same q/s, same weight: device 0.0449, reference -0.279). The row refuses until
     * that is resolved, so a container with a vocabulary-weighted head declares and fails loudly
     * at the first step instead of emitting wrong logits. See JOURNAL_RDNA3.md. */
    (void)a;
    (void)stream;
    return R3D_NO(RAD_E_UNSUPPORTED);
#if 0
    if (!a || a->n_t < 15 || !a->scratch) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0);
    const RadTensor* lg = r3d_opt(a, 14);
    if (!r3d_p(x) || !r3d_wp(lg)) return R3D_NO(RAD_E_INVAL);
    if (x->dtype != RAD_BF16 || lg->dtype != RAD_F32) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(x) || !r3d_contig(lg)) return R3D_NO(RAD_E_STRIDE);

    const long long n_embd = r3d_geti_or(a, "n_embd", 0);
    const long long n_vocab = r3d_geti_or(a, "n_vocab", 0);
    const long long voff = r3d_geti_or(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_embd % 128 || n_vocab <= 0 || voff < 0) return R3D_NO(RAD_E_SHAPE);

    long long M = 0, x_ld = 0;
    x_ld = r3d_rows_pitch(x, n_embd, &M);
    if (x_ld < n_embd || M <= 0 || M != r3d_geti_or(a, "M", M)) return R3D_NO(RAD_E_SHAPE);
    long long lrows = 0;
    const long long ld = r3d_rows_pitch(lg, n_vocab, &lrows);
    if (ld < n_vocab || lrows < M) return R3D_NO(RAD_E_SHAPE);

    long long rows[4] = { 0, 0, 0, 0 };
    long long maxrows = 0;
    for (int i = 0; i < 4; ++i) {
        rows[i] = r3d_geti_or(a, ("rows" + std::to_string(i)).c_str(), 0);
        maxrows = rows[i] > maxrows ? rows[i] : maxrows;
    }
    if (maxrows <= 0) return R3D_NO(RAD_E_SHAPE);

    /* The classes' weights: each present one is the IQR pair and its ids. */
    const RadTensor* w[4] = { r3d_opt(a, 1), r3d_opt(a, 4), r3d_opt(a, 7), r3d_opt(a, 10) };
    const RadTensor* sc[4] = { r3d_opt(a, 2), r3d_opt(a, 5), r3d_opt(a, 8), r3d_opt(a, 11) };
    const RadTensor* id[4] = { r3d_opt(a, 3), r3d_opt(a, 6), r3d_opt(a, 9), r3d_opt(a, 12) };
    for (int i = 0; i < 4; ++i) {
        if (rows[i] <= 0) { w[i] = sc[i] = id[i] = nullptr; continue; }
        if (!r3d_p(w[i]) || !r3d_p(sc[i]) || !r3d_p(id[i])) return R3D_NO(RAD_E_INVAL);
        if (!is_weight_dtype(w[i]->dtype) || !is_scale_dtype(sc[i]->dtype)) return R3D_NO(RAD_E_DTYPE);
        if (id[i]->dtype != RAD_I32 || r3d_numel(id[i]) < rows[i]) return R3D_NO(RAD_E_SHAPE);
        if (kernel_word(w[i]->dtype) != kernel_word(sc[i]->dtype)) return R3D_NO(RAD_E_DTYPE);
    }

    const int64_t need = r3d_logits_gemm_vocab_iqr_scratch(a);
    if (need <= 0 || a->scratch_bytes < need) return R3D_NO(RAD_E_SCRATCH);
    uint8_t* q = (uint8_t*)a->scratch;
    float* s = (float*)(q + align256(M * n_embd));
    uint16_t* y = (uint16_t*)((uint8_t*)s + align256(M * (n_embd / 128) * 4));

    const dim3 qgrid((uint32_t)M, (uint32_t)(n_embd / 128));
    R3D_KLAUNCH(R3D_KNAME(logits_quant_kernel), qgrid, dim3(kThreads), (hipStream_t)stream,
                (const uint16_t*)r3d_p(x), q, s, (uint32_t)n_embd, (uint32_t)x_ld);

    for (int i = 0; i < 4; ++i) {
        if (!w[i]) continue;
        const int rc = r3d_gemm_iqr_tiled((long)(uintptr_t)q, (long)(uintptr_t)s,
                                          (long)(uintptr_t)r3d_p(w[i]),
                                          (long)(uintptr_t)r3d_p(sc[i]), (long)(uintptr_t)y,
                                          (int)M, (int)rows[i], (int)n_embd, (int)n_embd,
                                          (int)rows[i], w[i]->dtype, (long)stream);
        if (rc) return R3D_NO(RAD_E_DEVICE);
        const dim3 sgrid((uint32_t)M, (uint32_t)((rows[i] + 127) / 128));
        R3D_KLAUNCH(R3D_KNAME(logits_scatter_kernel), sgrid, dim3(128), (hipStream_t)stream, y,
                    (const int32_t*)r3d_p(id[i]), (float*)r3d_wp(lg), (uint32_t)n_vocab,
                    (uint32_t)rows[i], (int32_t)voff, (uint32_t)rows[i], (uint32_t)ld);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : R3D_NO(RAD_E_DEVICE);
#endif
}

}  // extern "C"
