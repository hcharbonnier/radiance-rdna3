/* r3d_gemm_iqr_gated.hip -- gemm_nt_q_gated over a rotated int8 (i8a8h) IQR weight, the 27B's
 * feed-forward fold: the gate and up halves of a [2n, K] weight, the silu gate, the 128-wide
 * Hadamard and the per-group int8 quantiser in one op.
 *
 * The device pieces already exist and each is the pair's own kernel: gemm_iqr_m16 computes the
 * [M, 2n] bf16 gate_up with the IQR decode, and gated_had_quant_i8g reads it back, gates, rounds,
 * rotates and codes. The fold saves the graph edge and one host dispatch, not a device launch: both
 * kernels still run, and the [M, 2n] bf16 intermediate is the same buffer libr4d's fused gated GEMM
 * stages in its scratch (`r4d_gemm_iqr_gated`). Fusing the epilogue into the decode kernel is
 * W12's work.
 *
 * The rotator's width and the quantiser's group are both 128, as the ffn fold declares them.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"
#include <cstring>

extern "C" int r3d_rad_gemm_nt_q_iqr_m16(const RadArgs*, RadStream);
extern "C" int r3d_rad_gated_had_quant_i8g(const RadArgs*, RadStream);

namespace {

constexpr int64_t kGroup = 128;

int64_t align256(int64_t v) { return (v + 255) & ~(int64_t)255; }

}  // namespace

extern "C" {

int64_t r3d_rad_gemm_nt_q_iqr_gated_scratch(const RadArgs* a) {
    if (!a) return 0;
    const long long M = r3d_geti_or(a, "M", 0), N = r3d_geti_or(a, "N", 0);
    if (M < 1 || N < 2) return 0;
    return align256((int64_t)M * (int64_t)N * 2);
}

int r3d_rad_gemm_nt_q_iqr_gated(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 6) return R3D_NO(RAD_E_INVAL);
    const char* dt = r3d_gets(a, "dtype");
    if (!dt || std::strcmp(dt, "i8a8h") != 0) return R3D_NO(RAD_E_DTYPE);
    const char* act = r3d_gets(a, "act");
    if (!act || (std::strcmp(act, "silu") != 0 && std::strcmp(act, "sigmoid") != 0))
        return R3D_NO(RAD_E_UNSUPPORTED);

    const RadTensor* A = r3d_opt(a, 0), *AS = r3d_opt(a, 1), *B = r3d_opt(a, 2),
                    *BS = r3d_opt(a, 3), *Q = r3d_opt(a, 4), *SC = r3d_opt(a, 5),
                    *OB = r3d_opt(a, 6);
    if (!r3d_p(A) || !r3d_p(AS) || !r3d_p(B) || !r3d_p(BS) || !r3d_wp(Q) || !r3d_wp(SC))
        return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_i8(A) || !r3d_is_f32(AS) || !r3d_is_i8(Q) || !r3d_is_f32(SC))
        return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(OB) && OB->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);

    const long long N = r3d_geti_or(a, "N", 0), K = r3d_geti_or(a, "K", 0),
                    group = r3d_geti_or(a, "group", 0);
    if (N < 2 || (N & 1) || K < 128) return R3D_NO(RAD_E_SHAPE);
    if (group != kGroup) return R3D_NO(RAD_E_UNSUPPORTED);
    if (K % (long long)kGroup) return R3D_NO(RAD_E_SHAPE);
    const long long n = N / 2;
    long long qm = 0, qn = 0;
    if (r3d_rows_of(Q, &qm, &qn) != 0 || qn < n) return R3D_NO(RAD_E_SHAPE);
    /* The band's `M` is the largest step (16); the tensors hold this step's rows, which is what
     * the scratch's arithmetic uses. */
    const long long M = qm;
    if (M < 1 || M > 16) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(Q)) return R3D_NO(RAD_E_STRIDE);
    if (r3d_numel(SC) < M * (n / (long long)kGroup)) return R3D_NO(RAD_E_SHAPE);
    if (r3d_has(OB) && r3d_numel(OB) < M * n) return R3D_NO(RAD_E_SHAPE);

    const int64_t need = (int64_t)M * (int64_t)N * 2;
    if (!a->scratch || (int64_t)a->scratch_bytes < need) return R3D_NO(RAD_E_SCRATCH);

    RadTensor H;
    std::memset(&H, 0, sizeof H);
    H.data = a->scratch;
    H.dtype = RAD_BF16;
    H.rank = 2;
    H.shape[0] = M;
    H.shape[1] = N;
    H.stride[0] = N;
    H.stride[1] = 1;

    /* The gate_up half: the same row the unfused graph calls, with the fold's scratch as its y. */
    RadTensor t1[5] = { *A, *AS, *B, *BS, H };
    RadArgs a1 = { t1, 5, a->p, a->n_p, nullptr, 0, nullptr, 0, 1 };
    const int rc = r3d_rad_gemm_nt_q_iqr_m16(&a1, s);
    if (rc != RAD_OK) return rc;

    const RadParam pg[] = { { "group", RAD_P_INT, kGroup, 0, nullptr, 0.0 },
                            { "dtype", RAD_P_STR, 0, 0, "bf16", 0.0 },
                            { "act", RAD_P_STR, 0, 0, act, 0.0 } };
    RadTensor t2[4] = { H, *Q, *SC, r3d_has(OB) ? *OB : RadTensor{} };
    RadArgs a2 = { t2, 4, pg, 3, nullptr, 0, nullptr, 0, 1 };
    return r3d_rad_gated_had_quant_i8g(&a2, s);
}

}  // extern "C"
