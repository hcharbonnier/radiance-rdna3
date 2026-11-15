/* r3d_ar_hc_write.hip -- the collective HC write at one rank, and its gather-fused form.
 *
 * At world_size 1 an all-reduce is the identity, so ar_hc_write is hc_write over the message and
 * ar_hc_write's wire cannot change a byte. ar_gather_hc_write is moe_gather followed by
 * hc_write: the routed experts' rows are summed back to token order (the shared arm folded when
 * present) and that bf16 value is the message, then each stream takes `h[s] += inj[s] * y`.
 *
 * Only the one-rank forms are implemented here. Qwen4-Exp's target is tp:1, and a gfx11 card has
 * no peer fabric path in this library; a request for two or four ranks is refused rather than
 * run without its exchange. The schemas are libr4d's, copied slot for slot (the ops are
 * device-only and have no libref row).
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_hc_ple_common.h"
#include "r3d_launch.h"

namespace {

constexpr int kArThreads = 256;
constexpr int kArSlots = 64;

__global__ __launch_bounds__(kArThreads) void ar_gather_hc_write_kernel(
    const uint16_t* __restrict__ ye, const void* __restrict__ ew,
    const int32_t* __restrict__ sorted, int ew_f32, uint16_t* __restrict__ y,
    const uint16_t* __restrict__ inj, uint16_t* __restrict__ h, int M, int n, int hc, int top_k,
    int T, int ye_ld, int ew_ld, int y_ld, int inj_ld, int h_ld,
    const uint16_t* __restrict__ sh, const uint16_t* __restrict__ sg, int sh_ld, int sg_ld,
    int sig) {
    __shared__ int32_t s_i[kArSlots];
    __shared__ float s_w[kArSlots];
    const int t = (int)blockIdx.x;
    if (t >= M) return;

    float gate = 0.0f;
    if (sh && sg) {
        const float v0 = bf16_to_f32(sg[(long long)t * sg_ld]);
        gate = sig ? r3d_hcple::sigmoid(v0) : v0;
    }

    for (int j = (int)threadIdx.x; j < top_k; j += kArThreads) s_i[j] = -1;
    __syncthreads();
    const int32_t lo = (int32_t)(t * top_k), hi = lo + (int32_t)top_k;
    for (int i = (int)threadIdx.x; i < T; i += kArThreads) {
        const int32_t f = sorted[i];
        if (f >= lo && f < hi) s_i[f - lo] = i;
    }
    for (int j = (int)threadIdx.x; j < top_k; j += kArThreads)
        s_w[j] = ew_f32 ? ((const float*)ew)[(long long)t * ew_ld + j]
                        : bf16_to_f32(((const uint16_t*)ew)[(long long)t * ew_ld + j]);
    __syncthreads();

    for (int c = (int)(blockIdx.y * kArThreads + threadIdx.x); c < n; c += kArThreads * (int)gridDim.y) {
        float acc = 0.0f;
        for (int j = 0; j < top_k; ++j) {
            const int32_t i = s_i[j];
            if (i < 0) continue;
            acc += s_w[j] * bf16_to_f32(ye[(long long)i * ye_ld + c]);
        }
        const uint16_t g = f32_to_bf16(acc);
        uint16_t yv = g;
        if (sh && sg) {
            const uint16_t p = f32_to_bf16(bf16_to_f32(sh[(long long)t * sh_ld + c]) * gate);
            yv = f32_to_bf16(bf16_to_f32(p) + bf16_to_f32(g));
        }
        y[(long long)t * y_ld + c] = yv;
        const float yf = bf16_to_f32(yv);
        for (int s = 0; s < hc; ++s)
            h[(long long)t * h_ld + (long long)s * n + c] =
                f32_to_bf16(bf16_to_f32(h[(long long)t * h_ld + (long long)s * n + c]) +
                            bf16_to_f32(inj[(long long)t * inj_ld + s]) * yf);
    }
}

}  // namespace

extern "C" {

int r3d_ar_gather_hc_write(long ye, long ew, long sorted, long sh, long sg, long y, long inj,
                           long h, int M, int n, int hc, int top_k, int T, int ye_ld, int ew_ld,
                           int y_ld, int inj_ld, int h_ld, int ew_f32, int sh_ld, int sg_ld,
                           int sig, long stream) {
    if (!ye || !ew || !sorted || !y || !inj || !h) return -1;
    if (M <= 0 || n <= 0 || hc <= 1 || top_k <= 0 || top_k > kArSlots || T < M * top_k) return -2;
    if (ye_ld < n || ew_ld < top_k || y_ld < n || inj_ld < hc || h_ld < hc * n) return -3;
    dim3 grid((uint32_t)M, (uint32_t)((n + kArThreads - 1) / kArThreads));
    R3D_KLAUNCH(ar_gather_hc_write_kernel, grid, dim3(kArThreads), (hipStream_t)stream,
                (const uint16_t*)ye, (const void*)ew, (const int32_t*)sorted, ew_f32,
                (uint16_t*)y, (const uint16_t*)inj, (uint16_t*)h, M, n, hc, top_k, T, ye_ld,
                ew_ld, y_ld, inj_ld, h_ld, (const uint16_t*)sh, (const uint16_t*)sg, sh_ld, sg_ld,
                sig);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

extern "C" int r3d_rad_ar_hc_write(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    if (a->world_size != 1) return R3D_NO(RAD_E_UNSUPPORTED);
    if (r3d_geti_or(a, "wire", 0) != 0) return R3D_NO(RAD_E_UNSUPPORTED);
    const RadTensor* y = r3d_opt(a, 0), *inj = r3d_opt(a, 1), *h = r3d_opt(a, 2);
    if (!r3d_p(y) || !r3d_p(inj) || !r3d_wp(h)) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(y) || !r3d_is_bf16(inj) || !r3d_is_bf16(h)) return R3D_NO(RAD_E_DTYPE);
    const long long n = r3d_geti_or(a, "n", 0), hc = r3d_geti_or(a, "hc", 0);
    if (n <= 0 || hc <= 1) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0, Mi = 0;
    const long long h_ld = r3d_rows_pitch(h, hc * n, &M);
    const long long y_ld = r3d_rows_pitch(y, n, &My);
    const long long inj_ld = r3d_rows_pitch(inj, hc, &Mi);
    if (h_ld < 0 || y_ld < 0 || inj_ld < 0 || M <= 0) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(y) || !r3d_contig(inj) || !r3d_contig(h)) return R3D_NO(RAD_E_STRIDE);
    if (My != M || Mi != M) return R3D_NO(RAD_E_SHAPE);
    const int rc = r3d_hc_write((long)(uintptr_t)r3d_p(y), (long)(uintptr_t)r3d_p(inj),
                                (long)(uintptr_t)r3d_wp(h), (int)M, (int)n, (int)hc, (int)y_ld,
                                (int)inj_ld, (int)h_ld, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

extern "C" int r3d_rad_ar_gather_hc_write(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 8) return R3D_NO(RAD_E_INVAL);
    if (a->world_size != 1) return R3D_NO(RAD_E_UNSUPPORTED);
    if (r3d_geti_or(a, "wire", 0) != 0) return R3D_NO(RAD_E_UNSUPPORTED);
    const RadTensor* ye = r3d_opt(a, 0), *ew = r3d_opt(a, 1), *sorted = r3d_opt(a, 2),
                    *sh = r3d_opt(a, 3), *sg = r3d_opt(a, 4), *y = r3d_opt(a, 5),
                    *inj = r3d_opt(a, 6), *h = r3d_opt(a, 7);
    if (!r3d_p(ye) || !r3d_p(ew) || !r3d_p(sorted) || !r3d_wp(y) || !r3d_p(inj) || !r3d_wp(h))
        return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(ye) || !r3d_is_bf16(y) || !r3d_is_bf16(inj) || !r3d_is_bf16(h))
        return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_bf16(ew) && !r3d_is_f32(ew)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_i32(sorted)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(sh) != r3d_has(sg)) return R3D_NO(RAD_E_INVAL);
    if (r3d_has(sh) && (!r3d_is_bf16(sh) || !r3d_is_bf16(sg))) return R3D_NO(RAD_E_DTYPE);

    const long long n = r3d_geti_or(a, "n", 0), hc = r3d_geti_or(a, "hc", 0);
    const long long top_k = r3d_geti_or(a, "top_k", 0);
    if (n <= 0 || hc <= 1 || top_k <= 0) return R3D_NO(RAD_E_SHAPE);
    if (top_k > kArSlots) return R3D_NO(RAD_E_UNSUPPORTED);

    long long M = 0, My = 0, Mi = 0;
    const long long h_ld = r3d_rows_pitch(h, hc * n, &M);
    const long long y_ld = r3d_rows_pitch(y, n, &My);
    const long long inj_ld = r3d_rows_pitch(inj, hc, &Mi);
    long long mye = 0;
    const long long ye_ld = r3d_rows_pitch(ye, n, &mye);
    if (h_ld < 0 || y_ld < 0 || inj_ld < 0 || ye_ld < 0 || M <= 0) return R3D_NO(RAD_E_STRIDE);
    if (My != M || Mi != M) return R3D_NO(RAD_E_SHAPE);
    const long long T = r3d_numel(sorted);
    if (T < M * top_k || mye < T) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(ew) < M * top_k) return R3D_NO(RAD_E_SHAPE);

    const long long ew_ld = ew->rank >= 2 ? r3d_stride(ew, (int)ew->rank - 2) : top_k;
    if (ew_ld < top_k) return R3D_NO(RAD_E_STRIDE);
    int sh_ld = 0, sg_ld = 1;
    if (r3d_has(sh)) {
        long long msh = 0;
        sh_ld = (int)r3d_rows_pitch(sh, n, &msh);
        if (sh_ld < 0 || msh != M) return R3D_NO(RAD_E_SHAPE);
        long long msg = 0;
        const long long ld = r3d_rows_pitch(sg, 1, &msg);
        if (ld < 0 || msg != M) return R3D_NO(RAD_E_SHAPE);
        sg_ld = (int)ld;
    }
    int sig = 0;
    if (const char* act = r3d_gets(a, "act")) {
        if (!std::strcmp(act, "sigmoid")) sig = 1;
        else if (std::strcmp(act, "none")) return R3D_NO(RAD_E_INVAL);
    }

    const int rc = r3d_ar_gather_hc_write(
        (long)(uintptr_t)r3d_p(ye), (long)(uintptr_t)r3d_p(ew), (long)(uintptr_t)r3d_p(sorted),
        r3d_has(sh) ? (long)(uintptr_t)r3d_p(sh) : 0L,
        r3d_has(sg) ? (long)(uintptr_t)r3d_p(sg) : 0L, (long)(uintptr_t)r3d_wp(y),
        (long)(uintptr_t)r3d_p(inj), (long)(uintptr_t)r3d_wp(h), (int)M, (int)n, (int)hc,
        (int)top_k, (int)T, (int)ye_ld, (int)ew_ld, (int)y_ld, (int)inj_ld, (int)h_ld,
        ew->dtype == RAD_F32 ? 1 : 0, sh_ld, sg_ld, sig, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

}  /* extern "C" */
