/* r3d_qsa_block_key.hip -- qsa_block_key: the QSA indexer's compressed key, one vector per
 * `ratio` tokens of key history.
 *
 *   pooled = mean over the block's `ratio` raw keys, accumulated in f32 and rounded to bf16
 *   normed = pooled * rsqrt(mean(pooled^2) + eps) * (wadd + w)
 *   out    = NeoX rotary of normed's first `rotary_dim` lanes at the block's first position,
 *            the rest passed through, rounded once at the store
 *
 * The mean is rounded to bf16 before the norm, which is the reference's order
 * (`key_groups.float().mean(dim=1).to(raw_keys.dtype)`); carrying f32 through gives a small,
 * systematic difference on every block. `ss` is accumulated in double in ascending t, as libref
 * does, so the norm is the same number. The rotation is `rope`'s ladder with YaRN's blend and
 * magnitude, and under mrope/imrope each frequency reads the component its section names.
 *
 * `page[b]`, when present, is the output row; a negative entry writes nothing and a row past the
 * store is dropped rather than writing past it. `pos[b]` is the block's first-token position
 * (rank 2 holds a block's components side by side, read under `mode`), and without it block b is
 * at `pos0 + b*ratio`.
 *
 * One workgroup a block: n threads pooled, thread 0's double sum of the squares, then the norm
 * and rotation. n is bounded by the LDS (4096 floats); the row's constraint is n <= 4096.
 */
#include <hip/hip_runtime.h>

#include "rad_yarn.h"
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_launch.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kThreads = 256;

/* The component frequency i reads. mrope's sections are contiguous ([0, s0), then [s0, s0+s1),
 * then the rest); imrope's are interleaved (component 1 where i % 3 == 1 and i < 3*s1, component 2
 * where i % 3 == 2 and i < 3*s2, component 0 elsewhere), and only when at least three sections
 * were given. With one position component every mode is NeoX. */
__device__ __forceinline__ int qsa_bkey_comp(int mode, int n_sec, int i, int s0, int s1, int s2) {
    if (mode == 3 && n_sec >= 3) {
        if (i % 3 == 1 && i < 3 * s1) return 1;
        if (i % 3 == 2 && i < 3 * s2) return 2;
        return 0;
    }
    if (mode == 2 && n_sec > 0) return i < s0 ? 0 : (i < s0 + s1 ? 1 : 2);
    return 0;
}

__global__ __launch_bounds__(kThreads) void qsa_block_key_kernel(
    const uint16_t* __restrict__ k, const float* __restrict__ w,
    const int32_t* __restrict__ page, const int32_t* __restrict__ pos,
    uint16_t* __restrict__ out, int n, int ratio, int rot, float eps, float wadd, float theta,
    float rscale, long long pos0, int k_ld, int o_ld, int out_rows, int page_rs, int mode,
    int n_sec, int s0, int s1, int s2, int pos_rs, int pos_cs, int mc, RadYarn yarn) {
    R3D_DYN_LDS(float, sh);
    __shared__ float inv_sh;

    const int b = blockIdx.x;
    const int t0 = threadIdx.x;

    /* Pool the block's rows in f32 and round the mean to bf16 once. */
    for (int t = t0; t < n; t += kThreads) {
        float acc = 0.0f;
        const size_t base = (size_t)b * (size_t)ratio;
        for (int j = 0; j < ratio; ++j)
            acc += bf16_to_f32(k[(base + (size_t)j) * (size_t)k_ld + (size_t)t]);
        sh[t] = bf16_to_f32(f32_to_bf16(acc / (float)ratio));
    }
    __syncthreads();

    /* The reference's double sum, in ascending t. */
    if (t0 == 0) {
        double ss = 0.0;
        for (int t = 0; t < n; ++t) {
            const double x = (double)sh[t];
            ss += x * x;
        }
        inv_sh = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    }
    __syncthreads();

    const float inv = inv_sh;
    for (int t = t0; t < n; t += kThreads) sh[t] = sh[t] * inv * (wadd + w[t]);
    __syncthreads();

    /* A negative page is a block with no page; it writes nothing. */
    long long orow = b;
    if (page) {
        const int32_t p = page[(size_t)b * (size_t)page_rs];
        if (p < 0) return;
        orow = p;
    }
    if (orow < 0 || orow >= out_rows) return;
    uint16_t* __restrict__ dst = out + (size_t)orow * (size_t)o_ld;

    if (rot > 0) {
        const int half = rot >> 1;
        for (int i = t0; i < half; i += kThreads) {
            const int comp = mc ? qsa_bkey_comp(mode, n_sec, i, s0, s1, s2) : 0;
            long long bp;
            if (!pos)
                bp = pos0 + (long long)b * (long long)ratio;
            else
                bp = (long long)pos[(size_t)b * (size_t)pos_rs + (size_t)comp * (size_t)pos_cs];
            const float p = (float)bp / (rscale != 0.0f ? rscale : 1.0f);
            const float f =
                1.0f / (float)pow((double)theta, (double)((float)(2 * i) / (float)rot));
            const float iv = yarn.frequency(f, i);
            const float ang = p * iv;
            const float c = (float)cos((double)ang) * yarn.magnitude;
            const float s = (float)sin((double)ang) * yarn.magnitude;
            const float x0 = sh[i], x1 = sh[i + half];
            dst[i] = f32_to_bf16(x0 * c - x1 * s);
            dst[i + half] = f32_to_bf16(x0 * s + x1 * c);
        }
        for (int t = rot + t0; t < n; t += kThreads) dst[t] = f32_to_bf16(sh[t]);
    } else {
        for (int t = t0; t < n; t += kThreads) dst[t] = f32_to_bf16(sh[t]);
    }
}

}  // namespace

extern "C" {

int r3d_rad_qsa_block_key(const RadArgs* a, RadStream s)
{
    if (!a || a->n_t < 5) return R3D_NO(RAD_E_INVAL);
    const RadTensor* k = r3d_opt(a, 0);
    const RadTensor* w = r3d_opt(a, 1);
    const RadTensor* pg = r3d_opt(a, 2);
    const RadTensor* po = r3d_opt(a, 3);
    const RadTensor* out = r3d_opt(a, 4);
    if (!r3d_p(k) || !r3d_p(w) || !r3d_wp(out)) return R3D_NO(RAD_E_INVAL);

    const long long n = r3d_geti_or(a, "n", 0);
    const long long ratio = r3d_geti_or(a, "ratio", 0);
    const long long rot = r3d_geti_or(a, "rotary_dim", 0);
    const long long pos0 = r3d_geti_or(a, "pos0", -1);
    const bool has_page = r3d_has(pg);
    const bool has_pos = r3d_has(po);
    if (n <= 0 || n > 4096 || ratio <= 0 || (pos0 < 0 && !has_pos))
        return R3D_NO(RAD_E_INVAL);
    if (rot < 0 || rot > n || (rot & 1)) return R3D_NO(RAD_E_SHAPE);
    const long long bs = r3d_geti_or(a, "block_size", ratio);
    if (bs != ratio) return R3D_NO(RAD_E_SHAPE);

    if (k->rank != 2 || out->rank < 2 || out->shape[0] < 1) return R3D_NO(RAD_E_SHAPE);
    const long long M = has_page ? r3d_numel(pg) : out->shape[0];
    if (M < 1) return R3D_NO(RAD_E_SHAPE);
    /* The shape comparison below forms M*ratio; keep it inside int64. */
    if (M > 0x7fffffffLL || ratio > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(out) / out->shape[0] != n) return R3D_NO(RAD_E_SHAPE);
    if (k->shape[0] != M * ratio || k->shape[1] != n) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(w) < n) return R3D_NO(RAD_E_SHAPE);
    if (has_pos) {
        if (po->rank == 2) {
            if (po->shape[0] < M) return R3D_NO(RAD_E_SHAPE);
        } else if (po->rank == 1) {
            if (r3d_numel(po) < M) return R3D_NO(RAD_E_SHAPE);
        } else {
            return R3D_NO(RAD_E_SHAPE);
        }
    }

    if (k->dtype != RAD_BF16 || out->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (w->dtype != RAD_F32 || !r3d_contig(w)) return R3D_NO(RAD_E_DTYPE);
    if (has_page && pg->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);
    if (has_pos && po->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);

    static const char* kModes[] = { "neox", "mrope", "imrope" };
    int mode = 0;
    if (const char* nm = r3d_gets(a, "mode")) {
        int mi = -1;
        for (int i = 0; i < 3; ++i)
            if (!std::strcmp(nm, kModes[i])) { mi = i; break; }
        if (mi < 0) return R3D_NO(RAD_E_UNSUPPORTED);
        mode = mi == 1 ? 2 : mi == 2 ? 3 : 0;
    }
    long long sec[3] = {};
    int n_sec = 0;
    if (const char* v = r3d_gets(a, "sections")) {
        while (*v) {
            char* end = nullptr;
            const long long x = std::strtoll(v, &end, 10);
            if (end == v) break;
            if (x <= 0 || n_sec >= 3) return R3D_NO(RAD_E_SHAPE);
            sec[n_sec++] = x;
            v = end;
        }
    }
    const bool pos2 = has_pos && po->rank == 2;
    const bool mc = mode != 0 && n_sec > 0 && pos2 && po->shape[1] >= 3;
    if (mc && mode == 2) {
        long long total = 0;
        for (int i = 0; i < n_sec; ++i) total += sec[i];
        if (total != rot / 2) return R3D_NO(RAD_E_SHAPE);
    }
    /* The kernel compares i < 3*s, i < n/2; clamping a section to n keeps that truth and keeps
     * the int arithmetic away from overflow. A mrope total above has already refused a section
     * larger than n/2. */
    for (int i = 0; i < n_sec; ++i)
        if (sec[i] > n) sec[i] = n;

    const float eps = (float)r3d_getf(a, "eps", 1e-6);
    const float wadd = (float)r3d_getf(a, "wadd", 1.0);
    const float theta = (float)r3d_getf(a, "theta", 10000.0);
    const float rscale = (float)r3d_getf(a, "scale", 1.0);
    const RadYarn yarn = rad_yarn_args(a);
    if (!rad_yarn_valid(yarn) || (yarn.factor != 0.0f && (rot <= 0 || rscale != 1.0f)))
        return R3D_NO(RAD_E_INVAL);

    const long long k_ld = r3d_row_pitch(k);
    const long long o_ld = r3d_row_pitch(out);
    if (k_ld < 0 || k_ld < n) return R3D_NO(RAD_E_STRIDE);
    if (o_ld < 0 || o_ld < n) return R3D_NO(RAD_E_STRIDE);

    long long page_rs = 1;
    if (has_page) {
        if (pg->rank == 1) {
            if (pg->shape[0] != M) return R3D_NO(RAD_E_SHAPE);
            page_rs = pg->stride[0];
        } else if (pg->rank == 2 && pg->shape[1] == 1) {
            if (pg->shape[0] != M) return R3D_NO(RAD_E_SHAPE);
            page_rs = pg->stride[0];
        } else {
            return R3D_NO(RAD_E_SHAPE);
        }
        if (page_rs < 1) return R3D_NO(RAD_E_STRIDE);
    }

    long long pos_rs = 1, pos_cs = 0;
    if (has_pos) {
        pos_rs = po->stride[0];
        pos_cs = po->rank == 2 ? po->stride[1] : 0;
        if (pos_rs < 1 || pos_cs < 0) return R3D_NO(RAD_E_STRIDE);
    }

    if (M > 0x7fffffffLL || n > 0x7fffffffLL || ratio > 0x7fffffffLL ||
        rot > 0x7fffffffLL || k_ld > 0x7fffffffLL || o_ld > 0x7fffffffLL ||
        out->shape[0] > 0x7fffffffLL || page_rs > 0x7fffffffLL ||
        pos_rs > 0x7fffffffLL || pos_cs > 0x7fffffffLL)
        return R3D_NO(RAD_E_SHAPE);

    R3D_KLAUNCH_S(qsa_block_key_kernel, dim3((unsigned)M), dim3(kThreads),
                  (size_t)n * sizeof(float), (hipStream_t)s,
                  (const uint16_t*)r3d_p(k), (const float*)r3d_p(w),
                  has_page ? (const int32_t*)r3d_p(pg) : nullptr,
                  has_pos ? (const int32_t*)r3d_p(po) : nullptr,
                  (uint16_t*)r3d_wp(out), (int)n, (int)ratio, (int)rot, eps, wadd, theta,
                  rscale, pos0, (int)k_ld, (int)o_ld, (int)out->shape[0], (int)page_rs, mode,
                  n_sec, (int)sec[0], (int)sec[1], (int)sec[2], (int)pos_rs, (int)pos_cs,
                  mc ? 1 : 0, yarn);
    return (int)hipGetLastError();
}

}  // extern "C"
