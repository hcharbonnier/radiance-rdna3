/* r3d_rope_imrope_h128.hip -- `rope` at the QSA indexer's geometry: imrope over NeoX pairs on
 * head_dim 128, rotary_dim 64, n_head 4, n_head_kv 0.
 *
 * The layer's indexer rotates its four 128-wide query heads at the position's three components
 * (Qwen3-VL's interleaved rotary, sections 11 11 10); the fused attention block at head_dim 256 is
 * the separate `rope_imrope_h256` row and is not touched by this unit.
 *
 * The arithmetic is libref's ref_rope:
 *
 *   c(i)   = 1 when i % 3 == 1 and i < 3*s1, 2 when i % 3 == 2 and i < 3*s2, else 0
 *   inv(i) = 1 / (float)pow((double)theta, (double)((float)(2*i) / (float)rot))
 *   inv(i) = yarn.frequency(inv(i), i)                    YaRN's blend, off when factor 0
 *   ang    = ((float)pos[m][c(i)] / scale) * inv(i)
 *   x0     = row[h*head_dim + i], x1 = row[h*head_dim + i + rot/2]
 *   row[h*head_dim + i]         = x0 * cos(ang)*mag - x1 * sin(ang)*mag
 *   row[h*head_dim + i + rot/2] = x0 * sin(ang)*mag + x1 * cos(ang)*mag
 *
 * Every angle is built in double and narrowed once (ref_pow, ref_cos and ref_sin), as libref does;
 * the pair's two products are f32 and each store narrows once. The lanes past `rotary_dim` are not
 * read or written, so a partial rotary leaves the head's tail untouched.
 *
 * `positions` is one component per token (rank 1), or a component plane in either layout:
 * [M, 3] token-major (the QSA block-key path's `bpos` form) or [3, M] component-major (what
 * `RadBatch::rope_pos` is issued as). A rank-2 plane is read at its own row and component strides,
 * so a pitched view is read in place. With no `sections`, a rank-1 position plane reduces to
 * NeoX, exactly as ref_rope treats it; a rank-2 plane without sections is refused (ref does the
 * same).
 */
#include <hip/hip_runtime.h>

#include "rad_yarn.h"
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ int32_t ldg_i32(const int32_t* p) { return *(const int32_t*)p; }
__device__ __forceinline__ uint16_t ldg_u16_at(const uint16_t* p, int i) {
    return (uint16_t)ldg_u16(p + i);
}

/* imrope's component for frequency i (Qwen3-VL). A section past the rotary width covers every
 * frequency below rot/2, so the shim clamps each section to rot/2 and the product stays exact. */
__device__ __forceinline__ int rope_imrope_comp(int i, int s1, int s2) {
    if (i % 3 == 1 && (long long)i < 3LL * (long long)s1) return 1;
    if (i % 3 == 2 && (long long)i < 3LL * (long long)s2) return 2;
    return 0;
}

__global__ __launch_bounds__(kThreads) void rope_imrope_h128_kernel(
    uint16_t* __restrict__ qkv, const int32_t* __restrict__ pos, int head_dim,
    int n_rope_heads, int rot, float theta, float div, int qkv_ld, int pos_rs, int pos_cs,
    int pos_multi, int s1, int s2, RadYarn yarn) {
    const int m = blockIdx.x;
    const int half = rot >> 1;
    const int total = n_rope_heads * half;
    uint16_t* __restrict__ row = qkv + (size_t)m * (size_t)qkv_ld;

    for (int j = threadIdx.x; j < total; j += kThreads) {
        const int h = j / half, i = j - h * half;
        const int c = pos_multi ? rope_imrope_comp(i, s1, s2) : 0;
        const long long poff = (long long)m * (long long)pos_rs +
                               (long long)c * (long long)pos_cs;
        const float p = (float)ldg_i32(pos + poff) / div;

        /* ref's ladder: `1.0f / ref_pow(theta, (float)(2i) / den)`, the double pow narrowed
         * once, then YaRN's blend on the f32 frequency. */
        const float base =
            1.0f / (float)pow((double)theta, (double)((float)(2 * i) / (float)rot));
        const float inv = yarn.frequency(base, i);
        const float ang = p * inv;
        const float cv = (float)cos((double)ang) * yarn.magnitude;
        const float sv = (float)sin((double)ang) * yarn.magnitude;

        const int i0 = h * head_dim + i;
        const int i1 = h * head_dim + i + half;
        const float x0 = bf16_to_f32(ldg_u16_at(row, i0));
        const float x1 = bf16_to_f32(ldg_u16_at(row, i1));
        ((uint16_t*)row)[i0] = f32_to_bf16(x0 * cv - x1 * sv);
        ((uint16_t*)row)[i1] = f32_to_bf16(x0 * sv + x1 * cv);
    }
}

}  /* namespace */

extern "C" int r3d_rad_rope_imrope_h128(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 2) return R3D_NO(RAD_E_INVAL);
    const RadTensor* qkv = r3d_opt(a, 0);
    const RadTensor* pos = r3d_opt(a, 1);
    if (!r3d_has(qkv) || !r3d_has(pos)) return R3D_NO(RAD_E_INVAL);
    if (qkv->dtype != RAD_BF16 || pos->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);

    const long long head_dim = r3d_geti_or(a, "head_dim", 0);
    const long long n_head = r3d_geti_or(a, "n_head", 0);
    const long long n_head_kv = r3d_geti_or(a, "n_head_kv", 0);
    if (head_dim <= 0 || n_head <= 0 || n_head_kv < 0) return R3D_NO(RAD_E_SHAPE);
    if (head_dim & 1) return R3D_NO(RAD_E_SHAPE);
    long long rot = r3d_geti_or(a, "rotary_dim", head_dim);
    if (rot <= 0 || rot > head_dim || (rot & 1)) return R3D_NO(RAD_E_SHAPE);

    const char* mode = r3d_gets(a, "mode");
    if (!mode || std::strcmp(mode, "imrope")) return R3D_NO(RAD_E_UNSUPPORTED);
    if (const char* lad = r3d_gets(a, "ladder")) {
        if (std::strcmp(lad, "rotary")) return R3D_NO(RAD_E_UNSUPPORTED);
    }
    if (r3d_getf(a, "llama3_factor", 0.0) != 0.0) return R3D_NO(RAD_E_UNSUPPORTED);

    const float theta = (float)r3d_getf(a, "theta", 10000.0);
    const float scale = (float)r3d_getf(a, "scale", 1.0);
    if (!(theta > 0.0f) || !std::isfinite(theta)) return R3D_NO(RAD_E_INVAL);
    const float div = scale != 0.0f ? scale : 1.0f;
    const RadYarn yarn = rad_yarn_args(a);
    if (!rad_yarn_valid(yarn)) return R3D_NO(RAD_E_INVAL);
    /* ref_rope refuses YaRN together with a position scale, on every mode. */
    if (yarn.factor != 0.0f && scale != 1.0f) return R3D_NO(RAD_E_UNSUPPORTED);

    long long s1 = 0, s2 = 0;
    int n_sec = 0;
    if (const char* sec = r3d_gets(a, "sections")) {
        long long v[3] = {};
        for (const char* p = sec; *p; ) {
            char* end = nullptr;
            const long long x = std::strtoll(p, &end, 10);
            if (end == p) break;                /* trailing text ends the list, as ref does */
            if (x <= 0) return R3D_NO(RAD_E_SHAPE);
            if (n_sec < 3) v[n_sec++] = x;      /* imrope reads only the first three */
            p = end;
        }
        if (n_sec != 3) return R3D_NO(RAD_E_SHAPE);
        s1 = v[1];
        s2 = v[2];
        /* `i < 3*s` for i < rot/2 is true for every s >= rot/2. Clamping there keeps the
         * predicate exact and the section product inside int; a section past the width is not an
         * error (ref's imrope test simply never reaches it). */
        const long long half = rot / 2;
        if (s1 > half) s1 = half;
        if (s2 > half) s2 = half;
    }

    /* Positions: [M] is one component per token; [M, 3] is token-major and [3, M] is
     * component-major. Both rank-2 forms carry the strides the caller bound. */
    long long M = 0, pos_rs = 1, pos_cs = 0;
    int pos_multi = 0;
    if (pos->rank == 1) {
        M = r3d_numel(pos);
        pos_rs = r3d_stride(pos, 0);
        if (pos_rs < 1) return R3D_NO(RAD_E_STRIDE);
    } else if (pos->rank == 2) {
        if (r3d_dim(pos, 0) == 3) {
            M = r3d_dim(pos, 1);
            pos_rs = r3d_stride(pos, 1);
            pos_cs = r3d_stride(pos, 0);
        } else if (r3d_dim(pos, 1) == 3) {
            M = r3d_dim(pos, 0);
            pos_rs = r3d_stride(pos, 0);
            pos_cs = r3d_stride(pos, 1);
        } else {
            return R3D_NO(RAD_E_SHAPE);
        }
        pos_multi = 1;
        if (M <= 0 || pos_rs < 1 || pos_cs < 1) return R3D_NO(RAD_E_STRIDE);
        if (n_sec == 0) return R3D_NO(RAD_E_SHAPE);   /* ref refuses a multi plane with no sections */
    } else {
        return R3D_NO(RAD_E_SHAPE);
    }
    if (M <= 0 || M > 0x7fffffffLL) return R3D_NO(RAD_E_SHAPE);
    if (n_sec == 0) { s1 = 0; s2 = 0; }

    const long long width = (n_head + 2 * n_head_kv) * head_dim;
    long long Mq = 0;
    const long long qkv_ld = r3d_rows_pitch(qkv, width, &Mq);
    if (qkv_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (qkv_ld < width || Mq < M) return R3D_NO(RAD_E_SHAPE);
    if (qkv_ld > 0x7fffffffLL || pos_rs > 0x7fffffffLL || pos_cs > 0x7fffffffLL ||
        width > 0x7fffffffLL)
        return R3D_NO(RAD_E_SHAPE);

    R3D_KLAUNCH(rope_imrope_h128_kernel, dim3((unsigned)M), dim3(kThreads), (hipStream_t)s,
                (uint16_t*)r3d_wp(qkv), (const int32_t*)r3d_p(pos), (int)head_dim,
                (int)(n_head + n_head_kv), (int)rot, theta, div, (int)qkv_ld, (int)pos_rs,
                (int)pos_cs, pos_multi, (int)s1, (int)s2, yarn);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
