/* r3d_mtp_hc_e4m3.hip -- the MTP head's entry over a wide residual stream, its fc matrices stored
 * as E4M3 rows.
 *
 * The op is the one docs/OPS.md's `mtp_enter` row describes, and libref/ref_hc.cpp:344 is the
 * oracle:
 *
 *     ne[j]    = e[j] * rse * (w_e[j] + wadd),   rse = rsqrt(mean(e^2) + eps)
 *     fe[i]    = fc_e[i] . ne
 *     nh[k][j] = h[k*n+j] * rs[k] * (w_h[k*n+j] + wadd)
 *     x[k*n+i] = fc_h[i] . nh[k] + fe[i]
 *
 * `ngroup` is the checkpoint's one ambiguity: `hc` normalises every sub-stream on its own, `1`
 * normalises the whole wide row once. Both are accepted, as the reference.
 *
 * The stored fc row is libr4d's joined form (`r4d.mtp.e4m3.g128`): n E4M3 codes, then one f32
 * scale per 128 of K, then zero padding to 16 bytes. Publishing libr4d's tag keeps a container
 * laid out on RDNA4 readable here without a re-pack, as hc_read's E4M3 mix already does.
 *
 * One workgroup owns `kMtpChannels` output channels of one row. It recomputes the two norms and
 * stages `nh` and `ne` in LDS: 25 KiB at Flash-Next's shape against the 42 KiB of fc rows it then
 * reads, so a second launch and a scratch buffer for five scalars would buy nothing. A wave owns
 * one output channel and walks the two fc rows with a lane stride; the hc + 1 dots of that channel
 * share the one pass over `nh` and `ne`.
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_common.h"
#include "r3d_launch.h"

namespace {

constexpr int kMtpThreads = 256;
constexpr int kMtpWave = 32;
constexpr int kMtpWaves = kMtpThreads / kMtpWave;
constexpr int kMtpChannels = kMtpWaves;   /* one wave an output channel */
constexpr int kMtpHcMax = 4;              /* the op's constraint, and the row's */

__device__ __forceinline__ float mtp_fc(const uint8_t* __restrict__ row, int n, long j) {
    return e4m3_to_f32(row[j]) * ((const float*)(row + n))[j >> 7];
}

__global__ __launch_bounds__(kMtpThreads) void mtp_enter_kernel(
    const uint16_t* __restrict__ h, const uint16_t* __restrict__ e,
    const float* __restrict__ wh, const float* __restrict__ we,
    const uint8_t* __restrict__ fch, const uint8_t* __restrict__ fce, long fstride,
    uint16_t* __restrict__ x, long rows, long n, int hc, int flat, float eps, float wadd,
    long hrow, long erow, long xrow) {
    extern __shared__ uint16_t sM[];   /* [hc * n] nh, then [n] ne */
    __shared__ float s_part[kMtpWaves][kMtpHcMax];
    __shared__ float s_pe[kMtpWaves];
    __shared__ float s_scale[kMtpHcMax];
    __shared__ float s_rse;

    const long r = blockIdx.y;
    if (r >= rows) return;
    const long HN = (long)hc * n;
    const int tid = (int)threadIdx.x;
    const int wave = tid / kMtpWave;
    const int lane = tid % kMtpWave;

    const uint16_t* hp = h + r * hrow;
    const uint16_t* ep = e + r * erow;
    uint16_t* xp = x + r * xrow;
    uint16_t* s_nh = sM;
    uint16_t* s_ne = sM + HN;

    /* ---- the sums of squares, one partial a sub-stream plus the embedding's. */
    float part[kMtpHcMax];
#pragma unroll
    for (int c = 0; c < kMtpHcMax; ++c) part[c] = 0.0f;
    float pe = 0.0f;
    for (long i = tid; i < n; i += kMtpThreads) {
        const float ev = bf16_to_f32(ep[i]);
        pe += ev * ev;
        for (int c = 0; c < hc; ++c) {
            const float v = bf16_to_f32(hp[(long)c * n + i]);
            part[c] += v * v;
        }
    }
    for (int c = 0; c < hc; ++c) part[c] = wave_sum(part[c]);
    pe = wave_sum(pe);
    if (lane == 0) {
        for (int c = 0; c < hc; ++c) s_part[wave][c] = part[c];
        s_pe[wave] = pe;
    }
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int c = 0; c < hc; ++c) {
            float v = 0.0f;
            for (int w = 0; w < kMtpWaves; ++w) v += s_part[w][c];
            s_scale[c] = v;
            tot += v;
        }
        if (flat) {
            const float sc = 1.0f / sqrtf(tot / (float)HN + eps);
            for (int c = 0; c < hc; ++c) s_scale[c] = sc;
        } else {
            for (int c = 0; c < hc; ++c)
                s_scale[c] = 1.0f / sqrtf(s_scale[c] / (float)n + eps);
        }
        float ve = 0.0f;
        for (int w = 0; w < kMtpWaves; ++w) ve += s_pe[w];
        s_rse = 1.0f / sqrtf(ve / (float)n + eps);
    }
    __syncthreads();

    /* ---- the normalised operands, narrowed once and kept for every channel of the block. */
    for (long i = tid; i < n; i += kMtpThreads)
        s_ne[i] = f32_to_bf16(bf16_to_f32(ep[i]) * s_rse * (we[i] + wadd));
    for (int c = 0; c < hc; ++c) {
        const float sc = s_scale[c];
        const long base = (long)c * n;
        for (long i = tid; i < n; i += kMtpThreads)
            s_nh[base + i] = f32_to_bf16(bf16_to_f32(hp[base + i]) * sc * (wh[base + i] + wadd));
    }
    __syncthreads();

    /* ---- one output channel a wave: the two weight rows read once, hc + 1 dots. */
    const long i = (long)blockIdx.x * kMtpChannels + wave;
    if (i >= n) return;
    const uint8_t* rh = fch + i * fstride;
    const uint8_t* re = fce + i * fstride;
    float acc[kMtpHcMax];
#pragma unroll
    for (int c = 0; c < kMtpHcMax; ++c) acc[c] = 0.0f;
    float ae = 0.0f;
    for (long j = lane; j < n; j += kMtpWave) {
        const float wv = mtp_fc(rh, (int)n, j);
        for (int c = 0; c < hc; ++c) acc[c] += bf16_to_f32(s_nh[(long)c * n + j]) * wv;
        ae += bf16_to_f32(s_ne[j]) * mtp_fc(re, (int)n, j);
    }
    for (int c = 0; c < hc; ++c) acc[c] = wave_sum(acc[c]);
    ae = wave_sum(ae);
    if (lane == 0) {
        for (int c = 0; c < hc; ++c)
            xp[(long)c * n + i] = f32_to_bf16(acc[c] + ae);
    }
}

/* ---- the stored form -----------------------------------------------------------------------
 *
 * libr4d's joined E4M3 row (r4d_hc_fp8.h at K = n, G = 128): n codes, n / 128 f32 scales, zero
 * padding to 16 bytes. The tag is libr4d's, so an RDNA4-laid container is read in place.
 */
constexpr long long mtp_row_bytes(long long n) {
    return (n + 4 * (n / 128) + 15) / 16 * 16;
}

struct MtpGeom {
    long long n;
};

bool mtp_group_ok(long long n) { return n > 0 && (n % 128) == 0; }

bool mtp_selects(const RadEncoding* e, const int* sel, int n) {
    const auto role = [&](int i) {
        return (e && sel && sel[i] >= 0 && sel[i] < e->n_planes) ? e->plane[sel[i]].role : "";
    };
    return n == 2 && !std::strncmp(role(0), "codes", RAD_ENC_STR) &&
           !std::strncmp(role(1), "scale", RAD_ENC_STR);
}

bool mtp_dims(const RadTensor* t, long long rows, long long cols) {
    return t && t->rank == 2 && t->shape[0] == rows && t->shape[1] == cols;
}

bool mtp_geom(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
              const RadTensor* pl, int n, MtpGeom* g) {
    if (operand != 4 && operand != 5) return false;
    if (!enc || n != 2 || !rad_enc_is(enc, "affine")) return false;
    if (enc->n_planes != 2 || enc->plane[0].dtype != RAD_F8E4M3 || enc->plane[0].block[0] != 1 ||
        enc->plane[0].block[1] != 1)
        return false;
    const int s = rad_enc_find(enc, "scale");
    if (s < 0 || enc->plane[s].dtype != RAD_F32 || enc->plane[s].kind != RAD_PLANE_TILED)
        return false;
    const long long n_ = rad_param_geti(p, n_p, "n", 0);
    if (!mtp_group_ok(n_)) return false;
    if (enc->plane[s].block[0] != 1 || enc->plane[s].block[1] != 128) return false;
    if (!mtp_selects(enc, sel, n)) return false;
    if (!mtp_dims(&pl[0], n_, n_) || !mtp_dims(&pl[1], n_, n_ / 128)) return false;
    g->n = n_;
    return true;
}

}  // namespace

extern "C" int r3d_rad_layout_mtp_e4m3(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, RadLayout* out) {
    MtpGeom g;
    if (!out) return RAD_E_INVAL;
    if (operand != 4 && operand != 5) return RAD_E_UNSUPPORTED;
    if (!mtp_geom(p, n_p, operand, enc, sel, pl, n, &g)) return RAD_E_DTYPE;
    const long long stride = mtp_row_bytes(g.n);
    std::memset(out, 0, sizeof(*out));
    out->tag = "r4d.mtp.e4m3.g128";
    out->dtype = RAD_F8E4M3;
    out->rank = 2;
    out->shape[0] = g.n;
    out->shape[1] = stride;
    out->align = 16;
    out->bytes = g.n * stride;
    return RAD_OK;
}

extern "C" int r3d_rad_relayout_mtp_e4m3(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* pl, int n, void* dst,
                                         int64_t dst_bytes) {
    MtpGeom g;
    if (operand != 4 && operand != 5) return RAD_E_UNSUPPORTED;
    if (!mtp_geom(p, n_p, operand, enc, sel, pl, n, &g)) return RAD_E_DTYPE;
    const long long stride = mtp_row_bytes(g.n);
    const long long ng = g.n / 128;
    if (dst_bytes != g.n * stride || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* codes = (const uint8_t*)pl[0].data;
    const float* sc = (const float*)pl[1].data;
    uint8_t* d = (uint8_t*)dst;
    const long long used = g.n + 4 * ng;
    for (long long r = 0; r < g.n; ++r) {
        uint8_t* row = d + r * stride;
        std::memcpy(row, codes + r * g.n, (size_t)g.n);
        std::memcpy(row + g.n, sc + r * ng, (size_t)(4 * ng));
        if (stride > used) std::memset(row + used, 0, (size_t)(stride - used));
    }
    return RAD_OK;
}

extern "C" int r3d_rad_unrelayout_mtp_e4m3(const RadParam* p, int n_p, int operand,
                                           const RadEncoding* enc, const int* sel,
                                           const void* src, int64_t src_bytes,
                                           const RadTensor* pl, int n) {
    MtpGeom g;
    if (operand != 4 && operand != 5) return RAD_E_UNSUPPORTED;
    if (!mtp_geom(p, n_p, operand, enc, sel, pl, n, &g)) return RAD_E_DTYPE;
    const long long stride = mtp_row_bytes(g.n);
    const long long ng = g.n / 128;
    if (src_bytes != g.n * stride || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* codes = (uint8_t*)pl[0].data;
    float* sc = (float*)pl[1].data;
    for (long long r = 0; r < g.n; ++r) {
        std::memcpy(codes + r * g.n, s + r * stride, (size_t)g.n);
        std::memcpy(sc + r * ng, s + r * stride + g.n, (size_t)(4 * ng));
    }
    return RAD_OK;
}

extern "C" int r3d_rad_mtp_enter(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 7) return R3D_NO(RAD_E_INVAL);
    const RadTensor* h = r3d_opt(a, 0), *e = r3d_opt(a, 1), *wh = r3d_opt(a, 2),
                    *we = r3d_opt(a, 3), *fch = r3d_opt(a, 4), *fce = r3d_opt(a, 5),
                    *x = r3d_opt(a, 6);
    if (!r3d_p(h) || !r3d_p(e) || !r3d_p(wh) || !r3d_p(we) || !r3d_p(fch) || !r3d_p(fce) ||
        !r3d_wp(x))
        return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(h) || !r3d_is_bf16(e) || !r3d_is_bf16(x)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_f32(wh) || !r3d_is_f32(we)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_is_e4m3(fch) || !r3d_is_e4m3(fce)) return R3D_NO(RAD_E_DTYPE);
    const char* fc = r3d_gets(a, "fc");
    if (fc && std::strcmp(fc, "e4m3") != 0) return R3D_NO(RAD_E_INVAL);

    const long long n = r3d_geti_or(a, "n", 0);
    const long long hc = r3d_geti_or(a, "hc", 0);
    const long long ngroup = r3d_geti_or(a, "ngroup", hc);
    if (n <= 0 || hc < 2 || hc > kMtpHcMax) return R3D_NO(RAD_E_SHAPE);
    if (ngroup != 1 && ngroup != hc) return R3D_NO(RAD_E_SHAPE);
    if (!mtp_group_ok(n)) return R3D_NO(RAD_E_SHAPE);
    const long long HN = hc * n;
    if (HN > 5734 * kMtpHcMax) return R3D_NO(RAD_E_UNSUPPORTED);

    long long rows = 0, Mx = 0, Me = 0;
    const long long hrow = r3d_rows_pitch(h, HN, &rows);
    const long long erow = r3d_rows_pitch(e, n, &Me);
    const long long xrow = r3d_rows_pitch(x, HN, &Mx);
    if (hrow < 0 || erow < 0 || xrow < 0 || rows <= 0) return R3D_NO(RAD_E_STRIDE);
    if (Me < rows || Mx < rows) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(h) || !r3d_contig(e) || !r3d_contig(x)) return R3D_NO(RAD_E_STRIDE);
    if (r3d_numel(wh) < HN || r3d_numel(we) < n) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(wh) || !r3d_contig(we)) return R3D_NO(RAD_E_STRIDE);
    /* The declared fc extent is the logical [n, n]; the joined row's pitch follows the layout
     * rule, as in hc_read's E4M3 mix. */
    if (r3d_dim(fch, 0) != n || r3d_dim(fch, 1) != n || r3d_dim(fce, 0) != n ||
        r3d_dim(fce, 1) != n)
        return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(fch) || !r3d_contig(fce)) return R3D_NO(RAD_E_STRIDE);

    const float eps = (float)r3d_getf(a, "eps", 1e-6);
    const float wadd = (float)r3d_getf(a, "wadd", 0.0);
    const long fstride = (long)mtp_row_bytes(n);
    const size_t lds = (size_t)(HN + n) * sizeof(uint16_t);
    R3D_KLAUNCH_S(mtp_enter_kernel,
                  dim3((unsigned)((n + kMtpChannels - 1) / kMtpChannels), (unsigned)rows),
                  dim3(kMtpThreads), lds, (hipStream_t)s, (const uint16_t*)r3d_p(h),
                  (const uint16_t*)r3d_p(e), (const float*)r3d_p(wh), (const float*)r3d_p(we),
                  (const uint8_t*)r3d_p(fch), (const uint8_t*)r3d_p(fce), fstride,
                  (uint16_t*)r3d_wp(x), rows, n, (int)hc, ngroup == 1 ? 1 : 0, eps, wadd, hrow,
                  erow, xrow);
    return hipGetLastError() == hipSuccess ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}