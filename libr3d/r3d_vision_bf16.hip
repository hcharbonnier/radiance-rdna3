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
 *     W12 gave the kernel the walk the cost at M = 16384 asks for: a workgroup carries a block of
 *     query rows (and the whole of a small GQA group), one thread a row, and sweeps the segment's
 *     keys in LDS tiles of K and V widened to f32 once a tile element. The arithmetic order is
 *     untouched -- see "attn_dense, hd 72" below for what that buys and what it refuses to change.
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
/*
 * One thread owns one (query row, query head): 72 f32 of q live in registers beside 72 f32 of
 * accumulator. A workgroup owns one cu segment, one KV head (or the whole of a small GQA group of
 * query heads), and the segment's next kThr / heads-per-block query rows; its waves sweep the
 * segment's keys in tiles of kStage keys.
 *
 * ============================== the two walks ==============================
 *
 * The reference is two-pass: walk one takes the maximum of `dot * scale` over the segment's keys;
 * walk two recomputes the score, exponentiates against that maximum, accumulates `sum += w` and
 * `acc[d] += w * v[d]` ascending in key, then narrows `acc[d] * (1 / sum)` once. Every f32 sum
 * keeps the reference's bracketing -- dot over d ascending, one multiply by `scale` after it, both
 * sums ascending in key -- and `expf` is the file header's one divergence.
 *
 * Keeping the scores between the walks would need 4 bytes a (row, key): 64 KiB a row, 8 MiB for a
 * 16384-key image at a block's 128 rows, against 64 KiB of LDS a workgroup, so walk two recomputes
 * them with the same operands and the same order and the two walks produce the same bits. Online
 * (flash) softmax would instead rescale the running accumulator and the denominator by
 * `exp(m_old - m_new)`, bracketing every sum differently; r3d_attn_paged_h128.hip states why that
 * arm is not taken, and this kernel makes the same choice.
 *
 * ============================== the stage ==============================
 *
 * A workgroup stages the tile's K rows -- and, in walk two, its V rows -- into LDS as f32: kStage
 * rows of 72 f32, 4.5 KiB a plane. Every lane of a wave is reading the same key of the same KV
 * head, so a read is a broadcast of one address rather than 32 scattered rows and a tile costs a
 * wave one broadcast a dword, not a read a lane. That is why the codes are widened once a tile
 * element instead of once a (row, element): 128 rows share the widening, and the 16-byte global
 * read that feeds it moves eight bf16 codes at once. The stage holds f32 rather than bf16 codes
 * because a code read costs a shift and a mask a value on every use -- two ALU ops a value, where
 * an f32 dword serves two values with none -- and the same argument rejects f16 pairs and dot2
 * here: gfx11 has no bf16 dot2 (r3d_common.h), and narrowing to f16 would put the reference's
 * exact widening behind a lossy convert, which is not this row's contract.
 *
 * A tile costs two barriers: one after the workgroup-wide load, one before the next tile overwrites
 * it. Walk one stages K only; walk two stages K and V. When the shim sees every operand 16-byte
 * aligned and every pitch a multiple of eight, a tile row is kChunk 16-byte loads; otherwise the
 * same rows are loaded a halfword a thread, so an unaligned base is served rather than refused
 * (`align16` is the shim's promise, and the branch is workgroup-uniform).
 *
 * ============================== the grid ==============================
 *
 * grid = (ceil(total_q / rows-per-block), q_heads / heads-per-block, n_seq): one block a (segment,
 * KV head or group, query-row tile). A tile that starts past its segment's last row returns at
 * once, so a ragged batch pays one empty block a tile bound instead of a cu_seqlens scan a thread.
 * A block carries `hpb = min(gqa, kMaxHb)` of the group's query heads when the group fits in a
 * block, so with gqa > 1 each staged tile is reused in LDS across the whole group; a wider group
 * falls back to one head a block (its heads still meet in L2) rather than refusing the geometry.
 */

constexpr int kStage = 16;                  /* keys one staged tile carries */
constexpr int kQuad  = kHd / 4;             /* 16-byte reads a staged row: 18 */
constexpr int kChunk = kHd / 8;             /* 16-byte chunks a key row: 9 */
constexpr int kMaxHb = 8;                   /* query heads a block keeps of one KV group */

/* bf16 codes in a dword, widened in place: the low half is the even element, the high the odd. */
__device__ __forceinline__ void widen8(const uint32_t* d, float* f) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        f[2 * j]     = __uint_as_float(d[j] << 16);
        f[2 * j + 1] = __uint_as_float(d[j] & 0xffff0000u);
    }
}

/* One plane's tile: `nu` key rows from `row0` (already at the KV head's column), widened to f32 in
 * LDS. ALIGN16 is the shim's promise that every row starts on a 16-byte boundary. */
template <bool ALIGN16>
__device__ __forceinline__ void stage_plane(float* dst, const uint16_t* row0, long ld, int nu,
                                            int tid) {
    if constexpr (ALIGN16) {
        const int nchunk = nu * kChunk;
        for (int e = tid; e < nchunk; e += kThr) {
            const int u = e / kChunk, c = e - u * kChunk;
            const r3d_u32x4 raw = *(const __attribute__((address_space(1)))
                                    r3d_u32x4*)(const void*)(row0 + (long)u * ld + c * 8);
            const uint32_t d[4] = { raw.x, raw.y, raw.z, raw.w };
            float f[8];
            widen8(d, f);
            *(r3d_u32x4*)(void*)(dst + u * kHd + c * 8) =
                r3d_u32x4{ __float_as_uint(f[0]), __float_as_uint(f[1]),
                           __float_as_uint(f[2]), __float_as_uint(f[3]) };
            *(r3d_u32x4*)(void*)(dst + u * kHd + c * 8 + 4) =
                r3d_u32x4{ __float_as_uint(f[4]), __float_as_uint(f[5]),
                           __float_as_uint(f[6]), __float_as_uint(f[7]) };
        }
    } else {
        const int n = nu * kHd;
        for (int e = tid; e < n; e += kThr) {
            const int u = e / kHd, d = e - u * kHd;
            dst[u * kHd + d] = bf16_to_f32((uint16_t)ldg_u16(row0 + (long)u * ld + d));
        }
    }
}

/* The score's dot over d ascending, f32, no FMA: the reference's expression over a staged row. The
 * row is read a 16-byte dword at a time, which is one broadcast for the whole wave. */
__device__ __forceinline__ float dot_row(const float* qv, const float* krow) {
    float dot = 0.0f;
#pragma unroll
    for (int c = 0; c < kQuad; ++c) {
        const r3d_u32x4 raw = *(const r3d_u32x4*)(const void*)(krow + 4 * c);
        dot += qv[4 * c] * __uint_as_float(raw.x);
        dot += qv[4 * c + 1] * __uint_as_float(raw.y);
        dot += qv[4 * c + 2] * __uint_as_float(raw.z);
        dot += qv[4 * c + 3] * __uint_as_float(raw.w);
    }
    return dot;
}

__global__ __launch_bounds__(kThr) R3D_WAVES(4) void attn_dense_h72_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
    const uint16_t* __restrict__ v, uint16_t* __restrict__ o,
    const int32_t* __restrict__ cu, int total_q, int gqa, int hpb,
    long q_ld, long k_ld, long v_ld, long o_ld, float scale, int align16) {
    __shared__ __attribute__((aligned(16))) float sK[kStage * kHd];
    __shared__ __attribute__((aligned(16))) float sV[kStage * kHd];

    const int tid = (int)threadIdx.x;
    const int s = (int)blockIdx.z;
    const int r = tid / hpb, hl = tid - r * hpb;     /* row in the tile, head in the group */
    const int h = (int)blockIdx.y * hpb + hl;        /* the query head */
    const int kh = h / gqa;                          /* its KV head */
    const int rows = kThr / hpb;                     /* query rows a block carries */

    /* `cu` is the description's promise; a last boundary past the query plane is clamped here so a
     * long descriptor cannot make a key read leave K or V (`seq_of`'s rows are the reference's).
     * The row index is 64-bit because the tile bound is measured in rows and a plane of near-int
     * extent must not overflow it; `row` itself is a plane row and fits an int. */
    const int b0 = cu[s];
    int b1 = cu[s + 1];
    if (b1 > total_q) b1 = total_q;
    const long long row0 = (long long)b0 + (long long)(int)blockIdx.x * rows;
    if (b1 <= b0 || row0 >= b1) return;
    const long long i = row0 + r;
    /* A row the block does not carry -- the last tile's tail, or the padding `gqa` leaves when it
     * does not divide kThr -- loads a row that exists so the walk stays branch-free, and stores
     * nothing. */
    const bool live = r < rows && i < b1;
    const int row = (int)(live ? i : (long long)(b1 - 1));

    const uint16_t* qrow = q + (size_t)row * q_ld + (size_t)h * kHd;
    float qv[kHd];
    if (align16) {
#pragma unroll
        for (int c = 0; c < kChunk; ++c) {
            const r3d_u32x4 raw = *(const __attribute__((address_space(1)))
                                    r3d_u32x4*)(const void*)(qrow + c * 8);
            const uint32_t d[4] = { raw.x, raw.y, raw.z, raw.w };
            widen8(d, qv + c * 8);
        }
    } else {
#pragma unroll
        for (int d = 0; d < kHd; ++d) qv[d] = bf16_to_f32((uint16_t)ldg_u16(qrow + d));
    }

    const uint16_t* kbase = k + (size_t)kh * kHd;
    const uint16_t* vbase = v + (size_t)kh * kHd;

    /* Walk one: the maximum of the scaled scores, ascending in key. */
    float mx = -INFINITY;
    for (int j0 = b0; j0 < b1; j0 += kStage) {
        const int nu = b1 - j0 < kStage ? b1 - j0 : kStage;
        const uint16_t* src = kbase + (size_t)j0 * k_ld;
        if (align16) stage_plane<true>(sK, src, k_ld, nu, tid);
        else stage_plane<false>(sK, src, k_ld, nu, tid);
        group_barrier();
        /* A thread whose row the block does not carry still stages the tile and still meets every
         * barrier, but nothing it computes is stored: skip its arithmetic, so a short segment's idle
         * waves do not spend the block's issue slots on it. A wave is all-live or all-idle unless it
         * straddles the tile's end, so the test is uniform in the common case. */
        if (live)
            for (int u = 0; u < nu; ++u) {
                const float sc = dot_row(qv, sK + u * kHd) * scale;
                if (sc > mx) mx = sc;
            }
        group_barrier();
    }

    /* Walk two: the same scores, exponentiated against the maximum, accumulated ascending. */
    float acc[kHd];
#pragma unroll
    for (int d = 0; d < kHd; ++d) acc[d] = 0.0f;
    float sum = 0.0f;
    for (int j0 = b0; j0 < b1; j0 += kStage) {
        const int nu = b1 - j0 < kStage ? b1 - j0 : kStage;
        const uint16_t* ksrc = kbase + (size_t)j0 * k_ld;
        const uint16_t* vsrc = vbase + (size_t)j0 * v_ld;
        if (align16) {
            stage_plane<true>(sK, ksrc, k_ld, nu, tid);
            stage_plane<true>(sV, vsrc, v_ld, nu, tid);
        } else {
            stage_plane<false>(sK, ksrc, k_ld, nu, tid);
            stage_plane<false>(sV, vsrc, v_ld, nu, tid);
        }
        group_barrier();
        if (live)
            for (int u = 0; u < nu; ++u) {
                const float w = expf(dot_row(qv, sK + u * kHd) * scale - mx);
                sum += w;
                const float* vrow = sV + u * kHd;
#pragma unroll
                for (int c = 0; c < kQuad; ++c) {
                    const r3d_u32x4 raw = *(const r3d_u32x4*)(const void*)(vrow + 4 * c);
                    acc[4 * c] += w * __uint_as_float(raw.x);
                    acc[4 * c + 1] += w * __uint_as_float(raw.y);
                    acc[4 * c + 2] += w * __uint_as_float(raw.z);
                    acc[4 * c + 3] += w * __uint_as_float(raw.w);
                }
            }
        group_barrier();
    }

    if (!live) return;
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    uint16_t* orow = o + (size_t)row * o_ld + (size_t)h * kHd;
    if (align16) {
#pragma unroll
        for (int c = 0; c < kChunk; ++c) {
            uint32_t d[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int t = c * 8 + 2 * j;
                d[j] = (uint32_t)f32_to_bf16(acc[t] * inv) |
                       ((uint32_t)f32_to_bf16(acc[t + 1] * inv) << 16);
            }
            *(r3d_u32x4*)(void*)(orow + c * 8) = r3d_u32x4{ d[0], d[1], d[2], d[3] };
        }
    } else {
#pragma unroll
        for (int d = 0; d < kHd; ++d) orow[d] = f32_to_bf16(acc[d] * inv);
    }
}

}  // namespace

extern "C" int r3d_attn_dense_h72(long q, long k, long v, long o, long cu, int total_q,
                                  int n_seq, int max_seqlen, int q_heads, int gqa, long q_ld,
                                  long k_ld, long v_ld, long o_ld, float scale, long stream) {
    if (!q || !k || !v || !o || !cu) return -1;
    if (total_q <= 0 || n_seq <= 0 || q_heads <= 0 || gqa <= 0 || q_heads % gqa) return -2;
    if (max_seqlen < 0) return -2;
    if (q_ld < (long)q_heads * kHd || k_ld < (long)(q_heads / gqa) * kHd ||
        v_ld < (long)(q_heads / gqa) * kHd || o_ld < (long)q_heads * kHd)
        return -3;
    /* A GQA group up to kMaxHb heads is carried by one block, so its staged K and V are shared in
     * LDS; a wider group is served one head a block, which is a different grid, not a different
     * kernel. `rows` floors, so a gqa that does not divide kThr leaves its last threads idle. */
    const int hpb = gqa <= kMaxHb ? gqa : 1;
    const long rows = kThr / hpb;
    /* A segment is at most `max_seqlen` rows, so that bounds the x axis; without it the whole
     * batch's rows do, and the blocks past each segment's end return immediately. */
    const long long span = (max_seqlen > 0 && max_seqlen < total_q) ? max_seqlen : total_q;
    const long long tiles = (span + rows - 1) / rows;
    const long long heads = (long long)q_heads / hpb;
    if (tiles > 0x7fffffffLL) return -2;
    /* The launch's y and z axes carry the heads and the segments, and gfx11 caps both at 65535
     * (maxGridSize[1], [2]); a geometry past that is refused here rather than handed to the runtime
     * as a failed configuration. */
    if (heads > 65535 || n_seq > 65535) return -2;
    /* A row is kHd bf16 wide, so a pitch that is a multiple of eight keeps it 16-byte aligned
     * whenever the operand's base is; the kernel reads 16 bytes at a time when both hold. */
    const int align16 = r3d_aligned((const void*)(uintptr_t)q, 16) &&
                        r3d_aligned((const void*)(uintptr_t)k, 16) &&
                        r3d_aligned((const void*)(uintptr_t)v, 16) &&
                        r3d_aligned((const void*)(uintptr_t)o, 16) &&
                        (q_ld % 8 == 0) && (k_ld % 8 == 0) && (v_ld % 8 == 0) && (o_ld % 8 == 0);
    R3D_KLAUNCH(attn_dense_h72_kernel, dim3((uint32_t)tiles, (uint32_t)heads, (uint32_t)n_seq),
                dim3(kThr), (hipStream_t)(uintptr_t)stream, (const uint16_t*)(uintptr_t)q,
                (const uint16_t*)(uintptr_t)k, (const uint16_t*)(uintptr_t)v,
                (uint16_t*)(uintptr_t)o, (const int32_t*)(uintptr_t)cu, total_q, gqa, hpb,
                q_ld, k_ld, v_ld, o_ld, scale, align16);
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
    /* The optional band parameter bounds a segment's rows, so the launch does not tile the whole
     * batch for every segment; a caller that omits it keeps the total-rows bound. */
    const long long max_seqlen = r3d_geti_or(a, "max_seqlen", 0);
    if (max_seqlen < 0) return R3D_NO(RAD_E_SHAPE);
    const int msl = (max_seqlen > 0 && max_seqlen < q_tok) ? (int)max_seqlen : 0;
    return launch_ok(r3d_attn_dense_h72(r3d_addr(q), r3d_addr(k), r3d_addr(v), r3d_addr(o),
                                        r3d_addr(cu), (int)q_tok, (int)n_seq, msl, (int)q_nh,
                                        (int)gqa, (long)q_ld, (long)k_ld, (long)v_ld, (long)o_ld,
                                        scale, (long)(uintptr_t)s));
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
