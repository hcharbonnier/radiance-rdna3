/* r3d_attn_paged_h256.hip -- paged attention at head_dim 256 with a GQA 12 or 6 geometry and an
 * int8 KV cache, for RDNA3. Two entries: decode (q_len <= 16, the speculative-verify band) and
 * prefill (q_len >= 17, a chunk).
 *
 * ============================== the geometry ==============================
 *
 * The cache is the paged layout ref_attn.cpp documents, with an int8 slot:
 *
 *     kv_cache[blocks, kv_heads, block_size, 2 * 256 + 4]   i8
 *
 * A slot's row is 256 K codes, 256 V codes, then the two f16 scales (K's, then V's). A value is
 * its code times its scale; there are no k_descale / v_descale planes for an int8 cache, and this
 * row refuses them rather than ignore a caller error. `block_size` is 4 for the Flash-Next
 * gqa = 12 geometry (24 query heads / 2 KV heads) and 16 for the 27B gqa = 6 geometry (24 / 4),
 * which is the pair docs/ARCHITECTURES.md names. The QSA indexer is out of scope here: the block
 * table is read as an ordinary page table, so a 4-token table that QSA selected can be served as
 * soon as W7 writes it, but this row never chooses pages.
 *
 * ============================== the lane map ==============================
 *
 * head_dim 256 cannot give one lane a whole output row: 256 f32 accumulators plus the query row
 * do not fit even at one wave a SIMD. Instead four lanes own one output row, 64 dims each, and a
 * 64-lane wave carries sixteen rows (sixteen 4-lane groups). A workgroup is three waves (decode)
 * or six waves (prefill); because a row is (token, query head) and every row of a workgroup is in
 * the same KV head, the whole workgroup stages each cache block once. The tile is a whole number
 * of tokens:
 *
 *     decode   waves 3, rows 48, gqa 12 -> 4 tokens/tile;  gqa 6 -> 8 tokens/tile
 *     prefill  waves 6, rows 96, gqa 12 -> 8 tokens/tile;  gqa 6 -> 16 tokens/tile
 *
 * A wave's sixteen row slots are (token, head) pairs, head-major inside the tile, so the four lanes
 * of a group share the token (and its mask). A row slot past its sequence's length is a clamped
 * duplicate that stores nothing, exactly as the h128 row's padding slots.
 *
 * ============================== the arithmetic ==============================
 *
 * The query, the cache codes and the f16 scales are read in the reference's order:
 *
 *     dot += q[d] * (code[d] * k_scale)          d ascending, f32
 *     score = dot * attn_scale
 *     w     = exp(score - max)
 *     acc[d] += w * (v_code[d] * v_scale)        keys ascending, f32
 *     out[d] = bf16(acc[d] * (1 / sum))
 *
 * The dot is split over the four lanes: each lane sums its 64 dims in ascending order, then the
 * four partials are added in d-block order. That is not libref's single 256-add chain, so a
 * generic cache can land a few bf16 store ulps from the reference; the test below pins the parts
 * that must be bit-exact (the mask, the page walk, the per-slot scales, the value chain and its
 * key order) with a cache whose K row is the same for every key, and checks a random cache at one
 * bf16 code. `exp` is the h128 row's polynomial, within one f32 ulp of (float)std::exp(double).
 *
 * libref keeps each visible key's score from the max pass and reuses it in the value pass; this
 * row does too whenever the workgroup's key range fits the score cache in LDS (`kScores` floats,
 * one per (row slot, key)). The pass two stage then skips the K stores and the pass two walk reads
 * the score instead of recomputing the dot, which removes half the QK arithmetic. The scores are
 * bit-for-bit the pass one values, so nothing about the reference's order changes. When the range
 * does not fit (a long prefill with no split, or a decode slice past ~106 keys) the row walks the
 * reference's two-pass form: the dot is recomputed with the same operands in the same order.
 * `nkeys * kRows <= kScores` is a workgroup-uniform decision, so both paths compute the same bytes.
 *
 * Split-KV (decode only) stores four-lane partial rows of 260 floats (256 accumulators, sum,
 * max) and a second kernel rescales them. splits = 1 is the reference's key order; splits > 1
 * changes the softmax's parentheses and is the documented last-ulp departure. The shim lowers
 * splits until the arena the engine offers fits, down to 1, so the band is never refused for lack
 * of scratch.
 *
 * ============================== what the shims refuse ==============================
 *
 * q and the output are bf16; indices are i32. The cache must be the dense int8 slot above
 * (rank 4, last dim 2 * head_dim + 4, stride 1, contiguous, 4-aligned; 16-aligned q/out), the
 * block table's reach must be inside the plane, seqused must be contiguous, and a ragged
 * cu_seqlens is [n_seq + 1] i32. head_dim must be 256; (gqa, block_size) must be (12, 4) or
 * (6, 16). The h128 row stays the 128-dim body for bf16 and fp8 caches.
 */
#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_launch.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

enum : int {
    kHeadDim    = 256,
    kLanesRow   = 4,           /* lanes an output row is split over */
    kDimsLane   = kHeadDim / kLanesRow,
    kLanesWave  = 64,          /* an RDNA3 wavefront */
    kRowsWave   = kLanesWave / kLanesRow,
    kPartStride = kHeadDim + 4,
    kSplitMax   = 64,
    kScores     = 5120,        /* f32 score slots: one per (row, key) when the walk fits them */
    /* A staged slot's row pitch in LDS. 272 bytes (68 dwords, still 16-byte aligned for the b128
     * reads) puts successive key slots four banks apart: a 64-byte row pitch would put all sixteen
     * row slots of a wave on the same eight banks. It changes no value, only the address. */
    kPitch      = kHeadDim + 16,
};

__device__ __forceinline__ long long ld_ix(const int* p, size_t i) { return p[i]; }

/* The h128 row's exp: rn(x/ln2) with a two-term ln2 reduction, a degree-8 Estrin polynomial on
 * |r| <= 0.35, then 2^k by exponent assembly. Within one f32 ulp of (float)std::exp((double)x). */
__device__ __forceinline__ float exp_ref(float x)
{
    if (!(x < 88.7228390525719f))
        return x == x ? __int_as_float(0x7f800000) : __int_as_float(0x7fc00000);
    if (x < -103.9720770839916f) return 0.0f;
    const int k = (int)__float2int_rn(x * 1.4426950408889634f);
    const float LN_HI = 0.693145751953125f;
    const float LN_LO = 1.42860681039557e-06f;
    float r = x - (float)k * LN_HI;
    r = r - (float)k * LN_LO;
    const float c8 = 1.98412698412698e-04f, c7 = 1.38888888888889e-03f;
    const float c6 = 8.33333333333333e-03f, c5 = 4.16666666666667e-02f;
    const float c3 = 1.66666666666667e-01f;
    const float r2 = r * r;
    const float p4 = __fmaf_rn(__fmaf_rn(__fmaf_rn(c8, r, c7), r, c6), r, c5);
    const float p2 = __fmaf_rn(p4, r2, __fmaf_rn(c3, r, 5.0e-01f));
    const float e = __fmaf_rn(p2, r2, __fmaf_rn(r, 1.0f, 1.0f));
    const int be = __float_as_int(e);
    const int field = ((be >> 23) & 0xff) + k;
    if (field >= 1) return __int_as_float((be & 0x007fffff) | (field << 23));
    if (field <= -23) return 0.0f;
    const float t = __int_as_float((be & 0x007fffff) | ((field + 126) << 23));
    return t * __int_as_float((127 - 126) << 23);
}

/* Two bf16 in a dword, widened exactly (a shift, not a codec). */
__device__ __forceinline__ float bf16_lo(uint32_t w) { return bf16_to_f32((uint16_t)(w & 0xffffu)); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return bf16_to_f32((uint16_t)(w >> 16)); }

/* An f32 staging vector; arrays of HIP's float4 class spill, this ext_vector_type does not. */
typedef float r3d_f32x4 __attribute__((ext_vector_type(4)));

struct Args {
    const uint16_t* q;
    const uint8_t*  kv;        /* int8 cache: [blocks, kv_heads, block_size, 2*hd+4] bytes */
    const int*      bt;
    const int*      su;
    const int*      cu;
    uint16_t*       out;
    int             q_rs, o_rs;          /* row pitch in elements */
    int             bt_rs, bt_cs;
    int             sb, sh, so;          /* cache strides in bytes: block, kv head, slot */
    int             kv16;                /* the cache and both page strides are 16-aligned */
    int             n_seq, q_len, rows;
    int             n_head, kv_heads;
    int             block;               /* cache slots a page: 4 or 16 */
    int             jcap;                /* min(ctx, block table's reach) */
    int             causal, window;
    float           scale;
    int             splits;
    int             band;                /* 0 decode, 1 prefill: the entry's wave count */
    float*          part;
};

struct RowMap { int s, tile, start, i0, len, span; };

__device__ __forceinline__ RowMap map_seq(const Args& A, int tpb, int g)
{
    const int n = A.n_seq;
    for (int s = 0; s < n; ++s) {
        const int c0 = A.cu ? (int)ld_ix(A.cu, (size_t)s) : s * A.q_len;
        const int c1 = A.cu ? (int)ld_ix(A.cu, (size_t)s + 1) : c0 + A.q_len;
        /* With cu[0] > 0 sequence 0's tiles start at row 0, not cu[0]; the reference's seq_of
         * gives rows below cu[0] to sequence 0 too. `len` is the reference's qlen (positions count
         * from it), `span` the rows the tiles carry. */
        const int start = (s == 0 && c0 > 0) ? 0 : c0;
        const int nr = c1 - start;
        const int nt = nr > 0 ? (nr + tpb - 1) / tpb : 0;
        if (g < nt) return RowMap{ s, g, start, c0, c1 - c0, nr };
        if (s + 1 >= n) return RowMap{ s, -1, start, c0, c1 - c0, nr };
        g -= nt;
    }
    return RowMap{ n - 1, -1, 0, 0, 0, 0 };
}

__device__ __forceinline__ int seq_of_row(const Args& A, int row)
{
    if (!A.cu) return row / A.q_len;
    int lo = 0, hi = A.n_seq;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (ld_ix(A.cu, (size_t)mid + 1) <= row) lo = mid + 1; else hi = mid;
    }
    return lo < A.n_seq ? lo : A.n_seq - 1;
}

/* The staged tile lives in LDS. These address-space-qualified pointers keep every stage write and
 * every dot/value read an `ds_` operation; a plain `int8_t*` parameter would decay to generic and
 * let the compiler treat the store and the read as unrelated memory (the first device run showed
 * exactly that: only the wave that wrote a slot saw its value). */
typedef __attribute__((address_space(3))) int8_t   r3d_lds_i8;
typedef __attribute__((address_space(3))) uint32_t r3d_lds_u32;
typedef __attribute__((address_space(3))) float    r3d_lds_f32;

/* One 4-byte word of a staged page, routed by its flat word index: a slot is 129 words (64 K
 * codes, 64 V codes, then the two f16 scales packed in one dword). `STAGE_K` and `STAGE_V` select
 * the planes a pass consumes: the max pass reads K and K's scale only, and the cached value pass
 * reads V and V's scale only, so neither stores the other plane. */
template <bool STAGE_K, bool STAGE_V>
__device__ __forceinline__ void put_slot_word(int slot, int r, uint32_t v,
                                              r3d_lds_i8* __restrict__ sK,
                                              r3d_lds_i8* __restrict__ sV,
                                              r3d_lds_f32* __restrict__ sks,
                                              r3d_lds_f32* __restrict__ svs)
{
    if (r < kHeadDim / 4) {
        if (STAGE_K) *((r3d_lds_u32*)(sK + slot * kPitch) + r) = v;
    } else if (r < kHeadDim / 2) {
        if (STAGE_V) *((r3d_lds_u32*)(sV + slot * kPitch) + (r - kHeadDim / 4)) = v;
    } else {
        if (STAGE_K) sks[slot] = f16_to_f32((uint16_t)(v & 0xffffu));
        if (STAGE_V) svs[slot] = f16_to_f32((uint16_t)(v >> 16));
    }
}

/* Stage the `nu` visible int8 slots of one cache block (K codes, V codes) and their scales into
 * LDS. Every lane of the workgroup runs this; the caller barriers before and after.
 *
 * A slot row is 516 bytes, so a row is 4-aligned but not 16-aligned. The page as a whole is
 * 16-aligned whenever the cache is (A.sb and A.sh are multiples of 16 for both geometries), so the
 * staged slots are then read as one flat run of 16-byte chunks and each of the four words is
 * routed to its K, V or scale home: every global load is a 16-byte vector load inside the page.
 * The 4-byte loop is the general path, because the shim only promises 4-byte alignment. Both paths
 * move the same bytes to the same LDS words, so no arithmetic changes.
 *
 * A slot's K and V homes are `kPitch` (272) bytes apart while the row itself is 256 bytes: the 16
 * extra bytes skew successive slots across the LDS banks, so a wave's sixteen row slots no longer
 * all read the same eight banks on every dot/value step. Only addresses move; the staged bytes and
 * the arithmetic are unchanged. */
template <int NTH, bool STAGE_K, bool STAGE_V>
__device__ __forceinline__ void stage_i8(const Args& A, int blk, int kh, int nu, int tid,
                                         r3d_lds_i8* __restrict__ sK, r3d_lds_i8* __restrict__ sV,
                                         r3d_lds_f32* __restrict__ sks,
                                         r3d_lds_f32* __restrict__ svs)
{
    const long long base = (long long)blk * A.sb + (long long)kh * A.sh;
    const uint8_t* src = A.kv + base;
    constexpr int kWordsSlot = kHeadDim / 2 + 1;       /* 129 */
    const int nw = nu * kWordsSlot;
    if (A.kv16 && nu > 0) {
        const int nc = nw >> 2;
#pragma unroll 1
        for (int c = tid; c < nc; c += NTH) {
            const r3d_u32x4 v = *(const __attribute__((address_space(1))) r3d_u32x4*)
                                    (const void*)(src + (long long)c * 16);
            int slot = (c << 2) / kWordsSlot;
            int r = (c << 2) - slot * kWordsSlot;
            put_slot_word<STAGE_K, STAGE_V>(slot, r, v[0], sK, sV, sks, svs);
            if (++r == kWordsSlot) { r = 0; ++slot; }
            put_slot_word<STAGE_K, STAGE_V>(slot, r, v[1], sK, sV, sks, svs);
            if (++r == kWordsSlot) { r = 0; ++slot; }
            put_slot_word<STAGE_K, STAGE_V>(slot, r, v[2], sK, sV, sks, svs);
            if (++r == kWordsSlot) { r = 0; ++slot; }
            put_slot_word<STAGE_K, STAGE_V>(slot, r, v[3], sK, sV, sks, svs);
        }
#pragma unroll 1
        for (int idx = (nc << 2) + tid; idx < nw; idx += NTH) {
            const int slot = idx / kWordsSlot;
            const int r = idx - slot * kWordsSlot;
            put_slot_word<STAGE_K, STAGE_V>(slot, r, ldg_u32((const uint32_t*)(src + (long long)idx * 4)),
                                            sK, sV, sks, svs);
        }
    } else {
#pragma unroll 1
        for (int idx = tid; idx < nw; idx += NTH) {
            const int slot = idx / kWordsSlot;
            const int r = idx - slot * kWordsSlot;
            put_slot_word<STAGE_K, STAGE_V>(slot, r, ldg_u32((const uint32_t*)(src + (long long)idx * 4)),
                                            sK, sV, sks, svs);
        }
    }
}

/* One visible key's score. The four lanes of a row group each sum their own 64 dims (ascending)
 * and then the partials are added in d-block order. All lanes call this, including lanes whose
 * row is padding: only the shuffle is unconditional, the caller applies the visibility. */
__device__ __forceinline__ float dot_group(const float* __restrict__ qf,
                                           const r3d_lds_i8* __restrict__ krow,
                                           float ks, int gbase)
{
    float p = 0.0f;
#pragma unroll
    for (int i = 0; i < kDimsLane; ++i)
        p = p + qf[i] * ((float)(int8_t)krow[i] * ks);
    /* Every lane must start from the group's *first* partial, not its own: a lane that starts with
     * its own p would double its slice and drop lane 0's. */
    float dot = __shfl(p, gbase, kLanesWave);
#pragma unroll
    for (int b = 1; b < kLanesRow; ++b)
        dot = dot + __shfl(p, gbase + b, kLanesWave);
    return dot;
}

/* Two keys' scores in one pass. Each key's chain is exactly `dot_group`'s (its 64 dims ascending,
 * then the four d-block partials in order); the two chains are independent, so the compiler can
 * interleave them and hide the 64-add latency. Nothing is reassociated: key 0 and key 1 keep their
 * own partial sums and their own order. */
__device__ __forceinline__ void dot_group2(const float* __restrict__ qf,
                                           const r3d_lds_i8* __restrict__ k0,
                                           const r3d_lds_i8* __restrict__ k1,
                                           float ks0, float ks1, int gbase,
                                           float& o0, float& o1)
{
    float p0 = 0.0f, p1 = 0.0f;
#pragma unroll
    for (int i = 0; i < kDimsLane; ++i) {
        p0 = p0 + qf[i] * ((float)(int8_t)k0[i] * ks0);
        p1 = p1 + qf[i] * ((float)(int8_t)k1[i] * ks1);
    }
    o0 = __shfl(p0, gbase, kLanesWave);
    o1 = __shfl(p1, gbase, kLanesWave);
#pragma unroll
    for (int b = 1; b < kLanesRow; ++b) {
        o0 = o0 + __shfl(p0, gbase + b, kLanesWave);
        o1 = o1 + __shfl(p1, gbase + b, kLanesWave);
    }
}

__device__ __forceinline__ bool visible(int j, int pos, int causal, long long window)
{
    if (causal && j > pos) return false;
    if (window > 0) {
        const long long d = (long long)pos - j;
        if (causal ? (d >= window) : (d >= window || d <= -window)) return false;
    }
    return true;
}

template <bool CAUSAL, bool SPLIT, int GQA, int BLOCK, int WAVES>
__global__ __launch_bounds__(WAVES * kLanesWave) R3D_WAVES(WAVES)
void attn_kernel(const Args A)
{
    static_assert(kRowsWave * WAVES % GQA == 0, "a tile is whole tokens");
    constexpr int kRows = kRowsWave * WAVES;
    constexpr int TPT   = kRows / GQA;       /* tokens a tile carries */
    constexpr int NTH   = WAVES * kLanesWave;

    __shared__ __align__(16) int8_t sK[BLOCK * kPitch];
    __shared__ __align__(16) int8_t sV[BLOCK * kPitch];
    __shared__ float sks[BLOCK], svs[BLOCK];
    /* The max pass's scores, one per (row slot, key), when the workgroup's key range fits. A score
     * is bit-for-bit the value pass two would recompute, so the cached and the recompute paths
     * agree; `nkeys * kRows <= kScores` is a uniform decision. */
    __shared__ float sScore[kScores];

    const int tid  = (int)threadIdx.x;
    const int lane = tid & (kLanesWave - 1);
    const int wave = tid / kLanesWave;
    const int rr   = wave * kRowsWave + (lane >> 2);   /* row slot inside the workgroup */
    const int db   = lane & 3;                         /* which 64-dim slice */
    const int d0   = db * kDimsLane;
    const int gbase = lane & ~3;

    const int kh = (int)blockIdx.y;
    const RowMap M = map_seq(A, TPT, (int)blockIdx.x);
    /* A grid row past the batch's tiles, a sequence with no rows, or one starting past the query
     * plane. */
    if (M.tile < 0 || M.len <= 0 || M.start >= A.rows) return;

    const int tb = M.tile * TPT;                       /* the tile's first token */
    const int tok_in = tb + rr / GQA;                  /* this group's token inside the sequence */
    const bool row_ok = tok_in < M.span && M.start + tok_in < A.rows;
    /* Idle slots load a real row; they never store. */
    int tok = row_ok ? tok_in : (tb < M.span ? tb : M.span - 1);
    if (M.start + tok >= A.rows) tok = A.rows - 1 - M.start;
    if (tok < 0) tok = 0;
    const int h = kh * GQA + (rr % GQA);
    const int row = M.start + tok;
    const int ib = M.start - M.i0;
    const int ctx = (int)ld_ix(A.su, (size_t)M.s);
    const int pos = ctx - M.len + tok + ib;

    /* Every row of the workgroup is in the same KV head, so one staged block serves all of them. */
    float qf[kDimsLane];
    {
        const uint16_t* qp = A.q + (long long)row * A.q_rs + (long long)h * kHeadDim + d0;
#pragma unroll
        for (int u = 0; u < kDimsLane / 8; ++u) {
            const r3d_u32x4 v = *(const r3d_u32x4*)(const void*)(qp + 8 * u);
            qf[8 * u + 0] = bf16_lo(v[0]); qf[8 * u + 1] = bf16_hi(v[0]);
            qf[8 * u + 2] = bf16_lo(v[1]); qf[8 * u + 3] = bf16_hi(v[1]);
            qf[8 * u + 4] = bf16_lo(v[2]); qf[8 * u + 5] = bf16_hi(v[2]);
            qf[8 * u + 6] = bf16_lo(v[3]); qf[8 * u + 7] = bf16_hi(v[3]);
        }
    }

    const int jend_all = ctx < A.jcap ? ctx : A.jcap;

    /* Row slots past the sequence are clamped duplicates; they still run the dot for the shuffle,
     * but their visibility is off and they store nothing. */
    const bool active = row_ok;

    /* The workgroup's key range, so blocks fully outside the mask are never staged. All waves
     * share the staged tile, so the bounds are the workgroup's, not a wave's. `wmin..wmax` are the
     * tile's token positions; causal trims the end, a window trims both ends, and `jhi` is an
     * exclusive key bound. */
    const int gb = M.tile * TPT;
    const int wmin = ctx - M.len + gb + ib;
    const int wmax = wmin + TPT - 1;
    int jlo = 0, jhi = jend_all;
    if (CAUSAL && wmax + 1 < jhi) jhi = wmax + 1;
    if (A.window > 0) {
        const int lo = wmin - A.window + 1;
        if (lo > jlo) jlo = lo > 0 ? lo : 0;
        if (!CAUSAL) {
            const int hi = wmax + A.window;
            if (hi < jhi) jhi = hi > 0 ? hi : 0;
        }
    }
    if (jhi < 0) jhi = 0;
    const int cols = (A.jcap + BLOCK - 1) / BLOCK;
    int b0 = jlo / BLOCK;
    int b1 = (jhi + BLOCK - 1) / BLOCK;
    b1 = b1 > cols ? cols : b1;
    if (b0 < 0) b0 = 0;

    if (SPLIT) {
        const int nblk = (jend_all + BLOCK - 1) / BLOCK;
        const int tps = (nblk + A.splits - 1) / A.splits;
        const int sl = (int)blockIdx.z;
        const int s0 = sl * tps;
        const int s1 = s0 + tps < nblk ? s0 + tps : nblk;
        if (b0 < s0) b0 = s0;
        if (b1 > s1) b1 = s1;
    }

    /* The key range the workgroup walks, in flat key indices: blocks b0..b1-1, the last one cut at
     * `jend_all`. libref keeps a score a key; so does this row when `nkeys * kRows` fits the LDS
     * score cache. The decision and the index (`j - key_base`) are uniform across the workgroup. */
    const int key_base = b0 * BLOCK;
    const int key_end  = jend_all < b1 * BLOCK ? jend_all : b1 * BLOCK;
    const int nkeys    = key_end > key_base ? key_end - key_base : 0;
    const bool cache   = (long long)nkeys * kRows <= kScores;
    r3d_lds_f32* const srow = (r3d_lds_f32*)sScore + (size_t)rr * (size_t)nkeys;

    float mx = -INFINITY;
    /* Pass one: the maximum over visible scores. Ascending key order, same arithmetic as pass two,
     * so the score a key gets here is the score it gets there. This pass reads K and K's scale
     * only, so it stages no V (pass two stages the block's V rows anyway). When the scores fit,
     * lane 0 of each row group keeps each visible key's score for pass two. */
    for (int b = b0; b < b1; ++b) {
        const int blk = ld_ix(A.bt, (size_t)M.s * A.bt_rs + (size_t)b * A.bt_cs);
        const int j0 = b * BLOCK;
        const int nu = jend_all - j0 < BLOCK ? jend_all - j0 : BLOCK;
        group_barrier();
        if (blk >= 0) stage_i8<NTH, true, false>(A, blk, kh, nu, tid, (r3d_lds_i8*)sK, (r3d_lds_i8*)sV, (r3d_lds_f32*)sks, (r3d_lds_f32*)svs);
        group_barrier();
        if (blk < 0) continue;
        int u = 0;
        for (; u + 1 < nu; u += 2) {
            float dot0, dot1;
            dot_group2(qf, (const r3d_lds_i8*)(sK + u * kPitch + d0),
                       (const r3d_lds_i8*)(sK + (u + 1) * kPitch + d0),
                       ((const r3d_lds_f32*)sks)[u], ((const r3d_lds_f32*)sks)[u + 1],
                       gbase, dot0, dot1);
            const int j = j0 + u;
            const bool vis0 = active && visible(j, pos, CAUSAL, A.window);
            if (vis0) {
                const float sc = dot0 * A.scale;
                if (cache && db == 0) srow[j - key_base] = sc;
                if (sc > mx) mx = sc;
            }
            const bool vis1 = active && visible(j + 1, pos, CAUSAL, A.window);
            if (vis1) {
                const float sc = dot1 * A.scale;
                if (cache && db == 0) srow[j + 1 - key_base] = sc;
                if (sc > mx) mx = sc;
            }
        }
        if (u < nu) {
            const float dot = dot_group(qf, (const r3d_lds_i8*)(sK + u * kPitch + d0), ((const r3d_lds_f32*)sks)[u], gbase);
            const int j = j0 + u;
            const bool vis = active && visible(j, pos, CAUSAL, A.window);
            if (vis) {
                const float sc = dot * A.scale;
                if (cache && db == 0) srow[j - key_base] = sc;
                if (sc > mx) mx = sc;
            }
        }
    }

    /* A row with no visible key (an empty window, or a padding slot) still walks pass two: the
     * stage and the barriers are workgroup-wide, so an early return here would leave the surviving
     * row groups reading LDS that only some waves wrote. Its visibility is false for every key, so
     * sum stays 0 and the store below writes a zero row (or a zero partial record). */

    float acc[kDimsLane];
#pragma unroll
    for (int i = 0; i < kDimsLane; ++i) acc[i] = 0.0f;
    float sum = 0.0f;

    /* Pass two: exponentiate and accumulate in ascending key order. With the scores cached it
     * stages V only and reads the max pass's score; otherwise it restages K too and recomputes
     * the dot with the same operands and the same order. */
    for (int b = b0; b < b1; ++b) {
        const int blk = ld_ix(A.bt, (size_t)M.s * A.bt_rs + (size_t)b * A.bt_cs);
        const int j0 = b * BLOCK;
        const int nu = jend_all - j0 < BLOCK ? jend_all - j0 : BLOCK;
        group_barrier();
        if (blk >= 0) {
            if (cache) stage_i8<NTH, false, true>(A, blk, kh, nu, tid, (r3d_lds_i8*)sK, (r3d_lds_i8*)sV, (r3d_lds_f32*)sks, (r3d_lds_f32*)svs);
            else       stage_i8<NTH, true,  true>(A, blk, kh, nu, tid, (r3d_lds_i8*)sK, (r3d_lds_i8*)sV, (r3d_lds_f32*)sks, (r3d_lds_f32*)svs);
        }
        group_barrier();
        if (blk < 0) continue;
        for (int u = 0; u < nu; ++u) {
            const int j = j0 + u;
            const bool vis = active && visible(j, pos, CAUSAL, A.window);
            if (vis) {
                float sc;
                if (cache) {
                    sc = srow[j - key_base];
                } else {
                    const float dot = dot_group(qf, (const r3d_lds_i8*)(sK + u * kPitch + d0), ((const r3d_lds_f32*)sks)[u], gbase);
                    sc = dot * A.scale;
                }
                const float w = exp_ref(sc - mx);
                sum += w;
                const float vm = ((const r3d_lds_f32*)svs)[u];
                const r3d_lds_i8* vrow = (const r3d_lds_i8*)(sV + u * kPitch + d0);
#pragma unroll
                for (int i = 0; i < kDimsLane; ++i)
                    acc[i] = acc[i] + w * ((float)(int8_t)vrow[i] * vm);
            }
        }
    }

    /* Padding slots clamp to a real row; an unguarded partial store would add a writer to that
     * row's arena record. Store nothing for them (their lanes' sums/accs are duplicates anyway). */
    if (!row_ok) return;

    if (SPLIT) {
        float* pp = A.part + ((size_t)row * A.n_head + h) * ((size_t)A.splits * kPartStride)
                    + (size_t)blockIdx.z * kPartStride;
        for (int i = 0; i < kDimsLane; i += 4) {
            const r3d_f32x4 v = { acc[i], acc[i + 1], acc[i + 2], acc[i + 3] };
            *(r3d_f32x4*)(void*)(pp + d0 + i) = v;
        }
        if (db == 0) { pp[kHeadDim] = sum; pp[kHeadDim + 1] = mx; }
        return;
    }

    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;     /* no visible key: a zero row */
    uint16_t* op = A.out + (long long)row * A.o_rs + (long long)h * kHeadDim + d0;
#pragma unroll
    for (int u = 0; u < kDimsLane / 8; ++u) {
        uint32_t q4[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = 8 * u + 2 * j;
            q4[j] = bf16x2_rne(acc[d] * inv, acc[d + 1] * inv);
        }
        *(r3d_u32x4*)(void*)(op + 8 * u) = r3d_u32x4{ q4[0], q4[1], q4[2], q4[3] };
    }
}

/* ------------------------------------------------------------------ split-KV reduce
 *
 * A wave per (query row, head): 32 lanes x 8 dims. Slices combine ascending, each rescaled by
 * exp(m_slice - M) with M the slices' maximum, which is where split-KV leaves the reference's
 * parentheses.
 */
__global__ __launch_bounds__(128) R3D_WAVES(4)
void attn_reduce(const Args A, long long nrows)
{
    const int w = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;
    const long long idx = (long long)blockIdx.x * 4 + w;
    const long long total = nrows * A.n_head;
    if (idx >= total) return;
    const long long row = idx / A.n_head, h = idx - row * A.n_head;
    const int s = seq_of_row(A, (int)row);
    const long long ctx = ld_ix(A.su, (size_t)s);
    const int jend_all = (int)(ctx < A.jcap ? ctx : A.jcap);
    const int nblk = (jend_all + A.block - 1) / A.block;
    const int tps = (nblk + A.splits - 1) / A.splits;
    const int nslice = tps > 0 ? (nblk + tps - 1) / tps : 0;

    float* pp = A.part + (size_t)idx * ((size_t)A.splits * kPartStride);
    float M = -INFINITY;
    for (int u = 0; u < nslice; ++u) {
        const float m = pp[(size_t)u * kPartStride + kHeadDim + 1];
        if (m > M) M = m;
    }
    uint16_t* o = A.out + (long long)row * A.o_rs + (long long)h * kHeadDim;
    const int d0 = lane * 8;
    float a[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = 0.0f;
    float sum = 0.0f;
    if (M > -INFINITY) {
        for (int sl = 0; sl < nslice; ++sl) {
            const float* q = pp + (size_t)sl * kPartStride;
            const float f = exp_ref(q[kHeadDim + 1] - M);
            sum += q[kHeadDim] * f;
#pragma unroll
            for (int i = 0; i < 8; ++i) a[i] += q[d0 + i] * f;
        }
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    uint32_t q4[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) q4[j] = bf16x2_rne(a[2 * j] * inv, a[2 * j + 1] * inv);
    *(r3d_u32x4*)(void*)(o + d0) = r3d_u32x4{ q4[0], q4[1], q4[2], q4[3] };
}

}  // namespace

/* ================================================================== launchers
 *
 * Raw addresses and the dense layouts the shim has checked. The cache is
 * [blocks, kv_heads, block_size, 2*256+4] int8 bytes. With `cu`, `q_len` is the query plane's
 * row count (cu[num_seqs]); without it, the per-sequence length.
 */
namespace {

int fill(Args& A, long q, long kv, long bt, long seqused, long cu, long out, long rows,
         int num_seqs, int q_len, int head_dim, int gqa, int n_head, int block_size,
         int bt_cols, int bt_rs, int bt_cs, int max_ctx, int window, int causal, float scale,
         int kv_dt, long q_ld, long out_ld, int splits, long scratch)
{
    A = Args{};
    if (head_dim != kHeadDim) return -2;
    if (num_seqs <= 0 || n_head <= 0) return -2;
    if (cu) { if (rows < 1 || rows > 2147483000LL) return -2; }
    else if (q_len < 1 || (long long)num_seqs * q_len > 2147483000LL) return -2;
    if (gqa != 6 && gqa != 12) return -2;
    if (!((gqa == 12 && block_size == 4) || (gqa == 6 && block_size == 16))) return -2;
    if (n_head % gqa) return -2;
    if (window < 0 || max_ctx < 1) return -2;
    if (kv_dt != RAD_I8) return -1;
    if (!q || !kv || !bt || !seqused) return -1;

    const int kv_heads = n_head / gqa;
    const int cache_blocks = (max_ctx + block_size - 1) / block_size;
    if (bt_cols < 1 || bt_rs < 1 || bt_cs < 1) return -2;

    A.q  = (const uint16_t*)(uintptr_t)q;
    A.kv = (const uint8_t*)(uintptr_t)kv;
    A.bt = (const int*)(uintptr_t)bt;
    A.su = (const int*)(uintptr_t)seqused;
    A.cu = (const int*)(uintptr_t)cu;
    A.out = (uint16_t*)(uintptr_t)out;

    /* A slot is 256 K codes, 256 V codes, two f16 scales: 516 bytes. All strides stay 32-bit. */
    A.so = 2 * kHeadDim + 4;
    A.sh = block_size * A.so;
    A.sb = kv_heads * A.sh;
    /* A page's slots are 516 bytes, so a row is 4-aligned at best; the page base however is
     * 16-aligned whenever the cache is, because `sh` (4 * 516 or 16 * 516) and `sb` are multiples
     * of 16. The staging copy reads a page as a flat run of 16-byte chunks only under that
     * condition; otherwise it stays on 4-byte words. */
    A.kv16 = (((uintptr_t)A.kv | (uintptr_t)A.sb | (uintptr_t)A.sh) & 15) == 0;
    if ((long long)A.sb * cache_blocks > 2147483000LL) return -2;
    if (q_ld > 2147483000LL || out_ld > 2147483000LL) return -2;
    A.q_rs = (int)q_ld;
    A.o_rs = (int)out_ld;
    A.bt_rs = bt_rs;
    A.bt_cs = bt_cs;
    A.n_seq = num_seqs;
    A.q_len = cu ? 0 : q_len;
    A.rows = cu ? (int)rows : num_seqs * q_len;
    A.n_head = n_head;
    A.kv_heads = kv_heads;
    A.block = block_size;
    A.causal = causal ? 1 : 0;
    A.window = window;
    A.scale = scale > 0.0f ? scale : 1.0f / std::sqrt((float)kHeadDim);
    A.jcap = max_ctx;
    if ((long long)bt_cols * block_size < max_ctx) A.jcap = bt_cols * block_size;
    A.splits = splits > 0 ? splits : 1;
    A.part = A.splits > 1 ? (float*)(uintptr_t)scratch : nullptr;
    if (A.splits > 1 && !scratch) return -5;
    return 0;
}

/* Workgroups a 16-CU card is filled with: two a CU at the band's wave count, on a doubling ladder,
 * capped at 16 slices (a 24/2 decode has only two KV heads; eight slices would leave the card at
 * 16 workgroups). */
int pick_splits(int tiles, int num_seqs, int kv_heads)
{
    const long long blocks = (long long)tiles * num_seqs * kv_heads;
    int sp = 1;
    while (sp < 16 && blocks * sp < 32) sp *= 2;
    return sp;
}

int launch(const Args& A, long stream)
{
    const int gqa = A.n_head / A.kv_heads;
    const bool causal = A.causal != 0;
    const bool split = A.splits > 1;
    hipStream_t st = (hipStream_t)(uintptr_t)stream;

    /* One launch per (gqa, block_size, band) through the template's wave count. `A.band` is set by
     * the two entries, so decode (three waves) and prefill (six) never share an instantiation. */
#define R3D_ATTN2(GQAV, BLOCKV, WAVESV)                                                  \
    do {                                                                                 \
        constexpr int TPT = kRowsWave * WAVESV / GQAV;                                   \
        const int tiles = A.cu                                                           \
            ? (A.rows + TPT - 1) / TPT + A.n_seq - 1                                     \
            : (A.q_len + TPT - 1) / TPT * A.n_seq;                                       \
        dim3 grid((unsigned)tiles, (unsigned)A.kv_heads,                                 \
                  (unsigned)(split ? A.splits : 1));                                     \
        const dim3 block(WAVESV * kLanesWave);                                                   \
        if (causal) {                                                                    \
            if (split) R3D_KLAUNCH((attn_kernel<true, true, GQAV, BLOCKV, WAVESV>),      \
                                   grid, block, st, A);                                  \
            else       R3D_KLAUNCH((attn_kernel<true, false, GQAV, BLOCKV, WAVESV>),     \
                                   grid, block, st, A);                                  \
        } else {                                                                         \
            if (split) R3D_KLAUNCH((attn_kernel<false, true, GQAV, BLOCKV, WAVESV>),     \
                                   grid, block, st, A);                                  \
            else       R3D_KLAUNCH((attn_kernel<false, false, GQAV, BLOCKV, WAVESV>),    \
                                   grid, block, st, A);                                  \
        }                                                                                \
    } while (0)

    (void)hipGetLastError();
    if (gqa == 12) {
        if (A.band == 0) R3D_ATTN2(12, 4, 3);
        else             R3D_ATTN2(12, 4, 6);
    } else {
        if (A.band == 0) R3D_ATTN2(6, 16, 3);
        else             R3D_ATTN2(6, 16, 6);
    }
#undef R3D_ATTN2
    {
        const hipError_t ke = hipGetLastError();
        if (ke != hipSuccess) return -4;
    }

    if (split) {
        const long long rows = A.rows;
        const long long total = rows * A.n_head;
        dim3 rg((unsigned)((total + 3) / 4));
        (void)hipGetLastError();
        R3D_KLAUNCH(attn_reduce, rg, dim3(128), st, A, rows);
        if (hipGetLastError() != hipSuccess) return -4;
    }
    return 0;
}

}  // namespace

extern "C" int r3d_attn_decode_h256(long q, long kv, long bt, long seqused, long cu, long out,
                                    int num_seqs, int q_len, int head_dim, int gqa, int n_head,
                                    int block_size, int bt_cols, int bt_rs, int bt_cs,
                                    int max_ctx, int window, int causal, float scale, int kv_dt,
                                    long q_ld, long out_ld, int splits, long scratch, long stream)
{
    if (splits < 0) return -2;
    if (gqa <= 0 || n_head <= 0 || n_head % gqa) return -2;
    if (cu && num_seqs <= 0) return -2;
    Args A;
    const int kv_heads = n_head / gqa;
    const int per = cu ? (q_len + num_seqs - 1) / num_seqs : (q_len > 0 ? q_len : 1);
    const int tpb = kRowsWave * 3 / gqa;      /* decode's waves */
    const int tiles = per > 0 ? (per + tpb - 1) / tpb : 1;
    int sp = splits > 0 ? splits : pick_splits(tiles, num_seqs, kv_heads);
    const int e = fill(A, q, kv, bt, seqused, cu, out, cu ? q_len : 0, num_seqs,
                       cu ? 0 : q_len, head_dim, gqa, n_head, block_size, bt_cols, bt_rs, bt_cs,
                       max_ctx, window, causal, scale, kv_dt, q_ld, out_ld, sp, scratch);
    if (e) return e;
    A.band = 0;
    return launch(A, stream);
}

extern "C" int r3d_attn_prefill_h256(long q, long kv, long bt, long seqused, long cu, long out,
                                     int num_seqs, int q_len, int head_dim, int gqa, int n_head,
                                     int block_size, int bt_cols, int bt_rs, int bt_cs,
                                     int max_ctx, int window, int causal, float scale, int kv_dt,
                                     long q_ld, long out_ld, long scratch, long stream)
{
    (void)scratch;
    Args A;
    const int e = fill(A, q, kv, bt, seqused, cu, out, cu ? q_len : 0, num_seqs,
                       cu ? 0 : q_len, head_dim, gqa, n_head, block_size, bt_cols, bt_rs, bt_cs,
                       max_ctx, window, causal, scale, kv_dt, q_ld, out_ld, 1, 0);
    if (e) return e;
    A.band = 1;
    A.splits = 1;
    A.part = nullptr;
    return launch(A, stream);
}

/* ================================================================== shims */
namespace {

enum { P_Q = 0, P_KV = 1, P_BT = 2, P_SU = 3, P_KDS = 4, P_VDS = 5, P_CU = 6, P_OUT = 7 };

int attn_shim(const RadArgs* a, RadStream s, bool decode)
{
    if (a->n_t < 8) return R3D_NO(RAD_E_INVAL);
    const RadTensor* Q = r3d_opt(a, P_Q);
    const RadTensor* KV = r3d_opt(a, P_KV);
    const RadTensor* BT = r3d_opt(a, P_BT);
    const RadTensor* SU = r3d_opt(a, P_SU);
    const RadTensor* O = r3d_opt(a, P_OUT);
    const RadTensor* KDS = r3d_opt(a, P_KDS);
    const RadTensor* VDS = r3d_opt(a, P_VDS);
    const RadTensor* CU = r3d_opt(a, P_CU);
    if (!Q || !KV || !BT || !SU || !O) return R3D_NO(RAD_E_INVAL);
    if (!Q->data || !KV->data || !BT->data || !SU->data || !O->data) return R3D_NO(RAD_E_INVAL);

    const bool ragged = r3d_has(CU);
    const int head_dim = (int)r3d_geti_or(a, "head_dim", r3d_dim(Q, (int)Q->rank - 1));
    if (head_dim != kHeadDim) return R3D_NO(RAD_E_SHAPE);
    const int gqa = (int)r3d_geti_or(a, "gqa", 1);
    if (gqa != 6 && gqa != 12) return R3D_NO(RAD_E_SHAPE);
    const int causal = (int)r3d_geti_or(a, "causal", 1);
    const int window = (int)r3d_geti_or(a, "window", 0);
    const float scale = (float)r3d_getf(a, "scale", 0.0);

    if (Q->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (O->dtype != RAD_BF16) return R3D_NO(RAD_E_DTYPE);
    if (KV->dtype != RAD_I8) return R3D_NO(RAD_E_DTYPE);
    if (BT->dtype != RAD_I32 || SU->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);
    /* No descale planes on an int8 cache: its scales live in the slot. Refused, not ignored. */
    if ((KDS && KDS->data) || (VDS && VDS->data)) return R3D_NO(RAD_E_SHAPE);

    const int n_head = (int)r3d_geti_or(a, "n_head", r3d_dim(Q, (int)Q->rank - 2));
    if (n_head <= 0 || n_head % gqa) return R3D_NO(RAD_E_SHAPE);
    const int kv_heads = n_head / gqa;
    const int block_size = (int)r3d_geti_or(a, "block_size", r3d_dim(KV, 2));
    if (!((gqa == 12 && block_size == 4) || (gqa == 6 && block_size == 16)))
        return R3D_NO(RAD_E_SHAPE);
    if (KV->rank != 4) return R3D_NO(RAD_E_SHAPE);
    if (r3d_dim(KV, 1) != kv_heads || r3d_dim(KV, 2) != block_size) return R3D_NO(RAD_E_SHAPE);
    /* The slot row is fixed at 2 * 256 + 4 bytes (K, V, two f16 scales); the kernel's stride is
     * that constant, so a wider last dim would be read at the wrong offset. */
    if (r3d_dim(KV, 3) != 2 * kHeadDim + 4) return R3D_NO(RAD_E_SHAPE);
    const long long nb = r3d_dim(KV, 0);
    if (nb < 1) return R3D_NO(RAD_E_SHAPE);
    const int max_ctx = (int)(nb * block_size);
    if ((long long)max_ctx != nb * block_size) return R3D_NO(RAD_E_SHAPE);

    const long long n_seq = r3d_numel(SU);
    if (n_seq < 1 || n_seq > 65536) return R3D_NO(RAD_E_SHAPE);
    if (ragged) {
        if (CU->dtype != RAD_I32) return R3D_NO(RAD_E_DTYPE);
        if (!r3d_contig(CU)) return R3D_NO(RAD_E_STRIDE);
        if (r3d_numel(CU) != n_seq + 1) return R3D_NO(RAD_E_SHAPE);
        if (!r3d_aligned(CU->data, 4)) return R3D_NO(RAD_E_ALIGN);
    }
    long long qh = 0, qt = 0, qld = 0;
    if (!r3d_head_rows(Q, kHeadDim, &qt, &qh, &qld) || qh != n_head)
        return R3D_NO(RAD_E_STRIDE);
    long long oh = 0, ot = 0, old = 0;
    if (!r3d_head_rows(O, kHeadDim, &ot, &oh, &old) || oh != n_head)
        return R3D_NO(RAD_E_STRIDE);
    if (qld % 8 || old % 8) return R3D_NO(RAD_E_STRIDE);
    if (qt < 1 || qt > 2147483000LL) return R3D_NO(RAD_E_SHAPE);
    if (!ragged && qt % n_seq) return R3D_NO(RAD_E_SHAPE);
    const long long q_len = ragged ? qt : qt / n_seq;
    const long long rows = qt;
    if (q_len < 1) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(O) < rows * n_head * kHeadDim) return R3D_NO(RAD_E_SHAPE);

    if (!r3d_contig(KV)) return R3D_NO(RAD_E_STRIDE);
    const long long bt_cols = BT->rank >= 2 ? r3d_dim(BT, (int)BT->rank - 1)
                                            : (n_seq ? r3d_numel(BT) / n_seq : 0);
    const long long bt_rs = BT->rank >= 2 ? BT->stride[BT->rank - 2] : bt_cols;
    const long long bt_cs = BT->rank >= 2 ? BT->stride[BT->rank - 1] : 1;
    if (bt_cols < 1 || bt_rs < 1 || bt_cs < 1) return R3D_NO(RAD_E_STRIDE);
    const long long bt_reach = (n_seq - 1) * bt_rs + (bt_cols - 1) * bt_cs + 1;
    if (r3d_numel(BT) < bt_reach) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_contig(SU)) return R3D_NO(RAD_E_STRIDE);
    if (!r3d_aligned(Q->data, 16) || !r3d_aligned(KV->data, 4) || !r3d_aligned(O->data, 16))
        return R3D_NO(RAD_E_ALIGN);

    int splits = decode ? 0 : 1;
    const long long per = ragged ? (rows + n_seq - 1) / n_seq : q_len;
    const int waves = decode ? 3 : 6;
    const int tpb = kRowsWave * waves / gqa;
    const int tiles = (int)((per + tpb - 1) / tpb);
    if (splits == 0) splits = pick_splits(tiles, (int)n_seq, kv_heads);
    if (splits > kSplitMax) splits = kSplitMax;
    if (splits > 1) {
        for (;;) {
            if (splits <= 1) { splits = 1; break; }
            const long long need = qt * n_head * (long long)splits * kPartStride * 4;
            if (a->scratch && a->scratch_bytes >= need) break;
            splits /= 2;
        }
    }

    const int e = decode
        ? r3d_attn_decode_h256(r3d_addr(Q), r3d_addr(KV), r3d_addr(BT), r3d_addr(SU),
                               r3d_addr(CU), r3d_addr(O), (int)n_seq,
                               (int)(ragged ? rows : q_len), head_dim, gqa, n_head, block_size,
                               (int)bt_cols, (int)bt_rs, (int)bt_cs, max_ctx, window, causal,
                               scale, (int)KV->dtype, qld, old, splits,
                               (long)(uintptr_t)(splits > 1 ? a->scratch : nullptr),
                               (long)(uintptr_t)s)
        : r3d_attn_prefill_h256(r3d_addr(Q), r3d_addr(KV), r3d_addr(BT), r3d_addr(SU),
                                r3d_addr(CU), r3d_addr(O), (int)n_seq,
                                (int)(ragged ? rows : q_len), head_dim, gqa, n_head, block_size,
                                (int)bt_cols, (int)bt_rs, (int)bt_cs, max_ctx, window, causal,
                                scale, (int)KV->dtype, qld, old, (long)(uintptr_t)nullptr,
                                (long)(uintptr_t)s);
    if (e == 0) return RAD_OK;
    return R3D_NO(e == -1 ? RAD_E_DTYPE : e == -2 ? RAD_E_SHAPE
                        : e == -3 ? RAD_E_ALIGN : e == -4 ? RAD_E_DEVICE
                        : e == -5 ? RAD_E_SCRATCH : RAD_E_INVAL);
}

}  // namespace

extern "C" int r3d_rad_attn_decode_h256(const RadArgs* a, RadStream s)
{
    return attn_shim(a, s, true);
}

extern "C" int r3d_rad_attn_prefill_h256(const RadArgs* a, RadStream s)
{
    return attn_shim(a, s, false);
}
