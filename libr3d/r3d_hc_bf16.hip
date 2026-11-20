/* r3d_hc_bf16.hip -- HyperConnections' read, write, enter and projection.
 *
 *   h is [M, hc*n] bf16, one wide residual stream cut into hc connections of n.
 *   read:  N[c*n+i] = bf16(h * rsqrt(mean(h_c^2)+eps) * (w+wadd))
 *          d[j]      = bf16(silu(sum_i N[i] * down[j][i] / hc))
 *          gv[j]     = sum_k d[k] * up[j][k]                 j is a connection's column
 *          x[i]      = sum_c bf16(sigmoid(gv[c*n+i])) * N[c*n+i] / hc
 *          inj[c]    = bf16(2 * sigmoid(sum_i N[i] * injw[c][i] / hc))
 *          q / rq    = the quantised and rotated copies of x, exactly quant_act's and
 *                      had_quant_act's group arithmetic
 *   write:   h += bf16(inj[c]) * bf16(y[i])
 *   enter:   h[c*n+i] = bf16(x[i])
 *   project: for a connection with on[r] set, h -= g * v where
 *            g = (float)(sum_i h[i] * v[i] * scale)        (double sum, as libref writes it)
 *
 * libref/ref_hc.cpp is the oracle and docs/OPS.md the contract. The read is split over four
 * launches because each phase has a different natural grid:
 *
 *   norm    one block a row, hc threads: the hc double sums and their scales;
 *   down    one block a (8-row tile, 32-output tile), plus one a row tile for `inj`: every thread
 *           owns one (row, output) pair and accumulates it, so all 256 lanes are busy and the
 *           staged slice of 32 weight rows serves 8 rows. From 256 rows the block instead owns a
 *           16-row by 64-output tile, two outputs and two rows a lane: one staged weight feeds
 *           two rows and the grid needs five output tiles instead of ten. The block rebuilds N
 *           in LDS as bf16 one
 *           128-column chunk at a time (the normalisation is cheap beside the weight stream) and
 *           sums i ascending in f32 exactly as ref does. In the E4M3 form a lane reads one
 *           16-byte vector of codes beside its group's scale, so the slice is one load a lane
 *           instead of 32 lanes fetching 32 different rows at one offset (the bf16 form keeps the
 *           scalar walk). The injection gains have their own workgroup, one a row tile, because
 *           their chain would otherwise run behind a tile's;
 *   up      one block a (8-row tile, 32-column tile): each thread owns one (row, column) pair,
 *           sums c ascending, and the block stages those 32 weight rows into LDS 64 k at a time,
 *           again with all 256 threads, four lanes a row in the E4M3 form;
 *   quant   one block a (row, group): q is a sequential group scan, rq a wave transform in LDS.
 *
 * A launch narrower than a row tile is decode, where the tile's spare rows would be lanes doing a
 * dot nothing reads: it keeps the W28 one-row-a-block kernels instead (16 outputs by 256 columns a
 * slice for down, 64 columns by 32 k for up), so a one-row call is not slower for the tiling that
 * a prefill chunk needs.
 *
 * `d` and the per-stream scales travel in the row's scratch: M*(lowrank*2 + hc*4) bytes. The
 * weight stream is read once per row tile, coalesced, and the staged slice is decoded once. The
 * accumulations are ascending and f32 (double for the norm), with no FMA, so the read is
 * bit-identical to the reference, not merely close.
 */
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_common.h"
#include "r3d_hc_ple_common.h"
#include "r3d_launch.h"

namespace {

constexpr int kHcThreads = 256;
/* A workgroup owns one tile of rows by one tile of outputs; the two multiply to the block's
 * thread count times the pairs a lane owns (`P` outputs and `Q` rows), so a lane accumulates
 * several (row, output) pairs at once. That is what the staged slice's LDS traffic needs: the
 * weight read is shared by the lane's Q rows and the N read by its P outputs, so the two LDS
 * instructions a MAC otherwise costs fall to 1/P + 1/Q. The larger tiles also serve more rows
 * per staged weight slice. Below one row tile the W28 kernels take over. */
constexpr int kHcDownRows = 8;      /* stream rows a one-pair down workgroup owns */
constexpr int kHcUpRows = 8;        /* stream rows a one-pair up workgroup owns */
/* The 2x2 down tile stages a 64-output slice, so its grid has five tiles a row set instead of
 * ten. At 128 rows that leaves 48 workgroups and the narrow tile is faster; from 256 rows the
 * wider one measures equal and from 512 it is 5% ahead, reaching 8-9% at 2k and 4k rows. */
constexpr int kHcDownWide = 256;
/* The wide tile's output width, 32*P for the P=2 instantiation below (the kernel derives it from
 * its template arguments; the launch's tile count must match). */
constexpr int kHcDownTileWide = 32 * 2;
/* A launch narrower than a row tile keeps the W28 kernels, which are one row a block: no row of
 * the tile is an empty lane, and a one-row call keeps the latency it was measured at. */
constexpr int kHcW28DownTile = 16;
/* The decode band's slice. It stays 256 so the stage plus N's tile fits the 64 KiB LDS for every
 * declared width: at 512 the 16-row stage alone is 32.8 KiB and N's tile is two bytes an element,
 * so a 16,384-wide stream would need 65.6 KiB and such a model would be refused. The slice changes
 * no value: a column's sum stays ascending in i across the boundary. */
constexpr int kHcW28DownChunk = 256;
constexpr int kHcW28DownPitch = kHcW28DownChunk + 1;
constexpr int kHcW28DownStage = kHcW28DownTile * kHcW28DownPitch;
constexpr int kHcW28UpCols = 64;
/* The decode slice over the low rank. Each column's sum stays ascending in k across the slice
 * boundary, so 64 keeps the arithmetic of 32 exactly and halves the load/dot ladder (five rounds
 * for lr = 320) at 17.5 KiB of LDS. */
constexpr int kHcW28UpChunk = 64;
static_assert(kHcW28UpCols * (kHcW28UpChunk / 16) <= kHcThreads,
              "the vector staging path must cover every column of a block");
constexpr int kHcW28UpStage = kHcW28UpCols * (kHcW28UpChunk + 1);
constexpr int kHcQuantScratch = 256;   /* the deepest rotated group the quantiser transforms */

/* Scratch layout: d [M, lowrank] bf16, then the stream scales [M, hc] f32, 16-byte aligned. */
static inline long long hc_sc_off(long long M, long long lr) {
    return ((M * lr * 2 + 15) / 16) * 16;
}
static inline long long hc_scratch_bytes(long long M, long long lr, long long hc) {
    return hc_sc_off(M, lr) + M * hc * 4;
}

/* A mixing weight: bf16 row-major, or the E4M3 row with its scales in the row's tail (the
 * layout hook below). `K` is the dot length and `G` the scale group. */
struct MixW {
    const void* base;
    int f8;
    int ld;     /* bf16 row pitch in elements */
    int stride; /* E4M3 row stride in bytes */
    int K;
    int G;
};

__device__ __forceinline__ float mix_get(const MixW& w, long long row, int i) {
    if (!w.f8) return bf16_to_f32(((const uint16_t*)w.base)[row * w.ld + i]);
    const uint8_t* r = (const uint8_t*)w.base + row * w.stride;
    return e4m3_to_f32(r[i]) * ((const float*)(r + w.K))[i / w.G];
}

/* The reference's sum: `acc += a[i] * b[i]`, i ascending, f32, one chain. The two staged slices
 * are read into registers eight at a time before the adds, so the LDS round trip of a slice is
 * paid once every eight elements instead of on every add; the adds themselves keep their order and
 * the carry `acc` crosses the slice boundary exactly as the unbatched walk did. */
__device__ __forceinline__ float hc_dot_asc_f(float acc, const float* a, const float* b, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float x[8], y[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) x[u] = a[i + u];
#pragma unroll
        for (int u = 0; u < 8; ++u) y[u] = b[i + u];
#pragma unroll
        for (int u = 0; u < 8; ++u) acc += x[u] * y[u];
    }
    for (; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

/* ---- the per-stream norm scale ------------------------------------------------------------- */
/* One workgroup covers as many rows as its threads divide by hc, so every lane is an active
 * chain: the sum is a dependent double accumulation and exposes its latency, and one row a
 * workgroup left 252 of 256 lanes idle. Eight loads are issued before their adds, which leaves
 * the ascending double sum (the reference's) untouched.
 *
 * At decode there is one row and only `hc` lanes are active (four at Flash-Next's hc = 4), so the
 * scalar walk is a stream of 2,560 dependent 2-byte loads a lane: 107 us a launch, 96 launches a
 * step, 10.3 ms of the 92.4 ms step (measured, RADIANCE_AQL_STAMP). The same ascending double sum
 * read through 16-byte vectors, sixteen vectors in flight, leaves the order and every rounding
 * untouched while it turns 2,560 round trips into 20. A 16-byte vector is 8 bf16 on a boundary the
 * row start keeps (n and h_ld are multiples of 8 at every served geometry); the tail and a
 * misaligned row keep the scalar walk. */
constexpr int kHcNormBatch = 8;
constexpr int kHcNormVec    = 16;   /* 16-byte loads in flight (128 bf16 a round) */

__global__ __launch_bounds__(kHcThreads) void hc_norm_kernel(
    const uint16_t* __restrict__ h, const float* __restrict__ w, float* __restrict__ sc, int M,
    int n, int hc, int h_ld, float eps) {
    (void)w;
    const int t = (int)threadIdx.x;
    const int rows = kHcThreads / hc;
    const int c = t - (t / hc) * hc;
    const int r = t / hc;
    if (r >= rows) return;
    const int row = (int)blockIdx.x * rows + r;
    if (row >= M) return;
    const uint16_t* hp = h + (long long)row * h_ld + (long long)c * n;
    double ss = 0.0;
    int i = 0;
    if (((uintptr_t)hp & 15) == 0) {
        const int step = kHcNormVec * 8;
        for (; i + step <= n; i += step) {
            r3d_u32x4 v[kHcNormVec];
#pragma unroll
            for (int u = 0; u < kHcNormVec; ++u)
                v[u] = *(const r3d_u32x4*)(hp + i + u * 8);
#pragma unroll
            for (int u = 0; u < kHcNormVec; ++u) {
                /* Extract from the vector's words: taking the address of v[u] would spill the
                 * array to scratch and turn the walk back into dependent local-memory loads. */
                const uint32_t wv[4] = {v[u][0], v[u][1], v[u][2], v[u][3]};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const float lo = bf16_to_f32((uint16_t)(wv[q] & 0xffffu));
                    const float hi = bf16_to_f32((uint16_t)(wv[q] >> 16));
                    ss += (double)lo * (double)lo;
                    ss += (double)hi * (double)hi;
                }
            }
        }
    }
    for (; i + kHcNormBatch <= n; i += kHcNormBatch) {
        float v[kHcNormBatch];
#pragma unroll
        for (int u = 0; u < kHcNormBatch; ++u) v[u] = bf16_to_f32(hp[i + u]);
#pragma unroll
        for (int u = 0; u < kHcNormBatch; ++u) ss += (double)v[u] * (double)v[u];
    }
    for (; i < n; ++i) {
        const float v = bf16_to_f32(hp[i]);
        ss += (double)v * (double)v;
    }
    const float mean = (float)(ss / (double)n);
    sc[(long long)row * hc + c] = 1.0f / sqrtf(mean + eps);
}

/* ---- down: d and inj ----------------------------------------------------------------------- */
/* LDS: TILE staged weight rows of CHUNK floats (one float of pad a row, so an owner lane's
 * sequential walk and its neighbours start on distinct banks), then N bf16 [ROWS][chunk],
 * pitch-padded so the tile's rows do not collide in a bank. A lane owns P outputs of one row set
 * and Q rows: one staged weight value feeds its Q rows and one staged N value its P outputs, so a
 * MAC costs 1/P + 1/Q LDS instructions instead of two, and the adds stay ascending per (row,
 * output) in f32 without FMA. The injection block reuses the weight stage for its hc rows, since
 * it stages no tile. The decode of a staged element is `mix_get`'s, once, so the dot reads what
 * the per-element read produced. */
constexpr size_t hc_down_lds_bytes_impl(int tile, int rows, int chunk) {
    return (size_t)tile * (chunk + 1) * sizeof(float) +
           (size_t)rows * (chunk + 8) * sizeof(uint16_t);
}
template <int P, int Q, int CHUNK>
constexpr size_t hc_down_lds_bytes() {
    static_assert(32 * P * 8 * Q == kHcThreads * P * Q, "a down lane owns P outputs by Q rows");
    return hc_down_lds_bytes_impl(32 * P, 8 * Q, CHUNK);
}

template <int P, int Q, int CHUNK>
__device__ __forceinline__ void hc_down_body(
    unsigned char* __restrict__ hc_down_smem, const uint16_t* __restrict__ h,
    const float* __restrict__ w, const void* __restrict__ down,
    const uint16_t* __restrict__ injw, uint16_t* __restrict__ d, const float* __restrict__ sc,
    uint16_t* __restrict__ inj, int M, int n, int hc, int lr, int h_ld, int injw_ld, int inj_ld,
    int f8, int down_ld, int down_stride, int down_G, float wadd) {
    constexpr int TILE = 32 * P, ROWS = 8 * Q;
    static_assert(TILE * ROWS == kHcThreads * P * Q, "a down lane owns P outputs by Q rows");
    constexpr int kPitch = CHUNK + 1;
    constexpr int kNPitch = CHUNK + 8;   /* 16-byte pad per N row shifts the bank */
    constexpr int kStage = TILE * kPitch;
    const int r0 = (int)blockIdx.x * ROWS;
    if (r0 >= M) return;
    const int rv = min(ROWS, M - r0);   /* the tile's rows that exist */
    const long long HN = (long long)hc * n;
    float* stage = (float*)hc_down_smem;
    uint16_t* Nb = (uint16_t*)(stage + kStage);
    const int t = (int)threadIdx.x;
    const int lane = t & 31, wave = t >> 5;
    const int ntiles = (lr + TILE - 1) / TILE;
    const bool inj_block = (int)blockIdx.y == ntiles;
    const int j0 = (int)blockIdx.y * TILE;
    const int nr = inj_block ? 0 : min(TILE, lr - j0);
    MixW dw{down, f8, down_ld, down_stride, (int)HN, down_G};
    /* Every chunk is whole when the stream width divides it, which is the served shape; the
     * fallback keeps the per-element stream lookup for a width that does not. */
    const bool whole = (n % CHUNK) == 0;
    float acc[P][Q];
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
        for (int q = 0; q < Q; ++q) acc[p][q] = 0.0f;
    float iacc = 0.0f;
    /* The stream a chunk belongs to, advanced once a chunk boundary crosses it: a division by
     * `n` per chunk instead of one per staged element. */
    int cstream = 0;
    long long stream_end = n;
    for (long long i0 = 0; i0 < HN; i0 += CHUNK) {
        const int cn = (int)min((long long)CHUNK, HN - i0);
        if (whole && i0 >= stream_end) {
            ++cstream;
            stream_end += n;
        }
        /* N of the tile's rows, one chunk at a time: the whole-row fill's expression, narrowed
         * once. Rows past `rv` (and so past M) are neither staged nor stored; the dot's lanes past
         * them read whatever the stage holds and drop the value at the store guard. */
        if (whole) {
            const int c = cstream;
            for (int idx = t; idx < rv * CHUNK; idx += kHcThreads) {
                const int r = idx / CHUNK, ii = idx - r * CHUNK;
                const int row = r0 + r;
                const long long j = i0 + ii;
                float v = 0.0f;
                if (row < M)
                    v = bf16_to_f32(h[(long long)row * h_ld + j]) *
                        sc[(long long)row * hc + c] * (w[j] + wadd);
                Nb[r * kNPitch + ii] = f32_to_bf16(v);
            }
        } else {
            for (int idx = t; idx < rv * cn; idx += kHcThreads) {
                const int r = idx / cn, ii = idx - r * cn;
                const int row = r0 + r;
                float v = 0.0f;
                if (row < M) {
                    const long long j = i0 + ii;
                    v = bf16_to_f32(h[(long long)row * h_ld + j]) *
                        sc[(long long)row * hc + (int)(j / n)] * (w[j] + wadd);
                }
                Nb[r * kNPitch + ii] = f32_to_bf16(v);
            }
        }
        if (inj_block) {
            /* The injection gains are one dot a stream and their own workgroup: behind a tile's
             * dot they would serialise two full-length chains and set the launch's time. */
            if (whole) {
                for (int idx = t; idx < hc * CHUNK; idx += kHcThreads) {
                    const int r = idx / CHUNK, ii = idx - r * CHUNK;
                    stage[r * kPitch + ii] = bf16_to_f32(injw[(long long)r * injw_ld + i0 + ii]);
                }
            } else {
                for (int idx = t; idx < hc * cn; idx += kHcThreads) {
                    const int r = idx / cn, ii = idx - r * cn;
                    stage[r * kPitch + ii] = bf16_to_f32(injw[(long long)r * injw_ld + i0 + ii]);
                }
            }
        } else if (f8 && cn == CHUNK && down_G % 16 == 0) {
            /* Sixteen codes a lane, a 16-byte vector a load, with the group's scale beside it:
             * the slice is one coalesced pass instead of TILE lanes on TILE rows at one offset. */
            const int gpr = CHUNK / 16;
            for (int c = t; c < nr * gpr; c += kHcThreads) {
                const int r = c / gpr, g = c - r * gpr;
                const uint8_t* wr = (const uint8_t*)down + (long long)(j0 + r) * down_stride;
                const uint4 codes = *(const uint4*)(wr + i0 + g * 16);
                const float scale = ((const float*)(wr + HN))[(i0 + (long long)g * 16) / down_G];
                const uint32_t* cw = &codes.x;
                float* st = stage + r * kPitch + g * 16;
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const uint32_t v = cw[u];
                    #pragma unroll
                    for (int b = 0; b < 4; ++b)
                        st[u * 4 + b] = e4m3_to_f32((uint8_t)(v >> (8 * b))) * scale;
                }
            }
        } else {
            for (int idx = t; idx < nr * cn; idx += kHcThreads) {
                const int r = idx / cn, ii = idx - r * cn;
                stage[r * kPitch + ii] = mix_get(dw, j0 + r, i0 + ii);
            }
        }
        __syncthreads();
        if (inj_block) {
            if (t < hc * ROWS) {
                const int r = t / hc, c = t - r * hc;
                const uint16_t* nb = Nb + r * kNPitch;
                const float* st = stage + c * kPitch;
                float a = iacc;
                if (cn == CHUNK) {
                    #pragma unroll 16
                    for (int ii = 0; ii < CHUNK; ++ii) a += bf16_to_f32(nb[ii]) * st[ii];
                } else {
                    for (int ii = 0; ii < cn; ++ii) a += bf16_to_f32(nb[ii]) * st[ii];
                }
                iacc = a;
            }
        } else {
            /* A partial output tile still walks every lane's P columns; the rows past `nr` hold
             * whatever the stage left there and are never stored. */
            const uint16_t* np[Q];
            const float* wp[P];
#pragma unroll
            for (int q = 0; q < Q; ++q) np[q] = Nb + (wave * Q + q) * kNPitch;
#pragma unroll
            for (int p = 0; p < P; ++p) wp[p] = stage + (lane + 32 * p) * kPitch;
            if (cn == CHUNK) {
                /* Four MACs' products are formed first, so the loads of one group overlap the
                 * dependent adds of the previous one; the adds themselves keep the reference's
                 * ascending order. */
                for (int ii = 0; ii < CHUNK; ii += 4) {
                    float prod[P][Q][4];
#pragma unroll
                    for (int e = 0; e < 4; ++e)
#pragma unroll
                        for (int p = 0; p < P; ++p)
#pragma unroll
                            for (int q = 0; q < Q; ++q)
                                prod[p][q][e] = bf16_to_f32(np[q][ii + e]) * wp[p][ii + e];
#pragma unroll
                    for (int e = 0; e < 4; ++e)
#pragma unroll
                        for (int p = 0; p < P; ++p)
#pragma unroll
                            for (int q = 0; q < Q; ++q) acc[p][q] += prod[p][q][e];
                }
            } else {
                for (int ii = 0; ii < cn; ++ii)
#pragma unroll
                    for (int p = 0; p < P; ++p)
#pragma unroll
                        for (int q = 0; q < Q; ++q) acc[p][q] += bf16_to_f32(np[q][ii]) * wp[p][ii];
            }
        }
        __syncthreads();
    }
    if (inj_block) {
        if (injw && inj && t < hc * ROWS) {
            const int r = t / hc, c = t - r * hc;
            const int row = r0 + r;
            if (row < M)
                inj[(long long)row * inj_ld + c] =
                    f32_to_bf16(2.0f * r3d_hcple::sigmoid(iacc / (float)hc));
        }
        return;
    }
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
        for (int q = 0; q < Q; ++q)
            if (lane + 32 * p < nr && r0 + wave * Q + q < M)
                d[(long long)(r0 + wave * Q + q) * lr + j0 + lane + 32 * p] =
                    f32_to_bf16(r3d_hcple::silu(acc[p][q] / (float)hc));
}

/* A template-id cannot sit in R3D_KLAUNCH's first macro argument (its commas split the
 * arguments), so the tiled body is a device function and these kernels name it. */
#define R3D_HC_DOWN_ARGS                                                                          \
    const uint16_t* __restrict__ h, const float* __restrict__ w, const void* __restrict__ down,   \
        const uint16_t* __restrict__ injw, uint16_t* __restrict__ d,                              \
        const float* __restrict__ sc, uint16_t* __restrict__ inj, int M, int n, int hc, int lr,   \
        int h_ld, int injw_ld, int inj_ld, int f8, int down_ld, int down_stride, int down_G,      \
        float wadd
#define R3D_HC_DOWN_CALL                                                                          \
    h, w, down, injw, d, sc, inj, M, n, hc, lr, h_ld, injw_ld, inj_ld, f8, down_ld, down_stride,  \
        down_G, wadd
#define R3D_HC_DOWN_KERNEL(NAME, P_, Q_, CH_)                                                     \
    __global__ __launch_bounds__(kHcThreads) void NAME(R3D_HC_DOWN_ARGS) {                        \
        extern __shared__ __attribute__((aligned(16))) unsigned char smem[];                      \
        hc_down_body<P_, Q_, CH_>(smem, R3D_HC_DOWN_CALL);                                        \
    }
R3D_HC_DOWN_KERNEL(hc_down_kernel_p1q2c128, 1, 2, 128)
constexpr size_t kHcDownLdsP1Q2C128 = hc_down_lds_bytes<1, 2, 128>();
R3D_HC_DOWN_KERNEL(hc_down_kernel_p2q2c64, 2, 2, 64)
constexpr size_t kHcDownLdsP2Q2C64 = hc_down_lds_bytes<2, 2, 64>();
#undef R3D_HC_DOWN_KERNEL

/* W28's down, unchanged: one row a block, the whole of N in LDS, 16 rows by 256 columns a slice.
 * It is the decode shape (M below one row tile). */
static inline size_t hc_down_w28_lds_bytes(long long HN) {
    return (size_t)kHcW28DownStage * sizeof(float) + (size_t)HN * sizeof(uint16_t);
}

__global__ __launch_bounds__(kHcThreads) void hc_down_kernel_w28(R3D_HC_DOWN_ARGS) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long HN = (long long)hc * n;
    extern __shared__ __attribute__((aligned(16))) unsigned char hc_down_smem[];
    float* stage = (float*)hc_down_smem;
    uint16_t* Nb = (uint16_t*)(stage + kHcW28DownStage);
    const float* scr = sc + (long long)row * hc;
    const uint16_t* hr = h + (long long)row * h_ld;
    const int t = (int)threadIdx.x;
    for (long long j = t; j < HN; j += kHcThreads) {
        const int c = (int)(j / n);
        Nb[j] = f32_to_bf16(bf16_to_f32(hr[j]) * scr[c] * (w[j] + wadd));
    }
    __syncthreads();

    /* The injection gains are one dot a stream and their own workgroup: behind a tile's dot they
     * would serialise two full-length chains and set the launch's time. */
    const int ntiles = (lr + kHcW28DownTile - 1) / kHcW28DownTile;
    if ((int)blockIdx.y == ntiles) {
        float acc = 0.0f;
        for (long long i0 = 0; i0 < HN; i0 += kHcW28DownChunk) {
            const int cn = (int)min((long long)kHcW28DownChunk, HN - i0);
            for (int idx = t; idx < cn * hc; idx += kHcThreads) {
                const int r = idx / cn, ii = idx - r * cn;
                stage[r * kHcW28DownPitch + ii] =
                    bf16_to_f32(injw[(long long)r * injw_ld + i0 + ii]);
            }
            __syncthreads();
            if (t < hc) {
                float a = acc;
                const float* st = stage + t * kHcW28DownPitch;
                if (cn == kHcW28DownChunk) {
                    #pragma unroll 16
                    for (int ii = 0; ii < kHcW28DownChunk; ++ii)
                        a += bf16_to_f32(Nb[i0 + ii]) * st[ii];
                } else {
                    for (int ii = 0; ii < cn; ++ii) a += bf16_to_f32(Nb[i0 + ii]) * st[ii];
                }
                acc = a;
            }
            __syncthreads();
        }
        if (injw && inj && t < hc) {
            inj[(long long)row * inj_ld + t] =
                f32_to_bf16(2.0f * r3d_hcple::sigmoid(acc / (float)hc));
        }
        return;
    }

    const int j0 = (int)blockIdx.y * kHcW28DownTile;
    const int nr = min(j0 + kHcW28DownTile, lr) - j0;
    MixW dw{down, f8, down_ld, down_stride, (int)HN, down_G};
    float acc = 0.0f;
    for (int i0 = 0; i0 < HN; i0 += kHcW28DownChunk) {
        const int cn = (int)min((long long)kHcW28DownChunk, HN - i0);
        if (f8 && cn == kHcW28DownChunk && down_G % 16 == 0) {
            const int gpr = kHcW28DownChunk / 16;
            for (int c = t; c < nr * gpr; c += kHcThreads) {
                const int r = c / gpr, g = c - r * gpr;
                const uint8_t* wr = (const uint8_t*)down + (long long)(j0 + r) * down_stride;
                const uint4 codes = *(const uint4*)(wr + i0 + g * 16);
                const float scale = ((const float*)(wr + HN))[(i0 + g * 16) / down_G];
                const uint32_t* cw = &codes.x;
                float* st = stage + r * kHcW28DownPitch + g * 16;
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const uint32_t v = cw[u];
                    #pragma unroll
                    for (int b = 0; b < 4; ++b)
                        st[u * 4 + b] = e4m3_to_f32((uint8_t)(v >> (8 * b))) * scale;
                }
            }
        } else {
            for (int idx = t; idx < nr * cn; idx += kHcThreads) {
                const int r = idx / cn, ii = idx - r * cn;
                stage[r * kHcW28DownPitch + ii] = mix_get(dw, j0 + r, i0 + ii);
            }
        }
        __syncthreads();
        if (t < nr) {
            float a = acc;
            const float* st = stage + t * kHcW28DownPitch;
            if (cn == kHcW28DownChunk) {
                #pragma unroll 16
                for (int ii = 0; ii < kHcW28DownChunk; ++ii)
                    a += bf16_to_f32(Nb[i0 + ii]) * st[ii];
            } else {
                for (int ii = 0; ii < cn; ++ii) a += bf16_to_f32(Nb[i0 + ii]) * st[ii];
            }
            acc = a;
        }
        __syncthreads();
    }
    if (t < nr)
        d[(long long)row * lr + j0 + t] = f32_to_bf16(r3d_hcple::silu(acc / (float)hc));
}
#undef R3D_HC_DOWN_ARGS
#undef R3D_HC_DOWN_CALL

/* ---- up and x ------------------------------------------------------------------------------ */
/* LDS: TILE staged weight columns of CHUNK floats (one float of pad a row) and d for the tile's
 * rows, chunk by chunk with the same pad, so a lane's walk is sequential and its neighbours hit
 * distinct banks. A lane owns P columns and Q rows of the tile: one staged weight value feeds its
 * Q rows and one staged d value its P columns, so a MAC costs 1/P + 1/Q LDS instructions. The
 * sums over k and over the streams stay ascending, and the sigmoid's input is the same f32, so
 * the gate is bit-identical to the reference. */
template <int P, int Q, int CHUNK>
constexpr size_t hc_up_lds_bytes() {
    static_assert(32 * P * 8 * Q == kHcThreads * P * Q, "an up lane owns P columns by Q rows");
    return (size_t)(32 * P) * (CHUNK + 1) * sizeof(float) +
           (size_t)(8 * Q) * (CHUNK + 1) * sizeof(float);
}

template <int P, int Q, int CHUNK>
__device__ __forceinline__ void hc_up_body(
    unsigned char* __restrict__ hc_up_smem, const uint16_t* __restrict__ h,
    const float* __restrict__ w, const void* __restrict__ up, const uint16_t* __restrict__ d,
    const float* __restrict__ sc, void* __restrict__ x, int M, int n, int hc, int lr, int h_ld,
    int x_ld, int x_dt, int f8, int up_ld, int up_stride, int up_G, float wadd) {
    constexpr int TILE = 32 * P, ROWS = 8 * Q;
    static_assert(TILE * ROWS == kHcThreads * P * Q, "an up lane owns P columns by Q rows");
    constexpr int kPitch = CHUNK + 1;
    constexpr int kStage = TILE * kPitch;
    const int r0 = (int)blockIdx.x * ROWS;
    if (r0 >= M) return;
    const int rv = min(ROWS, M - r0);   /* the tile's rows that exist */
    const int t = (int)threadIdx.x;
    const int lane = t & 31, wave = t >> 5;
    float* stage = (float*)hc_up_smem;
    float* dsh = stage + kStage;
    /* Every thread of the block reaches every barrier, including the lanes past `n` and the
     * staging-only lanes: only the per-column work is guarded. */
    const int i0 = (int)blockIdx.y * TILE;
    const int ni = min(TILE, n - i0);
    MixW uw{up, f8, up_ld, up_stride, lr, up_G};
    float acc[P][Q];
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
        for (int q = 0; q < Q; ++q) acc[p][q] = 0.0f;
    for (int c = 0; c < hc; ++c) {
        float gv[P][Q];
#pragma unroll
        for (int p = 0; p < P; ++p)
#pragma unroll
            for (int q = 0; q < Q; ++q) gv[p][q] = 0.0f;
        for (int k0 = 0; k0 < lr; k0 += CHUNK) {
            const int kn = min(CHUNK, lr - k0);
            if (kn == CHUNK) {
                for (int idx = t; idx < rv * CHUNK; idx += kHcThreads) {
                    const int r = idx / CHUNK, kk = idx - r * CHUNK;
                    const int row = r0 + r;
                    dsh[r * kPitch + kk] =
                        row < M ? bf16_to_f32(d[(long long)row * lr + k0 + kk]) : 0.0f;
                }
            } else {
                for (int idx = t; idx < rv * kn; idx += kHcThreads) {
                    const int r = idx / kn, kk = idx - r * kn;
                    const int row = r0 + r;
                    dsh[r * kPitch + kk] =
                        row < M ? bf16_to_f32(d[(long long)row * lr + k0 + kk]) : 0.0f;
                }
            }
            if (f8 && kn == CHUNK && up_G > 0 && up_G % 16 == 0) {
                /* A 16-code vector starts on a 16-byte boundary and the scale group is a
                 * multiple of 16, so a vector never straddles two groups: one scale a vector, four
                 * lanes a row, the slice staged in one load each. */
                const int vpr = CHUNK / 16;
                for (int idx = t; idx < ni * vpr; idx += kHcThreads) {
                    const int r = idx / vpr, g = idx - r * vpr;
                    const uint8_t* wr =
                        (const uint8_t*)up + (long long)((long long)c * n + i0 + r) * up_stride;
                    const uint4 codes = *(const uint4*)(wr + k0 + g * 16);
                    const float scale = ((const float*)(wr + lr))[(k0 + g * 16) / up_G];
                    const uint32_t* cw = &codes.x;
                    float* st = stage + r * kPitch + g * 16;
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const uint32_t v = cw[u];
                        #pragma unroll
                        for (int b = 0; b < 4; ++b)
                            st[u * 4 + b] = e4m3_to_f32((uint8_t)(v >> (8 * b))) * scale;
                    }
                }
            } else {
                for (int idx = t; idx < ni * kn; idx += kHcThreads) {
                    const int r = idx / kn, kk = idx - r * kn;
                    stage[r * kPitch + kk] = mix_get(uw, (long long)c * n + i0 + r, k0 + kk);
                }
            }
            __syncthreads();
            {
                const float* wp[P];
                const float* dp[Q];
#pragma unroll
                for (int p = 0; p < P; ++p) wp[p] = stage + (lane + 32 * p) * kPitch;
#pragma unroll
                for (int q = 0; q < Q; ++q) dp[q] = dsh + (wave * Q + q) * kPitch;
                if (kn == CHUNK) {
                    for (int kk = 0; kk < CHUNK; kk += 4) {
                        float prod[P][Q][4];
#pragma unroll
                        for (int e = 0; e < 4; ++e)
#pragma unroll
                            for (int p = 0; p < P; ++p)
#pragma unroll
                                for (int q = 0; q < Q; ++q)
                                    prod[p][q][e] = dp[q][kk + e] * wp[p][kk + e];
#pragma unroll
                        for (int e = 0; e < 4; ++e)
#pragma unroll
                            for (int p = 0; p < P; ++p)
#pragma unroll
                                for (int q = 0; q < Q; ++q) gv[p][q] += prod[p][q][e];
                    }
                } else {
                    for (int kk = 0; kk < kn; ++kk)
#pragma unroll
                        for (int p = 0; p < P; ++p)
#pragma unroll
                            for (int q = 0; q < Q; ++q) gv[p][q] += dp[q][kk] * wp[p][kk];
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int p = 0; p < P; ++p)
#pragma unroll
            for (int q = 0; q < Q; ++q) {
                const int row = r0 + wave * Q + q;
                if (lane + 32 * p < ni && row < M) {
                    const long long j = (long long)c * n + i0 + lane + 32 * p;
                    const float Nj = r3d_hcple::narrow(
                        R3D_DT_BF16,
                        bf16_to_f32(h[(long long)row * h_ld + j]) * sc[(long long)row * hc + c] *
                            (w[j] + wadd));
                    acc[p][q] +=
                        r3d_hcple::narrow(R3D_DT_BF16, r3d_hcple::sigmoid(gv[p][q])) * Nj;
                }
            }
    }
#pragma unroll
    for (int p = 0; p < P; ++p)
#pragma unroll
        for (int q = 0; q < Q; ++q)
            if (lane + 32 * p < ni && r0 + wave * Q + q < M)
                r3d_hcple::st_elem(x_dt, x,
                                   (long long)(r0 + wave * Q + q) * x_ld + i0 + lane + 32 * p,
                                   acc[p][q] / (float)hc);
}

#define R3D_HC_UP_ARGS                                                                            \
    const uint16_t* __restrict__ h, const float* __restrict__ w, const void* __restrict__ up,     \
        const uint16_t* __restrict__ d, const float* __restrict__ sc, void* __restrict__ x, int M, \
        int n, int hc, int lr, int h_ld, int x_ld, int x_dt, int f8, int up_ld, int up_stride,    \
        int up_G, float wadd
#define R3D_HC_UP_CALL h, w, up, d, sc, x, M, n, hc, lr, h_ld, x_ld, x_dt, f8, up_ld, up_stride,  \
        up_G, wadd
#define R3D_HC_UP_KERNEL(NAME, P_, Q_, CH_)                                                       \
    __global__ __launch_bounds__(kHcThreads) void NAME(R3D_HC_UP_ARGS) {                          \
        extern __shared__ __attribute__((aligned(16))) unsigned char smem[];                      \
        hc_up_body<P_, Q_, CH_>(smem, R3D_HC_UP_CALL);                                            \
    }
R3D_HC_UP_KERNEL(hc_up_kernel_p2q2c64, 2, 2, 64)
#undef R3D_HC_UP_KERNEL
constexpr size_t kHcUpLdsP2Q2C64 = hc_up_lds_bytes<2, 2, 64>();

/* W28's up, unchanged: one row a block, d f32 in LDS whole, 64 columns by 32 k a slice. */
static inline size_t hc_up_w28_lds_bytes(long long lr) {
    return (((size_t)lr + 3) / 4) * 4 * sizeof(float) + (size_t)kHcW28UpStage * sizeof(float);
}

__global__ __launch_bounds__(kHcThreads) void hc_up_kernel_w28(R3D_HC_UP_ARGS) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const int t = (int)threadIdx.x;
    extern __shared__ __attribute__((aligned(16))) unsigned char hc_up_smem[];
    float* dsh = (float*)hc_up_smem;
    float* stage = (float*)(dsh + (((size_t)lr + 3) / 4) * 4);
    /* Every thread of the block reaches every barrier, including the lanes past `n` and the
     * staging-only lanes: only the per-column work is guarded. */
    for (int k = t; k < lr; k += kHcThreads)
        dsh[k] = bf16_to_f32(d[(long long)row * lr + k]);
    __syncthreads();

    const int i0 = (int)blockIdx.y * kHcW28UpCols;
    const int ni = min(kHcW28UpCols, n - i0);
    const float* scr = sc + (long long)row * hc;
    const uint16_t* hr = h + (long long)row * h_ld;
    MixW uw{up, f8, up_ld, up_stride, lr, up_G};
    float acc = 0.0f;
    for (int c = 0; c < hc; ++c) {
        float gv = 0.0f;
        for (int k0 = 0; k0 < lr; k0 += kHcW28UpChunk) {
            const int kn = min(kHcW28UpChunk, lr - k0);
            if (f8 && kn == kHcW28UpChunk && up_G > 0 && up_G % 16 == 0) {
                const int vpr = kHcW28UpChunk / 16;   /* 16-code vectors a row */
                if (t < ni * vpr) {
                    const int r = t / vpr, kk0 = (t - r * vpr) * 16;
                    const uint8_t* wr =
                        (const uint8_t*)up + (long long)((long long)c * n + i0 + r) * up_stride;
                    const uint4 codes = *(const uint4*)(wr + k0 + kk0);
                    const float scale = ((const float*)(wr + lr))[(k0 + kk0) / up_G];
                    const uint32_t* cw = &codes.x;
                    float* st = stage + r * (kHcW28UpChunk + 1) + kk0;
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const uint32_t v = cw[u];
                        #pragma unroll
                        for (int b = 0; b < 4; ++b)
                            st[u * 4 + b] = e4m3_to_f32((uint8_t)(v >> (8 * b))) * scale;
                    }
                }
            } else {
                for (int idx = t; idx < ni * kn; idx += kHcThreads) {
                    const int r = idx / kn, kk = idx - r * kn;
                    stage[r * (kHcW28UpChunk + 1) + kk] =
                        mix_get(uw, (long long)c * n + i0 + r, k0 + kk);
                }
            }
            __syncthreads();
            if (t < ni) gv = hc_dot_asc_f(gv, dsh + k0, stage + t * (kHcW28UpChunk + 1), kn);
            __syncthreads();
        }
        if (t < ni) {
            const long long j = (long long)c * n + i0 + t;
            const float Nj =
                r3d_hcple::narrow(R3D_DT_BF16, bf16_to_f32(hr[j]) * scr[c] * (w[j] + wadd));
            acc += r3d_hcple::narrow(R3D_DT_BF16, r3d_hcple::sigmoid(gv)) * Nj;
        }
    }
    if (t < ni) r3d_hcple::st_elem(x_dt, x, (long long)row * x_ld + i0 + t, acc / (float)hc);
}
#undef R3D_HC_UP_ARGS
#undef R3D_HC_UP_CALL

/* ---- q and rq ------------------------------------------------------------------------------ */
/* One workgroup a (row, group) pair, one thread a value: the amax is a block reduction and the
 * rotation's butterflies are spread over the block stage by stage. The stage order and every
 * butterfly are the reference's, so the codes and the scale are bit-identical; only the order in
 * which independent butterflies run changes. */
constexpr int kHcQuantThreads = 128;

__global__ __launch_bounds__(kHcQuantThreads) void hc_quant_kernel(
    const void* __restrict__ x, int x_dt, int M, int n, int x_ld, int qg, int ngrp, void* __restrict__ q,
    float* __restrict__ qs, int q_dt, int q_ld, int qs_ld, int rg, int rgrp, void* __restrict__ rq,
    float* __restrict__ rs, int rq_dt, int rq_ld, int rs_ld) {
    const int row = (int)blockIdx.x;
    const int gi = (int)blockIdx.y;
    if (row >= M) return;
    __shared__ float s[kHcQuantScratch];
    __shared__ float red[kHcQuantThreads];
    const int t = (int)threadIdx.x;

    if (gi < ngrp) {
        const long long k0 = (long long)gi * qg;
        const int kn = (int)min((long long)qg, (long long)n - k0);
        const long long base = (long long)row * x_ld + k0;
        float amax = 0.0f;
        for (int i = t; i < kn; i += kHcQuantThreads)
            amax = fmaxf(amax, fabsf(r3d_hcple::ld_elem(x_dt, x, base + i)));
        red[t] = amax;
        __syncthreads();
        for (int stp = kHcQuantThreads / 2; stp > 0; stp >>= 1) {
            if (t < stp) red[t] = fmaxf(red[t], red[t + stp]);
            __syncthreads();
        }
        const float top = q_dt == R3D_DT_I8 ? 127.0f : 448.0f;
        const float scale = red[0] > 0.0f ? red[0] * (1.0f / top) : 1.0f;
        const float inv = 1.0f / scale;
        for (int i = t; i < kn; i += kHcQuantThreads) {
            const float v = r3d_hcple::ld_elem(x_dt, x, base + i) * inv;
            const float c = fminf(fmaxf(v, -top), top);
            r3d_hcple::st_elem(q_dt, q, (long long)row * q_ld + k0 + i, c);
        }
        if (t == 0) qs[(long long)row * qs_ld + gi] = scale;
        __syncthreads();
    }

    if (gi < rgrp) {
        const long long k0 = (long long)gi * rg;
        const long long base = (long long)row * x_ld + k0;
        for (int i = t; i < rg; i += kHcQuantThreads)
            s[i] = r3d_hcple::ld_elem(x_dt, x, base + i);
        __syncthreads();
        const int half = rg >> 1;
        for (int h = 1; h < rg; h <<= 1) {
            for (int bi = t; bi < half; bi += kHcQuantThreads) {
                const int b = (bi / h) * (h << 1);
                const int j = b + (bi - (bi / h) * h);
                const float a = s[j], c = s[j + h];
                s[j] = a + c;
                s[j + h] = a - c;
            }
            __syncthreads();
        }
        float amax = 0.0f;
        for (int i = t; i < rg; i += kHcQuantThreads) amax = fmaxf(amax, fabsf(s[i]));
        red[t] = amax;
        __syncthreads();
        for (int stp = kHcQuantThreads / 2; stp > 0; stp >>= 1) {
            if (t < stp) red[t] = fmaxf(red[t], red[t + stp]);
            __syncthreads();
        }
        const float top = rq_dt == R3D_DT_I8 ? 127.0f : 448.0f;
        const float scale = red[0] > 0.0f ? red[0] * (1.0f / top) : 1.0f;
        const float inv = 1.0f / scale;
        for (int i = t; i < rg; i += kHcQuantThreads) {
            const float v = fminf(fmaxf(s[i] * inv, -top), top);
            r3d_hcple::st_elem(rq_dt, rq, (long long)row * rq_ld + k0 + i, v);
        }
        if (t == 0) rs[(long long)row * rs_ld + gi] = scale;
    }
}

/* ---- the write half ------------------------------------------------------------------------ */
__global__ __launch_bounds__(kHcThreads) void hc_write_kernel(
    const uint16_t* __restrict__ y, const uint16_t* __restrict__ inj, uint16_t* __restrict__ h,
    int M, int n, int hc, int y_ld, int inj_ld, int h_ld) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long HN = (long long)hc * n;
    const uint16_t* yr = y + (long long)row * y_ld;
    const uint16_t* ir = inj + (long long)row * inj_ld;
    uint16_t* hr = h + (long long)row * h_ld;
    for (long long j = threadIdx.x; j < HN; j += kHcThreads) {
        const int c = (int)(j / n);
        const int i = (int)(j - (long long)c * n);
        const float hv = bf16_to_f32(hr[j]);
        const float gain = bf16_to_f32(ir[c]);
        const float yv = bf16_to_f32(yr[i]);
        hr[j] = f32_to_bf16(hv + gain * yv);
    }
}

/* ---- enter --------------------------------------------------------------------------------- */
__global__ __launch_bounds__(kHcThreads) void hc_enter_kernel(
    const void* __restrict__ x, int x_dt, int x_ld, void* __restrict__ h, int h_dt, int h_ld,
    int M, int n, int hc) {
    const int row = (int)blockIdx.x;
    if (row >= M) return;
    const long long HN = (long long)hc * n;
    for (long long j = threadIdx.x; j < HN; j += kHcThreads) {
        const int i = (int)(j % n);
        r3d_hcple::st_elem(h_dt, h, (long long)row * h_ld + j,
                           r3d_hcple::ld_elem(x_dt, x, (long long)row * x_ld + i));
    }
}

/* ---- project ------------------------------------------------------------------------------- */
__global__ __launch_bounds__(kHcThreads) void hc_project_kernel(
    uint16_t* __restrict__ h, const float* __restrict__ v, const int32_t* __restrict__ on,
    int M, int n, int hc, int h_ld, int on_ld, float scale) {
    const int row = (int)blockIdx.x;
    const int c = (int)blockIdx.y;
    if (row >= M) return;
    if (on && on[(long long)row * on_ld] == 0) return;
    __shared__ float g;
    const long long base = (long long)c * n;
    uint16_t* hr = h + (long long)row * h_ld;
    if (threadIdx.x == 0) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i)
            acc += (double)bf16_to_f32(hr[base + i]) * (double)v[i];
        g = (float)(acc * (double)scale);
    }
    __syncthreads();
    for (int i = (int)threadIdx.x; i < n; i += kHcThreads)
        hr[base + i] = f32_to_bf16(bf16_to_f32(hr[base + i]) - g * v[i]);
}

}  // namespace

extern "C" {

int64_t r3d_hc_read_scratch_bytes(int M, int lowrank, int hc) {
    if (M <= 0 || lowrank <= 0 || hc <= 0) return -1;
    return hc_scratch_bytes(M, lowrank, hc);
}

int r3d_hc_read(long h, long w, long down, long up, long injw, long x, long inj, long q, long qs,
                long rq, long rs, long scratch, long long scratch_bytes, int M, int n, int hc,
                int lr, float eps, float wadd, int group, int h_ld, int x_ld, int inj_ld,
                int injw_ld, int q_ld, int qs_ld, int rq_ld, int rs_ld, int mix_e4m3, int q_dt, int rq_dt,
                int x_dt, int down_ld, int down_stride, int down_G, int up_ld, int up_stride,
                int up_G, long stream) {
    if (!h || !w || !down || !up || !x || !scratch) return -1;
    if (M <= 0 || n <= 0 || hc <= 0 || lr <= 0 || h_ld < hc * n || x_ld < n) return -2;
    if (hc > kHcThreads) return -2;
    if (scratch_bytes < hc_scratch_bytes(M, lr, hc)) return -2;

    const long long sc_off = hc_sc_off(M, lr);
    uint16_t* d = (uint16_t*)scratch;
    float* sc = (float*)((char*)scratch + sc_off);

    const int norm_rows = kHcThreads / hc;
    R3D_KLAUNCH(hc_norm_kernel, dim3((uint32_t)((M + norm_rows - 1) / norm_rows)),
                dim3(kHcThreads), (hipStream_t)stream, (const uint16_t*)h, (const float*)w, sc, M,
                n, hc, h_ld, eps);

    /* A launch narrower than a row tile is decode: the tiled kernels' spare rows would be lanes
     * doing a dot nothing reads, so the W28 one-row-a-block kernels keep that latency. */
    const long long HN = (long long)hc * n;
    const int inj_y = (injw && inj) ? 1 : 0;
    if (M < kHcDownRows) {
        const int tiles = (lr + kHcW28DownTile - 1) / kHcW28DownTile;
        R3D_KLAUNCH_S(hc_down_kernel_w28, dim3((uint32_t)M, (uint32_t)(tiles + inj_y)),
                      dim3(kHcThreads), hc_down_w28_lds_bytes(HN), (hipStream_t)stream,
                      (const uint16_t*)h, (const float*)w, (const void*)down,
                      injw ? (const uint16_t*)injw : nullptr, d, sc,
                      inj ? (uint16_t*)inj : nullptr, M, n, hc, lr, h_ld, injw ? injw_ld : 0,
                      inj ? inj_ld : 0, mix_e4m3, down_ld, down_stride, down_G, wadd);
    } else if (M >= kHcDownWide) {
        /* Two columns a lane over a 64-output tile: one staged weight feeds two rows and the
         * slice is 64 deep, so LDS and the 80-VGPR allocation still hold three workgroups a CU.
         * From 256 rows this measures equal to the narrow tile and 5% ahead at 512, 8-9% at 2k
         * and 4k. */
        const int tiles = (lr + kHcDownTileWide - 1) / kHcDownTileWide;
        R3D_KLAUNCH_S(hc_down_kernel_p2q2c64,
                      dim3((uint32_t)((M + 15) / 16), (uint32_t)(tiles + inj_y)),
                      dim3(kHcThreads), kHcDownLdsP2Q2C64, (hipStream_t)stream,
                      (const uint16_t*)h, (const float*)w, (const void*)down,
                      injw ? (const uint16_t*)injw : nullptr, d, sc,
                      inj ? (uint16_t*)inj : nullptr, M, n, hc, lr, h_ld, injw ? injw_ld : 0,
                      inj ? inj_ld : 0, mix_e4m3, down_ld, down_stride, down_G, wadd);
    } else {
        /* Two rows a lane: the staged weight slice feeds both, and one 32-output tile with 16
         * rows still fills the grid at 128 rows (8 tiles x 11 workgroups). A 128-output tile
         * was measured no faster: three tiles leave the last 64 columns on a half ladder. */
        const int tiles = (lr + 31) / 32;
        R3D_KLAUNCH_S(hc_down_kernel_p1q2c128,
                      dim3((uint32_t)((M + 15) / 16), (uint32_t)(tiles + inj_y)),
                      dim3(kHcThreads), kHcDownLdsP1Q2C128, (hipStream_t)stream,
                      (const uint16_t*)h, (const float*)w, (const void*)down,
                      injw ? (const uint16_t*)injw : nullptr, d, sc,
                      inj ? (uint16_t*)inj : nullptr, M, n, hc, lr, h_ld, injw ? injw_ld : 0,
                      inj ? inj_ld : 0, mix_e4m3, down_ld, down_stride, down_G, wadd);
    }

    if (M < kHcUpRows) {
        R3D_KLAUNCH_S(hc_up_kernel_w28,
                      dim3((uint32_t)M, (uint32_t)((n + kHcW28UpCols - 1) / kHcW28UpCols)),
                      dim3(kHcThreads), hc_up_w28_lds_bytes(lr), (hipStream_t)stream,
                      (const uint16_t*)h, (const float*)w, (const void*)up, d, sc, (void*)x, M, n,
                      hc, lr, h_ld, x_ld, x_dt, mix_e4m3, up_ld, up_stride, up_G, wadd);
    } else {
        const int cols = (n + 63) / 64;
        R3D_KLAUNCH_S(hc_up_kernel_p2q2c64, dim3((uint32_t)((M + 15) / 16), (uint32_t)cols),
                      dim3(kHcThreads), kHcUpLdsP2Q2C64, (hipStream_t)stream,
                      (const uint16_t*)h, (const float*)w, (const void*)up, d, sc, (void*)x, M, n,
                      hc, lr, h_ld, x_ld, x_dt, mix_e4m3, up_ld, up_stride, up_G, wadd);
    }

    if (q || rq) {
        const int qg = (group > 0 && group <= n) ? group : n;
        const int ngrp = q ? (n + qg - 1) / qg : 0;
        const int rgrp = rq ? n / (group > 0 ? group : n) : 0;
        const int gx = ngrp > rgrp ? ngrp : rgrp;
        if (gx > 0)
            R3D_KLAUNCH(hc_quant_kernel, dim3((uint32_t)M, (uint32_t)gx),
                        dim3(kHcQuantThreads),
                        (hipStream_t)stream, (const void*)x, x_dt, M, n, x_ld, qg, ngrp,
                        (void*)q, (float*)qs, q_dt, q_ld, qs_ld, rq ? (group > 0 ? group : n) : 0,
                        rgrp, (void*)rq, (float*)rs, rq_dt, rq_ld, rs_ld);
    }
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

int r3d_hc_write(long y, long inj, long h, int M, int n, int hc, int y_ld, int inj_ld, int h_ld,
                 long stream) {
    if (!y || !inj || !h) return -1;
    if (M <= 0 || n <= 0 || hc <= 0 || y_ld < n || inj_ld < hc || h_ld < hc * n) return -2;
    R3D_KLAUNCH(hc_write_kernel, dim3((uint32_t)M), dim3(kHcThreads), (hipStream_t)stream,
                (const uint16_t*)y, (const uint16_t*)inj, (uint16_t*)h, M, n, hc, y_ld, inj_ld,
                h_ld);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

int r3d_hc_enter(long x, long h, int M, int n, int hc, int x_ld, int h_ld, int x_dt, int h_dt,
                 long stream) {
    if (!x || !h) return -1;
    if (M <= 0 || n <= 0 || hc <= 0 || x_ld < n || h_ld < hc * n) return -2;
    R3D_KLAUNCH(hc_enter_kernel, dim3((uint32_t)M), dim3(kHcThreads), (hipStream_t)stream,
                (const void*)x, x_dt, x_ld, (void*)h, h_dt, h_ld, M, n, hc);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

int r3d_hc_project(long h, long v, long on, int M, int n, int hc, float scale, int h_ld, int on_ld,
                   long stream) {
    if (!h || !v) return -1;
    if (M <= 0 || n <= 0 || hc <= 0 || h_ld < hc * n) return -2;
    R3D_KLAUNCH(hc_project_kernel, dim3((uint32_t)M, (uint32_t)hc), dim3(kHcThreads),
                (hipStream_t)stream, (uint16_t*)h, (const float*)v, (const int32_t*)on, M, n, hc,
                h_ld, on_ld, scale);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

/* ---- the shims ----------------------------------------------------------------------------- */

extern "C" int64_t r3d_rad_hc_read_scratch(const RadArgs* a) {
    const long long M = r3d_geti_or(a, "M", 0);
    const long long n = r3d_geti_or(a, "n", 0);
    const long long hc = r3d_geti_or(a, "hc", 0);
    const long long lr = r3d_geti_or(a, "lowrank", 0);
    if (M <= 0 || n <= 0 || hc <= 0 || lr <= 0) return R3D_NO(RAD_E_SHAPE);
    return hc_scratch_bytes(M, lr, hc);
}

}  /* extern "C" */

/* The joined mixing weight: E4M3 codes and a f32 scale per 128 of K (down) or K/4 (up), both
 * row-major, packed into one row with the scales in its tail. The bytes and the tag are libr4d's
 * (`r4d.hc.e4m3.*`): a container baked on an RDNA4 build then loads on gfx11 without a re-pack,
 * and the two libraries' layouts are held byte-identical by construction. */
namespace {

struct HcGeom { long long rows, K, G; };

/* The role of selected plane i, and whether the selection names exactly these two roles. */
const char* hc_role(const RadEncoding* e, const int* sel, int i) {
    return (e && sel && sel[i] >= 0 && sel[i] < e->n_planes) ? e->plane[sel[i]].role : "";
}

bool hc_selects(const RadEncoding* e, const int* sel, int n) {
    return n == 2 && !std::strncmp(hc_role(e, sel, 0), "codes", RAD_ENC_STR) &&
           !std::strncmp(hc_role(e, sel, 1), "scale", RAD_ENC_STR);
}

bool hc_dims(const RadTensor* t, long long rows, long long cols) {
    return t && t->rank == 2 && t->shape[0] == rows && t->shape[1] == cols;
}

bool hc_mix_geom(const RadParam* p, int n_p, int operand, const RadEncoding* enc, const int* sel,
                 const RadTensor* pl, int n, HcGeom* g) {
    if (!enc || n != 2 || !rad_enc_is(enc, "affine")) return false;
    if (enc->n_planes != 2 || enc->plane[0].dtype != RAD_F8E4M3 || enc->plane[0].block[1] != 1)
        return false;
    const int s = rad_enc_find(enc, "scale");
    if (s < 0 || enc->plane[s].dtype != RAD_F32 || enc->plane[s].kind != RAD_PLANE_TILED)
        return false;
    const long long n_ = rad_param_geti(p, n_p, "n", 0);
    const long long hc = rad_param_geti(p, n_p, "hc", 0);
    const long long lr = rad_param_geti(p, n_p, "lowrank", 0);
    const long long HN = hc * n_;
    if (HN <= 0 || lr <= 0) return false;
    if (operand == 2) {
        g->rows = lr;
        g->K = HN;
        g->G = 128;
    } else if (operand == 3) {
        g->rows = HN;
        g->K = lr;
        g->G = lr / 4;
    } else {
        return false;
    }
    if (g->G <= 0 || g->G % 16 || g->K % g->G) return false;
    /* The down stream's scale group is 128; the up stream has one group per four lowrank
     * columns, so its length is the one that must be a multiple of 64. */
    if (operand == 2 && g->K % 128) return false;
    if (operand == 3 && g->K % 64) return false;
    if (enc->plane[s].block[0] != 1 || enc->plane[s].block[1] != g->G) return false;
    if (!hc_selects(enc, sel, n)) return false;
    if (!hc_dims(&pl[0], g->rows, g->K) || !hc_dims(&pl[1], g->rows, g->K / g->G))
        return false;
    return true;
}

}  // namespace

extern "C" int r3d_rad_layout_hc_mix(const RadParam* p, int n_p, int operand,
                                     const RadEncoding* enc, const int* sel, const RadTensor* pl,
                                     int n, RadLayout* out) {
    if (operand != 2 && operand != 3) return RAD_E_UNSUPPORTED;
    if (r3d_trace_on()) {
        static bool once = false;
        if (!once) {
            once = true;
            std::fprintf(stderr, "[r3d] layout hc mix operand=%d n=%d np=%d scheme=%s", operand, n,
                         enc ? enc->n_planes : -1, enc ? enc->scheme : "(none)");
            for (int i = 0; i < (enc ? enc->n_planes : 0); ++i)
                std::fprintf(stderr, " p%d=%s/%u/k%u/b%lld,%lld", i, enc->plane[i].role,
                             enc->plane[i].dtype, enc->plane[i].kind,
                             (long long)enc->plane[i].block[0], (long long)enc->plane[i].block[1]);
            std::fprintf(stderr, " sel=[%d,%d]", sel ? sel[0] : -9, sel ? sel[1] : -9);
            if (pl) for (int i = 0; i < 2; ++i)
                std::fprintf(stderr, " pl%d=[%lld,%lld]dt%u", i, (long long)pl[i].shape[0],
                             (long long)pl[i].shape[1], pl[i].dtype);
            std::fprintf(stderr, " params n=%lld hc=%lld lr=%lld\n",
                         (long long)rad_param_geti(p, n_p, "n", 0),
                         (long long)rad_param_geti(p, n_p, "hc", 0),
                         (long long)rad_param_geti(p, n_p, "lowrank", 0));
        }
    }
    HcGeom g;
    if (!hc_mix_geom(p, n_p, operand, enc, sel, pl, n, &g)) return RAD_E_DTYPE;
    const long long stride = ((g.K + 4 * (g.K / g.G)) + 15) / 16 * 16;
    std::memset(out, 0, sizeof(*out));
    out->tag = g.G == 128 ? "r4d.hc.e4m3.g128" : "r4d.hc.e4m3.q4";
    out->dtype = RAD_F8E4M3;
    out->rank = 2;
    out->shape[0] = g.rows;
    out->shape[1] = stride;
    out->align = 16;
    out->bytes = g.rows * stride;
    return RAD_OK;
}

extern "C" int r3d_rad_relayout_hc_mix(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, void* dst,
                                       int64_t dst_bytes) {
    if (operand != 2 && operand != 3) return RAD_E_UNSUPPORTED;
    HcGeom g;
    if (!hc_mix_geom(p, n_p, operand, enc, sel, pl, n, &g)) return RAD_E_DTYPE;
    const long long stride = ((g.K + 4 * (g.K / g.G)) + 15) / 16 * 16;
    const long long ng = g.K / g.G;
    if (dst_bytes != g.rows * stride || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* codes = (const uint8_t*)pl[0].data;
    const float* sc = (const float*)pl[1].data;
    uint8_t* d = (uint8_t*)dst;
    const long long used = g.K + 4 * ng;
    for (long long r = 0; r < g.rows; ++r) {
        uint8_t* row = d + r * stride;
        std::memcpy(row, codes + r * g.K, (size_t)g.K);
        std::memcpy(row + g.K, sc + r * ng, (size_t)(4 * ng));
        if (stride > used) std::memset(row + used, 0, (size_t)(stride - used));
    }
    return RAD_OK;
}

extern "C" int r3d_rad_unrelayout_hc_mix(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel, const void* src,
                                         int64_t src_bytes, const RadTensor* pl, int n) {
    if (operand != 2 && operand != 3) return RAD_E_UNSUPPORTED;
    HcGeom g;
    if (!hc_mix_geom(p, n_p, operand, enc, sel, pl, n, &g)) return RAD_E_DTYPE;
    const long long stride = ((g.K + 4 * (g.K / g.G)) + 15) / 16 * 16;
    const long long ng = g.K / g.G;
    if (src_bytes != g.rows * stride || !pl[0].data || !pl[1].data) return RAD_E_SHAPE;
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* codes = (uint8_t*)pl[0].data;
    float* sc = (float*)pl[1].data;
    for (long long r = 0; r < g.rows; ++r) {
        const uint8_t* row = s + r * stride;
        std::memcpy(codes + r * g.K, row, (size_t)g.K);
        std::memcpy(sc + r * ng, row + g.K, (size_t)(4 * ng));
    }
    return RAD_OK;
}

extern "C" int r3d_rad_hc_read(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 11) return R3D_NO(RAD_E_INVAL);
    const RadTensor* h = r3d_opt(a, 0), *w = r3d_opt(a, 1), *down = r3d_opt(a, 2),
                    *up = r3d_opt(a, 3), *injw = r3d_opt(a, 4), *x = r3d_opt(a, 5),
                    *inj = r3d_opt(a, 6), *q = r3d_opt(a, 7), *qs = r3d_opt(a, 8),
                    *rq = r3d_opt(a, 9), *rs = r3d_opt(a, 10);
    if (!r3d_p(h) || !r3d_p(w) || !r3d_p(down) || !r3d_p(up) || !r3d_wp(x))
        return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(h) || !r3d_is_f32(w)) return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(w) || !r3d_contig(h)) return R3D_NO(RAD_E_STRIDE);
    const int x_dt = r3d_dt_code(x->dtype);
    if (x_dt != R3D_DT_BF16 && x_dt != R3D_DT_F16 && x_dt != R3D_DT_F32)
        return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(inj) && !r3d_is_bf16(inj)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(injw) && !r3d_is_bf16(injw)) return R3D_NO(RAD_E_DTYPE);

    const long long n = r3d_geti_or(a, "n", 0);
    const long long hc = r3d_geti_or(a, "hc", 0);
    const long long lr = r3d_geti_or(a, "lowrank", 0);
    if (n <= 0 || hc <= 1 || lr <= 0) return R3D_NO(RAD_E_SHAPE);
    if (hc > 8) return R3D_NO(RAD_E_UNSUPPORTED);
    const long long HN = hc * n;
    if (HN > 12288 || lr > 12288) return R3D_NO(RAD_E_UNSUPPORTED);   /* one 48 KiB LDS stage */

    const char* mix = r3d_gets(a, "mix");
    bool mix_e4m3 = false;
    if (mix) {
        if (!std::strcmp(mix, "e4m3")) mix_e4m3 = true;
        else if (std::strcmp(mix, "bf16")) return R3D_NO(RAD_E_INVAL);
    }

    long long M = 0;
    const long long h_ld = r3d_rows_pitch(h, HN, &M);
    if (h_ld < 0 || M <= 0) return R3D_NO(RAD_E_STRIDE);
    long long Mx = 0;
    const long long x_ld = r3d_rows_pitch(x, n, &Mx);
    if (x_ld < 0 || Mx != M) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_contig(x)) return R3D_NO(RAD_E_STRIDE);
    if (r3d_numel(w) < HN) return R3D_NO(RAD_E_SHAPE);

    long long inj_ld = 0;
    if (r3d_has(inj)) {
        long long Mi = 0;
        inj_ld = r3d_rows_pitch(inj, hc, &Mi);
        if (inj_ld < 0 || Mi != M) return R3D_NO(RAD_E_SHAPE);
        if (!r3d_contig(inj)) return R3D_NO(RAD_E_STRIDE);
    }
    long long down_ld = HN, down_stride = 0, down_G = 0;
    if (mix_e4m3) {
        if (!r3d_is_e4m3(down) || !r3d_is_e4m3(up)) return R3D_NO(RAD_E_DTYPE);
        /* The core binds a re-laid weight with its declared (logical) shape: the joined row
         * stride is the kernel's, from n/hc/lowrank, and the load's layout tag holds the bytes to
         * it. Numel would be the declared codes count, not the joined row, so the geometry test is
         * the declared mix-matrix extent. */
        if (!hc_dims(down, lr, HN) || !hc_dims(up, HN, lr)) return R3D_NO(RAD_E_SHAPE);
        if (HN % 128 || lr % 64) return R3D_NO(RAD_E_SHAPE);
        down_G = 128;
        down_stride = (int)(((HN + 4 * (HN / 128)) + 15) / 16 * 16);
    } else {
        if (!r3d_is_bf16(down) || !r3d_is_bf16(up)) return R3D_NO(RAD_E_DTYPE);
        if (!r3d_contig(down) || !r3d_contig(up)) return R3D_NO(RAD_E_STRIDE);
        long long dr = 0;
        down_ld = (int)r3d_rows_pitch(down, HN, &dr);
        if (down_ld < 0 || dr != lr) return R3D_NO(RAD_E_SHAPE);
    }
    long long up_ld = lr, up_stride = 0, up_G = 0;
    if (mix_e4m3) {
        up_G = (int)(lr / 4);
        up_stride = (int)(((lr + 4 * (lr / up_G)) + 15) / 16 * 16);
    } else {
        long long ur = 0;
        up_ld = (int)r3d_rows_pitch(up, lr, &ur);
        if (up_ld < 0 || ur != HN) return R3D_NO(RAD_E_SHAPE);
    }

    long long injw_ld = 0;
    if (r3d_has(injw)) {
        long long ir = 0;
        injw_ld = (int)r3d_rows_pitch(injw, HN, &ir);
        if (injw_ld < 0 || ir != hc) return R3D_NO(RAD_E_SHAPE);
        if (!r3d_contig(injw)) return R3D_NO(RAD_E_STRIDE);
        /* A weight without an output would compute scalars nobody reads; the reference refuses
         * the pair. */
        if (!r3d_has(inj)) return R3D_NO(RAD_E_SHAPE);
    }

    const long long group = r3d_geti_or(a, "group", 0);
    long long q_ld = 0, qs_ld = 0, rq_ld = 0, rs_ld = 0, qg = n, ngrp = 0, rgrp = 0;
    int q_dt = R3D_DT_U8, rq_dt = R3D_DT_U8;
    if (r3d_has(q) || r3d_has(qs)) {
        if (!r3d_has(q) || !r3d_has(qs)) return R3D_NO(RAD_E_INVAL);
        q_dt = r3d_dt_code(q->dtype);
        if (q_dt != R3D_DT_U8 && q_dt != R3D_DT_I8) return R3D_NO(RAD_E_DTYPE);
        if (!r3d_is_f32(qs)) return R3D_NO(RAD_E_DTYPE);
        qg = (group > 0 && group <= n) ? group : n;
        ngrp = (n + qg - 1) / qg;
        long long Mq = 0, Ms = 0;
        q_ld = r3d_rows_pitch(q, n, &Mq);
        qs_ld = r3d_rows_pitch(qs, ngrp, &Ms);
        if (q_ld < 0 || qs_ld < 0 || Mq != M || Ms != M) return R3D_NO(RAD_E_SHAPE);
    }
    if (r3d_has(rq) || r3d_has(rs)) {
        if (!r3d_has(rq) || !r3d_has(rs)) return R3D_NO(RAD_E_INVAL);
        rq_dt = r3d_dt_code(rq->dtype);
        if (rq_dt != R3D_DT_U8 && rq_dt != R3D_DT_I8) return R3D_NO(RAD_E_DTYPE);
        if (!r3d_is_f32(rs)) return R3D_NO(RAD_E_DTYPE);
        if (group <= 0 || group > kHcQuantScratch || (group & (group - 1)) || n % group)
            return R3D_NO(RAD_E_UNSUPPORTED);
        rgrp = n / group;
        long long Mr = 0, Msr = 0;
        rq_ld = r3d_rows_pitch(rq, n, &Mr);
        rs_ld = r3d_rows_pitch(rs, rgrp, &Msr);
        if (rq_ld < 0 || rs_ld < 0 || Mr != M || Msr != M) return R3D_NO(RAD_E_SHAPE);
    }

    const long long need = hc_scratch_bytes(M, lr, hc);
    if (!a->scratch || a->scratch_bytes < need) return R3D_NO(RAD_E_SHAPE);

    const float eps = (float)r3d_getf(a, "eps", 1e-6);
    const float wadd = (float)r3d_getf(a, "wadd", 0.0);
    const int rc = r3d_hc_read((long)(uintptr_t)r3d_p(h), (long)(uintptr_t)r3d_p(w),
                               (long)(uintptr_t)r3d_p(down), (long)(uintptr_t)r3d_p(up),
                               r3d_has(injw) ? (long)(uintptr_t)r3d_p(injw) : 0L,
                               (long)(uintptr_t)r3d_wp(x),
                               r3d_has(inj) ? (long)(uintptr_t)r3d_wp(inj) : 0L,
                               r3d_has(q) ? (long)(uintptr_t)r3d_wp(q) : 0L,
                               r3d_has(qs) ? (long)(uintptr_t)r3d_wp(qs) : 0L,
                               r3d_has(rq) ? (long)(uintptr_t)r3d_wp(rq) : 0L,
                               r3d_has(rs) ? (long)(uintptr_t)r3d_wp(rs) : 0L,
                               (long)(uintptr_t)a->scratch, a->scratch_bytes, (int)M, (int)n,
                               (int)hc, (int)lr, eps, wadd, (int)group, (int)h_ld, (int)x_ld,
                               (int)inj_ld, (int)injw_ld, (int)q_ld, (int)qs_ld, (int)rq_ld,
                               (int)rs_ld, mix_e4m3 ? 1 : 0, q_dt, rq_dt, x_dt, (int)down_ld,
                               (int)down_stride, (int)down_G, (int)up_ld, (int)up_stride,
                               (int)up_G, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

extern "C" int r3d_rad_hc_write(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* y = r3d_opt(a, 0), *inj = r3d_opt(a, 1), *h = r3d_opt(a, 2);
    if (!r3d_p(y) || !r3d_p(inj) || !r3d_wp(h)) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(y) || !r3d_is_bf16(inj) || !r3d_is_bf16(h)) return R3D_NO(RAD_E_DTYPE);
    const long long n = r3d_geti_or(a, "n", 0), hc = r3d_geti_or(a, "hc", 0);
    if (n <= 0 || hc <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, My = 0, Mi = 0;
    const long long h_ld = r3d_rows_pitch(h, hc * n, &M);
    const long long y_ld = r3d_rows_pitch(y, n, &My);
    const long long inj_ld = r3d_rows_pitch(inj, hc, &Mi);
    if (h_ld < 0 || y_ld < 0 || inj_ld < 0 || M <= 0) return R3D_NO(RAD_E_STRIDE);
    if (My != M || Mi != M) return R3D_NO(RAD_E_SHAPE);
    const int rc = r3d_hc_write((long)(uintptr_t)r3d_p(y), (long)(uintptr_t)r3d_p(inj),
                                (long)(uintptr_t)r3d_wp(h), (int)M, (int)n, (int)hc, (int)y_ld,
                                (int)inj_ld, (int)h_ld, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

extern "C" int r3d_rad_hc_enter(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 2) return R3D_NO(RAD_E_INVAL);
    const RadTensor* x = r3d_opt(a, 0), *h = r3d_opt(a, 1);
    if (!r3d_p(x) || !r3d_wp(h)) return R3D_NO(RAD_E_INVAL);
    const int x_dt = r3d_dt_code(x->dtype), h_dt = r3d_dt_code(h->dtype);
    if (x_dt < 0 || h_dt != R3D_DT_BF16) return R3D_NO(RAD_E_DTYPE);
    const long long n = r3d_geti_or(a, "n", 0), hc = r3d_geti_or(a, "hc", 0);
    if (n <= 0 || hc <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0, Mx = 0;
    const long long h_ld = r3d_rows_pitch(h, hc * n, &M);
    const long long x_ld = r3d_rows_pitch(x, n, &Mx);
    if (h_ld < 0 || x_ld < 0 || M <= 0 || Mx != M) return R3D_NO(RAD_E_SHAPE);
    const int rc = r3d_hc_enter((long)(uintptr_t)r3d_p(x), (long)(uintptr_t)r3d_wp(h), (int)M,
                                (int)n, (int)hc, (int)x_ld, (int)h_ld, x_dt, h_dt, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

extern "C" int r3d_rad_hc_project(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 3) return R3D_NO(RAD_E_INVAL);
    const RadTensor* h = r3d_opt(a, 0), *v = r3d_opt(a, 1), *on = r3d_opt(a, 2);
    if (!r3d_wp(h) || !r3d_p(v)) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_bf16(h) || !r3d_is_f32(v)) return R3D_NO(RAD_E_DTYPE);
    if (r3d_has(on) && !r3d_is_i32(on)) return R3D_NO(RAD_E_DTYPE);
    const long long n = r3d_geti_or(a, "n", 0), hc = r3d_geti_or(a, "hc", 0);
    if (n <= 0 || hc <= 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long h_ld = r3d_rows_pitch(h, hc * n, &M);
    if (h_ld < 0 || M <= 0) return R3D_NO(RAD_E_STRIDE);
    if (r3d_numel(v) < n) return R3D_NO(RAD_E_SHAPE);
    long long on_ld = 1;
    if (r3d_has(on)) {
        long long Mo = 0;
        on_ld = r3d_rows_pitch(on, 1, &Mo);
        if (on_ld < 0 || Mo != M) return R3D_NO(RAD_E_SHAPE);
    }
    const float scale = (float)r3d_getf(a, "scale", 1.0);
    const int rc = r3d_hc_project((long)(uintptr_t)r3d_wp(h), (long)(uintptr_t)r3d_p(v),
                                  r3d_has(on) ? (long)(uintptr_t)r3d_p(on) : 0L, (int)M, (int)n,
                                  (int)hc, scale, (int)h_ld, (int)on_ld, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}
