/* iqr_ref.h -- the host side of the IQR tests: the canonical planes a container hands over, the
 * trellis decode those planes define, and the CPU references.
 *
 * This header is shared by tests/iqr_device_test.cpp (the dense IQR rows) and
 * tests/moe_iqr_test.cpp (the routed IQR experts). It holds no device code and no library state,
 * so a host-only test can include it without a card.
 *
 * ============================== the two references ==============================
 *
 * Both decode an expert's canonical planes to the int8 grid (docs/IQR.md, libr3d/r3d_iqr.h) and
 * then run the int8 GEMM's order: per 128-K group an exact i32 sub-sum, then
 * (float)sub * a_scale * w_scale_block, accumulated in f32 in ascending K-block order; a
 * kRowF32 weight takes its row's f32 scale after the K loop. That is the order libref's int8
 * reference states (ADR-4), not the batching the device kernels use to keep the bias in the fma,
 * so a kernel comparison allows a documented number of bf16 ulps (the tests state it).
 *
 * moe_gemm_reference adds what moe_gemm_q's grouped form owns: the expert-sorted row order, the
 * parity (or mod-4) classes, `parts` stacked output parts (part p's columns [p*fy, p*fy+fc),
 * fy = max N / parts, fc = class N / parts), a narrower class's zero fill past its width, dropped
 * slots and experts with no rows. moe_gather_reference is the token-level top-k sum, in ascending
 * slot order, as moe_gather defines it.
 */
#pragma once

#include "rad_test.h"
#include "rad_plugin.h"

#include "r3d_codecs.h"
#include "r3d_iqr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace iqr_ref {

using namespace r3d_iqr;

inline uint32_t code_dtype(uint32_t bits) {
    static const uint32_t kU[] = { 0, RAD_U1, RAD_U2, RAD_U3, RAD_U4, RAD_U5, RAD_U6, RAD_U7 };
    return bits >= 1 && bits <= 7 ? kU[bits] : 0;
}

/* One form's canonical planes, as a container hands them over. */
struct W {
    RadEncoding enc{};
    std::vector<std::vector<uint8_t>> data;
    std::vector<int64_t> rows, cols;
    int find(const char* r) const { return rad_enc_find(&enc, r); }
    RadTensor t(int i) {
        RadTensor x;
        std::memset(&x, 0, sizeof x);
        x.data = data[(size_t)i].data();
        x.dtype = enc.plane[i].dtype;
        x.rank = 2;
        x.shape[0] = rows[(size_t)i];
        x.shape[1] = cols[(size_t)i];
        rad_tensor_pack(&x);
        return x;
    }
    int64_t row_bytes(int i) const { return rad_enc_row_bytes(enc.plane[i].dtype, cols[(size_t)i]); }
    void size(int64_t n, int64_t k) {
        data.assign((size_t)enc.n_planes, {});
        rows.assign((size_t)enc.n_planes, 0);
        cols.assign((size_t)enc.n_planes, 0);
        for (int i = 0; i < enc.n_planes; ++i) {
            rad_enc_plane_dims(&enc.plane[i], n, k, &rows[(size_t)i], &cols[(size_t)i]);
            data[(size_t)i].assign((size_t)(rows[(size_t)i] * row_bytes(i)), 0);
        }
    }
};

inline int mad_value(uint32_t st, uint32_t a, uint32_t b) {
    const uint32_t h = ((st & 0xFFFFFFu) * (a & 0xFFFFFFu) + b) & 0x3F3F3F3Fu;
    return (int)((h & 0xFFu) + ((h >> 8) & 0xFFu) + ((h >> 16) & 0xFFu) + (h >> 24)) - 126;
}

/* `n` <= 32 bits of a byte array from bit `o`. */
inline uint32_t bits_at(const uint8_t* p, int64_t o, int n) {
    const uint8_t* b = p + (o >> 3);
    const int sh = (int)(o & 7);
    const int nb = (sh + n + 7) >> 3;
    uint64_t w = 0;
    for (int i = 0; i < nb; ++i) w |= (uint64_t)b[i] << (8 * i);
    return (uint32_t)((w >> sh) & ((1ull << n) - 1ull));
}

/* The canonical codes plane: row r's B-bit code for column c. */
inline uint32_t code_at(const W& w, int64_t r, int64_t c, uint32_t B) {
    const int i = w.find("codes");
    return bits_at(w.data[(size_t)i].data() + r * w.row_bytes(i), c * B, (int)B);
}

/* Row r's trellis state for column c: the L-bit window ending at r (row r lowest), tail-biting
 * over the run of T rows (the run's last span codes seed it). */
inline uint32_t canonical_state(const W& w, uint32_t B, uint32_t L, int64_t run, int64_t r,
                                int64_t c) {
    const uint32_t m = (1u << L) - 1u;
    const int64_t base = r / run * run;
    const int64_t nspan = (L + B - 1) / B;
    uint32_t acc = 0;
    for (int64_t j = 0; j < nspan; ++j)
        acc = ((acc << B) | code_at(w, base + run - nspan + j, c, B)) & m;
    for (int64_t j = 0; j <= r - base; ++j)
        acc = ((acc << B) | code_at(w, base + j, c, B)) & m;
    return acc;
}

/* The unnormalised 128-wide Hadamard, in the order r3d_fwht.h's wave transform applies it: the
 * same stride sequence 1, 2, 4 .. 64 over natural indices. */
inline void fwht128_host(float* v) {
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; ++i)
            if ((i & h) == 0) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
}

/* A class's decoded int8 weight (no tcq_rb rescale: the head's classes are off that grid). */
inline std::vector<int8_t> decode_class(const W& w, uint32_t B, uint32_t L, int64_t N, int64_t K,
                                        int64_t run = 256) {
    const uint32_t A = ((const uint32_t*)w.data[(size_t)w.find("trellis.mad")].data())[0];
    const uint32_t Bc = ((const uint32_t*)w.data[(size_t)w.find("trellis.mad")].data())[1];
    std::vector<int8_t> dec((size_t)(N * K));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const uint32_t st = canonical_state(w, B, L, run, n, k);
            dec[(size_t)(n * K + k)] = (int8_t)mad_value(st, A, Bc);
        }
    return dec;
}

/* The decoded weight, row-major [N, K]. The values are canonical_state's, one (n, k) at a time,
 * but the state advances by one code as n rises inside a run instead of restarting from the run's
 * seed: N * K steps rather than N * K * run. */
inline std::vector<int8_t> decode_rows(const W& w, uint32_t B, uint32_t L, uint32_t RB, Scale sc,
                                       int64_t N, int64_t K, const uint8_t* sb, int64_t nkb,
                                       int64_t run = 256) {
    const Form f{ kTrellisMad, B, 0, L, RB, sc, 0, 0 };
    const uint32_t A = ((const uint32_t*)(w.data[(size_t)w.find("trellis.mad")].data()))[0];
    const uint32_t Bc = ((const uint32_t*)(w.data[(size_t)w.find("trellis.mad")].data()))[1];
    std::vector<int8_t> dec((size_t)(N * K));
    const uint32_t mask = (1u << L) - 1u;
    const int64_t nspan = (L + B - 1) / B;
    for (int64_t base = 0; base < N; base += run) {
        const int64_t len = (N - base < run) ? N - base : run;
        for (int64_t k = 0; k < K; ++k) {
            uint32_t st = 0;
            for (int64_t j = 0; j < nspan; ++j)
                st = ((st << B) | code_at(w, base + run - nspan + j, k, B)) & mask;
            for (int64_t j = 0; j < len; ++j) {
                const int64_t n = base + j;
                st = ((st << B) | code_at(w, n, k, B)) & mask;
                int v = mad_value(st, A, Bc);
                if (RB) {
                    const int64_t i = tile_major(n, (uint32_t)(k / 128), nkb);
                    const float own = bf16_to_f32(*(const uint16_t*)(sb + i * 2));
                    const uint16_t* bp = (const uint16_t*)(sb + blk_off(f, N, K));
                    const float bb = bf16_to_f32(bp[(n >> RB) * nkb + k / 128]);
                    const float rq = bb > 0.0f ? own / bb : 0.0f;
                    v = (int)std::nearbyintf((float)v * rq);
                }
                dec[(size_t)(n * K + k)] = (int8_t)v;
            }
        }
    }
    return dec;
}

/* A computed trellis's canonical form, as libquant writes it (`tcq_code=mad`, one f32 scale a
 * row plus one f32 per 128-K block). `run` is the trellis run in rows (the table's `tcq_rows`). */
struct MadTrellis {
    uint32_t bits = 4, L = 16, lrb = 0;
    int64_t run = 256;
    Scale scale = kRowF32;
};

/* Build a canonical encoding and fill it from a splitmix64 stream: random codes, the hash's two
 * constants, the generated table, row scales and block factors. */
inline void make_mad_trellis(W& w, int64_t N, int64_t K, const MadTrellis& t, uint64_t seed) {
    rad_enc_clear(&w.enc);
    rad_enc_copy_str(w.enc.scheme, "trellis");
    rad_enc_add_plane(&w.enc, "codes", code_dtype(t.bits), 1, 1);
    if (t.scale == kRowF32) {
        rad_enc_add_plane(&w.enc, "scale", RAD_F32, 1, 0);
        rad_enc_add_plane(&w.enc, "scale.col", RAD_F32, 0, 128);
    } else {
        rad_enc_add_plane(&w.enc, "scale", RAD_BF16, 1, 128);
    }
    const int tab = rad_enc_add_table(&w.enc, "trellis", RAD_I8, (int64_t)1 << t.L, 1);
    w.enc.plane[tab].block[0] = t.run;
    w.enc.plane[tab].block[1] = t.lrb ? 32 : 0;
    rad_enc_add_table(&w.enc, "trellis.mad", RAD_U32, 2, 1);
    rad_enc_copy_str(w.enc.transform, "fwht128");
    w.size(N, K);

    uint64_t z = seed;
    auto rnd = [&]() {
        z += 0x9E3779B97F4A7C15ull;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    };
    for (auto& b : w.data[(size_t)w.find("codes")]) b = (uint8_t)rnd();
    const uint32_t A = (uint32_t)(rnd() & 0xFFFFFFu) | 1u, Bc = (uint32_t)rnd();
    uint32_t* ab = (uint32_t*)w.data[(size_t)w.find("trellis.mad")].data();
    ab[0] = A;
    ab[1] = Bc;
    int8_t* tabp = (int8_t*)w.data[(size_t)w.find("trellis")].data();
    for (uint32_t st = 0; st < (1u << t.L); ++st) tabp[st] = (int8_t)mad_value(st, A, Bc);
    if (t.scale == kRowF32) {
        float* rs = (float*)w.data[(size_t)w.find("scale")].data();
        for (int64_t n = 0; n < N; ++n) {
            const int mant = (int)(rnd() % 64);
            const int e = -9 - (int)(rnd() % 4);
            rs[n] = std::ldexp(1.0f + (float)mant / 64.0f, e);
        }
        float* cf = (float*)w.data[(size_t)w.find("scale.col")].data();
        for (int64_t kb = 0; kb < K / 128; ++kb) cf[kb] = std::ldexp(1.0f, (int)(rnd() % 4) - 1);
    } else {
        uint16_t* sc = (uint16_t*)w.data[(size_t)w.find("scale")].data();
        for (int64_t i = 0; i < N * (K / 128); ++i) {
            const int mant = (int)(rnd() % 64);
            const int e = -9 - (int)(rnd() % 4);
            sc[i] = f32_to_bf16(std::ldexp(1.0f + (float)mant / 64.0f, e));
        }
    }
}

/* ================================================================== the routed experts
 *
 * One expert's canonical planes and its laid-out scale buffer (the layout hook's output, whose
 * offsets are the form's). The reference decodes the canonical planes; the scale buffer is the
 * one the kernel reads, so the row/block folds are exercised in the same layout. */
struct MoeExpert {
    const W* w = nullptr;
    const uint8_t* sb = nullptr;
    int64_t N = 0, K = 0;
    MadTrellis form{};
};

struct MoeGeom {
    int64_t n_expert = 0, top_k = 1, ncls = 2, parts = 1;
    int64_t M = 0, T = 0;               /* tokens, and M * top_k sorted rows */
    int64_t N[4] = { 0, 0, 0, 0 };      /* each class's width */
    int64_t K[4] = { 0, 0, 0, 0 };      /* each class's K */
    int64_t a_group = 128;
    bool sorted_a = false;
};

/* The expert whose half-open offset range holds row i, or -1. A zero-row expert owns nothing. */
inline int64_t expert_of(const std::vector<int32_t>& off, int64_t n_expert, int64_t i) {
    if (i < 0 || (size_t)(n_expert + 1) > off.size()) return -1;
    if (i >= off[(size_t)n_expert]) return -1;
    const auto it = std::upper_bound(off.begin(), off.begin() + n_expert + 1, (int32_t)i);
    const int64_t e = (int64_t)(it - off.begin()) - 1;
    if (e < 0 || e >= n_expert) return -1;
    if ((int64_t)off[(size_t)e] > i || (int64_t)off[(size_t)e + 1] <= i) return -1;
    return e;
}

inline int64_t class_count(const MoeGeom& g, int64_t q) {
    return (g.n_expert + g.ncls - 1 - q) / g.ncls;
}

/* y[i, :] for sorted row i: a[sorted_tok[i] / top_k] (or row i at a_order=sorted) times expert
 * e(i)'s int8 weight, in the int8 grid's order. A dropped slot, a row no expert owns, and the
 * columns past a narrower class's width are zero; `live` marks the cells the GEMM computed. */
inline bool moe_gemm_reference(const std::vector<MoeExpert>& ex, const MoeGeom& g,
                               const std::vector<int8_t>& a, const std::vector<float>& as,
                               const std::vector<int32_t>& sorted,
                               const std::vector<int32_t>& off, std::vector<uint16_t>* out,
                               std::vector<uint8_t>* live_out = nullptr) {
    const int64_t ncls = g.ncls, parts = g.parts;
    if (!out || g.n_expert <= 0 || g.top_k <= 0 || (ncls != 2 && ncls != 4) || parts <= 0 ||
        parts > 255 || g.a_group <= 0 || (int64_t)ex.size() < g.n_expert ||
        (int64_t)off.size() < g.n_expert + 1 || (int64_t)sorted.size() < g.T)
        return false;
    const int64_t Arows = g.sorted_a ? g.T : g.M;
    if (g.T != g.M * g.top_k) return false;
    int64_t Nmax = 0, Kmax = 0, nsa = 0;
    for (int64_t q = 0; q < ncls; ++q) {
        if (g.N[q] < 0 || g.K[q] < 0 || g.N[q] % parts || g.K[q] % g.a_group) return false;
        Nmax = std::max(Nmax, g.N[q]);
        Kmax = std::max(Kmax, g.K[q]);
    }
    if (Nmax <= 0 || Kmax <= 0 || Kmax % g.a_group || Nmax % parts) return false;
    nsa = Kmax / g.a_group;
    if ((int64_t)a.size() < Arows * Kmax || (int64_t)as.size() < Arows * nsa) return false;

    for (int64_t e = 0; e < g.n_expert; ++e) {
        const MoeExpert& x = ex[(size_t)e];
        if (!x.w || !x.sb || x.N <= 0 || x.K <= 0 || x.K % 128 || x.form.scale != kRowF32)
            return false;
    }

    std::vector<std::vector<int8_t>> dec((size_t)g.n_expert);
    for (int64_t e = 0; e < g.n_expert; ++e) {
        if (off[(size_t)e] == off[(size_t)e + 1]) continue;
        const MoeExpert& x = ex[(size_t)e];
        const int64_t nkb = x.K / 128;
        dec[(size_t)e] = decode_rows(*x.w, x.form.bits, x.form.L, x.form.lrb, x.form.scale, x.N,
                                     x.K, x.sb, nkb, x.form.run);
    }

    out->assign((size_t)(g.T * Nmax), 0);
    if (live_out) live_out->assign((size_t)(g.T * Nmax), 0);
    const int64_t fy = Nmax / parts;

    for (int64_t i = 0; i < g.T; ++i) {
        const int64_t e = expert_of(off, g.n_expert, i);
        const int32_t slot = sorted[(size_t)i];
        if (e < 0 || slot < 0) continue;
        const int64_t src = g.sorted_a ? i : slot / g.top_k;
        if (src < 0 || src >= Arows) continue;
        const int64_t cls = e % ncls, q = e / ncls;
        if (q >= class_count(g, cls)) continue;
        const int64_t Nc = g.N[cls], Kc = g.K[cls];
        if (Nc <= 0 || Kc <= 0 || Kc % 128) continue;
        const MoeExpert& x = ex[(size_t)e];
        const Form f{ kTrellisMad, x.form.bits, 0, x.form.L, x.form.lrb, x.form.scale, 0, 0 };
        const float* cf = (const float*)(x.sb + cf_off(f, Nc, Kc));
        const float* rs = (const float*)x.sb;
        const int8_t* ad = a.data() + src * Kmax;
        const float* asd = as.data() + src * nsa;
        const std::vector<int8_t>& wd = dec[(size_t)e];
        const int64_t fc = Nc / parts;
        const int64_t nkb = Kc / 128;
        for (int64_t p = 0; p < parts; ++p) {
            for (int64_t j = 0; j < fc; ++j) {
                const int64_t n = p * fc + j;
                float acc = 0.0f;
                const int8_t* wr = wd.data() + n * Kc;
                for (int64_t kb = 0; kb < nkb; ++kb) {
                    int32_t sub = 0;
                    const int8_t* ar = ad + kb * 128;
                    const int8_t* wb = wr + kb * 128;
                    for (int64_t k = 0; k < 128; ++k) sub += (int32_t)ar[k] * (int32_t)wb[k];
                    const float t0 = (float)sub * asd[(kb * 128) / g.a_group];
                    acc += t0 * cf[kb];
                }
                acc *= rs[n];
                const int64_t col = p * fy + j;
                (*out)[(size_t)(i * Nmax + col)] = f32_to_bf16(acc);
                if (live_out) (*live_out)[(size_t)(i * Nmax + col)] = 1;
            }
        }
    }
    return true;
}

/* The token-level top-k sum (moe_gather): y[t, :] = sum over slot of gate[t, slot] * rows[slot],
 * ascending slot, every term in f32. A dropped or unowned slot contributes nothing. */
inline bool moe_gather_reference(const std::vector<uint16_t>& rows, const MoeGeom& g,
                                 const std::vector<int32_t>& sorted,
                                 const std::vector<int32_t>& off,
                                 const std::vector<float>& gate, std::vector<float>* out) {
    int64_t Nmax = 0;
    for (int64_t q = 0; q < g.ncls; ++q) Nmax = std::max(Nmax, g.N[q]);
    if (!out || Nmax <= 0 || (int64_t)rows.size() < g.T * Nmax || (int64_t)gate.size() < g.T)
        return false;
    out->assign((size_t)(g.M * Nmax), 0.0f);
    std::vector<int64_t> row_of((size_t)g.T, -1);
    for (int64_t i = 0; i < g.T; ++i) {
        const int32_t f = sorted[(size_t)i];
        if (f < 0 || f >= g.T) continue;
        if (expert_of(off, g.n_expert, i) < 0) continue;
        row_of[(size_t)f] = i;
    }
    for (int64_t m = 0; m < g.M; ++m)
        for (int64_t j = 0; j < g.top_k; ++j) {
            const int64_t slot = m * g.top_k + j;
            if (slot < 0 || slot >= g.T) continue;
            const int64_t r = row_of[(size_t)slot];
            if (r < 0) continue;
            const float w = gate[(size_t)slot];
            for (int64_t n = 0; n < Nmax; ++n)
                (*out)[(size_t)(m * Nmax + n)] +=
                    bf16_to_f32(rows[(size_t)(r * Nmax + n)]) * w;
        }
    return true;
}

}  // namespace iqr_ref
