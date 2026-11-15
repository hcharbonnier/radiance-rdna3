/* iqr_device_test.cpp -- libr3d's IQR GEMM (r3d_gemm_iqr_m16.hip) against a host reference written
 * from r3d_iqr.h, on the card.
 *
 * The reference decodes the container's canonical planes itself (the state recurrence and the
 * computed value's byte sum, a code at a time) and forms the int8 grid's group sum with the
 * reference's order: per 128-K group an exact i32 sub-sum, then (sub * a_scale) * w_scale, groups
 * ascending, the row's own factor last. The stored form is produced by the library's own layout
 * hooks, so the test also covers them end to end (their byte order is held to libr4d's by
 * iqr_parity_test). Above 64 rows the tiled row runs the int8 wave matrix, whose eight 16x16x16
 * steps of a group are one exact i32 sub-sum too, so those cases compare bit for bit as well.
 *
 * The library is dlopened, as the engine loads it: the test links only the ABI and the HIP
 * runtime.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include "r3d_codecs.h"
#include "r3d_iqr.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef RAD_R3D_SO
#define RAD_R3D_SO "libr3d.so"
#endif

namespace {

using namespace r3d_iqr;

typedef int (*LayoutFn)(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                        int, RadLayout*);
typedef int (*RelayoutFn)(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                          int, void*, int64_t);
typedef int (*LaunchFn)(const RadArgs*, RadStream);
typedef int64_t (*ScratchFn)(const RadArgs*);
typedef int (*SlicesFn)(long long, long long);

struct Lib {
    void* h = nullptr;
    LayoutFn layout = nullptr;
    RelayoutFn relayout = nullptr;
    LaunchFn gemm = nullptr;
    LaunchFn tiled = nullptr;
    LaunchFn gated = nullptr;
    LaunchFn fused = nullptr;
    LaunchFn logits = nullptr;
    ScratchFn gemm_scratch = nullptr;
    LaunchFn attn = nullptr;
    ScratchFn logits_scratch = nullptr;
    SlicesFn slices = nullptr;
    bool ok() const { return h && layout && relayout && gemm; }
};

Lib& lib() {
    static Lib l = [] {
        Lib x;
        x.h = dlopen(RAD_R3D_SO, RTLD_NOW | RTLD_LOCAL);
        if (!x.h) return x;
        x.layout = (LayoutFn)dlsym(x.h, "r3d_rad_layout_iqr");
        x.relayout = (RelayoutFn)dlsym(x.h, "r3d_rad_relayout_iqr");
        x.gemm = (LaunchFn)dlsym(x.h, "r3d_rad_gemm_nt_q_iqr_m16");
        x.tiled = (LaunchFn)dlsym(x.h, "r3d_rad_gemm_nt_q_iqr_tiled");
        x.gated = (LaunchFn)dlsym(x.h, "r3d_rad_gated_had_quant_i8g");
        x.fused = (LaunchFn)dlsym(x.h, "r3d_rad_gemm_nt_q_iqr_gated");
        x.logits = (LaunchFn)dlsym(x.h, "r3d_rad_logits_gemm_vocab_iqr");
        x.gemm_scratch = (ScratchFn)dlsym(x.h, "r3d_rad_gemm_nt_q_iqr_m16_scratch");
        x.attn = (LaunchFn)dlsym(x.h, "r3d_rad_attn_dense");
        x.logits_scratch = (ScratchFn)dlsym(x.h, "r3d_logits_gemm_vocab_iqr_scratch");
        x.slices = (SlicesFn)dlsym(x.h, "r3d_iqr_split_slices");
        return x;
    }();
    return l;
}

uint64_t g_z = 0x452821E638D01377ull;
uint64_t rnd() {
    g_z += 0x9E3779B97F4A7C15ull;
    uint64_t x = g_z;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

RadParam pint(const char* k, long long v) {
    RadParam p;
    std::memset(&p, 0, sizeof p);
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}
RadParam pstr(const char* k, const char* v) {
    RadParam p;
    std::memset(&p, 0, sizeof p);
    p.key = k;
    p.kind = RAD_P_STR;
    p.sval = v;
    return p;
}

const uint32_t kU[] = { 0, RAD_U1, RAD_U2, RAD_U3, RAD_U4, RAD_U5, RAD_U6, RAD_U7 };

/* The unsigned code dtype of a width up to 12 bits, for the grid forms. */
uint32_t udt(uint32_t b) {
    static const uint32_t t[] = { 0,      RAD_U1,  RAD_U2,  RAD_U3,  RAD_U4,  RAD_U5,  RAD_U6,
                                  RAD_U7, RAD_U8,  RAD_U9,  RAD_U10, RAD_U11, RAD_U12 };
    return b < (uint32_t)(sizeof t / sizeof t[0]) ? t[b] : 0;
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
uint32_t bits_at(const uint8_t* p, int64_t o, int n) {
    const uint8_t* b = p + (o >> 3);
    const int sh = (int)(o & 7);
    const int nb = (sh + n + 7) >> 3;
    uint64_t w = 0;
    for (int i = 0; i < nb; ++i) w |= (uint64_t)b[i] << (8 * i);
    return (uint32_t)((w >> sh) & ((1ull << n) - 1ull));
}

/* The canonical codes plane: row r's B-bit code for column c. */
uint32_t code_at(const W& w, int64_t r, int64_t c, uint32_t B) {
    const int i = w.find("codes");
    return bits_at(w.data[(size_t)i].data() + r * w.row_bytes(i), c * B, (int)B);
}

/* Row r's trellis state for column c: the L-bit window ending at r (row r lowest), tail-biting
 * over the run of T rows (the run's last span codes seed it). */
uint32_t canonical_state(const W& w, uint32_t B, uint32_t L, int64_t run, int64_t r, int64_t c) {
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
void fwht128_host(float* v) {
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; ++i)
            if ((i & h) == 0) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
}

/* A class's decoded int8 weight (no tcq_rb rescale: the head's classes are off that grid). */
std::vector<int8_t> decode_class(const W& w, uint32_t B, uint32_t L, int64_t N, int64_t K) {
    const uint32_t A = ((const uint32_t*)w.data[(size_t)w.find("trellis.mad")].data())[0];
    const uint32_t Bc = ((const uint32_t*)w.data[(size_t)w.find("trellis.mad")].data())[1];
    std::vector<int8_t> dec((size_t)(N * K));
    const int64_t run = 256;
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const uint32_t st = canonical_state(w, B, L, run, n, k);
            dec[(size_t)(n * K + k)] = (int8_t)mad_value(st, A, Bc);
        }
    return dec;
}

/* The high-precision head reference lives below, beside the class fixture it reads. */

struct Case {
    const char* what;
    uint32_t B, L, RB;
    Scale sc;
    int64_t N, K;
    int M;
    bool split = false;   /* ask the row for its split-K scratch (the GEMV's narrow band) */
    Kind kind = kTrellisMad;
    int pattern = 0;      /* 0: random codes, 1: all zero, 2: all one bits */
    uint32_t LV = 0;      /* Grid: log2 V */
};

/* Split-K sums S slab partials instead of the reference's one ascending chain: the same group
 * terms, associated differently. The seeded down fixture measures one bf16 ulp on one output of
 * 5120; two covers a second boundary flip and is still orders below any decode or scale error. */
constexpr int kSplitMaxUlps = 2;

/* The decoded weight, row-major [N, K]. A trellis's values are canonical_state's, one (n, k) at a
 * time, but the state advances by one code as n rises inside a 256-row run instead of restarting
 * from the run's seed: N * K steps rather than N * K * run. The down shape (5120 x 17408) is
 * otherwise too slow for the device test. A table code is its entry; a grid code is a V-vector of
 * entries (V = 4 for the built forms). */
std::vector<int8_t> decode_rows(const W& w, Kind kind, uint32_t B, uint32_t LV, uint32_t L,
                                uint32_t RB, Scale sc, int64_t N, int64_t K, const uint8_t* sb,
                                int64_t nkb) {
    std::vector<int8_t> dec((size_t)(N * K));
    if (kind == kTable) {
        const int8_t* tab = (const int8_t*)w.data[(size_t)w.find("table")].data();
        for (int64_t n = 0; n < N; ++n)
            for (int64_t k = 0; k < K; ++k)
                dec[(size_t)(n * K + k)] = tab[code_at(w, n, k, B)];
        return dec;
    }
    if (kind == kGrid) {
        const int8_t* grid = (const int8_t*)w.data[(size_t)w.find("grid")].data();
        const int64_t V = (int64_t)1 << LV, nc = K / V;
        for (int64_t n = 0; n < N; ++n)
            for (int64_t c = 0; c < nc; ++c) {
                const uint32_t code = code_at(w, n, c, B);
                for (int64_t j = 0; j < V; ++j)
                    dec[(size_t)(n * K + c * V + j)] = grid[code * V + j];
            }
        return dec;
    }
    const Form f{ kTrellisMad, B, 0, L, RB, sc, 0, 0 };
    const uint32_t A = ((const uint32_t*)(w.data[(size_t)w.find("trellis.mad")].data()))[0];
    const uint32_t Bc = ((const uint32_t*)(w.data[(size_t)w.find("trellis.mad")].data()))[1];
    const int64_t run = 256;
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

/* The host answer: the decoded weight, then the int8 grid's order. */
std::vector<uint16_t> reference(const W& w, const Case& c, const std::vector<int8_t>& a,
                                const std::vector<float>& as, const uint8_t* sb, int64_t sbytes) {
    const int64_t nkb = c.K / 128;
    (void)sbytes;
    const Form f{ c.kind, c.B, c.LV, c.L, c.RB, c.sc, 0, 0 };
    const std::vector<int8_t> dec =
        decode_rows(w, c.kind, c.B, c.LV, c.L, c.RB, c.sc, c.N, c.K, sb, nkb);
    /* The scales the GEMM folds. */
    std::vector<uint16_t> y((size_t)(c.M * c.N));
    for (int m = 0; m < c.M; ++m)
        for (int64_t n = 0; n < c.N; ++n) {
            float acc = 0.0f;
            for (int64_t kb = 0; kb < nkb; ++kb) {
                int sub = 0;
                for (int64_t k = 0; k < 128; ++k)
                    sub += (int)a[(size_t)(m * c.K + kb * 128 + k)] *
                           (int)dec[(size_t)(n * c.K + kb * 128 + k)];
                float sw;
                if (c.sc == kRowF32) {
                    const float* cf = (const float*)(sb + cf_off(f, c.N, c.K));
                    sw = cf[kb];
                } else {
                    const int64_t i = tile_major(n, kb, nkb);
                    sw = bf16_to_f32(*(const uint16_t*)(sb + i * 2));
                }
                const float t1 = (float)sub * as[(size_t)(m * nkb + kb)];
                const float t2 = t1 * sw;
                acc += t2;
            }
            if (c.sc == kRowF32) acc *= ((const float*)sb)[n];
            y[(size_t)(m * c.N + n)] = f32_to_bf16(acc);
        }
    return y;
}

bool run_case(const Case& c) {
    W w;
    rad_enc_clear(&w.enc);
    if (c.kind == kTrellisMad) {
        rad_enc_copy_str(w.enc.scheme, "trellis");
        rad_enc_add_plane(&w.enc, "codes", kU[c.B], 1, 1);
        if (c.sc == kRowF32) {
            rad_enc_add_plane(&w.enc, "scale", RAD_F32, 1, 0);
            rad_enc_add_plane(&w.enc, "scale.col", RAD_F32, 0, 128);
        } else {
            rad_enc_add_plane(&w.enc, "scale", RAD_BF16, 1, 128);
        }
        const int t = rad_enc_add_table(&w.enc, "trellis", RAD_I8, (int64_t)1 << c.L, 1);
        w.enc.plane[t].block[0] = 256;
        w.enc.plane[t].block[1] = c.RB ? 32 : 0;
        rad_enc_add_table(&w.enc, "trellis.mad", RAD_U32, 2, 1);
    } else {
        rad_enc_copy_str(w.enc.scheme, "affine");
        rad_enc_add_plane(&w.enc, "codes", udt(c.B), 1, c.kind == kGrid ? 4 : 1);
        if (c.sc == kRowF32) {
            rad_enc_add_plane(&w.enc, "scale", RAD_F32, 1, 0);
            rad_enc_add_plane(&w.enc, "scale.col", RAD_F32, 0, 128);
        } else {
            rad_enc_add_plane(&w.enc, "scale", RAD_BF16, 1, 128);
        }
        if (c.kind == kTable)
            rad_enc_add_table(&w.enc, "table", RAD_I8, 1, (int64_t)1 << c.B);
        else
            rad_enc_add_table(&w.enc, "grid", RAD_I8, (int64_t)1 << c.B, 4);
    }
    rad_enc_copy_str(w.enc.transform, "fwht128");
    w.size(c.N, c.K);

    const int ic = w.find("codes");
    if (c.pattern == 1) std::fill(w.data[(size_t)ic].begin(), w.data[(size_t)ic].end(), (uint8_t)0);
    else if (c.pattern == 2)
        std::fill(w.data[(size_t)ic].begin(), w.data[(size_t)ic].end(), (uint8_t)0xFF);
    else
        for (auto& b : w.data[(size_t)ic]) b = (uint8_t)rnd();
    if (c.kind == kTrellisMad) {
        const uint32_t A = (uint32_t)(rnd() & 0xFFFFFFu) | 1u, Bc = (uint32_t)rnd();
        uint32_t* ab = (uint32_t*)w.data[(size_t)w.find("trellis.mad")].data();
        ab[0] = A;
        ab[1] = Bc;
        int8_t* tab = (int8_t*)w.data[(size_t)w.find("trellis")].data();
        for (uint32_t st = 0; st < (1u << c.L); ++st) tab[st] = (int8_t)mad_value(st, A, Bc);
    } else {
        const char* role = c.kind == kTable ? "table" : "grid";
        int8_t* tab = (int8_t*)w.data[(size_t)w.find(role)].data();
        for (size_t i = 0; i < w.data[(size_t)w.find(role)].size(); ++i)
            tab[i] = (int8_t)((int)(rnd() % 255) - 127);
    }
    if (c.sc == kRowF32) {
        float* rs = (float*)w.data[(size_t)w.find("scale")].data();
        for (int64_t n = 0; n < c.N; ++n) rs[n] = 0.001f + 0.0005f * (float)(n % 7);
        float* cf = (float*)w.data[(size_t)w.find("scale.col")].data();
        for (int64_t kb = 0; kb < c.K / 128; ++kb) cf[kb] = 0.5f + 0.1f * (float)kb;
    } else {
        uint16_t* sc = (uint16_t*)w.data[(size_t)w.find("scale")].data();
        for (int64_t i = 0; i < c.N * (c.K / 128); ++i)
            sc[i] = f32_to_bf16(0.002f + 0.0001f * (float)(i % 11));
    }

    const std::vector<RadParam> lp = { pint("N", c.N), pint("K", c.K), pint("group", 128),
                                       pstr("dtype", "i8a8h") };
    /* Each operand gets the planes of its own role, as the loader selects them. */
    std::vector<int> selB, selS;
    std::vector<RadTensor> plB, plS;
    for (int i = 0; i < w.enc.n_planes; ++i) {
        const char* r = w.enc.plane[i].role;
        if (!std::strncmp(r, "scale", 5)) {
            selS.push_back(i);
            plS.push_back(w.t(i));
        } else {
            selB.push_back(i);
            plB.push_back(w.t(i));
        }
    }

    RadLayout L{};
    int lrc = lib().layout(lp.data(), (int)lp.size(), 2, &w.enc, selB.data(), plB.data(),
                           (int)plB.size(), &L);
    if (lrc != RAD_OK) {
        std::fprintf(stderr, "%s: weight layout refused rc=%d\n", c.what, lrc);
        return false;
    }
    std::vector<uint8_t> sb((size_t)L.bytes, 0);
    if (lib().relayout(lp.data(), (int)lp.size(), 2, &w.enc, selB.data(), plB.data(),
                       (int)plB.size(), sb.data(), L.bytes) != RAD_OK) {
        std::fprintf(stderr, "%s: weight relayout refused\n", c.what);
        return false;
    }
    RadLayout LS{};
    int lsrc = lib().layout(lp.data(), (int)lp.size(), 3, &w.enc, selS.data(), plS.data(),
                            (int)plS.size(), &LS);
    if (lsrc != RAD_OK) {
        std::fprintf(stderr, "%s: scale layout refused rc=%d (planes %d)\n", c.what, lsrc,
                     w.enc.n_planes);
        for (int i = 0; i < w.enc.n_planes; ++i)
            std::fprintf(stderr, "  plane %d role='%s' dt=%u block=%lld,%lld dims=%lld,%lld\n", i,
                         w.enc.plane[i].role, w.enc.plane[i].dtype,
                         (long long)w.enc.plane[i].block[0], (long long)w.enc.plane[i].block[1],
                         (long long)w.rows[(size_t)i], (long long)w.cols[(size_t)i]);
        return false;
    }
    std::vector<uint8_t> sbs((size_t)LS.bytes, 0);
    if (lib().relayout(lp.data(), (int)lp.size(), 3, &w.enc, selS.data(), plS.data(),
                       (int)plS.size(), sbs.data(), LS.bytes) != RAD_OK) {
        std::fprintf(stderr, "%s: scale relayout refused\n", c.what);
        return false;
    }

    const int64_t nkb = c.K / 128;
    std::vector<int8_t> a((size_t)(c.M * c.K));
    std::vector<float> as((size_t)(c.M * nkb));
    for (auto& v : a) v = (int8_t)((int)(rnd() % 255) - 127);
    for (auto& v : as) v = 0.01f + 0.001f * (float)(rnd() % 7);
    std::vector<uint16_t> y((size_t)(c.M * c.N), 0xDEADu);

    void *dA = nullptr, *dS = nullptr, *dB = nullptr, *dBS = nullptr, *dY = nullptr;
    void* dPart = nullptr;
    auto clean = [&] {
        hipFree(dA);
        hipFree(dS);
        hipFree(dB);
        hipFree(dBS);
        hipFree(dY);
        hipFree(dPart);
    };
    if (hipMalloc(&dA, a.size()) != hipSuccess) return false;
    if (hipMalloc(&dS, as.size() * 4) != hipSuccess) { clean(); return false; }
    if (hipMalloc(&dB, sb.size()) != hipSuccess) { clean(); return false; }
    if (hipMalloc(&dBS, sbs.size()) != hipSuccess) { clean(); return false; }
    if (hipMalloc(&dY, y.size() * 2) != hipSuccess) { clean(); return false; }
    hipMemcpy(dA, a.data(), a.size(), hipMemcpyHostToDevice);
    hipMemcpy(dS, as.data(), as.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dB, sb.data(), sb.size(), hipMemcpyHostToDevice);
    hipMemcpy(dBS, sbs.data(), sbs.size(), hipMemcpyHostToDevice);
    hipMemset(dY, 0, y.size() * 2);

    RadTensor ta{}, tas{}, tb{}, tbs{}, ty{};
    ta.data = dA;
    ta.dtype = RAD_I8;
    ta.rank = 2;
    ta.shape[0] = c.M;
    ta.shape[1] = c.K;
    rad_tensor_pack(&ta);
    tas.data = dS;
    tas.dtype = RAD_F32;
    tas.rank = 2;
    tas.shape[0] = c.M;
    tas.shape[1] = nkb;
    rad_tensor_pack(&tas);
    tb.data = dB;
    tb.dtype = L.dtype;
    tb.rank = 2;
    tb.shape[0] = c.N;
    tb.shape[1] = c.K;
    rad_tensor_pack(&tb);
    tbs.data = dBS;
    tbs.dtype = LS.dtype;
    tbs.rank = 2;
    tbs.shape[0] = c.N;
    tbs.shape[1] = nkb;
    rad_tensor_pack(&tbs);
    ty.data = dY;
    ty.dtype = RAD_BF16;
    ty.rank = 2;
    ty.shape[0] = c.M;
    ty.shape[1] = c.N;
    rad_tensor_pack(&ty);
    RadTensor ops[7] = { ta, tas, tb, tbs, ty, RadTensor{}, RadTensor{} };

    const RadParam params[] = { pint("M", c.M), pint("N", c.N), pint("K", c.K),
                                pint("group", 128), pstr("dtype", "i8a8h") };
    RadArgs args{};
    args.t = ops;
    args.n_t = 7;
    args.p = params;
    args.n_p = 5;
    args.kv_used_max_h = -1;

    /* The GEMV's split band: the row's own scratch hook sizes [splits][N] f32 partials. The hook
     * is queried for every GEMV case, so the wide shape (no split) must answer 0, and a case that
     * asks for the split must get a buffer at least the logical size. */
    int splits = 1;
    int64_t npart = 0;
    if (c.M <= 16 && lib().gemm_scratch) {
        npart = lib().gemm_scratch(&args);
        if (lib().slices) splits = lib().slices(c.N, c.K);
        if (c.split) {
            if (!lib().slices || splits < 2) {
                std::fprintf(stderr, "%s: the row does not split this geometry\n", c.what);
                clean();
                return false;
            }
            if (npart < (int64_t)splits * c.N * 4) {
                std::fprintf(stderr, "%s: scratch hook answered %lld bytes, needs %lld\n", c.what,
                             (long long)npart, (long long)((int64_t)splits * c.N * 4));
                clean();
                return false;
            }
            if (hipMalloc(&dPart, (size_t)npart) != hipSuccess) {
                clean();
                return false;
            }
            /* Poison: every partial the split writes must replace these NaNs. */
            hipMemset(dPart, 0x7F, (size_t)npart);
            args.scratch = dPart;
            args.scratch_bytes = npart;
        } else if (c.M == 1 && c.N > 8192 && npart != 0) {
            std::fprintf(stderr, "%s: the wide GEMV must not ask for split scratch (%lld)\n",
                         c.what, (long long)npart);
            clean();
            return false;
        }
    } else if (c.split) {
        std::fprintf(stderr, "%s: the row has no scratch hook\n", c.what);
        clean();
        return false;
    }

    const int rc = (c.M > 16 ? lib().tiled : lib().gemm)(&args, (RadStream)0);
    const hipError_t he = hipDeviceSynchronize();
    if (rc != RAD_OK || he != hipSuccess) {
        std::fprintf(stderr, "%s: launch rc=%d hip=%d\n", c.what, rc, (int)he);
        clean();
        return false;
    }
    hipMemcpy(y.data(), dY, y.size() * 2, hipMemcpyDeviceToHost);
    if (c.split) {
        std::vector<float> part((size_t)((int64_t)splits * c.N));
        hipMemcpy(part.data(), dPart, part.size() * 4, hipMemcpyDeviceToHost);
        size_t unwritten = 0;
        for (float v : part)
            if (!std::isfinite(v)) ++unwritten;
        if (unwritten) {
            std::fprintf(stderr, "%s: %zu of %zu split partials were never written\n", c.what,
                         unwritten, part.size());
            clean();
            return false;
        }
    }
    clean();

    const std::vector<uint16_t> want = reference(w, c, a, as, sbs.data(), (int64_t)sbs.size());
    /* bf16 ordered key: monotonically increasing across the sign, so a difference of 1 is one
     * representable bf16 step. */
    const auto order = [](uint16_t h) {
        return (h & 0x8000u) ? -(int)(h & 0x7fffu) : (int)h;
    };
    size_t bad = 0, first = y.size(), nonzero = 0;
    int worst = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        const int gap = std::abs(order(y[i]) - order(want[i]));
        if (gap) ++nonzero;
        if (gap > worst) worst = gap;
        const bool differs = c.split ? gap > kSplitMaxUlps : gap != 0;
        if (differs) {
            ++bad;
            if (first == y.size()) first = i;
        }
    }
    if (c.split)
        std::printf("%s: worst %d bf16 ulp, %zu of %zu outputs differ (tolerance %d)\n", c.what,
                    worst, nonzero, y.size(), kSplitMaxUlps);
    if (bad) {
        std::fprintf(stderr, "%s: %zu of %zu differ; first y[%zu]: got 0x%04x want 0x%04x "
                     "(got %.6f want %.6f)\n",
                     c.what, bad, y.size(), first, y[first], want[first],
                     bf16_to_f32(y[first]), bf16_to_f32(want[first]));
        if (c.M == 1) {
            std::fprintf(stderr, "  y[0..7]:");
            for (size_t i = 0; i < 8 && i < y.size(); ++i)
                std::fprintf(stderr, " %04x/%04x", y[i], want[i]);
            std::fprintf(stderr, "\n");
        }
        return false;
    }
    return true;
}

/* gemm_iqr_gated_i8a8h against its unfused chain: r3d_rad_gemm_nt_q_iqr_m16 writes the [M, 2n]
 * bf16 gate_up, r3d_rad_gated_had_quant_i8g gates, rotates and codes it; the fused row must give the
 * same code bytes and the same f32 scales with no intermediate and no scratch. The weight is the
 * 27B's feed-forward width (n = 17408, so 136 groups of 128) at a shorter K, and M in {1, 5, 16},
 * act in {silu, sigmoid} and out_bf16 present or absent are all run against it. */
bool run_gated_fold() {
    const int64_t n = 17408, N = 2 * n, K = 2560;
    const int64_t nkb = K / 128, ng = n / 128;
    const uint32_t B = 4, L16 = 16;
    W w;
    rad_enc_clear(&w.enc);
    rad_enc_copy_str(w.enc.scheme, "trellis");
    rad_enc_add_plane(&w.enc, "codes", kU[B], 1, 1);
    rad_enc_add_plane(&w.enc, "scale", RAD_F32, 1, 0);
    rad_enc_add_plane(&w.enc, "scale.col", RAD_F32, 0, 128);
    const int t = rad_enc_add_table(&w.enc, "trellis", RAD_I8, (int64_t)1 << L16, 1);
    w.enc.plane[t].block[0] = 256;
    w.enc.plane[t].block[1] = 0;
    rad_enc_add_table(&w.enc, "trellis.mad", RAD_U32, 2, 1);
    rad_enc_copy_str(w.enc.transform, "fwht128");
    w.size(N, K);

    for (auto& b : w.data[(size_t)w.find("codes")]) b = (uint8_t)rnd();
    const uint32_t A = (uint32_t)(rnd() & 0xFFFFFFu) | 1u, Bc = (uint32_t)rnd();
    uint32_t* ab = (uint32_t*)w.data[(size_t)w.find("trellis.mad")].data();
    ab[0] = A;
    ab[1] = Bc;
    int8_t* tab = (int8_t*)w.data[(size_t)w.find("trellis")].data();
    for (uint32_t st = 0; st < (1u << L16); ++st) tab[st] = (int8_t)mad_value(st, A, Bc);
    float* rs = (float*)w.data[(size_t)w.find("scale")].data();
    for (int64_t i = 0; i < N; ++i) rs[i] = 0.001f + 0.0005f * (float)(i % 7);
    float* cf = (float*)w.data[(size_t)w.find("scale.col")].data();
    for (int64_t kb = 0; kb < nkb; ++kb) cf[kb] = 0.5f + 0.1f * (float)kb;

    const std::vector<RadParam> lp = { pint("N", N), pint("K", K), pint("group", 128),
                                       pstr("dtype", "i8a8h") };
    std::vector<int> selB, selS;
    std::vector<RadTensor> plB, plS;
    for (int i = 0; i < w.enc.n_planes; ++i) {
        const char* r = w.enc.plane[i].role;
        if (!std::strncmp(r, "scale", 5)) {
            selS.push_back(i);
            plS.push_back(w.t(i));
        } else {
            selB.push_back(i);
            plB.push_back(w.t(i));
        }
    }
    RadLayout L{};
    if (lib().layout(lp.data(), (int)lp.size(), 2, &w.enc, selB.data(), plB.data(),
                     (int)plB.size(), &L) != RAD_OK) {
        std::fprintf(stderr, "gated fold: weight layout refused\n");
        return false;
    }
    std::vector<uint8_t> sb((size_t)L.bytes, 0);
    if (lib().relayout(lp.data(), (int)lp.size(), 2, &w.enc, selB.data(), plB.data(),
                       (int)plB.size(), sb.data(), L.bytes) != RAD_OK) {
        std::fprintf(stderr, "gated fold: weight relayout refused\n");
        return false;
    }
    RadLayout LS{};
    if (lib().layout(lp.data(), (int)lp.size(), 3, &w.enc, selS.data(), plS.data(),
                     (int)plS.size(), &LS) != RAD_OK) {
        std::fprintf(stderr, "gated fold: scale layout refused\n");
        return false;
    }
    std::vector<uint8_t> sbs((size_t)LS.bytes, 0);
    if (lib().relayout(lp.data(), (int)lp.size(), 3, &w.enc, selS.data(), plS.data(),
                       (int)plS.size(), sbs.data(), LS.bytes) != RAD_OK) {
        std::fprintf(stderr, "gated fold: scale relayout refused\n");
        return false;
    }

    const int Mmax = 16;
    std::vector<int8_t> a((size_t)(Mmax * K));
    std::vector<float> as((size_t)(Mmax * nkb));
    for (auto& v : a) v = (int8_t)((int)(rnd() % 255) - 127);
    for (auto& v : as) v = 0.01f + 0.001f * (float)(rnd() % 7);

    void *dA = nullptr, *dAS = nullptr, *dB = nullptr, *dBS = nullptr, *dGU = nullptr;
    void *dQ = nullptr, *dSC = nullptr, *dOB = nullptr;
    auto clean = [&] {
        hipFree(dA);  hipFree(dAS); hipFree(dB);  hipFree(dBS); hipFree(dGU);
        hipFree(dQ);  hipFree(dSC); hipFree(dOB);
    };
    if (hipMalloc(&dA, a.size()) != hipSuccess ||
        hipMalloc(&dAS, as.size() * 4) != hipSuccess ||
        hipMalloc(&dB, sb.size()) != hipSuccess ||
        hipMalloc(&dBS, sbs.size()) != hipSuccess ||
        hipMalloc(&dGU, (size_t)(Mmax * N) * 2) != hipSuccess ||
        hipMalloc(&dQ, (size_t)(Mmax * n)) != hipSuccess ||
        hipMalloc(&dSC, (size_t)(Mmax * ng) * 4) != hipSuccess ||
        hipMalloc(&dOB, (size_t)(Mmax * n) * 2) != hipSuccess) {
        clean();
        return false;
    }
    hipMemcpy(dA, a.data(), a.size(), hipMemcpyHostToDevice);
    hipMemcpy(dAS, as.data(), as.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dB, sb.data(), sb.size(), hipMemcpyHostToDevice);
    hipMemcpy(dBS, sbs.data(), sbs.size(), hipMemcpyHostToDevice);

    bool ok = true;
    const int32_t ms[3] = { 1, 5, 16 };
    for (int mi = 0; mi < 3; ++mi) {
        const int64_t M = ms[mi];
        for (int act = 0; act < 2; ++act) {
            for (int ob = 0; ob < 2; ++ob) {
                const char* act_s = act ? "sigmoid" : "silu";
                RadTensor ta{}, tas{}, tb{}, tbs{}, tgu{}, tq{}, tsc{}, tob{};
                ta.data = dA;  ta.dtype = RAD_I8;  ta.rank = 2;  ta.shape[0] = M; ta.shape[1] = K;
                rad_tensor_pack(&ta);
                tas.data = dAS; tas.dtype = RAD_F32; tas.rank = 2; tas.shape[0] = M;
                tas.shape[1] = nkb; rad_tensor_pack(&tas);
                tb.data = dB;  tb.dtype = L.dtype;  tb.rank = 2;  tb.shape[0] = N; tb.shape[1] = K;
                rad_tensor_pack(&tb);
                tbs.data = dBS; tbs.dtype = LS.dtype; tbs.rank = 2; tbs.shape[0] = N;
                tbs.shape[1] = nkb; rad_tensor_pack(&tbs);
                tgu.data = dGU; tgu.dtype = RAD_BF16; tgu.rank = 2; tgu.shape[0] = M;
                tgu.shape[1] = N; rad_tensor_pack(&tgu);
                tq.data = dQ;  tq.dtype = RAD_I8;  tq.rank = 2;  tq.shape[0] = M; tq.shape[1] = n;
                rad_tensor_pack(&tq);
                tsc.data = dSC; tsc.dtype = RAD_F32; tsc.rank = 2; tsc.shape[0] = M;
                tsc.shape[1] = ng; rad_tensor_pack(&tsc);
                if (ob) {
                    tob.data = dOB; tob.dtype = RAD_BF16; tob.rank = 2; tob.shape[0] = M;
                    tob.shape[1] = n; rad_tensor_pack(&tob);
                }

                /* The pair: m16 into the [M, 2n] bf16 scratch, then the gated quantiser. */
                RadTensor pair_ops[5] = { ta, tas, tb, tbs, tgu };
                const RadParam pair_p[] = { pint("M", M), pint("N", N), pint("K", K),
                                            pint("group", 128), pstr("dtype", "i8a8h") };
                RadArgs pa{};
                pa.t = pair_ops; pa.n_t = 5; pa.p = pair_p; pa.n_p = 5; pa.kv_used_max_h = -1;
                int rc = lib().gemm(&pa, (RadStream)0);
                if (rc != RAD_OK || hipDeviceSynchronize() != hipSuccess) {
                    std::fprintf(stderr, "gated fold M=%lld %s ob=%d: pair GEMM rc=%d\n",
                                 (long long)M, act_s, ob, rc);
                    ok = false;
                    continue;
                }
                RadTensor gated_ops[4] = { tgu, tq, tsc, ob ? tob : RadTensor{} };
                const RadParam gated_p[] = { pint("group", 128), pstr("dtype", "bf16"),
                                             pstr("act", act_s) };
                RadArgs pg{};
                pg.t = gated_ops; pg.n_t = 4; pg.p = gated_p; pg.n_p = 3; pg.kv_used_max_h = -1;
                rc = lib().gated(&pg, (RadStream)0);
                if (rc != RAD_OK || hipDeviceSynchronize() != hipSuccess) {
                    std::fprintf(stderr, "gated fold M=%lld %s ob=%d: pair quantiser rc=%d\n",
                                 (long long)M, act_s, ob, rc);
                    ok = false;
                    continue;
                }
                std::vector<uint8_t> qp((size_t)(M * n)), sp((size_t)(M * ng) * 4);
                std::vector<uint8_t> op(ob ? (size_t)(M * n) * 2 : 0);
                hipMemcpy(qp.data(), dQ, qp.size(), hipMemcpyDeviceToHost);
                hipMemcpy(sp.data(), dSC, sp.size(), hipMemcpyDeviceToHost);
                if (ob) hipMemcpy(op.data(), dOB, op.size(), hipMemcpyDeviceToHost);

                /* Poison the outputs the fused row must write every byte of: identical values either
                 * way would not prove the second launch ran. */
                hipMemset(dQ, 0x5A, (size_t)(M * n));
                hipMemset(dSC, 0x5A, (size_t)(M * ng) * 4);
                if (ob) hipMemset(dOB, 0xDE, (size_t)(M * n) * 2);

                /* The fused row: one launch, no scratch. */
                RadTensor fused_ops[7] = { ta, tas, tb, tbs, tq, tsc, ob ? tob : RadTensor{} };
                const RadParam fused_p[] = { pint("M", M), pint("N", N), pint("K", K),
                                             pint("group", 128), pstr("dtype", "i8a8h"),
                                             pstr("act", act_s) };
                RadArgs pf{};
                pf.t = fused_ops; pf.n_t = 7; pf.p = fused_p; pf.n_p = 6; pf.kv_used_max_h = -1;
                rc = lib().fused(&pf, (RadStream)0);
                if (rc != RAD_OK || hipDeviceSynchronize() != hipSuccess) {
                    std::fprintf(stderr, "gated fold M=%lld %s ob=%d: fused rc=%d\n",
                                 (long long)M, act_s, ob, rc);
                    ok = false;
                    continue;
                }
                std::vector<uint8_t> qf((size_t)(M * n)), sf((size_t)(M * ng) * 4);
                std::vector<uint8_t> of(ob ? (size_t)(M * n) * 2 : 0);
                hipMemcpy(qf.data(), dQ, qf.size(), hipMemcpyDeviceToHost);
                hipMemcpy(sf.data(), dSC, sf.size(), hipMemcpyDeviceToHost);
                if (ob) hipMemcpy(of.data(), dOB, of.size(), hipMemcpyDeviceToHost);

                const auto same = [&](const char* what, const std::vector<uint8_t>& got,
                                      const std::vector<uint8_t>& want, bool as_f32) {
                    size_t first = got.size();
                    for (size_t i = 0; i < got.size(); ++i)
                        if (got[i] != want[i]) { first = i; break; }
                    if (first == got.size()) return true;
                    if (as_f32) {
                        const float g = ((const float*)got.data())[first / 4];
                        const float h = ((const float*)want.data())[first / 4];
                        std::fprintf(stderr, "gated fold M=%lld %s ob=%d: %s differs at %zu "
                                     "(got %.9g want %.9g)\n", (long long)M, act_s, ob, what,
                                     first / 4, g, h);
                    } else {
                        std::fprintf(stderr, "gated fold M=%lld %s ob=%d: %s differs at byte %zu "
                                     "(got 0x%02x want 0x%02x)\n", (long long)M, act_s, ob, what,
                                     first, got[first], want[first]);
                    }
                    return false;
                };
                if (!same("codes", qf, qp, false)) ok = false;
                if (!same("scales", sf, sp, true)) ok = false;
                if (ob && !same("out_bf16", of, op, false)) ok = false;
            }
        }
    }
    clean();
    return ok;
}

}  // namespace

TEST(iqr_gemm_on_the_card_matches_the_host_reference) {
    if (!lib().ok()) {
        std::fprintf(stderr, "skip: %s not loadable here\n", RAD_R3D_SO);
        return;
    }
    int nd = 0;
    if (hipGetDeviceCount(&nd) != hipSuccess || nd < 1) {
        std::fprintf(stderr, "skip: no card\n");
        return;
    }
    const Case cases[] = {
        { "mad u4 L16 one-f32-row, GEMV", 4, 16, 0, kRowF32, 512, 384, 1 },
        { "mad u4 L16 one-f32-row, 16 tokens", 4, 16, 0, kRowF32, 512, 384, 16 },
        { "mad u5 L16 one-f32-row, wide GEMV 12288x2560 (no split)", 5, 16, 0, kRowF32, 12288,
          2560, 1 },
        { "mad u4 L16 one-f32-row, 2 tokens", 4, 16, 0, kRowF32, 256, 512, 2 },
        { "mad u4 L16 bf16, 5 tokens", 4, 16, 0, kBf16, 256, 256, 5 },
        { "mad u4 L16 bf16, 16 tokens", 4, 16, 0, kBf16, 256, 512, 16 },
        { "mad u4 L16 one-f32-row, 33 tokens (tiled v_dot4)", 4, 16, 0, kRowF32, 512, 384, 33 },
        { "mad u4 L16 one-f32-row, 64 tokens (resident)", 4, 16, 0, kRowF32, 256, 384, 64 },
        { "mad u4 L16 one-f32-row, 160 tokens (resident)", 4, 16, 0, kRowF32, 256, 384, 160 },
        { "mad u4 L16 bf16, 512 tokens (resident)", 4, 16, 0, kBf16, 256, 384, 512 },
        { "mad u4 L16 one-f32-row, 512 tokens (resident)", 4, 16, 0, kRowF32, 512, 384, 512 },
        { "mad u4 L16 bf16, 4096 tokens (resident)", 4, 16, 0, kBf16, 256, 384, 4096 },
        /* IQR-V1's scalar codebook (kTable): the entry a u2..u5 code names, under a bf16 scale a
         * (row, 128). Every width is covered; u4 also at M = 1 / 16 / 512, and the zero and
         * all-one-bits code planes hold the two ends of the table. */
        { "table u4 bf16, GEMV", 4, 0, 0, kBf16, 256, 384, 1, false, kTable },
        { "table u4 bf16, 16 tokens", 4, 0, 0, kBf16, 256, 384, 16, false, kTable },
        { "table u4 bf16, 512 tokens (resident)", 4, 0, 0, kBf16, 256, 384, 512, false, kTable },
        { "table u2 bf16, 16 tokens", 2, 0, 0, kBf16, 256, 384, 16, false, kTable },
        { "table u3 bf16, 16 tokens", 3, 0, 0, kBf16, 256, 384, 16, false, kTable },
        { "table u5 bf16, 16 tokens", 5, 0, 0, kBf16, 256, 384, 16, false, kTable },
        { "table u4 bf16 zero codes, 16 tokens", 4, 0, 0, kBf16, 256, 384, 16, false, kTable, 1 },
        { "table u4 bf16 zero codes, GEMV", 4, 0, 0, kBf16, 256, 384, 1, false, kTable, 1 },
        { "table u4 bf16 max codes, 16 tokens", 4, 0, 0, kBf16, 256, 384, 16, false, kTable, 2 },
        { "table u5 bf16 max codes, GEMV", 5, 0, 0, kBf16, 256, 384, 1, false, kTable, 2 },
        /* IQR-V1's vector codebook (kGrid, V = 4): the codeword a u8/u9/u10/u12 code names, under a
         * bf16 scale a (row, 128). u9 at M = 1 / 16 / 512, every built width, and the zero and
         * all-one-bits code planes. */
        { "grid u9 V4 bf16, GEMV", 9, 0, 0, kBf16, 256, 384, 1, false, kGrid, 0, 2 },
        { "grid u9 V4 bf16, 16 tokens", 9, 0, 0, kBf16, 256, 384, 16, false, kGrid, 0, 2 },
        { "grid u9 V4 bf16, 512 tokens (resident)", 9, 0, 0, kBf16, 256, 384, 512, false, kGrid, 0,
          2 },
        { "grid u8 V4 bf16, 16 tokens", 8, 0, 0, kBf16, 256, 384, 16, false, kGrid, 0, 2 },
        { "grid u10 V4 bf16, 16 tokens", 10, 0, 0, kBf16, 256, 384, 16, false, kGrid, 0, 2 },
        { "grid u12 V4 bf16, 16 tokens", 12, 0, 0, kBf16, 256, 384, 16, false, kGrid, 0, 2 },
        { "grid u9 V4 bf16 zero codes, 16 tokens", 9, 0, 0, kBf16, 256, 384, 16, false, kGrid, 1,
          2 },
        { "grid u9 V4 bf16 zero codes, GEMV", 9, 0, 0, kBf16, 256, 384, 1, false, kGrid, 1, 2 },
        { "grid u9 V4 bf16 max codes, 16 tokens", 9, 0, 0, kBf16, 256, 384, 16, false, kGrid, 2, 2 },
        { "grid u12 V4 bf16 max codes, GEMV", 12, 0, 0, kBf16, 256, 384, 1, false, kGrid, 2, 2 },
        /* The 27B's down vector (N = 5120, K = 17408, u4, one f32 scale a row, M = 1): the shape
         * the W0 table measures at 51.5 %. It takes the split band, so the comparison is in bf16
         * ulps (2). Every other case above stays bit-identical: it either passes no scratch (the
         * shim's unsplit fallback) or is wide enough that the split rule declines. */
        { "down 5120x17408 u4 L16 one-f32-row, GEMV split-K", 4, 16, 0, kRowF32, 5120, 17408, 1,
          true },
    };
    for (const Case& c : cases) CHECK(run_case(c));
}

TEST(iqr_gated_fold_is_bit_identical_to_the_unfused_pair) {
    if (!lib().ok() || !lib().gated || !lib().fused) {
        std::fprintf(stderr, "skip: %s or its gated rows not loadable here\n", RAD_R3D_SO);
        return;
    }
    int nd = 0;
    if (hipGetDeviceCount(&nd) != hipSuccess || nd < 1) {
        std::fprintf(stderr, "skip: no card\n");
        return;
    }
    CHECK(run_gated_fold());
}

/* ------------------------------------------------------------------ the vocabulary head
 *
 * logits_gemm_vocab_iqr: four IQR classes of the head, the rotated sixteen-bit activation
 * (q = 128 hi + lo, one f32 scale a group), and the ids scatter. The reference rotates the bf16
 * input in f32, decodes the class weights and accumulates the block sums in double, so it holds
 * the head's exact-integer accumulation, its f32 fold and the scatter; the only rounding it does
 * not model is the sixteen-bit activation quantiser, which the tolerance below covers.
 */
struct Cls {
    W w;
    std::vector<uint8_t> b, s;
    RadLayout L{}, LS{};
    std::vector<int32_t> ids;
    int64_t rows = 0;
    uint32_t bits = 4;
};

bool make_class(Cls& c, int64_t rows, int64_t K, int32_t id0, uint64_t seed, uint32_t bits = 4) {
    g_z = seed;
    c.rows = rows;
    c.bits = bits;
    rad_enc_clear(&c.w.enc);
    rad_enc_copy_str(c.w.enc.scheme, "trellis");
    rad_enc_add_plane(&c.w.enc, "codes", kU[bits], 1, 1);
    rad_enc_add_plane(&c.w.enc, "scale", RAD_F32, 1, 0);
    rad_enc_add_plane(&c.w.enc, "scale.col", RAD_F32, 0, 128);
    const int t = rad_enc_add_table(&c.w.enc, "trellis", RAD_I8, 1ll << 16, 1);
    c.w.enc.plane[t].block[0] = 256;
    rad_enc_add_table(&c.w.enc, "trellis.mad", RAD_U32, 2, 1);
    rad_enc_copy_str(c.w.enc.transform, "fwht128");
    c.w.size(rows, K);
    for (auto& v : c.w.data[(size_t)c.w.find("codes")]) v = (uint8_t)rnd();
    const uint32_t A = (uint32_t)(rnd() & 0xFFFFFFu) | 1u, Bc = (uint32_t)rnd();
    uint32_t* ab = (uint32_t*)c.w.data[(size_t)c.w.find("trellis.mad")].data();
    ab[0] = A;
    ab[1] = Bc;
    int8_t* tab = (int8_t*)c.w.data[(size_t)c.w.find("trellis")].data();
    for (uint32_t st = 0; st < (1u << 16); ++st) tab[st] = (int8_t)mad_value(st, A, Bc);
    float* rs = (float*)c.w.data[(size_t)c.w.find("scale")].data();
    for (int64_t n = 0; n < rows; ++n) rs[n] = 0.001f + 0.0002f * (float)(n % 5);
    float* cf = (float*)c.w.data[(size_t)c.w.find("scale.col")].data();
    for (int64_t kb = 0; kb < K / 128; ++kb) cf[kb] = 1.0f;
    c.ids.resize((size_t)rows);
    for (int64_t r = 0; r < rows; ++r) c.ids[(size_t)r] = id0 + (int32_t)((r * 37) % rows);

    const std::vector<RadParam> lp = { pint("N", rows), pint("K", K), pint("group", 128),
                                       pstr("dtype", "i8a8h") };
    std::vector<int> selB, selS;
    std::vector<RadTensor> plB, plS;
    for (int i = 0; i < c.w.enc.n_planes; ++i) {
        const char* r = c.w.enc.plane[i].role;
        if (!std::strncmp(r, "scale", 5)) {
            selS.push_back(i);
            plS.push_back(c.w.t(i));
        } else {
            selB.push_back(i);
            plB.push_back(c.w.t(i));
        }
    }
    if (lib().layout(lp.data(), 4, 2, &c.w.enc, selB.data(), plB.data(), (int)plB.size(), &c.L) != RAD_OK)
        return false;
    c.b.resize((size_t)c.L.bytes);
    if (lib().relayout(lp.data(), 4, 2, &c.w.enc, selB.data(), plB.data(), (int)plB.size(),
                       c.b.data(), c.L.bytes) != RAD_OK)
        return false;
    if (lib().layout(lp.data(), 4, 3, &c.w.enc, selS.data(), plS.data(), (int)plS.size(), &c.LS) != RAD_OK)
        return false;
    c.s.resize((size_t)c.LS.bytes);
    return lib().relayout(lp.data(), 4, 3, &c.w.enc, selS.data(), plS.data(), (int)plS.size(),
                          c.s.data(), c.LS.bytes) == RAD_OK;
}

/* The high-precision head reference: the activation rotated in f32 from its bf16 values (no
 * sixteen-bit quantisation), the decoded int8 weight, a double block sum, the f32 block factor,
 * and the row's own factor once at the end. Nothing is rounded to bf16. */
void reference16(const Cls& cl, const std::vector<float>& xr, int64_t M, int64_t K, int32_t voff,
                 int64_t n_vocab, float* want) {
    const int64_t rows = cl.rows, nkb = K / 128;
    const std::vector<int8_t> dec = decode_class(cl.w, cl.bits, 16, rows, K);
    const float* rs = (const float*)cl.w.data[(size_t)cl.w.find("scale")].data();
    const float* cf = (const float*)cl.w.data[(size_t)cl.w.find("scale.col")].data();
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < rows; ++n) {
            const int64_t rel = (int64_t)cl.ids[(size_t)n] - voff;
            if (rel < 0 || rel >= n_vocab) continue;
            double acc = 0.0;
            for (int64_t kb = 0; kb < nkb; ++kb) {
                double sub = 0.0;
                for (int64_t k = 0; k < 128; ++k)
                    sub += (double)xr[(size_t)(m * K + kb * 128 + k)] *
                           (double)dec[(size_t)(n * K + kb * 128 + k)];
                acc += sub * (double)cf[kb];
            }
            want[(size_t)(m * n_vocab + rel)] = (float)(acc * (double)rs[n]);
        }
}

/* The activation is held at sixteen bits (|q| <= 127 * 128), so the kernel and this f32 reference
 * differ by the quantiser's rounding alone: at most half a step, a step being amax / 16256, spread
 * over the 128-K dot and folded by the row's scale (0.001 .. 0.0019 here). The measured worst case
 * over M = 1 and 7 and four classes is 3.2e-3 absolute with wanted logits up to ~22 (1.5e-4
 * relative); 5e-3 + 2e-4 * |want| leaves >1.5x headroom. That is far below bf16's rounding at
 * these magnitudes (ulp 0.125 at 22), which is the point of the sixteen-bit activation. */
constexpr float kLogitAtol = 5e-3f, kLogitRtol = 2e-4f;

/* `row_counts` defaults to four equal classes; a class's ids keep their own contiguous range, and
 * the output window drops `voff` ids at each end, so the caller can also give the classes unequal
 * row counts (the one launch's empty-block path). */
bool run_vocab(int64_t M, std::vector<float>* out = nullptr, const uint32_t* bits = nullptr,
               const int64_t* row_counts = nullptr) {
    const int64_t K = 512, voff = 128;
    int64_t rows[4] = { 256, 256, 256, 256 };
    for (int i = 0; i < 4; ++i)
        if (row_counts) rows[i] = row_counts[i];
    int64_t total = 0;
    for (int i = 0; i < 4; ++i) total += rows[i];
    const int64_t n_vocab = total - 2 * voff;
    Cls cls[4];
    int64_t id0 = 0;
    for (int i = 0; i < 4; ++i) {
        if (!make_class(cls[i], rows[i], K, (int32_t)id0, 0x9E3779B97F4A7C15ull * (i + 3),
                        bits ? bits[i] : 4u))
            return false;
        id0 += rows[i];
    }
    const int64_t nkb = K / 128;
    std::vector<uint16_t> xb((size_t)(M * K));
    for (auto& v : xb) {
        const float f = (float)((int)(rnd() % 4096) - 2048) / 1024.0f;
        v = f32_to_bf16(f);
    }
    std::vector<float> logits((size_t)(M * n_vocab), 0.0f);

    void *dx = nullptr, *dl = nullptr, *db[4] = {}, *ds[4] = {}, *di[4] = {};
    hipMalloc(&dx, xb.size() * 2);
    hipMemcpy(dx, xb.data(), xb.size() * 2, hipMemcpyHostToDevice);
    hipMalloc(&dl, logits.size() * 4);
    hipMemset(dl, 0, logits.size() * 4);
    for (int i = 0; i < 4; ++i) {
        hipMalloc(&db[i], cls[i].b.size());
        hipMalloc(&ds[i], cls[i].s.size());
        hipMalloc(&di[i], cls[i].ids.size() * 4);
        hipMemcpy(db[i], cls[i].b.data(), cls[i].b.size(), hipMemcpyHostToDevice);
        hipMemcpy(ds[i], cls[i].s.data(), cls[i].s.size(), hipMemcpyHostToDevice);
        hipMemcpy(di[i], cls[i].ids.data(), cls[i].ids.size() * 4, hipMemcpyHostToDevice);
    }
    RadTensor ta{}, tl{}, tw[4]{}, ts[4]{}, ti[4]{}, tv{};
    ta.data = dx; ta.dtype = RAD_BF16; ta.rank = 2; ta.shape[0] = M; ta.shape[1] = K;
    tl.data = dl; tl.dtype = RAD_F32; tl.rank = 2; tl.shape[0] = M; tl.shape[1] = n_vocab;
    for (int i = 0; i < 4; ++i) {
        tw[i].data = db[i]; tw[i].dtype = cls[i].L.dtype; tw[i].rank = 2;
        tw[i].shape[0] = rows[i]; tw[i].shape[1] = K;
        ts[i].data = ds[i]; ts[i].dtype = cls[i].LS.dtype; ts[i].rank = 2;
        ts[i].shape[0] = rows[i]; ts[i].shape[1] = nkb;
        ti[i].data = di[i]; ti[i].dtype = RAD_I32; ti[i].rank = 1; ti[i].shape[0] = rows[i];
    }
    for (RadTensor* p : { &ta, &tl, &tw[0], &ts[0], &ti[0], &tw[1], &ts[1], &ti[1], &tw[2],
                          &ts[2], &ti[2], &tw[3], &ts[3], &ti[3] })
        rad_tensor_pack(p);
    RadTensor ops[15] = { ta, tw[0], ts[0], ti[0], tw[1], ts[1], ti[1], tw[2], ts[2], ti[2],
                          tw[3], ts[3], ti[3], tv, tl };
    const RadParam params[] = { pint("M", M), pint("n_vocab", n_vocab), pint("n_embd", K),
                                pstr("dtype", "bf16"), pint("vocab_offset", voff),
                                pint("rows0", rows[0]), pint("rows1", rows[1]),
                                pint("rows2", rows[2]), pint("rows3", rows[3]) };
    RadArgs args{};
    args.t = ops; args.n_t = 15; args.p = params; args.n_p = 9; args.kv_used_max_h = -1;
    const int64_t need = lib().logits_scratch(&args);
    void* dscratch = nullptr;
    if (need <= 0 || hipMalloc(&dscratch, (size_t)need) != hipSuccess) return false;
    args.scratch = dscratch;
    args.scratch_bytes = need;
    const int rc = lib().logits(&args, (RadStream)0);
    if (rc != RAD_OK || hipDeviceSynchronize() != hipSuccess) {
        std::fprintf(stderr, "vocab M=%lld launch rc=%d\n", (long long)M, rc);
        return false;
    }
    hipMemcpy(logits.data(), dl, logits.size() * 4, hipMemcpyDeviceToHost);
    if (out) *out = logits;

    /* The f32 reference's rotated activation comes from the same bf16 values the kernel reads. */
    std::vector<float> xr((size_t)(M * K));
    for (int64_t m = 0; m < M; ++m)
        for (int64_t kg = 0; kg < nkb; ++kg) {
            float v[128];
            for (int j = 0; j < 128; ++j)
                v[j] = bf16_to_f32(xb[(size_t)(m * K + kg * 128 + j)]);
            fwht128_host(v);
            std::memcpy(&xr[(size_t)(m * K + kg * 128)], v, sizeof v);
        }
    std::vector<float> want((size_t)(M * n_vocab), 0.0f);
    for (int i = 0; i < 4; ++i)
        reference16(cls[i], xr, M, K, (int32_t)voff, n_vocab, want.data());

    size_t bad = 0, first = want.size();
    double worst_abs = 0.0, worst_rel = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double d = std::fabs((double)logits[i] - (double)want[i]);
        const double tol = (double)kLogitAtol + (double)kLogitRtol * std::fabs((double)want[i]);
        worst_abs = d > worst_abs ? d : worst_abs;
        if (want[i] != 0.0f) {
            const double r = d / std::fabs((double)want[i]);
            worst_rel = r > worst_rel ? r : worst_rel;
        }
        if (d > tol) {
            ++bad;
            if (first == want.size()) first = i;
        }
    }
    if (bad)
        std::fprintf(stderr, "vocab M=%lld: %zu of %zu logits outside tol; first %zu got %.7g "
                             "want %.7g; worst |d| %.3g, worst rel %.3g\n",
                     (long long)M, bad, want.size(), first, logits[first], want[first], worst_abs,
                     worst_rel);
    for (int i = 0; i < 4; ++i) {
        hipFree(db[i]); hipFree(ds[i]); hipFree(di[i]);
    }
    hipFree(dx); hipFree(dl); hipFree(dscratch);
    return bad == 0;
}


/* ------------------------------------------------------------------ a timing pass
 *
 * Run by hand (RAD_IQR_BENCH=1), on the 27B's shapes, for the W0 real/ideal table: the weight is
 * read once a launch, so GEMV's ceiling is VRAM bandwidth; M = 16 adds the ALU term.
 */
void bench_one(int64_t N, int64_t K, int M, int B, int reps, bool tiled = false) {
    const Case c{ "bench", (uint32_t)B, 16, 0, kRowF32, N, K, M };
    W w;
    rad_enc_clear(&w.enc);
    rad_enc_copy_str(w.enc.scheme, "trellis");
    rad_enc_add_plane(&w.enc, "codes", B == 5 ? RAD_U5 : RAD_U4, 1, 1);
    rad_enc_add_plane(&w.enc, "scale", RAD_F32, 1, 0);
    rad_enc_add_plane(&w.enc, "scale.col", RAD_F32, 0, 128);
    const int t = rad_enc_add_table(&w.enc, "trellis", RAD_I8, 1ll << 16, 1);
    w.enc.plane[t].block[0] = 256;
    rad_enc_add_table(&w.enc, "trellis.mad", RAD_U32, 2, 1);
    rad_enc_copy_str(w.enc.transform, "fwht128");
    w.size(N, K);
    for (auto& b : w.data[(size_t)w.find("codes")]) b = (uint8_t)rnd();
    const uint32_t A = 0x00ABCDEFu, Bc = 0x12345678u;
    uint32_t* ab = (uint32_t*)w.data[(size_t)w.find("trellis.mad")].data();
    ab[0] = A;
    ab[1] = Bc;
    int8_t* tab = (int8_t*)w.data[(size_t)w.find("trellis")].data();
    for (uint32_t st = 0; st < (1u << 16); ++st) tab[st] = (int8_t)mad_value(st, A, Bc);
    float* rs = (float*)w.data[(size_t)w.find("scale")].data();
    for (int64_t n = 0; n < N; ++n) rs[n] = 0.001f;
    float* cf = (float*)w.data[(size_t)w.find("scale.col")].data();
    for (int64_t kb = 0; kb < K / 128; ++kb) cf[kb] = 1.0f;

    const std::vector<RadParam> lp = { pint("N", N), pint("K", K), pint("group", 128),
                                       pstr("dtype", "i8a8h") };
    std::vector<int> selB, selS;
    std::vector<RadTensor> plB, plS;
    for (int i = 0; i < w.enc.n_planes; ++i) {
        const char* r = w.enc.plane[i].role;
        if (!std::strncmp(r, "scale", 5)) {
            selS.push_back(i);
            plS.push_back(w.t(i));
        } else {
            selB.push_back(i);
            plB.push_back(w.t(i));
        }
    }
    RadLayout L{}, LS{};
    if (lib().layout(lp.data(), 4, 2, &w.enc, selB.data(), plB.data(), (int)plB.size(), &L) != RAD_OK)
        return;
    std::vector<uint8_t> sb((size_t)L.bytes);
    if (lib().relayout(lp.data(), 4, 2, &w.enc, selB.data(), plB.data(), (int)plB.size(),
                       sb.data(), L.bytes) != RAD_OK)
        return;
    if (lib().layout(lp.data(), 4, 3, &w.enc, selS.data(), plS.data(), (int)plS.size(), &LS) != RAD_OK)
        return;
    std::vector<uint8_t> sbs((size_t)LS.bytes);
    if (lib().relayout(lp.data(), 4, 3, &w.enc, selS.data(), plS.data(), (int)plS.size(),
                       sbs.data(), LS.bytes) != RAD_OK)
        return;

    const int64_t nkb = K / 128;
    std::vector<int8_t> a((size_t)(M * K));
    std::vector<float> as((size_t)(M * nkb));
    for (auto& v : a) v = (int8_t)((int)(rnd() % 255) - 127);
    for (auto& v : as) v = 0.01f;
    void *dA = nullptr, *dS = nullptr, *dB = nullptr, *dBS = nullptr, *dY = nullptr;
    hipMalloc(&dA, a.size());
    hipMalloc(&dS, as.size() * 4);
    hipMalloc(&dB, sb.size());
    hipMalloc(&dBS, sbs.size());
    hipMalloc(&dY, (size_t)(M * N * 2));
    hipMemcpy(dA, a.data(), a.size(), hipMemcpyHostToDevice);
    hipMemcpy(dS, as.data(), as.size() * 4, hipMemcpyHostToDevice);
    hipMemcpy(dB, sb.data(), sb.size(), hipMemcpyHostToDevice);
    hipMemcpy(dBS, sbs.data(), sbs.size(), hipMemcpyHostToDevice);
    RadTensor ta{}, tas{}, tb{}, tbs{}, ty{};
    ta.data = dA; ta.dtype = RAD_I8; ta.rank = 2; ta.shape[0] = M; ta.shape[1] = K;
    tas.data = dS; tas.dtype = RAD_F32; tas.rank = 2; tas.shape[0] = M; tas.shape[1] = nkb;
    tb.data = dB; tb.dtype = L.dtype; tb.rank = 2; tb.shape[0] = N; tb.shape[1] = K;
    tbs.data = dBS; tbs.dtype = LS.dtype; tbs.rank = 2; tbs.shape[0] = N; tbs.shape[1] = nkb;
    ty.data = dY; ty.dtype = RAD_BF16; ty.rank = 2; ty.shape[0] = M; ty.shape[1] = N;
    for (RadTensor* x : { &ta, &tas, &tb, &tbs, &ty }) rad_tensor_pack(x);
    RadTensor ops[7] = { ta, tas, tb, tbs, ty, RadTensor{}, RadTensor{} };
    const RadParam params[] = { pint("M", M), pint("N", N), pint("K", K), pint("group", 128),
                                pstr("dtype", "i8a8h") };
    RadArgs args{};
    args.t = ops; args.n_t = 7; args.p = params; args.n_p = 5; args.kv_used_max_h = -1;
    const LaunchFn run = (tiled && lib().tiled) ? lib().tiled : lib().gemm;
    const char* row = (tiled && lib().tiled) ? "tiled" : "m16";
    /* The row's split arena, so the M = 1 narrow shape prices the path the engine runs. Wide N
     * and M > 1 leave the hook at 0 and the launch on the unsplit kernel. */
    const int splits = (M == 1 && lib().slices) ? lib().slices(N, K) : 1;
    void* dPart = nullptr;
    if (M == 1 && lib().gemm_scratch) {
        const int64_t npart = lib().gemm_scratch(&args);
        if (npart > 0 && hipMalloc(&dPart, (size_t)npart) == hipSuccess) {
            args.scratch = dPart;
            args.scratch_bytes = npart;
        }
    }
    for (int i = 0; i < 5; ++i) run(&args, (RadStream)0);
    hipDeviceSynchronize();
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    std::vector<double> us;
    for (int b = 0; b < 5; ++b) {
        hipEventRecord(e0, 0);
        for (int i = 0; i < reps; ++i) run(&args, (RadStream)0);
        hipEventRecord(e1, 0);
        hipEventSynchronize(e1);
        float ms = 0;
        hipEventElapsedTime(&ms, e0, e1);
        us.push_back(ms * 1000.0 / reps);
    }
    std::sort(us.begin(), us.end());
    const double med = us[us.size() / 2];
    const double bytes = (double)weight_bytes(form(L.dtype), N, K);
    std::printf("%s N=%lld K=%lld M=%d split=%d: %.2f us, %.1f GB/s of weight, %.1f TFLOP/s\n",
                row, (long long)N, (long long)K, M, splits, med, bytes / med / 1e3,
                2.0 * M * N * K / med / 1e6);
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    hipFree(dA); hipFree(dS); hipFree(dB); hipFree(dBS); hipFree(dY); hipFree(dPart);
}

TEST(iqr_vocab_head_matches_its_reference) {
    if (!lib().ok() || !lib().logits || !lib().logits_scratch) return;
    int nd = 0;
    if (hipGetDeviceCount(&nd) != hipSuccess || nd < 1) return;
    CHECK(run_vocab(1));
    CHECK(run_vocab(2));
    CHECK(run_vocab(7));
    /* Four classes of different code widths, one launch: the width dispatch, not just the
     * per-class pointers. */
    const uint32_t mixed[4] = { 2, 3, 5, 7 };
    CHECK(run_vocab(2, nullptr, mixed));
    /* Classes of unequal row counts: grid.x is the widest class's tile count, and the narrower
     * classes' blocks past their rows must return without touching the tile or the logits. */
    const int64_t uneven[4] = { 256, 512, 768, 256 };
    CHECK(run_vocab(2, nullptr, nullptr, uneven));
}

/* The head's two tilings: M = 1 selects MT = 1 (the hi/lo wave split), M = 2 selects MT = 2. The
 * exact integer sums and the f32 fold are the same code and the same order, so the common token's
 * logits must be bit-identical, and both runs must stay within the reference's tolerance. The
 * fixture's four classes are the multi-class case; the second pair gives them different widths. */
TEST(iqr_vocab_head_is_bit_exact_across_mt) {
    if (!lib().ok() || !lib().logits || !lib().logits_scratch) return;
    int nd = 0;
    if (hipGetDeviceCount(&nd) != hipSuccess || nd < 1) return;
    const uint32_t fixed[4] = { 4, 4, 4, 4 };
    const uint32_t mixed[4] = { 2, 3, 5, 7 };
    std::vector<float> one, two, one_mixed, two_mixed;
    CHECK(run_vocab(1, &one, fixed));
    CHECK(run_vocab(2, &two, fixed));
    CHECK(run_vocab(1, &one_mixed, mixed));
    CHECK(run_vocab(2, &two_mixed, mixed));

    auto same = [](const std::vector<float>& a, const std::vector<float>& b, const char* what) {
        const bool shape = a.size() == b.size() / 2;   /* M = 2's first row */
        const bool exact = shape && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        if (!exact) {
            size_t first = a.size();
            for (size_t i = 0; shape && i < a.size(); ++i)
                if (std::memcmp(&a[i], &b[i], sizeof(float))) { first = i; break; }
            std::fprintf(stderr,
                         "vocab %s: M=1 vs M=2 not bit-exact (%zu vs %zu floats), first %zu: "
                         "%.9g vs %.9g\n",
                         what, a.size(), b.size(), first, first < a.size() ? a[first] : 0.0f,
                         first < b.size() ? b[first] : 0.0f);
        }
        return exact;
    };
    CHECK(same(one, two, "four classes of 4 bits"));
    CHECK(same(one_mixed, two_mixed, "classes of 2/3/5/7 bits"));
}

/* The vision tower's dense attention at the height an image asks for. One K row per (segment,
 * head) makes every key's score the same value s, so the maximum is s and every weight is
 * exp(0) = 1. The denominator is the segment length L; with L a power of two and a V row of
 * halves and quarters, k * v[d] and every partial sum are exact in f32, and (L * v[d]) * (1 / L)
 * is v[d]. The store is therefore exactly V, a bit-exact oracle at 16384 rows that forms no host
 * dot product: 2 segments of 8192 rows, 16 heads, 64 query tiles a segment, and max_seqlen below
 * the total row count. The two segments' K and V differ so a walk that crosses a boundary shows. */
TEST(attn_dense_hd72_is_exact_over_a_large_uniform_segment) {
    if (!lib().h || !lib().attn) {
        radtest::skip("attn_dense: the row is absent");
        return;
    }
    int nd = 0;
    if (hipGetDeviceCount(&nd) != hipSuccess || nd < 1) {
        radtest::skip("attn_dense: no device visible");
        return;
    }
    const int hd = 72, nh = 16, seg = 8192, M = 2 * seg;
    const size_t n = (size_t)M * nh * hd;
    std::vector<uint16_t> q(n), k(n), v(n), out(n, 0xffffu);
    for (int s = 0; s < 2; ++s)
        for (int r = s * seg; r < (s + 1) * seg; ++r)
            for (int h = 0; h < nh; ++h)
                for (int d = 0; d < hd; ++d) {
                    const size_t i = ((size_t)r * nh + h) * hd + d;
                    q[i] = rad_f32_to_bf16(0.002f * (float)(int)(rnd() % 1001) - 1.0f);
                    k[i] = rad_f32_to_bf16((float)((d + 2 * h + 5 * s) % 5 - 2) * 0.5f);
                    v[i] = rad_f32_to_bf16((float)((d + 3 * h + 7 * s) % 9 - 4) * 0.25f);
                }
    const int32_t cu[3] = { 0, seg, M };
    void *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr, *dcu = nullptr;
    auto clean = [&] {
        (void)hipFree(dq);
        (void)hipFree(dk);
        (void)hipFree(dv);
        (void)hipFree(dout);
        (void)hipFree(dcu);
    };
    if (hipMalloc(&dq, n * 2) != hipSuccess || hipMalloc(&dk, n * 2) != hipSuccess ||
        hipMalloc(&dv, n * 2) != hipSuccess || hipMalloc(&dout, n * 2) != hipSuccess ||
        hipMalloc(&dcu, sizeof cu) != hipSuccess) {
        clean();
        radtest::skip("attn_dense: no device buffers");
        return;
    }
    const bool copied = hipMemcpy(dq, q.data(), n * 2, hipMemcpyHostToDevice) == hipSuccess &&
                        hipMemcpy(dk, k.data(), n * 2, hipMemcpyHostToDevice) == hipSuccess &&
                        hipMemcpy(dv, v.data(), n * 2, hipMemcpyHostToDevice) == hipSuccess &&
                        hipMemcpy(dout, out.data(), n * 2, hipMemcpyHostToDevice) == hipSuccess &&
                        hipMemcpy(dcu, cu, sizeof cu, hipMemcpyHostToDevice) == hipSuccess;
    RadTensor tq{}, tk{}, tv{}, tcu{}, to{};
    RadTensor* t[5] = { &tq, &tk, &tv, &tcu, &to };
    void* d[5] = { dq, dk, dv, dcu, dout };
    const uint32_t dt[5] = { RAD_BF16, RAD_BF16, RAD_BF16, RAD_I32, RAD_BF16 };
    const int64_t s3[3] = { M, nh, hd }, s1[1] = { 3 };
    for (int i = 0; i < 5; ++i) {
        t[i]->data = d[i];
        t[i]->dtype = dt[i];
        t[i]->rank = i == 3 ? 1 : 3;
        if (i == 3) t[i]->shape[0] = s1[0];
        else {
            t[i]->shape[0] = s3[0];
            t[i]->shape[1] = s3[1];
            t[i]->shape[2] = s3[2];
        }
        rad_tensor_pack(t[i]);
    }
    const RadParam params[] = { pint("head_dim", hd), pint("gqa", 1), pint("causal", 0),
                                pstr("dtype", "bf16"), pint("max_seqlen", seg) };
    RadTensor ops[5] = { tq, tk, tv, tcu, to };
    RadArgs args{};
    args.t = ops;
    args.n_t = 5;
    args.p = params;
    args.n_p = 5;
    args.kv_used_max_h = -1;
    const int rc = copied ? lib().attn(&args, (RadStream)0) : RAD_E_DEVICE;
    const bool ran = rc == RAD_OK && hipDeviceSynchronize() == hipSuccess &&
                     hipMemcpy(out.data(), dout, n * 2, hipMemcpyDeviceToHost) == hipSuccess;
    clean();
    CHECK(ran);
    if (!ran) return;
    size_t bad = 0, first = n;
    for (int s = 0; s < 2; ++s)
        for (int r = s * seg; r < (s + 1) * seg; ++r)
            for (int h = 0; h < nh; ++h)
                for (int d = 0; d < hd; ++d) {
                    const size_t i = ((size_t)r * nh + h) * hd + d;
                    if (out[i] != v[i]) {
                        ++bad;
                        if (first == n) first = i;
                    }
                }
    if (bad) {
        const size_t i = first;
        std::fprintf(stderr,
                     "  attn_dense 16384: %zu of %zu values differ; first row %zu head %zu dim %zu: "
                     "got %g want %g\n",
                     bad, n, i / ((size_t)nh * hd), (i / hd) % nh, i % hd,
                     (double)rad_load_f32(out.data(), RAD_BF16, (int64_t)i),
                     (double)rad_load_f32(v.data(), RAD_BF16, (int64_t)i));
    }
    CHECK(bad == 0);
    if (!bad) std::printf("  attn_dense 16384: %zu uniform-segment values exact\n", n);
}

TEST(iqr_gemm_bench_on_the_27b_shapes_when_asked) {    if (!std::getenv("RAD_IQR_BENCH") || !lib().ok()) return;
    int nd = 0;
    if (hipGetDeviceCount(&nd) != hipSuccess || nd < 1) return;
    /* The report's q|gate shape (u5), then the widest down (u4). */
    bench_one(12288, 2560, 1, 5, 100);
    bench_one(12288, 2560, 16, 5, 100);
    bench_one(5120, 17408, 1, 4, 50);
    bench_one(5120, 17408, 16, 4, 50);
    /* The tiled row on the v_dot4 resident tile (the W13 wave matrix is off by default, measured
     * 25-30 % slower), same shapes. */
    bench_one(12288, 2560, 512, 5, 20, true);
    bench_one(12288, 2560, 4096, 5, 8, true);
    bench_one(5120, 17408, 512, 4, 20, true);
    bench_one(5120, 17408, 4096, 4, 8, true);
}

RAD_TEST_MAIN()
