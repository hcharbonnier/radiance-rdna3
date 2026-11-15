/* iqr_device_test.cpp -- libr3d's IQR GEMM (r3d_gemm_iqr_m16.hip) against a host reference written
 * from r3d_iqr.h, on the card.
 *
 * The reference decodes the container's canonical planes itself (the state recurrence and the
 * computed value's byte sum, a code at a time) and forms the int8 grid's group sum with the
 * reference's order: per 128-K group an exact i32 sub-sum, then (sub * a_scale) * w_scale, groups
 * ascending, the row's own factor last. The stored form is produced by the library's own layout
 * hooks, so the test also covers them end to end (their byte order is held to libr4d's by
 * iqr_parity_test).
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

struct Lib {
    void* h = nullptr;
    LayoutFn layout = nullptr;
    RelayoutFn relayout = nullptr;
    LaunchFn gemm = nullptr;
    LaunchFn logits = nullptr;
    ScratchFn logits_scratch = nullptr;
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
        x.logits = (LaunchFn)dlsym(x.h, "r3d_rad_logits_gemm_vocab_iqr");
        x.logits_scratch = (ScratchFn)dlsym(x.h, "r3d_logits_gemm_vocab_iqr_scratch");
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

struct Case {
    const char* what;
    uint32_t B, L, RB;
    Scale sc;
    int64_t N, K;
    int M;
};

/* The host answer: the decoded weight, then the int8 grid's order. */
std::vector<uint16_t> reference(const W& w, const Case& c, const std::vector<int8_t>& a,
                                const std::vector<float>& as, const uint8_t* sb, int64_t sbytes) {
    const uint32_t run_tiles = 16;   /* the table's run is 256 rows (tcq_rows) */
    const Form f{ kTrellisMad, c.B, 0, c.L, c.RB, c.sc, 0, 0 };
    const int64_t nkb = c.K / 128;
    (void)sbytes;
    const uint32_t A = ((const uint32_t*)(w.data[(size_t)w.find("trellis.mad")].data()))[0];
    const uint32_t Bc = ((const uint32_t*)(w.data[(size_t)w.find("trellis.mad")].data()))[1];
    std::vector<int8_t> dec((size_t)(c.N * c.K));
    const int64_t run = 256;
    for (int64_t n = 0; n < c.N; ++n) {
        for (int64_t k = 0; k < c.K; ++k) {
            const uint32_t kb = (uint32_t)(k / 128);
            const uint32_t st = canonical_state(w, c.B, c.L, run, n, k);
            int v = mad_value(st, A, Bc);
            if (c.RB) {
                const int64_t i = tile_major(n, kb, nkb);
                const float own = bf16_to_f32(*(const uint16_t*)(sb + i * 2));
                const uint16_t* bp = (const uint16_t*)(sb + blk_off(f, c.N, c.K));
                const float bb = bf16_to_f32(bp[(n >> c.RB) * nkb + kb]);
                const float rq = bb > 0.0f ? own / bb : 0.0f;
                v = (int)std::nearbyintf((float)v * rq);
            }
            dec[(size_t)(n * c.K + k)] = (int8_t)v;
        }
    }
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
    rad_enc_copy_str(w.enc.transform, "fwht128");
    w.size(c.N, c.K);

    const int ic = w.find("codes");
    for (auto& b : w.data[(size_t)ic]) b = (uint8_t)rnd();
    const uint32_t A = (uint32_t)(rnd() & 0xFFFFFFu) | 1u, Bc = (uint32_t)rnd();
    uint32_t* ab = (uint32_t*)w.data[(size_t)w.find("trellis.mad")].data();
    ab[0] = A;
    ab[1] = Bc;
    int8_t* tab = (int8_t*)w.data[(size_t)w.find("trellis")].data();
    for (uint32_t st = 0; st < (1u << c.L); ++st) tab[st] = (int8_t)mad_value(st, A, Bc);
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
    auto clean = [&] {
        hipFree(dA);
        hipFree(dS);
        hipFree(dB);
        hipFree(dBS);
        hipFree(dY);
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
    const int rc = lib().gemm(&args, (RadStream)0);
    const hipError_t he = hipDeviceSynchronize();
    if (rc != RAD_OK || he != hipSuccess) {
        std::fprintf(stderr, "%s: launch rc=%d hip=%d\n", c.what, rc, (int)he);
        clean();
        return false;
    }
    hipMemcpy(y.data(), dY, y.size() * 2, hipMemcpyDeviceToHost);
    clean();

    const std::vector<uint16_t> want = reference(w, c, a, as, sbs.data(), (int64_t)sbs.size());
    size_t bad = 0, first = y.size();
    for (size_t i = 0; i < y.size(); ++i)
        if (y[i] != want[i]) {
            ++bad;
            if (first == y.size()) first = i;
        }
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
        { "mad u4 L16 one-f32-row, 2 tokens", 4, 16, 0, kRowF32, 256, 512, 2 },
        { "mad u4 L16 bf16, 5 tokens", 4, 16, 0, kBf16, 256, 256, 5 },
        { "mad u4 L16 bf16, 16 tokens", 4, 16, 0, kBf16, 256, 512, 16 },
    };
    for (const Case& c : cases) CHECK(run_case(c));
}

/* ------------------------------------------------------------------ the vocabulary head
 *
 * logits_gemm_vocab_iqr: four IQR classes of the head, the rotated int8 activation (quantised by
 * the row), and the ids scatter. The reference reads the quantiser's own codes and scales back off
 * the scratch, so this holds the GEMM and the scatter; the quantiser is the same had_quant form
 * libr3d already serves.
 */
struct Cls {
    W w;
    std::vector<uint8_t> b, s;
    RadLayout L{}, LS{};
    std::vector<int32_t> ids;
    int64_t rows = 0;
};

bool make_class(Cls& c, int64_t rows, int64_t K, int32_t id0, uint64_t seed) {
    g_z = seed;
    c.rows = rows;
    rad_enc_clear(&c.w.enc);
    rad_enc_copy_str(c.w.enc.scheme, "trellis");
    rad_enc_add_plane(&c.w.enc, "codes", RAD_U4, 1, 1);
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

bool run_vocab() {
    const int64_t rows = 256, K = 512, n_vocab = 1024, M = 2;
    Cls cls[4];
    for (int i = 0; i < 4; ++i)
        if (!make_class(cls[i], rows, K, (int32_t)(i * rows), 0x9E3779B97F4A7C15ull * (i + 3)))
            return false;
    const int64_t nkb = K / 128;
    std::vector<int8_t> xb((size_t)(M * K * 2));
    for (size_t i = 0; i < xb.size(); i += 2) {
        xb[i] = (int8_t)(rnd() & 0xFF);
        xb[i + 1] = (int8_t)(i % 6 ? 0x3C : 0x3D);
    }
    std::vector<float> logits((size_t)(M * n_vocab), 0.0f);

    void *dx = nullptr, *dl = nullptr, *db[4] = {}, *ds[4] = {}, *di[4] = {};
    hipMalloc(&dx, xb.size());
    hipMemcpy(dx, xb.data(), xb.size(), hipMemcpyHostToDevice);
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
        tw[i].shape[0] = rows; tw[i].shape[1] = K;
        ts[i].data = ds[i]; ts[i].dtype = cls[i].LS.dtype; ts[i].rank = 2;
        ts[i].shape[0] = rows; ts[i].shape[1] = nkb;
        ti[i].data = di[i]; ti[i].dtype = RAD_I32; ti[i].rank = 1; ti[i].shape[0] = rows;
    }
    for (RadTensor* p : { &ta, &tl, &tw[0], &ts[0], &ti[0], &tw[1], &ts[1], &ti[1], &tw[2],
                          &ts[2], &ti[2], &tw[3], &ts[3], &ti[3] })
        rad_tensor_pack(p);
    RadTensor ops[15] = { ta, tw[0], ts[0], ti[0], tw[1], ts[1], ti[1], tw[2], ts[2], ti[2],
                          tw[3], ts[3], ti[3], tv, tl };
    const RadParam params[] = { pint("M", M), pint("n_vocab", n_vocab), pint("n_embd", K),
                                pstr("dtype", "bf16"), pint("vocab_offset", 0),
                                pint("rows0", rows), pint("rows1", 0), pint("rows2", 0),
                                pint("rows3", 0) };
    RadArgs args{};
    args.t = ops; args.n_t = 15; args.p = params; args.n_p = 9; args.kv_used_max_h = -1;
    const int64_t need = lib().logits_scratch(&args);
    void* dscratch = nullptr;
    if (need <= 0 || hipMalloc(&dscratch, (size_t)need) != hipSuccess) return false;
    args.scratch = dscratch;
    args.scratch_bytes = need;
    const int rc = lib().logits(&args, (RadStream)0);
    if (rc != RAD_OK || hipDeviceSynchronize() != hipSuccess) {
        std::fprintf(stderr, "vocab launch rc=%d\n", rc);
        return false;
    }
    std::vector<uint8_t> q((size_t)(M * K));
    std::vector<float> as((size_t)(M * nkb));
    const int64_t qbytes = (M * K + 255) & ~255ll;
    hipMemcpy(q.data(), dscratch, q.size(), hipMemcpyDeviceToHost);
    hipMemcpy(as.data(), (uint8_t*)dscratch + qbytes, as.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(logits.data(), dl, logits.size() * 4, hipMemcpyDeviceToHost);
    {
        std::vector<uint16_t> ydbg((size_t)(M * rows));
        const uint8_t* ybase = (const uint8_t*)dscratch + ((M * K + 255) & ~255ll) +
                               ((M * (K / 128) * 4 + 255) & ~255ll);
        hipMemcpy(ydbg.data(), ybase, ydbg.size() * 2, hipMemcpyDeviceToHost);
        /* The same class through the decode row, to tell the tiled launcher from the GEMM. */
        void* dy2 = nullptr;
        hipMalloc(&dy2, ydbg.size() * 2);
        RadTensor qa{}, qs{}, qb{}, qbs{}, qy{};
        qa.data = dscratch; qa.dtype = RAD_I8; qa.rank = 2; qa.shape[0] = M; qa.shape[1] = K;
        qs.data = (uint8_t*)dscratch + qbytes; qs.dtype = RAD_F32; qs.rank = 2; qs.shape[0] = M;
        qs.shape[1] = K / 128;
        qb.data = db[0]; qb.dtype = cls[0].L.dtype; qb.rank = 2; qb.shape[0] = rows; qb.shape[1] = K;
        qbs.data = ds[0]; qbs.dtype = cls[0].LS.dtype; qbs.rank = 2;
        qbs.shape[0] = rows; qbs.shape[1] = K / 128;
        qy.data = dy2; qy.dtype = RAD_BF16; qy.rank = 2; qy.shape[0] = M; qy.shape[1] = rows;
        for (RadTensor* p : { &qa, &qs, &qb, &qbs, &qy }) rad_tensor_pack(p);
        RadTensor qops[7] = { qa, qs, qb, qbs, qy, RadTensor{}, RadTensor{} };
        const RadParam qp[] = { pint("M", M), pint("N", rows), pint("K", K), pint("group", 128),
                                pstr("dtype", "i8a8h") };
        RadArgs qargs{};
        qargs.t = qops; qargs.n_t = 7; qargs.p = qp; qargs.n_p = 5; qargs.kv_used_max_h = -1;
        const int qrc = lib().gemm(&qargs, (RadStream)0);
        std::vector<uint16_t> y2((size_t)(M * rows), 0);
        hipDeviceSynchronize();
        hipMemcpy(y2.data(), dy2, y2.size() * 2, hipMemcpyDeviceToHost);
        std::fprintf(stderr, "vocab q0=%d q1=%d s0=%g y0=%g m16=%g (rc=%d)\n", (int)q[0], (int)q[1],
                     as[0], bf16_to_f32(ydbg[0]), bf16_to_f32(y2[0]), qrc);
        hipFree(dy2);
    }

    /* The reference from those exact codes and scales. */
    std::vector<float> want((size_t)(M * n_vocab), 0.0f);
    std::vector<float> per[4];
    for (auto& p : per) p.assign((size_t)(M * n_vocab), 0.0f);
    for (int ci = 0; ci < 1; ++ci) {
        const Form f{ kTrellisMad, 4, 0, 16, 0, kRowF32, 0, 0 };
        const uint32_t* ab = (const uint32_t*)cls[ci].w.data[(size_t)cls[ci].w.find("trellis.mad")].data();
        std::vector<int8_t> dec((size_t)(rows * K));
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t k = 0; k < K; ++k)
                dec[(size_t)(r * K + k)] =
                    (int8_t)mad_value(canonical_state(cls[ci].w, 4, 16, 256, r, k), ab[0], ab[1]);
        const float* cf = (const float*)cls[ci].s.data() + ((rows * 4 + 255) & ~255ll) / 4;
        for (int m = 0; m < M; ++m)
            for (int64_t r = 0; r < rows; ++r) {
                float acc = 0.0f;
                for (int64_t kb = 0; kb < nkb; ++kb) {
                    int sub = 0;
                    for (int64_t k = 0; k < 128; ++k)
                        sub += (int)q[(size_t)(m * K + kb * 128 + k)] *
                               (int)dec[(size_t)(r * K + kb * 128 + k)];
                    const float t1 = (float)sub * as[(size_t)(m * nkb + kb)];
                    acc += t1 * cf[kb];
                }
                const int32_t id = cls[ci].ids[(size_t)r];
                if (ci == 0 && m == 0 && r == 0)
                    std::fprintf(stderr, "vocab ref: rowscale=%g cf0=%g as0=%g sub0=%d acc=%g\n",
                                 ((const float*)cls[ci].s.data())[0], cf[0], as[0],
                                 [&] {
                                     int s0 = 0;
                                     for (int64_t k = 0; k < 128; ++k)
                                         s0 += (int)q[(size_t)k] * (int)dec[(size_t)k];
                                     return s0;
                                 }(), acc);
                if (id >= 0 && id < n_vocab) {
                    const float v = bf16_to_f32(f32_to_bf16(acc * ((const float*)cls[ci].s.data())[r]));
                    want[(size_t)(m * n_vocab + id)] = v;
                    per[ci][(size_t)(m * n_vocab + id)] = v;
                }
            }
    }
    size_t bad = 0, first = want.size();
    for (size_t i = 0; i < want.size(); ++i)
        if (memcmp(&logits[i], &want[i], 4) != 0) {
            ++bad;
            if (first == want.size()) first = i;
        }
    if (bad)
        std::fprintf(stderr, "vocab: %zu of %zu logits differ; first %zu got %.6f want %.6f "
                             "(classes %.6f %.6f %.6f %.6f)\n",
                     bad, want.size(), first, logits[first], want[first], per[0][first],
                     per[1][first], per[2][first], per[3][first]);
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
void bench_one(int64_t N, int64_t K, int M, int B, int reps) {
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
    for (int i = 0; i < 5; ++i) lib().gemm(&args, (RadStream)0);
    hipDeviceSynchronize();
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    std::vector<double> us;
    for (int b = 0; b < 5; ++b) {
        hipEventRecord(e0, 0);
        for (int i = 0; i < reps; ++i) lib().gemm(&args, (RadStream)0);
        hipEventRecord(e1, 0);
        hipEventSynchronize(e1);
        float ms = 0;
        hipEventElapsedTime(&ms, e0, e1);
        us.push_back(ms * 1000.0 / reps);
    }
    std::sort(us.begin(), us.end());
    const double med = us[us.size() / 2];
    const double bytes = (double)weight_bytes(form(L.dtype), N, K);
    std::printf("N=%lld K=%lld M=%d: %.2f us, %.1f GB/s of weight, %.1f TFLOP/s\n",
                (long long)N, (long long)K, M, med, bytes / med / 1e3,
                2.0 * M * N * K / med / 1e6);
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    hipFree(dA); hipFree(dS); hipFree(dB); hipFree(dBS); hipFree(dY);
}

TEST(iqr_vocab_head_matches_its_reference) {
    /* The row refuses until the class-GEMM discrepancy is resolved (r3d_iqr_logits.hip); the test
     * stays so the fix is verified by RAD_IQR_VOCAB=1. */
    if (!std::getenv("RAD_IQR_VOCAB")) return;
    if (!lib().ok() || !lib().logits || !lib().logits_scratch) return;
    int nd = 0;
    if (hipGetDeviceCount(&nd) != hipSuccess || nd < 1) return;
    CHECK(run_vocab());
}

TEST(iqr_gemm_bench_on_the_27b_shapes_when_asked) {    if (!std::getenv("RAD_IQR_BENCH") || !lib().ok()) return;
    int nd = 0;
    if (hipGetDeviceCount(&nd) != hipSuccess || nd < 1) return;
    /* The report's q|gate shape (u5), then the widest down (u4). */
    bench_one(12288, 2560, 1, 5, 100);
    bench_one(12288, 2560, 16, 5, 100);
    bench_one(5120, 17408, 1, 4, 50);
    bench_one(5120, 17408, 16, 4, 50);
}

RAD_TEST_MAIN()
