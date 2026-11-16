/* moe_w4nl_device_test.cpp -- libr3d's routed w4nl codebook GEMM (r3d_moe_w4nl.hip) against a host
 * reference built from the container's four canonical planes, on the card.
 *
 * The host generates an expert as the container stores it (docs/MOE-W4.md):
 *
 *   codes   u4 [1 x 1]        row-major packed, element 2i low
 *   scale   E4M3 [1 x 64]     one scale a (row, 64 of K)
 *   scale.1 f32  [1 x 1]      the fixed second level, 2^-13
 *   table   f32  [1 x 16]     the w4nl levels
 *
 * runs the planes through the library's MoE layout hooks (which join codes+table and
 * scale+scale.1), and forms the reference the way libref's `quant_dot` does: per 64-K group, the
 * f32 sum of `a_int8 * table[code]`, then `(sub * a_scale) * (w_scale * level)`, groups ascending,
 * the two multiplies and the add rounded separately. The comparison is bit-exact. The kernel's
 * exact-integer-eighths sub-sum rounds identically by construction (r3d_moe_w4nl.h), so a mismatch
 * is a real defect, not a tolerance.
 *
 * Coverage: both model geometries (down N=2560 K=640 parts 1; MTP N=1280 K=2560 parts 2), the
 * decode band and the run map, top_k 1/2/10, E=4/8 and the MTP's 48, a 512-expert decode case,
 * token and sorted a, a_group 128/256, zero activations, zero-row experts and dropped slots.
 * The layout's value gate (a table or level other than the w4nl one) is checked separately.
 *
 * The library is dlopened, as the engine loads it: the test links only the ABI and HIP runtime.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include "r3d_codecs.h"
#include "r3d_moe_w4nl.h"

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
    LaunchFn moe = nullptr;
    ScratchFn scratch = nullptr;
    bool ok() const { return h && layout && relayout && moe; }
};

Lib& lib() {
    static Lib l = [] {
        Lib x;
        x.h = dlopen(RAD_R3D_SO, RTLD_NOW | RTLD_LOCAL);
        if (!x.h) return x;
        x.layout = (LayoutFn)dlsym(x.h, "r3d_rad_layout_moe_w4nl");
        x.relayout = (RelayoutFn)dlsym(x.h, "r3d_rad_relayout_moe_w4nl");
        x.moe = (LaunchFn)dlsym(x.h, "r3d_rad_moe_gemm_w4nl");
        x.scratch = (ScratchFn)dlsym(x.h, "r3d_rad_moe_gemm_w4nl_scratch");
        return x;
    }();
    return l;
}

uint64_t g_z = 0x243F6A8885A308D3ull;
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

/* One expert's canonical container planes. */
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

void make_encoding(W* w) {
    rad_enc_clear(&w->enc);
    rad_enc_copy_str(w->enc.scheme, "affine");
    rad_enc_copy_str(w->enc.transform, "fwht128");
    rad_enc_add_plane(&w->enc, "codes", RAD_U4, 1, 1);
    rad_enc_add_plane(&w->enc, "scale", RAD_F8E4M3, 1, r3d_w4nl::kGroup);
    rad_enc_add_plane(&w->enc, "scale.1", RAD_F32, 0, 0);
    rad_enc_add_table(&w->enc, "table", RAD_F32, 1, 16);
}

/* The host answer, in libref's `quant_dot` order: per 64-K group the f32 sum of a_int8 * table, then
 * (sub * a_scale) * (w_scale * level), groups ascending, the two multiplies and the add rounded
 * separately. The 512-expert case reuses one generated weight: routing and the decode are what it
 * exercises. */
std::vector<uint16_t> reference(const std::vector<W>& ws, int64_t N, int64_t K, int64_t a_group,
                                const std::vector<int8_t>& a, const std::vector<float>& as,
                                const std::vector<int32_t>& sorted, const std::vector<int32_t>& off,
                                int64_t top_k, bool a_sorted) {
    const int64_t T = (int64_t)sorted.size(), nAs = K / a_group, ng = K / r3d_w4nl::kGroup;
    const uint32_t ash = a_group == 256 ? 2u : 1u;
    std::vector<uint16_t> y((size_t)(T * N), 0);
    for (int64_t i = 0; i < T; ++i) {
        const int32_t f = sorted[(size_t)i];
        if (f < 0) continue;
        int64_t e = -1;
        for (int64_t q = 0; q + 1 < (int64_t)off.size(); ++q)
            if (off[(size_t)q] <= i && i < off[(size_t)q + 1]) { e = q; break; }
        if (e < 0) continue;
        const W& w = ws[(size_t)e];
        const int ci = w.find("codes"), si = w.find("scale"), li = w.find("scale.1"),
                  ti = w.find("table");
        const uint8_t* codes = w.data[(size_t)ci].data();
        const uint8_t* sc = w.data[(size_t)si].data();
        const float lvl = ((const float*)w.data[(size_t)li].data())[0];
        const float* tab = (const float*)w.data[(size_t)ti].data();
        const int64_t srb = w.row_bytes(si);
        const int64_t crb = w.row_bytes(ci);
        const int64_t tok = a_sorted ? i : (int64_t)f / top_k;
        for (int64_t n = 0; n < N; ++n) {
            const uint8_t* crow = codes + n * crb;
            float acc = 0.0f;
            for (int64_t g = 0; g < ng; ++g) {
                float sub = 0.0f;
                for (int64_t k = 0; k < r3d_w4nl::kGroup; ++k) {
                    const uint8_t byte = crow[(size_t)((g * r3d_w4nl::kGroup + k) >> 1)];
                    const uint8_t code =
                        (uint8_t)((k & 1) ? (byte >> 4) : (byte & 15));
                    sub += (float)a[(size_t)(tok * K + g * r3d_w4nl::kGroup + k)] * tab[code];
                }
                const float sa = as[(size_t)(tok * nAs + (g >> ash))];
                const float sb = e4m3_to_f32(sc[(size_t)(n * srb + g)]) * lvl;
                acc += (sub * sa) * sb;
            }
            y[(size_t)(i * N + n)] = f32_to_bf16(acc);
        }
    }
    return y;
}

struct Case {
    const char* what;
    int64_t N, K, n_expert, top_k, M, parts, a_group;
    bool a_sorted, zero_a;
    int dist;          /* 0 interleaved, 1 contiguous ranges, 2 one wide run */
    int64_t drop;      /* dropped slots at the tail */
    bool share_codes;  /* the 512-expert case reuses one generated weight */
};

std::vector<int32_t> build_ids(const Case& c, int64_t T) {
    std::vector<int32_t> id((size_t)T, 0);
    const int64_t live = T - c.drop;
    for (int64_t f = 0; f < T; ++f) {
        if (f >= live) { id[(size_t)f] = -1; continue; }
        int64_t e;
        if (c.dist == 1) e = (f * c.n_expert) / live;
        else if (c.dist == 2 && c.n_expert > 1) {
            const int64_t small = std::min<int64_t>(c.n_expert - 1, live / 4);
            e = f < live - small ? 0 : 1 + (f - (live - small)) % (c.n_expert - 1);
        } else e = f % c.n_expert;
        id[(size_t)f] = (int32_t)e;
    }
    return id;
}

void build_sorted(const Case& c, const std::vector<int32_t>& id, std::vector<int32_t>* sorted,
                  std::vector<int32_t>* off) {
    const int64_t T = (int64_t)id.size();
    off->assign((size_t)c.n_expert + 1, 0);
    for (int64_t f = 0; f < T; ++f)
        if (id[(size_t)f] >= 0) ++(*off)[(size_t)id[(size_t)f] + 1];
    for (int64_t e = 0; e < c.n_expert; ++e) (*off)[(size_t)e + 1] += (*off)[(size_t)e];
    sorted->assign((size_t)T, -1);
    std::vector<int32_t> pos = *off;
    for (int64_t f = 0; f < T; ++f)
        if (id[(size_t)f] >= 0) (*sorted)[(size_t)pos[(size_t)id[(size_t)f]]++] = (int32_t)f;
}

/* Fill one expert's canonical planes. The table and level are the w4nl values; the scales are finite
 * E4M3 encodings, so the reference never meets a NaN. */
void fill_expert(W* w, int64_t N, int64_t K, int64_t e, const std::vector<uint8_t>* shared_codes) {
    make_encoding(w);
    w->size(N, K);
    const int ci = w->find("codes"), si = w->find("scale"), li = w->find("scale.1"),
              ti = w->find("table");
    auto& codes = w->data[(size_t)ci];
    if (shared_codes && shared_codes->size() == codes.size()) {
        codes = *shared_codes;
    } else {
        for (size_t i = 0; i < codes.size(); ++i)
            codes[i] = (uint8_t)(rnd() >> ((i & 7) * 8));
    }
    float* tab = (float*)w->data[(size_t)ti].data();
    for (int i = 0; i < 16; ++i) tab[i] = r3d_w4nl::kTable[i];
    ((float*)w->data[(size_t)li].data())[0] = r3d_w4nl::kLevel;
    uint8_t* sc = w->data[(size_t)si].data();
    const int64_t srb = w->row_bytes(si);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t g = 0; g < K / r3d_w4nl::kGroup; ++g)
            sc[(size_t)(n * srb + g)] =
                f32_to_e4m3(0.0007f + 0.0002f * (float)((n + g + e) % 9));
}

bool run_case(const Case& c) {
    if (!lib().ok()) {
        radtest::artifact_missing(__FILE__, __LINE__, "libr3d's MoE w4nl symbols");
        return true;
    }
    const int64_t T = c.M * c.top_k;
    const int64_t nAs = c.K / c.a_group, nkb = c.K / 128;
    const int64_t neven = (c.n_expert + 1) / 2, nodd = c.n_expert / 2;

    std::vector<W> ws((size_t)c.n_expert);
    std::vector<uint8_t> shared;
    if (c.share_codes) {
        W base;
        make_encoding(&base);
        base.size(c.N, c.K);
        shared.assign((size_t)(c.N * (c.K / 2)), 0);
        for (size_t i = 0; i < shared.size(); ++i) shared[i] = (uint8_t)(rnd() >> ((i & 7) * 8));
    }
    for (int64_t e = 0; e < c.n_expert; ++e)
        fill_expert(&ws[(size_t)e], c.N, c.K, e, c.share_codes ? &shared : nullptr);

    const std::vector<RadParam> lp = {
        pint("M", c.M), pint("N", c.N), pint("K", c.K), pint("n_expert", c.n_expert),
        pint("top_k", c.top_k), pint("group", 128), pstr("a_order", c.a_sorted ? "sorted" : "token"),
        pstr("dtype", "w4nl64a8i8h"), pint("N_odd", c.N), pint("K_odd", c.K),
        pint("parts", c.parts), pint("a_group", c.a_group),
    };

    /* The layout hooks, at the classes' geometry, produce exactly the loader's buffers. */
    std::vector<std::vector<uint8_t>> whost[2], shost[2];
    uint32_t wdt = 0, sdt = 0;
    for (int64_t e = 0; e < c.n_expert; ++e) {
        W& w = ws[(size_t)e];
        std::vector<int> selB, selS;
        std::vector<RadTensor> plB, plS;
        for (int i = 0; i < w.enc.n_planes; ++i) {
            const char* role = w.enc.plane[i].role;
            if (!std::strncmp(role, "scale", 5)) {
                selS.push_back(i);
                plS.push_back(w.t(i));
            } else {
                selB.push_back(i);
                plB.push_back(w.t(i));
            }
        }
        const int cls = (int)(e & 1);
        const int wop = cls ? 6 : 2, sop = cls ? 7 : 3;
        RadLayout Lw{}, Ls{};
        int rc = lib().layout(lp.data(), (int)lp.size(), wop, &w.enc, selB.data(), plB.data(),
                              (int)plB.size(), &Lw);
        if (rc != RAD_OK) {
            std::fprintf(stderr, "%s: weight layout refused rc=%d expert=%lld\n", c.what, rc,
                         (long long)e);
            return false;
        }
        rc = lib().layout(lp.data(), (int)lp.size(), sop, &w.enc, selS.data(), plS.data(),
                          (int)plS.size(), &Ls);
        if (rc != RAD_OK) {
            std::fprintf(stderr, "%s: scale layout refused rc=%d expert=%lld\n", c.what, rc,
                         (long long)e);
            return false;
        }
        wdt = Lw.dtype;
        sdt = Ls.dtype;
        std::vector<uint8_t> wb((size_t)Lw.bytes), sb((size_t)Ls.bytes);
        if (lib().relayout(lp.data(), (int)lp.size(), wop, &w.enc, selB.data(), plB.data(),
                           (int)plB.size(), wb.data(), Lw.bytes) != RAD_OK ||
            lib().relayout(lp.data(), (int)lp.size(), sop, &w.enc, selS.data(), plS.data(),
                           (int)plS.size(), sb.data(), Ls.bytes) != RAD_OK) {
            std::fprintf(stderr, "%s: relayout refused expert=%lld\n", c.what, (long long)e);
            return false;
        }
        whost[cls].push_back(std::move(wb));
        shost[cls].push_back(std::move(sb));
    }
    if (wdt != RAD_U4 || sdt != RAD_F8E4M3) {
        std::fprintf(stderr, "%s: joined dtypes %u/%u, expected u4/e4m3\n", c.what, wdt, sdt);
        return false;
    }

    std::vector<int32_t> id = build_ids(c, T), sorted, off;
    build_sorted(c, id, &sorted, &off);

    const int64_t arows = c.a_sorted ? T : c.M;
    std::vector<int8_t> a((size_t)(arows * c.K));
    std::vector<float> as((size_t)(arows * nAs));
    if (c.zero_a) {
        std::fill(a.begin(), a.end(), 0);
    } else {
        for (auto& v : a) v = (int8_t)((int)(rnd() % 255) - 127);
    }
    for (auto& v : as) v = 0.01f + 0.001f * (float)(rnd() % 7);

    std::vector<uint16_t> want = reference(ws, c.N, c.K, c.a_group, a, as, sorted, off, c.top_k,
                                           c.a_sorted);
    std::vector<uint16_t> y((size_t)(T * c.N), 0xDEADu);

    /* Device copies. */
    void *dA = nullptr, *dS = nullptr, *dY = nullptr;
    std::vector<void*> dW[2], dSc[2];
    void* dWtab[2] = { nullptr, nullptr };
    void* dStab[2] = { nullptr, nullptr };
    void* dMap = nullptr;
    auto clean = [&] {
        (void)hipFree(dA);
        (void)hipFree(dS);
        (void)hipFree(dY);
        for (int q = 0; q < 2; ++q)
            for (void* p : dW[q]) (void)hipFree(p);
        for (int q = 0; q < 2; ++q)
            for (void* p : dSc[q]) (void)hipFree(p);
        for (int q = 0; q < 2; ++q) (void)hipFree(dWtab[q]);
        for (int q = 0; q < 2; ++q) (void)hipFree(dStab[q]);
        (void)hipFree(dMap);
    };
    if (hipMalloc(&dA, a.size()) != hipSuccess) return false;
    if (hipMalloc(&dS, as.size() * 4) != hipSuccess) { clean(); return false; }
    if (hipMalloc(&dY, y.size() * 2) != hipSuccess) { clean(); return false; }
    (void)hipMemcpy(dA, a.data(), a.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(dS, as.data(), as.size() * 4, hipMemcpyHostToDevice);
    (void)hipMemset(dY, 0, y.size() * 2);

    const int64_t cnt[2] = { neven, nodd };
    for (int q = 0; q < 2; ++q) {
        std::vector<void*> hp, hs;
        for (int64_t i = 0; i < cnt[q]; ++i) {
            void* p = nullptr;
            if (hipMalloc(&p, whost[q][(size_t)i].size()) != hipSuccess) { clean(); return false; }
            (void)hipMemcpy(p, whost[q][(size_t)i].data(), whost[q][(size_t)i].size(),
                            hipMemcpyHostToDevice);
            dW[q].push_back(p);
            hp.push_back(p);
        }
        for (int64_t i = 0; i < cnt[q]; ++i) {
            void* p = nullptr;
            if (hipMalloc(&p, shost[q][(size_t)i].size()) != hipSuccess) { clean(); return false; }
            (void)hipMemcpy(p, shost[q][(size_t)i].data(), shost[q][(size_t)i].size(),
                            hipMemcpyHostToDevice);
            dSc[q].push_back(p);
            hs.push_back(p);
        }
        if (cnt[q] > 0) {
            if (hipMalloc(&dWtab[q], (size_t)cnt[q] * sizeof(void*)) != hipSuccess) {
                clean();
                return false;
            }
            if (hipMalloc(&dStab[q], (size_t)cnt[q] * sizeof(void*)) != hipSuccess) {
                clean();
                return false;
            }
            (void)hipMemcpy(dWtab[q], hp.data(), (size_t)cnt[q] * sizeof(void*),
                            hipMemcpyHostToDevice);
            (void)hipMemcpy(dStab[q], hs.data(), (size_t)cnt[q] * sizeof(void*),
                            hipMemcpyHostToDevice);
        }
    }

    RadTensor ta{}, tas{}, tw[2]{}, ts[2]{}, tsort{}, toff{}, ty{};
    ta.data = dA; ta.dtype = RAD_I8; ta.rank = 2; ta.shape[0] = arows; ta.shape[1] = c.K;
    rad_tensor_pack(&ta);
    tas.data = dS; tas.dtype = RAD_F32; tas.rank = 2; tas.shape[0] = arows; tas.shape[1] = nAs;
    rad_tensor_pack(&tas);
    for (int q = 0; q < 2; ++q) {
        tw[q].data = dWtab[q];
        tw[q].dtype = wdt;
        tw[q].rank = 3;
        tw[q].shape[0] = cnt[q];
        tw[q].shape[1] = c.N;
        tw[q].shape[2] = c.K;
        tw[q].stride[0] = 0;
        tw[q].stride[1] = c.K;
        tw[q].stride[2] = 1;
        ts[q].data = dStab[q];
        ts[q].dtype = sdt;
        ts[q].rank = 3;
        ts[q].shape[0] = cnt[q];
        ts[q].shape[1] = c.N;
        ts[q].shape[2] = nkb * 2;
        ts[q].stride[0] = 0;
        ts[q].stride[1] = nkb * 2;
        ts[q].stride[2] = 1;
    }
    void *dSorted = nullptr, *dOff = nullptr;
    if (hipMalloc(&dSorted, sorted.size() * 4) != hipSuccess) { clean(); return false; }
    if (hipMalloc(&dOff, off.size() * 4) != hipSuccess) { clean(); return false; }
    (void)hipMemcpy(dSorted, sorted.data(), sorted.size() * 4, hipMemcpyHostToDevice);
    (void)hipMemcpy(dOff, off.data(), off.size() * 4, hipMemcpyHostToDevice);
    tsort.data = dSorted; tsort.dtype = RAD_I32; tsort.rank = 1; tsort.shape[0] = T;
    rad_tensor_pack(&tsort);
    toff.data = dOff; toff.dtype = RAD_I32; toff.rank = 1; toff.shape[0] = c.n_expert + 1;
    rad_tensor_pack(&toff);
    ty.data = dY; ty.dtype = RAD_BF16; ty.rank = 2; ty.shape[0] = T; ty.shape[1] = c.N;
    rad_tensor_pack(&ty);

    RadTensor ops[9] = { ta, tas, tw[0], ts[0], tsort, toff, tw[1], ts[1], ty };
    const std::vector<RadParam> ap = {
        pint("M", c.M), pint("N", c.N), pint("K", c.K), pint("n_expert", c.n_expert),
        pint("top_k", c.top_k), pint("group", 128), pstr("a_order", c.a_sorted ? "sorted" : "token"),
        pstr("dtype", "w4nl64a8i8h"), pint("N_odd", c.N), pint("K_odd", c.K),
        pint("parts", c.parts), pint("a_group", c.a_group),
    };
    RadArgs args{};
    args.t = ops;
    args.n_t = 9;
    args.p = ap.data();
    args.n_p = (int)ap.size();
    args.kv_used_max_h = -1;

    const bool band = 2 * T <= 3 * c.n_expert;
    int64_t want_scratch = 0;
    if (lib().scratch) want_scratch = lib().scratch(&args);
    if (band) {
        if (want_scratch != 0) {
            std::fprintf(stderr, "%s: decode band must ask for no scratch (%lld)\n", c.what,
                         (long long)want_scratch);
            clean();
            return false;
        }
    } else {
        const int64_t cap = c.n_expert + (T + 63) / 64 + 1;
        const int64_t need = (1 + 2 * cap) * 4;
        if (want_scratch < need) {
            std::fprintf(stderr, "%s: map scratch hook answered %lld, needs %lld\n", c.what,
                         (long long)want_scratch, (long long)need);
            clean();
            return false;
        }
        if (hipMalloc(&dMap, (size_t)want_scratch) != hipSuccess) { clean(); return false; }
        (void)hipMemset(dMap, 0x7F, (size_t)want_scratch);
        args.scratch = dMap;
        args.scratch_bytes = want_scratch;
    }

    const int rc = lib().moe(&args, (RadStream)0);
    const hipError_t he = hipDeviceSynchronize();
    if (rc != RAD_OK || he != hipSuccess) {
        std::fprintf(stderr, "%s: launch rc=%d hip=%d\n", c.what, rc, (int)he);
        clean();
        return false;
    }
    (void)hipMemcpy(y.data(), dY, y.size() * 2, hipMemcpyDeviceToHost);
    clean();

    size_t bad = 0, first = y.size();
    for (size_t i = 0; i < y.size(); ++i)
        if (y[i] != want[i]) {
            ++bad;
            if (first == y.size()) first = i;
        }
    if (bad) {
        const size_t i = first;
        std::fprintf(stderr,
                     "%s: %zu of %zu outputs differ; first [%lld,%lld] got %04x want %04x\n",
                     c.what, bad, y.size(), (long long)(i / c.N), (long long)(i % c.N), y[i],
                     want[i]);
        return false;
    }
    std::printf("%s: %lld slots, %lld experts, N %lld K %lld, %lld bf16 exact\n", c.what,
                (long long)T, (long long)c.n_expert, (long long)c.N, (long long)c.K,
                (long long)y.size());
    return true;
}

}  // namespace

/* ---- the down projections: N 2560, K 640, parts 1 ------------------------------------------- */
TEST(moe_w4nl_decode_top1) {
    CHECK(run_case(Case{ "down decode top1", 2560, 640, 4, 1, 4, 1, 128, false, false, 0, 0, false }));
}
TEST(moe_w4nl_decode_top10_sorted) {
    CHECK(run_case(Case{ "down decode top10 sorted", 2560, 640, 8, 10, 1, 1, 128, true, false, 0, 1,
                         false }));
}
TEST(moe_w4nl_decode_zero_act_zero_runs) {
    CHECK(run_case(Case{ "down decode zero a, zero-row experts", 2560, 640, 8, 2, 2, 1, 128, false,
                         true, 1, 0, false }));
}
TEST(moe_w4nl_prefill_top2) {
    CHECK(run_case(Case{ "down prefill top2", 2560, 640, 8, 2, 16, 1, 128, false, false, 1, 2,
                         false }));
}
TEST(moe_w4nl_prefill_all_dropped) {
    CHECK(run_case(Case{ "down prefill all dropped", 2560, 640, 4, 1, 8, 1, 128, false, false, 1, 8,
                         false }));
}
TEST(moe_w4nl_decode_512_experts) {
    CHECK(run_case(Case{ "down decode E=512", 2560, 640, 512, 1, 64, 1, 128, false, false, 0, 0,
                         true }));
}

/* ---- the MTP head: N 1280, K 2560, parts 2, 48 experts -------------------------------------- */
TEST(moe_w4nl_mtp_decode_top2) {
    CHECK(run_case(Case{ "MTP decode top2", 1280, 2560, 48, 2, 4, 2, 128, false, false, 2, 0, false }));
}
TEST(moe_w4nl_mtp_prefill_top2) {
    CHECK(run_case(Case{ "MTP prefill top2", 1280, 2560, 48, 2, 64, 2, 128, false, false, 1, 5,
                         false }));
}
TEST(moe_w4nl_mtp_prefill_top10) {
    CHECK(run_case(Case{ "MTP prefill top10", 1280, 2560, 48, 10, 8, 2, 128, true, false, 0, 3,
                         false }));
}
TEST(moe_w4nl_mtp_decode_parts2_a_group256) {
    CHECK(run_case(Case{ "MTP decode parts2 a_group256", 1280, 2560, 8, 2, 2, 2, 256, false, false,
                         1, 0, false }));
}

/* ---- the layout's value gate ----------------------------------------------------------------- */
TEST(moe_w4nl_layout_refuses_foreign_table) {
    if (!lib().ok()) {
        radtest::artifact_missing(__FILE__, __LINE__, "libr3d's MoE w4nl layout hooks");
        return;
    }
    const int64_t N = 128, K = 256;
    W w;
    fill_expert(&w, N, K, 0, nullptr);
    const std::vector<RadParam> lp = {
        pint("M", 4), pint("N", N), pint("K", K), pint("n_expert", 2), pint("top_k", 1),
        pint("group", 128), pstr("a_order", "token"), pstr("dtype", "w4nl64a8i8h"),
        pint("N_odd", N), pint("K_odd", K), pint("parts", 1), pint("a_group", 128),
    };
    std::vector<int> selB, selS;
    std::vector<RadTensor> plB, plS;
    for (int i = 0; i < w.enc.n_planes; ++i) {
        if (!std::strncmp(w.enc.plane[i].role, "scale", 5)) {
            selS.push_back(i);
            plS.push_back(w.t(i));
        } else {
            selB.push_back(i);
            plB.push_back(w.t(i));
        }
    }
    RadLayout Lw{};
    CHECK(lib().layout(lp.data(), (int)lp.size(), 2, &w.enc, selB.data(), plB.data(),
                       (int)plB.size(), &Lw) == RAD_OK);
    std::vector<uint8_t> wb((size_t)Lw.bytes);
    CHECK(lib().relayout(lp.data(), (int)lp.size(), 2, &w.enc, selB.data(), plB.data(),
                         (int)plB.size(), wb.data(), Lw.bytes) == RAD_OK);

    float* tab = (float*)w.data[(size_t)w.find("table")].data();
    tab[3] += 0.5f;   /* not the w4nl table */
    CHECK(lib().relayout(lp.data(), (int)lp.size(), 2, &w.enc, selB.data(), plB.data(),
                         (int)plB.size(), wb.data(), Lw.bytes) == RAD_E_DTYPE);
    tab[3] -= 0.5f;

    RadLayout Ls{};
    CHECK(lib().layout(lp.data(), (int)lp.size(), 3, &w.enc, selS.data(), plS.data(),
                       (int)plS.size(), &Ls) == RAD_OK);
    std::vector<uint8_t> sb((size_t)Ls.bytes);
    float* lvl = (float*)w.data[(size_t)w.find("scale.1")].data();
    lvl[0] = 0x1p-12f;   /* not 2^-13 */
    CHECK(lib().relayout(lp.data(), (int)lp.size(), 3, &w.enc, selS.data(), plS.data(),
                         (int)plS.size(), sb.data(), Ls.bytes) == RAD_E_DTYPE);
}

RAD_TEST_MAIN()
