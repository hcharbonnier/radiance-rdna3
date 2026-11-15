/* moe_iqr_device_test.cpp -- libr3d's routed IQR GEMM (r3d_moe_iqr.hip) against a host reference
 * built from r3d_iqr.h, on the card.
 *
 * The host side generates the container's canonical trellis planes (codes, the computed table's
 * A and B, the one-f32-a-row scales and the per-128 block factors), runs them through the
 * library's MoE layout hooks, and forms the int8 grid's sums in the kernel's order: per 128-K
 * block an exact i32 sub-sum, then (sub * a_scale) * fold in f32, blocks ascending, the weight
 * row's own factor once at the store. The comparison is bit-exact: both sides decode the same
 * codes and round the same two multiplies.
 *
 * What it covers, and what it deliberately leaves to the reference test:
 *   - both grids: the 16-row decode band (runs of one or two rows, a tile straddling experts, a
 *     run longer than a decode tile, a wide run) and the run map above it (a run of up to 64 rows
 *     reused over four token tiles);
 *   - top_k 1/2/3, token and sorted a, n_expert from 1 to 12, zero-row experts, dropped slots,
 *     parts = 1 and 2, a_group 128 and 256, B = 2..7 (L = 16), N = 64/128/256.
 * It does not re-check the layout byte order (iqr_device_test and iqr_parity_test own that); it
 * uses the hooks as the loader does.
 *
 * The library is dlopened, as the engine loads it: the test links only the ABI and HIP runtime.
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
    LaunchFn moe = nullptr;
    ScratchFn scratch = nullptr;
    bool ok() const { return h && layout && relayout && moe; }
};

Lib& lib() {
    static Lib l = [] {
        Lib x;
        x.h = dlopen(RAD_R3D_SO, RTLD_NOW | RTLD_LOCAL);
        if (!x.h) return x;
        x.layout = (LayoutFn)dlsym(x.h, "r3d_rad_layout_moe_iqr");
        x.relayout = (RelayoutFn)dlsym(x.h, "r3d_rad_relayout_moe_iqr");
        x.moe = (LaunchFn)dlsym(x.h, "r3d_rad_moe_gemm_iqr");
        x.scratch = (ScratchFn)dlsym(x.h, "r3d_rad_moe_gemm_iqr_scratch");
        return x;
    }();
    return l;
}

uint64_t g_z = 0x5DEECE66Dull;
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

inline int mad_value(uint32_t st, uint32_t a, uint32_t b) {
    const uint32_t h = ((st & 0xFFFFFFu) * (a & 0xFFFFFFu) + b) & 0x3F3F3F3Fu;
    return (int)((h & 0xFFu) + ((h >> 8) & 0xFFu) + ((h >> 16) & 0xFFu) + (h >> 24)) - 126;
}

const uint32_t kU[] = { 0, RAD_U1, RAD_U2, RAD_U3, RAD_U4, RAD_U5, RAD_U6, RAD_U7 };

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

uint32_t bits_at(const uint8_t* p, int64_t o, int n) {
    const uint8_t* b = p + (o >> 3);
    const int sh = (int)(o & 7);
    const int nb = (sh + n + 7) >> 3;
    uint64_t w = 0;
    for (int i = 0; i < nb; ++i) w |= (uint64_t)b[i] << (8 * i);
    return (uint32_t)((w >> sh) & ((1ull << n) - 1ull));
}

uint32_t code_at(const W& w, int64_t r, int64_t c, uint32_t B) {
    const int i = w.find("codes");
    return bits_at(w.data[(size_t)i].data() + r * w.row_bytes(i), c * B, (int)B);
}

/* The decoded weight [N, K] from the canonical codes, advancing one code at a time inside a run
 * (the tail-biting seed is the run's last span codes). */
std::vector<int8_t> decode_weight(const W& w, uint32_t B, uint32_t L, int64_t run, int64_t N,
                                  int64_t K) {
    const uint32_t A = ((const uint32_t*)w.data[(size_t)w.find("trellis.mad")].data())[0];
    const uint32_t Bc = ((const uint32_t*)w.data[(size_t)w.find("trellis.mad")].data())[1];
    const uint32_t mask = (1u << L) - 1u;
    const int64_t nspan = (L + B - 1) / B;
    std::vector<int8_t> dec((size_t)(N * K));
    for (int64_t base = 0; base < N; base += run) {
        const int64_t len = (N - base < run) ? N - base : run;
        for (int64_t k = 0; k < K; ++k) {
            uint32_t st = 0;
            for (int64_t j = 0; j < nspan; ++j)
                st = ((st << B) | code_at(w, base + run - nspan + j, k, B)) & mask;
            for (int64_t j = 0; j < len; ++j) {
                st = ((st << B) | code_at(w, base + j, k, B)) & mask;
                dec[(size_t)((base + j) * K + k)] = (int8_t)mad_value(st, A, Bc);
            }
        }
    }
    return dec;
}

/* One case's geometry and data distribution. */
struct Case {
    const char* what;
    uint32_t B, lrun;      /* code bits; log2 rows a trellis run */
    int64_t N, K, a_group;
    int64_t n_expert, top_k, M;   /* T = M * top_k */
    int64_t parts;
    bool a_sorted;
    int dist;              /* 0 interleaved (decode), 1 contiguous ranges (prefill), 2 one hot run */
    int64_t drop;          /* dropped slots at the tail */
};

std::vector<int8_t> build_ids(const Case& c, int64_t T) {
    std::vector<int8_t> id((size_t)T, 0);
    for (int64_t f = 0; f < T; ++f) {
        if (f >= T - c.drop) {
            id[(size_t)f] = -1;
            continue;
        }
        const int64_t live = T - c.drop;
        int64_t e;
        if (c.dist == 1) {
            e = (f * c.n_expert) / live;
        } else if (c.dist == 2 && c.n_expert > 1) {
            /* Expert 0 takes all but a handful: a run longer than a decode tile. */
            const int64_t small = std::min<int64_t>(c.n_expert - 1, live / 4);
            e = f < live - small ? 0 : 1 + (f - (live - small)) % (c.n_expert - 1);
        } else {
            e = f % c.n_expert;
        }
        id[(size_t)f] = (int8_t)e;
    }
    return id;
}

void build_sorted(const Case& c, const std::vector<int8_t>& id, std::vector<int32_t>* sorted,
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

int expert_of(const std::vector<int32_t>& off, int64_t row) {
    for (size_t e = 0; e + 1 < off.size(); ++e)
        if (off[e] <= row && row < off[e + 1]) return (int)e;
    return -1;
}

/* The host answer: one row per sorted slot, an exact i32 group sub-sum and the same two f32
 * multiplies the kernel rounds. */
std::vector<uint16_t> reference(const Case& c, const std::vector<W>& ws,
                                const std::vector<std::vector<int8_t>>& dec,
                                const std::vector<int8_t>& a, const std::vector<float>& as,
                                const std::vector<int32_t>& sorted,
                                const std::vector<int32_t>& off) {
    const int64_t T = (int64_t)sorted.size(), nkb = c.K / 128, nAs = c.K / c.a_group;
    std::vector<uint16_t> y((size_t)(T * c.N), 0);
    for (int64_t i = 0; i < T; ++i) {
        const int32_t f = sorted[(size_t)i];
        if (f < 0) continue;   /* already zero */
        const int e = expert_of(off, i);
        if (e < 0) continue;
        const int64_t tok = c.a_sorted ? i : (int64_t)f / c.top_k;
        const int64_t nbase = i * c.N;
        for (int64_t n = 0; n < c.N; ++n) {
            float acc = 0.0f;
            for (int64_t kb = 0; kb < nkb; ++kb) {
                int sub = 0;
                for (int64_t k = 0; k < 128; ++k)
                    sub += (int)a[(size_t)(tok * c.K + kb * 128 + k)] *
                           (int)dec[(size_t)e][(size_t)(n * c.K + kb * 128 + k)];
                const uint32_t ash = c.a_group == 256 ? 1u : 0u;
                const float sa = as[(size_t)(tok * nAs + (kb >> ash))];
                const float sb = ((const float*)ws[(size_t)e]
                                      .data[(size_t)ws[(size_t)e].find("scale.col")]
                                      .data())[kb];
                const float t = (float)sub * sa;
                acc += t * sb;
            }
            const float rs = ((const float*)ws[(size_t)e]
                                  .data[(size_t)ws[(size_t)e].find("scale")]
                                  .data())[n];
            y[(size_t)(nbase + n)] = f32_to_bf16(acc * rs);
        }
    }
    return y;
}

bool run_case(const Case& c) {
    if (!lib().ok()) {
        radtest::artifact_missing(__FILE__, __LINE__, "libr3d's MoE IQR symbols");
        return true;
    }
    const int64_t T = c.M * c.top_k;
    const int64_t nkb = c.K / 128, nAs = c.K / c.a_group;
    const int64_t run = 1ll << c.lrun;
    const int64_t neven = (c.n_expert + 1) / 2, nodd = c.n_expert / 2;

    std::vector<W> ws((size_t)c.n_expert);
    std::vector<std::vector<int8_t>> dec((size_t)c.n_expert);
    const uint32_t A = (uint32_t)(rnd() & 0xFFFFFFu) | 1u, Bc = (uint32_t)rnd();
    for (int64_t e = 0; e < c.n_expert; ++e) {
        W& w = ws[(size_t)e];
        rad_enc_clear(&w.enc);
        rad_enc_copy_str(w.enc.scheme, "trellis");
        rad_enc_add_plane(&w.enc, "codes", kU[c.B], 1, 1);
        rad_enc_add_plane(&w.enc, "scale", RAD_F32, 1, 0);
        rad_enc_add_plane(&w.enc, "scale.col", RAD_F32, 0, 128);
        const int t = rad_enc_add_table(&w.enc, "trellis", RAD_I8, (int64_t)1 << 16, 1);
        w.enc.plane[t].block[0] = 1 << c.lrun;
        w.enc.plane[t].block[1] = 0;
        rad_enc_add_table(&w.enc, "trellis.mad", RAD_U32, 2, 1);
        rad_enc_copy_str(w.enc.transform, "fwht128");
        w.size(c.N, c.K);

        for (auto& b : w.data[(size_t)w.find("codes")]) b = (uint8_t)rnd();
        uint32_t* ab = (uint32_t*)w.data[(size_t)w.find("trellis.mad")].data();
        ab[0] = A;
        ab[1] = Bc;
        int8_t* tab = (int8_t*)w.data[(size_t)w.find("trellis")].data();
        for (uint32_t st = 0; st < (1u << 16); ++st) tab[st] = (int8_t)mad_value(st, A, Bc);
        float* rs = (float*)w.data[(size_t)w.find("scale")].data();
        for (int64_t n = 0; n < c.N; ++n) rs[n] = 0.001f + 0.0005f * (float)((n + e) % 7);
        float* cf = (float*)w.data[(size_t)w.find("scale.col")].data();
        for (int64_t kb = 0; kb < nkb; ++kb) cf[kb] = 0.5f + 0.1f * (float)((kb + e) % 5);
        dec[(size_t)e] = decode_weight(w, c.B, 16, run, c.N, c.K);
    }

    /* The layout hooks, at the classes' geometry, produce exactly the loader's buffers. */
    const std::vector<RadParam> lp = {
        pint("M", c.M), pint("N", c.N), pint("K", c.K), pint("n_expert", c.n_expert),
        pint("top_k", c.top_k), pint("group", 128), pstr("a_order", c.a_sorted ? "sorted" : "token"),
        pstr("dtype", "i8a8h"), pint("N_odd", c.N), pint("K_odd", c.K), pint("parts", c.parts),
        pint("a_group", c.a_group),
    };
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

    std::vector<int8_t> id = build_ids(c, T);
    std::vector<int32_t> sorted, off;
    build_sorted(c, id, &sorted, &off);

    const int64_t arows = c.a_sorted ? T : c.M;
    std::vector<int8_t> a((size_t)(arows * c.K));
    std::vector<float> as((size_t)(arows * nAs));
    for (auto& v : a) v = (int8_t)((int)(rnd() % 255) - 127);
    for (auto& v : as) v = 0.01f + 0.001f * (float)(rnd() % 7);

    std::vector<uint16_t> want = reference(c, ws, dec, a, as, sorted, off);
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

    /* Per class, one device buffer an entry and a device array of their pointers. */
    const int64_t cnt[2] = { neven, nodd };
    for (int q = 0; q < 2; ++q) {
        std::vector<void*> hp;
        for (int64_t i = 0; i < cnt[q]; ++i) {
            void* p = nullptr;
            if (hipMalloc(&p, whost[q][(size_t)i].size()) != hipSuccess) { clean(); return false; }
            (void)hipMemcpy(p, whost[q][(size_t)i].data(), whost[q][(size_t)i].size(), hipMemcpyHostToDevice);
            dW[q].push_back(p);
            hp.push_back(p);
        }
        std::vector<void*> hs;
        for (int64_t i = 0; i < cnt[q]; ++i) {
            void* p = nullptr;
            if (hipMalloc(&p, shost[q][(size_t)i].size()) != hipSuccess) { clean(); return false; }
            (void)hipMemcpy(p, shost[q][(size_t)i].data(), shost[q][(size_t)i].size(), hipMemcpyHostToDevice);
            dSc[q].push_back(p);
            hs.push_back(p);
        }
        if (cnt[q] > 0) {
            if (hipMalloc(&dWtab[q], (size_t)cnt[q] * sizeof(void*)) != hipSuccess) { clean(); return false; }
            if (hipMalloc(&dStab[q], (size_t)cnt[q] * sizeof(void*)) != hipSuccess) { clean(); return false; }
            (void)hipMemcpy(dWtab[q], hp.data(), (size_t)cnt[q] * sizeof(void*), hipMemcpyHostToDevice);
            (void)hipMemcpy(dStab[q], hs.data(), (size_t)cnt[q] * sizeof(void*), hipMemcpyHostToDevice);
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
        ts[q].shape[2] = nkb;
        ts[q].stride[0] = 0;
        ts[q].stride[1] = nkb;
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
        pstr("dtype", "i8a8h"), pint("N_odd", c.N), pint("K_odd", c.K), pint("parts", c.parts),
        pint("a_group", c.a_group),
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
    for (size_t i = 0; i < y.size(); ++i) {
        if (y[i] != want[i]) {
            ++bad;
            if (first == y.size()) first = i;
        }
    }
    if (bad) {
        const size_t i = first;
        std::fprintf(stderr, "%s: %zu of %zu outputs differ; first [%lld,%lld] got %04x want %04x\n",
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

TEST(moe_iqr_decode_interleaved) {
    CHECK(run_case(Case{ "decode interleaved", 4, 6, 128, 256, 128, 8, 2, 6, 1, 0, 0,
                             3 }));
}
TEST(moe_iqr_decode_sorted_order) {
    CHECK(run_case(Case{ "decode sorted a", 5, 6, 128, 256, 128, 12, 3, 4, 1, 0, 0, 2 }));
}
TEST(moe_iqr_decode_parts) {
    CHECK(run_case(Case{ "decode parts=2 a_group=256", 4, 7, 256, 512, 256, 8, 2, 4, 2, 0,
                             0, 3 }));
}
TEST(moe_iqr_decode_two_bit) {
    CHECK(run_case(Case{ "decode B=2 N=64", 2, 6, 64, 128, 128, 4, 1, 2, 1, 0, 0, 1 }));
}
TEST(moe_iqr_decode_all_dropped) {
    CHECK(run_case(Case{ "decode all dropped", 4, 6, 128, 128, 128, 4, 1, 4, 1, 0, 0, 4 }));
}
TEST(moe_iqr_decode_cross_tile_run) {
    CHECK(run_case(Case{ "decode cross-tile run", 4, 6, 128, 256, 128, 20, 1, 30, 1, 0, 2, 0 }));
}
TEST(moe_iqr_prefill_block_runs) {
    CHECK(run_case(Case{ "prefill block runs", 4, 6, 128, 384, 128, 8, 2, 64, 1, 0, 1, 5 }));
}
TEST(moe_iqr_prefill_parts) {
    CHECK(run_case(Case{ "prefill parts=2 a_group=256", 5, 7, 256, 512, 256, 4, 2, 40, 2, 0,
                             1, 3 }));
}
TEST(moe_iqr_prefill_single_expert) {
    CHECK(run_case(Case{ "prefill one expert", 4, 6, 128, 256, 128, 1, 1, 16, 1, 0, 1, 0 }));
}
TEST(moe_iqr_prefill_all_dropped) {
    CHECK(run_case(Case{ "prefill all dropped", 3, 6, 128, 128, 128, 4, 1, 32, 1, 0, 1, 32 }));
}
TEST(moe_iqr_prefill_seven_bit) {
    CHECK(run_case(Case{ "prefill B=7", 7, 6, 128, 256, 128, 6, 2, 32, 1, 0, 1, 2 }));
}
/* The model's two routed shapes: gate_up (K 2560, N 512 a rank, parts 2) and down (N 2560,
 * K 384). Wide K is 20 scale groups; down is the column count and the narrower K. */
TEST(moe_iqr_decode_gate_up_shape) {
    CHECK(run_case(Case{ "decode gate_up shape", 4, 6, 512, 2560, 128, 8, 2, 4, 2, 0, 0, 2 }));
}
TEST(moe_iqr_decode_down_shape) {
    CHECK(run_case(Case{ "decode down shape", 4, 7, 2560, 384, 128, 8, 2, 4, 1, 0, 0, 2 }));
}
TEST(moe_iqr_prefill_gate_up_shape) {
    CHECK(run_case(Case{ "prefill gate_up shape", 4, 6, 512, 2560, 128, 8, 2, 64, 2, 0, 1,
                         5 }));
}

RAD_TEST_MAIN()
