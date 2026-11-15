/* moe_iqr_test.cpp -- moe_gemm_iqr (libr3d/r3d_moe_iqr.hip, R4) against the host reference in
 * tests/iqr_ref.h, and the reference itself at the shapes W5 must cover.
 *
 * The reference decodes each expert's canonical IQR planes to the int8 grid and runs the int8
 * GEMM's order (ADR-4): exact i32 sub-sums on 128-K groups, then
 * (float)sub * a_scale * w_block_scale in f32 ascending K-block order, and a kRowF32 weight's
 * row scale after the K loop. libr3d's row does not exist yet on this branch; the card case is
 * compiled and skipped at run time when the row's symbol is absent, as iqr_device_test does for
 * the rows it probes. The host cases always run and cover the reference's contract: the sorted
 * run map, parity classes with N_odd/K_odd, parts, a narrower class's zero fill, zero-row
 * experts, dropped slots, and moe_gather's ascending-slot top-k sum.
 *
 * Tolerance on the card: two bf16 ulps. The kernel batches its int8 sums through the
 * 12582912-biased fma the WMMA and dot forms share, so its f32 association differs from the
 * reference's plain (sub * a_scale) * w_scale even when both are exact; two steps cover a second
 * boundary flip, as tests/iqr_device_test.cpp's split-K case does. The bounds are reported for
 * every shape, per ADR-4.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include "iqr_ref.h"
#include "r3d_codecs.h"
#include "r3d_iqr.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef RAD_R3D_SO
#define RAD_R3D_SO "libr3d.so"
#endif

namespace {

using namespace r3d_iqr;
using namespace iqr_ref;

typedef int (*LayoutFn)(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                        int, RadLayout*);
typedef int (*RelayoutFn)(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                          int, void*, int64_t);
typedef int (*UnrelayoutFn)(const RadParam*, int, int, const RadEncoding*, const int*, const void*,
                            int64_t, const RadTensor*, int);
typedef int (*LaunchFn)(const RadArgs*, RadStream);
typedef int64_t (*ScratchFn)(const RadArgs*);

struct Lib {
    void* h = nullptr;
    LayoutFn layout = nullptr;
    RelayoutFn relayout = nullptr;
    UnrelayoutFn unrelayout = nullptr;
    LaunchFn gemm = nullptr;
    ScratchFn scratch = nullptr;
    bool host_ok() const { return h && layout && relayout; }
    bool card_ok() const { return host_ok() && gemm; }
};

Lib& lib() {
    static Lib l = [] {
        Lib x;
        x.h = dlopen(RAD_R3D_SO, RTLD_NOW | RTLD_LOCAL);
        if (!x.h) return x;
        x.layout = (LayoutFn)dlsym(x.h, "r3d_rad_layout_moe_iqr");
        x.relayout = (RelayoutFn)dlsym(x.h, "r3d_rad_relayout_moe_iqr");
        x.unrelayout = (UnrelayoutFn)dlsym(x.h, "r3d_rad_unrelayout_moe_iqr");
        x.gemm = (LaunchFn)dlsym(x.h, "r3d_rad_moe_gemm_iqr");
        x.scratch = (ScratchFn)dlsym(x.h, "r3d_rad_moe_gemm_iqr_scratch");
        return x;
    }();
    return l;
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

struct Rng {
    uint64_t z;
    explicit Rng(uint64_t seed) : z(seed) {}
    uint64_t next() {
        z += 0x9E3779B97F4A7C15ull;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    }
    uint32_t u32() { return (uint32_t)(next() >> 32); }
};

constexpr int kMoeMaxUlps = 2;

struct Plan {
    const char* what;
    int64_t E, top_k, M, parts;
    int64_t N[2], K[2];
    bool sorted_a;
    int64_t drop;
    bool zero_rows;
    bool all_to_first;
    uint64_t seed;
};

/* One synthetic layer: the routing and activation, the canonical weights and the laid-out scale
 * buffers the reference and the card case share. */
struct Built {
    Plan plan{};
    int64_t ncls = 2;
    MoeGeom g{};
    std::vector<int32_t> sorted, off;
    std::vector<int8_t> a;
    std::vector<float> as, gate;
    std::vector<W> w;
    std::vector<std::vector<uint8_t>> wq, wsc;
    std::vector<RadLayout> lw, ls;
    std::vector<MoeExpert> ex;
    int64_t Kmax = 0;
};

bool excluded(const Plan& p, int64_t e) {
    return p.zero_rows && (e == 1 || e == p.E - 1);
}

bool build(const Plan& p, Built* b) {
    b->plan = p;
    b->ncls = 2;
    const int64_t E = p.E, top_k = p.top_k, M = p.M, T = M * top_k;
    if (E < 2 || top_k <= 0 || top_k > E || M <= 0 || (p.N[0] <= 0) || (p.N[1] <= 0) ||
        p.N[0] % 256 || p.N[1] % 256 || p.K[0] % 128 || p.K[1] % 128 || p.parts <= 0 ||
        p.N[0] % p.parts || p.N[1] % p.parts || (p.N[0] / p.parts) % 256 ||
        (p.N[1] / p.parts) % 256 || p.drop < 0 || p.drop > T)
        return false;
    if (p.zero_rows && p.top_k > E - 2) return false;

    b->g.n_expert = E;
    b->g.top_k = top_k;
    b->g.ncls = 2;
    b->g.parts = p.parts;
    b->g.M = M;
    b->g.T = T;
    b->g.N[0] = p.N[0];
    b->g.N[1] = p.N[1];
    b->g.K[0] = p.K[0];
    b->g.K[1] = p.K[1];
    b->g.sorted_a = p.sorted_a;
    b->g.a_group = 128;
    b->Kmax = std::max(p.K[0], p.K[1]);

    /* The routing: top_k distinct experts a token, the last `drop` slots dropped. */
    Rng rnd(p.seed);
    std::vector<std::vector<int32_t>> rows_of((size_t)E);
    for (int64_t t = 0; t < M; ++t) {
        std::vector<int64_t> used;
        for (int64_t j = 0; j < top_k; ++j) {
            const int64_t flat = t * top_k + j;
            if (flat >= T - p.drop) continue;
            int64_t e;
            if (p.all_to_first) {
                e = 0;
            } else {
                do {
                    e = (int64_t)(rnd.u32() % (uint32_t)E);
                } while (std::find(used.begin(), used.end(), e) != used.end() || excluded(p, e));
            }
            used.push_back(e);
            rows_of[(size_t)e].push_back((int32_t)flat);
        }
    }
    b->sorted.assign((size_t)T, -1);
    b->off.assign((size_t)E + 1, 0);
    {
        int64_t o = 0;
        for (int64_t e = 0; e < E; ++e) {
            b->off[(size_t)e] = (int32_t)o;
            for (int32_t f : rows_of[(size_t)e]) b->sorted[(size_t)o++] = f;
        }
        b->off[(size_t)E] = (int32_t)o;
    }

    /* The activation: one row a token (gate_up's a_order) or one a sorted slot (down's). */
    const int64_t Arows = p.sorted_a ? T : M;
    b->a.assign((size_t)(Arows * b->Kmax), 0);
    b->as.assign((size_t)(Arows * (b->Kmax / 128)), 0.0f);
    for (auto& v : b->a) v = (int8_t)((int)(rnd.u32() % 255) - 127);
    for (auto& v : b->as) v = std::ldexp(1.0f + (float)(rnd.u32() % 256) / 256.0f, -7);
    b->gate.assign((size_t)T, 0.0f);
    for (auto& v : b->gate) v = 0.5f + (float)(rnd.u32() % 256) / 256.0f;

    /* Each expert's canonical planes, then the class hook's laid-out weight and scale buffers. */
    const std::vector<RadParam> lp = {
        pint("N", p.N[0]), pint("K", p.K[0]), pint("N_odd", p.N[1]), pint("K_odd", p.K[1]),
        pint("group", 128), pstr("dtype", "i8a8h"), pint("parts", p.parts), pint("a_group", 128),
    };
    b->w.resize((size_t)E);
    b->wq.resize((size_t)E);
    b->wsc.resize((size_t)E);
    b->lw.assign(2, RadLayout{});
    b->ls.assign(2, RadLayout{});
    b->ex.resize((size_t)E);
    for (int64_t e = 0; e < E; ++e) {
        const int64_t q = e & 1;
        MadTrellis f;
        f.bits = 4;
        f.L = 16;
        f.lrb = 0;
        f.run = 256;
        f.scale = kRowF32;
        make_mad_trellis(b->w[(size_t)e], p.N[q], p.K[q], f, 0x9E3779B97F4A7C15ull * (uint64_t)(e + 3) +
                                                              p.seed);
        W& w = b->w[(size_t)e];
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
        const int opB = q ? 6 : 2, opS = q ? 7 : 3;
        RadLayout lw{}, ls{};
        int rc = lib().layout(lp.data(), (int)lp.size(), opB, &w.enc, selB.data(), plB.data(),
                              (int)plB.size(), &lw);
        if (rc == RAD_OK)
            rc = lib().layout(lp.data(), (int)lp.size(), opS, &w.enc, selS.data(), plS.data(),
                              (int)plS.size(), &ls);
        if (rc != RAD_OK) return false;
        b->lw[(size_t)q] = lw;
        b->ls[(size_t)q] = ls;
        b->wq[(size_t)e].assign((size_t)lw.bytes, 0);
        b->wsc[(size_t)e].assign((size_t)ls.bytes, 0);
        rc = lib().relayout(lp.data(), (int)lp.size(), opB, &w.enc, selB.data(), plB.data(),
                            (int)plB.size(), b->wq[(size_t)e].data(), lw.bytes);
        if (rc == RAD_OK)
            rc = lib().relayout(lp.data(), (int)lp.size(), opS, &w.enc, selS.data(), plS.data(),
                                (int)plS.size(), b->wsc[(size_t)e].data(), ls.bytes);
        if (rc != RAD_OK) return false;
        MoeExpert x;
        x.w = &b->w[(size_t)e];
        x.sb = b->wsc[(size_t)e].data();
        x.N = p.N[q];
        x.K = p.K[q];
        x.form = f;
        b->ex[(size_t)e] = x;
    }
    return true;
}

std::vector<uint16_t> reference(const Built& b, std::vector<uint8_t>* live = nullptr) {
    std::vector<uint16_t> y;
    if (!moe_gemm_reference(b.ex, b.g, b.a, b.as, b.sorted, b.off, &y, live)) return {};
    return y;
}

/* ================================================================== the host cases */

/* A few computed cells recomputed without the class/part mapping: the decoded expert's own row n,
 * the activation row the sorted row names, the reference's int8-grid order. */
int bad_reference_cells(const Built& b, const std::vector<uint16_t>& y) {
    const Plan& p = b.plan;
    const int64_t Nmax = std::max(p.N[0], p.N[1]), fy = Nmax / p.parts;
    int bad = 0;
    for (int64_t i = 0; i < b.g.T; i += 3) {
        const int64_t e = expert_of(b.off, p.E, i);
        if (e < 0) continue;
        const int64_t q = e & 1, Nc = p.N[q], Kc = p.K[q], nkb = Kc / 128;
        const int64_t src = p.sorted_a ? i : b.sorted[(size_t)i] / p.top_k;
        if (src < 0 || src >= (p.sorted_a ? b.g.T : p.M)) continue;
        const MoeExpert& x = b.ex[(size_t)e];
        const Form f{ kTrellisMad, x.form.bits, 0, x.form.L, x.form.lrb, x.form.scale, 0, 0 };
        const std::vector<int8_t> dec =
            decode_rows(*x.w, x.form.bits, x.form.L, x.form.lrb, x.form.scale, Nc, Kc, x.sb, nkb,
                        x.form.run);
        const float* cf = (const float*)(x.sb + cf_off(f, Nc, Kc));
        const float* rs = (const float*)x.sb;
        const int8_t* ad = b.a.data() + src * b.Kmax;
        const float* asd = b.as.data() + src * (b.Kmax / b.g.a_group);
        for (int64_t n : { int64_t(0), Nc / 2, Nc - 1 }) {
            float acc = 0.0f;
            for (int64_t kb = 0; kb < nkb; ++kb) {
                int32_t sub = 0;
                for (int64_t k = 0; k < 128; ++k)
                    sub += (int32_t)ad[kb * 128 + k] * (int32_t)dec[(size_t)(n * Kc + kb * 128 + k)];
                acc += ((float)sub * asd[(kb * 128) / b.g.a_group]) * cf[kb];
            }
            acc *= rs[n];
            const int64_t part = n / (Nc / p.parts), local = n % (Nc / p.parts);
            bad += y[(size_t)(i * Nmax + part * fy + local)] != f32_to_bf16(acc);
        }
    }
    return bad;
}

TEST(moe_iqr_reference_covers_runs_zero_rows_and_topk) {
    Plan p{ "host contract", 8, 2, 3, 1, { 256, 512 }, { 256, 128 }, false, 2, true, false,
            0x51A7E123u };
    Built b;
    if (!lib().host_ok()) {
        radtest::artifact_missing(__FILE__, __LINE__, "libr3d's moe_gemm_iqr layout hooks");
        return;
    }
    CHECK(build(p, &b));
    std::vector<uint8_t> live;
    const std::vector<uint16_t> y = reference(b, &live);
    CHECK(!y.empty());
    CHECK_EQ((int64_t)b.off.size(), p.E + 1);

    /* The run map: offsets monotone, the prefix is a permutation of the live slots, the tail is
     * -1. Experts 1 and E-1 were kept out of the routing and own no row. */
    bool monotone = true;
    for (int64_t e = 0; e < p.E; ++e) monotone &= b.off[(size_t)e] <= b.off[(size_t)e + 1];
    CHECK(monotone);
    std::vector<int> seen((size_t)(p.M * p.top_k), 0);
    int64_t tail = 0;
    for (int64_t e = 0; e < p.E; ++e) {
        if (e == 1 || e == p.E - 1) CHECK_EQ(b.off[(size_t)e], b.off[(size_t)e + 1]);
        for (int64_t i = b.off[(size_t)e]; i < b.off[(size_t)e + 1]; ++i) {
            const int32_t f = b.sorted[(size_t)i];
            CHECK(f >= 0 && f < p.M * p.top_k);
            if (f >= 0 && f < p.M * p.top_k) ++seen[(size_t)f];
        }
    }
    for (int64_t i = b.off[(size_t)p.E]; i < p.M * p.top_k; ++i) {
        CHECK_EQ(b.sorted[(size_t)i], -1);
        ++tail;
    }
    CHECK_EQ(tail, p.drop);
    int dup = 0;
    for (int v : seen) dup += v > 1;
    CHECK_EQ(dup, 0);

    /* A row an expert owns is computed where its class is wide; a narrower class's part is zero
     * past its width; a dropped tail row is zero everywhere. */
    const int64_t Nmax = std::max(p.N[0], p.N[1]), fy = Nmax / p.parts;
    int bad_fill = 0, bad_live = 0;
    for (int64_t i = 0; i < b.g.T; ++i) {
        const int64_t e = expert_of(b.off, p.E, i);
        if (e < 0) {
            for (int64_t n = 0; n < Nmax; ++n) {
                bad_fill += y[(size_t)(i * Nmax + n)] != 0;
                bad_live += live[(size_t)(i * Nmax + n)] != 0;
            }
            continue;
        }
        const int64_t q = e & 1, fc = p.N[q] / p.parts;
        for (int64_t part = 0; part < p.parts; ++part)
            for (int64_t j = 0; j < fy; ++j) {
                const int64_t at = i * Nmax + part * fy + j;
                if (j < fc) bad_live += live[(size_t)at] != 1;
                else {
                    bad_fill += y[(size_t)at] != 0;
                    bad_live += live[(size_t)at] != 0;
                }
            }
    }
    CHECK_EQ(bad_fill, 0);
    CHECK_EQ(bad_live, 0);

    /* Pick a few computed cells and recompute them without the class/part mapping: the decoded
     * expert's own row n, the activation row the sorted row names, the same int8-grid order. */
    CHECK_EQ(bad_reference_cells(b, y), 0);

    /* The token-level sum is moe_gather's: ascending slot, every term multiplied by the slot's
     * gate weight. Recompute it with a direct slot scan. */
    std::vector<float> sum;
    CHECK(moe_gather_reference(y, b.g, b.sorted, b.off, b.gate, &sum));
    std::vector<float> want((size_t)(p.M * Nmax), 0.0f);
    for (int64_t m = 0; m < p.M; ++m)
        for (int64_t j = 0; j < p.top_k; ++j) {
            const int64_t slot = m * p.top_k + j;
            int64_t row = -1;
            for (int64_t i = 0; i < p.M * p.top_k; ++i)
                if (b.sorted[(size_t)i] == slot && expert_of(b.off, p.E, i) >= 0) row = i;
            if (row < 0) continue;
            for (int64_t n = 0; n < Nmax; ++n)
                want[(size_t)(m * Nmax + n)] +=
                    bf16_to_f32(y[(size_t)(row * Nmax + n)]) * b.gate[(size_t)slot];
        }
    int bad_sum = 0;
    for (size_t i = 0; i < sum.size(); ++i) bad_sum += sum[i] != want[i];
    CHECK_EQ(bad_sum, 0);
}

TEST(moe_iqr_reference_sorted_rows_and_parts) {
    if (!lib().host_ok()) {
        radtest::artifact_missing(__FILE__, __LINE__, "libr3d's moe_gemm_iqr layout hooks");
        return;
    }
    Plan p{ "sorted rows, two parts", 4, 2, 2, 2, { 512, 512 }, { 256, 256 }, true, 0, false, false,
            0xBEEF1234u };
    Built b;
    CHECK(build(p, &b));
    std::vector<uint8_t> live;
    const std::vector<uint16_t> y = reference(b, &live);
    CHECK(!y.empty());
    CHECK_EQ(bad_reference_cells(b, y), 0);
    const int64_t Nmax = 512, fy = Nmax / p.parts;
    int bad_live = 0;
    for (int64_t i = 0; i < b.g.T; ++i)
        for (int64_t part = 0; part < p.parts; ++part)
            for (int64_t j = 0; j < fy; ++j)
                bad_live += live[(size_t)(i * Nmax + part * fy + j)] != 1;
    CHECK_EQ(bad_live, 0);
}

TEST(moe_iqr_layout_round_trips_both_classes) {
    if (!lib().host_ok() || !lib().unrelayout) {
        radtest::artifact_missing(__FILE__, __LINE__, "libr3d's moe_gemm_iqr layout hooks");
        return;
    }
    Plan p{ "layout round trip", 4, 2, 1, 1, { 256, 512 }, { 256, 128 }, false, 0, false, false,
            0x1234ABCDu };
    Built b;
    CHECK(build(p, &b));
    const std::vector<RadParam> lp = {
        pint("N", p.N[0]), pint("K", p.K[0]), pint("N_odd", p.N[1]), pint("K_odd", p.K[1]),
        pint("group", 128), pstr("dtype", "i8a8h"), pint("parts", p.parts), pint("a_group", 128),
    };
    int bad = 0;
    for (int64_t e = 0; e < std::min<int64_t>(2, p.E); ++e) {
        W& w = b.w[(size_t)e];
        const int q = (int)(e & 1);
        const int opB = q ? 6 : 2, opS = q ? 7 : 3;
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
        std::vector<std::vector<uint8_t>> want = w.data;
        int rc = lib().unrelayout(lp.data(), (int)lp.size(), opB, &w.enc, selB.data(),
                                  b.wq[(size_t)e].data(), (int64_t)b.wq[(size_t)e].size(),
                                  plB.data(), (int)plB.size());
        if (rc == RAD_OK)
            rc = lib().unrelayout(lp.data(), (int)lp.size(), opS, &w.enc, selS.data(),
                                  b.wsc[(size_t)e].data(), (int64_t)b.wsc[(size_t)e].size(),
                                  plS.data(), (int)plS.size());
        CHECK_EQ(rc, RAD_OK);
        for (size_t i = 0; i < want.size(); ++i) bad += want[i] != w.data[i];
    }
    CHECK_EQ(bad, 0);
}

/* ================================================================== the card */

struct Dev {
    std::vector<void*> bufs;
    ~Dev() {
        for (void* p : bufs) (void)hipFree(p);
    }
    void* alloc(size_t n) {
        void* p = nullptr;
        if (hipMalloc(&p, n) != hipSuccess) return nullptr;
        bufs.push_back(p);
        return p;
    }
};

void to_dev(void* dst, const void* src, size_t n) {
    (void)hipMemcpy(dst, src, n, hipMemcpyHostToDevice);
}
void from_dev(void* dst, const void* src, size_t n) {
    (void)hipMemcpy(dst, src, n, hipMemcpyDeviceToHost);
}

/* A device buffer padded with a canary on both sides: the row must not read or write past the
 * plan the layout hook states (CHECKLIST 2.7's "aucun accès hors plan"). */
constexpr size_t kGuardBytes = 256;
constexpr uint8_t kGuardFill = 0xA5;

struct Guarded {
    void* base = nullptr;
    uint8_t* p = nullptr;
    size_t bytes = 0;
};

Guarded galloc(Dev& dev, size_t bytes) {
    Guarded g;
    g.base = dev.alloc(bytes + 2 * kGuardBytes);
    if (!g.base) return g;
    g.p = (uint8_t*)g.base + kGuardBytes;
    g.bytes = bytes;
    std::vector<uint8_t> fill(bytes + 2 * kGuardBytes, kGuardFill);
    to_dev(g.base, fill.data(), fill.size());
    return g;
}

bool guards_intact(const Guarded& g) {
    if (!g.base) return false;
    std::vector<uint8_t> v(g.bytes + 2 * kGuardBytes);
    from_dev(v.data(), g.base, v.size());
    for (size_t i = 0; i < kGuardBytes; ++i)
        if (v[i] != kGuardFill) return false;
    for (size_t i = kGuardBytes + g.bytes; i < v.size(); ++i)
        if (v[i] != kGuardFill) return false;
    return true;
}

RadTensor dense(void* data, uint32_t dtype, std::vector<int64_t> shape) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = data;
    t.dtype = dtype;
    t.rank = (uint32_t)shape.size();
    for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
    rad_tensor_pack(&t);
    return t;
}

bool run_card(const Plan& p, int* worst_out, int64_t* diff_out) {
    Built b;
    if (!build(p, &b)) {
        std::fprintf(stderr, "%s: the fixture did not build\n", p.what);
        return false;
    }
    std::vector<uint8_t> live;
    const std::vector<uint16_t> want = reference(b, &live);
    if (want.empty()) {
        std::fprintf(stderr, "%s: the host reference refused the fixture\n", p.what);
        return false;
    }
    const int64_t E = p.E, T = b.g.T, Nmax = std::max(p.N[0], p.N[1]);
    const int64_t Arows = p.sorted_a ? T : p.M;

    Dev dev;
    std::vector<Guarded> g_w((size_t)E), g_s((size_t)E);
    for (int64_t e = 0; e < E; ++e) {
        g_w[(size_t)e] = galloc(dev, b.wq[(size_t)e].size());
        g_s[(size_t)e] = galloc(dev, b.wsc[(size_t)e].size());
        if (!g_w[(size_t)e].base || !g_s[(size_t)e].base) return false;
        to_dev(g_w[(size_t)e].p, b.wq[(size_t)e].data(), b.wq[(size_t)e].size());
        to_dev(g_s[(size_t)e].p, b.wsc[(size_t)e].data(), b.wsc[(size_t)e].size());
    }
    RadTensor wt[2], st[2];
    for (int q = 0; q < 2; ++q) {
        std::vector<const void*> wp, sp;
        for (int64_t e = q; e < E; e += 2) {
            wp.push_back(g_w[(size_t)e].p);
            sp.push_back(g_s[(size_t)e].p);
        }
        void* dwp = dev.alloc(std::max<size_t>(1, wp.size()) * sizeof(void*));
        void* dsp = dev.alloc(std::max<size_t>(1, sp.size()) * sizeof(void*));
        if (!dwp || !dsp) return false;
        if (!wp.empty()) to_dev(dwp, wp.data(), wp.size() * sizeof(void*));
        if (!sp.empty()) to_dev(dsp, sp.data(), sp.size() * sizeof(void*));
        wt[q] = dense(dwp, b.lw[(size_t)q].dtype, { (int64_t)wp.size(), p.N[q], p.K[q] });
        st[q] = dense(dsp, b.ls[(size_t)q].dtype,
                      { (int64_t)sp.size(), p.N[q], p.K[q] / 128 });
        wt[q].stride[0] = 0;
        st[q].stride[0] = 0;
    }

    const int64_t Kmax = b.Kmax, nsa = Kmax / b.g.a_group;
    Guarded gA = galloc(dev, (size_t)(Arows * Kmax));
    Guarded gAS = galloc(dev, (size_t)(Arows * nsa) * 4);
    Guarded gSO = galloc(dev, (size_t)T * 4);
    Guarded gOF = galloc(dev, (size_t)(E + 1) * 4);
    Guarded gY = galloc(dev, (size_t)(T * Nmax) * 2);
    if (!gA.base || !gAS.base || !gSO.base || !gOF.base || !gY.base) return false;
    to_dev(gA.p, b.a.data(), b.a.size());
    to_dev(gAS.p, b.as.data(), b.as.size() * 4);
    to_dev(gSO.p, b.sorted.data(), b.sorted.size() * 4);
    to_dev(gOF.p, b.off.data(), b.off.size() * 4);
    std::vector<uint16_t> poison((size_t)(T * Nmax), 0x7FC0u);
    to_dev(gY.p, poison.data(), poison.size() * 2);

    RadTensor ops[9] = {
        dense(gA.p, RAD_I8, { Arows, Kmax }),
        dense(gAS.p, RAD_F32, { Arows, nsa }),
        wt[0], st[0],
        dense(gSO.p, RAD_I32, { T }),
        dense(gOF.p, RAD_I32, { E + 1 }),
        wt[1], st[1],
        dense(gY.p, RAD_BF16, { T, Nmax }),
    };
    const std::vector<RadParam> ps = {
        pint("M", p.M), pint("N", p.N[0]), pint("K", p.K[0]), pint("n_expert", E),
        pint("top_k", p.top_k), pint("group", 128),
        pstr("a_order", p.sorted_a ? "sorted" : "token"), pstr("dtype", "i8a8h"),
        pint("N_odd", p.N[1]), pint("K_odd", p.K[1]), pint("parts", p.parts), pint("a_group", 128),
    };
    RadArgs args;
    std::memset(&args, 0, sizeof args);
    args.t = ops;
    args.n_t = 9;
    args.p = ps.data();
    args.n_p = (int)ps.size();
    args.kv_used_max_h = -1;
    args.world_size = 1;
    if (lib().scratch) {
        const int64_t need = lib().scratch(&args);
        if (need > 0) {
            args.scratch = dev.alloc((size_t)need);
            args.scratch_bytes = need;
        }
    }
    const int rc = lib().gemm(&args, (RadStream)0);
    const hipError_t he = hipDeviceSynchronize();
    if (rc != RAD_OK || he != hipSuccess) {
        std::fprintf(stderr, "%s: launch rc=%d hip=%d\n", p.what, rc, (int)he);
        return false;
    }
    std::vector<uint16_t> y((size_t)(T * Nmax));
    from_dev(y.data(), gY.p, y.size() * 2);

    bool guards = true;
    for (const Guarded& g : g_w) guards &= guards_intact(g);
    for (const Guarded& g : g_s) guards &= guards_intact(g);
    guards &= guards_intact(gA) && guards_intact(gAS) && guards_intact(gSO) &&
              guards_intact(gOF) && guards_intact(gY);
    if (!guards) {
        std::fprintf(stderr, "%s: a canary past an operand changed\n", p.what);
        return false;
    }

    const auto order = [](uint16_t h) {
        return (h & 0x8000u) ? -(int)(h & 0x7fffu) : (int)h;
    };
    int worst = 0;
    int64_t bad = 0, first = -1;
    for (size_t i = 0; i < y.size(); ++i) {
        if (!live[i]) {
            if (y[i] != 0) {
                ++bad;
                if (first < 0) first = (int64_t)i;
            }
            continue;
        }
        const int gap = std::abs(order(y[i]) - order(want[i]));
        worst = std::max(worst, gap);
        if (gap > kMoeMaxUlps) {
            ++bad;
            if (first < 0) first = (int64_t)i;
        }
    }
    if (worst_out) *worst_out = worst;
    if (diff_out) *diff_out = bad;
    std::printf("%s: worst %d bf16 ulp, %lld of %lld outputs outside the tolerance\n", p.what,
                worst, (long long)bad, (long long)y.size());
    if (bad) {
        std::fprintf(stderr, "%s: first [%lld][%lld]: got 0x%04x want 0x%04x (%.6f vs %.6f)\n",
                     p.what, (long long)(first / Nmax), (long long)(first % Nmax),
                     first >= 0 ? y[(size_t)first] : 0, first >= 0 ? want[(size_t)first] : 0,
                     first >= 0 ? bf16_to_f32(y[(size_t)first]) : 0.0f,
                     first >= 0 ? bf16_to_f32(want[(size_t)first]) : 0.0f);
    }
    return bad == 0;
}

TEST(moe_iqr_on_the_card_matches_the_host_reference) {
    /* The reproducible fixture: every case is a seeded plan, so the same weights, routing and
     * activation are drawn on any host. A data/kernels*.rkb case is not recorded here: rad-kbench's
     * --record-rows takes its answers from libref, which has no IQR MoE row, and data/kernels_moe.rkb
     * was recorded from a non-IQR expert layer. Once the row lands, this list is the set a recording
     * should cover. */
    static const Plan kPlans[] = {
        { "decode top_k 1, E 4, M 1", 4, 1, 1, 1, { 256, 512 }, { 256, 128 }, false, 0, false, false,
          0x1111u },
        { "decode top_k 2, E 4, M 1", 4, 2, 1, 1, { 256, 512 }, { 256, 128 }, false, 0, false, false,
          0x2222u },
        { "decode run of 8, E 8, M 8", 8, 1, 8, 1, { 256, 512 }, { 256, 128 }, false, 0, false, true,
          0x3333u },
        { "map top_k 2, E 4, M 16", 4, 2, 16, 1, { 256, 512 }, { 256, 128 }, false, 0, false, false,
          0x4444u },
        { "map top_k 4, E 4, M 33", 4, 4, 33, 1, { 256, 512 }, { 256, 128 }, false, 0, false, false,
          0x5555u },
        { "map zero rows and drops, E 8", 8, 2, 16, 1, { 256, 512 }, { 256, 128 }, false, 5, true,
          false, 0x6666u },
        { "map top_k 1, E 8, M 33", 8, 1, 33, 1, { 256, 512 }, { 256, 128 }, false, 0, false, false,
          0x7777u },
        { "map sorted activation, E 4", 4, 2, 16, 1, { 256, 512 }, { 256, 128 }, true, 0, false,
          false, 0x8888u },
        { "parts 2, E 4, M 16", 4, 2, 16, 2, { 512, 512 }, { 256, 256 }, false, 0, false, false,
          0x9999u },
        { "parts 2 sorted, E 4, M 33", 4, 4, 33, 2, { 512, 512 }, { 256, 256 }, true, 3, false,
          false, 0xAAAAu },
    };
    if (!lib().card_ok()) {
        SKIP("libr3d has no moe_gemm_iqr row yet (r3d_rad_moe_gemm_iqr absent)");
        return;
    }
    int worst_all = 0;
    int64_t bad_all = 0;
    for (const Plan& p : kPlans) {
        int worst = 0;
        int64_t bad = 0;
        const bool ok = run_card(p, &worst, &bad);
        worst_all = std::max(worst_all, worst);
        bad_all += bad;
        if (!ok) std::fprintf(stderr, "  %s failed\n", p.what);
        CHECK(ok);
    }
    std::printf("moe_gemm_iqr: %zu shapes, worst %d bf16 ulp (tolerance %d), %lld outside\n",
                sizeof kPlans / sizeof *kPlans, worst_all, kMoeMaxUlps, (long long)bad_all);
    CHECK(bad_all == 0);
}

}  // namespace

RAD_TEST_MAIN()
