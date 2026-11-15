/* ple_hc_device_test.cpp -- libr3d's PLE (W9) and HyperConnections (W8) rows against libref.
 *
 * Each case runs libref's host row and libr3d's device row on the same operands and compares what
 * the contract fixes:
 *
 *   ple_conv     y and the written window, bit for bit (both sum the taps in j order with the same
 *                silu and one bf16 store);
 *   ple_gate     gv/gvn within bf16's last bits: both accumulate in double but the reductions
 *                differ in order (libref sequential, the kernel a workgroup tree);
 *   embed_q      x bit for bit;
 *   hc_enter / hc_write / hc_project   bit for bit;
 *   hc_read      x, inj, q/scale and rq/rscale bit for bit: every dot is ascending f32 and the
 *                norm is sequential double, so the split launch must reproduce the reference;
 *   ar_hc_write  bit for bit against hc_write at one rank (the identity all-reduce);
 *   ar_gather_hc_write  bit for bit against moe_gather then hc_write;
 *   ngram_hint   against the documented hash written out in the test (libref has no hint row).
 *
 * The libraries are dlopened and their rows found through rad_kernel_at, as the engine loads
 * them. On a machine without the card or the plugins every case skips; the test is meant to be
 * built everywhere and run where a gfx11 card is free.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include "r3d_codecs.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifndef RAD_R3D_SO
#define RAD_R3D_SO "libr3d.so"
#endif
#ifndef RAD_LIBREF_SO
#define RAD_LIBREF_SO "libref.so"
#endif

namespace {

typedef int (*LaunchFn)(const RadArgs*, RadStream);
typedef int64_t (*ScratchFn)(const RadArgs*);

struct Plugin {
    void* h = nullptr;
    int (*count)() = nullptr;
    const RadKernelInfo* (*at)(int) = nullptr;

    bool open(const char* path) {
        h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) return false;
        count = (int (*)())dlsym(h, "rad_kernel_count");
        at = (const RadKernelInfo* (*)(int))dlsym(h, "rad_kernel_at");
        return count && at;
    }
    const RadKernelInfo* find(const char* op, int domain, const char* prefix) const {
        for (int i = 0; count && i < count(); ++i) {
            const RadKernelInfo* k = at(i);
            if (!k || !k->op || !k->launch || k->domain != domain) continue;
            if (std::strcmp(k->op, op) != 0) continue;
            if (prefix && std::strncmp(k->name, prefix, std::strlen(prefix)) != 0) continue;
            return k;
        }
        return nullptr;
    }
    LaunchFn launch(const char* op, int domain, const char* prefix) const {
        const RadKernelInfo* k = find(op, domain, prefix);
        return k ? k->launch : nullptr;
    }
    ScratchFn scratch(const char* op, int domain, const char* prefix) const {
        const RadKernelInfo* k = find(op, domain, prefix);
        return k ? k->scratch : nullptr;
    }
};

Plugin g_r3d, g_ref;

bool ready() {
    static int state = -1;
    static bool ok = false;
    if (state >= 0) return ok;
    state = 0;
    int nd = 0;
    if (hipGetDeviceCount(&nd) != hipSuccess || nd < 1) {
        std::fprintf(stderr, "skip: no card\n");
        return false;
    }
    if (!g_r3d.open(RAD_R3D_SO) || !g_ref.open(RAD_LIBREF_SO)) {
        std::fprintf(stderr, "skip: %s / %s not loadable here\n", RAD_R3D_SO, RAD_LIBREF_SO);
        return false;
    }
    ok = true;
    state = 1;
    return ok;
}

/* ---- buffers and tensors ------------------------------------------------------------------ */

struct Dev {
    void* p = nullptr;
    size_t bytes = 0;
    Dev() = default;
    explicit Dev(size_t b) { alloc(b); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { if (p) hipFree(p); }
    void alloc(size_t b) {
        bytes = b;
        if (b && hipMalloc(&p, b) != hipSuccess) {
            std::fprintf(stderr, "hipMalloc(%zu) failed\n", b);
            std::abort();
        }
    }
};

void up(Dev& d, const void* src, size_t bytes) {
    if (bytes) hipMemcpy(d.p, src, bytes, hipMemcpyHostToDevice);
}
void down(Dev& d, void* dst, size_t bytes) {
    if (bytes) hipMemcpy(dst, d.p, bytes, hipMemcpyDeviceToHost);
}

RadTensor ten(void* p, int dtype, std::initializer_list<long long> dims) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = p;
    t.dtype = dtype;
    t.rank = (uint32_t)dims.size();
    int i = 0;
    for (long long d : dims) t.shape[i++] = d;
    rad_tensor_pack(&t);
    return t;
}

RadParam pi(const char* k, long long v) { return RAD_INT(k, v); }
RadParam pf(const char* k, double v) { return RAD_F64(k, v); }
RadParam ps(const char* k, const char* v) { return RAD_STR(k, v); }

std::vector<uint16_t> rand_bf16(size_t n, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(lo, hi);
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = f32_to_bf16(U(rng));
    return v;
}
std::vector<float> rand_f32(size_t n, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = U(rng);
    return v;
}
std::vector<uint8_t> rand_e4m3(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> U(0, 126);   /* finite, non-NaN bytes */
    std::vector<uint8_t> v(n);
    for (auto& x : v) x = (uint8_t)U(rng);
    return v;
}

int run(const RadKernelInfo* k, std::vector<RadTensor>& t, std::vector<RadParam>& p, void* scratch,
        int64_t scratch_bytes, int world = 1) {
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.scratch = scratch;
    a.scratch_bytes = scratch_bytes;
    a.world_size = world;
    a.rank = 0;
    const int rc = k->launch(&a, (RadStream)0);
    hipDeviceSynchronize();
    return rc;
}

bool same_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const char* what,
               size_t* first = nullptr) {
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            if (first) *first = i;
            std::fprintf(stderr, "%s differs at %zu: %04x vs %04x\n", what, i, a[i], b[i]);
            return false;
        }
    return true;
}
bool same_f32(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 4) != 0) {
            std::fprintf(stderr, "%s differs at %zu: %.9g vs %.9g\n", what, i, a[i], b[i]);
            return false;
        }
    return true;
}
bool close_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const char* what,
                float rel, float abs) {
    for (size_t i = 0; i < a.size(); ++i) {
        const float x = bf16_to_f32(a[i]), y = bf16_to_f32(b[i]);
        if (std::fabs(x - y) > abs + rel * std::fabs(y)) {
            std::fprintf(stderr, "%s differs at %zu: %.6g vs %.6g\n", what, i, x, y);
            return false;
        }
    }
    return true;
}

/* ================================================================== ple_conv */
TEST(ple_conv_bf16_matches_libref) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("ple_conv", RAD_DOMAIN_DEVICE, "ple_conv");
    const RadKernelInfo* kr = g_ref.find("ple_conv", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!kd || !kr) return;

    const int n = 32, width = 4, dilation = 3, hist = 9;
    const int T = 12, N = 1, slots = 2, state_len = 16;
    auto xh = rand_bf16((size_t)T * n, 1);
    auto wh = rand_bf16((size_t)n * width, 2);
    auto rh = rand_bf16((size_t)T * n, 3);
    auto sh = rand_bf16((size_t)slots * n * state_len, 4);
    std::vector<int32_t> cu = { 0, T };

    Dev dx((size_t)T * n * 2), dw((size_t)n * width * 2), dr((size_t)T * n * 2),
        ds((size_t)slots * n * state_len * 2), dcu(2 * 4);
    Dev dy0((size_t)T * n * 2), dy1((size_t)T * n * 2), ds0(ds.bytes), ds1(ds.bytes);
    up(dx, xh.data(), dx.bytes); up(dw, wh.data(), dw.bytes); up(dr, rh.data(), dr.bytes);
    up(ds0, sh.data(), ds0.bytes); up(ds1, sh.data(), ds1.bytes); up(dcu, cu.data(), dcu.bytes);

    auto p = std::vector<RadParam>{ pi("M", T), pi("n", n), pi("width", width),
                                    pi("dilation", dilation), ps("dtype", "bf16") };
    for (int which = 0; which < 2; ++which) {
        Dev& y = which ? dy1 : dy0;
        Dev& st = which ? ds1 : ds0;
        std::vector<RadTensor> t = {
            ten(dx.p, RAD_BF16, { T, n }), ten(dw.p, RAD_BF16, { n, width }),
            ten(st.p, RAD_BF16, { slots, n, state_len }), ten(dcu.p, RAD_I32, { 2 }),
            ten(dr.p, RAD_BF16, { T, n }), ten(nullptr, RAD_I32, { 0 }),
            ten(nullptr, RAD_I32, { 0 }), ten(nullptr, RAD_I32, { 0 }),
            ten(y.p, RAD_BF16, { T, n })
        };
        CHECK_EQ(run(which ? kr : kd, t, p, nullptr, 0), 0);
    }
    std::vector<uint16_t> y0((size_t)T * n), y1((size_t)T * n), st0(sh.size()), st1(sh.size());
    down(dy0, y0.data(), dy0.bytes); down(dy1, y1.data(), dy1.bytes);
    down(ds0, st0.data(), ds0.bytes); down(ds1, st1.data(), ds1.bytes);
    CHECK(same_bf16(y0, y1, "ple_conv y"));
    CHECK(same_bf16(st0, st1, "ple_conv state"));
}

/* ================================================================== ple_gate */
TEST(ple_gate_bf16_matches_libref) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("ple_gate", RAD_DOMAIN_DEVICE, "ple_gate");
    const RadKernelInfo* kr = g_ref.find("ple_gate", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!kd || !kr) return;

    const int n = 64, hc = 2, T = 2, hn = hc * n;
    auto kh = rand_bf16((size_t)T * hn, 11), qh = rand_bf16((size_t)T * hn, 12);
    auto vh = rand_bf16((size_t)T * n, 13);
    auto wkh = rand_f32(hn, 14, 0.5f, 1.5f), wqh = rand_f32(hn, 15, 0.5f, 1.5f),
         wch = rand_f32(hn, 16, 0.5f, 1.5f);

    Dev dk((size_t)T * hn * 2), dq((size_t)T * hn * 2), dv((size_t)T * n * 2);
    Dev dwk(hn * 4), dwq(hn * 4), dwc(hn * 4);
    Dev g0((size_t)T * hn * 2), g1((size_t)T * hn * 2), n0((size_t)T * hn * 2),
        n1((size_t)T * hn * 2);
    up(dk, kh.data(), dk.bytes); up(dq, qh.data(), dq.bytes); up(dv, vh.data(), dv.bytes);
    up(dwk, wkh.data(), dwk.bytes); up(dwq, wqh.data(), dwq.bytes); up(dwc, wch.data(), dwc.bytes);

    auto p = std::vector<RadParam>{ pi("M", T), pi("n", n), pi("hc", hc), pf("eps", 1e-6),
                                    ps("dtype", "bf16"), pf("wadd", 0.1), pf("gate_eps", 1e-6) };
    for (int which = 0; which < 2; ++which) {
        Dev& gv = which ? g1 : g0;
        Dev& gvn = which ? n1 : n0;
        std::vector<RadTensor> t = {
            ten(dk.p, RAD_BF16, { T, hn }), ten(dq.p, RAD_BF16, { T, hn }),
            ten(dv.p, RAD_BF16, { T, n }), ten(dwk.p, RAD_F32, { hn }),
            ten(dwq.p, RAD_F32, { hn }), ten(dwc.p, RAD_F32, { hn }),
            ten(gv.p, RAD_BF16, { T, hn }), ten(gvn.p, RAD_BF16, { T, hn })
        };
        CHECK_EQ(run(which ? kr : kd, t, p, nullptr, 0), 0);
    }
    std::vector<uint16_t> a((size_t)T * hn), b((size_t)T * hn), c((size_t)T * hn),
        d((size_t)T * hn);
    down(g0, a.data(), g0.bytes); down(g1, b.data(), g1.bytes);
    down(n0, c.data(), n0.bytes); down(n1, d.data(), n1.bytes);
    /* bf16 relative resolution is 2^-8 = 0.0039; one reduction reorder moves the last bit or two. */
    CHECK(close_bf16(a, b, "ple_gate gv", 0.01f, 0.01f));
    CHECK(close_bf16(c, d, "ple_gate gvn", 0.01f, 0.01f));
}

/* ================================================================== embed_lookup_q */
TEST(embed_lookup_q_e4m3_matches_libref) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("embed_lookup_q", RAD_DOMAIN_DEVICE, "embed_lookup_q");
    const RadKernelInfo* kr = g_ref.find("embed_lookup_q", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!kd || !kr) return;

    const int M = 4, ne = 32, nv = 8;
    std::vector<int32_t> tok = { 0, 3, 7, 1 };
    auto codes = rand_e4m3((size_t)nv * ne, 21);
    uint16_t scale = f32_to_bf16(0.017f);

    Dev dtok(M * 4), dtab(codes.size()), dsc(2), dx0((size_t)M * ne * 2), dx1((size_t)M * ne * 2);
    up(dtok, tok.data(), dtok.bytes); up(dtab, codes.data(), dtab.bytes); up(dsc, &scale, 2);

    auto p = std::vector<RadParam>{ pi("M", M), pi("n_embd", ne), pi("n_vocab", nv),
                                    ps("dtype", "bf16"), pi("vocab_offset", 0), pf("wscale", 1.0) };
    for (int which = 0; which < 2; ++which) {
        Dev& x = which ? dx1 : dx0;
        std::vector<RadTensor> t = {
            ten(dtok.p, RAD_I32, { M }), ten(dtab.p, RAD_F8E4M3, { nv, ne }),
            ten(dsc.p, RAD_BF16, { 1 }), ten(x.p, RAD_BF16, { M, ne }),
            ten(nullptr, RAD_I32, { 0 })
        };
        CHECK_EQ(run(which ? kr : kd, t, p, nullptr, 0), 0);
    }
    std::vector<uint16_t> a((size_t)M * ne), b((size_t)M * ne);
    down(dx0, a.data(), dx0.bytes); down(dx1, b.data(), dx1.bytes);
    CHECK(same_bf16(a, b, "embed_lookup_q x"));
}

/* ================================================================== hc_enter/write/project */
TEST(hc_stream_ops_match_libref) {
    if (!ready()) return;
    const RadKernelInfo* ke_d = g_r3d.find("hc_enter", RAD_DOMAIN_DEVICE, "hc_enter");
    const RadKernelInfo* ke_r = g_ref.find("hc_enter", RAD_DOMAIN_HOST, "ref_");
    const RadKernelInfo* kw_d = g_r3d.find("hc_write", RAD_DOMAIN_DEVICE, "hc_write");
    const RadKernelInfo* kw_r = g_ref.find("hc_write", RAD_DOMAIN_HOST, "ref_");
    const RadKernelInfo* kp_d = g_r3d.find("hc_project", RAD_DOMAIN_DEVICE, "hc_project");
    const RadKernelInfo* kp_r = g_ref.find("hc_project", RAD_DOMAIN_HOST, "ref_");
    CHECK(ke_d && ke_r && kw_d && kw_r && kp_d && kp_r);
    if (!(ke_d && ke_r && kw_d && kw_r && kp_d && kp_r)) return;

    const int M = 3, n = 32, hc = 4, hn = hc * n;

    /* enter */
    {
        auto xh = rand_bf16((size_t)M * n, 31);
        Dev dx((size_t)M * n * 2), dh0((size_t)M * hn * 2), dh1((size_t)M * hn * 2);
        up(dx, xh.data(), dx.bytes);
        auto p = std::vector<RadParam>{ pi("M", M), pi("n", n), pi("hc", hc), ps("dtype", "bf16") };
        for (int which = 0; which < 2; ++which) {
            Dev& h = which ? dh1 : dh0;
            std::vector<RadTensor> t = { ten(dx.p, RAD_BF16, { M, n }),
                                         ten(h.p, RAD_BF16, { M, hn }) };
            CHECK_EQ(run(which ? ke_r : ke_d, t, p, nullptr, 0), 0);
        }
        std::vector<uint16_t> a((size_t)M * hn), b((size_t)M * hn);
        down(dh0, a.data(), dh0.bytes); down(dh1, b.data(), dh1.bytes);
        CHECK(same_bf16(a, b, "hc_enter h"));
    }

    /* write */
    {
        auto yh = rand_bf16((size_t)M * n, 32), ijh = rand_bf16((size_t)M * hc, 33),
             hh = rand_bf16((size_t)M * hn, 34);
        Dev dy((size_t)M * n * 2), dij((size_t)M * hc * 2), dh0((size_t)M * hn * 2),
            dh1((size_t)M * hn * 2);
        up(dy, yh.data(), dy.bytes); up(dij, ijh.data(), dij.bytes);
        up(dh0, hh.data(), dh0.bytes); up(dh1, hh.data(), dh1.bytes);
        auto p = std::vector<RadParam>{ pi("M", M), pi("n", n), pi("hc", hc), ps("dtype", "bf16") };
        for (int which = 0; which < 2; ++which) {
            Dev& h = which ? dh1 : dh0;
            std::vector<RadTensor> t = { ten(dy.p, RAD_BF16, { M, n }),
                                         ten(dij.p, RAD_BF16, { M, hc }),
                                         ten(h.p, RAD_BF16, { M, hn }) };
            CHECK_EQ(run(which ? kw_r : kw_d, t, p, nullptr, 0), 0);
        }
        std::vector<uint16_t> a((size_t)M * hn), b((size_t)M * hn);
        down(dh0, a.data(), dh0.bytes); down(dh1, b.data(), dh1.bytes);
        CHECK(same_bf16(a, b, "hc_write h"));
    }

    /* project */
    {
        auto hh = rand_bf16((size_t)M * hn, 35);
        auto vh = rand_f32(n, 36);
        std::vector<int32_t> on = { 1, 0, 1 };
        Dev dh0((size_t)M * hn * 2), dh1((size_t)M * hn * 2), dv(n * 4), don(M * 4);
        up(dh0, hh.data(), dh0.bytes); up(dh1, hh.data(), dh1.bytes);
        up(dv, vh.data(), dv.bytes); up(don, on.data(), don.bytes);
        auto p = std::vector<RadParam>{ pi("M", M), pi("n", n), pi("hc", hc), pf("scale", 2.5),
                                        ps("dtype", "bf16") };
        for (int which = 0; which < 2; ++which) {
            Dev& h = which ? dh1 : dh0;
            std::vector<RadTensor> t = { ten(h.p, RAD_BF16, { M, hn }), ten(dv.p, RAD_F32, { n }),
                                         ten(don.p, RAD_I32, { M }) };
            CHECK_EQ(run(which ? kp_r : kp_d, t, p, nullptr, 0), 0);
        }
        std::vector<uint16_t> a((size_t)M * hn), b((size_t)M * hn);
        down(dh0, a.data(), dh0.bytes); down(dh1, b.data(), dh1.bytes);
        CHECK(same_bf16(a, b, "hc_project h"));
    }
}

/* ================================================================== hc_read */
TEST(hc_read_bf16_matches_libref) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("hc_read", RAD_DOMAIN_DEVICE, "hc_read");
    const RadKernelInfo* kr = g_ref.find("hc_read", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr && kd->scratch);
    if (!(kd && kr && kd->scratch)) return;

    /* lr = 128 > n % 256 = 64: the last column tile's lanes 64..127 retire past `n`, so they
     * must load the shared window and reach the barrier before they go. lr = 16 would not
     * cover that staging. */
    const int M = 2, n = 64, hc = 2, hn = hc * n, lr = 128, group = 16;
    const int ngrp = n / group;
    auto hh = rand_bf16((size_t)M * hn, 41), dh = rand_bf16((size_t)lr * hn, 42),
         uh = rand_bf16((size_t)hn * lr, 43), ijh = rand_bf16((size_t)hc * hn, 44);
    auto wh = rand_f32(hn, 45, 0.5f, 1.5f);

    Dev dh0((size_t)M * hn * 2), ddown((size_t)lr * hn * 2), dup((size_t)hn * lr * 2),
        dijw((size_t)hc * hn * 2), dw(hn * 4);
    up(dh0, hh.data(), dh0.bytes); up(ddown, dh.data(), ddown.bytes); up(dup, uh.data(), dup.bytes);
    up(dijw, ijh.data(), dijw.bytes); up(dw, wh.data(), dw.bytes);

    auto p = std::vector<RadParam>{
        pi("M", M), pi("n", n), pi("hc", hc), pi("lowrank", lr), pf("eps", 1e-6),
        ps("dtype", "bf16"), pf("wadd", 0.0), pi("inject", 1), pi("group", group),
        pi("rotate", 1), ps("mix", "bf16")
    };

    for (int which = 0; which < 2; ++which) {
        Dev x((size_t)M * n * 2), inj((size_t)M * hc * 2), q((size_t)M * n),
            qs((size_t)M * ngrp * 4), rq((size_t)M * n), rs((size_t)M * ngrp * 4);
        Dev scratch;
        int64_t need = 0;
        if (which == 0) {
            std::vector<RadTensor> t = {
                ten(dh0.p, RAD_BF16, { M, hn }), ten(dw.p, RAD_F32, { hn }),
                ten(ddown.p, RAD_BF16, { lr, hn }), ten(dup.p, RAD_BF16, { hn, lr }),
                ten(dijw.p, RAD_BF16, { hc, hn }), ten(x.p, RAD_BF16, { M, n }),
                ten(inj.p, RAD_BF16, { M, hc }), ten(q.p, RAD_F8E4M3, { M, n }),
                ten(qs.p, RAD_F32, { M, ngrp }), ten(rq.p, RAD_F8E4M3, { M, n }),
                ten(rs.p, RAD_F32, { M, ngrp })
            };
            RadArgs a;
            std::memset(&a, 0, sizeof a);
            a.t = t.data(); a.n_t = (int)t.size(); a.p = p.data(); a.n_p = (int)p.size();
            a.world_size = 1;
            need = kd->scratch(&a);
            CHECK(need > 0);
            if (need <= 0) return;
            scratch.alloc((size_t)need);
            CHECK_EQ(run(kd, t, p, scratch.p, need), 0);
        } else {
            std::vector<RadTensor> t = {
                ten(dh0.p, RAD_BF16, { M, hn }), ten(dw.p, RAD_F32, { hn }),
                ten(ddown.p, RAD_BF16, { lr, hn }), ten(dup.p, RAD_BF16, { hn, lr }),
                ten(dijw.p, RAD_BF16, { hc, hn }), ten(x.p, RAD_BF16, { M, n }),
                ten(inj.p, RAD_BF16, { M, hc }), ten(q.p, RAD_F8E4M3, { M, n }),
                ten(qs.p, RAD_F32, { M, ngrp }), ten(rq.p, RAD_F8E4M3, { M, n }),
                ten(rs.p, RAD_F32, { M, ngrp })
            };
            CHECK_EQ(run(kr, t, p, nullptr, 0), 0);
        }
        /* Compare inside the loop: each side's buffers die at the iteration's end. */
        static std::vector<uint16_t> rx, ri, rqbuf, rrq;
        static std::vector<float> rqs, rrs;
        if (which == 0) {
            rx.resize((size_t)M * n); ri.resize((size_t)M * hc); rqbuf.resize((size_t)M * n);
            rrq.resize((size_t)M * n); rqs.resize((size_t)M * ngrp); rrs.resize((size_t)M * ngrp);
            down(x, rx.data(), x.bytes); down(inj, ri.data(), inj.bytes);
            down(q, rqbuf.data(), q.bytes); down(qs, rqs.data(), qs.bytes);
            down(rq, rrq.data(), rq.bytes); down(rs, rrs.data(), rs.bytes);
        } else {
            std::vector<uint16_t> x2((size_t)M * n), i2((size_t)M * hc), q2((size_t)M * n),
                rq2((size_t)M * n);
            std::vector<float> qs2((size_t)M * ngrp), rs2((size_t)M * ngrp);
            down(x, x2.data(), x.bytes); down(inj, i2.data(), inj.bytes);
            down(q, q2.data(), q.bytes); down(qs, qs2.data(), qs.bytes);
            down(rq, rq2.data(), rq.bytes); down(rs, rs2.data(), rs.bytes);
            CHECK(same_bf16(rx, x2, "hc_read x"));
            CHECK(same_bf16(ri, i2, "hc_read inj"));
            CHECK(std::memcmp(rqbuf.data(), q2.data(), q2.size()) == 0);
            CHECK(same_f32(rqs, qs2, "hc_read q scale"));
            CHECK(std::memcmp(rrq.data(), rq2.data(), rq2.size()) == 0);
            CHECK(same_f32(rrs, rs2, "hc_read rq scale"));
        }
    }
}

/* ================================================================== the one-rank collectives */
TEST(ar_hc_write_one_rank_matches_libref) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("ar_hc_write", RAD_DOMAIN_DEVICE, "ar_hc_write");
    const RadKernelInfo* kr = g_ref.find("hc_write", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!kd || !kr) return;

    const int M = 3, n = 32, hc = 4, hn = hc * n;
    auto yh = rand_bf16((size_t)M * n, 51), ijh = rand_bf16((size_t)M * hc, 52),
         hh = rand_bf16((size_t)M * hn, 53);
    Dev dy((size_t)M * n * 2), dij((size_t)M * hc * 2), dh0((size_t)M * hn * 2),
        dh1((size_t)M * hn * 2);
    up(dy, yh.data(), dy.bytes); up(dij, ijh.data(), dij.bytes);
    up(dh0, hh.data(), dh0.bytes); up(dh1, hh.data(), dh1.bytes);

    auto pa = std::vector<RadParam>{ pi("M", M), pi("n", n), pi("hc", hc),
                                     pi("world_size", 1), ps("dtype", "bf16"), pi("wire", 0) };
    {
        std::vector<RadTensor> t = { ten(dy.p, RAD_BF16, { M, n }),
                                     ten(dij.p, RAD_BF16, { M, hc }),
                                     ten(dh0.p, RAD_BF16, { M, hn }) };
        CHECK_EQ(run(kd, t, pa, nullptr, 0), 0);
    }
    auto pw = std::vector<RadParam>{ pi("M", M), pi("n", n), pi("hc", hc), ps("dtype", "bf16") };
    {
        std::vector<RadTensor> t = { ten(dy.p, RAD_BF16, { M, n }),
                                     ten(dij.p, RAD_BF16, { M, hc }),
                                     ten(dh1.p, RAD_BF16, { M, hn }) };
        CHECK_EQ(run(kr, t, pw, nullptr, 0), 0);
    }
    std::vector<uint16_t> a((size_t)M * hn), b((size_t)M * hn);
    down(dh0, a.data(), dh0.bytes); down(dh1, b.data(), dh1.bytes);
    CHECK(same_bf16(a, b, "ar_hc_write h"));
}

TEST(ar_gather_hc_write_one_rank_matches_libref_pair) {
    if (!ready()) return;
    const RadKernelInfo* kd =
        g_r3d.find("ar_gather_hc_write", RAD_DOMAIN_DEVICE, "ar_gather_hc_write");
    const RadKernelInfo* kg = g_ref.find("moe_gather", RAD_DOMAIN_HOST, "ref_");
    const RadKernelInfo* kw = g_ref.find("hc_write", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kg && kw);
    if (!(kd && kg && kw)) return;

    const int M = 3, n = 32, hc = 2, hn = hc * n, top_k = 2, T = M * top_k;
    auto yeh = rand_bf16((size_t)T * n, 61), ewh = rand_bf16((size_t)M * top_k, 62),
         shh = rand_bf16((size_t)M * n, 63);
    auto sgh = rand_bf16((size_t)M, 64);
    auto ijh = rand_bf16((size_t)M * hc, 65), hh = rand_bf16((size_t)M * hn, 66);
    std::vector<int32_t> sorted(T);
    for (int i = 0; i < T; ++i) sorted[(size_t)i] = i;

    Dev dye((size_t)T * n * 2), dew((size_t)M * top_k * 2), dsort(T * 4),
        dsh((size_t)M * n * 2), dsg(M * 2), dij((size_t)M * hc * 2);
    Dev hy0((size_t)M * n * 2), hy1((size_t)M * n * 2);
    Dev hh0((size_t)M * hn * 2), hh1((size_t)M * hn * 2);
    up(dye, yeh.data(), dye.bytes); up(dew, ewh.data(), dew.bytes);
    up(dsort, sorted.data(), dsort.bytes); up(dsh, shh.data(), dsh.bytes);
    up(dsg, sgh.data(), dsg.bytes); up(dij, ijh.data(), dij.bytes);
    up(hh0, hh.data(), hh0.bytes); up(hh1, hh.data(), hh1.bytes);

    /* libr3d: the fused row over copies. */
    {
        auto p = std::vector<RadParam>{ pi("M", M), pi("n", n), pi("hc", hc),
                                        pi("top_k", top_k), pi("world_size", 1),
                                        ps("dtype", "bf16"), ps("act", "sigmoid"), pi("wire", 0) };
        std::vector<RadTensor> t = {
            ten(dye.p, RAD_BF16, { T, n }), ten(dew.p, RAD_BF16, { M, top_k }),
            ten(dsort.p, RAD_I32, { T }), ten(dsh.p, RAD_BF16, { M, n }),
            ten(dsg.p, RAD_BF16, { M }), ten(hy0.p, RAD_BF16, { M, n }),
            ten(dij.p, RAD_BF16, { M, hc }), ten(hh0.p, RAD_BF16, { M, hn })
        };
        CHECK_EQ(run(kd, t, p, nullptr, 0), 0);
    }
    /* libref: moe_gather into hy1, then hc_write into hh1. */
    {
        auto pg = std::vector<RadParam>{ pi("M", M), pi("n", n), pi("top_k", top_k),
                                         ps("dtype", "bf16"), ps("act", "sigmoid") };
        std::vector<RadTensor> t = {
            ten(dye.p, RAD_BF16, { T, n }), ten(dew.p, RAD_BF16, { M, top_k }),
            ten(dsort.p, RAD_I32, { T }), ten(dsh.p, RAD_BF16, { M, n }),
            ten(dsg.p, RAD_BF16, { M }), ten(hy1.p, RAD_BF16, { M, n })
        };
        CHECK_EQ(run(kg, t, pg, nullptr, 0), 0);
        auto pw = std::vector<RadParam>{ pi("M", M), pi("n", n), pi("hc", hc),
                                         ps("dtype", "bf16") };
        std::vector<RadTensor> tw = { ten(hy1.p, RAD_BF16, { M, n }),
                                      ten(dij.p, RAD_BF16, { M, hc }),
                                      ten(hh1.p, RAD_BF16, { M, hn }) };
        CHECK_EQ(run(kw, tw, pw, nullptr, 0), 0);
    }
    std::vector<uint16_t> y0((size_t)M * n), y1((size_t)M * n), a((size_t)M * hn),
        b((size_t)M * hn);
    down(hy0, y0.data(), hy0.bytes); down(hy1, y1.data(), hy1.bytes);
    down(hh0, a.data(), hh0.bytes); down(hh1, b.data(), hh1.bytes);
    CHECK(same_bf16(y0, y1, "ar_gather_hc_write y"));
    CHECK(same_bf16(a, b, "ar_gather_hc_write h"));
}

/* ================================================================== ngram_hint */
TEST(ngram_hint_matches_the_documented_hash) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("ngram_hint", RAD_DOMAIN_DEVICE, "ngram_hint");
    CHECK(kd);
    if (!kd) return;

    const int N = 2, ngram = 2, heads = 1, per = 1, slots = 2, ctx = ngram - 1;
    std::vector<int32_t> tok = { 5, 9 };
    std::vector<int32_t> pos = { 0, 1 };
    std::vector<int32_t> cu = { 0, 1, 2 };
    std::vector<int64_t> mult = { 0x7F4A7C15LL, 1 };
    std::vector<int64_t> vsz = { 1000 };
    std::vector<int64_t> off = { 7 };
    std::vector<int32_t> tail((size_t)slots * 2 * ctx, 0);
    std::vector<int32_t> ids((size_t)N * heads, -1);

    Dev dtok(2 * 4), dpos(2 * 4), dcu(3 * 4), dmult((size_t)ngram * 8), dvsz(8), doff(8), dtail(tail.size() * 4),
        dids(ids.size() * 4);
    up(dtok, tok.data(), dtok.bytes); up(dpos, pos.data(), dpos.bytes);
    up(dcu, cu.data(), dcu.bytes); up(dmult, mult.data(), dmult.bytes);
    up(dvsz, vsz.data(), dvsz.bytes); up(doff, off.data(), doff.bytes);
    up(dtail, tail.data(), dtail.bytes);

    auto p = std::vector<RadParam>{ pi("M", N), pi("heads", heads), pi("ngram", ngram),
                                    pi("eos", 1000), pi("hist", 1) };
    std::vector<RadTensor> t = {
        ten(dtok.p, RAD_I32, { 2 }), ten(dpos.p, RAD_I32, { 2 }), ten(dcu.p, RAD_I32, { 3 }),
        ten(dmult.p, RAD_I64, { ngram }), ten(dvsz.p, RAD_I64, { heads }),
        ten(doff.p, RAD_I64, { heads }), ten(nullptr, RAD_I32, { 0 }),
        ten(dtail.p, RAD_I32, { slots, 2 * ctx }), ten(dids.p, RAD_I32, { N, heads })
    };
    CHECK_EQ(run(kd, t, p, nullptr, 0), 0);
    down(dids, ids.data(), dids.bytes);

    /* ngram 2: the mix is x * mult[0], independently per sequence. */
    for (int s = 0; s < N; ++s) {
        const long long x = tok[(size_t)s];
        const long long m = mult[0];
        const unsigned long long prod = (unsigned long long)x * (unsigned long long)m;
        long long mixed = (long long)prod;
        long long r = mixed % vsz[0];
        if (r < 0) r += vsz[0];
        const int want = (int)(r + off[0]);
        CHECK_EQ(ids[(size_t)s], want);
    }
}

}  // namespace

RAD_TEST_MAIN()
