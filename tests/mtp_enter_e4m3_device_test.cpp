/* mtp_enter_e4m3_device_test.cpp -- libr3d's `mtp_enter_e4m3` row against libref's `mtp_enter`,
 * on the card, at Flash-Next's shape (n = 2560, hc = 4, eps 1e-6, wadd 1, ngroup 4, E4M3 fc rows).
 *
 * The row reads the two fc matrices in libr4d's joined form (`r4d.mtp.e4m3.g128`): n E4M3 codes,
 * one f32 scale per 128 of K, zero padding to 16 bytes. The test calls the row's layout hook and
 * its relayout writer, as rad-kbench and the engine do, then uploads the joined operand. libref is
 * called on the plain f32 matrices those rows decode to, so the two read the same weights.
 *
 * Covered: M = 1..32 including non-powers of two; the grouped norm (ngroup = hc) and a flat one
 * (ngroup = 1); and a call whose M parameter is the band's ceiling while the operands carry three
 * rows (the `--band-capacity` condition, where M is never a row count).
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef RAD_R3D_SO
#define RAD_R3D_SO "libr3d.so"
#endif
#ifndef RAD_LIBREF_SO
#define RAD_LIBREF_SO "libref.so"
#endif

namespace {

typedef int (*LaunchFn)(const RadArgs*, RadStream);

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

struct Dev {
    void* p = nullptr;
    size_t bytes = 0;
    Dev() = default;
    explicit Dev(size_t b) { alloc(b); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { if (p) (void)hipFree(p); }
    void alloc(size_t b) {
        bytes = b;
        if (b && hipMalloc(&p, b) != hipSuccess) {
            std::fprintf(stderr, "hipMalloc(%zu) failed\n", b);
            std::abort();
        }
    }
};

void up(Dev& d, const void* src, size_t bytes) {
    if (bytes) (void)hipMemcpy(d.p, src, bytes, hipMemcpyHostToDevice);
}
void down(Dev& d, void* dst, size_t bytes) {
    if (bytes) (void)hipMemcpy(dst, d.p, bytes, hipMemcpyDeviceToHost);
}

RadTensor ten2(void* p, int dtype, long long rows, long long cols, long long rs, long long cs) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = p;
    t.dtype = (uint32_t)dtype;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.stride[0] = rs;
    t.stride[1] = cs;
    return t;
}

RadParam pint(const char* k, long long v) {
    RadParam p;
    std::memset(&p, 0, sizeof p);
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}
RadParam pf64(const char* k, double v) {
    RadParam p;
    std::memset(&p, 0, sizeof p);
    p.key = k;
    p.kind = RAD_P_F64;
    p.dval = v;
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

uint64_t g_z = 0x452821E638D01377ull;
uint64_t rnd() {
    g_z += 0x9E3779B97F4A7C15ull;
    uint64_t x = g_z;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
uint16_t rnd_bf16() {
    const int q = (int)(rnd() % 4001) - 2000;
    return rad_f32_to_bf16((float)q / 64.0f);
}

/* An E4M3 byte that is not a NaN (0x7f/0xff) and has a spread of exponents. */
uint8_t rnd_e4m3() {
    uint8_t b = (uint8_t)(rnd() & 0xffu);
    if ((b & 0x7fu) == 0x7fu) b ^= 0x01u;
    return b;
}

struct Fc {
    long long n;
    std::vector<uint8_t> codes;   /* [n, n] E4M3 */
    std::vector<float> scale;     /* [n, n/128] f32 */
    std::vector<float> plain;     /* [n, n] f32, what libref reads */
    std::vector<uint8_t> joined;  /* the row's stored form */
};

bool make_fc(const RadKernelInfo* kd, long long n, Fc* fc, int operand, const char* what) {
    fc->n = n;
    const long long ng = n / 128;
    fc->codes.resize((size_t)(n * n));
    for (auto& c : fc->codes) c = rnd_e4m3();
    fc->scale.resize((size_t)(n * ng));
    for (auto& s : fc->scale) s = 0.001f + 0.0002f * (float)(rnd() % 13);
    fc->plain.resize((size_t)(n * n));
    for (long long r = 0; r < n; ++r)
        for (long long j = 0; j < n; ++j)
            fc->plain[(size_t)(r * n + j)] =
                rad_fp8e4m3_to_f32(fc->codes[(size_t)(r * n + j)]) * fc->scale[(size_t)(r * ng + j / 128)];

    RadEncoding enc = rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 128);
    const int sel[2] = { 0, 1 };
    const std::vector<RadParam> lp = {
        pint("M", 32), pint("n", n), pint("hc", 4), pf64("eps", 1e-6), pstr("dtype", "bf16"),
        pf64("wadd", 1.0), pint("ngroup", 4), pstr("fc", "e4m3") };
    RadTensor pl[2] = { ten2(fc->codes.data(), RAD_F8E4M3, n, n, n, 1),
                        ten2(fc->scale.data(), RAD_F32, n, ng, ng, 1) };
    RadLayout lay{};
    int rc = kd->layout ? kd->layout(lp.data(), (int)lp.size(), operand, &enc, sel, pl, 2, &lay)
                        : RAD_E_UNSUPPORTED;
    if (rc != RAD_OK) {
        std::fprintf(stderr, "%s: fc layout refused rc=%d\n", what, rc);
        return false;
    }
    if (!kd->relayout) {
        std::fprintf(stderr, "%s: the row has no relayout\n", what);
        return false;
    }
    fc->joined.assign((size_t)lay.bytes, 0);
    rc = kd->relayout(lp.data(), (int)lp.size(), operand, &enc, sel, pl, 2, fc->joined.data(),
                      lay.bytes);
    if (rc != RAD_OK) {
        std::fprintf(stderr, "%s: fc relayout refused rc=%d\n", what, rc);
        return false;
    }
    return true;
}

struct Case {
    int M;          /* rows in the operands */
    int Mparam;     /* the band ceiling the op carries */
    int ngroup;
};

bool run_case(const RadKernelInfo* kd, const RadKernelInfo* kr, const Fc& fch, const Fc& fce,
              const Case& c, int hc, double tol, const char* what) {
    const long long n = fch.n, HN = (long long)hc * n;
    std::vector<uint16_t> h((size_t)c.M * HN), e((size_t)c.M * n), x((size_t)c.M * HN, 0);
    std::vector<float> wh((size_t)HN), we((size_t)n);
    for (auto& v : h) v = rnd_bf16();
    for (auto& v : e) v = rnd_bf16();
    for (auto& v : wh) v = 0.5f + 0.01f * (float)(rnd() % 100);
    for (auto& v : we) v = 0.5f + 0.01f * (float)(rnd() % 100);

    std::vector<uint16_t> want((size_t)c.M * HN, 0);
    std::vector<RadParam> p = { pint("M", c.Mparam), pint("n", n), pint("hc", hc),
                                pf64("eps", 1e-6), pstr("dtype", "bf16"), pf64("wadd", 1.0),
                                pint("ngroup", c.ngroup), pstr("fc", "e4m3") };
    const int n_p = (int)p.size();

    RadArgs ar;
    std::memset(&ar, 0, sizeof ar);
    RadTensor tr[7] = { ten2(h.data(), RAD_BF16, c.M, HN, HN, 1),
                        ten2(e.data(), RAD_BF16, c.M, n, n, 1),
                        ten2(wh.data(), RAD_F32, 1, HN, HN, 1),
                        ten2(we.data(), RAD_F32, 1, n, n, 1),
                        ten2((void*)fch.plain.data(), RAD_F32, n, n, n, 1),
                        ten2((void*)fce.plain.data(), RAD_F32, n, n, n, 1),
                        ten2(want.data(), RAD_BF16, c.M, HN, HN, 1) };
    ar.t = tr;
    ar.n_t = 7;
    ar.p = p.data();
    ar.n_p = n_p;

    Dev dh((size_t)c.M * HN * 2), de((size_t)c.M * n * 2), dwh((size_t)HN * 4),
        dwe((size_t)n * 4), dfh(fch.joined.size()), dfe(fce.joined.size()), dx((size_t)c.M * HN * 2);
    up(dh, h.data(), dh.bytes);
    up(de, e.data(), de.bytes);
    up(dwh, wh.data(), dwh.bytes);
    up(dwe, we.data(), dwe.bytes);
    up(dfh, fch.joined.data(), dfh.bytes);
    up(dfe, fce.joined.data(), dfe.bytes);

    std::vector<RadParam> pd = p;
    /* The core binds a re-laid weight with its DECLARED (logical) extent, as hc_read's E4M3 mix
 * documents; the joined bytes are behind that descriptor and the kernel reads its own row pitch. */
    RadTensor td[7] = { ten2(dh.p, RAD_BF16, c.M, HN, HN, 1),
                        ten2(de.p, RAD_BF16, c.M, n, n, 1),
                        ten2(dwh.p, RAD_F32, 1, HN, HN, 1),
                        ten2(dwe.p, RAD_F32, 1, n, n, 1),
                        ten2(dfh.p, RAD_F8E4M3, n, n, n, 1),
                        ten2(dfe.p, RAD_F8E4M3, n, n, n, 1),
                        ten2(dx.p, RAD_BF16, c.M, HN, HN, 1) };
    RadArgs ad;
    std::memset(&ad, 0, sizeof ad);
    ad.t = td;
    ad.n_t = 7;
    ad.p = pd.data();
    ad.n_p = n_p;

    const int rd = kd->launch(&ad, (RadStream)0);
    (void)hipDeviceSynchronize();
    const int rr = kr->launch(&ar, (RadStream)0);
    if (rd != 0 || rr != 0) {
        std::fprintf(stderr, "%s M=%d ngroup=%d: launch returned device %d, reference %d\n", what,
                     c.M, c.ngroup, rd, rr);
        return false;
    }
    std::vector<uint16_t> got((size_t)c.M * HN, 0);
    down(dx, got.data(), (size_t)c.M * HN * 2);

    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double g = rad_bf16_to_f32(got[i]), w = rad_bf16_to_f32(want[i]);
        if (!std::isfinite(g) || !std::isfinite(w)) {
            std::fprintf(stderr, "%s M=%d: non-finite at %zu\n", what, c.M, i);
            return false;
        }
        const double d = g - w;
        num += d * d;
        den += w * w;
        mx = std::fmax(mx, std::fabs(d));
    }
    const double rel = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
    if (rel > tol) {
        std::fprintf(stderr, "%s M=%d Mparam=%d ngroup=%d: rel_l2 %.3e over %.3e (max_abs %.3e)\n",
                     what, c.M, c.Mparam, c.ngroup, rel, tol, mx);
        return false;
    }
    return true;
}

}  // namespace

/* Flash-Next's real shape, every decode height the container declares, both norms. */
TEST(mtp_enter_e4m3_matches_libref_at_flash_next) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("mtp_enter", RAD_DOMAIN_DEVICE, "mtp_enter_e4m3");
    const RadKernelInfo* kr = g_ref.find("mtp_enter", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return;

    Fc fch, fce;
    CHECK(make_fc(kd, 2560, &fch, 4, "fc_h"));
    CHECK(make_fc(kd, 2560, &fce, 5, "fc_e"));
    if (fch.joined.empty() || fce.joined.empty()) return;

    const int Ms[] = { 1, 2, 3, 5, 7, 8, 16, 17, 31, 32 };
    for (int M : Ms)
        CHECK(run_case(kd, kr, fch, fce, { M, M, 4 }, 4, 5e-3, "grouped norm"));
    CHECK(run_case(kd, kr, fch, fce, { 3, 3, 1 }, 4, 5e-3, "flat norm"));
}

/* `M` is a band ceiling, never a row count: a step with 3 rows under the widest band must read the
 * operands' rows, not launch the ceiling. */
TEST(mtp_enter_e4m3_reads_the_operands_rows_under_a_wider_band) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("mtp_enter", RAD_DOMAIN_DEVICE, "mtp_enter_e4m3");
    const RadKernelInfo* kr = g_ref.find("mtp_enter", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return;

    Fc fch, fce;
    CHECK(make_fc(kd, 2560, &fch, 4, "fc_h"));
    CHECK(make_fc(kd, 2560, &fce, 5, "fc_e"));
    if (fch.joined.empty() || fce.joined.empty()) return;
    CHECK(run_case(kd, kr, fch, fce, { 3, 32, 4 }, 4, 5e-3, "band ceiling"));
}

/* hc = 2, a smaller width, so the kernel's two-stream path and its norm indices are covered away
 * from the 4-stream deployment. */
TEST(mtp_enter_e4m3_matches_libref_at_hc_two) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("mtp_enter", RAD_DOMAIN_DEVICE, "mtp_enter_e4m3");
    const RadKernelInfo* kr = g_ref.find("mtp_enter", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return;

    Fc fch, fce;
    CHECK(make_fc(kd, 1280, &fch, 4, "fc_h"));
    CHECK(make_fc(kd, 1280, &fce, 5, "fc_e"));
    if (fch.joined.empty() || fce.joined.empty()) return;
    CHECK(run_case(kd, kr, fch, fce, { 3, 3, 2 }, 2, 5e-3, "hc two"));
}

RAD_TEST_MAIN()