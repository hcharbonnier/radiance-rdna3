/* dflash_conv_device_test.cpp -- libr3d's `dflash_conv` row against libref's `ref_dflash_conv`.
 *
 * The 27B's DFlash2 drafter convolves its residual stream with a two-tap dynamic kernel whose
 * gains are base[j, h] + delta[t, j, h / 16]. The row exists so that the drafter's weight group is
 * served on the card at all: without it the group's conv is host-only, the plan pins the whole
 * group to the container's mapping, and the widened norms refuse to load (R47).
 *
 * The operands are bf16 throughout and both libraries round once to bf16 at the store, so the
 * comparison is on the raw 16-bit words: this is a bit-exactness test, not a tolerance test.
 *
 * Covered: the 27B band (H = 5120, NG = 320, block 4, both halves of the coefficient projection);
 * a block that does not divide T and a block that is not a power of two (the mask is a modulo);
 * block 1 (the shifted tap never applies); a block wider than T (only row 0 masked); T = 1; and a
 * small H (128, the row's divisibility minimum). The two refusals the shim owns are checked too:
 * `out` aliasing `x`, and a delta plane at the packed pitch instead of the projection's.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
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

RadTensor ten(void* p, uint32_t dtype, std::initializer_list<long long> dims) {
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
RadParam ps(const char* k, const char* v) { return RAD_STR(k, v); }

int run(const RadKernelInfo* k, std::vector<RadTensor>& t, std::vector<RadParam>& p) {
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    const int rc = k->launch(&a, (RadStream)0);
    hipDeviceSynchronize();
    return rc;
}

const RadConstraint* constraint(const RadKernelInfo* k, const char* key) {
    if (!k) return nullptr;
    for (int i = 0; i < k->n_constraints; ++i) {
        const RadConstraint* c = &k->constraints[i];
        if (c->key && std::strcmp(c->key, key) == 0) return c;
    }
    return nullptr;
}

/* A bf16 plane of small values that exercises the sign, the exponent range and exact zeros; the
 * coefficient gains are drawn near 1 so both taps matter. */
std::vector<uint16_t> draw(size_t n, uint32_t seed, float scale) {
    std::vector<uint16_t> v(n);
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        const float f = scale * ((float)((s >> 8) & 0xffffu) / 32768.0f - 1.0f);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        v[i] = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    return v;
}

// One geometry. `half` selects which half of the [T, 2, taps, NG] coefficient projection the view
// starts at, so the row pitch (2*taps*NG) is exercised rather than the packed width.
struct Case {
    const char* name;
    int T, H, block, half;
};

bool run_case(const RadKernelInfo* k, const RadKernelInfo* kr, const Case& c) {
    const int taps = 2, grp = 16;
    const int NG = c.H / grp;
    const long long dplane = (long long)c.T * 2 * taps * NG;   /* the whole projection */
    const long long dview = (long long)c.T * taps * NG;        /* the slice's own extent */

    std::vector<uint16_t> x = draw((size_t)c.T * c.H, 0x1234u, 2.0f);
    std::vector<uint16_t> coef = draw((size_t)dplane, 0x5678u, 1.5f);
    std::vector<uint16_t> base = draw((size_t)taps * c.H, 0x9abcu, 1.0f);

    Dev dx((size_t)c.T * c.H * 2), dc((size_t)dplane * 2), db((size_t)taps * c.H * 2);
    Dev dout((size_t)c.T * c.H * 2), dref((size_t)c.T * c.H * 2);
    up(dx, x.data(), x.size() * 2);
    up(dc, coef.data(), coef.size() * 2);
    up(db, base.data(), base.size() * 2);

    /* Both libraries receive the same slice: the half's offset with the projection's row pitch. */
    const size_t off = (size_t)c.half * taps * NG * 2;
    RadTensor tx = ten(dx.p, RAD_BF16, { c.T, c.H });
    RadTensor td = ten((char*)dc.p + off, RAD_BF16, { c.T, taps, NG });
    td.stride[0] = 2LL * taps * NG;
    td.stride[1] = NG;
    td.stride[2] = 1;
    RadTensor tb = ten(db.p, RAD_BF16, { taps, c.H });
    RadTensor to = ten(dout.p, RAD_BF16, { c.T, c.H });
    RadTensor tr = ten(dref.p, RAD_BF16, { c.T, c.H });
    (void)dview;

    std::vector<RadParam> p = { pi("T", c.T), pi("hidden_size", c.H), pi("taps", taps),
                                pi("group_size", grp), pi("NG", NG), pi("block_size", c.block),
                                ps("dtype", "bf16") };

    std::vector<RadTensor> a3 = { tx, td, tb, to };
    const int rc = run(k, a3, p);
    if (rc != RAD_OK) {
        std::fprintf(stderr, "  %s: libr3d refused rc=%d\n", c.name, rc);
        return false;
    }
    std::vector<RadTensor> r3 = { tx, td, tb, tr };
    const int rr = run(kr, r3, p);
    if (rr != RAD_OK) {
        std::fprintf(stderr, "  %s: libref refused rc=%d\n", c.name, rr);
        return false;
    }

    std::vector<uint16_t> got((size_t)c.T * c.H), want((size_t)c.T * c.H);
    down(dout, got.data(), got.size() * 2);
    down(dref, want.data(), want.size() * 2);
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] != want[i]) {
            const size_t t = i / (size_t)c.H, h = i % (size_t)c.H;
            std::fprintf(stderr,
                         "  %s: first difference at [%zu][%zu] (t %% %d = %zu): libr3d 0x%04x, "
                         "libref 0x%04x\n", c.name, t, h, c.block, t % (size_t)c.block,
                         (unsigned)got[i], (unsigned)want[i]);
            return false;
        }
    }
    return true;
}

TEST(dflash_conv_is_declared) {
    if (!ready()) return;
    const RadKernelInfo* k = g_r3d.find("dflash_conv", RAD_DOMAIN_DEVICE, "dflash_conv");
    CHECK(k != nullptr);
    if (!k) return;

    const RadConstraint* taps = constraint(k, "taps");
    const RadConstraint* grp = constraint(k, "group_size");
    const RadConstraint* h = constraint(k, "hidden_size");
    CHECK(taps && taps->op == RAD_C_EQ && taps->ival == 2);
    CHECK(grp && grp->op == RAD_C_EQ && grp->ival == 16);
    CHECK(h && h->op == RAD_C_DIV && h->ival == 128);

    /* The description kbench and the plan size through: four operands, delta at the projection's
     * rank (so the descriptor carries the pitch) and the base weight at [taps, H]. */
    const RadParam pp[] = { pi("T", 4), pi("hidden_size", 5120), pi("taps", 2),
                            pi("group_size", 16), pi("NG", 320), pi("block_size", 4),
                            ps("dtype", "bf16") };
    CHECK(k->opd_shape != nullptr);
    if (!k->opd_shape) return;
    RadOpdDesc x, d, b, o;
    CHECK_EQ(k->opd_shape(pp, 7, 0, &x), RAD_OK);
    CHECK_EQ(x.dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(x.rank, 2u);
    CHECK_EQ(x.shape[0], 4);
    CHECK_EQ(x.shape[1], 5120);
    CHECK_EQ(k->opd_shape(pp, 7, 1, &d), RAD_OK);
    CHECK_EQ(d.rank, 4u);
    CHECK_EQ(d.shape[0], 4);
    CHECK_EQ(d.shape[1], 2);
    CHECK_EQ(d.shape[2], 2);
    CHECK_EQ(d.shape[3], 320);
    CHECK_EQ(k->opd_shape(pp, 7, 2, &b), RAD_OK);
    CHECK_EQ(b.rank, 2u);
    CHECK_EQ(b.shape[0], 2);
    CHECK_EQ(b.shape[1], 5120);
    CHECK_EQ(k->opd_shape(pp, 7, 3, &o), RAD_OK);
    CHECK_EQ(o.shape[0], 4);
    CHECK_EQ(o.shape[1], 5120);
}

TEST(dflash_conv_matches_libref) {
    if (!ready()) return;
    const RadKernelInfo* k = g_r3d.find("dflash_conv", RAD_DOMAIN_DEVICE, "dflash_conv");
    const RadKernelInfo* kr = g_ref.find("dflash_conv", RAD_DOMAIN_HOST, "ref_");
    CHECK(k != nullptr);
    CHECK(kr != nullptr);
    if (!k || !kr) return;

    const Case cases[] = {
        /* The 27B's draft band, both halves of the coefficient projection. */
        { "27b_block4_h1", 4, 5120, 4, 1 },
        { "27b_block4_h0", 4, 5120, 4, 0 },
        /* T not a multiple of the block: rows 0 and 4 must drop the shifted tap. */
        { "block4_T10", 10, 5120, 4, 0 },
        /* A block that is not a power of two: the script takes a modulo, not a mask. */
        { "block3_T12", 12, 128, 3, 1 },
        /* Block 1: every row but row 0 sees only the unshifted tap, so out = gain * x. */
        { "block1_T8", 8, 5120, 1, 0 },
        /* A block wider than T: only row 0 is masked. */
        { "block32_T8", 8, 5120, 32, 1 },
        /* One row, and the divisibility minimum (H = 16 * 8). */
        { "T1", 1, 5120, 4, 0 },
        { "h128", 8, 128, 4, 0 },
    };
    for (const Case& c : cases) CHECK(run_case(k, kr, c));
}

TEST(dflash_conv_refuses_alias_and_packed_delta) {
    if (!ready()) return;
    const RadKernelInfo* k = g_r3d.find("dflash_conv", RAD_DOMAIN_DEVICE, "dflash_conv");
    CHECK(k != nullptr);
    if (!k) return;

    const int T = 4, H = 128, taps = 2, NG = 8;
    Dev dx((size_t)T * H * 2), dc((size_t)T * 2 * taps * NG * 2), db((size_t)taps * H * 2);
    std::vector<uint16_t> z((size_t)T * H, 0);
    up(dx, z.data(), z.size() * 2);
    std::vector<RadParam> p = { pi("T", T), pi("hidden_size", H), pi("taps", taps),
                                pi("group_size", 16), pi("NG", NG), pi("block_size", 4),
                                ps("dtype", "bf16") };

    RadTensor tx = ten(dx.p, RAD_BF16, { T, H });
    RadTensor td = ten(dc.p, RAD_BF16, { T, taps, NG });
    td.stride[0] = 2 * taps * NG;
    td.stride[1] = NG;
    td.stride[2] = 1;
    RadTensor tb = ten(db.p, RAD_BF16, { taps, H });

    /* out aliasing x: tap 1 would read what the block has already overwritten. */
    RadTensor alias = tx;
    std::vector<RadTensor> a = { tx, td, tb, alias };
    CHECK(run(k, a, p) != RAD_OK);

    /* A packed delta pitch is the other half's coefficients one row over. */
    td.stride[0] = (long long)taps * NG;
    Dev dout((size_t)T * H * 2);
    RadTensor to = ten(dout.p, RAD_BF16, { T, H });
    std::vector<RadTensor> b = { tx, td, tb, to };
    CHECK(run(k, b, p) != RAD_OK);

    /* The projection's pitch is served. `b` holds copies of the tensors, so it carries the fix. */
    td.stride[0] = 2 * taps * NG;
    b[1].stride[0] = td.stride[0];
    CHECK_EQ(run(k, b, p), RAD_OK);
}

}  // namespace

RAD_TEST_MAIN()
