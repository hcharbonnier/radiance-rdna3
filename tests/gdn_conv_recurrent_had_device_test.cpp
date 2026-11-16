/* gdn_conv_recurrent_had_device_test.cpp -- libr3d's gdn_conv_recurrent_update_had row against
 * the pair it fuses and against libref, on the card.
 *
 * The row (docs/OPS.md) is gdn_conv_recurrent_update with had_quant_act_i8g's codes of `o` in its
 * epilogue: int8 after a 128-wide Hadamard before the per-(token, head) amax, at group 128 (one
 * scale a value head) or group 256 (IQ8R, one scale a pair of value heads). The cases here run:
 *
 *   - one sequence and four, uneven token counts and q_len up to the eight-token window;
 *   - two steps, so the conv window shifts and the arrival counters must reset;
 *   - conv slots that are not sequence ids, a conv-state ring of eight, and state slots high in
 *     the cache (the long-context / paged shape), in both state forms;
 *   - group 128 and 256, and every byte the fused row writes compared with the non-fused pair;
 *   - a tolerance comparison of `o`, the recurrent state and the conv state against libref's
 *     gdn_conv_recurrent_update, and the int8 codes byte-for-byte against libref's
 *     had_quant_act_i8g over the pair's `o`.
 *
 * The libraries are dlopened and their rows found through rad_kernel_at, as the engine loads them.
 * libref's row reads and writes the same device buffers; on a machine without the card or the
 * plugins every case skips.
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
        radtest::skip("no card");
        return false;
    }
    if (!g_r3d.open(RAD_R3D_SO) || !g_ref.open(RAD_LIBREF_SO)) {
        radtest::artifact_missing(__FILE__, __LINE__,
                                  std::string(RAD_R3D_SO) + " / " + RAD_LIBREF_SO +
                                      " not loadable here");
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

const char* g_cmp = "";
void up(Dev& d, const void* src, size_t bytes) {
    if (bytes && hipMemcpy(d.p, src, bytes, hipMemcpyHostToDevice) != hipSuccess) {
        std::fprintf(stderr, "hipMemcpy H2D %zu failed: %s\n", bytes,
                     hipGetErrorName(hipGetLastError()));
        std::abort();
    }
}
void down(Dev& d, void* dst, size_t bytes) {
    if (bytes && hipMemcpy(dst, d.p, bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        std::fprintf(stderr, "hipMemcpy D2H %zu failed: %s (%s)\n", bytes,
                     hipGetErrorName(hipGetLastError()), g_cmp);
        std::abort();
    }
}

RadTensor ten(void* p, int dtype, std::initializer_list<long long> dims) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = p;
    t.dtype = (uint32_t)dtype;
    t.rank = (uint32_t)dims.size();
    int i = 0;
    for (long long d : dims) t.shape[i++] = d;
    rad_tensor_pack(&t);
    return t;
}

/* A dense [rows, cols] view at `base + col0`, its rows `pitch` elements apart. */
RadTensor view(void* base, int dtype, long long rows, long long cols, long long pitch,
               long long col0, int esz) {
    RadTensor t = ten((char*)base + (size_t)col0 * esz, dtype, { rows, cols });
    t.stride[0] = pitch;
    return t;
}

RadParam pi(const char* k, long long v) { return RAD_INT(k, v); }
RadParam ps(const char* k, const char* v) { return RAD_STR(k, v); }
RadParam pf(const char* k, double v) { return RAD_F64(k, v); }

uint16_t bfv(float v) { return rad_f32_to_bf16(v); }
float bf2f(uint16_t h) { return rad_bf16_to_f32(h); }

int esz(int dt) {
    switch (dt) {
        case RAD_BF16: return 2;
        case RAD_F32: return 4;
        case RAD_I32: return 4;
        default: return 1;
    }
}

int run(const RadKernelInfo* k, std::vector<RadTensor> t, std::vector<RadParam> p) {
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.world_size = 1;
    (void)hipGetLastError();                 /* a stale error is not this launch's */
    const int rc = k->launch(&a, (RadStream)0);
    if (rc != RAD_OK) {
        std::fprintf(stderr, "  launch %s rc=%d (%s)\n", k->name, rc,
                     rc < 0 ? rad_strerror(rc) : hipGetErrorName((hipError_t)rc));
        return rc;
    }
    const hipError_t he = hipDeviceSynchronize();
    if (he != hipSuccess) {
        std::fprintf(stderr, "  %s synchronize: %s\n", k->name, hipGetErrorName(he));
        return RAD_E_DEVICE;
    }
    return RAD_OK;
}

/* ---- comparisons -------------------------------------------------------------------------- */

bool diff_bytes(Dev& a, Dev& b, size_t n, const char* what) {
    g_cmp = what;
    std::vector<uint8_t> A(n), B(n);
    down(a, A.data(), n);
    down(b, B.data(), n);
    size_t diff = 0, first = 0;
    for (size_t i = 0; i < n; ++i)
        if (A[i] != B[i]) { if (!diff) first = i; ++diff; }
    if (diff) {
        std::fprintf(stderr, "%s differs in %zu of %zu bytes (first %zu: %02x vs %02x)\n", what,
                     diff, n, first, A[first], B[first]);
        return false;
    }
    return true;
}

/* Max |a-b| and the worst relative miss against `atol + rtol*|b|`, for bf16 buffers. */
bool close_bf16(Dev& a, Dev& b, size_t n, const char* what, double rtol, double atol) {
    g_cmp = what;
    std::vector<uint16_t> A(n), B(n);
    down(a, A.data(), n * 2);
    down(b, B.data(), n * 2);
    double worst = 0.0, worst_d = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        const double fa = bf2f(A[i]), fb = bf2f(B[i]);
        const double d = std::fabs(fa - fb);
        const double lim = atol + rtol * std::fmax(std::fabs(fa), std::fabs(fb));
        const double r = lim > 0.0 ? d / lim : d;
        if (d > worst_d) { worst_d = d; }
        if (r > worst) worst = r;
        if (d > lim) ++bad;
    }
    if (bad) {
        std::fprintf(stderr, "%s: %zu of %zu beyond %g + %g*|x| (worst |d| %g, ratio %g)\n",
                     what, bad, n, atol, rtol, worst_d, worst);
        return false;
    }
    return true;
}

bool close_f32(Dev& a, Dev& b, size_t n, const char* what, double rtol, double atol) {
    g_cmp = what;
    std::vector<float> A(n), B(n);
    down(a, A.data(), n * 4);
    down(b, B.data(), n * 4);
    double worst = 0.0, worst_d = 0.0;
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)A[i] - (double)B[i]);
        const double lim = atol + rtol * std::fmax(std::fabs((double)A[i]), std::fabs((double)B[i]));
        const double r = lim > 0.0 ? d / lim : d;
        if (d > worst_d) { worst_d = d; }
        if (r > worst) worst = r;
        if (d > lim) ++bad;
    }
    if (bad) {
        std::fprintf(stderr, "%s: %zu of %zu beyond %g + %g*|x| (worst |d| %g, ratio %g)\n",
                     what, bad, n, atol, rtol, worst_d, worst);
        return false;
    }
    return true;
}

/* ---- one geometry ------------------------------------------------------------------------- */

struct Cfg {
    const char* name;
    int N;
    int lens[4];
    int cidx[4];
    int H;
    int Hg;
    int qg;              /* the codes' scale group: 128 or 256 */
    const char* form;    /* "per_candidate" or "anchor" */
    int MQ;              /* q_len: the largest token count in one step */
};

void run_case(const Cfg& c) {
    const int64_t K = 128, V = 128, W = 4;
    const int64_t H = c.H, Hg = c.Hg, N = c.N;
    const int64_t Td = [&] {
        int64_t n = 0;
        for (int i = 0; i < c.N; ++i) n += c.lens[i];
        return n;
    }();
    const int64_t conv_dim = 2 * Hg * K + H * V;
    const int64_t in_w = conv_dim + H * V;
    const int64_t SL = W - 2 + c.MQ;       /* the conv window the launcher sizes */
    const int64_t CSLOTS = 8;              /* the conv-state ring */
    const int64_t COLS = 8;                /* one per candidate, as the engine sizes it (>= 6 for anchor) */
    const int64_t SBASE = 40;              /* the state pool is used high, as a long context is */
    const int64_t SSLOTS = SBASE + N * COLS + 8;
    const bool wide = c.qg == 256;

    const RadKernelInfo* rhad  = g_r3d.find("gdn_conv_recurrent_update_had", RAD_DOMAIN_DEVICE,
                                            "gdn_conv_recurrent_update_had");
    const RadKernelInfo* rconv = g_r3d.find("gdn_conv_update", RAD_DOMAIN_DEVICE,
                                            "gdn_conv_update");
    const RadKernelInfo* rrec  = g_r3d.find("gdn_recurrent_update", RAD_DOMAIN_DEVICE,
                                            "gdn_recurrent_update");
    const RadKernelInfo* rref  = g_ref.find("gdn_conv_recurrent_update", RAD_DOMAIN_HOST, "ref_");
    const RadKernelInfo* rq    = g_ref.find("had_quant_act_i8g", RAD_DOMAIN_HOST, "ref_");
    CHECK(rhad && rconv && rrec && rref && rq);
    if (!(rhad && rconv && rrec && rref && rq)) return;

    uint32_t seed = 0x9e3779b9u ^ (uint32_t)c.qg ^ (uint32_t)(N * 131 + c.MQ);
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (float)((seed >> 8) & 0xFFFFu) * (1.0f / 32768.0f) - 1.0f;
    };
    auto fill_bf = [&](size_t n, float s) {
        std::vector<uint16_t> v(n);
        for (auto& e : v) e = bfv(s * rnd());
        return v;
    };

    const std::vector<uint16_t> hw  = fill_bf((size_t)(conv_dim * W), 0.5f);
    const std::vector<uint16_t> hcs = fill_bf((size_t)(CSLOTS * conv_dim * SL), 1.0f);
    std::vector<float> hst((size_t)(SSLOTS * H * V * K));
    for (auto& e : hst) e = 0.1f * rnd();
    std::vector<float> halog((size_t)H), hdtb((size_t)H), hnw((size_t)V);
    for (int64_t h = 0; h < H; ++h) {
        halog[(size_t)h] = -2.0f + 0.25f * (float)h;
        hdtb[(size_t)h]  = -1.0f + 0.5f * (float)h;
    }
    for (auto& e : hnw) e = 1.0f + 0.25f * rnd();

    std::vector<int32_t> hsidx((size_t)(N * COLS));
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < COLS; ++k)
            hsidx[(size_t)(n * COLS + k)] =
                c.cidx[n] < 0 ? -1 : (int32_t)(SBASE + n * COLS + k);

    /* The anchor form starts from a zeroed factor block: its marker reads as absent, and the
     * scratch replays nothing. The committed anchor columns keep the random state. */
    if (std::strcmp(c.form, "anchor") == 0) {
        for (int64_t n = 0; n < N; ++n)
            for (int64_t k = 3; k < COLS; ++k)
                std::memset(&hst[(size_t)(SBASE + n * COLS + k) * H * V * K], 0,
                            (size_t)(H * V * K) * 4);
    }

    std::vector<int32_t> hcu((size_t)(N + 1), 0);
    for (int64_t n = 0; n < N; ++n) hcu[(size_t)n + 1] = hcu[(size_t)n] + c.lens[n];

    const size_t in_b  = (size_t)(Td * in_w) * 2;
    const size_t ab_b  = (size_t)(Td * 2 * H) * 2;
    const size_t st_b  = hst.size() * 4;
    const size_t cs_b  = hcs.size() * 2;
    const size_t o_b   = (size_t)(Td * H * V) * 2;
    const size_t oq_b  = (size_t)(Td * H * V);
    const size_t os_b  = (size_t)(Td * H * V / c.qg) * 4;

    Dev d_in(in_b), d_ab(ab_b), d_w(hw.size() * 2), d_alog(H * 4), d_dtb(H * 4), d_nw(V * 4);
    Dev d_cidx(N * 4), d_sidx(N * COLS * 4), d_cu((N + 1) * 4), d_nacc(N * 4);
    Dev d_q((size_t)(Td * Hg * K) * 2), d_k((size_t)(Td * Hg * K) * 2),
        d_v((size_t)(Td * H * V) * 2);
    up(d_w, hw.data(), d_w.bytes);
    up(d_alog, halog.data(), d_alog.bytes);
    up(d_dtb, hdtb.data(), d_dtb.bytes);
    up(d_nw, hnw.data(), d_nw.bytes);
    up(d_cidx, c.cidx, d_cidx.bytes);
    up(d_sidx, hsidx.data(), d_sidx.bytes);
    up(d_cu, hcu.data(), d_cu.bytes);
    if (!d_in.p || !d_ab.p || !d_w.p || !d_cu.p) { CHECK(false); return; }

    Dev d_cs[3];
    Dev d_st[3];
    Dev d_o[3];
    Dev d_oq[3];
    Dev d_os[3];
    for (int i = 0; i < 3; ++i) {
        d_cs[i].alloc(cs_b);
        d_st[i].alloc(st_b);
        d_o[i].alloc(o_b);
        d_oq[i].alloc(oq_b);
        d_os[i].alloc(os_b);
        up(d_cs[i], hcs.data(), cs_b);
        up(d_st[i], hst.data(), st_b);
        /* o_q / o_scale are outputs only. */
    }

    const RadTensor tx  = view(d_in.p, RAD_BF16, Td, conv_dim, in_w, 0, 2);
    const RadTensor tz  = view(d_in.p, RAD_BF16, Td, H * V, in_w, conv_dim, 2);
    const RadTensor ta  = view(d_ab.p, RAD_BF16, Td, H, 2 * H, 0, 2);
    const RadTensor tb  = view(d_ab.p, RAD_BF16, Td, H, 2 * H, H, 2);
    const RadTensor tw  = ten(d_w.p, RAD_BF16, { conv_dim, W });
    const RadTensor tal = ten(d_alog.p, RAD_F32, { H });
    const RadTensor tdt = ten(d_dtb.p, RAD_F32, { H });
    const RadTensor tnw = ten(d_nw.p, RAD_F32, { V });
    const RadTensor tci = ten(d_cidx.p, RAD_I32, { N });
    const RadTensor tsi = ten(d_sidx.p, RAD_I32, { N, COLS });
    const RadTensor tcu = ten(d_cu.p, RAD_I32, { N + 1 });
    const RadTensor tna = ten(d_nacc.p, RAD_I32, { N });
    const RadTensor tq  = ten(d_q.p, RAD_BF16, { Td, Hg, K });
    const RadTensor tk  = ten(d_k.p, RAD_BF16, { Td, Hg, K });
    const RadTensor tv  = ten(d_v.p, RAD_BF16, { Td, H, V });
    RadTensor none;
    std::memset(&none, 0, sizeof none);

    auto tcs = [&](int i) { return ten(d_cs[i].p, RAD_BF16, { CSLOTS, conv_dim, SL }); };
    auto tst = [&](int i) { return ten(d_st[i].p, RAD_F32, { SSLOTS, H, V, K }); };
    auto to  = [&](int i) { return ten(d_o[i].p, RAD_BF16, { Td, H, V }); };
    auto toq = [&](int i) { return ten(d_oq[i].p, RAD_I8, { Td, H * V }); };
    auto tos = [&](int i) { return ten(d_os[i].p, RAD_F32, { Td, H * V / c.qg }); };
    auto toflat = [&](int i) { return ten(d_o[i].p, RAD_BF16, { Td, H * V }); };

    std::vector<RadParam> pconv = { pi("q_len", c.MQ), pi("head_k", K), pi("head_v", V),
                                    pi("conv_width", W) };
    std::vector<RadParam> prec = { pi("q_len", c.MQ), pi("head_k", K), pi("head_v", V),
                                   pi("n_head_v", H), pf("scale", 0.08838834764831845),
                                   pf("eps", 1e-6), ps("act", "silu"),
                                   ps("state_form", c.form) };
    std::vector<RadParam> pfused = prec;
    pfused.push_back(pi("conv_width", W));
    pfused.push_back(pi("group", c.qg));
    std::vector<RadParam> pref = prec;
    pref.push_back(pi("conv_width", W));
    std::vector<RadParam> pq = { pi("M", Td), pi("n", H * V), pi("group", c.qg),
                                 ps("dtype", "bf16") };

    std::string tag = std::string(c.name) + " (" + c.form + ", group " +
                      std::to_string(c.qg) + ")";
    bool ok = true;

    for (int step = 0; step < 2 && ok; ++step) {
        /* A decode step's x, a and b; each sequence's accepted prefix is one token at step 0 and
         * two at step 1 (the previous step committed both). */
        const std::vector<uint16_t> hin = fill_bf((size_t)(Td * in_w), 1.0f);
        const std::vector<uint16_t> hab = fill_bf((size_t)(Td * 2 * H), 1.0f);
        up(d_in, hin.data(), in_b);
        up(d_ab, hab.data(), ab_b);
        std::vector<int32_t> nacc((size_t)N);
        for (int64_t n = 0; n < N; ++n)
            nacc[(size_t)n] = step == 0 ? 1 : 2;
        up(d_nacc, nacc.data(), d_nacc.bytes);

        int rc = run(rconv, { tx, tw, none, tcs(0), tci, tna, tcu, tq, tk, tv }, pconv);
        if (rc != RAD_OK) { std::fprintf(stderr, "%s step %d: pair conv rc=%d\n", tag.c_str(),
                                         step, rc); ok = false; break; }
        rc = run(rrec, { tq, tk, tv, ta, tb, tal, tdt, tst(0), tsi, tna, tcu, tz, tnw, to(0),
                         none, none }, prec);
        if (rc != RAD_OK) { std::fprintf(stderr, "%s step %d: pair recurrent rc=%d\n", tag.c_str(),
                                         step, rc); ok = false; break; }
        /* libref's rotated quantiser over the pair's `o`: the expected bytes. */
        rc = run(rq, { toflat(0), toq(0), tos(0) }, pq);
        if (rc != RAD_OK) { std::fprintf(stderr, "%s step %d: quantiser rc=%d\n", tag.c_str(),
                                         step, rc); ok = false; break; }

        rc = run(rhad, { tx, tw, none, tcs(1), tci, ta, tb, tal, tdt, tst(1), tsi, tna, tcu,
                         tz, tnw, to(1), toq(1), tos(1) }, pfused);
        if (rc != RAD_OK) { std::fprintf(stderr, "%s step %d: had rc=%d\n", tag.c_str(), step, rc);
                            ok = false; break; }

        rc = run(rref, { tx, tw, none, tcs(2), tci, ta, tb, tal, tdt, tst(2), tsi, tna, tcu,
                         tz, tnw, to(2), none, none }, pref);
        if (rc != RAD_OK) { std::fprintf(stderr, "%s step %d: libref rc=%d\n", tag.c_str(), step,
                                         rc); ok = false; break; }

        const size_t real = (size_t)hcu[(size_t)N];
        if (!diff_bytes(d_o[0], d_o[1], o_b, (tag + " step " + std::to_string(step) +
                                              ": o vs pair").c_str())) ok = false;
        if (!diff_bytes(d_cs[0], d_cs[1], cs_b, (tag + " step " + std::to_string(step) +
                                                 ": conv_state vs pair").c_str())) ok = false;
        if (!diff_bytes(d_st[0], d_st[1], st_b, (tag + " step " + std::to_string(step) +
                                                 ": state vs pair").c_str())) ok = false;
        if (!diff_bytes(d_oq[0], d_oq[1], real * (size_t)(H * V),
                        (tag + " step " + std::to_string(step) + ": o_q vs libref quantiser over "
                         "the pair's o").c_str())) ok = false;
        if (!diff_bytes(d_os[0], d_os[1], real * (size_t)(H * V / c.qg) * 4,
                        (tag + " step " + std::to_string(step) + ": o_scale vs libref quantiser "
                         "over the pair's o").c_str())) ok = false;

        /* The int8 contract: symmetric (the clamp is +-127, so -128 never appears), and a group's
         * amax reaches the top code. Every scale of a real token is positive and finite. */
        {
            std::vector<int8_t> code((size_t)(real * H * V));
            down(d_oq[1], code.data(), code.size());
            int lo = 127, hi = -128;
            for (int8_t v : code) { lo = v < lo ? v : lo; hi = v > hi ? v : hi; }
            std::vector<float> sc((size_t)(real * (H * V / c.qg)));
            down(d_os[1], sc.data(), sc.size() * 4);
            bool pos = true;
            for (float s : sc) if (!(s > 0.0f) || !std::isfinite(s)) pos = false;
            std::fprintf(stderr, "  %s step %d: codes [%d, %d], scales all positive %d\n",
                         tag.c_str(), step, lo, hi, (int)pos);
            CHECK(lo >= -127 && hi <= 127 && hi == 127 && lo <= -126 && pos);
        }

        /* Against libref's own gdn_conv_recurrent_update (its conv silu and exp differ from the
         * device's fast ones by a last bit, so this is a tolerance, not byte equality). */
        if (!close_bf16(d_o[1], d_o[2], o_b / 2, (tag + " step " + std::to_string(step) +
                                              ": o vs libref").c_str(), 1e-2, 2e-3)) ok = false;
        if (!close_bf16(d_cs[1], d_cs[2], cs_b / 2, (tag + " step " + std::to_string(step) +
                                                 ": conv_state vs libref").c_str(), 2e-2, 4e-3))
            ok = false;
        if (!close_f32(d_st[1], d_st[2], st_b / 4, (tag + " step " + std::to_string(step) +
                                                ": state vs libref").c_str(), 1e-3, 3e-4))
            ok = false;
    }

    CHECK(ok);
}

}  // namespace

/* ---- cases -------------------------------------------------------------------------------- */

static Cfg one_sequence(int qg, const char* form) {
    Cfg c{};
    c.name = qg == 256 ? "one sequence, group 256" : "one sequence";
    c.N = 1;
    c.lens[0] = 4;
    c.cidx[0] = 7;              /* a conv slot that is not the sequence id */
    c.H = 4;
    c.Hg = 2;
    c.qg = qg;
    c.form = form;
    c.MQ = 4;
    return c;
}

static Cfg four_sequences(int qg, const char* form, int MQ) {
    Cfg c{};
    c.name = qg == 256 ? "four sequences, group 256" : "four sequences";
    c.N = 4;
    c.lens[0] = MQ; c.lens[1] = 2; c.lens[2] = 3; c.lens[3] = 1;
    c.cidx[0] = 7; c.cidx[1] = 3; c.cidx[2] = 5; c.cidx[3] = 1;
    c.H = 4;
    c.Hg = 2;
    c.qg = qg;
    c.form = form;
    c.MQ = MQ;
    return c;
}

TEST(gdn_conv_recurrent_had_matches_the_pair_one_sequence) {
    if (!ready()) return;
    run_case(one_sequence(128, "per_candidate"));
    run_case(one_sequence(128, "anchor"));
}

TEST(gdn_conv_recurrent_had_matches_the_pair_four_sequences) {
    if (!ready()) return;
    run_case(four_sequences(128, "per_candidate", 4));
    run_case(four_sequences(128, "anchor", 4));
}

TEST(gdn_conv_recurrent_had_group_256_iq8r) {
    if (!ready()) return;
    run_case(one_sequence(256, "per_candidate"));
    run_case(four_sequences(256, "per_candidate", 4));
    run_case(four_sequences(256, "anchor", 4));
}

/* The largest window the fold carries: eight tokens in one sequence, with a variable M. */
TEST(gdn_conv_recurrent_had_full_window) {
    if (!ready()) return;
    Cfg c{};
    c.name = "full window, two sequences";
    c.N = 2;
    c.lens[0] = 8; c.lens[1] = 5;
    c.cidx[0] = 6; c.cidx[1] = 2;
    c.H = 4; c.Hg = 2;
    c.qg = 128;
    c.form = "per_candidate";
    c.MQ = 8;
    run_case(c);

    c.name = "full window, group 256";
    c.qg = 256;
    c.form = "anchor";
    run_case(c);
}

/* ---- refusals ----------------------------------------------------------------------------- */

/* A minimal valid set for the had row, so each mutation reaches the check it is meant for. */
struct Refusal {
    Dev x{ (size_t)(1 * 1024) * 2 }, w{ (size_t)(1024 * 4) * 2 }, cs{ (size_t)(1024 * 6) * 2 };
    Dev a{ 1 * 4 * 2 }, b{ 1 * 4 * 2 }, alog{ 4 * 4 }, dtb{ 4 * 4 }, st{ (size_t)(12 * 4 * 128 * 128) * 4 };
    Dev cidx{ 4 }, sidx{ 6 * 4 }, nacc{ 4 }, cu{ 2 * 4 }, z{ 1 * 512 * 2 }, nw{ 128 * 4 };
    Dev o{ 1 * 4 * 128 * 2 }, oq{ 1 * 512 }, os{ 1 * 4 * 4 };

    std::vector<RadTensor> tensors(bool with_scale = true) {
        std::vector<RadTensor> t = {
            ten(x.p, RAD_BF16, { 1, 1024 }), ten(w.p, RAD_BF16, { 1024, 4 }),
            RadTensor{}, ten(cs.p, RAD_BF16, { 1, 1024, 6 }),
            ten(cidx.p, RAD_I32, { 1 }), ten(a.p, RAD_BF16, { 1, 4 }),
            ten(b.p, RAD_BF16, { 1, 4 }), ten(alog.p, RAD_F32, { 4 }),
            ten(dtb.p, RAD_F32, { 4 }), ten(st.p, RAD_F32, { 12, 4, 128, 128 }),
            ten(sidx.p, RAD_I32, { 1, 6 }), ten(nacc.p, RAD_I32, { 1 }),
            ten(cu.p, RAD_I32, { 2 }), ten(z.p, RAD_BF16, { 1, 512 }),
            ten(nw.p, RAD_F32, { 128 }), ten(o.p, RAD_BF16, { 1, 4, 128 }),
            ten(oq.p, RAD_I8, { 1, 512 }),
            with_scale ? ten(os.p, RAD_F32, { 1, 4 }) : RadTensor{},
        };
        return t;
    }
    std::vector<RadParam> params() {
        return { pi("q_len", 1), pi("head_k", 128), pi("head_v", 128), pi("conv_width", 4),
                 pi("n_head_v", 4), pf("scale", 0.08838834764831845), pf("eps", 1e-6),
                 ps("act", "silu"), ps("state_form", "per_candidate"), pi("group", 128) };
    }
};

TEST(gdn_conv_recurrent_had_refuses_forms_it_does_not_serve) {
    if (!ready()) return;
    const RadKernelInfo* rhad = g_r3d.find("gdn_conv_recurrent_update_had", RAD_DOMAIN_DEVICE,
                                           "gdn_conv_recurrent_update_had");
    CHECK(rhad);
    if (!rhad) return;

    Refusal r;
    for (Dev* d : { &r.x, &r.w, &r.cs, &r.a, &r.b, &r.alog, &r.dtb, &r.st, &r.cidx, &r.sidx,
                    &r.nacc, &r.cu, &r.z, &r.nw, &r.o, &r.oq, &r.os })
        hipMemset(d->p, 0, d->bytes);
    int32_t cidx = 0, sidx[6] = { 0, 1, 2, 3, 4, 5 }, nacc = 1, cu[2] = { 0, 1 };
    int32_t z[512] = { 0 };
    (void)z;
    up(r.cidx, &cidx, 4);
    up(r.sidx, sidx, sizeof sidx);
    up(r.nacc, &nacc, 4);
    up(r.cu, cu, sizeof cu);

    /* The valid set resolves (the row launches). */
    {
        std::vector<RadTensor> t = r.tensors();
        std::vector<RadParam> p = r.params();
        CHECK_EQ(run(rhad, t, p), RAD_OK);
    }
    /* `group` other than 128 or 256. */
    {
        std::vector<RadTensor> t = r.tensors();
        std::vector<RadParam> p = r.params();
        for (auto& q : p) if (!std::strcmp(q.key, "group")) q.ival = 64;
        CHECK_EQ(run(rhad, t, p), RAD_E_UNSUPPORTED);
    }
    /* An fp8 `o_q` is the base row's, not this one's. */
    {
        std::vector<RadTensor> t = r.tensors();
        std::vector<RadParam> p = r.params();
        t[16].dtype = RAD_F8E4M3;
        CHECK_EQ(run(rhad, t, p), RAD_E_INVAL);
    }
    /* Without `o_scale` the pair is incomplete. */
    {
        std::vector<RadTensor> t = r.tensors(false);
        std::vector<RadParam> p = r.params();
        CHECK_EQ(run(rhad, t, p), RAD_E_INVAL);
    }
    /* A conv width the fold does not carry. */
    {
        std::vector<RadTensor> t = r.tensors();
        std::vector<RadParam> p = r.params();
        for (auto& q : p) if (!std::strcmp(q.key, "conv_width")) q.ival = 3;
        CHECK_EQ(run(rhad, t, p), RAD_E_UNSUPPORTED);
    }
    /* head_k/head_v off the compiled geometry. */
    {
        std::vector<RadTensor> t = r.tensors();
        std::vector<RadParam> p = r.params();
        for (auto& q : p) if (!std::strcmp(q.key, "head_k")) q.ival = 64;
        CHECK_EQ(run(rhad, t, p), RAD_E_SHAPE);
    }
    /* A flattened `o_q` is not [tokens, heads * head_v]. */
    {
        std::vector<RadTensor> t = r.tensors();
        std::vector<RadParam> p = r.params();
        t[16] = ten(r.oq.p, RAD_I8, { 1 * 512 });
        CHECK_EQ(run(rhad, t, p), RAD_E_SHAPE);
    }
    /* A `o_q` base the four-byte stores cannot address. */
    {
        std::vector<RadTensor> t = r.tensors();
        std::vector<RadParam> p = r.params();
        t[16].data = (char*)t[16].data + 2;
        CHECK_EQ(run(rhad, t, p), RAD_E_ALIGN);
    }
    /* A flattened `o_scale` cannot carry one scale a head. */
    {
        std::vector<RadTensor> t = r.tensors();
        std::vector<RadParam> p = r.params();
        t[17] = ten(r.os.p, RAD_F32, { 4 });
        CHECK_EQ(run(rhad, t, p), RAD_E_SHAPE);
    }
}

RAD_TEST_MAIN()
