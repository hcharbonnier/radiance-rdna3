/* logits_bf16_device_test.cpp -- libr3d's `logits_gemm_bf16` row against libref's `logits_gemm`,
 * on the card, at the Flash-Next band (n_vocab 248320, n_embd 2560, bf16 activation).
 *
 * The row reads the head in either stored form: plain bf16, or codes (E4M3 or int8) with one
 * float scale per scale cell. The container's `output.weight` is `i8*bf16[1x128]` -- one scale a
 * vocabulary row and 128 of K -- while the block-scaled heads use `{128, 128}`. The scale's row
 * and column blocks decide where `scaled_rows`' cover reads it, so both are exercised here, over
 * the decode arm (M = 1, 2, 8, 16) and the prefill tile (M = 32).
 *
 * The device row needs the plan its layout hook publishes at declare; the test calls that hook,
 * as rad-kbench and the engine do, then relayouts the two planes into the joined operand of
 * abi/rad_planes.h and uploads it. libref is called on the host operands. Every case compares the
 * f32 logits by relative L2, the same measure rad-kbench uses.
 *
 * The libraries are dlopened and their rows found through rad_kernel_info, as the engine loads
 * them; on a machine without the card or the plugins every case skips.
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

/* ---- buffers and tensors ------------------------------------------------------------------ */

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

/* The joined operand: one rank, its byte count as the shape, as kf_describe_laid stores it. */
RadTensor ten_joined(void* p, int dtype, long long bytes) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = p;
    t.dtype = (uint32_t)dtype;
    t.rank = 1;
    t.shape[0] = bytes;
    rad_tensor_pack(&t);
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

/* A bf16 value with a spread of exponents and signs, but no infinities or NaNs. */
uint16_t rnd_bf16() {
    const int q = (int)(rnd() % 4001) - 2000;
    return rad_f32_to_bf16((float)q / 64.0f);
}

/* A deterministic int8 plane, filled four bytes a step so the 635 MiB head is cheap to draw. */
void fill_codes(std::vector<int8_t>& v) {
    uint32_t s = 0x9E3779B9u;
    size_t i = 0;
    for (; i + 4 <= v.size(); i += 4) {
        s = s * 1664525u + 1013904223u;
        const uint32_t b = s;
        v[i + 0] = (int8_t)(uint8_t)(b >> 24);
        v[i + 1] = (int8_t)(uint8_t)(b >> 16);
        v[i + 2] = (int8_t)(uint8_t)(b >> 8);
        v[i + 3] = (int8_t)(uint8_t)b;
    }
    for (; i < v.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        v[i] = (int8_t)(uint8_t)(s >> 24);
    }
}

/* The joined form of an affine int8-codes / bf16-scale head with block (sbr, sbc), and the plan
 * the row's layout hook publishes for it. The scale cell is what `scaled_rows`' cover reads: a
 * row block of 1 is Flash-Next's per-row group-128 scale, 128 the block-scaled heads'. */
bool make_joined(const RadKernelInfo* kd, long long N, long long K, int sbr, int sbc,
                 std::vector<uint8_t>* joined, const char* what) {
    const long long sc_rows = (N + sbr - 1) / sbr;
    const long long sc_cols = (K + sbc - 1) / sbc;

    std::vector<int8_t> codes((size_t)(N * K));
    fill_codes(codes);
    std::vector<uint16_t> scale((size_t)(sc_rows * sc_cols));
    for (size_t i = 0; i < scale.size(); ++i)
        scale[i] = rad_f32_to_bf16(0.001f + 0.0002f * (float)(i % 13));

    RadEncoding enc = rad_enc_affine(RAD_I8, RAD_BF16, sbr, sbc);
    const int sel[2] = { 0, 1 };
    const std::vector<RadParam> lp = { pint("n_vocab", N), pint("n_embd", K),
                                       pint("M", 1), pstr("dtype", "bf16") };
    RadTensor pl[2] = { ten2(codes.data(), RAD_I8, N, K, K, 1),
                        ten2(scale.data(), RAD_BF16, sc_rows, sc_cols, sc_cols, 1) };

    RadLayout lay{};
    int rc = kd->layout ? kd->layout(lp.data(), (int)lp.size(), 1, &enc, sel, pl, 2, &lay)
                        : RAD_E_UNSUPPORTED;
    if (rc != RAD_OK) {
        std::fprintf(stderr, "%s: head layout refused rc=%d\n", what, rc);
        return false;
    }
    if (!kd->relayout) {
        std::fprintf(stderr, "%s: the row has no relayout\n", what);
        return false;
    }
    joined->assign((size_t)lay.bytes, 0);
    rc = kd->relayout(lp.data(), (int)lp.size(), 1, &enc, sel, pl, 2, joined->data(), lay.bytes);
    if (rc != RAD_OK) {
        std::fprintf(stderr, "%s: head relayout refused rc=%d\n", what, rc);
        return false;
    }
    return true;
}

struct Result {
    double rel_l2 = 0.0;
    double max_abs = 0.0;
};

/* One launch of both rows over the same host inputs and head; the device row's operands are
 * uploaded, libref reads the host copies. Returns the relative L2 of the device logits. */
bool run_point(const RadKernelInfo* kd, const RadKernelInfo* kr, const std::vector<uint16_t>& x,
               const std::vector<uint16_t>& plain, const std::vector<uint8_t>& joined, bool is_plain,
               int M, long long N, long long K, double tol, Result* out, const char* what) {
    const size_t x_elems = (size_t)M * K;
    const size_t y_elems = (size_t)M * N;
    const size_t w_bytes = is_plain ? (size_t)N * K * 2 : joined.size();

    Dev dx(x_elems * 2), dw(w_bytes), dy(y_elems * 4);
    up(dx, x.data(), dx.bytes);
    up(dw, is_plain ? (const void*)plain.data() : (const void*)joined.data(), dw.bytes);

    std::vector<float> want(y_elems, 0.0f);
    std::vector<float> got(y_elems, 0.0f);

    std::vector<RadParam> p = { pint("M", M), pint("n_vocab", N), pint("n_embd", K),
                                pstr("dtype", "bf16"), pint("impl", 0) };

    RadTensor wd = is_plain ? ten2(dw.p, RAD_BF16, N, K, K, 1)
                            : ten_joined(dw.p, RAD_I8, (long long)joined.size());
    RadTensor wr = is_plain ? ten2((void*)plain.data(), RAD_BF16, N, K, K, 1)
                            : ten_joined((void*)joined.data(), RAD_I8, (long long)joined.size());
    RadTensor td[3] = { ten2(dx.p, RAD_BF16, M, K, K, 1), wd,
                        ten2(dy.p, RAD_F32, M, N, N, 1) };
    RadTensor tr[3] = { ten2((void*)x.data(), RAD_BF16, M, K, K, 1), wr,
                        ten2(want.data(), RAD_F32, M, N, N, 1) };

    RadArgs ad;
    std::memset(&ad, 0, sizeof ad);
    ad.t = td;
    ad.n_t = 3;
    ad.p = p.data();
    ad.n_p = (int)p.size();
    RadArgs ar = ad;
    ar.t = tr;

    const int rd = kd->launch(&ad, (RadStream)0);
    (void)hipDeviceSynchronize();
    const int rr = kr->launch(&ar, (RadStream)0);
    if (rd != 0 || rr != 0) {
        std::fprintf(stderr, "%s M=%d: launch returned device %d, reference %d\n", what, M, rd, rr);
        return false;
    }
    down(dy, got.data(), dy.bytes);

    double num = 0.0, den = 0.0, mx = 0.0;
    for (size_t i = 0; i < y_elems; ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
            std::fprintf(stderr, "%s M=%d: non-finite logit at %zu: %.9g vs %.9g\n", what, M, i,
                         (double)got[i], (double)want[i]);
            return false;
        }
        const double d = (double)got[i] - (double)want[i];
        num += d * d;
        den += (double)want[i] * (double)want[i];
        if (std::fabs(d) > mx) mx = std::fabs(d);
    }
    out->rel_l2 = den > 0.0 ? std::sqrt(num / den) : (num > 0.0 ? 1.0 : 0.0);
    out->max_abs = mx;
    if (out->rel_l2 > tol) {
        std::fprintf(stderr, "%s M=%d: rel_l2 %.3e over tolerance %.3e (max_abs %.3e)\n", what, M,
                     out->rel_l2, tol, out->max_abs);
        return false;
    }
    return true;
}

/* A whole head form, once, over the given decode and prefill heights. */
bool check_form(const RadKernelInfo* kd, const RadKernelInfo* kr, bool is_plain, long long N,
                long long K, int sbr, int sbc, const std::vector<int>& Ms, double tol,
                const char* what) {
    std::vector<uint16_t> plain;
    std::vector<uint8_t> joined;
    if (is_plain) {
        plain.resize((size_t)(N * K));
        for (auto& v : plain) v = rnd_bf16();
    } else if (!make_joined(kd, N, K, sbr, sbc, &joined, what)) {
        return false;
    }

    bool ok = true;
    for (int M : Ms) {
        std::vector<uint16_t> x((size_t)M * K);
        for (auto& v : x) v = rnd_bf16();
        Result r;
        if (!run_point(kd, kr, x, plain, joined, is_plain, M, N, K, tol, &r, what)) ok = false;
    }
    return ok;
}

/* ---- cases -------------------------------------------------------------------------------- */

/* The head the checkpoint actually stores: int8 codes with one bf16 scale a vocabulary row and
 * 128 of K. Before the scale cover was read from the plane's own block, this indexed the plane as
 * a 128x128 grid and every row past the first 128 borrowed another block's scales (rel_l2 ~0.5). */
TEST(logits_bf16_per_row_scale_matches_libref) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("logits_gemm", RAD_DOMAIN_DEVICE, "logits_gemm_bf16");
    const RadKernelInfo* kr = g_ref.find("logits_gemm", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return;

    CHECK(check_form(kd, kr, false, 248320, 2560, 1, 128, { 1, 2, 8, 16, 32 }, 8e-3,
                     "per-row i8 head"));
}

/* The block-scaled form the row served before Flash-Next, unchanged: codes times one scale per
 * 128x128 block. */
TEST(logits_bf16_block_scale_matches_libref) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("logits_gemm", RAD_DOMAIN_DEVICE, "logits_gemm_bf16");
    const RadKernelInfo* kr = g_ref.find("logits_gemm", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return;

    CHECK(check_form(kd, kr, false, 1024, 2560, 128, 128, { 1, 2, 8, 32 }, 8e-3,
                     "block i8 head"));
}

/* The plain bf16 head: `w->dtype == RAD_BF16` reads the rows as stored, at the Flash-Next band. */
TEST(logits_bf16_plain_head_matches_libref) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("logits_gemm", RAD_DOMAIN_DEVICE, "logits_gemm_bf16");
    const RadKernelInfo* kr = g_ref.find("logits_gemm", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return;

    CHECK(check_form(kd, kr, true, 248320, 2560, 0, 0, { 1, 2, 8, 32 }, 1e-4, "plain bf16 head"));
}

/* The row refuses a scale K-block that does not divide the decode arm's 128-k chunk, rather than
 * reading a scale over codes it does not cover. The plan is what the launch consults, so record
 * one whose K-block is 64 and drive a plain-bf16-typed call through the joined form. */
TEST(logits_bf16_refuses_a_fractional_scale_chunk) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("logits_gemm", RAD_DOMAIN_DEVICE, "logits_gemm_bf16");
    CHECK(kd);
    if (!kd) return;

    std::vector<uint8_t> joined;
    if (!make_joined(kd, 256, 2560, 1, 64, &joined, "fractional scale chunk")) return;

    Dev dx((size_t)2560 * 2), dw(joined.size()), dy(256 * 4);
    std::vector<uint16_t> x(2560);
    for (auto& v : x) v = rnd_bf16();
    up(dx, x.data(), dx.bytes);
    up(dw, joined.data(), dw.bytes);

    std::vector<RadParam> p = { pint("M", 1), pint("n_vocab", 256), pint("n_embd", 2560),
                                pstr("dtype", "bf16"), pint("impl", 0) };
    RadTensor td[3] = { ten2(dx.p, RAD_BF16, 1, 2560, 2560, 1),
                        ten_joined(dw.p, RAD_I8, (long long)joined.size()),
                        ten2(dy.p, RAD_F32, 1, 256, 256, 1) };
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = td;
    a.n_t = 3;
    a.p = p.data();
    a.n_p = (int)p.size();
    CHECK_EQ(kd->launch(&a, (RadStream)0), (int)RAD_E_SHAPE);
}

}  // namespace

RAD_TEST_MAIN()
