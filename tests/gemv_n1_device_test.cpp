/* gemv_n1_device_test.cpp -- libr3d's N = 1 bf16 GEMV row against libref's gemm_nt, on the card.
 *
 * The row serves the Flash-Next shared expert's gate: `gemm_nt` over a [M, K] activation and a
 * one-row [1, K] weight, at M = 1, the skinny decode heights and whole prefill chunks. Its
 * contract is libref's ascending-k chain, not merely the same arithmetic: every case compares the
 * two output planes bit for bit, including the padding columns a pitched output leaves behind.
 *
 * The libraries are dlopened and their rows found through rad_kernel_at, as the engine loads them.
 * libref's row reads and writes the same device buffers; on a machine without the card or the
 * plugins every case skips.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

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

/* A 2-D view with its own row and column pitches, which rad_tensor_pack would overwrite. */
RadTensor ten2(void* p, int dtype, long long rows, long long cols, long long rs, long long cs) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = p;
    t.dtype = dtype;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    t.stride[0] = rs;
    t.stride[1] = cs;
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
    (void)hipDeviceSynchronize();
    return rc;
}

uint64_t g_z = 0x243F6A8885A308D3ull;
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

bool same_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const char* what,
               int y_ld) {
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            std::fprintf(stderr, "%s differs at %zu (row %zu, col %zu): %04x vs %04x\n", what, i,
                         y_ld > 0 ? i / (size_t)y_ld : 0, y_ld > 0 ? i % (size_t)y_ld : 0,
                         (unsigned)a[i], (unsigned)b[i]);
            return false;
        }
    return true;
}

/* One launch of both rows over the same inputs, with separate sentinel-filled outputs. */
bool run_case(const RadKernelInfo* kd, const RadKernelInfo* kr, int M, int K, int a_ld, int y_ld,
              const char* what) {
    const size_t a_elems = (size_t)M * a_ld;
    const size_t b_elems = (size_t)K;
    const size_t y_elems = (size_t)M * y_ld;

    std::vector<uint16_t> a(a_elems), b(b_elems);
    for (auto& v : a) v = rnd_bf16();
    for (auto& v : b) v = rnd_bf16();

    Dev da(a_elems * 2), db(b_elems * 2), dy0(y_elems * 2), dy1(y_elems * 2);
    up(da, a.data(), da.bytes);
    up(db, b.data(), db.bytes);
    const std::vector<uint16_t> fill(y_elems, 0x5a5au);
    up(dy0, fill.data(), dy0.bytes);
    up(dy1, fill.data(), dy1.bytes);

    const std::vector<RadParam> p = { pi("M", M), pi("N", 1), pi("K", K), ps("dtype", "bf16") };
    std::vector<RadParam> pp = p;
    std::vector<RadTensor> td = { ten2(da.p, RAD_BF16, M, K, a_ld, 1),
                                  ten2(db.p, RAD_BF16, 1, K, K, 1),
                                  ten2(dy0.p, RAD_BF16, M, 1, y_ld, 1) };
    std::vector<RadTensor> tr = { ten2(da.p, RAD_BF16, M, K, a_ld, 1),
                                  ten2(db.p, RAD_BF16, 1, K, K, 1),
                                  ten2(dy1.p, RAD_BF16, M, 1, y_ld, 1) };

    const int rd = run(kd, td, pp);
    const int rr = run(kr, tr, pp);
    if (rd != 0 || rr != 0) {
        std::fprintf(stderr, "%s: launch returned device %d, reference %d\n", what, rd, rr);
        return false;
    }

    std::vector<uint16_t> outd(y_elems), outr(y_elems);
    down(dy0, outd.data(), dy0.bytes);
    down(dy1, outr.data(), dy1.bytes);
    return same_bf16(outd, outr, what, y_ld);
}

void fill_inputs(std::vector<uint16_t>& a, std::vector<uint16_t>& b) {
    for (auto& v : a) v = rnd_bf16();
    for (auto& v : b) v = rnd_bf16();
}

/* ---- cases -------------------------------------------------------------------------------- */

/* M = 1, 16 and a whole prefill chunk, at the gate's K: dense operands and pitched ones, because
 * the gate is a column slice of the shared projection and the engine binds its row pitch. */
TEST(gemv_n1_matches_libref) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("gemm_nt", RAD_DOMAIN_DEVICE, "gemm_nt_bf16_n1");
    const RadKernelInfo* kr = g_ref.find("gemm_nt", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return;

    const int K = 2560;
    for (int M : { 1, 16, 2048 }) {
        char what[64];
        std::snprintf(what, sizeof what, "dense M=%d", M);
        CHECK(run_case(kd, kr, M, K, K, 1, what));
        std::snprintf(what, sizeof what, "pitched M=%d", M);
        CHECK(run_case(kd, kr, M, K, K + 16, 3, what));
    }
}

/* The row's refusals, reached through its own launch pointer rather than the selector. */
TEST(gemv_n1_refuses_other_bands) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("gemm_nt", RAD_DOMAIN_DEVICE, "gemm_nt_bf16_n1");
    CHECK(kd);
    if (!kd) return;

    const int M = 8, K = 2560;
    std::vector<uint16_t> a((size_t)M * K), b(K);
    fill_inputs(a, b);
    Dev da(a.size() * 2), db(b.size() * 2);
    up(da, a.data(), da.bytes);
    up(db, b.data(), db.bytes);

    {
        /* N is not 1: the m16/tiled rows own any other band. */
        Dev y((size_t)M * 2 * 2);
        std::vector<RadTensor> t = { ten2(da.p, RAD_BF16, M, K, K, 1),
                                     ten2(db.p, RAD_BF16, 2, K, K, 1),
                                     ten2(y.p, RAD_BF16, M, 2, 2, 1) };
        std::vector<RadParam> p = { pi("M", M), pi("N", 2), pi("K", K), ps("dtype", "bf16") };
        CHECK_EQ(run(kd, t, p), (int)RAD_E_SHAPE);
    }
    {
        /* The row reads bf16 only; f32 is another row's band. */
        std::vector<float> af(a.size(), 1.0f), bf((size_t)K, 1.0f), yf((size_t)M, 0.0f);
        Dev daf(af.size() * 4), dbf(bf.size() * 4), dyf(yf.size() * 4);
        up(daf, af.data(), daf.bytes);
        up(dbf, bf.data(), dbf.bytes);
        up(dyf, yf.data(), dyf.bytes);
        std::vector<RadTensor> t = { ten2(daf.p, RAD_F32, M, K, K, 1),
                                     ten2(dbf.p, RAD_F32, 1, K, K, 1),
                                     ten2(dyf.p, RAD_F32, M, 1, 1, 1) };
        std::vector<RadParam> p = { pi("M", M), pi("N", 1), pi("K", K), ps("dtype", "f32") };
        CHECK_EQ(run(kd, t, p), (int)RAD_E_DTYPE);
    }
    {
        /* K not a multiple of 128. */
        const int K2 = K + 64;
        std::vector<uint16_t> a2((size_t)M * K2), b2((size_t)K2);
        fill_inputs(a2, b2);
        Dev da2(a2.size() * 2), db2(b2.size() * 2), dy2((size_t)M * 2);
        up(da2, a2.data(), da2.bytes);
        up(db2, b2.data(), db2.bytes);
        std::vector<RadTensor> t = { ten2(da2.p, RAD_BF16, M, K2, K2, 1),
                                     ten2(db2.p, RAD_BF16, 1, K2, K2, 1),
                                     ten2(dy2.p, RAD_BF16, M, 1, 1, 1) };
        std::vector<RadParam> p = { pi("M", M), pi("N", 1), pi("K", K2), ps("dtype", "bf16") };
        CHECK_EQ(run(kd, t, p), (int)RAD_E_SHAPE);
    }
    {
        /* A row pitch that leaves the row base off the 16-byte load grid. */
        Dev dy((size_t)M * 2);
        std::vector<RadTensor> t = { ten2(da.p, RAD_BF16, M, K, K + 1, 1),
                                     ten2(db.p, RAD_BF16, 1, K, K, 1),
                                     ten2(dy.p, RAD_BF16, M, 1, 1, 1) };
        std::vector<RadParam> p = { pi("M", M), pi("N", 1), pi("K", K), ps("dtype", "bf16") };
        CHECK_EQ(run(kd, t, p), (int)RAD_E_STRIDE);
    }
    {
        /* A misaligned activation base. */
        Dev da_odd(a.size() * 2 + 2), dy((size_t)M * 2);
        std::vector<RadTensor> t = {
            ten2((char*)da_odd.p + 2, RAD_BF16, M, K, K, 1),
            ten2(db.p, RAD_BF16, 1, K, K, 1), ten2(dy.p, RAD_BF16, M, 1, 1, 1) };
        std::vector<RadParam> p = { pi("M", M), pi("N", 1), pi("K", K), ps("dtype", "bf16") };
        CHECK_EQ(run(kd, t, p), (int)RAD_E_ALIGN);
    }
}

}  // namespace

RAD_TEST_MAIN()
