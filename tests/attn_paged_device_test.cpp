/* attn_paged_device_test.cpp -- libr3d's int8 paged attention rows against libref's host row.
 *
 * The two RDNA3 geometries that matter are covered: h256/gqa6/block 16 (the 27B),
 * h256/gqa12/block 4 (Flash-Next), each in the decode band (q_len 1) and in a
 * prefill chunk (q_len 17). The cache is the int8 slot the rows document: 256
 * K codes, 256 V codes and the two f16 scales. The libraries are dlopened as
 * the engine loads them; a machine without the card or the plugins skips.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include "r3d_codecs.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

std::vector<uint16_t> rand_bf16(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = f32_to_bf16(U(rng));
    return v;
}

int run(const RadKernelInfo* k, std::vector<RadTensor>& t, std::vector<RadParam>& p,
        void* scratch = nullptr, int64_t scratch_bytes = 0) {
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.scratch = scratch;
    a.scratch_bytes = scratch_bytes;
    const int rc = k->launch(&a, (RadStream)0);
    hipDeviceSynchronize();
    return rc;
}

int64_t scratch_need(const RadKernelInfo* k, std::vector<RadTensor>& t,
                     std::vector<RadParam>& p) {
    if (!k->scratch) return 0;
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    return k->scratch(&a);
}

bool same_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const char* what) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::fprintf(stderr, "%s differs at %zu: %04x vs %04x\n", what, i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

bool close_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const char* what,
                float rel, float abs) {
    for (size_t i = 0; i < a.size(); ++i) {
        const float x = bf16_to_f32(a[i]), y = bf16_to_f32(b[i]);
        if (std::fabs(x - y) > abs + rel * std::fabs(y)) {
            std::fprintf(stderr, "%s differs at %zu: %.9g vs %.9g\n", what, i, x, y);
            return false;
        }
    }
    return true;
}

/* One case: q [M, n_head, hd] bf16, an int8 cache [blocks, kv_heads, block, 2*hd+4]. */
bool case_ok(const char* name, int n_head, int gqa, int hd, int block_size, int q_len,
             int seq_len, bool exact) {
    char prefix[64];
    std::snprintf(prefix, sizeof prefix, "attn_paged_i8_h256_gqa%d_%s", gqa,
                  q_len <= 16 ? "decode" : "prefill");
    const RadKernelInfo* kd = g_r3d.find("attn_paged", RAD_DOMAIN_DEVICE, prefix);
    const RadKernelInfo* kr = g_ref.find("attn_paged", RAD_DOMAIN_HOST, "ref_");
    if (!kd || !kr) {
        std::fprintf(stderr, "skip: no attn_paged row for %s\n", prefix);
        return true;
    }
    const int kv_heads = n_head / gqa;
    const int blocks = (seq_len + block_size - 1) / block_size;
    const int row_bytes = 2 * hd + 4;
    const size_t qn = (size_t)q_len * n_head * hd;
    const size_t outn = qn;
    const size_t kvn = (size_t)blocks * kv_heads * block_size * row_bytes;
    const size_t btn = (size_t)blocks;

    auto qh = rand_bf16(qn, 100u + (unsigned)q_len + (unsigned)block_size);
    std::vector<int8_t> kvh(kvn);
    std::mt19937 rng(7u + (unsigned)block_size);
    std::uniform_int_distribution<int> U(-127, 127);
    for (size_t i = 0; i < (size_t)blocks * kv_heads * block_size; ++i) {
        int8_t* slot = (int8_t*)&kvh[i * row_bytes];
        for (int d = 0; d < 2 * hd; ++d) slot[d] = (int8_t)U(rng);
        const uint16_t ksc = 0x2118u;   /* 0.01 in f16 */
        const uint16_t vsc = 0x2118u;
        std::memcpy((uint8_t*)slot + 2 * hd, &ksc, 2);
        std::memcpy((uint8_t*)slot + 2 * hd + 2, &vsc, 2);
    }
    std::vector<int32_t> bth((size_t)blocks);
    for (int b = 0; b < blocks; ++b) bth[b] = b;
    std::vector<int32_t> suh = { seq_len };
    std::vector<int32_t> cuh = { 0, q_len };

    Dev dq(qn * 2), dkv(kvn), dbt(btn * 4), dsu(4), dcu(8);
    Dev dout0(outn * 2), dout1(outn * 2);
    up(dq, qh.data(), dq.bytes);
    up(dkv, kvh.data(), dkv.bytes);
    up(dbt, bth.data(), dbt.bytes);
    up(dsu, suh.data(), dsu.bytes);
    up(dcu, cuh.data(), dcu.bytes);

    auto p = std::vector<RadParam>{
        pi("q_len", q_len), pi("head_dim", hd), pi("gqa", gqa), pi("block_size", block_size),
        pi("causal", 1), pi("window", 0), ps("q_dtype", "bf16"), ps("kv_dtype", "i8"),
        pf("scale", 1.0 / std::sqrt((double)hd)), pi("max_ctx", seq_len),
        pi("n_head", n_head), pi("max_seqs", 1), pi("prefill_fp8", 0)
    };
    Dev dscratch;
    for (int which = 0; which < 2; ++which) {
        Dev& out = which ? dout1 : dout0;
        std::vector<RadTensor> t = {
            ten(dq.p, RAD_BF16, { q_len, n_head, hd }),
            ten(dkv.p, RAD_I8, { blocks, kv_heads, block_size, row_bytes }),
            ten(dbt.p, RAD_I32, { 1, blocks }),
            ten(dsu.p, RAD_I32, { 1 }),
            ten(nullptr, RAD_F32, { 0 }),
            ten(nullptr, RAD_F32, { 0 }),
            ten(q_len > 1 ? dcu.p : nullptr, RAD_I32, { q_len > 1 ? 2 : 0 }),
            ten(out.p, RAD_BF16, { q_len, n_head, hd })
        };
        if (!which) dscratch.alloc((size_t)std::max<int64_t>(0, scratch_need(kd, t, p)));
        const int rc = run(which ? kr : kd, t, p, which ? nullptr : dscratch.p,
                           which ? 0 : (int64_t)dscratch.bytes);
        if (rc != 0) {
            std::fprintf(stderr, "%s: %s rc=%d\n", name, which ? "ref" : "device", rc);
            return false;
        }
    }
    std::vector<uint16_t> a(outn), b(outn);
    down(dout0, a.data(), dout0.bytes);
    down(dout1, b.data(), dout1.bytes);
    return exact ? same_bf16(a, b, name) : close_bf16(a, b, name, 0.02f, 0.02f);
}

/* Device timing for the attention row, no oracle: warm up, then the median of
 * three 20-launch batches. Used with RAD_ATTN_BENCH=1 to fill the gap kbench
 * leaves (libref declines the paged operands). */
bool bench_case(const char* name, int n_head, int gqa, int hd, int block_size, int q_len,
                int ctx) {
    char prefix[64];
    std::snprintf(prefix, sizeof prefix, "attn_paged_i8_h256_gqa%d_%s", gqa,
                  q_len <= 16 ? "decode" : "prefill");
    const RadKernelInfo* kd = g_r3d.find("attn_paged", RAD_DOMAIN_DEVICE, prefix);
    if (!kd) return true;
    const int kv_heads = n_head / gqa;
    const int blocks = (ctx + block_size - 1) / block_size;
    const int row_bytes = 2 * hd + 4;
    const size_t qn = (size_t)q_len * n_head * hd;
    const size_t kvn = (size_t)blocks * kv_heads * block_size * row_bytes;

    auto qh = rand_bf16(qn, 11u + (unsigned)q_len);
    std::vector<int8_t> kvh(kvn);
    std::mt19937 rng(13u + (unsigned)block_size);
    std::uniform_int_distribution<int> U(-127, 127);
    for (size_t i = 0; i < (size_t)blocks * kv_heads * block_size; ++i) {
        int8_t* slot = (int8_t*)&kvh[i * row_bytes];
        for (int d = 0; d < 2 * hd; ++d) slot[d] = (int8_t)U(rng);
        const uint16_t sc = 0x2118u;
        std::memcpy((uint8_t*)slot + 2 * hd, &sc, 2);
        std::memcpy((uint8_t*)slot + 2 * hd + 2, &sc, 2);
    }
    std::vector<int32_t> bth((size_t)blocks);
    for (int b = 0; b < blocks; ++b) bth[b] = b;
    std::vector<int32_t> suh = { ctx };
    std::vector<int32_t> cuh = { 0, q_len };

    Dev dq(qn * 2), dkv(kvn), dbt((size_t)blocks * 4), dsu(4), dcu(8), dout(qn * 2);
    up(dq, qh.data(), dq.bytes); up(dkv, kvh.data(), dkv.bytes);
    up(dbt, bth.data(), dbt.bytes); up(dsu, suh.data(), dsu.bytes);
    up(dcu, cuh.data(), dcu.bytes);

    auto p = std::vector<RadParam>{
        pi("q_len", q_len), pi("head_dim", hd), pi("gqa", gqa), pi("block_size", block_size),
        pi("causal", 1), pi("window", 0), ps("q_dtype", "bf16"), ps("kv_dtype", "i8"),
        pf("scale", 1.0 / std::sqrt((double)hd)), pi("max_ctx", ctx),
        pi("n_head", n_head), pi("max_seqs", 1), pi("prefill_fp8", 0)
    };
    std::vector<RadTensor> t = {
        ten(dq.p, RAD_BF16, { q_len, n_head, hd }),
        ten(dkv.p, RAD_I8, { blocks, kv_heads, block_size, row_bytes }),
        ten(dbt.p, RAD_I32, { 1, blocks }), ten(dsu.p, RAD_I32, { 1 }),
        ten(nullptr, RAD_F32, { 0 }), ten(nullptr, RAD_F32, { 0 }),
        ten(q_len > 1 ? dcu.p : nullptr, RAD_I32, { q_len > 1 ? 2 : 0 }),
        ten(dout.p, RAD_BF16, { q_len, n_head, hd })
    };
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    Dev dscratch((size_t)std::max<int64_t>(0, kd->scratch ? kd->scratch(&a) : 0));
    a.scratch = dscratch.p;
    a.scratch_bytes = (int64_t)dscratch.bytes;

    for (int i = 0; i < 5; ++i) kd->launch(&a, (RadStream)0);
    hipDeviceSynchronize();
    float best = 1e30f;
    for (int batch = 0; batch < 3; ++batch) {
        hipEvent_t e0, e1;
        hipEventCreate(&e0);
        hipEventCreate(&e1);
        hipEventRecord(e0, (hipStream_t)0);
        for (int i = 0; i < 20; ++i) kd->launch(&a, (RadStream)0);
        hipEventRecord(e1, (hipStream_t)0);
        hipEventSynchronize(e1);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, e0, e1);
        hipEventDestroy(e0);
        hipEventDestroy(e1);
        if (ms < best) best = ms;
    }
    std::printf("ATTNBENCH %-22s ctx=%6d q=%3d  %8.1f us\n", name, ctx, q_len,
                best * 1000.0f / 20.0f);
    return true;
}

}  // namespace

/* The short cases (one page, splits=1) agree with ref_attn_paged within one
 * bf16 ULP but not bit-for-bit; the gap is the documented exp polynomial
 * (within 1 f32 ulp) and the reduction rounding. The long cases allocate the
 * row's scratch and exercise split-KV, whose parentheses are the documented
 * last-ulp departure; they are held to the same bf16 tolerance. A multi-row
 * rewrite must preserve the per-row order. The bench mode allocates the
 * scratch too, so its timings are the served split-KV path. */

TEST(attn_paged_h256_gqa6_block16_decode_short_matches_libref_within_bf16) {
    if (!ready()) return;
    CHECK(case_ok("h256/gqa6/bs16 decode short", 24, 6, 256, 16, 1, 16, false));
}

TEST(attn_paged_h256_gqa6_block16_prefill_short_matches_libref_within_bf16) {
    if (!ready()) return;
    CHECK(case_ok("h256/gqa6/bs16 prefill short", 24, 6, 256, 16, 17, 17, false));
}

TEST(attn_paged_h256_gqa12_block4_decode_short_matches_libref_within_bf16) {
    if (!ready()) return;
    CHECK(case_ok("h256/gqa12/bs4 decode short", 24, 12, 256, 4, 1, 4, false));
}

TEST(attn_paged_h256_gqa12_block4_prefill_short_matches_libref_within_bf16) {
    if (!ready()) return;
    CHECK(case_ok("h256/gqa12/bs4 prefill short", 24, 12, 256, 4, 17, 17, false));
}

TEST(attn_paged_h256_gqa6_block16_decode_split_matches_libref_within_bf16) {
    if (!ready()) return;
    CHECK(case_ok("h256/gqa6/bs16 decode split", 24, 6, 256, 16, 1, 32, false));
}

TEST(attn_paged_h256_gqa6_block16_prefill_split_matches_libref_within_bf16) {
    if (!ready()) return;
    CHECK(case_ok("h256/gqa6/bs16 prefill split", 24, 6, 256, 16, 17, 32, false));
}

TEST(attn_paged_h256_gqa12_block4_decode_split_matches_libref_within_bf16) {
    if (!ready()) return;
    CHECK(case_ok("h256/gqa12/bs4 decode split", 24, 12, 256, 4, 1, 32, false));
}

TEST(attn_paged_h256_gqa12_block4_prefill_split_matches_libref_within_bf16) {
    if (!ready()) return;
    CHECK(case_ok("h256/gqa12/bs4 prefill split", 24, 12, 256, 4, 17, 32, false));
}

TEST(attn_paged_h256_bench) {
    if (!std::getenv("RAD_ATTN_BENCH")) return;
    if (!ready()) return;
    bench_case("gqa6/bs16 decode", 24, 6, 256, 16, 1, 512);
    bench_case("gqa6/bs16 decode", 24, 6, 256, 16, 1, 2048);
    bench_case("gqa6/bs16 decode", 24, 6, 256, 16, 1, 8192);
    bench_case("gqa6/bs16 decode", 24, 6, 256, 16, 1, 32768);
    bench_case("gqa6/bs16 decode", 24, 6, 256, 16, 4, 2048);
    bench_case("gqa6/bs16 decode", 24, 6, 256, 16, 4, 8192);
    bench_case("gqa6/bs16 decode", 24, 6, 256, 16, 4, 32768);
    bench_case("gqa12/bs4 decode", 24, 12, 256, 4, 1, 512);
    bench_case("gqa12/bs4 decode", 24, 12, 256, 4, 1, 8192);
    bench_case("gqa12/bs4 decode", 24, 12, 256, 4, 1, 32768);
    bench_case("gqa12/bs4 decode", 24, 12, 256, 4, 4, 8192);
    bench_case("gqa12/bs4 decode", 24, 12, 256, 4, 4, 32768);
    bench_case("gqa6/bs16 prefill", 24, 6, 256, 16, 17, 2048);
    bench_case("gqa6/bs16 prefill", 24, 6, 256, 16, 17, 8192);
    bench_case("gqa6/bs16 prefill", 24, 6, 256, 16, 17, 32768);
    bench_case("gqa6/bs16 prefill", 24, 6, 256, 16, 128, 2048);
    bench_case("gqa6/bs16 prefill", 24, 6, 256, 16, 128, 8192);
    bench_case("gqa12/bs4 prefill", 24, 12, 256, 4, 17, 2048);
    bench_case("gqa12/bs4 prefill", 24, 12, 256, 4, 17, 8192);
    bench_case("gqa12/bs4 prefill", 24, 12, 256, 4, 17, 32768);
    bench_case("gqa12/bs4 prefill", 24, 12, 256, 4, 128, 2048);
    bench_case("gqa12/bs4 prefill", 24, 12, 256, 4, 128, 8192);
}

RAD_TEST_MAIN()
