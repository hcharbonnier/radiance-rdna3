/* qsa_block_key_device_test.cpp -- libr3d's qsa_block_key (r3d_qsa_block_key.hip) against libref.
 *
 * Each case runs libref's host row and libr3d's device row on the same operands and compares the
 * output planes. The contract under test is docs/OPS.md's: the block mean is accumulated in f32
 * and rounded to bf16 before the norm, the RMS norm uses (wadd + w), the key is rotated at the
 * block's first-token position, `mode`/`sections` pick the component each frequency reads, YaRN
 * blends the ladder and scales the pair, and `page[b] < 0` writes nothing.
 *
 * The two libraries are dlopened and their rows found through rad_kernel_at, as the engine loads
 * them. On a machine without the card or the plugins every case skips.
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
RadParam pf(const char* k, double v) { return RAD_F64(k, v); }
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

/* One positive case. `reject` cases only check that both libraries refuse the launch. */
struct Case {
    const char* name;
    int M, n, ratio, rot;
    long long pos0;
    bool has_page, has_pos;
    int pos_cols;          /* 1 for a rank-1 list, 3 for an [M, 3] component plane */
    int out_rows;
    int k_pad, o_pad, pos_pad;   /* extra row pitch; 0 means dense */
    bool explicit_block_size;
    const char* mode;      /* nullptr: absent */
    const char* sections;
    bool yarn_present;
    double yarn_factor, yarn_low, yarn_high, yarn_magnitude;
    int negative_page;     /* -1: none */
    bool two_segments;
};

std::vector<RadParam> params_for(const Case& c) {
    std::vector<RadParam> p = { pi("M", c.M), pi("n", c.n), pi("ratio", c.ratio),
                                pi("rotary_dim", c.rot) };
    if (c.pos0 >= 0) p.push_back(pi("pos0", c.pos0));
    if (c.explicit_block_size) p.push_back(pi("block_size", c.ratio));
    p.push_back(pf("eps", 1e-6));
    p.push_back(pf("theta", 10000000.0));
    p.push_back(pf("scale", 1.0));
    p.push_back(pf("wadd", 1.0));
    p.push_back(ps("dtype", "bf16"));
    if (c.mode) p.push_back(ps("mode", c.mode));
    if (c.sections) p.push_back(ps("sections", c.sections));
    if (c.yarn_present) {
        p.push_back(pf("yarn_factor", c.yarn_factor));
        p.push_back(pf("yarn_low", c.yarn_low));
        p.push_back(pf("yarn_high", c.yarn_high));
        p.push_back(pf("yarn_magnitude", c.yarn_magnitude));
    }
    return p;
}

bool close_enough(float got, float want) {
    const double tol = 1e-5 + 2.0 * 0x1p-8 * std::fabs((double)want);
    return std::fabs((double)got - (double)want) <= tol;
}

bool run_case(const RadKernelInfo* kd, const RadKernelInfo* kr, const Case& c) {
    const int k_pad = c.k_pad > c.n ? c.k_pad : c.n;
    const int o_pad = c.o_pad > c.n ? c.o_pad : c.n;
    const int pos_pad = c.pos_cols == 3 ? (c.pos_pad > 3 ? c.pos_pad : 3) : 0;

    const size_t k_rows = (size_t)c.M * (size_t)c.ratio;
    std::vector<uint16_t> kh(k_rows * (size_t)k_pad);
    for (size_t i = 0; i < k_rows; ++i)
        for (int d = 0; d < c.n; ++d) {
            float v = (float)(((int)(i * 37 + (size_t)d * 11) % 97) - 48) / 53.0f;
            if (d % 31 == 0) v *= 5.0f;
            kh[i * (size_t)k_pad + (size_t)d] = f32_to_bf16(v);
        }
    std::vector<float> wh((size_t)c.n);
    for (int d = 0; d < c.n; ++d) wh[(size_t)d] = 0.1f + (float)((d * 7) % 13) / 40.0f;

    std::vector<int32_t> ph;
    if (c.has_page) {
        ph.resize((size_t)c.M);
        for (int b = 0; b < c.M; ++b) ph[(size_t)b] = c.out_rows - 1 - b;
        if (c.negative_page >= 0 && c.negative_page < c.M)
            ph[(size_t)c.negative_page] = -1;
    }

    std::vector<int32_t> xh;
    if (c.has_pos) {
        if (c.pos_cols == 3) {
            xh.assign((size_t)c.M * (size_t)pos_pad, 0);
            for (int b = 0; b < c.M; ++b) {
                xh[(size_t)b * (size_t)pos_pad + 0] = 10 + 4 * b;
                xh[(size_t)b * (size_t)pos_pad + 1] = 100 + 4 * b;
                xh[(size_t)b * (size_t)pos_pad + 2] = 1000 + 4 * b;
            }
        } else {
            xh.resize((size_t)c.M);
            for (int b = 0; b < c.M; ++b)
                xh[(size_t)b] = c.two_segments && b >= c.M / 2
                                    ? 3 + 4 * (b - c.M / 2)
                                    : 100 + 4 * b;
        }
    }

    const uint16_t sentinel = f32_to_bf16(-7.0f);
    std::vector<uint16_t> out_ref((size_t)c.out_rows * (size_t)o_pad, sentinel);
    std::vector<uint16_t> out_got((size_t)c.out_rows * (size_t)o_pad, sentinel);

    std::vector<RadParam> p = params_for(c);

    /* libref: host pointers straight into the vectors above. */
    {
        std::vector<RadTensor> t = {
            ten(kh.data(), RAD_BF16, { (long long)k_rows, c.n }),
            ten(wh.data(), RAD_F32, { c.n }),
            c.has_page ? ten(ph.data(), RAD_I32, { c.M }) : ten(nullptr, RAD_I32, { c.M }),
            c.has_pos
                ? (c.pos_cols == 3
                       ? ten(xh.data(), RAD_I32, { c.M, c.pos_cols })
                       : ten(xh.data(), RAD_I32, { c.M }))
                : ten(nullptr, RAD_I32, { c.M }),
            ten(out_ref.data(), RAD_BF16, { c.out_rows, c.n }),
        };
        if (c.k_pad) t[0].stride[0] = k_pad;
        if (c.o_pad) t[4].stride[0] = o_pad;
        if (c.has_pos && c.pos_cols == 3 && c.pos_pad) t[3].stride[0] = pos_pad;
        const int rc = run(kr, t, p);
        if (rc != RAD_OK) {
            std::fprintf(stderr, "  %s: libref refused rc=%d\n", c.name, rc);
            return false;
        }
    }

    /* libr3d: the same tensors over device copies. */
    {
        Dev dk(kh.size() * 2), dw(wh.size() * 4), dout(out_got.size() * 2);
        up(dk, kh.data(), dk.bytes);
        up(dw, wh.data(), dw.bytes);
        up(dout, out_got.data(), dout.bytes);
        Dev dpg(c.has_page ? ph.size() * 4 : 0), dpo(c.has_pos ? xh.size() * 4 : 0);
        if (c.has_page) up(dpg, ph.data(), dpg.bytes);
        if (c.has_pos) up(dpo, xh.data(), dpo.bytes);

        std::vector<RadTensor> t = {
            ten(dk.p, RAD_BF16, { (long long)k_rows, c.n }),
            ten(dw.p, RAD_F32, { c.n }),
            c.has_page ? ten(dpg.p, RAD_I32, { c.M }) : ten(nullptr, RAD_I32, { c.M }),
            c.has_pos
                ? (c.pos_cols == 3 ? ten(dpo.p, RAD_I32, { c.M, c.pos_cols })
                                   : ten(dpo.p, RAD_I32, { c.M }))
                : ten(nullptr, RAD_I32, { c.M }),
            ten(dout.p, RAD_BF16, { c.out_rows, c.n }),
        };
        if (c.k_pad) t[0].stride[0] = k_pad;
        if (c.o_pad) t[4].stride[0] = o_pad;
        if (c.has_pos && c.pos_cols == 3 && c.pos_pad) t[3].stride[0] = pos_pad;
        const int rc = run(kd, t, p);
        if (rc != RAD_OK) {
            std::fprintf(stderr, "  %s: libr3d refused rc=%d\n", c.name, rc);
            return false;
        }
        down(dout, out_got.data(), dout.bytes);
    }

    size_t bad = 0, first = 0, exact = 0;
    double worst = 0.0;
    for (int r = 0; r < c.out_rows; ++r)
        for (int d = 0; d < c.n; ++d) {
            const size_t i = (size_t)r * (size_t)o_pad + (size_t)d;
            const float a = bf16_to_f32(out_got[i]);
            const float b = bf16_to_f32(out_ref[i]);
            if (out_got[i] == out_ref[i]) {
                ++exact;
                continue;
            }
            if (!close_enough(a, b)) {
                if (!bad) first = i;
                ++bad;
                worst = std::max(worst, std::fabs((double)a - (double)b) /
                                            std::max(1e-12, std::fabs((double)b)));
            }
        }
    if (bad) {
        const float a = bf16_to_f32(out_got[first]);
        const float b = bf16_to_f32(out_ref[first]);
        std::fprintf(stderr,
                     "  %s: %zu of %zu differ; first row %zu dim %zu: got %.9g want %.9g "
                     "(worst rel %.3g)\n",
                     c.name, bad, (size_t)c.out_rows * (size_t)c.n, first / (size_t)o_pad,
                     first % (size_t)o_pad, (double)a, (double)b, worst);
        return false;
    }
    std::printf("  %-24s %6zu values, %zu exact, %zu within 2 bf16 ulps\n", c.name,
                (size_t)c.out_rows * (size_t)c.n, exact,
                (size_t)c.out_rows * (size_t)c.n - exact);
    return true;
}

bool run_refusal(const RadKernelInfo* kd, const RadKernelInfo* kr, const Case& c,
                 const char* mutation) {
    std::vector<RadParam> p = params_for(c);
    if (!std::strcmp(mutation, "block_size")) p.push_back(pi("block_size", c.ratio + 1));
    if (!std::strcmp(mutation, "odd_rot")) {
        for (RadParam& q : p)
            if (q.key && !std::strcmp(q.key, "rotary_dim")) q.ival = c.rot + 1;
    }
    if (!std::strcmp(mutation, "yarn_scale")) {
        p.push_back(pf("yarn_factor", 2.0));
        p.push_back(pf("yarn_low", 1.0));
        p.push_back(pf("yarn_high", 3.0));
        p.push_back(pf("yarn_magnitude", 1.0));
        for (RadParam& q : p)
            if (q.key && !std::strcmp(q.key, "scale")) q.dval = 2.0;
    }

    const int k_pad = c.k_pad > c.n ? c.k_pad : c.n;
    const size_t k_rows = (size_t)c.M * (size_t)c.ratio;
    std::vector<uint16_t> kh(k_rows * (size_t)k_pad, f32_to_bf16(1.0f));
    std::vector<float> wh((size_t)c.n, 1.0f);
    std::vector<uint16_t> out_ref((size_t)c.out_rows * (size_t)c.n, f32_to_bf16(-7.0f));
    std::vector<RadTensor> t = {
        ten(kh.data(), RAD_BF16, { (long long)k_rows, c.n }),
        ten(wh.data(), RAD_F32, { c.n }),
        ten(nullptr, RAD_I32, { c.M }),
        ten(nullptr, RAD_I32, { c.M }),
        ten(out_ref.data(), RAD_BF16, { c.out_rows, c.n }),
    };
    if (c.k_pad) t[0].stride[0] = k_pad;
    const int ref_rc = run(kr, t, p);

    Dev dk(kh.size() * 2), dw(wh.size() * 4), dout(out_ref.size() * 2);
    up(dk, kh.data(), dk.bytes);
    up(dw, wh.data(), dw.bytes);
    up(dout, out_ref.data(), dout.bytes);
    std::vector<RadTensor> td = {
        ten(dk.p, RAD_BF16, { (long long)k_rows, c.n }),
        ten(dw.p, RAD_F32, { c.n }),
        ten(nullptr, RAD_I32, { c.M }),
        ten(nullptr, RAD_I32, { c.M }),
        ten(dout.p, RAD_BF16, { c.out_rows, c.n }),
    };
    if (c.k_pad) td[0].stride[0] = k_pad;
    const int got_rc = run(kd, td, p);
    const bool ok = ref_rc != RAD_OK && got_rc != RAD_OK;
    if (!ok) {
        std::fprintf(stderr, "  refusal %s: libref rc=%d, libr3d rc=%d\n", mutation, ref_rc,
                     got_rc);
    }
    return ok;
}

TEST(qsa_block_key_matches_libref) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("qsa_block_key", RAD_DOMAIN_DEVICE, "qsa_block_key");
    const RadKernelInfo* kr = g_ref.find("qsa_block_key", RAD_DOMAIN_HOST, "ref_qsa_block_key");
    CHECK(kd && kr);
    if (!kd || !kr) return;

    const Case cases[] = {
        /* One block, and a contiguous run of several. */
        { "one_block_pos0", 1, 128, 4, 64, 0, false, false, 1, 1, 0, 0, 0, false, nullptr, nullptr,
          false, 0, 0, 0, 0, -1, false },
        { "many_blocks_pos0", 9, 128, 4, 64, 36, false, false, 1, 9, 0, 0, 0, false, nullptr,
          nullptr, false, 0, 0, 0, 0, -1, false },
        /* A page list that scatters blocks over a taller store, with one block still open. */
        { "scattered_pages", 7, 128, 4, 64, 36, true, false, 1, 26, 0, 0, 0, false, nullptr,
          nullptr, false, 0, 0, 0, 0, 3, false },
        /* pos as a list, the two runs belonging to different sequences. */
        { "pos_list_two_segments", 6, 128, 4, 64, -1, false, true, 1, 6, 0, 0, 0, false, nullptr,
          nullptr, false, 0, 0, 0, 0, -1, true },
        /* Multi-component rotaries: mrope's contiguous sections, imrope's interleaved ones. */
        { "mrope_sections", 5, 128, 4, 64, -1, true, true, 3, 5, 0, 0, 4, false, "mrope",
          "16 8 8", false, 0, 0, 0, 0, -1, false },
        { "imrope_sections", 5, 128, 4, 64, -1, true, true, 3, 5, 0, 0, 4, false, "imrope",
          "11 11 10", false, 0, 0, 0, 0, -1, false },
        /* YaRN off (the coefficients are present and ignored) and on, alone and with mrope. */
        { "yarn_inactive", 5, 128, 4, 64, 40, false, false, 1, 5, 0, 0, 0, false, nullptr, nullptr,
          true, 0, -5, -9, 2, -1, false },
        { "yarn_active", 5, 128, 4, 64, 40, false, false, 1, 5, 0, 0, 0, false, nullptr, nullptr,
          true, 2, 1, 3, 1.1, -1, false },
        { "yarn_mrope", 4, 128, 4, 64, -1, true, true, 3, 4, 0, 0, 4, false, "mrope", "16 8 8",
          true, 2, 1, 3, 1.25, -1, false },
        /* Non-tight row pitches on k and the output store, and an explicit block_size. */
        { "pitched", 5, 128, 4, 64, 36, true, false, 1, 20, 170, 131, 0, false, nullptr, nullptr,
          false, 0, 0, 0, 0, -1, false },
        { "block_size_explicit", 4, 128, 4, 64, 36, true, false, 1, 4, 0, 0, 0, true, nullptr,
          nullptr, false, 0, 0, 0, 0, -1, false },
        /* No rotary, and mrope with a rank-1 position (which reduces to NeoX). */
        { "rot_zero", 3, 128, 4, 0, 36, false, false, 1, 3, 0, 0, 0, false, nullptr, nullptr, false,
          0, 0, 0, 0, -1, false },
        { "mrope_rank1", 3, 128, 4, 64, -1, false, true, 1, 3, 0, 0, 0, false, "mrope", "16 8 8",
          false, 0, 0, 0, 0, -1, false },
        /* The edges of the width: the whole row rotated, a pooling block of one, a narrow row. */
        { "rot_full", 4, 128, 4, 128, 40, true, false, 1, 4, 0, 0, 0, false, nullptr, nullptr,
          false, 0, 0, 0, 0, -1, false },
        { "ratio_one", 5, 128, 1, 64, 40, false, false, 1, 5, 0, 0, 0, false, nullptr, nullptr,
          false, 0, 0, 0, 0, -1, false },
        { "n_64", 4, 64, 4, 32, 40, false, false, 1, 4, 0, 0, 0, false, nullptr, nullptr, false, 0,
          0, 0, 0, -1, false },
    };
    for (const Case& c : cases) CHECK(run_case(kd, kr, c));
}

TEST(qsa_block_key_refuses_what_libref_refuses) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("qsa_block_key", RAD_DOMAIN_DEVICE, "qsa_block_key");
    const RadKernelInfo* kr = g_ref.find("qsa_block_key", RAD_DOMAIN_HOST, "ref_qsa_block_key");
    CHECK(kd && kr);
    if (!kd || !kr) return;

    const Case base = { "refusal", 4, 128, 4, 64, 36, false, false, 1, 4, 0, 0, 0, false,
                        nullptr, nullptr, false, 0, 0, 0, 0, -1, false };
    CHECK(run_refusal(kd, kr, base, "block_size"));
    CHECK(run_refusal(kd, kr, base, "odd_rot"));
    CHECK(run_refusal(kd, kr, base, "yarn_scale"));
}

}  // namespace

RAD_TEST_MAIN()
