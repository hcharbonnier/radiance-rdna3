/* r3d_rope_imrope_h128_device_test.cpp -- libr3d's rope_imrope_h128 (r3d_rope_imrope_h128.hip)
 * against libref's ref_rope.
 *
 * The band under test is the QSA indexer's query rotary: head_dim 128, rotary_dim 64, n_head 4,
 * n_head_kv 0, mode imrope and sections 11 11 10. Each case builds the same qkv rows and position
 * components, runs ref_rope on the host and the device row on the card, and compares the whole
 * qkv plane (padding included, so a stray write outside a row is caught).
 *
 * Covered here: one token and a long batch, two sequences (non-monotonic positions), the rank-1
 * `[M]` position form, both component-plane layouts (`[M, 3]` token-major and `[3, M]`
 * component-major) with dense and padded pitches, a padded qkv row pitch, YaRN off (coefficients
 * present and ignored) and on, a position scale, and a partial rotary whose head tail must not
 * move. The token-major device plane is transposed to `[3, M]` for ref_rope, which is the layout
 * libref's schema states for every multi-component mode.
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

enum { POS_RANK1 = 0, POS_COMP_MAJOR = 1, POS_TOKEN_MAJOR = 2 };

/* One positive case. `pos_pad` is the row pitch of a rank-2 component plane (0 means dense:
 * M for [3, M], 3 for [M, 3]). `qkv_pad` is the fused row's pitch in bf16 elements. */
struct Case {
    const char* name;
    int M;
    int qkv_pad;
    int pos_layout;
    int pos_pad;
    bool two_segments;
    bool yarn_present;
    double yarn_factor, yarn_low, yarn_high, yarn_magnitude;
    double scale;
    const char* sections;   /* nullptr: absent */
    bool ladder_rotary;
};

std::vector<RadParam> params_for(const Case& c, int hd, int nh, int nkv, int rot) {
    std::vector<RadParam> p = { pi("M", c.M), pi("head_dim", hd), pi("n_head", nh),
                                pi("n_head_kv", nkv), pf("theta", 10000000.0),
                                pf("scale", c.scale), ps("mode", "imrope"),
                                pi("rotary_dim", rot) };
    if (c.sections) p.push_back(ps("sections", c.sections));
    if (c.yarn_present) {
        p.push_back(pf("yarn_factor", c.yarn_factor));
        p.push_back(pf("yarn_low", c.yarn_low));
        p.push_back(pf("yarn_high", c.yarn_high));
        p.push_back(pf("yarn_magnitude", c.yarn_magnitude));
    }
    if (c.ladder_rotary) p.push_back(ps("ladder", "rotary"));
    return p;
}

bool close_enough(float got, float want) {
    const double tol = 1e-5 + 2.0 * 0x1p-8 * std::fabs((double)want);
    return std::fabs((double)got - (double)want) <= tol;
}

/* Position component c of token m. The second sequence restarts, so the batch is not monotonic. */
int32_t pos_at(int M, int m, int c, bool two_segments) {
    const int half = M / 2;
    const int base = (two_segments && m >= half) ? 3 + 4 * (m - half) : 100 + 4 * m;
    return (int32_t)(base + 37 * c);
}

bool run_case(const RadKernelInfo* kd, const RadKernelInfo* kr, const Case& c, int hd, int nh,
              int nkv, int rot) {
    const int width = (nh + 2 * nkv) * hd;
    const int qkv_pad = c.qkv_pad > width ? c.qkv_pad : width;
    const int pos_pitch = c.pos_layout == POS_COMP_MAJOR
                              ? (c.pos_pad > c.M ? c.pos_pad : c.M)
                              : c.pos_layout == POS_TOKEN_MAJOR
                                    ? (c.pos_pad > 3 ? c.pos_pad : 3)
                                    : 0;

    const size_t qkv_n = (size_t)c.M * (size_t)qkv_pad;
    const uint16_t sentinel = f32_to_bf16(-13.0f);
    std::vector<uint16_t> qkv0(qkv_n, sentinel);
    for (int m = 0; m < c.M; ++m)
        for (int d = 0; d < width; ++d) {
            float v = (float)(((m * 53 + d * 29) % 211) - 105) / 71.0f;
            if (d % 37 == 0) v *= 6.0f;
            if (d >= rot) v = (float)(((m * 17 + d * 7) % 89) - 44) / 23.0f;   /* the tail */
            qkv0[(size_t)m * (size_t)qkv_pad + (size_t)d] = f32_to_bf16(v);
        }

    /* Rank-1 positions carry one component; rank-2 carry the three components. */
    std::vector<int32_t> pos1((size_t)c.M);
    for (int m = 0; m < c.M; ++m) pos1[(size_t)m] = pos_at(c.M, m, 0, c.two_segments);

    std::vector<int32_t> pos_cm;
    if (c.pos_layout == POS_COMP_MAJOR)
        pos_cm.assign((size_t)3 * (size_t)pos_pitch, 0);
    std::vector<int32_t> pos_tm;
    if (c.pos_layout == POS_TOKEN_MAJOR)
        pos_tm.assign((size_t)c.M * (size_t)pos_pitch, 0);
    if (c.pos_layout == POS_COMP_MAJOR)
        for (int m = 0; m < c.M; ++m)
            for (int cc = 0; cc < 3; ++cc)
                pos_cm[(size_t)cc * (size_t)pos_pitch + (size_t)m] =
                    pos_at(c.M, m, cc, c.two_segments);
    if (c.pos_layout == POS_TOKEN_MAJOR)
        for (int m = 0; m < c.M; ++m)
            for (int cc = 0; cc < 3; ++cc)
                pos_tm[(size_t)m * (size_t)pos_pitch + (size_t)cc] =
                    pos_at(c.M, m, cc, c.two_segments);

    /* libref: the plane its schema states. A token-major device plane is transposed here. */
    std::vector<int32_t> pos_ref;
    RadTensor tpos_ref;
    if (c.pos_layout == POS_RANK1) {
        tpos_ref = ten(pos1.data(), RAD_I32, { c.M });
    } else {
        const int pitch = c.pos_layout == POS_COMP_MAJOR ? pos_pitch : c.M;
        pos_ref.assign((size_t)3 * (size_t)pitch, 0);
        for (int m = 0; m < c.M; ++m)
            for (int cc = 0; cc < 3; ++cc)
                pos_ref[(size_t)cc * (size_t)pitch + (size_t)m] =
                    pos_at(c.M, m, cc, c.two_segments);
        tpos_ref = ten(pos_ref.data(), RAD_I32, { 3, c.M });
        tpos_ref.stride[0] = pitch;
    }

    std::vector<uint16_t> out_ref = qkv0;
    std::vector<uint16_t> out_got = qkv0;
    std::vector<RadParam> p = params_for(c, hd, nh, nkv, rot);

    {
        RadTensor tq = ten(out_ref.data(), RAD_BF16, { c.M, qkv_pad });
        tq.stride[0] = qkv_pad;
        std::vector<RadTensor> t = { tq, tpos_ref };
        const int rc = run(kr, t, p);
        if (rc != RAD_OK) {
            std::fprintf(stderr, "  %s: libref refused rc=%d\n", c.name, rc);
            return false;
        }
    }

    {
        Dev dq(qkv_n * 2), dp;
        if (c.pos_layout == POS_RANK1)
            dp.alloc(pos1.size() * 4);
        else if (c.pos_layout == POS_COMP_MAJOR)
            dp.alloc(pos_cm.size() * 4);
        else
            dp.alloc(pos_tm.size() * 4);
        up(dq, qkv0.data(), dq.bytes);
        if (c.pos_layout == POS_RANK1) up(dp, pos1.data(), dp.bytes);
        else if (c.pos_layout == POS_COMP_MAJOR) up(dp, pos_cm.data(), dp.bytes);
        else up(dp, pos_tm.data(), dp.bytes);

        RadTensor tq = ten(dq.p, RAD_BF16, { c.M, width });
        tq.stride[0] = qkv_pad;
        RadTensor tp;
        if (c.pos_layout == POS_RANK1)
            tp = ten(dp.p, RAD_I32, { c.M });
        else if (c.pos_layout == POS_COMP_MAJOR) {
            tp = ten(dp.p, RAD_I32, { 3, c.M });
            tp.stride[0] = pos_pitch;
        } else {
            tp = ten(dp.p, RAD_I32, { c.M, 3 });
            tp.stride[0] = pos_pitch;
        }
        std::vector<RadTensor> t = { tq, tp };
        const int rc = run(kd, t, p);
        if (rc != RAD_OK) {
            std::fprintf(stderr, "  %s: libr3d refused rc=%d\n", c.name, rc);
            return false;
        }
        down(dq, out_got.data(), dq.bytes);
    }

    size_t bad = 0, first = 0, exact = 0;
    double worst = 0.0;
    for (size_t i = 0; i < qkv_n; ++i) {
        if (out_got[i] == out_ref[i]) {
            ++exact;
            continue;
        }
        const float a = bf16_to_f32(out_got[i]);
        const float b = bf16_to_f32(out_ref[i]);
        if (!close_enough(a, b)) {
            if (!bad) first = i;
            ++bad;
            worst = std::max(worst, std::fabs((double)a - (double)b) /
                                        std::max(1e-12, std::fabs((double)b)));
        }
    }
    if (bad) {
        const size_t m = first / (size_t)qkv_pad, d = first % (size_t)qkv_pad;
        std::fprintf(stderr,
                     "  %s: %zu of %zu differ; first row %zu dim %zu: got %.9g want %.9g "
                     "(worst rel %.3g)\n", c.name, bad, qkv_n, m, d,
                     (double)bf16_to_f32(out_got[first]), (double)bf16_to_f32(out_ref[first]),
                     worst);
        return false;
    }
    std::printf("  %-26s %7zu values, %zu exact, %zu within 2 bf16 ulps\n", c.name, qkv_n,
                exact, qkv_n - exact);
    return true;
}

bool run_refusal(const RadKernelInfo* kd, const RadKernelInfo* kr, const Case& c, int hd, int nh,
                 int nkv, int rot, const char* mutation) {
    std::vector<RadParam> p = params_for(c, hd, nh, nkv, rot);
    if (!std::strcmp(mutation, "multi_no_sections")) {
        p.erase(std::remove_if(p.begin(), p.end(), [](const RadParam& q) {
            return q.key && !std::strcmp(q.key, "sections");
        }), p.end());
    }
    if (!std::strcmp(mutation, "odd_rot")) {
        for (RadParam& q : p)
            if (q.key && !std::strcmp(q.key, "rotary_dim")) q.ival = 63;
    }
    if (!std::strcmp(mutation, "yarn_scale")) {
        for (RadParam& q : p)
            if (q.key && !std::strcmp(q.key, "scale")) q.dval = 2.5;
        p.push_back(pf("yarn_factor", 2.0));
        p.push_back(pf("yarn_low", 1.0));
        p.push_back(pf("yarn_high", 5.0));
        p.push_back(pf("yarn_magnitude", 1.0));
    }

    const int width = (nh + 2 * nkv) * hd;
    std::vector<uint16_t> qkv((size_t)c.M * (size_t)width, f32_to_bf16(1.0f));
    std::vector<int32_t> pos1((size_t)c.M);
    for (int m = 0; m < c.M; ++m) pos1[(size_t)m] = 100 + m;
    std::vector<int32_t> pos_cm((size_t)3 * (size_t)c.M);
    for (int m = 0; m < c.M; ++m)
        for (int cc = 0; cc < 3; ++cc) pos_cm[(size_t)cc * c.M + m] = 100 + m + 37 * cc;

    const bool multi = c.pos_layout != POS_RANK1;
    RadTensor tq = ten(qkv.data(), RAD_BF16, { c.M, width });
    RadTensor tp = multi ? ten(pos_cm.data(), RAD_I32, { 3, c.M })
                         : ten(pos1.data(), RAD_I32, { c.M });
    std::vector<RadTensor> t = { tq, tp };
    const int ref_rc = run(kr, t, p);

    Dev dq(qkv.size() * 2), dp(multi ? pos_cm.size() * 4 : pos1.size() * 4);
    up(dq, qkv.data(), dq.bytes);
    up(dp, multi ? (const void*)pos_cm.data() : (const void*)pos1.data(), dp.bytes);
    RadTensor dq_t = ten(dq.p, RAD_BF16, { c.M, width });
    RadTensor dp_t = multi ? ten(dp.p, RAD_I32, { 3, c.M }) : ten(dp.p, RAD_I32, { c.M });
    std::vector<RadTensor> td = { dq_t, dp_t };
    const int got_rc = run(kd, td, p);
    const bool ok = ref_rc != RAD_OK && got_rc != RAD_OK;
    if (!ok) {
        std::fprintf(stderr, "  refusal %s: libref rc=%d, libr3d rc=%d\n", mutation, ref_rc,
                     got_rc);
    }
    return ok;
}

TEST(rope_imrope_h128_matches_libref) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("rope", RAD_DOMAIN_DEVICE, "rope_imrope_h128");
    const RadKernelInfo* kr = g_ref.find("rope", RAD_DOMAIN_HOST, "ref_rope");
    CHECK(kd && kr);
    if (!kd || !kr) return;

    const int HD = 128, NH = 4, NKV = 0, ROT = 64;
    const Case cases[] = {
        /* One token, then a run of tokens as one sequence. */
        { "one_token", 1, 0, POS_RANK1, 0, false, false, 0, 0, 0, 0, 1.0, "11 11 10", false },
        { "many_tokens", 7, 0, POS_RANK1, 0, false, false, 0, 0, 0, 0, 1.0, "11 11 10", false },
        { "two_sequences", 8, 0, POS_RANK1, 0, true, false, 0, 0, 0, 0, 1.0, "11 11 10", false },
        /* Both component-plane layouts, dense and pitched. */
        { "components_3xM", 5, 0, POS_COMP_MAJOR, 0, false, false, 0, 0, 0, 0, 1.0, "11 11 10",
          false },
        { "components_3xM_pitched", 6, 0, POS_COMP_MAJOR, 9, false, false, 0, 0, 0, 0, 1.0,
          "11 11 10", false },
        { "components_Mx3", 5, 0, POS_TOKEN_MAJOR, 0, false, false, 0, 0, 0, 0, 1.0, "11 11 10",
          false },
        { "components_Mx3_pitched", 5, 0, POS_TOKEN_MAJOR, 7, false, false, 0, 0, 0, 0, 1.0,
          "11 11 10", false },
        /* A padded fused row: the padding and the head tail past rotary_dim must not move. */
        { "qkv_pitch", 6, 528, POS_RANK1, 0, false, false, 0, 0, 0, 0, 1.0, "11 11 10", false },
        { "qkv_pitch_components", 4, 544, POS_COMP_MAJOR, 0, false, false, 0, 0, 0, 0, 1.0,
          "11 11 10", false },
        /* YaRN on and off, with the explicit `rotary` ladder. */
        { "yarn_off", 5, 0, POS_RANK1, 0, false, true, 0.0, -5.0, -9.0, 2.0, 1.0, "11 11 10",
          false },
        { "yarn_on", 5, 0, POS_RANK1, 0, false, true, 2.0, 1.0, 5.0, 1.1, 1.0, "11 11 10", false },
        { "yarn_on_components", 4, 0, POS_TOKEN_MAJOR, 0, false, true, 3.0, 2.0, 8.0, 1.25, 1.0,
          "11 11 10", false },
        /* A linear position scale without YaRN. */
        { "scale_2_5", 5, 0, POS_RANK1, 0, false, false, 0, 0, 0, 0, 2.5, "11 11 10", false },
        { "ladder_rotary", 4, 0, POS_RANK1, 0, false, false, 0, 0, 0, 0, 1.0, "11 11 10", true },
        /* A different section split, to show the interleaved component map is the reference's. */
        { "sections_8_12_12", 4, 0, POS_COMP_MAJOR, 0, false, false, 0, 0, 0, 0, 1.0, "8 12 12",
          false },
        /* Sections past the rotary width and a trailing separator: ref clamps nothing but the
         * predicate never reaches the excess, and the shim's clamp must keep that truth. */
        { "sections_huge", 4, 0, POS_COMP_MAJOR, 0, false, false, 0, 0, 0, 0, 1.0,
          "10000000000 10000000000 10000000000", false },
        { "sections_trailing_space", 4, 0, POS_COMP_MAJOR, 0, false, false, 0, 0, 0, 0, 1.0,
          "11 11 10 ", false },
        /* The production capacity: M = 4096 at the band's geometry. */
        { "m_4096", 4096, 0, POS_RANK1, 0, false, false, 0, 0, 0, 0, 1.0, "11 11 10", false },
    };
    for (const Case& c : cases) CHECK(run_case(kd, kr, c, HD, NH, NKV, ROT));
}

TEST(rope_imrope_h128_refuses_what_libref_refuses) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("rope", RAD_DOMAIN_DEVICE, "rope_imrope_h128");
    const RadKernelInfo* kr = g_ref.find("rope", RAD_DOMAIN_HOST, "ref_rope");
    CHECK(kd && kr);
    if (!kd || !kr) return;

    const int HD = 128, NH = 4, NKV = 0, ROT = 64;
    const Case multi = { "refusal", 4, 0, POS_COMP_MAJOR, 0, false, false, 0, 0, 0, 0, 1.0,
                         "11 11 10", false };
    const Case one = { "refusal", 4, 0, POS_RANK1, 0, false, false, 0, 0, 0, 0, 1.0, "11 11 10",
                       false };
    CHECK(run_refusal(kd, kr, multi, HD, NH, NKV, ROT, "multi_no_sections"));
    CHECK(run_refusal(kd, kr, one, HD, NH, NKV, ROT, "odd_rot"));
    CHECK(run_refusal(kd, kr, one, HD, NH, NKV, ROT, "yarn_scale"));
}

}  // namespace

RAD_TEST_MAIN()
