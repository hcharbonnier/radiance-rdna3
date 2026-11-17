/* act_i8g_group256_device_test.cpp -- libr3d's rotated int8 activation producers at IQ8R's group
 * 256, against an independent host reference, on the card.
 *
 * The three entries are rmsnorm_had_quant_i8g, had_quant_act_i8g and gated_had_quant_i8g. At group
 * 256 a group is two independent 128-wide FWHTs (each half of the group), then one amax over the
 * whole 256 columns, one f32 scale amax / 127 per (row, 256), and int8 codes in plain row-major
 * order in both halves. The 1/128 factor of the unnormalised transform stays on the weight's
 * scale, so nothing is folded into the activation scale.
 *
 * The reference here is written out in the test: the same butterfly network and pairing as libref's
 * fwht, the sequential amax, the same clamp-then-round-to-nearest-even code, and for the fused
 * norm the same workgroup reduction order as r3d_common.h's block_sum. Cases cover M = 1 and
 * several rows, group 256 and group 128 (the non-regression case), the gated silu and sigmoid
 * modes with and without out_bf16, the residual and no-residual norm forms, bf16 and f32 gains, and
 * the shapes each shim must refuse.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include "r3d_codecs.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef RAD_R3D_SO
#define RAD_R3D_SO "libr3d.so"
#endif

namespace {

typedef int (*LaunchFn)(const RadArgs*, RadStream);

struct Lib {
    void* h = nullptr;
    LaunchFn had = nullptr;
    LaunchFn gated = nullptr;
    LaunchFn gate = nullptr;
    LaunchFn rms = nullptr;
    bool open() {
        h = dlopen(RAD_R3D_SO, RTLD_NOW | RTLD_LOCAL);
        if (!h) return false;
        had = (LaunchFn)dlsym(h, "r3d_rad_had_quant_act_i8g");
        gated = (LaunchFn)dlsym(h, "r3d_rad_gated_had_quant_i8g");
        gate = (LaunchFn)dlsym(h, "r3d_rad_gate_had_quant_i8g");
        rms = (LaunchFn)dlsym(h, "r3d_rad_rmsnorm_had_quant_i8g");
        return had && gated && gate && rms;
    }
};

Lib g_lib;

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
    if (!g_lib.open()) {
        radtest::artifact_missing(__FILE__, __LINE__,
                                  std::string(RAD_R3D_SO) + " not loadable here");
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
    if (bytes && hipMemcpy(d.p, src, bytes, hipMemcpyHostToDevice) != hipSuccess) {
        std::fprintf(stderr, "hipMemcpy H2D %zu failed: %s\n", bytes,
                     hipGetErrorName(hipGetLastError()));
        std::abort();
    }
}
void down(const Dev& d, void* dst, size_t bytes) {
    if (bytes && hipMemcpy(dst, d.p, bytes, hipMemcpyDeviceToHost) != hipSuccess) {
        std::fprintf(stderr, "hipMemcpy D2H %zu failed: %s\n", bytes,
                     hipGetErrorName(hipGetLastError()));
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

RadParam pi(const char* k, long long v) { return RAD_INT(k, v); }
RadParam ps(const char* k, const char* v) { return RAD_STR(k, v); }
RadParam pf(const char* k, double v) { return RAD_F64(k, v); }

uint16_t bfv(float v) { return rad_f32_to_bf16(v); }
float bf2f(uint16_t h) { return rad_bf16_to_f32(h); }

int run(LaunchFn fn, std::vector<RadTensor> t, std::vector<RadParam> p) {
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.world_size = 1;
    (void)hipGetLastError();
    const int rc = fn(&a, (RadStream)0);
    if (rc != RAD_OK) return rc;
    return hipDeviceSynchronize() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

uint32_t g_seed = 0x8f3c2a17u;
float rnd() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (float)((g_seed >> 8) & 0xFFFFu) * (1.0f / 32768.0f) - 1.0f;
}

/* ---- the reference ------------------------------------------------------------------------ */

/* libref's fwht network: stride doubles, the low half keeps a + b, the high half a - b. The
 * device's r3d_fwht_wave applies the same stages in the same order, so the results are bit-exact. */
void fwht(float* v, int g) {
    for (int h = 1; h < g; h <<= 1)
        for (int i = 0; i < g; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
}

int8_t i8code(float v, float inv) {
    float c = v * inv;
    if (c > 127.0f) c = 127.0f;
    if (c < -127.0f) c = -127.0f;
    return f32_to_i8_rtne(c);
}

/* The group's scale amax / 127 (1 for an all-zero group) and its row-major codes. */
float quant_group(const float* v, int g, int8_t* q) {
    float m = 0.0f;
    for (int i = 0; i < g; ++i) {
        const float a = std::fabs(v[i]);
        if (a > m) m = a;
    }
    const float sc = m > 0.0f ? m / 127.0f : 1.0f;
    const float inv = 1.0f / sc;
    for (int i = 0; i < g; ++i) q[i] = i8code(v[i], inv);
    return sc;
}

/* Two 128-wide rotations a group, one amax and one scale over the group's 256 columns. */
void ref_had(const std::vector<uint16_t>& x, int M, int n, int group, std::vector<int8_t>& q,
             std::vector<float>& s) {
    q.assign((size_t)M * n, 0);
    s.assign((size_t)M * (n / group), 0.0f);
    for (int r = 0; r < M; ++r) {
        for (int gi = 0; gi < n / group; ++gi) {
            float buf[256];
            const size_t xb = (size_t)r * n + (size_t)gi * group;
            for (int i = 0; i < group; ++i) buf[i] = bf2f(x[xb + (size_t)i]);
            for (int b = 0; b < group; b += 128) fwht(buf + b, 128);
            s[(size_t)r * (n / group) + (size_t)gi] =
                quant_group(buf, group, &q[xb]);
        }
    }
}

void ref_gated(const std::vector<uint16_t>& gu, int M, int n, long long gu_ld, int group,
               bool sigmoid, std::vector<uint16_t>* y, std::vector<int8_t>& q,
               std::vector<float>& s) {
    q.assign((size_t)M * n, 0);
    s.assign((size_t)M * (n / group), 0.0f);
    if (y) y->assign((size_t)M * n, 0);
    for (int r = 0; r < M; ++r) {
        for (int gi = 0; gi < n / group; ++gi) {
            float buf[256];
            const size_t gb = (size_t)r * (size_t)gu_ld + (size_t)gi * group;
            for (int i = 0; i < group; ++i) {
                const float xg = bf2f(gu[gb + (size_t)i]);
                const float up = bf2f(gu[gb + (size_t)n + (size_t)i]);
                const float sg = 1.0f / (1.0f + (float)std::exp((double)-xg));
                const float p = sigmoid ? sg : xg * sg;
                const uint16_t hb = rad_f32_to_bf16(p * up);
                if (y) (*y)[(size_t)r * n + (size_t)gi * group + (size_t)i] = hb;
                buf[i] = bf2f(hb);
            }
            for (int b = 0; b < group; b += 128) fwht(buf + b, 128);
            s[(size_t)r * (n / group) + (size_t)gi] =
                quant_group(buf, group, &q[(size_t)r * n + (size_t)gi * group]);
        }
    }
}

/* The same product with the gate and x in separate planes of their own pitch. */
void ref_gate(const std::vector<uint16_t>& gate, const std::vector<uint16_t>& x, int M, int n,
              long long gate_ld, long long x_ld, int group, bool sigmoid,
              std::vector<uint16_t>* y, std::vector<int8_t>& q, std::vector<float>& s) {
    q.assign((size_t)M * n, 0);
    s.assign((size_t)M * (n / group), 0.0f);
    if (y) y->assign((size_t)M * n, 0);
    for (int r = 0; r < M; ++r) {
        for (int gi = 0; gi < n / group; ++gi) {
            float buf[256];
            const size_t gb = (size_t)r * (size_t)gate_ld + (size_t)gi * group;
            const size_t xb = (size_t)r * (size_t)x_ld + (size_t)gi * group;
            for (int i = 0; i < group; ++i) {
                const float xg = bf2f(gate[gb + (size_t)i]);
                const float sg = 1.0f / (1.0f + (float)std::exp((double)-xg));
                const float p = sigmoid ? sg : xg * sg;
                const uint16_t hb = rad_f32_to_bf16(p * bf2f(x[xb + (size_t)i]));
                if (y) (*y)[(size_t)r * n + (size_t)gi * group + (size_t)i] = hb;
                buf[i] = bf2f(hb);
            }
            for (int b = 0; b < group; b += 128) fwht(buf + b, 128);
            s[(size_t)r * (n / group) + (size_t)gi] =
                quant_group(buf, group, &q[(size_t)r * n + (size_t)gi * group]);
        }
    }
}

/* The fused norm: the residual sum rounded to bf16, the variance over the stored bytes, then
 * block_sum's order (each thread's strided partial, the wave's xor butterflies, the waves added in
 * order), the bf16 norm output, and the same rotation and quantiser. */
void ref_rms(const std::vector<uint16_t>& x, const std::vector<uint16_t>* res,
             const std::vector<uint16_t>* wb, const std::vector<float>* wf, int M, int n,
             int group, float eps, float wadd, std::vector<uint16_t>& nr,
             std::vector<uint16_t>& ob, std::vector<int8_t>& q, std::vector<float>& s) {
    nr.assign((size_t)M * n, 0);
    ob.assign((size_t)M * n, 0);
    q.assign((size_t)M * n, 0);
    s.assign((size_t)M * (n / group), 0.0f);
    const int THR = n <= 256 ? 32 : 256;
    const int NW = THR / 32;
    for (int r = 0; r < M; ++r) {
        const size_t xb = (size_t)r * n;
        for (int i = 0; i < n; ++i) {
            nr[xb + (size_t)i] =
                res ? rad_f32_to_bf16(bf2f(x[xb + (size_t)i]) + bf2f((*res)[xb + (size_t)i]))
                    : x[xb + (size_t)i];
        }
        float ss[256];
        for (int t = 0; t < THR; ++t) {
            float a = 0.0f;
            for (int i = t; i < n; i += THR) {
                const float v = bf2f(nr[xb + (size_t)i]);
                a += v * v;
            }
            ss[t] = a;
        }
        float wv[32];
        float waves[8];
        for (int wv_i = 0; wv_i < NW; ++wv_i) {
            for (int l = 0; l < 32; ++l) wv[l] = ss[wv_i * 32 + l];
            for (int N = 16; N >= 1; N >>= 1) {
                float t2[32];
                for (int l = 0; l < 32; ++l) t2[l] = wv[l] + wv[l ^ N];
                for (int l = 0; l < 32; ++l) wv[l] = t2[l];
            }
            waves[wv_i] = wv[0];
        }
        float total = waves[0];
        for (int i = 1; i < NW; ++i) total += waves[i];
        const float rs = 1.0f / std::sqrt(total / (float)n + eps);
        for (int i = 0; i < n; ++i) {
            const float gain = wf ? (*wf)[(size_t)i] : bf2f((*wb)[(size_t)i]);
            const float a = bf2f(nr[xb + (size_t)i]) * rs;
            const float b = gain + wadd;
            ob[xb + (size_t)i] = rad_f32_to_bf16(a * b);
        }
        for (int gi = 0; gi < n / group; ++gi) {
            float buf[256];
            for (int i = 0; i < group; ++i) buf[i] = bf2f(ob[xb + (size_t)gi * group + (size_t)i]);
            for (int b = 0; b < group; b += 128) fwht(buf + b, 128);
            s[(size_t)r * (n / group) + (size_t)gi] =
                quant_group(buf, group, &q[xb + (size_t)gi * group]);
        }
    }
}

/* ---- comparisons -------------------------------------------------------------------------- */

bool same_bytes(const void* a, const void* b, size_t bytes, const char* what) {
    if (std::memcmp(a, b, bytes) != 0) {
        std::fprintf(stderr, "  %s: bytes differ\n", what);
        return false;
    }
    return true;
}

/* ---- cases -------------------------------------------------------------------------------- */

void run_had(int M, int n, int group, uint32_t seed) {
    g_seed = seed;
    std::vector<uint16_t> hx((size_t)M * n);
    for (auto& e : hx) e = bfv(1.5f * rnd());
    std::vector<int8_t> hq((size_t)M * n, 0x55);
    std::vector<float> hs((size_t)M * (n / group), -1.0f);
    std::vector<int8_t> rq;
    std::vector<float> rs;
    ref_had(hx, M, n, group, rq, rs);

    Dev dx(hx.size() * 2), dq(hq.size()), ds(hs.size() * 4);
    up(dx, hx.data(), dx.bytes);
    up(dq, hq.data(), dq.bytes);
    up(ds, hs.data(), ds.bytes);

    std::vector<RadTensor> t = { ten(dx.p, RAD_BF16, { M, n }), ten(dq.p, RAD_I8, { M, n }),
                                 ten(ds.p, RAD_F32, { M, n / group }) };
    std::vector<RadParam> p = { pi("M", M), pi("n", n), pi("group", group), ps("dtype", "bf16") };
    const int rc = run(g_lib.had, t, p);
    CHECK_EQ(rc, RAD_OK);
    if (rc != RAD_OK) return;

    down(dq, hq.data(), dq.bytes);
    down(ds, hs.data(), ds.bytes);
    CHECK(same_bytes(hq.data(), rq.data(), hq.size(), "had_quant_act_i8g codes"));
    CHECK(same_bytes(hs.data(), rs.data(), hs.size() * 4, "had_quant_act_i8g scales"));
}

void run_gated(int M, int n, long long pitch, int group, bool sigmoid, bool with_y, uint32_t seed) {
    g_seed = seed;
    const long long gu_ld = pitch > 0 ? pitch : 2 * n;
    std::vector<uint16_t> hgu((size_t)M * (size_t)gu_ld, 0);
    for (int r = 0; r < M; ++r)
        for (int i = 0; i < 2 * n; ++i) hgu[(size_t)r * (size_t)gu_ld + (size_t)i] = bfv(1.5f * rnd());
    std::vector<int8_t> hq((size_t)M * n, 0x55);
    std::vector<float> hs((size_t)M * (n / group), -1.0f);
    std::vector<uint16_t> hy((size_t)M * n, 0);
    std::vector<int8_t> rq;
    std::vector<float> rs;
    std::vector<uint16_t> ry;
    ref_gated(hgu, M, n, gu_ld, group, sigmoid, with_y ? &ry : nullptr, rq, rs);

    Dev dgu(hgu.size() * 2), dq(hq.size()), ds(hs.size() * 4), dy(hy.size() * 2);
    up(dgu, hgu.data(), dgu.bytes);
    up(dq, hq.data(), dq.bytes);
    up(ds, hs.data(), ds.bytes);

    std::vector<RadTensor> t = {
        ten(dgu.p, RAD_BF16, { M, gu_ld }), ten(dq.p, RAD_I8, { M, n }),
        ten(ds.p, RAD_F32, { M, n / group }),
        with_y ? ten(dy.p, RAD_BF16, { M, n }) : RadTensor{},
    };
    std::vector<RadParam> p = { pi("M", M), pi("n", n), pi("group", group), ps("dtype", "bf16"),
                                ps("act", sigmoid ? "sigmoid" : "silu") };
    const int rc = run(g_lib.gated, t, p);
    CHECK_EQ(rc, RAD_OK);
    if (rc != RAD_OK) return;

    down(dq, hq.data(), dq.bytes);
    down(ds, hs.data(), ds.bytes);
    CHECK(same_bytes(hq.data(), rq.data(), hq.size(), "gated_had_quant_i8g codes"));
    CHECK(same_bytes(hs.data(), rs.data(), hs.size() * 4, "gated_had_quant_i8g scales"));
    if (with_y) {
        down(dy, hy.data(), dy.bytes);
        CHECK(same_bytes(hy.data(), ry.data(), hy.size() * 2, "gated_had_quant_i8g out_bf16"));
    }
}

void run_gate(int M, int n, long long gpitch, long long xpitch, int group, bool sigmoid,
              bool with_y, uint32_t seed) {
    g_seed = seed;
    const long long gate_ld = gpitch > 0 ? gpitch : n;
    const long long x_ld = xpitch > 0 ? xpitch : n;
    std::vector<uint16_t> hg((size_t)M * (size_t)gate_ld, 0), hx((size_t)M * (size_t)x_ld, 0);
    for (int r = 0; r < M; ++r)
        for (int i = 0; i < n; ++i) {
            hg[(size_t)r * (size_t)gate_ld + (size_t)i] = bfv(1.5f * rnd());
            hx[(size_t)r * (size_t)x_ld + (size_t)i] = bfv(1.5f * rnd());
        }
    std::vector<int8_t> hq((size_t)M * n, 0x55);
    std::vector<float> hs((size_t)M * (n / group), -1.0f);
    std::vector<uint16_t> hy((size_t)M * n, 0);
    std::vector<int8_t> rq;
    std::vector<float> rs;
    std::vector<uint16_t> ry;
    ref_gate(hg, hx, M, n, gate_ld, x_ld, group, sigmoid, with_y ? &ry : nullptr, rq, rs);

    Dev dg(hg.size() * 2), dx(hx.size() * 2), dq(hq.size()), ds(hs.size() * 4), dy(hy.size() * 2);
    up(dg, hg.data(), dg.bytes);
    up(dx, hx.data(), dx.bytes);
    up(dq, hq.data(), dq.bytes);
    up(ds, hs.data(), ds.bytes);

    std::vector<RadTensor> t = {
        ten(dg.p, RAD_BF16, { M, gate_ld }), ten(dx.p, RAD_BF16, { M, x_ld }),
        ten(dq.p, RAD_I8, { M, n }), ten(ds.p, RAD_F32, { M, n / group }),
        with_y ? ten(dy.p, RAD_BF16, { M, n }) : RadTensor{},
    };
    std::vector<RadParam> p = { pi("M", M), pi("n", n), pi("group", group), ps("dtype", "bf16"),
                                ps("act", sigmoid ? "sigmoid" : "silu") };
    const int rc = run(g_lib.gate, t, p);
    CHECK_EQ(rc, RAD_OK);
    if (rc != RAD_OK) return;

    down(dq, hq.data(), dq.bytes);
    down(ds, hs.data(), ds.bytes);
    CHECK(same_bytes(hq.data(), rq.data(), hq.size(), "gate_had_quant_i8g codes"));
    CHECK(same_bytes(hs.data(), rs.data(), hs.size() * 4, "gate_had_quant_i8g scales"));
    if (with_y) {
        down(dy, hy.data(), dy.bytes);
        CHECK(same_bytes(hy.data(), ry.data(), hy.size() * 2, "gate_had_quant_i8g out_bf16"));
    }
}

void run_rms(int M, int n, int group, bool with_res, bool gain_f32, float wadd, uint32_t seed,
             bool with_ob = true) {
    g_seed = seed;
    std::vector<uint16_t> hx((size_t)M * n), hres((size_t)M * n);
    std::vector<uint16_t> hwb((size_t)n);
    std::vector<float> hwf((size_t)n);
    for (auto& e : hx) e = bfv(1.5f * rnd());
    for (auto& e : hres) e = bfv(0.5f * rnd());
    for (int i = 0; i < n; ++i) {
        const float g = 0.5f + 0.5f * rnd();
        hwb[(size_t)i] = bfv(g);
        hwf[(size_t)i] = g + 0.125f * rnd();
    }
    std::vector<int8_t> hq((size_t)M * n, 0x55);
    std::vector<float> hs((size_t)M * (n / group), -1.0f);
    std::vector<uint16_t> hob((size_t)M * n, 0);
    std::vector<uint16_t> rnr, rob;
    std::vector<int8_t> rq;
    std::vector<float> rs;
    ref_rms(hx, with_res ? &hres : nullptr, gain_f32 ? nullptr : &hwb,
            gain_f32 ? &hwf : nullptr, M, n, group, 1e-5f, wadd, rnr, rob, rq, rs);

    Dev dx(hx.size() * 2), dres(hres.size() * 2), dw((gain_f32 ? hwf.size() * 4 : hwb.size() * 2));
    Dev dq(hq.size()), ds(hs.size() * 4), dob(hob.size() * 2);
    up(dx, hx.data(), dx.bytes);
    up(dres, hres.data(), dres.bytes);
    if (gain_f32) up(dw, hwf.data(), dw.bytes); else up(dw, hwb.data(), dw.bytes);
    up(dq, hq.data(), dq.bytes);
    up(ds, hs.data(), ds.bytes);

    std::vector<RadTensor> t = {
        ten(dx.p, RAD_BF16, { M, n }),
        with_res ? ten(dres.p, RAD_BF16, { M, n }) : RadTensor{},
        ten(dw.p, gain_f32 ? RAD_F32 : RAD_BF16, { n }),
        ten(dq.p, RAD_I8, { M, n }),
        ten(ds.p, RAD_F32, { M, n / group }),
        with_ob ? ten(dob.p, RAD_BF16, { M, n }) : RadTensor{},
    };
    std::vector<RadParam> p = { pi("M", M), pi("n", n), pf("eps", 1e-5), pi("group", group),
                                ps("dtype", "bf16"), pf("wadd", wadd) };
    const int rc = run(g_lib.rms, t, p);
    CHECK_EQ(rc, RAD_OK);
    if (rc != RAD_OK) return;

    down(dq, hq.data(), dq.bytes);
    down(ds, hs.data(), ds.bytes);
    if (with_ob) down(dob, hob.data(), dob.bytes);
    if (with_res) {
        std::vector<uint16_t> got_res(hres.size());
        down(dres, got_res.data(), dres.bytes);
        CHECK(same_bytes(got_res.data(), rnr.data(), got_res.size() * 2,
                         "rmsnorm_had_quant_i8g residual"));
    }
    CHECK(same_bytes(hq.data(), rq.data(), hq.size(), "rmsnorm_had_quant_i8g codes"));
    CHECK(same_bytes(hs.data(), rs.data(), hs.size() * 4, "rmsnorm_had_quant_i8g scales"));
    if (with_ob)
        CHECK(same_bytes(hob.data(), rob.data(), hob.size() * 2,
                         "rmsnorm_had_quant_i8g out_bf16"));
}

}  // namespace

/* ---- cases -------------------------------------------------------------------------------- */

TEST(had_quant_act_i8g_group_256) {
    if (!ready()) return;
    run_had(1, 256, 256, 0x1111u);
    run_had(5, 512, 256, 0x2222u);
    run_had(3, 1024, 256, 0x3333u);
    run_had(4, 512, 128, 0x4444u);   /* group 128, the non-regression case */
}

TEST(gated_had_quant_i8g_group_256) {
    if (!ready()) return;
    run_gated(1, 256, 0, 256, false, true, 0x5555u);
    run_gated(3, 512, 0, 256, true, true, 0x6666u);
    run_gated(4, 512, 0, 256, false, false, 0x7777u);   /* no out_bf16 */
    run_gated(2, 512, 2 * 512 + 64, 256, true, true, 0x8888u);   /* a wider gate_up pitch */
    run_gated(2, 512, 0, 128, false, true, 0x9999u);    /* group 128 */
}

TEST(gate_had_quant_i8g_group_256) {
    if (!ready()) return;
    run_gate(1, 256, 0, 0, 256, false, true, 0x5a5au);
    run_gate(3, 512, 0, 0, 256, true, true, 0x5b5bu);
    run_gate(4, 512, 0, 0, 256, false, false, 0x5c5cu);   /* no out_bf16 */
    run_gate(2, 512, 2 * 512 + 64, 512 + 32, 256, true, true, 0x5d5du);   /* distinct pitches */
    run_gate(2, 512, 0, 0, 128, false, true, 0x5e5eu);    /* group 128 */
}

TEST(rmsnorm_had_quant_i8g_group_256) {
    if (!ready()) return;
    run_rms(1, 256, 256, true, false, 0.0f, 0xaaaau);
    run_rms(3, 512, 256, true, false, 0.25f, 0xbbbbu);
    run_rms(2, 512, 256, false, true, -0.5f, 0xccccu);   /* no residual, f32 gain */
    run_rms(2, 512, 256, true, false, 0.0f, 0xbd00u, false);   /* no out_bf16 */
    run_rms(2, 512, 128, true, false, 0.0f, 0xddddu);    /* group 128 */
}

TEST(act_i8g_group_256_refuses_forms) {
    if (!ready()) return;

    /* A minimal valid set for each entry, so each mutation reaches the check it targets. */
    Dev x(16 * 2048 * 2), q(16 * 2048), sc(16 * 128 * 4), w(2048 * 2), ob(16 * 2048 * 2);
    std::vector<uint16_t> hx((size_t)16 * 2048, bfv(0.1f));
    std::vector<float> hw((size_t)2048, 1.0f);
    up(x, hx.data(), x.bytes);
    up(w, hw.data(), w.bytes);
    (void)hipMemset(q.p, 0, q.bytes);
    (void)hipMemset(sc.p, 0, sc.bytes);

    /* had_quant_act_i8g: a valid call, then group 192, n % group, an f32 input, and no scale. */
    {
        std::vector<RadTensor> t = { ten(x.p, RAD_BF16, { 16, 1024 }), ten(q.p, RAD_I8, { 16, 1024 }),
                                     ten(sc.p, RAD_F32, { 16, 4 }) };
        std::vector<RadParam> p = { pi("M", 16), pi("n", 1024), pi("group", 256),
                                    ps("dtype", "bf16") };
        CHECK_EQ(run(g_lib.had, t, p), RAD_OK);
        for (auto& e : p) if (!std::strcmp(e.key, "group")) e.ival = 192;
        CHECK_EQ(run(g_lib.had, t, p), RAD_E_SHAPE);
        {
            std::vector<RadTensor> t2 = { ten(x.p, RAD_BF16, { 16, 384 }),
                                          ten(q.p, RAD_I8, { 16, 384 }),
                                          ten(sc.p, RAD_F32, { 16, 1 }) };
            std::vector<RadParam> p2 = { pi("M", 16), pi("n", 384), pi("group", 256),
                                         ps("dtype", "bf16") };
            CHECK_EQ(run(g_lib.had, t2, p2), RAD_E_SHAPE);   /* n % group */
        }
        for (auto& e : p) if (!std::strcmp(e.key, "group")) e.ival = 256;
        t[0].dtype = RAD_F32;
        CHECK_EQ(run(g_lib.had, t, p), RAD_E_DTYPE);
        t[0].dtype = RAD_BF16;
        t[2] = RadTensor{};
        CHECK_EQ(run(g_lib.had, t, p), RAD_E_INVAL);
    }

    /* gated_had_quant_i8g: group 192, n % group, an f32 gate_up. */
    {
        std::vector<RadTensor> t = { ten(x.p, RAD_BF16, { 16, 2048 }), ten(q.p, RAD_I8, { 16, 1024 }),
                                     ten(sc.p, RAD_F32, { 16, 4 }), ten(ob.p, RAD_BF16, { 16, 1024 }) };
        std::vector<RadParam> p = { pi("M", 16), pi("n", 1024), pi("group", 256),
                                    ps("dtype", "bf16"), ps("act", "silu") };
        CHECK_EQ(run(g_lib.gated, t, p), RAD_OK);
        for (auto& e : p) if (!std::strcmp(e.key, "group")) e.ival = 192;
        CHECK_EQ(run(g_lib.gated, t, p), RAD_E_SHAPE);
        {
            std::vector<RadTensor> t2 = { ten(x.p, RAD_BF16, { 16, 768 }),
                                          ten(q.p, RAD_I8, { 16, 384 }),
                                          ten(sc.p, RAD_F32, { 16, 1 }),
                                          ten(ob.p, RAD_BF16, { 16, 384 }) };
            std::vector<RadParam> p2 = { pi("M", 16), pi("n", 384), pi("group", 256),
                                         ps("dtype", "bf16"), ps("act", "silu") };
            CHECK_EQ(run(g_lib.gated, t2, p2), RAD_E_SHAPE);   /* n % group */
        }
        for (auto& e : p) if (!std::strcmp(e.key, "group")) e.ival = 256;
        t[0].dtype = RAD_F32;
        CHECK_EQ(run(g_lib.gated, t, p), RAD_E_DTYPE);
    }

    /* gate_had_quant_i8g: group 192, n % group, an f32 gate, a missing scale. */
    {
        std::vector<RadTensor> t = { ten(x.p, RAD_BF16, { 16, 1024 }),
                                     ten(x.p, RAD_BF16, { 16, 1024 }), ten(q.p, RAD_I8, { 16, 1024 }),
                                     ten(sc.p, RAD_F32, { 16, 4 }) };
        std::vector<RadParam> p = { pi("M", 16), pi("n", 1024), pi("group", 256),
                                    ps("dtype", "bf16"), ps("act", "sigmoid") };
        CHECK_EQ(run(g_lib.gate, t, p), RAD_OK);
        for (auto& e : p) if (!std::strcmp(e.key, "group")) e.ival = 192;
        CHECK_EQ(run(g_lib.gate, t, p), RAD_E_SHAPE);
        {
            std::vector<RadTensor> t2 = { ten(x.p, RAD_BF16, { 16, 384 }),
                                          ten(x.p, RAD_BF16, { 16, 384 }),
                                          ten(q.p, RAD_I8, { 16, 384 }),
                                          ten(sc.p, RAD_F32, { 16, 1 }) };
            std::vector<RadParam> p2 = { pi("M", 16), pi("n", 384), pi("group", 256),
                                         ps("dtype", "bf16"), ps("act", "silu") };
            CHECK_EQ(run(g_lib.gate, t2, p2), RAD_E_SHAPE);   /* n % group */
        }
        for (auto& e : p) if (!std::strcmp(e.key, "group")) e.ival = 256;
        t[0].dtype = RAD_F32;
        CHECK_EQ(run(g_lib.gate, t, p), RAD_E_DTYPE);
        t[0].dtype = RAD_BF16;
        t[3] = RadTensor{};
        CHECK_EQ(run(g_lib.gate, t, p), RAD_E_INVAL);
    }

    /* rmsnorm_had_quant_i8g: group 192, n % group, and a missing gain. */
    {
        std::vector<RadTensor> t = { ten(x.p, RAD_BF16, { 16, 1024 }),
                                     RadTensor{}, ten(w.p, RAD_BF16, { 1024 }),
                                     ten(q.p, RAD_I8, { 16, 1024 }),
                                     ten(sc.p, RAD_F32, { 16, 4 }),
                                     ten(ob.p, RAD_BF16, { 16, 1024 }) };
        std::vector<RadParam> p = { pi("M", 16), pi("n", 1024), pf("eps", 1e-5), pi("group", 256),
                                    ps("dtype", "bf16") };
        CHECK_EQ(run(g_lib.rms, t, p), RAD_OK);
        for (auto& e : p) if (!std::strcmp(e.key, "group")) e.ival = 192;
        CHECK_EQ(run(g_lib.rms, t, p), RAD_E_UNSUPPORTED);
        {
            std::vector<RadTensor> t2 = { ten(x.p, RAD_BF16, { 16, 384 }),
                                          RadTensor{}, ten(w.p, RAD_BF16, { 384 }),
                                          ten(q.p, RAD_I8, { 16, 384 }),
                                          ten(sc.p, RAD_F32, { 16, 1 }),
                                          ten(ob.p, RAD_BF16, { 16, 384 }) };
            std::vector<RadParam> p2 = { pi("M", 16), pi("n", 384), pf("eps", 1e-5),
                                         pi("group", 256), ps("dtype", "bf16") };
            CHECK_EQ(run(g_lib.rms, t2, p2), RAD_E_UNSUPPORTED);   /* n % group */
        }
        for (auto& e : p) if (!std::strcmp(e.key, "group")) e.ival = 256;
        t[2] = RadTensor{};
        CHECK_EQ(run(g_lib.rms, t, p), RAD_E_INVAL);
    }
}

RAD_TEST_MAIN()
