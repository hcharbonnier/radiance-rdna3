/* vision_rows_device_test.cpp -- libr3d's gemm_nt_bias and grid_embed rows against libref, on
 * the card.
 *
 * These are the two ops the 27B's vision tower (arch/common/rad_block_vit.h) issues and W10-lite
 * left on the host:
 *
 *   gemm_nt_bias  every linear of the tower: patch embedding, qkv, the attention projection and
 *                 both MLP layers, with a bias and, on the projection and down layers, the
 *                 residual stream in the epilogue. Cases cover the tower's geometries (hidden
 *                 width, 3x qkv, the ffn, the merger and the patch projection), a variable M
 *                 including tails, both activations, the residual present and absent, zero and
 *                 non-zero biases, row pitches and unaligned bases, the finite bf16 domain of the
 *                 tanh GELU exhaustively, and both tile heights around their dispatch boundary.
 *
 *   grid_embed    the learned position table added before the first block. Cases cover the
 *                 48 x 48 table of the 27B resampled to patch grids (including 1-wide and
 *                 clamped coordinates), several M, row pitches on x and the table, and a
 *                 coordinate plane at its own strides.
 *
 * Every output is compared bit for bit: both rows are transcriptions of libref's arithmetic, and
 * the comparison is the contract. The libraries are dlopened and their rows found through
 * rad_kernel_at, as the engine loads them; on a machine without the card or the plugins every
 * case skips.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
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
RadParam ps(const char* k, const char* v) { return RAD_STR(k, v); }

uint16_t bfv(float v) { return rad_f32_to_bf16(v); }

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

/* A host [rows, n] bf16 plane with a row pitch and a leading element offset. */
struct HBuf {
    std::vector<uint16_t> v;
    int off = 0;
    uint16_t* p() { return v.data() + off; }
    const uint16_t* p() const { return v.data() + off; }
};

struct Gen {
    std::mt19937 rng;
    explicit Gen(uint32_t seed) : rng(seed) {}
    float val(float scale = 1.0f) {
        return ((float)(rng() % 2001) - 1000.0f) / 1000.0f * scale;
    }
    uint16_t bf(float scale = 1.0f) { return bfv(val(scale)); }
};

bool same_bf16(const uint16_t* a, int a_ld, const uint16_t* b, int b_ld, int rows, int cols,
               const char* what) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const uint16_t x = a[(size_t)r * a_ld + c], y = b[(size_t)r * b_ld + c];
            if (x != y) {
                std::fprintf(stderr, "%s differs at (%d,%d): %04x vs %04x\n", what, r, c, x, y);
                return false;
            }
        }
    return true;
}

/* ---- gemm_nt_bias ------------------------------------------------------------------------- */

struct GemmCase {
    int M = 0, N = 0, K = 0;
    int a_ld = 0, b_ld = 0, res_ld = 0, y_ld = 0;   /* 0 = dense */
    int a_off = 0, b_off = 0, bias_off = 0, res_off = 0, y_off = 0;
    bool res = false;
    bool res_alias = false;                         /* res is y itself, the stream in place */
    bool zero_bias = false;
    bool exhaustive_gelu_domain = false;            /* A holds every bf16 code once, B = 1 */
    const char* act = nullptr;                      /* null: the optional parameter is omitted */
    uint32_t seed = 1;
};

/* Fills a plane's logical entries from `g`; the pitch padding and the leading offset stay at their
 * sentinel, so a kernel that reads outside the row shows up as a mismatch. */
void fill_plane(HBuf* h, int rows, int cols, int pitch, int off, Gen* g, bool zero = false) {
    const size_t bytes = (size_t)(off + rows * pitch + 16);
    h->v.assign(bytes, 0x5a5a);
    h->off = off;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            h->p()[(size_t)r * pitch + c] = zero ? 0u : g->bf();
}

bool run_gemm(const GemmCase& c, const char* what) {
    const int a_ld = c.a_ld ? c.a_ld : c.K;
    const int b_ld = c.b_ld ? c.b_ld : c.K;
    const int y_ld = c.y_ld ? c.y_ld : c.N;
    const int res_ld = c.res_ld ? c.res_ld : c.N;
    const int r_ld = c.res_alias ? y_ld : res_ld;

    const RadKernelInfo* kd = g_r3d.find("gemm_nt_bias", RAD_DOMAIN_DEVICE, "gemm_nt_bias");
    const RadKernelInfo* kr = g_ref.find("gemm_nt_bias", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return false;

    Gen g(c.seed);
    HBuf A, B, bias, res, y0;
    fill_plane(&A, c.M, c.K, a_ld, c.a_off, &g);
    fill_plane(&B, c.N, c.K, b_ld, c.b_off, &g);
    fill_plane(&bias, 1, c.N, c.N, c.bias_off, &g, c.zero_bias);
    if (c.exhaustive_gelu_domain) {
        /* K = N = 1 and B = 1: every output is one bf16 input, so A carries each finite bf16 code
         * once and the row's gelu_tanh epilogue sees its whole finite domain. The non-finite codes
         * map to +0: their NaN payloads are not part of the contract and would depend on where the
         * host and the device round them. */
        for (int r = 0; r < c.M; ++r) {
            const uint16_t code = (uint16_t)r;
            A.p()[(size_t)r * a_ld + 0] = ((code & 0x7f80u) == 0x7f80u) ? 0 : code;
        }
        B.p()[0] = 0x3f80;                    /* 1.0f */
        bias.p()[0] = 0;
    }
    if (c.res) fill_plane(&res, c.M, c.N, res_ld, c.res_off, &g);
    fill_plane(&y0, c.M, c.N, y_ld, c.y_off, &g);
    if (c.res_alias) res = y0;
    HBuf y_ref = y0, y_dev = y0;

    auto params = [&]() {
        std::vector<RadParam> p{ pi("M", c.M), pi("N", c.N), pi("K", c.K), ps("dtype", "bf16") };
        if (c.act) p.push_back(ps("act", c.act));
        return p;
    };

    /* libref, on the host, writes y_ref. */
    {
        std::vector<RadTensor> t{
            ten(A.p(), RAD_BF16, { c.M, c.K }), ten(B.p(), RAD_BF16, { c.N, c.K }),
            ten(bias.p(), RAD_BF16, { c.N }),
            c.res ? ten(c.res_alias ? y_ref.p() : res.p(), RAD_BF16, { c.M, c.N }) : RadTensor{},
            ten(y_ref.p(), RAD_BF16, { c.M, c.N })
        };
        t[0].stride[0] = a_ld;
        t[1].stride[0] = b_ld;
        if (c.res) t[3].stride[0] = r_ld;
        t[4].stride[0] = y_ld;
        std::vector<RadParam> p = params();
        RadArgs a;
        std::memset(&a, 0, sizeof a);
        a.t = t.data();
        a.n_t = (int)t.size();
        a.p = p.data();
        a.n_p = (int)p.size();
        const int rc = kr->launch(&a, (RadStream)0);
        if (rc != RAD_OK) {
            std::fprintf(stderr, "%s: reference returned %d\n", what, rc);
            return false;
        }
    }

    /* The device row writes y_dev over the same inputs. */
    Dev da(A.v.size() * 2), db(B.v.size() * 2), dbias(bias.v.size() * 2);
    Dev dres(c.res ? res.v.size() * 2 : 16), dy(y_dev.v.size() * 2);
    up(da, A.v.data(), da.bytes);
    up(db, B.v.data(), db.bytes);
    up(dbias, bias.v.data(), dbias.bytes);
    if (c.res && !c.res_alias) up(dres, res.v.data(), dres.bytes);
    up(dy, y_dev.v.data(), dy.bytes);

    {
        std::vector<RadTensor> t{
            ten((uint8_t*)da.p + (size_t)c.a_off * 2, RAD_BF16, { c.M, c.K }),
            ten((uint8_t*)db.p + (size_t)c.b_off * 2, RAD_BF16, { c.N, c.K }),
            ten((uint8_t*)dbias.p + (size_t)c.bias_off * 2, RAD_BF16, { c.N }),
            c.res ? ten(c.res_alias ? (uint8_t*)dy.p + (size_t)c.y_off * 2
                                     : (uint8_t*)dres.p + (size_t)c.res_off * 2,
                        RAD_BF16, { c.M, c.N })
                  : RadTensor{},
            ten((uint8_t*)dy.p + (size_t)c.y_off * 2, RAD_BF16, { c.M, c.N })
        };
        t[0].stride[0] = a_ld;
        t[1].stride[0] = b_ld;
        if (c.res) t[3].stride[0] = r_ld;
        t[4].stride[0] = y_ld;
        std::vector<RadParam> p = params();
        const int rc = run(kd, t, p);
        if (rc != RAD_OK) {
            std::fprintf(stderr, "%s: device returned %d\n", what, rc);
            return false;
        }
    }

    std::vector<uint16_t> out(y_dev.v.size());
    down(dy, out.data(), out.size() * 2);
    y_dev.v = out;

    const bool ok = same_bf16(y_ref.p(), y_ld, y_dev.p(), y_ld, c.M, c.N, what);
    if (!ok) std::fprintf(stderr, "%s failed\n", what);
    return ok;
}

/* Device timing for the row, no oracle: warm up, then the median and the minimum of fifteen
 * individually timed launches. Used with RAD_VISION_BENCH=1 on the tower's served shapes. */
bool bench_gemm(const char* name, int M, int N, int K, const char* act, bool res) {
    const RadKernelInfo* kd = g_r3d.find("gemm_nt_bias", RAD_DOMAIN_DEVICE, "gemm_nt_bias");
    if (!kd) return true;

    Gen g(9001u + (uint32_t)M + (uint32_t)N + (uint32_t)K);
    HBuf A, B, bias, rb, y;
    fill_plane(&A, M, K, K, 0, &g);
    fill_plane(&B, N, K, K, 0, &g);
    fill_plane(&bias, 1, N, N, 0, &g);
    if (res) fill_plane(&rb, M, N, N, 0, &g);
    fill_plane(&y, M, N, N, 0, &g);

    Dev da(A.v.size() * 2), db(B.v.size() * 2), dbias(bias.v.size() * 2);
    Dev dres(res ? rb.v.size() * 2 : 16), dy(y.v.size() * 2);
    up(da, A.v.data(), da.bytes);
    up(db, B.v.data(), db.bytes);
    up(dbias, bias.v.data(), dbias.bytes);
    if (res) up(dres, rb.v.data(), dres.bytes);
    up(dy, y.v.data(), dy.bytes);

    std::vector<RadTensor> t{
        ten(da.p, RAD_BF16, { M, K }), ten(db.p, RAD_BF16, { N, K }),
        ten(dbias.p, RAD_BF16, { N }),
        res ? ten(dres.p, RAD_BF16, { M, N }) : RadTensor{},
        ten(dy.p, RAD_BF16, { M, N })
    };
    std::vector<RadParam> p{ pi("M", M), pi("N", N), pi("K", K), ps("dtype", "bf16") };
    if (act) p.push_back(ps("act", act));
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();

    for (int i = 0; i < 3; ++i) kd->launch(&a, (RadStream)0);
    hipDeviceSynchronize();

    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    std::vector<float> us(15);
    for (int i = 0; i < 15; ++i) {
        hipEventRecord(e0, (hipStream_t)0);
        kd->launch(&a, (RadStream)0);
        hipEventRecord(e1, (hipStream_t)0);
        hipEventSynchronize(e1);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, e0, e1);
        us[(size_t)i] = ms * 1000.0f;
    }
    hipEventDestroy(e0);
    hipEventDestroy(e1);

    std::sort(us.begin(), us.end());
    const float med = us[us.size() / 2], best = us[0];
    const double flops = 2.0 * (double)M * (double)N * (double)K;
    std::printf("GEMMBENCH %-26s M=%4d N=%4d K=%4d act=%-4s res=%d  med=%9.1f us "
                "min=%9.1f us  %6.3f / %6.3f TFLOP/s\n",
                name, M, N, K, act ? act : "none", res ? 1 : 0, med, best,
                flops / ((double)med * 1e-6) / 1e12, flops / ((double)best * 1e-6) / 1e12);
    return true;
}

/* The tower's geometries. They are the 27B's shape family: width = 16 heads x 72, ff 4304,
 * patch 16 with temporal 2 and 3 channels (patch_dim 1536) and merge^2 = 4. */
TEST(gemm_nt_bias_vision_shapes_match_libref) {
    if (!ready()) return;
    GemmCase c;
    c.M = 1; c.N = 1152; c.K = 1152; c.seed = 11;        /* decode row: qkv/proj width */
    CHECK(run_gemm(c, "m1 qkv"));

    c.M = 7; c.N = 3456; c.K = 1152; c.res = true; c.seed = 12;   /* qkv + residual projection */
    CHECK(run_gemm(c, "m7 qkv res"));

    c.M = 33; c.N = 4304; c.K = 1152; c.act = "gelu_tanh"; c.seed = 13;   /* fc1 */
    CHECK(run_gemm(c, "m33 fc1"));

    c.M = 64; c.N = 1152; c.K = 4304; c.res = true; c.act = "none"; c.seed = 14;  /* fc2 + res */
    CHECK(run_gemm(c, "m64 fc2 res"));

    c.M = 129; c.N = 1152; c.K = 1536; c.act = "none"; c.seed = 15;      /* patch embedding */
    CHECK(run_gemm(c, "m129 patch"));

    c.M = 17; c.N = 1152; c.K = 4608; c.act = "gelu"; c.seed = 16;       /* merger fc1 */
    CHECK(run_gemm(c, "m17 merger1"));

    c.M = 17; c.N = 1152; c.K = 4608; c.res = true; c.act = "none"; c.seed = 17;
    CHECK(run_gemm(c, "m17 merger2 res"));

    /* The tower's projection and down layers add the residual into the stream itself. */
    c.M = 33; c.N = 1152; c.K = 1152; c.res = true; c.res_alias = true; c.act = "none";
    c.seed = 18;
    CHECK(run_gemm(c, "m33 residual in place"));
}

/* M is the pass's patch count: it moves with the image, and the last row/column tiles are
 * partial at most heights. */
TEST(gemm_nt_bias_m_tails_match_libref) {
    if (!ready()) return;
    for (int M : { 2, 3, 15, 16, 31, 32, 47, 48, 49, 63, 65 }) {
        GemmCase c;
        c.M = M; c.N = 67; c.K = 71; c.seed = 100u + (uint32_t)M;
        c.act = (M & 1) ? "gelu_tanh" : nullptr;
        c.res = (M % 3) == 0;
        CHECK(run_gemm(c, "tail"));
    }
}

/* The tower hands a column slice of a wider activation and a residual stream at a wider pitch; a
 * weight may be a jumped plane too. The leading offsets put the operands two bytes off a 16-byte
 * boundary, which the halfword reads must serve. */
TEST(gemm_nt_bias_pitches_and_alignments_match_libref) {
    if (!ready()) return;
    GemmCase c;
    c.M = 19; c.N = 69; c.K = 83;
    c.a_ld = c.K + 5; c.b_ld = c.K + 1; c.res_ld = c.N + 7; c.y_ld = c.N + 3;
    c.a_off = 1; c.b_off = 3; c.bias_off = 1; c.res_off = 5; c.y_off = 1;
    c.res = true; c.act = "gelu_tanh"; c.seed = 21;
    CHECK(run_gemm(c, "pitched"));

    c.act = nullptr; c.res = false; c.a_off = 0; c.y_off = 0; c.seed = 22;
    CHECK(run_gemm(c, "pitched no epilogue"));
}

/* A zero bias is the same chain as gemm_nt plus 0; a non-zero one is added in f32 before the one
 * rounding. */
TEST(gemm_nt_bias_zero_and_nonzero_bias_match_libref) {
    if (!ready()) return;
    GemmCase c;
    c.M = 5; c.N = 40; c.K = 64; c.seed = 31;
    CHECK(run_gemm(c, "bias"));
    c.zero_bias = true; c.seed = 32;
    CHECK(run_gemm(c, "zero bias"));
}

/* The tanh GELU's input is always a bf16 value (the biased sum is narrowed before the epilogue),
 * so its reachable domain is the finite bf16 codes. The row evaluates them with tanhf; this case
 * runs the whole finite domain through the row against libref, which uses tanh in double, and
 * compares the stored bytes -- including the negative tail where 1+th is only a few f32 ulps. */
TEST(gemm_nt_bias_gelu_tanh_covers_every_finite_bf16_input) {
    if (!ready()) return;
    GemmCase c;
    c.M = 65536; c.N = 1; c.K = 1;
    c.act = "gelu_tanh"; c.exhaustive_gelu_domain = true; c.seed = 41;
    CHECK(run_gemm(c, "gelu_tanh all finite bf16"));
}

/* The tall (64-row) tile takes over at kTallMin; a pass just above it has a one-row M tail, and a
 * row/column/k tail on top of that. The two rows around the dispatch boundary are checked so the
 * switch itself is covered, each against libref. */
TEST(gemm_nt_bias_tall_tile_tails_match_libref) {
    if (!ready()) return;
    GemmCase c;
    c.M = 1025; c.N = 67; c.K = 71; c.act = "gelu_tanh"; c.seed = 42;
    CHECK(run_gemm(c, "tall tile tails"));
    c.act = nullptr; c.res = true; c.seed = 43;
    CHECK(run_gemm(c, "tall tile tails res"));
    c.res = false; c.act = "gelu_tanh"; c.seed = 44;
    c.M = 1023;
    CHECK(run_gemm(c, "short tile at the boundary"));
    c.M = 1024; c.seed = 45;
    CHECK(run_gemm(c, "tall tile at the boundary"));
}

/* The served tower's whole-image pass, timed with no oracle (RAD_VISION_BENCH=1). The shapes are
 * the ones the 2048^2 pass issues: the patch projection, the attention projection at the patch
 * count of the 27B's 2048^2 image, and the deepstack merger's GELU layer. */
TEST(vision_gemm_bench) {
    if (!std::getenv("RAD_VISION_BENCH")) return;
    if (!ready()) return;
    CHECK(bench_gemm("patch embed", 512, 1152, 1536, nullptr, false));
    CHECK(bench_gemm("attn proj + res", 1045, 1152, 1152, nullptr, true));
    CHECK(bench_gemm("merger fc gelu", 1024, 4608, 4608, "gelu", false));
}

/* ---- grid_embed --------------------------------------------------------------------------- */

struct GridCase {
    int M = 0, n = 0, side = 0;
    int x_ld = 0, t_ld = 0, ccs = 0;
    int x_off = 0, t_off = 0, c_off = 0;
    int grid_h = 0, grid_w = 0;      /* the patch grid each row's coordinate names */
    bool clamp_rows = false;         /* put a few rows past the grid's edge */
    bool single_axis = false;        /* one row names a 1 x 1 grid */
    uint32_t seed = 1;
};

bool run_grid(const GridCase& c, const char* what) {
    const int x_ld = c.x_ld ? c.x_ld : c.n;
    const int t_ld = c.t_ld ? c.t_ld : c.n;
    const int ccs = c.ccs ? c.ccs : c.M + 3;
    const int gh = c.grid_h ? c.grid_h : 1, gw = c.grid_w ? c.grid_w : 1;

    const RadKernelInfo* kd = g_r3d.find("grid_embed", RAD_DOMAIN_DEVICE, "grid_embed");
    const RadKernelInfo* kr = g_ref.find("grid_embed", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return false;

    Gen g(c.seed);
    HBuf x, tab;
    std::vector<int32_t> crd;
    fill_plane(&x, c.M, c.n, x_ld, c.x_off, &g);
    fill_plane(&tab, c.side * c.side, c.n, t_ld, c.t_off, &g);

    /* The coordinate plane: component-major [4, M] at `ccs` elements a component and rows one
     * apart, so a view of a wider plane is read in place. It is i32, so its buffer is too. */
    crd.assign((size_t)(c.c_off + 3 * ccs + c.M + 8), 0);
    for (int r = 0; r < c.M; ++r) {
        int gr = r % gh, gc = (r / gh) % gw;
        if (c.single_axis) { gr = 0; gc = 0; }
        if (c.clamp_rows && (r % 7) == 6) { gr = gh + 2; gc = gw + 3; }
        int32_t* row = crd.data() + c.c_off;
        row[0 * ccs + r] = gr;
        row[1 * ccs + r] = gc;
        row[2 * ccs + r] = gh;
        row[3 * ccs + r] = gw;
    }

    HBuf x_ref = x, x_dev = x;
    auto params = [&]() {
        return std::vector<RadParam>{ pi("M", c.M), pi("n", c.n), pi("side", c.side),
                                      ps("dtype", "bf16") };
    };

    /* libref on the host. */
    {
        std::vector<RadTensor> t{
            ten(x_ref.p(), RAD_BF16, { c.M, c.n }),
            ten(tab.p(), RAD_BF16, { c.side * c.side, c.n }),
            ten(crd.data() + c.c_off, RAD_I32, { 4, c.M })
        };
        t[0].stride[0] = x_ld;
        t[1].stride[0] = t_ld;
        t[2].stride[0] = ccs;
        t[2].stride[1] = 1;
        std::vector<RadParam> p = params();
        RadArgs a;
        std::memset(&a, 0, sizeof a);
        a.t = t.data();
        a.n_t = (int)t.size();
        a.p = p.data();
        a.n_p = (int)p.size();
        const int rc = kr->launch(&a, (RadStream)0);
        if (rc != RAD_OK) {
            std::fprintf(stderr, "%s: reference returned %d\n", what, rc);
            return false;
        }
    }

    Dev dx(x_dev.v.size() * 2), dtab(tab.v.size() * 2), dcrd(crd.size() * 4);
    up(dx, x_dev.v.data(), dx.bytes);
    up(dtab, tab.v.data(), dtab.bytes);
    up(dcrd, crd.data(), dcrd.bytes);
    {
        std::vector<RadTensor> t{
            ten((uint8_t*)dx.p + (size_t)c.x_off * 2, RAD_BF16, { c.M, c.n }),
            ten((uint8_t*)dtab.p + (size_t)c.t_off * 2, RAD_BF16, { c.side * c.side, c.n }),
            ten((uint8_t*)dcrd.p + (size_t)c.c_off * 4, RAD_I32, { 4, c.M })
        };
        t[0].stride[0] = x_ld;
        t[1].stride[0] = t_ld;
        t[2].stride[0] = ccs;
        t[2].stride[1] = 1;
        std::vector<RadParam> p = params();
        const int rc = run(kd, t, p);
        if (rc != RAD_OK) {
            std::fprintf(stderr, "%s: device returned %d\n", what, rc);
            return false;
        }
    }

    std::vector<uint16_t> out(x_dev.v.size());
    down(dx, out.data(), out.size() * 2);
    x_dev.v = out;

    const bool ok = same_bf16(x_ref.p(), x_ld, x_dev.p(), x_ld, c.M, c.n, what);
    if (!ok) std::fprintf(stderr, "%s failed\n", what);
    return ok;
}

/* The 27B's 48 x 48 table onto a 24 x 48 patch grid (1152 patches): the tower's real pass. */
TEST(grid_embed_vision_table_matches_libref) {
    if (!ready()) return;
    GridCase c;
    c.M = 1152; c.n = 1152; c.side = 48; c.grid_h = 24; c.grid_w = 48; c.seed = 41;
    CHECK(run_grid(c, "48x48 1152x1152"));

    c.M = 1024; c.n = 1152; c.side = 48; c.grid_h = 32; c.grid_w = 32; c.seed = 42;
    CHECK(run_grid(c, "48x48 1024x1152"));
}

/* Small grids, one-row grids and coordinates past the edge (the taps clamp, as libref's do). */
TEST(grid_embed_grid_tails_and_clamped_taps_match_libref) {
    if (!ready()) return;
    GridCase c;
    c.M = 7; c.n = 37; c.side = 8; c.grid_h = 1; c.grid_w = 7;
    c.clamp_rows = true; c.seed = 51;
    CHECK(run_grid(c, "1x7 clamped"));

    c.M = 11; c.n = 5; c.side = 5; c.grid_h = 3; c.grid_w = 2; c.single_axis = true;
    c.clamp_rows = false; c.seed = 52;
    CHECK(run_grid(c, "3x2 with 1x1 row"));

    c.M = 1; c.n = 3; c.side = 2; c.grid_h = 1; c.grid_w = 1; c.seed = 53;
    CHECK(run_grid(c, "single row"));
}

/* x and the table at wider row pitches, the coordinate plane at its own component stride, and
 * the bases two to six bytes off a 16-byte boundary. */
TEST(grid_embed_pitches_and_alignments_match_libref) {
    if (!ready()) return;
    GridCase c;
    c.M = 23; c.n = 41; c.side = 7; c.grid_h = 5; c.grid_w = 6;
    c.x_ld = c.n + 9; c.t_ld = c.n + 3; c.ccs = c.M + 11;
    c.x_off = 1; c.t_off = 3; c.c_off = 1;
    c.clamp_rows = true; c.seed = 61;
    CHECK(run_grid(c, "pitched"));
}

/* ---- refusals ----------------------------------------------------------------------------- */

TEST(vision_rows_refuse_unsupported_calls) {
    if (!ready()) return;
    const RadKernelInfo* kg = g_r3d.find("gemm_nt_bias", RAD_DOMAIN_DEVICE, "gemm_nt_bias");
    const RadKernelInfo* kx = g_r3d.find("grid_embed", RAD_DOMAIN_DEVICE, "grid_embed");
    CHECK(kg && kx);
    if (!(kg && kx)) return;

    /* gemm_nt_bias without a bias: the schema requires it, and the shim must refuse rather than
     * read a null plane. */
    {
        Gen g(71);
        HBuf A, B, bias, y;
        fill_plane(&A, 2, 4, 4, 0, &g);
        fill_plane(&B, 3, 4, 4, 0, &g);
        fill_plane(&bias, 1, 3, 3, 0, &g);
        fill_plane(&y, 2, 3, 3, 0, &g);
        Dev da(A.v.size() * 2), db(B.v.size() * 2), dy(y.v.size() * 2);
        up(da, A.v.data(), da.bytes); up(db, B.v.data(), db.bytes); up(dy, y.v.data(), dy.bytes);
        std::vector<RadTensor> t{
            ten(da.p, RAD_BF16, { 2, 4 }), ten(db.p, RAD_BF16, { 3, 4 }), RadTensor{},
            RadTensor{}, ten(dy.p, RAD_BF16, { 2, 3 })
        };
        std::vector<RadParam> p{ pi("M", 2), pi("N", 3), pi("K", 4), ps("dtype", "bf16") };
        CHECK_EQ(run(kg, t, p), RAD_E_INVAL);
    }
    /* An unknown activation is refused, not silently dropped. */
    {
        Gen g(72);
        HBuf A, B, bias, y;
        fill_plane(&A, 2, 4, 4, 0, &g);
        fill_plane(&B, 3, 4, 4, 0, &g);
        fill_plane(&bias, 1, 3, 3, 0, &g);
        fill_plane(&y, 2, 3, 3, 0, &g);
        Dev da(A.v.size() * 2), db(B.v.size() * 2), dbias(bias.v.size() * 2), dy(y.v.size() * 2);
        up(da, A.v.data(), da.bytes); up(db, B.v.data(), db.bytes);
        up(dbias, bias.v.data(), dbias.bytes); up(dy, y.v.data(), dy.bytes);
        std::vector<RadTensor> t{
            ten(da.p, RAD_BF16, { 2, 4 }), ten(db.p, RAD_BF16, { 3, 4 }),
            ten(dbias.p, RAD_BF16, { 3 }), RadTensor{}, ten(dy.p, RAD_BF16, { 2, 3 })
        };
        std::vector<RadParam> p{ pi("M", 2), pi("N", 3), pi("K", 4), ps("dtype", "bf16"),
                                 ps("act", "gelu_quick") };
        CHECK_EQ(run(kg, t, p), RAD_E_UNSUPPORTED);
        /* A bias whose last axis is not one element apart would be read as a different plane. */
        t[2] = ten(dbias.p, RAD_BF16, { 3 });
        t[2].stride[0] = 2;
        CHECK_EQ(run(kg, t, p), RAD_E_STRIDE);
    }

    /* grid_embed with a table too short for its side, a coordinate plane of the wrong rank, and
     * a non-bf16 x. */
    {
        Gen g(73);
        HBuf x, tab;
        std::vector<int32_t> crd(4 * 3, 0);
        fill_plane(&x, 3, 4, 4, 0, &g);
        fill_plane(&tab, 4, 4, 4, 0, &g);      /* 2 x 2 table, one row short for side 3 */
        Dev dx(x.v.size() * 2), dtab(tab.v.size() * 2), dcrd(crd.size() * 4);
        up(dx, x.v.data(), dx.bytes); up(dtab, tab.v.data(), dtab.bytes);
        up(dcrd, crd.data(), dcrd.bytes);
        std::vector<RadTensor> t{
            ten(dx.p, RAD_BF16, { 3, 4 }), ten(dtab.p, RAD_BF16, { 4, 4 }),
            ten(dcrd.p, RAD_I32, { 4, 3 })
        };
        std::vector<RadParam> p{ pi("M", 3), pi("n", 4), pi("side", 3), ps("dtype", "bf16") };
        CHECK_EQ(run(kx, t, p), RAD_E_SHAPE);

        t[2] = ten(dcrd.p, RAD_I32, { 12 });
        CHECK_EQ(run(kx, t, p), RAD_E_SHAPE);

        t[2] = ten(dcrd.p, RAD_I32, { 4, 3 });
        t[0] = ten(dx.p, RAD_F32, { 3, 4 });
        CHECK_EQ(run(kx, t, p), RAD_E_DTYPE);
        /* The last axis of x must be one element apart. */
        t[0] = ten(dx.p, RAD_BF16, { 3, 4 });
        t[0].stride[1] = 2;
        CHECK_EQ(run(kx, t, p), RAD_E_STRIDE);
    }
}

/* ---- attn_dense hd72 device timing -------------------------------------------------------- */

/* Device timing for the vision tower's dense attention, no oracle: warm up, then the fastest of
 * three batches, as attn_paged_device_test's RAD_ATTN_BENCH mode does. `rows` is the q plane's
 * extent, `heads` its query heads, `cu` bounds the real segments, and `msl` is the band's
 * `max_seqlen` (0 omits it). The FLOP count is 4 * sum(len^2) * head_dim over the cu segments,
 * all heads (QK^T and PV); the per-head figure the profile-ops report uses omits the head count.
 * The kbench reference case is one head (four 512 x 72 bf16 planes), so both are measured. */
bool bench_attn_dense(const char* name, int rows, int heads, const std::vector<int32_t>& cu,
                      int msl, int iters) {
    /* RAD_ATTN_DENSE_ONLY narrows the matrix to the cases whose name contains it, so a counter
     * run or a quick re-measure pays one geometry instead of the whole walk. */
    if (const char* only = std::getenv("RAD_ATTN_DENSE_ONLY")) {
        if (!std::strstr(name, only)) return true;
    }
    if (const char* only_rows = std::getenv("RAD_ATTN_DENSE_ROWS")) {
        if (std::atoi(only_rows) != rows) return true;
    }
    const RadKernelInfo* kd = g_r3d.find("attn_dense", RAD_DOMAIN_DEVICE, "attn_dense_bf16_h72");
    if (!kd) return true;
    const int hd = 72, nh = heads, gqa = 1;
    const size_t n = (size_t)rows * nh * hd;
    std::vector<uint16_t> q(n), k(n), v(n, 0x3c00u);
    std::mt19937 rng(91u + (unsigned)rows);
    std::uniform_int_distribution<int> U(-1024, 1024);
    for (size_t i = 0; i < n; ++i) {
        q[i] = bfv((float)U(rng) / 1024.0f);
        k[i] = bfv((float)U(rng) / 1024.0f);
    }
    Dev dq(n * 2), dk(n * 2), dv(n * 2), dout(n * 2), dcu(cu.size() * 4);
    up(dq, q.data(), dq.bytes);
    up(dk, k.data(), dk.bytes);
    up(dv, v.data(), dv.bytes);
    up(dcu, cu.data(), dcu.bytes);

    std::vector<RadParam> p{ pi("head_dim", hd), pi("gqa", gqa), pi("causal", 0),
                             ps("dtype", "bf16") };
    if (msl > 0) p.push_back(pi("max_seqlen", msl));
    std::vector<RadTensor> t{
        ten(dq.p, RAD_BF16, { rows, nh, hd }), ten(dk.p, RAD_BF16, { rows, nh, hd }),
        ten(dv.p, RAD_BF16, { rows, nh, hd }), ten(dcu.p, RAD_I32, { (long long)cu.size() }),
        ten(dout.p, RAD_BF16, { rows, nh, hd })
    };
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.kv_used_max_h = -1;

    const int rc = kd->launch(&a, (RadStream)0);
    hipDeviceSynchronize();
    if (rc != RAD_OK) {
        std::fprintf(stderr, "attn_dense bench %s: launch returned %d\n", name, rc);
        return false;
    }
    for (int i = 0; i < 2; ++i) kd->launch(&a, (RadStream)0);
    hipDeviceSynchronize();

    float best = 1e30f;
    for (int batch = 0; batch < 3; ++batch) {
        hipEvent_t e0, e1;
        hipEventCreate(&e0);
        hipEventCreate(&e1);
        hipEventRecord(e0, (hipStream_t)0);
        for (int i = 0; i < iters; ++i) kd->launch(&a, (RadStream)0);
        hipEventRecord(e1, (hipStream_t)0);
        hipEventSynchronize(e1);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, e0, e1);
        hipEventDestroy(e0);
        hipEventDestroy(e1);
        best = std::min(best, ms / (float)iters);
    }

    /* The real segment rows are the cu spans, not the padded plane; a many-segment pass does
     * sum(len^2) key pairs, not total^2. */
    double sq = 0.0;
    for (size_t s = 0; s + 1 < cu.size(); ++s) {
        const double len = (double)(cu[s + 1] - cu[s]);
        sq += len * len;
    }
    const double flops = 4.0 * sq * (double)hd * (double)nh;
    const double us = (double)best * 1000.0;
    std::printf("VISIONBENCH %-28s rows=%5d nh=%2d cu=%zu keys=%9.0f msl=%5d  %9.1f us  "
                "%7.1f GFLOP/s all heads  %6.1f GFLOP/s a head\n",
                name, rows, nh, cu.size() - 1, sq, msl, us, flops / (us * 1e3),
                flops / (double)nh / (us * 1e3));
    return true;
}

/* RAD_ATTN_DENSE_BENCH=1 walks M=512..16384 at one and sixteen heads, plus the served launch's
 * declared band and a many-segment pass. The 512/one-head case is the kbench reference; the
 * padded and many-segment cases separate the key walk's length from the workgroup count and from
 * the declared band. */
TEST(attn_dense_hd72_device_timing) {
    if (!std::getenv("RAD_ATTN_DENSE_BENCH")) return;
    if (!ready()) return;
    int nd = 0;
    if (hipGetDeviceCount(&nd) != hipSuccess || nd < 1) {
        radtest::skip("attn_dense bench: no device visible");
        return;
    }
    hipDeviceProp_t prop;
    std::memset(&prop, 0, sizeof prop);
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
        radtest::skip("attn_dense bench: device properties unavailable");
        return;
    }
    /* HIP reports the WGP count on RDNA, not the 96 CUs rocminfo lists. */
    std::printf("VISIONBENCH device=%s wgp=%d clock=%d kHz\n", prop.name,
                prop.multiProcessorCount, prop.clockRate);

    for (int M : { 512, 1024, 2048, 4096, 16384 })
        CHECK(bench_attn_dense("one segment, 1 head", M, 1, { 0, M }, M,
                               M <= 1024 ? 20 : (M <= 4096 ? 4 : 2)));

    for (int M : { 512, 1024, 2048, 4096, 16384 })
        CHECK(bench_attn_dense("one segment, 16 heads", M, 16, { 0, M }, M,
                               M <= 1024 ? 20 : (M <= 4096 ? 4 : 2)));

    /* The served launch declares max_seqlen 16384 with a 4096-row plane: the shim drops a band
     * past the row count, so the grid is the same as the unpadded case. */
    CHECK(bench_attn_dense("msl 16384, 4096 rows", 4096, 16, { 0, 4096 }, 16384, 4));

    /* A q plane padded to the declared band, one real segment of 4096 rows. */
    CHECK(bench_attn_dense("padded plane 16384", 16384, 16, { 0, 4096 }, 16384, 4));

    /* The same work as 32 segments of 128 rows: many short walks, 512 workgroups. */
    {
        std::vector<int32_t> cu;
        for (int s = 0; s <= 32; ++s) cu.push_back(s * 128);
        CHECK(bench_attn_dense("32 segments of 128", 4096, 16, cu, 128, 4));
    }
}

}  // namespace

RAD_TEST_MAIN()
