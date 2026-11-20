/* router_fused_device_test.cpp -- libr3d's `router_gemm_topk_scatter` row on the card, against
 * libref's fused oracle and against the device pair it is composed of (gemm_nt + the
 * router_topk_scatter row).
 *
 * The audit's Flash-Next graph asks for `router_gemm_topk_scatter` at M <= 64, n_expert 512 with
 * top_k 10 and `protect`; the plugin declares it beside the pair, and libr3d had only the host
 * fallback. This row serves the op in device. The cases here cover M = 1/16/64, top_k = 2/10,
 * `protect` off and on, the shared expert's gate present and absent, and exact-score ties (three
 * experts share the maximum, so the selection must break them to the lower id in both the device
 * row and libref).
 *
 * The data is built so the router GEMM is exact bit for bit: `a` is one-hot on the diagonal
 * (a[t,t] = 1), so logits[t,e] = b[e,t] and every dot is a single product whatever the split. The
 * router weights are k/512, all exactly representable in bf16, so the selection has no near-tie
 * fuzz: ids, sorted_tok, expert_offset and expert_count compare exactly, the bf16 gate compares
 * exactly, and the softmax weights (the one place the device's __expf and libref's double exp
 * differ) are compared to a small relative band.
 *
 * The libraries are dlopened and their rows found through rad_kernel_at, as the engine loads
 * them. On a machine without the card or the plugins every case skips.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <algorithm>
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
typedef int64_t (*ScratchFn)(const RadArgs*);

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

uint16_t bfv(float v) { return rad_f32_to_bf16(v); }

float bf2f(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/* The scratch a row asks for at this geometry: a real RadArgs, so the hook reads the same
 * parameters the launch will. */
int64_t scratch_for(const RadKernelInfo* k, std::vector<RadTensor>& t, std::vector<RadParam>& p) {
    if (!k->scratch) return 0;
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.kv_used_max_h = -1;
    return k->scratch(&a);
}

int run(const RadKernelInfo* k, std::vector<RadTensor>& t, std::vector<RadParam>& p, void* scratch,
        int64_t sbytes) {
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    a.scratch = scratch;
    a.scratch_bytes = sbytes;
    a.kv_used_max_h = -1;
    const int rc = k->launch(&a, (RadStream)0);
    hipDeviceSynchronize();
    return rc;
}

bool same_i32(const std::vector<int32_t>& a, const std::vector<int32_t>& b, const char* what) {
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            std::fprintf(stderr, "%s differs at %zu: %d vs %d\n", what, i, a[i], b[i]);
            return false;
        }
    return true;
}

bool same_bf16_exact(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b,
                     const char* what) {
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            std::fprintf(stderr, "%s differs at %zu: %04x vs %04x\n", what, i, a[i], b[i]);
            return false;
        }
    return true;
}

/* bf16 routing weights: the device narrows the probabilities and normalises by the narrowed sum,
 * libref sums the f32 values, so a percent of relative slack is the contract. Every weight is
 * O(1/top_k). */
bool close_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const char* what) {
    for (size_t i = 0; i < a.size(); ++i) {
        const float x = bf2f(a[i]), y = bf2f(b[i]);
        const float tol = 0.02f * std::max(std::fabs(y), 1e-4f) + 1e-5f;
        if (std::fabs(x - y) > tol) {
            std::fprintf(stderr, "%s differs at %zu: %.8g vs %.8g\n", what, i, (double)x,
                         (double)y);
            return false;
        }
    }
    return true;
}

/* score(t,e) is k/512 for a permutation of k over [0,512), except that two experts a row are
 * pinned to the maximum: three experts then tie at the top, which the selection must break to
 * the lower id. Every value is exactly representable in bf16. */
float score(int t, int e) {
    int rank = (int)(((long long)e * 73 + 19LL * t) % 512);
    if (e == (2 * t) % 512 || e == (2 * t + 1) % 512) rank = 511;
    return (float)(rank - 256) * (1.0f / 512.0f);
}

/* ---- the case ----------------------------------------------------------------------------- */

struct Inputs {
    int M = 0, top_k = 0;
    bool protect = false;
    bool gate = false;
    int xpad = 0;   /* extra elements per activation row: a padded row slice */
};

/* Runs the fused row against libref, and (when `with_pair`) against the device gemm_nt +
 * router_topk_scatter launches it composes. Returns false on any mismatch. */
bool run_case(const Inputs& in, bool with_pair, const char* what) {
    static const int E = 512, K = 2560;
    const int M = in.M, tk = in.top_k;

    const size_t b_elems = (size_t)E * K;
    const size_t logits_elems = (size_t)M * E;
    const size_t T = (size_t)M * tk;

    const int ap = K + in.xpad;   /* the activation's row pitch */
    std::vector<uint16_t> a((size_t)M * ap, 0), b(b_elems, 0), gb(in.gate ? (size_t)K : 0, 0);
    for (int t = 0; t < M; ++t) a[(size_t)t * ap + t] = bfv(1.0f);
    for (int e = 0; e < E; ++e)
        for (int t = 0; t < M; ++t) b[(size_t)e * K + t] = bfv(score(t, e));
    for (int t = 0; t < M; ++t)
        if (in.gate) gb[(size_t)t] = bfv(0.25f * (float)(t + 1));

    std::vector<RadParam> p = {
        RAD_INT("M", M), RAD_INT("K", K), RAD_INT("n_expert", E), RAD_INT("n_local", E),
        RAD_INT("top_k", tk), RAD_INT("norm", 1), RAD_STR("dtype", "bf16"),
        RAD_INT("expert_base", 0)
    };
    if (in.protect) p.push_back(RAD_STR("protect", "4,200,511"));

    const RadKernelInfo* kd =
        g_r3d.find("router_gemm_topk_scatter", RAD_DOMAIN_DEVICE, "moe_router_gemm_topk_scatter");
    const RadKernelInfo* kr = g_ref.find("router_gemm_topk_scatter", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return false;

    Dev da(a.size() * 2), db(b_elems * 2), dgb(gb.size() * 2);
    up(da, a.data(), da.bytes);
    up(db, b.data(), db.bytes);
    if (!gb.empty()) up(dgb, gb.data(), dgb.bytes);

    const std::vector<uint16_t> logits_fill(logits_elems, 0xA5A5u);
    const std::vector<uint16_t> w_fill(T, 0xA5A5u);
    const std::vector<uint16_t> gate_fill(M, 0xA5A5u);
    const std::vector<int32_t> ids_fill(T, 0x6e6e6e6e);
    const std::vector<int32_t> sorted_fill(T, 0x5d5d5d5d);
    const std::vector<int32_t> off_fill((size_t)E + 1, 0x4c4c4c4c);
    const std::vector<int32_t> cnt_fill((size_t)E, 0x3b3b3b3b);

    Dev lg0(logits_elems * 2), w0(T * 2), gate0(M * 2), ids0(T * 4), st0(T * 4),
        off0(((size_t)E + 1) * 4), cnt0((size_t)E * 4);
    Dev lg1(logits_elems * 2), w1(T * 2), gate1(M * 2), ids1(T * 4), st1(T * 4),
        off1(((size_t)E + 1) * 4), cnt1((size_t)E * 4);
    for (Dev* d : { &lg0, &lg1 }) up(*d, logits_fill.data(), d->bytes);
    for (Dev* d : { &w0, &w1 }) up(*d, w_fill.data(), d->bytes);
    for (Dev* d : { &gate0, &gate1 }) up(*d, gate_fill.data(), d->bytes);
    for (Dev* d : { &ids0, &ids1 }) up(*d, ids_fill.data(), d->bytes);
    for (Dev* d : { &st0, &st1 }) up(*d, sorted_fill.data(), d->bytes);
    for (Dev* d : { &off0, &off1 }) up(*d, off_fill.data(), d->bytes);
    for (Dev* d : { &cnt0, &cnt1 }) up(*d, cnt_fill.data(), d->bytes);

    auto operands = [&](Dev& lg, Dev& w, Dev& gate, Dev& ids, Dev& st, Dev& off, Dev& cnt) {
        RadTensor xt = ten(da.p, RAD_BF16, { M, K });
        xt.stride[0] = ap;   /* a padded row slice keeps its pitch */
        std::vector<RadTensor> v;
        v.push_back(xt);
        v.push_back(ten(db.p, RAD_BF16, { E, K }));
        v.push_back(in.gate ? ten(dgb.p, RAD_BF16, { 1, K }) : RadTensor{});
        v.push_back(ten(lg.p, RAD_BF16, { M, E }));
        v.push_back(ten(ids.p, RAD_I32, { M, tk }));
        v.push_back(ten(w.p, RAD_BF16, { M, tk }));
        v.push_back(ten(st.p, RAD_I32, { (long long)T }));
        v.push_back(ten(off.p, RAD_I32, { E + 1 }));
        v.push_back(ten(cnt.p, RAD_I32, { E }));
        if (in.gate) v.push_back(ten(gate.p, RAD_BF16, { M, 1 }));
        return v;
    };

    std::vector<RadTensor> td = operands(lg0, w0, gate0, ids0, st0, off0, cnt0);
    std::vector<RadTensor> tr = operands(lg1, w1, gate1, ids1, st1, off1, cnt1);

    const int64_t need = scratch_for(kd, td, p);
    Dev dscratch(need > 0 ? (size_t)need : 0);
    const int rd = run(kd, td, p, dscratch.p, need);
    const int rr = run(kr, tr, p, nullptr, 0);
    if (rd != 0 || rr != 0) {
        std::fprintf(stderr, "%s: fused device %d, reference %d\n", what, rd, rr);
        return false;
    }

    std::vector<uint16_t> dlg(logits_elems), rlg(logits_elems), dw(T), rw(T), dgate(M), rgate(M);
    std::vector<int32_t> dids(T), rids(T), dst(T), rst(T), doff((size_t)E + 1),
        roff((size_t)E + 1), dcnt((size_t)E), rcnt((size_t)E);
    down(lg0, dlg.data(), lg0.bytes);
    down(lg1, rlg.data(), lg1.bytes);
    down(w0, dw.data(), w0.bytes);
    down(w1, rw.data(), w1.bytes);
    down(gate0, dgate.data(), gate0.bytes);
    down(gate1, rgate.data(), gate1.bytes);
    down(ids0, dids.data(), ids0.bytes);
    down(ids1, rids.data(), ids1.bytes);
    down(st0, dst.data(), st0.bytes);
    down(st1, rst.data(), st1.bytes);
    down(off0, doff.data(), off0.bytes);
    down(off1, roff.data(), off1.bytes);
    down(cnt0, dcnt.data(), cnt0.bytes);
    down(cnt1, rcnt.data(), cnt1.bytes);

    bool ok = same_bf16_exact(dlg, rlg, "logits");
    ok &= same_i32(dids, rids, "expert_ids");
    ok &= same_i32(dst, rst, "sorted_tok");
    ok &= same_i32(doff, roff, "expert_offset");
    ok &= same_i32(dcnt, rcnt, "expert_count");
    ok &= close_bf16(dw, rw, "expert_w");
    if (in.gate) ok &= same_bf16_exact(dgate, rgate, "gate");

    /* The ids must be the top_k of the score table, ties to the lower id -- computed here rather
     * than read from either library. */
    for (int t = 0; t < M; ++t) {
        std::vector<std::pair<float, int>> all;
        for (int e = 0; e < E; ++e) all.emplace_back(score(t, e), e);
        std::sort(all.begin(), all.end(), [](const auto& x, const auto& y) {
            return x.first != y.first ? x.first > y.first : x.second < y.second;
        });
        for (int j = 0; j < tk; ++j)
            if (dids[(size_t)t * tk + j] != all[(size_t)j].second) {
                std::fprintf(stderr, "%s: row %d slot %d: id %d, expected %d\n", what, t, j,
                             dids[(size_t)t * tk + j], all[(size_t)j].second);
                ok = false;
            }
    }

    /* The sort's shape: offsets nondecreasing, the last the kept count, counts summing to it, and
     * sorted_tok a permutation of [0, T) (every id here is in this rank's slice). */
    long long kept = 0;
    for (int e = 0; e < E; ++e) {
        ok &= doff[(size_t)e] <= doff[(size_t)e + 1];
        kept += dcnt[(size_t)e];
    }
    ok &= doff[(size_t)E] == kept;
    if (doff[(size_t)E] != (long long)T) {
        std::fprintf(stderr, "%s: kept %lld of %zu slots\n", what, (long long)doff[(size_t)E], T);
        ok = false;
    }
    std::vector<int> seen(T, 0);
    for (size_t i = 0; i < T; ++i) {
        const long long s = dst[i];
        if (s < 0 || s >= (long long)T || seen[(size_t)s]++) {
            std::fprintf(stderr, "%s: sorted_tok[%zu] = %lld is not a permutation\n", what, i, s);
            ok = false;
            break;
        }
    }

    /* The device pair the fused row stands in for, compared byte for byte: the skinny m16 GEMM at
     * M <= 16, the tiled row above, then the router_topk_scatter row. */
    if (with_pair) {
        const RadKernelInfo* kg = M <= 16
            ? g_r3d.find("gemm_nt", RAD_DOMAIN_DEVICE, "gemm_nt_bf16_m16")
            : g_r3d.find("gemm_nt", RAD_DOMAIN_DEVICE, "gemm_nt_bf16_tiled");
        const RadKernelInfo* kp =
            g_r3d.find("router_topk_scatter", RAD_DOMAIN_DEVICE, "moe_router_topk_scatter");
        CHECK(kg && kp);
        if (!(kg && kp)) return false;

        Dev plg(logits_elems * 2), pw(T * 2), pids(T * 4), pst(T * 4), poff(((size_t)E + 1) * 4),
            pcnt((size_t)E * 4);
        RadTensor px = ten(da.p, RAD_BF16, { M, K });
        px.stride[0] = ap;
        std::vector<RadTensor> g_t = { px, ten(db.p, RAD_BF16, { E, K }),
                                       ten(plg.p, RAD_BF16, { M, E }) };
        std::vector<RadParam> g_p = { RAD_INT("M", M), RAD_INT("N", E), RAD_INT("K", K),
                                      RAD_STR("dtype", "bf16") };
        const int64_t gneed = scratch_for(kg, g_t, g_p);
        Dev gscratch(gneed > 0 ? (size_t)gneed : 0);
        const int rg = run(kg, g_t, g_p, gscratch.p, gneed);
        std::vector<RadTensor> p_t = { ten(plg.p, RAD_BF16, { M, E }),
                                       ten(pids.p, RAD_I32, { M, tk }),
                                       ten(pw.p, RAD_BF16, { M, tk }),
                                       ten(pst.p, RAD_I32, { (long long)T }),
                                       ten(poff.p, RAD_I32, { E + 1 }),
                                       ten(pcnt.p, RAD_I32, { E }) };
        const int rp = run(kp, p_t, p, nullptr, 0);
        if (rg != 0 || rp != 0) {
            std::fprintf(stderr, "%s: pair device %d / %d\n", what, rg, rp);
            return false;
        }
        std::vector<uint16_t> plogits(logits_elems), pwv(T);
        std::vector<int32_t> pidsv(T), pstv(T), poffv((size_t)E + 1), pcntv((size_t)E);
        down(plg, plogits.data(), plg.bytes);
        down(pw, pwv.data(), pw.bytes);
        down(pids, pidsv.data(), pids.bytes);
        down(pst, pstv.data(), pst.bytes);
        down(poff, poffv.data(), poff.bytes);
        down(pcnt, pcntv.data(), pcnt.bytes);
        ok &= same_bf16_exact(dlg, plogits, "logits vs pair");
        ok &= same_i32(dids, pidsv, "expert_ids vs pair");
        ok &= same_i32(dst, pstv, "sorted_tok vs pair");
        ok &= same_i32(doff, poffv, "expert_offset vs pair");
        ok &= same_i32(dcnt, pcntv, "expert_count vs pair");
        ok &= same_bf16_exact(dw, pwv, "expert_w vs pair");
    }
    return ok;
}

/* ================================================================== cases */

/* The audit's decode band: M = 1, 16 and 64, also against the device pair. */
TEST(router_fused_m1_top2_matches_libref_and_pair) {
    if (!ready()) return;
    Inputs in;
    in.M = 1; in.top_k = 2; in.protect = false;
    CHECK(run_case(in, true, "M1 k2"));
}

TEST(router_fused_m16_top2_protect_matches_libref_and_pair) {
    if (!ready()) return;
    Inputs in;
    in.M = 16; in.top_k = 2; in.protect = true;
    CHECK(run_case(in, true, "M16 k2 protect"));
}

TEST(router_fused_m64_top10_matches_libref_and_pair) {
    if (!ready()) return;
    Inputs in;
    in.M = 64; in.top_k = 10; in.protect = false;
    CHECK(run_case(in, true, "M64 k10"));
}

/* The band crossing: M = 17 is the first step the router serves with the tiled GEMM instead of
 * the m16 arm, and the shared gate's 16-row chunking takes a two-row second chunk. */
TEST(router_fused_m17_crosses_to_tiled_and_pair) {
    if (!ready()) return;
    Inputs in;
    in.M = 17; in.top_k = 10; in.protect = true; in.gate = true;
    CHECK(run_case(in, true, "M17 k10 protect"));
}

/* A padded activation row slice: the row pitch is K + 8, as a slice of a wider buffer would be.
 * Both the router GEMM (the m16 arm at M = 16 and the tiled arm at 17) and the gate chunks must
 * stride by the tensor's pitch, not by K. */
TEST(router_fused_pitched_activation_matches_libref_and_pair) {
    if (!ready()) return;
    for (int M : { 16, 17 }) {
        Inputs in;
        in.M = M; in.top_k = 10; in.protect = true; in.gate = true; in.xpad = 8;
        CHECK(run_case(in, true, "pitched act"));
    }
}

/* The rest of the matrix: every M with both top_k, protect on and off, and the shared gate
 * present. The ties are in every row (three experts share the maximum). */
TEST(router_fused_matrix_matches_libref) {
    if (!ready()) return;
    for (int M : { 1, 16, 64 })
        for (int tk : { 2, 10 })
            for (int prot : { 0, 1 }) {
                Inputs in;
                in.M = M; in.top_k = tk; in.protect = prot != 0; in.gate = true;
                if (!run_case(in, false, "matrix")) return;
            }
}

TEST(router_fused_gate_absent_matches_libref) {
    if (!ready()) return;
    Inputs in;
    in.M = 16; in.top_k = 10; in.protect = true; in.gate = false;
    CHECK(run_case(in, false, "no gate"));
}

/* The contract's refusals: the gate weight and its output come together, M stays at or below 64,
 * top_k at or below 32, and K on the tiled arm's 32-code chunk. Every one is a launch-time code,
 * returned before a kernel runs, so the descriptors alone are enough. */
TEST(router_fused_refuses_malformed_calls) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("router_gemm_topk_scatter", RAD_DOMAIN_DEVICE,
                                         "moe_router_gemm_topk_scatter");
    REQUIRE(kd);

    uintptr_t next = 0x1000000;
    auto dummy = [&](int dtype, std::initializer_list<long long> dims) {
        RadTensor t = ten((void*)next, dtype, dims);
        next += 0x100000;   /* stays non-null and 16-byte aligned; the shim refuses before reading */
        return t;
    };
    /* gate_mode: 0 none, 1 both, 2 gate_b only, 3 gate only. */
    auto call = [&](long long M, long long K, long long tk, int gate_mode) {
        std::vector<RadTensor> t;
        t.push_back(dummy(RAD_BF16, { M, K }));
        t.push_back(dummy(RAD_BF16, { 512, K }));
        t.push_back(gate_mode == 1 || gate_mode == 2 ? dummy(RAD_BF16, { 1, K }) : RadTensor{});
        t.push_back(dummy(RAD_BF16, { M, 512 }));
        t.push_back(dummy(RAD_I32, { M, tk }));
        t.push_back(dummy(RAD_BF16, { M, tk }));
        t.push_back(dummy(RAD_I32, { M * tk }));
        t.push_back(dummy(RAD_I32, { 513 }));
        t.push_back(dummy(RAD_I32, { 512 }));
        if (gate_mode == 1 || gate_mode == 3) t.push_back(dummy(RAD_BF16, { M, 1 }));
        std::vector<RadParam> p = { RAD_INT("M", M), RAD_INT("K", K), RAD_INT("n_expert", 512),
                                    RAD_INT("n_local", 512), RAD_INT("top_k", tk),
                                    RAD_INT("norm", 1), RAD_STR("dtype", "bf16"),
                                    RAD_INT("expert_base", 0) };
        return run(kd, t, p, nullptr, 0);
    };

    CHECK_EQ(call(64, 2560, 10, 2), (int)RAD_E_INVAL);   /* gate_b without gate */
    CHECK_EQ(call(64, 2560, 10, 3), (int)RAD_E_INVAL);   /* gate without gate_b */
    CHECK_EQ(call(65, 2560, 10, 0), (int)RAD_E_SHAPE);   /* M past the band */
    CHECK_EQ(call(16, 2560, 33, 0), (int)RAD_E_SHAPE);   /* top_k past 32 */
    CHECK_EQ(call(16, 2552, 10, 0), (int)RAD_E_SHAPE);   /* K not on the 32-code chunk */
}

}  // namespace

RAD_TEST_MAIN()
