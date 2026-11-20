/* logit_rerank_iqr_device_test.cpp -- libr3d's `logit_rerank_iqr` row against libref's
 * `logit_rerank`, on the card, over the lm form r3d_iqr.h stores: B-bit codes, one bf16 scale per
 * 128 of K in the row's tail, and the int8 table after the last row.
 *
 * The test builds the stored head itself (the layout hook is exercised elsewhere), decodes the same
 * rows to the plain f32 head libref reads, and compares the re-sorted ids and values. Flash-Next's
 * shape (n_vocab 248320, n_embd 2560, R 8) is covered for u4 and u3 at every M the container
 * declares, 1..8; the smaller cases cover a vocabulary shard (`vocab_off`), ids at both edges of
 * the rank's rows, pads, duplicates, and an exact tie that must go to the lower id.
 */
#include "rad_test.h"
#include "rad_plugin.h"
#include "r3d_iqr.h"

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

using r3d_iqr::Form;

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

RadTensor ten_joined(void* p, uint32_t dtype, long long bytes) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = p;
    t.dtype = dtype;
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
uint16_t rnd_bf16() {
    const int q = (int)(rnd() % 4001) - 2000;
    return rad_f32_to_bf16((float)q / 64.0f);
}

/* One head in the lm form, with the plain f32 head the same rows decode to. Rows listed in
 * `same_as` are copied from an earlier row, so an exact tie can be built. */
struct Head {
    long long N = 0, K = 0;
    uint32_t bits = 0;
    Form f{};
    std::vector<uint8_t> joined;
    std::vector<float> plain;
    int64_t tab_off = 0;
};

bool build_head(long long N, long long K, uint32_t bits, Head* h, bool decode_plain,
                const std::vector<std::pair<long long, long long>>& same_as, const char* what) {
    h->N = N;
    h->K = K;
    h->bits = bits;
    h->f = Form{};
    h->f.kind = r3d_iqr::kTable;
    h->f.bits = bits;
    h->f.scale = r3d_iqr::kBf16;
    const long long stride = r3d_iqr::lm_row(h->f, K);
    const long long tab = r3d_iqr::lm_tab(h->f, N, K);
    const long long bytes = r3d_iqr::lm_bytes(h->f, N, K);
    h->tab_off = tab;
    h->joined.assign((size_t)bytes, 0);
    if (decode_plain) h->plain.assign((size_t)(N * K), 0.0f);

    std::vector<int8_t> table((size_t)(1u << bits));
    for (auto& t : table) t = (int8_t)((int)(rnd() % 241) - 120);
    std::memcpy(h->joined.data() + tab, table.data(), table.size());

    const long long cb = K * (long long)bits / 8;
    std::vector<uint8_t> codes((size_t)K);
    std::vector<uint16_t> sb((size_t)(K / 128));
    for (long long r = 0; r < N; ++r) {
        uint8_t* row = h->joined.data() + r * stride;
        auto copy_from = [&](long long src) {
            const uint8_t* s = h->joined.data() + src * stride;
            std::memcpy(row, s, (size_t)stride);
        };
        bool copied = false;
        for (const auto& p : same_as)
            if (p.first == r) { copy_from(p.second); copied = true; break; }
        if (copied) {
            if (decode_plain)
                std::memcpy(h->plain.data() + (size_t)(r * K),
                            h->plain.data() + (size_t)(same_as.front().second * K),
                            (size_t)K * sizeof(float));
            continue;
        }
        std::memset(row, 0, (size_t)stride);
        for (auto& c : codes) c = (uint8_t)(rnd() & ((1u << bits) - 1u));
        for (long long k = 0; k < K; ++k) {
            const long long bit = k * (long long)bits;
            const uint32_t v = (uint32_t)codes[(size_t)k] << (bit & 7);
            row[bit >> 3] |= (uint8_t)(v & 0xffu);
            row[(bit >> 3) + 1] |= (uint8_t)((v >> 8) & 0xffu);
        }
        for (auto& s : sb) s = rad_f32_to_bf16(0.001f + 0.0002f * (float)(rnd() % 13));
        std::memcpy(row + cb, sb.data(), (size_t)(2 * sb.size()));
        if (decode_plain)
            for (long long k = 0; k < K; ++k)
                h->plain[(size_t)(r * K + k)] =
                    (float)table[(size_t)codes[(size_t)k]] * rad_bf16_to_f32(sb[(size_t)(k / 128)]);
    }
    (void)what;
    return true;
}

/* The candidate list: values inside the rank's slice, its two edges, one below it, one above it, a
 * duplicate and -1. */
void fill_ids(std::vector<int32_t>* in, long long M, long long R, long long vo, long long nv) {
    in->assign((size_t)(M * R), -1);
    for (long long m = 0; m < M; ++m) {
        for (long long r = 0; r < R; ++r) {
            int32_t v;
            switch (r) {
                case 0: v = (int32_t)(vo + (m * 7 + 3) % nv); break;
                case 1: v = (int32_t)vo; break;
                case 2: v = (int32_t)(vo + nv - 1); break;
                case 3: v = (int32_t)(vo - 1); break;
                case 4: v = (int32_t)(vo + nv); break;
                case 5: v = (int32_t)(vo - 1); break;
                case 6: v = (int32_t)(vo + (m * 11 + 5) % nv); break;
                default: v = ((m + r) & 1) ? (int32_t)(vo + nv - 1) : (int32_t)vo; break;
            }
            (*in)[(size_t)(m * R + r)] = v;
        }
    }
}

/* The container's int8 head, joined through the row's own layout hook (K codes then K/128 bf16
 * scales a row), with the logical f32 head libref reads. */
bool build_i8_head(const RadKernelInfo* kd, long long N, long long K, Head* h, const char* what) {
    const long long ng = K / 128;
    std::vector<int8_t> codes((size_t)(N * K));
    for (auto& c : codes) c = (int8_t)((int)(rnd() % 255) - 127);
    std::vector<uint16_t> sb((size_t)(N * ng));
    for (auto& s : sb) s = rad_f32_to_bf16(0.001f + 0.0002f * (float)(rnd() % 13));
    h->N = N;
    h->K = K;
    h->bits = 0;
    h->f = Form{};
    h->plain.resize((size_t)(N * K));
    for (long long r = 0; r < N; ++r)
        for (long long k = 0; k < K; ++k)
            h->plain[(size_t)(r * K + k)] =
                (float)codes[(size_t)(r * K + k)] * rad_bf16_to_f32(sb[(size_t)(r * ng + k / 128)]);

    RadEncoding enc = rad_enc_affine(RAD_I8, RAD_BF16, 1, 128);
    const int sel[2] = { 0, 1 };
    const std::vector<RadParam> lp = { pint("M", 8), pint("R", 8), pint("n_vocab", N),
                                       pint("n_embd", K), pint("vocab_off", 0),
                                       pstr("dtype", "bf16") };
    RadTensor pl[2] = { ten2(codes.data(), RAD_I8, N, K, K, 1),
                        ten2(sb.data(), RAD_BF16, N, ng, ng, 1) };
    RadLayout lay{};
    int rc = kd->layout ? kd->layout(lp.data(), (int)lp.size(), 1, &enc, sel, pl, 2, &lay)
                        : RAD_E_UNSUPPORTED;
    if (rc != RAD_OK) {
        std::fprintf(stderr, "%s: head layout refused rc=%d\n", what, rc);
        return false;
    }
    if (lay.bytes <= N * K) {
        std::fprintf(stderr, "%s: head layout is %lld bytes, not the joined planes\n", what,
                     (long long)lay.bytes);
        return false;
    }
    h->joined.assign((size_t)lay.bytes, 0);
    rc = kd->relayout(lp.data(), (int)lp.size(), 1, &enc, sel, pl, 2, h->joined.data(), lay.bytes);
    if (rc != RAD_OK) {
        std::fprintf(stderr, "%s: head relayout refused rc=%d\n", what, rc);
        return false;
    }
    h->tab_off = -1;
    return true;
}

bool run_case(const RadKernelInfo* kd, const RadKernelInfo* kr, const Head& h, int M, int R,
              long long vo, const std::vector<int32_t>& in, const char* what,
              bool one_hot = false, bool i8form = false) {
    const long long N = h.N, K = h.K;
    std::vector<uint16_t> x((size_t)(M * K));
    if (one_hot) {
        for (int m = 0; m < M; ++m) x[(size_t)(m * K)] = rnd_bf16();
    } else {
        for (auto& v : x) v = rnd_bf16();
    }

    std::vector<RadParam> p = { pint("M", M), pint("R", R), pint("n_vocab", N),
                                pint("n_embd", K), pint("vocab_off", vo), pstr("dtype", "bf16") };
    const int n_p = (int)p.size();

    std::vector<int32_t> want_i((size_t)(M * R), 0);
    std::vector<float> want_v((size_t)(M * R), 0.0f);
    RadArgs ar;
    std::memset(&ar, 0, sizeof ar);
    RadTensor tr[6] = { ten2(x.data(), RAD_BF16, M, K, K, 1),
                        ten2((void*)h.plain.data(), RAD_F32, N, K, K, 1),
                        ten2((void*)in.data(), RAD_I32, M, R, R, 1),
                        ten2(want_i.data(), RAD_I32, M, R, R, 1),
                        ten2(want_v.data(), RAD_F32, M, R, R, 1),
                        RadTensor{} };
    tr[5].data = nullptr;
    ar.t = tr;
    ar.n_t = 6;
    ar.p = p.data();
    ar.n_p = n_p;

    Dev dx((size_t)(M * K) * 2), dh((size_t)h.joined.size()), di((size_t)(M * R) * 4),
        doi((size_t)(M * R) * 4), dov((size_t)(M * R) * 4);
    up(dx, x.data(), dx.bytes);
    up(dh, h.joined.data(), dh.bytes);
    up(di, in.data(), di.bytes);

    std::vector<RadParam> pd = p;
    RadTensor td[6] = { ten2(dx.p, RAD_BF16, M, K, K, 1),
                        i8form ? ten2(dh.p, RAD_I8, N, K, K, 1)
                               : ten_joined(dh.p, r3d_iqr::R3D_DT_IQR_W | r3d_iqr::word(h.f),
                                            (long long)h.joined.size()),
                        ten2(di.p, RAD_I32, M, R, R, 1),
                        ten2(doi.p, RAD_I32, M, R, R, 1),
                        ten2(dov.p, RAD_F32, M, R, R, 1),
                        RadTensor{} };
    td[5].data = nullptr;
    RadArgs ad;
    std::memset(&ad, 0, sizeof ad);
    ad.t = td;
    ad.n_t = 6;
    ad.p = pd.data();
    ad.n_p = n_p;

    const int rd = kd->launch(&ad, (RadStream)0);
    (void)hipDeviceSynchronize();
    const int rr = kr->launch(&ar, (RadStream)0);
    if (rd != 0 || rr != 0) {
        std::fprintf(stderr, "%s M=%d: launch returned device %d, reference %d\n", what, M, rd, rr);
        return false;
    }
    std::vector<int32_t> got_i((size_t)(M * R), 0);
    std::vector<float> got_v((size_t)(M * R), 0.0f);
    down(doi, got_i.data(), doi.bytes);
    down(dov, got_v.data(), dov.bytes);

    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < got_i.size(); ++i) {
        if (got_i[i] != want_i[i]) {
            std::fprintf(stderr, "%s M=%d: id %zu is %d, reference %d\n", what, M, i, got_i[i],
                         want_i[i]);
            return false;
        }
        const bool gi = std::isinf(got_v[i]), wi = std::isinf(want_v[i]);
        if (gi != wi || (gi && (got_v[i] < 0) != (want_v[i] < 0))) {
            std::fprintf(stderr, "%s M=%d: value %zu is %g, reference %g\n", what, M, i,
                         (double)got_v[i], (double)want_v[i]);
            return false;
        }
        if (gi) continue;
        const double d = (double)got_v[i] - (double)want_v[i];
        num += d * d;
        den += (double)want_v[i] * (double)want_v[i];
        mx = std::fmax(mx, std::fabs(d));
    }
    const double rel = den > 0 ? std::sqrt(num / den) : (num > 0 ? 1.0 : 0.0);
    if (rel > 2e-5) {
        std::fprintf(stderr, "%s M=%d R=%d vo=%lld: rel_l2 %.3e (max_abs %.3e)\n", what, M, R, vo,
                     rel, mx);
        return false;
    }
    return true;
}

}  // namespace

/* Flash-Next's head at the container's widths, u4, every declared decode height. */
TEST(logit_rerank_iqr_u4_matches_libref_at_flash_next) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("logit_rerank", RAD_DOMAIN_DEVICE, "logit_rerank_iqr");
    const RadKernelInfo* kr = g_ref.find("logit_rerank", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return;

    Head h;
    CHECK(build_head(248320, 2560, 4, &h, true, {}, "u4 head"));
    if (h.joined.empty()) return;
    for (int M = 1; M <= 8; ++M) {
        std::vector<int32_t> in;
        fill_ids(&in, M, 8, 0, h.N);
        CHECK(run_case(kd, kr, h, M, 8, 0, in, "u4 real"));
    }
}

/* Flash-Next's stored head is `i8*bf16[1x128]` (`output.weight`), not IQR; the i8 row is the one
 * that resolves on this container. Same shape and heights. */
TEST(logit_rerank_i8_matches_libref_at_flash_next) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("logit_rerank", RAD_DOMAIN_DEVICE, "logit_rerank_i8");
    const RadKernelInfo* kr = g_ref.find("logit_rerank", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return;

    Head h;
    CHECK(build_i8_head(kd, 248320, 2560, &h, "i8 head"));
    if (h.joined.empty()) return;
    for (int M = 1; M <= 8; ++M) {
        std::vector<int32_t> in;
        fill_ids(&in, M, 8, 0, h.N);
        CHECK(run_case(kd, kr, h, M, 8, 0, in, "i8 real", false, true));
    }
}

/* The u3 form the container's trellis-free heads use (logits_gemm_vocab_iqr's narrow classes). */
TEST(logit_rerank_iqr_u3_matches_libref_at_flash_next) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("logit_rerank", RAD_DOMAIN_DEVICE, "logit_rerank_iqr");
    const RadKernelInfo* kr = g_ref.find("logit_rerank", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return;

    Head h;
    CHECK(build_head(248320, 2560, 3, &h, true, {}, "u3 head"));
    if (h.joined.empty()) return;
    for (int M : { 1, 3, 5, 7, 8 }) {
        std::vector<int32_t> in;
        fill_ids(&in, M, 8, 0, h.N);
        CHECK(run_case(kd, kr, h, M, 8, 0, in, "u3 real"));
    }
}

/* A vocabulary shard: ids below `vocab_off` and above its last row are pads, and the ids written
 * back are global. */
TEST(logit_rerank_iqr_handles_a_vocabulary_shard) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("logit_rerank", RAD_DOMAIN_DEVICE, "logit_rerank_iqr");
    const RadKernelInfo* kr = g_ref.find("logit_rerank", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return;

    const long long N = 1000, K = 512, vo = 700;
    Head h;
    CHECK(build_head(N, K, 4, &h, true, {}, "shard head"));
    if (h.joined.empty()) return;
    for (int M : { 1, 3, 4 }) {
        std::vector<int32_t> in;
        fill_ids(&in, M, 8, vo, N);
        CHECK(run_case(kd, kr, h, M, 8, vo, in, "shard"));
    }
}

/* Two candidates whose head rows are identical and an activation with one nonzero column: the two
 * scores are exactly equal, and the reference's law takes the lower id first. */
TEST(logit_rerank_iqr_breaks_an_exact_tie_to_the_lower_id) {
    if (!ready()) return;
    const RadKernelInfo* kd = g_r3d.find("logit_rerank", RAD_DOMAIN_DEVICE, "logit_rerank_iqr");
    const RadKernelInfo* kr = g_ref.find("logit_rerank", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return;

    const long long N = 256, K = 512;
    Head h;
    CHECK(build_head(N, K, 4, &h, true, { { 11, 10 } }, "tie head"));
    if (h.joined.empty()) return;

    /* A one-hot x makes the dot a single product: the two identical head rows score the same. */
    std::vector<int32_t> in(8, -1);
    in[0] = 11;
    in[1] = 10;
    in[2] = 20;
    in[3] = 21;
    in[4] = 0;
    in[5] = 255;
    in[6] = -1;
    in[7] = 256;

    for (int M : { 1, 3 }) {
        std::vector<int32_t> ids((size_t)(M * 8), 0);
        for (int m = 0; m < M; ++m)
            for (int r = 0; r < 8; ++r) ids[(size_t)(m * 8 + r)] = in[(size_t)r];
        CHECK(run_case(kd, kr, h, M, 8, 0, ids, "tie", true));
    }
}

RAD_TEST_MAIN()