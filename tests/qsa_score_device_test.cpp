/* qsa_score_device_test.cpp -- libr3d's qsa_score row against libref's host oracle.
 *
 * The op is docs/OPS.md's:
 *
 *   score[m][b] = (sum over heads h of relu(q[m][h] . bk[bt[seq(m)][b]])) / sqrt(n)
 *
 * for b below min(nc[seq(m)], the table's width, the score plane's width). The test holds libr3d
 * and libref to the same answer bit for bit on the same bytes: the kernel walks a block's heads
 * and lanes in the reference's ascending f32 order, so no tolerance is needed. Cases cover one
 * and several sequences, query rows no sequence holds, cu starting at zero or not, several heads,
 * M from 1 to 19, block counts from 1 to 4096, incomplete blocks (nc below the width), a live
 * count above the width, negative page ids (-inf), a score plane wider than the table and the
 * reverse, a padded query row (a column slice of a wider projection), a rank-4 block-key store,
 * and bf16/f16/f32 activations with the block keys in another dtype.
 *
 * The libraries are dlopened and their rows found through rad_kernel_at, as the engine loads
 * them. On a machine without the card or the plugins every case skips; the test is meant to be
 * built everywhere and run where a gfx11 card is free.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include "r3d_codecs.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

size_t dt_bytes(int dt) { return (dt == RAD_BF16 || dt == RAD_F16) ? 2 : 4; }

/* A whole activation plane in `dt`, drawn in [lo, hi]. */
std::vector<uint8_t> rand_act(size_t nel, int dt, unsigned seed, float lo = -0.9f,
                              float hi = 0.9f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(lo, hi);
    std::vector<uint8_t> b(nel * dt_bytes(dt));
    for (size_t i = 0; i < nel; ++i) {
        const float v = U(rng);
        if (dt == RAD_BF16) {
            const uint16_t h = f32_to_bf16(v);
            std::memcpy(&b[i * 2], &h, 2);
        } else if (dt == RAD_F16) {
            const uint16_t h = f32_to_f16(v);
            std::memcpy(&b[i * 2], &h, 2);
        } else {
            std::memcpy(&b[i * 4], &v, 4);
        }
    }
    return b;
}

bool f32_same_bits(float a, float b) {
    uint32_t x = 0, y = 0;
    std::memcpy(&x, &a, 4);
    std::memcpy(&y, &b, 4);
    return x == y;
}

/* One geometry and its host operands. Everything is described here; run_case() builds the
 * libref and libr3d calls from it. */
struct Case {
    const char* what = "";
    int dt = RAD_BF16;          /* q's dtype */
    int kdt = RAD_BF16;         /* bk's dtype */
    long long rows = 1;
    long long n = 128;
    long long heads = 4;
    long long blocks = 8;       /* the table's width (bt's columns) */
    long long seqs = 1;
    long long bt_pitch = 0;     /* 0 = blocks */
    long long score_pitch = 0;  /* 0 = blocks */
    long long pages = 0;        /* 0 = blocks; bk's page count */
    bool q_slice = false;       /* q a column slice of a (heads+1)*n wide row */
    bool zero_q = false;        /* q all zeros: every head's dot is +0, so the score is +0 */
    int bk_rank = 2;
    long long bk_ratio = 4;     /* rank-4 store: [pages, 2, ratio, n/(2*ratio)] */
    std::vector<int32_t> cu, nc, bt;
    unsigned seed = 1;
};

const float kSentinel = 1.2345678e30f;

/* The sequence that owns row m, or -1 (libref's seq_holding). */
long long seq_holding(const std::vector<int32_t>& cu, long long m) {
    const long long nseq = (long long)cu.size() - 1;
    for (long long s = 0; s < nseq; ++s)
        if (m >= cu[(size_t)s] && m < cu[(size_t)s + 1]) return s;
    return -1;
}

void build_bt(Case& c, unsigned seed) {
    std::mt19937 rng(seed);
    const long long pitch = c.bt_pitch ? c.bt_pitch : c.blocks;
    const long long pages = c.pages ? c.pages : c.blocks;
    c.bt.assign((size_t)(c.seqs * pitch), -1);
    for (long long s = 0; s < c.seqs; ++s) {
        const long long live = c.nc[(size_t)s] < c.blocks ? c.nc[(size_t)s] : c.blocks;
        for (long long b = 0; b < live; ++b) {
            const long long v = (long long)(rng() % (unsigned)pages);
            c.bt[(size_t)(s * pitch + b)] = (int32_t)v;
        }
    }
}

bool check_case(const Case& c) {
    const long long nb = c.blocks < (c.bt_pitch ? c.bt_pitch : c.blocks)
                             ? c.blocks
                             : (c.bt_pitch ? c.bt_pitch : c.blocks);
    const long long bt_pitch = c.bt_pitch ? c.bt_pitch : c.blocks;
    const long long score_pitch = c.score_pitch ? c.score_pitch : c.blocks;
    const long long pages = c.pages ? c.pages : c.blocks;
    const long long q_pitch = c.q_slice ? (c.heads + 1) * c.n : c.heads * c.n;

    /* Host operands, in the layouts the op describes. */
    std::vector<uint8_t> q = rand_act((size_t)(c.rows * q_pitch), c.dt, c.seed + 1);
    if (c.zero_q) std::memset(q.data(), 0, q.size());
    std::vector<uint8_t> bk;
    if (c.bk_rank == 4) {
        const long long per_page = c.n;
        bk = rand_act((size_t)(pages * per_page), c.kdt, c.seed + 2);
    } else {
        bk = rand_act((size_t)(pages * c.n), c.kdt, c.seed + 2);
    }
    std::vector<int32_t> cu = c.cu, nc = c.nc, bt = c.bt;
    std::vector<float> out((size_t)(c.rows * score_pitch), kSentinel);

    RadTensor qh = ten(q.data(), c.dt, { c.rows, c.heads * c.n });
    if (c.q_slice) {
        qh.stride[0] = q_pitch;
        qh.stride[1] = 1;
    }
    RadTensor bkh;
    if (c.bk_rank == 4)
        bkh = ten(bk.data(), c.kdt, { pages, 2, c.bk_ratio, c.n / (2 * c.bk_ratio) });
    else
        bkh = ten(bk.data(), c.kdt, { pages, c.n });
    RadTensor bth = ten(bt.data(), RAD_I32, { c.seqs, bt_pitch });
    RadTensor cuh = ten(cu.data(), RAD_I32, { c.seqs + 1 });
    RadTensor nch = ten(nc.data(), RAD_I32, { c.seqs });
    RadTensor oh = ten(out.data(), RAD_F32, { c.rows, score_pitch });

    const char* dname = c.dt == RAD_BF16 ? "bf16" : c.dt == RAD_F16 ? "f16" : "f32";
    std::vector<RadParam> p = { pi("M", c.rows), pi("n", c.n), pi("heads", c.heads),
                                pi("blocks", c.blocks), pi("seqs", c.seqs), ps("dtype", dname) };

    const RadKernelInfo* kr = g_ref.find("qsa_score", RAD_DOMAIN_HOST, "ref_");
    const RadKernelInfo* kd = g_r3d.find("qsa_score", RAD_DOMAIN_DEVICE, "qsa_score");
    CHECK(kr && kd);
    if (!kr || !kd) return false;

    std::vector<RadTensor> th = { qh, bkh, bth, cuh, nch, oh };
    CHECK_EQ(run(kr, th, p), 0);

    /* The same bytes on the device, the output plane pre-filled with the same sentinel. */
    Dev dq(q.size()), dbk(bk.size()), dbt(bt.size() * 4), dcu(cu.size() * 4),
        dnc(nc.size() * 4), dout(out.size() * 4);
    up(dq, q.data(), q.size());
    up(dbk, bk.data(), bk.size());
    up(dbt, bt.data(), dbt.bytes);
    up(dcu, cu.data(), dcu.bytes);
    up(dnc, nc.data(), dnc.bytes);
    std::vector<float> sentinel_out((size_t)(c.rows * score_pitch), kSentinel);
    up(dout, sentinel_out.data(), dout.bytes);

    RadTensor qd = ten(dq.p, c.dt, { c.rows, c.heads * c.n });
    if (c.q_slice) {
        qd.stride[0] = q_pitch;
        qd.stride[1] = 1;
    }
    RadTensor bkd;
    if (c.bk_rank == 4)
        bkd = ten(dbk.p, c.kdt, { pages, 2, c.bk_ratio, c.n / (2 * c.bk_ratio) });
    else
        bkd = ten(dbk.p, c.kdt, { pages, c.n });
    RadTensor btd = ten(dbt.p, RAD_I32, { c.seqs, bt_pitch });
    RadTensor cud = ten(dcu.p, RAD_I32, { c.seqs + 1 });
    RadTensor ncd = ten(dnc.p, RAD_I32, { c.seqs });
    RadTensor od = ten(dout.p, RAD_F32, { c.rows, score_pitch });
    std::vector<RadTensor> td = { qd, bkd, btd, cud, ncd, od };
    CHECK_EQ(run(kd, td, p), 0);

    std::vector<float> got((size_t)(c.rows * score_pitch), 0.0f);
    down(dout, got.data(), dout.bytes);

    /* Coverage: exactly the rows a sequence holds, and below its live count, are written. */
    bool cover_ok = true;
    for (long long m = 0; m < c.rows; ++m) {
        const long long s = seq_holding(cu, m);
        const long long live = s < 0 ? 0 : (nc[(size_t)s] < nb ? nc[(size_t)s] : nb);
        for (long long b = 0; b < score_pitch; ++b) {
            const bool want = b < live;
            const size_t at = (size_t)(m * score_pitch + b);
            if (want && out[at] == kSentinel) {
                std::fprintf(stderr, "%s: libref left (row %lld, block %lld) unwritten\n",
                             c.what, m, b);
                cover_ok = false;
            }
            if (!want && out[at] != kSentinel) {
                std::fprintf(stderr, "%s: libref wrote past the bound at (row %lld, block %lld)\n",
                             c.what, m, b);
                cover_ok = false;
            }
        }
    }
    CHECK(cover_ok);

    /* And libref's answer, bit for bit. */
    bool same = true;
    size_t first = 0, diffs = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        if (!f32_same_bits(out[i], got[i])) {
            if (!diffs) first = i;
            ++diffs;
        }
    }
    if (diffs) {
        const size_t m = first / (size_t)score_pitch, b = first % (size_t)score_pitch;
        std::fprintf(stderr,
                     "%s: %zu of %zu scores differ; first at (row %zu, block %zu): "
                     "libr3d %.9g (0x%08x), libref %.9g (0x%08x)\n",
                     c.what, diffs, out.size(), m, b, got[first],
                     [&] { uint32_t u; std::memcpy(&u, &got[first], 4); return u; }(),
                     out[first],
                     [&] { uint32_t u; std::memcpy(&u, &out[first], 4); return u; }());
        same = false;
    }
    CHECK(same);
    return cover_ok && same;
}

/* ================================================================== the cases */

TEST(qsa_score_one_sequence_partial_bf16) {
    if (!ready()) return;
    Case c;
    c.what = "one sequence, 3 of 5 blocks complete, a negative page, bf16";
    c.rows = 3; c.n = 128; c.heads = 4; c.blocks = 8; c.seqs = 1; c.dt = c.kdt = RAD_BF16;
    c.cu = { 0, 3 };
    c.nc = { 5 };
    build_bt(c, 11);
    c.bt[1] = -1;
    c.bt[4] = -1;
    c.seed = 21;
    CHECK(check_case(c));
}

TEST(qsa_score_three_sequences_in_one_step_bf16) {
    if (!ready()) return;
    Case c;
    c.what = "three sequences in one step, unequal live counts, a wider score plane, bf16";
    c.rows = 7; c.n = 40; c.heads = 2; c.blocks = 12; c.seqs = 3; c.dt = c.kdt = RAD_BF16;
    c.score_pitch = 15;
    c.cu = { 0, 2, 5, 7 };
    c.nc = { 12, 3, 0 };
    build_bt(c, 12);
    c.bt[0] = -1;        /* a negative page in the first sequence's history */
    c.seed = 22;
    CHECK(check_case(c));
}

TEST(qsa_score_unheld_rows_and_cu_offset_bf16) {
    if (!ready()) return;
    Case c;
    c.what = "cu from 2, four rows no sequence holds, a dead sequence, bf16";
    c.rows = 12; c.n = 72; c.heads = 3; c.blocks = 10; c.seqs = 3; c.dt = c.kdt = RAD_BF16;
    c.cu = { 2, 5, 6, 10 };
    c.nc = { 4, 2, 7 };
    build_bt(c, 13);
    c.bt[5] = -1;
    c.seed = 23;
    CHECK(check_case(c));
}

TEST(qsa_score_f32_scattered_pages_and_rank4_store) {
    if (!ready()) return;
    Case c;
    c.what = "19 rows of one sequence, f32, a rank-4 block-key store, a narrower plane";
    c.rows = 19; c.n = 72; c.heads = 3; c.blocks = 41; c.seqs = 1; c.dt = c.kdt = RAD_F32;
    c.pages = 70;
    c.bk_rank = 4;
    c.score_pitch = 37;      /* nb = 37: the last four table entries are not scored */
    c.cu = { 0, 19 };
    c.nc = { 41 };
    build_bt(c, 14);
    c.bt[3] = -1;
    c.seed = 24;
    CHECK(check_case(c));
}

TEST(qsa_score_f16_query_against_f32_keys) {
    if (!ready()) return;
    Case c;
    c.what = "f16 queries against f32 keys, five heads, n 127, a table wider than the plane";
    c.rows = 4; c.n = 127; c.heads = 5; c.blocks = 5; c.seqs = 1;
    c.dt = RAD_F16; c.kdt = RAD_F32;
    c.bt_pitch = 7;          /* nb = 5: the last two table columns are ignored */
    c.pages = 9;
    c.cu = { 0, 4 };
    c.nc = { 5 };
    build_bt(c, 15);
    c.bt[2] = -1;
    c.seed = 25;
    CHECK(check_case(c));
}

TEST(qsa_score_long_context_two_sequences_bf16) {
    if (!ready()) return;
    Case c;
    c.what = "two sequences, 4000 and 17 live blocks of 4096, bf16";
    c.rows = 2; c.n = 128; c.heads = 4; c.blocks = 4096; c.seqs = 2; c.dt = c.kdt = RAD_BF16;
    c.cu = { 0, 1, 2 };
    c.nc = { 4000, 17 };
    build_bt(c, 16);
    c.seed = 26;
    CHECK(check_case(c));
}

TEST(qsa_score_one_block_one_row_bf16) {
    if (!ready()) return;
    Case c;
    c.what = "one block, one row, bf16";
    c.rows = 1; c.n = 128; c.heads = 4; c.blocks = 1; c.seqs = 1; c.dt = c.kdt = RAD_BF16;
    c.cu = { 0, 1 };
    c.nc = { 1 };
    c.bt = { 0 };
    c.seed = 27;
    CHECK(check_case(c));
}

TEST(qsa_score_zero_live_blocks_bf16) {
    if (!ready()) return;
    Case c;
    c.what = "a sequence with no complete block writes nothing, bf16";
    c.rows = 2; c.n = 64; c.heads = 2; c.blocks = 4; c.seqs = 1; c.dt = c.kdt = RAD_BF16;
    c.cu = { 0, 2 };
    c.nc = { 0 };
    c.bt = { 0, 1, 2, 3 };
    c.seed = 28;
    CHECK(check_case(c));
}

TEST(qsa_score_live_count_above_the_width_bf16) {
    if (!ready()) return;
    Case c;
    c.what = "nc above the declared width is clamped, bf16";
    c.rows = 3; c.n = 128; c.heads = 4; c.blocks = 5; c.seqs = 1; c.dt = c.kdt = RAD_BF16;
    c.cu = { 0, 3 };
    c.nc = { 999 };
    build_bt(c, 17);
    c.seed = 29;
    CHECK(check_case(c));
}

TEST(qsa_score_padded_query_row_f32) {
    if (!ready()) return;
    Case c;
    c.what = "q a column slice of a (heads+1)*n wide row, f32";
    c.rows = 19; c.n = 40; c.heads = 2; c.blocks = 70; c.seqs = 1;
    c.dt = c.kdt = RAD_F32;
    c.q_slice = true;
    c.cu = { 0, 19 };
    c.nc = { 70 };
    build_bt(c, 18);
    c.seed = 30;
    CHECK(check_case(c));
}

TEST(qsa_score_zero_dot_is_positive_zero_bf16) {
    if (!ready()) return;
    Case c;
    c.what = "q all zeros: every dot and score is +0, not -0, bf16";
    c.rows = 2; c.n = 128; c.heads = 4; c.blocks = 2; c.seqs = 1; c.dt = c.kdt = RAD_BF16;
    c.zero_q = true;
    c.cu = { 0, 2 };
    c.nc = { 2 };
    c.bt = { 0, 1 };
    c.seed = 31;
    CHECK(check_case(c));
}

TEST(qsa_score_more_rows_than_grid_y_bf16) {
    if (!ready()) return;
    Case c;
    c.what = "70000 rows: the launch splits rows past gridDim.y's 65535 onto z, bf16";
    c.rows = 70000; c.n = 1; c.heads = 1; c.blocks = 1; c.seqs = 1; c.dt = c.kdt = RAD_BF16;
    c.cu = { 0, 70000 };
    c.nc = { 1 };
    c.bt = { 0 };
    c.seed = 32;
    CHECK(check_case(c));
}

}  // namespace

RAD_TEST_MAIN()
