/* qsa_select_device_test.cpp -- libr3d's qsa_select against libref, on the card.
 *
 * The op turns the indexer's per-block scores into the page-id table the sparse attention reads:
 * the topk complete blocks a query has passed, ties to the lower block index, then the partial
 * page holding the query. This test builds the operands by hand (page ids, complete counts,
 * positions and the cu join are not values a generic draw produces) and launches both libraries
 * through their row tables, so the comparison covers the schema path too.
 *
 * The cases pin: several sequences and a prefill chunk (one query row each), the query's own
 * (pos + 1) / ratio bound, partial pages with and without a page id, topk above the block count,
 * a topk sweep across the tie threshold, signed and unsigned rankings (negative scores, -0/+0,
 * non-finite scores, the -1e30 sentinel), an empty selection, blocks the table width does not
 * cover, and a score plane narrower than the table. `sel`, `nsel` and `seqused` are compared
 * byte for byte: the attention sums pages in table order, so the order is part of the contract.
 *
 * The libraries are dlopened, as the engine loads them: this test links only the ABI and the HIP
 * runtime.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifndef RAD_R3D_SO
#define RAD_R3D_SO "libr3d.so"
#endif
#ifndef RAD_LIBREF_SO
#define RAD_LIBREF_SO "libref.so"
#endif

namespace {

struct Plugin {
    void* h = nullptr;
    const RadKernelInfo* row = nullptr;
    bool ok() const { return h && row && row->launch; }
};

Plugin load_plugin(const char* path, const char* op, int domain)
{
    Plugin p;
    p.h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!p.h) return p;
    auto count = (int (*)(void))dlsym(p.h, "rad_kernel_count");
    auto at = (const RadKernelInfo* (*)(int))dlsym(p.h, "rad_kernel_at");
    if (!count || !at) return p;
    for (int i = 0, n = count(); i < n; ++i) {
        const RadKernelInfo* k = at(i);
        if (k && k->op && std::strcmp(k->op, op) == 0 && k->domain == domain && k->launch) {
            p.row = k;
            break;
        }
    }
    return p;
}

Plugin& ours()
{
    static Plugin p = load_plugin(RAD_R3D_SO, "qsa_select", RAD_DOMAIN_DEVICE);
    return p;
}

Plugin& reference()
{
    static Plugin p = load_plugin(RAD_LIBREF_SO, "qsa_select", RAD_DOMAIN_HOST);
    return p;
}

bool have_card()
{
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

bool ready()
{
    if (!have_card()) {
        ::radtest::skip("no device visible");
        return false;
    }
    if (!ours().ok()) {
        ::radtest::artifact_missing(__FILE__, __LINE__, "libr3d qsa_select device row");
        return false;
    }
    if (!reference().ok()) {
        ::radtest::artifact_missing(__FILE__, __LINE__, "libref qsa_select row");
        return false;
    }
    return true;
}

constexpr int32_t kSentinel = 0x5A5A5A5A;

struct Fixture {
    int M = 0, nb = 0, snb = 0, topk = 0, ratio = 0, nseq = 0, sgn = 0;
    std::vector<float> score;
    std::vector<int32_t> bt, nc, pos, cu;
    std::vector<int32_t> sel, nsel, seqused;

    Fixture() = default;
    Fixture(int M_, int nb_, int topk_, int ratio_, int nseq_, int sgn_)
        : M(M_), nb(nb_), snb(nb_), topk(topk_), ratio(ratio_), nseq(nseq_), sgn(sgn_)
    {
        score.assign((size_t)M * snb, 0.0f);
        bt.assign((size_t)nseq * nb, -1);
        nc.assign((size_t)nseq, 0);
        pos.assign((size_t)M, 0);
        cu.assign((size_t)nseq + 1, 0);
        reset_out();
    }

    void reset_out()
    {
        sel.assign((size_t)M * (topk + 1), kSentinel);
        nsel.assign((size_t)M, kSentinel);
        seqused.assign((size_t)M, kSentinel);
    }

    void score_width(int w)
    {
        snb = w;
        score.assign((size_t)M * snb, 0.0f);
    }

    float& sc(int m, int b) { return score[(size_t)m * snb + b]; }
    int32_t& page(int s, int b) { return bt[(size_t)s * nb + b]; }
};

RadParam pint(const char* k, long long v)
{
    RadParam p{};
    p.key = k;
    p.kind = RAD_P_INT;
    p.ival = v;
    return p;
}

RadTensor tensor(void* p, uint32_t dt, std::initializer_list<int64_t> dims)
{
    RadTensor t{};
    t.data = p;
    t.dtype = dt;
    t.rank = (uint32_t)dims.size();
    int i = 0;
    for (int64_t d : dims) t.shape[i++] = d;
    rad_tensor_pack(&t);
    return t;
}

size_t tbytes(const RadTensor& t)
{
    return (size_t)rad_dtype_bytes(t.dtype, rad_tensor_numel(&t));
}

struct Dev {
    void* p[8]{};
    ~Dev()
    {
        for (void*& q : p)
            if (q) (void)hipFree(q);
    }
};

void expect_same(const char* what, const char* plane, const std::vector<int32_t>& want,
                 const std::vector<int32_t>& got, int topk)
{
    ++::radtest::checks();
    if (want.size() != got.size()) {
        ::radtest::fail(__FILE__, __LINE__,
                        std::string(what) + ": " + plane + " has " + std::to_string(got.size()) +
                            " entries against " + std::to_string(want.size()));
        return;
    }
    for (size_t i = 0; i < want.size(); ++i) {
        if (want[i] == got[i]) continue;
        char msg[512];
        if (std::strcmp(plane, "sel") == 0)
            std::snprintf(msg, sizeof msg,
                          "%s: sel[%zu][%zu] is %d, libref has %d (nsel row %zu)",
                          what, i / (size_t)(topk + 1), i % (size_t)(topk + 1), (int)got[i],
                          (int)want[i], i / (size_t)(topk + 1));
        else
            std::snprintf(msg, sizeof msg, "%s: %s[%zu] is %d, libref has %d", what, plane, i,
                          (int)got[i], (int)want[i]);
        ::radtest::fail(__FILE__, __LINE__, msg);
        return;
    }
}

void run_fixture(const char* what, Fixture& f)
{
    RadParam pp[6] = { pint("M", f.M),      pint("blocks", f.nb), pint("topk", f.topk),
                       pint("ratio", f.ratio), pint("seqs", f.nseq), pint("signed", f.sgn) };

    RadTensor th[8] = {
        tensor(f.score.data(), RAD_F32, { f.M, f.snb }),
        tensor(f.bt.data(), RAD_I32, { f.nseq, f.nb }),
        tensor(f.nc.data(), RAD_I32, { f.nseq }),
        tensor(f.pos.data(), RAD_I32, { f.M }),
        tensor(f.cu.data(), RAD_I32, { f.nseq + 1 }),
        tensor(f.sel.data(), RAD_I32, { f.M, f.topk + 1 }),
        tensor(f.nsel.data(), RAD_I32, { f.M }),
        tensor(f.seqused.data(), RAD_I32, { f.M }),
    };
    RadArgs ra{ th, 8, pp, 6, nullptr, 0, nullptr, 0, 1 };
    const int rc = reference().row->launch(&ra, nullptr);
    ++::radtest::checks();
    if (rc != RAD_OK) {
        ::radtest::fail(__FILE__, __LINE__,
                        std::string(what) + ": libref refused the case (" + std::to_string(rc) + ")");
        return;
    }
    const std::vector<int32_t> want_sel = f.sel, want_ns = f.nsel, want_su = f.seqused;

    Dev dev;
    RadTensor td[8];
    for (int i = 0; i < 8; ++i) {
        const size_t b = tbytes(th[i]);
        if (hipMalloc(&dev.p[i], b) != hipSuccess) {
            ::radtest::fail(__FILE__, __LINE__,
                            std::string(what) + ": hipMalloc(" + std::to_string(b) + ") failed");
            return;
        }
        if (hipMemcpy(dev.p[i], th[i].data, b, hipMemcpyHostToDevice) != hipSuccess) {
            ::radtest::fail(__FILE__, __LINE__,
                            std::string(what) + ": the operand copy to the device failed");
            return;
        }
        td[i] = th[i];
        td[i].data = dev.p[i];
    }

    RadArgs da{ td, 8, pp, 6, nullptr, 0, nullptr, 0, 1 };
    const int drc = ours().row->launch(&da, nullptr);
    ++::radtest::checks();
    if (drc != RAD_OK) {
        ::radtest::fail(__FILE__, __LINE__,
                        std::string(what) + ": the device row refused the launch (" +
                            std::to_string(drc) + ")");
        return;
    }
    ++::radtest::checks();
    if (hipDeviceSynchronize() != hipSuccess) {
        ::radtest::fail(__FILE__, __LINE__, std::string(what) + ": the launch did not complete");
        return;
    }
    for (int i = 5; i < 8; ++i) {
        if (hipMemcpy(th[i].data, dev.p[i], tbytes(th[i]), hipMemcpyDeviceToHost) != hipSuccess) {
            ::radtest::fail(__FILE__, __LINE__,
                            std::string(what) + ": the output copy back failed");
            return;
        }
    }
    expect_same(what, "sel", want_sel, f.sel, f.topk);
    expect_same(what, "nsel", want_ns, f.nsel, f.topk);
    expect_same(what, "seqused", want_su, f.seqused, f.topk);
}

int32_t& sel_at(Fixture& f, int m, int j) { return f.sel[(size_t)m * (f.topk + 1) + j]; }

void expect_row(const char* what, Fixture& f, int m, std::initializer_list<int32_t> want)
{
    int j = 0;
    if ((int)want.size() > f.topk + 1) {
        ::radtest::fail(__FILE__, __LINE__, std::string(what) + ": expectation is wider than the row");
        return;
    }
    for (int32_t v : want) {
        ++::radtest::checks();
        if (sel_at(f, m, j) != v) {
            char msg[256];
            std::snprintf(msg, sizeof msg, "%s: row %d slot %d is %d, expected %d", what, m, j,
                          (int)sel_at(f, m, j), (int)v);
            ::radtest::fail(__FILE__, __LINE__, msg);
            return;
        }
        ++j;
    }
}

TEST(qsa_select_unsigned_multiple_sequences)
{
    if (!ready()) return;
    Fixture f(3, 8, 3, 4, 3, 0);
    f.cu = { 0, 1, 2, 3 };
    f.nc = { 8, 5, 8 };
    f.pos = { 31, 19, 30 };   /* n = 8, 5, 7 (the third adds bt[7] as a partial page) */
    for (int s = 0; s < 3; ++s)
        for (int b = 0; b < f.nb; ++b) f.page(s, b) = 100 * (s + 1) + b;
    const float rows[3][8] = {
        { 1, 9, 3, 7, 2, 8, 4, 6 },
        { 9, 1, 8, 2, 7, 3, 6, 4 },
        { 2, 4, 9, 1, 8, 3, 7, 5 },
    };
    for (int m = 0; m < 3; ++m)
        for (int b = 0; b < f.nb; ++b) f.sc(m, b) = rows[m][b];
    run_fixture("unsigned, three sequences", f);
}

TEST(qsa_select_ties_break_to_lower_block)
{
    if (!ready()) return;
    Fixture f(3, 12, 5, 4, 3, 0);
    f.cu = { 0, 1, 2, 3 };
    f.nc = { 12, 12, 12 };
    f.pos = { 47, 47, 47 };
    for (int s = 0; s < 3; ++s)
        for (int b = 0; b < f.nb; ++b) f.page(s, b) = 100 * (s + 1) + b;
    for (int b = 0; b < f.nb; ++b) f.sc(0, b) = 5.0f;               /* every block ties */
    const float tie_cut[12] = { 9, 9, 9, 5, 5, 5, 5, 1, 1, 1, 1, 1 };
    const float tie_tail[12] = { 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6 };
    for (int b = 0; b < f.nb; ++b) { f.sc(1, b) = tie_cut[b]; f.sc(2, b) = tie_tail[b]; }
    run_fixture("ties at and above the threshold", f);
    expect_row("all-equal scores", f, 0, { 100, 101, 102, 103, 104, -1 });
    expect_row("ties cut at 5", f, 1, { 200, 201, 202, 203, 204, -1 });
    expect_row("one above the tie", f, 2, { 300, 301, 302, 303, 304, -1 });
}

TEST(qsa_select_topk_above_blocks_and_partial_page)
{
    if (!ready()) return;
    Fixture f(3, 5, 9, 4, 3, 0);
    f.cu = { 0, 1, 2, 3 };
    f.nc = { 5, 5, 3 };
    f.pos = { 19, 18, 2 };   /* n = 5 (full), 4 + partial bt[4], 0 + partial bt[0] */
    for (int s = 0; s < 3; ++s)
        for (int b = 0; b < f.nb; ++b) f.page(s, b) = 100 * (s + 1) + b;
    f.page(1, 4) = 144;
    f.page(2, 0) = 200;
    const float rows[3][5] = {
        { 1, 9, 3, 7, 5 },
        { 9, 1, 8, 2, 7 },
        { 2, 4, 9, 1, 8 },
    };
    for (int m = 0; m < 3; ++m)
        for (int b = 0; b < f.nb; ++b) f.sc(m, b) = rows[m][b];
    run_fixture("topk above the block count", f);
    /* Five live blocks are all taken in block order, then five -1 pads. */
    expect_row("full table under topk", f, 0,
               { 100, 101, 102, 103, 104, -1, -1, -1, -1 });
    /* Four selected blocks then the partial page. */
    expect_row("partial page after four", f, 1,
               { 200, 201, 202, 203, 144, -1, -1, -1, -1, -1 });
    /* No complete block: the partial page is the only entry. */
    expect_row("partial page only", f, 2, { 200, -1, -1, -1, -1, -1, -1, -1, -1, -1 });
}

TEST(qsa_select_topk_sweep)
{
    if (!ready()) return;
    const int nb = 17;
    for (int ratio : { 1, 2, 3, 4, 8 }) {
        Fixture base(1, nb, 1, ratio, 1, 0);
        base.cu = { 0, 1 };
        base.nc = { nb };
        base.pos = { nb * ratio - 1 };   /* n = nb, on a block boundary */
        for (int b = 0; b < nb; ++b) {
            base.page(0, b) = 10 + b;
            base.sc(0, b) = (float)(b % 5);   /* a tie at nearly every threshold */
        }
        for (int topk = 1; topk <= nb + 3; ++topk) {
            base.topk = topk;
            base.reset_out();
            char what[96];
            std::snprintf(what, sizeof what, "topk %d of %d, ratio %d", topk, nb, ratio);
            run_fixture(what, base);
        }
    }
}

TEST(qsa_select_signed_ranking)
{
    if (!ready()) return;
    Fixture f(1, 12, 4, 4, 1, 1);
    f.cu = { 0, 1 };
    f.nc = { 12 };
    f.pos = { 47 };   /* n = 12, no partial */
    for (int b = 0; b < f.nb; ++b) f.page(0, b) = 20 + b;
    const float ns = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float v[12] = { -5.0f, -1e30f, -2e30f, -0.0f, 0.0f, 3.0f,
                          -2.0f, inf, -inf, ns, 1e30f, -1e-30f };
    for (int b = 0; b < f.nb; ++b) f.sc(0, b) = v[b];
    run_fixture("signed ranking", f);
    /* Finite only: 1e30 (block 10), 3 (5), -1e-30 (11), then the -0/+0 tie at block 3
     * (lower index). Non-finite scores are dead. */
    expect_row("signed order", f, 0, { 25, 30, 23, 24, -1 });
    CHECK_EQ(f.nsel[0], 4);
    CHECK_EQ(f.seqused[0], 16);
}

TEST(qsa_select_unsigned_clamp_and_dead_scores)
{
    if (!ready()) return;
    Fixture f(1, 12, 4, 4, 1, 0);
    f.cu = { 0, 1 };
    f.nc = { 12 };
    f.pos = { 47 };
    for (int b = 0; b < f.nb; ++b) f.page(0, b) = 20 + b;
    const float ns = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float v[12] = { -5.0f, -1e30f, -2e30f, -0.0f, 0.0f, 3.0f,
                          -2.0f, inf, -inf, ns, 1e30f, -1e-30f };
    for (int b = 0; b < f.nb; ++b) f.sc(0, b) = v[b];
    run_fixture("unsigned ranking", f);
    /* Live: everything above -1e30, clamped to max(score, 0). Above the threshold: 3 (5),
     * inf (7), 1e30 (10); the first zero is block 0. */
    expect_row("unsigned order", f, 0, { 25, 27, 30, 20, -1 });
    CHECK_EQ(f.nsel[0], 4);
    CHECK_EQ(f.seqused[0], 16);
}

TEST(qsa_select_all_zero_scores)
{
    if (!ready()) return;
    Fixture f(1, 10, 4, 4, 1, 0);
    f.cu = { 0, 1 };
    f.nc = { 10 };
    f.pos = { 39 };
    for (int b = 0; b < f.nb; ++b) f.page(0, b) = 70 + b;
    run_fixture("all scores zero", f);
    expect_row("zero ties", f, 0, { 70, 71, 72, 73, -1 });
}

TEST(qsa_select_prefill_query_bounds)
{
    if (!ready()) return;
    Fixture f(7, 8, 3, 4, 2, 0);
    f.cu = { 0, 3, 7 };
    f.nc = { 8, 8 };
    f.pos = { 0, 1, 2, 7, 8, 9, 10 };
    for (int s = 0; s < 2; ++s)
        for (int b = 0; b < f.nb; ++b) f.page(s, b) = 100 * (s + 1) + b;
    f.page(1, 2) = 152;
    for (int m = 0; m < 7; ++m)
        for (int b = 0; b < f.nb; ++b) f.sc(m, b) = (float)((m * 7 + b * 3) % 11);
    /* The partial page is never a candidate, however high its score. */
    f.sc(0, 0) = 1e9f;
    f.sc(4, 2) = 1e9f;
    f.sc(5, 2) = 1e9f;
    f.sc(6, 2) = 1e9f;
    run_fixture("prefill query bounds", f);
    /* Rows 0..2 see no complete block: only the partial page bt[0]. */
    expect_row("n = 0, partial", f, 0, { 100, -1, -1, -1 });
    expect_row("n = 0, partial", f, 1, { 100, -1, -1, -1 });
    expect_row("n = 0, partial", f, 2, { 100, -1, -1, -1 });
}

TEST(qsa_select_partial_page_rules)
{
    if (!ready()) return;
    Fixture f(5, 6, 2, 4, 5, 0);
    f.cu = { 0, 1, 2, 3, 4, 5 };
    f.nc = { 6, 6, 6, 6, 0 };
    f.pos = { 23, 22, 22, 0, 0 };
    for (int s = 0; s < 5; ++s)
        for (int b = 0; b < f.nb; ++b) f.page(s, b) = 100 * (s + 1) + b;
    f.page(1, 5) = 177;    /* appended */
    f.page(2, 5) = -1;     /* absent: no partial page */
    for (int b = 0; b < f.nb; ++b) f.page(4, b) = -1;
    for (int m = 0; m < 5; ++m)
        for (int b = 0; b < f.nb; ++b) f.sc(m, b) = (float)(1 + (m * 5 + b * 7) % 13);
    run_fixture("partial page rules", f);
    /* n = 6 on a boundary: no partial even though n == nb. */
    CHECK_EQ(f.nsel[0], 2);
    /* n = 5 with (pos + 1) % ratio != 0: the partial page follows the selection. */
    CHECK_EQ(f.nsel[1], 3);
    CHECK_EQ(sel_at(f, 1, 2), 177);
    /* The same shape with bt[n] = -1 appends nothing. */
    CHECK_EQ(f.nsel[2], 2);
    /* n = 0 with (pos + 1) % ratio != 0: the partial page is the whole table. */
    CHECK_EQ(f.nsel[3], 1);
    CHECK_EQ(sel_at(f, 3, 0), 400);
    /* n = 0 and bt[0] = -1: no complete block and no partial page, so an empty table. */
    CHECK_EQ(f.nsel[4], 0);
    CHECK_EQ(f.seqused[4], 0);
    CHECK_EQ(sel_at(f, 4, 0), -1);
}

TEST(qsa_select_score_narrower_than_table)
{
    if (!ready()) return;
    Fixture f(1, 10, 3, 4, 1, 0);
    f.cu = { 0, 1 };
    f.nc = { 10 };
    f.pos = { 39 };   /* the query's bound is 10, but the score plane holds 6 columns */
    f.score_width(6);
    for (int b = 0; b < f.nb; ++b) f.page(0, b) = 90 + b;
    const float v[6] = { 4, 9, 1, 8, 2, 7 };
    for (int b = 0; b < f.snb; ++b) f.sc(0, b) = v[b];
    run_fixture("score narrower than the table", f);
    /* n stops at the score plane's width; blocks 6..9 are outside every query. */
    CHECK_EQ(f.nsel[0], 3);
    CHECK_EQ(sel_at(f, 0, 0), 91);   /* score 9 */
    CHECK_EQ(sel_at(f, 0, 1), 93);   /* score 8 */
    CHECK_EQ(sel_at(f, 0, 2), 95);   /* score 7 */
}

TEST(qsa_select_row_outside_cu_reads_sequence_zero)
{
    if (!ready()) return;
    Fixture f(3, 7, 2, 4, 2, 0);
    f.cu = { 0, 1, 1 };   /* rows 1 and 2 belong to no sequence (libref reads sequence 0) */
    f.nc = { 7, 0 };
    f.pos = { 27, 5, 9 };
    for (int s = 0; s < 2; ++s)
        for (int b = 0; b < f.nb; ++b) f.page(s, b) = 100 * (s + 1) + b;
    for (int m = 0; m < 3; ++m)
        for (int b = 0; b < f.nb; ++b) f.sc(m, b) = (float)(1 + (m * 3 + b * 5) % 17);
    run_fixture("row outside cu", f);
}

TEST(qsa_select_large_topk_across_rounds)
{
    if (!ready()) return;
    const int nb = 600, topk = 300;
    Fixture f(2, nb, topk, 4, 2, 0);
    f.cu = { 0, 1, 2 };
    f.nc = { nb, nb };
    f.pos = { nb * 4 - 1, nb * 4 - 2 };
    for (int s = 0; s < 2; ++s) {
        for (int b = 0; b < nb; ++b) {
            f.page(s, b) = 1000 * (s + 1) + b;
            /* Many exact ties, and dead pages on the second sequence. */
            f.sc(s, b) = (float)((b * 37) % 101);
        }
    }
    for (int b = 3; b < nb; b += 17) f.page(1, b) = -1;
    run_fixture("topk 300 over 600 blocks", f);
}

}  // namespace

RAD_TEST_MAIN()
