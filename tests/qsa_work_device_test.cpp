/* qsa_work_device_test.cpp -- libr3d's qsa_work row against libref, on the card.
 *
 * qsa_work is the gather that turns a step's batch into the flat block list qsa_block_key reads
 * (docs/OPS.md): one output row per completing block, its `ratio` raw keys staged contiguously,
 * its destination page, its first-token position and each sequence's complete-block count. The
 * cases here cover a whole prefill chunk, a block straddling the step's start (keys from the tail
 * state at q % ring), several sequences in one launch with one of them having no tail slot, a
 * ring that wraps across the step, ratio > 1, variable token counts, and the M-RoPE `rope` form
 * (a block whose first token is in the step and one from an earlier step).
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

bool same_bf16(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const char* what) {
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            std::fprintf(stderr, "%s differs at %zu: %04x vs %04x\n", what, i, a[i], b[i]);
            return false;
        }
    return true;
}
bool same_i32(const std::vector<int32_t>& a, const std::vector<int32_t>& b, const char* what) {
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            std::fprintf(stderr, "%s differs at %zu: %d vs %d\n", what, i, a[i], b[i]);
            return false;
        }
    return true;
}

/* ---- the case ----------------------------------------------------------------------------- */

struct Inputs {
    int T = 0, n = 0, ratio = 0, ring = 0, seqs = 0, slots = 0, work = 0;
    int sidx_cols = 1, bt_cols = 8;
    /* 0: rank 1 without rope, [work,3] with it. A positive value forces the engine's no-plane
     * form: shape [work,1] at that row pitch, so only component 0 is written. */
    int bpos_stride = 0;
    std::vector<int32_t> pos, cu, sidx, bt, rope;   /* rope empty: the one-component form */
    std::vector<uint16_t> k, state;
};

struct Outputs {
    std::vector<uint16_t> stage;
    std::vector<int32_t> page, bpos, nc;
};

/* Builds one launch's operands, runs `kd` and `kr` on equal inputs and equal output sentinels,
 * and returns both output sets. */
bool run_case(const Inputs& in, Outputs* dev, Outputs* ref, const char* what) {
    const size_t k_elems = (size_t)in.T * in.n;
    const size_t state_elems = (size_t)in.slots * in.ring * in.n;
    const size_t stage_elems = (size_t)in.work * in.ratio * in.n;
    const size_t page_elems = (size_t)in.work;
    const int bpos_pitch = in.rope.empty() ? (in.bpos_stride > 0 ? in.bpos_stride : 1) : 3;
    const size_t bpos_elems = (size_t)in.work * bpos_pitch;
    const size_t nc_elems = (size_t)in.seqs;

    auto p = std::vector<RadParam>{ pi("M", in.T), pi("n", in.n), pi("ratio", in.ratio),
                                    pi("ring", in.ring), pi("seqs", in.seqs),
                                    pi("work", in.work), ps("dtype", "bf16") };

    const RadKernelInfo* kd = g_r3d.find("qsa_work", RAD_DOMAIN_DEVICE, "qsa_work");
    const RadKernelInfo* kr = g_ref.find("qsa_work", RAD_DOMAIN_HOST, "ref_");
    CHECK(kd && kr);
    if (!(kd && kr)) return false;

    Dev dk(k_elems * 2), dstate(state_elems * 2), dpos((size_t)in.T * 4),
        dcu((size_t)(in.seqs + 1) * 4), dsidx((size_t)in.seqs * in.sidx_cols * 4),
        dbt((size_t)in.seqs * in.bt_cols * 4);
    Dev drope(in.rope.size() * 4);
    up(dk, in.k.data(), dk.bytes);
    up(dstate, in.state.data(), dstate.bytes);
    up(dpos, in.pos.data(), dpos.bytes);
    up(dcu, in.cu.data(), dcu.bytes);
    up(dsidx, in.sidx.data(), dsidx.bytes);
    up(dbt, in.bt.data(), dbt.bytes);
    if (in.rope.size()) up(drope, in.rope.data(), drope.bytes);

    Dev stage0(stage_elems * 2), stage1(stage_elems * 2);
    Dev page0(page_elems * 4), page1(page_elems * 4);
    Dev bpos0(bpos_elems * 4), bpos1(bpos_elems * 4);
    Dev nc0(nc_elems * 4), nc1(nc_elems * 4);
    std::vector<uint16_t> sfill(stage_elems, 0xA5A5u);
    std::vector<int32_t> ifill(page_elems, 0x7f7f7f7f);
    std::vector<int32_t> bfill(bpos_elems, 0x6e6e6e6e);
    std::vector<int32_t> nfill(nc_elems, 0x5d5d5d5d);
    for (Dev* d : { &stage0, &stage1 }) up(*d, sfill.data(), d->bytes);
    for (Dev* d : { &page0, &page1 }) up(*d, ifill.data(), d->bytes);
    for (Dev* d : { &bpos0, &bpos1 }) up(*d, bfill.data(), d->bytes);
    for (Dev* d : { &nc0, &nc1 }) up(*d, nfill.data(), d->bytes);

    auto tensors = [&](Dev& stage, Dev& page, Dev& bpos, Dev& nc) {
        std::vector<RadTensor> t;
        t.push_back(ten(dk.p, RAD_BF16, { in.T, in.n }));
        t.push_back(ten(dpos.p, RAD_I32, { in.T }));
        t.push_back(ten(dcu.p, RAD_I32, { in.seqs + 1 }));
        t.push_back(ten(dstate.p, RAD_BF16, { in.slots, 1, in.ring, in.n }));
        t.push_back(ten(dsidx.p, RAD_I32, { in.seqs, in.sidx_cols }));
        t.push_back(ten(dbt.p, RAD_I32, { in.seqs, in.bt_cols }));
        t.push_back(in.rope.empty() ? ten(nullptr, RAD_I32, { 0 })
                                    : ten(drope.p, RAD_I32, { 3, in.T }));
        t.push_back(ten(stage.p, RAD_BF16, { in.work * in.ratio, in.n }));
        t.push_back(ten(page.p, RAD_I32, { in.work }));
        if (!in.rope.empty())
            t.push_back(ten(bpos.p, RAD_I32, { in.work, 3 }));
        else if (in.bpos_stride > 1) {
            /* The engine's no-plane pass on an M-RoPE model: shape [work, 1] at the [work, 3]
             * plane's row pitch, so only column 0 is written. */
            RadTensor b = ten(bpos.p, RAD_I32, { in.work, 1 });
            b.stride[0] = in.bpos_stride;
            t.push_back(b);
        } else
            t.push_back(ten(bpos.p, RAD_I32, { in.work }));
        t.push_back(ten(nc.p, RAD_I32, { in.seqs }));
        return t;
    };

    std::vector<RadTensor> td = tensors(stage0, page0, bpos0, nc0);
    std::vector<RadTensor> tr = tensors(stage1, page1, bpos1, nc1);
    const int rd = run(kd, td, p);
    const int rr = run(kr, tr, p);
    if (rd != 0 || rr != 0) {
        std::fprintf(stderr, "%s: launch returned device %d, reference %d\n", what, rd, rr);
        return false;
    }

    dev->stage.assign(stage_elems, 0);
    ref->stage.assign(stage_elems, 0);
    dev->page.assign(page_elems, 0);
    ref->page.assign(page_elems, 0);
    dev->bpos.assign(bpos_elems, 0);
    ref->bpos.assign(bpos_elems, 0);
    dev->nc.assign(nc_elems, 0);
    ref->nc.assign(nc_elems, 0);
    down(stage0, dev->stage.data(), stage0.bytes);
    down(stage1, ref->stage.data(), stage1.bytes);
    down(page0, dev->page.data(), page0.bytes);
    down(page1, ref->page.data(), page1.bytes);
    down(bpos0, dev->bpos.data(), bpos0.bytes);
    down(bpos1, ref->bpos.data(), bpos1.bytes);
    down(nc0, dev->nc.data(), nc0.bytes);
    down(nc1, ref->nc.data(), nc1.bytes);

    bool ok = same_bf16(dev->stage, ref->stage, "stage");
    ok &= same_i32(dev->page, ref->page, "page");
    ok &= same_i32(dev->bpos, ref->bpos, "bpos");
    ok &= same_i32(dev->nc, ref->nc, "nc");
    return ok;
}

/* The tail state's element for (slot, row, d), so a gathered tail row is recognisable. */
uint16_t tail_val(int slot, int row, int d) { return bfv((float)(100 * slot + row) + 0.01f * (float)d); }

void fill_tail(std::vector<uint16_t>& state, int slots, int ring, int n) {
    for (int s = 0; s < slots; ++s)
        for (int r = 0; r < ring; ++r)
            for (int d = 0; d < n; ++d)
                state[((size_t)s * ring + r) * n + d] = tail_val(s, r, d);
}

void fill_keys(std::vector<uint16_t>& k, int T, int n, int base) {
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < n; ++d)
            k[(size_t)t * n + d] = bfv((float)(base + t) + 0.01f * (float)d);
}

/* ================================================================== cases */

/* A whole prefill chunk: every block completes from this step's keys, and the slot past the real
 * count reports page -1. */
TEST(qsa_work_prefill_matches_libref) {
    if (!ready()) return;
    Inputs in;
    in.T = 12; in.n = 8; in.ratio = 4; in.ring = 8; in.seqs = 1; in.slots = 2;
    in.work = in.T / in.ratio + in.seqs;   /* 4 */
    in.pos = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    in.cu = { 0, in.T };
    in.sidx = { 0 };
    in.bt.resize((size_t)in.seqs * in.bt_cols);
    for (int s = 0; s < in.seqs; ++s)
        for (int b = 0; b < in.bt_cols; ++b) in.bt[(size_t)s * in.bt_cols + b] = 1000 + 100 * s + b;
    in.k.resize((size_t)in.T * in.n);
    in.state.assign((size_t)in.slots * in.ring * in.n, 0);
    fill_keys(in.k, in.T, in.n, 0);
    fill_tail(in.state, in.slots, in.ring, in.n);

    Outputs d, r;
    CHECK(run_case(in, &d, &r, "prefill"));
    if (d.nc.size() != (size_t)in.seqs) return;
    CHECK_EQ(d.nc[0], 3);
    CHECK_EQ(d.page[0], 1000);
    CHECK_EQ(d.page[1], 1001);
    CHECK_EQ(d.page[2], 1002);
    CHECK_EQ(d.page[3], -1);
    CHECK_EQ(d.bpos[0], 0);
    CHECK_EQ(d.bpos[1], 4);
    CHECK_EQ(d.bpos[2], 8);
    CHECK_EQ(d.bpos[3], 0);
    for (int t = 0; t < in.T; ++t)
        for (int x = 0; x < in.n; ++x)
            CHECK_EQ(d.stage[(size_t)t * in.n + x], in.k[(size_t)t * in.n + x]);
}

/* The engine's M-RoPE model on a pass with no rotary planes hands bpos as [work, 1] at the
 * [work, 3] plane's row pitch; only component 0 is written, and the rest of the plane is not
 * qsa_work's. */
TEST(qsa_work_bpos_pitch_without_rope_matches_libref) {
    if (!ready()) return;
    Inputs in;
    in.T = 8; in.n = 8; in.ratio = 4; in.ring = 8; in.seqs = 1; in.slots = 1;
    in.work = in.T / in.ratio + in.seqs;   /* 3 */
    in.bpos_stride = 3;
    in.pos = { 0, 1, 2, 3, 4, 5, 6, 7 };
    in.cu = { 0, in.T };
    in.sidx = { 0 };
    in.bt.assign((size_t)in.bt_cols, 0);
    for (int b = 0; b < in.bt_cols; ++b) in.bt[(size_t)b] = 500 + b;
    in.k.resize((size_t)in.T * in.n);
    in.state.assign((size_t)in.slots * in.ring * in.n, 0);
    fill_keys(in.k, in.T, in.n, 700);
    fill_tail(in.state, in.slots, in.ring, in.n);

    Outputs d, r;
    CHECK(run_case(in, &d, &r, "bpos pitch"));
    if (d.nc.size() != (size_t)in.seqs) return;
    CHECK_EQ(d.nc[0], 2);
    CHECK_EQ(d.page[0], 500);
    CHECK_EQ(d.page[1], 501);
    CHECK_EQ(d.page[2], -1);
    CHECK_EQ(d.bpos[0], 0);
    CHECK_EQ(d.bpos[3], 4);
    CHECK_EQ(d.bpos[6], 0);
    for (size_t w = 0; w < (size_t)in.work; ++w)
        for (int c = 1; c < 3; ++c) CHECK_EQ(d.bpos[w * 3 + (size_t)c], 0x6e6e6e6e);
}

/* A decode step that completes a block straddling its start: key 8 is in the tail at row 8 % 6,
 * keys 9..11 are this step's rows. */
TEST(qsa_work_tail_straddle_and_ring_wrap_match_libref) {
    if (!ready()) return;
    Inputs in;
    in.T = 3; in.n = 8; in.ratio = 4; in.ring = 6; in.seqs = 1; in.slots = 1;
    in.work = (in.T + in.ratio - 1) / in.ratio + in.seqs;   /* 2 */
    in.pos = { 9, 10, 11 };
    in.cu = { 0, in.T };
    in.sidx = { 0 };
    in.bt.assign((size_t)in.bt_cols, 0);
    for (int b = 0; b < in.bt_cols; ++b) in.bt[(size_t)b] = 70 + b;
    in.k.resize((size_t)in.T * in.n);
    in.state.assign((size_t)in.slots * in.ring * in.n, 0);
    fill_keys(in.k, in.T, in.n, 500);
    fill_tail(in.state, in.slots, in.ring, in.n);

    Outputs d, r;
    CHECK(run_case(in, &d, &r, "tail straddle"));
    if (d.nc.size() != (size_t)in.seqs) return;
    CHECK_EQ(d.nc[0], 3);
    CHECK_EQ(d.page[0], 72);
    CHECK_EQ(d.page[1], -1);
    CHECK_EQ(d.bpos[0], 8);
    CHECK_EQ(d.bpos[1], 0);
    for (int x = 0; x < in.n; ++x) {
        CHECK_EQ(d.stage[(size_t)0 * in.n + x], tail_val(0, 8 % in.ring, x));
        for (int i = 1; i < in.ratio; ++i)
            CHECK_EQ(d.stage[(size_t)i * in.n + x], in.k[(size_t)(i - 1) * in.n + x]);
    }
    /* The spare work item left `stage` alone. */
    CHECK_EQ(d.stage[(size_t)in.ratio * in.n + 0], 0xA5A5u);
}

/* Two sequences in one launch, sequence 0 with two completing blocks and sequence 1 with one and
 * no tail slot; the state row has a pitch wider than one column. */
TEST(qsa_work_multiple_sequences_match_libref) {
    if (!ready()) return;
    Inputs in;
    in.T = 12; in.n = 8; in.ratio = 4; in.ring = 8; in.seqs = 2; in.slots = 2;
    in.work = in.T / in.ratio + in.seqs;   /* 5 */
    in.sidx_cols = 3;
    in.pos = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3 };
    in.cu = { 0, 8, 12 };
    in.sidx = { 1, -7, -7,   -1, -7, -7 };
    in.bt.assign((size_t)in.seqs * in.bt_cols, 0);
    for (int s = 0; s < in.seqs; ++s)
        for (int b = 0; b < in.bt_cols; ++b) in.bt[(size_t)s * in.bt_cols + b] = 300 + 100 * s + b;
    in.k.resize((size_t)in.T * in.n);
    in.state.assign((size_t)in.slots * in.ring * in.n, 0);
    fill_keys(in.k, in.T, in.n, 0);
    fill_tail(in.state, in.slots, in.ring, in.n);

    Outputs d, r;
    CHECK(run_case(in, &d, &r, "multiple sequences"));
    if (d.nc.size() != (size_t)in.seqs) return;
    CHECK_EQ(d.nc[0], 2);
    CHECK_EQ(d.nc[1], 1);
    CHECK_EQ(d.page[0], 300);
    CHECK_EQ(d.page[1], 301);
    CHECK_EQ(d.page[2], -1);            /* sequence 1 has no tail slot */
    CHECK_EQ(d.page[3], -1);
    CHECK_EQ(d.page[4], -1);
    CHECK_EQ(d.bpos[0], 0);
    CHECK_EQ(d.bpos[1], 4);
    CHECK_EQ(d.bpos[2], 0);
    CHECK_EQ(d.bpos[3], 0);
    CHECK_EQ(d.bpos[4], 0);
    for (int t = 0; t < 8; ++t)
        for (int x = 0; x < in.n; ++x)
            CHECK_EQ(d.stage[(size_t)t * in.n + x], in.k[(size_t)t * in.n + x]);
    /* Sequence 1's work item had no slot: its stage rows were not written. */
    for (size_t i = (size_t)8 * in.n; i < (size_t)in.work * in.ratio * in.n; ++i)
        CHECK_EQ(d.stage[i], 0xA5A5u);
}

/* M-RoPE: block 0 starts before the step (its first token is at index 0 of the step's shift) and
 * block 1 starts inside it. */
TEST(qsa_work_rope_matches_libref) {
    if (!ready()) return;
    Inputs in;
    in.T = 6; in.n = 8; in.ratio = 4; in.ring = 8; in.seqs = 1; in.slots = 1;
    in.work = (in.T + in.ratio - 1) / in.ratio + in.seqs;   /* 3 */
    in.pos = { 3, 4, 5, 6, 7, 8 };
    in.cu = { 0, in.T };
    in.sidx = { 0 };
    in.bt.assign((size_t)in.bt_cols, 0);
    for (int b = 0; b < in.bt_cols; ++b) in.bt[(size_t)b] = 40 + b;
    in.rope.resize((size_t)3 * in.T);
    for (int c = 0; c < 3; ++c)
        for (int t = 0; t < in.T; ++t)
            in.rope[(size_t)c * in.T + t] = (c + 1) * 100 + t;
    in.k.resize((size_t)in.T * in.n);
    in.state.assign((size_t)in.slots * in.ring * in.n, 0);
    fill_keys(in.k, in.T, in.n, 900);
    fill_tail(in.state, in.slots, in.ring, in.n);

    Outputs d, r;
    CHECK(run_case(in, &d, &r, "rope"));
    if (d.nc.size() != (size_t)in.seqs) return;
    CHECK_EQ(d.nc[0], 2);
    CHECK_EQ(d.page[0], 40);
    CHECK_EQ(d.page[1], 41);
    CHECK_EQ(d.page[2], -1);
    /* f = 0 < p0 = 3: every component is f + rope[0][0] - p0. */
    for (int c = 0; c < 3; ++c) CHECK_EQ(d.bpos[(size_t)0 * 3 + c], 0 + 100 - 3);
    /* f = 4 >= p0: component c reads rows 0 + (4 - 3). */
    for (int c = 0; c < 3; ++c) CHECK_EQ(d.bpos[(size_t)1 * 3 + c], (c + 1) * 100 + 1);
    for (int c = 0; c < 3; ++c) CHECK_EQ(d.bpos[(size_t)2 * 3 + c], 0);
    for (int x = 0; x < in.n; ++x) {
        /* Block 0: q = 0..2 from the tail, q = 3 from this step. */
        for (int i = 0; i < 3; ++i)
            CHECK_EQ(d.stage[(size_t)i * in.n + x], tail_val(0, i % in.ring, x));
        CHECK_EQ(d.stage[(size_t)3 * in.n + x], in.k[(size_t)0 * in.n + x]);
        /* Block 1: q = 4..7, all in this step. */
        for (int i = 0; i < in.ratio; ++i)
            CHECK_EQ(d.stage[(size_t)(4 + i) * in.n + x], in.k[(size_t)(1 + i) * in.n + x]);
    }
}

/* ratio 2: more blocks per token count, and the same gather law. */
TEST(qsa_work_ratio_two_matches_libref) {
    if (!ready()) return;
    Inputs in;
    in.T = 5; in.n = 8; in.ratio = 2; in.ring = 4; in.seqs = 1; in.slots = 1;
    in.work = (in.T + in.ratio - 1) / in.ratio + in.seqs;   /* 4 */
    in.pos = { 0, 1, 2, 3, 4 };
    in.cu = { 0, in.T };
    in.sidx = { 0 };
    in.bt.assign((size_t)in.bt_cols, 0);
    for (int b = 0; b < in.bt_cols; ++b) in.bt[(size_t)b] = 20 + b;
    in.k.resize((size_t)in.T * in.n);
    in.state.assign((size_t)in.slots * in.ring * in.n, 0);
    fill_keys(in.k, in.T, in.n, 100);
    fill_tail(in.state, in.slots, in.ring, in.n);

    Outputs d, r;
    CHECK(run_case(in, &d, &r, "ratio two"));
    if (d.nc.size() != (size_t)in.seqs) return;
    CHECK_EQ(d.nc[0], 2);
    CHECK_EQ(d.page[0], 20);
    CHECK_EQ(d.page[1], 21);
    CHECK_EQ(d.page[2], -1);
    CHECK_EQ(d.page[3], -1);
    CHECK_EQ(d.bpos[0], 0);
    CHECK_EQ(d.bpos[1], 2);
    CHECK_EQ(d.bpos[2], 0);
    CHECK_EQ(d.bpos[3], 0);
    for (int i = 0; i < in.ratio * 2; ++i)
        for (int x = 0; x < in.n; ++x)
            CHECK_EQ(d.stage[(size_t)i * in.n + x], in.k[(size_t)i * in.n + x]);
    CHECK_EQ(d.stage[(size_t)(in.ratio * 2) * in.n + 0], 0xA5A5u);
}

}  // namespace

RAD_TEST_MAIN()
