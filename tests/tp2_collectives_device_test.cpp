/* tp2_collectives_device_test.cpp -- libr3d's tensor-parallel collectives at two ranks.
 *
 * The engine's tp>1 path needs two cards, which this host does not have. The kernels themselves do
 * not: every rank is a thread on its own stream, and a peer on the same card needs no peer access
 * (r3dh_ar_entry.hip, "a peer on this rank's own card needs none"). This test builds the peer
 * instance the way the engine does (init under the family's rendezvous, one stream a rank) and
 * checks the two rows a tp>1 decode issues: all_reduce and all_gather, both exact, flat.
 *
 * The message is fp32 so every sum is exact and the comparison can be bitwise: rank 0 sends 1.0
 * and rank 1 sends 2.0, and each rank must read 3.0 everywhere. The gather moves a rank-tagged
 * pattern, so a byte in the wrong rank's slice is caught even when the total count is right.
 */
#include "rad_test.h"
#include "rad_abi.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifndef RAD_R3D_SO
#define RAD_R3D_SO "libr3d.so"
#endif

namespace {

constexpr long kNumel = 4096;          /* fp32: 16 KiB a message, a whole number of 16-byte words */
constexpr int  kRanks = 2;

typedef int  (*CountFn)(void);
typedef const RadKernelInfo* (*AtFn)(int);
typedef int  (*LaunchFn)(const RadArgs*, RadStream);

struct Lib {
    void* h = nullptr;
    CountFn count = nullptr;
    AtFn at = nullptr;
    const RadKernelInfo* row(const char* name) {
        if (!h) return nullptr;
        const int n = count ? count() : 0;
        for (int i = 0; i < n; ++i) {
            const RadKernelInfo* k = at(i);
            if (k && k->name && !std::strcmp(k->name, name)) return k;
        }
        return nullptr;
    }
    bool open() {
        h = dlopen(RAD_R3D_SO, RTLD_NOW | RTLD_LOCAL);
        if (!h) return false;
        count = (CountFn)dlsym(h, "rad_kernel_count");
        at = (AtFn)dlsym(h, "rad_kernel_at");
        return count && at;
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

/* One rank's view of the run: its stream, its buffers, the instance init() returned and both
 * return codes, so the main thread can report them. */
struct Rank {
    int rank = 0;
    const RadKernelInfo* row = nullptr;
    hipStream_t stream = nullptr;
    float* x = nullptr;
    float* y = nullptr;
    void* inst = nullptr;
    int rc_init = 0;
    int rc_launch = 0;
};

/* Both ranks must leave the kernels before either frees its exchange buffers: a rank's own
 * scratch is the peer's read buffer, and a stream sync only waits for this rank's launch. */
std::atomic<int> g_arrived{0};

void run_rank(Rank* r) {
    (void)hipSetDevice(0);
    std::vector<RadParam> p = {
        RAD_INT("world_size", kRanks),
        RAD_INT("numel", kNumel),
        RAD_STR("dtype", "fp32"),
        RAD_INT("exact", 1),
        RAD_INT("nblocks", 4),
        RAD_INT("nthreads", 256),
        RAD_INT("drain", 3),
        RAD_INT("acq", 0),
    };
    const bool stream_ok =
        hipStreamCreateWithFlags(&r->stream, hipStreamNonBlocking) == hipSuccess;
    if (!stream_ok) r->rc_init = -1;
    if (stream_ok) {
        r->rc_init = r->row->init(p.data(), (int)p.size(), r->rank, kRanks, &r->inst);
    }
    if (r->rc_init == RAD_OK) {
        RadTensor t[2]{};
        t[0].data = r->x;
        t[0].dtype = RAD_F32;
        t[0].rank = 1;
        t[0].shape[0] = kNumel;
        rad_tensor_pack(&t[0]);
        if (r->y) {
            t[1].data = r->y;
            t[1].dtype = RAD_F32;
            t[1].rank = 1;
            t[1].shape[0] = kNumel * kRanks;
            rad_tensor_pack(&t[1]);
        }
        RadArgs a{};
        a.t = t;
        a.n_t = r->y ? 2 : 1;
        a.p = p.data();
        a.n_p = (int)p.size();
        a.instance = r->inst;
        a.rank = r->rank;
        a.world_size = kRanks;

        r->rc_launch = r->row->launch(&a, r->stream);
        if (r->rc_launch == 0 && hipStreamSynchronize(r->stream) != hipSuccess) r->rc_launch = -2;
    }
    /* Leave both launches before either frees: a rank's scratch is its peer's read buffer. */
    g_arrived.fetch_add(1);
    while (g_arrived.load() < kRanks) std::this_thread::yield();
    if (r->inst) r->row->fini(r->inst);
    if (r->stream) (void)hipStreamDestroy(r->stream);
}

/* Both ranks at once: init() rendezvouses inside (r3dh_ar_entry.hip) and waits for the peer, so a
 * serial start would hang on the first barrier. */
void run_both(Rank r[kRanks]) {
    g_arrived.store(0);
    std::thread t0(run_rank, &r[0]), t1(run_rank, &r[1]);
    t0.join();
    t1.join();
}

void fill_rank(float* d, int rank, bool gather) {
    std::vector<float> h(kNumel);
    for (long i = 0; i < kNumel; ++i) h[(size_t)i] = gather ? (float)(rank * 10000 + i) : (float)(rank + 1);
    if (hipMemcpy(d, h.data(), h.size() * sizeof(float), hipMemcpyHostToDevice) != hipSuccess)
        radtest::fail(__FILE__, __LINE__, "hipMemcpy H2D failed");
}

std::vector<float> read_back(const float* d, long n) {
    std::vector<float> h((size_t)n);
    if (hipMemcpy(h.data(), d, h.size() * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess)
        radtest::fail(__FILE__, __LINE__, "hipMemcpy D2H failed");
    return h;
}

void alloc_ranks(Rank r[kRanks], bool gather) {
    for (int i = 0; i < kRanks; ++i) {
        r[i].rank = i;
        if (hipMalloc((void**)&r[i].x, kNumel * sizeof(float)) != hipSuccess ||
            (gather && hipMalloc((void**)&r[i].y, kNumel * kRanks * sizeof(float)) != hipSuccess)) {
            radtest::fail(__FILE__, __LINE__, "hipMalloc failed");
            return;
        }
        fill_rank(r[i].x, i, gather);
    }
}

void free_ranks(Rank r[kRanks]) {
    for (int i = 0; i < kRanks; ++i) {
        if (r[i].x) (void)hipFree(r[i].x);
        if (r[i].y) (void)hipFree(r[i].y);
        r[i].x = r[i].y = nullptr;
    }
}

}  /* namespace */

TEST(all_reduce_two_ranks_one_card) {
    if (!ready()) return;
    const RadKernelInfo* row = g_lib.row("ar_oneshot_2rank_exact");
    if (!row) { radtest::artifact_missing(__FILE__, __LINE__, "ar_oneshot_2rank_exact absent"); return; }
    Rank r[kRanks];
    alloc_ranks(r, false);
    for (int i = 0; i < kRanks; ++i) r[i].row = row;
    run_both(r);
    CHECK_EQ(r[0].rc_init, (int)RAD_OK);
    CHECK_EQ(r[1].rc_init, (int)RAD_OK);
    CHECK_EQ(r[0].rc_launch, (int)RAD_OK);
    CHECK_EQ(r[1].rc_launch, (int)RAD_OK);
    for (int i = 0; i < kRanks; ++i) {
        const std::vector<float> got = read_back(r[i].x, kNumel);
        int bad = 0;
        for (float v : got) if (v != 3.0f) ++bad;
        CHECK_EQ(bad, 0);
        if (bad) std::printf("  rank %d: first mismatches %g\n", i, (double)got[0]);
    }
    free_ranks(r);
}

TEST(all_gather_two_ranks_one_card) {
    if (!ready()) return;
    const RadKernelInfo* row = g_lib.row("ar_gather_2rank");
    if (!row) { radtest::artifact_missing(__FILE__, __LINE__, "ar_gather_2rank absent"); return; }
    Rank r[kRanks];
    alloc_ranks(r, true);
    for (int i = 0; i < kRanks; ++i) r[i].row = row;
    run_both(r);
    CHECK_EQ(r[0].rc_init, (int)RAD_OK);
    CHECK_EQ(r[1].rc_init, (int)RAD_OK);
    CHECK_EQ(r[0].rc_launch, (int)RAD_OK);
    CHECK_EQ(r[1].rc_launch, (int)RAD_OK);
    for (int i = 0; i < kRanks; ++i) {
        const std::vector<float> got = read_back(r[i].y, kNumel * kRanks);
        int bad = 0;
        for (int src = 0; src < kRanks; ++src)
            for (long j = 0; j < kNumel; ++j)
                if (got[(size_t)(src * kNumel + j)] != (float)(src * 10000 + j)) ++bad;
        CHECK_EQ(bad, 0);
        if (bad) std::printf("  rank %d: %d gathered words wrong (y[0]=%g y[%ld]=%g)\n",
                             i, bad, (double)got[0], kNumel, (double)got[(size_t)kNumel]);
    }
    free_ranks(r);
}

RAD_TEST_MAIN()
