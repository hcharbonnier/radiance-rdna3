/* gather_rows_u8_device_test.cpp -- libr3d's opaque-byte row moves, which the KV tier issues.
 *
 * The host tier copies one layer's fragment of one block out of (and back into) a paged pool with
 * gather_rows/scatter_rows at a `dtype` of u8. The bytes are opaque: a KV fragment may hold any
 * pattern, including a NaN payload in a bf16 plane, and the move must be exact. The bf16/f32 rows
 * pass every element through ld_elem/st_elem, so they are not usable here; these cases hold the u8
 * rows to a byte-for-byte host oracle instead of libref, whose u8 path goes through float.
 *
 * Covered: the 4128-byte QSA fragment of a 4-token page, several M (1, 8, 2048), padded source
 * and destination pitches, every one of the 256 byte values, indices that gather a zero row
 * (-1 and at/over the source), and a scatter whose destination keeps the rows no index names. The
 * row's declared constraints and its operand description are checked too: that is what the tier's
 * resolve() reads when it asks for a u8 row at the fragment width.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include <dlfcn.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef RAD_R3D_SO
#define RAD_R3D_SO "libr3d.so"
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
    const RadKernelInfo* find(const char* op, int domain, const char* name) const {
        for (int i = 0; count && i < count(); ++i) {
            const RadKernelInfo* k = at(i);
            if (!k || !k->op || !k->launch || k->domain != domain) continue;
            if (std::strcmp(k->op, op) != 0) continue;
            if (name && std::strcmp(k->name, name) != 0) continue;
            return k;
        }
        return nullptr;
    }
};

Plugin g_r3d;

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
    if (!g_r3d.open(RAD_R3D_SO)) {
        std::fprintf(stderr, "skip: %s not loadable here\n", RAD_R3D_SO);
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
RadParam ps(const char* k, const char* v) { return RAD_STR(k, v); }

int run(const RadKernelInfo* k, std::vector<RadTensor>& t, std::vector<RadParam>& p) {
    RadArgs a;
    std::memset(&a, 0, sizeof a);
    a.t = t.data();
    a.n_t = (int)t.size();
    a.p = p.data();
    a.n_p = (int)p.size();
    const int rc = k->launch(&a, (RadStream)0);
    (void)hipDeviceSynchronize();
    return rc;
}

const RadConstraint* constraint(const RadKernelInfo* k, const char* key) {
    if (!k) return nullptr;
    for (int i = 0; i < k->n_constraints; ++i) {
        const RadConstraint* c = &k->constraints[i];
        if (c->key && std::strcmp(c->key, key) == 0) return c;
    }
    return nullptr;
}

/* A byte that depends on its position, so a row read from the wrong place or a misaddressed pitch
 * differs at the first byte rather than coincidentally matching. */
uint8_t pat(size_t row, size_t col) {
    return (uint8_t)((row * 197u + col * 37u + (col >> 3) * 11u) & 0xFFu);
}

struct Case {
    const char* name;
    int M;          /* rows moved */
    int n;          /* bytes a row */
    int src_rows;   /* rows in the gather's source / the scatter's destination */
    int src_pad;    /* source row pitch in bytes beyond n (0: dense) */
    int dst_pad;    /* destination row pitch in bytes beyond n (0: dense) */
    bool wild_idx;  /* include -1 and an index at/over the source */
};

bool check(const char* what, const std::vector<uint8_t>& got, const std::vector<uint8_t>& want) {
    for (size_t i = 0; i < got.size(); ++i) {
        if (got[i] == want[i]) continue;
        std::fprintf(stderr, "  %s: first difference at byte %zu (want %d, got %d)\n", what, i,
                     (int)want[i], (int)got[i]);
        return false;
    }
    return true;
}

bool run_gather(const Case& c) {
    const RadKernelInfo* k = g_r3d.find("gather_rows", RAD_DOMAIN_DEVICE, "gather_rows_u8");
    if (!k) {
        std::fprintf(stderr, "  %s: the row is absent\n", c.name);
        return false;
    }
    const int x_pad = c.n + c.src_pad, y_pad = c.n + c.dst_pad;

    std::vector<uint8_t> src((size_t)c.src_rows * (size_t)x_pad, 0);
    for (int r = 0; r < c.src_rows; ++r)
        for (int i = 0; i < c.n; ++i) src[(size_t)r * (size_t)x_pad + (size_t)i] = pat((size_t)r, (size_t)i);

    std::vector<int32_t> idx((size_t)c.M);
    for (int m = 0; m < c.M; ++m) idx[(size_t)m] = (int32_t)((m * 7 + 1) % c.src_rows);
    if (c.wild_idx && c.M >= 4) {
        idx[1] = -1;                 /* padding: the zero row */
        idx[c.M - 2] = c.src_rows;   /* at the end: the zero row, not a read past the plane */
    }

    std::vector<uint8_t> want((size_t)c.M * (size_t)y_pad, 0xA5);
    for (int m = 0; m < c.M; ++m) {
        uint8_t* dst = want.data() + (size_t)m * (size_t)y_pad;
        const int32_t r = idx[(size_t)m];
        if (r < 0 || r >= c.src_rows) {
            std::memset(dst, 0, (size_t)c.n);
            continue;
        }
        std::memcpy(dst, src.data() + (size_t)r * (size_t)x_pad, (size_t)c.n);
    }

    Dev dx((size_t)c.src_rows * (size_t)x_pad), di((size_t)c.M * sizeof(int32_t)),
        dy((size_t)c.M * (size_t)y_pad);
    up(dx, src.data(), src.size());
    up(di, idx.data(), idx.size() * sizeof(int32_t));
    std::vector<uint8_t> fill(dy.bytes, 0xA5);
    up(dy, fill.data(), fill.size());

    std::vector<RadParam> p = { pi("M", c.M), pi("n", c.n), ps("dtype", "u8") };
    RadTensor tx = ten(dx.p, RAD_U8, { c.src_rows, c.n });
    RadTensor ti = ten(di.p, RAD_I32, { c.M });
    RadTensor ty = ten(dy.p, RAD_U8, { c.M, c.n });
    tx.stride[0] = x_pad;
    ty.stride[0] = y_pad;
    std::vector<RadTensor> t = { tx, ti, ty };
    const int rc = run(k, t, p);
    if (rc != RAD_OK) {
        std::fprintf(stderr, "  %s: libr3d refused rc=%d\n", c.name, rc);
        return false;
    }
    std::vector<uint8_t> got((size_t)c.M * (size_t)y_pad);
    down(dy, got.data(), got.size());
    return check(c.name, got, want);
}

bool run_scatter(const Case& c) {
    const RadKernelInfo* k = g_r3d.find("scatter_rows", RAD_DOMAIN_DEVICE, "scatter_rows_u8");
    if (!k) {
        std::fprintf(stderr, "  %s: the row is absent\n", c.name);
        return false;
    }
    const int v_pad = c.n + c.src_pad, x_pad = c.n + c.dst_pad;

    std::vector<uint8_t> src((size_t)c.M * (size_t)v_pad, 0);
    for (int m = 0; m < c.M; ++m)
        for (int i = 0; i < c.n; ++i) src[(size_t)m * (size_t)v_pad + (size_t)i] = pat((size_t)m, (size_t)i);

    /* Distinct destinations over 4M rows, so a destination wider than the moved rows is the norm. */
    std::vector<int32_t> idx((size_t)c.M);
    for (int m = 0; m < c.M; ++m) idx[(size_t)m] = (int32_t)((m * 3 + 1) % (4 * c.M));
    if (c.wild_idx && c.M >= 4) {
        idx[1] = -1;              /* writes nothing */
        idx[c.M - 2] = 4 * c.M;   /* past the destination: dropped, never a write outside it */
    }

    std::vector<uint8_t> want((size_t)c.src_rows * (size_t)x_pad, 0x5C);
    for (int m = 0; m < c.M; ++m) {
        const int32_t r = idx[(size_t)m];
        if (r < 0 || r >= c.src_rows) continue;
        std::memcpy(want.data() + (size_t)r * (size_t)x_pad, src.data() + (size_t)m * (size_t)v_pad,
                    (size_t)c.n);
    }

    Dev dv((size_t)c.M * (size_t)v_pad), di((size_t)c.M * sizeof(int32_t)),
        dx((size_t)c.src_rows * (size_t)x_pad);
    up(dv, src.data(), src.size());
    up(di, idx.data(), idx.size() * sizeof(int32_t));
    std::vector<uint8_t> fill(dx.bytes, 0x5C);
    up(dx, fill.data(), fill.size());

    std::vector<RadParam> p = { pi("M", c.M), pi("n", c.n), ps("dtype", "u8") };
    RadTensor tv = ten(dv.p, RAD_U8, { c.M, c.n });
    RadTensor ti = ten(di.p, RAD_I32, { c.M });
    RadTensor tx = ten(dx.p, RAD_U8, { c.src_rows, c.n });
    tv.stride[0] = v_pad;
    tx.stride[0] = x_pad;
    std::vector<RadTensor> t = { tv, ti, tx };
    const int rc = run(k, t, p);
    if (rc != RAD_OK) {
        std::fprintf(stderr, "  %s: libr3d refused rc=%d\n", c.name, rc);
        return false;
    }
    std::vector<uint8_t> got((size_t)c.src_rows * (size_t)x_pad);
    down(dx, got.data(), got.size());
    return check(c.name, got, want);
}

TEST(gather_rows_u8_is_declared) {
    if (!ready()) return;
    const RadKernelInfo* g = g_r3d.find("gather_rows", RAD_DOMAIN_DEVICE, "gather_rows_u8");
    const RadKernelInfo* s = g_r3d.find("scatter_rows", RAD_DOMAIN_DEVICE, "scatter_rows_u8");
    CHECK(g != nullptr);
    CHECK(s != nullptr);
    if (!g || !s) return;

    for (const RadKernelInfo* k : { g, s }) {
        const RadConstraint* dt = constraint(k, "dtype");
        const RadConstraint* n = constraint(k, "n");
        CHECK(dt && dt->op == RAD_C_IN && dt->sval && std::strcmp(dt->sval, "u8") == 0);
        CHECK(n && n->op == RAD_C_DIV && n->ival == 4);
    }

    /* The tier's geometry: a 4128-byte QSA fragment. The description's operands are opaque u8, and
     * the source holds 4M rows so a gather cannot pass by being the identity. */
    const RadParam pp[] = { pi("M", 2048), pi("n", 4128), ps("dtype", "u8") };
    RadOpdDesc x, idx, y;
    CHECK_EQ(g->opd_shape(pp, 3, 0, &x), RAD_OK);
    CHECK_EQ(x.dtype, (uint32_t)RAD_U8);
    CHECK_EQ(x.rank, 2u);
    CHECK_EQ(x.shape[0], 4 * 2048);
    CHECK_EQ(x.shape[1], 4128);
    CHECK_EQ(g->opd_shape(pp, 3, 1, &idx), RAD_OK);
    CHECK_EQ(idx.rank, 1u);
    CHECK_EQ(idx.shape[0], 2048);
    CHECK_EQ(g->opd_shape(pp, 3, 2, &y), RAD_OK);
    CHECK_EQ(y.dtype, (uint32_t)RAD_U8);
    CHECK_EQ(y.shape[0], 2048);
    CHECK_EQ(y.shape[1], 4128);
    CHECK_EQ(s->opd_shape(pp, 3, 2, &y), RAD_OK);
    CHECK_EQ(y.shape[0], 4 * 2048);
    CHECK((constraint(s, "n") && constraint(s, "n")->ival == 4));
}

TEST(gather_rows_u8_moves_fragments_whole) {
    if (!ready()) return;
    const Case cases[] = {
        { "gather_M1_dense",       1, 4128, 8,  0,  0,  false },
        { "gather_M8_dense",       8, 4128, 32, 0,  0,  true  },
        { "gather_M8_pitched",     8, 4128, 32, 16, 32, true  },
        { "gather_M2048_band",     2048, 4128, 512, 16, 32, false },
        { "gather_M7_small_row",   7, 4,    9,  0,  4,  true  },
        { "gather_M3_256B",        3, 256,  5,  8,  8,  true  },
    };
    for (const Case& c : cases) CHECK(run_gather(c));
}

TEST(scatter_rows_u8_keeps_unnamed_rows) {
    if (!ready()) return;
    const Case cases[] = {
        { "scatter_M1_dense",      1, 4128, 4,   0,  0,  false },
        { "scatter_M8_dense",      8, 4128, 32,  0,  0,  true  },
        { "scatter_M8_pitched",    8, 4128, 32,  16, 32, true  },
        { "scatter_M2048_band",    2048, 4128, 8200, 16, 32, false },
        { "scatter_M7_small_row",  7, 4,    28,  0,  4,  true  },
        { "scatter_M3_256B",       3, 256,  12,  8,  8,  true  },
    };
    for (const Case& c : cases) CHECK(run_scatter(c));
}

}  // namespace

RAD_TEST_MAIN()
