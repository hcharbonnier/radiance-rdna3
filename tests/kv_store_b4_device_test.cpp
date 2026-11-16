/* kv_store_b4_device_test.cpp -- libr3d's 4-token paged int8 store against libref.
 *
 * Flash-Next's 13 attention layers page their int8 KV cache at block_size 4, head_dim 256 and
 * n_head_kv 2 (FLASH_NEXT_AUDIT.md §7.2, M 4096). The row that served the 27B's 16-token pages
 * constrained block_size to 16, so this row, `kv_store_i8_b4`, is the same store at the 4-token
 * geometry. The kernel already splits a slot by the block_size argument, so the bytes a slot holds
 * do not depend on the page size; the test holds both rows to libref at their own geometry.
 *
 * Every case runs libref's host `kv_store` and libr3d's device row on identical operands and
 * compares the whole cache plane byte for byte, sentinels included: mapped slots must carry the
 * same codes and f16 scales, unmapped slots and holes must survive untouched. The cases cover
 * several sequences interleaved through separate block ranges, partial pages at both ends, M not
 * a multiple of the page size and M = 4096, and non-tight pitches on k, v and the cache's three
 * strides (a column slice of a fused projection and a padded page plane).
 *
 * The row's operand description is checked at the 4-token geometry too: the cache operand is
 * [blocks, kv_heads, 4, 2*head_dim + 4] i8 and slot_mapping is the rank-1 [M] unique-index draw
 * bounded by blocks*4. A machine without the card or the plugins skips.
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

/* One geometry. `k_pad` and `v_pad` are the k/v row pitches in elements (0 means dense); the
 * cache strides are dense plus the three offsets, in bytes because the cache is i8. `slots` selects
 * the slot-mapping pattern and `values` the quantisation edges the operands exercise. */
struct Case {
    const char* name;
    int M, hd, kvh, bs, nb;
    int k_pad, v_pad;
    int so_off, sh_off, sb_off;
    int slots;
    int values;
    bool block16_row;      /* run the 16-token row (the 27B's unchanged band) */
};

std::vector<int> slot_map(const Case& c) {
    const int slots = c.nb * c.bs;
    std::vector<int> sm((size_t)c.M);
    if (c.slots == 1) {
        /* Four sequences interleaved through their own block ranges: each sees its own pages,
         * and the ranges leave holes between them so pages stay partial. */
        const int S = 4, per = slots / S;
        for (int t = 0; t < c.M; ++t) sm[(size_t)t] = (t % S) * per + t / S;
    } else if (c.slots == 2) {
        /* A step that adopts a prefix: its tail starts at the allocator's global slot bs, so the
         * first page is untouched. */
        for (int t = 0; t < c.M; ++t) sm[(size_t)t] = c.bs + t;
    } else {
        /* Dense from slot 0, with holes, so the last page is partial when M is not a multiple. */
        for (int t = 0; t < c.M; ++t) sm[(size_t)t] = t;
    }
    /* Padding rows: both libraries skip a negative slot and leave the target byte untouched. */
    for (int t = 0; t < c.M; ++t)
        if (t % 7 == 3) sm[(size_t)t] = -1;
    if (c.M > 1) sm[0] = -1;
    return sm;
}

float value_at(const Case& c, int t, int h, int d, bool is_v) {
    float x = std::sin(0.037f * (float)(t * 97 + h * 13 + d * 7 + (is_v ? 31 : 0)));
    if (c.values == 1) {
        /* Quantisation edges, placed apart so each head keeps its own scale:
         *  - a zero head stores scale 0 and codes 0;
         *  - an absmax above 515*127 clamps the scale at 515 and the code at 127;
         *  - an all-negative head keeps the sign;
         *  - a head with its maximum only in the d % 64 >= 32 half catches a 64-lane reduction. */
        const int t_zero = c.M / 4, t_huge = c.M / 2, t_neg = (3 * c.M) / 4, t_up = c.M / 3;
        if (!is_v && h == 0 && t == t_zero) return 0.0f;
        if (!is_v && h == 0 && t == t_up) {
            if (d == 32) return 1.0f;
            return d % 64 < 32 ? 0.01f : 0.02f;
        }
        if (!is_v && h == c.kvh - 1 && t == t_huge) return d == 17 ? 1.0e6f : 0.0f;
        if (!is_v && h == c.kvh - 1 && t == t_neg) return -0.25f - 0.001f * (float)d;
        if (is_v && h == c.kvh - 1 && t == 1) return 0.0f;
        if (is_v && h == 0 && t == c.M - 1) return d == 9 ? -2.0e6f : 0.0f;
    }
    return x;
}

bool run_case(const Case& c) {
    const RadKernelInfo* kd = g_r3d.find("kv_store", RAD_DOMAIN_DEVICE,
                                          c.block16_row ? "kv_store_i8" : "kv_store_i8_b4");
    const RadKernelInfo* kr = g_ref.find("kv_store", RAD_DOMAIN_HOST, "ref_kv_store");
    if (!kd || !kr) {
        std::fprintf(stderr, "  %s: a row is absent (libr3d %p, libref %p)\n", c.name,
                     (const void*)kd, (const void*)kr);
        return false;
    }

    const int width = c.kvh * c.hd;
    const int k_pad = c.k_pad > width ? c.k_pad : width;
    const int v_pad = c.v_pad > width ? c.v_pad : width;
    const int rowe = 2 * c.hd + 4;
    const int so = rowe + c.so_off;
    const int sh = c.bs * so + c.sh_off;
    const int sb = c.kvh * sh + c.sb_off;
    const size_t cache_bytes = (size_t)(c.nb - 1) * (size_t)sb +
                               (size_t)(c.kvh - 1) * (size_t)sh +
                               (size_t)(c.bs - 1) * (size_t)so + (size_t)rowe;

    const uint16_t pad_sentinel = f32_to_bf16(12345.0f);
    std::vector<uint16_t> kh((size_t)c.M * (size_t)k_pad, pad_sentinel);
    std::vector<uint16_t> vh((size_t)c.M * (size_t)v_pad, pad_sentinel);
    for (int t = 0; t < c.M; ++t)
        for (int h = 0; h < c.kvh; ++h)
            for (int d = 0; d < c.hd; ++d) {
                kh[(size_t)t * (size_t)k_pad + (size_t)h * (size_t)c.hd + (size_t)d] =
                    f32_to_bf16(value_at(c, t, h, d, false));
                vh[(size_t)t * (size_t)v_pad + (size_t)h * (size_t)c.hd + (size_t)d] =
                    f32_to_bf16(value_at(c, t, h, d, true));
            }

    const std::vector<int> sm = slot_map(c);
    std::vector<RadParam> params = {
        pi("M", c.M), pi("head_dim", c.hd), pi("n_head_kv", c.kvh),
        pi("block_size", c.bs), ps("kv_dtype", "i8")
    };

    const uint8_t cache_fill = 0x5a;
    std::vector<uint8_t> want(cache_bytes, cache_fill), got(cache_bytes, cache_fill);

    /* libref: host pointers straight into the vectors above, with the same non-tight pitches. */
    {
        std::vector<RadTensor> t = {
            ten(kh.data(), RAD_BF16, { c.M, c.kvh, c.hd }),
            ten(vh.data(), RAD_BF16, { c.M, c.kvh, c.hd }),
            ten((void*)sm.data(), RAD_I32, { c.M }),
            ten(want.data(), RAD_I8, { c.nb, c.kvh, c.bs, rowe })
        };
        t[0].stride[0] = k_pad;
        t[1].stride[0] = v_pad;
        t[3].stride[0] = sb;
        t[3].stride[1] = sh;
        t[3].stride[2] = so;
        const int rc = run(kr, t, params);
        if (rc != RAD_OK) {
            std::fprintf(stderr, "  %s: libref refused rc=%d\n", c.name, rc);
            return false;
        }
    }

    /* libr3d: the same tensors over device copies. */
    {
        Dev dk(kh.size() * 2), dv(vh.size() * 2), dsm(sm.size() * 4), dkv(cache_bytes);
        up(dk, kh.data(), dk.bytes);
        up(dv, vh.data(), dv.bytes);
        up(dsm, sm.data(), dsm.bytes);
        up(dkv, got.data(), dkv.bytes);
        std::vector<RadTensor> t = {
            ten(dk.p, RAD_BF16, { c.M, c.kvh, c.hd }),
            ten(dv.p, RAD_BF16, { c.M, c.kvh, c.hd }),
            ten(dsm.p, RAD_I32, { c.M }),
            ten(dkv.p, RAD_I8, { c.nb, c.kvh, c.bs, rowe })
        };
        t[0].stride[0] = k_pad;
        t[1].stride[0] = v_pad;
        t[3].stride[0] = sb;
        t[3].stride[1] = sh;
        t[3].stride[2] = so;
        const int rc = run(kd, t, params);
        if (rc != RAD_OK) {
            std::fprintf(stderr, "  %s: libr3d refused rc=%d\n", c.name, rc);
            return false;
        }
        down(dkv, got.data(), dkv.bytes);
    }

    size_t first = got.size();
    for (size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i]) { first = i; break; }
    if (first == got.size()) {
        std::printf("  %-26s M=%4d bs=%2d kvh=%d nb=%4d bytes=%zu identical\n", c.name, c.M,
                    c.bs, c.kvh, c.nb, got.size());
        return true;
    }
    const size_t within = first % (size_t)sb;
    const size_t block = first / (size_t)sb;
    const size_t in_block = within % (size_t)sh;
    const size_t head = in_block / (size_t)so;
    std::fprintf(stderr,
                 "  %s: first difference byte %zu (block %zu, head %zu, slot %zu, +%zu): "
                 "libr3d %d, libref %d\n", c.name, first, block, head,
                 (in_block % (size_t)so) / (size_t)rowe, within % (size_t)rowe, (int)got[first],
                 (int)want[first]);
    return false;
}

const RadConstraint* constraint(const RadKernelInfo* k, const char* key) {
    if (!k) return nullptr;
    for (int i = 0; i < k->n_constraints; ++i) {
        const RadConstraint* c = &k->constraints[i];
        if (c->key && std::strcmp(c->key, key) == 0) return c;
    }
    return nullptr;
}

TEST(kv_store_i8_b4_row_is_declared) {
    if (!ready()) return;
    const RadKernelInfo* b4 = g_r3d.find("kv_store", RAD_DOMAIN_DEVICE, "kv_store_i8_b4");
    const RadKernelInfo* b16 = g_r3d.find("kv_store", RAD_DOMAIN_DEVICE, "kv_store_i8");
    CHECK(b4 != nullptr);
    CHECK(b16 != nullptr);
    if (!b4 || !b16) return;

    /* The new band is exactly the 4-token one; the 16-token row keeps its own constraint. */
    const RadConstraint* c4 = constraint(b4, "block_size");
    const RadConstraint* c16 = constraint(b16, "block_size");
    CHECK(c4 && c4->op == RAD_C_EQ && c4->ival == 4);
    CHECK(c16 && c16->op == RAD_C_EQ && c16->ival == 16);
    CHECK(constraint(b4, "head_dim") && constraint(b4, "head_dim")->ival == 256);
    CHECK(constraint(b4, "kv_dtype") && constraint(b4, "kv_dtype")->op == RAD_C_IN);

    /* The operand description the block manager sizes a 4-token pool from. M = 4096 and two KV
     * heads are the audit's geometry: 1026 blocks of 4 slots, 2 * 256 + 4 bytes a slot. */
    const RadParam pp[] = { pi("M", 4096), pi("head_dim", 256), pi("n_head_kv", 2),
                            pi("block_size", 4), ps("kv_dtype", "i8") };
    CHECK(b4->opd_shape != nullptr);
    if (!b4->opd_shape) return;
    RadOpdDesc cache, slots, k;
    CHECK_EQ(b4->opd_shape(pp, 5, 3, &cache), RAD_OK);
    CHECK_EQ(cache.dtype, (uint32_t)RAD_I8);
    CHECK_EQ(cache.rank, 4u);
    CHECK_EQ(cache.shape[0], 1026);
    CHECK_EQ(cache.shape[1], 2);
    CHECK_EQ(cache.shape[2], 4);
    CHECK_EQ(cache.shape[3], 516);
    CHECK_EQ(b4->opd_shape(pp, 5, 2, &slots), RAD_OK);
    CHECK_EQ(slots.rank, 1u);
    CHECK_EQ(slots.shape[0], 4096);
    CHECK_EQ(slots.idx_max, 4104);
    CHECK((slots.flags & RAD_OPD_F_IDX_UNIQUE) != 0);
    CHECK_EQ(b4->opd_shape(pp, 5, 0, &k), RAD_OK);
    CHECK_EQ(k.dtype, (uint32_t)RAD_BF16);
    CHECK_EQ(k.shape[0], 4096);
    CHECK_EQ(k.shape[1], 2);
    CHECK_EQ(k.shape[2], 256);
}

TEST(kv_store_block4_matches_libref) {
    if (!ready()) return;
    const Case cases[] = {
        /* M variable around whole pages and the 4096-token audit shape. */
        { "M1_b4", 1, 256, 2, 4, 3, 0, 0, 0, 0, 0, 0, 0, false },
        { "M3_partial", 3, 256, 2, 4, 3, 0, 0, 0, 0, 0, 0, 0, false },
        { "M4_exact_page", 4, 256, 2, 4, 3, 0, 0, 0, 0, 0, 0, 0, false },
        { "M5_one_over", 5, 256, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, false },
        { "M7_partial", 7, 256, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, false },
        { "M16_four_pages", 16, 256, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, false },
        { "M17_one_over", 17, 256, 2, 4, 7, 0, 0, 0, 0, 0, 0, 0, false },
        { "M33_partial", 33, 256, 2, 4, 11, 0, 0, 0, 0, 0, 0, 0, false },
        { "M255_partial", 255, 256, 2, 4, 66, 0, 0, 0, 0, 0, 0, 0, false },
        { "M4096_audit", 4096, 256, 2, 4, 1026, 0, 0, 0, 0, 0, 0, 0, false },
        { "M4093_audit_partial", 4093, 256, 2, 4, 1026, 0, 0, 0, 0, 0, 0, 0, false },
        /* Several sequences, each in its own block range, with holes and partial pages. */
        { "seqs_M25_b4", 25, 256, 2, 4, 9, 0, 0, 0, 0, 0, 1, 1, false },
        { "seqs_M4096_b4", 4096, 256, 2, 4, 1026, 11, 13, 0, 0, 0, 1, 1, false },
        /* A tail written at the allocator's global slots; the first page stays untouched. */
        { "adopted_tail_b4", 37, 256, 2, 4, 12, 0, 0, 0, 0, 0, 2, 1, false },
        /* Non-tight k/v pitches and cache strides: a fused-projection slice and a padded page. */
        { "pitched_b4", 17, 256, 2, 4, 7, 280, 273, 8, 16, 32, 0, 1, false },
        { "pitched_seq_b4", 129, 256, 2, 4, 35, 264, 280, 4, 8, 16, 1, 1, false },
        /* The 16-token row is untouched: one canonical case plus a pitched one. */
        { "M4096_b16", 4096, 256, 2, 16, 258, 0, 0, 0, 0, 0, 0, true },
        { "pitched_b16", 33, 256, 2, 16, 5, 288, 272, 8, 16, 32, 1, 1, true },
    };
    for (const Case& c : cases) CHECK(run_case(c));
}

}  // namespace

RAD_TEST_MAIN()
