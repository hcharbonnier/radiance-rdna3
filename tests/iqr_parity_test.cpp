/* iqr_parity_test.cpp -- libr3d's IQR storage hooks (libr3d/r3d_iqr_layout.cpp) against libr4d's
 * (libr4d/r4d_iqr_layout.cpp) on the same container planes: the stored bytes must be identical
 * (ADR-1), so a fixture recorded against one library replays on the other, and both inverses must
 * give the planes back.
 *
 * Both libraries' layout sources are host-only C++ and are compiled into this test; neither
 * library includes the other.
 */
#include "rad_test.h"
#include "rad_plugin.h"
#include "rad_encoding.h"
#include "r4d_iqr.h"
#include "r3d_iqr.h"

#include <cstring>
#include <vector>

extern "C" int r4d_rad_layout_iqr(const RadParam*, int, int, const RadEncoding*, const int*,
                                  const RadTensor*, int, RadLayout*);
extern "C" int r4d_rad_relayout_iqr(const RadParam*, int, int, const RadEncoding*, const int*,
                                    const RadTensor*, int, void*, int64_t);
extern "C" int r4d_rad_unrelayout_iqr(const RadParam*, int, int, const RadEncoding*, const int*,
                                      const void*, int64_t, const RadTensor*, int);
extern "C" int r3d_rad_layout_iqr(const RadParam*, int, int, const RadEncoding*, const int*,
                                  const RadTensor*, int, RadLayout*);
extern "C" int r3d_rad_relayout_iqr(const RadParam*, int, int, const RadEncoding*, const int*,
                                    const RadTensor*, int, void*, int64_t);
extern "C" int r3d_rad_unrelayout_iqr(const RadParam*, int, int, const RadEncoding*, const int*,
                                      const void*, int64_t, const RadTensor*, int);

namespace {

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

uint64_t g_z = 0x243F6A8885A308D3ull;
uint64_t rnd() {
    g_z += 0x9E3779B97F4A7C15ull;
    uint64_t x = g_z;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

const uint32_t kU[] = { 0, RAD_U1, RAD_U2, RAD_U3, RAD_U4, RAD_U5, RAD_U6, RAD_U7,
                        RAD_U8, RAD_U9, RAD_U10, RAD_U11, RAD_U12 };

struct W {
    RadEncoding enc{};
    std::vector<std::vector<uint8_t>> data;
    std::vector<int64_t> rows, cols;
    int find(const char* r) const { return rad_enc_find(&enc, r); }
    RadTensor t(int i) {
        RadTensor x;
        std::memset(&x, 0, sizeof x);
        x.data = data[(size_t)i].data();
        x.dtype = enc.plane[i].dtype;
        x.rank = 2;
        x.shape[0] = rows[(size_t)i];
        x.shape[1] = cols[(size_t)i];
        rad_tensor_pack(&x);
        return x;
    }
    int64_t row_bytes(int i) const { return rad_enc_row_bytes(enc.plane[i].dtype, cols[(size_t)i]); }
    void size(int64_t n, int64_t k) {
        data.assign((size_t)enc.n_planes, {});
        rows.assign((size_t)enc.n_planes, 0);
        cols.assign((size_t)enc.n_planes, 0);
        for (int i = 0; i < enc.n_planes; ++i) {
            rad_enc_plane_dims(&enc.plane[i], n, k, &rows[(size_t)i], &cols[(size_t)i]);
            data[(size_t)i].assign((size_t)(rows[(size_t)i] * row_bytes(i)), 0);
        }
    }
};

using r4d_iqr::is_trellis;

struct Case { uint32_t kind; int B, V, L, RB; bool rot; };

W make(const Case& c, int64_t N, int64_t K) {
    W w;
    const bool tr = is_trellis(c.kind);
    if (tr) {
        rad_enc_clear(&w.enc);
        rad_enc_copy_str(w.enc.scheme, "trellis");
        rad_enc_add_plane(&w.enc, "codes", kU[c.B], 1, 1);
        rad_enc_add_plane(&w.enc, "scale", RAD_F8E4M3, 1, 128);
        rad_enc_add_plane(&w.enc, "scale.1", RAD_F32, 0, 0);
        const int t = rad_enc_add_table(&w.enc, "trellis", RAD_I8, (int64_t)1 << c.L, 1);
        w.enc.plane[t].block[0] = 256;
        w.enc.plane[t].block[1] = c.RB ? 32 : 0;
        if (c.kind == r4d_iqr::kTrellisMad) rad_enc_add_table(&w.enc, "trellis.mad", RAD_U32, 2, 1);
    } else {
        w.enc = rad_enc_affine(kU[c.B], RAD_BF16, 1, 128);
        if (c.kind == r4d_iqr::kGrid) {
            w.enc.plane[0].block[1] = c.V;
            rad_enc_add_table(&w.enc, "grid", RAD_I8, (int64_t)1 << c.B, c.V);
        } else if (c.kind == r4d_iqr::kTable) {
            rad_enc_add_table(&w.enc, "table", RAD_I8, 1, (int64_t)1 << c.B);
        } else if (c.kind == r4d_iqr::kZero) {
            rad_enc_add_plane(&w.enc, "zero", RAD_U6, 1, 128);
        }
    }
    if (c.rot) rad_enc_copy_str(w.enc.transform, "fwht128");
    w.size(N, K);
    const int ic = w.find("codes");
    for (auto& b : w.data[(size_t)ic]) b = (uint8_t)rnd();
    const int ia = tr ? w.find("trellis")
                 : c.kind == r4d_iqr::kGrid ? w.find("grid")
                 : c.kind == r4d_iqr::kTable ? w.find("table")
                 : c.kind == r4d_iqr::kZero ? w.find("zero") : -1;
    if (c.kind == r4d_iqr::kZero) {
        for (int64_t r = 0; r < N; ++r)
            for (int64_t kb = 0; kb < K / 128; ++kb)
                rad_store_bits(w.data[(size_t)ia].data() + r * w.row_bytes(ia), 6, kb,
                               (uint32_t)(rnd() & 63u));
    } else if (c.kind == r4d_iqr::kTrellisMad) {
        const uint32_t A = (uint32_t)(rnd() & 0xFFFFFFu) | 1u, Bc = (uint32_t)rnd();
        uint32_t* ab = (uint32_t*)w.data[(size_t)w.find("trellis.mad")].data();
        ab[0] = A;
        ab[1] = Bc;
        for (uint32_t st = 0; st < (1u << c.L); ++st) {
            const uint32_t h = ((st & 0xFFFFFFu) * A + Bc) & 0x3F3F3F3Fu;
            w.data[(size_t)ia][st] = (uint8_t)(int8_t)((int)((h & 0xFFu) + ((h >> 8) & 0xFFu) +
                                                             ((h >> 16) & 0xFFu) + (h >> 24)) - 126);
        }
    } else if (ia >= 0) {
        for (auto& b : w.data[(size_t)ia]) b = (uint8_t)(int8_t)((int)(rnd() % 255) - 127);
    }
    return w;
}

const char* aux_role(const Case& c) {
    return is_trellis(c.kind) ? "trellis"
         : c.kind == r4d_iqr::kGrid ? "grid"
         : c.kind == r4d_iqr::kTable ? "table"
         : c.kind == r4d_iqr::kZero ? "zero" : nullptr;
}

struct Sel {
    std::vector<int> sel;
    std::vector<RadTensor> t;
};
Sel select(W& w, const Case& c) {
    Sel s;
    std::vector<const char*> roles = { "codes" };
    if (const char* a = aux_role(c)) roles.push_back(a);
    if (c.kind == r4d_iqr::kTrellisMad) roles.push_back("trellis.mad");
    for (const char* r : roles) {
        s.sel.push_back(w.find(r));
        s.t.push_back(w.t(s.sel.back()));
    }
    return s;
}

std::vector<RadParam> params(int64_t N, int64_t K, bool rot) {
    return { pint("N", N), pint("K", K), pint("group", 128), pstr("dtype", rot ? "i8a8h" : "i8a8") };
}

/* The form the case asks for, as r3d_iqr.h spells the word; -1 when libr3d does not build it. */
int64_t form_word(const Case& c) {
    if (c.kind == r4d_iqr::kTrellisMad)
        return (int64_t)r3d_iqr::word(r3d_iqr::Form{ r3d_iqr::kTrellisMad, (uint32_t)c.B, 0,
                                                     (uint32_t)c.L, (uint32_t)(c.RB ? 5 : 0),
                                                     r3d_iqr::kE4m3Tensor, 0, 0 });
    if (c.kind == r4d_iqr::kTrellis)
        return (int64_t)r3d_iqr::word(r3d_iqr::Form{ r3d_iqr::kTrellis, (uint32_t)c.B, 0,
                                                     (uint32_t)c.L, (uint32_t)(c.RB ? 5 : 0),
                                                     r3d_iqr::kE4m3Tensor, 0, 0 });
    if (c.kind == r4d_iqr::kTable)
        return (int64_t)r3d_iqr::word(r3d_iqr::Form{ r3d_iqr::kTable, (uint32_t)c.B, 0, 0, 0,
                                                     r3d_iqr::kBf16, 0, 0 });
    if (c.kind == r4d_iqr::kGrid)
        return (int64_t)r3d_iqr::word(r3d_iqr::Form{ r3d_iqr::kGrid, (uint32_t)c.B,
                                                     (uint32_t)(c.V == 4 ? 2 : 1), 0, 0,
                                                     r3d_iqr::kBf16, 0, 0 });
    if (c.kind == r4d_iqr::kZero)
        return (int64_t)r3d_iqr::word(r3d_iqr::Form{ r3d_iqr::kZero, (uint32_t)c.B, 0, 0, 0,
                                                     r3d_iqr::kBf16, 0, 0 });
    return -1;
}

void check(const Case& c, int64_t N, int64_t K) {
    W w = make(c, N, K);
    Sel s = select(w, c);
    const std::vector<RadParam> lp = params(N, K, c.rot);

    RadLayout L4, L3;
    std::memset(&L4, 0, sizeof L4);
    std::memset(&L3, 0, sizeof L3);
    REQUIRE_EQ(r4d_rad_layout_iqr(lp.data(), (int)lp.size(), 2, &w.enc, s.sel.data(), s.t.data(),
                                  (int)s.t.size(), &L4), RAD_OK);
    const int w3 = (int)form_word(c);
    if (w3 < 0) return;
    REQUIRE_EQ(r3d_rad_layout_iqr(lp.data(), (int)lp.size(), 2, &w.enc, s.sel.data(), s.t.data(),
                                  (int)s.t.size(), &L3), RAD_OK);
    CHECK_EQ(L3.bytes, L4.bytes);
    CHECK_EQ(r3d_iqr::kernel_word(L3.dtype), (uint32_t)w3);
    CHECK_EQ(r3d_iqr::kernel_word(L4.dtype), (uint32_t)w3);

    std::vector<uint8_t> d4((size_t)L4.bytes, 0xA5), d3((size_t)L3.bytes, 0x5A);
    REQUIRE_EQ(r4d_rad_relayout_iqr(lp.data(), (int)lp.size(), 2, &w.enc, s.sel.data(), s.t.data(),
                                    (int)s.t.size(), d4.data(), L4.bytes), RAD_OK);
    REQUIRE_EQ(r3d_rad_relayout_iqr(lp.data(), (int)lp.size(), 2, &w.enc, s.sel.data(), s.t.data(),
                                    (int)s.t.size(), d3.data(), L3.bytes), RAD_OK);
    size_t first = d3.size();
    for (size_t i = 0; i < d3.size(); ++i)
        if (d3[i] != d4[i]) { first = i; break; }
    if (first != d3.size())
        std::fprintf(stderr, "kind %u B=%d V=%d L=%d RB=%d: first differing byte %zu of %zu\n",
                     c.kind, c.B, c.V, c.L, c.RB, first, d3.size());
    CHECK_EQ(first, d3.size());

    /* both inverses give the planes back */
    W back = w;
    for (auto& p : back.data) std::fill(p.begin(), p.end(), (uint8_t)0x5A);
    Sel sb = select(back, c);
    REQUIRE_EQ(r3d_rad_unrelayout_iqr(lp.data(), (int)lp.size(), 2, &back.enc, sb.sel.data(),
                                      d3.data(), L3.bytes, sb.t.data(), (int)sb.t.size()), RAD_OK);
    for (size_t i = 0; i < s.sel.size(); ++i) CHECK(back.data[(size_t)s.sel[i]] == w.data[(size_t)s.sel[i]]);
}

const Case kForms[] = {
    { r4d_iqr::kTable, 2, 1, 0, 0, false }, { r4d_iqr::kTable, 3, 1, 0, 0, false },
    { r4d_iqr::kTable, 4, 1, 0, 0, true },  { r4d_iqr::kTable, 5, 1, 0, 0, false },
    { r4d_iqr::kGrid, 8, 4, 0, 0, false },  { r4d_iqr::kGrid, 9, 4, 0, 0, false },
    { r4d_iqr::kGrid, 10, 4, 0, 0, true },  { r4d_iqr::kGrid, 12, 4, 0, 0, false },
    { r4d_iqr::kZero, 6, 1, 0, 0, false },
    { r4d_iqr::kTrellis, 2, 1, 12, 0, true }, { r4d_iqr::kTrellis, 3, 1, 12, 0, true },
    { r4d_iqr::kTrellis, 5, 1, 12, 0, false }, { r4d_iqr::kTrellis, 7, 1, 12, 0, true },
    { r4d_iqr::kTrellis, 4, 1, 14, 1, true }, { r4d_iqr::kTrellis, 6, 1, 14, 1, false },
    { r4d_iqr::kTrellisMad, 2, 1, 16, 0, true }, { r4d_iqr::kTrellisMad, 3, 1, 16, 0, true },
    { r4d_iqr::kTrellisMad, 4, 1, 16, 0, true }, { r4d_iqr::kTrellisMad, 5, 1, 16, 0, true },
    { r4d_iqr::kTrellisMad, 6, 1, 16, 0, true }, { r4d_iqr::kTrellisMad, 7, 1, 16, 0, true },
    { r4d_iqr::kTrellisMad, 4, 1, 16, 1, true }, { r4d_iqr::kTrellisMad, 5, 1, 16, 1, true },
};

}  // namespace

TEST(iqr_layout_bytes_match_libr4d_on_every_form_both_build) {
    for (const Case& c : kForms) check(c, 512, 384);
}

TEST(iqr_layout_bytes_match_libr4d_at_a_wide_shape) {
    check({ r4d_iqr::kTrellisMad, 4, 1, 16, 0, true }, 256, 1024);
    check({ r4d_iqr::kTrellisMad, 5, 1, 16, 1, true }, 512, 384);
}

RAD_TEST_MAIN()
