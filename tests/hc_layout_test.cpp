/* hc_layout_test.cpp -- libr3d's joined E4M3 mixing weight, on the host.
 *
 * The row packs each weight row as [codes K][f32 scales K/G][pad to 16B]. The down matrix's group
 * is 128 of hc*n; the up matrix's is lowrank/4. This checks the geometry, the tag, the byte total,
 * the packing and the inverse, with no card: the hooks are host functions, so this is the part of
 * W8's e4m3 form that can be proven without a GPU.
 */
#include "rad_test.h"
#include "rad_plugin.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <vector>

#ifndef RAD_R3D_SO
#define RAD_R3D_SO "libr3d.so"
#endif

namespace {

typedef int (*LayoutFn)(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                        int, RadLayout*);
typedef int (*RelayoutFn)(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*,
                          int, void*, int64_t);
typedef int (*UnrelayoutFn)(const RadParam*, int, int, const RadEncoding*, const int*, const void*,
                            int64_t, const RadTensor*, int);

struct Hooks {
    LayoutFn layout = nullptr;
    RelayoutFn relayout = nullptr;
    UnrelayoutFn unrelayout = nullptr;
    bool ok() const { return layout && relayout && unrelayout; }
};

Hooks load() {
    Hooks h;
    void* so = dlopen(RAD_R3D_SO, RTLD_NOW | RTLD_LOCAL);
    if (!so) {
        std::fprintf(stderr, "skip: %s not loadable\n", RAD_R3D_SO);
        return h;
    }
    h.layout = (LayoutFn)dlsym(so, "r3d_rad_layout_hc_mix");
    h.relayout = (RelayoutFn)dlsym(so, "r3d_rad_relayout_hc_mix");
    h.unrelayout = (UnrelayoutFn)dlsym(so, "r3d_rad_unrelayout_hc_mix");
    return h;
}

RadTensor mat(void* p, uint32_t dtype, long long rows, long long cols) {
    RadTensor t;
    std::memset(&t, 0, sizeof t);
    t.data = p;
    t.dtype = dtype;
    t.rank = 2;
    t.shape[0] = rows;
    t.shape[1] = cols;
    rad_tensor_pack(&t);
    return t;
}

bool check_operand(const Hooks& h, int operand, long long n, long long hc, long long lr,
                   long long rows, long long K, long long G) {
    RadParam p[] = { RAD_INT("n", n), RAD_INT("hc", hc), RAD_INT("lowrank", lr) };
    RadEncoding enc;
    std::memset(&enc, 0, sizeof enc);
    std::strcpy(enc.scheme, "affine");
    enc.n_planes = 2;
    std::strcpy(enc.plane[0].role, "codes");
    enc.plane[0].dtype = RAD_F8E4M3;
    enc.plane[0].kind = RAD_PLANE_TILED;
    enc.plane[0].block[0] = 1;
    enc.plane[0].block[1] = 1;
    std::strcpy(enc.plane[1].role, "scale");
    enc.plane[1].dtype = RAD_F32;
    enc.plane[1].kind = RAD_PLANE_TILED;
    enc.plane[1].block[0] = 1;
    enc.plane[1].block[1] = G;
    const int sel[2] = { 0, 1 };

    std::vector<uint8_t> codes((size_t)(rows * K));
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = (uint8_t)(i % 127);
    std::vector<float> scales((size_t)(rows * (K / G)));
    for (size_t i = 0; i < scales.size(); ++i) scales[i] = 0.25f * (float)(i + 1);
    RadTensor pl[2] = { mat(codes.data(), RAD_F8E4M3, rows, K),
                        mat(scales.data(), RAD_F32, rows, K / G) };

    RadLayout out;
    std::memset(&out, 0, sizeof out);
    CHECK_EQ(h.layout(p, 3, operand, &enc, sel, pl, 2, &out), 0);
    const long long stride = ((K + 4 * (K / G)) + 15) / 16 * 16;
    CHECK_EQ(out.rank, 2u);
    CHECK_EQ(out.shape[0], rows);
    CHECK_EQ(out.shape[1], stride);
    CHECK_EQ(out.bytes, rows * stride);
    CHECK_EQ(out.dtype, (uint32_t)RAD_F8E4M3);
    CHECK(std::strcmp(out.tag, G == 128 ? "r4d.hc.e4m3.g128" : "r4d.hc.e4m3.q4") == 0);

    std::vector<uint8_t> packed((size_t)out.bytes, 0xAA);
    CHECK_EQ(h.relayout(p, 3, operand, &enc, sel, pl, 2, packed.data(),
                        (int64_t)packed.size()), 0);
    const long long used = K + 4 * (K / G);
    bool bytes_ok = true;
    for (long long r = 0; r < rows; ++r) {
        const uint8_t* row = packed.data() + r * stride;
        if (std::memcmp(row, codes.data() + r * K, (size_t)K) != 0) bytes_ok = false;
        if (std::memcmp(row + K, scales.data() + r * (K / G), (size_t)(4 * (K / G))) != 0)
            bytes_ok = false;
        for (long long i = used; i < stride; ++i)
            if (row[i] != 0) bytes_ok = false;
    }
    CHECK(bytes_ok);

    std::vector<uint8_t> codes2(codes.size(), 0);
    std::vector<float> scales2(scales.size(), 0.0f);
    RadTensor back[2] = { mat(codes2.data(), RAD_F8E4M3, rows, K),
                          mat(scales2.data(), RAD_F32, rows, K / G) };
    CHECK_EQ(h.unrelayout(p, 3, operand, &enc, sel, packed.data(), (int64_t)packed.size(), back,
                          2), 0);
    CHECK(codes2 == codes);
    CHECK(std::memcmp(scales2.data(), scales.data(), scales.size() * 4) == 0);
    return true;
}

}  // namespace

TEST(hc_e4m3_mix_packs_codes_and_scales_in_the_row_tail) {
    const Hooks h = load();
    if (!h.ok()) return;
    /* Qwen4-Exp's geometry: n 2560, hc 4, lowrank 320. Down is [320, 10240] with a scale per 128;
     * up is [10240, 320] with a scale per 80. A smaller pair covers both branches' arithmetic. */
    CHECK(check_operand(h, 2, 64, 2, 64, 64, 128, 128));
    CHECK(check_operand(h, 3, 64, 2, 64, 128, 64, 16));
}

RAD_TEST_MAIN()
