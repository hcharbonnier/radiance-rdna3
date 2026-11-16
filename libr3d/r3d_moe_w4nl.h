/* r3d_moe_w4nl.h -- the `w4nl64a8i8h` expert form's decode constants (docs/MOE-W4.md).
 *
 * The container stores a codebook expert as four canonical planes:
 *
 *   codes   u4 [1 x 1]        one nibble a weight, row-major packed, two a byte
 *   scale   fp8_e4m3 [1 x 64] one scale a (row, 64 of K)
 *   scale.1 f32  [* x *]      one fixed level for the whole weight (2^-13)
 *   table   f32  [1 x 16]     the w4nl levels, magnitudes ascending then their negatives
 *
 * and a weight element is `table[code] * scale * scale.1`. The GEMM reads the codes and the table
 * as one joined operand and the two scale planes as another (abi/rad_planes.h), which the row's
 * layout hook asks for. The `w4nl64a8i8h` name is that stored form met by an int8 activation.
 *
 * `kW4NL` is this form's tag. It is deliberately not an `r3d_iqr::Kind`: the w4nl experts are an
 * `affine` encoding with a table, not an `R3D_DT_IQR_W | word` plane, and they are not in
 * `R3D_IQR_FORMS`. The tag lets the row table and the tests name the form without reusing an IQR
 * kind value.
 *
 * ---- the exact integer grid -------------------------------------------------------------------
 *
 * Every entry of the shipped table is an exact multiple of 1/8 (1.125 = 9/8 ... 22 = 176/8), so the
 * row decodes a code to `table8[code] = round(table[code] * 8)` and folds 1/8 into the scale. The
 * group sub-sum is then an exact i32, while the reference's f32 sum of `a * table[code]` is exact
 * for the same reason: its partial sums are multiples of 1/8 below 2^18, which f32 represents
 * exactly. `sub * sa * sb` with `sub = 8 * sub_ref` and `sb = sb_ref / 8` rounds identically,
 * because scaling by a power of two is exact. That is what makes the device result bit-exact
 * against libref's `quant_dot` chain rather than merely close.
 *
 * Pure C++ (no HIP), shared by the kernel, the layout hook and the tests.
 */
#pragma once

#include "rad_planes.h"

#include <cstdint>

namespace r3d_w4nl {

/* The form's tag (see the file comment). */
enum Form : uint32_t { kW4NL = 1 };

constexpr int     kBits = 4;             /* u4 codes */
constexpr int64_t kGroup = 64;           /* K per E4M3 scale */
constexpr int64_t kBaseGroup = 128;      /* the op's `group` parameter */
constexpr float   kLevel = 0x1p-13f;     /* the fixed second level, libr4d's kW4G64Scale */
constexpr float   kEighth = 0.125f;

/* The w4nl levels, exactly as the container's table plane states them (r3d_args.h's kW4nlTable). */
constexpr float kTable[16] = { 1.125f, 3.25f, 5.5f, 8.0f, 11.0f, 14.0f, 18.0f, 22.0f,
                               -1.125f, -3.25f, -5.5f, -8.0f, -11.0f, -14.0f, -18.0f, -22.0f };

/* The same levels in eighths: the row's exact integer operand. */
constexpr int16_t kTable8[16] = { 9, 26, 44, 64, 88, 112, 144, 176,
                                  -9, -26, -44, -64, -88, -112, -144, -176 };

/* ---- the joined planes' offsets ---------------------------------------------------------------
 *
 * rad_planes plans `[header][entry 0][entry 1][plane 0][plane 1]`, each part 64-byte aligned, and
 * both views here have exactly two planes (codes,table) and (scale,scale.1). The offsets are the
 * same for every expert of a class because every entry has the class's geometry.
 */
constexpr int64_t round64(int64_t n) { return (n + 63) & ~63ll; }
constexpr int64_t first_plane_off() {
    return round64((int64_t)sizeof(RadPlanesHeader) + 2 * (int64_t)sizeof(RadPlanesEntry));
}
constexpr int64_t codes_off() { return first_plane_off(); }
constexpr int64_t codes_bytes(int64_t N, int64_t K) { return N * (K / 2); }
constexpr int64_t table_off(int64_t N, int64_t K) {
    return round64(codes_off() + codes_bytes(N, K));
}
constexpr int64_t scale_off() { return first_plane_off(); }
constexpr int64_t scale_bytes(int64_t N, int64_t K) { return N * (K / kGroup); }
constexpr int64_t level_off(int64_t N, int64_t K) {
    return round64(scale_off() + scale_bytes(N, K));
}

/* Whether `v` is the shipped table, value for value. The layout hook's relayout calls this before
 * it stores the joined form, so a container quantised against another table is refused at load. */
inline bool table_is_w4nl(const float* v, int n) {
    if (!v || n != 16) return false;
    for (int i = 0; i < 16; ++i)
        if (v[i] != kTable[i]) return false;
    return true;
}

}  // namespace r3d_w4nl
