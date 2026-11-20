// DFlash2 grouped dynamic depthwise convolution -- device math only.
//
// Free of <hip/hip_runtime.h> so the same source compiles standalone to gfx1100 assembly with a
// stock clang (no ROCm) for ISA review, as well as into libr3d.so. r3dh_common.h declares the same
// 16-byte vector type for the source that includes both; the typedefs here are private to this
// header so the two can coexist.
//
// The op, per DFlash2's _grouped_conv, over blocks of `block` rows starting at row 0:
//
//   out[t,h] = SUM over j in [0, taps) with j == 0 or t % block >= j of
//                  (base[j,h] + delta[t,j,g]) * x[t-j,h],   g = h / group
//
// in ascending j, each term (base + delta) * x rounded as written in f32, the sum rounded once to
// bf16. That is libref's ref_dflash_conv with -ffp-contract=off (the whole library is compiled
// that way), so the two agree bit for bit.
//
// One thread owns 8 contiguous channels: a dwordx4 in and out, 16 bytes each. Swizzled groups
// (group 16) are a multiple of 8, so the group index and both delta scalars are uniform per thread
// and read once.
#pragma once

typedef unsigned short r3dhc_u16;
typedef unsigned int   r3dhc_u32;
typedef r3dhc_u32 r3dhc_u32x4 __attribute__((ext_vector_type(4)));

// A 16-byte load addressed both ways. Subscript the union member, never a bit-cast vector:
// bit-casting a vector subscript silently yields element 0's bits.
union R3dhcV8 { r3dhc_u32x4 v; r3dhc_u16 h[8]; };

// Device functions under hipcc; plain always-inline functions under a standalone amdgcn clang.
#ifndef R3DHC_DEV
#ifdef __HIP__
#define R3DHC_DEV __device__ __forceinline__
#else
#define R3DHC_DEV static __attribute__((always_inline))
#endif
#endif

R3DHC_DEV float r3dhc_bf2f(r3dhc_u16 v) {
    return __builtin_bit_cast(float, (r3dhc_u32)v << 16);
}

// f32 -> bf16 round-to-nearest-even, the rounding libref's store performs.
R3DHC_DEV r3dhc_u16 r3dhc_f2bf(float f) {
    r3dhc_u32 u = __builtin_bit_cast(r3dhc_u32, f);
    return (r3dhc_u16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// x and out are [T, H] bf16 at row pitch H; delta is [T, taps, NG] bf16 read at row pitch dpitch (a
// slice of the [T, 2, taps, NG] coefficient projection, so the pitch is 2*taps*NG and not the
// slice's own width); base is the [taps, H] weight; out must not alias x, because tap j reads row
// t - j. `block` is the query block: a block's first rows never convolve with the previous block.
template <int TAPS, int GROUP>
R3DHC_DEV void r3dhc_dflash_conv_body(const r3dhc_u16* __restrict__ x,
                                      const r3dhc_u16* __restrict__ delta,
                                      const r3dhc_u16* __restrict__ base,
                                      r3dhc_u16* __restrict__ out,
                                      int H, int dpitch, int NG, int vpr, int block,
                                      int t, int hv) {
    if (hv >= vpr) return;

    const int off = hv * 8;                       /* the thread's first channel in the row */
    const int g   = off / GROUP;                  /* GROUP is 16, so the group is uniform */

    const r3dhc_u16* __restrict__ xr = x + (long)t * H + off;
    r3dhc_u16*       __restrict__ orow = out + (long)t * H + off;
    const r3dhc_u16* __restrict__ dr = delta + (long)t * dpitch + g;

    R3dhcV8 x0; x0.v = *(const r3dhc_u32x4*)xr;
    R3dhcV8 b0; b0.v = *(const r3dhc_u32x4*)(base + off);
    const float d0 = r3dhc_bf2f(*dr);

    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
        acc[j] = (r3dhc_bf2f(b0.h[j]) + d0) * r3dhc_bf2f(x0.h[j]);

    // Tap `tap` contributes only where the position within the block is >= tap; `t` is the
    // workgroup's y index, so this is a scalar branch and a skipped row issues no shifted load.
#pragma unroll
    for (int tap = 1; tap < TAPS; ++tap) {
        const r3dhc_u16* __restrict__ xprev = xr - (long)tap * H;
        const r3dhc_u16* __restrict__ btap  = base + (long)tap * H + off;
        if ((t % block) >= tap) {
            R3dhcV8 xs; xs.v = *(const r3dhc_u32x4*)xprev;
            R3dhcV8 bt; bt.v = *(const r3dhc_u32x4*)btap;
            const float dt = r3dhc_bf2f(dr[tap * NG]);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[j] += (r3dhc_bf2f(bt.h[j]) + dt) * r3dhc_bf2f(xs.h[j]);
        }
    }

    R3dhcV8 o;
#pragma unroll
    for (int j = 0; j < 8; ++j) o.h[j] = r3dhc_f2bf(acc[j]);
    *(r3dhc_u32x4*)orow = o.v;
}
