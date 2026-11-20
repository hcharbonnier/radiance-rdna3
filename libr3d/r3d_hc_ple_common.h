/* r3d_hc_ple_common.h -- the small conversions the HC and PLE kernels share.
 *
 * These rows read and write ten dtypes (the two block families, the row-direct mix streams, the
 * n-gram table's E4M3 and the activations), so the load/store switch lives here rather than in
 * each unit. Only the dtypes a row's constraints admit are reachable; an unsupported code is
 * refused by the shim before launch.
 *
 * The activation arithmetic is libref's spelling on purpose: `act_silu` is `x * sigmoid(x)` with
 * `std::exp`, and the narrowings are one RTNE store. That is what makes the HC/PLE ladders
 * comparable layer by layer against the host reference instead of only in distribution.
 */
#ifndef R3D_HC_PLE_COMMON_H
#define R3D_HC_PLE_COMMON_H

#include <hip/hip_runtime.h>

#include <cstdint>

#include "r3d.h"
#include "r3d_codecs.h"

namespace r3d_hcple {

/* A value in the dtype `dt` (R3D_DT_*), one element at `i`. E4M3 is the container's fp8 byte,
 * decoded exactly; I8 is a signed code, not a scale. */
__device__ __forceinline__ float ld_elem(int dt, const void* p, long long i) {
    switch (dt) {
        case R3D_DT_BF16: return bf16_to_f32(((const uint16_t*)p)[i]);
        case R3D_DT_F16:  return f16_to_f32(((const uint16_t*)p)[i]);
        case R3D_DT_F32:  return ((const float*)p)[i];
        case R3D_DT_I32:  return (float)((const int32_t*)p)[i];
        case R3D_DT_I64:  return (float)((const int64_t*)p)[i];
        case R3D_DT_U8:   return e4m3_to_f32(((const uint8_t*)p)[i]);
        case R3D_DT_I8:   return (float)((const int8_t*)p)[i];
        default:          return 0.0f;
    }
}

/* One narrowing to the dtype `dt`; the float dtypes round once, as libref's `stt` does. */
__device__ __forceinline__ void st_elem(int dt, void* p, long long i, float v) {
    switch (dt) {
        case R3D_DT_BF16: ((uint16_t*)p)[i] = f32_to_bf16(v); break;
        case R3D_DT_F16:  ((uint16_t*)p)[i] = f32_to_f16(v); break;
        case R3D_DT_F32:  ((float*)p)[i] = v; break;
        case R3D_DT_I32:  ((int32_t*)p)[i] = (int32_t)v; break;
        case R3D_DT_I64:  ((int64_t*)p)[i] = (int64_t)v; break;
        case R3D_DT_U8:   ((uint8_t*)p)[i] = f32_to_e4m3(v); break;
        case R3D_DT_I8:   ((int8_t*)p)[i] = f32_to_i8_rtne(v); break;
        default:          break;
    }
}

/* The narrowed value kept in f32: what libref's `round_dt` hands the next product. */
__device__ __forceinline__ float narrow(int dt, float v) {
    switch (dt) {
        case R3D_DT_BF16: return bf16_to_f32(f32_to_bf16(v));
        case R3D_DT_F16:  return f16_to_f32(f32_to_f16(v));
        case R3D_DT_F32:  return v;
        case R3D_DT_U8:   return e4m3_to_f32(f32_to_e4m3(v));
        default:          return v;
    }
}

/* libref computes exp in double and narrows the result, so its sigmoid is the correctly rounded
 * one. `expf` is a single-precision approximation; on the gate/ladder paths a one-ulp difference
 * there would show up as a changed bf16 byte, which the bit-exact tests would catch. The double
 * routine is slower, but it runs once a gate value, not once a weight element. */
__device__ __forceinline__ float ref_exp(float x) { return (float)exp((double)x); }

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + ref_exp(-x)); }

/* libref's act_silu: the sigmoid first, then the multiply. */
__device__ __forceinline__ float silu(float x) { return x * sigmoid(x); }

__device__ __forceinline__ float rsqrtf_exact(float x) { return 1.0f / sqrtf(x); }

}  // namespace r3d_hcple

#endif  /* R3D_HC_PLE_COMMON_H */
