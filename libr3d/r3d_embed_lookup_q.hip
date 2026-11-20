/* r3d_embed_lookup_q.hip -- embed_lookup_q: gather rows from a quantised table.
 *
 *   x[r, :] = wte[tok[r] - vocab_offset, :] * scale        (or a zero row)
 *
 * The table is the checkpoint's n-gram embedding: E4M3 codes with one bf16 (or f16/f32) scale for
 * the whole table, or a plain bf16 table read in place with no scale. The op exists separately
 * from embed_lookup because the scale is an operand, not a parameter, so a device row cannot fold
 * it into a later op. One workgroup per output row; the token becomes zero when it is outside the
 * table, as docs/OPS.md says the device writes.
 *
 * The optional `ahead` operand is a prefetch hint for the mapped tier. This row is fed a device
 * pointer, so it ignores it: there is nothing to pull into a host arena here.
 */
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r3d.h"
#include "r3d_args.h"
#include "r3d_hc_ple_common.h"
#include "r3d_launch.h"

namespace {

__global__ __launch_bounds__(256) void embed_q_kernel(
    const int32_t* __restrict__ tok, const void* __restrict__ wte, const void* __restrict__ scale,
    void* __restrict__ x, int M, int n_embd, int n_vocab, int vocab_offset, int x_ld,
    long long w_ld, int sc_dt, int tbl_dt, int x_dt) {
    const int m = (int)blockIdx.x;
    if (m >= M) return;
    const int id = tok[m] - vocab_offset;
    const bool have = id >= 0 && id < n_vocab;
    float sc = 1.0f;
    if (scale) sc = r3d_hcple::ld_elem(sc_dt, scale, 0);
    for (int i = (int)threadIdx.x; i < n_embd; i += 256) {
        float v = 0.0f;
        if (have) v = r3d_hcple::ld_elem(tbl_dt, wte, (long long)id * w_ld + i) * sc;
        r3d_hcple::st_elem(x_dt, x, (long long)m * x_ld + i, v);
    }
}

}  // namespace

extern "C" {

int r3d_embed_lookup_q(long tok, long wte, long scale, long x, int M, int n_embd, int n_vocab,
                       int vocab_offset, int x_ld, int w_ld, int sc_dt, int tbl_dt, int x_dt,
                       long stream) {
    if (!tok || !wte || !x) return -1;
    if (M <= 0 || n_embd <= 0 || n_vocab <= 0 || x_ld < n_embd || w_ld < n_embd) return -2;
    if (vocab_offset < 0) return -2;
    const int thr = 256;
    R3D_KLAUNCH(embed_q_kernel, dim3((uint32_t)M), dim3(thr), (hipStream_t)stream,
                (const int32_t*)tok, (const void*)wte, (const void*)scale, (void*)x, M, n_embd,
                n_vocab, vocab_offset, x_ld, (long long)w_ld, sc_dt, tbl_dt, x_dt);
    return hipGetLastError() == hipSuccess ? 0 : -4;
}

int r3d_rad_embed_lookup_q(const RadArgs* a, RadStream s) {
    if (!a || a->n_t < 4) return R3D_NO(RAD_E_INVAL);
    const RadTensor* tok = r3d_opt(a, 0), *wte = r3d_opt(a, 1), *sc = r3d_opt(a, 2),
                    *x = r3d_opt(a, 3);
    if (!r3d_p(tok) || !r3d_p(wte) || !r3d_wp(x)) return R3D_NO(RAD_E_INVAL);
    if (!r3d_is_i32(tok)) return R3D_NO(RAD_E_DTYPE);
    const int tbl_dt = r3d_dt_code(wte->dtype);
    if (tbl_dt != R3D_DT_BF16 && tbl_dt != R3D_DT_F16 && tbl_dt != R3D_DT_F32 &&
        tbl_dt != R3D_DT_U8)
        return R3D_NO(RAD_E_DTYPE);
    const int x_dt = r3d_dt_code(x->dtype);
    if (x_dt != R3D_DT_BF16 && x_dt != R3D_DT_F16 && x_dt != R3D_DT_F32)
        return R3D_NO(RAD_E_DTYPE);

    int sc_dt = R3D_DT_F32;
    const void* scp = nullptr;
    if (r3d_has(sc)) {
        sc_dt = r3d_dt_code(sc->dtype);
        if (sc_dt != R3D_DT_BF16 && sc_dt != R3D_DT_F16 && sc_dt != R3D_DT_F32)
            return R3D_NO(RAD_E_DTYPE);
        scp = r3d_p(sc);
    }

    const long long n_embd = r3d_geti_or(a, "n_embd", wte->rank >= 2 ? r3d_dim(wte, (int)wte->rank - 1) : 0);
    const long long n_vocab = r3d_geti_or(a, "n_vocab", wte->rank >= 2 ? r3d_dim(wte, (int)wte->rank - 2) : 0);
    const long long voff = r3d_geti_or(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_vocab <= 0 || voff < 0) return R3D_NO(RAD_E_SHAPE);
    /* The table's row pitch: a view's rows can be wider than the shape. E4M3 codes are one byte,
     * and the element pitch is the row's column stride. */
    const long long w_ld = wte->rank >= 2 ? r3d_stride(wte, (int)wte->rank - 2) : n_embd;
    if (w_ld < n_embd) return R3D_NO(RAD_E_STRIDE);

    long long rows = 0;
    const long long x_ld = r3d_rows_pitch(x, n_embd, &rows);
    if (x_ld < 0 || rows <= 0) return R3D_NO(RAD_E_STRIDE);
    if (r3d_numel(tok) < rows) return R3D_NO(RAD_E_SHAPE);
    if (r3d_numel(wte) < n_vocab * n_embd) return R3D_NO(RAD_E_SHAPE);

    const int rc = r3d_embed_lookup_q((long)(uintptr_t)r3d_p(tok), (long)(uintptr_t)r3d_p(wte),
                                      (long)(uintptr_t)scp, (long)(uintptr_t)r3d_wp(x), (int)rows,
                                      (int)n_embd, (int)n_vocab, (int)voff, (int)x_ld, (int)w_ld,
                                      sc_dt, tbl_dt, x_dt, (long)s);
    return rc == 0 ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

}  /* extern "C" */
