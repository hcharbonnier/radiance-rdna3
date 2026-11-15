/* r3d_iqr_embed.hip -- embed_lookup over an IQR table (r3d_iqr.h's embed_off), on gfx11.
 *
 * The RDNA3 port of libr4d/r4d_iqr_embed.hip: the gather decodes each requested row from its
 * codes, so VRAM holds the table at its codes' width. A row's value is the form's int8 value times
 * its (row, 128) scale in f32, the 128-wide Hadamard undone in f32 on a rotated table (span 1, 2,
 * .., 64), rounded to bf16 to nearest even.
 *
 * The trellis's state for vocabulary row r is its last ceil(L/B) codes in its column, the run of
 * 256 rows wrapping; the launcher refuses another run, as libr4d does.
 */
#include "r3d_args.h"
#include "r3d_codecs.h"
#include "r3d_common.h"
#include "r3d_iqr.h"
#include "r3d_iqr_dec.h"
#include "r3d_launch.h"

namespace {

using namespace r3d_iqr;
using namespace r3d_iqr_dev;

/* `n` <= 16 bits of a byte stream from bit `o`. */
__device__ __forceinline__ uint32_t bits_at(const uint8_t* p, uint64_t o, uint32_t n) {
    const uint8_t* b = p + (o >> 3);
    const uint32_t w = (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16;
    return (w >> (o & 7u)) & ((1u << n) - 1u);
}

template <uint32_t KIND, uint32_t B, uint32_t LV, uint32_t L, uint32_t LRB, uint32_t SC, uint32_t ROT>
__global__ __launch_bounds__(128) void iqr_embed_kernel(const int32_t* __restrict__ tok,
                                                        const uint8_t* __restrict__ W,
                                                        uint16_t* __restrict__ x, uint32_t n_embd,
                                                        int32_t n_vocab, int32_t vocab_off,
                                                        uint32_t x_ld, float mult) {
    constexpr Form f{ KIND, B, LV, L, LRB, SC, 0, ROT };
    __shared__ float s_v[128];
    const uint32_t m = blockIdx.x, g = blockIdx.y, t = threadIdx.x, k = g * 128u + t;
    const int32_t id = tok[m] - vocab_off;
    uint16_t* dst = x + (size_t)m * x_ld + k;
    if (id < 0 || id >= n_vocab) {
        *dst = 0;
        return;
    }
    const EmbedOff o = embed_off(f, n_vocab, n_embd);
    const uint32_t nkb = n_embd / 128u, r = (uint32_t)id;
    auto code = [&](uint32_t row, uint32_t col) {
        return bits_at(W + (size_t)row * o.codes_row, (uint64_t)col * B, B);
    };
    const int8_t* aux = (const int8_t*)(W + o.aux);
    int32_t v;
    if constexpr (KIND == kTable) {
        v = aux[code(r, k)];
    } else if constexpr (KIND == kGrid) {
        v = aux[code(r, k >> LV) * (1u << LV) + (k & ((1u << LV) - 1u))];
    } else if constexpr (KIND == kZero) {
        v = (int32_t)code(r, k) - (int32_t)((const uint8_t*)aux)[(size_t)r * nkb + g];
    } else if constexpr (KIND == kSigned) {
        v = (int32_t)(code(r, k) << (32u - B)) >> (32u - B);
    } else {
        constexpr uint32_t C = (L + B - 1) / B, run = 256u;
        const uint32_t r0 = r / run * run;
        uint32_t st = 0;
#pragma unroll
        for (uint32_t q = C; q-- > 0;) {
            const uint32_t rr = r0 + (r - r0 + run - q) % run;
            st = ((st << B) | code(rr, k)) & ((1u << L) - 1u);
        }
        v = aux[st];
    }
    uint32_t own;
    if constexpr (SC == kBf16) {
        own = ((const uint16_t*)(W + o.sc))[(size_t)r * nkb + g];
    } else {
        const float s2 = ((const float*)(W + o.s2))[SC == kE4m3Row ? r : 0u];
        own = bf16_bits(s2 * e4m3_to_f32((W + o.sc)[(size_t)r * nkb + g]));
    }
    uint32_t sb = own;
    if constexpr (LRB != 0) {
        sb = ((const uint16_t*)(W + o.blk))[(size_t)(r >> LRB) * nkb + g];
        const float so = __uint_as_float(own << 16), sbf = __uint_as_float(sb << 16);
        v = sbf > 0.0f ? (int32_t)rintf(__fmul_rn((float)v, __fdiv_rn(so, sbf))) : 0;
    }
    float val = __fmul_rn((float)v, __uint_as_float(sb << 16));
    if constexpr (ROT != 0) {
        s_v[t] = val;
        __syncthreads();
#pragma unroll
        for (uint32_t h = 1; h < 128u; h <<= 1) {
            const float a = s_v[t & ~h], b = s_v[t | h];
            const float y = (t & h) ? __fsub_rn(a, b) : __fadd_rn(a, b);
            __syncthreads();
            s_v[t] = y;
            __syncthreads();
        }
        val = s_v[t];
    }
    uint32_t bits = bf16_bits(val);
    if (mult != 1.0f) bits = f32_to_bf16(__uint_as_float(bits << 16) * mult);
    *dst = (uint16_t)bits;
}

/* The run, which the kernel assumes, is libquant's tcq_rows = 256. */
int run_ok(const Form& f) { return f.kind != kTrellis || f.lrun == 8u; }

}  // namespace

using namespace r3d_iqr;

enum { EL_TOK = 0, EL_WTE, EL_X };

extern "C" int r3d_rad_embed_lookup_iqr(const RadArgs* a, RadStream stream) {
    const RadTensor* tok = r3d_opt(a, EL_TOK);
    const RadTensor* wte = r3d_opt(a, EL_WTE);
    const RadTensor* x = r3d_opt(a, EL_X);
    if (!tok || !wte || !x) return R3D_NO(RAD_E_INVAL);
    if (tok->dtype != RAD_I32 || x->dtype != RAD_BF16 || !is_weight_dtype(wte->dtype))
        return R3D_NO(RAD_E_DTYPE);
    if (!r3d_contig(tok)) return R3D_NO(RAD_E_STRIDE);
    const long long n_embd = r3d_geti_or(a, "n_embd", 0), n_vocab = r3d_geti_or(a, "n_vocab", 0);
    const long long voff = r3d_geti_or(a, "vocab_offset", 0);
    if (n_embd <= 0 || n_embd % 128 || n_vocab <= 0 || voff < 0) return R3D_NO(RAD_E_SHAPE);
    long long M = 0;
    const long long x_ld = r3d_rows_pitch(x, n_embd, &M);
    if (x_ld < 0) return R3D_NO(RAD_E_STRIDE);
    if (M <= 0 || r3d_numel(tok) < M) return R3D_NO(RAD_E_SHAPE);
    const float mult = (float)r3d_getf(a, "mult", 1.0);
    const uint32_t kw = kernel_word(wte->dtype & kWordMask);
    const Form f = form(wte->dtype & kWordMask);
    if (!run_ok(f)) return R3D_NO(RAD_E_SHAPE);
    if (!r3d_aligned(r3d_p(tok), 4) || !r3d_aligned(r3d_p(wte), 16) || !r3d_aligned(r3d_p(x), 2))
        return R3D_NO(RAD_E_ALIGN);
    const dim3 grid((uint32_t)M, (uint32_t)(n_embd / 128));
    int rc = RAD_E_UNSUPPORTED;
#define R3D_IQR_EGO(k, b, lv, l, lrb, sc, r)                                                      \
    if (kw == word(Form{ k, b, lv, l, lrb, sc, 0, r })) {                                          \
        R3D_KLAUNCH(R3D_KNAME((iqr_embed_kernel<k, b, lv, l, lrb, sc, r>)), grid, dim3(128),       \
                    (hipStream_t)stream, (const int32_t*)tok->data,                                \
                    (const uint8_t*)wte->data, (uint16_t*)x->data, (uint32_t)n_embd,               \
                    (int32_t)n_vocab, (int32_t)voff, (uint32_t)x_ld, mult);                        \
        rc = RAD_OK;                                                                               \
    } else
    R3D_IQR_EMBED_FORMS(R3D_IQR_EGO) { return R3D_NO(RAD_E_UNSUPPORTED); }
#undef R3D_IQR_EGO
    return rc == RAD_OK && hipGetLastError() == hipSuccess ? RAD_OK : R3D_NO(RAD_E_DEVICE);
}

/* W4/W11: the vocabulary-weighted head's kernel. The row exists so the architecture declares (its
 * classes' layout, the op's schema, and the weights are all resolved), and refuses at run time
 * until the kernel lands. */
extern "C" int r3d_rad_logits_gemm_vocab_iqr(const RadArgs* a, RadStream s) {
    (void)a;
    (void)s;
    return R3D_NO(RAD_E_UNSUPPORTED);
}
