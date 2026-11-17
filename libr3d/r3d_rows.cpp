/* r3d_rows.cpp -- the libr3d row table: what each kernel computes and for which geometry.
 *
 * One row per (kernel, band). A row is a `RadKernelInfo`: names, prose, the geometry it serves as
 * constraints, the tuning axes and the entry points that launch it. The engine reads this table at
 * load, matches a requested op and geometry against it and ranks the candidates by `priority`. It
 * never calls into this file to select, and nothing outside libr3d names libr3d's kernels.
 *
 * ============================================================ the schema table
 *
 * The first plugin in hierarchy order to declare an op fixes its schema, and a plugin whose schema
 * disagrees is refused at load (rad_abi.h, spec §2.3): arguments are positional, so two schemas
 * would give one call two meanings. libref declares the vocabulary (`docs/OPS.md`), so every schema
 * below is libref's: the same keys, types, requiredness, operand roles and optionality, in the same
 * order. The prose describes this library's rows; the loader compares only the positional slots.
 *
 * libr3d declares no new op: an op declared only by a kernel plugin has no reference to be checked
 * against.
 *
 * ============================================================ the constraint table
 *
 * A constraint is a necessary condition on the geometry (spec §2.1). An operand's dtype is a
 * property of the call, and the shim checks it on the tensor. Constraints state what the kernel
 * cannot do (the divisibility its lane map assumes, the band its grid is sized for, the scheme its
 * decoders implement) and what the selector needs to choose between rows. A constraint the kernel
 * does not actually honour is a promise the tests will break.
 *
 * ============================================================ rows are positional
 *
 * `RadKernelInfo` is initialised positionally, as `rad_abi.h` requires: a field added anywhere but
 * the end shifts every later initialiser, and because `nullptr` converts to any function pointer,
 * part of the shift still compiles. Unused trailing slots are left off and zero-initialised. A row's
 * entry points are, in order: init, fini, launch, scratch, layout, relayout, opd_shape, unrelayout,
 * replaces.
 *
 * The `layout` and `relayout` hooks here only present the container's own planes as the joined
 * operands they store (the w2a8 row's `b_scale`, the fp8 weight's codes and scale, the logits head)
 * and repack nothing, so a container converted on this card keeps row-major codes and scales. A
 * dot2 GEMM reads a weight in stored order; libr4d's fragment permutations exist for WMMA, which
 * these kernels do not use (see r3d_common.h).
 *
 * Pure host C++ (no r3d.h, no HIP), so the table compiles and can be tested without ROCm.
 */

#include "r3d_plugin.h"
#include "r3dh_args.h"
#include "rad_abi_layout.h"

#ifndef R3D_BUILD_TARGET
#define R3D_BUILD_TARGET "gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151"
#endif

#include <dirent.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define ROW(a)  (a), (int)(sizeof(a) / sizeof((a)[0]))
#define NELEM(a) ((int)(sizeof(a) / sizeof((a)[0])))

/* Parameter and constraint spellings, shared with libref's table. */
#define P_INT(k)   { (k), RAD_P_INT, RAD_REQUIRED }
#define P_INTO(k)  { (k), RAD_P_INT, RAD_OPTIONAL }
#define P_INTD(k)  { (k), RAD_P_INT, RAD_DERIVED }
/* The sequence-chunk parameter: derived, and the one the core reads by meaning (RAD_PROLE_SEQ_CHUNK).
 * Roles are part of a schema, so this must match libref's spelling, not only P_INTD's. */
#define P_CHUNK(k) { (k), RAD_P_INT, RAD_DERIVED, RAD_PROLE_SEQ_CHUNK }
#define P_STR(k)   { (k), RAD_P_STR, RAD_REQUIRED }
#define P_STRO(k)  { (k), RAD_P_STR, RAD_OPTIONAL }
#define P_F64(k)   { (k), RAD_P_F64, RAD_REQUIRED }
#define P_F64O(k)  { (k), RAD_P_F64, RAD_OPTIONAL }
/* A count the kernel is handed as a bound and never writes past its operands for (rad_abi.h,
 * RAD_PROLE_CAPACITY). */
#define P_CAP(k)   { (k), RAD_P_INT, RAD_REQUIRED, RAD_PROLE_CAPACITY }

#define C_EQ(k, v)  { (k), RAD_C_EQ,  (long long)(v), nullptr }
#define C_LE(k, v)  { (k), RAD_C_LE,  (long long)(v), nullptr }
#define C_GE(k, v)  { (k), RAD_C_GE,  (long long)(v), nullptr }
#define C_DIV(k, v) { (k), RAD_C_DIV, (long long)(v), nullptr }
#define C_IN(k, s)  { (k), RAD_C_IN,  0, (s) }
/* The engine sets RAD_FACT_P2P to 1 when every rank can reach every peer's VRAM. A collective
 * that moves its payload through the peer's IPC slot cannot serve the host-mailbox fallback. */
#define C_P2P       C_EQ(RAD_FACT_P2P, 1)

/* A row with one implementation and nothing to measure. */
#define NOTUNE nullptr, 0, nullptr

#define SCHEMA(op, p, o, doc) { (op), ROW(p), ROW(o), (doc) }

/* ================================================================== schemas
 *
 * Operand names are diagnostics only; roles and optionality are positional and must match.
 * Operand names may differ from libref's (`b` for a gate, say) without changing the schema.
 */

/* --- norms and elementwise --- */
/* r3d_rope_imrope_h128.hip: the QSA query's imrope row. Kept out of r3d_plugin.h so the h256 and
 * h72 rows' translation units and entry points stay untouched. */
extern "C" int r3d_rad_rope_imrope_h128(const RadArgs*, RadStream);

static const RadParamSpec pRmsNorm[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype"),
                                         P_F64O("wadd"), P_F64O("mult") };
static const RadOperandSpec oRmsNorm[] = { { "x", RAD_OPD_IN, 0 },
                                           { "w", RAD_OPD_WEIGHT, 1 },
                                           { "y", RAD_OPD_OUT, 0 } };

static const RadParamSpec pLayernorm[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype") };
static const RadOperandSpec oLayernorm[] = { { "x", RAD_OPD_IN, 0 },
                                             { "w", RAD_OPD_WEIGHT, 0 },
                                             { "b", RAD_OPD_WEIGHT, 1 },
                                             { "y", RAD_OPD_OUT, 0 } };

static const RadParamSpec pAdd[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_F64O("div") };
static const RadParamSpec pMnDtype[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
static const RadOperandSpec oBinary[] = { { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_IN, 0 },
                                          { "y", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oUnary[] = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };

static const RadParamSpec pActMul[] = { P_INT("M"), P_INT("n"), P_STR("act"), P_STR("dtype") };

static const RadParamSpec pCast[] = { P_INT("M"), P_INT("n"), P_STR("from"), P_STR("to") };

static const RadParamSpec pScale[] = { P_INT("M"), P_INT("n"), P_STR("dtype") };
/* `s` is a weight operand, as the reference declares it: the issue resolves a handle and waits on
 * the mover, because a scale plane comes from the container or a quantiser, not from a buffer the
 * planner allocated for the step. The loader compares roles positionally, so declaring it IN would
 * get the whole library refused. */
static const RadOperandSpec oScale[] = { { "x", RAD_OPD_IN, 0 }, { "s", RAD_OPD_WEIGHT, 0 },
                                         { "y", RAD_OPD_OUT, 0 } };

static const RadParamSpec pFill[] = { P_INT("M"), P_INT("n"), P_STR("dtype"), P_F64O("value") };
static const RadOperandSpec oFill[] = { { "y", RAD_OPD_OUT, 0 } };

/* --- the model's odds and ends --- */
static const RadParamSpec pRope[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head"),
                                      P_INT("n_head_kv"), P_F64("theta"), P_F64("scale"),
                                      P_STR("mode"), P_INTO("rotary_dim"), P_STRO("sections"),
                                      P_F64O("yarn_factor"), P_F64O("yarn_low"),
                                      P_F64O("yarn_high"), P_F64O("yarn_magnitude"),
                                      P_STRO("ladder"), P_F64O("llama3_factor"),
                                      P_F64O("llama3_low"), P_F64O("llama3_high"),
                                      P_F64O("llama3_orig") };
static const RadOperandSpec oRope[] = { { "qkv", RAD_OPD_INOUT, 0 },
                                        { "positions", RAD_OPD_IN, 0 } };

static const RadParamSpec pRopeTable[] = { P_INT("M"), P_INT("rot"), P_F64("theta"),
                                           P_F64("scale"), P_STR("dtype"),
                                           P_F64O("yarn_factor"), P_F64O("yarn_low"),
                                           P_F64O("yarn_high"), P_F64O("yarn_magnitude"),
                                           P_INTO("head_dim"), P_STRO("ladder") };
static const RadOperandSpec oRopeTable[] = { { "positions", RAD_OPD_IN, 0 },
                                             { "cos_sin", RAD_OPD_OUT, 0 } };

static const RadParamSpec pKvStore[] = { P_INT("M"), P_INT("head_dim"), P_INT("n_head_kv"),
                                         P_INT("block_size"), P_STR("kv_dtype"),
                                         P_F64O("k_scale"), P_F64O("v_scale") };
static const RadOperandSpec oKvStore[] = { { "k", RAD_OPD_IN, 0 }, { "v", RAD_OPD_IN, 0 },
                                           { "slot_mapping", RAD_OPD_IN, 0 },
                                           { "kv_cache", RAD_OPD_INOUT, 0 } };

static const RadParamSpec pEmbed[] = { P_INT("M"), P_INT("n_embd"), P_INT("n_vocab"),
                                       P_STR("dtype"), P_INTO("vocab_offset"), P_F64O("wscale"),
                                       P_F64O("mult") };
static const RadOperandSpec oEmbed[] = { { "tokens", RAD_OPD_IN, 0 },
                                         { "wte", RAD_OPD_WEIGHT, 0 },
                                         { "x", RAD_OPD_OUT, 0 } };
/* embed_lookup_q's parameter list: embed_lookup's without `mult`, since the quantised table's
 * scale is an operand and the row has no separate per-row factor. */
static const RadParamSpec pEmbedQ[] = { P_INT("M"), P_INT("n_embd"), P_INT("n_vocab"),
                                        P_STR("dtype"), P_INTO("vocab_offset"), P_F64O("wscale") };

/* The lm_head as up to four width classes, each a (codes, scales, ids) triple (r3d_iqr.h's lm
 * form); `vrow` is the inverse. Copied from libr4d's schema: the op is device-only and no
 * reference declares it. */
static const RadParamSpec pLogitsVocab[] = {
    { "M", RAD_P_INT, RAD_REQUIRED },       { "n_vocab", RAD_P_INT, RAD_REQUIRED },
    { "n_embd", RAD_P_INT, RAD_REQUIRED },  { "dtype", RAD_P_STR, RAD_REQUIRED },
    { "vocab_offset", RAD_P_INT, RAD_REQUIRED }, { "rows0", RAD_P_INT, RAD_REQUIRED },
    { "rows1", RAD_P_INT, RAD_OPTIONAL },   { "rows2", RAD_P_INT, RAD_OPTIONAL },
    { "rows3", RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oLogitsVocab[] = {
    { "x", RAD_OPD_IN, 0 },      { "w0", RAD_OPD_WEIGHT, 0 }, { "s0", RAD_OPD_WEIGHT, 0 },
    { "ids0", RAD_OPD_WEIGHT, 0 }, { "w1", RAD_OPD_WEIGHT, 1 }, { "s1", RAD_OPD_WEIGHT, 1 },
    { "ids1", RAD_OPD_WEIGHT, 1 }, { "w2", RAD_OPD_WEIGHT, 1 }, { "s2", RAD_OPD_WEIGHT, 1 },
    { "ids2", RAD_OPD_WEIGHT, 1 }, { "w3", RAD_OPD_WEIGHT, 1 }, { "s3", RAD_OPD_WEIGHT, 1 },
    { "ids3", RAD_OPD_WEIGHT, 1 }, { "vrow", RAD_OPD_WEIGHT, 0 }, { "logits", RAD_OPD_OUT, 0 },
};

static const RadOperandSpec oGather[] = { { "x", RAD_OPD_IN, 0 }, { "idx", RAD_OPD_IN, 0 },
                                          { "y", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oScatter[] = { { "v", RAD_OPD_IN, 0 }, { "idx", RAD_OPD_IN, 0 },
                                           { "x", RAD_OPD_INOUT, 0 } };

/* --- quantisation --- */
static const RadParamSpec pQuant[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype") };
static const RadOperandSpec oQuantFp8[] = { { "x", RAD_OPD_IN, 0 }, { "q", RAD_OPD_OUT, 0 },
                                            { "scale", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oQuant[] = { { "x", RAD_OPD_IN, 0 }, { "q", RAD_OPD_OUT, 0 },
                                         { "scale", RAD_OPD_OUT, 0 },
                                         { "asum", RAD_OPD_OUT, 1 } };

static const RadParamSpec pGatedQuant[] = { P_INT("M"), P_INT("n"), P_INT("group"), P_STR("dtype"),
                                            P_STRO("act") };

 /* router_topk_scatter: the pair as one op -- logits, the selection (expert_ids, expert_w), and
  * the sort (sorted_tok, expert_offset, expert_count). */
static const RadOperandSpec oRouterTopkScatter[] = { { "logits", RAD_OPD_IN, 0 },
                                                     { "expert_ids", RAD_OPD_OUT, 0 },
                                                     { "expert_w", RAD_OPD_OUT, 0 },
                                                     { "sorted_tok", RAD_OPD_OUT, 0 },
                                                     { "expert_offset", RAD_OPD_OUT, 0 },
                                                     { "expert_count", RAD_OPD_OUT, 0 } };
static const RadParamSpec pRouterTopkScatter[] = { P_INT("M"), P_INT("n_expert"), P_INTO("n_local"),
                                                   P_INT("top_k"), P_INT("norm"), P_STR("dtype"),
                                                   P_INTO("expert_base"), P_STRO("protect") };

 /* router_gemm_topk_scatter: the router GEMM, the pair, and -- when gate_b and gate come together
  * -- the shared expert's one-column gate. The union of the three operand lists, `logits` named
  * once as the GEMM's output, as libref's schema. */
static const RadOperandSpec oRouterGemmTopkScatter[] = { { "a", RAD_OPD_IN, 0 },
                                                         { "b", RAD_OPD_WEIGHT, 0 },
                                                         { "gate_b", RAD_OPD_WEIGHT, 1 },
                                                         { "logits", RAD_OPD_OUT, 0 },
                                                         { "expert_ids", RAD_OPD_OUT, 0 },
                                                         { "expert_w", RAD_OPD_OUT, 0 },
                                                         { "sorted_tok", RAD_OPD_OUT, 0 },
                                                         { "expert_offset", RAD_OPD_OUT, 0 },
                                                         { "expert_count", RAD_OPD_OUT, 0 },
                                                         { "gate", RAD_OPD_OUT, 1 } };
static const RadParamSpec pRouterGemmTopkScatter[] = { P_INT("M"), P_INT("K"), P_INT("n_expert"),
                                                       P_INTO("n_local"), P_INT("top_k"),
                                                       P_INT("norm"), P_STR("dtype"),
                                                       P_INTO("expert_base"), P_STRO("protect") };

 /* router_sigmoid_topk: the GLM-style router -- logits, the per-expert correction bias, and the
  * selection (expert_ids, expert_w). */
static const RadOperandSpec oRouterSigmoid[] = { { "logits", RAD_OPD_IN, 0 },
                                                 { "bias", RAD_OPD_WEIGHT, 0 },
                                                 { "ids", RAD_OPD_OUT, 0 },
                                                 { "weights", RAD_OPD_OUT, 0 } };
static const RadParamSpec pRouterSigmoid[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                               P_INT("norm"), P_F64("route_scale"), P_STR("dtype") };
static const RadParamSpec pRouterSigmoidGroup[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
    P_INT("n_group"), P_INT("topk_group"), P_INT("norm"), P_F64("route_scale"), P_STR("dtype") };
static const RadOperandSpec oGatedQuant[] = { { "gate_up", RAD_OPD_IN, 0 },
                                              { "q", RAD_OPD_OUT, 0 },
                                              { "scale", RAD_OPD_OUT, 0 },
                                              { "out_bf16", RAD_OPD_OUT, 1 } };
/* The gate in its own operand, as a gated attention's [q|gate] projection leaves it. */
static const RadOperandSpec oGateQuant[] = { { "gate", RAD_OPD_IN, 0 },
                                             { "x", RAD_OPD_IN, 0 },
                                             { "q", RAD_OPD_OUT, 0 },
                                             { "scale", RAD_OPD_OUT, 0 },
                                             { "out_bf16", RAD_OPD_OUT, 1 } };
static const RadOperandSpec oGeluMulQuantI8[] = { { "gate_up", RAD_OPD_IN, 0 },
                                                  { "q", RAD_OPD_OUT, 0 },
                                                  { "scale", RAD_OPD_OUT, 0 },
                                                  { "out_bf16", RAD_OPD_OUT, 1 } };
static const RadParamSpec pKvLenTrim[] = { P_INT("trim"), P_INT("max_seqs") };
static const RadOperandSpec oKvLenTrim[] = { { "seqused", RAD_OPD_IN, 0 },
                                             { "tail", RAD_OPD_IN, 0 },
                                             { "out", RAD_OPD_OUT, 0 } };
static const RadParamSpec pQsaTailStore[] = { P_INT("M"), P_INT("n"), P_INT("ratio"),
                                              P_INT("ring"), P_INT("seqs"), P_STR("dtype") };
static const RadOperandSpec oQsaTailStore[] = { { "k", RAD_OPD_IN, 0 },
                                                { "pos", RAD_OPD_IN, 0 },
                                                { "cu", RAD_OPD_IN, 0 },
                                                { "state", RAD_OPD_INOUT, 0 },
                                                { "sidx", RAD_OPD_IN, 0 } };
/* The QSA indexer's compressed key, libr4d's schema exactly: `block_size` spells `ratio` for the
 * block-key store's KV group, and `page`/`pos` are optional. */
static const RadParamSpec pQsaBlockKey[] = { P_INT("M"), P_INT("n"), P_INT("ratio"),
                                             P_INT("rotary_dim"), P_INTO("pos0"),
                                             P_INTO("block_size"), P_F64("eps"), P_F64("theta"),
                                             P_F64O("scale"), P_F64O("wadd"), P_STR("dtype"),
                                             P_STRO("mode"), P_STRO("sections"),
                                             P_F64O("yarn_factor"), P_F64O("yarn_low"),
                                             P_F64O("yarn_high"), P_F64O("yarn_magnitude") };
static const RadOperandSpec oQsaBlockKey[] = { { "k", RAD_OPD_IN, 0 },
                                               { "w", RAD_OPD_WEIGHT, 0 },
                                               { "page", RAD_OPD_IN, 1 },
                                               { "pos", RAD_OPD_IN, 1 },
                                               { "out", RAD_OPD_OUT, 0 } };
static const RadParamSpec pQsaScore[] = { P_INT("M"), P_INT("n"), P_INT("heads"), P_INT("blocks"),
                                          P_INT("seqs"), P_STR("dtype") };
static const RadOperandSpec oQsaScore[] = { { "q", RAD_OPD_IN, 0 }, { "bk", RAD_OPD_IN, 0 },
                                            { "bt", RAD_OPD_IN, 0 }, { "cu", RAD_OPD_IN, 0 },
                                            { "nc", RAD_OPD_IN, 0 }, { "score", RAD_OPD_OUT, 0 } };
static const RadParamSpec pQsaSelect[] = { P_INT("M"), P_INT("blocks"), P_INT("topk"),
                                           P_INT("ratio"), P_INT("seqs"), P_INTO("signed") };
static const RadOperandSpec oQsaSelect[] = { { "score", RAD_OPD_IN, 0 },
                                             { "bt", RAD_OPD_IN, 0 },
                                             { "nc", RAD_OPD_IN, 0 },
                                             { "pos", RAD_OPD_IN, 0 },
                                             { "cu", RAD_OPD_IN, 0 },
                                             { "sel", RAD_OPD_OUT, 0 },
                                             { "nsel", RAD_OPD_OUT, 0 },
                                             { "seqused", RAD_OPD_OUT, 0 } };
/* `work` bounds the blocks that can complete; slots past the real count get page -1. */
static const RadParamSpec pQsaWork[] = { P_INT("M"), P_INT("n"), P_INT("ratio"), P_INT("ring"),
                                         P_INT("seqs"), P_CAP("work"), P_STR("dtype") };
static const RadOperandSpec oQsaWork[] = { { "k", RAD_OPD_IN, 0 },
                                           { "pos", RAD_OPD_IN, 0 },
                                           { "cu", RAD_OPD_IN, 0 },
                                           { "state", RAD_OPD_IN, 0 },
                                           { "sidx", RAD_OPD_IN, 0 },
                                           { "bt", RAD_OPD_IN, 0 },
                                           { "rope", RAD_OPD_IN, 1 },
                                           { "stage", RAD_OPD_OUT, 0 },
                                           { "page", RAD_OPD_OUT, 0 },
                                           { "bpos", RAD_OPD_OUT, 0 },
                                           { "nc", RAD_OPD_OUT, 0 } };
static const RadParamSpec pGatedHadQuantI8[] = { P_INT("M"), P_INT("n"), P_INT("group"),
                                                 P_STR("dtype"), P_STRO("act"), P_INTO("mode") };
static const RadOperandSpec oGatedHadQuantI8[] = { { "a", RAD_OPD_IN, 0 },
                                                   { "b", RAD_OPD_IN, 1 },
                                                   { "q", RAD_OPD_OUT, 0 },
                                                   { "scale", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oQuantCols[] = { { "x", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 },
                                             { "scale", RAD_OPD_OUT, 0 } };
static const RadParamSpec pRmsQuantI8[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_STR("dtype"),
                                            P_F64O("wadd") };
static const RadOperandSpec oRmsQuantI8[] = { { "x", RAD_OPD_IN, 0 },
                                              { "residual", RAD_OPD_INOUT, 1 },
                                              { "w", RAD_OPD_WEIGHT, 0 },
                                              { "q", RAD_OPD_OUT, 0 },
                                              { "scale", RAD_OPD_OUT, 0 },
                                              { "out_bf16", RAD_OPD_OUT, 1 },
                                              { "w_post", RAD_OPD_WEIGHT, 1 },
                                              { "scalar", RAD_OPD_WEIGHT, 1 } };

static const RadOperandSpec oDequant[] = { { "q", RAD_OPD_IN, 0 }, { "scale", RAD_OPD_IN, 0 },
                                           { "y", RAD_OPD_OUT, 0 } };

/* --- gemm --- */
static const RadParamSpec pGemm[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_STR("dtype") };

/* pGemmNt/oGemmNt (gemm_nt_f32) reuse pGemm/oGemm above: the pair is byte-identical to
 * libr4d's, and the loader compares positionally. */
static const RadOperandSpec oGemm[] = { { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_WEIGHT, 0 },
                                        { "y", RAD_OPD_OUT, 0 } };

/* The biased form: the product with a per-column bias, then an optional activation and residual,
 * each rounded where a separate tensor would be. `res` is an optional input and may be `y`
 * itself, the stream updated in place. */
static const RadParamSpec pGemmBias[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_STR("dtype"),
                                          P_STRO("act") };
static const RadOperandSpec oGemmBias[] = { { "a", RAD_OPD_IN, 0 }, { "b", RAD_OPD_WEIGHT, 0 },
                                            { "bias", RAD_OPD_WEIGHT, 0 },
                                            { "res", RAD_OPD_IN, 1 }, { "y", RAD_OPD_OUT, 0 } };

/* --- elementwise --- */
static const RadParamSpec pGridEmbed[] = { P_INT("M"), P_INT("n"), P_INT("side"), P_STR("dtype") };
static const RadOperandSpec oGridEmbed[] = { { "x", RAD_OPD_INOUT, 0 },
                                             { "table", RAD_OPD_WEIGHT, 0 },
                                             { "coord", RAD_OPD_IN, 0 } };

/* `w_rb` is libr4d's (docs/OPS.md), accepted and ignored: an int8 weight's per-row scale plane
 * holds each block's value in every row, and the trellis forms that need it are IQR dtypes, which
 * no row here serves. */
static const RadParamSpec pGemmQ[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"),
                                       P_STR("dtype"), P_INTO("pb1"), P_INTO("pb2"),
                                       P_INTO("split_n"), P_INTO("w_rb") };
static const RadOperandSpec oGemmQ[] = { { "a", RAD_OPD_IN, 0 }, { "a_scale", RAD_OPD_IN, 1 },
                                         { "b", RAD_OPD_WEIGHT, 0 },
                                         { "b_scale", RAD_OPD_WEIGHT, 0 },
                                         { "y", RAD_OPD_OUT, 0 }, { "a_sum", RAD_OPD_IN, 1 },
                                         { "b_ref", RAD_OPD_WEIGHT, 1 } };

/* --- gated-delta-net --- */
static const RadParamSpec pGdnKktSolve[] = { P_INT("M"), P_INT("head_k"), P_CHUNK("chunk") };
static const RadOperandSpec oGdnKktSolve[] = { { "k", RAD_OPD_IN, 0 },
                                               { "beta", RAD_OPD_IN, 0 },
                                               { "g", RAD_OPD_IN, 0 },
                                               { "cu", RAD_OPD_IN, 0 },
                                               { "A", RAD_OPD_OUT, 0 } };

static const RadParamSpec pGdnChunkScan[] = { P_INT("M"), P_INT("head_k"), P_INT("head_v"),
                                              P_CHUNK("chunk"), P_F64O("scale"), P_INTO("state_fp16") };
static const RadOperandSpec oGdnChunkScan[] = { { "q", RAD_OPD_IN, 0 },
                                                { "k", RAD_OPD_IN, 0 },
                                                { "v", RAD_OPD_IN, 0 },
                                                { "A", RAD_OPD_IN, 0 },
                                                { "g", RAD_OPD_IN, 0 },
                                                { "beta", RAD_OPD_IN, 0 },
                                                { "h0", RAD_OPD_IN, 1 },
                                                { "cu", RAD_OPD_IN, 0 },
                                                { "o", RAD_OPD_OUT, 0 },
                                                { "ht", RAD_OPD_OUT, 1 },
                                                { "state_idx", RAD_OPD_IN, 1 } };

static const RadParamSpec pGdnGatedRmsnorm[] = { P_INT("M"), P_INT("channels"),
                                                P_F64("eps"), P_STR("act") };
static const RadOperandSpec oGdnGatedRmsnorm[] = { { "x", RAD_OPD_IN, 0 },
                                                  { "z", RAD_OPD_IN, 0 },
                                                  { "w", RAD_OPD_WEIGHT, 0 },
                                                  { "o", RAD_OPD_OUT, 0 } };

static const RadParamSpec pGdnGatedNormHq[] = { P_INT("M"), P_INT("n"), P_INT("head_dim"),
                                               P_INT("group"), P_F64("eps"), P_STR("dtype"),
                                               P_INTO("zgrp"), P_INTO("zblk") };
static const RadOperandSpec oGdnGatedNormHq[] = { { "x", RAD_OPD_IN, 0 },
                                                 { "z", RAD_OPD_IN, 0 },
                                                 { "w", RAD_OPD_WEIGHT, 0 },
                                                 { "q", RAD_OPD_OUT, 0 },
                                                 { "scale", RAD_OPD_OUT, 0 } };

static const RadParamSpec pGdnConvPrep[] = { P_INT("M"), P_INT("head_k"), P_INT("head_v"),
                                            P_CHUNK("chunk"), P_INT("conv_width"),
                                            P_F64O("l2_eps"), P_F64O("softplus_thr") };
static const RadOperandSpec oGdnConvPrep[] = { { "x", RAD_OPD_IN, 0 },
                                              { "w", RAD_OPD_WEIGHT, 0 },
                                              { "b", RAD_OPD_WEIGHT, 1 },
                                              { "A_log", RAD_OPD_WEIGHT, 0 },
                                              { "dt_bias", RAD_OPD_WEIGHT, 0 },
                                              { "conv_state", RAD_OPD_INOUT, 0 },
                                              { "cu", RAD_OPD_IN, 0 },
                                              { "a", RAD_OPD_IN, 1 },
                                              { "b_gate", RAD_OPD_IN, 1 },
                                              { "cache_idx", RAD_OPD_IN, 1 },
                                              { "has_init", RAD_OPD_IN, 1 },
                                              { "q", RAD_OPD_OUT, 0 },
                                              { "k", RAD_OPD_OUT, 0 },
                                              { "v", RAD_OPD_OUT, 0 },
                                              { "g", RAD_OPD_OUT, 0 },
                                              { "beta", RAD_OPD_OUT, 0 } };

static const RadParamSpec pGdnConvUpdate[] = { P_INT("q_len"), P_INT("head_k"), P_INT("head_v"),
                                              P_INT("conv_width"), P_INTO("max_query_len") };
static const RadOperandSpec oGdnConvUpdate[] = { { "x", RAD_OPD_IN, 0 },
                                                { "w", RAD_OPD_WEIGHT, 0 },
                                                { "b", RAD_OPD_WEIGHT, 1 },
                                                { "conv_state", RAD_OPD_INOUT, 0 },
                                                { "state_idx", RAD_OPD_IN, 0 },
                                                { "num_accepted", RAD_OPD_IN, 0 },
                                                { "cu", RAD_OPD_IN, 0 },
                                                { "q", RAD_OPD_OUT, 0 },
                                                { "k", RAD_OPD_OUT, 0 },
                                                { "v", RAD_OPD_OUT, 0 } };

static const RadParamSpec pPleConv[] = { P_INT("M"), P_INT("n"), P_INT("width"),
                                         P_INT("dilation"), P_STR("dtype") };
/* The optionals unite gdn_conv_prep's and gdn_conv_update's contracts: `has_init` serves
 * prefill, `num_accepted` decode. */
static const RadOperandSpec oPleConv[] = {
    { "x",            RAD_OPD_IN,     0 },
    { "w",            RAD_OPD_WEIGHT, 0 },
    { "conv_state",   RAD_OPD_INOUT,  0 },
    { "cu",           RAD_OPD_IN,     0 },
    { "resid",        RAD_OPD_IN,     1 },
    { "cache_idx",    RAD_OPD_IN,     1 },
    { "has_init",     RAD_OPD_IN,     1 },
    { "num_accepted", RAD_OPD_IN,     1 },
    { "y",            RAD_OPD_OUT,    0 },
};

static const RadParamSpec pExpertScale[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                             P_STR("dtype") };
static const RadOperandSpec oExpertScale[] = { { "ids", RAD_OPD_IN, 0 },
                                               { "w_e", RAD_OPD_WEIGHT, 0 },
                                               { "ew", RAD_OPD_INOUT, 0 } };

static const RadParamSpec pMoeRouterTopk[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                               P_INT("norm"), P_STR("dtype") };
static const RadOperandSpec oMoeRouterTopk[] = { { "logits", RAD_OPD_IN, 0 },
                                                 { "expert_ids", RAD_OPD_OUT, 0 },
                                                 { "expert_w", RAD_OPD_OUT, 0 } };

static const RadParamSpec pMoeScatter[] = { P_INT("M"), P_INT("n_expert"), P_INT("top_k"),
                                            P_INTO("expert_base"), P_STRO("protect") };
static const RadOperandSpec oMoeScatter[] = { { "expert_ids", RAD_OPD_IN, 0 },
                                              { "sorted_tok", RAD_OPD_OUT, 0 },
                                              { "expert_offset", RAD_OPD_OUT, 0 },
                                              { "expert_count", RAD_OPD_OUT, 0 } };
 /* moe_gather: fold the expert outputs back to the token. y_expert [Te, n], expert_w [Mw, top_k],
  * the slot table (sorted_tok), the optional shared arm, and y [M, n]. */
static const RadOperandSpec oMoeGather[] = { { "y_expert", RAD_OPD_IN, 0 },
                                              { "expert_w", RAD_OPD_IN, 0 },
                                              { "sorted_tok", RAD_OPD_IN, 0 },
                                              { "shared", RAD_OPD_IN, 1 },
                                              { "shared_gate", RAD_OPD_IN, 1 },
                                              { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pMoeGather[] = { P_INT("M"), P_INT("n"), P_INT("top_k"), P_STR("dtype"),
                                           P_STRO("act") };

/* moe_gemm_q: the grouped quantised GEMM, libref's schema. The experts are weight tables
 * (RAD_OPD_WTAB): the even experts in `w` / `w_scale`, the odd in `w_odd` / `w_odd_scale`, expert e
 * at entry e/2 (r3d_moe_tab.h). */
static const RadOperandSpec oMoeGemmQ[] = { { "a", RAD_OPD_IN, 0 },
                                            { "a_scale", RAD_OPD_IN, 1 },
                                            { "w", RAD_OPD_WTAB, 0 },
                                            { "w_scale", RAD_OPD_WTAB, 0 },
                                            { "sorted_tok", RAD_OPD_IN, 0 },
                                            { "expert_offset", RAD_OPD_IN, 0 },
                                            { "w_odd", RAD_OPD_WTAB, 0 },
                                            { "w_odd_scale", RAD_OPD_WTAB, 0 },
                                            { "y", RAD_OPD_OUT, 0 } };
/* `a_group` is an int8 activation's scale group, which no row here reads: the dtypes this library
 * serves take E4M3 or int4 activations, and the op's schema is one for every library. */
static const RadParamSpec pMoeGemmQ[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"),
                                          P_INT("top_k"), P_INT("group"), P_STR("a_order"),
                                          P_STR("dtype"), P_INTO("N_odd"), P_INTO("K_odd"),
                                          P_INTO("parts"), P_INTO("a_group") };

/* moe_gemm_q dtype w4a8: the grouped int4 GEMM (r3d_moe_gemm_w4a8.hip) is a row of the moe_gemm_q
 * op above (two's-complement int4 experts, a bf16 scale per (row, 128 of K), bf16 out; the int8 or
 * E4M3 activation arm is picked by `a`'s dtype). It reuses moe_gemm_q's nine-operand parity schema. */

 /* moe_gemm: the grouped GEMM with unquantised experts, libref's schema; one weight table. */
static const RadOperandSpec oMoeGemm[] = { { "a", RAD_OPD_IN, 0 },
                                          { "w", RAD_OPD_WTAB, 0 },
                                          { "sorted_tok", RAD_OPD_IN, 0 },
                                          { "expert_offset", RAD_OPD_IN, 0 },
                                          { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pMoeGemm[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("n_expert"),
                                         P_INT("top_k"), P_STR("a_order"), P_STR("dtype") };

static const RadParamSpec pGdnRecurrent[] = { P_INT("q_len"), P_INT("head_k"), P_INT("head_v"),
                                             P_INTO("n_head_v"), P_F64O("scale"), P_F64O("eps"),
                                             P_F64O("l2_eps"), P_F64O("softplus_thr"),
                                             P_STRO("act"), P_INTO("state_fp16"), P_STRO("state_form") };
static const RadOperandSpec oGdnRecurrent[] = { { "q", RAD_OPD_IN, 0 },
                                               { "k", RAD_OPD_IN, 0 },
                                               { "v", RAD_OPD_IN, 0 },
                                               { "a", RAD_OPD_IN, 0 },
                                               { "b", RAD_OPD_IN, 0 },
                                               { "A_log", RAD_OPD_WEIGHT, 0 },
                                               { "dt_bias", RAD_OPD_WEIGHT, 0 },
                                               { "state", RAD_OPD_INOUT, 0 },
                                               { "state_idx", RAD_OPD_IN, 0 },
                                               { "num_accepted", RAD_OPD_IN, 0 },
                                               { "cu", RAD_OPD_IN, 0 },
                                               { "z", RAD_OPD_IN, 1 },
                                               { "norm_w", RAD_OPD_WEIGHT, 1 },
                                               { "o", RAD_OPD_OUT, 0 },
                                               { "o_q", RAD_OPD_OUT, 1 },
                                               { "o_scale", RAD_OPD_OUT, 1 } };
static const RadParamSpec pKdaUpdate[] = {
    P_INT("M"), P_INT("heads"), P_INT("head_k"), P_INT("head_v"), P_F64("lower_bound"),
    P_F64("scale"), P_F64("l2_eps"), P_STR("mode"), P_STR("dtype"),
    P_STRO("state_form"),
};
static const RadOperandSpec oKdaUpdate[] = {
    {"q", RAD_OPD_IN, 0}, {"k", RAD_OPD_IN, 0}, {"v", RAD_OPD_IN, 0},
    {"a", RAD_OPD_IN, 0}, {"b", RAD_OPD_IN, 0}, {"A_log", RAD_OPD_WEIGHT, 0},
    {"dt_bias", RAD_OPD_WEIGHT, 0}, {"state", RAD_OPD_INOUT, 0}, {"state_idx", RAD_OPD_IN, 0},
    {"cu", RAD_OPD_IN, 0}, {"num_accepted", RAD_OPD_IN, 1}, {"has_init", RAD_OPD_IN, 1},
    {"o", RAD_OPD_OUT, 0},
};

/* gdn_conv_recurrent_update: the decode conv and the recurrent update as one kernel, the
 * convolution's operands then the recurrence's minus q, k and v (never materialised). */
static const RadParamSpec pGdnConvRecurrent[] = { P_INT("q_len"), P_INT("head_k"), P_INT("head_v"),
                                                 P_INT("conv_width"), P_INTO("n_head_v"),
                                                 P_F64O("scale"), P_F64O("eps"), P_F64O("l2_eps"),
                                                 P_F64O("softplus_thr"), P_STRO("act"),
                                                 P_INTO("state_fp16"), P_STRO("state_form") };
static const RadOperandSpec oGdnConvRecurrent[] = { { "x", RAD_OPD_IN, 0 },
                                                   { "conv_w", RAD_OPD_WEIGHT, 0 },
                                                   { "conv_b", RAD_OPD_WEIGHT, 1 },
                                                   { "conv_state", RAD_OPD_INOUT, 0 },
                                                   { "conv_idx", RAD_OPD_IN, 0 },
                                                   { "a", RAD_OPD_IN, 0 },
                                                   { "b", RAD_OPD_IN, 0 },
                                                   { "A_log", RAD_OPD_WEIGHT, 0 },
                                                   { "dt_bias", RAD_OPD_WEIGHT, 0 },
                                                   { "state", RAD_OPD_INOUT, 0 },
                                                   { "state_idx", RAD_OPD_IN, 0 },
                                                   { "num_accepted", RAD_OPD_IN, 0 },
                                                   { "cu", RAD_OPD_IN, 0 },
                                                   { "z", RAD_OPD_IN, 0 },
                                                   { "norm_w", RAD_OPD_WEIGHT, 0 },
                                                   { "o", RAD_OPD_OUT, 0 },
                                                   { "o_q", RAD_OPD_OUT, 1 },
                                                   { "o_scale", RAD_OPD_OUT, 1 } };

/* gdn_conv_recurrent_update_had: the same operands, `o_q` / `o_scale` required (they are
 * had_quant_act_i8g's codes of `o`), and the codes' scale group. */
static const RadParamSpec pGdnConvRecurrentHad[] = {
                                                   P_INT("q_len"), P_INT("head_k"), P_INT("head_v"),
                                                   P_INT("conv_width"), P_INTO("n_head_v"),
                                                   P_F64O("scale"), P_F64O("eps"), P_F64O("l2_eps"),
                                                   P_F64O("softplus_thr"), P_STRO("act"),
                                                   P_INTO("state_fp16"), P_STRO("state_form"),
                                                   P_INTO("group") };
static const RadOperandSpec oGdnConvRecurrentHad[] = { { "x", RAD_OPD_IN, 0 },
                                                   { "conv_w", RAD_OPD_WEIGHT, 0 },
                                                   { "conv_b", RAD_OPD_WEIGHT, 1 },
                                                   { "conv_state", RAD_OPD_INOUT, 0 },
                                                   { "conv_idx", RAD_OPD_IN, 0 },
                                                   { "a", RAD_OPD_IN, 0 },
                                                   { "b", RAD_OPD_IN, 0 },
                                                   { "A_log", RAD_OPD_WEIGHT, 0 },
                                                   { "dt_bias", RAD_OPD_WEIGHT, 0 },
                                                   { "state", RAD_OPD_INOUT, 0 },
                                                   { "state_idx", RAD_OPD_IN, 0 },
                                                   { "num_accepted", RAD_OPD_IN, 0 },
                                                   { "cu", RAD_OPD_IN, 0 },
                                                   { "z", RAD_OPD_IN, 0 },
                                                   { "norm_w", RAD_OPD_WEIGHT, 0 },
                                                   { "o", RAD_OPD_OUT, 0 },
                                                   { "o_q", RAD_OPD_OUT, 0 },
                                                   { "o_scale", RAD_OPD_OUT, 0 } };


/* ================================================================== the vocab and the sampler
 *
 * The vocab and sampler schemas, transcribed from libref. `r3d_selftest`'s schema phase compares
 * every declared schema against libref's, slot for slot, when the library loads, so a misspelled
 * key is caught at load rather than when a row first uses it.
 *
 * The sampler's behaviour is fully specified: the candidate set, the top-k tie-break, the order of
 * the renormalisations and the seeded draw are all observable, and libref is the oracle for each.
 */

/* The gated MLP's activation as llama-family graphs emit it: one fused [M, 2n] projection, the
 * gate half activated and multiplied by the other half. `act_mul` is the same arithmetic over two
 * operands; an architecture emits one op or the other. */
static const RadOperandSpec oGateUp[] = { { "gate_up", RAD_OPD_IN, 0 }, { "y", RAD_OPD_OUT, 0 } };

/* The gated MLP band: one weight of N = 2n rows (gate first, then up), with the GEMM, the gate and
 * the output quantiser in one launch so the [M, N] intermediate is never stored. `act` is a required
 * string because an implementation compiles one gate; an optional key would let a row resolve for a
 * gate it does not have. */
static const RadParamSpec pGemmQGated[] = { P_INT("M"), P_INT("N"), P_INT("K"),
                                            P_INT("group"), P_STR("dtype"), P_STR("act") };
/* The vocabulary projection's schema, slot for slot with the reference's: the geometry is the
 * head's (`n_vocab` x `n_embd`), the dtype is the activation's, and `logits` is f32, which is why
 * the op is separate from gemm_nt. */
static const RadParamSpec pLogits[] = { P_INT("M"), P_INT("n_vocab"), P_INT("n_embd"),
                                        P_STR("dtype") };
static const RadOperandSpec oLogits[] = {
    { "x", RAD_OPD_IN, 0 }, { "lm_head", RAD_OPD_WEIGHT, 0 }, { "logits", RAD_OPD_OUT, 0 } };

static const RadOperandSpec oGemmQGated[] = {
    { "a", RAD_OPD_IN, 0 },        { "a_scale", RAD_OPD_IN, 0 },
    { "b", RAD_OPD_WEIGHT, 0 },    { "b_scale", RAD_OPD_WEIGHT, 0 },
    { "q", RAD_OPD_OUT, 0 },       { "scale", RAD_OPD_OUT, 0 },
    { "out_bf16", RAD_OPD_OUT, 1 },
};

/* DFlash2's candidate walk: a draft block of `steps` proposals with `top_k` candidates each,
 * rescored against the target's rank-d codebooks. `anchor_stride` is optional and defaults to 1.
 * `hp` is an optional per-edge scaling of the predecessor's codebook row; absent, it is a plane of
 * ones, as the reference computes. */
static const RadParamSpec pDflashSelect[] = { P_INT("M"), P_INT("steps"), P_INT("top_k"),
                                              P_INT("rank"), P_INT("n_vocab"),
                                              P_INTO("anchor_stride"), P_STR("dtype") };
static const RadOperandSpec oDflashSelect[] = {
    { "cand", RAD_OPD_IN, 0 },    { "unary", RAD_OPD_IN, 0 },
    { "hp", RAD_OPD_IN, 1 },      { "anchor", RAD_OPD_IN, 0 },
    { "pred", RAD_OPD_WEIGHT, 0 },{ "succ", RAD_OPD_WEIGHT, 0 },
    { "tokens", RAD_OPD_OUT, 0 },
};

/* Paged attention. `block_size` is RAD_DERIVED (the container's cache geometry, admitted by a
 * constraint that only refuses its absence). `n_head`, `max_seqs`, `max_ctx` and `prefill_fp8` are
 * listed because a plugin's scratch hook sizes the split-KV buffer from them and both libraries must
 * agree on the parameter list; libref ignores them. */
static const RadParamSpec pAttnPaged[] = {
    P_INT("q_len"), P_INT("head_dim"), P_INT("gqa"), P_INTD("block_size"), P_INT("causal"),
    P_INT("window"), P_STR("q_dtype"), P_STR("kv_dtype"), P_F64O("scale"), P_INTO("max_ctx"),
    P_INTO("n_head"), P_INTO("max_seqs"), P_INTO("prefill_fp8")
};
static const RadOperandSpec oAttnPaged[] = {
    { "q", RAD_OPD_IN, 0 }, { "kv_cache", RAD_OPD_IN, 0 }, { "block_table", RAD_OPD_IN, 0 },
    { "seqused", RAD_OPD_IN, 0 }, { "k_descale", RAD_OPD_IN, 1 }, { "v_descale", RAD_OPD_IN, 1 },
    { "cu_seqlens", RAD_OPD_IN, 1 }, { "out", RAD_OPD_OUT, 0 }
};

static const RadParamSpec pMlaProject[] = { P_INT("M"),          P_INT("heads"), P_INT("head_dim"),
                                             P_INT("latent_dim"), P_STR("mode"),  P_STR("dtype") };

static const RadParamSpec pMlaAttnPaged[] = { P_INT("M"), P_INT("heads"), P_INT("latent_dim"),
                                              P_INT("rope_dim"), P_INT("block_size"), P_F64("scale"),
                                              P_STR("dtype") };
static const RadOperandSpec oMlaAttnPaged[] = { { "q", RAD_OPD_IN, 0 },
                                                { "q_rope", RAD_OPD_IN, 1 },
                                                { "kv", RAD_OPD_IN, 0 },
                                                { "block_table", RAD_OPD_IN, 0 },
                                                { "seqused", RAD_OPD_IN, 0 },
                                                { "cu_seqlens", RAD_OPD_IN, 0 },
                                                { "gate", RAD_OPD_IN, 1 },
                                                { "out", RAD_OPD_OUT, 0 } };
/* Dense, unpaged attention over cu_seqlens segments (the vision tower's), transcribed from libref:
 * the same keys, types, requiredness and operand roles, in the same order. `max_seqlen`, `window`,
 * `cu_div`, `block` and `window_right` are optional and ignored by a caller that does not need
 * them; the vision band states `max_seqlen` only. */
static const RadParamSpec pAttnDense[] = {
    P_INT("M"), P_INT("head_dim"), P_INT("gqa"), P_INT("causal"), P_STR("dtype"),
    P_F64O("scale"), P_INTO("max_seqlen"), P_INTO("window"), P_INTO("cu_div"),
    P_INTO("block"), P_INTO("window_right")
};
static const RadOperandSpec oAttnDense[] = {
    { "q", RAD_OPD_IN, 0 }, { "k", RAD_OPD_IN, 0 }, { "v", RAD_OPD_IN, 0 },
    { "cu_seqlens", RAD_OPD_IN, 0 }, { "out", RAD_OPD_OUT, 0 }
};

/* The row's top-R over a vocabulary slice, and the sampler's stages. The history stages write this
 * rank's columns and read global ids, so they carry `vocab_off`; the narrowing stages work on the
 * candidate set the top-k produced and need only its width. */
static const RadParamSpec pRowTopk[]  = { P_INT("M"), P_INT("N"), P_INT("R"), P_INTO("vocab_off"),
                                          P_STR("dtype") };
static const RadOperandSpec oRowTopk[] = { { "x", RAD_OPD_IN, 0 }, { "idx", RAD_OPD_OUT, 0 },
                                           { "val", RAD_OPD_OUT, 0 },
                                           { "pairs", RAD_OPD_OUT, 1 } };
static const RadParamSpec pMVocab[]  = { P_INT("M"), P_INT("n_vocab") };
static const RadParamSpec pMCand[]   = { P_INT("M"), P_INT("n_cand") };
static const RadParamSpec pPenal[]   = { P_INT("M"), P_INT("n_vocab"), P_INTO("vocab_off") };
static const RadParamSpec pTopk[]    = { P_INT("M"), P_INT("n_vocab"), P_INTO("vocab_off") };
static const RadParamSpec pLogprobs[] = { P_INT("M"), P_INT("n_vocab"), P_INT("k"),
                                          P_INTO("vocab_off") };
static const RadOperandSpec oTemp[]      = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 } };
static const RadOperandSpec oMask[]      = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "bitmask", RAD_OPD_IN, 0 } };
static const RadOperandSpec oPenalties[] = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "history", RAD_OPD_IN, 0 } };
static const RadOperandSpec oDry[]       = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "history", RAD_OPD_IN, 0 },
                                             { "breakers", RAD_OPD_IN, 1 } };
static const RadOperandSpec oNgramBan[]  = { { "logits", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "history", RAD_OPD_IN, 0 },
                                             { "ngram", RAD_OPD_IN, 0 },
                                             { "whitelist", RAD_OPD_IN, 1 } };
static const RadOperandSpec oTopk[]      = { { "logits", RAD_OPD_IN, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "cand_idx", RAD_OPD_OUT, 0 },
                                             { "cand_val", RAD_OPD_OUT, 0 },
                                             { "pairs", RAD_OPD_OUT, 1 } };
static const RadOperandSpec oArgmax[]    = { { "logits", RAD_OPD_IN, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "token", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oLogprobs[]  = { { "logits", RAD_OPD_IN, 0 },
                                             { "params", RAD_OPD_IN, 1 },
                                             { "stats", RAD_OPD_OUT, 0 } };
static const RadOperandSpec oNarrow[]    = { { "cand_idx", RAD_OPD_INOUT, 0 },
                                             { "cand_val", RAD_OPD_INOUT, 0 },
                                             { "params", RAD_OPD_IN, 0 } };
static const RadOperandSpec oPick[]      = { { "cand_idx", RAD_OPD_IN, 0 },
                                             { "cand_val", RAD_OPD_IN, 0 },
                                             { "params", RAD_OPD_IN, 0 },
                                             { "token", RAD_OPD_OUT, 0 } };

/* The MTP head's entry (`ngroup` hc for a grouped norm, 1 for a flat one) and the exact rescoring
 * of a coarse head's candidates. Both transcribe libref's schemas slot for slot, optionality
 * included: the loader compares them. */
static const RadParamSpec pMtpEnter[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_F64("eps"),
                                          P_STR("dtype"), P_F64O("wadd"), P_INTO("ngroup"),
                                          P_STRO("fc") };
static const RadOperandSpec oMtpEnter[] = { { "h", RAD_OPD_IN, 0 },      { "e", RAD_OPD_IN, 0 },
                                            { "w_h", RAD_OPD_WEIGHT, 0 }, { "w_e", RAD_OPD_WEIGHT, 0 },
                                            { "fc_h", RAD_OPD_WEIGHT, 0 }, { "fc_e", RAD_OPD_WEIGHT, 0 },
                                            { "x", RAD_OPD_OUT, 0 } };
static const RadParamSpec pLogitRerank[] = { P_INT("M"), P_INT("R"), P_INT("n_vocab"),
                                             P_INT("n_embd"), P_INTO("vocab_off"),
                                             P_STR("dtype") };
static const RadOperandSpec oLogitRerank[] = { { "x", RAD_OPD_IN, 0 },
                                               { "head", RAD_OPD_WEIGHT, 0 },
                                               { "idx_in", RAD_OPD_IN, 0 },
                                               { "idx", RAD_OPD_OUT, 0 },
                                               { "val", RAD_OPD_OUT, 0 },
                                               { "pairs", RAD_OPD_OUT, 1 } };


/* The fused norm-quantiser from the model unit, libref's slot for slot. `residual` is INOUT because
 * the sum it stores is the one the variance reads. */
static const RadParamSpec pRmsQuantFp8[] = { P_INT("M"), P_INT("n"), P_F64("eps"), P_INT("group"),
                                             P_STR("dtype"), P_F64O("wadd") };
/* `residual` is optional because the op has a no-residual form (the launcher's `res` pointer is 0);
 * libref declares it INOUT optional and the loader compares the flag. */
static const RadOperandSpec oRmsQuantFp8[] = {
    { "x", RAD_OPD_IN, 0 },        { "residual", RAD_OPD_INOUT, 1 }, { "w", RAD_OPD_WEIGHT, 0 },
    { "q", RAD_OPD_OUT, 0 },       { "scale", RAD_OPD_OUT, 0 },      { "out_bf16", RAD_OPD_OUT, 1 }
};

/* --- new Tier-1 ops --- */
static const RadParamSpec pDequantW4[] = { P_INT("N"), P_INT("K"), P_INT("group"), P_STR("dtype"),
                                           P_INT("twos") };
static const RadOperandSpec oDequantW4[] = { { "wq", RAD_OPD_WEIGHT, 0 },
                                             { "ws", RAD_OPD_WEIGHT, 0 },
                                             { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pRowTopkMerge[] = { P_INT("M"), P_INT("R"), P_INT("world_size"),
                                              P_STR("dtype") };
static const RadOperandSpec oRowTopkMerge[] = { { "gathered", RAD_OPD_IN, 0 },
                                                { "idx", RAD_OPD_OUT, 0 },
                                                { "val", RAD_OPD_OUT, 0 } };
static const RadParamSpec pNgramIds[] = { P_INT("M"), P_INT("heads"), P_INT("ngram"), P_INT("eos") };
static const RadOperandSpec oNgramIds[] = { { "tok", RAD_OPD_IN, 0 },
                                            { "state", RAD_OPD_INOUT, 0 },
                                            { "cu", RAD_OPD_IN, 0 },
                                            { "mult", RAD_OPD_WEIGHT, 0 },
                                            { "vocab_sizes", RAD_OPD_WEIGHT, 0 },
                                            { "offsets", RAD_OPD_WEIGHT, 0 },
                                            { "cache_idx", RAD_OPD_IN, 1 },
                                            { "has_init", RAD_OPD_IN, 1 },
                                            { "num_accepted", RAD_OPD_IN, 1 },
                                            { "ids", RAD_OPD_OUT, 0 },
                                            { "ahead_ids", RAD_OPD_OUT, 1 } };
static const RadParamSpec pGemmQBias[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"), P_STR("dtype") };
static const RadOperandSpec oGemmQBias[] = { { "a", RAD_OPD_IN, 0 }, { "a_scale", RAD_OPD_IN, 0 },
                                             { "b", RAD_OPD_WEIGHT, 0 },
                                             { "b_scale", RAD_OPD_WEIGHT, 0 },
                                             { "bias", RAD_OPD_WEIGHT, 0 },
                                             { "y", RAD_OPD_OUT, 0 } };
static const RadParamSpec pGemmQShort[] = { P_INT("M"), P_INT("N"), P_INT("K"), P_INT("group"),
                                            P_STR("dtype"), P_INTO("pb1"), P_INTO("pb2"),
                                            P_INTO("split_n"), P_INTO("w_rb") };
static const RadOperandSpec oGemmQShort[] = { { "a", RAD_OPD_IN, 0 }, { "a_scale", RAD_OPD_IN, 1 },
                                              { "b", RAD_OPD_WEIGHT, 0 },
                                              { "b_scale", RAD_OPD_WEIGHT, 0 },
                                              { "y", RAD_OPD_OUT, 0 },
                                              { "a_sum", RAD_OPD_IN, 1 },
                                              { "b_ref", RAD_OPD_WEIGHT, 1 } };
static const RadParamSpec pMoeGemmQMod4[] = { P_INT("M"), P_INT("N"), P_INT("K"),
                                              P_INT("n_expert"), P_INT("top_k"), P_INT("group"),
                                              P_STR("a_order"), P_STR("dtype"), P_INTO("N_1"),
                                              P_INTO("K_1"), P_INTO("N_2"), P_INTO("K_2"),
                                              P_INTO("N_3"), P_INTO("K_3"), P_INTO("parts"),
                                              P_INTO("a_group") };
static const RadOperandSpec oMoeGemmQMod4[] = { { "a", RAD_OPD_IN, 0 }, { "a_scale", RAD_OPD_IN, 1 },
                                                { "w", RAD_OPD_WTAB, 0 }, { "w_scale", RAD_OPD_WTAB, 0 },
                                                { "sorted_tok", RAD_OPD_IN, 0 }, { "expert_offset", RAD_OPD_IN, 0 },
                                                { "w_1", RAD_OPD_WTAB, 0 }, { "w_1_scale", RAD_OPD_WTAB, 0 },
                                                { "w_2", RAD_OPD_WTAB, 0 }, { "w_2_scale", RAD_OPD_WTAB, 0 },
                                                { "w_3", RAD_OPD_WTAB, 0 }, { "w_3_scale", RAD_OPD_WTAB, 0 },
                                                { "y", RAD_OPD_OUT, 0 } };

/* --- Per-Layer Embedding and HyperConnections (Qwen4-Exp) ---
 *
 * The schemas are libref's for the ops it declares; ngram_hint and the two collective writes are
 * libr4d's device-only ops, copied slot for slot so a build that loads both plugins agrees. */
static const RadOperandSpec oEmbedQ[] = { { "tokens", RAD_OPD_IN, 0 },
                                          { "wte", RAD_OPD_WEIGHT, 0 },
                                          { "scale", RAD_OPD_WEIGHT, 1 },
                                          { "x", RAD_OPD_OUT, 0 },
                                          { "ahead", RAD_OPD_IN, 1 } };
static const RadParamSpec pPleGate[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_F64("eps"),
                                         P_STR("dtype"), P_F64O("wadd"), P_F64O("gate_eps") };
static const RadOperandSpec oPleGate[] = { { "k", RAD_OPD_IN, 0 },
                                           { "q", RAD_OPD_IN, 0 },
                                           { "v", RAD_OPD_IN, 0 },
                                           { "w_key", RAD_OPD_WEIGHT, 0 },
                                           { "w_query", RAD_OPD_WEIGHT, 0 },
                                           { "w_conv", RAD_OPD_WEIGHT, 0 },
                                           { "gv", RAD_OPD_OUT, 0 },
                                           { "gvn", RAD_OPD_OUT, 0 } };
static const RadParamSpec pHcRead[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_INT("lowrank"),
                                        P_F64("eps"), P_STR("dtype"), P_F64O("wadd"),
                                        P_INTO("inject"), P_INTO("group"), P_INTO("rotate"),
                                        P_STRO("mix") };
static const RadOperandSpec oHcRead[] = { { "h", RAD_OPD_IN, 0 },
                                          { "w", RAD_OPD_WEIGHT, 0 },
                                          { "mix_down", RAD_OPD_WEIGHT, 0 },
                                          { "mix_up", RAD_OPD_WEIGHT, 0 },
                                          { "inject_w", RAD_OPD_WEIGHT, 1 },
                                          { "x", RAD_OPD_OUT, 0 },
                                          { "inj", RAD_OPD_OUT, 1 },
                                          { "q", RAD_OPD_OUT, 1 },
                                          { "scale", RAD_OPD_OUT, 1 },
                                          { "rq", RAD_OPD_OUT, 1 },
                                          { "rscale", RAD_OPD_OUT, 1 } };
static const RadParamSpec pHcWrite[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_STR("dtype") };
static const RadOperandSpec oHcWrite[] = { { "y", RAD_OPD_IN, 0 }, { "inj", RAD_OPD_IN, 0 },
                                           { "h", RAD_OPD_INOUT, 0 } };
static const RadOperandSpec oHcEnter[] = { { "x", RAD_OPD_IN, 0 }, { "h", RAD_OPD_OUT, 0 } };
static const RadParamSpec pHcProject[] = { P_INT("M"), P_INT("n"), P_INT("hc"), P_F64("scale"),
                                           P_STR("dtype") };
static const RadOperandSpec oHcProject[] = { { "h", RAD_OPD_INOUT, 0 }, { "v", RAD_OPD_IN, 0 },
                                             { "on", RAD_OPD_IN, 0 } };
static const RadParamSpec pNgramHint[] = { P_INT("M"), P_INT("heads"), P_INT("ngram"),
                                           P_INT("eos"), P_INT("hist") };
static const RadOperandSpec oNgramHint[] = { { "tok", RAD_OPD_IN, 0 },
                                             { "pos", RAD_OPD_IN, 0 },
                                             { "cu", RAD_OPD_IN, 0 },
                                             { "mult", RAD_OPD_WEIGHT, 0 },
                                             { "vocab_sizes", RAD_OPD_WEIGHT, 0 },
                                             { "offsets", RAD_OPD_WEIGHT, 0 },
                                             { "cache_idx", RAD_OPD_IN, 1 },
                                             { "tail", RAD_OPD_INOUT, 0 },
                                             { "ids", RAD_OPD_OUT, 0 } };
static const RadParamSpec pArHcWrite[] = { P_INT("M"), P_INT("n"), P_INT("hc"),
                                           P_INT("world_size"), P_STR("dtype"), P_INTO("wire") };
static const RadOperandSpec oArHcWrite[] = { { "y", RAD_OPD_INOUT, 0 }, { "inj", RAD_OPD_IN, 0 },
                                             { "h", RAD_OPD_INOUT, 0 } };
static const RadParamSpec pArGatherHcWrite[] = { P_INT("M"), P_INT("n"), P_INT("hc"),
                                                 P_INT("top_k"), P_INT("world_size"),
                                                 P_STR("dtype"), P_STRO("act"), P_INTO("wire") };
static const RadOperandSpec oArGatherHcWrite[] = {
    { "y_expert", RAD_OPD_IN, 0 },  { "expert_w", RAD_OPD_IN, 0 },
    { "sorted_tok", RAD_OPD_IN, 0 }, { "shared", RAD_OPD_IN, 1 },
    { "shared_gate", RAD_OPD_IN, 1 }, { "y", RAD_OPD_INOUT, 0 },
    { "inj", RAD_OPD_IN, 0 },       { "h", RAD_OPD_INOUT, 0 }
};

/* --- the tensor-parallel collectives (schema is libr4d's, slot for slot) --- */
static const RadParamSpec pAllReduce[] = {
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    /* The message in elements. A band resolves at its upper bound, so every bound the rows below
     * state is a multiple of the largest DIV they use (128), or the band would have no row. */
    { "numel",      RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    { "exact",      RAD_P_INT, RAD_REQUIRED },
    { "min_bytes",  RAD_P_INT, RAD_OPTIONAL },
    { "hops",       RAD_P_INT, RAD_OPTIONAL },
    { "overlap",    RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oAllReduce[] = {
    { "x", RAD_OPD_INOUT, 0 },
    { "y", RAD_OPD_OUT,   1 },
};
static const RadParamSpec pAllGather[] = {
    { "world_size", RAD_P_INT, RAD_REQUIRED },
    { "numel",      RAD_P_INT, RAD_REQUIRED },
    { "dtype",      RAD_P_STR, RAD_REQUIRED },
    { "row",        RAD_P_INT, RAD_OPTIONAL },
};
static const RadOperandSpec oAllGather[] = {
    { "x", RAD_OPD_IN,  0 },
    { "y", RAD_OPD_OUT, 0 },
};

static const RadOpSchema kSchemas[] = {
    SCHEMA("rmsnorm", pRmsNorm, oRmsNorm,
           "y = x * rsqrt(mean(x^2) + eps) * (w + wadd) * mult. `wadd` is GemmaRMSNorm's `1 +`, "
           "added to the gain in f32; `mult` scales the normed row before its one rounding. No "
           "gain operand at all means a gain of 1 and a `wadd` of 0."),
    SCHEMA("layernorm", pLayernorm, oLayernorm,
           "y = (x - mean) / sqrt(var + eps) * w + b, the mean and the variance taken in two "
           "passes over the row."),
    SCHEMA("grid_embed", pGridEmbed, oGridEmbed,
           "x[m] += the side x side table resampled at the row's grid coordinate, bilinearly with "
           "the grid's corners on the table's (align_corners), edge taps clamped; `coord` is "
           "[4, M] -- row, column, grid height, grid width. The taps are summed in f32 and "
           "rounded to x's dtype, then added to x and rounded again."),
    SCHEMA("add", pAdd, oBinary,
           "y = a + b. `b` carrying exactly n elements against an output of M rows broadcasts "
           "along rows, and nothing else broadcasts at all. With `div`, y = rd(rd(a + b) / div)."),
    SCHEMA("mul", pMnDtype, oBinary, "y = a * b, with the same one broadcast rule as `add`."),
    SCHEMA("act_mul", pActMul, oBinary,
           "y = act(a) * b over two [M, n] operands, `act` naming silu, gelu (the exact erf form), "
           "gelu_tanh, sigmoid or quick_gelu. The feed-forward's gated middle."),
    SCHEMA("silu", pMnDtype, oUnary, "y = x * sigmoid(x)."),
    SCHEMA("gelu", pMnDtype, oUnary,
           "y = 0.5 * x * (1 + erf(x / sqrt(2))), the exact form; the tanh approximation is a "
           "different op name."),
    SCHEMA("sigmoid", pMnDtype, oUnary, "y = 1 / (1 + exp(-x))."),
    SCHEMA("softmax", pMnDtype, oUnary, "Row-wise, f32, with the maximum subtracted."),
    SCHEMA("cast", pCast, oUnary,
           "y = x, from the dtype `from` to the dtype `to`, one rounding on the way. An integer "
           "pair goes through without touching f32, so ids above 2^24 survive."),
    SCHEMA("scale", pScale, oScale,
           "y = x * s, `s` one element broadcast over the whole output. A row of scales is "
           "`mul_weight`."),
    SCHEMA("fill", pFill, oFill, "y = value over [M, n]. The output is the only operand."),
    SCHEMA("rope", pRope, oRope,
           "Rotary positions applied in place to a fused qkv row: q heads, then the key heads, the "
           "value heads untouched. `mode` is neox, gptj, mrope, imrope or axial; the yarn, llama3 "
           "and ladder parameters adjust the frequency ladder."),
    SCHEMA("kv_store", pKvStore, oKvStore,
           "K and V into a paged cache at slot_mapping's slots, converting to the cache's dtype. A "
           "negative slot is skipped; the cache is [blocks, kv_heads, block_size, 2*head_dim]."),
    SCHEMA("rope_table", pRopeTable, oRopeTable,
           "The cos/sin plane a fused prologue reads: cos in the first rot/2 values of a table "
           "row, sin in the second, at each position's own row. Only rows named by positions are "
           "written; a negative or past-the-table position writes nothing."),
    SCHEMA("embed_lookup", pEmbed, oEmbed,
           "x[r] = wte[tokens[r] - vocab_offset] * wscale * mult, the id global and the table "
           "sharded by `vocab_offset`. An id outside the shard's range yields a zero row."),
    SCHEMA("logits_gemm_vocab", pLogitsVocab, oLogitsVocab,
           "logits[M, n_vocab] = x[M, n_embd] @ lm_head^T over a head stored as width classes of "
           "the vocabulary's rows: class i is a weight of rows<i> rows on this rank (w<i> its "
           "codes, s<i> its scales) and ids<i> the global id of each, whose logit it writes; ids "
           "outside [vocab_offset, vocab_offset + n_vocab) are dropped. vrow is the inverse."),
    SCHEMA("gather_rows", pMnDtype, oGather, "y[r] = x[idx[r]], rows of n elements."),
    SCHEMA("scatter_rows", pMnDtype, oScatter, "x[idx[r]] = v[r], rows of n elements."),
    SCHEMA("quant_act_fp8", pQuant, oQuantFp8,
           "Symmetric per-(row, group) E4M3: scale = amax * (1/448) with the reciprocal "
           "multiplied, not divided; an all-zero group gets scale 1 and the codes 0."),
    SCHEMA("quant_act_i8", pQuant, oQuant,
           "Symmetric int8 with one scale per row, max(amax, 1e-8) / 127, codes round-to-nearest-"
           "ties-to-even and saturating, stored in the step order the int8 WMMA reads. `asum` takes "
           "the group-major row sums the zero-pointed GEMMs reduce."),
    SCHEMA("had_quant_act_i8", pQuant, oQuantFp8,
           "A block Hadamard along K (width `group`), then per-row symmetric int8: the rotation "
           "shrinks the row maximum, so int8 beats e4m3 on these activations."),
    SCHEMA("gelu_mul_quant_i8", pMnDtype, oGeluMulQuantI8,
           "gelu(gate) * up over a fused [M, 2n] projection, then per-row symmetric int8: the "
           "tanh-form product rounded to bf16, the codes and scale taken over those values."),
    SCHEMA("kv_len_trim", pKvLenTrim, oKvLenTrim,
           "A batch's per-sequence context lengths less `trim`, floored at zero: the prefill "
           "overlap's slice sees its own last row, not the step's. `trim` is the tail's extent."),
    SCHEMA("qsa_tail_store", pQsaTailStore, oQsaTailStore,
           "The QSA indexer's tail: the raw index keys of tokens that have not completed a block, "
           "into a per-layer linear KV state. Position p lands at row p % ring; only the last "
           "ring tokens of a sequence's chunk write."),
    SCHEMA("qsa_block_key", pQsaBlockKey, oQsaBlockKey,
           "The QSA indexer's compressed key: one vector per `ratio` rows of k. The block's mean "
           "is accumulated in f32 and rounded to bf16 before the norm; the RMS norm uses gain "
           "(wadd + w); the key is rotated at the block's first-token position (`pos[b]` under "
           "mode/sections, or `pos0 + b*ratio`) with YaRN when given. `page[b]` is the output row; "
           "a negative entry writes nothing."),
    SCHEMA("qsa_score", pQsaScore, oQsaScore,
           "The QSA indexer's block score: score[m][b] = (sum over heads of relu(q[m][h] . "
           "bk[bt[seq(m)][b]])) / sqrt(n), for b below min(nc[seq], the table's and the score "
           "plane's widths). The relu is per (block, head) and the sum is over the heads, in "
           "that order. `bt` holds page ids; a negative entry scores -inf, never zero."),
    SCHEMA("qsa_select", pQsaSelect, oQsaSelect,
           "The QSA selection: the topk highest-scoring complete blocks a query has passed, as "
           "page ids, ties to the lower block index, then the partial page holding the query."),
    SCHEMA("qsa_work", pQsaWork, oQsaWork,
           "The blocks completing in this batch: per block its `ratio` raw keys (this step's rows, "
           "or the tail's at q % ring) staged contiguously, its block-key page and its first-token "
           "position; slots past the count get page -1. nc is each sequence's complete-block "
           "count. Under M-RoPE an optional `rope` gives each block's first-token components."),
    SCHEMA("gated_had_quant_i8", pGatedHadQuantI8, oGatedHadQuantI8,
           "A gated product (mode 0 silu(a)*b, mode 1 sigmoid(a)*b), a block Hadamard along n, "
           "then per-row symmetric int8: the rotation shrinks the row maximum, so int8 beats e4m3."),
    SCHEMA("had_quant_act_fp8", pQuant, oQuantFp8,
           "A 128-wide block Hadamard, then per-group E4M3: the rotation shrinks the group maximum "
           "so e4m3's exponent bits carry the outliers better. A separate op, not a parameter."),
    SCHEMA("gated_had_quant_fp8", pGatedQuant, oGatedQuant,
           "A gated product (silu or sigmoid), a 128-wide block Hadamard, then per-group E4M3: the "
           "rotation shrinks the group maximum so e4m3's exponent bits carry the outliers better."),
    SCHEMA("quant_cols_fp8", pMnDtype, oQuantCols,
           "E4M3 codes with one scale per column over every row: an attention's V for its fp8 P.V, "
           "the group running down a column. Two passes over the rows, split into chunks."),
    SCHEMA("had_quant_act_i8g", pQuant, oQuantFp8,
           "A 128-wide block Hadamard, then per-group int8: the rotated form for the int8 trunk "
           "GEMMs, the codes row-major (the i8a8 dot2 GEMM reads stored order)."),
    SCHEMA("gated_had_quant_i8g", pGatedQuant, oGatedQuant,
           "A gated product (silu or sigmoid), a 128-wide block Hadamard, then per-group int8: the "
           "rotated int8 down projection, the codes row-major for the i8a8 dot2 GEMM."),
    SCHEMA("gate_had_quant_i8g", pGatedQuant, oGateQuant,
           "The same gated product with the gate in its own operand rather than the first half of "
           "a fused gate_up plane: act(gate) * x, a 128-wide block Hadamard, then per-group int8 "
           "for an out projection stored rotated (i8a8h). A group of 256 is IQ8R's: two independent "
           "128-wide rotations under one amax and scale."),
    SCHEMA("rmsnorm_quant_i8", pRmsQuantI8, oRmsQuantI8,
           "The RMS norm and the per-row int8 quantiser in one pass, optionally with the residual "
           "junction in front (a post-norm, the residual add, a scalar): the sandwich-norm chain "
           "as one dispatch, byte-identical to the unfused chain."),
    SCHEMA("quant_act_i8g", pQuant, oQuantFp8,
           "The same int8 code at `group` granularity: one f32 scale per (row, group), amax / 127, "
           "1 for an all-zero group, codes row-major."),
    SCHEMA("gated_quant_fp8", pGatedQuant, oGatedQuant,
           "act(gate) * up over a fused [M, 2n] projection, quantised in the same pass: the codes "
           "are taken over the rounded product, which `out_bf16` receives when present."),
    SCHEMA("dequant", pQuant, oDequant,
           "y = code * scale, the read of a container's own quantised weight: `group`-wide f32 "
           "scales against E4M3, E5M2 or int8 codes."),
    SCHEMA("gemm_nt", pGemm, oGemm, "y[M,N] = a[M,K] @ b[N,K]^T, f32 accumulate."),
    SCHEMA("gemm_nt_f32", pGemm, oGemm,
           "BF16 activations times F32 weights, F32 accumulation and output; preserves routing "
           "and index gate precision."),
    SCHEMA("gemm_nt_bias", pGemmBias, oGemmBias,
           "gemm_nt with a per-column bias added in f32 and rounded to y's dtype; then, "
           "optionally, `act` of that (none, gelu exact, gelu_tanh) rounded, then `res` added and "
           "rounded. Each step is what a separate tensor would hold."),
    SCHEMA("gemm_nt_q", pGemmQ, oGemmQ,
           "The same product against a quantised weight. `dtype` names the scheme: fp8a16 a bf16 "
           "activation against E4M3 codes, fp8a8 quant_act_fp8's codes and scales, i8a8 int8 on "
           "both axes. `group` is the activation's scale width; `pb1` and `pb2` the weight's block, "
           "which the container states. `a_sum` and `b_ref` are the row sums a zero-pointed form "
           "reduces."),

    SCHEMA("gdn_kkt_solve", pGdnKktSolve, oGdnKktSolve,
           "A = (I + strict_lower(diag(beta) K K^T e^{g_i - g_j}))^-1 per 64-chunk: the gram once "
           "per k head, the blocked inverse 16 -> 32 -> 64."),

    SCHEMA("gdn_chunk_scan", pGdnChunkScan, oGdnChunkScan,
           "The gated delta net's chunked scan: the chunk's output and its closing state, the "
           "recurrent state held in the wave matrix across the chunk loop."),

    SCHEMA("gdn_gated_rmsnorm", pGdnGatedRmsnorm, oGdnGatedRmsnorm,
            "The gated RMS norm a gated-delta-net layer applies to its own output: "
            "x * rsqrt(mean(x^2) + eps) * w * act(z), one (token, head) row of `channels`."),

    SCHEMA("gdn_gated_norm_had_quant_i8", pGdnGatedNormHq, oGdnGatedNormHq,
            "The GDN block's gated RMS norm, block Hadamard and per-row int8 quantiser in one pass: "
            "x * rsqrt(mean_head(x^2) + eps) * w * silu(z), rotated over `group` and quantised to "
            "int8 with a per-row f32 `scale`."),

    SCHEMA("gdn_conv_prep", pGdnConvPrep, oGdnConvPrep,
            "The GDN depthwise causal convolution (width 4, silu) plus the split, l2norm, gate "
            "(g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b)) and the per-chunk cumsum of "
            "g, in one prefill kernel; the conv state is read at a sequence's first chunk and "
            "rewritten from its last tokens."),

    SCHEMA("gdn_conv_update", pGdnConvUpdate, oGdnConvUpdate,
            "The GDN depthwise causal convolution at decode: a speculative window over a rolling "
            "conv-state buffer, read at the slot the last accepted token left and rewritten "
            "shifted by one; writes q, k and v in their own layouts."),

    SCHEMA("ple_conv", pPleConv, oPleConv,
           "y[t] = resid[t] + silu(sum_j w[c][j] * x[t - (width-1-j)*dilation]), a DEPTHWISE "
           "CAUSAL convolution whose taps are `dilation` apart -- at width 4 and dilation 3 they "
           "are 9, 6, 3 and 0 timesteps back, so the history a sequence carries is "
           "(width-1)*dilation and not width-1. The state is [n_slots, n, state_len] with slot i "
           "holding the value at relative position i - (width-1)*dilation, and `has_init` 0 means "
           "that window is ZEROS rather than whatever the slot last held. `num_accepted` shifts "
           "the READ offset within a deeper window and marks a decode step: the window is then "
           "rewritten as gdn_conv_update rewrites its own -- the history read shifted down by one, "
           "then every input of the step -- so a next step that kept k tokens reads at k - 1. "
           "Absent, every token commits: the read is at offset zero and the last "
           "(width-1)*dilation inputs land at offset zero, which is the prefill chunk. "
           "`resid` absent drops the add and leaves the plain convolution."),
    SCHEMA("kda_update", pKdaUpdate, oKdaUpdate,
           "Bounded per-channel KDA recurrence. verify stores candidate states; prefill commits final state. "
           "Absolute cu offsets, accepted-prefix read slot, optional cold initialization, no output norm."),
    SCHEMA("gdn_recurrent_update", pGdnRecurrent, oGdnRecurrent,
            "The gated-delta-net recurrent state update at decode: g = -exp(A_log) softplus(a + "
            "dt_bias), beta = sigmoid(b), q/k l2-normalised, S *= e^g, u = beta (v - S k), S += u "
            "k^T, o = S q, per token against a paged fp32 state cache; the gated RMS norm is "
            "folded in when `z` and `norm_w` are present."),

    SCHEMA("gdn_conv_recurrent_update", pGdnConvRecurrent, oGdnConvRecurrent,
            "gdn_conv_update followed by gdn_recurrent_update, with q, k and v never written: the "
            "operands are the convolution's (`conv_idx` is its state_idx) and then the recurrence's "
            "without those three."),
    SCHEMA("gdn_conv_recurrent_update_had", pGdnConvRecurrentHad, oGdnConvRecurrentHad,
            "gdn_conv_recurrent_update whose `o_q` / `o_scale` are had_quant_act_i8g's codes of "
            "`o` (int8, a 128-wide Hadamard before the per-(token, head) amax), for an out "
            "projection stored rotated (i8a8h). Byte-identical to gdn_conv_recurrent_update "
            "followed by had_quant_act_i8g. `group` 256 is IQ8R's: one scale per two value heads "
            "(`o_scale` [tokens, n_head_v / 2]), each head still rotated on its own."),
    SCHEMA("expert_scale", pExpertScale, oExpertScale,
           "Each routing weight times its expert's own per-expert scale, in place: the router's "
           "renormalised top-k weight scaled by the [n_expert] table, an out-of-table id reading "
           "zero."),
    SCHEMA("router_topk", pMoeRouterTopk, oMoeRouterTopk,
           "Softmax over the expert logits, the top_k by value (ties to the lower id), and the "
           "kept weights renormalised to sum to one when `norm`."),
    SCHEMA("moe_scatter", pMoeScatter, oMoeScatter,
           "The routed slots' stable counting sort: expert_ids to sorted_tok, the exclusive "
           "expert_offset prefix, and the expert_count. An id outside the slice is dropped to the "
           "tail; `protect` places its experts last."),

    SCHEMA("moe_gather", pMoeGather, oMoeGather,
            "Fold the expert outputs back to the token: y[t,c] = sum_j expert_w[t,j] * "
            "y_expert[slot(t,j), c], the slots from sorted_tok. An optional shared arm adds "
            "shared[t,c] * act(gate[t]). bf16 in and out."),

    SCHEMA("router_topk_scatter", pRouterTopkScatter, oRouterTopkScatter,
            "The router top-k and the counting sort as one op: the selection over the whole "
            "n_expert logits, then the sort over this rank's n_local slice. expert_w is the "
            "selection output the sort reads."),

    SCHEMA("router_sigmoid_group_topk", pRouterSigmoidGroup, oRouterSigmoid,
           "Group-limited sigmoid routing: groups rank by the sum of their two best p + bias, "
           "selection runs over the topk_group best groups; correction bias selects, original "
           "scores weight."),
    SCHEMA("router_gemm_topk_scatter", pRouterGemmTopkScatter, oRouterGemmTopkScatter,
            "The router GEMM, its top-k and the counting sort as one op -- `gemm_nt` of `a` "
            "against the router `b`, then `router_topk_scatter` over those logits -- and, with "
            "`gate_b`/`gate` present, the shared expert's one-column `gemm_nt` over the same `a`."),

    SCHEMA("router_sigmoid_topk", pRouterSigmoid, oRouterSigmoid,
            "The GLM-style router: scores = sigmoid(logits), the selection on scores + bias, the "
            "kept weights the scores scaled by route_scale (renormalised if `norm`). Ties to the "
            "lower id; n_expert <= 512."),

    SCHEMA("moe_gemm_q", pMoeGemmQ, oMoeGemmQ,
            "The grouped quantised GEMM: y[i] = a[sorted_tok[i]/top_k] @ w_e^T with a per-(row, "
            "grid-of-K) scale fold, e the expert of sorted row i. E4M3 operands upcast to f16 in "
            "registers; the even experts are `w` / `w_scale`, the odd `w_odd` / `w_odd_scale`."),

    SCHEMA("moe_gemm", pMoeGemm, oMoeGemm,
            "The grouped GEMM with unquantised experts: y[i] = a[sorted_tok[i]/top_k] @ w_e^T, e "
            "the expert of sorted row i. bf16 a and w, f32 accumulate over the whole of K, bf16 out."),

    SCHEMA("rmsnorm_quant_fp8", pRmsQuantFp8, oRmsQuantFp8,
           "RMS norm with an optional residual add, then the E4M3 quantiser, in one pass. The "
           "residual sum is stored as bf16 before the variance reads it, so the norm is taken over "
           "the rounded sum, as the `add` + `rmsnorm` pair computes it (`rmsnorm_add` normalises "
           "the unrounded sum). `wadd` is added to every "
           "gain in f32. `out_bf16` holds exactly the value the codes were taken over."),

    SCHEMA("rmsnorm_had_quant_i8g", pRmsQuantFp8, oRmsQuantFp8,
           "The same fused add and norm with had_quant_act_i8g's quantiser on its pass: a "
           "128-wide Hadamard between the bf16-rounded norm output and the int8 absmax, one "
           "f32 scale per scale group (128, or 256 for IQ8R, whose two halves rotate "
           "independently under one amax). Byte-identical to `rmsnorm` + `had_quant_act_i8g`; "
           "`out_bf16` holds the unrotated norm output, for a bf16 consumer."),

    /* ---- the vocabulary and the sampler: schemas first, rows with their kernels ---- */
    SCHEMA("silu_mul", pMnDtype, oGateUp,
           "y[i] = silu(gate_up[i]) * gate_up[n + i] over one fused [M, 2n] plane, the gate the "
           "first half -- rad-convert's fusion order -- and the product of an activation that was "
           "never narrowed. act_mul is the same arithmetic over two operands, narrowing the "
           "activation first, as a stored activation is."),
    SCHEMA("attn_paged", pAttnPaged, oAttnPaged,
           "Scaled dot-product attention over a paged KV cache: q [tokens, n_head, head_dim], the "
           "cache [blocks, kv_heads, block_size, 2*head_dim], `block_table` a row's block list, "
           "`seqused` its length. `gqa` is the head ratio; `causal` and `window` are the mask; "
           "`scale` is 1/sqrt(head_dim) unless the graph says otherwise. k_descale and v_descale "
           "come with an fp8 cache, cu_seqlens with a packed prefill."),
    SCHEMA("mla_attn_paged", pMlaAttnPaged, oMlaAttnPaged,
           "Dense causal latent attention over a paged cache whose token row is the latent then the "
           "rotated key part: score = scale * (q . latent + q_rope . rope), value = the latent; "
           "sequences as in attn_paged. With `gate`, out = bf16(bf16(out) * bf16(sigmoid(gate[head])))."),
    SCHEMA("mla_project", pMlaProject, oGemm,
           "Per-head query absorption or value expansion through the shared [head,(K|V),latent] BF16 matrix."),
    SCHEMA("attn_dense", pAttnDense, oAttnDense,
           "Dense varlen attention over cu_seqlens segments: q, k and v may be [tokens, heads, "
           "head_dim] or [tokens, heads * head_dim] (a column slice of a fused projection at its "
           "row pitch); nothing is paged. `gqa` is the query/key head ratio, `causal` and `window` "
           "the mask, `scale` 1/sqrt(head_dim) by default. `window` is attn_paged's rule over a "
           "segment's own positions, `block` cuts a segment into blocks, `cu_div` reads the "
           "boundaries in units of rows, and `window_right` is the non-causal window's right side."),
    SCHEMA("row_topk", pRowTopk, oRowTopk,
           "Exact per-row top-R, descending, ties to the lower column. `vocab_off` shifts every "
           "column reported, so a rank's slice reports global ids; `pairs` is the [M, R, 2] plane an "
           "all_gather moves, with id -1 for a slot the row could not fill."),
    SCHEMA("sample_argmax", pMVocab, oArgmax,
           "The most likely token of each row, the lowest id on a tie."),
    SCHEMA("sample_temp", pMVocab, oTemp,
           "logits / temperature, in place. The temperature is a row parameter, so a batch mixes "
           "greedy and sampled rows."),
    SCHEMA("sample_topk", pTopk, oTopk,
           "The k most likely candidates a row: ids and values, descending, ties to the lower id. "
           "`pairs` is the vocab-parallel form the merge reads after the all-gather."),
    SCHEMA("sample_topp", pMCand, oNarrow,
           "Narrow the candidate set to the smallest prefix whose probabilities reach p, in the "
           "order the candidates arrive -- not in sorted order, which is a different set at a tie."),
    SCHEMA("sample_minp", pMCand, oNarrow,
           "Drop a candidate whose probability is under min_p times the best candidate's."),
    SCHEMA("sample_typical", pMCand, oNarrow,
           "Drop the candidates furthest from the set's entropy: keep those within `sigma` of the "
           "mean surprise, the token set a reader would not have predicted least and most alike."),
    SCHEMA("sample_xtc", pMCand, oNarrow,
           "The exclusion-the-model-would-not-choose filter: drop every candidate above p, keeping "
           "the last one standing when all of them are."),
    SCHEMA("sample_pick", pMCand, oPick,
           "The seeded inverse-CDF draw over the candidate set: the law the request's seed selects, "
           "in the order the candidates arrive. A kernel that draws from the same distribution "
           "differs from the reference on every token, and kbench reads that as a failure, because "
           "for a sampler it is one."),
    SCHEMA("sample_penalties", pPenal, oPenalties,
           "Presence, frequency and repetition penalties in place, over the row's history, with the "
           "DRY and penalty terms libref spells out."),
    SCHEMA("sample_dry", pPenal, oDry,
           "Distributed Repetition Penalty: a sequence of `last_n` that repeats a sequence of "
           "`baseline` is penalised, growing with the length of the repeat. `breakers` marks the "
           "positions that end a sequence and are never penalised."),
    SCHEMA("sample_ngram_ban", pPenal, oNgramBan,
           "Ban the n-grams of length `n` that the history shows inside `window`, to -inf. "
           "`whitelist` exempts a set of ids from the ban."),
    SCHEMA("sample_mask", pMVocab, oMask,
           "The grammar's and the tool-caller's doing: set to -inf every id whose bit is clear in "
           "the row's bitmask, addressed by the row's mask_row, which is -1 for a row with no "
           "constraint and so shares a short plane."),
    SCHEMA("sample_logprobs", pLogprobs, oLogprobs,
           "The row's log-softmax statistics for the response: max, sum of exp, and the k best "
           "(id, logit) pairs, in [M, 2 + 2k]."),
    SCHEMA("logits_gemm", pLogits, oLogits,
           "logits[M, n_vocab] = x[M, n_embd] @ lm_head[n_vocab, n_embd]^T. Separate from gemm_nt "
           "because lm_head is its own access class and the sampler wants to fuse against it: the "
           "head is the largest read in a decode step and the output is f32, which every sampler "
           "stage downstream reads."),
    SCHEMA("mtp_enter", pMtpEnter, oMtpEnter,
           "The MTP head's entry for a wide residual stream: x[k*n+i] = fc_h[i] . grouped_norm(h)[k] "
           "+ fc_e[i] . norm(e), for each of the hc sub-streams k. `ngroup` == hc is one rsqrt per "
           "sub-stream and 1 is a single rsqrt over the whole row."),
    SCHEMA("logit_rerank", pLogitRerank, oLogitRerank,
           "Rescore R candidate ids a row exactly against the full-precision head and re-sort on "
           "those values: the second half of a coarse top-R. `idx_in` carries global ids, "
           "`vocab_off` is this rank's first column, and an id outside its rows becomes a pad."),
    SCHEMA("gemm_nt_q_gated", pGemmQGated, oGemmQGated,
           "The gated MLP band in one launch: `b` is N = 2n rows, gate over up, the dot products are "
           "narrowed to bf16 as the unfused pair would leave them, the gate is `gt * sigmoid(gt)`, the "
           "product is narrowed, and the row is coded per group of `group` columns with the amax that "
           "group's scale comes from. `out_bf16`, when asked for, is the narrowed gated tensor before "
           "the coding."),
    SCHEMA("dflash_select", pDflashSelect, oDflashSelect,
           "DFlash2's greedy walk over a draft block: at each of `steps` steps the candidate with "
           "the best `unary + pred[pid] . (hp * succ[cand])` is picked, and becomes the predecessor "
           "of the next step. `cand`/`unary` have one row per (sequence, step); `anchor` is the "
           "block's first predecessor, at `m * anchor_stride`."),

    SCHEMA("dequant_w4", pDequantW4, oDequantW4,
           "Packed int4 weight to row-major bf16. Each 32-bit word holds 8 codes; the scale is "
           "f16 in the low 16 bits, the zero point -(1024+zero) as f16 in the high 16 bits."),
    SCHEMA("row_topk_merge", pRowTopkMerge, oRowTopkMerge,
           "Top-R of gathered per-rank candidate sets. Each rank contributes a sorted list; the "
           "merge selects the top-R global candidates by score."),
    SCHEMA("ngram_ids", pNgramIds, oNgramIds,
           "Hashed n-gram row ids for PLE. The FNV-1a mix over the ngram-1 token window is "
            "folded into each head's embedding table; per, W and the ids pitch derive from the "
            "operands, and `eos` cuts the hash window."),
    SCHEMA("gemm_nt_q_bias", pGemmQBias, oGemmQBias,
           "Narrow fp8 GEMM (M <= 16) with per-column bf16 bias fused into the epilogue."),
    SCHEMA("gemm_nt_q_short", pGemmQShort, oGemmQShort,
           "Narrow fp8 GEMM (M <= 16) for short-prompt decode, unsplit K."),
    SCHEMA("moe_gemm_q_mod4", pMoeGemmQMod4, oMoeGemmQMod4,
           "Grouped GEMM over the expert-sorted rows, quantised operands, four classes by "
           "expert number mod 4. The E4M3 wave-matrix core over 8 R3dTab tables (class q = "
           "entry e/4 of tables q); unequal class shapes are refused."),
    /* --- W9: Per-Layer Embedding --- */
    SCHEMA("embed_lookup_q", pEmbedQ, oEmbedQ,
           "x[m] = wte[id] * scale[0] from a table quantised with ONE scale for all of it; an "
           "absent scale is 1, so a plain bf16 table takes the same op. Padding and bounds are "
           "embed_lookup's."),
    SCHEMA("ngram_hint", pNgramHint, oNgramHint,
           "ids[s][h] = ngram_ids' hash of the n-gram ending at sequence s's newest token on a "
           "draft pass. `hist` takes the history pass's last row; otherwise the round's one row. "
           "`tail` keeps the two tails ngram_ids maintains. A prefetch hint only."),
    SCHEMA("ple_gate", pPleGate, oPleGate,
           "Per-Layer Embedding's gate: K and Q are grouped rmsnorms times their gains, "
           "s[c] = K[c].Q[c]/sqrt(n), the signed square root with `gate_eps`, then "
           "gv[c] = sigmoid(s[c]) * v and gvn = grouped_rmsnorm(gv) * w_conv."),
    /* --- W8: HyperConnections --- */
    SCHEMA("hc_read", pHcRead, oHcRead,
           "The read half of a gated residual: N = grouped_rmsnorm(h), g = sigmoid(mix_up @ "
           "silu(mix_down @ N / hc)), x = mean_c(g * N), and with inject_w, "
           "inj[c] = 2*sigmoid(inject_w[c] . N / hc). Optional q/rq are the fp8 and rotated fp8 "
           "forms of the stored x."),
    SCHEMA("hc_write", pHcWrite, oHcWrite,
           "The write half: h[c] += inj[c] * y for each of the hc streams, on the unnormed "
           "residual the paired read was handed."),
    SCHEMA("hc_enter", pHcWrite, oHcEnter,
           "h[c] = x for each of the hc streams: the wide residual stream's first value."),
    SCHEMA("hc_project", pHcProject, oHcProject,
           "For each stream of a row whose `on` flag is 1, h[c] -= scale * (h[c] . v) * v, with "
           "v the layer's unit direction and scale its stored norm."),
    SCHEMA("ar_hc_write", pArHcWrite, oArHcWrite,
           "The all-reduce and the gated residual's write that follows it, in one kernel. At one "
           "rank the reduction is the identity, so this is hc_write over the message."),
    SCHEMA("ar_gather_hc_write", pArGatherHcWrite, oArGatherHcWrite,
           "moe_gather, then ar_hc_write: the routed experts' rows summed back to token order "
           "(with the shared arm folded) ARE the message, and the gated write follows."),
    SCHEMA("all_reduce", pAllReduce, oAllReduce,
           "Sum over the tensor-parallel group. `exact` separates the exact sum from the quantised "
           "wire and a caller that does not declare it gets no kernel. At 4 and 8 ranks the exact "
           "sum is one-shot up to 524288 elements (at 8 ranks 8192) and two-shot above, chosen by "
           "the message's band; `hops` is accepted and no row here reads it."),
    SCHEMA("all_gather", pAllGather, oAllGather,
           "y is [world_size * numel] with rank r's contribution at offset r * numel. The bytes "
           "are moved and never summed: the vocab-parallel sampler's payload is (u32 token id, "
           "f32 logit) pairs and an id lane read as f32 is a denormal."),
};

extern "C" int rad_kernel_schema_count(void) {
    return (int)(sizeof(kSchemas) / sizeof(kSchemas[0]));
}
extern "C" const RadOpSchema* rad_kernel_schema_at(int i) {
    if (i < 0 || i >= rad_kernel_schema_count()) return nullptr;
    return &kSchemas[i];
}

/* ================================================================== tuning
 *
 * The axes a kernel is instantiated along, measured per card by `rad-tune` (spec §15). Without a
 * matrix instruction to size a tile against, the rows have few axes: how the work is split across
 * the card and how the weight stream is read.
 */

/* Split-K. 0 is not a point in the space but the kernel's rule, "the largest split legal at this
 * K" (r3d_args.h: libr3d's tunables). */
static const int64_t kSk[] = { 0, 1, 2, 4, 8, 16 };
/* Waves per block. 8 is 256 threads, which is the largest block that keeps the register budget of
 * a wave-per-N-tile kernel inside gfx11's 1024 VGPRs a SIMD at four waves. */
static const int64_t kWv[] = { 1, 2, 4, 8 };
/* Non-temporal weight streaming: a decode GEMM reads a weight once per token, so the L2 serves
 * latency, not reuse. RDNA3 has no transposed-load instruction, so this axis only switches the cache
 * hints on the plain 128-bit stream. */
static const int64_t kNt[] = { 0, 1 };
/* The implementation arm. 0 is the dot2 pair path the row has always run; 1 is the 16x16x16
 * wave matrix (V_WMMA) where the ISA has a tile for the row's dtype. The arm is a point like
 * the rest: the tuner walks it per shape per machine, an untuned install runs the dot2 arm,
 * and a dtype with no tile simply has no legal point 1 (gemm_tune_valid). */
static const int64_t kImpl[] = { 0, 1 };
/* The m64 band's axes: row tiles per block and column tiles a wave. NPW 1 and 4 only: the
 * kernel's activation re-read is 4*MT/NPW, and the instantiations in between cost registers
 * without buying a ratio. */
static const int64_t kMb[]  = { 1, 2, 3, 4 };
static const int64_t kNpw[] = { 1, 4 };
/* The accumulator: 0 = the f32 arm, 1 = the bf16 arm (the 16x16x16 half-precision-accumulate
 * tile; its 8 live elements sit in the even indices of the v16short accumulator, the epilogue
 * reads only those). The axis is the selector; the tuner walks it per shape per machine. */
static const int64_t kAcc[] = { 0, 1 };
/* The kernel-implementation arms of a row that ships several kernels for one contract. Two arms:
 * the table/codec decode and its per-byte ALU twin. Three arms: the grouped E4M3 GEMM's 16x16 tile
 * (LDS table), the same tile with the ALU decode, and the 16x128 block. Four arms: the bf16 tiled
 * row's fdot2 tile (table), the same tile (ALU), the 128x128 wave-matrix tile, and the 64-row
 * wave-matrix kernel. The axis is the selector, as the rest are: an untuned install runs arm 0. */
static const int64_t kImpl2[] = { 0, 1 };
static const int64_t kImpl3[] = { 0, 1, 2 };
static const int64_t kImpl4[] = { 0, 1, 2, 3 };
/* The MoE GEMM's row block: 16 (decode), 32, 64 (a tall prefill). A taller block reads the weight
 * T/mrows times, so the tune picks per shape; 64 is the untuned default (the tall band's best). */
static const int64_t kMrowsMoE[] = { 8, 16 };
static const int64_t kWvMoE[] = { 4, 8 };
static const RadTunable tMoeGemmQW4[] = {
    { "mrows", kMrowsMoE, NELEM(kMrowsMoE), 16, "rows per block: 8 for decode, 16 for prefill" },
    { "wv", kWvMoE, NELEM(kWvMoE), 8, "waves per block: 8 = 128-col tile, 4 = 64-col tile" },
};
static const int64_t kMrows[] = { 16, 32, 64 };
static const RadTunable tMoeGemm[] = {
    { "mrows", kMrows, NELEM(kMrows), 64, "the block's row count; the weight is read T/mrows times" },
};
/* The per-wave register budget: 128 is the 8-wave cap on gfx11 (1024 VGPRs/SIMD), 512 the natural
 * allocation. It is a compile-time attribute, so each legal point is its own compiled kernel; see
 * the kernels' budget comments for the measurements. */
static const int64_t kVgpr[] = { 128, 512 };
static const RadTunable tMoeGemmBf16[] = {
    { "mrows", kMrows, NELEM(kMrows), 64, "the block's row count; the weight is read T/mrows times" },
    { "vgpr",  kVgpr,  NELEM(kVgpr),  512, "the per-wave register budget: 128 is the 8-wave cap, 512 natural" },
    { "acc",   kAcc,   NELEM(kAcc),   0,
      "the WMMA accumulator: 0 = f32, 1 = the half-precision tile (a documented low-precision mode: "
      "the loss grows with k depth, no throughput gain; see the README)" },
};
static const RadTunable tMoeGemmFp8[] = {
    { "vgpr", kVgpr, NELEM(kVgpr), 128, "the per-wave register budget: 128 is the 8-wave cap, 512 natural" },
    { "acc",  kAcc,  NELEM(kAcc),  0,
      "the WMMA accumulator: 0 = f32, 1 = the half-precision tile (a documented low-precision mode: "
      "the loss grows with k depth, no throughput gain; see the README)" },
};
/* Every mrows is legal: the kernel clamps the tail, so the tune is free to pick any point. */
static int moe_tune_valid(const RadParam* p, int n_p, const RadParam* c, int n_c) {
    (void)p; (void)n_p; (void)c; (void)n_c; return 1;
}
/* Only the compiled (mrows, vgpr) points are legal: <32> has both budgets (the capped one measures
 * +38% at T=N=K=4096 NE=8), while <16> and <64> have the natural allocation only -- they measure
 * 7-8% slower under the cap, because their spill traffic outweighs the extra wave. */
static int moe_bf16_tune_valid(const RadParam* p, int n_p, const RadParam* c, int n_c) {
    (void)p; (void)n_p;
    const long long mrows = rad_param_geti(c, n_c, "mrows", 64);
    const long long vgpr = rad_param_geti(c, n_c, "vgpr", 512);
    if (vgpr == 128 && mrows != 32) return 0;
    return 1;
}
/* A single implementation axis: the kernel the row runs. No geometry interaction, so any arm is
 * legal at any point. */
static const RadTunable tImpl2[] = {
    { "impl", kImpl2, NELEM(kImpl2), 0,
      "0 = the table/codec decode, 1 = the per-byte ALU decode" } };
static const RadTunable tImpl4[] = {
    { "impl", kImpl4, NELEM(kImpl4), 0,
      "0 = the 128x128 fdot2 tile (table decode), 1 = the same tile, per-byte ALU decode, 2 = the "
      "128x128 wave-matrix tile, 3 = the 64-row wave-matrix kernel" } };
static int impl_tune_valid(const RadParam* p, int n_p, const RadParam* c, int n_c) {
    (void)p; (void)n_p;
    const long long impl = rad_param_geti(c, n_c, "impl", 0);
    return impl >= 0 && impl <= 3;
}
/* The grouped E4M3 GEMM's three kernels: arm 0 the 16x16 tile with the LDS decode table, arm 1
 * the same tile with the per-byte ALU decode, arm 2 the 16x128 block with the F32 scales folded.
 * The vgpr / acc axes are arm 2's (the block's register budget and accumulator); on arms 0 and 1
 * they are pinned to their defaults so the tuner does not spend time on an axis the arm ignores. */
static const RadTunable tMoeGemmQ[] = {
    { "impl", kImpl3, NELEM(kImpl3), 0,
      "0 = the 16x16 tile, LDS decode table, 1 = the same tile, per-byte ALU decode, 2 = the 16x128 "
      "block, F32 scales folded in f32" },
    { "vgpr", kVgpr, NELEM(kVgpr), 128, "the block's per-wave register budget (arm 2 only)" },
    { "acc",  kAcc,  NELEM(kAcc),  0,
      "the WMMA accumulator: 0 = f32, 1 = the half-precision tile (arm 2 only; a documented "
      "low-precision mode, the loss grows with k depth)" } };
static int moe_q_tune_valid(const RadParam* p, int n_p, const RadParam* c, int n_c) {
    (void)p; (void)n_p;
    const long long impl = rad_param_geti(c, n_c, "impl", 0);
    const long long vgpr = rad_param_geti(c, n_c, "vgpr", 128);
    const long long acc  = rad_param_geti(c, n_c, "acc", 0);
    if (impl < 0 || impl > 2 || acc < 0 || acc > 1) return 0;
    if (impl == 2) return (vgpr == 128 || vgpr == 512);
    return vgpr == 128;
}
/* No `kb` axis: only a GEMM that stages a K slab through LDS has a slab width to tune, and the
 * skinny rows stream the weight through registers. */

static const RadTunable tGemmM16[] = {
    { "sk", kSk, NELEM(kSk), 0, "split-K slices; 0 is the widest legal for this K" },
    { "wv", kWv, NELEM(kWv), 8, "waves per block" },
    { "nt", kNt, NELEM(kNt), 1, "non-temporal weight streaming" },
    { "impl", kImpl, NELEM(kImpl), 0, "0 = dot2 pairs, 1 = the 16x16x16 wave matrix" },
};
/* The m64 band: the wave-matrix kernel's own axes. There is no `impl` -- the band is the
 * wave matrix, and the dot2 row's band ends where this one begins. The block is `wv * 32`
 * threads; the k split is a block split over grid.y, so `sk` costs grid size, not workgroup
 * size, and the validity check bounds the two separately. */
static const RadTunable tGemmM64[] = {
    { "sk", kSk, NELEM(kSk), 1, "split-K slabs of whole 128-wide groups" },
    { "wv", kWv, NELEM(kWv), 8, "waves per block" },
    { "mb", kMb, NELEM(kMb), 1, "row tiles per block" },
    { "npw", kNpw, NELEM(kNpw), 1, "column tiles a wave: 1 or 4" },
    { "nt", kNt, NELEM(kNt), 1, "non-temporal weight streaming" },
    { "acc", kAcc, NELEM(kAcc), 0,
      "accumulator: 0 = f32, 1 = the half-precision tile (a documented low-precision mode: the "
      "loss grows with k depth, no throughput gain; see the README)" },
};
/* No `wv` axis: the quantised kernel's launcher uses a fixed block of eight waves, because its
 * weight decode (sixteen codes a lane into eight dot2) fills a wave's issue budget at any block
 * size. An axis the kernel ignores would only spend tuner time. */
static const RadTunable tGemmQ[] = {
    { "sk", kSk, NELEM(kSk), 0, "split-K slices; 0 is the widest legal for this K" },
    { "nt", kNt, NELEM(kNt), 1, "non-temporal weight streaming" },
    { "impl", kImpl, NELEM(kImpl), 0, "0 = dot2 pairs, 1 = the 16x16x16 wave matrix" },
};

/* Cross-axis legality, in one place so the tuner, the scratch hook and the launch agree. A split's
 * slice must end on a k multiple the kernel's inner loop consumes: 8 elements for a bf16 dot2 pair
 * group, and one whole scale group for a quantised weight, since a slice that ends mid-block cannot
 * apply its block scale. */
static int gemm_tune_valid(const RadParam* p, int n_p, const RadParam* c, int n_c) {
    const long long K = rad_param_getdim(p, n_p, "K", 0);
    const long long sk = rad_param_geti(c, n_c, "sk", 0);
    const long long wv = rad_param_geti(c, n_c, "wv", 8);
    const long long impl = rad_param_geti(c, n_c, "impl", 0);
    if (K <= 0 || wv < 1 || wv > 8 || impl < 0 || impl > 1) return 0;
    /* The wave-matrix arm exists where the ISA has a tile for the row's dtype: the bf16 and f16
     * skinny rows and the w8a8 grid. The other dtypes have no 16x16x16 form on this ISA, and a
     * point the kernel would refuse is a point the tuner should not spend. */
    if (impl == 1) {
        const char* dt = rad_param_gets(p, n_p, "dtype", 0);
        if (!dt || (std::strcmp(dt, "bf16") && std::strcmp(dt, "f16") && std::strcmp(dt, "w8a8")))
            return 0;
    }
    long long unit = rad_param_getdim(p, n_p, "group", 0) > 0
                         ? rad_param_getdim(p, n_p, "group", 0)
                         : 8;
    /* The wave-matrix arm's slab is a 16-k step, wider than the dot2 pair group's 8. */
    if (impl == 1 && unit < 16) unit = 16;
    return sk >= 0 && (sk == 0 || K % (sk * unit) == 0);
}

/* The m64 band's cross-axis legality, in one place so the tuner and the launch agree. The slab
 * is whole 128-wide groups; the split is a block split, so the workgroup is WV*32 threads, and
 * the one shape the kernel's lowering cannot carry -- four column tiles a wave with more than
 * one row tile -- is declined here as well as at the launch. */
static int gemm_tune_valid_m64(const RadParam* p, int n_p, const RadParam* c, int n_c)
{
    const long long K = rad_param_getdim(p, n_p, "K", 0);
    const long long sk = rad_param_geti(c, n_c, "sk", 1);
    const long long wv = rad_param_geti(c, n_c, "wv", 8);
    const long long mb = rad_param_geti(c, n_c, "mb", 1);
    const long long npw = rad_param_geti(c, n_c, "npw", 1);
    const long long nt = rad_param_geti(c, n_c, "nt", 1);
    const long long acc = rad_param_geti(c, n_c, "acc", 0);
    if (K <= 0) return 0;
    if (sk < 1 || sk > 32) return 0;
    if (wv < 1 || wv > 32) return 0;
    if (mb < 1 || mb > 4) return 0;
    if (npw != 1 && npw != 4) return 0;
    if (nt < 0 || nt > 1) return 0;
    if (acc != 0) return 0;
    if (npw == 4 && mb > 1) return 0;
    return K % (sk * 128) == 0;
}

/* The chosen point, read back. `TUNE` falls back to the axis's default when the core appended
 * nothing, so a caller with no tuned key runs what an untuned install runs. */
static long long tune_i(const RadArgs* a, const RadTunable* t, int n, const char* key) {
    if (a && a->p)
        for (int i = 0; i < a->n_p; ++i)
            if (a->p[i].key && !std::strcmp(a->p[i].key, key) && a->p[i].kind == RAD_P_INT)
                return a->p[i].ival;
    for (int i = 0; i < n; ++i)
        if (!std::strcmp(t[i].key, key)) return t[i].deflt;
    return 0;
}
#define TUNE(a, tab, key) ((int)tune_i((a), (tab), NELEM(tab), (key)))

/* ------------------------------------------------------------------------------ collectives
 *
 * The all-reduce family's launch geometry. The defaults are libhalo's measured points and change
 * no result: nthreads 1024, drain 3 (s_wait_storecnt, which fine-grained uncached scratch needs),
 * acq 0, pub 1, and nblocks 0 for the size rule (r3dh_ar_entry.hip clamps it). `ts_min` is the
 * element count from which the four-rank exact one-shot hands its message to the exact two-shot on
 * the same instance; it is declared by every all-reduce table so one tune pin is legal for every
 * row the op resolves to. `overlap` 1 rows take the narrow launch (tArLane), which leaves the CUs
 * to the compute beside them. */
static const int64_t kArNblocks[]    = { 0, 2, 4, 8, 16, 24 };
static const int64_t kArNthreads[]   = { 256, 512, 1024 };
static const int64_t kArDrain[]      = { 1, 3 };
static const int64_t kArAcq[]        = { 0, 1 };
static const int64_t kArTsMin[]      = { 0, 16384, 32768, 65536, 131072, 262144 };
static const int64_t kArQ84Tail[]    = { 1, 0 };
static const int64_t kArNblocksLane4[] = { 0, 4, 8, 16, 24 };
static const RadTunable tAr[] = {
    { "nblocks",  kArNblocks,  NELEM(kArNblocks),  0, "blocks in the grid; 0 scales with the message and clamps" },
    { "nthreads", kArNthreads, NELEM(kArNthreads), 1024, "threads per block" },
    { "drain",    kArDrain,    NELEM(kArDrain),     3, "store-drain wait before the flag: 3 = s_wait_storecnt, 1 = vmcnt" },
    { "acq",      kArAcq,      NELEM(kArAcq),       0, "acquire fence after the flag read" },
    { "ts_min",   kArTsMin,    NELEM(kArTsMin), 32768, "elements from which the four-rank one-shot takes the exact two-shot; 0 = never" },
};
static const RadTunable tArQ84[] = {
    { "nblocks",  kArNblocks,  NELEM(kArNblocks),  0, "blocks in the grid; 0 is one a row, clamped" },
    { "nthreads", kArNthreads, NELEM(kArNthreads), 1024, "unused: the norm's width sets the block for bit-exactness" },
    { "drain",    kArDrain,    NELEM(kArDrain),     3, "store-drain wait before the flag: 3 = s_wait_storecnt, 1 = vmcnt" },
    { "acq",      kArAcq,      NELEM(kArAcq),       0, "acquire fence after the flag read" },
    { "tail",     kArQ84Tail,  NELEM(kArQ84Tail),   1, "from 5 to 32 rows, the standalone two-shot with the junction as its tail; 0 = the one-shot" },
};
static const RadTunable tArTs4Lane[] = {
    { "nblocks",  kArNblocksLane4, NELEM(kArNblocksLane4), 4, "blocks in the grid; 0 scales with the message and clamps" },
    { "nthreads", kArNthreads, NELEM(kArNthreads), 1024, "threads per block" },
    { "drain",    kArDrain,    NELEM(kArDrain),     3, "store-drain wait before the flag: 3 = s_wait_storecnt, 1 = vmcnt" },
    { "acq",      kArAcq,      NELEM(kArAcq),       0, "acquire fence after the flag read" },
    { "ts_min",   kArTsMin,    NELEM(kArTsMin), 32768, "unused here (tAr says why it is declared)" },
};
static const RadTunable tArLane[] = {
    { "nblocks",  kArNblocks,  NELEM(kArNblocks),  8, "blocks in the grid; 0 scales with the message and clamps" },
    { "nthreads", kArNthreads, NELEM(kArNthreads), 256, "threads per block" },
    { "drain",    kArDrain,    NELEM(kArDrain),     3, "store-drain wait before the flag: 3 = s_wait_storecnt, 1 = vmcnt" },
    { "acq",      kArAcq,      NELEM(kArAcq),       0, "acquire fence after the flag read" },
    { "ts_min",   kArTsMin,    NELEM(kArTsMin), 32768, "unused here (tAr says why it is declared)" },
};

extern "C" void r3dh_cfg_ar(const RadArgs* a, HaloArCfg* out) {
    out->nblocks  = TUNE(a, tAr, "nblocks");
    out->nthreads = TUNE(a, tAr, "nthreads");
    out->drain    = TUNE(a, tAr, "drain");
    out->acq      = TUNE(a, tAr, "acq");
    out->pub      = 1;
    /* Operand 0 is the message on every collective that reads this. */
    if (a && a->t && a->n_t > 0 && rad_tensor_numel(&a->t[0]) >= 131072 &&
        rad_args_geti_or(a, "world_size", a->world_size) == 4)
        out->drain = 1;
}
extern "C" int r3dh_cfg_ar_tail(const RadArgs* a) {
    return TUNE(a, tArQ84, "tail");
}
extern "C" long r3dh_cfg_ar_ts_min(const RadArgs* a) {
    return (long)tune_i(a, tAr, NELEM(tAr), "ts_min");
}

extern "C" void r3d_cfg_gemm(const RadArgs* a, R3dGemmCfg* out) {
    out->wv = TUNE(a, tGemmM16, "wv");
    out->sk = TUNE(a, tGemmM16, "sk");
    out->kb = 0;
    out->nt = TUNE(a, tGemmM16, "nt");
    out->impl = TUNE(a, tGemmM16, "impl");
}
extern "C" void r3d_cfg_moe(const RadArgs* a, R3dMoeCfg* out) {
    out->mrows = TUNE(a, tMoeGemm, "mrows");
    out->vgpr = TUNE(a, tMoeGemmBf16, "vgpr");   /* 512 natural where the row has no vgpr axis */
    out->acc = TUNE(a, tMoeGemmBf16, "acc");     /* 0 where the row has no acc axis */
}
extern "C" void r3d_cfg_moe_q(const RadArgs* a, R3dMoeQCfg* out) {
    out->impl = TUNE(a, tMoeGemmQ, "impl");
    out->vgpr = TUNE(a, tMoeGemmQ, "vgpr");
    out->acc = TUNE(a, tMoeGemmQ, "acc");
}
extern "C" void r3d_cfg_split(const RadArgs* a, R3dSplitCfg* out) {
    out->splits = TUNE(a, tGemmM16, "sk");
}
extern "C" void r3d_cfg_topk(const RadArgs* a, R3dTopkCfg* out) {
    out->rc = 0;
    out->nch = 0;
}

/* Split-KV partials: one record of 128 accumulator floats plus the running max and sum (132
 * floats) per (query row, head, slice). The shim takes the slice count from the bytes the engine
 * offers, halving it until they fit, so this is the request at the slice count the walk benefits
 * from, capped at 16 MiB. Attaching scratch makes the row non-concurrent (rad_kernel_concurrent is
 * `scratch == nullptr`), which is the cost of the split. */
static int64_t attn_scratch(const RadArgs* a) {
    /* Sized from the shape: a scratch hook runs at declare, when the operands' shapes are bound
     * but every tensor's `data` is still null. A hook that required data would answer 0 and the
     * split pass would find no buffer. gemm_scratch sizes from RadParam dims for the same reason. */
    if (!a || a->n_t < 1 || a->t[0].rank == 0) return 0;
    long long n_head = 0, head_dim = 0;
    const int64_t nh = rad_args_geti(a, "n_head", &n_head) ? n_head : 0;
    if (!rad_args_geti(a, "head_dim", &head_dim) || !head_dim || !nh) return 0;
    const long long rows = rad_tensor_numel(&a->t[0]) / (nh * head_dim);
    if (rows <= 0) return 0;
    const long long want = rows * nh * 8 * 132 * (long long)sizeof(float);
    const long long cap = 16LL << 20;
    return want < cap ? want : cap;
}

/* Split-KV partials for the head_dim-256 int8 decode row: 256 accumulator floats plus the
 * running max and sum (260 floats) per (query row, head, slice), at the band's widest split (16)
 * and capped at 16 MiB. Attaching scratch makes the row non-concurrent, as for the h128 decode. */
static int64_t attn_h256_scratch(const RadArgs* a) {
    if (!a || a->n_t < 1 || a->t[0].rank == 0) return 0;
    long long n_head = 0, head_dim = 0;
    const int64_t nh = rad_args_geti(a, "n_head", &n_head) ? n_head : 0;
    if (!rad_args_geti(a, "head_dim", &head_dim) || !head_dim || !nh) return 0;
    const long long rows = rad_tensor_numel(&a->t[0]) / (nh * head_dim);
    if (rows <= 0) return 0;
    const long long want = rows * nh * 16 * 260 * (long long)sizeof(float);
    const long long cap = 16LL << 20;
    return want < cap ? want : cap;
}

/* The IQR GEMV's split-K partials (r3d_gemm_iqr_m16.hip): [splits][N] f32 for the narrow band, 0
 * otherwise. Declared here because the plugin's own header is not the ABI. */
extern "C" int64_t r3d_rad_gemm_nt_q_iqr_m16_scratch(const RadArgs* a);

/* The N = 1 GEMV (r3d_gemm_nt_n1.hip). Declared here because the plugin's own header is not the
 * ABI. */
extern "C" int r3d_rad_gemm_nt_bf16_n1(const RadArgs* a, RadStream s);

/* The routed IQR GEMM (r3d_moe_iqr.hip), its run-map scratch hook and the IQR storage hooks at the
 * MoE classes' geometry (r3d_iqr_layout.cpp). Declared here because the plugin's own header is
 * not the ABI. */
extern "C" int r3d_rad_moe_gemm_iqr(const RadArgs* a, RadStream s);
extern "C" int64_t r3d_rad_moe_gemm_iqr_scratch(const RadArgs* a);
extern "C" int r3d_rad_layout_moe_iqr(const RadParam* p, int n_p, int operand,
                                      const RadEncoding* enc, const int* sel,
                                      const RadTensor* pl, int n, RadLayout* out);
extern "C" int r3d_rad_relayout_moe_iqr(const RadParam* p, int n_p, int operand,
                                        const RadEncoding* enc, const int* sel,
                                        const RadTensor* pl, int n, void* dst,
                                        int64_t dst_bytes);
extern "C" int r3d_rad_unrelayout_moe_iqr(const RadParam* p, int n_p, int operand,
                                          const RadEncoding* enc, const int* sel, const void* src,
                                          int64_t src_bytes, const RadTensor* pl, int n);

/* The grouped w4nl codebook GEMM (r3d_moe_w4nl.hip): the same routed-row walk as the IQR row, over
 * experts stored as u4 codes, a sixteen-level f32 table and an E4M3 scale per (row, 64 of K) under
 * one f32 2^-13. Its layout hooks (r3d_iqr_layout.cpp) join the codes and table and the two scale
 * planes, and refuse a table or level that is not the w4nl one. Declared here because the plugin's
 * own header is not the ABI. */
extern "C" int r3d_rad_moe_gemm_w4nl(const RadArgs* a, RadStream s);
extern "C" int64_t r3d_rad_moe_gemm_w4nl_scratch(const RadArgs* a);
extern "C" int r3d_rad_layout_moe_w4nl(const RadParam* p, int n_p, int operand,
                                       const RadEncoding* enc, const int* sel,
                                       const RadTensor* pl, int n, RadLayout* out);
extern "C" int r3d_rad_relayout_moe_w4nl(const RadParam* p, int n_p, int operand,
                                         const RadEncoding* enc, const int* sel,
                                         const RadTensor* pl, int n, void* dst,
                                         int64_t dst_bytes);

/* A split's partials are f32 over the output, plus one arrival counter per split. Sized at the
 * band's bounds and the axis's widest split, because a scratch hook runs at declare, where the
 * range may still be open: the buffer must hold any point the tuner can pick. */
static int64_t gemm_scratch(const RadParam* p, int n_p) {
    const long long M = rad_param_getdim(p, n_p, "M", 0), N = rad_param_getdim(p, n_p, "N", 0);
    const long long sk = kSk[NELEM(kSk) - 1];
    if (M <= 0 || N <= 0) return 0;
    return sk * M * N * 4 + sk * 64;
}

/* The fused router GEMM's scratch: the m16 arm's f32 partial arena at the band's M, whose split
 * cap is kSk's top (r3d_gemm_bf16_nt_m16.hip's ladder). The tiled arm above M = 16 asks for none,
 * and the gate's N = 1 arena is far smaller; the shared arena holds whichever arm the step picks. */
static int64_t router_fused_scratch(const RadArgs* a) {
    if (!a) return 0;
    const long long M = rad_param_getdim(a->p, a->n_p, "M", 0);
    const long long ne = rad_param_getdim(a->p, a->n_p, "n_expert", 0);
    const long long sk = kSk[NELEM(kSk) - 1];
    if (M <= 0 || ne <= 0) return 0;
    return sk * M * ne * 4 + sk * 64;
}

/* The fused router row (r3d_moe_router_topk_bf16.hip), declared here because the plugin's own
 * header is not the ABI. */
extern "C" int r3d_rad_router_gemm_topk_scatter(const RadArgs* a, RadStream s);

/* ================================================================== constraints
 *
 * What the kernel cannot do, and what the selector needs. Divisibility follows the unit's lane map:
 * a wave covering a bf16 row reads four elements a lane (two pairs), so a row of 4m elements needs
 * no tail; where the kernel has a tail path the constraint says 2, not 4.
 */

/* bf16 elementwise, norms and the GEMMs: rows of an even width, so a lane owns a pair. */
static const RadConstraint cBf16Row[]  = { C_IN("dtype", "bf16"), C_DIV("n", 2) };
static const RadConstraint cF32Row[]   = { C_IN("dtype", "f32"), C_DIV("n", 2) };
static const RadConstraint cNormBf16[] = { C_IN("dtype", "bf16"), C_DIV("n", 2),
                                           C_GE("n", 2) };
static const RadConstraint cNormF32[]  = { C_IN("dtype", "f32"), C_DIV("n", 2), C_GE("n", 2) };
static const RadConstraint cSoftmax[]  = { C_IN("dtype", "f32"), C_GE("n", 2) };

/* A cast row is its pair of dtype names. */
static const RadConstraint cCastBf16F32[] = { C_IN("from", "bf16"), C_IN("to", "f32"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastF32Bf16[] = { C_IN("from", "f32"), C_IN("to", "bf16"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastBf16[]    = { C_IN("from", "bf16"), C_IN("to", "bf16"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastF32[]     = { C_IN("from", "f32"), C_IN("to", "f32"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastI64F32[]  = { C_IN("from", "i64"), C_IN("to", "f32"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastI32I64[]  = { C_IN("from", "i32"), C_IN("to", "i64"),
                                              C_DIV("n", 2) };
static const RadConstraint cCastI64I32[]  = { C_IN("from", "i64"), C_IN("to", "i32"),
                                              C_DIV("n", 2) };

/* act_mul names its activation, because a row for gelu and a row for silu are different template
 * instantiations and a model asks for one of them. */
static const RadConstraint cActSilu[]    = { C_IN("act", "silu"), C_IN("dtype", "bf16"),
                                             C_DIV("n", 2) };
static const RadConstraint cActGelu[]    = { C_IN("act", "gelu"), C_IN("dtype", "bf16"),
                                             C_DIV("n", 2) };
static const RadConstraint cActSigmoid[] = { C_IN("act", "sigmoid"), C_IN("dtype", "bf16"),
                                             C_DIV("n", 2) };

/* A fill row is its dtype; the value is a parameter, and every dtype takes every value. */
static const RadConstraint cFillBf16[] = { C_IN("dtype", "bf16") };
static const RadConstraint cFillF32[]  = { C_IN("dtype", "f32") };
static const RadConstraint cDequantW4[] = { C_DIV("N", 16), C_DIV("K", 64) };
static const RadConstraint cRowTopkMerge[] = { C_GE("R", 1), C_LE("R", 32), C_IN("dtype", "f32") };
static const RadConstraint cNgramIds[] = { C_GE("ngram", 2), C_LE("ngram", 8) };

static const RadConstraint cGemmQBias[] = { C_IN("dtype", "fp8a8"), C_GE("M", 1), C_LE("M", 16) };
static const RadConstraint cGemmQShort[] = { C_IN("dtype", "fp8a8"), C_GE("M", 1), C_LE("M", 64) };
static const RadConstraint cMoeGemmQMod4[] = { C_GE("n_expert", 4) };

static const RadConstraint cFillI32[]  = { C_IN("dtype", "i32") };
static const RadConstraint cFillI64[]  = { C_IN("dtype", "i64") };

/* rope: the two modes this kernel implements and an even head_dim. The yarn, llama3 and ladder
 * parameters are absent from these rows' bands, so a geometry carrying them is not routed here.
 * `rotary_dim` cannot be constrained: it is optional, a constraint on a key the band does not carry
 * does not hold (see r3d_selftest.cpp's band-reachability phase), and llama-family graphs, which
 * rotate the whole head, omit it. The shim refuses an odd rotary_dim (RAD_E_SHAPE); absent, it is
 * head_dim, which the constraint keeps even. */
static const RadConstraint cRope[] = { C_IN("mode", "neox gptj"), C_DIV("head_dim", 2) };

/* The cache's dtype is the row: a bf16 cache stores the value, an E4M3 cache encodes it with the
 * store's scale. head_dim even because a lane writes a pair; block_size a multiple of 16 so a
 * block's slice of a row is a whole number of lanes. */
static const RadConstraint cKvBf16[] = { C_IN("kv_dtype", "bf16"), C_DIV("head_dim", 2),
                                         C_DIV("block_size", 16) };
static const RadConstraint cKvFp8[]  = { C_IN("kv_dtype", "fp8_e4m3"), C_DIV("head_dim", 2),
                                         C_DIV("block_size", 16) };
static const RadConstraint cKvF16[]  = { C_IN("kv_dtype", "f16"), C_DIV("head_dim", 2),
                                         C_DIV("block_size", 16) };
/* head_dim is 128 or 256 (the launcher refuses anything else); the constraint system has no
 * int-set form, so the row states the range and the kernel enforces the set. */
static const RadConstraint cMlaProject[] = { C_GE("M", 1),
                                             C_LE("M", 65535),
                                             C_GE("heads", 1),
                                             C_LE("heads", 64),
                                             C_GE("head_dim", 128),
                                             C_LE("head_dim", 256),
                                             C_EQ("latent_dim", 512),
                                             C_IN("mode", "absorb expand"),
                                             C_IN("dtype", "bf16") };
static const RadConstraint cMlaAttnPaged[] = {
    C_GE("M", 1),       C_LE("M", 65535),     C_EQ("latent_dim", 512), C_EQ("rope_dim", 64),
    C_GE("heads", 1),   C_LE("heads", 64),    C_GE("block_size", 1),   C_LE("block_size", 256),
    C_IN("dtype", "bf16")
};

/* The 27B bands: an int8 slot cache at head_dim 256 and 16-token pages, the cos/sin table the
 * fused prologue reads (rot 64, theta 1e7, 64 rows), and imrope with Qwen3-VL's 11 11 10
 * sections at 24 query heads over 4 KV heads.
 *
 * Flash-Next's 13 attention layers page the same int8 slot cache at 4 tokens a block
 * (FLASH_NEXT_AUDIT.md §7.2: head_dim 256, n_head_kv 2, block_size 4, M 4096). The store's
 * addressing already reads block_size as an argument, so the band is a second row over the same
 * entry points, not a second kernel: the 16-token row and its bytes stay as they were. */
static const RadConstraint cKvI8[] = { C_IN("kv_dtype", "i8"), C_EQ("head_dim", 256),
                                       C_EQ("block_size", 16) };
static const RadConstraint cKvI8B4[] = { C_IN("kv_dtype", "i8"), C_EQ("head_dim", 256),
                                         C_EQ("block_size", 4) };
static const RadConstraint cRopeTableF32[] = { C_IN("dtype", "f32"), C_EQ("rot", 64),
                                               C_GE("M", 1), C_LE("M", 64) };
/* The 27B calls imrope once for q (24 heads, no KV) and once for k (4 heads, no KV); the fused
 * q+k form (24/4) is Flash-Next's. The shim reads the width from the pair and refuses the rest. */
static const RadConstraint cRopeImrope[] = { C_IN("mode", "imrope"), C_EQ("head_dim", 256),
                                             C_EQ("rotary_dim", 64), C_GE("n_head", 1),
                                             C_LE("n_head", 24), C_LE("n_head_kv", 4) };
/* Flash-Next's QSA query: four indexer heads of 128 lanes, no key or value heads, a 64-lane
 * partial rotary and Qwen3-VL's 11 11 10 sections. The fused attention block keeps the h256 row;
 * the two bands cannot overlap on head_dim. */
static const RadConstraint cRopeImropeH128[] = { C_IN("mode", "imrope"), C_EQ("head_dim", 128),
                                                 C_EQ("rotary_dim", 64), C_EQ("n_head", 4),
                                                 C_EQ("n_head_kv", 0), C_GE("M", 1) };

static const RadConstraint cEmbed[] = { C_IN("dtype", "bf16"), C_DIV("n_embd", 2) };
/* An IQR table: the gather decodes the rows it is asked for, so n_embd is whole 128-blocks. */
static const RadConstraint cEmbedIqr[] = { C_IN("dtype", "bf16"), C_DIV("n_embd", 128) };
/* The vocabulary-weighted IQR head: its classes and the activation quantiser assume whole 256. */
static const RadConstraint cLogitsVocabIqr[] = { C_IN("dtype", "bf16"), C_DIV("n_embd", 256) };

/* The gather's and scatter's dtype, and a row wide enough for a lane to own a pair. */
static const RadConstraint cGatherBf16[]   = { C_IN("dtype", "bf16"), C_DIV("n", 2) };
static const RadConstraint cGatherF32[]    = { C_IN("dtype", "f32"), C_DIV("n", 2) };
static const RadConstraint cScatterBf16[]  = { C_IN("dtype", "bf16"), C_DIV("n", 2) };
/* Opaque byte rows: the KV tier's fragments, copied whole and never through the E4M3 codec. A
 * dword a thread, so the row is a whole number of dwords. */
static const RadConstraint cOpaqueRowU8[]  = { C_IN("dtype", "u8"), C_DIV("n", 4) };

/* The quantisers: the scale group is the checkpoint's block, 128, and a row wide enough that a
 * wave's four-elements-a-lane store is whole. */
static const RadConstraint cQuantFp8[]   = { C_IN("dtype", "bf16"), C_EQ("group", R3D_FP8_BLOCK),
                                             C_DIV("n", 4) };
/* `quant_act_i8` is the per-row int8 whose codes are in the int8 WMMA's step order, so its width
 * is a whole 16-byte step. `quant_act_i8g` is per group with row-major codes, as a dot2 GEMM reads
 * them; it is the activation of this library's i8a8 row. */
static const RadConstraint cQuantI8[]    = { C_IN("dtype", "bf16"), C_EQ("group", R3D_W8_GROUP),
                                             C_DIV("n", 16) };
static const RadConstraint cQuantI8g[]   = { C_IN("dtype", "bf16"), C_EQ("group", R3D_W8_GROUP),
                                             C_DIV("n", 4) };
static const RadConstraint cHadQuantI8[] = { C_IN("dtype", "bf16"), C_GE("group", 128),
                                             C_LE("group", 512), C_DIV("n", 16) };
static const RadConstraint cGeluMulQuantI8[] = { C_IN("dtype", "bf16"), C_DIV("n", 16) };
static const RadConstraint cKvLenTrim[] = { C_GE("max_seqs", 1), C_LE("max_seqs", 64) };
static const RadConstraint cQsaTailStore[] = { C_IN("dtype", "bf16"), C_GE("seqs", 1) };
/* The block key's LDS is one f32 row, so n is bounded by it; the board's index head is 128. */
static const RadConstraint cQsaBlockKey[] = { C_IN("dtype", "bf16"), C_LE("n", 4096) };
/* The score row takes the indexer's activation dtype for q; `bk` may be the block-key group's
 * own bf16 whichever row runs. */
static const RadConstraint cQsaScoreBf16[] = { C_IN("dtype", "bf16") };
static const RadConstraint cQsaScoreF16[]  = { C_IN("dtype", "f16") };
static const RadConstraint cQsaScoreF32[]  = { C_IN("dtype", "f32") };
static const RadConstraint cQsaSelect[] = { C_GE("blocks", 1), C_GE("topk", 1),
                                            C_GE("ratio", 1), C_GE("seqs", 1) };
static const RadConstraint cQsaWork[] = { C_IN("dtype", "bf16"), C_GE("n", 1), C_GE("ratio", 1),
                                          C_GE("seqs", 1), C_GE("work", 1) };
/* group is the block width: the kernel instantiates 128, 256 and 512 (the shim declines the
 * DIV-legal 384, which no wave form serves). */
static const RadConstraint cGatedHadQuantI8[] = { C_IN("dtype", "bf16"), C_DIV("group", 128),
                                                  C_GE("group", 128), C_LE("group", 512),
                                                  C_DIV("n", 16) };
static const RadConstraint cHadQuantFp8[] = { C_IN("dtype", "bf16"), C_EQ("group", 128),
                                              C_EQ("n", 128) };
static const RadConstraint cGatedHadQuantFp8[] = { C_IN("dtype", "bf16"), C_EQ("group", 128),
                                                   C_EQ("n", 128) };
static const RadConstraint cQuantCols[] = { C_IN("dtype", "bf16"), C_DIV("n", 8) };
static const RadConstraint cHadQuantI8g[] = { C_IN("dtype", "bf16"), C_EQ("group", 128),
                                              C_DIV("n", 128) };
static const RadConstraint cGatedHadQuantI8g[] = { C_IN("dtype", "bf16"), C_EQ("group", 128),
                                                   C_DIV("n", 128) };
/* The gate in its own operand at the same widths. `act` is optional on the op, so it is not bound:
 * the shim reads it and defaults to silu, as libref does. */
static const RadConstraint cGateHadQuantI8g[] = { C_IN("dtype", "bf16"), C_EQ("group", 128),
                                                  C_DIV("n", 128) };
/* IQ8R's group: one scale per 256 columns, each 128-wide half rotated on its own. */
static const RadConstraint cHadQuantI8g256[] = { C_IN("dtype", "bf16"), C_EQ("group", 256),
                                                  C_DIV("n", 256) };
static const RadConstraint cGatedHadQuantI8g256[] = { C_IN("dtype", "bf16"), C_EQ("group", 256),
                                                       C_DIV("n", 256) };
static const RadConstraint cGateHadQuantI8g256[] = { C_IN("dtype", "bf16"), C_EQ("group", 256),
                                                     C_DIV("n", 256) };
static const RadConstraint cRmsQuantI8[] = { C_IN("dtype", "bf16"), C_DIV("n", 16) };
/* Not `C_IN("act", "silu sigmoid")`: `act` is optional on the op (a graph may carry libref's
 * integer `mode` instead, and libref defaults to silu), so a band with neither would match no row.
 * The shim reads `act`, then `mode`, then defaults to silu, and refuses gelu, which this unit has no
 * kernel for, with RAD_E_UNSUPPORTED. */
static const RadConstraint cGateQuant[]  = { C_IN("dtype", "bf16"),
                                             C_EQ("group", R3D_FP8_BLOCK), C_DIV("n", 4) };
static const RadConstraint cDequantFp8[] = { C_IN("dtype", "fp8_e4m3"),
                                             C_EQ("group", R3D_FP8_BLOCK), C_DIV("n", 4) };
static const RadConstraint cDequantE5[]  = { C_IN("dtype", "fp8_e5m2"),
                                             C_EQ("group", R3D_FP8_BLOCK), C_DIV("n", 4) };
static const RadConstraint cDequantI8[]  = { C_IN("dtype", "i8"), C_EQ("group", R3D_W8_GROUP),
                                             C_DIV("n", 4) };

/* The GEMMs. The skinny row's band is the decode and verify height, M <= 16: it covers a
 * speculative verify with room to spare and stays below the height where staging the activations
 * through LDS pays. K even because the inner loop consumes pairs. N has no bound: a wave covering N
 * handles the tail, and feed-forward and vocabulary widths (11008, 14336, 128256, a checkpoint's
 * `intermediate_size`) vary. */
static const RadConstraint cGemmNtF32[] = { C_GE("M", 1),         C_LE("M", 65535),
                                            C_GE("N", 1),         C_LE("N", 512),
                                            C_GE("K", 1),         C_LE("K", 16384),
                                            C_IN("dtype", "bf16") };
static const RadConstraint cGemmM16[] = { C_IN("dtype", "bf16"), C_GE("M", 1), C_LE("M", 16),
                                          C_DIV("K", 2), C_GE("N", 1) };
static const RadConstraint cGemmM16F16[] = { C_IN("dtype", "f16"), C_GE("M", 1), C_LE("M", 16),
                                             C_DIV("K", 2), C_GE("N", 1) };
/* The one-column GEMV: the shared expert's gate at prefill heights, where the m16 row's wave map
 * has one column to cover. Its band is N = 1 and any M; K is the 16-element load step's multiple
 * (128 is the container's block, and the row's store order does not need a split). */
static const RadConstraint cGemmN1[] = { C_IN("dtype", "bf16"), C_EQ("N", 1), C_GE("M", 1),
                                         C_DIV("K", 128) };
/* The a16 band: the 16x16x16 f16 tile's M, the group's K, the tile's N. K divisible by 128
 * because the slab is a whole (scale, zero) group; N by 16 because a wave's column tile is a
 * tile. */
static const RadConstraint cGemmM64W4A16A[] = { C_GE("M", 1), C_LE("M", 64), C_DIV("K", 128),
                                                C_DIV("N", 16), C_IN("dtype", "w4a16_asym") };
/* The 5-bit row: the 4-bit row's band, ParoQuant's int5. */
static const RadConstraint cGemmM64W5A16[] = { C_GE("M", 1), C_LE("M", 64), C_DIV("K", 128),
                                               C_DIV("N", 16), C_IN("dtype", "w5a16") };

/* The gated-delta-net preamble: the chunk width and the key head size are compiled in. */
static const RadConstraint cGdnKktC64[] = { C_GE("M", 1), C_EQ("head_k", 128), C_EQ("chunk", 64) };

/* The chunked scan: the tile is compiled in for both head sizes and the chunk width. */
static const RadConstraint cGdnChunkScanC64[] = { C_GE("M", 1), C_EQ("head_k", 128),
                                                  C_EQ("head_v", 128), C_EQ("chunk", 64) };
static const RadConstraint cGdnGatedRmsnorm[] = { C_GE("M", 1), C_EQ("channels", 128) };
static const RadConstraint cGdnGatedNormHq[] = { C_GE("M", 1), C_EQ("head_dim", 128) };
static const RadConstraint cGdnConvW4[] = { C_GE("M", 1), C_EQ("head_k", 128), C_EQ("head_v", 128),
                                           C_EQ("conv_width", 4) };
static const RadConstraint cGdnConvUpdW4[] = { C_GE("q_len", 1), C_EQ("head_k", 128),
                                              C_EQ("head_v", 128), C_EQ("conv_width", 4) };
/* PLE's convolution. The kernel bounds (width-1)*dilation, the state write-back's register
 * window, at 16; a constraint cannot express a product, so these are the tightest separate caps
 * admitting every combination under it. */
static const RadConstraint cPleConv[]   = { C_IN("dtype", "bf16"), C_GE("width", 1),
                                            C_LE("width", 5), C_GE("dilation", 1),
                                            C_LE("dilation", 4) };
static const RadConstraint cGdnRecurrent[] = { C_GE("q_len", 1), C_EQ("head_k", 128),
                                              C_EQ("head_v", 128) };
static const RadConstraint cKdaUpdate[] = {
    C_GE("M", 1),          C_GE("heads", 1),    C_LE("heads", 65535),
    C_EQ("head_k", 128),   C_EQ("head_v", 128), C_IN("mode", "prefill verify"),
    C_IN("dtype", "bf16"),
};

static const RadConstraint cGdnConvRecurrent[] = { C_GE("q_len", 1), C_EQ("head_k", 128),
                                                   C_EQ("head_v", 128), C_EQ("conv_width", 4) };
static const RadConstraint cExpertScaleBf16[] = { C_IN("dtype", "bf16"), C_GE("top_k", 1),
                                                  C_LE("top_k", 64), C_GE("n_expert", 1) };
static const RadConstraint cMoeRouterTopkBf16[] = { C_IN("dtype", "bf16"), C_GE("top_k", 1),
                                                    C_LE("top_k", 32), C_GE("n_expert", 1),
                                                    C_LE("n_expert", 512) };
static const RadConstraint cMoeScatter[] = { C_GE("n_expert", 1), C_LE("n_expert", 512),
                                             C_GE("top_k", 1) };
/* The grouped E4M3 GEMM: the model's `fp8a8f32` experts (E4M3 codes, f32 scales per 128 of K,
 * the kServed table in arch/common/rad_block_moe_fp8.h). The `mxfp4a8` experts (E2M1 weights,
 * E8M0 scales) are a different dtype and a different row. */
static const RadConstraint cMoeGemmQ[] = { C_IN("dtype", "fp8a8f32"), C_GE("n_expert", 1),
                                           C_GE("top_k", 1) };
static const RadConstraint cMoeGemmMxfp4[] = { C_IN("dtype", "mxfp4a8"), C_GE("n_expert", 1),
                                               C_GE("top_k", 1) };
static const RadConstraint cMoeGemmQW4A8[] = { C_IN("dtype", "w4a8"), C_GE("n_expert", 1),
                                               C_GE("top_k", 1) };
/* The w4nl codebook experts: an E4M3 scale a 64 of K under a fixed f32 second level and a
 * sixteen-entry table, so `group` stays the op's base 128 (the scale grid is 1 x group/2, as
 * libref derives it) while the row folds every 64. `a_group` is optional and left unconstrained:
 * its default is `group`, and the launcher accepts 128 or 256. */
static const RadConstraint cMoeGemmQW4NL[] = { C_IN("dtype", "w4nl64a8i8h"), C_GE("n_expert", 1),
                                               C_LE("n_expert", 512), C_GE("top_k", 1),
                                               C_EQ("group", 128) };
static const RadConstraint cMoeGemmW4A4[] = { C_IN("dtype", "w4a4"), C_GE("n_expert", 1),
                                                C_GE("top_k", 1) };
static const RadConstraint cMoeGather[] = { C_GE("M", 1), C_GE("n", 1), C_GE("top_k", 1),
                                             C_IN("dtype", "bf16") };
/* No constraint on `n_local`: it is optional, and a band that leaves it out (every expert local)
 * would not resolve to a row that constrains it; the kernel reads it absent as `n_expert`. */
static const RadConstraint cRouterTopkScatter[] = { C_GE("M", 1), C_GE("n_expert", 1),
                                                    C_GE("top_k", 1), C_LE("n_expert", 512),
                                                    C_IN("dtype", "bf16") };
/* The fused router GEMM's decode band: M <= 64 (past it the router is the tiled GEMM the caller
 * issues separately), K on the 32-code chunk the tiled arm's swizzle needs, and the top-k/sort
 * bounds of router_topk_scatter. No constraint on n_local, expert_base or protect: all optional. */
static const RadConstraint cRouterGemmTopkScatter[] = { C_GE("M", 1), C_LE("M", 64),
                                                        C_GE("K", 1), C_DIV("K", 32),
                                                        C_GE("n_expert", 1), C_LE("n_expert", 512),
                                                        C_GE("top_k", 1), C_LE("top_k", 32),
                                                        C_IN("dtype", "bf16") };
static const RadConstraint cRouterSigmoid[] = { C_GE("M", 1), C_GE("n_expert", 1),
                                                C_GE("top_k", 1), C_LE("n_expert", 512),
                                                C_IN("dtype", "bf16") };
static const RadConstraint cRouterSigmoidGroup[] = { C_GE("M", 1),        C_LE("M", 65535),
                                                     C_GE("n_expert", 1), C_LE("n_expert", 512),
                                                     C_GE("top_k", 1),    C_LE("top_k", 32),
                                                     C_GE("n_group", 1),  C_LE("n_group", 32),
                                                     C_IN("dtype", "f32") };
static const RadConstraint cMoeGemmBf16[] = { C_IN("dtype", "bf16"), C_GE("n_expert", 1),
                                              C_GE("top_k", 1) };
/* The routed IQR GEMM: the model's `i8a8h` experts (a computed trellis under one f32 scale a row,
 * the FFN form arch/common/rad_block_moe_fp8.h selects), with the activation's 128-K scale block. */
static const RadConstraint cMoeGemmIqr[] = { C_IN("dtype", "i8a8h"), C_GE("n_expert", 1),
                                             C_LE("n_expert", 512), C_GE("top_k", 1),
                                             C_EQ("group", 128) };
/* The vocabulary projection has no `dtype` constraint because its band carries no `dtype` key: the
 * head's dtype is a property of the weight (libref checks it with a `check_dtypes` hook). A
 * constraint on a key the band lacks does not hold, which would leave only the host kernel and the
 * engine would refuse to start, so the shim refuses an unsupported dtype instead.
 * `n_embd % 32` is the entry's chunk rule, stated so a band outside it is refused at declare rather
 * at launch. `n_vocab >= 4` is the store's granularity (16-byte runs); a narrower band (a
 * shard's padding, a 1-id test vocabulary) is refused here. Above that floor the grid spans the
 * vocabulary at every M, so every vocabulary width runs the same kernel. */
static const RadConstraint cLogitsGemm[] = { C_GE("M", 1), C_GE("n_vocab", 4),
                                             C_DIV("n_embd", 32) };
/* The quantised GEMM above decode heights. Here `dtype` is the operand scheme (gemm_q_shape
 * switches on it), so C_IN filters rows, unlike logits_gemm, where the band's dtype names the
 * activation. M >= 17 complements gemm_nt_q_fp8a8_m16's M <= 16; a higher floor would leave batched
 * decode at 17..64 rows unserved. The tile's k-chunk (32) divides the container's scale block
 * (128), so `group` is an equality and `K` a divisibility. */
static const RadConstraint cGemmQFP8Tiled[] = { C_IN("dtype", "fp8a8 fp8a8f32"),
                                                C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 17),
                                                C_DIV("K", 128) };
/* The OCP-MXFP4 rows: e2m1 weight codes with one e8m0 scale per 32 of K, the activation scaled per
 * 128. Both are fixed by the format, so the `group` key has no kernel-side meaning and is not
 * constrained (the reference derives the planes from it). K % 128 covers the 32-block, the bands
 * split at M = 16 / 17 as for fp8a8, and the dtype names the whole scheme. */
static const RadConstraint cGemmQMxfp4a8M16[]  = { C_IN("dtype", "mxfp4a8"),
                                                   C_GE("M", 1),
                                                   C_LE("M", 16), C_DIV("K", 128), C_GE("N", 1) };
static const RadConstraint cGemmQMxfp4a8Tiled[] = { C_IN("dtype", "mxfp4a8"),
                                                    C_GE("M", 17),
                                                    C_DIV("K", 128) };

/* The symmetric int4 / int8 weight grids at decode heights: the block is the weight's scale group
 * (128) and the band is M <= 16. No row serves the 8-bit grid at prefill heights. */
static const RadConstraint cGemmQW4A8M16[]  = { C_IN("dtype", "w4a8"),
                                                C_GE("M", 1), C_LE("M", 16),
                                                C_DIV("K", 128), C_GE("N", 1) };
static const RadConstraint cGemmQW8A8M16[]  = { C_IN("dtype", "w8a8"),
                                                C_GE("M", 1), C_LE("M", 16),
                                                C_DIV("K", 128), C_GE("N", 1) };

/* The IQR weight at decode heights (r3d_iqr.h): the rotated int8 activation against a computed
 * trellis or a table decoded where it is read. `group` is the activation's scale width and the
 * weight's 128-K block; N is whole 16-row tiles and a trellis run divides it, which the shim
 * checks because the run rides in the weight's private dtype. */
static const RadConstraint cGemmQIqrM16[]   = { C_IN("dtype", "i8a8h"), C_EQ("group", 128),
                                                C_GE("M", 1), C_LE("M", 16), C_DIV("K", 128),
                                                C_DIV("N", 16) };
/* The same weight at any M: declare resolves the architecture at its widest bucket, so a row for
 * M > 16 must exist even before the tiled kernel does. */
static const RadConstraint cGemmQIqrTiled[] = { C_IN("dtype", "i8a8h"), C_EQ("group", 128),
                                                C_GE("M", 1), C_DIV("K", 128), C_DIV("N", 16) };
/* IQ8R's 256-wide scale group (arch/common/rad_fp8.h's linear_is_iq8r): the same rotated int8
 * grid and code order, one bf16 scale per (row, 256) folded once over two 128-K blocks. */
static const RadConstraint cGemmQIqrM16W256[]   = { C_IN("dtype", "i8a8h"), C_EQ("group", 256),
                                                    C_GE("M", 1), C_LE("M", 16), C_DIV("K", 256),
                                                    C_DIV("N", 16) };
static const RadConstraint cGemmQIqrTiledW256[] = { C_IN("dtype", "i8a8h"), C_EQ("group", 256),
                                                    C_GE("M", 1), C_DIV("K", 256), C_DIV("N", 16) };

/* The w8a8 wave-matrix arm at prefill heights. 17 <= M <= 127 complements cGemmQW8A8M16's
 * M <= 16, so the two rows never compete for a band. K % 128 is the scale group. */
static const RadConstraint cGemmQW8A8M64[]  = { C_IN("dtype", "w8a8"),
                                                C_GE("M", 17), C_LE("M", 127),
                                                C_DIV("K", 128), C_GE("N", 1) };

/* The tiled bf16 GEMM at prefill heights. M >= 17 complements cGemmM16's M <= 16, so the two rows
 * never compete for a band: the skinny shape (a wave a run of columns, the weight streamed through
 * registers) suits up to 16 rows and the 128x128 tile suits more. K % 32 is the tile's k-chunk and
 * N >= 8 the bf16 store run. */
static const RadConstraint cGemmNTTiled[] = { C_IN("dtype", "bf16"), C_GE("M", 17), C_DIV("K", 32),
                                              C_GE("N", 8) };
/* The quantised GEMM: the scheme names the row, the weight's block is the container's, and a split
 * must end on a block boundary (gemm_tune_valid). `group` is required and every band names it, so
 * the row constrains it. `pb1` and `pb2` are optional, and a constraint on a key the band omits does
 * not hold, so constraining them would send every GEMM band of a container using the default block
 * to the reference's host kernel. The launcher defaults them to `group` (r3d_gemm_fp8a16_nt_m1.hip)
 * and refuses a pb2 that differs from the code group, the container's own invariant. */
static const RadConstraint cGemmQFp8a16[] = { C_IN("dtype", "fp8a16"),
                                              C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 1),
                                              C_LE("M", 16), C_DIV("K", 128), C_GE("N", 1) };
static const RadConstraint cGemmQFp8a8[]  = { C_IN("dtype", "fp8a8"),
                                              C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 1),
                                              C_LE("M", 16), C_DIV("K", 128), C_GE("N", 1) };
/* The int8 GEMM's constraints. No row uses them: a row must name an entry point the library
 * defines, or the plugin fails to load. */
static const RadConstraint cGemmQI8[]     = { C_IN("dtype", "i8a8"),
                                              C_EQ("group", R3D_W8_GROUP), C_GE("M", 1),
                                              C_LE("M", 16), C_DIV("K", 128), C_GE("N", 1) };

/* --- Per-Layer Embedding and HyperConnections (Qwen4-Exp) --- */
static const RadConstraint cPleGate[] = { C_IN("dtype", "bf16"), C_GE("n", 1),
                                          C_LE("n", 4096), C_GE("hc", 2), C_LE("hc", 8) };
static const RadConstraint cEmbedQ[] = { C_IN("dtype", "bf16"), C_GE("n_embd", 1) };
static const RadConstraint cNgramHint[] = { C_GE("heads", 1), C_GE("ngram", 2), C_LE("ngram", 8),
                                            C_GE("hist", 0) };
/* The stream rows are shared by hc_write, hc_enter and hc_project. */
static const RadConstraint cHcStream[] = { C_IN("dtype", "bf16"), C_GE("M", 1),
                                           C_GE("hc", 2), C_LE("hc", 8), C_DIV("n", 2) };
/* The read's plain form also takes a band with no `mix` key; the E4M3 row is the higher-priority
 * one and its constraint on the key is what splits the two. */
static const RadConstraint cHcReadBf16[] = { C_IN("dtype", "bf16"), C_GE("M", 1),
                                             C_GE("hc", 2), C_LE("hc", 8), C_DIV("n", 128),
                                             C_GE("lowrank", 16) };
static const RadConstraint cHcReadE4m3[] = { C_IN("dtype", "bf16"), C_IN("mix", "e4m3"),
                                             C_GE("M", 1), C_GE("hc", 2), C_LE("hc", 8),
                                             C_DIV("n", 128), C_DIV("lowrank", 64) };
/* The MTP head's fc matrices as E4M3 rows: n a multiple of the 128-column scale group, and the
 * whole row's LDS stage (hc + 1 streams) inside a workgroup's 64 KiB. */
static const RadConstraint cMtpEnterE4m3[] = {
    C_GE("M", 1), C_DIV("n", 128), C_LE("n", 5734), C_GE("hc", 2), C_LE("hc", 4),
    C_IN("dtype", "bf16"), C_IN("fc", "e4m3"),
};
/* The rescore's IQR lm rows carry a bf16 scale per 128 of n_embd, so the band's width must be a
 * multiple of 128; the rest is logit_rerank's (R <= 32). */
static const RadConstraint cLogitRerankIqr[] = {
    C_GE("M", 1), C_GE("R", 1), C_LE("R", 32), C_GE("n_vocab", 1), C_DIV("n_embd", 128),
    C_IN("dtype", "bf16"),
};
static const RadConstraint cArHc[] = { C_IN("dtype", "bf16"), C_GE("M", 1), C_EQ("world_size", 1),
                                       C_GE("hc", 2), C_LE("hc", 8), C_DIV("n", 8) };
static const RadConstraint cArGatherHc[] = { C_IN("dtype", "bf16"), C_GE("M", 1),
                                             C_EQ("world_size", 1), C_GE("hc", 2), C_LE("hc", 8),
                                             C_DIV("n", 8), C_GE("top_k", 1),
                                             C_LE("top_k", 64) };

/* The tp collectives' message bands, the widths libr4d states: at 4 and 8 ranks the exact sum is
 * one-shot up to these element counts and two-shot above, so the two-shot band starts one past the
 * bound (a multiple of every DIV below) and the message size, not the caller, picks the form. */
enum {
    R3D_AR_WIDE_ONESHOT_MAX_ELEMS = 524288,
    R3D_AR_8RANK_ONESHOT_MAX_ELEMS = 8192,
    /* The two-shot's upper bound: the kernel's word offsets are int32 at fp32. */
    R3D_AR_TWOSHOT_WIDE_MAX_ELEMS = 268435456
};
static const RadConstraint cArExact[] = {
    C_EQ("world_size", 2), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"), C_P2P,
};
static const RadConstraint cArGather2[] = {
    C_EQ("world_size", 2), C_IN("dtype", "bf16 fp16 f16 fp32 f32"), C_P2P,
};
static const RadConstraint cArGather3[] = {
    C_EQ("world_size", 3), C_IN("dtype", "bf16 fp16 f16 fp32 f32"), C_P2P,
};
static const RadConstraint cArGather4[] = {
    C_EQ("world_size", 4), C_IN("dtype", "bf16 fp16 f16 fp32 f32"), C_P2P,
};
static const RadConstraint cArGather8[] = {
    C_EQ("world_size", 8), C_IN("dtype", "bf16 fp16 f16 fp32 f32"), C_P2P,
};
static const RadConstraint cArExact3[] = {
    C_EQ("world_size", 3), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"), C_P2P,
};
static const RadConstraint cArExact4[] = {
    C_EQ("world_size", 4), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"), C_P2P,
};
static const RadConstraint cArExact8[] = {
    C_EQ("world_size", 8), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"), C_P2P,
};
/* No DIV at three ranks: the two-shot shards unevenly in whole words. */
static const RadConstraint cArTsExact3[] = {
    C_EQ("world_size", 3), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_GE("numel", R3D_AR_WIDE_ONESHOT_MAX_ELEMS + 1),
    C_LE("numel", R3D_AR_TWOSHOT_WIDE_MAX_ELEMS), C_P2P,
};
static const RadConstraint cArTsExact4[] = {
    C_EQ("world_size", 4), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_DIV("numel", 64), C_GE("numel", R3D_AR_WIDE_ONESHOT_MAX_ELEMS + 1),
    C_LE("numel", R3D_AR_TWOSHOT_WIDE_MAX_ELEMS), C_P2P,
};
static const RadConstraint cArTsExact4Lane[] = {
    C_EQ("world_size", 4), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_DIV("numel", 64), C_GE("numel", R3D_AR_WIDE_ONESHOT_MAX_ELEMS + 1),
    C_LE("numel", R3D_AR_TWOSHOT_WIDE_MAX_ELEMS), C_EQ("overlap", 1), C_P2P,
};
static const RadConstraint cArTsExact8[] = {
    C_EQ("world_size", 8), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_DIV("numel", 64), C_GE("numel", R3D_AR_8RANK_ONESHOT_MAX_ELEMS + 1),
    C_LE("numel", R3D_AR_TWOSHOT_WIDE_MAX_ELEMS), C_P2P,
};
static const RadConstraint cArWht6[] = {
    C_EQ("world_size", 2), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 64),
    C_P2P,
};
static const RadConstraint cArExactLane[] = {
    C_EQ("world_size", 2), C_EQ("exact", 1), C_IN("dtype", "bf16 fp16 fp32"),
    C_EQ("overlap", 1), C_P2P,
};
static const RadConstraint cArWht6Lane[] = {
    C_EQ("world_size", 2), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 64),
    C_EQ("overlap", 1), C_P2P,
};
/* DIV 32, not 96: the three-rank split is uneven in whole 32-element groups. */
static const RadConstraint cArTi8Ts3[] = {
    C_EQ("world_size", 3), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 32),
    C_P2P,
};
static const RadConstraint cArTi8Ts4[] = {
    C_EQ("world_size", 4), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 128),
    C_P2P,
};
static const RadConstraint cArTi8Ts4Lane[] = {
    C_EQ("world_size", 4), C_EQ("exact", 0), C_IN("dtype", "bf16 fp16"), C_DIV("numel", 128),
    C_EQ("overlap", 1), C_P2P,
};

/* ================================================================== the rows
 *
 * Declaration order is the tie-break within one priority, so rows that share an op are declared
 * narrowest first.
 */
/* Constraint arrays are declared ahead of the rows that use them, each with a comment naming the
 * row it serves. */

/* ================================================================== the fused-MLP rows
 *
 * Two rows llama graphs ask for that the model unit serves. `silu_mul` constrains only the dtype
 * and an even `n`: `act` is not in this op's parameter list (the op fuses one fixed activation), and
 * a constraint on a key the band lacks does not hold, so the act_mul constraints would match no band.
 * `rmsnorm_quant_fp8` requires the group the kernel covers with a wave to divide `n`; the standalone
 * quantiser's `n % 4` would admit bands this kernel refuses at launch.
 */
static const RadConstraint cSiluMul[]     = { C_IN("dtype", "bf16"), C_DIV("n", 2) };
static const RadConstraint cRmsQuantFp8[] = { C_IN("dtype", "bf16"), C_EQ("group", R3D_FP8_BLOCK),
                                              C_DIV("n", R3D_FP8_BLOCK) };
/* The rotated int8 twin: the rotation is 128 wide and one scale covers it, so the group is fixed
 * and divides n. */
static const RadConstraint cRmsQuantI8g[] = { C_IN("dtype", "bf16"), C_EQ("group", 128),
                                              C_DIV("n", 128) };
/* IQ8R's group: one scale per 256 columns, each 128-wide half rotated on its own. */
static const RadConstraint cRmsQuantI8g256[] = { C_IN("dtype", "bf16"), C_EQ("group", 256),
                                                 C_DIV("n", 256) };

/* The sampler's pre-filters constrain only `n_vocab`. Their bands carry `n_vocab` and `M` (plus
 * `vocab_off`, and `k` for logprobs) and no `dtype` key, so a dtype constraint would match no band;
 * the shims check that the tensors are f32. `k <= 64` is the depth of the logprobs kernel's pool, so
 * the row serves exactly the bands its launcher accepts.
 */
static const RadConstraint cSampleF32[]      = { C_GE("n_vocab", 1) };
static const RadConstraint cSampleLogprobs[] = { C_GE("n_vocab", 1), C_GE("k", 1), C_LE("k", 64) };

/* The attention rows bind the two dtype keys, the head width the kernel's tile is built for, and
 * the q_len bucket boundary the two kernels are split at. `gqa`, `window` and `causal` are left
 * free: the kernel walks the mask for any gqa the shim accepts and any window, and a constraint on
 * them would only exclude bands it serves. */
static const RadConstraint cAttnBf16Decode[]  = { C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
                                                  C_EQ("head_dim", 128), C_LE("q_len", 16) };
static const RadConstraint cAttnBf16Prefill[] = { C_IN("q_dtype", "bf16"), C_IN("kv_dtype", "bf16"),
                                                  C_EQ("head_dim", 128), C_GE("q_len", 17) };
static const RadConstraint cAttnFp8Decode[]   = { C_IN("q_dtype", "bf16"),
                                                  C_IN("kv_dtype", "fp8_e4m3"),
                                                  C_EQ("head_dim", 128), C_LE("q_len", 16) };
static const RadConstraint cAttnFp8Prefill[]  = { C_IN("q_dtype", "bf16"),
                                                  C_IN("kv_dtype", "fp8_e4m3"),
                                                  C_EQ("head_dim", 128), C_GE("q_len", 17) };
/* head_dim 256's int8 rows. The geometry pairs are the two the docs name: Flash-Next's 24 query
 * heads over 2 KV heads with 4-token pages, and the 27B's 24 over 4 with 16-token pages. The
 * block_size constraint is RAD_DERIVED, so the selector skips it while matching and
 * rad_kv_block_size() reads it back off the resolved row. */
static const RadConstraint cAttnH256Gqa12Decode[]  = { C_IN("q_dtype", "bf16"),
                                                       C_IN("kv_dtype", "i8"),
                                                       C_EQ("head_dim", 256), C_EQ("gqa", 12),
                                                       C_EQ("block_size", 4), C_LE("q_len", 16) };
static const RadConstraint cAttnH256Gqa12Prefill[] = { C_IN("q_dtype", "bf16"),
                                                       C_IN("kv_dtype", "i8"),
                                                       C_EQ("head_dim", 256), C_EQ("gqa", 12),
                                                       C_EQ("block_size", 4), C_GE("q_len", 17) };
static const RadConstraint cAttnH256Gqa6Decode[]   = { C_IN("q_dtype", "bf16"),
                                                       C_IN("kv_dtype", "i8"),
                                                       C_EQ("head_dim", 256), C_EQ("gqa", 6),
                                                       C_EQ("block_size", 16), C_LE("q_len", 16) };
static const RadConstraint cAttnH256Gqa6Prefill[]  = { C_IN("q_dtype", "bf16"),
                                                       C_IN("kv_dtype", "i8"),
                                                       C_EQ("head_dim", 256), C_EQ("gqa", 6),
                                                       C_EQ("block_size", 16), C_GE("q_len", 17) };

/* `row_topk` is banded on dtype, so it has two rows. The candidate stages are banded on `n_cand`,
 * capped at the kernel's ceiling: five f32 arrays of n_cand in LDS is 20 KiB of the 64 KiB, and the
 * shim refuses a wider band (RAD_E_SHAPE). The vocab stages carry `n_vocab` and no dtype key; the
 * shim checks that the logits are f32.
 */
static const RadConstraint cRowTopkBf16[] = { C_IN("dtype", "bf16"), C_GE("N", 1), C_GE("R", 1) };
static const RadConstraint cRowTopkF32[]  = { C_IN("dtype", "f32"),  C_GE("N", 1), C_GE("R", 1) };
static const RadConstraint cSampleCand[]  = { C_GE("n_cand", 1), C_LE("n_cand", 1024) };

/* The container's 2-bit vocab weights. `group` is both the weight's K block and the scale's block.
 * K must also be a multiple of the launcher's chunk (2048 / M-tile), which the launcher refuses and
 * the row does not express. */
static const RadConstraint cGemmQW2A8[]  = { C_IN("dtype", "w2a8"), C_EQ("group", R3D_FP8_BLOCK),
                                             C_GE("M", 1), C_LE("M", 16), C_DIV("K", 128),
                                             C_GE("N", 1) };

/* The packed-code grids at the heights gemm_nt_q_w2a8_m16 declines. The grid is the container's --
 * one scale and one zero per `group` of codes on a {1, group} plane -- so `group` is the block, and
 * K divides by it because a staged 32-code chunk must sit inside one scale group. */
static const RadConstraint cGemmQW2A8Tiled[] = { C_IN("dtype", "w2a8 w4a8_asym"),
                                                 C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 17),
                                                 C_DIV("K", 128) };

/* `group` is one key for three things: the weight's K block, its row block and the quantiser's
 * group. `act` is C_IN because it is a required string here, not the optional `mode` form. M <= 64
 * is the launcher's ceiling. */
static const RadConstraint cGemmGated[]  = { C_IN("dtype", "fp8a8"),
                                             C_IN("act", "silu sigmoid"),
                                             C_EQ("group", R3D_FP8_BLOCK), C_GE("M", 1),
                                             C_LE("M", 64), C_DIV("K", 128), C_DIV("N", 2),
                                             C_GE("N", 2) };
/* The same fold over a rotated int8 (i8a8h) IQR weight: the ffn block declares it for M up to 16,
 * N and K multiples of 256 (whole 128-blocks on both sides). */
static const RadConstraint cGemmGatedI8[] = { C_IN("dtype", "i8a8h"),
                                              C_IN("act", "silu sigmoid"),
                                              C_EQ("group", 128), C_GE("M", 1),
                                              C_LE("M", 16), C_DIV("K", 256), C_DIV("N", 256) };
/* IQ8R's gated fold: one i32 sub-sum over two 128-K blocks, one amax and scale over two 128-column
 * halves. The gated width is N/2, so N is a whole 512 for the 256-wide quantiser group. */
static const RadConstraint cGemmGatedI8W256[] = { C_IN("dtype", "i8a8h"),
                                                  C_IN("act", "silu sigmoid"),
                                                  C_EQ("group", 256), C_GE("M", 1),
                                                  C_LE("M", 16), C_DIV("K", 256),
                                                  C_DIV("N", 512) };

/* Every key the walk reads is required and in the band, so all are bound; `top_k <= 256` gives each
 * candidate one lane. `anchor_stride` is deliberately unbound: it is optional, defaults to 1, and
 * the kernel serves any value >= 1, so a band that omits it must still resolve. */
static const RadConstraint cDflashSelect[] = { C_IN("dtype", "bf16"), C_GE("M", 1),
                                               C_GE("steps", 1), C_GE("top_k", 1),
                                               C_LE("top_k", 256), C_GE("rank", 1),
                                               C_GE("n_vocab", 1) };

/* The vision tower's two remaining bands (W10-lite). `attn_dense` is non-causal full attention
 * within each cu segment, q/k/v as the column slices of the fused qkv projection (three separate
 * GEMMs' outputs in the reference; the tower's buffer form in the graph). `rope` is the 2-D axial
 * rotary the ViT applies before that attention, over [2, M] (row, column) positions and two
 * 18-frequency ladders.
 *
 * The selector would otherwise fall back to the unconstrained rope row, which refuses axial at
 * launch; the axial row's priority is what puts it first. Its constraints name only parameters the
 * band states, because a constraint on an optional parameter the band omits does not hold. */
static const RadConstraint cAttnDenseH72[] = {
    C_EQ("head_dim", 72), C_EQ("gqa", 1), C_EQ("causal", 0), C_IN("dtype", "bf16")
};
static const RadConstraint cRopeAxialH72[] = {
    C_EQ("head_dim", 72), C_EQ("rotary_dim", 72), C_IN("mode", "axial")
};

/* The N = 1 GEMV serves a band the m16 row also matches, so it must outrank it. The two rows are
 * disjoint on N only for this row's exact equality, which the selector reads before priority; the
 * extra rank states the intent for a reader and keeps a future m16 tweak from taking the gate. */
enum { R3D_PRIO_GEMM_N1 = R3D_PRIO_GEMM_M16 + 10 };

/* The vision tower's linear and its position table, the two ops W10-lite left to the host. Both
 * are bf16 in the tower, and `M` is the pass's patch count, so the band is a floor: a kernel that
 * cannot carry an extent refuses it at launch. */
static const RadConstraint cGemmVision[] = {
    C_IN("dtype", "bf16"), C_GE("M", 1), C_GE("N", 1), C_GE("K", 1)
};
static const RadConstraint cGridEmbedVision[] = {
    C_IN("dtype", "bf16"), C_GE("M", 1), C_GE("n", 1), C_GE("side", 1)
};

static const RadKernelInfo kKernels[] = {
/* ------------------------------------------------------------------ r3d_model_bf16.hip
 *
 * One workgroup a row. The `shape` column states the geometry the grid is built for; the
 * `computes` column states the arithmetic, in the order the kernel performs it.
 */
{ "rmsnorm_bf16", "rmsnorm", "norm",
  "root-mean-square norm with a gain, two passes over the row",
  "any n; a workgroup a row; the sum of squares accumulates in f32 and only the store narrows",
  "bf16 x and y, bf16 or f32 gain, fp32 reduction",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cNormBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_rmsnorm, nullptr, nullptr, nullptr,
  r3d_shape_rmsnorm },

{ "rmsnorm_f32", "rmsnorm", "norm",
  "root-mean-square norm with a gain, two passes over the row",
  "any n; a workgroup a row",
  "f32 x, y and gain, fp32 reduction",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cNormF32), NOTUNE,
  nullptr, nullptr, r3d_rad_rmsnorm, nullptr, nullptr, nullptr,
  r3d_shape_rmsnorm },

{ "layernorm_bf16", "layernorm", "norm",
  "layer norm with a gain and an optional bias, mean and variance in two passes",
  "any n; a workgroup a row; both passes accumulate in f32",
  "bf16 x, y, gain and bias, fp32 reduction",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cNormBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_layernorm, nullptr, nullptr, nullptr,
  r3d_shape_layernorm },

{ "add_bf16", "add", "elem",
  "a + b, with b's row broadcast and an optional divisor",
  "any n; a workgroup a row; f32 arithmetic, one narrowing at the store",
  "bf16 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_add, nullptr, nullptr, nullptr,
  r3d_shape_add },

{ "add_f32", "add", "elem",
  "a + b, with b's row broadcast and an optional divisor",
  "any n; a workgroup a row",
  "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r3d_rad_add, nullptr, nullptr, nullptr,
  r3d_shape_add },

{ "mul_bf16", "mul", "elem",
  "a * b, with b's row broadcast",
  "any n; a workgroup a row",
  "bf16 in and out, f32 product, one narrowing at the store",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_mul, nullptr, nullptr, nullptr,
  r3d_shape_mul },

{ "mul_f32", "mul", "elem",
  "a * b, with b's row broadcast",
  "any n; a workgroup a row",
  "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r3d_rad_mul, nullptr, nullptr, nullptr,
  r3d_shape_mul },

/* The casts, one row per pair the engine issues. bf16 to f32 and back is the residual stream's
 * width change; i64 to f32 is a position table read by a rotary; the integer pairs move token ids
 * between planes and must not pass through f32, which loses the low bits above 2^24. */
{ "cast_bf16_f32", "cast", "elem", "bf16 to f32, exact", "any n; a wave 128 elements",
  "bf16 in, f32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastBf16F32), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_f32_bf16", "cast", "elem", "f32 to bf16, one rounding", "any n; a wave 128 elements",
  "f32 in, bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastF32Bf16), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_bf16_bf16", "cast", "elem", "bf16 to bf16, a copy", "any n; a wave 128 elements",
  "bf16 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_f32_f32", "cast", "elem", "f32 to f32, a copy", "any n; a wave 128 elements",
  "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastF32), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_i64_f32", "cast", "elem", "i64 to f32", "any n; a wave 128 elements",
  "i64 in, f32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastI64F32), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_i32_i64", "cast", "elem", "i32 to i64, exact", "any n; a wave 128 elements",
  "i32 in, i64 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastI32I64), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "cast_i64_i32", "cast", "elem", "i64 to i32, the ids' narrowing", "any n; a wave 128 elements",
  "i64 in, i32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cCastI64I32), NOTUNE,
  nullptr, nullptr, r3d_rad_cast, nullptr, nullptr, nullptr, r3d_shape_cast },

{ "silu_bf16", "silu", "elem", "x * sigmoid(x), the sigmoid in f32", "any n; a wave 128 elements",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_silu, nullptr, nullptr, nullptr, r3d_shape_silu },

{ "silu_f32", "silu", "elem", "x * sigmoid(x)", "any n; a wave 128 elements", "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r3d_rad_silu, nullptr, nullptr, nullptr, r3d_shape_silu },

{ "gelu_bf16", "gelu", "elem",
  "the exact erf GELU: 0.5 * x * (1 + erf(x / sqrt(2)))",
  "any n; a wave 128 elements; erff, the accurate form, not the tanh approximation",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_gelu, nullptr, nullptr, nullptr, r3d_shape_gelu },

{ "sigmoid_bf16", "sigmoid", "elem", "1 / (1 + exp(-x))", "any n; a wave 128 elements",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_sigmoid, nullptr, nullptr, nullptr, r3d_shape_sigmoid },

{ "act_mul_silu_bf16", "act_mul", "elem",
  "silu(gate) * up, the gated feed-forward's middle",
  "any n; a workgroup a row; the gate is operand 0",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cActSilu), NOTUNE,
  nullptr, nullptr, r3d_rad_act_mul, nullptr, nullptr, nullptr, r3d_shape_act_mul },

{ "act_mul_gelu_bf16", "act_mul", "elem",
  "gelu(gate) * up, the exact erf form",
  "any n; a workgroup a row",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cActGelu), NOTUNE,
  nullptr, nullptr, r3d_rad_act_mul, nullptr, nullptr, nullptr, r3d_shape_act_mul },

{ "act_mul_sigmoid_bf16", "act_mul", "elem", "sigmoid(gate) * up", "any n; a workgroup a row",
  "bf16 in and out, fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cActSigmoid), NOTUNE,
  nullptr, nullptr, r3d_rad_act_mul, nullptr, nullptr, nullptr, r3d_shape_act_mul },

{ "softmax_f32", "softmax", "elem",
  "row softmax with the maximum subtracted, the denominator summed in ascending order",
  "any n; a workgroup a row",
  "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSoftmax), NOTUNE,
  nullptr, nullptr, r3d_rad_softmax, nullptr, nullptr, nullptr, r3d_shape_softmax },

{ "scale_bf16", "scale", "elem", "x * s, one scalar over the whole output",
  "any n; a wave 128 elements", "bf16 in and out, fp32 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cBf16Row), NOTUNE,
  nullptr, nullptr, r3d_rad_scale, nullptr, nullptr, nullptr, r3d_shape_scale },

{ "scale_f32", "scale", "elem", "x * s, one scalar over the whole output",
  "any n; a wave 128 elements", "f32 in and out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cF32Row), NOTUNE,
  nullptr, nullptr, r3d_rad_scale, nullptr, nullptr, nullptr, r3d_shape_scale },

{ "fill_bf16", "fill", "elem", "the value over [M, n]", "any n; a wave 128 elements",
  "bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cFillBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_fill, nullptr, nullptr, nullptr, r3d_shape_fill },

{ "fill_f32", "fill", "elem", "the value over [M, n]", "any n; a wave 128 elements", "f32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cFillF32), NOTUNE,
  nullptr, nullptr, r3d_rad_fill, nullptr, nullptr, nullptr, r3d_shape_fill },

{ "fill_i32", "fill", "elem", "the value over [M, n], the value as an integer",
  "any n; a wave 128 elements", "i32 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cFillI32), NOTUNE,
  nullptr, nullptr, r3d_rad_fill, nullptr, nullptr, nullptr, r3d_shape_fill },

{ "fill_i64", "fill", "elem", "the value over [M, n], the value as an integer",
  "any n; a wave 128 elements", "i64 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cFillI64), NOTUNE,
  nullptr, nullptr, r3d_rad_fill, nullptr, nullptr, nullptr, r3d_shape_fill },

{ "rope_bf16", "rope", "attn",
  "rotary positions applied in place to a fused qkv row, neox and gptj pair layouts",
  "head_dim even; rotary_dim optional and even, defaulting to head_dim; a wave a head; the "
  "angles from accurate powf and sincosf, the position divided by `scale`",
  "bf16 qkv in place, i32 positions, fp32 angles",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRope), NOTUNE,
  nullptr, nullptr, r3d_rad_rope, nullptr, nullptr, nullptr, r3d_shape_rope },

{ "kv_store_bf16", "kv_store", "attn",
  "K and V into a paged cache at the slot mapping, bf16 to bf16",
  "any head_dim and block size; a wave a (token, head) pair; a negative slot skipped",
  "bf16 k and v, i32 slot mapping, bf16 cache",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKvBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_kv_store, nullptr, nullptr, nullptr, r3d_shape_kv_store },

{ "kv_store_f16", "kv_store", "attn",
  "K and V into a paged cache at the slot mapping, bf16 to f16",
  "any head_dim and block size; a wave a (token, head) pair",
  "bf16 k and v, i32 slot mapping, f16 cache",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKvF16), NOTUNE,
  nullptr, nullptr, r3d_rad_kv_store, nullptr, nullptr, nullptr, r3d_shape_kv_store },

{ "kv_store_fp8", "kv_store", "attn",
  "K and V into a paged cache at the slot mapping, encoding E4M3 with the store's scale",
  "any head_dim and block size; a wave a (token, head) pair; the scale divides before the encode",
  "bf16 k and v, i32 slot mapping, fp8_e4m3 cache",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKvFp8), NOTUNE,
  nullptr, nullptr, r3d_rad_kv_store, nullptr, nullptr, nullptr, r3d_shape_kv_store },

{ "embed_lookup_bf16", "embed_lookup", "gemm",
  "the codebook row an id names, scaled, with out-of-shard ids yielding a zero row",
  "any n_embd; a workgroup a row of the table",
  "bf16 table and output, i32 ids",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cEmbed), NOTUNE,
  nullptr, nullptr, r3d_rad_embed_lookup, nullptr, nullptr, nullptr, r3d_shape_embed_lookup },

{ "embed_lookup_iqr", "embed_lookup", "gemm",
  "embed_lookup over an IQR table (r3d_iqr.h's embed_off): each requested vocabulary row is "
  "decoded from its codes, the rotated table's 128-wide Hadamard undone in f32, and scaled",
  "n_embd a multiple of 128; a block a (token, 128-block) pair, 128 threads; a trellis run of 256 "
  "rows",
  "i32 tokens, an IQR table, bf16 x; ids outside the shard yield a zero row",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cEmbedIqr), NOTUNE,
  nullptr, nullptr, r3d_rad_embed_lookup_iqr,
  nullptr, r3d_rad_layout_iqr_embed, r3d_rad_relayout_iqr_embed,
  r3d_shape_embed_lookup, nullptr },

{ "logits_gemm_vocab_iqr", "logits_gemm_vocab", "vocab",
  "the vocabulary-weighted IQR head: each class's computed-trellis codes decoded once a 128-K "
  "block and dotted against the rotated, sixteen-bit activation in exact int8 sums; the class's "
  "ids scatter the f32 logits",
  "any M; n_embd a multiple of 256; each class a multiple of 256 rows; runs of 256 rows",
  "bf16 x rotated by fwht128 and quantised per (token, 128) to q = 128 hi + lo in the call's "
  "scratch; classes of tcq_code=mad codes under one f32 scale a row; F32 logits out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cLogitsVocabIqr), NOTUNE,
  nullptr, nullptr, r3d_rad_logits_gemm_vocab_iqr, r3d_logits_gemm_vocab_iqr_scratch,
  r3d_rad_layout_iqr_vocab, r3d_rad_relayout_iqr_vocab,
  nullptr, r3d_rad_unrelayout_iqr_vocab },

{ "gather_rows_bf16", "gather_rows", "elem", "y[r] = x[idx[r]]",
  "any n; a workgroup a row", "bf16 rows, i32 indices",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGatherBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_gather_rows, nullptr, nullptr, nullptr, r3d_shape_gather_rows },

{ "gather_rows_f32", "gather_rows", "elem", "y[r] = x[idx[r]]",
  "any n; a workgroup a row", "f32 rows, i32 indices",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGatherF32), NOTUNE,
  nullptr, nullptr, r3d_rad_gather_rows, nullptr, nullptr, nullptr, r3d_shape_gather_rows },

{ "scatter_rows_bf16", "scatter_rows", "elem", "x[idx[r]] = v[r]",
  "any n; a workgroup a row; distinct indices, so no two rows write one destination",
  "bf16 rows, i32 indices",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cScatterBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_scatter_rows, nullptr, nullptr, nullptr, r3d_shape_scatter_rows },

/* Opaque byte rows, the KV tier's traffic: one layer's fragment of one block, copied without an
 * arithmetic round trip. The u8 pair is separate from the bf16/f32 rows because ld_elem's E4M3
 * codec need not preserve a NaN payload, and a KV fragment is bytes, not values. */
{ "gather_rows_u8", "gather_rows", "elem", "y[r] = x[idx[r]] over opaque bytes",
  "n a multiple of 4; a workgroup a row, four bytes a thread",
  "u8 rows, i32 indices",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cOpaqueRowU8), NOTUNE,
  nullptr, nullptr, r3d_rad_gather_rows, nullptr, nullptr, nullptr, r3d_shape_gather_rows },

{ "scatter_rows_u8", "scatter_rows", "elem", "x[idx[r]] = v[r] over opaque bytes",
  "n a multiple of 4; distinct indices; a workgroup a row, four bytes a thread",
  "u8 rows, i32 indices",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cOpaqueRowU8), NOTUNE,
  nullptr, nullptr, r3d_rad_scatter_rows, nullptr, nullptr, nullptr, r3d_shape_scatter_rows },

/* ------------------------------------------------------------------ r3d_quant_fp8.hip
 *
 * Plain row-major code planes with one f32 scale per (row, group). A dot2 GEMM reads codes in
 * stored order, so they carry no fragment permutation, and a container converted on this card is
 * readable by libref and by `rad-convert` on any card. Only `quant_act_i8` keeps libref's
 * permutation, because its order is part of the op (see the row).
 */
{ "quant_act_fp8_bf16", "quant_act_fp8", "quant",
  "symmetric per-(row, group) E4M3: scale = amax * (1/448), the codes the value times 1/scale",
  "group 128; n a multiple of 4; a wave covers four columns a lane, so a group is a quarter-wipe",
  "bf16 in, fp8_e4m3 codes, f32 scales",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQuantFp8), NOTUNE,
  nullptr, nullptr, r3d_rad_quant_act_fp8, nullptr, nullptr, nullptr,
  r3d_shape_quant_act_fp8 },

/* The per-row int8, in libref's step order. The order belongs to the op: the codes are the WMMA
 * A-fragment's and the oracle compares the plane, so the kernel's store address applies the
 * permutation. libr3d's own int8 GEMM reads the row-major `quant_act_i8g` plane (two rows below);
 * this row exists so an architecture that emits the op resolves to a kernel here rather than to the
 * reference. */
{ "quant_act_i8_bf16", "quant_act_i8", "quant",
  "symmetric int8 with one scale per row: scale = max(amax, 1e-8) / 127, ties to even, saturating",
  "n a multiple of 16, a whole WMMA step; a wave a row of the row sums; the codes stored in the "
  "step order, the middle two dwords of every 16 bytes exchanged",
  "bf16 in, i8 codes, f32 scale per row and the group-major row sums",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQuantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_quant_act_i8, nullptr, nullptr, nullptr,
  r3d_shape_quant_act_i8 },

{ "quant_act_i8g_bf16", "quant_act_i8g", "quant",
  "symmetric per-(row, group) int8: scale = amax / 127, 1 for an all-zero group, ties to even",
  "group 128; n a multiple of 4; a wave covers four columns a lane, so a lane's codes are one "
  "dword and a wave's are 128 contiguous bytes",
  "bf16 in, i8 codes row-major, f32 scales",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQuantI8g), NOTUNE,
  nullptr, nullptr, r3d_rad_quant_act_i8g, nullptr, nullptr, nullptr,
  r3d_shape_quant_act_i8g },

{ "had_quant_act_i8_bf16", "had_quant_act_i8", "quant",
  "a block Hadamard along K, then per-row symmetric int8: the rotation shrinks the row maximum",
  "group 128/256/512, n a multiple of 16; the FWHT runs once per wave (K/group <= 8) so its DPP "
  "exchanges stay out of a loop; the codes in the int8 WMMA step order",
  "bf16 in, i8 codes, f32 scale per row",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cHadQuantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_had_quant_act_i8, nullptr, nullptr, nullptr,
  r3d_shape_had_quant_act_i8 },

{ "gelu_mul_quant_i8_bf16", "gelu_mul_quant_i8", "quant",
  "gelu(gate) * up over a fused [M, 2n] projection, then per-row symmetric int8",
  "the tanh-form product (0.5*g*(1+tanh(k0*(g+k1*g^3)))) rounded to bf16, the absmax, scale and "
  "codes taken over those values; n a multiple of 16, codes in the int8 WMMA step order",
  "bf16 gate_up, i8 codes, f32 scale per row, optional bf16 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGeluMulQuantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_gelu_mul_quant_i8, nullptr, nullptr, nullptr,
  r3d_shape_gelu_mul_quant_i8 },

{ "kv_len_trim_i32", "kv_len_trim", "kv",
  "per-sequence context lengths less `trim`, floored at zero: the slice sees its own last row",
  "one workgroup of 64, a thread a sequence; `trim` is the tail operand's extent, its values "
  "unread; max_seqs <= 64",
  "i32 seqused, tail (extent), out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKvLenTrim), NOTUNE,
  nullptr, nullptr, r3d_rad_kv_len_trim, nullptr, nullptr, nullptr,
  r3d_shape_kv_len_trim },

{ "qsa_tail_store_bf16", "qsa_tail_store", "kv",
  "the QSA indexer's tail into a per-layer linear KV state: position p lands at row p % ring",
  "one block a token; the sequence by a linear cu_seqlens scan; only the last ring tokens of a "
  "chunk write (p and p + ring share a row); ring = ratio + n_spec",
  "bf16 k, i32 pos/cu/sidx, bf16 state [slots,1,ring,n] inout",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQsaTailStore), NOTUNE,
  nullptr, nullptr, r3d_rad_qsa_tail_store, nullptr, nullptr, nullptr,
  r3d_shape_qsa_tail_store },

{ "qsa_block_key_bf16", "qsa_block_key", "attn",
  "the QSA indexer's compressed key: mean of `ratio` raw keys, RMS-normed, rotated at the "
  "block's first position",
  "one workgroup a block; the mean is rounded to bf16 before the norm (the reference's order); "
  "`pos` is a per-block list or [M,3] components under mode/sections, `pos0` the contiguous "
  "tiling; page -1 writes nothing, n <= 4096 (one LDS row)",
  "bf16 k [M*ratio, n], f32 gain, optional i32 page/pos; bf16 out, a block a row",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQsaBlockKey), NOTUNE,
  nullptr, nullptr, r3d_rad_qsa_block_key, nullptr, nullptr, nullptr,
  r3d_shape_qsa_block_key },

/* The indexer's score. A thread owns one (query, block) pair and walks its heads and lanes in the
 * reference's ascending f32 order, so the scores are bit-identical to libref's; the workgroup's
 * lanes cover the block axis and share the row's sequence lookup. `nc` bounds the block loop, and
 * a negative page id writes -inf, which qsa_select reads as absent. */
{ "qsa_score_bf16", "qsa_score", "kv",
  "the QSA indexer's block score: relu per (block, head), summed over heads, over 1/sqrt(n)",
  "any n, head count and block count; a thread a (query, block); the dot is an ascending f32 "
  "chain, the relus summed in head order; bt's page ids bound the store",
  "bf16 q, bf16/f16/f32 bk (widened exactly to f32), i32 bt/cu/nc, f32 scores; a "
  "negative page scores -inf and a page past the store is left unwritten",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQsaScoreBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_qsa_score, nullptr, nullptr, nullptr,
  r3d_shape_qsa_score },

{ "qsa_score_f16", "qsa_score", "kv",
  "the QSA indexer's block score: relu per (block, head), summed over heads, over 1/sqrt(n)",
  "any n, head count and block count; a thread a (query, block); f16 queries widened exactly",
  "f16 q, bf16/f16/f32 bk, i32 bt/cu/nc, f32 scores; a negative page scores -inf",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQsaScoreF16), NOTUNE,
  nullptr, nullptr, r3d_rad_qsa_score, nullptr, nullptr, nullptr,
  r3d_shape_qsa_score },

{ "qsa_score_f32", "qsa_score", "kv",
  "the QSA indexer's block score: relu per (block, head), summed over heads, over 1/sqrt(n)",
  "any n, head count and block count; a thread a (query, block); the reference's f32 chain",
  "f32 q, bf16/f16/f32 bk, i32 bt/cu/nc, f32 scores; a negative page scores -inf and a page "
  "past the store is left unwritten",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQsaScoreF32), NOTUNE,
  nullptr, nullptr, r3d_rad_qsa_score, nullptr, nullptr, nullptr,
  r3d_shape_qsa_score },

{ "qsa_select_bf16", "qsa_select", "attn",
  "the topk highest-scoring complete blocks a query has passed, as page ids, then the partial page",
  "a workgroup a query; four-digit radix threshold and workgroup prefix-count emit; ties to the "
  "lower block index, never to atomic order; signed ranks by the float total order, unsigned by "
  "max(score, 0)",
  "f32 score [M,blocks], i32 bt [seqs,blocks]/nc/pos/cu, i32 sel [M,topk+1]/nsel/seqused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQsaSelect), NOTUNE,
  nullptr, nullptr, r3d_rad_qsa_select, nullptr, nullptr, nullptr,
  r3d_shape_qsa_select },
{ "qsa_work_bf16", "qsa_work", "kv",
  "the blocks completing this step gathered into the flat list qsa_block_key reads",
  "one workgroup a potential block: ceil(T / ratio) + seqs bounds the count; keys come from this "
  "step's k where the position is in the chunk and from the tail state where it is not (q % ring); "
  "slots past the real count report page -1; nc is each sequence's complete-block count",
  "bf16 k, tail state and stage; i32 pos/cu/sidx/bt/page/bpos/nc; optional i32 rope for M-RoPE",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQsaWork), NOTUNE,
  nullptr, nullptr, r3d_rad_qsa_work, nullptr, nullptr, nullptr,
  r3d_shape_qsa_work },

{ "gated_had_quant_i8_bf16", "gated_had_quant_i8", "quant",
  "a gated product (silu(a)*b or sigmoid(a)*b), a block Hadamard along n, then per-row int8",
  "mode 0 down_proj's input, mode 1 o_proj's input; the rotation shrinks the row maximum; the "
  "FWHT runs once per wave (n == group) so its DPP stages stay out of a loop; codes in step order",
  "bf16 a (gate) and b (up, optional fused), i8 codes, f32 scale per row",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGatedHadQuantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_gated_had_quant_i8, nullptr, nullptr, nullptr,
  r3d_shape_gated_had_quant_i8 },

{ "had_quant_act_fp8_bf16", "had_quant_act_fp8", "quant",
  "a 128-wide block Hadamard, then per-group E4M3: the rotation shrinks the group maximum",
  "n == group == 128 (single-group form, the FWHT runs once per wave so its DPP stages stay out of "
  "a loop); the codes row-major, the scale amax/448 (the rotation's 1/128 is the weight plane's)",
  "bf16 in, E4M3 codes, f32 scale per group",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cHadQuantFp8), NOTUNE,
  nullptr, nullptr, r3d_rad_had_quant_act_fp8, nullptr, nullptr, nullptr,
  r3d_shape_had_quant_act_fp8 },

{ "gated_had_quant_fp8_bf16", "gated_had_quant_fp8", "quant",
  "a gated product (silu or sigmoid), a 128-wide block Hadamard, then per-group E4M3",
  "n == group == 128 (single-group form, the FWHT runs once per wave so its DPP stages stay out of "
  "a loop); the gated product rounded to bf16, the codes row-major, the scale amax/448",
  "bf16 gate_up [M,2n], E4M3 codes, f32 scale per group, optional bf16 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGatedHadQuantFp8), NOTUNE,
  nullptr, nullptr, r3d_rad_gated_had_quant_fp8, nullptr, nullptr, nullptr,
  r3d_shape_gated_had_quant_fp8 },

{ "quant_cols_fp8_bf16", "quant_cols_fp8", "quant",
  "E4M3 codes with one scale per column over every row: an attention's V for its fp8 P.V",
  "two passes over the rows split into 32 chunks: the column amax partials, then the scales and "
  "codes; a thread owns eight columns; the software e4m3 codec (RDNA3 has no fp8 builtin)",
  "bf16 in, E4M3 codes, f32 scale per column; kColChunks * n floats of scratch",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cQuantCols), NOTUNE,
  nullptr, nullptr, r3d_rad_quant_cols_fp8, r3d_quant_cols_fp8_scratch, nullptr, nullptr,
  r3d_shape_quant_cols_fp8 },

{ "had_quant_act_i8g_bf16", "had_quant_act_i8g", "quant",
  "a 128-wide block Hadamard, then per-group int8: the rotated form for the int8 trunk GEMMs",
  "n a multiple of 128, group 128 (one wave a (token, 128-group): the FWHT's DPP stages stay out "
  "of a loop); the codes row-major for the i8a8 dot2 GEMM, the scale amax/127 (the rotation's "
  "1/128 is the weight plane's)",
  "bf16 in, int8 codes (row-major), f32 scale per group",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cHadQuantI8g), NOTUNE,
  nullptr, nullptr, r3d_rad_had_quant_act_i8g, nullptr, nullptr, nullptr,
  r3d_shape_had_quant_act_i8g },

/* IQ8R's group 256: two independent 128-wide rotations, one amax and scale over the pair. */
{ "had_quant_act_i8g_256_bf16", "had_quant_act_i8g", "quant",
  "had_quant_act_i8g at IQ8R's group: each 128-wide half of a 256-column group rotated on its "
  "own, one int8 scale for the pair",
  "n a multiple of 256, group 256 (one wave a (token, 256-group)); the codes row-major for the "
  "i8a8 dot2 GEMM, the scale amax/127 over the pair (the rotation's 1/128 is the weight plane's)",
  "bf16 in, int8 codes (row-major), f32 scale per (row, 256 columns)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cHadQuantI8g256), NOTUNE,
  nullptr, nullptr, r3d_rad_had_quant_act_i8g, nullptr, nullptr, nullptr,
  r3d_shape_had_quant_act_i8g },

{ "gated_had_quant_i8g_bf16", "gated_had_quant_i8g", "quant",
  "a gated product (silu or sigmoid), a 128-wide block Hadamard, then per-group int8",
  "n a multiple of 128, group 128 (one wave a (token, 128-group): the FWHT's DPP stages stay out "
  "of a loop); the gated product rounded to bf16, the codes row-major for the i8a8 dot2 GEMM, the "
  "scale amax/127",
  "bf16 gate_up [M,2n], int8 codes (row-major), f32 scale per group, optional bf16 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGatedHadQuantI8g), NOTUNE,
  nullptr, nullptr, r3d_rad_gated_had_quant_i8g, nullptr, nullptr, nullptr,
  r3d_shape_gated_had_quant_i8g },

/* IQ8R's group 256: two independent 128-wide rotations, one amax and scale over the pair. */
{ "gated_had_quant_i8g_256_bf16", "gated_had_quant_i8g", "quant",
  "gated_had_quant_i8g at IQ8R's group: one int8 scale per 256 columns, each 128-wide half "
  "rotated on its own",
  "n a multiple of 256, group 256 (one wave a (token, 256-group)); the gated product rounded to "
  "bf16, the codes row-major for the i8a8 dot2 GEMM, the scale amax/127 over the pair",
  "bf16 gate_up [M,2n], int8 codes (row-major), f32 scale per (row, 256 columns), optional bf16 "
  "product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGatedHadQuantI8g256), NOTUNE,
  nullptr, nullptr, r3d_rad_gated_had_quant_i8g, nullptr, nullptr, nullptr,
  r3d_shape_gated_had_quant_i8g },

{ "gate_had_quant_i8g_bf16", "gate_had_quant_i8g", "quant",
  "act(gate) * x with the gate in its own operand, a 128-wide block Hadamard, then per-group int8",
  "n a multiple of 128, group 128 (one wave a (token, 128-group)); the product rounded to bf16, the "
  "codes row-major for the i8a8 dot2 GEMM, the scale amax/127",
  "bf16 gate [M,n] and x [M,n], int8 codes (row-major), f32 scale per group, optional bf16 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGateHadQuantI8g), NOTUNE,
  nullptr, nullptr, r3d_rad_gate_had_quant_i8g, nullptr, nullptr, nullptr,
  r3d_shape_gate_had_quant_i8g },

/* IQ8R's group 256: two independent 128-wide rotations, one amax and scale over the pair. */
{ "gate_had_quant_i8g_256_bf16", "gate_had_quant_i8g", "quant",
  "gate_had_quant_i8g at IQ8R's group: one int8 scale per 256 columns of a (token, head) row",
  "n a multiple of 256, group 256 (one wave a (token, 256-group)); the product rounded to bf16, the "
  "codes row-major for the i8a8 dot2 GEMM, the scale amax/127 over the pair",
  "bf16 gate [M,n] and x [M,n], int8 codes (row-major), f32 scale per (row, 256 columns), optional "
  "bf16 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGateHadQuantI8g256), NOTUNE,
  nullptr, nullptr, r3d_rad_gate_had_quant_i8g, nullptr, nullptr, nullptr,
  r3d_shape_gate_had_quant_i8g },

{ "rmsnorm_quant_i8_bf16", "rmsnorm_quant_i8", "quant",
  "the RMS norm and the per-row int8 quantiser in one pass, optionally with the residual junction",
  "post-norm, residual add, scalar, then the norm and the int8 codes in the w4a8 fragment order; "
  "every intermediate rounded to bf16 so the chain is byte-identical; one workgroup a row",
  "bf16 x and optional residual (inout), weight, i8 codes, f32 scale per row, optional bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRmsQuantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_rmsnorm_quant_i8, nullptr, nullptr, nullptr,
  r3d_shape_rmsnorm_quant_i8 },

{ "gated_quant_fp8_bf16", "gated_quant_fp8", "quant",
  "silu(gate) * up over a fused [M, 2n] projection, quantised in the same pass",
  "group 128; n a multiple of 4; the product is rounded to bf16 before its amax, as every fused "
  "implementation rounds it",
  "bf16 gate_up in, fp8_e4m3 codes, f32 scales, optional bf16 product",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGateQuant), NOTUNE,
  nullptr, nullptr, r3d_rad_gated_quant_fp8, nullptr, nullptr, nullptr,
  r3d_shape_gated_quant_fp8 },

{ "dequant_fp8", "dequant", "quant",
  "E4M3 code times its group's f32 scale",
  "group 128; n a multiple of 4",
  "fp8_e4m3 codes and f32 scales in, bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cDequantFp8), NOTUNE,
  nullptr, nullptr, r3d_rad_dequant, nullptr, nullptr, nullptr, r3d_shape_dequant },

{ "dequant_e5m2", "dequant", "quant",
  "E5M2 code times its group's f32 scale",
  "group 128; n a multiple of 4",
  "fp8_e5m2 codes and f32 scales in, bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cDequantE5), NOTUNE,
  nullptr, nullptr, r3d_rad_dequant, nullptr, nullptr, nullptr, r3d_shape_dequant },

{ "dequant_i8", "dequant", "quant",
  "int8 code times its group's f32 scale",
  "group 128; n a multiple of 4",
  "i8 codes and f32 scales in, bf16 out",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cDequantI8), NOTUNE,
  nullptr, nullptr, r3d_rad_dequant, nullptr, nullptr, nullptr, r3d_shape_dequant },

/* ------------------------------------------------------------------ r3d_gemm_bf16_nt_m16.hip
 *
 * The decode GEMM: M is 1 to 16, so a wave owns a run of N columns and streams the weight's rows
 * past an activation kept in registers. There is no fragment layout, so the kernel reads the
 * checkpoint's rows in order.
 */
{ "gemm_nt_bf16_m16", "gemm_nt", "gemm",
  "a[M,K] @ b[N,K]^T for the decode and verify heights, f32 accumulate, one narrowing at the store",
  "M <= 16; a wave a run of N columns, one output element a lane, k ascending in pairs; split-K "
  "over scratch with the partials reduced in split order; a ragged K runs the scalar tail at sk=1",
  "bf16 a, b and y, fp32 accumulation; the pair is folded into the accumulator by one rounding, so "
  "about 0.3% of a long-K output lands one bf16 store ulp from the reference's, and no output is "
  "further. bf16 reaches the dot as f16, exact over f16's normal range, which every checkpoint's "
  "weights and activations live inside",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_M16,
  ROW(cGemmM16), tGemmM16, NELEM(tGemmM16), gemm_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_bf16_m16,
  [](const RadArgs* a) -> int64_t { return gemm_scratch(a->p, a->n_p); },
  nullptr, nullptr, r3d_shape_gemm_nt },

/* ------------------------------------------------------------------ r3d_gemm_nt_n1.hip
 *
 * The shared expert's gate: gemm_nt at N = 1, where the m16 row's wave-per-run-of-columns map has
 * one column and the tiled row's M floor is out of band. One lane an output row, the dot walked
 * ascending in 16-element steps with no cross-lane reduction or split, so the result is libref's
 * chain bit for bit.
 */
{ "gemm_nt_bf16_n1", "gemm_nt", "gemm",
  "y[m] = a[m,K] . b[0,K] for N = 1, f32 accumulate in the reference's ascending-k chain, one "
  "narrowing at the store",
  "N == 1; any M; one lane an output row, K read in 16-element steps; K a multiple of 128, a and "
  "b 16-byte aligned, the activation row pitch a multiple of 8 elements",
  "bf16 a, b and y; every product and its add round in f32 and neither is contracted "
  "(-ffp-contract=off), so y is bit-identical to libref's gemm_nt at N = 1",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_N1,
  ROW(cGemmN1), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_bf16_n1, nullptr, nullptr, nullptr, r3d_shape_gemm_nt },

/* ------------------------------------------------------------------ r3d_gemm_f16_nt_m16.hip
 *
 * The same shape over an f16 checkpoint: the bytes are the dot's native pair, so the dot2 arm
 * stages a straight copy and the weight streams into the fold with no widening between. The
 * wave-matrix arm rides the ISA's 16x16x16 f16 tile, the `impl` point, the way the bf16 row's
 * does on its tile.
 */
{ "gemm_nt_f16_m16", "gemm_nt", "gemm",
  "a[M,K] @ b[N,K]^T for the decode and verify heights, f32 accumulate, one narrowing at the store",
  "M <= 16; a wave a run of N columns, one output element a lane, k ascending in pairs; split-K "
  "over scratch with the partials reduced in split order; a ragged K runs the scalar tail at sk=1",
  "f16 a, b and y, fp32 accumulation; the pair is folded into the accumulator by one rounding, so "
  "about 0.3% of a long-K output lands one f16 store ulp from the reference's, and no output is "
  "further. f16 is the pair the dot takes, so there is no widening to price and the row's range "
  "is f16's: a checkpoint whose values sit outside it is not a checkpoint this row serves",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_M16,
  ROW(cGemmM16F16), tGemmM16, NELEM(tGemmM16), gemm_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_f16_m16,
  [](const RadArgs* a) -> int64_t { return gemm_scratch(a->p, a->n_p); },
  nullptr, nullptr, r3d_shape_gemm_nt },

/* ------------------------------------------------------------------ r3d_gemm_nt_f32.hip
 *
 * The router's F32 projection: a bf16 activation against an f32 weight, f32 out, no narrowing.
 * One wave R rows by C columns, lane l folding k = l, l+32, ... with fmaf and the xor butterfly;
 * the tile form its rule picks for M, every form byte-identical.
 */
{ "gemm_nt_f32_bf16", "gemm_nt_f32", "gemm",
  "F32 routing and index gate projection",
  "N<=512, K<=16384; pitched rows",
  "bf16 input, f32 weights and output",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGemmNtF32), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_f32, nullptr, nullptr, nullptr,
  r3d_shape_gemm_nt_f32 },

/* ------------------------------------------------------------------ r3d_gemm_w4a16_nt_m64.hip
 *
 * The 4-bit weight against an f16 activation, at the verify and small-prefill heights: the
 * weight's codes run the 16x16x16 f16 tile through the 1024-offset dequant, and the group
 * scale lands on the f32 accumulator after the zero point's cancellation. The weight arrives
 * in the hook's fragment order, permuted at load.
 */
{ "gemm_nt_q_w4a16asym_m64", "gemm_nt_q", "gemm",
  "a[M,K] @ dequant(wq)[N,K]^T: unsigned 4-bit codes with an f16 scale and an integer zero per "
  "128 of K, against an f16 activation, f32 accumulate, one narrowing at the store",
  "1 <= M <= 64, N a multiple of 16, K a multiple of the group and of the split's slab; the "
  "weight read in the container's stored order, no permutation; split-K slabs reduced by a "
  "second kernel over a grow-only pool",
  "f16 a, U4 wq with the f16 scale and the integer zero per (row, 128 of K), bf16 y, fp32 "
  "accumulation; the dequant is the integer code minus the zero, exact, and the store is the "
  "only narrowing",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmM64W4A16A), tGemmM64, NELEM(tGemmM64), gemm_tune_valid_m64,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w4a16_m64,
  nullptr, r3d_layout_gemm_nt_q_w4a16, r3d_relayout_gemm_nt_q_w4a16,
  nullptr, nullptr },

/* ------------------------------------------------------------------ r3d_gemm_w5a16_nt_m64.hip
 *
 * The 5-bit weight against an f16 activation: the 4-bit row with a fifth bit per code,
 * ParoQuant's int5. The fifth bits ride a plane of their own, one word a (tile, 64 of K,
 * lane), joined behind the (scale, zero) pair.
 */
{ "gemm_nt_q_w5a16_m64", "gemm_nt_q", "gemm",
  "a[M,K] @ dequant(wq, hi)[N,K]^T: 5-bit codes with an f16 scale and an integer zero per "
  "128 of K, against an f16 activation, f32 accumulate, one narrowing at the store",
  "1 <= M <= 64, N a multiple of 16, K a multiple of the group and of the split's slab; the "
  "weight read in the container's stored order, the fifth bits a plane of their own, no "
  "permutation; split-K slabs reduced by a second kernel over a grow-only pool",
  "f16 a, U4 wq with the f16 scale, the integer zero per (row, 128 of K) and the fifth-bit "
  "plane per (tile, 64 of K, lane), bf16 y, fp32 accumulation; the dequant is the integer "
  "code minus the zero, exact, and the store is the only narrowing",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmM64W5A16), tGemmM64, NELEM(tGemmM64), gemm_tune_valid_m64,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w5a16_m64,
  nullptr, r3d_layout_gemm_nt_q_w5a16, r3d_relayout_gemm_nt_q_w5a16,
  nullptr, nullptr },

/* ------------------------------------------------------------------ r3d_gdn_kkt_solve_c64.hip
 *
 * The gated-delta-net chunk preamble: the gram and the blocked inverse in one kernel, the
 * FLA pair's 64x64 fp32 intermediate kept in LDS.
 */
{ "gdn_kkt_solve_c64", "gdn_kkt_solve", "gdn",
  "A = (I + strict_lower(diag(beta) K K^T e^{g_i - g_j}))^-1 per 64-chunk of a gated delta net",
  "head_k 128, chunk 64, varlen; one gram per k head, applied to every v head of the group, the "
  "inverse blocked 16 -> 32 -> 64 on the wave matrix (libhalo's kernel, r3dh_gdn_kkt_solve.hip)",
  "bf16 k and A, f32 beta, g, gram and inverse, i32 cu",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnKktC64), NOTUNE,
  nullptr, nullptr, r3dh_rad_gdn_kkt_solve,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

/* ------------------------------------------------------------------ r3d_gdn_chunk_scan_k128_v128_c64.hip
 *
 * The gated-delta-net chunked scan: the chunk's output and its closing state, the recurrent
 * state in wave-matrix accumulators across the chunk loop.
 */
{ "gdn_chunk_scan_k128_v128_c64", "gdn_chunk_scan", "gdn",
  "O = scale*ge*(Q@S) + sA@V', S = e^{g_last}S + V'^T@(gs*K) per 64-chunk of a gated delta net, "
  "V' = A@W, W = beta*(V - ge*U), U = K@S, sA = the kkt_solve row's output with the "
  "scale*e^{g_i-g_j} score decay",
  "head_k 128, head_v 128, chunk 64, H a multiple of head groups; the state stays in "
  "accumulators across the chunk loop, so only q, k, v, A, g, beta, the state cache and o "
  "touch HBM; two workgroups per v head each carry 64 of the v dimension",
  "bf16 q, k, v, A and o, f32 g and beta, f32 or f16 state cache (the operand's dtype), i32 cu "
  "and the optional state slot index (libhalo's kernel, r3dh_gdn_chunk_scan.hip)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnChunkScanC64), NOTUNE,
  nullptr, nullptr, r3dh_rad_gdn_chunk_scan,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

 /* ------------------------------------------------------------------ r3d_gdn_gated_rmsnorm_h128.hip
  *
  * The gated RMS norm a gated-delta-net layer applies to its own output before the output
  * projection: one wave per (token, head) row, a barrier-free wave sum and one 8-byte access per
  * lane for each load and the store.
  */
{ "gdn_gated_rmsnorm_h128", "gdn_gated_rmsnorm", "gdn",
  "x * rsqrt(mean(x^2) + eps) * w * act(z), one (token, head) row of 128 channels",
  "channels 128; act is silu (the default) or sigmoid; the gate may be a flat projection row or a "
  "column slice of a fused [q|k|v|z] row",
  "bf16 x, z and o, f32 w; one wave per row, four channels per lane",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnGatedRmsnorm), NOTUNE,
  nullptr, nullptr, r3d_rad_gdn_gated_rmsnorm,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

 /* ------------------------------------------------------------------ r3d_gdn_gated_norm_had_quant_i8.hip
  *
  * The GDN block's gated RMS norm, block Hadamard and per-row int8 quantiser in one pass: the
  * out_proj input on the linear-attention layers. One workgroup per row; the rotation is a
  * wave-local FWHT in registers, the int8 scale a whole-row max.
  */
{ "gdn_gated_norm_had_quant_i8", "gdn_gated_norm_had_quant_i8", "gdn",
  "x * rsqrt(mean_head(x^2) + eps) * w * silu(z), Hadamard-rotated over `group`, int8-quantised",
  "head_dim 128, `group` the rotation width (128, 256 or 512); the RMS is per 128-head, the "
  "rotation and the int8 scale per token over the flattened row",
  "bf16 x and z, f32 w, i8 q, f32 scale; the gate may be a column slice of a fused [q|k|v|z] row "
  "(`zgrp`, `zblk`)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnGatedNormHq), NOTUNE,
  nullptr, nullptr, r3d_rad_gdn_gated_norm_hq_i8,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

 /* ------------------------------------------------------------------ r3d_gdn_conv_w4_h128.hip
  *
  * The GDN depthwise causal convolution (width 4, silu) and the work fused around it: the prefill
  * prep (conv + split + l2norm + gate + cumsum) and the decode update (a speculative window over a
  * rolling conv-state buffer).
  */
{ "gdn_conv_prep_w4_h128", "gdn_conv_prep", "gdn",
  "The GDN depthwise causal convolution (width 4, silu) plus split, l2norm, gate and per-chunk "
  "cumsum in one prefill kernel",
  "head_k 128, head_v 128, conv_width 4, chunk 64; the conv state is read at a sequence's first "
  "chunk and rewritten from its last tokens",
  "bf16 x, w, conv_state and q/k/v, f32 A_log, dt_bias, g and beta, a and b_gate bf16 or f32; the "
  "conv output is rounded to bf16 before the norm and the store",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnConvW4), NOTUNE,
  nullptr, nullptr, r3d_rad_gdn_conv_prep,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

{ "gdn_conv_update_w4_h128", "gdn_conv_update", "gdn",
  "The GDN depthwise causal convolution at decode: a speculative window over a rolling conv-state "
  "buffer",
  "head_k 128, head_v 128, conv_width 4; the state buffer is read at the slot the last accepted "
  "token left and rewritten shifted by one",
  "bf16 x, w, conv_state and q/k/v; `state_idx` the rolling buffer's slot, `num_accepted` the "
  "window's read offset",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnConvUpdW4), NOTUNE,
  nullptr, nullptr, r3d_rad_gdn_conv_update,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

{ "ple_conv_bf16", "ple_conv", "norm",
  "PLE's dilated depthwise conv, silu and residual add; one thread a channel",
  "width up to 5 and dilation up to 4 -- (width-1)*dilation is a register window",
  "bf16 activations, weights and conv state; i32 cu, slots, has_init and num_accepted",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cPleConv), NOTUNE,
  nullptr, nullptr, r3d_rad_ple_conv, nullptr, nullptr, nullptr, nullptr },

{ "expert_scale_bf16", "expert_scale", "moe",
  "Each routing weight times its expert's own per-expert scale, in place",
  "the router's renormalised top-k weight scaled by the [n_expert] table, an out-of-table id "
  "reading zero",
  "bf16 ids (i32), w_e and ew; one thread per element",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cExpertScaleBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_expert_scale_bf16,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

{ "moe_router_topk_bf16", "router_topk", "moe",
  "Softmax over the expert logits, the top_k by value, renormalised if `norm`",
  "one wave per row; top_k passes of a wave minimum (ties to the lower id)",
  "bf16 logits and expert_w; i32 expert_ids; n_expert <= 512, top_k <= 32",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cMoeRouterTopkBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_moe_router_topk_bf16,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

 { "moe_router_topk_scatter", "router_topk_scatter", "moe",
  "The router top-k and the counting sort as one op: the selection over the whole n_expert logits, "
  "then the sort over this rank's n_local slice",
  "phase 1 the top-k (one wave per row), phase 2 the small counting sort (one workgroup); the "
  "selection's expert_w is the sort's input",
  "bf16 logits [M,ne] and expert_w [M,k]; i32 expert_ids [M,k], sorted_tok [M*k], expert_offset "
  "[nl+1], expert_count [nl]; n_expert <= 512, n_local <= n_expert, T <= 1024",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRouterTopkScatter), NOTUNE,
  nullptr, nullptr, r3d_rad_router_topk_scatter,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

 { "moe_router_gemm_topk_scatter", "router_gemm_topk_scatter", "moe",
  "The router GEMM, its top-k and the counting sort as one device op: gemm_nt of a [M,K] against "
  "the router b [n_expert,K], then router_topk_scatter over those logits, and the shared expert's "
  "one-column gate beside them when gate_b/gate are present",
  "the device launches the pair would run, in order -- gemm_nt_bf16_m16 at M <= 16 and "
  "gemm_nt_bf16_tiled above (the same split and impl the unfused rows pick), the gate in 16-row "
  "m16 chunks at N = 1, then moe_router_topk_scatter; not a single-launch fusion",
  "bf16 a [M,K], router b [ne,K], optional gate_b [1,K]; bf16 logits [M,ne], expert_w [M,k] and "
  "optional gate [M,1]; i32 expert_ids [M,k], sorted_tok [M*k], expert_offset [nl+1], "
  "expert_count [nl]; M <= 64, K%32 == 0, n_expert <= 512, top_k <= 32, M*top_k <= 1024",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRouterGemmTopkScatter), NOTUNE,
  nullptr, nullptr, r3d_rad_router_gemm_topk_scatter,
  router_fused_scratch, nullptr, nullptr,
  nullptr, nullptr },

 { "moe_router_sigmoid_topk", "router_sigmoid_topk", "moe",
  "The GLM-style router: scores = sigmoid(logits), the selection on scores + bias, the kept "
  "weights the scores scaled by route_scale (renormalised if norm)",
  "one wave per row; a 64-bit selection key (the select monotone bits, the complemented id), the "
  "scores staged in LDS; ties to the lower id",
  "bf16 logits [M,ne], f32 bias [ne]; i32 expert_ids [M,k], bf16 expert_w [M,k]; n_expert <= 512, "
  "top_k <= 32",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRouterSigmoid), NOTUNE,
  nullptr, nullptr, r3d_rad_router_sigmoid_topk,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },
{ "moe_router_sigmoid_group_topk_f32", "router_sigmoid_group_topk", "moe",
  "group-limited sigmoid correction-bias routing, one wave a row",
  "experts <=512, groups <=32, top_k<=32",
  "f32 logits, bias and weights; i32 ids",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRouterSigmoidGroup), NOTUNE,
  nullptr, nullptr, r3d_rad_router_sigmoid_group_topk, nullptr, nullptr, nullptr,
  r3d_shape_router_sigmoid_group_topk },

{ "moe_scatter_small", "moe_scatter", "moe",
  "The routed slots' stable counting sort",
  "up to 1024 slots one workgroup (histogram in LDS, an exclusive prefix over the expert counts, "
  "each slot placed by its rank inside its expert's run); above, three passes over 256-slot tiles "
  "(histogram, scan across tiles, placement) with the per-tile counts in scratch",
  "i32 expert_ids, sorted_tok, expert_offset, expert_count; n_expert <= 512",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cMoeScatter), NOTUNE,
  nullptr, nullptr, r3d_rad_moe_scatter,
  r3d_rad_moe_scatter_scratch, nullptr, nullptr,
  nullptr, nullptr },

  /* ------------------------------------------------------------------ r3d_moe_gather.hip */
 { "moe_gather", "moe_gather", "moe",
  "Fold the expert outputs back to the token: y[t,c] = sum_j expert_w[t,j] * "
  "y_expert[slot(t,j),c]",
  "one block per token, the slot table (the inverse of sorted_tok, scanned in LDS) and the expert "
  "weights hoisted so the per-column loads pipeline; the optional shared arm adds shared*act(gate)",
  "bf16 y_expert [Te,n], expert_w [Mw,top_k] (bf16 or f32), sorted_tok [T], optional shared/gate, "
  "bf16 y [M,n]; top_k <= 64",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cMoeGather), NOTUNE,
  nullptr, nullptr, r3d_rad_moe_gather,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

 /* ------------------------------------------------------------------ r3d_moe_gemm_q.hip
  *
  * The grouped quantised GEMM: y[i] = a[sorted_tok[i]/top_k] @ w_e^T with a per-(row, grid-of-K)
  * scale fold, e the expert of sorted row i. E4M3 operands staged compact and upcast to f16 in
  * registers (the 256-entry decode table), the f16 wave-matrix, the even experts in `w` and the
  * odd in `w_odd`.
  */
 { "moe_gemm_q_e4m3", "moe_gemm_q", "moe",
  "The grouped quantised GEMM over the model's fp8a8f32 experts: y[i] = a[sorted_tok[i]/top_k] @ "
  "w_e^T with a per-(row, grid-of-K) scale fold, e the expert of sorted row i",
  "three kernels, one contract, the tune's impl axis picks: 0 = 16x16 tiles through LDS, the E4M3 "
  "operands staged compact (8-bit) and upcast to f16 in registers through the block's 512-byte "
  "decode table; 1 = the same tile, the E4M3 decode in registers (the per-byte ALU formula, no LDS "
  "table); 2 = the 16x128 block, the E4M3 operands widened exactly to f16 and the F32 scales "
  "folded in f32 per 128 of K. All arms run the f16 wave-matrix (f32 accumulate) and walk the runs "
  "in the block's 16 rows, so a block spanning several experts (short runs) is correct; the even "
  "experts are entry e/2 of `w` / `w_scale`, the odd of `w_odd` / `w_odd_scale`",
  "E4M3 a and the weight tables, f32 a_scale per (row, 128 of K), f32 w_scale per 128x128 tile; "
  "K % 128; equal classes (N_odd = N, K_odd = K); token or sorted a; y bf16 in sorted order",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cMoeGemmQ), tMoeGemmQ, NELEM(tMoeGemmQ), moe_q_tune_valid,
  nullptr, nullptr, r3d_rad_moe_gemm_q,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },


 /* ------------------------------------------------------------------ r3d_moe_gemm_bf16.hip */
 { "moe_gemm_bf16", "moe_gemm", "moe",
   "The grouped GEMM with unquantised experts: y[i] = a[sorted_tok[i]/top_k] @ w_e^T, e the expert "
   "of sorted row i",
   "16x16 tiles through LDS, the bf16 operands staged and loaded straight into the bf16 wave-matrix "
   "fragments (f32 accumulate over the whole of K); the block walks the runs in its 16 rows, so a "
   "block spanning several experts (short runs) is correct; the weight is a table of the "
   "checkpoint's own [N, K] row-major planes (no layout hook)",
   "bf16 a and w, bf16 y in sorted order; token or sorted a; K a multiple of 64, the N tail "
   "clamped",
   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
   ROW(cMoeGemmBf16), tMoeGemmBf16, NELEM(tMoeGemmBf16), moe_bf16_tune_valid,
   nullptr, nullptr, r3d_rad_moe_gemm_bf16,
   nullptr, nullptr, nullptr,
   nullptr, nullptr },

 /* ------------------------------------------------------------------ r3d_moe_gemm_mxfp4a8.hip */
 { "moe_gemm_mxfp4a8", "moe_gemm_q", "moe",
   "The grouped GEMM with canonical MXFP4 experts: y[i] = a[sorted_tok[i]/top_k] @ w_e^T, e the "
   "expert of sorted row i",
   "16x16 tiles through LDS; the E4M3 activation and the E2M1 weight (decoded exactly to E4M3) "
   "upcast to f16 in registers, the f16 wave-matrix (f32 accumulate); the E8M0 weight scales fold "
   "in f32 per 32 of K; the block walks the runs in its 16 rows; the even experts are entry e/2 "
   "of `w` / `w_scale`, the odd of `w_odd` / `w_odd_scale`",
   "E4M3 a with f32 a_scale per (row, 128 of K); E2M1 (packed, two a byte) weight tables with E8M0 "
   "w_scale per (row, 32 of K); K % 128; equal classes; token or sorted a; y bf16 in sorted order",
   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
   ROW(cMoeGemmMxfp4), NOTUNE,
   nullptr, nullptr, r3d_rad_moe_gemm_mxfp4a8,
   nullptr, nullptr, nullptr,
   nullptr, nullptr },

  /* ------------------------------------------------------------------ r3d_moe_iqr.hip
   *
   * The grouped IQR GEMM: each expert's weight is a computed-trellis IQR plane at its codes' width,
   * decoded where it is read. A block owns 16 sorted rows and four 16-column weight tiles; the
   * decode band walks the runs in its tile, above it a run map in the row's scratch gives a block
   * one run of up to 64 rows so a decoded weight tile serves four token tiles. The arithmetic is
   * the dense IQR row's: an exact i32 group sub-sum, then (sub * a_scale) * fold in f32.
   */
  { "moe_gemm_iqr", "moe_gemm_q", "moe",
    "The grouped GEMM over experts stored as IQR weights: y[i] = a[sorted_tok[i]/top_k] @ w_e^T, "
    "e the expert of sorted row i, each 128-K group's exact int8 sub-sum folded by the activation's "
    "scale and the weight row's block factor",
    "16 sorted rows and four 16-column weight tiles a block, a wave a tile; the decode band (M * "
    "top_k <= 1.5 * n_expert) walks the expert runs in its tile; above it a run map in the row's "
    "scratch gives a block one run of up to 64 rows and a decoded 16x128 unit is reused over four "
    "token tiles, the dense resident tile's residency; a 128-K scale block's factor and a row's own "
    "f32 scale are folded in the dense row's order; equal classes (N_odd = N, K_odd = K), N % 64, "
    "K % 128, the trellis run divides N, parts divides N in 16-row tiles",
    "I8 a with an f32 scale per (row, 128 or 256 of K), IQR weight tables (computed trellis, L = 16, "
    "one f32 scale a row, no tcq_rb, 2..7 code bits) and their scale tables, bf16 y in sorted order; "
    "token or sorted a; zero-row experts and dropped slots handled",
    RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
    ROW(cMoeGemmIqr), NOTUNE,
    nullptr, nullptr, r3d_rad_moe_gemm_iqr,
    r3d_rad_moe_gemm_iqr_scratch, r3d_rad_layout_moe_iqr, r3d_rad_relayout_moe_iqr,
    nullptr, r3d_rad_unrelayout_moe_iqr,
    nullptr, 0 },

  /* ------------------------------------------------------------------ r3d_moe_w4nl.hip
   *
   * The grouped w4nl codebook GEMM: the routed-row walk of the IQR row over experts whose weight is
   * u4 codes indexing a sixteen-entry f32 table, with an E4M3 scale per (row, 64 of K) under one
   * fixed f32 level. A row of moe_gemm_q selected by dtype=w4nl64a8i8h; the even experts in `w` /
   * `w_scale`, the odd in `w_odd` / `w_odd_scale`. The layout hook stores the codes and table as one
   * joined operand and the two scale planes as another.
   */
  { "moe_gemm_w4nl", "moe_gemm_q", "moe",
    "The grouped GEMM over experts stored in the w4nl codebook form: y[i] = a[sorted_tok[i]/top_k] "
    "@ w_e^T, e the expert of sorted row i, each 64-K group's exact int8*table sub-sum folded by "
    "the activation's scale and the weight's E4M3 scale times its fixed f32 level",
    "16 sorted rows and four 16-column weight tiles a block, a wave a tile; the decode band (M * "
    "top_k <= 1.5 * n_expert) walks the expert runs in its tile; above it a run map in the row's "
    "scratch gives a block one run of up to 64 rows and a decoded 16x128 unit is reused over four "
    "token tiles; the group fold is libref's (sub * a_scale) * (w_scale * level / 8), groups "
    "ascending, so the int8 result is bit-exact; equal classes (N_odd = N, K_odd = K), N % 64, "
    "K % 128, `group` 128, `a_group` 128 or 256, parts divides N in 64-column blocks",
    "I8 a with an f32 scale per (row, 128 or 256 of K); u4 weight tables (two codes a byte) joined "
    "with the f32[16] table, and E4M3 scale tables joined with the fixed f32 2^-13; bf16 y in "
    "sorted order; token or sorted a; zero-row experts and dropped slots handled; bf16 protected "
    "experts are the bf16 row's",
    RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
    ROW(cMoeGemmQW4NL), NOTUNE,
    nullptr, nullptr, r3d_rad_moe_gemm_w4nl,
    r3d_rad_moe_gemm_w4nl_scratch, r3d_rad_layout_moe_w4nl, r3d_rad_relayout_moe_w4nl,
    nullptr, nullptr,
    nullptr, 0 },

  /* ------------------------------------------------------------------ r3d_moe_gemm_w4a8.hip
   *
   * The grouped int4 GEMM: the two's-complement int4 weight (nibbles, -8..7) against an int8 or
   * E4M3 activation, one 128-block of K per scale fold. The int8 arm runs the exact i32 wave-matrix
   * (no f16 rounding); the E4M3 arm the f16 wave-matrix (libr4d's form). A row of moe_gemm_q,
   * selected by dtype=w4a8; the even experts in `w` / `w_scale`, the odd in `w_odd` / `w_odd_scale`.
   */
  { "moe_gemm_w4a8", "moe_gemm_q", "moe",
   "The grouped int4 GEMM over the expert-sorted rows: y[i] = a[sorted_tok[i]/top_k] @ w_e^T, e the "
   "expert of sorted row i; the two's-complement int4 weight (nibbles, -8..7) against an int8 or "
   "E4M3 activation, one 128-block of K per scale fold",
   "16 sorted rows per block, eight waves of 16 columns; the block walks its expert runs, the "
   "padding rows reading the run's first; the decode band (M * top_k <= 16 the tuned point, more "
   "runs as extra 16-row blocks, prefill falls back to libavx); K % 128, N % 64; the even experts "
   "are entry e/2 of `w` / `w_scale`, the odd of `w_odd` / `w_odd_scale`; token or sorted a",
   "I4 (packed, two a byte) weight tables with a bf16 w_scale per (row, 128 of K); int8 a (an f32 "
   "a_scale per (row, 128 of K)) or E4M3 a (an f32 a_scale); bf16 y in sorted order; the int8 arm is "
   "bit-exact against the reference, the E4M3 arm to the f16 store ulp",
   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
   ROW(cMoeGemmQW4A8), tMoeGemmQW4, NELEM(tMoeGemmQW4), nullptr,
   nullptr, nullptr, r3d_rad_moe_gemm_w4a8,
   nullptr, nullptr, nullptr,
   nullptr, nullptr },
   /* ------------------------------------------------------------------ r3d_moe_gemm_w4a4.hip
   *
   * The grouped int4xint4 GEMM: the two's-complement int4 weight AND int4 activation, one 128-block
   * of K per scale fold. Both are widened to int8 and run on the exact i32 wave-matrix (no f16
   * rounding). A row of moe_gemm_q, selected by dtype=w4a4; the even experts in `w` / `w_scale`, the
   * odd in `w_odd` / `w_odd_scale`.
   */
   { "moe_gemm_w4a4", "moe_gemm_q", "moe",
    "The grouped int4xint4 GEMM over the expert-sorted rows: y[i] = a[sorted_tok[i]/top_k] @ w_e^T, e "
    "the expert of sorted row i; the two's-complement int4 weight and int4 activation, one 128-block "
    "of K per scale fold",
    "16 sorted rows per block, eight waves of 16 columns; the block walks its expert runs, the "
    "padding rows reading the run's first; the decode band (M * top_k <= 16 the tuned point, more "
    "runs as extra 16-row blocks, prefill falls back to libavx); K % 128, N % 64; the even experts "
    "are entry e/2 of `w` / `w_scale`, the odd of `w_odd` / `w_odd_scale`; token or sorted a",
    "I4 (packed, two a byte) weight tables with a bf16 w_scale per (row, 128 of K); int4 a (packed, "
    "two a byte, an f32 a_scale per (row, 128 of K)); bf16 y in sorted order; bit-exact against the "
    "reference",
    RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
    ROW(cMoeGemmW4A4), tMoeGemmQW4, NELEM(tMoeGemmQW4), nullptr,
    nullptr, nullptr, r3d_rad_moe_gemm_w4a4,
    nullptr, nullptr, nullptr,
    nullptr, nullptr },

    /* ------------------------------------------------------------------ r3d_dequant_w4_bf16.hip
   *
   * The packed 4-bit weight back to a plain row-major bf16 [N, K], for the prefill path where
   * W4A8 gains nothing.
   */
  { "dequant_w4_bf16", "dequant_w4", "quant",
   "The packed 4-bit weight back to a plain row-major bf16 [N, K]",
   "One thread per packed dword scatters its eight nibbles to row-major bf16; the scale dword holds "
   "the f16 scale low and -(1024+zero) high; N % 16, K % 64, group divides K",
   "I4 (packed, two a byte) wq with an f16 ws scale per (row, group); bf16 y [N, K]; twos 1 = "
   "symmetric two's-complement nibble, 0 = asymmetric unsigned nibble minus a per-group zero",
   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
   ROW(cDequantW4), NOTUNE,
   nullptr, nullptr, r3d_rad_dequant_w4,
   nullptr, nullptr, nullptr,
   nullptr, nullptr, nullptr },

  { "rowtopk_merge", "row_topk_merge", "select",
   "The R largest of the all-gathered per-rank candidate sets, one wave per row",
   "One wave per row over world_size * R pairs; tk_insert into a 32-slot top-R heap, tk_round "
   "extracts via a 5-step xor-butterfly; ties break on the lower id; R <= 32",
   "f32 g (M * world_size * R * 2), i32 idx (M * R), f32 val (M * R); R <= 32",
   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
   ROW(cRowTopkMerge), NOTUNE,
   nullptr, nullptr, r3d_rad_row_topk_merge,
   nullptr, nullptr, nullptr,
   nullptr, nullptr, nullptr },

{ "ngram_ids_i32", "ngram_ids", "vocab",
  "The hashed row ids of the Per-Layer Embedding: mixed(k) = XOR of t_s * mult[s], "
  "ids[h] = mixed % vocab_sizes[h] + offsets[h]",
  "One thread a token, looping its heads; n-gram shifts with EOS walk; 64-bit wrapping "
  "multiply and floor-mod; ngram <= 8, heads divisible by (ngram-1)",
  "i32 tok, i32 state, i32 cu, i64 mult, i64 vsz, i64 off, i32 ids; "
  "optional i32 cidx, hini, nacc, ahead",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cNgramIds), NOTUNE,
  nullptr, nullptr, r3d_rad_ngram_ids,
  nullptr, nullptr, nullptr,
  nullptr, nullptr, nullptr },

  { "gemm_nt_q_bias_fp8a8_m16", "gemm_nt_q_bias", "gemm",
   "The narrow fp8 GEMM with a per-column bias epilogue: y = a @ b^T + bias",
   "M <= 16, K a multiple of the block; fp8a8 codes and scales, bf16 bias and output; "
   "GEMM then bias-add in two passes",
   "fp8_e4m3 a and b, f32 scales, bf16 bias [N], bf16 y [M,N]",
   RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
   ROW(cGemmQBias), tGemmM16, NELEM(tGemmM16), nullptr,
   nullptr, nullptr, r3d_rad_gemm_nt_q_bias_fp8,
   nullptr, nullptr, nullptr,
   nullptr, nullptr, nullptr },

  { "gemm_nt_q_short_fp8a8_m16", "gemm_nt_q_short", "gemm",
   "The narrow fp8 GEMM with the short-prompt split rule: unsplit at N <= 512, K <= 5120, "
   "41..64 rows",
   "M <= 64, K a multiple of 128, N a multiple of 16; fp8a8 codes and scales, bf16 output; "
   "sk=1 forced (no split)",
   "fp8_e4m3 a and b, f32/bf16 scales, bf16 y [M,N]",
   RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
   ROW(cGemmQShort), tGemmM16, NELEM(tGemmM16), nullptr,
   nullptr, nullptr, r3d_rad_gemm_nt_q_short_fp8,
   nullptr, nullptr, nullptr,
   nullptr, nullptr, nullptr },

  { "moe_gemm_q_mod4_e4m3", "moe_gemm_q_mod4", "moe",
   "The grouped quantised GEMM with 4 expert classes: y[i] = a[sorted_tok[i]/top_k] @ w_e^T, "
   "e%4 picks the class",
   "same as moe_gemm_q_e4m3 but 4-class routing; group = 128; the class shapes default to N/K "
   "and an unequal N_1..K_3 is refused; token or sorted a",
   "E4M3 a and weight tables (4 classes), f32 a_scale and w_scale; y bf16 in sorted order",
   RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
   ROW(cMoeGemmQMod4), tMoeGemmQW4, NELEM(tMoeGemmQW4), nullptr,
   nullptr, nullptr, r3d_rad_moe_gemm_q_mod4,
   nullptr, nullptr, nullptr,
   nullptr, nullptr, nullptr },

   /* ------------------------------------------------------------------ r3d_kda_update.hip
 *
 * GLM's per-channel KDA recurrence over a paged f32 state cache: the decay from the softplus-
 * limited gate, the l2-normalised q and k, the delta-rule step and o = S q per token, with the
 * speculative candidate snapshots. The state never touches LDS: a wave owns four value rows and
 * keeps each row's 128 state elements in four registers a lane, so the RDNA3 64 KiB stage is not
 * in play.
 */
{ "kda_update_k128_v128_bf16", "kda_update", "gdn",
  "per-channel KDA recurrence and candidate rollback",
  "heads 1..65535, K=V=128, prefill and verify",
  "bf16 activations, f32 gates and state",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKdaUpdate), NOTUNE,
  nullptr, nullptr, r3d_rad_kda_update, nullptr, nullptr, nullptr, r3d_shape_kda_update },

/* ------------------------------------------------------------------ r3d_gdn_recurrent_update_k128_v128.hip
  *
  * The gated-delta-net recurrent state update at decode: the per-token state step against a paged
  * fp32 state cache, the state in registers (one thread per v row and half its k range), the
  * gated RMS norm folded into the epilogue.
  */
{ "gdn_recurrent_update_k128_v128", "gdn_recurrent_update", "gdn",
  "The gated-delta-net recurrent state update at decode: S *= e^g, u = beta (v - S k), S += u k^T, "
  "o = S q per token",
  "head_k 128, head_v 128, paged state in either form: per_candidate (one state per candidate "
  "token) or anchor (one committed state and a replay scratch slot a sequence); the gated RMS "
  "norm folded in when z and norm_w are present (libhalo's kernel, r3dh_gdn_recurrent_update.hip)",
  "bf16 q, k, v and o, f32 a/b, A_log and dt_bias; the state cache is f32 or f16 and the operand "
  "says which",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnRecurrent), NOTUNE,
  nullptr, nullptr, r3dh_rad_gdn_recurrent_update,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

{ "gdn_conv_recurrent_update_k128_v128", "gdn_conv_recurrent_update", "gdn",
  "The gated-delta-net decode step as one kernel: the rolling-window conv and the recurrent state "
  "update, with q, k and v never materialised",
  "head_k 128, head_v 128, conv width 4; the speculative window (state_len_max - 2) must fit the "
  "register buffer; the gated RMS norm is folded in (z and norm_w are required); the state cache "
  "is f32 or f16 (libhalo's kernel, r3dh_gdn_recurrent_update.hip)",
  "bf16 x and o, f32 a, b, A_log, dt_bias and norm_w, the conv state bf16, the state f32 or f16",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnConvRecurrent), NOTUNE,
  nullptr, nullptr, r3dh_rad_gdn_conv_recurrent_update,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

{ "gdn_conv_recurrent_update_had_k128_v128", "gdn_conv_recurrent_update_had", "gdn",
  "The gated-delta-net decode step and had_quant_act_i8g's codes of its output in one kernel: "
  "the conv fold and recurrent update of gdn_conv_recurrent_update, then the epilogue's int8 "
  "codes (a 128-wide Hadamard before the amax), for an out projection stored rotated (i8a8h)",
  "head_k 128, head_v 128, conv width 4, group 128 (one scale a value head) or 256 (IQ8R: one "
  "scale a pair of value heads); the speculative window (state_len_max - 2) must fit the register "
  "buffer; the gated RMS norm is folded in (z and norm_w are required); the state cache is f32 or "
  "f16. o_q (int8) and o_scale (f32) are required; other groups, an fp8 o_q, and head or conv "
  "widths off the band are refused",
  "bf16 x and o, f32 a, b, A_log, dt_bias and norm_w, the conv state and conv weight bf16, the "
  "state f32 or f16; int8 codes row-major, f32 scale per (token, group). Byte-identical to "
  "gdn_conv_recurrent_update followed by had_quant_act_i8g",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGdnConvRecurrent), NOTUNE,
  nullptr, nullptr, r3dh_rad_gdn_conv_recurrent_update_had,
  nullptr, nullptr, nullptr,
  nullptr, nullptr },

/* ------------------------------------------------------------------ r3d_gemm_fp8a16_nt_m1.hip
 *
 * The same shape against a block-scaled weight. The E4M3 codes are widened to f16 exactly and
 * accumulated by dot2; the block scale is applied once per 128-element slab, after the slab's
 * products, where the reference applies it.
 */
{ "gemm_nt_q_fp8a16_m16", "gemm_nt_q", "gemm",
  "a[M,K] @ b[N,K]^T against E4M3 codes with a scale per 128x128 block",
  "M <= 16, K a multiple of the block, group = pb2 = 128 and pb1 the rows sharing a scale row; a "
  "wave a run of N columns; the codes widened to f16 exactly and the block scale applied after each "
  "slab; a split ends on a block boundary (gemm_tune_valid)",
  "bf16 a, fp8_e4m3 b, f32 b_scale, bf16 y, fp32 accumulation; an f32 a, a bf16 scale plane, a "
  "ragged last block and the zero-point planes a_sum and b_ref are refused, not approximated",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQFp8a16), tGemmQ, NELEM(tGemmQ), gemm_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_fp8,
  [](const RadArgs* a) -> int64_t { return gemm_scratch(a->p, a->n_p); },
  nullptr, nullptr, r3d_shape_gemm_nt_q },

{ "gemm_nt_q_fp8a8_m16", "gemm_nt_q", "gemm",
  "quant_act_fp8's codes and scales against E4M3 codes with a scale per 128x128 block",
  "M <= 16, K a multiple of the block; both axes coded, so both scales apply",
  "fp8_e4m3 a and b, f32 scales on both, bf16 y, fp32 accumulation",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQFp8a8), tGemmQ, NELEM(tGemmQ), gemm_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_fp8,
  [](const RadArgs* a) -> int64_t { return gemm_scratch(a->p, a->n_p); },
  nullptr, nullptr, r3d_shape_gemm_nt_q },

{ "gemm_nt_q_mxfp4a8_m16", "gemm_nt_q", "gemm",
  "fp8 a against OCP-MXFP4 weights: e2m1 codes, two a byte, with an e8m0 scale per 32 of K, the "
  "block's partials scaled and summed in ascending k",
  "M <= 16, K a multiple of 128; one wave a weight row, 64 codes a lane, the tune's impl axis "
  "picks the E4M3 decode (0 = the LDS table, 1 = the per-byte ALU formula), the lane's two 32-blocks "
  "scaled at the block boundary",
  "fp8_e4m3 a with an f32 scale per (row, 128 of K), fp4_e2m1 codes with an e8m0 scale per (row, "
  "32 of K) -- `b_ref`, the same exponent plane, is accepted and ignored, bf16 y, fp32 "
  "accumulation; a reserved e8 (0, 255) is refused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQMxfp4a8M16), tImpl2, NELEM(tImpl2), impl_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_mxfp4a8_m16,
  nullptr, nullptr, nullptr, r3d_shape_gemm_nt_q_mxfp4a8 },

{ "gemm_nt_q_w4a8_m16", "gemm_nt_q", "gemm",
  "the symmetric 4-bit weight grid at decode: signed int4 codes against int8 activations, the "
  "group's integer sub-sum scaled once by the activation's row scale and the weight's group scale",
  "M <= 16, K a multiple of 128; one wave a weight row, 64 codes a lane from two 16-byte loads, "
  "the pair fold at the group boundary, the group terms scaled in the epilogue",
  "I8 a in the step order with an f32 scale per row, I4 codes b (two a byte) with an f16 scale "
  "per (row, 128 of K), bf16 y, fp32 accumulation; the sub-sums are exact integers under 2^24, "
  "and the store is the only narrowing; no zero point, so no a_sum",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQW4A8M16), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w4a8_m16,
  nullptr, nullptr, nullptr, r3d_shape_gemm_nt_q_w4a8 },

{ "gemm_nt_q_w8a8_m16", "gemm_nt_q", "gemm",
  "the symmetric 8-bit weight grid at decode: signed int8 codes against int8 activations, the "
  "group's integer sub-sum scaled once by the activation's row scale and the weight's group scale",
  "M <= 16, K a multiple of 128; one wave a weight row, 64 codes a lane from four 16-byte loads, "
  "the pair fold at the group boundary, the group terms scaled in the epilogue",
  "I8 a in the step order with an f32 scale per row, I8 codes b with an f16 scale per (row, 128 "
  "of K), bf16 y, fp32 accumulation; the sub-sums are exact integers under 2^24, and the store is "
  "the only narrowing; no zero point, so no a_sum; M > 16 is declared out. The tune's impl axis "
  "picks the arm: 0 = the dot2 pair path, 1 = the 16x16x16 wave matrix",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQW8A8M16), tImpl2, NELEM(tImpl2), impl_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w8a8_m16,
  nullptr, nullptr, nullptr, r3d_shape_gemm_nt_q_w8a8 },

{ "gemm_iqr_nt_m64", "gemm_nt_q", "gemm",
  "gemm_nt_q over an IQR weight (r3d_iqr.h) kept at its codes' width, decoded where it is read: a "
  "wave a 16-row tile, each lane its 32-lane segment of a unit; at M = 1 the lanes dot their four "
  "columns and fold the group in LDS, above 1 the tile is staged and each lane owns whole "
  "(token, row) pairs",
  "M <= 16; K a multiple of 128 and N of 16; a trellis run divides N. At M = 1 with N <= 8192 "
  "the GEMV splits K over 8, 4 or 2 whole-group slices through the row's scratch, reduced in "
  "split order; wider N and M > 1 keep the unsplit kernel. W0 reads the computed "
  "trellis (tcq_code=mad, L = 16) under one f32 scale a row or a bf16 scale a (row, 128); the "
  "table and grid forms are refused until the MoE lot",
  "I8 a row-major with an f32 scale per (row, 128 of K), W an IQR form's codes decoding to int8 "
  "values, bf16 y; per group an exact i32 sub-sum and the reference's two f32 factors, the row's "
  "own factor in the epilogue under kRowF32",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_M16,
  ROW(cGemmQIqrM16), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_iqr_m16, r3d_rad_gemm_nt_q_iqr_m16_scratch,
  r3d_rad_layout_iqr, r3d_rad_relayout_iqr,
  r3d_shape_gemm_nt_q, r3d_rad_unrelayout_iqr },

{ "gemm_iqr_tiled", "gemm_nt_q", "gemm",
  "gemm_nt_q over an IQR weight above the decode band, resident: one workgroup owns a 16-row "
  "weight tile and four 16-token slices, decodes the 16x128 unit once per 128-K block into LDS "
  "and reuses it across all four waves, so the weight is read once per 64 tokens instead of once "
  "per 16 (the chunked gemm_iqr_nt_m64 launch it replaces)",
  "M >= 1, K a multiple of 128 and N of 16, a trellis run dividing N; same forms as the decode "
  "band",
  "I8 a row-major with an f32 scale per (row, 128 of K), W an IQR form's codes, bf16 y; per group "
  "an exact i32 sub-sum and the reference's two f32 factors",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_TILED,
  ROW(cGemmQIqrTiled), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_iqr_tiled,
  nullptr, r3d_rad_layout_iqr, r3d_rad_relayout_iqr,
  r3d_shape_gemm_nt_q, r3d_rad_unrelayout_iqr },

{ "gemm_iqr_nt_m16_w256", "gemm_nt_q", "gemm",
  "gemm_nt_q over an IQ8R weight at decode heights: the rotated int8 grid at a 256-wide scale "
  "group, so an exact i32 sub-sum runs over two 128-K blocks and folds once with the "
  "(token, 256) and (row, 256) scales",
  "M <= 16; K a multiple of 256 and N of 16; group 256",
  "I8 a row-major with an f32 scale per (row, 256 of K), I8 codes b with one bf16 scale per "
  "(row, 256) (the container's u8 and bf16 levels' product), bf16 y; groups ascend, each "
  "group's sub is one i32 dot and its two f32 factors multiply left to right",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_M16,
  ROW(cGemmQIqrM16W256), NOTUNE,
  /* No scratch hook: the 256-wide group never splits, and a hook that always answered 0 would
   * mark this decode row non-concurrent (rad_kernel_concurrent is `scratch == nullptr`). The
   * m16 hook itself still answers 0 for group 256. */
  nullptr, nullptr, r3d_rad_gemm_nt_q_iqr_m16, nullptr,
  r3d_rad_layout_iqr, r3d_rad_relayout_iqr,
  r3d_shape_gemm_nt_q, r3d_rad_unrelayout_iqr },

{ "gemm_iqr_tiled_w256", "gemm_nt_q", "gemm",
  "gemm_iqr_tiled at IQ8R's 256-wide scale group: the resident 16-row tile decodes each 128-K "
  "block once and the group's two blocks fold once",
  "M >= 1, K a multiple of 256 and N of 16; group 256",
  "I8 a row-major with an f32 scale per (row, 256 of K), I8 codes b with one bf16 scale per "
  "(row, 256), bf16 y; per group an exact i32 sub-sum over its two blocks and the two f32 "
  "factors",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_TILED,
  ROW(cGemmQIqrTiledW256), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_iqr_tiled,
  nullptr, r3d_rad_layout_iqr, r3d_rad_relayout_iqr,
  r3d_shape_gemm_nt_q, r3d_rad_unrelayout_iqr },

 { "gemm_nt_q_w8a8_m64", "gemm_nt_q", "gemm",
   "the symmetric 8-bit weight grid at prefill heights: signed int8 codes against int8 "
   "activations, the group's integer sub-sum scaled once by the activation's row scale and the "
   "weight's group scale",
   "17 <= M <= 127, K a multiple of 128; two 16-row tiles a wave, the weight column loaded once "
   "per 16-k step and reused over the tiles, the group terms scaled in the epilogue",
   "I8 a in the step order with an f32 scale per row, I8 codes b with an f16 scale per (row, 128 "
   "of K), bf16 y, fp32 accumulation; the sub-sums are exact integers under 2^24, and the store is "
   "the only narrowing; no zero point, so no a_sum",
   RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
   ROW(cGemmQW8A8M64), NOTUNE,
   nullptr, nullptr, r3d_rad_gemm_nt_q_w8a8_m64,
   nullptr, nullptr, nullptr, r3d_shape_gemm_nt_q_w8a8 },

    { "silu_mul_bf16", "silu_mul", "elem",
      "silu(gate) * up over one fused gate_up plane, the gate the first half",
      "any n; a workgroup a row; the activation is not narrowed before the product, as ref's is not "
      "(act_mul, over two operands, narrows it, as a stored activation is)",
      "bf16 in and out, fp32 arithmetic",
      RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
      ROW(cSiluMul), NOTUNE,
      nullptr, nullptr, r3d_rad_silu_mul, nullptr, nullptr, nullptr, r3d_shape_silu_mul },

    { "rmsnorm_quant_fp8_bf16", "rmsnorm_quant_fp8", "quant",
      "residual add, RMS norm and the E4M3 quantiser in one pass: a 5120-element bf16 plane a token "
      "is written once and read once inside one kernel, where the pair writes it and reads it back",
      "group 128 and a multiple of 128; a workgroup a row; the residual sum is stored before the "
      "variance reads it, so the norm is over the bf16 sum, and the codes are taken over the bf16 "
      "normalised value, which out_bf16 receives",
      "bf16 x, residual, gain and optional out_bf16, fp8_e4m3 codes, f32 scales, fp32 arithmetic",
      RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
      ROW(cRmsQuantFp8), NOTUNE,
      nullptr, nullptr, r3d_rad_rmsnorm_quant_fp8, nullptr, nullptr, nullptr,
      r3d_shape_rmsnorm_quant_fp8 },

    { "rmsnorm_had_quant_i8g_bf16", "rmsnorm_had_quant_i8g", "quant",
      "residual add, RMS norm, 128-wide Hadamard and per-group int8 in one pass: the int8 trunk's "
      "rotated producer, byte-identical to rmsnorm followed by had_quant_act_i8g",
      "group 128 and n a multiple of 128; a workgroup a row, a wave a group after the norm "
      "reduction; the residual sum is stored before the variance reads it, so the norm is over the "
      "bf16 sum, and the codes are taken over the bf16 norm output, which out_bf16 receives",
      "bf16 x, residual, gain and optional out_bf16, int8 codes (row-major), f32 scales amax/127 "
      "(the rotation's 1/128 is the weight plane's), fp32 arithmetic",
      RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
      ROW(cRmsQuantI8g), NOTUNE,
      nullptr, nullptr, r3d_rad_rmsnorm_had_quant_i8g, nullptr, nullptr, nullptr,
      r3d_shape_rmsnorm_had_quant_i8g },

    /* IQ8R's group 256: two independent 128-wide rotations, one amax and scale over the pair. */
    { "rmsnorm_had_quant_i8g_256_bf16", "rmsnorm_had_quant_i8g", "quant",
      "rmsnorm_had_quant_i8g at IQ8R's group: each 128-wide half of a 256-column group rotated on "
      "its own, one int8 scale for the pair",
      "group 256 and n a multiple of 256; a workgroup a row, a wave a group after the norm "
      "reduction; the residual sum is stored before the variance reads it, the codes are taken "
      "over the bf16 norm output, which out_bf16 receives",
      "bf16 x, residual, gain and optional out_bf16, int8 codes (row-major), f32 scales amax/127 "
      "over each 256-column pair (the rotation's 1/128 is the weight plane's), fp32 arithmetic",
      RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
      ROW(cRmsQuantI8g256), NOTUNE,
      nullptr, nullptr, r3d_rad_rmsnorm_had_quant_i8g, nullptr, nullptr, nullptr,
      r3d_shape_rmsnorm_had_quant_i8g },


/* ------------------------------------------------------------------ r3d_sample_pre_f32.hip
 *
 * The sampler's pre-filters: one workgroup a row over the logits plane. The history stages are
 * sparse (a row's history is a few hundred tokens against a vocabulary of about 130K), so walking
 * the window rather than the row, the reference's shape, is also the fast one.
 */
{ "sample_penalties_f32", "sample_penalties", "sample",
  "presence, frequency and repetition penalties from the row's history, in place",
  "a workgroup a row; the count is taken from the token's FIRST occurrence forward, and the "
  "repetition penalty is sign-dependent -- v > 0 divides, v < 0 multiplies -- as the reference has "
  "it; a penalty of zero writes nothing",
  "f32 logits, f32 params, i32 history; writes only inside the row's own n_vocab columns",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_penalties, nullptr, nullptr, nullptr,
  r3d_shape_sample_penalties },

{ "sample_dry_f32", "sample_dry", "sample",
  "the DRY penalty: a suffix of the history that repeats after a token penalises it, by its length",
  "a workgroup a row; the longest repeat per token, capped at dry_allowed_length; the multiplier "
  "grows by repeated multiply and stops at FLT_MAX/base, which is where an exponent would overflow; "
  "one penalty per token, from its longest repeat, and the FIRST occurrence is the one written",
  "f32 logits, f32 params, i32 history, u8 breakers; a breaker position ends a sequence and is "
  "never penalised; a history no longer than dry_allowed_length penalises nothing",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_dry, nullptr, nullptr, nullptr,
  r3d_shape_sample_dry },

{ "sample_ngram_ban_f32", "sample_ngram_ban", "sample",
  "the n-grams the history shows inside the window are banned to -inf",
  "a workgroup a row; the prefix compare over n-1 tokens, the ban landing on s[idx+n-1]; the "
  "whitelist is removed from the banned set BEFORE the ban, so an exempt id survives; the window is "
  "the row's own ngram pair's second word (0 clamps to the whole staged span), and n = 0 bans "
  "nothing",
  "f32 logits, f32 params, i32 history, i32 n-gram pairs, an optional i32 whitelist plane",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_ngram_ban, nullptr, nullptr, nullptr,
  r3d_shape_sample_ngram_ban },

{ "sample_mask_f32", "sample_mask", "sample",
  "the grammar's and the tool-caller's doing: every id whose bit is clear becomes -inf",
  "a workgroup a row over 4080 words at this vocabulary; a word of all ones is skipped and the last "
  "partial word honours n_vocab; a NEGATIVE mask_row skips the row, which is how an unconstrained "
  "request ends up with no mask applied -- the descriptor calls the same thing "
  "a shared row, and the body is the spec; the bound is the bitmask plane's own row count, not M, "
  "so a row addressing the plane's second half is masked as the reference masks it",
  "f32 logits, f32 params, i32 bitmask; a plane with fewer rows than the logits is refused "
  "(RAD_E_SHAPE), the case the geometry never produces",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_mask, nullptr, nullptr, nullptr,
  r3d_shape_sample_mask },

{ "sample_logprobs_f32", "sample_logprobs", "sample",
  "the row's summary for the response: max, the sum of exp, and the k best (id, logit) pairs",
  "one read of the logits plane where the reference reads it three (max, sum, k-list); the sum is "
  "SUM exp(v - max) in DOUBLE, the reference's law, and a row's k-list is one insertion list in "
  "registers, not an upper_bound per element; stats is [M][2 + 2k], and rows the flag leaves alone "
  "are not written (no params at all: every row runs)",
  "f32 logits, f32 params, f32 stats; k <= 64 is the pool's depth and above it the shim refuses "
  "(RAD_E_UNSUPPORTED); on a row that contains a NaN the reference's list freezes at the first k it "
  "saw: once a NaN sits in the k-list's tail, every later real value fails the tail's compare "
  "(anything against NaN is false) and is never inserted -- this kernel reports the row's true "
  "k-best, a divergence a plane has to be fed NaNs to reach",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleLogprobs), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_logprobs, nullptr, nullptr, nullptr,
  r3d_shape_sample_logprobs },


/* ------------------------------------------------------------------ r3d_attn_paged_h128.hip
 *
 * Paged attention, head_dim 128, banded on q_len. The two kernels share the block walk and differ
 * in what a workgroup is: decode is one head group over 16 query rows with split-KV, prefill a
 * 16-token query tile. The walk's block bounds are per workgroup: a tile is staged once into LDS
 * and every wave consumes it, so a wave must not narrow the walk to its own query positions, or the
 * waves would stage different blocks into the same LDS.
 */
{ "attn_paged_bf16_decode", "attn_paged", "attn",
  "paged attention over a bf16 cache: one head group a workgroup, 16-token tiles, split-KV",
  "head_dim 128; q_len <= 16; two passes over the blocks -- scores are recomputed in the second so "
  "the softmax sum accumulates ascending like the reference, and a masked key is never exponentiated; "
  "the block bounds come from the workgroup's 16 rows, because the staged tile is shared",
  "bf16 q and out, bf16 cache, i32 block_table and seqused; a descale on a bf16 cache is refused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_DECODE,
  ROW(cAttnBf16Decode), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_decode_h128, attn_scratch, nullptr, nullptr,
  r3d_shape_attn_paged },

{ "attn_paged_bf16_prefill", "attn_paged", "attn",
  "the same walk over a 16-token query tile, for a chunk that is wider than the decode bucket",
  "head_dim 128; q_len >= 17; a workgroup is 4 waves over 16 query rows and one KV head; the tile's "
  "idle lanes clamp to the last query row instead of reading past the q plane",
  "bf16 q and out, bf16 cache, i32 block_table and seqused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_PREFILL,
  ROW(cAttnBf16Prefill), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_prefill_h128, nullptr, nullptr, nullptr,   /* no split: no scratch */
  r3d_shape_attn_paged },

{ "attn_paged_fp8_decode", "attn_paged", "attn",
  "paged attention over an E4M3 cache, the descale riding the widening into the tile",
  "head_dim 128; q_len <= 16; a 16-code chunk is the aligned 16-byte unit -- eight codes are 8 bytes "
  "and a 16-byte load at a slot's odd half returns garbage; the descale "
  "multiplies into the f16 tile, exact for a power-of-two scale inside f16's range",
  "bf16 q and out, fp8_e4m3 cache, i32 block_table and seqused, f32 k_descale and v_descale "
  "optional (absent is scale 1.0)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_DECODE,
  ROW(cAttnFp8Decode), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_decode_h128, attn_scratch, nullptr, nullptr,
  r3d_shape_attn_paged },

{ "attn_paged_fp8_prefill", "attn_paged", "attn",
  "the E4M3 cache over a 16-token query tile",
  "head_dim 128; q_len >= 17; the same staged walk as the bf16 prefill, with the codes widened as "
  "they land in the tile",
  "bf16 q and out, fp8_e4m3 cache, i32 block_table and seqused, f32 descales optional (absent is "
  "scale 1.0)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_PREFILL,
  ROW(cAttnFp8Prefill), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_prefill_h128, nullptr, nullptr, nullptr,   /* no split: no scratch */
  r3d_shape_attn_paged },


{ "mla_project_bf16", "mla_project", "attn", "per-head absorbed MLA projections",
  "head_dim 128 or 256, latent_dim=512; pitched tensors", "bf16 input, weights and output; fp32 arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY, ROW(cMlaProject), NOTUNE,
  nullptr, nullptr, r3d_rad_mla_project, nullptr, nullptr, nullptr,
  r3d_shape_mla_project },

{ "mla_attn_paged_bf16", "mla_attn_paged", "attn",
  "dense causal latent attention with a rotary part, a wave a (row, head), split history",
  "latent_dim=512, rope_dim=64, heads<=64, block_size<=256; one page table a sequence",
  "bf16 Q/rotary/cache/gate/output, f32 softmax and partials",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY, ROW(cMlaAttnPaged), NOTUNE,
  nullptr, nullptr, r3d_rad_mla_attn_paged, r3d_rad_mla_attn_paged_scratch, nullptr, nullptr,
  r3d_shape_mla_attn_paged },

/* ------------------------------------------------------------------ r3d_sample_f32.hip
 *
 * The sampler's spine: the row's top-R and the stages that turn its candidate plane into one token.
 * The candidate set is the contract between them: `cand_val` carries logits, so each narrowing
 * stage takes its own softmax and the stages compose in any order; a dropped candidate is
 * `cand_idx` -1 and `cand_val` -inf; the set is descending by value, then ascending by id; and every
 * stage leaves a greedy row byte-identical, because a greedy row's token is `sample_argmax`'s.
 * `topk_next` is a sort by (-value, +column) written as a loop (`v == *pv && i <= *pi` steps through
 * a run of equal values), so one pass keeping the best R returns the reference's R pairs, in its
 * order and with its tie-breaks to the lower column.
 */
{ "row_topk_bf16", "row_topk", "moe",
  "the row's top R, with the reference's tie-breaks, in one scan of the row",
  "one pass keeps the best R in the reference's order, where the reference scans the row R times; "
  "R beyond a pass's pool rescans with the chain's state carried, so the answer is the reference's "
  "at any R and only the time grows; a slot a short row cannot fill is idx -1 and val -inf; a row "
  "that opens with a NaN reports (0, NaN) R times, because the reference's seed takes the first "
  "element it sees and nothing displaces a NaN",
  "bf16 x, i32 idx [M,R], f32 val [M,R], an optional f32 pairs [M,R,2] (the id's bits and the value)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRowTopkBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_row_topk, nullptr, nullptr, nullptr, r3d_shape_row_topk },

{ "row_topk_f32", "row_topk", "moe",
  "the row's top R over an f32 plane, in one scan",
  "the same one-pass insertion as the bf16 row; the band's router row costs the reference 7.0 GB of "
  "scans and this kernel 468 MB, and at the band it is the sifts that cost, not the bytes: the f32 "
  "plane moves half again the traffic for the same 23 ms",
  "f32 x, i32 idx, f32 val, optional f32 pairs",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRowTopkF32), NOTUNE,
  nullptr, nullptr, r3d_rad_row_topk, nullptr, nullptr, nullptr, r3d_shape_row_topk },

{ "logit_rerank_i8", "logit_rerank", "select",
  "logit_rerank over the container's int8 head: the coarse head's candidates rescored against "
  "`output.weight` and re-sorted on the exact values",
  "logit_rerank's: any M, R <= 32, bf16 x, a multiple-of-128 n_embd; the head is `i8*bf16[1x128]` "
  "joined into K codes and K/128 bf16 scales a row",
  "bf16 in, f32 values + i32 ids out, descending, ties to the lower id, an id outside this rank's "
  "rows a pad (-1, -inf)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cLogitRerankIqr), NOTUNE,
  nullptr, nullptr, r3d_rad_logit_rerank_i8, nullptr,
  r3d_rad_layout_i8_logits, r3d_relayout_logits_head, nullptr, nullptr },

{ "logit_rerank_iqr", "logit_rerank", "select",
  "logit_rerank over an IQR head: the coarse head's candidates rescored against logits_gemm_iqr's "
  "rows, then re-sorted on the exact values",
  "logit_rerank's: any M, R <= 32, bf16 x, a multiple-of-128 n_embd; the head is the IQR lm form "
  "with a bf16 scale per 128 of K",
  "bf16 in, f32 values + i32 ids out, descending, ties to the lower id, an id outside this rank's "
  "rows a pad (-1, -inf)",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cLogitRerankIqr), NOTUNE,
  nullptr, nullptr, r3d_rad_logit_rerank_iqr, nullptr,
  r3d_rad_layout_iqr_logits, r3d_rad_relayout_iqr_logits,
  nullptr, r3d_rad_unrelayout_iqr_logits },

{ "sample_argmax_f32", "sample_argmax", "sample",
  "the greedy token: the row's argmax, ties to the lower id",
  "a workgroup a row; the max is a strict `>`, so a NaN is never adopted -- the reference's scan "
  "keeps the first element it sees, which is why a row opening with a NaN yields token 0, and a row "
  "of all -inf yields 0; the stage that reads no scalar at all, so a missing params operand is not "
  "a refusal: the reference describes the operand and does not read it",
  "f32 logits, i32 token [M]; the params plane is described, not read",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_argmax, nullptr, nullptr, nullptr,
  r3d_shape_sample_argmax },

{ "sample_temp_f32", "sample_temp", "sample",
  "temperature, in place: every logit multiplied by the reciprocal of temp",
  "a workgroup a row; a row is skipped when temp is not positive or is exactly 1; the reciprocal is "
  "formed ONCE, as the reference forms it, not a divide per element -- a divide per element is a "
  "different f32, and the argmax after it is only bit-exact under the transcription",
  "f32 logits INOUT, f32 params; the most expensive row in the slice at 11 ms a step, and the cost "
  "is the plane: 1 GiB in and 1 GiB out for one multiply an element",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_temp, nullptr, nullptr, nullptr,
  r3d_shape_sample_temp },

{ "sample_topk_f32", "sample_topk", "sample",
  "the candidate set the rest of the sampler draws from, written whole every step",
  "the set's width is the BUFFER's, not the row's: top_k 40 in a 1024-wide plane is 40 candidates "
  "and 984 holes, and a stale candidate left in a hole is a token the pick can draw; the row's "
  "top_k only narrows inside the width, and `<= 0` or a width above it takes the whole width, as "
  "the reference clamps; cand_val carries the LOGIT, which is what lets the narrowing stages "
  "compose in any order; a greedy row takes none",
  "f32 logits, f32 params, i32 cand_idx [M,n_cand] (local columns), f32 cand_val, an optional i32 "
  "pairs [M,2*n_cand] of (global id, logit bits) with 0xffffffff for a hole; n_cand <= 1024",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleF32), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_topk, nullptr, nullptr, nullptr,
  r3d_shape_sample_topk },

{ "sample_pick_f32", "sample_pick", "sample",
  "the draw: one uniform over the candidate set's own softmax, in index order",
  "splitmix64 over the seed and the position, the top 24 bits over 2^24, keyed by the pick's own "
  "key; the exps are computed by the wave in parallel and the CDF accumulates on one lane in "
  "candidate index order, because the order is the law -- parallel values, serial sums; an empty "
  "set writes -1; a set whose mass falls below u falls back to the last candidate above -infinity, "
  "which is why a masked tail is never the fallback",
  "i32 cand_idx, f32 cand_val, f32 params (seed, pos), i32 token [M]; only token is written",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_pick, nullptr, nullptr, nullptr,
  r3d_shape_sample_pick },

{ "sample_topp_f32", "sample_topp", "sample",
  "nucleus: the shortest prefix whose cumulative probability reaches top_p",
  "the stage takes its own softmax over the set (max by strict `>` over live candidates, then exp, "
  "then the predicate), because the set carries logits; at least one candidate survives; it is a "
  "PREFIX rule in the set's order, which is why the set's order is contract and why no stage sorts "
  "it",
  "i32 cand_idx INOUT, f32 cand_val INOUT, f32 params; off when top_p is out of range; a greedy row "
  "and an all-holes row are left alone",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_topp, nullptr, nullptr, nullptr,
  r3d_shape_sample_topp },

{ "sample_minp_f32", "sample_minp", "sample",
  "min-p: drop what is below p times the best candidate's probability",
  "the stage's own softmax again; the best always survives; a relative cut under the best, not a "
  "prefix, so it composes with top-p in either order",
  "i32 cand_idx INOUT, f32 cand_val INOUT, f32 params; off when min_p is out of range",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_minp, nullptr, nullptr, nullptr,
  r3d_shape_sample_minp },

{ "sample_typical_f32", "sample_typical", "sample",
  "locally typical: drop the candidates whose surprisal is furthest from the set's entropy",
  "the set's entropy from the same exps, the kept mass overshooting p by design and not depending "
  "on the order, ties kept -- a stage that sorted the set would genuinely disagree with the schema",
  "i32 cand_idx INOUT, f32 cand_val INOUT, f32 params; off when typical_p is out of range",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_typical, nullptr, nullptr, nullptr,
  r3d_shape_sample_typical },

{ "sample_xtc_f32", "sample_xtc", "sample",
  "the exclusion of top candidates: a coin, then drop all but the least likely of the clearers",
  "its own coin (the seed keyed by XTC) decides whether the stage fires at all; when it fires it "
  "drops every candidate at or above the threshold EXCEPT the least likely of them, and does nothing "
  "when fewer than two clear it -- a stage that dropped unconditionally would fire on 100% of steps",
  "i32 cand_idx INOUT, f32 cand_val INOUT, f32 params; off when the probability or the threshold is "
  "out of range",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cSampleCand), NOTUNE,
  nullptr, nullptr, r3d_rad_sample_xtc, nullptr, nullptr, nullptr,
  r3d_shape_sample_xtc },

/* ------------------------------------------------------------------ r3d_dflash_f32.hip
 *
 * The speculative decoder's accept step. It is a gather, not arithmetic: two 63.8 MiB codebooks and
 * `M * steps * rank` gathered elements, so each score's rank sum is one lane's serial chain. There
 * is no work to hide the gather behind, and a tree reduction would change the sum for no gain.
 */
/* ------------------------------------------------------------------ r3d_gemm_nt_q_w2a8.hip
 *
 * `output.weight`: U2 codes, int8 activations, one f32 scale per activation row (ZP_I8_ROW), and a
 * joined `b_scale` stored as two planes (scale f16, zero u8) behind one operand, which only a layout
 * hook can present. libref's shape hook declines the dtype for that reason, so kbench has no case
 * for the band: the row's opd_shape is null (the tool's named skip), and the joined plane is built
 * with rad_planes_relayout, the same builder the driver uses.
 *
 * Exactness: `v_fdot2_f32_f16` carries about 22 bits of mantissa on integer operands, not 24, so an
 * integer dot through it is inexact above 2^20. The error is about 2^-22 of the magnitude, so a
 * smaller bias shrinks the tail; the bias is 128 (sampled worst |err| 0.015625, against 0.125 at
 * bias 1024). 0.1-0.2% of outputs land one bf16 ulp off the reference (rel_l2 5.0e-3, max_abs
 * 9.8e-4, no non-finite values). This row is not bit-exact; the unit's dot probe states the bound.
 *
 * M <= 16 is decode.
 */
{ "gemm_nt_q_w2a8_m16", "gemm_nt_q", "gemm",
  "the vocab projection at decode: 2-bit codes, int8 activations, the zero removed with a_sum",
  "a wave takes one weight row; lane l owns group l&15 and half of it, 64 codes to a 16-byte load, "
  "the pair (l, l^16) folding the group in one shuffle; lanes 0..15 hold the per-group terms and "
  "lane 0 sums the groups in ascending order in one accumulator, which is the reference's order. "
  "The codes widen to f16 through a bias of 128, so the dot's integer answer comes back with "
  "B*sum(a) on it and a_sum -- group-major, the reference's own 16-row pad -- removes it",
  "U2 codes b, I8 activations a, f32 a_scale (one per row), the joined b_scale (f16 scale + u8 zero "
  "as rad_planes stores them), f32 a_sum, bf16 y. fp8a8, fp8a16 and w4a8 take their own rows; a zero "
  "plane stored sub-byte (U2/U4/U5) is refused, not approximated -- the bit-extract path is 10 lines "
  "of rad_load_bits and is not written; M > 16 is refused, prefill's vocab is logits_gemm",
  RAD_DOMAIN_DEVICE, R3D_PRIO_VOCAB_M1,
  ROW(cGemmQW2A8), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w2a8, nullptr,
  r3d_layout_gemm_nt_q_w2a8, r3d_relayout_gemm_nt_q_w2a8, nullptr },

{ "gemm_nt_q_w2a8_tiled", "gemm_nt_q", "gemm",
  "the packed-code grids at the heights the skinny row declines: 2-bit and 4-bit codes, int8 "
  "activations, the batched decode and the draft head of a 2-bit vocab projection",
  "the same 128x128 tile as the fp8a8 arm, with the codes unpacked on the way into LDS -- 8 codes "
  "from a 4-byte load at U2, 16 from a 16-byte load at U4 -- and the epilogue folding "
  "(code - zero) * scale per staged element, so the dot runs over values and no a_sum term is "
  "reduced: the zero comes off term by term, which is why Σ a(q-z) needs no Σ a here while the m16 "
  "row needs it to lift the integer bias off its f16-as-int dot. The row's scale and zero are one "
  "per ROW on a {1, group} plane, so the staging reads 128 scalars a chunk where the fp8a8 arm reads "
  "one, at the row the grid puts it; issued one chunk ahead of the fold that consumes them, because "
  "a scalar load with nothing to overlap it dominates the band otherwise (the output bits are "
  "the same either way). r3d_tiling.h's swizzle is shared, proven byte-identical",
  "M <= 16 is gemm_nt_q_w2a8_m16's by its declared C_LE(\"M\", 16); fp8a8 and fp8a16 take their own "
  "rows, and the fp8a8 arm refuses a band carrying a_sum at its own line -- a row that silently "
  "ignores a band operand passes a checkpoint whose a_sum is garbage. A zero plane stored sub-byte "
  "(U2/U4) is refused and named, not approximated (rad_load_bits reads codes, not zeros). At "
  "kKC=32 a staged chunk draws 8 bytes from a row of pitch 512, so the band moves 8x the bytes it "
  "uses and is far from its read floor; a wider k-chunk, not more loads in flight, is what would "
  "close it",
  RAD_DOMAIN_DEVICE, R3D_PRIO_VOCAB_M1,
  ROW(cGemmQW2A8Tiled), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_w2a8_tiled, nullptr,
  r3d_layout_gemm_q_fp8_weight, r3d_relayout_gemm_q_fp8_weight, r3d_shape_gemm_nt_q },

/* ------------------------------------------------------------------ r3d_gemm_fp8a8_gated.hip
 *
 * The gate couples column i to column n+i, so a lane owns both dots and the coupling costs no extra
 * weight traffic. The block is one group by 256/group rows so the group's amax closes in-block. That
 * block shape is the quantiser's rule, so there is nothing to tune, and the row states its M ceiling
 * in the prose. Throughput is low (about 0.7% of the card's fp8 peak at M=64), but the alternative
 * is the reference's host GEMM.
 */
{ "gemm_nt_q_gated_fp8a8", "gemm_nt_q_gated", "gemm",
  "the gated MLP band's GEMM, its gate and its output quantiser in one launch",
  "a block is one 128-wide group by 256/group rows, so the group's amax closes in-block: wave max "
  "then an LDS fold over the lanes that share a group. The gate couples column i to column n+i, so a "
  "lane owns both dots and the coupling costs no extra weight traffic. No [M,N] bf16 intermediate "
  "exists: the row is narrowed, gated and coded from registers, which is why there is no scratch and "
  "no second pass. The narrowing matches the unfused chain -- both halves bf16 before the gate, the "
  "activation NOT narrowed before the product, the product formed in f32 then narrowed -- so the "
  "codes are the codes the chain would have written, byte for byte at the band",
  "fp8_e4m3 a and b, f32 a_scale, bf16 b_scale (f32 accepted and equal bit for bit; fp8a8f32 is a "
  "different dtype and takes no row here), f32 scale, E4M3 q, bf16 out_bf16 optional; f32 "
  "accumulation with -ffp-contract=off; f16x4 dot pairing, the code widening exact. M > 64, N odd, K "
  "% group != 0, group outside {32,64,128,256} or wider than n, sk > 1 and an absent act are refused, "
  "not approximated",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmGated), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_gated, nullptr, nullptr, nullptr,
  r3d_shape_gemm_nt_q_gated },

{ "gemm_iqr_gated_i8a8h", "gemm_nt_q_gated", "gemm",
  "the gated MLP fold over a rotated int8 (i8a8h) IQR weight: the gate's and the up's IQR decode, "
  "the bf16 gate_up rounding, silu, the 128-wide Hadamard and the per-group int8 quantiser in one "
  "launch",
  "M <= 16, N a multiple of 256, K a multiple of 128, group 128; the same IQR forms and private "
  "layout as gemm_iqr_m16. One workgroup is one 128-column group, sixteen waves: eight decode the "
  "gate's 16-row tiles and eight the up's, each with the m16 kernel's K loop and order, then the "
  "gate and up values meet in LDS and a wave a token runs the Hadamard and the codes. No [M,2n] "
  "bf16 intermediate and no scratch (the hook answers 0), so the row is one device launch where the "
  "pair is two, and its codes and scales are the pair's bytes",
  "I8 a row-major with an f32 scale per (row, 128 of K), IQR b/b_scale, I8 q (row-major), f32 "
  "scale per (row, 128 of n), optional bf16 out_bf16; fp32 arithmetic in the m16 order, a bf16 "
  "round trip a value before the gate, as the pair stores it",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_M16,
  ROW(cGemmGatedI8), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_iqr_gated, r3d_rad_gemm_nt_q_iqr_gated_scratch,
  r3d_rad_layout_iqr, r3d_rad_relayout_iqr,
  r3d_shape_gemm_nt_q_gated_i8a8h, r3d_rad_unrelayout_iqr },

/* IQ8R's gated fold: the same one-launch chain at a 256-wide scale group, one i32 sub-sum over
 * the group's two 128-K blocks and one amax/scale over the two 128-column halves. */
{ "gemm_iqr_gated_w256_i8a8h", "gemm_nt_q_gated", "gemm",
  "gemm_iqr_gated_i8a8h at IQ8R's 256-wide scale group: an exact i32 sub-sum over two 128-K "
  "blocks and one amax over two 128-column halves",
  "M <= 16, N a multiple of 512 (the gated width's whole 256 groups), K a multiple of 256, group "
  "256; the same IQR forms and private layout as gemm_iqr_nt_m16_w256",
  "I8 a row-major with an f32 scale per (row, 256 of K), IQR b/b_scale, I8 q (row-major), f32 "
  "scale per (row, 256 of n), optional bf16 out_bf16; per group one exact i32 sub-sum and the two "
  "f32 factors, the quantiser's two rotations under one amax",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_M16,
  ROW(cGemmGatedI8W256), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_q_iqr_gated, r3d_rad_gemm_nt_q_iqr_gated_scratch,
  r3d_rad_layout_iqr, r3d_rad_relayout_iqr,
  r3d_shape_gemm_nt_q_gated_i8a8h, r3d_rad_unrelayout_iqr },

{ "dflash_select_bf16", "dflash_select", "draft",
  "DFlash2's greedy walk over the candidate lattice, one lane a (sequence, candidate) pair",
  "score[l][c] = unary[l][c] + SUM_r pred[pid(l)][r] * hp[l][r] * succ[cand[l][c]][r], the rank sum "
  "ascending in r in f32 with ax[r] = pred * hp rounded to f32 BEFORE it meets succ, because the "
  "reference writes it to a float array first; argmax strict `>` over c ascending, so the earliest "
  "candidate wins a tie and a NaN never takes the lead; pid(0) is the anchor at m * anchor_stride "
  "and pid(l) the walk's own previous pick, so the steps loop is serial with the two LDS round "
  "trips its dependence requires; hp absent is a plane of ones; a predecessor outside the vocabulary "
  "contributes zero and a candidate outside it is scored by unary alone, which is what makes a "
  "shard's padding rows legal input",
  "bf16 codebooks (pred, succ, optional hp) with contiguous rows, f32 unary, i32 cand, anchor and "
  "tokens; pitched rows served on cand, unary, tokens and hp, a row stride INSIDE a codebook row "
  "refused (RAD_E_STRIDE); top_k > 256 refused (RAD_E_UNSUPPORTED); a cand plane describing more "
  "sequences than the band's M is refused, not truncated",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cDflashSelect), NOTUNE,
  nullptr, nullptr, r3d_rad_dflash_select, nullptr, nullptr, nullptr,
  r3d_shape_dflash_select },

/* ------------------------------------------------------------------ r3d_vocab_bf16.hip
 *
 * The head is the largest read in a decode step (510 MiB at 130560 x 2048), and the sampler reads
 * f32, so the store is f32 rather than narrowed and a prefill band writes about 1 GB. Against the
 * reference the f32 plane is at rel_l2 1.77e-6 (about 30 f32 ulps at K=2048), from dot2 folding two
 * products into one rounding; widening the operands to f16 first changes nothing. The head's own
 * bf16 quantisation (4.9e-4 an element) is 300x coarser.
 */
{ "logits_gemm_bf16", "logits_gemm", "vocab",
  "logits[M, n_vocab] = x[M, n_embd] @ lm_head[n_vocab, n_embd]^T, the head read once a token",
  "M <= 16: a wave a run of vocabulary columns, one output element a lane, the head's rows streamed "
  "past an activation kept in registers, no wave reduction. M > 16: 128x128 tiles through LDS, 8x8 "
  "f32 accumulators a lane, the head staged with an 8-element swizzle, k ascending in 32-code "
  "chunks. The grid is vocab-wide at every M, so there is no split-K and no scratch",
  "bf16 x, f32 logits: every sampler stage reads f32, so the narrowing the schema does not ask for "
  "is not performed, and a prefill band's output plane is M x n_vocab f32. The head arrives in the "
  "form the container stores and the row reads both, chosen by the operand's header, not by a "
  "parameter: bf16 rows in the order they are stored, or -- what this checkpoint's output.weight "
  "actually is -- E4M3 codes with a bf16 scale per 128x128 block as two container planes, which the "
  "row's layout hook presents as the one operand the plan says they are, moving no byte. E4M3 and "
  "int8 widen to f16 exactly, so the fold into f16 is the only arithmetic the codes pay: 2.77e-04 "
  "from the reference's exact law, one f16 rounding, 250x finer than the code's own quantum. A head "
  "with a zero plane is w2a8's grid and is refused at the shim's own line; so is a scale block that "
  "is not 128x128, which is gemm_nt_q's grid",
  RAD_DOMAIN_DEVICE, R3D_PRIO_VOCAB_M1,
  ROW(cLogitsGemm), tImpl2, NELEM(tImpl2), impl_tune_valid,
  nullptr, nullptr, r3d_rad_logits_gemm_bf16, nullptr,
  r3d_layout_logits_head, r3d_relayout_logits_head, r3d_shape_logits_gemm_bf16 },

/* ------------------------------------------------------------------ r3d_gemm_fp8a8_tiled.hip
 *
 * The prefill projection. At these densities the dot product, not memory bandwidth, is the limit
 * (about 27-36% of the card's dense f16 peak at the graph's bands), which is why the E4M3 decoder is
 * a table: the decode ALU is on the critical path.
 */
{ "gemm_nt_q_fp8a8_tiled", "gemm_nt_q", "gemm",
  "y[M,N] = a[M,K] @ dequant(b)^T at prefill heights, the group sub-sum from the raw codes and one "
  "scale per group, which is the reference's law and the reason the codes are never pre-scaled",
  "the tune's impl axis picks the decode: 0 = the block's 512-byte E4M3->f16 table, 1 = the "
  "per-byte ALU formula. 128x128 tiles through LDS, 64 f32 accumulators a lane, k in 32-code chunks, "
  "the family's 8-element "
  "XOR swizzle on the staged rows; both sides decode through the block's 512-byte E4M3->f16 table "
  "and leave staging as f16 with their scale folded in -- a_scale per staged row, b_scale one scalar "
  "per tile, because the tile's 128 columns are exactly the scale plane's row extent and 32 divides "
  "128. One accumulator, no per-group partials, 237 VGPRs and no spill",
  "E4M3 codes both sides, f32 a_scale per (row, group of 128), bf16 b_scale per 128x128 block -- the "
  "container's own block; a {1,group} scale block is w2a8's grid and is refused, as is a zero plane "
  "and an a_sum, fp8a8 being symmetric. y bf16 with one narrowing at the store, and at a bf16 output "
  "the store's ulp (3.9e-3) is the binding floor, not the fold's (9.8e-4 of the quadrature scale): "
  "83% of outputs are bit-exact, 99.93% land within the fold plus one store ulp, none beyond. The "
  "fold is v_mul_f16, never f16x2_pack, which rounds toward zero and flushes subnormals",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQFP8Tiled), tImpl2, NELEM(tImpl2), impl_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_fp8_tiled, nullptr,
  r3d_layout_gemm_q_fp8_weight, r3d_relayout_gemm_q_fp8_weight, r3d_shape_gemm_nt_q },

{ "gemm_nt_q_mxfp4a8_tiled", "gemm_nt_q", "gemm",
  "y[M,N] = a[M,K] @ dequant(b)^T at prefill heights, the weight in OCP MXFP4: e2m1 codes, two a "
  "byte, one e8m0 scale per 32 of K; the 32-block partial scales at the block boundary and the "
  "blocks sum in ascending k",
  "the tune's impl axis picks the decode: 0 = the block's 512-byte table, 1 = the per-byte ALU "
  "formula. 128x128 tiles through LDS, 64 f32 accumulators a lane for the block partials and 64 more "
  "for the running sum, k in 32-code chunks -- one OCP 32-block a chunk -- the family's 8-element XOR "
  "swizzle on the staged rows; the codes unpack to e4m3 through eight fixed magnitude bytes and "
  "decode through the block's 512-byte table, a_scale folded per staged row, the e8 read at "
  "staging and held in registers for the flush",
  "E4M3 a with an f32 scale per (row, 128 of K), fp4_e2m1 codes with an e8m0 scale per (row, 32 "
  "of K) -- `b_ref`, the same exponent plane, is accepted and ignored, bf16 y, fp32 accumulation. "
  "The e2m1 magnitude is exact in e4m3, the code products are "
  "exact in f16, the 32-block partial scales by a power of two in f32, and the store is the only "
  "narrowing; a reserved e8 (0, 255) is refused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_FP8,
  ROW(cGemmQMxfp4a8Tiled), tImpl2, NELEM(tImpl2), impl_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_q_mxfp4a8_tiled, nullptr,
  nullptr, nullptr, r3d_shape_gemm_nt_q_mxfp4a8 },

/* The same tiling on gemm_nt's bf16 operands, the prefill projection the reference runs on the
 * host. The exactness is the m16 row's, because it is the same dot. */
{ "gemm_nt_bf16_tiled", "gemm_nt", "gemm",
  "a[M,K] @ b[N,K]^T at prefill heights, f32 accumulate, one narrowing at the store",
  "the tune's impl axis picks the kernel: 0 = 128x128 tiles through LDS, 8x8 f32 accumulators a "
  "lane, k ascending in 32-code chunks, the partials closed in the tile; 1 = the same tile, the bf16 "
  "decode in registers (the per-byte ALU formula); 2 = the same tile on the wave-matrix (eight 16x16 "
  "tiles a wave); 3 = the 16-column-tile wave-matrix kernel, MT 16-row tiles a block (the M = 17..127 "
  "band); M <= 16 stays with gemm_nt_bf16_m16, whose wave-per-run-of-columns "
  "shape is the right one there -- the two rows' M ranges are complements, so neither competes for "
  "the other's band",
  "bf16 a, b and y, fp32 accumulation; the pair is folded into the accumulator by one rounding, so "
  "about 0.3% of a long-K output lands one bf16 store ulp from the reference's, and no output is "
  "further",
  RAD_DOMAIN_DEVICE, R3D_PRIO_GEMM_TILED,
  ROW(cGemmNTTiled), tImpl4, NELEM(tImpl4), impl_tune_valid,
  nullptr, nullptr, r3d_rad_gemm_nt_bf16_tiled, nullptr,
  nullptr, nullptr, r3d_shape_gemm_nt },

/* ------------------------------------------------------------------ W9: r3d_embed_lookup_q.hip,
 * r3d_ngram_hint.hip, r3d_ple_gate_bf16.hip (ple_conv is upstream's r3d_ple_conv.hip)
 *
 * Qwen4-Exp's Per-Layer Embedding. The table is the 320-million-row n-gram embedding, E4M3 with
 * one scale for the whole table (or a plain bf16 table read in place). The hint is a prefetch
 * only; the gate and convolution are the same arithmetic as libref, with the sums in double where
 * libref accumulates in double.
 */
{ "embed_lookup_q_bf16", "embed_lookup_q", "gemm",
  "the table row an id names, times the table's one scale; an id outside the shard is zero",
  "any n_embd; one workgroup a row; E4M3 codes with an optional bf16/f16/f32 scale, or a bf16 "
  "table with no scale",
  "codes decoded exactly and multiplied once, then one narrowing to bf16/f16/f32",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cEmbedQ), NOTUNE,
  nullptr, nullptr, r3d_rad_embed_lookup_q, nullptr, nullptr, nullptr, nullptr },

{ "ngram_hint_i32", "ngram_hint", "elem",
  "the n-gram row ids of each sequence's newest token, one draft pass before they are hashed for "
  "real; a prefetch hint, nothing reads them as model ids",
  "one thread a sequence; heads split into ngram-1 equal blocks; tail keeps the two committed tails",
  "the ngram_ids hash and narrowing, with the same EOS cut",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cNgramHint), NOTUNE,
  nullptr, nullptr, r3d_rad_ngram_hint, nullptr, nullptr, nullptr, nullptr },

{ "ple_gate_bf16", "ple_gate", "norm",
  "the signed-square-root dot between two normed streams, its sigmoid times v, and the rms of that "
  "product; one workgroup a (row, stream)",
  "bf16 k/q/v/gv/gvn, f32 gains; n <= 4096 and hc 2..8, the row and register budget of the "
  "workgroup reduction",
  "both norms and the dot in double, the scale returned as f32, expf rather than __expf",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cPleGate), NOTUNE,
  nullptr, nullptr, r3d_rad_ple_gate, nullptr, nullptr, nullptr, nullptr },

/* ------------------------------------------------------------------ W8: r3d_hc_bf16.hip
 *
 * The wide residual stream. The read is four launches (norm, down+inj, up, quant) so the weight
 * stream spreads over the card; `d` and the stream scales travel in the row's scratch. The E4M3
 * mixed form packs each row's codes and scales through the host layout hook below.
 */
{ "hc_read_e4m3", "hc_read", "norm",
  "the gated residual's read over an E4M3 mixing pair: grouped norm, low-rank silu gate, stream "
  "mean, injection gains, and the optional q/rq forms of x",
  "n a multiple of 128, lowrank a multiple of 64, hc 2..8, hc*n within one 48 KiB LDS stage; "
  "the two mix weights are the joined E4M3 form with a f32 scale per 128 of K (down) or K/4 (up)",
  "the norm in double; each dot ascending in f32 with no FMA; the same narrowings as libref, so "
  "the read is bit-identical to the reference",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY + 1,
  ROW(cHcReadE4m3), NOTUNE,
  nullptr, nullptr, r3d_rad_hc_read, r3d_rad_hc_read_scratch,
  r3d_rad_layout_hc_mix, r3d_rad_relayout_hc_mix, nullptr, r3d_rad_unrelayout_hc_mix },

{ "hc_read_bf16", "hc_read", "norm",
  "the same read over plain bf16 mixing matrices",
  "n a multiple of 128, hc 2..8, hc*n within one 48 KiB LDS stage",
  "the norm in double; each dot ascending in f32 with no FMA; one bf16 narrowing per value",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cHcReadBf16), NOTUNE,
  nullptr, nullptr, r3d_rad_hc_read, r3d_rad_hc_read_scratch,
  nullptr, nullptr, nullptr, nullptr },

/* ------------------------------------------------------------------ W25: the MTP head's entry
 *
 * The drafter's first op at N > 0. The two square fc matrices are read as E4M3 rows (libr4d's
 * joined form), so a Flash-Next container converted there serves here without a second layout.
 */
{ "mtp_enter_e4m3", "mtp_enter", "norm",
  "the MTP head's entry with fc_hidden and fc_embedding stored as E4M3 rows: half the bytes of the "
  "two square matrices every draft round reads",
  "any M, n a multiple of 128 up to 5734, hc 2..4",
  "bf16 activations, E4M3 fc rows with f32 128-column group scales in each row's tail, f32 gains",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cMtpEnterE4m3), NOTUNE,
  nullptr, nullptr, r3d_rad_mtp_enter, nullptr,
  r3d_rad_layout_mtp_e4m3, r3d_rad_relayout_mtp_e4m3,
  nullptr, r3d_rad_unrelayout_mtp_e4m3 },

{ "hc_write_bf16", "hc_write", "norm",
  "h[c] += inj[c] * y on each of the hc streams, in place on the unnormed residual",
  "bf16, hc 2..8, any even n",
  "the product in f32 and one bf16 store, the write half's arithmetic",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cHcStream), NOTUNE,
  nullptr, nullptr, r3d_rad_hc_write, nullptr, nullptr, nullptr, nullptr },

{ "hc_enter_bf16", "hc_enter", "norm",
  "hc identical copies of the embedding: the wide stream's first value",
  "bf16 x and h, hc 2..8, any even n",
  "one narrowing, the same value into each stream",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cHcStream), NOTUNE,
  nullptr, nullptr, r3d_rad_hc_enter, nullptr, nullptr, nullptr, nullptr },

{ "hc_project_bf16", "hc_project", "norm",
  "the control vector's projection on the rows whose flag is set: h[c] -= scale*(h[c].v)*v",
  "bf16 h, f32 v, i32 on; hc 2..8, any even n",
  "the dot in double and the scale applied in double, as libref writes it; one bf16 store",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cHcStream), NOTUNE,
  nullptr, nullptr, r3d_rad_hc_project, nullptr, nullptr, nullptr, nullptr },

/* ------------------------------------------------------------------ W8: r3d_ar_hc_write.hip
 *
 * The collective pair at one rank. At tp:1 the reduction is the identity; the two- and four-rank
 * forms stay with libr4d (a gfx11 card has no peer fabric path in this library).
 */
{ "ar_hc_write_1rank", "ar_hc_write", "ar",
  "the one-rank all-reduce (identity) and the gated residual's write that follows it",
  "exactly world_size 1; bf16, hc 2..8, n a multiple of 8",
  "hc_write's arithmetic over the message, byte-identical to the unfused pair at one rank",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cArHc), NOTUNE,
  nullptr, nullptr, r3d_rad_ar_hc_write, nullptr, nullptr, nullptr, nullptr },

{ "ar_gather_hc_write_1rank", "ar_gather_hc_write", "ar",
  "moe_gather folded in front of the one-rank ar_hc_write: the routed experts' rows summed back "
  "to token order, then the gated write",
  "exactly world_size 1; bf16, top_k <= 64, hc 2..8, n a multiple of 8",
  "the gather's sums and narrows are moe_gather's; the write is hc_write's over that bf16 value",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cArGatherHc), NOTUNE,
  nullptr, nullptr, r3d_rad_ar_gather_hc_write, nullptr, nullptr, nullptr, nullptr },

/* --------------------------------------------------------------- tp: the collectives
 *
 * The engine's tp>1 path: the ports of libhalo's all-reduce and all-gather (r3dh_ar_*, the
 * halo_ -> r3dh_ rename). They move the payload through the peer's IPC scratch and the start-up
 * exchange buffers, so a row needs `device.p2p` 1. At four and eight ranks the exact sum is
 * one-shot up to its bound and two-shot above; the message's band picks the form, as in libr4d.
 */
{ "ar_twoshot_3rank_exact", "all_reduce", "ar",
  "all-reduce (sum), reduce-scatter + all-gather over P2P IPC scratch",
  "exactly 3 ranks, all peer-accessible; messages from "
  "524289 elements; shards as even as the word count allows",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_TWOSHOT,
  ROW(cArTsExact3), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_twoshot_3rank_exact, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_twoshot_4rank_exact", "all_reduce", "ar",
  "all-reduce (sum), reduce-scatter + all-gather over P2P IPC scratch",
  "exactly 4 ranks, all peer-accessible; messages from "
  "524289 elements, sharding evenly",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_TWOSHOT,
  ROW(cArTsExact4), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_twoshot_4rank_exact, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_twoshot_4rank_exact_lane", "all_reduce", "ar",
  "all-reduce (sum) with a narrow launch beside compute",
  "exactly 4 ranks, peer-accessible, overlap 1; messages from "
  "524289 elements, sharding evenly",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, same bytes as the wide row",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_TWOSHOT + 1,
  ROW(cArTsExact4Lane), ROW(tArTs4Lane), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_twoshot_4rank_exact, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_twoshot_8rank_exact", "all_reduce", "ar",
  "all-reduce (sum), reduce-scatter + all-gather over P2P IPC scratch",
  "exactly 8 ranks, all peer-accessible; messages from "
  "8193 elements, sharding evenly",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_TWOSHOT,
  ROW(cArTsExact8), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_twoshot_8rank_exact, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_oneshot_2rank_exact", "all_reduce", "ar",
  "all-reduce (sum), one-shot push over P2P IPC scratch",
  "exactly 2 ranks, peer-accessible; scratch sized by the full message; cudagraph-safe",
  "bf16 / fp16 / fp32 payload, fp32 accumulate, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT,
  ROW(cArExact), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_oneshot_2rank_exact, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_oneshot_2rank_wht6", "all_reduce", "ar",
  "all-reduce (sum) with a Walsh-Hadamard-rotated 6-bit wire payload -- lossy",
  "exactly 2 ranks, peer-accessible; groups of 64 elements",
  "bf16 / fp16 payload, 6 bits + bf16 scale per group on the wire, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT,
  ROW(cArWht6), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_oneshot_2rank_wht6, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_oneshot_2rank_exact_lane", "all_reduce", "ar",
  "all-reduce (sum), one-shot push with a narrow launch beside compute",
  "exactly 2 ranks, peer-accessible, overlap 1; scratch sized by the full message",
  "bf16 / fp16 / fp32 payload, fp32 accumulate, same bytes as the wide row",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT + 1,
  ROW(cArExactLane), ROW(tArLane), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_oneshot_2rank_exact, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_oneshot_2rank_wht6_lane", "all_reduce", "ar",
  "all-reduce (sum) on the rotated 6-bit wire -- lossy; narrow launch beside compute",
  "exactly 2 ranks, peer-accessible, overlap 1; groups of 64 elements",
  "bf16 / fp16 payload, 6 bits + bf16 scale per group on the wire, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT + 1,
  ROW(cArWht6Lane), ROW(tArLane), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_oneshot_2rank_wht6, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_oneshot_3rank_exact", "all_reduce", "ar",
  "all-reduce (sum), one-shot push over P2P IPC scratch",
  "exactly 3 ranks, peer-accessible; above "
  "524288 elements ar_twoshot_3rank_exact outranks it",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT,
  ROW(cArExact3), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_oneshot_3rank_exact, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_oneshot_4rank_exact", "all_reduce", "ar",
  "all-reduce (sum), one-shot push over P2P IPC scratch",
  "exactly 4 ranks, peer-accessible, at most "
  "524288 elements",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT,
  ROW(cArExact4), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_oneshot_4rank_exact, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_oneshot_8rank_exact", "all_reduce", "ar",
  "all-reduce (sum), one-shot push over P2P IPC scratch",
  "exactly 8 ranks, peer-accessible, at most "
  "8192 elements",
  "bf16 / fp16 / fp32 payload, fp32 accumulate in ascending rank order, bit-identical across ranks",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT,
  ROW(cArExact8), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_oneshot_8rank_exact, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_twoshot_3rank_ti8", "all_reduce", "ar",
  "all-reduce (sum) on the tiered-int8 wire -- lossy",
  "exactly 3 ranks, peer-accessible; whole 32-element groups, split over the shards as evenly as "
  "they divide",
  "bf16 / fp16 payload, int8 + fp16 scale + u32 record per group on the wire, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT,
  ROW(cArTi8Ts3), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_twoshot_3rank_ti8, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_twoshot_4rank_ti8", "all_reduce", "ar",
  "all-reduce (sum) on the tiered-int8 wire -- lossy",
  "exactly 4 ranks, peer-accessible; numel a multiple of 128",
  "bf16 / fp16 payload, int8 + fp16 scale + u32 record per group on the wire, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT,
  ROW(cArTi8Ts4), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_twoshot_4rank_ti8, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_twoshot_4rank_ti8_lane", "all_reduce", "ar",
  "all-reduce (sum) on the tiered-int8 wire -- lossy; narrow launch beside compute",
  "exactly 4 ranks, peer-accessible, overlap 1; numel a multiple of 128",
  "bf16 / fp16 payload, int8 + fp16 scale + u32 record per group on the wire, fp32 accumulate",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT + 1,
  ROW(cArTi8Ts4Lane), ROW(tArTs4Lane), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_twoshot_4rank_ti8, nullptr, nullptr, nullptr,
  r3d_shape_all_reduce },

{ "ar_gather_2rank", "all_gather", "ar",
  "all-gather, one-shot push over P2P IPC scratch",
  "exactly 2 ranks, peer-accessible; y is [2 * numel] with rank r's input at r * numel",
  "any payload: the bytes are moved, never summed, so (u32 id, f32 logit) pairs survive intact",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT,
  ROW(cArGather2), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_gather_2rank, nullptr, nullptr, nullptr,
  r3d_shape_all_gather },

{ "ar_gather_3rank", "all_gather", "ar",
  "all-gather, one-shot push over P2P IPC scratch",
  "exactly 3 ranks, peer-accessible; y is [3 * numel] with rank r's input at r * numel, or "
  "row-interleaved [rows][3][row] when `row` is given",
  "any payload: the bytes are moved, never summed, so (u32 id, f32 logit) pairs survive intact",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT,
  ROW(cArGather3), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_gather_3rank, nullptr, nullptr, nullptr,
  r3d_shape_all_gather },

{ "ar_gather_4rank", "all_gather", "ar",
  "all-gather, one-shot push over P2P IPC scratch",
  "exactly 4 ranks, peer-accessible; y is [4 * numel] with rank r's input at r * numel",
  "any payload: the bytes are moved, never summed, so (u32 id, f32 logit) pairs survive intact",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT,
  ROW(cArGather4), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_gather_4rank, nullptr, nullptr, nullptr,
  r3d_shape_all_gather },

{ "ar_gather_8rank", "all_gather", "ar",
  "all-gather, one-shot push over P2P IPC scratch",
  "exactly 8 ranks, peer-accessible; y is [8 * numel] with rank r's input at r * numel",
  "any payload: the bytes are moved, never summed, so (u32 id, f32 logit) pairs survive intact",
  RAD_DOMAIN_DEVICE, R3D_PRIO_AR_ONESHOT,
  ROW(cArGather8), ROW(tAr), nullptr,
  r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_gather_8rank, nullptr, nullptr, nullptr,
  r3d_shape_all_gather },


/* ------------------------------------------------------------------ r3d_attn_paged_h256.hip
 *
 * The head_dim-256 int8 body, banded on q_len and paired by geometry. Four lanes own one output
 * row (64 dims each); a workgroup carries 4 tokens (gqa 12 / decode), 8 (gqa 6 / decode,
 * gqa 12 / prefill) or 16 (gqa 6 / prefill), so every staged cache block is shared by all of a
 * tile's rows in one KV head. The decode row split-KVs into 260-float records; the prefill row is
 * a single pass. Both read the cache's per-slot f16 K/V scales and have no descale planes.
 */
{ "attn_paged_i8_h256_gqa12_decode", "attn_paged", "attn",
  "paged attention over an int8 head_dim-256 cache, 24 query heads over 2 KV heads, 4-token "
  "pages: split-KV decode",
  "head_dim 256; gqa 12; block_size 4; q_len <= 16; four lanes an output row (64 dims each) and "
  "three waves a workgroup (4 tokens a tile); the block walk is the workgroup's token range, and "
  "the four lane partials are summed in d-block order -- the documented departure from libref's "
  "single 256-add chain",
  "bf16 q and out, i8 cache with a slot row of 256 K codes, 256 V codes and the two f16 scales "
  "(no k_descale / v_descale; a descale is refused), i32 block_table and seqused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_DECODE,
  ROW(cAttnH256Gqa12Decode), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_decode_h256, attn_h256_scratch, nullptr, nullptr,
  r3d_shape_attn_paged_i8 },

{ "attn_paged_i8_h256_gqa12_prefill", "attn_paged", "attn",
  "the same int8 head_dim-256 body over a prefill chunk, 24 query heads over 2 KV heads",
  "head_dim 256; gqa 12; block_size 4; q_len >= 17; six waves a workgroup (8 tokens a tile); no "
  "split pass",
  "bf16 q and out, i8 cache with per-slot f16 K/V scales (no descales), i32 block_table and "
  "seqused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_PREFILL,
  ROW(cAttnH256Gqa12Prefill), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_prefill_h256, nullptr, nullptr, nullptr,
  r3d_shape_attn_paged_i8 },

{ "attn_paged_i8_h256_gqa6_decode", "attn_paged", "attn",
  "paged attention over an int8 head_dim-256 cache, the 27B's 24 query heads over 4 KV heads, "
  "16-token pages: split-KV decode",
  "head_dim 256; gqa 6; block_size 16; q_len <= 16; four lanes an output row (64 dims each) and "
  "three waves a workgroup (8 tokens a tile)",
  "bf16 q and out, i8 cache with per-slot f16 K/V scales (no descales), i32 block_table and "
  "seqused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_DECODE,
  ROW(cAttnH256Gqa6Decode), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_decode_h256, attn_h256_scratch, nullptr, nullptr,
  r3d_shape_attn_paged_i8 },

{ "attn_paged_i8_h256_gqa6_prefill", "attn_paged", "attn",
  "the same int8 head_dim-256 body over a prefill chunk at the 27B geometry",
  "head_dim 256; gqa 6; block_size 16; q_len >= 17; six waves a workgroup (16 tokens a tile); no "
  "split pass",
  "bf16 q and out, i8 cache with per-slot f16 K/V scales (no descales), i32 block_table and "
  "seqused",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ATTN_PREFILL,
  ROW(cAttnH256Gqa6Prefill), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_prefill_h256, nullptr, nullptr, nullptr,
  r3d_shape_attn_paged_i8 },


/* ------------------------------------------------------------------ the 27B's int8 store and rope
 *
 * kv_store_i8: one wave a (token, KV head), the head's absmax reduced across the wave and rounded
 * to an f16 scale, then round(x/scale) codes. rope_table builds the cos/sin plane the fused
 * prologue reads; rope_imrope is Qwen3-VL's interleaved three-component rotary (sections 11 11
 * 10) over NeoX pairs, q heads then k heads, v untouched.
 */
{ "kv_store_i8", "kv_store", "attn",
  "K and V into an int8 paged cache: each head stores 256 K codes, 256 V codes and the two f16 "
  "scales, the scale the head's absmax / 127 rounded to f16 and the codes round(x/scale)",
  "head_dim 256; block_size 16; a wave a (token, KV head); a negative slot skipped; a zero head "
  "stores scale 0 and codes 0",
  "bf16 k and v, i32 slot_mapping, i8 cache with per-slot f16 K/V scales",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKvI8), NOTUNE,
  nullptr, nullptr, r3d_rad_kv_store, nullptr, nullptr, nullptr, r3d_shape_kv_store_i8 },

/* Flash-Next's 4-token pages: the same int8 slot layout and quantisation as `kv_store_i8`, with
 * the slot split by 4 rather than 16. Kept a separate row because the block size is part of the
 * KV group's geometry; the kernel is the same one, so a slot's bytes are unchanged. */
{ "kv_store_i8_b4", "kv_store", "attn",
  "K and V into an int8 paged cache at 4-token pages: each head stores 256 K codes, 256 V codes "
  "and the two f16 scales, the scale the head's absmax / 127 rounded to f16 and the codes "
  "round(x/scale)",
  "head_dim 256; block_size 4; a wave a (token, KV head); a negative slot skipped; a zero head "
  "stores scale 0 and codes 0",
  "bf16 k and v, i32 slot_mapping, i8 cache with per-slot f16 K/V scales",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cKvI8B4), NOTUNE,
  nullptr, nullptr, r3d_rad_kv_store, nullptr, nullptr, nullptr, r3d_shape_kv_store_i8 },

{ "rope_table_f32", "rope_table", "attn",
  "the cos/sin plane a fused prologue reads, cos in the first rot/2 values of a row and sin in "
  "the second, written at each position's own row",
  "rot 64; M 64; one block a position; only named rows are written and a negative or "
  "past-the-table position writes nothing; ref's double cos/sin and pow, narrowed once",
  "i32 positions, f32 cos/sin table",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRopeTableF32), NOTUNE,
  nullptr, nullptr, r3d_rad_rope_table, nullptr, nullptr, nullptr, r3d_shape_rope_table },

{ "rope_imrope_h256", "rope", "attn",
  "Qwen3-VL's interleaved three-component rotary over NeoX pairs, sections 11 11 10, in place on "
  "a fused qkv row: q heads then k heads, v untouched",
  "head_dim 256; rotary_dim 64; n_head 24; n_head_kv 4; mode imrope; positions [3, M] "
  "component-major; one block a token",
  "bf16 qkv in place, i32 positions, fp32 angles",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRopeImrope), NOTUNE,
  nullptr, nullptr, r3d_rad_rope, nullptr, nullptr, nullptr, r3d_shape_rope },

{ "rope_imrope_h128", "rope", "attn",
  "the QSA indexer's interleaved three-component rotary over NeoX pairs, sections 11 11 10, in "
  "place on four 128-wide query heads at rotary_dim 64; the head past the rotary is untouched",
  "head_dim 128; rotary_dim 64; n_head 4; n_head_kv 0; mode imrope; positions [M], [M, 3] "
  "token-major or [3, M] component-major at their own strides; YaRN's blend and magnitude; one "
  "block a token",
  "bf16 qkv in place, i32 positions, fp32 angles",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cRopeImropeH128), NOTUNE,
  nullptr, nullptr, r3d_rad_rope_imrope_h128, nullptr, nullptr, nullptr, r3d_shape_rope },


/* ------------------------------------------------------------------ the vision tower (W10-lite)
 *
 * The two rows the 27B's vision tower needs, and the constraints defined above the table.
 */

{ "attn_dense_bf16_h72", "attn_dense", "attn",
  "dense varlen attention over cu_seqlens segments: one thread a (query row, query head), two "
  "passes with the score recomputed, every f32 sum in ascending key order, softmax f32 and one "
  "bf16 store; an empty segment writes zeros",
  "head_dim 72; gqa 1; causal 0; no window, block, window_right or cu_div; q/k/v bf16 over their "
  "row pitch (a column slice of a fused qkv is read in place); positions come from cu_seqlens "
  "(i32); the softmax uses expf, one bf16 code from libref's double exp",
  "bf16 q, k, v and out; i32 cu_seqlens; softmax f32, output narrowed once",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cAttnDenseH72), NOTUNE,
  nullptr, nullptr, r3d_rad_attn_dense, nullptr, nullptr, nullptr, r3d_shape_attn_dense },

{ "rope_axial_bf16_h72", "rope", "attn",
  "the vision tower's 2-D axial rotary over NeoX pairs, two sections and a ladder restarted in "
  "each, in place on a fused qkv row: q heads then k heads, v untouched",
  "head_dim 72; rotary_dim 72; mode axial; two sections summing 36 (18 18 on the 27B); positions "
  "[2, M] component-major (row, column) read at their own strides; the angles are f32 products of "
  "double cos/sin and double pow, narrowed once",
  "bf16 qkv in place, i32 positions, fp32 angles",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ROPE_AXIAL,
  ROW(cRopeAxialH72), NOTUNE,
  nullptr, nullptr, r3d_rad_rope_axial, nullptr, nullptr, nullptr, r3d_shape_rope_axial },

{ "gemm_nt_bias_bf16", "gemm_nt_bias", "gemm",
  "the vision tower's linear layer: a 16-row x 32-column tile a block, both operands staged as "
  "f32 in LDS, two output columns a thread walking k ascending with one multiply and one add a "
  "step; the bias is added in f32 and the result narrowed, then act (exact or tanh GELU) and the "
  "residual each narrowed where a separate tensor would round",
  "bf16; any M, N and K (tails clamp); a/b/y/res at their row pitches; bias a dense [N]; act "
  "none, gelu or gelu_tanh; no split-K, no scratch",
  "bf16 a, b, bias, res and y; every output's chain is libref's ascending k with separate "
  "multiply and add, so the row is bit-identical to the reference at every band",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGemmVision), NOTUNE,
  nullptr, nullptr, r3d_rad_gemm_nt_bias, nullptr, nullptr, nullptr, r3d_shape_gemm_nt_bias },

{ "grid_embed_bf16", "grid_embed", "elem",
  "the vision tower's learned position table: a block a row, the [4, M] coordinate read once and "
  "its two-by-two taps formed in shared memory, then each thread adds the f32 tap sum (rounded to "
  "bf16) into x's row",
  "bf16; side x side table at its row pitch; [4, M] i32 coord at its own strides; x and table "
  "last axis contiguous; no alignment requirement",
  "bf16 x in place and table; taps summed in f32 in libref's (h,w) order and rounded once before "
  "the add, so the row is bit-identical to the reference",
  RAD_DOMAIN_DEVICE, R3D_PRIO_ONLY,
  ROW(cGridEmbedVision), NOTUNE,
  nullptr, nullptr, r3d_rad_grid_embed, nullptr, nullptr, nullptr, r3d_shape_grid_embed },

};

/* ================================================================== the exports
 *
 * The two exports every plugin needs, the four that make a kernel library, and
 * `rad_kernel_concurrent` and `rad_kernel_collective`.
 *
 * No `rad_kernel_chain`: chaining replaces the queue's ordering between two dependent launches with
 * a spin in the kernel, and no row here has a chained form, so every answer would be 0. Omitting the
 * symbol means the same (ABI, `rad_kernel_chain`: "without the symbol, or with 0, the engine never
 * asks").
 *
 * `rad_kernel_collective`: the all-reduce and all-gather rows exchange through IPC scratch and peer
 * flags, so the engine must order them against the graph and hold the link for them. They are
 * threads of one process in the selftest, but a serving engine runs one rank a card.
 */
extern "C" int rad_kernel_count(void) {
    return (int)(sizeof(kKernels) / sizeof(kKernels[0]));
}
extern "C" const RadKernelInfo* rad_kernel_at(int i) {
    if (i < 0 || i >= rad_kernel_count()) return nullptr;
    return &kKernels[i];
}

/* Which rows touch nothing but their operands and their scratch.
 *
 * The elementwise, norm, quant, cast, rope, cache and gather rows: one launch, one pass, no state
 * between launches, so two may run at once and the engine can start one before an independent
 * earlier one finishes.
 *
 * Not the GEMM rows, because of the scratch: the arena belongs to the op instance, so two layers'
 * decode GEMMs launched together share it and their split-K partials and arrival counters would
 * collide. With sk = 1 a GEMM has no partials, but the promise is per row and sk is chosen per
 * geometry, so the row states the worst case.
 */
extern "C" int rad_kernel_concurrent(int i) {
    const RadKernelInfo* k = rad_kernel_at(i);
    if (!k) return 0;
    /* The collectives keep per-block handshake and parity state in the instance, outside every
     * operand, so two launches must not overlap. */
    if (k->family && !std::strcmp(k->family, "ar")) return 0;
    return k->scratch == nullptr;
}

/* Collective rows: every launch exchanges with a peer rank through the instance's IPC scratch and
 * flags. Answer 1: the engine orders them and releases the link after the launch (no row here
 * returns RAD_LINK_ENDED early; that form needs the launch to hold the gate itself). */
extern "C" int rad_kernel_collective(int i) {
    const RadKernelInfo* k = rad_kernel_at(i);
    if (!k || !k->op) return 0;
    return !std::strcmp(k->op, "all_reduce") || !std::strcmp(k->op, "all_gather");
}

static const RadPluginInfo kInfo = {
    RAD_PLUGIN_KERNEL,
    "libr3d",
    "0.1.0",
    "the RDNA3 (gfx11) kernel library: the elementwise, norm, rope, cache and embedding rows, fp8 "
    "and int8 activation quantisation with its dequant and gated forms, and the skinny bf16 and "
    "fp8a16 GEMMs, the matrix work through v_dot2acc_f32_f16 rather than RDNA3's WMMA "
    "fragment layout",
    /* The architectures this build's code objects run on, spelled as `RadDeviceProps::arch`. The
     * loader warns once when the code object it picked names a card this list does not
     * (`target_covers`), so the string must be the build's real targets. R3D_BUILD_TARGET comes
     * from CMakeLists.txt as a compile definition; the fallback is the whole family. */
    R3D_BUILD_TARGET,
};

extern "C" uint32_t rad_plugin_abi_version(void) { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" const RadPluginInfo* rad_plugin_info(void) { return &kInfo; }

/* libr3d declines a machine with no gfx11 card. The loader would leave its device rows out there
 * (no code object fits), but rad-convert and the other tools resolve kernels without that check,
 * and libr3d, ahead of libr4d in file-name order, would choose the stored form of every weight it
 * has a row for: an RDNA4 container laid out for RDNA3 kernels, or one weight wanted in two forms.
 *
 * The test reads the KFD topology (gfx_target_version 110000..119999 is the gfx11 family) rather
 * than asking a runtime, which open must not initialise. KFD lists only the cards the process may
 * open, so a container without /dev/kfd sees the CPU node alone; with no topology at all there is
 * no card, and the library declines too. */
extern "C" int rad_plugin_open(void) {
    const char* root = "/sys/class/kfd/kfd/topology/nodes";
    DIR* d = opendir(root);
    if (!d) return RAD_E_UNSUPPORTED;
    bool gfx11 = false;
    while (const dirent* e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        const std::string path = std::string(root) + "/" + e->d_name + "/properties";
        FILE* f = std::fopen(path.c_str(), "r");
        if (!f) continue;
        char key[64];
        unsigned long long v = 0;
        while (std::fscanf(f, "%63s %llu", key, &v) == 2)
            if (!std::strcmp(key, "gfx_target_version") && v >= 110000ull && v < 120000ull)
                gfx11 = true;
        std::fclose(f);
    }
    closedir(d);
    return gfx11 ? RAD_OK : RAD_E_UNSUPPORTED;
}