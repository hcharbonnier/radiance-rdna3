/* r3dh_ar.h -- the all-reduce / all-gather entry points of the libhalo-derived RDNA3 port.
 * Generated from libhalo/halo.h (the halo_ -> r3dh_ rename), ar section only. */
#pragma once
#include <hip/hip_runtime.h>
#include "r3dh_args.h"

extern "C" {

// ---- all-reduce: one-shot, push, 2 ranks over P2P ------------------------------------------
// Each rank pushes its whole input into the peer's IPC scratch, then reduces locally; no ring, so
// the scratch is sized by the full message. Exactly 2 ranks: the handshake is a single peer flag.
// Every 2-rank kernel on one buffer set shares its counters: `seq_ctrs` is 2 * r3dh_ar_max_blocks()
// zeroed words, a handshake and a slot-parity counter per block of capacity (r3dh_ar_seq.h).
enum { R3DH_AR_HANDLE_BYTES = 64 };
// IPC scratch helpers, startup-only: alloc returns the device pointer and writes the IPC handle
// (up to HANDLE_BYTES) into out_handle. A refused export (beside another engine on the same cards)
// leaves it zeroed with *out_len 0: libhalo's ranks are threads of one process and share the pointer.
long r3dh_ar_ipc_alloc(long size, int finegrained, char* out_handle, int* out_len);
// Strided peer (or local) device-to-device copy through the copy engines; pitches and width in
// bytes, height in rows. Returns 0 or -100 - hipError_t.
int r3dh_p2p_copy2d_kernel(long dst, long src, long dpitch, long spitch, long width, long height,
                          int bx, int by, int threads, long stream);
int r3dh_p2p_fence(long stream);
int r3dh_p2p_copy2d(long dst, long src, long dpitch, long spitch, long width, long height,
                   long stream);
void r3dh_ar_ipc_free(long p);
void r3dh_ar_ipc_enable_peer(long peer);
// Exchange buffers in pinned, fine-grained host memory, for cards that cannot reach each other's
// VRAM. Zeroed; the address is valid on every card and the CPU. Startup-only.
long r3dh_ar_host_alloc(long size);
void r3dh_ar_host_free(long p);
// Exact sum, fp32 accumulate, bf16 / fp16 / fp32 payload (dtype 0 / 1 / 2).
void r3dh_ar_oneshot_2rank_exact(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                                long seq_ctrs, long slot_stride16, long inp, long out, long n_elem,
                                long dtype, long stream, long nblocks, long nthreads, long drain,
                                long acq);
// Same topology, with the wire payload Walsh-Hadamard rotated and quantised to 6 bits per element
// in groups of 64 (plus a bf16 scale per group). Lossy; bf16 / fp16 payload only. loc_pack is this
// rank's own packed copy, so the reduce folds exactly the bytes it sent.
void r3dh_ar_oneshot_2rank_wht6(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                               long seq_ctrs, long loc_pack, long slot_stride_bytes,
                               long scale_off_bytes, long inp, long out, long n_elem, long dtype,
                               long stream, long nblocks, long nthreads, long drain, long acq);
// The exact all-reduce of `y` [M][n] bf16 with hc_write as its epilogue: h[r][c*n + i] +=
// inj[r][c] * y[r][i] over the reduced y, byte-identical to all_reduce then hc_write. `irow` and
// `hrow` are the pitches of inj [M][hc] and h [M][hc*n] in elements; y is tight. A non-zero `gate`
// is the engine's link word, ended once the peer's push has landed (r3dh_ar_link_landed);
// `gate_ctr` is its block counter. The gather form below takes the same pair.
void r3dh_ar_hc_write(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                     long seq_ctrs, long slot_stride16, long y, long inj, long h, long M, long n,
                     long hc, long irow, long hrow, long stream, long nblocks, long nthreads,
                     long drain, long acq, long gate, long gate_ctr);
// The same with the MoE gather as its prologue: y is produced, not read, from the routed experts'
// rows `ye` [M*top_k][ye_ld] summed to token order by weights `ew` [M][ew_ld] and `sorted`
// [M*top_k], with the shared arm `sh` [M][sh_ld] and its gate `sg` folded when both are non-zero.
// Byte-identical to moe_gather, all_reduce, hc_write. A non-zero `inv` is M*top_k int32 device
// scratch for the inverse slot table a prefill-sized call reads instead of a per-block LDS one.
// Returns 0, or a negative code for a shape it does not serve.
int  r3dh_ar_gather_hc_write(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                            long seq_ctrs, long slot_stride16, long y, long inj, long h, long M,
                            long n, long hc, long irow, long hrow, long ye, long ew, long sorted,
                            long sh, long sg, long ye_ld, long ew_ld, long sh_ld, long sg_ld,
                            long top_k, long sig, long stream, long nblocks, long nthreads,
                            long drain, long acq, long inv, long gate, long gate_ctr);
// r3dh_ar_hc_write and r3dh_ar_gather_hc_write on the rotated 6-bit wire (r3dh_ar_oneshot_2rank_wht6's
// format, loc_pack and scratch split): byte-identical to all_reduce on that wire then hc_write,
// with `ye` non-zero preceded by moe_gather. Gather operands as r3dh_ar_gather_hc_write; its slot
// table is built in LDS, and the grid is widened until a block's table fits. M*n must be a
// multiple of 64. `gate` / `gate_ctr` as r3dh_ar_hc_write. Returns 0, or a negative code for a
// shape it does not serve.
int  r3dh_ar_hc_write_wht6(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                          long seq_ctrs, long loc_pack, long slot_stride_bytes,
                          long scale_off_bytes, long y, long inj, long h, long M, long n, long hc,
                          long irow, long hrow, long ye, long ew, long sorted, long sh, long sg,
                          long ye_ld, long ew_ld, long sh_ld, long sg_ld, long top_k, long sig,
                          long stream, long nblocks, long nthreads, long drain, long acq,
                          long gate, long gate_ctr);
int  r3dh_ar_max_blocks(void);                                   // the two 2-rank kernels
void r3dh_ar_wht6_dims(int* group, int* bits, int* chunk_elems); // rotated-6-bit payload only

// ---- all-gather: one-shot, push, exactly 2 ranks over P2P ----------------------------------
// The push and per-block flag handshake of r3dh_ar_oneshot_2rank_exact, with placement instead of
// reduction: `out` is 2 * n_elem elements, this rank's input at rank * n_elem and the peer's at
// (1 - rank) * n_elem. `dtype` only gives the element size; the payload is never summed, because
// the vocab-parallel sampler gathers (u32 token id, f32 logit) pairs.
void r3dh_ar_gather_2rank(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                         long seq_ctrs, long slot_stride16, long inp, long out, long n_elem,
                         long dtype, long rank, long row_elems, long stream, long nblocks,
                         long nthreads, long drain, long acq, long dpitch,
                         long spitch);

// ---- all-reduce: one-shot, push, 4 or 8 ranks over P2P -------------------------------------
// The same algorithm at width (r3dh_ar_oneshot_Nrank_exact.hip). Peer arguments are in ascending
// global rank with the current rank removed. Scratch is 2*world-size slots of slot_stride16 16B
// words (parity x source rank, this rank's own slot unused); flags are max_blocks*ws words indexed
// [block][source rank]; seq_ctrs is 2 * max_blocks words, a handshake and a slot-parity counter
// per block of capacity (r3dh_ar_seq.h).
// Three ranks is the same kernel at WS=3; the wide family does not need a power of two. The
// two-shot serves 3 too, with shards one word uneven where the message does not divide.
void r3dh_ar_oneshot_3rank_exact(long peer_scratch0, long peer_scratch1,
                                long peer_flags0, long peer_flags1,
                                long my_scratch, long my_flags, long seq_ctrs, long slot_stride16,
                                long inp, long out, long n_elem, long dtype, long rank,
                                long stream, long nblocks, long nthreads, long drain, long acq,
                                long pub);
void r3dh_ar_oneshot_4rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_flags0, long peer_flags1, long peer_flags2,
                                long my_scratch, long my_flags, long seq_ctrs, long slot_stride16,
                                long inp, long out, long n_elem, long dtype, long rank,
                                long stream, long nblocks, long nthreads, long drain, long acq,
                                long pub, long gate, long gate_ctr);
// The two wide releases' extra pair at four ranks: `gate` is the engine's link word (0 when this
// launch holds nothing -- the host mailbox, a recorded pass, or a width the row does not release),
// `gate_ctr` the block counter beside the sequence counters (r3dh_ar_seq.h). The 3- and 8-rank
// entries take neither: their rows answer 1 and the engine releases by barrier.
// ---- all-gather at width (r3dh_ar_gather_Nrank.hip) ------------------------------------------
// The 2-rank gather's placement on the wide family's transport: `out` is world * n_elem elements,
// rank r's slice at r * n_elem, or row-interleaved [rows][world][row] when row_elems > 0. Shares
// the wide all-reduce's instance (scratch, [block][source] flags, seq counters).
void r3dh_ar_gather_3rank(long peer_scratch0, long peer_scratch1, long peer_flags0,
                         long peer_flags1, long my_scratch, long my_flags, long seq_ctrs,
                         long slot_stride16, long inp, long out, long n_elem, long dtype,
                         long rank, long row_elems, long stream, long nblocks, long nthreads,
                         long drain, long acq, long pub, long dpitch,
                         long spitch);
void r3dh_ar_oneshot_8rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_scratch3, long peer_scratch4, long peer_scratch5,
                                long peer_scratch6, long peer_flags0, long peer_flags1,
                                long peer_flags2, long peer_flags3, long peer_flags4,
                                long peer_flags5, long peer_flags6, long my_scratch,
                                long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                long out, long n_elem, long dtype, long rank, long stream,
                                long nblocks, long nthreads, long drain, long acq, long pub);
// r3dh_ar_hc_write at four ranks: the 4-rank exact one-shot of `y` [M][n] bf16, in place, with
// hc_write as its epilogue, byte-identical to that all-reduce then hc_write. Buffers, peer order,
// flag values and slots match r3dh_ar_oneshot_4rank_exact, so the two (and the two-shots) may share
// one instance. `irow`/`hrow` as r3dh_ar_hc_write; y is tight.
void r3dh_ar_hc_write_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                           long peer_flags0, long peer_flags1, long peer_flags2,
                           long my_scratch, long my_flags, long seq_ctrs, long slot_stride16,
                           long y, long inj, long h, long M, long n, long hc, long irow,
                           long hrow, long rank, long stream, long nblocks, long nthreads,
                           long drain, long acq, long pub, long gate, long gate_ctr);
// r3dh_ar_hc_write_4rank with the MoE gather as its prologue, byte-identical to moe_gather, the
// 4-rank exact all-reduce, then hc_write. Gather operands, `inv` and the grid rule are
// r3dh_ar_gather_hc_write's; buffers, peer order, flag values and slots are
// r3dh_ar_oneshot_4rank_exact's. Exact wire only.
void r3dh_ar_gather_hc_write_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                  long peer_flags0, long peer_flags1, long peer_flags2,
                                  long my_scratch, long my_flags, long seq_ctrs,
                                  long slot_stride16, long y, long inj, long h, long M, long n,
                                  long hc, long irow, long hrow, long ye, long ew, long sorted,
                                  long sh, long sg, long ye_ld, long ew_ld, long sh_ld,
                                  long sg_ld, long top_k, long sig, long rank, long stream,
                                  long nblocks, long nthreads, long drain, long acq, long pub,
                                  long inv, long gate, long gate_ctr);
// The inverse slot table both widths' gather forms read at a prefill-sized call: inv[f] = the
// sorted row holding slot f, -1 where none does, over T = M*top_k int32. One block, on `stream`.
// Returns 0 or a negative / HIP error code.
int  r3dh_ar_gather_inv(long inv, long sorted, long T, long stream);
void r3dh_ar_wide_dims(int* max_blocks, int* nb_design, int* max_peers);

// ---- all-reduce: two-shot, 3, 4 or 8 ranks over P2P ----------------------------------------
// Reduce-scatter + all-gather on the wide one-shot's buffer trio and peer order
// (r3dh_ar_twoshot_Nrank_exact.hip): shard s, a contiguous run of 16-byte words, is reduced by
// rank s (fp32, ascending rank order), then gathered. Sends 2*(ws-1)/ws*N bytes per rank in two
// hops, for sizes where the one-shot's (ws-1)*N send is too heavy. Shards are as even as the word
// count allows (the first n16 % ws take one more word), so the message need not divide by the
// rank count. At 8 ranks `shard_weights` (8 ints, or null for the even split) sizes the shards per
// rank instead (r3dh_ar_shards.h); every rank must pass the same weights. Regions A and B sit at 0
// and the longest shard, so the slot stride must hold twice that. `pieces` (8 ranks, 1 to
// R3DH_AR_TS_MAX_PIECES, the same on every rank) pipelines each block's words in that many runs so
// phase 1 of one run and phase 2 of the previous share the link; 1 is the unpipelined kernel.
// Above the bound below the caller falls back to the one-shot; it only guards the kernel's int
// word offsets (2^28 elements is 2^26 words at fp32). A lower bound would send large prefill
// chunks (e.g. 8192 tokens at Gemma 4's 2816 width, 23M-46M elements) to the one-shot.
enum { R3DH_AR_TWOSHOT_WIDE_MAX_ELEMS = 268435456, R3DH_AR_TS_MAX_PIECES = 16 };
void r3dh_ar_twoshot_3rank_exact(long peer_scratch0, long peer_scratch1, long peer_flags0,
                                long peer_flags1, long my_scratch, long my_flags, long seq_ctrs,
                                long slot_stride16, long inp, long out, long n_elem, long dtype,
                                long rank, long stream, long nblocks, long nthreads, long drain,
                                long acq, long pub);
void r3dh_ar_twoshot_4rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_flags0, long peer_flags1, long peer_flags2,
                                long my_scratch, long my_flags, long seq_ctrs,
                                long slot_stride16, long inp, long out, long n_elem, long dtype,
                                long rank, long stream, long nblocks, long nthreads, long drain,
                                long acq, long pub, long gate, long gate_ctr, long chain);
// The four-rank exact two-shot with both pushes coded losslessly at ~12.2 bits a bf16 value (the
// same bytes out; r3dh_ar_twoshot_Nrank_exact.hip). r3dh_ar_twoshot_coded_ok says whether a message
// and slot fit the coded form: bf16, shards of whole 256-value waves, two coded regions a slot.
int r3dh_ar_twoshot_coded_ok(long n_elem, long dtype, long world, long slot_stride16);
void r3dh_ar_twoshot_4rank_exact_coded(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                      long peer_flags0, long peer_flags1, long peer_flags2,
                                      long my_scratch, long my_flags, long seq_ctrs,
                                      long slot_stride16, long inp, long out, long n_elem,
                                      long rank, long stream, long nblocks, long nthreads,
                                      long drain, long acq, long pub, long gate, long gate_ctr,
                                      long chain);
void r3dh_ar_twoshot_8rank_exact(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                long peer_scratch3, long peer_scratch4, long peer_scratch5,
                                long peer_scratch6, long peer_flags0, long peer_flags1,
                                long peer_flags2, long peer_flags3, long peer_flags4,
                                long peer_flags5, long peer_flags6, long my_scratch,
                                long my_flags, long seq_ctrs, long slot_stride16, long inp,
                                long out, long n_elem, long dtype, long rank, long stream,
                                long nblocks, long nthreads, long drain, long acq, long pub,
                                const int* shard_weights, long pieces, long timeline,
                                long form);
// `form` is R3DH_AR8_FORM_* (r3dh_ar_shards.h), the same on every rank: lockstep, ring or split,
// all with the same bits. `timeline`, a diagnostic, is 0 or a device buffer of
// nblocks * R3DH_AR_TL_WORDS int64 that every block fills with wall-clock stamps (r3dh_ar_wide.h):
// its start, each handshake's arrival and release, its end. r3dh_selftest --perfar8 --timeline
// reads it.
// The tiered int8 wire on both two-shot hops (r3dh_ar_twoshot_Nrank_ti8.hip; codec in
// r3dh_ar_ti8.h), 3 or 4 ranks. Int8 groups of 32 with an fp16 scale and a per-group record
// (T1 plain, T2 top-1-exact, T3 wht32), 38 bytes/group; tiering is deterministic and
// bit-identical across ranks. Peer order and flag layout as the exact pair; slot stride in bytes
// (the wire is not 16B-aligned), holding two wire images; the encoder writes its own slot too (no
// loc_pack). numel must be whole 32-element groups, split over the ranks as evenly as they divide
// (the first ng % ws shards one group longer), with every slot laid out for the longest shard.
// The 4-rank entry still requires numel % 128 == 0, the even split.
void r3dh_ar_twoshot_3rank_ti8(long peer_scratch0, long peer_scratch1, long peer_flags0,
                              long peer_flags1, long my_scratch, long my_flags, long seq_ctrs,
                              long slot_stride_bytes, long inp, long out, long n_elem,
                              long dtype, long rank, long stream, long nblocks,
                              long nthreads, long drain, long acq, long pub);
void r3dh_ar_twoshot_4rank_ti8(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                              long peer_flags0, long peer_flags1, long peer_flags2,
                              long my_scratch, long my_flags, long seq_ctrs,
                              long slot_stride_bytes, long inp, long out, long n_elem,
                              long dtype, long rank, long stream, long nblocks,
                              long nthreads, long drain, long acq, long pub, long gate,
                              long gate_ctr);
void r3dh_ar_ti8_dims(int* group, int* group_bytes);  // tiered-int8 wire only



// The same norm with the preceding tensor-parallel all-reduce folded in. Every row-parallel
// linear's output is all-reduced then normalised (128 adjacent dispatches per decode step); fused,
// the all-reduce's bf16 output is never read back and the peer's contribution is a second operand
// on the norm's load. IPC arguments are r3dh_ar_oneshot_2rank_exact's; `nblocks` is clamped to M
// because a block owns whole rows, and `x` takes no pitch because it is also the message. Like the
// int8 twin r3dh_ar_ln_had_quant_i8 it has two wires: `wire` 0 is the exact bf16 payload, 6 the
// rotated 6-bit one. Wire 6 needs `n` to be a multiple of r3dh_ar_rmsnorm_quant_fp8_chunk_elems()
// so a chunk never straddles a row.
int  r3dh_ar_rmsnorm_quant_fp8(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                              long seq_ctrs, long slot_stride16,
                              long x, long res, long w, long y, long q, long s,
                              int M, int n, float eps, float wadd, int w_f32,
                              int r_ld, int y_ld, int q_ld, int s_ld, long stream,
                              int nblocks, int nthreads, int drain, int acq, int wire,
                              long loc_pack, long loc_scale_off, long wp = 0);
int  r3dh_ar_rmsnorm_quant_fp8_chunk_elems(void);
// Its rotated int8 form, as r3dh_rmsnorm_had_quant_i8g: the reduction and the norm unchanged, the
// norm's pass quantised by had_quant_act_i8g's rule. No post-norm.
int  r3dh_ar_rmsnorm_had_quant_i8g(long peer_scratch, long my_scratch, long peer_flags,
                                  long my_flags, long seq_ctrs, long slot_stride16,
                                  long x, long res, long w, long y, long q, long s,
                                  int M, int n, float eps, float wadd, int w_f32,
                                  int r_ld, int y_ld, int q_ld, int s_ld, long stream,
                                  int nblocks, int nthreads, int drain, int acq, int wire,
                                  long loc_pack, long loc_scale_off);

// The same IPC with the sandwich-norm junction after it (r3dh_rmsnorm_quant_i8.hip): post-norm of
// the reduced `x` (`wp`), the residual add, the [1] scalar (`sc`), the norm (`w`) and the w4a8
// GEMM's per-row int8 quantiser. Exact wire only; `x` is the message and takes no pitch.
int  r3dh_ar_rmsnorm_quant_i8(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                             long seq_ctrs, long slot_stride16, long x, long res, long w, long wp,
                             long sc, int sc_f32, long y, long q, long s, int M, int n, float eps,
                             float wadd, int w_f32, int r_ld, int y_ld, long stream, int nblocks,
                             int drain, int acq);
// ...and at four ranks, on the wide family's transport: r3dh_ar_oneshot_4rank_exact's IPC
// arguments (peers in ascending rank, this rank's own, the counters, the slot stride), `pub`, the
// engine's link gate with the instance's landed counter (0 for none; r3dh_ar_seq.h), and, with
// `tail_nbt` blocks (positive), the standalone two-shot on that many with the junction as its tail,
// counting on the two words at `tail_ctr` (ar_rmsnorm_quant_i8_tail_kernel). With `sp` 1 or 2 the
// all-reduce split around the junction instead (ar_rmsnorm_quant_i8_sp_kernel): q and S on every
// row, the residual on this rank's rows (every row at 2), `x` left as it came.
int  r3dh_ar_rmsnorm_quant_i8_4rank(long peer_scratch0, long peer_scratch1, long peer_scratch2,
                                   long peer_flags0, long peer_flags1, long peer_flags2,
                                   long my_scratch, long my_flags, long seq_ctrs,
                                   long slot_stride16, long x, long res, long w, long wp, long sc,
                                   int sc_f32, long y, long q, long s, int M, int n, float eps,
                                   float wadd, int w_f32, int r_ld, int y_ld, int rank,
                                   long stream, int nblocks, int drain, int acq, int pub,
                                   long gate, long gate_ctr, long tail_ctr, int tail_nbt,
                                   int chain, int sp);

// ---- sampling ----------------------------------------------------------------------------------
//
// One workgroup a position: temperature, softmax, top_k, top_p, min_p, renormalise and the draw,
// in vLLM's order (reordering changes which tokens survive). Three outputs for a greedy-draft
// speculative verify: a draw, a draw with the proposed token removed and the rest renormalised,
// and the proposal's filtered probability, which is the rejection-sampling accept probability.
struct HaloSpecDraw {
    unsigned int  tok;        // a draw from the filtered distribution
    unsigned int  tok_excl;   // a draw with `query` removed and the remainder renormalised
    float         p_query;    // filtered probability of `query` -- the accept probability
};
// `logits[pos*col_stride + v*row_stride]`, so either head layout is read without a transpose.
// `query` may be 0; `u` is two uniforms a position. `n_vocab` is the full vocabulary: a
// vocab-parallel rank gathers the other bands first, since a nucleus needs the whole distribution.
int  r3dh_sample_chain_f32(long logits, long query, long u, long out,
                          int M, int n_vocab, int row_stride, int col_stride,
                          float temperature, float top_p, int top_k, float min_p, long stream);

// ---------------------------------------------------------------------------- the quantisers
//
// These take the plugin ABI's arguments: `int r3dh_rad_<name>(const RadArgs*, RadStream)`, declared
// in libhalo/r3dh_plugin.h because the row table is the only caller and that header is host-only.
// The entry point named under each note is the symbol. See libhalo/r3dh_args.h for why this is
// libhalo's only calling convention.

// Per-row symmetric int8 quantisation of a bf16 activation, in the A-fragment byte order. The
// optional fourth operand asks for the per-row per-group sum of the int8 codes an asymmetric
// weight grid needs -- [K/group][Mpad] float, group-major; see r3dh_gemm_w4a8_asym_nt_m64.
//   r3dh_rad_quant_act_i8
// Block Hadamard of width `had` along K, then the same per-row int8 quantisation. `had` must be a
// power of two dividing K; a rotation block cannot cross a tensor-parallel shard, so it must
// divide the per-rank K (5120 / 3072 / 8704 here, which rules 1024 out).
//   r3dh_rad_had_quant_act_i8
// The residual add, the RMS norm, the same block Hadamard and int8 quantiser in one pass.
// `residual` and `out_bf16` may be null: pass a residual for vLLM's fused_add_rms_norm form, and
// out_bf16 only on layers with a bf16 consumer of the post-norm tensor (a GDN block's in_proj_a /
// in_proj_b, which stay bf16 and read it unrotated).
// `wadd` is added to every gain element in fp32 after the bf16 load: GemmaRMSNorm (Qwen3.5,
// Qwen3-Next, Gemma) stores a zero-centred gamma and uses `1 + w`, and forming that in bf16 costs
// 0.3% per channel. Pass 0.0f for a plain RMS norm.
//   r3dh_rad_rmsnorm_had_quant_i8
// The same norm with the preceding all-reduce folded in, as in r3dh_ar_rmsnorm_quant_fp8: IPC
// arguments, double-buffered scratch, per-block flags and device-resident sequence counters
// are r3dh_ar_oneshot_2rank_exact's; `nblocks` is clamped to M because a block owns whole rows.
void r3dh_ar_ln_wire_dims(int* stage_bits, int* chunk_elems, int* chunk_bytes,
                         int* scale_bytes);
// The fewest blocks at which [M, K] fits the kernel's LDS at this width and wire (`nthreads` 0
// means the norm's own width), or -1 when no block count does. A block count below it is one the
// launcher refuses.
int  r3dh_ar_ln_blocks_min(int M, int K, int had, int nthreads, int wire);
void r3dh_ar_ln_had_quant_i8(long peer_scratch, long my_scratch, long peer_flags, long my_flags,
                            long seq_ctrs, long slot_stride16,
                            long x, long residual, long wgain, long out_bf16, long q, long s,
                            int M, int K, int had, float eps, float wadd,
                            long stream, int nblocks, int nthreads, int drain, int acq,
                            int wire = 0);
int  r3dh_quant_act_fp8_group(void);

}  /* extern "C" */
