/* r3dh_ar_stubs.hip -- the fused all-reduce + quantiser launchers the ported ar entry names but
 * that this library does not carry yet.
 *
 * The collectives landed without their fused junctions: the rows below declare only all_reduce and
 * all_gather, whose kernels are in r3dh_ar_*.hip. The entry's junctions call standalone producers
 * (rmsnorm_quant_fp8, rmsnorm_had_quant_i8g, rmsnorm_quant_i8) that libr3d implements under its own
 * R3d producers with different signatures, so the call sites are unresolved here. These stubs
 * resolve the entry and refuse at runtime; a row that wants a junction must replace them with the
 * real kernels rather than expose these symbols.
 */
#include "r3dh_ar.h"

extern "C" int r3dh_ar_rmsnorm_quant_fp8(
    long, long, long, long, long, long, long, long, long, long, long, long, int, int, float, float,
    int, int, int, int, int, long, int, int, int, int, int, long, long, long) {
    return -1;
}

extern "C" int r3dh_ar_rmsnorm_had_quant_i8g(
    long, long, long, long, long, long, long, long, long, long, long, long, int, int, float, float,
    int, int, int, int, int, long, int, int, int, int, int, long, long) {
    return -1;
}

extern "C" int r3dh_ar_rmsnorm_quant_i8(
    long, long, long, long, long, long, long, long, long, long, long, int, long, long, long, int,
    int, float, float, int, int, int, long, int, int, int) {
    return -1;
}

extern "C" int r3dh_ar_rmsnorm_quant_i8_4rank(
    long, long, long, long, long, long, long, long, long, long, long, long, long, long, long, int,
    long, long, long, int, int, float, float, int, int, int, int, long, int, int, int, int, long,
    long, long, int, int, int) {
    return -1;
}