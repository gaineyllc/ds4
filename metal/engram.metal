#include <metal_stdlib>
using namespace metal;

/* Fused Engram gather-gate (spec 3.C).
 *
 * Reads FP8 rows straight out of the file-backed, no-copy Engram mapping,
 * dequantizes with the row's block scales, applies the scalar gate, and
 * accumulates into the residual in place. No staging buffer, no intermediate
 * device allocation, no host round trip between the gather and the gate.
 *
 * Row layout, 264 B, matching ds4_engram.h:
 *   [0   .. 255]  256 x E4M3 values
 *   [256 .. 263]  8 x E8M0 block scales, one per 32-value block
 *
 * Addressing: the host passes one absolute GPU address per gathered row. A
 * table is ~94 GiB and cannot be a single Metal buffer, so the host resolves
 * the windowed view and all boundary arithmetic; by the time a row id reaches
 * the GPU it is a flat pointer. A zero address means DS4_ENGRAM_DEAD and the
 * row contributes nothing. */

constant uint ENGRAM_DIM = 256;
constant uint ENGRAM_BLK = 32;   /* values per block scale */

struct engram_gather_params {
    uint tokens;   /* residual rows to accumulate into */
    uint cols;     /* rows gathered per token (DS4_ENGRAM_COLS) */
    uint dim;      /* ENGRAM_DIM */
    uint stride;   /* residual row stride in floats */
};

/* E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa. */
static inline float engram_e4m3_to_float(uchar b) {
    const uint s = (uint)(b >> 7) & 1u;
    const uint e = ((uint)b >> 3) & 0xFu;
    const uint m = (uint)b & 0x7u;
    float v;
    if (e == 0u) {
        v = ldexp((float)m, -9);              /* subnormal: m * 2^-9 */
    } else {
        v = ldexp((float)(m + 8u), (int)e - 10); /* (1 + m/8) * 2^(e-7) */
    }
    return s ? -v : v;
}

/* E8M0: the byte is the exponent, value is 2^(b-127). There is no zero case --
 * ds4_engram.c applies ldexpf(v, scale - 127) unconditionally, and treats only
 * scale == 255 as an error, so a 0 byte means 2^-127, not 0. */
static inline float engram_e8m0_to_float(uchar b) {
    return ldexp(1.0f, (int)b - 127);
}

/* Bit-exact port of ds4_engram.c's engram_decode_row.
 *
 * The CPU reference does NOT stop at the dequantized float: it rounds the
 * result to BF16 (round-to-nearest-even in integer bit space) and stores that
 * in an F32. Skipping this rounds nothing and produces values that differ from
 * the pread path in the low 16 mantissa bits, which shows up as a diverged
 * token a few hundred steps into a greedy decode. Keep the exponent arithmetic
 * folded into one ldexp so no intermediate rounds, and mirror the CPU's order:
 * negate, then round. */
static inline float engram_decode(uchar code, uchar scale) {
    const uint s = (uint)(code >> 7) & 1u;
    const uint e = ((uint)code >> 3) & 0xFu;
    const uint m = (uint)code & 0x7u;

    const float mant = e ? (float)(m + 8u) : (float)m;
    const int   exp2 = (e ? (int)e - 10 : -9) + (int)scale - 127;
    float v = ldexp(mant, exp2);
    if (s) v = -v;

    uint bits = as_type<uint>(v);
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    return as_type<float>(bits);
}

static inline float engram_fast_sigmoid(float x) {
    return 1.0f / (1.0f + fast::exp(-x));
}

/* Gather + dequantize only, no gate: the drop-in for the CPU pread +
 * dequantize + tensor_write path at the Engram layers. ds4 applies the gate
 * downstream inside the graph layer, so folding it in here would change where
 * the math happens and break parity. Output is [cols][dim] floats, exactly the
 * layout ds4_engram_read produces.
 *
 * One threadgroup per gathered row, one thread per dimension. */
kernel void kernel_engram_gather_rows(
        device float                  *out         [[buffer(0)]],
        device const ulong            *row_addr    [[buffer(1)]],
        constant engram_gather_params &p           [[buffer(3)]],
        uint                           tid         [[thread_position_in_threadgroup]],
        uint                           tg          [[threadgroup_position_in_grid]])
{
    const uint c = tg;
    if (c >= p.cols || tid >= p.dim) return;

    float v = 0.0f;
    const ulong addr = row_addr[c];
    if (addr != 0ul) {   /* 0 never reaches here: the host fails the gather */
        device const uchar *row = (device const uchar *)addr;
        v = engram_decode(row[tid], row[ENGRAM_DIM + tid / ENGRAM_BLK]);
    }
    out[c * p.dim + tid] = v;
}

/* One threadgroup per token, one thread per output dimension.
 *
 * Thread d reads byte d of each of the token's rows, so reads across a row are
 * coalesced and each row's page is touched once by the whole threadgroup
 * rather than once per lane -- which matters more than the arithmetic here,
 * because these are file-backed pages. */
kernel void kernel_engram_fused_gather_gate(
        device float                  *residual    [[buffer(0)]],
        device const ulong            *row_addr    [[buffer(1)]],
        device const float            *gate_logits [[buffer(2)]],
        constant engram_gather_params &p           [[buffer(3)]],
        uint                           tid         [[thread_position_in_threadgroup]],
        uint                           tg          [[threadgroup_position_in_grid]])
{
    const uint token = tg;
    if (token >= p.tokens || tid >= p.dim) return;

    const uint blk = tid / ENGRAM_BLK;
    const uint base = token * p.cols;
    float acc = 0.0f;

    for (uint c = 0; c < p.cols; ++c) {
        const ulong addr = row_addr[base + c];
        if (addr == 0ul) continue;   /* DS4_ENGRAM_DEAD */

        device const uchar *row = (device const uchar *)addr;
        const float gate = engram_fast_sigmoid(gate_logits[base + c]);
        acc = fma(gate, engram_decode(row[tid], row[ENGRAM_DIM + blk]), acc);
    }

    residual[token * p.stride + tid] += acc;
}

/* Same math, store instead of accumulate: lets the fused path be diffed
 * against the CPU pread path without a read-modify-write in the way. */
kernel void kernel_engram_fused_gather_gate_store(
        device float                  *out         [[buffer(0)]],
        device const ulong            *row_addr    [[buffer(1)]],
        device const float            *gate_logits [[buffer(2)]],
        constant engram_gather_params &p           [[buffer(3)]],
        uint                           tid         [[thread_position_in_threadgroup]],
        uint                           tg          [[threadgroup_position_in_grid]])
{
    const uint token = tg;
    if (token >= p.tokens || tid >= p.dim) return;

    const uint blk = tid / ENGRAM_BLK;
    const uint base = token * p.cols;
    float acc = 0.0f;

    for (uint c = 0; c < p.cols; ++c) {
        const ulong addr = row_addr[base + c];
        if (addr == 0ul) continue;

        device const uchar *row = (device const uchar *)addr;
        const float gate = engram_fast_sigmoid(gate_logits[base + c]);
        acc = fma(gate, engram_decode(row[tid], row[ENGRAM_DIM + blk]), acc);
    }

    out[token * p.stride + tid] = acc;
}
