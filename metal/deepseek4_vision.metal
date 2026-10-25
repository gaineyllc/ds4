// DeepSeek V4 Flash Vision-Exp operations not covered by shared BF16 matmuls.

struct deepseek4_vision_rows_args {
    uint width;
    uint rows;
};

struct deepseek4_vision_qkv_args {
    uint rows;
    uint grid_width;
};

struct deepseek4_vision_align_args {
    uint grid_height;
    uint grid_width;
    uint output_rows;
};

static inline float deepseek4_vision_round_bf16(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return value;
    bits += 0x00007fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float deepseek4_vision_erf(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float a = abs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float p = (((((1.061405429f * t - 1.453152027f) * t) +
                       1.421413741f) * t - 0.284496736f) * t +
                       0.254829592f) * t;
    return sign * (1.0f - p * exp(-a * a));
}

kernel void kernel_deepseek4_vision_qkv_rope(
        constant deepseek4_vision_qkv_args &args,
        device const float                 *qkv,
        device const ushort                *bias,
        device float                       *q,
        device float                       *k,
        device float                       *v,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint head = group.y;
    if (row >= args.rows || head >= 16u) return;
    const ulong qkv_base = (ulong)row * 3072u + (ulong)head * 64u;
    const ulong out_base = (ulong)row * 1024u + (ulong)head * 64u;
    const float q0 = deepseek4_vision_round_bf16(
        qkv[qkv_base + lane] +
        glm53_bf16_to_f32(bias[(ulong)head * 64u + lane]));
    const float q1 = deepseek4_vision_round_bf16(
        qkv[qkv_base + lane + 32u] +
        glm53_bf16_to_f32(bias[(ulong)head * 64u + lane + 32u]));
    const float k0 = deepseek4_vision_round_bf16(
        qkv[qkv_base + 1024u + lane] +
        glm53_bf16_to_f32(bias[1024u + (ulong)head * 64u + lane]));
    const float k1 = deepseek4_vision_round_bf16(
        qkv[qkv_base + 1024u + lane + 32u] +
        glm53_bf16_to_f32(bias[1024u + (ulong)head * 64u + lane + 32u]));
    const uint y = row / args.grid_width;
    const uint x = row - y * args.grid_width;
    const uint pos = lane < 16u ? y : x;
    const uint freq_index = lane & 15u;
    const float inv_freq = powr(10000.0f, -(float)freq_index / 16.0f);
    const float angle = (float)pos * inv_freq;
    const float cs = cos(angle);
    const float sn = sin(angle);
    q[out_base + lane] = deepseek4_vision_round_bf16(q0 * cs - q1 * sn);
    q[out_base + lane + 32u] =
        deepseek4_vision_round_bf16(q1 * cs + q0 * sn);
    k[out_base + lane] = deepseek4_vision_round_bf16(k0 * cs - k1 * sn);
    k[out_base + lane + 32u] =
        deepseek4_vision_round_bf16(k1 * cs + k0 * sn);
    v[out_base + lane] = deepseek4_vision_round_bf16(
        qkv[qkv_base + 2048u + lane] +
        glm53_bf16_to_f32(bias[2048u + (ulong)head * 64u + lane]));
    v[out_base + lane + 32u] = deepseek4_vision_round_bf16(
        qkv[qkv_base + 2048u + lane + 32u] +
        glm53_bf16_to_f32(
            bias[2048u + (ulong)head * 64u + lane + 32u]));
}

kernel void kernel_deepseek4_vision_round_bf16(
        constant deepseek4_vision_rows_args &args,
        device float                       *x,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    x[off] = deepseek4_vision_round_bf16(x[off]);
}

kernel void kernel_deepseek4_vision_add_residual(
        constant deepseek4_vision_rows_args &args,
        device float                       *x,
        device const float                 *residual,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    x[off] = deepseek4_vision_round_bf16(x[off] + residual[off]);
}

kernel void kernel_deepseek41_vision_bias_residual(
        constant deepseek4_vision_rows_args &args,
        device float                       *x,
        device const ushort                *bias,
        device const float                 *residual,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    const float projected = deepseek4_vision_round_bf16(
        x[off] + glm53_bf16_to_f32(bias[gid.x]));
    x[off] = deepseek4_vision_round_bf16(projected + residual[off]);
}

static inline void deepseek4_vision_swiglu_split(
        constant deepseek4_vision_rows_args &args,
        device const float                 *gate_up,
        device float                       *out,
        uint2 gid,
        bool round_silu) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong source = (ulong)gid.y * args.width * 2u + gid.x;
    const float gate = gate_up[source];
    const float up = gate_up[source + args.width];
    float activated = gate / (1.0f + exp(-gate));
    if (round_silu) activated = deepseek4_vision_round_bf16(activated);
    out[(ulong)gid.y * args.width + gid.x] = deepseek4_vision_round_bf16(activated * up);
}

kernel void kernel_deepseek4_vision_swiglu_split(
        constant deepseek4_vision_rows_args &args,
        device const float                 *gate_up,
        device float                       *out,
        uint2 gid [[thread_position_in_grid]]) {
    deepseek4_vision_swiglu_split(args, gate_up, out, gid, false);
}

kernel void kernel_deepseek41_vision_swiglu_split(
        constant deepseek4_vision_rows_args &args,
        device const float                 *gate_up,
        device float                       *out,
        uint2 gid [[thread_position_in_grid]]) {
    deepseek4_vision_swiglu_split(args, gate_up, out, gid, true);
}

/* Match F.unfold(x, 3, stride=3): channels are outermost, followed by the
 * row-major 3x3 neighborhood. Values outside the ViT grid are zero. */
kernel void kernel_deepseek4_vision_aligner_reorder(
        constant deepseek4_vision_align_args &args,
        device const float                  *x,
        device float                        *out,
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint row = gid.y;
    if (d >= 9216u || row >= args.output_rows) return;
    const uint output_width = (args.grid_width + 2u) / 3u;
    const uint block_y = row / output_width;
    const uint block_x = row - block_y * output_width;
    const uint channel = d / 9u;
    const uint within = d - channel * 9u;
    const uint source_y = block_y * 3u + within / 3u;
    const uint source_x = block_x * 3u + within % 3u;
    float value = 0.0f;
    if (source_y < args.grid_height && source_x < args.grid_width) {
        const ulong source_row = (ulong)source_y * args.grid_width + source_x;
        value = x[source_row * 1024u + channel];
    }
    out[(ulong)row * 9216u + d] = value;
}

kernel void kernel_deepseek4_vision_gelu_bias(
        constant deepseek4_vision_rows_args &args,
        device const float                 *x,
        device const ushort                *bias,
        device float                       *out,
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.width || gid.y >= args.rows) return;
    const ulong off = (ulong)gid.y * args.width + gid.x;
    const float value = deepseek4_vision_round_bf16(
        x[off] + glm53_bf16_to_f32(bias[gid.x]));
    out[off] = deepseek4_vision_round_bf16(
        0.5f * value *
        (1.0f + deepseek4_vision_erf(value * 0.7071067811865475f)));
}

// ---------------------------------------------------------------------------
// Tiled flash attention for the vision ViT: 16 heads x 64 dims, rows laid out as
// [row][head*64 + d] (1024 floats per row), full bidirectional attention over the
// image. One threadgroup = 32 query rows of one head (4 simdgroups x 8 rows); keys
// and values stream through threadgroup memory 32 at a time; QK^T and PV use 8x8
// float simdgroup matrices; online softmax per row; the running output is rescaled
// with a diagonal-matrix multiply. fp32 throughout (matches the scalar kernel's
// numerics to rounding). Replaces O(N^2) device-memory streaming with tiled reuse.
// ---------------------------------------------------------------------------
struct vision_flash_attn_args {
    uint rows;
    float scale;
};

#define VFA_D   64u
#define VFA_BR  32u
#define VFA_BC  32u
#define VFA_ROW 1024u

kernel void kernel_vision_flash_attn_d64(
        constant vision_flash_attn_args &args,
        device const float *q,
        device const float *k,
        device const float *v,
        device float       *out,
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sQ[VFA_BR * VFA_D];        // 8 KB, later reused as the output stage
    threadgroup float sK[VFA_BC * VFA_D];        // 8 KB
    threadgroup float sV[VFA_BC * VFA_D];        // 8 KB
    threadgroup float sS[4 * 8 * VFA_BC];        // 4 KB: per-simdgroup scores / probs
    threadgroup float sDiag[4 * 64];             // 1 KB: per-simdgroup 8x8 diagonal

    const uint rows = args.rows;
    const uint head = tg.y;
    const uint q0 = tg.x * VFA_BR;
    const uint hoff = head * VFA_D;

    for (uint i = tid; i < VFA_BR * VFA_D; i += 128u) {
        const uint r = q0 + i / VFA_D;
        sQ[i] = r < rows ? q[(ulong)r * VFA_ROW + hoff + (i % VFA_D)] * args.scale : 0.0f;
    }

    simdgroup_float8x8 O[8];
    for (uint j = 0; j < 8u; j++) O[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    threadgroup float *mS = sS + sg * (8u * VFA_BC);
    threadgroup float *mD = sDiag + sg * 64u;
    threadgroup const float *mQ = sQ + sg * 8u * VFA_D;

    const uint my_row = lane >> 2;          // 0..7: the row this lane tracks
    const uint my_c0  = (lane & 3u) * 8u;   // 8 of the tile's 32 columns
    float m_run = -INFINITY, l_run = 0.0f;

    for (uint kb = 0; kb < rows; kb += VFA_BC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < VFA_BC * VFA_D; i += 128u) {
            const uint r = kb + i / VFA_D;
            const ulong off = (ulong)r * VFA_ROW + hoff + (i % VFA_D);
            sK[i] = r < rows ? k[off] : 0.0f;
            sV[i] = r < rows ? v[off] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S (8 x 32) = Q_sg (8 x 64) * K_tile^T (64 x 32)
        simdgroup_float8x8 S[4];
        for (uint j = 0; j < 4u; j++) S[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d = 0; d < 8u; d++) {
            simdgroup_float8x8 qa;
            simdgroup_load(qa, mQ + d * 8u, VFA_D);
            for (uint j = 0; j < 4u; j++) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, sK + (j * 8u) * VFA_D + d * 8u, VFA_D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[j], qa, kt, S[j]);
            }
        }
        for (uint j = 0; j < 4u; j++) simdgroup_store(S[j], mS + j * 8u, VFA_BC);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax for this lane's row segment.
        float sv[8];
        float mx = -INFINITY;
        for (uint t = 0; t < 8u; t++) {
            const uint col = my_c0 + t;
            const float s = (kb + col) < rows ? mS[my_row * VFA_BC + col] : -INFINITY;
            sv[t] = s;
            mx = max(mx, s);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float m_new = max(m_run, mx);
        const float alpha = m_run == -INFINITY ? 0.0f : exp(m_run - m_new);
        float psum = 0.0f;
        for (uint t = 0; t < 8u; t++) {
            const float p = sv[t] == -INFINITY ? 0.0f : exp(sv[t] - m_new);
            mS[my_row * VFA_BC + my_c0 + t] = p;
            psum += p;
        }
        psum += simd_shuffle_xor(psum, 1);
        psum += simd_shuffle_xor(psum, 2);
        l_run = l_run * alpha + psum;
        m_run = m_new;

        // diag(alpha) for the 8 rows of this simdgroup
        for (uint e = lane * 2u; e < lane * 2u + 2u; e++) {
            const uint rr = e >> 3, cc = e & 7u;
            const float a = simd_shuffle(alpha, (ushort)(rr * 4u));
            mD[e] = rr == cc ? a : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 dg;
        simdgroup_load(dg, mD, 8u);
        for (uint j = 0; j < 8u; j++) {
            simdgroup_float8x8 t;
            simdgroup_multiply(t, dg, O[j]);
            O[j] = t;
        }
        // O += P (8 x 32) * V_tile (32 x 64)
        for (uint kk = 0; kk < 4u; kk++) {
            simdgroup_float8x8 pa;
            simdgroup_load(pa, mS + kk * 8u, VFA_BC);
            for (uint j = 0; j < 8u; j++) {
                simdgroup_float8x8 vb;
                simdgroup_load(vb, sV + (kk * 8u) * VFA_D + j * 8u, VFA_D);
                simdgroup_multiply_accumulate(O[j], pa, vb, O[j]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O /= l, staged through this simdgroup's own (now unused) Q rows.
    const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
    for (uint e = lane * 2u; e < lane * 2u + 2u; e++) {
        const uint rr = e >> 3, cc = e & 7u;
        const float a = simd_shuffle(inv_l, (ushort)(rr * 4u));
        mD[e] = rr == cc ? a : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dg;
    simdgroup_load(dg, mD, 8u);
    threadgroup float *stage = sQ + sg * 8u * VFA_D;
    for (uint j = 0; j < 8u; j++) {
        simdgroup_float8x8 t;
        simdgroup_multiply(t, dg, O[j]);
        simdgroup_store(t, stage + j * 8u, VFA_D);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 8u * VFA_D; e += 32u) {
        const uint r = q0 + sg * 8u + e / VFA_D;
        if (r < rows) out[(ulong)r * VFA_ROW + hoff + (e % VFA_D)] = stage[e];
    }
}

// v2: 64 query rows per threadgroup (8 simdgroups), Q held in registers, so each
// streamed K/V tile is reused by twice as many queries (half the tile traffic of v1).
#define VFA2_BR 64u

kernel void kernel_vision_flash_attn_d64_br64(
        constant vision_flash_attn_args &args,
        device const float *q,
        device const float *k,
        device const float *v,
        device float       *out,
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sK[VFA_BC * VFA_D];        // 8 KB (also Q / output staging)
    threadgroup float sV[VFA_BC * VFA_D];        // 8 KB
    threadgroup float sS[8 * 8 * VFA_BC];        // 8 KB
    threadgroup float sDiag[8 * 64];             // 2 KB

    const uint rows = args.rows;
    const uint head = tg.y;
    const uint q0 = tg.x * VFA2_BR;
    const uint hoff = head * VFA_D;
    const uint nthreads = 256u;

    // Q (64 x 64) -> registers, staged 32 rows at a time through sK/sV.
    simdgroup_float8x8 QA[8];
    threadgroup float *stage = sg < 4u ? sK : sV;
    for (uint i = tid; i < 32u * VFA_D; i += nthreads) {
        const uint r0 = q0 + i / VFA_D, r1 = r0 + 32u;
        sK[i] = r0 < rows ? q[(ulong)r0 * VFA_ROW + hoff + (i % VFA_D)] * args.scale : 0.0f;
        sV[i] = r1 < rows ? q[(ulong)r1 * VFA_ROW + hoff + (i % VFA_D)] * args.scale : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 0; d < 8u; d++)
        simdgroup_load(QA[d], stage + (sg & 3u) * 8u * VFA_D + d * 8u, VFA_D);

    simdgroup_float8x8 O[8];
    for (uint j = 0; j < 8u; j++) O[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    threadgroup float *mS = sS + sg * (8u * VFA_BC);
    threadgroup float *mD = sDiag + sg * 64u;
    const uint my_row = lane >> 2;
    const uint my_c0  = (lane & 3u) * 8u;
    float m_run = -INFINITY, l_run = 0.0f;

    for (uint kb = 0; kb < rows; kb += VFA_BC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < VFA_BC * VFA_D; i += nthreads) {
            const uint r = kb + i / VFA_D;
            const ulong off = (ulong)r * VFA_ROW + hoff + (i % VFA_D);
            sK[i] = r < rows ? k[off] : 0.0f;
            sV[i] = r < rows ? v[off] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 S[4];
        for (uint j = 0; j < 4u; j++) S[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d = 0; d < 8u; d++) {
            for (uint j = 0; j < 4u; j++) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, sK + (j * 8u) * VFA_D + d * 8u, VFA_D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[j], QA[d], kt, S[j]);
            }
        }
        for (uint j = 0; j < 4u; j++) simdgroup_store(S[j], mS + j * 8u, VFA_BC);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float sv[8];
        float mx = -INFINITY;
        for (uint t = 0; t < 8u; t++) {
            const uint col = my_c0 + t;
            const float s = (kb + col) < rows ? mS[my_row * VFA_BC + col] : -INFINITY;
            sv[t] = s;
            mx = max(mx, s);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float m_new = max(m_run, mx);
        const float alpha = m_run == -INFINITY ? 0.0f : exp(m_run - m_new);
        float psum = 0.0f;
        for (uint t = 0; t < 8u; t++) {
            const float p = sv[t] == -INFINITY ? 0.0f : exp(sv[t] - m_new);
            mS[my_row * VFA_BC + my_c0 + t] = p;
            psum += p;
        }
        psum += simd_shuffle_xor(psum, 1);
        psum += simd_shuffle_xor(psum, 2);
        l_run = l_run * alpha + psum;
        m_run = m_new;
        for (uint e = lane * 2u; e < lane * 2u + 2u; e++) {
            const uint rr = e >> 3, cc = e & 7u;
            const float a = simd_shuffle(alpha, (ushort)(rr * 4u));
            mD[e] = rr == cc ? a : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 dg;
        simdgroup_load(dg, mD, 8u);
        for (uint j = 0; j < 8u; j++) {
            simdgroup_float8x8 t;
            simdgroup_multiply(t, dg, O[j]);
            O[j] = t;
        }
        for (uint kk = 0; kk < 4u; kk++) {
            simdgroup_float8x8 pa;
            simdgroup_load(pa, mS + kk * 8u, VFA_BC);
            for (uint j = 0; j < 8u; j++) {
                simdgroup_float8x8 vb;
                simdgroup_load(vb, sV + (kk * 8u) * VFA_D + j * 8u, VFA_D);
                simdgroup_multiply_accumulate(O[j], pa, vb, O[j]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
    for (uint e = lane * 2u; e < lane * 2u + 2u; e++) {
        const uint rr = e >> 3, cc = e & 7u;
        const float a = simd_shuffle(inv_l, (ushort)(rr * 4u));
        mD[e] = rr == cc ? a : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);   // everyone is done with sK/sV
    simdgroup_float8x8 dg;
    simdgroup_load(dg, mD, 8u);
    threadgroup float *ostage = (sg < 4u ? sK : sV) + (sg & 3u) * 8u * VFA_D;
    for (uint j = 0; j < 8u; j++) {
        simdgroup_float8x8 t;
        simdgroup_multiply(t, dg, O[j]);
        simdgroup_store(t, ostage + j * 8u, VFA_D);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lane; e < 8u * VFA_D; e += 32u) {
        const uint r = q0 + sg * 8u + e / VFA_D;
        if (r < rows) out[(ulong)r * VFA_ROW + hoff + (e % VFA_D)] = ostage[e];
    }
}

// v3: 128 query rows per threadgroup (16 simdgroups, 512 threads), Q in registers, K/V
// tiles stored as half in threadgroup memory and multiplied half x float -> float (the
// activations are bf16-rounded, which half represents exactly for |x| >= 6e-5). Half the
// tile traffic of v2 again.
#define VFA3_BR 128u

kernel void kernel_vision_flash_attn_d64_br128(
        constant vision_flash_attn_args &args,
        device const float *q,
        device const float *k,
        device const float *v,
        device float       *out,
        uint2  tg   [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup half  hK[VFA_BC * VFA_D];        // 4 KB
    threadgroup half  hV[VFA_BC * VFA_D];        // 4 KB
    threadgroup float sS[16 * 8 * VFA_BC];       // 16 KB (also Q / output staging)
    threadgroup float sDiag[16 * 64];            // 4 KB

    const uint rows = args.rows;
    const uint head = tg.y;
    const uint q0 = tg.x * VFA3_BR;
    const uint hoff = head * VFA_D;
    const uint nthreads = 512u;

    // Q (128 x 64) -> registers, staged 64 rows at a time through sS (64*64 floats = 16 KB).
    simdgroup_float8x8 QA[8];
    for (uint half_blk = 0; half_blk < 2u; half_blk++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 64u * VFA_D; i += nthreads) {
            const uint r = q0 + half_blk * 64u + i / VFA_D;
            sS[i] = r < rows ? q[(ulong)r * VFA_ROW + hoff + (i % VFA_D)] * args.scale : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((sg >> 3) == half_blk)
            for (uint d = 0; d < 8u; d++)
                simdgroup_load(QA[d], sS + (sg & 7u) * 8u * VFA_D + d * 8u, VFA_D);
    }

    simdgroup_float8x8 O[8];
    for (uint j = 0; j < 8u; j++) O[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    threadgroup float *mS = sS + sg * (8u * VFA_BC);
    threadgroup float *mD = sDiag + sg * 64u;
    const uint my_row = lane >> 2;
    const uint my_c0  = (lane & 3u) * 8u;
    float m_run = -INFINITY, l_run = 0.0f;

    for (uint kb = 0; kb < rows; kb += VFA_BC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < VFA_BC * VFA_D; i += nthreads) {
            const uint r = kb + i / VFA_D;
            const ulong off = (ulong)r * VFA_ROW + hoff + (i % VFA_D);
            hK[i] = r < rows ? (half)k[off] : (half)0.0f;
            hV[i] = r < rows ? (half)v[off] : (half)0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 S[4];
        for (uint j = 0; j < 4u; j++) S[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint d = 0; d < 8u; d++) {
            for (uint j = 0; j < 4u; j++) {
                simdgroup_half8x8 kt;
                simdgroup_load(kt, hK + (j * 8u) * VFA_D + d * 8u, VFA_D, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S[j], QA[d], kt, S[j]);
            }
        }
        for (uint j = 0; j < 4u; j++) simdgroup_store(S[j], mS + j * 8u, VFA_BC);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float sv[8];
        float mx = -INFINITY;
        for (uint t = 0; t < 8u; t++) {
            const uint col = my_c0 + t;
            const float s = (kb + col) < rows ? mS[my_row * VFA_BC + col] : -INFINITY;
            sv[t] = s;
            mx = max(mx, s);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float m_new = max(m_run, mx);
        const float alpha = m_run == -INFINITY ? 0.0f : exp(m_run - m_new);
        float psum = 0.0f;
        for (uint t = 0; t < 8u; t++) {
            const float p = sv[t] == -INFINITY ? 0.0f : exp(sv[t] - m_new);
            mS[my_row * VFA_BC + my_c0 + t] = p;
            psum += p;
        }
        psum += simd_shuffle_xor(psum, 1);
        psum += simd_shuffle_xor(psum, 2);
        l_run = l_run * alpha + psum;
        m_run = m_new;
        for (uint e = lane * 2u; e < lane * 2u + 2u; e++) {
            const uint rr = e >> 3, cc = e & 7u;
            const float a = simd_shuffle(alpha, (ushort)(rr * 4u));
            mD[e] = rr == cc ? a : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 dg;
        simdgroup_load(dg, mD, 8u);
        for (uint j = 0; j < 8u; j++) {
            simdgroup_float8x8 t;
            simdgroup_multiply(t, dg, O[j]);
            O[j] = t;
        }
        for (uint kk = 0; kk < 4u; kk++) {
            simdgroup_float8x8 pa;
            simdgroup_load(pa, mS + kk * 8u, VFA_BC);
            for (uint j = 0; j < 8u; j++) {
                simdgroup_half8x8 vb;
                simdgroup_load(vb, hV + (kk * 8u) * VFA_D + j * 8u, VFA_D);
                simdgroup_multiply_accumulate(O[j], pa, vb, O[j]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_l = l_run > 0.0f ? 1.0f / l_run : 0.0f;
    for (uint e = lane * 2u; e < lane * 2u + 2u; e++) {
        const uint rr = e >> 3, cc = e & 7u;
        const float a = simd_shuffle(inv_l, (ushort)(rr * 4u));
        mD[e] = rr == cc ? a : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dg;
    simdgroup_load(dg, mD, 8u);
    // Each simdgroup owns 8 x 32 floats of sS (its score scratch); stage the 8 x 64 output
    // in two 32-column halves.
    for (uint hcol = 0; hcol < 2u; hcol++) {
        for (uint j = 0; j < 4u; j++) {
            simdgroup_float8x8 t;
            simdgroup_multiply(t, dg, O[hcol * 4u + j]);
            simdgroup_store(t, mS + j * 8u, VFA_BC);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 8u * VFA_BC; e += 32u) {
            const uint r = q0 + sg * 8u + e / VFA_BC;
            if (r < rows) out[(ulong)r * VFA_ROW + hoff + hcol * 32u + (e % VFA_BC)] = mS[e];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
