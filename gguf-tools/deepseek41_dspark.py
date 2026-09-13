#!/usr/bin/env python3
"""Convert the DeepSeek V4.1 Flash DSpark draft module into a support GGUF.

V4.1 ships DSpark inside the main checkpoint under `mtp.0`, `mtp.1` and
`mtp.2`; deepseek41_quantize.py omits that namespace from the text GGUF.
This writes the three stages as a standalone support file using the same
quantization recipe the backbone uses, so `ds4 --dspark --mtp-model FILE`
has something to load.

The draft ties its token embedding and output head to the backbone, so
neither is copied here. Only shards holding `mtp.*` need to be present.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from deepseek41_metadata import GGUF_ALIGNMENT
from deepseek41_quantize import NativeQuantizer, scale_name, write_gguf  # noqa: F401
from glm53_manifest import load_index, load_safetensors_header
from glm53_quantize import (
    QTYPE_F32, QTYPE_F16, QTYPE_BF16, QTYPE_Q8_0, QTYPE_Q2_K, QTYPE_Q4_K,
    QTYPE_IQ2_XXS, SourceDB, TensorPlan, align, conversion_signature, fail,
    kv_string, kv_u32, kv_u32_array, load_resume_state, print_plan,
    qtype_nbytes, save_resume_state, tensor_header,
)

QTYPE_MXFP4 = 39

# The shared quantizer tables list every type the GLM and V4.1 recipes emit;
# MXFP4 is a GGUF type neither of them produces, so register it here rather
# than fork them. 32 values per 17-byte block: one E8M0 scale, then 16 bytes
# of nibble pairs.
import glm53_quantize as _shared
_shared.QTYPE_LAYOUT.setdefault(QTYPE_MXFP4, (32, 17))
_shared.QTYPE_NAMES.setdefault(QTYPE_MXFP4, "mxfp4")

SOURCE_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"
ARCH = "deepseek4-dspark"

# The drafter is small next to the target, and draft quality is what sets the
# accepted-prefix length, so BF16 -- the released FP8/FP4 weights dequantized
# and not requantized -- is the default. The quantized recipes exist for hosts
# that cannot spare the resident bytes.
PRECISION = {
    # The released routed experts are already MXFP4: the same E2M1 value set
    # (0, .5, 1, 1.5, 2, 3, 4, 6 and negatives) under one E8M0 scale per 32
    # elements of a row. Storing them as MXFP4 is a repack, not a
    # quantization -- every weight keeps its exact released value -- and it is
    # a routed-expert type ds4's Metal MoE kernels already execute. Dense
    # tensors are FP8 E4M3, which F16 holds exactly.
    "native": dict(att=QTYPE_F16, shared=QTYPE_F16, exp_gate=QTYPE_MXFP4,
                   exp_up=QTYPE_MXFP4, exp_down=QTYPE_MXFP4, head=QTYPE_F32,
                   hcfn=QTYPE_F32,
                   note="native: MXFP4 experts repacked bit-exact, F16 dense"),
    "bf16": dict(att=QTYPE_BF16, shared=QTYPE_BF16, exp_gate=QTYPE_BF16,
                 exp_up=QTYPE_BF16, exp_down=QTYPE_BF16, head=QTYPE_BF16,
                 hcfn=QTYPE_F32,
                 note="BF16 throughout; released weights are not requantized"),
    "q8": dict(att=QTYPE_Q8_0, shared=QTYPE_Q8_0, exp_gate=QTYPE_Q8_0,
               exp_up=QTYPE_Q8_0, exp_down=QTYPE_Q8_0, head=QTYPE_F16,
               hcfn=QTYPE_F16,
               note="Q8_0 throughout"),
    "q4": dict(att=QTYPE_Q8_0, shared=QTYPE_Q8_0, exp_gate=QTYPE_Q4_K,
               exp_up=QTYPE_Q4_K, exp_down=QTYPE_Q4_K, head=QTYPE_F16,
               hcfn=QTYPE_F16,
               note="Q4_K gate/up/down; Q8_0 attention/shared/projection"),
    "q2": dict(att=QTYPE_Q8_0, shared=QTYPE_Q8_0, exp_gate=QTYPE_IQ2_XXS,
               exp_up=QTYPE_IQ2_XXS, exp_down=QTYPE_Q2_K, head=QTYPE_F16,
               hcfn=QTYPE_F16,
               note="IQ2_XXS gate/up; Q2_K down; Q8_0 attention/shared/projection"),
}


class MtpSourceDB(SourceDB):
    """SourceDB restricted to the mtp.* namespace.

    The stock loader insists every shard in the index is present. DSpark
    lives in three of forty-eight, so the rest are never needed.
    """

    def __init__(self, hf_dir):
        import threading

        self.hf_dir = hf_dir
        document, weight_map = load_index(os.path.join(hf_dir, "model.safetensors.index.json"))
        self.weight_map = {k: v for k, v in weight_map.items() if k.startswith("mtp.")}
        if not self.weight_map:
            fail("index contains no mtp.* tensors; this is not a DSpark checkpoint")
        self.declared_bytes = None
        self.tensors = {}
        self._fds = {}
        self._fd_lock = threading.Lock()
        for shard in sorted(set(self.weight_map.values())):
            path = os.path.join(hf_dir, shard)
            if not os.path.isfile(path):
                fail(f"missing source shard {path} (only mtp shards are required)")
            for name, info in load_safetensors_header(path).items():
                if not name.startswith("mtp."):
                    fail(f"{shard} holds non-DSpark tensor {name}; expected an mtp-only shard")
                if self.weight_map.get(name) != shard:
                    fail(f"index assigns {name} to {self.weight_map.get(name)!r}, not {shard}")
                if name in self.tensors:
                    fail(f"duplicate source tensor {name}")
                self.tensors[name] = dict(info, shard=shard)
        if set(self.tensors) != set(self.weight_map):
            missing = sorted(set(self.weight_map) - set(self.tensors))
            fail(f"source headers are incomplete; first missing tensor is {missing[0]}")
        validate_scales(self.tensors)


def validate_scales(tensors):
    for name, info in tensors.items():
        dtype, shape = info["dtype"], info["shape"]
        if dtype not in ("F8_E4M3", "I8") or not name.endswith(".weight"):
            continue
        if len(shape) != 2:
            raise ValueError(f"{name}: expected a matrix")
        expected = [shape[0], shape[1] // 16] if dtype == "I8" else [(d + 31) // 32 for d in shape]
        scale = tensors.get(scale_name(name))
        if not scale or scale["dtype"] != "F8_E8M0" or scale["shape"] != expected:
            raise ValueError(f"{name}: expected E8M0 scales {expected}")


def dspark_config(hf_dir):
    config = json.loads((Path(hf_dir) / "config.json").read_text())
    if config.get("model_type") != "deepseek_v41":
        raise ValueError("expected the DeepSeek V4.1 Flash checkpoint")
    c = config["text_config"]
    if config["quantization_config"]["weight_block_size"] != [32, 32]:
        raise ValueError("expected native 32x32 FP8 blocks")
    for key in ("num_nextn_predict_layers", "dspark_block_size", "dspark_markov_rank",
                "dspark_noise_token_id", "dspark_target_layer_ids",
                "dspark_n_routed_experts", "dspark_num_experts_per_tok"):
        if key not in c:
            raise ValueError(f"config.json has no {key}; this checkpoint has no DSpark module")
    targets = list(c["dspark_target_layer_ids"])
    if not targets or sorted(targets) != targets or len(set(targets)) != len(targets):
        raise ValueError("dspark_target_layer_ids must be strictly increasing")
    if targets[-1] >= c["num_hidden_layers"]:
        raise ValueError("a DSpark target layer lies outside the backbone")
    return config, c


def build_plan(db, c, quant="bf16"):
    if quant not in PRECISION:
        raise ValueError(f"unknown precision recipe: {quant}")
    qt_of = PRECISION[quant]
    dim, inter = c["hidden_size"], c["moe_intermediate_size"]
    heads, hd = c["num_attention_heads"], c["head_dim"]
    qrank, orank, groups = c["q_lora_rank"], c["o_lora_rank"], c["o_groups"]
    hc, vocab = c["hc_mult"], c["vocab_size"]
    stages, experts = c["num_nextn_predict_layers"], c["dspark_n_routed_experts"]
    rank, targets = c["dspark_markov_rank"], list(c["dspark_target_layer_ids"])
    plan, consumed = [], set()

    def claim(name, expected, dtype=None):
        info = db.info(name)
        if info["shape"] != list(expected) or (dtype and info["dtype"] != dtype):
            raise ValueError(f"{name}: unexpected {info['dtype']} {info['shape']}, "
                             f"expected {dtype} {expected}")
        consumed.add(name)
        if info["dtype"] in ("I8", "F8_E4M3"):
            consumed.add(scale_name(name))
        return info

    def regular(name, source, shape, qtype, role):
        claim(source, shape)
        plan.append(TensorPlan(name, tuple(reversed(shape)), qtype, role, source=source))

    for stage in range(stages):
        src, dst = f"mtp.{stage}", f"mtp.{stage}"
        for site in ("attn", "ffn"):
            for part, shape, qt in (("fn", (hc * (hc + 2), hc * dim), qt_of["hcfn"]),
                                    ("base", (hc * (hc + 2),), QTYPE_F32),
                                    ("scale", (3,), QTYPE_F32)):
                regular(f"{dst}.hc_{site}_{part}.weight", f"{src}.hc_{site}_{part}", shape, qt, "mhc")
            regular(f"{dst}.{site}_norm.weight", f"{src}.{site}_norm.weight", (dim,), QTYPE_F32, "norm")
        regular(f"{dst}.attn_sinks.weight", f"{src}.attn.attn_sink", (heads,), QTYPE_F32, "attention")
        for target, source, shape, qt in (
            ("attn_q_a.weight", "wq_a.weight", (qrank, dim), qt_of["att"]),
            ("attn_q_a_norm.weight", "q_norm.weight", (qrank,), QTYPE_F32),
            ("attn_q_b.weight", "wq_b.weight", (heads * hd, qrank), qt_of["att"]),
            ("attn_kv.weight", "wkv.weight", (hd, dim), qt_of["att"]),
            ("attn_kv_a_norm.weight", "kv_norm.weight", (hd,), QTYPE_F32),
            ("attn_output_a.weight", "wo_a.weight", (groups * orank, heads * hd // groups), qt_of["att"]),
            ("attn_output_b.weight", "wo_b.weight", (dim, groups * orank), qt_of["att"]),
        ):
            regular(f"{dst}.{target}", f"{src}.attn.{source}", shape, qt, "attention")
        regular(f"{dst}.ffn_gate_inp.weight", f"{src}.ffn.gate.weight", (experts, dim), QTYPE_F32, "router")
        for suffix, target in (("bias", "exp_probs_b.bias"), ("bias_vl", "exp_probs_b_vl.bias")):
            regular(f"{dst}.{target}", f"{src}.ffn.gate.{suffix}", (experts,), QTYPE_F32, "router")
        for part, source, shape, qt in (("gate", "w1", (inter, dim), qt_of["exp_gate"]),
                                        ("up", "w3", (inter, dim), qt_of["exp_up"]),
                                        ("down", "w2", (dim, inter), qt_of["exp_down"])):
            regular(f"{dst}.ffn_{part}_shexp.weight", f"{src}.ffn.shared_experts.{source}.weight",
                    shape, qt_of["shared"], "shared")
            pattern = f"{src}.ffn.experts.{{expert}}.{source}.weight"
            for expert in range(experts):
                claim(pattern.format(expert=expert), (shape[0], shape[1] // 2), "I8")
            plan.append(TensorPlan(f"{dst}.ffn_{part}_exps.weight", (*reversed(shape), experts),
                                   qt,
                                   "experts", source=pattern, expert_layer=stage,
                                   expert_part=part, expert_count=experts,
                                   transform="mxfp4" if qt == QTYPE_MXFP4 else None))
        if stage == 0:
            regular(f"{dst}.main_proj.weight", f"{src}.main_proj.weight",
                    (dim, dim * len(targets)), qt_of["att"], "dspark_head")
            regular(f"{dst}.main_norm.weight", f"{src}.main_norm.weight", (dim,), QTYPE_F32, "norm")
        if stage == stages - 1:
            regular(f"{dst}.norm.weight", f"{src}.norm.weight", (dim,), QTYPE_F32, "norm")
            regular(f"{dst}.markov_head.markov_w1.weight", f"{src}.markov_head.embed.weight",
                    (vocab, rank), qt_of["head"], "dspark_head")
            regular(f"{dst}.markov_head.markov_w2.weight", f"{src}.markov_head.head.weight",
                    (vocab, rank), qt_of["head"], "dspark_head")
            regular(f"{dst}.confidence_head.proj.weight", f"{src}.confidence_head.proj.weight",
                    (1, dim + rank), QTYPE_F32, "dspark_head")

    if consumed != set(db.tensors):
        extra = sorted(set(db.tensors) - consumed)[:10]
        raise ValueError(f"unclaimed source tensors: {extra}")
    offset = 0
    for item in plan:
        item.offset = offset
        item.nbytes = qtype_nbytes(item.qtype, item.shape)
        offset += align(item.nbytes, GGUF_ALIGNMENT)
    return plan


def mxfp4_from_native(np, codes, scales):
    """Repack one released FP4 expert row-block into ds4's MXFP4 blocks.

    Source: one byte per weight pair, element 2i in the low nibble and 2i+1 in
    the high nibble, with one E8M0 byte per 32 elements of a row.
    MXFP4: 17-byte blocks of one E8M0 byte then 16 bytes holding element j in
    the low nibble and element j+16 in the high nibble. Same value set, same
    scale encoding, so this only moves nibbles.
    """
    rows, packed_cols = codes.shape
    cols = packed_cols * 2
    if cols % 32:
        raise ValueError("expert row is not a whole number of MXFP4 blocks")
    blocks = cols // 32
    if scales.shape != (rows, blocks):
        raise ValueError(f"expected {rows}x{blocks} block scales, found {scales.shape}")
    if np.any(scales == 255):
        raise ValueError("nonfinite E8M0 block scale")
    if np.any(scales == 0):
        # ds4 reads e == 0 as 2^-126 where the checkpoint means 2^-127. No
        # released tensor uses it; refuse rather than shift a weight silently.
        raise ValueError("E8M0 exponent 0 has no exact MXFP4 encoding")
    nibbles = np.empty((rows, cols), dtype=np.uint8)
    nibbles[:, 0::2] = codes & 0x0F
    nibbles[:, 1::2] = codes >> 4
    nibbles = nibbles.reshape(rows, blocks, 32)
    out = np.empty((rows, blocks, 17), dtype=np.uint8)
    out[:, :, 0] = scales
    out[:, :, 1:] = nibbles[:, :, :16] | (nibbles[:, :, 16:] << 4)
    return out.reshape(rows, blocks * 17)


def read_native_expert(db, np, name):
    codes = np.frombuffer(db.read(name), dtype=np.uint8)
    info = db.info(name)
    if info["dtype"] != "I8":
        raise ValueError(f"{name}: expected packed FP4 (I8), found {info['dtype']}")
    codes = codes.reshape(info["shape"])
    sinfo = db.info(scale_name(name))
    scales = np.frombuffer(db.read(scale_name(name)), dtype=np.uint8).reshape(sinfo["shape"])
    return mxfp4_from_native(np, codes, scales)


def write_native_gguf(args, plan, records, db):
    """Same journal and fsync discipline as the backbone writer, with one
    extra branch: MXFP4 experts are repacked from the release, not encoded."""
    import concurrent.futures, hashlib, shutil, struct, time
    quantizer = NativeQuantizer(args.quants_library)
    np = quantizer.np
    data_start, data_bytes = print_plan(plan, records, [], GGUF_ALIGNMENT)
    partial, journal = args.out + ".partial", args.out + ".partial.json"
    signature = conversion_signature(plan, records, [], None)
    source_identity = [(name, db.info(name)) for name in sorted(db.tensors)]
    signature = hashlib.sha256(
        (signature + json.dumps(source_identity, sort_keys=True)).encode()).hexdigest()
    if os.path.exists(args.out):
        raise ValueError(f"refusing to overwrite {args.out}")
    completed = 0
    if os.path.exists(partial) or os.path.exists(journal):
        if not args.resume or not (os.path.exists(partial) and os.path.exists(journal)):
            raise ValueError("partial file and journal require --resume")
        completed = load_resume_state(journal, signature, plan)
    end = data_start + (plan[completed - 1].offset +
                        align(plan[completed - 1].nbytes, GGUF_ALIGNMENT) if completed else 0)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(args.out))).free
    if free < data_start + data_bytes - end + (8 << 30):
        raise ValueError("insufficient disk space for remaining output plus 8 GiB reserve")
    header = b"GGUF" + struct.pack("<IQQ", 3, len(plan), len(records))
    header += b"".join(records) + b"".join(tensor_header(item) for item in plan)
    header += bytes(data_start - len(header))
    if os.path.exists(partial):
        with open(partial, "rb") as fp:
            if fp.read(data_start) != header or os.fstat(fp.fileno()).st_size < end:
                raise ValueError("partial GGUF is truncated or has a different header")
    else:
        with open(partial, "xb") as fp:
            fp.write(header)
            fp.flush()
            os.fsync(fp.fileno())
        save_resume_state(journal, signature, 0)
    with open(partial, "r+b") as fp, \
            concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        fp.truncate(end)
        fp.seek(end)
        for index in range(completed, len(plan)):
            item = plan[index]
            started = time.monotonic()
            if fp.tell() != data_start + item.offset:
                raise ValueError(f"incorrect offset for {item.name}")
            if item.is_expert and item.transform == "mxfp4":
                def repack(expert):
                    return read_native_expert(db, np, item.source.format(expert=expert)).tobytes()
                for first in range(0, item.expert_count, args.threads):
                    last = min(first + args.threads, item.expert_count)
                    for future in [pool.submit(repack, e) for e in range(first, last)]:
                        data = future.result()
                        if len(data) != item.nbytes // item.expert_count:
                            raise ValueError(f"{item.name}: wrong repacked expert size")
                        fp.write(data)
            elif item.is_expert:
                for first in range(0, item.expert_count, args.threads):
                    last = min(first + args.threads, item.expert_count)
                    futures = [pool.submit(lambda e: quantizer.encode(
                        quantizer.to_f32(db, item.source.format(expert=e)), item.qtype), e)
                        for e in range(first, last)]
                    for future in futures:
                        fp.write(future.result())
            else:
                fp.write(quantizer.encode(quantizer.to_f32(db, item.source), item.qtype))
            if fp.tell() != data_start + item.offset + item.nbytes:
                raise ValueError(f"incorrect payload size for {item.name}")
            fp.write(bytes(align(item.nbytes, GGUF_ALIGNMENT) - item.nbytes))
            fp.flush()
            os.fsync(fp.fileno())
            save_resume_state(journal, signature, index + 1)
            print(f"[{index + 1}/{len(plan)}] {item.name}: "
                  f"{item.nbytes / (1 << 30):.3f} GiB, {time.monotonic() - started:.1f}s",
                  flush=True)
    os.rename(partial, args.out)
    os.unlink(journal)


def records_for(c, revision, quant):
    targets = list(c["dspark_target_layer_ids"])
    records = [
        kv_string("general.architecture", ARCH),
        kv_string("general.name", "DeepSeek V4.1 Flash DSpark"),
        kv_u32("general.alignment", GGUF_ALIGNMENT),
        kv_string("general.source.url", SOURCE_URL),
        kv_string("general.source.revision", revision),
        kv_string("deepseek4.checkpoint_variant", "v4.1-flash"),
        kv_string("dspark.checkpoint_variant", "v4.1-flash"),
        kv_u32("dspark.block_size", c["dspark_block_size"]),
        kv_u32("dspark.markov_rank", c["dspark_markov_rank"]),
        kv_u32("dspark.noise_token_id", c["dspark_noise_token_id"]),
        kv_u32_array("dspark.target_layer_ids", targets),
        kv_u32("dspark.stage_count", c["num_nextn_predict_layers"]),
        kv_u32("dspark.n_layers", c["num_nextn_predict_layers"]),
        kv_u32("dspark.expert_count", c["dspark_n_routed_experts"]),
        kv_u32("dspark.expert_used_count", c["dspark_num_experts_per_tok"]),
        kv_u32("dspark.window_size", c["sliding_window"]),
        kv_u32("dspark.embedding_length", c["hidden_size"]),
        kv_u32("dspark.hc_mult", c["hc_mult"]),
        kv_u32("dspark.vocab_size", c["vocab_size"]),
        kv_u32("dspark.target_block_count", c["num_hidden_layers"]),
        kv_string("dspark.quantization", PRECISION[quant]["note"]),
        kv_string("dspark.precision", quant),
        kv_string("dspark.tied_weights", "token_embd,output"),
    ]
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="V4.1 Flash snapshot (mtp shards suffice)")
    parser.add_argument("--out", help="write this DSpark support GGUF")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--quant", choices=PRECISION, default="native",
                        help="drafter precision; native keeps every released weight bit-exact")
    parser.add_argument("--imatrix")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    suffix = "dylib" if sys.platform == "darwin" else "so"
    parser.add_argument("--quants-library",
                        default=os.path.join(os.path.dirname(__file__), f"libds4quants.{suffix}"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        parser.error("source revision must be a full commit hash")
    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run")
    if not 1 <= args.threads <= 32:
        parser.error("threads must be between 1 and 32")
    _, c = dspark_config(args.hf)
    db = MtpSourceDB(args.hf)
    try:
        plan = build_plan(db, c, args.quant)
        records = records_for(c, args.source_revision, args.quant)
        if args.dry_run:
            print_plan(plan, records, [], GGUF_ALIGNMENT)
            for item in plan:
                print(f"{item.name}\t{item.shape}\t{item.role}\t{item.nbytes}")
        elif args.quant == "native":
            write_native_gguf(args, plan, records, db)
        else:
            write_gguf(args, plan, records, db)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError) as error:
        sys.exit(f"deepseek41-dspark: {error}")
