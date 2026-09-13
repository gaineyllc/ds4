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
    QTYPE_IQ2_XXS, SourceDB, TensorPlan, align, fail, kv_string, kv_u32,
    kv_u32_array, print_plan, qtype_nbytes,
)

SOURCE_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"
ARCH = "deepseek4-dspark"

# The drafter is small next to the target, and draft quality is what sets the
# accepted-prefix length, so BF16 -- the released FP8/FP4 weights dequantized
# and not requantized -- is the default. The quantized recipes exist for hosts
# that cannot spare the resident bytes.
PRECISION = {
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
                                   expert_part=part, expert_count=experts))
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
    parser.add_argument("--quant", choices=PRECISION, default="bf16",
                        help="drafter precision; bf16 keeps the released weights intact")
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
        else:
            write_gguf(args, plan, records, db)
    finally:
        db.close()


if __name__ == "__main__":
    try:
        main()
    except (KeyError, OSError, ValueError) as error:
        sys.exit(f"deepseek41-dspark: {error}")
