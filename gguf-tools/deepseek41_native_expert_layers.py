#!/usr/bin/env python3
"""Replace selected layers' routed experts in a DeepSeek V4.1 Flash GGUF with the
released FP4 experts, repacked bit-exact as MXFP4 (the DSpark-native recipe,
deepseek41_dspark.mxfp4_from_native, applied to the main model).

The output is an APFS clone of the base (cp -c: no bytes copied); the new
expert payloads are appended at the end of the file and the nine tensor
records (gate/up/down x layers) are patched in place -- same name, same dims,
new type and offset -- so the header keeps its size and every other byte of
the file is the base's. The old payloads stay in the file unreferenced.

Only the HF shards holding the chosen layers are needed:
  layers.N.ffn.experts.E.w1|w3|w2.weight (+ .scale), one shard per layer.

Usage:
  python3 deepseek41_native_expert_layers.py --base gguf/DeepSeek-V4.1-Flash-Q2.gguf \
      --hf ~/hf/DeepSeek-V4.1-Flash --layers 37-39 \
      --out gguf/DeepSeek-V4.1-Flash-Q2-L37-39native.gguf
"""
import argparse, json, os, struct, subprocess, sys
import numpy as np

GGUF_MAGIC = b"GGUF"
QTYPE_MXFP4 = 39
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
SCALAR = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i",
          T_F32: "<f", T_BOOL: "<?", T_U64: "<Q", T_I64: "<q", T_F64: "<d"}


def parse_layers(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


class GGUFHeader:
    """Reads a GGUF v3 header; remembers where each tensor record's type and
    offset fields live so they can be patched in place."""

    def __init__(self, path):
        self.path = path
        f = self.f = open(path, "r+b")
        assert f.read(4) == GGUF_MAGIC, "not a GGUF"
        self.version, = struct.unpack("<I", f.read(4))
        assert self.version == 3, f"GGUF v{self.version}"
        self.n_tensors, self.n_kv = struct.unpack("<QQ", f.read(16))
        self.alignment = 32
        for _ in range(self.n_kv):
            key = self._str()
            t, = struct.unpack("<I", f.read(4))
            val = self._val(t)
            if key == "general.alignment":
                self.alignment = int(val)
        self.tensors = {}
        for _ in range(self.n_tensors):
            name = self._str()
            nd, = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            type_pos = f.tell()
            qtype, = struct.unpack("<I", f.read(4))
            offset, = struct.unpack("<Q", f.read(8))
            self.tensors[name] = dict(dims=dims, type=qtype, offset=offset, type_pos=type_pos)
        header_end = f.tell()
        self.data_start = (header_end + self.alignment - 1) // self.alignment * self.alignment

    def _str(self):
        n, = struct.unpack("<Q", self.f.read(8))
        return self.f.read(n).decode("utf-8", errors="replace")

    def _val(self, t):
        if t == T_STR:
            return self._str()
        if t == T_ARR:
            et, = struct.unpack("<I", self.f.read(4))
            n, = struct.unpack("<Q", self.f.read(8))
            return [self._val(et) for _ in range(n)]
        fmt = SCALAR[t]
        return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]

    def patch(self, name, qtype, offset):
        rec = self.tensors[name]
        self.f.seek(rec["type_pos"])
        self.f.write(struct.pack("<IQ", qtype, offset))
        rec["type"], rec["offset"] = qtype, offset


class Safetensors:
    def __init__(self, path):
        self.f = open(path, "rb")
        n, = struct.unpack("<Q", self.f.read(8))
        self.hdr = json.loads(self.f.read(n))
        self.base = 8 + n

    def read(self, name):
        info = self.hdr[name]
        a, b = info["data_offsets"]
        self.f.seek(self.base + a)
        return np.frombuffer(self.f.read(b - a), dtype=np.uint8), info


def mxfp4_from_native(codes, scales):
    """From gguf-tools/deepseek41_dspark.py: nibble move, no value change."""
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
        raise ValueError("E8M0 exponent 0 has no exact MXFP4 encoding")
    nibbles = np.empty((rows, cols), dtype=np.uint8)
    nibbles[:, 0::2] = codes & 0x0F
    nibbles[:, 1::2] = codes >> 4
    nibbles = nibbles.reshape(rows, blocks, 32)
    out = np.empty((rows, blocks, 17), dtype=np.uint8)
    out[:, :, 0] = scales
    out[:, :, 1:] = nibbles[:, :, :16] | (nibbles[:, :, 16:] << 4)
    return out.reshape(rows, blocks * 17)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    layers = parse_layers(args.layers)
    index = json.load(open(os.path.join(args.hf, "model.safetensors.index.json")))["weight_map"]

    base = GGUFHeader(args.base)
    n_expert = None
    plan = []
    for il in layers:
        shards = set()
        for part, src in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
            name = f"blk.{il}.ffn_{part}_exps.weight"
            if name not in base.tensors:
                sys.exit(f"{name} not in base GGUF (tensor naming?)")
            rec = base.tensors[name]
            dims = rec["dims"]  # (in, out, n_expert)
            n_expert = dims[2]
            hf0 = f"layers.{il}.ffn.experts.0.{src}.weight"
            if hf0 not in index:
                sys.exit(f"{hf0} not in HF index")
            shards.add(index[hf0])
            nbytes = dims[0] // 32 * 17 * dims[1] * n_expert
            plan.append((il, part, src, name, dims, nbytes, index[hf0]))
        print(f"layer {il}: shards {sorted(shards)}", file=sys.stderr)
    total = sum(p[5] for p in plan)
    print(f"{len(plan)} tensors, {total / 2**30:.2f} GiB of MXFP4 payload to append", file=sys.stderr)
    for p in plan:
        old = base.tensors[p[3]]
        print(f"  {p[3]} dims={p[4]} type {old['type']} -> {QTYPE_MXFP4} bytes={p[5]}", file=sys.stderr)
    base.f.close()
    if args.dry_run:
        return
    for p in plan:
        if not os.path.exists(os.path.join(args.hf, p[6])):
            sys.exit(f"missing shard {p[6]}")
    if os.path.exists(args.out):
        sys.exit(f"refusing to overwrite {args.out}")
    print("cloning base (cp -c)", file=sys.stderr)
    subprocess.check_call(["cp", "-c", args.base, args.out])
    out = GGUFHeader(args.out)
    f = out.f
    f.seek(0, os.SEEK_END)
    end = f.tell()
    shard_cache = {}
    for il, part, src, name, dims, nbytes, shard in plan:
        st = shard_cache.get(shard) or shard_cache.setdefault(shard, Safetensors(os.path.join(args.hf, shard)))
        # data offset relative to data_start, aligned
        pos = (end + out.alignment - 1) // out.alignment * out.alignment
        f.seek(pos)
        written = 0
        for e in range(n_expert):
            codes, info = st.read(f"layers.{il}.ffn.experts.{e}.{src}.weight")
            if info["dtype"] != "I8":
                sys.exit(f"expected packed FP4 (I8) for layer {il} expert {e}, got {info['dtype']}")
            codes = codes.reshape(info["shape"])
            scales, sinfo = st.read(f"layers.{il}.ffn.experts.{e}.{src}.scale")
            scales = scales.reshape(sinfo["shape"])
            if codes.shape != (dims[1], dims[0] // 2):
                sys.exit(f"{name}: expert {e} shape {codes.shape} vs dims {dims}")
            blob = mxfp4_from_native(codes, scales).tobytes()
            f.write(blob)
            written += len(blob)
        if written != nbytes:
            sys.exit(f"{name}: wrote {written} bytes, expected {nbytes}")
        out.patch(name, QTYPE_MXFP4, pos - out.data_start)
        end = pos + written
        print(f"  {name}: appended {written / 2**30:.2f} GiB at {pos}", file=sys.stderr)
    f.flush()
    os.fsync(f.fileno())
    f.close()
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
