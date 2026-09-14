#!/usr/bin/env python3
"""Requantize selected dense Q8_0 tensors of a DeepSeek V4.1 GGUF in place.

Decode on Apple silicon is memory-bandwidth bound and the Q8_0 attention
projections (attn_q_b, attn_output_a, attn_output_b) are ~48% of the bytes a
token reads. This tool makes a copy-on-write clone of the GGUF (APFS
`cp -c`, so only rewritten blocks cost disk), dequantizes the chosen Q8_0
tensors, requantizes them to Q4_K (or Q6_K/Q5_K) with the same quantizer the
DS4 converter uses, writes the smaller rows back at the tensor's existing
offset and patches the tensor's type id in the header. Nothing else moves:
offsets, dims and the Engram tables are untouched, so the result is a valid
GGUF with a gap after each shrunk tensor.

    python3 gguf-tools/deepseek41_requant_dense.py \
        --src gguf/DeepSeek-V4.1-Flash-Q2.gguf \
        --dst gguf/DeepSeek-V4.1-Flash-Q2-attnQ4K.gguf \
        --match 'attn_q_b|attn_output_a|attn_output_b' --type Q4_K
"""
import argparse, ctypes, os, re, struct, subprocess, sys, threading
import numpy as np

TYPES = {"Q4_K": 12, "Q5_K": 13, "Q6_K": 14, "Q8_0": 8}

def read_header(f):
    """Return (kv, tensors, data_offset). tensors: list of dict with name, dims,
    type, offset, type_pos (byte position of the type field in the file)."""
    def rd(fmt):
        b = f.read(struct.calcsize(fmt)); return struct.unpack("<" + fmt, b)
    def rstr():
        (n,) = rd("Q"); return f.read(n).decode("utf-8", "replace")
    magic = f.read(4)
    if magic != b"GGUF": raise SystemExit("not a GGUF file")
    (ver,) = rd("I"); (nt,) = rd("Q"); (nkv,) = rd("Q")
    T = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}
    def rval(t):
        if t == 8: return rstr()
        if t == 9:
            (et,) = rd("I"); (n,) = rd("Q"); return [rval(et) for _ in range(n)]
        return rd(T[t])[0]
    kv = {}
    for _ in range(nkv):
        k = rstr(); (t,) = rd("I"); kv[k] = rval(t)
    align = kv.get("general.alignment", 32)
    tensors = []
    for _ in range(nt):
        name = rstr(); (nd,) = rd("I"); dims = rd("Q" * nd)
        type_pos = f.tell(); (ty,) = rd("I"); (off,) = rd("Q")
        tensors.append(dict(name=name, dims=dims, type=ty, offset=off, type_pos=type_pos))
    data0 = (f.tell() + align - 1) // align * align
    return kv, tensors, data0

def dequant_q8_0(raw, rows, cols):
    """raw: bytes of rows*cols/32 Q8_0 blocks (34 B each) -> float32 [rows, cols]."""
    nb = rows * cols // 32
    blk = np.frombuffer(raw, dtype=np.dtype([("d", "<f2"), ("q", "i1", (32,))]), count=nb)
    return (blk["d"].astype(np.float32)[:, None] * blk["q"].astype(np.float32)).reshape(rows, cols)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--match", default=r"^blk\.\d+\.(attn_q_b|attn_output_a|attn_output_b)\.weight$")
    ap.add_argument("--type", default="Q4_K", choices=sorted(TYPES))
    ap.add_argument("--rows-per-chunk", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--quants-library",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "libds4quants.dylib" if sys.platform == "darwin" else "libds4quants.so"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    qt = TYPES[a.type]
    lib = ctypes.CDLL(a.quants_library)
    lib.ds4q_quantize_init.argtypes = [ctypes.c_int]
    lib.ds4q_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]; lib.ds4q_row_size.restype = ctypes.c_size_t
    lib.ds4q_block_size.argtypes = [ctypes.c_int]; lib.ds4q_block_size.restype = ctypes.c_int64
    lib.ds4q_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    lib.ds4q_quantize_chunk.restype = ctypes.c_size_t
    lib.ds4q_quantize_init(qt)
    blk = lib.ds4q_block_size(qt)

    if not os.path.exists(a.dst):
        if a.dry_run: print("would clone", a.src, "->", a.dst)
        else:
            print("cloning (copy-on-write)...", flush=True)
            subprocess.check_call(["cp", "-c", a.src, a.dst])
    with open(a.src, "rb") as f:
        kv, tensors, data0 = read_header(f)
    pat = re.compile(a.match)
    todo = [t for t in tensors if pat.search(t["name"]) and t["type"] == 8]
    skipped = [t for t in tensors if pat.search(t["name"]) and t["type"] != 8]
    for t in skipped: print("skip (not Q8_0):", t["name"], "type", t["type"])
    total_in = sum(np.prod(t["dims"]) for t in todo)
    print(f"{len(todo)} tensors, {total_in/1e9:.2f} G weights -> {a.type}")
    if a.dry_run: return
    fd = os.open(a.dst, os.O_RDWR)
    try:
        for ti, t in enumerate(todo):
            cols = t["dims"][0]; rows = int(np.prod(t["dims"][1:]))
            if cols % blk: raise SystemExit(f"{t['name']}: cols {cols} not a multiple of {blk}")
            q8_row = cols // 32 * 34; q_row = lib.ds4q_row_size(qt, cols)
            base = data0 + t["offset"]
            assert rows * q_row <= rows * q8_row
            print(f"[{ti+1}/{len(todo)}] {t['name']} {rows}x{cols} {rows*q8_row/2**20:.0f} MiB -> {rows*q_row/2**20:.0f} MiB", flush=True)
            # Dequantize whole tensor chunk by chunk, quantize in parallel threads, write back in order.
            chunks = [(r0, min(a.rows_per_chunk, rows - r0)) for r0 in range(0, rows, a.rows_per_chunk)]
            outs = [None] * len(chunks)
            def work(i):
                r0, n = chunks[i]
                raw = os.pread(fd, n * q8_row, base + r0 * q8_row)
                x = np.ascontiguousarray(dequant_q8_0(raw, n, cols))
                dst = (ctypes.c_uint8 * (n * q_row))()
                got = lib.ds4q_quantize_chunk(qt, x.ctypes.data, ctypes.addressof(dst), 0, n, cols, None)
                if got != n * q_row: raise RuntimeError(f"quantize_chunk returned {got}, expected {n*q_row}")
                outs[i] = bytes(dst)
            # Rows are read at the OLD stride and written at the NEW (smaller) stride at the same base:
            # every write lands at or before the bytes it replaces, and chunk i only overwrites rows < r0+n
            # of the old layout after those rows were read -- so read the whole tensor first.
            lock = threading.Lock(); nxt = [0]
            def runner():
                while True:
                    with lock:
                        i = nxt[0]; nxt[0] += 1
                    if i >= len(chunks): return
                    work(i)
            th = [threading.Thread(target=runner) for _ in range(a.threads)]
            [x.start() for x in th]; [x.join() for x in th]
            pos = base
            for i, (r0, n) in enumerate(chunks):
                os.pwrite(fd, outs[i], pos); pos += len(outs[i])
            os.pwrite(fd, struct.pack("<I", qt), t["type_pos"])
        os.fsync(fd)
    finally:
        os.close(fd)
    print("done:", a.dst)

if __name__ == "__main__":
    main()
