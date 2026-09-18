#!/usr/bin/env python3
"""Serialize an Engram table into a contiguous, 64-byte row-aligned FP8 file.

Spec Phase 1. The GGUF's Engram table is 264 B rows laid out as

    [0 .. 255]  256 x E4M3 values
    [256 .. 263]  8 x E8M0 block scales (one per 32-value block)

which is dense but not aligned: a 264 B row straddles a 64 B boundary three
times out of eight, so a gather touches one extra cache line per row about
half the time. This tool rewrites the table with the scales hoisted into a
prepended index and each row padded to 320 B (5 x 64), so every row starts on
a 64 B boundary and a row's values are one aligned 256 B run.

Output layout:

    [0 .. 63]            header (magic, version, rows, row_stride, dim, blocks,
                         scale_offset, data_offset), little endian, zero padded
    [scale_offset ..]    rows x 8 B of E8M0 scales, contiguous
    [data_offset ..]     rows x 320 B, first 256 B values, remainder zero

Streaming: reads and writes in chunks, so peak RSS is the chunk size, not the
table. A table is ~94 GiB and the output is larger (320/264 = 1.21x), so the
output path is required rather than defaulted next to the input -- filling the
boot volume is how this machine wedges.

Usage:
  deepseek41_engram_serialize.py --input MODEL.gguf --offset N --rows N \\
      --output /Volumes/external/engram_L1.bin [--chunk-rows 65536] [--verify]
"""

import argparse
import os
import struct
import sys

MAGIC = b"DS4ENGR1"
HEADER_BYTES = 64
DIM = 256
BLOCKS = 8
SRC_ROW_BYTES = 264
DST_ROW_BYTES = 320   # 64 B aligned
SCALE_BYTES = 8


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0


def build_header(rows):
    scale_offset = HEADER_BYTES
    data_offset = scale_offset + rows * SCALE_BYTES
    # align the data region itself to 64 B so row r sits at data_offset + r*320
    if data_offset % 64:
        data_offset += 64 - (data_offset % 64)
    head = MAGIC + struct.pack(
        "<IIQIIQQ",
        1,                 # version
        DIM,
        rows,
        DST_ROW_BYTES,
        BLOCKS,
        scale_offset,
        data_offset,
    )
    if len(head) > HEADER_BYTES:
        raise RuntimeError("header overflow")
    return head + b"\x00" * (HEADER_BYTES - len(head)), scale_offset, data_offset


def serialize(args):
    rows = args.rows
    header, scale_offset, data_offset = build_header(rows)
    total_out = data_offset + rows * DST_ROW_BYTES

    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    st = os.statvfs(out_dir)
    free = st.f_bavail * st.f_frsize
    if free < total_out:
        sys.exit(
            f"refusing to write: {args.output} needs {human(total_out)}, "
            f"{out_dir} has {human(free)} free"
        )

    src_bytes = rows * SRC_ROW_BYTES
    print(f"rows       {rows}")
    print(f"input      {args.input} @ {args.offset} ({human(src_bytes)})")
    print(f"output     {args.output} ({human(total_out)})")
    print(f"scales     {scale_offset}")
    print(f"data       {data_offset}, stride {DST_ROW_BYTES}")

    chunk_rows = args.chunk_rows
    pad = b"\x00" * (DST_ROW_BYTES - DIM)

    with open(args.input, "rb", buffering=0) as src, \
         open(args.output, "wb", buffering=0) as dst:
        dst.write(header)

        # pass 1: scales
        src.seek(args.offset)
        written = 0
        done = 0
        while done < rows:
            n = min(chunk_rows, rows - done)
            buf = src.read(n * SRC_ROW_BYTES)
            if len(buf) != n * SRC_ROW_BYTES:
                sys.exit(f"short read at row {done}")
            out = bytearray(n * SCALE_BYTES)
            for i in range(n):
                base = i * SRC_ROW_BYTES + DIM
                out[i * SCALE_BYTES:(i + 1) * SCALE_BYTES] = buf[base:base + SCALE_BYTES]
            dst.write(out)
            written += len(out)
            done += n
            if args.progress and done % (chunk_rows * 16) == 0:
                print(f"  scales {done}/{rows}", flush=True)

        # pad to data_offset
        here = HEADER_BYTES + written
        if here < data_offset:
            dst.write(b"\x00" * (data_offset - here))

        # pass 2: values, padded
        src.seek(args.offset)
        done = 0
        while done < rows:
            n = min(chunk_rows, rows - done)
            buf = src.read(n * SRC_ROW_BYTES)
            if len(buf) != n * SRC_ROW_BYTES:
                sys.exit(f"short read at row {done}")
            out = bytearray()
            for i in range(n):
                base = i * SRC_ROW_BYTES
                out += buf[base:base + DIM]
                out += pad
            dst.write(out)
            done += n
            if args.progress and done % (chunk_rows * 16) == 0:
                print(f"  values {done}/{rows}", flush=True)

    print(f"wrote {human(total_out)}")

    if args.verify:
        verify(args, scale_offset, data_offset)


def verify(args, scale_offset, data_offset):
    """Spot-check that a sample of rows round-trips byte for byte."""
    import random

    rows = args.rows
    sample = sorted(random.sample(range(rows), min(args.verify_rows, rows)))
    bad = 0
    with open(args.input, "rb", buffering=0) as src, \
         open(args.output, "rb", buffering=0) as dst:
        for r in sample:
            src.seek(args.offset + r * SRC_ROW_BYTES)
            ref = src.read(SRC_ROW_BYTES)
            dst.seek(data_offset + r * DST_ROW_BYTES)
            vals = dst.read(DIM)
            dst.seek(scale_offset + r * SCALE_BYTES)
            scales = dst.read(SCALE_BYTES)
            if vals != ref[:DIM] or scales != ref[DIM:DIM + SCALE_BYTES]:
                bad += 1
                if bad <= 5:
                    print(f"  MISMATCH row {r}")
    print(f"verify: {len(sample) - bad}/{len(sample)} rows match")
    if bad:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="GGUF holding the Engram table")
    p.add_argument("--offset", required=True, type=int,
                   help="absolute byte offset of the table (tensor abs_offset)")
    p.add_argument("--rows", required=True, type=int, help="row count for this table")
    p.add_argument("--output", required=True,
                   help="output path; required, and checked against free space")
    p.add_argument("--chunk-rows", type=int, default=65536)
    p.add_argument("--progress", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="spot-check sampled rows against the source after writing")
    p.add_argument("--verify-rows", type=int, default=4096)
    args = p.parse_args()

    if args.rows <= 0 or args.offset < 0:
        sys.exit("rows must be positive and offset non-negative")
    serialize(args)


if __name__ == "__main__":
    main()
