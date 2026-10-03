#!/usr/bin/env python3
"""Write a copy of a model's gguf_header.bin that describes ANOTHER upload.

    plant_mixed_header.py <gguf_header.bin> <out> [extra_bytes | shorter]

FUTURE_WORK #156: a node whose header came from one upload of a model and
whose parts (and their tensor table) from another reads every tensor from the
wrong place and answers garbage or NaN — the reader never fails, because each
shifted position still lands inside a part. The .220 gate measured the real
case: two uploads of Llama-3.2-1B Q8_0 whose headers differ by 3,808 bytes
(gotcha #776), so every tensor sat 3,808 bytes later than the table said.

This makes that header from any model's own: the same tensors, the same
types, the same metadata, plus one string entry (`swarmllm.test.planted`) of
`extra_bytes` (default 3808) — so the tensor data starts that much later,
rounded to the file's alignment. Planted beside a node's parts (replace the
file, never edit it in place: the rig's files are HARD LINKS to the live
node's), the shard loader must refuse with "Mixed model copy" rather than load.

TWO SHAPES, and they fail differently on a build without the check:
- LONGER (a number): every read lands later. The file's last tensor then runs
  past the end of the data — "failed to fill whole buffer", an ERROR. Gotcha
  #776's shape.
- `shorter`: drops the `tokenizer.chat_template` entry, so every read lands
  EARLIER — into the previous tensor, or the header, all of it mapped on a node
  holding every part. Nothing fails; the model computes on the wrong bytes and
  answers garbage or NaN. FUTURE_WORK #156's shape, the silent one — what the
  rig's `mixed` mode plants by default. (A node using the template is a
  coordinator; the planted node only computes layers.)


Standard library only; GGUF v2/v3 as written by llama.cpp's gguf.cpp.
"""
import struct
import sys

# value type -> struct format for the fixed-size scalars
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
          10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9


class Reader:
    def __init__(self, data):
        self.d, self.p = data, 0

    def take(self, n):
        b = self.d[self.p:self.p + n]
        if len(b) != n:
            raise SystemExit(f"truncated header at byte {self.p}")
        self.p += n
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8")

    def value(self, vtype):
        if vtype in SCALAR:
            return struct.unpack(SCALAR[vtype], self.take(struct.calcsize(SCALAR[vtype])))[0]
        if vtype == STRING:
            return self.string()
        if vtype == ARRAY:
            etype, n = self.u32(), self.u64()
            return [self.value(etype) for _ in range(n)]
        raise SystemExit(f"unknown GGUF value type {vtype} at byte {self.p}")


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def align(n, a):
    return (n + a - 1) // a * a


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    shape = sys.argv[3] if len(sys.argv) == 4 else "3808"
    data = open(src, "rb").read()
    r = Reader(data)
    if r.take(4) != b"GGUF":
        raise SystemExit(f"{src} is not a GGUF header")
    version = r.u32()
    n_tensors, n_kv = r.u64(), r.u64()
    alignment = 32
    kvs = []  # (key, raw bytes of the whole entry)
    for _ in range(n_kv):
        start = r.p
        key = r.string()
        vtype = r.u32()
        val = r.value(vtype)
        kvs.append((key, data[start:r.p]))
        if key == "general.alignment":
            alignment = int(val)
    kv_end = r.p
    for _ in range(n_tensors):
        r.string()
        n_dims = r.u32()
        r.take(8 * n_dims + 4 + 8)  # dims, type, offset
    infos_end = r.p

    if shape == "shorter":
        dropped = "tokenizer.chat_template"
        if not any(k == dropped for k, _ in kvs):
            raise SystemExit(f"{src} has no {dropped} to drop — plant a longer header instead")
        kept = [raw for k, raw in kvs if k != dropped]
    else:
        planted = (gguf_string("swarmllm.test.planted") + struct.pack("<I", STRING)
                   + gguf_string("x" * int(shape)))
        kept = [raw for _, raw in kvs] + [planted]
    head = b"GGUF" + struct.pack("<I", version) + struct.pack("<QQ", n_tensors, len(kept))
    body = b"".join(kept) + data[kv_end:infos_end]
    new_end = len(head) + len(body)
    blob = head + body + b"\0" * (align(new_end, alignment) - new_end)
    open(out, "wb").write(blob)
    shift = align(new_end, alignment) - align(infos_end, alignment)
    if shift == 0:
        raise SystemExit("the planted header starts its data where the real one does — nothing would be shifted")
    print(f"{out}: {n_tensors} tensors, tensor data now starts {abs(shift)} bytes "
          f"{'later' if shift > 0 else 'EARLIER'} ({align(infos_end, alignment)} -> {align(new_end, alignment)})")


if __name__ == "__main__":
    main()
