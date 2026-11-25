"""The Merkle shred wire layout, as in Agave v4.4.0.

Each fact below was read in the public Agave source at the v4.4.0 tag (file:line):

  ledger/src/shred.rs         header sizes 83 / 88 / 89 (:114-116); 32 data + 32 coding shreds per erasure set
                              (:123-124); data flags (:183-185): reference tick mask 0x3F, DATA_COMPLETE 0x40,
                              LAST_IN_SLOT 0xC0; variants (:705-756): 0x60 code, 0x70 code resigned, 0x90 data,
                              0xB0 data resigned, low nibble = Merkle proof size
  ledger/src/shred/wire.rs    variant b[64] (:79), slot b[65:73] (:92), index b[73:77] (:98), version b[77:79]
                              (:104), fec_set_index b[79:83] (:110); data: parent_offset b[83:85] (:118), flags
                              b[85] (:137), size b[86:88] (:150); code: num_data b[83:85], num_coding b[85:87]
                              (:176-181), then position b[87:89]
  ledger/src/shred/merkle.rs  payload 1203 data (:434-435) / 1228 code (:491); capacity (:223-234); proof offset
                              (:248-249); chained root offset (:263-268); retransmitter signature after the proof
                              (:374-383); erasure shards: data [64, 88 + capacity), code [89, 89 + capacity) (:70,
                              :150, :386-399); data = b[88:size] (:74-90); shard index: data = index -
                              fec_set_index (:101-107), code = num_data + position (:161-171)

A shred, in byte order:

  [0, 64)                 leader signature over the erasure set's Merkle root
  [64, 83)                common header: variant, slot, index, version, fec_set_index
  [83, 88) or [83, 89)    data header (parent_offset, flags, size) or code header (num_data, num_coding, position)
  [hdr, hdr + capacity)   data bytes (data shred, zero-filled past `size`) or Reed-Solomon parity (code shred)
  next 32                 chained Merkle root: the root of the previous erasure set
  next 20 * proof_size    Merkle proof, one 20-byte sibling per tree level
  next 64 (resigned only) retransmitter signature
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .errors import ShredError

SIGNATURE_SIZE = 64
DATA_HEADER_SIZE = 88
CODE_HEADER_SIZE = 89
DATA_PAYLOAD_SIZE = 1203
CODE_PAYLOAD_SIZE = 1228
MERKLE_ROOT_SIZE = 32
PROOF_ENTRY_SIZE = 20
SHREDS_PER_SET = 32
MAX_DATA_SHREDS_PER_SLOT = 32_768

FLAG_REFERENCE_TICK_MASK = 0x3F
FLAG_DATA_COMPLETE = 0x40
FLAG_LAST_IN_SLOT = 0xC0

_COMMON = struct.Struct("<BQIHI")    # variant, slot, index, version, fec_set_index at b[64:83]
_DATA_TAIL = struct.Struct("<HBH")   # parent_offset, flags, size at b[83:88]
_CODE_TAIL = struct.Struct("<HHH")   # num_data, num_coding, position at b[83:89]

_HIGH_NIBBLE = {0x60: (False, False), 0x70: (False, True), 0x90: (True, False), 0xB0: (True, True)}


@dataclass(frozen=True)
class Variant:
    """The variant byte decoded. Only the chained Merkle variants exist in this layout; resigned ones carry an
    extra 64-byte retransmitter signature, so every offset after the header depends on (is_data, resigned,
    proof_size) and lives here."""

    is_data: bool
    resigned: bool
    proof_size: int

    @classmethod
    def from_byte(cls, byte: int) -> Variant:
        kind = _HIGH_NIBBLE.get(byte & 0xF0)
        if kind is None:
            raise ShredError(f"unsupported shred variant 0x{byte:02x}")
        return cls(kind[0], kind[1], byte & 0x0F)

    def to_byte(self) -> int:
        high = {v: k for k, v in _HIGH_NIBBLE.items()}[(self.is_data, self.resigned)]
        return high | self.proof_size

    def sibling(self, is_data: bool) -> Variant:
        """The data or code variant of the same erasure set (same proof size and resigned flag)."""
        return Variant(is_data, self.resigned, self.proof_size)

    @property
    def header_size(self) -> int:
        return DATA_HEADER_SIZE if self.is_data else CODE_HEADER_SIZE

    @property
    def payload_size(self) -> int:
        return DATA_PAYLOAD_SIZE if self.is_data else CODE_PAYLOAD_SIZE

    @property
    def capacity(self) -> int:
        """Bytes left for data (or parity) after the header, root, proof and optional retransmitter signature."""
        used = self.header_size + MERKLE_ROOT_SIZE + PROOF_ENTRY_SIZE * self.proof_size
        return self.payload_size - used - (SIGNATURE_SIZE if self.resigned else 0)

    @property
    def root_offset(self) -> int:
        return self.header_size + self.capacity

    @property
    def proof_offset(self) -> int:
        return self.root_offset + MERKLE_ROOT_SIZE

    @property
    def retransmitter_offset(self) -> int:
        return self.proof_offset + PROOF_ENTRY_SIZE * self.proof_size

    @property
    def erasure_range(self) -> tuple[int, int]:
        """Data shards start at the variant byte, so a recovered data shard brings its own header back; code
        shards are the parity only. Both have the same length, which is what Reed-Solomon needs."""
        if self.is_data:
            return SIGNATURE_SIZE, DATA_HEADER_SIZE + self.capacity
        return CODE_HEADER_SIZE, CODE_HEADER_SIZE + self.capacity


@dataclass(frozen=True)
class Shred:
    """One parsed shred. `payload` is exactly `variant.payload_size` bytes; the header fields are decoded from it."""

    payload: bytes
    variant: Variant
    slot: int
    index: int
    version: int
    fec_set_index: int
    parent_offset: int = 0
    flags: int = 0
    size: int = 0
    num_data: int = 0
    num_coding: int = 0
    position: int = 0

    @property
    def is_data(self) -> bool:
        return self.variant.is_data

    @property
    def signature(self) -> bytes:
        return self.payload[:SIGNATURE_SIZE]

    @property
    def shard_index(self) -> int:
        """Position of this shred among the set's erasure shards and Merkle leaves: data first, then code."""
        if self.is_data:
            return self.index - self.fec_set_index
        return self.num_data + self.position

    @property
    def data_complete(self) -> bool:
        return self.is_data and bool(self.flags & FLAG_DATA_COMPLETE)

    @property
    def last_in_slot(self) -> bool:
        return self.is_data and (self.flags & FLAG_LAST_IN_SLOT) == FLAG_LAST_IN_SLOT

    @property
    def reference_tick(self) -> int:
        return self.flags & FLAG_REFERENCE_TICK_MASK

    def data(self) -> bytes:
        """The entry-batch bytes this data shred carries: b[88:size]."""
        if not self.is_data:
            raise ShredError("code shreds carry parity, not data")
        return self.payload[DATA_HEADER_SIZE:self.size]

    def erasure_shard(self) -> bytes:
        start, end = self.variant.erasure_range
        return self.payload[start:end]

    def chained_root(self) -> bytes:
        offset = self.variant.root_offset
        return self.payload[offset:offset + MERKLE_ROOT_SIZE]

    def merkle_leaf_bytes(self) -> bytes:
        """Everything the leaf hash covers: from the variant byte to the end of the chained root."""
        return self.payload[SIGNATURE_SIZE:self.variant.proof_offset]

    def proof(self) -> list[bytes]:
        offset = self.variant.proof_offset
        return [self.payload[offset + PROOF_ENTRY_SIZE * k:offset + PROOF_ENTRY_SIZE * (k + 1)]
                for k in range(self.variant.proof_size)]


def parse_shred(buf: bytes) -> Shred:
    """Parse and sanity-check one shred.

    Bytes past the variant's payload size are ignored (repair responses, for example, append a nonce after the
    shred). Range checks follow what the validator rejects, so a parsed shred has a usable index, size and shard
    position; whether it is authentic is a separate question (merkle.verify_leader_signature).
    """
    if len(buf) < CODE_HEADER_SIZE:
        raise ShredError(f"truncated shred: {len(buf)} bytes, a header needs {CODE_HEADER_SIZE}")
    variant = Variant.from_byte(buf[SIGNATURE_SIZE])
    if variant.capacity <= 0:
        raise ShredError(f"proof size {variant.proof_size} leaves no room for data")
    if len(buf) < variant.payload_size:
        raise ShredError(f"truncated shred: {len(buf)} bytes, variant 0x{variant.to_byte():02x} needs "
                         f"{variant.payload_size}")
    payload = bytes(buf[:variant.payload_size])
    _, slot, index, version, fec_set_index = _COMMON.unpack_from(payload, SIGNATURE_SIZE)
    common = dict(payload=payload, variant=variant, slot=slot, index=index, version=version,
                  fec_set_index=fec_set_index)
    if variant.is_data:
        parent_offset, flags, size = _DATA_TAIL.unpack_from(payload, 83)
        _check_data_header(variant, slot, index, fec_set_index, parent_offset, flags, size)
        return Shred(parent_offset=parent_offset, flags=flags, size=size, **common)
    num_data, num_coding, position = _CODE_TAIL.unpack_from(payload, 83)
    _check_code_header(num_data, num_coding, position)
    return Shred(num_data=num_data, num_coding=num_coding, position=position, **common)


def _check_data_header(variant: Variant, slot: int, index: int, fec_set_index: int, parent_offset: int,
                       flags: int, size: int) -> None:
    if not DATA_HEADER_SIZE <= size <= DATA_HEADER_SIZE + variant.capacity:
        raise ShredError(f"data size {size} outside [{DATA_HEADER_SIZE}, {DATA_HEADER_SIZE + variant.capacity}]")
    if index < fec_set_index:
        raise ShredError(f"index {index} below its fec_set_index {fec_set_index}")
    if index >= MAX_DATA_SHREDS_PER_SLOT:
        raise ShredError(f"index {index} out of range")
    if (flags & FLAG_LAST_IN_SLOT) == 0x80:
        raise ShredError("LAST_IN_SLOT bit set without DATA_COMPLETE")
    if parent_offset > slot or (parent_offset == 0 and slot != 0):
        raise ShredError(f"parent offset {parent_offset} invalid for slot {slot}")


def _check_code_header(num_data: int, num_coding: int, position: int) -> None:
    if num_data < 1 or num_coding < 1 or num_data + num_coding > 256:
        raise ShredError(f"erasure set {num_data} + {num_coding} is not codable over GF(2^8)")
    if position >= num_coding:
        raise ShredError(f"code position {position} >= num_coding {num_coding}")


def data_header(variant: Variant, slot: int, index: int, version: int, fec_set_index: int, parent_offset: int,
                flags: int, size: int) -> bytes:
    """b[64:88] of a data shred."""
    return _COMMON.pack(variant.to_byte(), slot, index, version, fec_set_index) + \
        _DATA_TAIL.pack(parent_offset, flags, size)


def code_header(variant: Variant, slot: int, index: int, version: int, fec_set_index: int, num_data: int,
                num_coding: int, position: int) -> bytes:
    """b[64:89] of a code shred."""
    return _COMMON.pack(variant.to_byte(), slot, index, version, fec_set_index) + \
        _CODE_TAIL.pack(num_data, num_coding, position)


def assemble_payload(variant: Variant, signature: bytes, body: bytes, chained_root: bytes, proof: list[bytes],
                     retransmitter_signature: bytes = b"") -> bytes:
    """signature + body (b[64:root_offset]: header and data or parity) + chained root + proof (+ retransmitter
    signature). Used by the synthetic builder and by recovery, so both lay shreds out the same way."""
    if len(body) != variant.root_offset - SIGNATURE_SIZE:
        raise ShredError(f"body is {len(body)} bytes, variant needs {variant.root_offset - SIGNATURE_SIZE}")
    if len(proof) != variant.proof_size or any(len(p) != PROOF_ENTRY_SIZE for p in proof):
        raise ShredError("proof does not match the variant's proof size")
    tail = (retransmitter_signature or bytes(SIGNATURE_SIZE)) if variant.resigned else b""
    out = signature + body + chained_root + b"".join(proof) + tail
    if len(out) != variant.payload_size:
        raise ShredError(f"assembled {len(out)} bytes, expected {variant.payload_size}")
    return out
