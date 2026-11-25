"""Entry batches: from data shreds to entries and transactions.

A leader turns a run of entries into one "batch": the bincode serialization of Vec<Entry> (agave v4.4.0
ledger/src/shred/merkle.rs:942, "Serialized &[Entry]"), cut across consecutive data shreds. The last data shred
of the batch has the DATA_COMPLETE flag. An Entry (agave v4.4.0 entry/src/entry.rs:48-60) is

  num_hashes u64, hash [32], transactions: u64 count, then each transaction in its wire format.

Deshredding (solana-ledger 4.4.1 src/shredder.rs:199-243, Shredder::deshred) requires consecutive indexes, no
DATA_COMPLETE before the last shred, DATA_COMPLETE on the last one, then concatenates each shred's b[88:size].

Trailing bytes. The validator's decoder reads the Vec<Entry> and does not require it to end at the last byte of
the batch. Agave's own deshred relies on that: for a batch with no data it returns a zero-filled buffer so that
it decodes as an empty Vec<Entry> (shredder.rs:233-242). Some producers do send zero padding after the entries,
inside each shred's declared size. shredkit therefore decodes the entries, then classifies what follows:
nothing ("exact"), only zero bytes ("zero_padded", accepted), or anything else ("trailing_bytes", not accepted:
the decoder would tolerate it, but it is not something a producer should send and it is worth a look).

Block markers. Since Agave 4.0 a batch whose entry count is 0 and which is followed by more data is a block marker
(u16 version, then marker data) rather than entries (solana-entry 4.0.0 src/block_component.rs). shredkit reports
such a batch as "marker" with its version and does not decode the marker.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .codec import ByteReader
from .errors import DecodeError
from .layout import Shred
from .tx import Transaction, parse_transaction

ENTRY_FIXED_SIZE = 8 + 32 + 8           # num_hashes, hash, transaction count
MIN_TRANSACTION_SIZE = 1 + 64           # a count byte (or version byte) and one signature, as a loose lower bound


@dataclass(frozen=True)
class Entry:
    num_hashes: int
    hash: bytes
    transactions: tuple[Transaction, ...]


@dataclass(frozen=True)
class EntryBatch:
    entries: tuple[Entry, ...]
    end: int                            # bytes used by the entries (or the marker's count and version)
    size: int                           # bytes in the batch
    tail_is_zero: bool
    marker_version: int | None = None

    @property
    def tail_length(self) -> int:
        return self.size - self.end

    @property
    def status(self) -> str:
        if self.marker_version is not None:
            return "marker"
        if self.tail_length == 0:
            return "exact"
        return "zero_padded" if self.tail_is_zero else "trailing_bytes"

    @property
    def ok(self) -> bool:
        return self.status != "trailing_bytes"

    @property
    def transactions(self) -> list[Transaction]:
        return [tx for entry in self.entries for tx in entry.transactions]


def serialize_entries(entries: Iterable[tuple[int, bytes, Sequence[bytes]]]) -> bytes:
    """[(num_hashes, hash32, [transaction bytes])] -> a batch. For synthetic data and tests."""
    entries = list(entries)
    out = bytearray(struct.pack("<Q", len(entries)))
    for num_hashes, entry_hash, txs in entries:
        if len(entry_hash) != 32:
            raise ValueError("entry hash must be 32 bytes")
        out += struct.pack("<Q", num_hashes) + entry_hash + struct.pack("<Q", len(txs)) + b"".join(txs)
    return bytes(out)


def parse_entry_batch(blob: bytes) -> EntryBatch:
    """Decode a whole batch. Raises DecodeError if the entries themselves do not decode; a non-zero tail after
    valid entries is reported in the result, not raised, so the caller can count it."""
    if not blob:
        return EntryBatch((), 0, 0, True)
    r = ByteReader(blob)
    count = r.u64()
    if count == 0:
        return _empty_or_marker(blob, r)
    if count > r.remaining() // ENTRY_FIXED_SIZE:
        raise DecodeError(f"entry count {count} cannot fit in {r.remaining()} bytes")
    entries = tuple(_parse_entry(r) for _ in range(count))
    return EntryBatch(entries, r.pos, len(blob), not any(blob[r.pos:]))


def _empty_or_marker(blob: bytes, r: ByteReader) -> EntryBatch:
    if not any(blob[r.pos:]):
        return EntryBatch((), r.pos, len(blob), True)
    version = r.u16()
    return EntryBatch((), r.pos, len(blob), False, marker_version=version)


def _parse_entry(r: ByteReader) -> Entry:
    num_hashes = r.u64()
    entry_hash = r.take(32)
    n_tx = r.u64()
    if n_tx > r.remaining() // MIN_TRANSACTION_SIZE:
        raise DecodeError(f"transaction count {n_tx} cannot fit in {r.remaining()} bytes")
    txs = []
    for _ in range(n_tx):
        tx, r.pos = parse_transaction(r.buf, r.pos)
        txs.append(tx)
    return Entry(num_hashes, entry_hash, tuple(txs))


def deshred(shreds: Sequence[Shred]) -> EntryBatch:
    """One batch from its data shreds, in index order, with the same checks as Agave's Shredder::deshred."""
    if not shreds:
        raise DecodeError("no shreds")
    for prev, cur in zip(shreds, shreds[1:]):
        if not cur.is_data or cur.slot != prev.slot or cur.index != prev.index + 1:
            raise DecodeError(f"shreds are not consecutive data shreds of one slot at index {cur.index}")
        if prev.data_complete:
            raise DecodeError(f"DATA_COMPLETE at index {prev.index} before the end of the batch")
    if not shreds[0].is_data or not shreds[-1].data_complete:
        raise DecodeError("the last shred of a batch must have DATA_COMPLETE")
    return parse_entry_batch(b"".join(s.data() for s in shreds))


def batch_ranges(present: Iterable[int], data_complete: Iterable[int]) -> tuple[list[range], int]:
    """Split one slot's data-shred indexes into batches.

    A batch starts at index 0 or right after a DATA_COMPLETE shred, and ends at a DATA_COMPLETE shred. Returns
    the complete batches (every index present, both ends known) and the number of batches that are known to exist
    but have missing shreds, counting shreds after the last DATA_COMPLETE as one unfinished batch.
    """
    present = set(present)
    ends = sorted(set(data_complete))
    complete, incomplete = [], 0
    start = 0
    for end in ends:
        span = range(start, end + 1)
        if all(i in present for i in span):
            complete.append(span)
        else:
            incomplete += 1
        start = end + 1
    if any(i >= start for i in present):
        incomplete += 1
    return complete, incomplete
