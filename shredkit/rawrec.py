"""Deshred a rawrec recording: the payload files plus the batch boundaries from the record files.

rawrec (https://github.com/njarzmik/rawrec) is a UDP shred recorder. Its run directory holds, among others:

  rec_<period>.bin   64-byte file header, then one 64-byte record per datagram: slot at offset 16 (u64), index at
                     24 (u32), the 6 header bytes after the common header at 46 (data: parent offset u16,
                     flags u8, size u16, ...), variant at 53 (u8), rflags at 56 (u16)
  pay_<period>.bin   64-byte file header, then for each data shred at its first arrival: {u64 slot, u32 index,
                     u16 n, u8 arm, u8 pad} and n bytes = the shred's b[88:size]

Every file header is {magic "RAWREC01", u16 version, u16 record size, u32 kind (1 rec, 2 pay), 16-byte run id,
u64 realtime, u64 monotonic, 16 reserved}. All little-endian. The layouts are documented in rawrec's README;
they are parsed here directly so this package does not depend on rawrec's Python reader.

A payload record has the data bytes but not the flags, and the DATA_COMPLETE flag is what ends a batch. The flags
are in the record files, so both are read: the record files once (only the DATA_COMPLETE data shreds are kept, a
few per slot), then the payload files streamed slot by slot. Payloads carry no signature or proof and rawrec does
not store code shreds' payloads, so batches from a recording are neither signature-checked nor recoverable here.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .entries import EntryBatch, batch_ranges, parse_entry_batch
from .errors import DecodeError
from .layout import FLAG_DATA_COMPLETE, parse_shred

MAGIC = b"RAWREC01"
KIND_REC, KIND_PAY = 1, 2
FILE_HEADER = struct.Struct("<8sHHI16sQQ16s")
PAY_HEADER = struct.Struct("<QIHBB")
RF_PAYLOAD_STORED, RF_SHORT, RF_PARSE_FAIL = 1, 8, 16

REC_DTYPE = np.dtype([
    ("t_user", "<u8"), ("t_kern", "<u8"), ("slot", "<u8"), ("index", "<u4"), ("fec", "<u4"),
    ("rxq_ovfl", "<u4"), ("src_ip", "<u4"), ("src_port", "<u2"), ("len", "<u2"), ("version", "<u2"),
    ("hdr", "u1", (6,)), ("arm", "u1"), ("variant", "u1"), ("ttl", "u1"), ("batch_n", "u1"),
    ("rflags", "<u2"), ("batch_i", "<u2"), ("seq", "<u4"),
])
assert REC_DTYPE.itemsize == 64 and FILE_HEADER.size == 64 and PAY_HEADER.size == 16


@dataclass
class RunStats:
    payload_records: int = 0
    duplicate_payloads: int = 0        # the same (slot, index) stored twice (rawrec does that for very late slots)
    conflicting_payloads: int = 0      # ... with different bytes: the first copy is kept
    late_payloads: int = 0             # arrived after their slot was already finished (see `window`)
    torn_bytes: int = 0                # an incomplete last record, e.g. after a crash


@dataclass(frozen=True)
class BatchResult:
    slot: int
    first_index: int
    last_index: int
    batch: EntryBatch | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.batch is not None and self.batch.ok


@dataclass(frozen=True)
class SlotResult:
    slot: int
    data_shreds: int
    batches: tuple[BatchResult, ...]
    incomplete_batches: int


def _check_header(raw: bytes, kind: int, path: Path) -> None:
    if len(raw) < FILE_HEADER.size:
        raise ValueError(f"{path.name}: shorter than the 64-byte file header")
    magic, _version, _recsize, file_kind, *_ = FILE_HEADER.unpack(raw)
    if magic != MAGIC or file_kind != kind:
        raise ValueError(f"{path.name}: not a rawrec file of kind {kind}")


def data_complete_indexes(run_dir: Path, chunk_records: int = 1 << 20) -> dict[int, set[int]]:
    """{slot: indexes of data shreds with DATA_COMPLETE}, from every rec_*.bin, read in chunks."""
    out: dict[int, set[int]] = {}
    for path in sorted(run_dir.glob("rec_*.bin")):
        with open(path, "rb") as f:
            _check_header(f.read(FILE_HEADER.size), KIND_REC, path)
            while chunk := f.read(REC_DTYPE.itemsize * chunk_records):
                recs = np.frombuffer(chunk, dtype=REC_DTYPE, count=len(chunk) // REC_DTYPE.itemsize)
                high = recs["variant"] >> 4
                keep = ((high == 0x9) | (high == 0xB)) & ((recs["hdr"][:, 2] & FLAG_DATA_COMPLETE) != 0)
                keep &= (recs["rflags"] & (RF_SHORT | RF_PARSE_FAIL)) == 0
                for slot, index in zip(recs["slot"][keep].tolist(), recs["index"][keep].tolist()):
                    out.setdefault(slot, set()).add(index)
    return out


def iter_payloads(path: Path, stats: RunStats) -> Iterator[tuple[int, int, bytes]]:
    """(slot, index, data) per record of one pay_*.bin; stops at a torn tail and counts it."""
    with open(path, "rb", buffering=1 << 20) as f:
        _check_header(f.read(FILE_HEADER.size), KIND_PAY, path)
        while head := f.read(PAY_HEADER.size):
            if len(head) < PAY_HEADER.size:
                stats.torn_bytes += len(head)
                return
            slot, index, n, _arm, _pad = PAY_HEADER.unpack(head)
            body = f.read(n)
            if len(body) < n:
                stats.torn_bytes += len(head) + len(body)
                return
            yield slot, index, body


def deshred_run(run_dir: str | Path, window: int = 256, stats: RunStats | None = None) -> Iterator[SlotResult]:
    """Yield each slot's batches once the recording has moved `window` slots past it, then the rest at the end.
    Memory stays at about `window` slots of payloads however long the recording is."""
    run_dir = Path(run_dir)
    stats = stats if stats is not None else RunStats()
    pay_files = sorted(run_dir.glob("pay_*.bin"))
    if not pay_files or not any(run_dir.glob("rec_*.bin")):
        raise FileNotFoundError(f"{run_dir}: needs pay_*.bin and rec_*.bin (the flags that end a batch)")
    ends = data_complete_indexes(run_dir)
    pending: dict[int, dict[int, bytes]] = {}
    finished_through, newest = -1, -1
    for path in pay_files:
        for slot, index, data in iter_payloads(path, stats):
            stats.payload_records += 1
            if slot <= finished_through:
                stats.late_payloads += 1
                continue
            _add(pending.setdefault(slot, {}), index, data, stats)
            if slot > newest:
                newest = slot
                if newest - window > finished_through:
                    finished_through = newest - window
                    yield from _finish(pending, ends, lambda s: s <= finished_through)
    yield from _finish(pending, ends, lambda s: True)


def _add(slot_data: dict[int, bytes], index: int, data: bytes, stats: RunStats) -> None:
    prev = slot_data.setdefault(index, data)
    if prev is not data:
        stats.duplicate_payloads += 1
        stats.conflicting_payloads += prev != data


def _finish(pending: dict[int, dict[int, bytes]], ends: dict[int, set[int]],
            due: Callable[[int], bool]) -> Iterator[SlotResult]:
    for slot in sorted(s for s in pending if due(s)):
        shreds = pending.pop(slot)
        complete, incomplete = batch_ranges(shreds, ends.get(slot, ()))
        batches = tuple(_deshred_range(slot, span, shreds) for span in complete)
        yield SlotResult(slot, len(shreds), batches, incomplete)


def _deshred_range(slot: int, span: range, shreds: dict[int, bytes]) -> BatchResult:
    try:
        batch = parse_entry_batch(b"".join(shreds[i] for i in span))
    except DecodeError as e:
        return BatchResult(slot, span.start, span.stop - 1, None, str(e))
    return BatchResult(slot, span.start, span.stop - 1, batch)


# ---------------------------------------------------------------------------------------------- writing (tests)

def write_run(run_dir: str | Path, payloads: Iterable[bytes], period: str = "20260101T00",
              run_id: bytes = b"SYNTHETIC0000000") -> None:
    """Write rec_ and pay_ files as rawrec would if these shreds arrived once each on arm 0, in this order.
    For tests and the synth-run command; only the fields this module reads are filled in."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shreds = [parse_shred(p) for p in payloads]
    recs = np.zeros(len(shreds), dtype=REC_DTYPE)
    for i, s in enumerate(shreds):
        recs[i] = (0, 0, s.slot, s.index, s.fec_set_index, 0, 0, 0, len(s.payload), s.version,
                   tuple(s.payload[83:89]), 0, s.variant.to_byte(), 0, 0, RF_PAYLOAD_STORED if s.is_data else 0, 0, i)
    header = FILE_HEADER.pack(MAGIC, 1, 64, KIND_REC, run_id, 0, 0, b"")
    (run_dir / f"rec_{period}.bin").write_bytes(header + recs.tobytes())
    with open(run_dir / f"pay_{period}.bin", "wb") as f:
        f.write(FILE_HEADER.pack(MAGIC, 1, 0, KIND_PAY, run_id, 0, 0, b""))
        for s in shreds:
            if s.is_data:
                f.write(PAY_HEADER.pack(s.slot, s.index, len(s.data()), 0, 0) + s.data())
