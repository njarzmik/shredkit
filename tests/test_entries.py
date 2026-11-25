import random
import struct

import pytest
from conftest import make_signed_batch

from shredkit.build import build_batch
from shredkit.entries import batch_ranges, deshred, parse_entry_batch, serialize_entries
from shredkit.errors import DecodeError
from shredkit.layout import parse_shred
from shredkit.synth import random_batch


def _blob(seed: int = 3, padding: int = 0):
    return random_batch(random.Random(seed), n_entries=5, txs_per_entry=3, zero_padding=padding)


def test_round_trip():
    entries = [(5, bytes([1]) * 32, []), (7, bytes([2]) * 32, [])]
    out = parse_entry_batch(serialize_entries(entries))
    assert [(e.num_hashes, e.hash) for e in out.entries] == [(5, bytes([1]) * 32), (7, bytes([2]) * 32)]
    blob, txs = _blob()
    out = parse_entry_batch(blob)
    assert out.status == "exact" and [t.raw for t in out.transactions] == txs
    assert len(out.entries) == 6 and out.entries[-1].transactions == ()        # the trailing tick entry


def test_zero_padded_batch_is_accepted_and_reported():
    """The real-bytes failure: entries that end before the batch does, followed by zeros only. The old
    'parse to the last byte' rule rejects this batch; the validator does not."""
    blob, txs = _blob(padding=700)
    out = parse_entry_batch(blob)
    assert out.end != len(blob)                       # what the old rule tested, and failed on
    assert out.status == "zero_padded" and out.ok and out.tail_length == 700
    assert [t.raw for t in out.transactions] == txs


def test_non_zero_trailing_bytes_are_flagged():
    blob, _ = _blob()
    out = parse_entry_batch(blob + bytes(50) + b"\x01")
    assert out.status == "trailing_bytes" and not out.ok and out.tail_length == 51


@pytest.mark.parametrize("cut", random.Random(4).sample(range(1, 2_000), 40))
def test_truncated_batch_raises(cut):
    blob, _ = _blob()
    with pytest.raises(DecodeError):
        parse_entry_batch(blob[:min(cut, len(blob) - 1)])


def test_impossible_counts_raise_before_looping():
    with pytest.raises(DecodeError, match="entry count"):
        parse_entry_batch(struct.pack("<Q", 10**12) + bytes(100))
    one_entry = struct.pack("<QQ", 1, 1) + bytes(32) + struct.pack("<Q", 10**9)    # 1 entry claiming 10^9 txs
    with pytest.raises(DecodeError, match="transaction count"):
        parse_entry_batch(one_entry + bytes(500))


def test_empty_batches_and_block_markers():
    assert parse_entry_batch(b"").status == "exact"
    assert parse_entry_batch(bytes(8)).status == "exact"
    assert parse_entry_batch(bytes(900)).status == "zero_padded"
    marker = parse_entry_batch(struct.pack("<QH", 0, 1) + b"marker data")
    assert marker.status == "marker" and marker.marker_version == 1 and marker.entries == ()


def test_deshred_checks_order_and_completion(batch):
    data = [s for s in batch.shreds if s.is_data]
    assert [t.raw for t in deshred(data).transactions] == batch.txs
    with pytest.raises(DecodeError, match="consecutive"):
        deshred(data[:10] + data[11:])
    with pytest.raises(DecodeError, match="DATA_COMPLETE"):
        deshred(data[:-1])
    nxt = build_batch(batch.leader, 5_000, b"\x00" * 8, first_data_index=64, first_code_index=64)
    second = [parse_shred(p) for p in nxt.shreds[:32]]
    with pytest.raises(DecodeError, match="before the end"):
        deshred(data + second)


def test_deshred_a_padded_signed_batch():
    b = make_signed_batch(5, n_entries=4, padding=1_500)
    out = deshred([s for s in b.shreds if s.is_data])
    assert out.status == "zero_padded" and out.tail_length == 1_500
    assert [t.raw for t in out.transactions] == b.txs


def test_batch_ranges():
    assert batch_ranges(range(10), [4, 9]) == ([range(0, 5), range(5, 10)], 0)
    assert batch_ranges([0, 1, 2, 3, 4, 5, 7, 8, 9], [4, 9]) == ([range(0, 5)], 1)
    assert batch_ranges(range(12), [4, 9]) == ([range(0, 5), range(5, 10)], 1)     # 10, 11: unfinished
    assert batch_ranges(range(1, 10), [4, 9]) == ([range(5, 10)], 1)               # index 0 missing
