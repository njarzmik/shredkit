import pytest
from conftest import patch

from shredkit.errors import ShredError
from shredkit.layout import (
    CODE_HEADER_SIZE,
    CODE_PAYLOAD_SIZE,
    DATA_PAYLOAD_SIZE,
    FLAG_DATA_COMPLETE,
    SIGNATURE_SIZE,
    Variant,
    parse_shred,
)


def test_sizes_match_the_agave_layout():
    data, code = Variant(True, False, 6), Variant(False, False, 6)
    assert (data.capacity, code.capacity, Variant(True, True, 6).capacity) == (963, 987, 899)
    assert (data.proof_offset, code.proof_offset) == (1083, 1108)
    # Reed-Solomon needs data and code erasure shards of one set to be the same length (merkle.rs:434-435).
    assert DATA_PAYLOAD_SIZE - SIGNATURE_SIZE == CODE_PAYLOAD_SIZE - CODE_HEADER_SIZE
    for resigned in (False, True):
        (a, b), (c, d) = Variant(True, resigned, 6).erasure_range, Variant(False, resigned, 6).erasure_range
        assert b - a == d - c


@pytest.mark.parametrize("byte", [0x66, 0x76, 0x96, 0xB6, 0x65, 0x9F])
def test_variant_byte_round_trip(byte):
    assert Variant.from_byte(byte).to_byte() == byte


@pytest.mark.parametrize("byte", [0x00, 0x46, 0x5A, 0x86, 0xA5, 0xC6, 0xFF])
def test_unknown_variant_is_rejected(byte, batch):
    """Legacy (0x5A / 0xA5) and unchained Merkle (0x4_ / 0x8_) shreds are not supported, and say so."""
    with pytest.raises(ShredError, match="unsupported shred variant"):
        Variant.from_byte(byte)
    with pytest.raises(ShredError, match="unsupported shred variant"):
        parse_shred(patch(batch.built.shreds[0], 64, "<B", byte))


def test_built_headers_parse_back(batch):
    data = [s for s in batch.shreds if s.is_data]
    code = [s for s in batch.shreds if not s.is_data]
    assert [s.index for s in data] == list(range(64))
    assert [s.position for s in code] == list(range(32)) * 2
    assert {s.fec_set_index for s in data} == {0, 32}
    assert all(s.reference_tick == 7 and s.slot == 5_000 and s.parent_offset == 1 for s in data)
    assert [s.index for s in data if s.data_complete] == [63]
    assert b"".join(s.data() for s in data) == batch.blob


@pytest.mark.parametrize("which,length", [(0, 0), (0, 88), (0, 89), (0, 1202), (40, 1227)])
def test_truncated_shred_is_rejected(batch, which, length):
    with pytest.raises(ShredError, match="truncated"):
        parse_shred(batch.built.shreds[which][:length])


def test_bytes_after_the_payload_are_ignored(batch):
    """A repair response appends a nonce; the shred itself is unchanged."""
    raw = batch.built.shreds[3]
    assert parse_shred(raw + b"\x01\x02\x03\x04") == parse_shred(raw)


@pytest.mark.parametrize("offset,fmt,value,message", [
    (86, "<H", 87, "data size"),                       # below the header
    (86, "<H", 88 + 964, "data size"),                 # beyond capacity
    (79, "<I", 4, "below its fec_set_index"),          # index 0 < fec_set_index 4
    (85, "<B", 0x80, "LAST_IN_SLOT"),                  # 0x80 without the 0x40 bit
    (83, "<H", 6_000, "parent offset"),                # parent offset beyond the slot
    (83, "<H", 0, "parent offset"),                    # parent offset 0 in a non-zero slot
])
def test_data_header_range_checks(batch, offset, fmt, value, message):
    with pytest.raises(ShredError, match=message):
        parse_shred(patch(batch.built.shreds[0], offset, fmt, value))


@pytest.mark.parametrize("offset,value,message", [(87, 32, "position"), (83, 0, "not codable")])
def test_code_header_range_checks(batch, offset, value, message):
    with pytest.raises(ShredError, match=message):
        parse_shred(patch(batch.built.shreds[32], offset, "<H", value))


def test_last_in_slot_includes_data_complete(resigned_batch):
    last = [s for s in resigned_batch.shreds if s.is_data][-1]
    assert last.last_in_slot and last.data_complete and last.flags & FLAG_DATA_COMPLETE
    assert last.variant.resigned and last.variant.capacity == 899
