import random

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from shredkit.codec import ByteReader, b58encode, encode_compact_u16
from shredkit.errors import DecodeError
from shredkit.synth import make_key
from shredkit.tx import AddressTableLookup, Instruction, build_transaction, parse_transaction, public_key_bytes

RNG = random.Random(11)
SIGNERS = [make_key(RNG), make_key(RNG)]
OTHERS = [RNG.randbytes(32) for _ in range(3)]
IXS = [Instruction(3, bytes([0, 2, 4]), b"\x01\x02\x03"), Instruction(4, b"", RNG.randbytes(70))]
BLOCKHASH = RNG.randbytes(32)
LOOKUP = AddressTableLookup(RNG.randbytes(32), bytes([0, 5]), bytes([1]))


def _build(version: str) -> bytes:
    extra = {}
    if version == "v0":
        extra = {"lookups": [LOOKUP]}
    if version == "v1":
        extra = {"config_mask": 0b1011, "config_values": RNG.randbytes(12)}
    return build_transaction(version, SIGNERS, OTHERS, IXS, BLOCKHASH, readonly_unsigned=1, **extra)


@pytest.mark.parametrize("version", ["legacy", "v0", "v1"])
def test_round_trip(version):
    raw = _build(version)
    tx, end = parse_transaction(raw + b"tail", 0)
    assert end == len(raw) and tx.raw == raw and tx.version == version
    assert tx.account_keys == tuple(public_key_bytes(k) for k in SIGNERS) + tuple(OTHERS)
    assert tx.instructions == tuple(IXS) and tx.recent_blockhash == BLOCKHASH
    assert (tx.num_required_signatures, tx.num_readonly_signed, tx.num_readonly_unsigned) == (2, 0, 1)
    assert tx.address_table_lookups == ((LOOKUP,) if version == "v0" else ())
    assert tx.verify_signatures()


def test_v1_signatures_come_last():
    """The control for the mistake this format invites: reading a v1 transaction signature-first."""
    raw = _build("v1")
    tx, _ = parse_transaction(raw)
    assert tx.signatures == (raw[-128:-64], raw[-64:])
    assert tx.message == raw[:-128]
    wrong = raw[1:65]                                  # where a legacy reader would look
    assert wrong != tx.first_signature
    with pytest.raises(InvalidSignature):
        Ed25519PublicKey.from_public_bytes(tx.account_keys[0]).verify(wrong, tx.message)


@pytest.mark.parametrize("version", ["legacy", "v0", "v1"])
def test_every_truncation_raises(version):
    raw = _build(version)
    for cut in range(len(raw)):
        with pytest.raises(DecodeError):
            parse_transaction(raw[:cut])


def test_tampered_message_fails_verification():
    raw = bytearray(_build("legacy"))
    raw[-1] ^= 0x01                                    # last byte of the last instruction's data
    tx, _ = parse_transaction(bytes(raw))
    assert not tx.verify_signatures()


@pytest.mark.parametrize("first", [0x80, 0x82, 0xFF])
def test_unknown_format_byte_is_rejected(first):
    with pytest.raises(DecodeError, match="format"):
        parse_transaction(bytes([first]) + _build("legacy")[1:])


def test_unknown_message_version_is_rejected():
    raw = bytearray(_build("v0"))
    raw[1 + 2 * 64] = 0x81                             # the v0 prefix after the signatures, now "version 1"
    with pytest.raises(DecodeError, match="message version 1"):
        parse_transaction(bytes(raw))


def test_signature_count_must_match_the_header():
    raw = bytearray(_build("legacy"))
    raw[1 + 2 * 64] = 3                                # header says 3 signers, 2 signatures present
    with pytest.raises(DecodeError, match="requires 3 signatures"):
        parse_transaction(bytes(raw))


@pytest.mark.parametrize("value", [0, 1, 127, 128, 255, 16_383, 16_384, 65_535])
def test_compact_u16_round_trip(value):
    enc = encode_compact_u16(value)
    assert ByteReader(enc).compact_u16() == value and len(enc) == 1 + (value > 127) + (value > 16_383)


@pytest.mark.parametrize("raw,message", [(b"\x80\x00", "non-canonical"), (b"\xff\xff\x04", "overflow"),
                                         (b"\x80\x80\x80\x01", "longer"), (b"\x80", "truncated")])
def test_compact_u16_rejects_bad_encodings(raw, message):
    with pytest.raises(DecodeError, match=message):
        ByteReader(raw).compact_u16()


def test_base58_known_values():
    assert b58encode(b"hello world") == "StV1DL6CwTryKyV"
    assert b58encode(b"\x00\x00\x01") == "112"
    assert b58encode(b"") == ""
