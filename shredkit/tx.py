"""Transaction wire formats: legacy, v0 and v1.

legacy / v0   compact-u16 signature count, the signatures, then the message. A v0 message starts with the byte
              0x80; it adds address-table lookups after the instructions.
v1            SIMD-0385. The transaction starts with 0x81, and the signatures come LAST:
                0x81, header (3 x u8), config mask (u32), lifetime / blockhash (32), n_instructions (u8),
                n_addresses (u8), addresses (32 each), config values (4 bytes per set mask bit),
                instruction headers (u8 program index, u8 n_accounts, u16 n_data each), then per instruction
                its account indexes and data, then num_required_signatures x 64-byte signatures.
              The signed message is everything before the signatures.

Telling them apart from the first byte: a legacy or v0 transaction starts with its signature count. A count
under 128 is one byte with the high bit clear, and 128 signatures (8 KiB) cannot fit in a transaction, so a first
byte with the high bit set is a version prefix. Reading a v1 transaction as if its signature came first gives a
wrong "signature" without any error, which is why the format is decided before anything else is read.

This module parses and checks signatures. It does not sanitize the way the runtime does (account index ranges,
duplicate keys, lookup resolution).
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .codec import ByteReader, encode_compact_u16
from .errors import DecodeError

V0_PREFIX = 0x80
V1_PREFIX = 0x81
SIGNATURE_SIZE = 64
KEY_SIZE = 32


@dataclass(frozen=True)
class Instruction:
    program_index: int
    accounts: bytes
    data: bytes


@dataclass(frozen=True)
class AddressTableLookup:
    account_key: bytes
    writable_indexes: bytes
    readonly_indexes: bytes


@dataclass(frozen=True)
class Transaction:
    version: str                              # "legacy", "v0" or "v1"
    signatures: tuple[bytes, ...]
    num_required_signatures: int
    num_readonly_signed: int
    num_readonly_unsigned: int
    account_keys: tuple[bytes, ...]
    recent_blockhash: bytes
    instructions: tuple[Instruction, ...]
    message: bytes                            # the bytes the signatures sign
    raw: bytes                                # the whole transaction as it was on the wire
    address_table_lookups: tuple[AddressTableLookup, ...] = ()
    config_mask: int = 0                      # v1 only
    config_values: bytes = b""                # v1 only

    @property
    def first_signature(self) -> bytes:
        """The transaction id."""
        return self.signatures[0]

    def verify_signatures(self) -> bool:
        """Signature i must be account key i's Ed25519 signature over the message."""
        for sig, key in zip(self.signatures, self.account_keys):
            try:
                Ed25519PublicKey.from_public_bytes(key).verify(sig, self.message)
            except (InvalidSignature, ValueError):
                return False
        return True


def parse_transaction(buf: bytes, offset: int = 0) -> tuple[Transaction, int]:
    """Parse one transaction at buf[offset:]. Returns it and the offset just past it (transactions are
    self-delimiting, which is how an entry's transactions are walked)."""
    reader = ByteReader(buf, offset)
    first = reader.peek_u8()
    if first == V1_PREFIX:
        return _parse_v1(reader)
    if first & 0x80:
        raise DecodeError(f"unknown transaction format byte 0x{first:02x}")
    return _parse_legacy_or_v0(reader)


def _parse_legacy_or_v0(r: ByteReader) -> tuple[Transaction, int]:
    start = r.pos
    n_sig = r.compact_u16()
    if n_sig == 0:
        raise DecodeError("transaction has no signatures")
    signatures = tuple(r.take(SIGNATURE_SIZE) for _ in range(n_sig))
    message_start = r.pos
    version = "legacy"
    if r.peek_u8() & 0x80:
        prefix = r.u8()
        if prefix != V0_PREFIX:
            raise DecodeError(f"unsupported message version {prefix & 0x7F}")
        version = "v0"
    header = (r.u8(), r.u8(), r.u8())
    keys = tuple(r.take(KEY_SIZE) for _ in range(r.compact_u16()))
    blockhash = r.take(32)
    instructions = tuple(_legacy_instruction(r) for _ in range(r.compact_u16()))
    lookups = tuple(_lookup(r) for _ in range(r.compact_u16())) if version == "v0" else ()
    _check_signers(header[0], n_sig, len(keys))
    tx = Transaction(version, signatures, *header, keys, blockhash, instructions,
                     message=bytes(r.buf[message_start:r.pos]), raw=bytes(r.buf[start:r.pos]),
                     address_table_lookups=lookups)
    return tx, r.pos


def _legacy_instruction(r: ByteReader) -> Instruction:
    program_index = r.u8()
    accounts = r.take(r.compact_u16())
    return Instruction(program_index, accounts, r.take(r.compact_u16()))


def _lookup(r: ByteReader) -> AddressTableLookup:
    key = r.take(KEY_SIZE)
    writable = r.take(r.compact_u16())
    return AddressTableLookup(key, writable, r.take(r.compact_u16()))


def _parse_v1(r: ByteReader) -> tuple[Transaction, int]:
    start = r.pos
    r.u8()
    header = (r.u8(), r.u8(), r.u8())
    mask = r.u32()
    blockhash = r.take(32)
    n_ix, n_addr = r.u8(), r.u8()
    keys = tuple(r.take(KEY_SIZE) for _ in range(n_addr))
    config = r.take(4 * mask.bit_count())
    heads = [(r.u8(), r.u8(), r.u16()) for _ in range(n_ix)]
    instructions = tuple(Instruction(p, r.take(n_acc), r.take(n_data)) for p, n_acc, n_data in heads)
    message_end = r.pos
    _check_signers(header[0], header[0], n_addr)
    signatures = tuple(r.take(SIGNATURE_SIZE) for _ in range(header[0]))
    tx = Transaction("v1", signatures, *header, keys, blockhash, instructions,
                     message=bytes(r.buf[start:message_end]), raw=bytes(r.buf[start:r.pos]),
                     config_mask=mask, config_values=config)
    return tx, r.pos


def _check_signers(required: int, n_sig: int, n_keys: int) -> None:
    if required == 0:
        raise DecodeError("no required signatures")
    if required != n_sig:
        raise DecodeError(f"header requires {required} signatures, transaction has {n_sig}")
    if required > n_keys:
        raise DecodeError(f"{required} signers but only {n_keys} account keys")


# ---------------------------------------------------------------------------------------------- building (tests)

def public_key_bytes(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def build_transaction(version: str, signers: Sequence[Ed25519PrivateKey], other_keys: Sequence[bytes],
                      instructions: Sequence[Instruction], blockhash: bytes, *, readonly_signed: int = 0,
                      readonly_unsigned: int = 0, lookups: Sequence[AddressTableLookup] = (),
                      config_mask: int = 0, config_values: bytes = b"") -> bytes:
    """Serialize and sign a transaction in any of the three formats, for synthetic data and tests."""
    keys = [public_key_bytes(k) for k in signers] + list(other_keys)
    header = bytes([len(signers), readonly_signed, readonly_unsigned])
    if version == "v1":
        if len(config_values) != 4 * config_mask.bit_count():
            raise ValueError("v1 config values must be 4 bytes per set mask bit")
        msg = bytearray([V1_PREFIX]) + header + struct.pack("<I", config_mask) + blockhash
        msg += bytes([len(instructions), len(keys)]) + b"".join(keys) + config_values
        msg += b"".join(struct.pack("<BBH", ix.program_index, len(ix.accounts), len(ix.data)) for ix in instructions)
        msg += b"".join(ix.accounts + ix.data for ix in instructions)
        return bytes(msg) + b"".join(k.sign(bytes(msg)) for k in signers)
    if version not in ("legacy", "v0"):
        raise ValueError(f"unknown version {version!r}")
    msg = bytearray([V0_PREFIX]) if version == "v0" else bytearray()
    msg += header + encode_compact_u16(len(keys)) + b"".join(keys) + blockhash
    msg += encode_compact_u16(len(instructions))
    for ix in instructions:
        msg += bytes([ix.program_index]) + encode_compact_u16(len(ix.accounts)) + ix.accounts
        msg += encode_compact_u16(len(ix.data)) + ix.data
    if version == "v0":
        msg += encode_compact_u16(len(lookups))
        for lk in lookups:
            msg += lk.account_key + encode_compact_u16(len(lk.writable_indexes)) + lk.writable_indexes
            msg += encode_compact_u16(len(lk.readonly_indexes)) + lk.readonly_indexes
    return encode_compact_u16(len(signers)) + b"".join(k.sign(bytes(msg)) for k in signers) + bytes(msg)
