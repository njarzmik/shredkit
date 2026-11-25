"""Shared synthetic fixtures. Every byte the tests use is built here or in the test itself, from fixed seeds."""

from __future__ import annotations

import random
import struct
from dataclasses import dataclass

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from shredkit.build import BuiltBatch, build_batch
from shredkit.layout import Shred, parse_shred
from shredkit.synth import make_key, random_batch
from shredkit.tx import public_key_bytes


@dataclass
class SignedBatch:
    blob: bytes
    txs: list[bytes]
    built: BuiltBatch
    shreds: list[Shred]
    leader: Ed25519PrivateKey
    leader_pub: bytes


def make_signed_batch(seed: int, *, n_entries: int = 30, last_in_slot: bool = False, padding: int = 0,
                      slot: int = 5_000) -> SignedBatch:
    rng = random.Random(seed)
    leader = make_key(rng)
    blob, txs = random_batch(rng, n_entries=n_entries, txs_per_entry=4, zero_padding=padding)
    retransmitter = make_key(rng) if last_in_slot else None
    built = build_batch(leader, slot, blob, reference_tick=7, last_in_slot=last_in_slot, retransmitter=retransmitter)
    return SignedBatch(blob, txs, built, [parse_shred(p) for p in built.shreds], leader, public_key_bytes(leader))


@pytest.fixture(scope="session")
def batch() -> SignedBatch:
    """30 entries of 4 transactions: two erasure sets (128 shreds)."""
    return make_signed_batch(1)


@pytest.fixture(scope="session")
def resigned_batch() -> SignedBatch:
    """A last-in-slot batch: resigned sets with a retransmitter signature."""
    return make_signed_batch(2, n_entries=6, last_in_slot=True)


def patch(payload: bytes, offset: int, fmt: str, *values: int) -> bytes:
    """A copy of `payload` with struct-packed values written at `offset`."""
    buf = bytearray(payload)
    struct.pack_into(fmt, buf, offset, *values)
    return bytes(buf)
