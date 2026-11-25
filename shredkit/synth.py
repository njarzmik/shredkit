"""Seeded synthetic content: keys, transactions in all three formats, entry batches and whole slots.

Everything is random bytes with valid structure and real signatures. Nothing here resembles real accounts or
programs, on purpose: the point is to exercise the formats, not to model traffic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .build import build_batch
from .entries import serialize_entries
from .tx import AddressTableLookup, Instruction, build_transaction

VERSIONS = ("legacy", "v0", "v1")


def make_key(rng: random.Random) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(rng.randbytes(32))


def random_transaction(rng: random.Random, version: str) -> bytes:
    signers = [make_key(rng) for _ in range(rng.randint(1, 2))]
    others = [rng.randbytes(32) for _ in range(rng.randint(1, 4))]
    n_keys = len(signers) + len(others)
    instructions = [Instruction(rng.randrange(n_keys), bytes(rng.randrange(n_keys) for _ in range(rng.randint(0, 4))),
                                rng.randbytes(rng.randint(0, 40))) for _ in range(rng.randint(1, 3))]
    extra: dict = {}
    if version == "v0":
        extra["lookups"] = [AddressTableLookup(rng.randbytes(32), bytes([0, 1]), bytes([2]))]
    if version == "v1":
        extra["config_mask"] = 0b101
        extra["config_values"] = rng.randbytes(8)
    return build_transaction(version, signers, others, instructions, rng.randbytes(32), **extra)


def random_batch(rng: random.Random, n_entries: int, txs_per_entry: int, zero_padding: int = 0) -> tuple[bytes, list[bytes]]:
    """A serialized Vec<Entry> (versions in rotation, one tick entry with no transactions at the end) plus
    optional zero padding, and the transactions in order."""
    entries, txs = [], []
    for e in range(n_entries):
        batch_txs = [random_transaction(rng, VERSIONS[(e * txs_per_entry + t) % 3]) for t in range(txs_per_entry)]
        entries.append((rng.randint(1, 12_500), rng.randbytes(32), batch_txs))
        txs += batch_txs
    entries.append((12_500, rng.randbytes(32), []))
    return serialize_entries(entries) + bytes(zero_padding), txs


@dataclass(frozen=True)
class SyntheticSlot:
    payloads: list[bytes]             # every shred, data and code, in build order
    transactions: list[bytes]         # every transaction, in slot order
    batch_sizes: list[int]            # bytes per batch, padding included


def synthetic_slot(rng: random.Random, leader: Ed25519PrivateKey, slot: int, n_batches: int = 3,
                   zero_padding: dict[int, int] | None = None) -> SyntheticSlot:
    """`n_batches` batches chained through their Merkle roots. The first batch is large enough to span two
    erasure sets; the last one is last-in-slot (resigned sets)."""
    retransmitter = make_key(rng)
    payloads, txs, sizes = [], [], []
    root, data_index, code_index = bytes(32), 0, 0
    for b in range(n_batches):
        n_entries = 30 if b == 0 else rng.randint(3, 8)
        blob, batch_txs = random_batch(rng, n_entries=n_entries, txs_per_entry=rng.randint(4, 6),
                                       zero_padding=(zero_padding or {}).get(b, 0))
        built = build_batch(leader, slot, blob, chained_root=root, first_data_index=data_index,
                            first_code_index=code_index, reference_tick=b, last_in_slot=b == n_batches - 1,
                            retransmitter=retransmitter)
        payloads += built.shreds
        txs += batch_txs
        sizes.append(len(blob))
        root, data_index, code_index = built.merkle_root, built.next_data_index, built.next_code_index
    return SyntheticSlot(payloads, txs, sizes)
