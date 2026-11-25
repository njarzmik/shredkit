"""A signed synthetic batch builder, so every check in this package can be tested without real data.

It follows the order of agave v4.4.0 ledger/src/shred/merkle.rs make_shreds_from_data (:939-1102) and
finish_erasure_batch (:1143-1212): split the batch into full erasure sets of 32 data shreds plus one last set
(padded with short or empty data shreds), put DATA_COMPLETE (or LAST_IN_SLOT) on the very last data shred, then
per set: Reed-Solomon encode, write the chained root (the previous set's root), build the Merkle tree, and have
the leader sign the root. Sets in the last batch of a slot are "resigned": they reserve 64 bytes for a
retransmitter signature, which costs data capacity (899 instead of 963 bytes per data shred).

It is a test fixture, not a shredder: it does not try to match how Agave sizes the data in a short last set.
"""

from __future__ import annotations

from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from . import rs
from .layout import (
    DATA_HEADER_SIZE,
    FLAG_DATA_COMPLETE,
    FLAG_LAST_IN_SLOT,
    FLAG_REFERENCE_TICK_MASK,
    MAX_DATA_SHREDS_PER_SLOT,
    MERKLE_ROOT_SIZE,
    SHREDS_PER_SET,
    Variant,
    assemble_payload,
    code_header,
    data_header,
)
from .merkle import hash_leaf, make_proof, tree_levels

PROOF_SIZE = 6     # ceil(log2(32 + 32)) levels


@dataclass(frozen=True)
class BuiltBatch:
    shreds: list[bytes]          # per set: 32 data payloads, then 32 code payloads
    merkle_root: bytes           # root of the last set: the chained root for the next batch
    next_data_index: int
    next_code_index: int


def build_batch(leader: Ed25519PrivateKey, slot: int, data: bytes, *, parent_offset: int = 1, version: int = 0,
                chained_root: bytes = bytes(MERKLE_ROOT_SIZE), first_data_index: int = 0,
                first_code_index: int = 0, reference_tick: int = 0, last_in_slot: bool = False,
                retransmitter: Ed25519PrivateKey | None = None) -> BuiltBatch:
    """Shreds for one entry batch (`data` = serialized Vec<Entry>, or any bytes)."""
    variant = Variant(True, last_in_slot, PROOF_SIZE)
    per_set = SHREDS_PER_SET * variant.capacity
    n_sets = max(1, -(-len(data) // per_set))
    if first_data_index + n_sets * SHREDS_PER_SET > MAX_DATA_SHREDS_PER_SLOT:
        raise ValueError("batch does not fit in the slot's index space")
    shreds: list[bytes] = []
    root = chained_root
    data_index, code_index = first_data_index, first_code_index
    for s in range(n_sets):
        last_flags = (FLAG_LAST_IN_SLOT if last_in_slot else FLAG_DATA_COMPLETE) if s == n_sets - 1 else 0
        payloads, root = _build_set(leader, retransmitter, variant, slot, parent_offset, version, data_index,
                                    code_index, data[s * per_set:(s + 1) * per_set], root,
                                    reference_tick & FLAG_REFERENCE_TICK_MASK, last_flags)
        shreds += payloads
        data_index += SHREDS_PER_SET
        code_index += SHREDS_PER_SET
    return BuiltBatch(shreds, root, data_index, code_index)


def _build_set(leader: Ed25519PrivateKey, retransmitter: Ed25519PrivateKey | None, data_variant: Variant,
               slot: int, parent_offset: int, version: int, fec_set_index: int, code_index: int, chunk: bytes,
               chained_root: bytes, flags: int, last_flags: int) -> tuple[list[bytes], bytes]:
    cap = data_variant.capacity
    data_bodies = []
    for j in range(SHREDS_PER_SET):
        piece = chunk[j * cap:(j + 1) * cap]
        shred_flags = flags | (last_flags if j == SHREDS_PER_SET - 1 else 0)
        head = data_header(data_variant, slot, fec_set_index + j, version, fec_set_index, parent_offset,
                           shred_flags, DATA_HEADER_SIZE + len(piece))
        data_bodies.append(head + piece.ljust(cap, b"\x00"))
    code_variant = data_variant.sibling(False)
    parity = rs.encode(data_bodies, SHREDS_PER_SET)
    code_bodies = [code_header(code_variant, slot, code_index + j, version, fec_set_index, SHREDS_PER_SET,
                               SHREDS_PER_SET, j) + parity[j] for j in range(SHREDS_PER_SET)]
    bodies = data_bodies + code_bodies
    levels = tree_levels([hash_leaf(b + chained_root) for b in bodies])
    root = levels[-1][0]
    signature = leader.sign(root)
    retrans_sig = retransmitter.sign(root) if (retransmitter and data_variant.resigned) else b""
    payloads = [assemble_payload(data_variant if i < SHREDS_PER_SET else code_variant, signature, body,
                                 chained_root, make_proof(levels, i), retrans_sig)
                for i, body in enumerate(bodies)]
    return payloads, root
